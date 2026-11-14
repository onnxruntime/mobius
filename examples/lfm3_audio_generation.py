# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Export and run offline float32 LFM3 Audio ASR or speech synthesis.

Examples::

    python examples/lfm3_audio_generation.py --audio recording.wav --trust-processor-code
    python examples/lfm3_audio_generation.py --text "Hello world." --trust-processor-code

The graph export reads configurations as data. The explicitly trusted, pinned
upstream processor supplies the audio frontend and chat template. This example
does not implement live audio, interleaved responses, or stochastic sampling.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from transformers import AutoProcessor

from mobius.integrations.lfm3_audio import Lfm3AudioPipeline, build_lfm3_audio_package

MODEL_ID = "LiquidAI/LFM3-Audio-2B"
REVISION = "2d6c1f99c129b3ef07a8d168bc0ff08e2bdb93f1"


class _Session:
    def __init__(self, path: Path, provider: str):
        self.session = ort.InferenceSession(str(path), providers=[provider])
        if provider not in self.session.get_providers():
            raise RuntimeError(
                f"{provider} could not initialize for {path.name}; install a compatible "
                "ONNX Runtime build and check its CUDA dependencies, or select --device cpu."
            )

    def run(self, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        names = [output.name for output in self.session.get_outputs()]
        return dict(zip(names, self.session.run(names, feeds), strict=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", type=Path)
    source.add_argument("--text")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--export-dir", type=Path, default=Path("lfm3-audio-export"))
    parser.add_argument("--output-wav", type=Path, default=Path("speech.wav"))
    parser.add_argument("--trust-processor-code", action="store_true")
    args = parser.parse_args()
    if not args.trust_processor_code:
        parser.error(
            "Pass --trust-processor-code only after reviewing the pinned upstream processor."
        )
    provider = "CUDAExecutionProvider" if args.device == "cuda" else "CPUExecutionProvider"
    if provider not in ort.get_available_providers():
        parser.error(f"{provider} is unavailable; install the matching ONNX Runtime package.")
    package = build_lfm3_audio_package(execution_provider=args.device)
    package.save(args.export_dir)
    sessions = {
        name: _Session(args.export_dir / name / "model.onnx", provider) for name in package
    }
    pipeline = Lfm3AudioPipeline(package.config, sessions)
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True
    )
    if args.audio is not None:
        inputs = processor.apply_chat_template(
            [
                {"role": "system", "content": "Perform ASR in English."},
                {"role": "user", "content": [{"type": "audio", "audio": str(args.audio)}]},
            ],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="np",
        )
        pipeline.prefill(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            input_features=inputs["input_features"].astype(np.float32),
            feature_lengths=inputs["feature_attention_mask"].sum(-1).astype(np.int64),
        )
        print(processor.decode(pipeline.generate_text(256), skip_special_tokens=True))
    else:
        inputs = processor.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": "Perform TTS in American English using a female voice.",
                },
                {"role": "user", "content": args.text},
                {"role": "assistant", "content": "<|audio_start|>"},
            ],
            continue_final_message=True,
            tokenize=True,
            return_dict=True,
            return_tensors="np",
        )
        pipeline.prefill(inputs["input_ids"], attention_mask=inputs["attention_mask"])
        codes = pipeline.generate_speech(256)
        waveform = pipeline.synthesize(codes)
        sf.write(args.output_wav, waveform[0], 24000)
        print(args.output_wav)


if __name__ == "__main__":
    main()
