# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Export and run offline float32 LFM3-Omni: image questions, ASR, TTS and voice chat.

Examples::

    python examples/lfm3_omni_generation.py --image photo.jpg --text "What is it?" --trust-processor-code
    python examples/lfm3_omni_generation.py --audio recording.wav --trust-processor-code
    python examples/lfm3_omni_generation.py --text "Hello world." --speak --trust-processor-code
    python examples/lfm3_omni_generation.py --audio question.wav --speak --trust-processor-code

The graph export reads configurations as data. The explicitly trusted, pinned
upstream processor supplies the image and audio frontends and the chat
template; speech uses the checkpoint's sampling settings.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
from PIL import Image
from transformers import AutoProcessor

from mobius.integrations.lfm3_audio import build_lfm3_audio_package, load_generation_config
from mobius.integrations.lfm3_omni import Lfm3OmniPipeline

MODEL_ID = "LiquidAI/LFM3-Omni-2B"
REVISION = "21d75123f86c1f7ffb4bef1ea202710c4e7ebe67"


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


def _conversation(args) -> tuple[list[dict], str]:
    """The model card's system prompt and user turn for each use, and the generation mode."""
    if args.speak and args.audio is None:
        system = "Perform TTS in American English using a female voice."
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": args.text},
        ], "audio"
    content = []
    if args.image is not None:
        content.append({"type": "image", "image": Image.open(args.image).convert("RGB")})
    if args.audio is not None:
        waveform, _ = librosa.load(args.audio, sr=16000)
        content.append({"type": "audio", "audio": waveform})
    if args.text is not None:
        content.append({"type": "text", "text": args.text})
    if args.speak:
        system = "Respond with interleaved text and audio."
    elif args.image is not None:
        system = "Answer questions about the provided image. Base every answer only on what can be seen."
    elif args.text is None:
        system = "Perform ASR in English."
    else:
        system = "Respond with text."
    mode = "interleaved" if args.speak else "text"
    return [{"role": "system", "content": system}, {"role": "user", "content": content}], mode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--text")
    parser.add_argument("--speak", action="store_true", help="Answer in speech.")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--export-dir", type=Path, default=Path("lfm3-omni-export"))
    parser.add_argument("--output-wav", type=Path, default=Path("speech.wav"))
    parser.add_argument("--trust-processor-code", action="store_true")
    args = parser.parse_args()
    if not args.trust_processor_code:
        parser.error(
            "Pass --trust-processor-code only after reviewing the pinned upstream processor."
        )
    if args.image is None and args.audio is None and args.text is None:
        parser.error("Give an --image, an --audio clip or a --text prompt.")
    provider = "CUDAExecutionProvider" if args.device == "cuda" else "CPUExecutionProvider"
    if provider not in ort.get_available_providers():
        parser.error(f"{provider} is unavailable; install the matching ONNX Runtime package.")
    package = build_lfm3_audio_package(
        MODEL_ID, revision=REVISION, execution_provider=args.device
    )
    package.save(args.export_dir)
    sessions = {
        name: _Session(args.export_dir / name / "model.onnx", provider) for name in package
    }
    pipeline = Lfm3OmniPipeline(
        package.config, sessions, load_generation_config(MODEL_ID, revision=REVISION)
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=True
    )
    conversation, mode = _conversation(args)
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="np",
        processor_kwargs={"sampling_rate": 16000},
    )
    feeds = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }
    if "input_features" in inputs:
        feeds["input_features"] = inputs["input_features"].astype(np.float32)
        feeds["feature_lengths"] = inputs["feature_attention_mask"].sum(-1).astype(np.int64)
    if "pixel_values" in inputs:
        feeds["pixel_values"] = inputs["pixel_values"].astype(np.float32)
        feeds["image_grid_thw"] = inputs["image_grid_thw"].astype(np.int64)
    tokens, codes = pipeline.generate(
        max_new_tokens=args.max_new_tokens, generation_mode=mode, **feeds
    )
    text = processor.decode(
        [token for token in tokens if token != package.config.audio_out_token_id],
        skip_special_tokens=True,
    )
    if text.strip():
        print(text)
    if codes.shape[-1]:
        sf.write(args.output_wav, pipeline.synthesize(codes)[0], 24000)
        print(args.output_wav)


if __name__ == "__main__":
    main()
