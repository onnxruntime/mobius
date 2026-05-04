# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SenseVoiceSmall speech recognition with ONNX models.

CTC-based encoder-only ASR with language control. Supports Chinese,
English, Japanese, Korean, and Cantonese, with Hakka fine-tuned variant.

Usage::

    python examples/sensevoice_small.py --audio recording.wav --language zh
    python examples/sensevoice_small.py --audio recording.wav --language hakka

The model uses the same LFR fbank frontend as Fun-ASR-Nano (7-frame stack,
stride 6, 80 mel bins → 560-dim). CTC decoding: argmax → collapse repeats
→ remove blank token (index 0).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from asr_utils import (
    LFR_M,
    N_MELS,
    SAMPLE_RATE,
    load_audio_file,
    load_cmvn,
    preprocess_audio,
)

from mobius._configs import ArchitectureConfig, AudioConfig
from mobius._testing.ort_inference import OnnxModelSession

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "justinchuby/SenseVoiceSmall-Hakka"

# Language ID mapping (from SenseVoiceSmall source)
LANGUAGE_MAP: dict[str, int] = {
    "auto": 0,
    "zh": 3,
    "chinese": 3,
    "en": 4,
    "english": 4,
    "yue": 7,
    "cantonese": 7,
    "ja": 11,
    "japanese": 11,
    "ko": 12,
    "korean": 12,
    "hakka": 3,  # Hakka model trained with zh language ID
}

# Textnorm query IDs
TEXTNORM_WOITN = 15  # without ITN (default)

# ---------------------------------------------------------------------------
# CTC Decoding
# ---------------------------------------------------------------------------


def ctc_greedy_decode(logits: np.ndarray, tokens: list[str], blank_id: int = 0) -> str:
    """CTC greedy decoding: argmax → collapse repeats → remove blanks.

    Args:
        logits: (T, vocab_size) CTC log-probabilities
        tokens: vocabulary list (index → token string)
        blank_id: blank token index (default 0 for SenseVoice)

    Returns:
        Decoded text string.
    """
    # Skip first 4 tokens (query positions: language, event, emo, textnorm)
    logits = logits[4:]
    pred_ids = np.argmax(logits, axis=-1)  # (T,)

    # Collapse consecutive duplicates
    collapsed = []
    prev = -1
    for idx in pred_ids:
        if idx != prev:
            collapsed.append(int(idx))
        prev = idx

    # Remove blanks and decode
    text_tokens = []
    for idx in collapsed:
        if idx == blank_id:
            continue
        if idx < len(tokens):
            text_tokens.append(tokens[idx])

    # Join tokens — sentencepiece uses ▁ as word boundary
    text = "".join(text_tokens).replace("\u2581", " ").strip()
    return text


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def build_sensevoice_config(model_id: str, dtype: str = "f32") -> ArchitectureConfig:
    """Build ArchitectureConfig from HF repo (config.json or config.yaml)."""
    from huggingface_hub import hf_hub_download

    # Try config.json first (mlx-community format), fall back to config.yaml
    try:
        cfg_path = hf_hub_download(model_id, "config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
    except Exception:  # config.json missing — try config.yaml (expected for FunASR repos)
        import yaml

        cfg_path = hf_hub_download(model_id, "config.yaml")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

    enc = cfg.get("encoder_conf", {})
    frontend = cfg.get("frontend_conf", {})

    lfr_m = frontend.get("lfr_m", LFR_M)
    n_mels = frontend.get("n_mels", N_MELS)
    input_size = cfg.get("input_size", lfr_m * n_mels)
    vocab_size = cfg.get("vocab_size", None)

    # If vocab_size not in config, try tokens.json
    if vocab_size is None:
        tok_path = hf_hub_download(model_id, "tokens.json")
        with open(tok_path) as f:
            tokens_list = json.load(f)
        vocab_size = len(tokens_list)

    from onnx_ir import DataType

    dtype_map = {"f32": DataType.FLOAT, "f16": DataType.FLOAT16, "bf16": DataType.BFLOAT16}

    return ArchitectureConfig(
        model_type="sensevoice_small",
        vocab_size=vocab_size,
        hidden_size=enc.get("output_size", 512),
        num_hidden_layers=1,  # Unused — encoder-only, validation requires > 0
        num_attention_heads=enc.get("attention_heads", 4),
        num_key_value_heads=enc.get("attention_heads", 4),
        intermediate_size=enc.get("linear_units", 2048),
        head_dim=enc.get("output_size", 512) // enc.get("attention_heads", 4),
        dtype=dtype_map.get(dtype, DataType.FLOAT),
        audio=AudioConfig(
            input_size=input_size,
            attention_dim=enc.get("output_size", 512),
            attention_heads=enc.get("attention_heads", 4),
            linear_units=enc.get("linear_units", 2048),
            kernel_size=enc.get("kernel_size", 11),
            num_blocks=enc.get("num_blocks", 50),
            tp_num_blocks=enc.get("tp_blocks", 20),
        ),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="SenseVoiceSmall speech recognition with ONNX models.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help="HuggingFace model ID (default: %(default)s).",
    )
    parser.add_argument("--audio", help="Path to audio file.")
    parser.add_argument(
        "--language",
        default="auto",
        help="Language: auto, zh, en, ja, ko, yue, hakka (default: auto).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Inference device (default: cpu).",
    )
    parser.add_argument(
        "--dtype",
        default="f32",
        choices=["f32", "f16", "bf16"],
        help="Model precision (default: f32).",
    )
    parser.add_argument(
        "--chunk-length",
        type=float,
        default=30.0,
        help="Audio chunk length in seconds for long files (default: 30).",
    )
    args = parser.parse_args()

    # Resolve language
    lang_key = args.language.lower().strip()
    if lang_key not in LANGUAGE_MAP:
        supported = ", ".join(sorted(set(LANGUAGE_MAP.keys()) - {"auto"}))
        parser.error(f"Unknown language {args.language!r}. Supported: auto, {supported}")
    language_id = LANGUAGE_MAP[lang_key]

    # Build config
    print(f"Building ONNX model from {args.model!r} (dtype={args.dtype}) ...")
    config = build_sensevoice_config(args.model, dtype=args.dtype)

    # Build ONNX model
    from mobius import build_from_module
    from mobius.models.sensevoice_small import SenseVoiceSmallModel
    from mobius.tasks import AudioCTCTask

    module = SenseVoiceSmallModel(config)
    pkg = build_from_module(module, config, task=AudioCTCTask())

    # Load weights
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    weights_path = hf_hub_download(args.model, "model.safetensors")
    state_dict = load_file(weights_path)
    state_dict = module.preprocess_weights(state_dict)
    pkg.apply_weights(state_dict)

    # Create ORT session
    print(f"Creating inference session (device={args.device}) ...")
    session = OnnxModelSession(pkg["model"], device=args.device)

    # Load tokens vocabulary
    tokens = None
    try:
        tok_path = hf_hub_download(args.model, "tokens.json")
        with open(tok_path) as f:
            tokens = json.load(f)
    except Exception:  # tokens.json not available — fall back to sentencepiece
        pass  # tokens remains None, handled below

    if tokens is None:
        # Fall back to sentencepiece BPE model
        try:
            import sentencepiece as spm

            bpe_path = hf_hub_download(args.model, "chn_jpn_yue_eng_ko_spectok.bpe.model")
            sp = spm.SentencePieceProcessor(model_file=bpe_path)
            tokens = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
            # Pad to vocab_size if needed
            while len(tokens) < config.vocab_size:
                tokens.append(f"<extra_{len(tokens)}>")
        except Exception as e:
            print(f"Warning: Could not load tokens ({e})")
            tokens = [f"<id_{i}>" for i in range(config.vocab_size)]

    # Load CMVN stats
    try:
        cmvn_path = hf_hub_download(args.model, "am.mvn")
        cmvn = load_cmvn(cmvn_path)
        print("CMVN: loaded from am.mvn")
    except Exception:  # am.mvn not available — skip CMVN (identity normalization)
        cmvn = None
        print("CMVN: not available (skipping)")

    print("Ready.\n")
    lang_name = {0: "auto", 3: "zh", 4: "en", 7: "yue", 11: "ja", 12: "ko"}.get(
        language_id, str(language_id)
    )
    print(f"Language: {lang_name} (id={language_id})")

    if not args.audio:
        print("No --audio specified. Pass --audio <file> to transcribe.")
        return

    print(f"Loading audio: {args.audio}")
    audio = load_audio_file(args.audio)
    print(f"Audio: {len(audio) / SAMPLE_RATE:.1f}s\n")

    language_ids = np.array([[language_id]], dtype=np.int64)

    def transcribe_chunk(chunk_audio: np.ndarray) -> str:
        """Transcribe a single audio chunk."""
        input_features = preprocess_audio(chunk_audio, cmvn=cmvn).astype(np.float32)
        out = session.run({"input_features": input_features, "language_id": language_ids})
        logits = out["logits"][0]  # (T, vocab)
        return ctc_greedy_decode(logits, tokens, blank_id=0)

    # Chunked transcription for long audio
    samples_per_chunk = int(args.chunk_length * SAMPLE_RATE)
    total_samples = len(audio)

    if total_samples <= samples_per_chunk:
        text = transcribe_chunk(audio)
    else:
        num_chunks = (total_samples + samples_per_chunk - 1) // samples_per_chunk
        results = []
        for i in range(num_chunks):
            start = i * samples_per_chunk
            end = min(start + samples_per_chunk, total_samples)
            chunk = audio[start:end]
            if len(chunk) < SAMPLE_RATE * 0.3:
                continue
            print(f"[Chunk {i + 1}/{num_chunks}] ", end="", flush=True)
            chunk_text = transcribe_chunk(chunk)
            print(chunk_text)
            results.append(chunk_text.strip())
        text = " ".join(results)

    print(f"\U0001f4dd Result: {text}")


if __name__ == "__main__":
    main()
