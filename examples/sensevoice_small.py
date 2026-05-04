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

import numpy as np

from mobius._configs import ArchitectureConfig, AudioConfig
from mobius._testing.ort_inference import OnnxModelSession

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "justinchuby/SenseVoiceSmall-Hakka"
SAMPLE_RATE = 16000
LFR_M = 7  # LFR stack factor
LFR_N = 6  # LFR stride
N_MELS = 80

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
# Audio frontend: fbank → LFR (same as Fun-ASR)
# ---------------------------------------------------------------------------


def compute_fbank(
    waveform: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
) -> np.ndarray:
    """Compute log-mel filterbank features using torchaudio."""
    import torch
    import torchaudio

    wav = torch.from_numpy(waveform).float().unsqueeze(0)
    fbank = torchaudio.compliance.kaldi.fbank(
        wav,
        num_mel_bins=n_mels,
        sample_frequency=sample_rate,
        window_type="hamming",
        frame_length=25.0,
        frame_shift=10.0,
        dither=0.0,
    )
    return fbank.numpy()  # (T, n_mels)


def apply_lfr(fbank: np.ndarray, lfr_m: int = LFR_M, lfr_n: int = LFR_N) -> np.ndarray:
    """Apply Low Frame Rate (LFR) stacking: stack lfr_m frames, stride lfr_n."""
    t, _d = fbank.shape
    num_lfr = (t + lfr_n - 1) // lfr_n
    # Pad to multiple of lfr_n
    pad_len = num_lfr * lfr_n + (lfr_m - lfr_n) - t
    if pad_len > 0:
        fbank = np.pad(fbank, ((0, pad_len), (0, 0)), mode="edge")
    lfr_frames = []
    for i in range(num_lfr):
        start = i * lfr_n
        lfr_frames.append(fbank[start : start + lfr_m].reshape(-1))
    return np.stack(lfr_frames, axis=0)  # (T_lfr, lfr_m * d)


def load_cmvn(cmvn_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load CMVN stats from Kaldi am.mvn file.

    Returns (means, vars) arrays of shape (560,) each.
    CMVN is applied as: features = (features + means) * vars
    """
    with open(cmvn_path) as f:
        lines = f.readlines()
    means = None
    variances = None
    for i, line in enumerate(lines):
        parts = line.split()
        if parts[0] == "<AddShift>":
            next_parts = lines[i + 1].split()
            if next_parts[0] == "<LearnRateCoef>":
                means = np.array(next_parts[3:-1], dtype=np.float32)
        elif parts[0] == "<Rescale>":
            next_parts = lines[i + 1].split()
            if next_parts[0] == "<LearnRateCoef>":
                variances = np.array(next_parts[3:-1], dtype=np.float32)
    assert means is not None and variances is not None, "Failed to parse am.mvn"
    return means, variances


def preprocess_audio(
    waveform: np.ndarray,
    cmvn: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Full frontend: fbank → LFR → CMVN → (1, T, 560)."""
    fbank = compute_fbank(waveform)
    lfr = apply_lfr(fbank)
    if cmvn is not None:
        means, variances = cmvn
        lfr = (lfr + means) * variances
    return lfr[np.newaxis, :, :]  # (1, T, 560)


def load_audio_file(path: str) -> np.ndarray:
    """Load audio file and resample to 16kHz mono."""
    import torchaudio

    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    return waveform.squeeze(0).numpy()


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
    """Build ArchitectureConfig from HF repo config.yaml."""
    import yaml
    from huggingface_hub import hf_hub_download

    yaml_path = hf_hub_download(model_id, "config.yaml")
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    enc = cfg.get("encoder_conf", {})
    frontend = cfg.get("frontend_conf", {})

    lfr_m = frontend.get("lfr_m", LFR_M)
    n_mels = frontend.get("n_mels", N_MELS)
    input_size = lfr_m * n_mels

    # Load tokens.json for vocab_size
    tok_path = hf_hub_download(model_id, "tokens.json")
    with open(tok_path) as f:
        tokens = json.load(f)
    vocab_size = len(tokens)

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

    # Load tokens
    tok_path = hf_hub_download(args.model, "tokens.json")
    with open(tok_path) as f:
        tokens = json.load(f)

    # Load CMVN stats
    try:
        cmvn_path = hf_hub_download(args.model, "am.mvn")
        cmvn = load_cmvn(cmvn_path)
        print("CMVN: loaded from am.mvn")
    except Exception:
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
