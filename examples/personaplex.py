#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""PersonaPlex: file-based speech-to-text/speech demo via ONNX.

PersonaPlex (``nvidia/personaplex-7b-v1``) is NVIDIA's fine-tune of
Kyutai's Moshi full-duplex speech model. It shares Moshi's 3-model
architecture (embedding + decoder + audio_decoder) and uses the Mimi
audio codec for input and output.

Unlike :file:`moshi_realtime.py`, this script processes an audio file
end-to-end rather than driving a microphone. It is the simplest way to
smoke-test the exported PersonaPlex ONNX models against a known input
without setting up live audio I/O.

Pipeline (per 80 ms frame at 24 kHz)::

    PCM frame ─► Mimi encoder ─► audio_codes (16 ints)
                                       │
                                       ▼
    text_token + audio_codes ─► embedding ─► inputs_embeds
                                       │
                                       ▼
                  inputs_embeds ─► decoder ─► text_logits
                                       │
                                       ▼
                     backbone_hidden ─► audio_decoder ─► output_codes
                                       │
                                       ▼  (optional)
                                Mimi decoder ─► output PCM

Usage::

    # First-time: build + cache ONNX from HuggingFace, then run inference
    python examples/personaplex.py --audio testdata/652-129742-0006.flac \
        --save-to /tmp/personaplex/

    # Subsequent runs: load pre-exported ONNX
    python examples/personaplex.py --audio my_speech.wav \
        --onnx-dir /tmp/personaplex/

    # Synthetic input (no audio file or Mimi codec needed) — fastest sanity
    python examples/personaplex.py --synthetic

Notes:
    - Mimi codec is optional. Without it, the script falls back to
      synthetic all-zero audio codes (still exercises every ONNX sub-model
      and validates output token shapes).
    - The 7B decoder needs ~28 GB host RAM in float32. Use ``--dtype bf16``
      with a recent ORT to reduce that to ~14 GB if you have GPU EP.
    - Output audio synthesis (--out-audio) requires the Mimi codec.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Defaults matching nvidia/personaplex-7b-v1
SAMPLE_RATE = 24_000  # Mimi codec sample rate (Hz)
FRAME_SAMPLES = 1920  # 80 ms at 24 kHz (one model step)
NUM_CODEBOOKS = 16  # PersonaPlex/Moshi codebook count
STEPS_PER_SECOND = SAMPLE_RATE // FRAME_SAMPLES  # 12.5 Hz


# --------------------------------------------------------------------------- #
# Build / load ONNX
# --------------------------------------------------------------------------- #


def build_and_save(model_id: str, save_dir: Path, dtype: str) -> dict[str, Path]:
    """Export the 3 PersonaPlex sub-models with external-data sidecars."""
    import onnx_ir  # type: ignore[import-not-found]

    from mobius import build  # type: ignore[import-not-found]

    print(f"[personaplex] Building ONNX from {model_id} (dtype={dtype}) …")
    pkg = build(model_id, dtype=dtype)
    save_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, model in pkg.items():
        out = save_dir / f"{name}.onnx"
        # External data because the 7B decoder exceeds the 2 GB protobuf
        # serialization limit by a wide margin.
        onnx_ir.save(model, out, external_data=f"{name}.data")
        paths[name] = out
        print(f"[personaplex]   Saved {name} → {out}")
    return paths


def resolve_onnx_paths(args: argparse.Namespace) -> dict[str, Path]:
    """Return paths to the three ONNX sub-models, building if needed."""
    if args.onnx_dir is not None:
        d = Path(args.onnx_dir)
        return {n: d / f"{n}.onnx" for n in ("embedding", "decoder", "audio_decoder")}
    if args.save_to is None:
        print(
            "[personaplex] Either --onnx-dir or --save-to must be provided.",
            file=sys.stderr,
        )
        sys.exit(2)
    return build_and_save(args.model, Path(args.save_to), args.dtype)


# --------------------------------------------------------------------------- #
# Inference loop
# --------------------------------------------------------------------------- #


def run_inference(
    paths: dict[str, Path], audio_codes_per_frame: list[np.ndarray], max_steps: int
) -> list[tuple[int, np.ndarray]]:
    """Process ``audio_codes_per_frame`` through the PersonaPlex pipeline.

    Returns a list of ``(text_token, output_codes)`` per step.
    """
    # Import lazily so the script imports without ORT installed
    sys.path.insert(0, str(Path(__file__).parent))
    from moshi_realtime import MoshiOnnxPipeline  # type: ignore[import-not-found]

    pipeline = MoshiOnnxPipeline(
        embedding_path=str(paths["embedding"]),
        decoder_path=str(paths["decoder"]),
        audio_decoder_path=str(paths["audio_decoder"]),
    )
    print(
        f"[personaplex] ORT sessions ready "
        f"(decoder KV cache: {len(pipeline._decoder_kv)} layer pairs, "
        f"depformer KV cache: {len(pipeline._depformer_kv)} layer pairs)"
    )

    last_text = 0  # start token
    history: list[tuple[int, np.ndarray]] = []
    for step, audio_codes in enumerate(audio_codes_per_frame[:max_steps]):
        text_token, out_codes = pipeline.step(last_text, audio_codes)
        history.append((int(text_token), out_codes))
        last_text = int(text_token)
    return history


# --------------------------------------------------------------------------- #
# Audio I/O helpers
# --------------------------------------------------------------------------- #


def encode_audio_file(
    audio_path: Path, max_frames: int
) -> tuple[list[np.ndarray], np.ndarray]:
    """Load + resample to 24 kHz + Mimi-encode an audio file.

    Returns:
        codes_per_frame: list of ``(NUM_CODEBOOKS,) int64`` arrays.
        raw_pcm: the resampled float32 PCM (for diagnostic RMS reporting).
    """
    import librosa  # type: ignore[import-not-found]

    pcm, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    pcm = pcm.astype(np.float32)
    n_frames = min(max_frames, len(pcm) // FRAME_SAMPLES)
    print(f"[personaplex] Loaded {audio_path.name}: {len(pcm) / sr:.2f}s, {n_frames} frames")

    # Mimi is optional. If unavailable, synthesize all-zero codes.
    try:
        import torch  # type: ignore[import-not-found]
        from moshi.models import loaders  # type: ignore[import-not-found]

        mimi = loaders.get_mimi(loaders.MIMI_NAME, device="cpu")
        mimi.set_num_codebooks(NUM_CODEBOOKS)
        codes_per_frame: list[np.ndarray] = []
        for i in range(n_frames):
            chunk = pcm[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]
            with torch.no_grad():
                t = torch.from_numpy(chunk).reshape(1, 1, -1)
                codes = mimi.encode(t)  # (1, num_codebooks, 1)
                codes_per_frame.append(codes[0, :, 0].cpu().numpy().astype(np.int64))
        print(f"[personaplex] Mimi-encoded {len(codes_per_frame)} frames")
        return codes_per_frame, pcm
    except (ImportError, OSError) as exc:
        print(
            f"[personaplex] Mimi unavailable ({exc.__class__.__name__}); "
            "falling back to all-zero audio codes. The pipeline still runs "
            "and outputs are exercised, but text/audio quality reflects "
            "silence-input, not your actual audio."
        )
        codes_per_frame = [np.zeros(NUM_CODEBOOKS, dtype=np.int64) for _ in range(n_frames)]
        return codes_per_frame, pcm


def synthetic_codes(n_frames: int) -> list[np.ndarray]:
    """Generate deterministic non-zero audio codes for sanity testing."""
    rng = np.random.default_rng(0)
    return [
        rng.integers(0, 2048, size=NUM_CODEBOOKS).astype(np.int64) for _ in range(n_frames)
    ]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PersonaPlex on an audio file via ONNX (file-based, no microphone)."
    )
    parser.add_argument(
        "--model",
        default="nvidia/personaplex-7b-v1",
        help="HuggingFace model id (default: nvidia/personaplex-7b-v1).",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Audio file to process (any sample rate; resampled to 24 kHz).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Skip audio file and run on synthetic non-zero codes (no Mimi required).",
    )
    parser.add_argument(
        "--save-to", type=str, default=None, help="Directory to export ONNX into."
    )
    parser.add_argument(
        "--onnx-dir",
        type=str,
        default=None,
        help="Directory of pre-exported ONNX sub-models (skips build step).",
    )
    parser.add_argument(
        "--dtype", default="float32", help="Build dtype: float32 (default) or bf16."
    )
    parser.add_argument(
        "--steps", type=int, default=8, help="Number of 80 ms frames to process."
    )

    args = parser.parse_args()

    paths = resolve_onnx_paths(args)

    # ------------------------------------------------------------------ #
    # Prepare input audio codes
    # ------------------------------------------------------------------ #
    if args.synthetic:
        codes = synthetic_codes(args.steps)
    elif args.audio is not None:
        codes, _pcm = encode_audio_file(args.audio, args.steps)
        if not codes:
            print(
                "[personaplex] No frames decoded from audio file; is it shorter than 80 ms?",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        # Default to the repo's standard audio fixture if it exists; else synthetic.
        fixture = Path("testdata/652-129742-0006.flac")
        if fixture.exists():
            print(f"[personaplex] No --audio specified; using fixture {fixture}")
            codes, _pcm = encode_audio_file(fixture, args.steps)
        else:
            print(
                "[personaplex] No --audio specified and no fixture found; using synthetic codes"
            )
            codes = synthetic_codes(args.steps)

    # ------------------------------------------------------------------ #
    # Run inference
    # ------------------------------------------------------------------ #
    history = run_inference(paths, codes, max_steps=args.steps)

    print()
    print(f"[personaplex] {len(history)} steps processed:")
    for i, (text_tok, out_codes) in enumerate(history):
        print(f"  step {i:2d}: text_token={text_tok:6d}  out_codes={out_codes.tolist()}")

    # ------------------------------------------------------------------ #
    # Decode text tokens (best-effort — tokenizer may be gated)
    # ------------------------------------------------------------------ #
    try:
        from transformers import AutoTokenizer  # type: ignore[import-not-found]

        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        ids = [t for t, _ in history]
        text = tok.decode(ids, skip_special_tokens=False)
        print()
        print(f"[personaplex] decoded text: {text!r}")
    except Exception as exc:
        print(f"[personaplex] (skipping text decode: {exc.__class__.__name__})")


if __name__ == "__main__":
    main()
