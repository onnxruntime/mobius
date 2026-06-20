# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Regenerate the Mimi codec golden reference used by the L4 parity test.

This produces ``testdata/golden/audio/mimi-personaplex.json`` by running the
*real* Kyutai Mimi codec (the ground-truth reference) on a deterministic,
in-code waveform. The waveform is reproduced byte-for-byte by the parity test,
so only a few KB of golden (exact codes + short decoded slices) is committed —
no audio file or ``.npz`` binary.

It must be run inside an environment that has the Kyutai ``moshi`` package
installed (it is **not** a mobius runtime dependency)::

    python -m venv /tmp/moshi-venv
    /tmp/moshi-venv/bin/pip install moshi
    /tmp/moshi-venv/bin/python scripts/generate_mimi_golden.py \
        --model nvidia/personaplex-7b-v1 \
        --out testdata/golden/audio/mimi-personaplex.json

The committed JSON stores exact integer codes ``(8, Tf)``, the decoded shape,
and 32-sample head/tail hex-float slices so the reference is self-describing
and auditable.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

N_FRAMES = 8
SAMPLE_RATE = 24000
SAMPLES_PER_FRAME = 1920


def make_input(n_frames: int = N_FRAMES, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Deterministic test waveform (must match the parity test exactly).

    A small mix of harmonics plus seeded low-amplitude noise exercises all
    eight residual codebooks while avoiding the degenerate argmin ties a pure
    tone would produce.
    """
    n = n_frames * SAMPLES_PER_FRAME
    t = np.arange(n, dtype=np.float64) / sr
    wav = (
        0.5 * np.sin(2 * np.pi * 220.0 * t)
        + 0.3 * np.sin(2 * np.pi * 440.0 * t)
        + 0.2 * np.sin(2 * np.pi * 660.0 * t)
        + 0.1 * np.sin(2 * np.pi * 1320.0 * t)
    )
    rng = np.random.RandomState(1234)
    wav = wav + 0.02 * rng.standard_normal(n)
    return wav.astype(np.float32)


def _resolve_mimi(model: str) -> str:
    """Resolve the Mimi ``tokenizer-*.safetensors`` file from a local HF cache."""
    if os.path.isfile(model):
        return model
    snaps = glob.glob(
        os.path.expanduser(
            f"~/.cache/huggingface/hub/models--{model.replace('/', '--')}/snapshots/*"
        )
    )
    if not snaps:
        raise FileNotFoundError(f"No local HF snapshot for {model!r}; download it first.")
    files = glob.glob(os.path.join(snaps[0], "tokenizer-*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No Mimi tokenizer checkpoint under {snaps[0]!r}")
    return files[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="nvidia/personaplex-7b-v1")
    parser.add_argument("--out", default="testdata/golden/audio/mimi-personaplex.json")
    args = parser.parse_args()

    from moshi.models.loaders import get_mimi

    mimi = get_mimi(_resolve_mimi(args.model), device="cpu")
    mimi.eval()

    wav = make_input()
    x = torch.from_numpy(wav)[None, None, :]  # (1, 1, T)
    with torch.no_grad():
        codes = mimi.encode(x)  # (1, 8, Tf)
        dec = mimi.decode(codes)  # (1, 1, T)

    codes_np = codes[0].cpu().numpy().astype(np.int64)
    dec_np = dec[0, 0].cpu().numpy().astype(np.float64)

    golden = {
        "description": (
            "Mimi codec (nvidia/personaplex-7b-v1) parity golden. Input is a "
            "deterministic harmonic + seeded-noise waveform (8 frames)."
        ),
        "sample_rate": SAMPLE_RATE,
        "n_frames": N_FRAMES,
        "num_codebooks": int(codes_np.shape[0]),
        "codes": codes_np.tolist(),
        "dec_shape": [int(d) for d in dec.shape],
        "dec_head_hex": [float(v).hex() for v in dec_np[:32]],
        "dec_tail_hex": [float(v).hex() for v in dec_np[-32:]],
    }

    with open(args.out, "w") as f:
        json.dump(golden, f, indent=2)
        f.write("\n")
    print(f"wrote {args.out} ({os.path.getsize(args.out)} bytes)")


if __name__ == "__main__":
    main()
