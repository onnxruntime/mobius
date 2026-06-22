# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Regenerate the Moshi LM golden reference used by the L4 parity test.

Produces ``testdata/golden/audio/moshi-lm-personaplex.json`` by running the
*real* Kyutai Moshi LM (the ground-truth reference) on a deterministic,
in-code token frame sequence. The inputs are reproduced byte-for-byte by the
parity test, so only a few KB of golden (hidden slices, text/depformer argmax,
and short logit slices) is committed -- no model weights or binary blobs.

It must be run inside an environment that has the Kyutai ``moshi`` package
installed (it is **not** a mobius runtime dependency)::

    python -m venv /tmp/moshi-venv
    /tmp/moshi-venv/bin/pip install moshi
    /tmp/moshi-venv/bin/python scripts/generate_moshi_lm_golden.py \
        --model nvidia/personaplex-7b-v1 \
        --out testdata/golden/audio/moshi-lm-personaplex.json

The golden captures both stacks: the temporal transformer (hidden + text
logits for a short frame sequence) and the depformer (per-substep audio
codebook logits, teacher-forced).
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

N_CH = 17
N_STEPS = 4
DEP_Q = 16
AUDIO_CARD = 2048
TEXT_CARD = 32000


def make_frames() -> np.ndarray:
    """Deterministic (1, 17, N_STEPS) frame. ch0=text, ch1..16=audio codebooks."""
    rng = np.random.RandomState(20240607)
    frames = np.zeros((1, N_CH, N_STEPS), dtype=np.int64)
    frames[0, 0, :] = rng.randint(0, TEXT_CARD, size=N_STEPS)
    frames[0, 1:, :] = rng.randint(0, AUDIO_CARD, size=(N_CH - 1, N_STEPS))
    return frames


def make_prev_tokens() -> list[int]:
    """Deterministic teacher-forced previous tokens for the 16 depformer substeps.

    Substep 0's previous token is a text token; substeps 1..15 use audio tokens.
    """
    rng = np.random.RandomState(7)
    text_prev = int(rng.randint(0, TEXT_CARD))
    audio_prev = rng.randint(0, AUDIO_CARD, size=DEP_Q - 1).tolist()
    return [text_prev, *audio_prev]


def _resolve_lm(model: str) -> str:
    if os.path.isfile(model):
        return model
    snaps = glob.glob(
        os.path.expanduser(
            f"~/.cache/huggingface/hub/models--{model.replace('/', '--')}/snapshots/*"
        )
    )
    if snaps:
        cand = os.path.join(snaps[0], "model.safetensors")
        if os.path.isfile(cand):
            return cand
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="nvidia/personaplex-7b-v1")
    parser.add_argument("--out", default="testdata/golden/audio/moshi-lm-personaplex.json")
    args = parser.parse_args()

    from moshi.models.loaders import get_moshi_lm

    lm_path = _resolve_lm(args.model)
    lm = get_moshi_lm(lm_path, device="cpu", dtype=torch.float32)
    lm.eval()

    frames = make_frames()
    prev_tokens = make_prev_tokens()
    seq = torch.from_numpy(frames)

    with torch.no_grad():
        transformer_out, text_logits = lm.forward_codes(seq)
    hid = transformer_out[0].cpu().numpy().astype(np.float64)  # (S, 4096)
    tl = text_logits[0, 0].cpu().numpy().astype(np.float64)  # (S, 32000)
    last_hidden = transformer_out[:, -1:]

    dep_argmax: list[int] = []
    dep_max_hex: list[str] = []
    dep0_head_hex: list[str] = []
    with torch.no_grad(), lm.depformer.streaming(1):
        for cb in range(DEP_Q):
            prev = torch.tensor([[[prev_tokens[cb]]]], dtype=torch.long)
            logits = lm.forward_depformer(cb, prev, last_hidden)
            lg = logits[0, 0, 0].cpu().numpy().astype(np.float64)
            dep_argmax.append(int(np.argmax(lg)))
            dep_max_hex.append(float(np.max(lg)).hex())
            if cb == 0:
                dep0_head_hex = [float(v).hex() for v in lg[:32]]

    golden = {
        "description": (
            "Moshi LM (nvidia/personaplex-7b-v1) parity golden. Temporal "
            "transformer + depformer outputs on a deterministic 17-channel "
            "frame sequence (4 steps) with teacher-forced depformer tokens."
        ),
        "n_channels": N_CH,
        "n_steps": N_STEPS,
        "dep_q": DEP_Q,
        "audio_card": AUDIO_CARD,
        "text_card": TEXT_CARD,
        "temporal": {
            "hidden_shape": list(hid.shape),
            "hidden_last_head_hex": [float(v).hex() for v in hid[-1, :32]],
            "hidden_last_tail_hex": [float(v).hex() for v in hid[-1, -32:]],
            "text_argmax": [int(np.argmax(tl[s])) for s in range(N_STEPS)],
            "text_max_hex": [float(np.max(tl[s])).hex() for s in range(N_STEPS)],
        },
        "depformer": {
            "logits_argmax": dep_argmax,
            "logits_max_hex": dep_max_hex,
            "logits0_head_hex": dep0_head_hex,
        },
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(golden, f, indent=2)
        f.write("\n")
    print("temporal text_argmax:", golden["temporal"]["text_argmax"])
    print("depformer argmax:", dep_argmax)
    print("wrote", args.out, os.path.getsize(args.out), "bytes")


if __name__ == "__main__":
    main()
