# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Integration parity test for the Moshi LM vs the Kyutai reference.

Builds the Moshi temporal transformer and depformer ONNX graphs from the real
native checkpoint (``nvidia/personaplex-7b-v1``) and compares against a
committed JSON golden (``testdata/golden/audio/moshi-lm-personaplex.json``)
generated from the Kyutai ``moshi`` reference (see
``scripts/generate_moshi_lm_golden.py``).

The golden stores hidden-state slices, text-stream argmax, and per-substep
depformer argmax for a deterministic in-code token frame sequence, so the
comparison is tight (``atol`` on slices) and exact (argmax). The depformer is
stepped autoregressively for all 16 substeps with teacher-forced tokens.

These tests are CPU-only: H200 / Ampere+ GPUs default to TF32 for fp32 matmul,
which can flip the text argmax; the ORT example sets ``use_tf32=0`` on CUDA.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import onnxruntime as ort
import pytest

pytestmark = pytest.mark.integration

_MODEL_ID = "nvidia/personaplex-7b-v1"
_GOLDEN = os.path.join(
    os.path.dirname(__file__),
    "..",
    "testdata",
    "golden",
    "audio",
    "moshi-lm-personaplex.json",
)

_N_CH = 17
_N_STEPS = 4
_DEP_Q = 16
_AUDIO_CARD = 2048
_TEXT_CARD = 32000
# Temporal transformer / depformer head geometry (personaplex-7b-v1).
_T_LAYERS, _T_HEADS, _T_HEAD_DIM = 32, 32, 128
_D_LAYERS, _D_HEADS, _D_HEAD_DIM = 6, 16, 64


def _make_frames() -> np.ndarray:
    """Deterministic frame, identical to ``scripts/generate_moshi_lm_golden.py``."""
    rng = np.random.RandomState(20240607)
    frames = np.zeros((1, _N_CH, _N_STEPS), dtype=np.int64)
    frames[0, 0, :] = rng.randint(0, _TEXT_CARD, size=_N_STEPS)
    frames[0, 1:, :] = rng.randint(0, _AUDIO_CARD, size=(_N_CH - 1, _N_STEPS))
    return frames


def _make_prev_tokens() -> list[int]:
    """Deterministic teacher-forced depformer tokens (matches the golden script)."""
    rng = np.random.RandomState(7)
    text_prev = int(rng.randint(0, _TEXT_CARD))
    audio_prev = rng.randint(0, _AUDIO_CARD, size=_DEP_Q - 1).tolist()
    return [text_prev, *audio_prev]


def _find_onnx(root: str) -> str:
    for dirpath, _, files in os.walk(root):
        for name in files:
            if name.endswith(".onnx"):
                return os.path.join(dirpath, name)
    raise FileNotFoundError(root)


@pytest.mark.integration_slow
def test_moshi_lm_parity():
    from mobius.integrations.moshi._builder import (
        _PERSONAPLEX_REVISION,
        _build_moshi_lm,
    )

    with open(_GOLDEN) as f:
        golden = json.load(f)
    g_temporal = golden["temporal"]
    g_dep = golden["depformer"]

    frames = _make_frames()
    prev_tokens = _make_prev_tokens()

    pkg = _build_moshi_lm(_MODEL_ID, revision=_PERSONAPLEX_REVISION)
    assert set(pkg) == {"temporal", "depformer"}

    with tempfile.TemporaryDirectory() as td:
        pkg.save(td)
        tdir = os.path.join(td, "temporal")
        ddir = os.path.join(td, "depformer")

        # --- Temporal transformer: 17-channel frame -> hidden + text_logits. ---
        tsess = ort.InferenceSession(_find_onnx(tdir), providers=["CPUExecutionProvider"])
        feeds = {
            "input_frame": frames,
            "attention_mask": np.ones((1, _N_STEPS), np.int64),
            "position_ids": np.arange(_N_STEPS, dtype=np.int64)[None],
        }
        for i in range(_T_LAYERS):
            feeds[f"past_key_values.{i}.key"] = np.zeros(
                (1, _T_HEADS, 0, _T_HEAD_DIM), np.float32
            )
            feeds[f"past_key_values.{i}.value"] = np.zeros(
                (1, _T_HEADS, 0, _T_HEAD_DIM), np.float32
            )
        hidden, text_logits = tsess.run(["hidden", "text_logits"], feeds)
        hidden, text_logits = hidden[0], text_logits[0]

        g_head = np.array([float.fromhex(x) for x in g_temporal["hidden_last_head_hex"]])
        g_tail = np.array([float.fromhex(x) for x in g_temporal["hidden_last_tail_hex"]])
        np.testing.assert_allclose(hidden[-1, :32], g_head, atol=2e-4)
        np.testing.assert_allclose(hidden[-1, -32:], g_tail, atol=2e-4)
        text_argmax = [int(np.argmax(text_logits[s])) for s in range(_N_STEPS)]
        assert text_argmax == g_temporal["text_argmax"]

        # --- Depformer: 16 autoregressive substeps, teacher-forced tokens. ---
        dsess = ort.InferenceSession(_find_onnx(ddir), providers=["CPUExecutionProvider"])
        last_hidden = hidden[None, -1:].astype(np.float32)  # (1, 1, 4096)
        past = [
            (
                np.zeros((1, _D_HEADS, 0, _D_HEAD_DIM), np.float32),
                np.zeros((1, _D_HEADS, 0, _D_HEAD_DIM), np.float32),
            )
            for _ in range(_D_LAYERS)
        ]
        out_names = [o.name for o in dsess.get_outputs()]
        dep_argmax = []
        head0 = None
        for cb in range(_DEP_Q):
            dfeeds = {
                "hidden": last_hidden,
                "prev_token": np.array([[prev_tokens[cb]]], np.int64),
                "substep_index": np.array(cb, np.int64),
            }
            for i in range(_D_LAYERS):
                dfeeds[f"past_key_values.{i}.key"] = past[i][0]
                dfeeds[f"past_key_values.{i}.value"] = past[i][1]
            outs = dsess.run(None, dfeeds)
            named = dict(zip(out_names, outs))
            logits = named["logits"][0, 0]
            past = [
                (named[f"present.{i}.key"], named[f"present.{i}.value"])
                for i in range(_D_LAYERS)
            ]
            dep_argmax.append(int(np.argmax(logits)))
            if cb == 0:
                head0 = logits[:32].copy()

        g_dep_head = np.array([float.fromhex(x) for x in g_dep["logits0_head_hex"]])
        np.testing.assert_allclose(head0, g_dep_head, atol=2e-4)
        assert dep_argmax == g_dep["logits_argmax"]
