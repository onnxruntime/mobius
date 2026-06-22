# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Integration parity test for the Mimi codec vs the Kyutai reference.

Builds the Mimi encoder/decoder ONNX graphs from the real native checkpoint
(``nvidia/personaplex-7b-v1``) and compares against a committed JSON golden
(``testdata/golden/audio/mimi-personaplex.json``) generated from the Kyutai
``moshi`` reference (see ``scripts/generate_mimi_golden.py``).

The golden stores exact integer codes plus short decoded waveform slices for a
deterministic in-code waveform, so the comparison is exact for codes and tight
(``atol=1e-4``) for the decoded audio.
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
    "mimi-personaplex.json",
)

_N_FRAMES = 8
_SAMPLE_RATE = 24000
_SAMPLES_PER_FRAME = 1920


def _make_input(n_frames: int = _N_FRAMES, sr: int = _SAMPLE_RATE) -> np.ndarray:
    """Deterministic waveform identical to ``scripts/generate_mimi_golden.py``."""
    n = n_frames * _SAMPLES_PER_FRAME
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


def _find_onnx(root: str, sub: str) -> str:
    for dirpath, _, files in os.walk(root):
        for name in files:
            if name.endswith(".onnx") and os.path.basename(dirpath) == sub:
                return os.path.join(dirpath, name)
    raise FileNotFoundError(sub)


def _run(path: str, feeds: dict):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return sess.run(None, feeds)


@pytest.mark.integration_slow
def test_mimi_codec_parity():
    from mobius.integrations.moshi import build_mimi

    with open(_GOLDEN) as f:
        golden = json.load(f)
    gold_codes = np.array(golden["codes"], dtype=np.int64)
    gold_head = np.array([float.fromhex(x) for x in golden["dec_head_hex"]])
    gold_tail = np.array([float.fromhex(x) for x in golden["dec_tail_hex"]])

    pkg = build_mimi(_MODEL_ID)
    assert set(pkg.keys()) == {"encoder", "decoder"}

    with tempfile.TemporaryDirectory() as td:
        pkg.save(td)

        # Encoder: waveform (B, 1, T) -> codes (B, 8, Tf). Exact integer match.
        wav = _make_input()[None, None, :]
        codes = _run(_find_onnx(td, "encoder"), {"waveform": wav})[0][0]
        np.testing.assert_array_equal(codes, gold_codes)

        # Decoder: codes (B, 8, Tf) -> waveform (B, 1, T). Compare slices.
        dec = _run(
            _find_onnx(td, "decoder"),
            {"codes": gold_codes[None].astype(np.int64)},
        )[0][0, 0]
        np.testing.assert_allclose(dec[:32], gold_head, atol=1e-4)
        np.testing.assert_allclose(dec[-32:], gold_tail, atol=1e-4)
