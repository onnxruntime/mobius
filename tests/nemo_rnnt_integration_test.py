# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Integration parity test for the FastConformer-RNNT model vs NeMo.

Builds the three ONNX sub-models from the real ``.nemo`` archive
(``nvidia/nemotron-speech-streaming-en-0.6b``, resolved from HuggingFace Hub)
and compares encoder / prediction / joint outputs against a pre-computed NeMo
reference (``testdata/golden/speech/nemotron_fastconformer_rnnt.npz``).

The reference was generated with ``nemo_toolkit`` (see the session script
``gen_nemo_golden.py``); regenerate it if the architecture changes.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import onnxruntime as ort
import pytest

pytestmark = pytest.mark.integration

_MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b"
# Pin the HF revision so the test always validates against the exact model the
# committed golden was generated from (see scripts/generate_nemo_rnnt_golden.py).
_REVISION = "7a9b763e6c5fb103da690219c049fac917aa50b1"
_GOLDEN = os.path.join(
    os.path.dirname(__file__),
    "..",
    "testdata",
    "golden",
    "speech",
    "nemotron_fastconformer_rnnt.npz",
)
_STREAMING_GOLDEN = os.path.join(
    os.path.dirname(__file__),
    "..",
    "testdata",
    "golden",
    "speech",
    "nemotron_fastconformer_rnnt_streaming.npz",
)
_SOS_ID = 1025  # rnnt_num_classes (1024) + 1; zero start-of-sequence embedding


def _find_onnx(root: str, sub: str) -> str:
    for dirpath, _, files in os.walk(root):
        for name in files:
            if name.endswith(".onnx") and os.path.basename(dirpath) == sub:
                return os.path.join(dirpath, name)
    raise FileNotFoundError(sub)


def _run(path: str, feeds: dict):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return sess.run(None, feeds)


def _full_length(feats: np.ndarray) -> np.ndarray:
    """Per-sample length covering all feature frames (no padding)."""
    return np.full((feats.shape[0],), feats.shape[2], dtype=np.int64)


@pytest.mark.integration_slow
@pytest.mark.skipif(
    "CUDAExecutionProvider" not in ort.get_available_providers(),
    reason="half-precision FastConformer kernels require CUDA",
)
def test_fastconformer_rnnt_fp16_encoder_parity():
    """The f16 encoder (built via dtype='f16') matches the f32 golden on CUDA."""
    from mobius.integrations.nemo import build_from_nemo

    golden = np.load(_GOLDEN)
    pkg = build_from_nemo(_MODEL_ID, revision=_REVISION, dtype="f16")

    with tempfile.TemporaryDirectory() as td:
        pkg.save(td)
        sess = ort.InferenceSession(
            _find_onnx(td, "encoder"), providers=["CUDAExecutionProvider"]
        )
        feats16 = golden["feats"].astype(np.float16)
        (enc, _enc_len) = sess.run(
            None, {"audio_signal": feats16, "length": _full_length(feats16)}
        )
        # f16 accumulates more rounding error than f32; allow a looser tolerance.
        np.testing.assert_allclose(enc.astype(np.float32), golden["enc_out"], atol=5e-2)


@pytest.mark.integration_slow
def test_fastconformer_rnnt_parity():
    from mobius.integrations.nemo import build_from_nemo

    golden = np.load(_GOLDEN)
    pkg = build_from_nemo(_MODEL_ID, revision=_REVISION)
    assert set(pkg.keys()) == {"encoder", "encoder_streaming", "decoder", "joint"}

    with tempfile.TemporaryDirectory() as td:
        pkg.save(td)

        # Encoder: mel features -> encoded frames
        feats = golden["feats"].astype(np.float32)
        enc = _run(
            _find_onnx(td, "encoder"),
            {"audio_signal": feats, "length": _full_length(feats)},
        )[0]
        np.testing.assert_allclose(enc, golden["enc_out"], atol=1e-4)

        # Prediction net: SOS + tokens -> g (matches NeMo add_sos=True)
        sos = np.full((1, 1), _SOS_ID, dtype=np.int64)
        targets = np.concatenate([sos, golden["tokens"].astype(np.int64)], axis=1)
        zeros = np.zeros((2, 1, 640), dtype=np.float32)
        dec = _run(
            _find_onnx(td, "decoder"),
            {"targets": targets, "state_h": zeros, "state_c": zeros},
        )[0]
        np.testing.assert_allclose(dec, golden["pred_out"], atol=1e-4)

        # Joint: log-softmaxed logits
        joint = _run(
            _find_onnx(td, "joint"),
            {
                "encoder_outputs": golden["enc_out"].astype(np.float32),
                "decoder_outputs": golden["pred_out"].astype(np.float32),
            },
        )[0]
        np.testing.assert_allclose(joint, golden["joint_out"], atol=1e-3)


@pytest.mark.integration_slow
def test_fastconformer_rnnt_streaming_parity():
    """Cache-aware streaming encoder matches NeMo over chained chunks.

    Drives the ``encoder_streaming`` graph chunk-by-chunk: chunk 0 starts from
    zero caches, and chunk 1 consumes chunk 0's *own* output caches. Both chunk
    encodings and the running cache-length scalars must match the NeMo
    reference, validating the per-layer attention/conv cache update logic.
    """
    from mobius.integrations.nemo import build_from_nemo

    golden = np.load(_STREAMING_GOLDEN)
    pkg = build_from_nemo(_MODEL_ID, revision=_REVISION)

    with tempfile.TemporaryDirectory() as td:
        pkg.save(td)
        sess = ort.InferenceSession(
            _find_onnx(td, "encoder_streaming"), providers=["CPUExecutionProvider"]
        )

        def step(feats, ch, ct, cl):
            return sess.run(
                None,
                {
                    "audio_signal": feats.astype(np.float32),
                    "length": _full_length(feats),
                    "cache_last_channel": ch,
                    "cache_last_time": ct,
                    "cache_last_channel_len": cl,
                },
            )

        f0 = golden["f0"]
        f1 = golden["f1"]

        # Initial caches are zeros (NeMo get_initial_cache_state). Concrete
        # shapes come from the graph's declared cache inputs with the symbolic
        # batch dim resolved to 1.
        def _concrete(shape):
            return [1 if not isinstance(d, int) else d for d in shape]

        shapes = {i.name: _concrete(i.shape) for i in sess.get_inputs()}
        ch0 = np.zeros(shapes["cache_last_channel"], dtype=np.float32)
        ct0 = np.zeros(shapes["cache_last_time"], dtype=np.float32)
        cl0 = np.zeros((1,), dtype=np.int64)

        out0, len0, ch1, ct1, cl1 = step(f0, ch0, ct0, cl0)
        np.testing.assert_allclose(out0, golden["out0"], atol=1e-4)
        assert len0.tolist() == golden["len0"].tolist()
        assert cl1.tolist() == golden["cl1"].tolist()

        # Chunk 1 consumes chunk 0's ONNX-produced caches (self-chaining).
        out1, len1, _ch2, _ct2, cl2 = step(f1, ch1, ct1, cl1)
        np.testing.assert_allclose(out1, golden["out1"], atol=1e-4)
        assert len1.tolist() == golden["len1"].tolist()
        assert cl2.tolist() == golden["cl2"].tolist()


@pytest.mark.integration_slow
def test_fastconformer_rnnt_greedy_decode():
    """L5 smoke test: run an RNN-T greedy decode loop over the 3 ONNX models."""
    from mobius.integrations.nemo import build_from_nemo

    golden = np.load(_GOLDEN)
    blank_id = 1024  # rnnt_num_classes; vocab index reserved for the blank label
    pkg = build_from_nemo(_MODEL_ID, revision=_REVISION)

    with tempfile.TemporaryDirectory() as td:
        pkg.save(td)
        enc_sess = ort.InferenceSession(
            _find_onnx(td, "encoder"), providers=["CPUExecutionProvider"]
        )
        dec_sess = ort.InferenceSession(
            _find_onnx(td, "decoder"), providers=["CPUExecutionProvider"]
        )
        joint_sess = ort.InferenceSession(
            _find_onnx(td, "joint"), providers=["CPUExecutionProvider"]
        )

        feats = golden["feats"].astype(np.float32)
        enc = enc_sess.run(None, {"audio_signal": feats, "length": _full_length(feats)})[0]
        n_frames = enc.shape[2]

        # Prime the prediction net with the zero start-of-sequence embedding.
        h = np.zeros((2, 1, 640), dtype=np.float32)
        c = np.zeros((2, 1, 640), dtype=np.float32)
        last_token = np.array([[_SOS_ID]], dtype=np.int64)
        g, h, c = dec_sess.run(None, {"targets": last_token, "state_h": h, "state_c": c})

        tokens: list[int] = []
        max_symbols = 5
        for t in range(n_frames):
            enc_t = enc[:, :, t : t + 1]
            emitted = 0
            while emitted < max_symbols:
                logits = joint_sess.run(
                    None,
                    {"encoder_outputs": enc_t, "decoder_outputs": g},
                )[0]
                k = int(np.argmax(logits.reshape(-1)))
                if k == blank_id:
                    break
                tokens.append(k)
                emitted += 1
                g, h, c = dec_sess.run(
                    None,
                    {
                        "targets": np.array([[k]], dtype=np.int64),
                        "state_h": h,
                        "state_c": c,
                    },
                )

        # The loop must terminate and only emit valid non-blank vocab ids.
        assert all(0 <= tok < blank_id for tok in tokens)

        # Incremental decoding must match a single-shot decode of the same
        # token sequence: feeding tokens one at a time while carrying the LSTM
        # state should reproduce the one-shot prediction outputs and final
        # state. This validates the decoder's stateful streaming contract.
        seq = [_SOS_ID, 3, 5, 7, 9]
        hi = np.zeros((2, 1, 640), dtype=np.float32)
        ci = np.zeros((2, 1, 640), dtype=np.float32)
        step_g = []
        for tok in seq:
            gi, hi, ci = dec_sess.run(
                None,
                {
                    "targets": np.array([[tok]], dtype=np.int64),
                    "state_h": hi,
                    "state_c": ci,
                },
            )
            step_g.append(gi[:, :, 0])  # (1, 640)
        incremental = np.stack(step_g, axis=-1)  # (1, 640, len(seq))

        one_shot, oh, oc = dec_sess.run(
            None,
            {
                "targets": np.array([seq], dtype=np.int64),
                "state_h": np.zeros((2, 1, 640), dtype=np.float32),
                "state_c": np.zeros((2, 1, 640), dtype=np.float32),
            },
        )
        np.testing.assert_allclose(incremental, one_shot, atol=1e-4)
        np.testing.assert_allclose(hi, oh, atol=1e-4)
        np.testing.assert_allclose(ci, oc, atol=1e-4)
