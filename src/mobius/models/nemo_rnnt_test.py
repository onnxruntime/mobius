# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for the FastConformer-RNNT model (graph construction).

These build the three ONNX sub-graphs from a tiny config (no weights, no
network) and run them with ONNX Runtime to verify shapes and that the graph
is structurally valid.
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

from mobius._builder import build_from_module
from mobius._configs import ArchitectureConfig
from mobius.models import EncDecRNNTModel


def _tiny_config() -> ArchitectureConfig:
    return ArchitectureConfig(
        vocab_size=33,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=16,
        num_key_value_heads=4,
        intermediate_size=128,
        dtype=ir.DataType.FLOAT,
        fastconformer_feat_in=32,
        fastconformer_subsampling_conv_channels=16,
        fastconformer_conv_kernel_size=9,
        fastconformer_att_context_size=(6, 1),
        rnnt_pred_hidden=48,
        rnnt_pred_rnn_layers=2,
        rnnt_joint_hidden=48,
        rnnt_num_classes=32,
    )


def _build():
    config = _tiny_config()
    pkg = build_from_module(EncDecRNNTModel(config), config, task="fastconformer-rnnt")
    return config, pkg


def _session(model: ir.Model) -> ort.InferenceSession:
    _fill_random_weights(model)
    proto = ir.to_proto(model)
    return ort.InferenceSession(proto.SerializeToString(), providers=["CPUExecutionProvider"])


def _fill_random_weights(model: ir.Model) -> None:
    """Assign random constant values to any uninitialised graph initializers."""
    rng = np.random.default_rng(0)
    for value in model.graph.initializers.values():
        if value.const_value is not None:
            continue
        shape = [d if isinstance(d, int) else 1 for d in value.shape]
        value.const_value = ir.tensor(rng.standard_normal(shape).astype(np.float32) * 0.05)


def test_package_has_three_models():
    _, pkg = _build()
    assert set(pkg.keys()) == {"encoder", "decoder", "joint"}


def test_encoder_io_shapes():
    config, pkg = _build()
    sess = _session(pkg["encoder"])
    assert [i.name for i in sess.get_inputs()] == ["audio_signal", "length"]
    t = 50
    feats = np.random.randn(1, config.fastconformer_feat_in, t).astype(np.float32)
    length = np.array([t], dtype=np.int64)
    out, enc_len = sess.run(None, {"audio_signal": feats, "length": length})
    # 8x causal subsampling: each stride-2 stage maps n -> n // 2 + 1
    expected_t = t
    for _ in range(3):
        expected_t = expected_t // 2 + 1
    assert out.shape[0] == 1
    assert out.shape[1] == config.hidden_size
    assert out.shape[2] == expected_t
    assert enc_len.tolist() == [expected_t]


def test_decoder_io_shapes_and_state():
    config, pkg = _build()
    sess = _session(pkg["decoder"])
    assert [i.name for i in sess.get_inputs()] == ["targets", "state_h", "state_c"]
    layers = config.rnnt_pred_rnn_layers
    h = config.rnnt_pred_hidden
    u = 4
    targets = np.array([[1, 2, 3, 4]], dtype=np.int64)
    z = np.zeros((layers, 1, h), dtype=np.float32)
    g, new_h, new_c = sess.run(None, {"targets": targets, "state_h": z, "state_c": z})
    assert g.shape == (1, h, u)
    assert new_h.shape == (layers, 1, h)
    assert new_c.shape == (layers, 1, h)


def test_joint_io_shapes():
    config, pkg = _build()
    sess = _session(pkg["joint"])
    t, u = 6, 4
    enc = np.random.randn(1, config.hidden_size, t).astype(np.float32)
    dec = np.random.randn(1, config.rnnt_pred_hidden, u).astype(np.float32)
    (logits,) = sess.run(None, {"encoder_outputs": enc, "decoder_outputs": dec})
    # (B, T', U, vocab + blank)
    assert logits.shape == (1, t, u, config.rnnt_num_classes + 1)


def test_joint_output_is_log_probs():
    config, pkg = _build()
    sess = _session(pkg["joint"])
    enc = np.random.randn(1, config.hidden_size, 3).astype(np.float32)
    dec = np.random.randn(1, config.rnnt_pred_hidden, 2).astype(np.float32)
    (logits,) = sess.run(None, {"encoder_outputs": enc, "decoder_outputs": dec})
    # log_softmax over the last axis: exp sums to 1
    probs = np.exp(logits)
    np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-4)


@pytest.mark.parametrize("batch", [1, 2])
def test_encoder_supports_batch(batch):
    config, pkg = _build()
    sess = _session(pkg["encoder"])
    t = 40
    feats = np.random.randn(batch, config.fastconformer_feat_in, t).astype(np.float32)
    length = np.full((batch,), t, dtype=np.int64)
    out, enc_len = sess.run(None, {"audio_signal": feats, "length": length})
    assert out.shape[0] == batch
    assert enc_len.shape == (batch,)


def test_encoder_padding_mask_consistency():
    """Length-aware attention masking keeps a sample's valid region stable.

    A short sample's valid region must be identical whether it is run alone or
    padded into a longer batch.
    """
    config, pkg = _build()
    sess = _session(pkg["encoder"])
    rng = np.random.default_rng(0)
    feat = config.fastconformer_feat_in
    short_t, long_t = 24, 56
    short = rng.standard_normal((1, feat, short_t)).astype(np.float32)

    # Run the short sample alone.
    solo, solo_len = sess.run(
        None, {"audio_signal": short, "length": np.array([short_t], dtype=np.int64)}
    )

    # Pad the short sample into a 2-sample batch and declare its true length.
    other = rng.standard_normal((1, feat, long_t)).astype(np.float32)
    padded_short = np.concatenate(
        [short, np.zeros((1, feat, long_t - short_t), dtype=np.float32)], axis=2
    )
    batch = np.concatenate([padded_short, other], axis=0)
    lengths = np.array([short_t, long_t], dtype=np.int64)
    batched, batched_len = sess.run(None, {"audio_signal": batch, "length": lengths})

    valid = int(solo_len[0])
    assert int(batched_len[0]) == valid
    np.testing.assert_allclose(batched[0, :, :valid], solo[0, :, :valid], atol=1e-4)


@pytest.mark.parametrize(
    "dtype,expected",
    [(ir.DataType.FLOAT16, ir.DataType.FLOAT16), (ir.DataType.BFLOAT16, ir.DataType.BFLOAT16)],
)
def test_builds_in_half_precision(dtype, expected):
    """f16/bf16 graphs build and type-check (Sin/Cos stay f32, then cast).

    CPU EP lacks half-precision LSTM/Conv kernels, so this validates graph
    construction + shape/type inference rather than executing the models.
    """
    config = _tiny_config()
    config.dtype = dtype
    pkg = build_from_module(EncDecRNNTModel(config), config, task="fastconformer-rnnt")
    assert set(pkg.keys()) == {"encoder", "decoder", "joint"}
    for name in ("encoder", "decoder", "joint"):
        model = pkg[name]
        for out in model.graph.outputs:
            assert out.dtype is not None, f"{name} output {out.name} has no dtype"
        # Float outputs must carry the requested compute dtype.
        float_outs = [
            o for o in model.graph.outputs if o.dtype not in (ir.DataType.INT64, None)
        ]
        assert any(o.dtype == expected for o in float_outs), (
            f"{name} has no {expected} output: {[o.dtype for o in float_outs]}"
        )
