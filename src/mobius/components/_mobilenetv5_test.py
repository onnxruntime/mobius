# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for the MobileNet-V5 vision encoder (Gemma 3n vision tower).

The reference implementation lives in timm, which is deliberately *not* a
mobius dependency, so these tests cannot diff against it.  What they can pin
down without it:

* the weight-name/shape contract against the ``mobilenetv5_300m_enc``
  checkpoint layout (548 tensors — the numbers below are transcribed from
  ``google/gemma-3n-E4B-it``);
* the resolution flow through the tower, which is where a wrong stride or a
  wrong SAME-padding amount shows up;
* end-to-end ONNX execution, which catches the shape-inference and
  broadcast mistakes that a pure construction test would miss.

Numerical parity against timm was validated out-of-band at 256x256 and
768x768 (max abs diff ~5e-6, cosine 1.0).
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
from onnxscript import GraphBuilder

from mobius._constants import OPSET_VERSION
from mobius.components._conv import RmsNorm2d
from mobius.components._mobilenetv5 import (
    _MOBILENETV5_300M_ENC_BLOCKS,
    MobileNetV5Encoder,
    _EdgeResidualSpec,
    _make_block,
    _MobileNetV5MSFA,
    _MQASpec,
    _same_padding,
    _UIBSpec,
    _UniversalInvertedBottleneck,
)

# E4B ships 768x768 images. 256 is the smallest resolution the tower accepts
# (below it the stage-2 grid stops being a multiple of the 16x16 MSFA output),
# and it still exercises every stride-2 reduction plus the average-pool path.
_SMALL_IMAGE_SIZE = 256
# A narrow output width keeps the MSFA projections small; the block spec (and
# therefore everything except msfa.ffn.pw_proj / msfa.norm) is unaffected.
_SMALL_HIDDEN_SIZE = 32
_HIDDEN_SIZE = 2048


def _fill_random_weights(module, *, seed: int = 0) -> None:
    """Assign plausible random values to every parameter of ``module``.

    1-D parameters (norm scales, layer-scale gammas) are set to ones and conv
    kernels are scaled by ``1 / sqrt(fan_in)``, so activations stay O(1) and
    float32 comparisons remain meaningful rather than saturating.
    """
    rng = np.random.default_rng(seed)
    for param in module.parameters():
        dims = [int(d) for d in param.shape]
        if len(dims) == 1:
            data = np.ones(dims, dtype=np.float32)
        else:
            fan_in = int(np.prod(dims[1:]))
            data = (rng.standard_normal(dims) / np.sqrt(fan_in)).astype(np.float32)
        param.const_value = ir.tensor(data)


def _build_session(module, input_shape: tuple[int | str, ...], *, seed: int = 0):
    """Build ``module`` into an in-memory ONNX session with random weights.

    ``input_shape`` entries may be strings for dynamic (symbolic) dimensions.
    Returns ``(session, input_name)``.  Serialising to protobuf in memory
    (rather than via ``ir.save`` + tempfile) avoids Windows PermissionError
    under concurrent test execution.
    """
    x_input = ir.Value(
        name="x",
        shape=ir.Shape(list(input_shape)),
        type=ir.TensorType(ir.DataType.FLOAT),
    )
    graph = ir.Graph(
        inputs=[x_input],
        outputs=[],
        nodes=[],
        name="test_mobilenetv5",
        opset_imports={"": OPSET_VERSION},
    )
    gb = GraphBuilder(graph)
    result = module(gb.op, x_input)
    result.name = "output"
    graph.outputs.append(result)

    _fill_random_weights(module, seed=seed)

    model = ir.Model(graph, ir_version=11)
    proto = ir.serde.serialize_model(model)
    session = ort.InferenceSession(
        proto.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return session, "x"


# ---------------------------------------------------------------------------
# SAME padding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kernel", "stride", "input_size", "expected"),
    [
        # Stride 1: total pad is k-1, split evenly for odd kernels.
        (3, 1, 8, (1, 1, 1, 1)),
        (5, 1, 8, (2, 2, 2, 2)),
        # Stride 2 on an even input: total pad is k-2 -> asymmetric for odd k.
        (3, 2, 8, (0, 0, 1, 1)),
        (5, 2, 8, (1, 1, 2, 2)),
        # 1x1 convs never pad.
        (1, 1, 8, (0, 0, 0, 0)),
        (1, 2, 8, (0, 0, 0, 0)),
    ],
)
def test_same_padding_amounts(kernel, stride, input_size, expected):
    """SAME padding puts the extra pixel on the *end*, as TensorFlow does.

    Symmetric ``k // 2`` padding is wrong for even total pads and would shift
    every strided feature map by one pixel.
    """
    assert _same_padding(kernel, stride, input_size) == expected


@pytest.mark.parametrize("kernel", [1, 3, 5])
@pytest.mark.parametrize("stride", [1, 2])
def test_same_padding_preserves_expected_output_size(kernel, stride):
    """``ceil(input / stride)`` output size, which is what SAME guarantees."""
    input_size = 32
    top, left, bottom, right = _same_padding(kernel, stride, input_size)
    padded = input_size + top + bottom
    out = (padded - kernel) // stride + 1
    assert out == -(-input_size // stride)
    assert (top, left) == (bottom, right) or bottom == top + 1
    assert top == left and bottom == right  # square kernels pad symmetrically


# ---------------------------------------------------------------------------
# RmsNorm2d
# ---------------------------------------------------------------------------


def test_rms_norm_2d_has_only_a_scale():
    """No bias and no running statistics — see the class docstring."""
    norm = RmsNorm2d(8)
    assert [n for n, _ in norm.named_parameters()] == ["weight"]


def test_rms_norm_2d_normalizes_over_the_channel_axis():
    """Channel-axis (not last-axis) RMS, matching timm's ``RmsNorm2d``."""
    norm = RmsNorm2d(4, eps=1e-6)
    session, name = _build_session(norm, (1, 4, 3, 3))
    x = np.arange(1, 37, dtype=np.float32).reshape(1, 4, 3, 3)

    got = session.run(None, {name: x})[0]

    expected = x / np.sqrt(np.mean(x**2, axis=1, keepdims=True) + 1e-6)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_rms_norm_2d_applies_per_channel_scale():
    """The ``[C]`` weight broadcasts over NCHW, not over the last axis."""
    norm = RmsNorm2d(3, eps=1e-6)
    x_input = ir.Value(
        name="x", shape=ir.Shape([1, 3, 2, 2]), type=ir.TensorType(ir.DataType.FLOAT)
    )
    graph = ir.Graph(
        inputs=[x_input],
        outputs=[],
        nodes=[],
        name="t",
        opset_imports={"": OPSET_VERSION},
    )
    gb = GraphBuilder(graph)
    out = norm(gb.op, x_input)
    out.name = "output"
    graph.outputs.append(out)
    scale = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    norm.weight.const_value = ir.tensor(scale)
    proto = ir.serde.serialize_model(ir.Model(graph, ir_version=11))
    session = ort.InferenceSession(
        proto.SerializeToString(), providers=["CPUExecutionProvider"]
    )

    x = np.ones((1, 3, 2, 2), dtype=np.float32)
    got = session.run(None, {"x": x})[0]

    # All-ones input normalizes to ones, so the output *is* the scale,
    # broadcast over H and W.
    expected = np.broadcast_to(scale.reshape(1, 3, 1, 1), (1, 3, 2, 2))
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Block spec table
# ---------------------------------------------------------------------------


def test_block_spec_matches_checkpoint_block_counts():
    """84 blocks in a 3/5/37/39 split, as the E4B checkpoint indices show."""
    counts = [len(stage) for stage in _MOBILENETV5_300M_ENC_BLOCKS]
    assert counts == [3, 5, 37, 39]
    assert sum(counts) == 84


def test_block_spec_stage_output_channels():
    """Stage outputs 128/256/640/1280; the last two sum to the MSFA's 1920."""
    outs = [stage[-1].out_chs for stage in _MOBILENETV5_300M_ENC_BLOCKS]
    assert outs == [128, 256, 640, 1280]
    assert outs[-2] + outs[-1] == 1920


def test_block_spec_attention_block_counts():
    """33 MQA blocks total; only stage 2's 14 downsample K/V.

    Cross-checks the checkpoint: 33 ``attn.*.proj`` groups but only 14
    ``key.down_conv`` / ``value.down_conv`` pairs.
    """
    mqa = [
        spec
        for stage in _MOBILENETV5_300M_ENC_BLOCKS
        for spec in stage
        if isinstance(spec, _MQASpec)
    ]
    assert len(mqa) == 33
    assert sum(1 for spec in mqa if spec.kv_stride > 1) == 14
    # Stage 2 uses 12 heads of 64; stage 3 uses 16 heads of 96.
    assert {(s.num_heads, s.kv_dim) for s in mqa} == {(12, 64), (16, 96)}


def test_block_spec_stage_zero_is_edge_residual_only():
    """Timm's arch_def gives stage 0 ``er_`` blocks; later stages never do."""
    assert all(isinstance(s, _EdgeResidualSpec) for s in _MOBILENETV5_300M_ENC_BLOCKS[0])
    later = [s for stage in _MOBILENETV5_300M_ENC_BLOCKS[1:] for s in stage]
    assert not any(isinstance(s, _EdgeResidualSpec) for s in later)


def test_block_spec_each_stage_starts_with_a_stride_two_block():
    """One stride-2 reduction per stage (5 total counting the stem).

    Only the first block of each stage downsamples; MQA blocks carry no
    ``stride`` at all (they preserve the grid, downsampling K/V instead).
    """
    for stage_idx, stage in enumerate(_MOBILENETV5_300M_ENC_BLOCKS):
        assert not isinstance(stage[0], _MQASpec)
        assert stage[0].stride == 2, f"stage {stage_idx} does not downsample"
        for spec in stage[1:]:
            assert isinstance(spec, _MQASpec) or spec.stride == 1


def test_block_spec_mqa_blocks_do_not_change_extent():
    """MQA blocks preserve channels, so the interleaved UIB sets the width."""
    for stage in _MOBILENETV5_300M_ENC_BLOCKS:
        chs = None
        for spec in stage:
            if isinstance(spec, _MQASpec) and chs is not None:
                assert spec.out_chs == chs
            chs = spec.out_chs


# ---------------------------------------------------------------------------
# Encoder construction
# ---------------------------------------------------------------------------


def test_encoder_parameter_count_matches_checkpoint():
    """548 tensors under ``model.vision_tower.timm_model.`` in E4B."""
    enc = MobileNetV5Encoder(hidden_size=_HIDDEN_SIZE, image_size=768)
    assert len(list(enc.named_parameters())) == 548


def test_encoder_conv_stem_is_the_only_conv_with_a_bias():
    """Every other conv in the tower is bias-free (checkpoint has one bias)."""
    enc = MobileNetV5Encoder(hidden_size=_HIDDEN_SIZE, image_size=768)
    biases = [n for n, _ in enc.named_parameters() if n.endswith(".bias")]
    assert biases == ["conv_stem.conv.bias"]


def test_encoder_norm_weights_have_no_batchnorm_companions():
    """``bn.*`` names carry only ``weight`` — they are RMSNorms, not BatchNorms."""
    enc = MobileNetV5Encoder(hidden_size=_HIDDEN_SIZE, image_size=768)
    names = {n for n, _ in enc.named_parameters()}
    assert sum(1 for n in names if n.endswith("bn.weight")) == 116
    for suffix in ("running_mean", "running_var", "num_batches_tracked", "bn.bias"):
        assert not any(n.endswith(suffix) for n in names)


def test_encoder_layer_scale_gamma_count():
    """81 blocks carry a ``layer_scale.gamma``; the MSFA's FFN does not."""
    enc = MobileNetV5Encoder(hidden_size=_HIDDEN_SIZE, image_size=768)
    gammas = [n for n, _ in enc.named_parameters() if n.endswith("layer_scale.gamma")]
    assert len(gammas) == 81
    assert not any(n.startswith("msfa.") for n in gammas)


def test_encoder_msfa_projection_shapes():
    """MSFA fuses 640 + 1280 -> 1920, expands 2x, projects to hidden_size."""
    enc = MobileNetV5Encoder(hidden_size=_HIDDEN_SIZE, image_size=768)
    shapes = {n: [int(d) for d in p.shape] for n, p in enc.named_parameters()}
    assert shapes["msfa.ffn.pw_exp.conv.weight"] == [3840, 1920, 1, 1]
    assert shapes["msfa.ffn.pw_proj.conv.weight"] == [_HIDDEN_SIZE, 3840, 1, 1]
    assert shapes["msfa.norm.weight"] == [_HIDDEN_SIZE]


def test_encoder_block_names_are_stage_qualified():
    """Nested ModuleLists must yield ``blocks.{stage}.{idx}.`` as timm does."""
    enc = MobileNetV5Encoder(hidden_size=_HIDDEN_SIZE, image_size=768)
    names = {n for n, _ in enc.named_parameters()}
    assert "blocks.0.0.conv_exp.weight" in names
    assert "blocks.3.38.pw_proj.conv.weight" in names
    # Stage 2's MQA blocks downsample K/V; stage 3's do not.
    assert "blocks.2.9.attn.key.down_conv.weight" in names
    assert not any(n.startswith("blocks.3.") and "down_conv" in n for n in names)


@pytest.mark.parametrize("image_size", [768, 512, 256])
def test_encoder_accepts_supported_image_sizes(image_size):
    enc = MobileNetV5Encoder(hidden_size=_HIDDEN_SIZE, image_size=image_size)
    assert enc.image_size == image_size


@pytest.mark.parametrize("image_size", [100, 224, 384, 64])
def test_encoder_rejects_unsupported_image_sizes(image_size):
    """5 stride-2 reductions and a 16x16 MSFA grid constrain the input size.

    224, 384 and 64 are divisible by 32 but give stage-2 resolutions
    (14, 24, 4) that are not multiples of the 16x16 MSFA output, so the
    average-pool path cannot produce 256 soft tokens.
    """
    with pytest.raises(ValueError):
        MobileNetV5Encoder(hidden_size=_HIDDEN_SIZE, image_size=image_size)


# ---------------------------------------------------------------------------
# ONNX execution
#
# The real tower is 300M parameters (>1 GB of initializers), past protobuf's
# 2 GB serialization ceiling once weights are attached, so the whole-encoder
# checks below stay at the graph level and the per-block tests carry the
# runtime coverage.  Every block type appears, at the resolutions and channel
# counts the spec table actually uses.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "in_chs", "resolution", "expected_res"),
    [
        # Stage 0's strided EdgeResidual and its two residual siblings.
        (_EdgeResidualSpec(16, 32, 3, 2), 8, 16, 8),
        (_EdgeResidualSpec(8, 32, 3, 1), 8, 16, 16),
        # UIB: strided dw_start+dw_mid, then FFN-only, then dw_start-only.
        (_UIBSpec(16, 32, 3, 5, 2), 8, 16, 8),
        (_UIBSpec(8, 32, 0, 0, 1), 8, 16, 16),
        (_UIBSpec(16, 32, 5, 0, 2), 8, 16, 8),
        # MQA with and without the K/V downsample (stage 2 vs stage 3).
        (_MQASpec(8, 4, 8, 2, 3), 8, 16, 16),
        (_MQASpec(8, 2, 4, 1, 3), 8, 16, 16),
    ],
)
def test_block_runs_and_preserves_expected_shape(spec, in_chs, resolution, expected_res):
    """Each block type executes and reduces the grid exactly as specified."""
    block = _make_block(in_chs, spec, resolution, norm_eps=1e-6)
    session, name = _build_session(block, (2, in_chs, resolution, resolution), seed=3)
    shape = (2, in_chs, resolution, resolution)
    x = np.random.default_rng(4).standard_normal(shape).astype(np.float32)

    got = session.run(None, {name: x})[0]

    assert got.shape == (2, spec.out_chs, expected_res, expected_res)
    assert np.isfinite(got).all()


def test_mqa_block_mixes_across_spatial_positions():
    """Attention must be spatial, not per-pixel.

    A per-pixel bug (e.g. attending over channels instead of positions) still
    produces the right shape, so perturbing one pixel and checking that a
    distant pixel moves is the discriminating check.
    """
    spec = _MQASpec(out_chs=8, num_heads=4, kv_dim=8, kv_stride=1, dw_kernel_size=3)
    block = _make_block(8, spec, 8, norm_eps=1e-6)
    session, name = _build_session(block, (1, 8, 8, 8), seed=5)
    x = np.random.default_rng(6).standard_normal((1, 8, 8, 8)).astype(np.float32)
    perturbed = x.copy()
    perturbed[0, :, 0, 0] += 10.0

    base = session.run(None, {name: x})[0]
    moved = session.run(None, {name: perturbed})[0]

    # The far corner is influenced by the perturbed pixel through attention.
    assert not np.allclose(base[0, :, 7, 7], moved[0, :, 7, 7], rtol=1e-3, atol=1e-3)


def test_msfa_fuses_two_resolutions_into_the_soft_token_grid():
    """MSFA upsamples the lower-res input, concatenates, and pools to 16x16."""
    msfa = _MobileNetV5MSFA(in_chs=12, out_chs=8, input_resolutions=(32, 16), norm_eps=1e-6)
    high = ir.Value(
        name="high", shape=ir.Shape([1, 4, 32, 32]), type=ir.TensorType(ir.DataType.FLOAT)
    )
    low = ir.Value(
        name="low", shape=ir.Shape([1, 8, 16, 16]), type=ir.TensorType(ir.DataType.FLOAT)
    )
    graph = ir.Graph(
        inputs=[high, low],
        outputs=[],
        nodes=[],
        name="t",
        opset_imports={"": OPSET_VERSION},
    )
    gb = GraphBuilder(graph)
    out = msfa(gb.op, [high, low])
    out.name = "output"
    graph.outputs.append(out)
    _fill_random_weights(msfa, seed=8)
    proto = ir.serde.serialize_model(ir.Model(graph, ir_version=11))
    session = ort.InferenceSession(
        proto.SerializeToString(), providers=["CPUExecutionProvider"]
    )

    rng = np.random.default_rng(9)
    got = session.run(
        None,
        {
            "high": rng.standard_normal((1, 4, 32, 32)).astype(np.float32),
            "low": rng.standard_normal((1, 8, 16, 16)).astype(np.float32),
        },
    )[0]

    assert got.shape == (1, 8, 16, 16)
    assert np.isfinite(got).all()


def test_msfa_rejects_non_divisible_input_resolutions():
    """The upsample factor must be an exact integer for static `scales`."""
    with pytest.raises(ValueError, match="does not divide"):
        _MobileNetV5MSFA(in_chs=12, out_chs=8, input_resolutions=(32, 20))


def _build_graph(image_size: int, hidden_size: int):
    """Build the encoder graph (no weights, no session) for shape assertions."""
    enc = MobileNetV5Encoder(hidden_size=hidden_size, image_size=image_size)
    x_input = ir.Value(
        name="pixel_values",
        shape=ir.Shape([1, 3, image_size, image_size]),
        type=ir.TensorType(ir.DataType.FLOAT),
    )
    graph = ir.Graph(
        inputs=[x_input],
        outputs=[],
        nodes=[],
        name="t",
        opset_imports={"": OPSET_VERSION},
    )
    return enc, GraphBuilder(graph), graph, x_input


def test_encoder_graph_shapes_are_statically_inferable():
    """Every intermediate must keep a concrete spatial shape.

    SAME padding is folded into the ``Conv`` ``pads`` attribute rather than
    emitted as ``Pad`` nodes precisely so that shape inference survives; a
    dynamic H/W here would mean that folding regressed.
    """
    enc, gb, graph, x_input = _build_graph(_SMALL_IMAGE_SIZE, _SMALL_HIDDEN_SIZE)

    out = enc(gb.op, x_input)

    assert out.shape is not None
    assert [int(d) for d in out.shape] == [1, _SMALL_HIDDEN_SIZE, 16, 16]
    assert not any(node.op_type == "Pad" for node in graph)


def test_encoder_resolution_flow():
    """Each stage halves the grid: 256 -> stem 128 -> 64 -> 32 -> 16 -> 8."""
    enc, gb, _graph, x_input = _build_graph(_SMALL_IMAGE_SIZE, _SMALL_HIDDEN_SIZE)
    op = gb.op

    x = enc.conv_stem(op, x_input)
    assert [int(d) for d in x.shape] == [1, 64, 128, 128]

    expected = [(128, 64), (256, 32), (640, 16), (1280, 8)]
    for stage, (chs, res) in zip(enc.blocks, expected):
        for block in stage:
            x = block(op, x)
        assert [int(d) for d in x.shape] == [1, chs, res, res]


def test_uib_block_without_depthwise_convs_has_no_dw_weights():
    """The ``(0, 0)`` kernel spec is what selects the FFN-only UIB shape."""
    spec = _UIBSpec(out_chs=64, exp_chs=128, dw_start_kernel=0, dw_mid_kernel=0, stride=1)
    block = _UniversalInvertedBottleneck(64, spec, input_size=8)
    names = {n for n, _ in block.named_parameters()}
    assert not any("dw_start" in n or "dw_mid" in n for n in names)
    assert "pw_exp.conv.weight" in names
    assert "pw_proj.conv.weight" in names
