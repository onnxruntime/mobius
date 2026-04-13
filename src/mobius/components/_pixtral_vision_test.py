# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for Pixtral vision encoder components."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort

from mobius._configs import ArchitectureConfig, VisionConfig
from mobius._constants import OPSET_VERSION
from mobius.components._pixtral_vision import (
    Mistral3MultiModalProjector,
    Mistral3PatchMerger,
    PixtralRoPE2D,
    PixtralTransformerEncoder,
    PixtralVisionTower,
)


def test_pixtral_rope_2d_cache_shapes():
    """2D RoPE produces correct cache shapes."""
    rope = PixtralRoPE2D(head_dim=16, max_grid_size=4)
    # 4x4 grid = 16 positions, cache dim = head_dim/2 = 8
    assert list(rope.cos_cache.shape) == [16, 8]
    assert list(rope.sin_cache.shape) == [16, 8]


def test_pixtral_rope_2d_cache_values():
    """2D RoPE cache has non-trivial values at non-zero positions."""
    rope = PixtralRoPE2D(head_dim=16, max_grid_size=4)
    cos_data = rope.cos_cache.const_value.numpy()
    sin_data = rope.sin_cache.const_value.numpy()
    # Position (0,0) should have cos=1, sin=0 (freq*0=0)
    np.testing.assert_allclose(cos_data[0], 1.0, atol=1e-6)
    np.testing.assert_allclose(sin_data[0], 0.0, atol=1e-6)
    # Non-zero positions should differ
    assert not np.allclose(cos_data[5], cos_data[0])


def test_pixtral_rope_2d_small_grid():
    """2D RoPE works with smallest possible grid (1x1)."""
    rope = PixtralRoPE2D(head_dim=8, max_grid_size=1)
    assert list(rope.cos_cache.shape) == [1, 4]


def test_pixtral_vision_tower_builds():
    """PixtralVisionTower constructs without errors."""
    config = ArchitectureConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=100,
        hidden_act="silu",
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        vision=VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=28,
            patch_size=14,
            model_type="pixtral",
        ),
    )
    tower = PixtralVisionTower(config)
    assert tower.ln_pre is not None
    assert tower.transformer is not None
    assert tower.rope is not None
    assert tower.patch_conv is not None


def test_mistral3_projector_builds():
    """Mistral3MultiModalProjector constructs without errors."""
    proj = Mistral3MultiModalProjector(
        vision_hidden_size=32,
        text_hidden_size=64,
        spatial_merge_size=2,
    )
    assert proj.patch_merger is not None
    assert proj.norm is not None
    assert proj.linear_1 is not None
    assert proj.linear_2 is not None


def test_transformer_encoder_layer_count():
    """Encoder creates the correct number of layers."""
    enc = PixtralTransformerEncoder(
        num_layers=3,
        hidden_size=32,
        intermediate_size=64,
        num_heads=2,
        head_dim=16,
    )
    assert len(list(enc.layers)) == 3


def test_patch_merger_builds():
    """PatchMerger reduces from merged_dim to hidden_size."""
    merger = Mistral3PatchMerger(hidden_size=32, spatial_merge_size=2)
    # input_dim = 32 * 2 * 2 = 128, output_dim = 32
    assert list(merger.merging_layer.weight.shape) == [32, 128]


def test_patch_merger_matches_hf_unfold_ordering():
    """PatchMerger element ordering matches HuggingFace F.unfold (dim-major).

    HF uses ``F.unfold(image_grid, kernel_size=ms, stride=ms)`` which
    groups elements as ``[D, ms_h, ms_w]`` per spatial position (dim is
    the outermost loop). The ONNX implementation must reproduce this
    ordering so the learned ``merging_layer`` projection is correct.
    """
    import torch
    from onnxscript._internal.builder import GraphBuilder

    hidden_size = 8
    ms = 2
    grid_h, grid_w = 4, 4
    seq_len = grid_h * grid_w

    rng = np.random.default_rng(42)
    x = rng.standard_normal((1, seq_len, hidden_size)).astype(np.float32)

    # HF reference: F.unfold ordering
    x_torch = torch.from_numpy(x.squeeze(0))  # (seq_len, D)
    image_grid = x_torch.view(grid_h, grid_w, hidden_size).permute(2, 0, 1).unsqueeze(0)
    grid = torch.nn.functional.unfold(image_grid, kernel_size=ms, stride=ms)
    hf_merged = grid.view(hidden_size * ms * ms, -1).t().numpy()  # (num_merged, D*ms*ms)

    # Build ONNX model that performs only the reshape+transpose+flatten
    # (no linear projection) so we can compare the raw merge ordering.
    x_input = ir.Value(
        name="x",
        shape=ir.Shape([1, seq_len, hidden_size]),
        type=ir.TensorType(ir.DataType.FLOAT),
    )
    gh_input = ir.Value(
        name="grid_h",
        shape=ir.Shape([]),
        type=ir.TensorType(ir.DataType.INT64),
    )
    gw_input = ir.Value(
        name="grid_w",
        shape=ir.Shape([]),
        type=ir.TensorType(ir.DataType.INT64),
    )
    graph = ir.Graph(
        inputs=[x_input, gh_input, gw_input],
        outputs=[],
        nodes=[],
        name="test_merge_ordering",
        opset_imports={"": OPSET_VERSION},
    )
    gb = GraphBuilder(graph)
    op = gb.op

    # Reproduce the PatchMerger reshape+transpose+flatten logic
    batch = op.Shape(x_input, start=0, end=1)
    d = op.Shape(x_input, start=2, end=3)
    ms_scalar = op.Constant(value_int=ms)
    h_m = op.Div(gh_input, ms_scalar)
    w_m = op.Div(gw_input, ms_scalar)
    ms_1d = op.Constant(value_ints=[ms])
    h_m_1d = op.Reshape(h_m, op.Constant(value_ints=[1]))
    w_m_1d = op.Reshape(w_m, op.Constant(value_ints=[1]))
    shape_6d = op.Concat(batch, h_m_1d, ms_1d, w_m_1d, ms_1d, d, axis=0)
    merged = op.Reshape(x_input, shape_6d)
    merged = op.Transpose(merged, perm=[0, 1, 3, 5, 2, 4])
    merged_count = op.Mul(h_m_1d, w_m_1d)
    shape_3d = op.Concat(batch, merged_count, op.Constant(value_ints=[-1]), axis=0)
    result = op.Reshape(merged, shape_3d)
    result.name = "output"
    graph.outputs.append(result)

    model = ir.Model(graph, ir_version=11)

    # Serialize to protobuf in-memory (avoids Windows PermissionError
    # from concurrent file access with tempfile + ir.save).
    proto = ir.serde.serialize_model(model)
    sess = ort.InferenceSession(
        proto.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    onnx_out = sess.run(
        None,
        {
            "x": x,
            "grid_h": np.array(grid_h, dtype=np.int64),
            "grid_w": np.array(grid_w, dtype=np.int64),
        },
    )[0]

    np.testing.assert_allclose(
        onnx_out.squeeze(0),
        hf_merged,
        atol=1e-5,
        rtol=1e-5,
        err_msg="PatchMerger ordering does not match HF F.unfold",
    )
