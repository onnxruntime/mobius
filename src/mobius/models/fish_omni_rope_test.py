# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Weight-free CPU math checks against independently evaluated pinned-source formulas.

These are component proofs, not full-model or CUDA 13.0 admission. Optimizations
are disabled so this profile does not silently re-fuse a RotaryEmbedding.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import onnx_ir as ir
import pytest
import torch

from mobius import build_from_module
from mobius._testing.ort_inference import OnnxModelSession
from mobius.components import apply_partial_rotary_pos_emb
from mobius.components._rotary_embedding import DefaultRope
from mobius.models import FishOmniModel
from mobius.models.fish_omni import (
    _FishAttention,
    _FishSourceAttention,
    _FishSourceRope,
    _FishSourceTowerAttention,
    _FishTowerSourceRope,
    _TowerAttention,
)
from mobius.models.fish_omni_test import (
    learned_parameter_shapes,
    reduced_precision_prototype_task,
    tiny_config,
)
from mobius.tasks._base import _make_graph


def _reference(values, positions, *, width, theta, tower):
    """Evaluate the two source expressions without calling candidate helpers.

    Fish Speech 541566932abbaebcb0a40e562a1f846738383f38:
    backbone.py:234-247 + layers.py:58-66; tower.py TowerAttention.rotate.
    """
    x = torch.from_numpy(values.astype(np.float32))
    pos = torch.from_numpy(positions).float()
    exponents = torch.arange(0, width, 2, dtype=torch.float32)
    if tower:
        frequencies = theta ** (-exponents / width)
    else:
        frequencies = 1.0 / (theta ** (exponents / width))
    angles = pos[..., None] * frequencies
    cosine, sine = angles.cos().unsqueeze(-2), angles.sin().unsqueeze(-2)
    if tower:
        even, odd = x[..., :width:2], x[..., 1:width:2]
        rotated = torch.stack(
            (even * cosine - odd * sine, odd * cosine + even * sine), -1
        ).flatten(-2)
    else:
        prefix = x[..., :width]
        first, second = prefix.chunk(2, -1)
        rotated = prefix * torch.cat((cosine, cosine), -1) + torch.cat(
            (-second, first), -1
        ) * torch.cat((sine, sine), -1)
    output = np.concatenate(
        (rotated.numpy().astype(values.dtype), values[..., width:]), axis=-1
    )
    return output, angles.cos().numpy(), angles.sin().numpy()


@pytest.mark.parametrize("tower", [False, True])
@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
@pytest.mark.parametrize(
    "heads,head_dim,width,theta",
    [(3, 12, 8, 12345.0), (1, 16, 16, 500000.0), (2, 256, 64, 1e7), (16, 256, 64, 1e7)],
)
def test_fish_source_rope_matches_independent_torch(
    dtype, tower, heads, head_dim, width, theta
):
    config = dataclasses.replace(
        tiny_config(),
        dtype=dtype,
        head_dim=head_dim,
        partial_rotary_factor=width / head_dim,
        rope_theta=theta,
    )
    rope = (_FishTowerSourceRope if tower else _FishSourceRope)(config)
    assert not list(rope.parameters())
    graph, builder = _make_graph("source_rope")
    x = builder.input("x", dtype, ["batch", "time", heads * head_dim])
    pos = builder.input("positions", ir.DataType.INT64, ["batch", "time"])
    cosine, sine = rope(builder.op, pos)
    output = apply_partial_rotary_pos_emb(
        builder.op,
        x,
        (cosine, sine),
        num_heads=heads,
        head_dim=head_dim,
        rotary_embedding_dim=width,
        interleaved=tower,
    )
    builder.add_output(output, "y")
    builder.add_output(cosine, "cos")
    builder.add_output(sine, "sin")
    powers = [node for node in graph if node.op_type == "Pow"]
    assert len(powers) == 1
    expansion = powers[0].inputs[0].producer()
    assert expansion.op_type == "Expand"
    shape = expansion.inputs[1].producer()
    assert shape.op_type == "Concat"
    assert shape.inputs[0].producer().inputs[0] is pos
    assert sum(node.op_type == "Reciprocal" for node in graph) == int(not tower)
    assert all(not node.domain and node.op_type != "RotaryEmbedding" for node in graph)
    assert not any("cache" in name for name in graph.initializers)
    session = OnnxModelSession(
        ir.Model(graph, ir_version=11),
        intra_op_num_threads=1,
        graph_optimization_level="disabled",
    )
    rng = np.random.default_rng(411)
    try:
        # Reuse a single symbolic graph across B/T and fresh/nonzero/late positions.
        for batch, length, start in ((1, 1, 0), (2, 7, 19), (1, 33, 32700)):
            positions = np.broadcast_to(
                np.arange(start, start + length, dtype=np.int64), (batch, length)
            ).copy()
            values = rng.standard_normal((batch, length, heads, head_dim)).astype(
                dtype.numpy()
            )
            expected, cos, sin = _reference(
                values, positions, width=width, theta=theta, tower=tower
            )
            actual = session.run(
                {"x": values.reshape(batch, length, -1), "positions": positions}
            )
            assert actual["y"].shape == (batch, length, heads * head_dim)
            assert actual["y"].dtype == dtype.numpy()
            np.testing.assert_allclose(
                actual["y"].astype(np.float32).reshape(values.shape),
                expected.astype(np.float32),
                atol=1e-2,
                rtol=1e-2,
            )
            # Trigonometry is evaluated after FP32 position multiplication,
            # including large positions. Backend libm bits are not portable.
            np.testing.assert_allclose(actual["cos"], cos, atol=1e-2, rtol=1e-2)
            np.testing.assert_allclose(actual["sin"], sin, atol=1e-2, rtol=1e-2)
            np.testing.assert_array_equal(
                actual["y"].reshape(values.shape)[..., width:].view(np.uint16),
                values[..., width:].view(np.uint16),
            )
    finally:
        session.close()


@pytest.mark.parametrize(
    "dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16]
)
def test_fish_rope_module_selection_preserves_learned_parameters(dtype):
    baseline = FishOmniModel(tiny_config())
    module = FishOmniModel(dataclasses.replace(tiny_config(), dtype=dtype))
    assert learned_parameter_shapes(module) == learned_parameter_shapes(baseline)
    text_rope = module.decoder.model.rotary_emb
    tower_rope = module.audio_decoder.rotary_emb
    text_attention = module.decoder.model.layers[1].self_attn
    tower_attention = module.audio_decoder.shared_layers[0].transformer_layer.attention
    if dtype == ir.DataType.FLOAT:
        assert type(text_rope) is type(tower_rope) is DefaultRope
        assert type(text_attention) is _FishAttention
        assert type(tower_attention) is _TowerAttention
    else:
        assert type(text_rope) is _FishSourceRope
        assert type(tower_rope) is _FishTowerSourceRope
        assert type(text_attention) is _FishSourceAttention
        assert type(tower_attention) is _FishSourceTowerAttention
        assert not list(text_rope.parameters()) and not list(tower_rope.parameters())
    # Other model families retain their canonical cached RoPE.
    assert len(list(DefaultRope(tiny_config()).parameters())) == 2


@pytest.mark.parametrize(
    "dtype,ep", [(ir.DataType.FLOAT16, "cuda"), (ir.DataType.BFLOAT16, "onnx-standard")]
)
@pytest.mark.parametrize("task", ["fish-omni", "fish-omni-generation"])
def test_low_fish_packages_emit_runtime_rope_without_custom_rotation(dtype, ep, task):
    config = dataclasses.replace(tiny_config(), dtype=dtype)
    package = build_from_module(
        FishOmniModel(config),
        config,
        task=reduced_precision_prototype_task(task),
        execution_provider=ep,
    )
    for role in (
        "decoder",
        "audio_generator" if task.endswith("generation") else "audio_decoder",
    ):
        graph = package[role].graph
        nodes = list(graph.all_nodes())
        assert not any(node.op_type == "RotaryEmbedding" for node in nodes)
        assert any(node.op_type == "Cos" and not node.domain for node in nodes)
        assert any(node.op_type == "Sin" and not node.domain for node in nodes)
        assert not any(
            name.endswith(("cos_cache", "sin_cache")) for name in graph.initializers
        )
