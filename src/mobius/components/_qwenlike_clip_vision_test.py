# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Executable tiny parity tests for Qwen-like CLIP sidecar components."""

from __future__ import annotations

import math

import numpy as np
import onnx_ir as ir
import torch
import torch.nn.functional as torch_functional

from mobius._testing import create_test_builder, create_test_input
from mobius._testing.ort_inference import OnnxModelSession
from mobius.components._qwen25_vl_vision import Qwen25VLVisionRotaryEmbedding
from mobius.components._qwenlike_clip_vision import (
    DualTemporalPatchEmbedding,
    Exaone45VisionSidecar,
    FusedQKVVisionAttention,
    GroupedQueryVisionAttention,
    KimiK25VisionSidecar,
    KimiVLVisionSidecar,
    LearnedPositionGrid3D,
    PatchMergeMLPProjector,
    SplitVisionRotaryEmbedding,
)


def _randomize(module: object, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    state: dict[str, np.ndarray] = {}
    for name, parameter in module.named_parameters():
        if parameter.const_value is not None:
            values = parameter.const_value.numpy()
        else:
            shape = tuple(int(dim) for dim in parameter.shape)
            if name.endswith(("input_norm.weight", "norm.weight")):
                values = (1.0 + rng.normal(0.0, 0.05, shape)).astype(np.float32)
            else:
                values = rng.normal(0.0, 0.12, shape).astype(np.float32)
            parameter.const_value = ir.tensor(values)
        state[name] = values
    return state


def _run(module: object, feeds: dict[str, np.ndarray], call):
    builder, op, graph = create_test_builder()
    values = {
        name: create_test_input(
            builder,
            name,
            list(value.shape),
            dtype=ir.DataType.INT64 if value.dtype == np.int64 else ir.DataType.FLOAT,
        )
        for name, value in feeds.items()
    }
    output = call(op, values)
    output.name = "output"
    graph.outputs.append(output)
    state = _randomize(module, seed=11)
    model = ir.Model(graph, ir_version=11)
    actual = OnnxModelSession(model, device="cpu").run(feeds)["output"]
    return actual, state, graph


def _linear(value: torch.Tensor, state: dict[str, np.ndarray], stem: str) -> torch.Tensor:
    bias = state.get(f"{stem}.bias")
    return torch_functional.linear(
        value,
        torch.from_numpy(state[f"{stem}.weight"]),
        None if bias is None else torch.from_numpy(bias),
    )


def _layer_norm(
    value: torch.Tensor, state: dict[str, np.ndarray], stem: str, eps: float
) -> torch.Tensor:
    return torch_functional.layer_norm(
        value,
        (value.shape[-1],),
        torch.from_numpy(state[f"{stem}.weight"]),
        torch.from_numpy(state[f"{stem}.bias"]),
        eps,
    )


def test_dual_temporal_patch_embedding_matches_two_convolutions():
    module = DualTemporalPatchEmbedding(hidden_size=5, in_channels=2, patch_size=2)
    rng = np.random.default_rng(3)
    patches = rng.normal(size=(3, 2, 2, 2, 2)).astype(np.float32)
    flat = patches.reshape(3, -1)
    actual, state, _ = _run(
        module,
        {"pixel_values": flat},
        lambda op, value: module(op, value["pixel_values"]),
    )
    expected = torch_functional.conv2d(
        torch.from_numpy(patches[:, 0]),
        torch.from_numpy(state["weight_0"]),
        stride=2,
    ) + torch_functional.conv2d(
        torch.from_numpy(patches[:, 1]),
        torch.from_numpy(state["weight_1"]),
        stride=2,
    )
    np.testing.assert_allclose(actual, expected.reshape(3, 5), rtol=1e-5, atol=1e-5)


def _split_rotate(value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    half = value.shape[-1] // 2
    first, second = value[..., :half], value[..., half:]
    return torch.cat(
        (
            first * cos[:, None, :half] - second * sin[:, None, :half],
            first * sin[:, None, half:] + second * cos[:, None, half:],
        ),
        dim=-1,
    )


def test_grouped_query_attention_matches_reference_with_2d_rope():
    hidden_size, heads, kv_heads = 32, 4, 2
    attention = GroupedQueryVisionAttention(hidden_size, heads, kv_heads)
    rope = Qwen25VLVisionRotaryEmbedding(hidden_size // heads // 2)

    class AttentionWithRoPE(torch.nn.Module):
        def named_parameters(self):
            yield from attention.named_parameters()
            yield from rope.named_parameters()

    owner = AttentionWithRoPE()
    rng = np.random.default_rng(7)
    hidden = rng.normal(size=(4, hidden_size)).astype(np.float32)
    positions = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    actual, state, _ = _run(
        owner,
        {"hidden_states": hidden, "position_ids": positions},
        lambda op, value: attention(
            op,
            value["hidden_states"],
            *rope(op, value["position_ids"]),
        ),
    )

    x = torch.from_numpy(hidden)
    qkv = _linear(x, state, "qkv")
    head_dim = hidden_size // heads
    q = qkv[:, :hidden_size].reshape(4, heads, head_dim)
    k = qkv[:, hidden_size : hidden_size + kv_heads * head_dim].reshape(4, kv_heads, head_dim)
    v = qkv[:, hidden_size + kv_heads * head_dim :].reshape(4, kv_heads, head_dim)
    table = torch.from_numpy(state["freq_table"])
    pos = torch.from_numpy(positions)
    half_freq = torch.cat((table[pos[:, 0]], table[pos[:, 1]]), dim=-1)
    frequencies = torch.cat((half_freq, half_freq), dim=-1)
    q = _split_rotate(q, frequencies.cos(), frequencies.sin())
    k = _split_rotate(k, frequencies.cos(), frequencies.sin())
    # GQA repeats each KV head over its contiguous query-head group.
    k = k.repeat_interleave(heads // kv_heads, dim=1)
    v = v.repeat_interleave(heads // kv_heads, dim=1)
    scores = torch.einsum("nhd,mhd->hnm", q, k) / math.sqrt(head_dim)
    context = torch.einsum("hnm,mhd->nhd", scores.softmax(dim=-1), v)
    expected = _linear(context.reshape(4, hidden_size), state, "proj")
    np.testing.assert_allclose(actual, expected.numpy(), rtol=2e-5, atol=2e-5)


def test_kimik25_converted_qk_layout_rotates_adjacent_split_axis_pairs():
    attention = FusedQKVVisionAttention(hidden_size=8, num_heads=2)
    rope = SplitVisionRotaryEmbedding(head_dim=4)

    class RotaryPath:
        def named_parameters(self):
            yield from attention.named_parameters()
            yield from rope.named_parameters()

    owner = RotaryPath()
    values = np.arange(2 * 2 * 4, dtype=np.float32).reshape(2, 2, 4) / 7
    positions = np.array([[1, 2], [3, 4]], dtype=np.int64)
    actual, state, _ = _run(
        owner,
        {"values": values, "position_ids": positions},
        lambda op, value: attention._rotate(
            op,
            value["values"],
            *rope(op, value["position_ids"]),
        ),
    )
    inv = torch.from_numpy(state["inv_freq"])
    pos = torch.from_numpy(positions).float()
    # Converted order is all X complex pairs followed by all Y pairs.
    scalar_freq = torch.cat((pos[:, 1:2] * inv, pos[:, 0:1] * inv), dim=-1)
    source = torch.from_numpy(values).reshape(2, 2, 2, 2)
    complex_source = torch.view_as_complex(source)
    expected = torch.view_as_real(
        complex_source * torch.polar(torch.ones_like(scalar_freq), scalar_freq)[:, None]
    ).reshape(2, 2, 4)
    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-5, atol=1e-5)


def test_patch_merge_norm_and_projector_order_matches_reference():
    module = PatchMergeMLPProjector(
        hidden_size=3,
        projector_hidden_size=7,
        output_size=5,
        merge_size=2,
        norm_eps=1e-5,
    )
    hidden = np.arange(4 * 4 * 3, dtype=np.float32).reshape(16, 3) / 17
    actual, state, _ = _run(
        module,
        {
            "hidden_states": hidden,
            "grid_height": np.array(4, dtype=np.int64),
            "grid_width": np.array(4, dtype=np.int64),
        },
        lambda op, value: module(
            op,
            value["hidden_states"],
            value["grid_height"],
            value["grid_width"],
        ),
    )
    reference = _layer_norm(torch.from_numpy(hidden), state, "input_norm", 1e-5)
    # Row-major H,W patches become [hm,wm,h_inner,w_inner,C] before flattening.
    reference = reference.reshape(2, 2, 2, 2, 3).permute(0, 2, 1, 3, 4).reshape(4, 12)
    reference = _linear(
        torch_functional.gelu(_linear(reference, state, "linear_1")),
        state,
        "linear_2",
    )
    np.testing.assert_allclose(actual, reference.numpy(), rtol=2e-5, atol=2e-5)


def test_kimik25_cwh_position_resize_matches_torch_bicubic():
    module = LearnedPositionGrid3D(hidden_size=2, stored_height=2, stored_width=3)
    actual, state, _ = _run(
        module,
        {
            "grid_height": np.array(3, dtype=np.int64),
            "grid_width": np.array(2, dtype=np.int64),
        },
        lambda op, value: module(op, value["grid_height"], value["grid_width"]),
    )
    # Stored layout is C,W,H, not C,H,W.
    table = torch.from_numpy(state["position_embeddings"]).permute(0, 2, 1)[None]
    expected = (
        torch_functional.interpolate(table, size=(3, 2), mode="bicubic", align_corners=False)
        .reshape(2, -1)
        .t()
    )
    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-4, atol=1e-4)


def _tiny_kimi(sidecar_type):
    return sidecar_type(
        depth=1,
        hidden_size=8,
        intermediate_size=12,
        num_heads=2,
        patch_size=2,
        in_channels=3,
        stored_height=2,
        stored_width=2,
        projector_hidden_size=16,
        output_size=6,
    )


def test_kimik25_and_kimivl_sidecars_execute_distinct_qkv_paths():
    pixels = np.random.default_rng(17).normal(size=(1, 3, 4, 4)).astype(np.float32)
    for sidecar_type, expected_qkv_name in (
        (KimiK25VisionSidecar, "layers.0.attn.qkv.weight"),
        (KimiVLVisionSidecar, "layers.0.attn.q_proj.weight"),
    ):
        module = _tiny_kimi(sidecar_type)
        actual, state, graph = _run(
            module,
            {"pixel_values": pixels},
            lambda op, value, module=module: module(op, value["pixel_values"]),
        )
        assert expected_qkv_name in state
        assert actual.shape == (1, 6)
        assert np.isfinite(actual).all()
        assert sum(1 for node in graph if node.op_type == "Attention") == 1


def test_exaone_sidecar_executes_dual_patch_gqa_window_schedule():
    module = Exaone45VisionSidecar(
        depth=2,
        hidden_size=16,
        intermediate_size=24,
        num_heads=2,
        num_kv_heads=1,
        patch_size=1,
        in_channels=3,
        output_size=10,
        fullatt_block_indexes=[1],
        window_size=4,
    )
    pixels = np.random.default_rng(23).normal(size=(16, 6)).astype(np.float32)
    grid = np.array([[1, 4, 4]], dtype=np.int64)
    actual, _, graph = _run(
        module,
        {"pixel_values": pixels, "image_grid_thw": grid},
        lambda op, value: module(op, value["pixel_values"], value["image_grid_thw"]),
    )
    names = {name for name, _ in module.named_parameters()}
    assert "patch_embed.weight_0" in names
    assert "patch_embed.weight_1" in names
    first_block = next(iter(module.blocks))
    assert first_block.attn.num_heads == 2
    assert first_block.attn.num_kv_heads == 1
    assert actual.shape == (4, 10)
    assert np.isfinite(actual).all()
    assert sum(1 for node in graph if node.op_type == "Attention") == 2
    assert any(node.op_type == "Scan" for node in graph)
