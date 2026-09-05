# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Executable tiny parity tests for the MiMoVL and MiniMax-M3 sidecar math."""

from __future__ import annotations

from collections.abc import Mapping

import ml_dtypes
import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import torch
import torch.nn.functional as torch_functional

from mobius._testing import create_test_builder, create_test_input
from mobius._testing.ort_inference import _numpy_to_ort_value, _ort_value_to_numpy
from mobius.components._mimo_minimax_vision import (
    DualTemporalPatchEmbedding,
    F32AccumulationLinear,
    MergeUnitReorder,
    MiMoVLAttentionCore,
    MiMoVLBlock,
    MiMoVLProjector,
    MiMoVLRotaryEmbedding,
    MiniMaxM3PartialRotaryEmbedding,
    MiniMaxM3Projector,
    MiniMaxM3VisionBlock,
    MiniMaxM3VisionSidecar,
    SpatialMergeOrder,
    minimax_m3_qk_permutation,
)


def _session(
    module,
    input_specs: Mapping[str, tuple[list[int], ir.DataType]],
    parameters: Mapping[str, np.ndarray] | None = None,
) -> ort.InferenceSession:
    builder, op, graph = create_test_builder()
    inputs = {
        name: create_test_input(builder, name, shape, dtype=dtype)
        for name, (shape, dtype) in input_specs.items()
    }
    output = module(op, **inputs)
    output.name = "output"
    graph.outputs.append(output)
    if parameters is not None:
        for name, parameter in module.named_parameters():
            parameter.const_value = ir.tensor(parameters[name])
    model = ir.Model(graph, ir_version=11)
    serialized = ir.serde.serialize_model(model)
    return ort.InferenceSession(
        serialized.SerializeToString(), providers=["CPUExecutionProvider"]
    )


def _random_parameters(module, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        name: rng.normal(0.0, 0.2, tuple(int(dim) for dim in parameter.shape)).astype(
            np.float32
        )
        for name, parameter in module.named_parameters()
    }


def _torch_linear(
    x: torch.Tensor, state: Mapping[str, np.ndarray], prefix: str
) -> torch.Tensor:
    bias = state.get(f"{prefix}.bias")
    return torch_functional.linear(
        x,
        torch.from_numpy(state[f"{prefix}.weight"]),
        None if bias is None else torch.from_numpy(bias),
    )


def test_dual_temporal_patch_halves_execute_like_two_conv2d_slices():
    module = DualTemporalPatchEmbedding(in_channels=1, hidden_size=2, patch_size=2)
    state = {
        "weight_0": np.array(
            [[[[1.0, 2.0], [3.0, 4.0]]], [[[0.5, -1.0], [2.0, 0.0]]]],
            dtype=np.float32,
        ),
        "weight_1": np.array(
            [[[[0.0, 1.0], [-1.0, 2.0]]], [[[3.0, 1.0], [0.0, -2.0]]]],
            dtype=np.float32,
        ),
    }
    patches = np.arange(1, 17, dtype=np.float32).reshape(2, 8)
    session = _session(
        module,
        {"pixel_patches": ([2, 8], ir.DataType.FLOAT)},
        state,
    )
    actual = session.run(None, {"pixel_patches": patches})[0]

    temporal = torch.from_numpy(patches).reshape(2, 2, 1, 2, 2)
    expected = torch_functional.conv2d(temporal[:, 0], torch.from_numpy(state["weight_0"]))
    expected += torch_functional.conv2d(temporal[:, 1], torch.from_numpy(state["weight_1"]))
    np.testing.assert_allclose(actual, expected.reshape(2, 2).numpy(), rtol=1e-6)


def test_mimovl_window_bias_and_sink_change_softmax_denominator():
    module = MiMoVLAttentionCore(num_query_heads=2, num_kv_heads=1, head_dim=1)
    specs = {
        "query": ([2, 2, 1], ir.DataType.FLOAT),
        "key": ([2, 1, 1], ir.DataType.FLOAT),
        "value": ([2, 1, 1], ir.DataType.FLOAT),
        "attention_bias": ([2, 2], ir.DataType.FLOAT),
        "sinks": ([2], ir.DataType.FLOAT),
    }
    session = _session(module, specs)
    feeds = {
        "query": np.zeros((2, 2, 1), dtype=np.float32),
        "key": np.zeros((2, 1, 1), dtype=np.float32),
        "value": np.array([[[1.0]], [[3.0]]], dtype=np.float32),
        "attention_bias": np.array([[0.0, -100.0], [-100.0, 0.0]], dtype=np.float32),
        "sinks": np.zeros(2, dtype=np.float32),
    }
    actual = session.run(None, feeds)[0]

    # Each query sees one real logit and one equal sink logit whose value is zero.
    expected = np.array([[[0.5], [0.5]], [[1.5], [1.5]]], dtype=np.float32)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_mimovl_two_dimensional_rope_uses_row_and_column_sections():
    module = MiMoVLRotaryEmbedding(head_dim=8, max_grid_size=8)
    session = _session(
        module,
        {
            "hidden_states": ([2, 1, 8], ir.DataType.FLOAT),
            "position_ids": ([2, 2], ir.DataType.INT64),
        },
        {
            name: np.asarray(parameter.const_value)
            for name, parameter in module.named_parameters()
        },
    )
    states = np.arange(1, 17, dtype=np.float32).reshape(2, 1, 8)
    positions = np.array([[1, 2], [3, 1]], dtype=np.int64)
    actual = session.run(None, {"hidden_states": states, "position_ids": positions})[0]

    inv = 1.0 / (10000.0 ** (np.arange(2, dtype=np.float32) * 2.0 / 4.0))
    angles = np.concatenate([positions[:, :1] * inv, positions[:, 1:] * inv], axis=1)[
        :, None, :
    ]
    expected = np.concatenate(
        [
            states[..., :4] * np.cos(angles) - states[..., 4:] * np.sin(angles),
            states[..., :4] * np.sin(angles) + states[..., 4:] * np.cos(angles),
        ],
        axis=-1,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_merge_unit_reorder_preserves_internal_patch_order_and_is_invertible():
    module = MergeUnitReorder(hidden_size=1)
    session = _session(
        module,
        {
            "hidden_states": ([12, 1], ir.DataType.FLOAT),
            "unit_indices": ([3], ir.DataType.INT64),
        },
    )
    tokens = np.arange(12, dtype=np.float32).reshape(12, 1)
    column = np.array([2, 0, 1], dtype=np.int64)
    reordered = session.run(None, {"hidden_states": tokens, "unit_indices": column})[0]
    np.testing.assert_array_equal(reordered[:, 0], [8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7])
    inverse = np.argsort(column)
    restored = session.run(None, {"hidden_states": reordered, "unit_indices": inverse})[0]
    np.testing.assert_array_equal(restored, tokens)


def test_mimovl_column_window_mode_reorders_units_and_restores_row_order():
    constructor = {
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_query_heads": 2,
        "num_kv_heads": 1,
        "head_dim": 4,
    }
    row_module = MiMoVLBlock(**constructor, window_mode=0)
    column_module = MiMoVLBlock(**constructor, window_mode=1)
    state = _random_parameters(row_module, seed=23)
    specs = {
        "hidden_states": ([8, 8], ir.DataType.FLOAT),
        "row_position_ids": ([8, 2], ir.DataType.INT64),
        "column_position_ids": ([8, 2], ir.DataType.INT64),
        "window_bias": ([8, 8], ir.DataType.FLOAT),
        "column_indices": ([2], ir.DataType.INT64),
        "inverse_column_indices": ([2], ir.DataType.INT64),
    }
    row_session = _session(row_module, specs, state)
    column_session = _session(column_module, specs, state)
    feeds = {
        "hidden_states": np.arange(64, dtype=np.float32).reshape(8, 8) / 31.0,
        "row_position_ids": np.zeros((8, 2), dtype=np.int64),
        "column_position_ids": np.zeros((8, 2), dtype=np.int64),
        "window_bias": np.zeros((8, 8), dtype=np.float32),
        "column_indices": np.array([1, 0], dtype=np.int64),
        "inverse_column_indices": np.array([1, 0], dtype=np.int64),
    }
    row = row_session.run(None, feeds)[0]
    column = column_session.run(None, feeds)[0]
    # With a permutation-invariant window, column mode must be the same layer
    # conjugated by the merge-unit permutation and restored on exit.
    np.testing.assert_allclose(column, row, rtol=2e-5, atol=2e-6)


def test_f32_accumulation_linear_executes_bfloat16_cast_path():
    module = F32AccumulationLinear(4, 1, bias=False)
    dtype = np.dtype(ml_dtypes.bfloat16)
    state = {"weight": np.array([[1.5, -2.25, 0.75, 4.0]], dtype=dtype)}
    session = _session(module, {"hidden_states": ([1, 4], ir.DataType.BFLOAT16)}, state)
    x = np.array([[1.25, -0.5, 3.0, 0.125]], dtype=dtype)
    outputs = session.run_with_ort_values(
        ["output"], {"hidden_states": _numpy_to_ort_value(x)}
    )
    actual = _ort_value_to_numpy(outputs[0])
    expected = (x.astype(np.float32) @ state["weight"].astype(np.float32).T).astype(dtype)
    np.testing.assert_array_equal(actual, expected)


def test_mimovl_post_ln_merge_four_projector_matches_torch():
    module = MiMoVLProjector(hidden_size=2, intermediate_size=3, output_size=2)
    state = _random_parameters(module, seed=3)
    # LayerNorm scale must not be centred around zero for a useful parity fixture.
    state["post_ln.weight"] += 1.0
    inputs = np.arange(16, dtype=np.float32).reshape(8, 2) / 7.0
    session = _session(module, {"hidden_states": ([8, 2], ir.DataType.FLOAT)}, state)
    actual = session.run(None, {"hidden_states": inputs})[0]

    expected = torch.from_numpy(inputs)
    expected = torch_functional.layer_norm(
        expected,
        (2,),
        torch.from_numpy(state["post_ln.weight"]),
        bias=None,
        eps=1e-6,
    ).reshape(2, 8)
    expected = _torch_linear(expected, state, "fc1")
    expected = _torch_linear(torch_functional.gelu(expected, approximate="none"), state, "fc2")
    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-5, atol=1e-6)


def test_minimax_converter_permutation_and_partial_two_axis_rope():
    assert minimax_m3_qk_permutation(12) == [0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11]
    module = MiniMaxM3PartialRotaryEmbedding(head_dim=12, max_grid_size=8)
    session = _session(
        module,
        {
            "hidden_states": ([1, 2, 1, 12], ir.DataType.FLOAT),
            "position_h": ([2], ir.DataType.INT64),
            "position_w": ([2], ir.DataType.INT64),
        },
        {
            name: np.asarray(parameter.const_value)
            for name, parameter in module.named_parameters()
        },
    )
    states = np.arange(1, 25, dtype=np.float32).reshape(1, 2, 1, 12)
    pos_h = np.array([1, 2], dtype=np.int64)
    pos_w = np.array([3, 1], dtype=np.int64)
    actual = session.run(
        None,
        {"hidden_states": states, "position_h": pos_h, "position_w": pos_w},
    )[0]

    def rotate(part: np.ndarray, positions: np.ndarray) -> np.ndarray:
        inv_freq = 1.0 / (10000.0 ** (np.arange(0, 4, 2, dtype=np.float32) / 4.0))
        angle = positions[None, :, None, None].astype(np.float32) * inv_freq
        return np.concatenate(
            [
                part[..., :2] * np.cos(angle) - part[..., 2:] * np.sin(angle),
                part[..., :2] * np.sin(angle) + part[..., 2:] * np.cos(angle),
            ],
            axis=-1,
        )

    expected = np.concatenate(
        [
            states[..., :4],
            rotate(states[..., 4:8], pos_h),
            rotate(states[..., 8:12], pos_w),
        ],
        axis=-1,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(actual[..., :4], states[..., :4])


def test_minimax_pre_vit_spatial_merge_orders_2x2_tiles():
    module = SpatialMergeOrder(4, 4, hidden_size=1, merge_size=2)
    session = _session(module, {"hidden_states": ([1, 16, 1], ir.DataType.FLOAT)})
    grid = np.arange(16, dtype=np.float32).reshape(1, 16, 1)
    actual = session.run(None, {"hidden_states": grid})[0]
    np.testing.assert_array_equal(
        actual[0, :, 0],
        [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15],
    )


def test_minimax_normal_norm_gelu_erf_vit_block_matches_torch():
    module = MiniMaxM3VisionBlock(
        hidden_size=12,
        intermediate_size=16,
        num_heads=1,
        norm_eps=1e-5,
    )
    state = _random_parameters(module, seed=17)
    state["norm1.weight"] += 1.0
    state["norm2.weight"] += 1.0
    inputs = np.arange(36, dtype=np.float32).reshape(1, 3, 12) / 19.0
    pos_h = np.array([0, 1, 2], dtype=np.int64)
    pos_w = np.array([2, 1, 0], dtype=np.int64)
    session = _session(
        module,
        {
            "hidden_states": ([1, 3, 12], ir.DataType.FLOAT),
            "position_h": ([3], ir.DataType.INT64),
            "position_w": ([3], ir.DataType.INT64),
        },
        state,
    )
    actual = session.run(
        None,
        {"hidden_states": inputs, "position_h": pos_h, "position_w": pos_w},
    )[0]

    def layer_norm(x: torch.Tensor, prefix: str) -> torch.Tensor:
        return torch_functional.layer_norm(
            x,
            (12,),
            torch.from_numpy(state[f"{prefix}.weight"]),
            torch.from_numpy(state[f"{prefix}.bias"]),
            1e-5,
        )

    def rope(x: torch.Tensor) -> torch.Tensor:
        values = x.numpy().reshape(1, 3, 1, 12)
        inv = 1.0 / (10000.0 ** (np.arange(0, 4, 2, dtype=np.float32) / 4.0))

        def rotate(part: np.ndarray, positions: np.ndarray) -> np.ndarray:
            angles = positions[None, :, None, None] * inv
            return np.concatenate(
                [
                    part[..., :2] * np.cos(angles) - part[..., 2:] * np.sin(angles),
                    part[..., :2] * np.sin(angles) + part[..., 2:] * np.cos(angles),
                ],
                axis=-1,
            )

        return torch.from_numpy(
            np.concatenate(
                [
                    values[..., :4],
                    rotate(values[..., 4:8], pos_h),
                    rotate(values[..., 8:12], pos_w),
                ],
                axis=-1,
            )
            .astype(np.float32)
            .reshape(1, 3, 12)
        )

    x = torch.from_numpy(inputs)
    normed = layer_norm(x, "norm1")
    query = rope(_torch_linear(normed, state, "attn.q_proj"))
    key = rope(_torch_linear(normed, state, "attn.k_proj"))
    value = _torch_linear(normed, state, "attn.v_proj")
    scores = torch.matmul(query, key.transpose(-1, -2)) / np.sqrt(12.0)
    attended = torch.matmul(torch.softmax(scores, dim=-1), value)
    x = x + _torch_linear(attended, state, "attn.out_proj")
    mlp = _torch_linear(layer_norm(x, "norm2"), state, "mlp.fc1")
    expected = x + _torch_linear(
        torch_functional.gelu(mlp, approximate="none"), state, "mlp.fc2"
    )
    np.testing.assert_allclose(actual, expected.numpy(), rtol=2e-5, atol=2e-6)


def test_minimax_per_patch_then_merger_mlp_matches_torch():
    module = MiniMaxM3Projector(
        hidden_size=2,
        patch_mlp_size=3,
        projected_size=2,
        merger_mlp_size=4,
        output_size=3,
    )
    state = _random_parameters(module, seed=9)
    inputs = np.arange(16, dtype=np.float32).reshape(1, 8, 2) / 11.0
    session = _session(module, {"hidden_states": ([1, 8, 2], ir.DataType.FLOAT)}, state)
    actual = session.run(None, {"hidden_states": inputs})[0]

    expected = _torch_linear(torch.from_numpy(inputs), state, "patch_mlp.fc1")
    expected = _torch_linear(
        torch_functional.gelu(expected, approximate="none"), state, "patch_mlp.fc2"
    )
    expected = expected.reshape(2, 8)
    expected = _torch_linear(expected, state, "merger_mlp.fc1")
    expected = _torch_linear(
        torch_functional.gelu(expected, approximate="none"), state, "merger_mlp.fc2"
    )
    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-5, atol=1e-6)


def test_minimax_full_sidecar_uses_runtime_grid_geometry():
    module = MiniMaxM3VisionSidecar(
        hidden_size=12,
        intermediate_size=16,
        num_heads=1,
        num_layers=0,
        patch_size=1,
        grid_height=2,
        grid_width=2,
        patch_mlp_size=12,
        projected_size=12,
        merger_mlp_size=16,
        output_size=8,
    )
    state = _random_parameters(module, seed=61)
    builder, op, graph = create_test_builder()
    tokens = ir.SymbolicDim("tokens")
    pixels = create_test_input(builder, "pixel_values", [tokens, 6])
    grid = create_test_input(builder, "grid_size", [2], dtype=ir.DataType.INT64)
    output = module(op, pixels, grid)
    output.name = "output"
    graph.outputs.append(output)
    for name, parameter in module.named_parameters():
        parameter.const_value = ir.tensor(state[name])
    model = ir.Model(graph, ir_version=11)
    session = ort.InferenceSession(
        ir.serde.serialize_model(model).SerializeToString(),
        providers=["CPUExecutionProvider"],
    )

    pixel_values = np.random.default_rng(62).normal(size=(8, 6)).astype(np.float32)
    actual = session.run(
        None,
        {
            "pixel_values": pixel_values,
            "grid_size": np.array([2, 4], dtype=np.int64),
        },
    )[0]
    pixels = torch.from_numpy(pixel_values)
    first_weight = torch.from_numpy(state["patch_embed.weight_0"])[:, :, 0, 0]
    second_weight = torch.from_numpy(state["patch_embed.weight_1"])[:, :, 0, 0]
    expected = torch_functional.linear(pixels[:, :3], first_weight)
    expected += torch_functional.linear(pixels[:, 3:], second_weight)
    expected = _torch_linear(expected, state, "projector.patch_mlp.fc1")
    expected = _torch_linear(
        torch_functional.gelu(expected, approximate="none"),
        state,
        "projector.patch_mlp.fc2",
    )
    expected = expected.reshape(2, 48)
    expected = _torch_linear(expected, state, "projector.merger_mlp.fc1")
    expected = _torch_linear(
        torch_functional.gelu(expected, approximate="none"),
        state,
        "projector.merger_mlp.fc2",
    )
    assert actual.shape == (2, 8)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-5, atol=1e-6)
