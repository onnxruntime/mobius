# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Executable parity tests for the HunyuanVL and Step3VL CLIP sidecars."""

from __future__ import annotations

import math

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import torch
import torch.nn.functional as functional

from mobius._testing import create_test_builder, create_test_input
from mobius.components._hunyuan_step_vision import (
    HunyuanVLClipSidecar,
    Step3VLClipSidecar,
    _Step3VLAttention,
)


def _state_and_session(component, graph, seed: int):
    rng = np.random.default_rng(seed)
    state: dict[str, np.ndarray] = {}
    for name, parameter in component.named_parameters():
        shape = tuple(int(dim) for dim in parameter.shape)
        if name.endswith("norm.weight"):
            values = rng.uniform(0.7, 1.3, shape).astype(np.float32)
        else:
            values = rng.normal(0.0, 0.2, shape).astype(np.float32)
        parameter.const_value = ir.tensor(values)
        state[name] = values
    model = ir.Model(graph, ir_version=11)
    proto = ir.serde.serialize_model(model)
    return state, ort.InferenceSession(
        proto.SerializeToString(), providers=["CPUExecutionProvider"]
    )


def _conv(x, state, stem, stride, padding):
    return functional.conv2d(
        x,
        torch.from_numpy(state[f"{stem}.weight"]),
        (torch.from_numpy(state[f"{stem}.bias"]) if f"{stem}.bias" in state else None),
        stride=stride,
        padding=padding,
    )


def _linear(x, state, stem, bias=True):
    return functional.linear(
        x,
        torch.from_numpy(state[f"{stem}.weight"]),
        torch.from_numpy(state[f"{stem}.bias"]) if bias else None,
    )


def _rms(x, weight, eps=1e-5):
    return x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps) * weight


def _layer_norm(x, state, stem):
    return functional.layer_norm(
        x,
        (x.shape[-1],),
        torch.from_numpy(state[f"{stem}.weight"]),
        torch.from_numpy(state[f"{stem}.bias"]),
        1e-5,
    )


def _rope_axis(x, positions, theta):
    pairs = x.shape[-1] // 2
    paired = x.reshape(*x.shape[:-1], pairs, 2)
    frequency = theta ** (-torch.arange(pairs, dtype=torch.float32) / pairs)
    angles = torch.from_numpy(positions).float().unsqueeze(-1) * frequency
    cos = angles.cos().reshape(1, 1, -1, pairs)
    sin = angles.sin().reshape(1, 1, -1, pairs)
    even, odd = paired[..., 0], paired[..., 1]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def _attention(x, state, stem, *, pos_h=None, pos_w=None, theta=10000.0):
    hidden_size = x.shape[-1]
    num_heads = 1 if hidden_size == 4 else 2
    head_dim = hidden_size // num_heads
    qkv = _linear(x, state, f"{stem}.in_proj").reshape(
        x.shape[0], x.shape[1], 3, num_heads, head_dim
    )
    q, k, v = qkv.permute(2, 0, 3, 1, 4)
    if pos_h is not None and pos_w is not None:
        q = torch.cat(
            (
                _rope_axis(q[..., : head_dim // 2], pos_w, theta),
                _rope_axis(q[..., head_dim // 2 :], pos_h, theta),
            ),
            dim=-1,
        )
        k = torch.cat(
            (
                _rope_axis(k[..., : head_dim // 2], pos_w, theta),
                _rope_axis(k[..., head_dim // 2 :], pos_h, theta),
            ),
            dim=-1,
        )
    scores = torch.matmul(q, k.transpose(-1, -2)) * head_dim**-0.5
    context = torch.matmul(scores.softmax(dim=-1), v)
    context = context.transpose(1, 2).reshape(x.shape)
    return _linear(context, state, f"{stem}.out_proj")


def test_hunyuanvl_position_interpolation_projector_and_token_order():
    """Position resize and ``(ow+1)*oh+2`` ordering match pinned mtmd."""
    component = HunyuanVLClipSidecar(
        vision_hidden_size=4,
        intermediate_size=7,
        num_heads=1,
        num_layers=1,
        patch_size=1,
        grid_height=4,
        grid_width=4,
        position_grid_size=2,
        projector_hidden_size=5,
        output_size=3,
    )
    builder, op, graph = create_test_builder()
    pixels_value = create_test_input(builder, "pixel_values", [1, 3, 4, 4])
    output = component(op, pixels_value)
    output.name = "image_features"
    graph.outputs.append(output)
    state, session = _state_and_session(component, graph, 11)

    rng = np.random.default_rng(12)
    pixels = rng.normal(size=(1, 3, 4, 4)).astype(np.float32)
    source_positions = state["position_embedding"].reshape(1, 2, 2, 4).transpose(0, 3, 1, 2)
    positions_t = (
        functional.interpolate(
            torch.from_numpy(source_positions),
            size=(4, 4),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        .permute(0, 2, 3, 1)
        .reshape(1, 16, 4)
    )
    actual = session.run(None, {"pixel_values": pixels})[0]
    x = _conv(torch.from_numpy(pixels), state, "patch_embedding.proj", 1, 0)
    x = x.flatten(2).transpose(1, 2) + positions_t
    residual = x
    x = _layer_norm(x, state, "layers.0.norm1")
    x = residual + _attention(x, state, "layers.0.attn")
    residual = x
    x = _layer_norm(x, state, "layers.0.norm2")
    x = _linear(x, state, "layers.0.mlp_up")
    x = functional.gelu(x, approximate="tanh")
    x = residual + _linear(x, state, "layers.0.mlp_down")
    x = _rms(x, torch.from_numpy(state["pre_projector_norm.weight"]))
    x = x.reshape(1, 4, 4, 4).permute(0, 3, 1, 2)
    x = functional.gelu(_conv(x, state, "projector_conv1", 2, 0), approximate="tanh")
    x = _conv(x, state, "projector_conv2", 1, 0).permute(0, 2, 3, 1)
    newline = torch.from_numpy(state["image_newline"]).reshape(1, 1, 1, 10)
    x = torch.cat((x, newline.expand(1, 2, 1, 10)), dim=2).reshape(1, 6, 10)
    x = _linear(x, state, "projector")
    x = torch.cat(
        (
            torch.from_numpy(state["image_begin"]).reshape(1, 1, 3),
            x,
            torch.from_numpy(state["image_end"]).reshape(1, 1, 3),
        ),
        dim=1,
    )
    expected = _rms(x, torch.from_numpy(state["post_projector_norm.weight"])).numpy()

    assert actual.shape == (1, (2 + 1) * 2 + 2, 3)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_step3vl_resized_positions_and_biased_conv_downsamplers():
    """Learned-position interpolation and both k3s2p1 convs execute exactly."""
    component = Step3VLClipSidecar(
        vision_hidden_size=8,
        intermediate_size=8,
        num_heads=2,
        num_layers=1,
        patch_size=1,
        grid_height=5,
        grid_width=7,
        position_grid_size=3,
        downsample_hidden_size=5,
        output_size=3,
    )
    builder, op, graph = create_test_builder()
    pixels_value = create_test_input(builder, "pixel_values", [1, 3, 5, 7])
    pos_h_value = create_test_input(builder, "pos_h", [35], dtype=ir.DataType.INT64)
    pos_w_value = create_test_input(builder, "pos_w", [35], dtype=ir.DataType.INT64)
    output = component(op, pixels_value, pos_h_value, pos_w_value)
    output.name = "image_features"
    graph.outputs.append(output)
    state, session = _state_and_session(component, graph, 21)

    rng = np.random.default_rng(22)
    pixels = rng.normal(size=(1, 3, 5, 7)).astype(np.float32)
    pos_h, pos_w = np.indices((5, 7), dtype=np.int64)
    actual = session.run(
        None,
        {
            "pixel_values": pixels,
            "pos_h": pos_h.reshape(-1),
            "pos_w": pos_w.reshape(-1),
        },
    )[0]

    x = _conv(torch.from_numpy(pixels), state, "patch_embedding.proj", 1, 0)
    x = x.flatten(2).transpose(1, 2)
    learned = torch.from_numpy(state["position_embedding"]).reshape(3, 3, 8)
    learned = (
        functional.interpolate(
            learned.permute(2, 0, 1).unsqueeze(0),
            size=(5, 7),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        .permute(0, 2, 3, 1)
        .reshape(1, 35, 8)
    )
    x = _layer_norm(x + learned, state, "pre_layer_norm")
    residual = x
    x = _layer_norm(x, state, "layers.0.norm1")
    x = residual + _attention(
        x,
        state,
        "layers.0.attn",
        pos_h=pos_h.reshape(-1),
        pos_w=pos_w.reshape(-1),
    ) * torch.from_numpy(state["layers.0.ls_1"])
    residual = x
    x = _layer_norm(x, state, "layers.0.norm2")
    x = _linear(x, state, "layers.0.mlp_up")
    x = x * torch.sigmoid(1.702 * x)
    x = residual + _linear(x, state, "layers.0.mlp_down") * torch.from_numpy(
        state["layers.0.ls_2"]
    )
    x = x.reshape(1, 5, 7, 8).permute(0, 3, 1, 2)
    x = _conv(x, state, "downsample1", 2, 1)
    x = _conv(x, state, "downsample2", 2, 1)
    x = x.permute(0, 2, 3, 1).reshape(1, 4, 10)
    expected = _linear(x, state, "projector", bias=False).numpy()

    assert actual.shape == (1, 4, 3)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_step3vl_axial_rope_attention_matches_torch():
    """The first head half uses W and the second uses H, with adjacent pairs."""
    component = _Step3VLAttention(hidden_size=8, num_heads=2, rope_theta=100.0)
    builder, op, graph = create_test_builder()
    hidden_value = create_test_input(builder, "hidden_states", [1, 6, 8])
    pos_h_value = create_test_input(builder, "pos_h", [6], dtype=ir.DataType.INT64)
    pos_w_value = create_test_input(builder, "pos_w", [6], dtype=ir.DataType.INT64)
    output = component(op, hidden_value, pos_h_value, pos_w_value)
    output.name = "attention_output"
    graph.outputs.append(output)
    state, session = _state_and_session(component, graph, 31)

    rng = np.random.default_rng(32)
    hidden = rng.normal(size=(1, 6, 8)).astype(np.float32)
    pos_h = np.array([0, 0, 1, 1, 3, 2], dtype=np.int64)
    pos_w = np.array([0, 2, 0, 2, 1, 4], dtype=np.int64)
    actual = session.run(None, {"hidden_states": hidden, "pos_h": pos_h, "pos_w": pos_w})[0]

    x = torch.from_numpy(hidden)
    qkv = _linear(x, state, "in_proj").reshape(1, 6, 3, 2, 4)
    q, k, v = qkv.permute(2, 0, 3, 1, 4)
    q = torch.cat(
        (_rope_axis(q[..., :2], pos_w, 100.0), _rope_axis(q[..., 2:], pos_h, 100.0)), dim=-1
    )
    k = torch.cat(
        (_rope_axis(k[..., :2], pos_w, 100.0), _rope_axis(k[..., 2:], pos_h, 100.0)), dim=-1
    )
    scores = torch.matmul(q, k.transpose(-1, -2)) * 0.5
    context = torch.matmul(scores.softmax(dim=-1), v)
    context = context.transpose(1, 2).reshape(1, 6, 8)
    expected = _linear(context, state, "out_proj").numpy()

    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)


def test_hunyuanvl_runtime_grid_controls_newline_token_count():
    component = HunyuanVLClipSidecar(
        vision_hidden_size=4,
        intermediate_size=7,
        num_heads=1,
        num_layers=0,
        patch_size=1,
        grid_height=4,
        grid_width=4,
        position_grid_size=2,
        projector_hidden_size=5,
        output_size=3,
    )
    builder, op, graph = create_test_builder()
    height = ir.SymbolicDim("height")
    width = ir.SymbolicDim("width")
    pixels = create_test_input(builder, "pixel_values", [1, 3, height, width])
    output = component(op, pixels)
    output.name = "image_features"
    graph.outputs.append(output)
    _, session = _state_and_session(component, graph, 41)

    rng = np.random.default_rng(42)
    for h, w in ((4, 4), (2, 4)):
        actual = session.run(
            None,
            {"pixel_values": rng.normal(size=(1, 3, h, w)).astype(np.float32)},
        )[0]
        assert actual.shape == (1, (w // 2 + 1) * (h // 2) + 2, 3)


def test_hunyuanvl_position_downsampling_matches_torch_antialias():
    class _PositionResizeProbe(HunyuanVLClipSidecar):
        def forward(self, op, pixel_values):
            del pixel_values
            return self._resize_positions(
                op,
                op.Constant(value_int=2),
                op.Constant(value_int=2),
            )

    component = _PositionResizeProbe(
        vision_hidden_size=4,
        intermediate_size=7,
        num_heads=1,
        num_layers=0,
        patch_size=1,
        grid_height=4,
        grid_width=4,
        position_grid_size=4,
        projector_hidden_size=5,
        output_size=3,
    )
    _, op, graph = create_test_builder()
    output = component(op, op.Constant(value_float=0.0))
    output.name = "positions"
    graph.outputs.append(output)
    state, session = _state_and_session(component, graph, 43)

    source = torch.from_numpy(state["position_embedding"]).reshape(1, 4, 4, 4)
    expected = functional.interpolate(
        source.permute(0, 3, 1, 2),
        size=(2, 2),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).permute(0, 2, 3, 1)
    actual = session.run(None, {})[0]
    np.testing.assert_allclose(actual, expected.reshape(1, 4, 4).numpy(), rtol=1e-6)


def test_step3vl_runtime_grid_controls_position_resize_and_downsampling():
    component = Step3VLClipSidecar(
        vision_hidden_size=8,
        intermediate_size=8,
        num_heads=2,
        num_layers=0,
        patch_size=1,
        grid_height=4,
        grid_width=4,
        position_grid_size=2,
        downsample_hidden_size=5,
        output_size=3,
    )
    builder, op, graph = create_test_builder()
    height = ir.SymbolicDim("height")
    width = ir.SymbolicDim("width")
    patches = ir.SymbolicDim("patches")
    pixels = create_test_input(builder, "pixel_values", [1, 3, height, width])
    pos_h = create_test_input(builder, "pos_h", [patches], dtype=ir.DataType.INT64)
    pos_w = create_test_input(builder, "pos_w", [patches], dtype=ir.DataType.INT64)
    output = component(op, pixels, pos_h, pos_w)
    output.name = "image_features"
    graph.outputs.append(output)
    _, session = _state_and_session(component, graph, 43)

    rng = np.random.default_rng(44)
    for h, w in ((4, 4), (4, 8)):
        rows, columns = np.indices((h, w), dtype=np.int64)
        actual = session.run(
            None,
            {
                "pixel_values": rng.normal(size=(1, 3, h, w)).astype(np.float32),
                "pos_h": rows.reshape(-1),
                "pos_w": columns.reshape(-1),
            },
        )[0]
        assert actual.shape == (1, math.ceil(h / 4) * math.ceil(w / 4), 3)
