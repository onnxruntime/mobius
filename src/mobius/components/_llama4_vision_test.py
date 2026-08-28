# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import torch
import torch.nn.functional as torch_functional

from mobius._configs import VisionConfig
from mobius._testing import create_test_builder, create_test_input
from mobius.components._llama4_vision import Llama4VisionTower


def _tiny_tower() -> Llama4VisionTower:
    return Llama4VisionTower(
        VisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=4,
            patch_size=2,
            norm_eps=1e-5,
            rope_theta=10_000.0,
        )
    )


def test_parameter_inventory_matches_serialized_families() -> None:
    names = {name for name, _ in _tiny_tower().named_parameters()}
    assert "embeddings.patch_embedding" in names
    assert "embeddings.class_embedding" in names
    assert "embeddings.position_embedding" in names
    assert "encoder.0.attn.q_proj.weight" in names
    assert "encoder.0.mlp.up_proj.weight" in names
    assert "pre_layernorm.weight" in names
    assert "post_layernorm.bias" in names


def test_tiny_tower_executes_nonzero_pixels() -> None:
    tower = _tiny_tower()
    builder, op, graph = create_test_builder()
    pixels = create_test_input(builder, "pixel_values", [1, 3, 4, 4])
    output = tower(op, pixels)
    output.name = "image_features"
    graph.outputs.append(output)

    rng = np.random.default_rng(9)
    for name, parameter in tower.named_parameters():
        if parameter.const_value is not None:
            continue
        shape = tuple(int(dim) for dim in parameter.shape)
        if name.endswith(".weight") and "layernorm" in name:
            values = np.ones(shape, dtype=np.float32)
        else:
            values = rng.normal(0.0, 0.1, shape).astype(np.float32)
        parameter.const_value = ir.tensor(values)

    session = ort.InferenceSession(
        ir.serde.serialize_model(ir.Model(graph, ir_version=11)).SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    (actual,) = session.run(
        None,
        {"pixel_values": np.linspace(-1, 1, 48, dtype=np.float32).reshape(1, 3, 4, 4)},
    )

    assert actual.shape == (1, 4, 16)
    assert np.isfinite(actual).all()
    assert np.any(actual != 0)


def test_tiny_tower_matches_independent_torch_reference() -> None:
    tower = _tiny_tower()
    builder, op, graph = create_test_builder()
    pixels_value = create_test_input(builder, "pixel_values", [1, 3, 4, 4])
    output = tower(op, pixels_value)
    output.name = "image_features"
    graph.outputs.append(output)

    rng = np.random.default_rng(33)
    state: dict[str, np.ndarray] = {}
    for name, parameter in tower.named_parameters():
        if parameter.const_value is not None:
            continue
        shape = tuple(int(dim) for dim in parameter.shape)
        if name.endswith(".weight") and ("layernorm" in name or ".ln" in name):
            values = (1.0 + rng.normal(0, 0.1, shape)).astype(np.float32)
        else:
            values = rng.normal(0, 0.1, shape).astype(np.float32)
        parameter.const_value = ir.tensor(values)
        state[name] = values

    session = ort.InferenceSession(
        ir.serde.serialize_model(ir.Model(graph, ir_version=11)).SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    pixels = rng.normal(size=(1, 3, 4, 4)).astype(np.float32)
    (actual,) = session.run(None, {"pixel_values": pixels})

    def linear(x: torch.Tensor, stem: str) -> torch.Tensor:
        return torch_functional.linear(
            x,
            torch.from_numpy(state[f"{stem}.weight"]),
            torch.from_numpy(state[f"{stem}.bias"]),
        )

    def layer_norm(x: torch.Tensor, stem: str) -> torch.Tensor:
        return torch_functional.layer_norm(
            x,
            (x.shape[-1],),
            torch.from_numpy(state[f"{stem}.weight"]),
            torch.from_numpy(state[f"{stem}.bias"]),
            1e-5,
        )

    hidden = torch_functional.conv2d(
        torch.from_numpy(pixels),
        torch.from_numpy(state["embeddings.patch_embedding"]).reshape(16, 3, 2, 2),
        stride=2,
    )
    hidden = hidden.flatten(2).transpose(1, 2)
    cls = torch.from_numpy(state["embeddings.class_embedding"])[None, None]
    hidden = torch.cat((hidden, cls), dim=1)
    hidden = hidden + torch.from_numpy(state["embeddings.position_embedding"])[None]
    hidden = layer_norm(hidden, "pre_layernorm")

    residual = hidden
    normed = layer_norm(hidden, "encoder.0.ln1")
    q = linear(normed, "encoder.0.attn.q_proj").reshape(1, 5, 2, 8)
    k = linear(normed, "encoder.0.attn.k_proj").reshape(1, 5, 2, 8)
    v = linear(normed, "encoder.0.attn.v_proj").reshape(1, 5, 2, 8)

    pos_w = torch.tensor([1, 2, 1, 2, 0], dtype=torch.float32)
    pos_h = torch.tensor([1, 1, 2, 2, 0], dtype=torch.float32)

    def rope_half(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        inv_freq = 1.0 / (10_000.0 ** (torch.arange(0, x.shape[-1], 2) / float(x.shape[-1])))
        angles = positions[:, None] * inv_freq[None]
        cos = torch.cos(angles).repeat_interleave(2, dim=-1)[:, None]
        sin = torch.sin(angles).repeat_interleave(2, dim=-1)[:, None]
        pairs = x.reshape(*x.shape[:-1], -1, 2)
        rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
        return x * cos + rotated * sin

    q = torch.cat((rope_half(q[..., :4], pos_w), rope_half(q[..., 4:], pos_h)), dim=-1)
    k = torch.cat((rope_half(k[..., :4], pos_w), rope_half(k[..., 4:], pos_h)), dim=-1)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(8.0)
    attended = torch.matmul(torch.softmax(scores, dim=-1), v)
    attended = attended.transpose(1, 2).reshape(1, 5, 16)
    hidden = residual + linear(attended, "encoder.0.attn.out_proj")

    residual = hidden
    hidden = layer_norm(hidden, "encoder.0.ln2")
    hidden = linear(
        torch_functional.gelu(linear(hidden, "encoder.0.mlp.up_proj")),
        "encoder.0.mlp.down_proj",
    )
    expected = layer_norm(residual + hidden, "post_layernorm")[:, :4]

    np.testing.assert_allclose(
        actual,
        expected.detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
