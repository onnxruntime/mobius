# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Executable tiny parity tests for the CogVLM/Nemotron clip sidecars."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import torch
import torch.nn.functional as torch_functional

from mobius._testing.ort_inference import OnnxModelSession
from mobius.components._cog_nemotron_clip import (
    CogVLMClipSidecar,
    NemotronV2VLClipSidecar,
)
from mobius.integrations._weight_loading import apply_weights
from mobius.tasks._base import _make_graph, _make_model

_IMAGE_SIZE = 4
_PATCH_SIZE = 2
_HIDDEN = 8
_INTERMEDIATE = 12
_HEADS = 2


def _build(module, name: str) -> ir.Model:
    graph, builder = _make_graph(name=name)
    pixel_values = builder.input(
        "pixel_values",
        dtype=ir.DataType.FLOAT,
        shape=[1, 3, _IMAGE_SIZE, _IMAGE_SIZE],
    )
    builder.add_output(module(builder.op, pixel_values), "image_features")
    return _make_model(graph)


def _randomize(model: ir.Model, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    state: dict[str, np.ndarray] = {}
    for name, initializer in model.graph.initializers.items():
        if name.startswith("const"):
            continue
        shape = tuple(int(dim) for dim in initializer.shape)
        values = rng.normal(0.0, 0.12, shape).astype(np.float32)
        if (name.endswith(".weight") and (".ln" in name or "post_fc_norm" in name)) or (
            name == "mm.model.mlp.0.weight"
        ):
            values += 1.0
        state[name] = values
    apply_weights(model, {name: torch.from_numpy(value) for name, value in state.items()})
    return state


def _key(state: dict[str, np.ndarray], name: str) -> str:
    if name in state:
        return name
    matches = [candidate for candidate in state if candidate.endswith(f".{name}")]
    assert len(matches) == 1, (name, matches)
    return matches[0]


def _linear(
    x: torch.Tensor,
    state: dict[str, np.ndarray],
    stem: str,
    *,
    bias: bool,
) -> torch.Tensor:
    return torch_functional.linear(
        x,
        torch.from_numpy(state[_key(state, f"{stem}.weight")]),
        torch.from_numpy(state[_key(state, f"{stem}.bias")]) if bias else None,
    )


def _layer_norm(
    x: torch.Tensor, state: dict[str, np.ndarray], stem: str, eps: float
) -> torch.Tensor:
    return torch_functional.layer_norm(
        x,
        (x.shape[-1],),
        torch.from_numpy(state[_key(state, f"{stem}.weight")]),
        torch.from_numpy(state[_key(state, f"{stem}.bias")]),
        eps,
    )


def _attention(x: torch.Tensor, state: dict[str, np.ndarray], stem: str) -> torch.Tensor:
    # Independent split/reshape makes a wrong fused-QKV ordering observable.
    qkv = _linear(x, state, f"{stem}.attn_qkv", bias=True)
    query, key, value = qkv.chunk(3, dim=-1)
    head_dim = _HIDDEN // _HEADS

    def heads(value_: torch.Tensor) -> torch.Tensor:
        return value_.reshape(1, value_.shape[1], _HEADS, head_dim).transpose(1, 2)

    query, key, value = heads(query), heads(key), heads(value)
    scores = query @ key.transpose(-1, -2) / head_dim**0.5
    attended = (scores.softmax(dim=-1) @ value).transpose(1, 2).reshape_as(x)
    return _linear(attended, state, f"{stem}.attn_out", bias=True)


def _patches(
    pixels: torch.Tensor, state: dict[str, np.ndarray], *, bias: bool
) -> torch.Tensor:
    patches = torch_functional.conv2d(
        pixels,
        torch.from_numpy(state[_key(state, "v.patch_embd.weight")]),
        torch.from_numpy(state[_key(state, "v.patch_embd.bias")]) if bias else None,
        stride=_PATCH_SIZE,
    )
    return patches.flatten(2).transpose(1, 2)


def _run(model: ir.Model, pixels: np.ndarray) -> np.ndarray:
    session = OnnxModelSession(model)
    actual = session.run({"pixel_values": pixels})["image_features"]
    session.close()
    return actual


def test_cogvlm_sidecar_matches_post_norm_swiglu_reference():
    module = CogVLMClipSidecar(
        image_size=_IMAGE_SIZE,
        patch_size=_PATCH_SIZE,
        hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
        num_layers=1,
        num_heads=_HEADS,
        projector_hidden_size=6,
        projector_intermediate_size=9,
        output_size=5,
    )
    model = _build(module, "cogvlm_clip")
    state = _randomize(model, 1)
    pixels = np.random.default_rng(2).normal(size=(1, 3, 4, 4)).astype(np.float32)

    x = _patches(torch.from_numpy(pixels), state, bias=True)
    x = torch.cat(
        (x, torch.from_numpy(state["v.class_embd"]).unsqueeze(0)),
        dim=1,
    )
    x = x + torch.from_numpy(state["v.position_embd.weight"]).unsqueeze(0)
    stem = "v.blk.0"
    x = x + _layer_norm(_attention(x, state, stem), state, f"{stem}.ln1", 1e-6)
    up = _linear(x, state, f"{stem}.ffn_up", bias=True)
    gate = torch_functional.silu(_linear(x, state, f"{stem}.ffn_gate", bias=True))
    ffn = _linear(up * gate, state, f"{stem}.ffn_down", bias=True)
    x = x + _layer_norm(ffn, state, f"{stem}.ln2", 1e-6)

    # The appended CLS row is removed before the linear/LN/GELU projector.
    x = _linear(x[:, :-1], state, "mm.model.fc", bias=False)
    x = _layer_norm(x, state, "mm.post_fc_norm", 1e-5)
    x = torch_functional.gelu(x, approximate="tanh")
    up = _linear(x, state, "mm.up", bias=False)
    gate = torch_functional.silu(_linear(x, state, "mm.gate", bias=False))
    x = _linear(up * gate, state, "mm.down", bias=False)
    expected = torch.cat(
        (torch.from_numpy(state["v.boi"]), x, torch.from_numpy(state["v.eoi"])),
        dim=1,
    ).numpy()

    actual = _run(model, pixels)
    assert model.graph.num_nodes() > 0
    assert actual.shape == (1, 6, 5)
    # The endpoints separately lock BOI -> patches -> EOI ordering.
    np.testing.assert_allclose(actual[:, :1], state["v.boi"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual[:, -1:], state["v.eoi"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_nemotron_sidecar_matches_register_merge_relu_squared_reference():
    module = NemotronV2VLClipSidecar(
        image_size=_IMAGE_SIZE,
        patch_size=_PATCH_SIZE,
        hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
        num_layers=1,
        num_heads=_HEADS,
        num_register_tokens=3,
        projector_hidden_size=7,
        output_size=5,
    )
    model = _build(module, "nemotron_v2_vl_clip")
    parameter_names = set(model.graph.initializers)
    state = _randomize(model, 3)
    pixels = np.random.default_rng(4).normal(size=(1, 3, 4, 4)).astype(np.float32)

    patches = _patches(torch.from_numpy(pixels), state, bias=False)
    patches = patches + torch.from_numpy(state["v.position_embd.weight"])
    x = torch.cat(
        (torch.from_numpy(state["v.class_embd"]).unsqueeze(0), patches),
        dim=1,
    )
    stem = "v.blk.0"
    normalized = _layer_norm(x, state, f"{stem}.ln1", 1e-6)
    x = x + _attention(normalized, state, stem)
    normalized = _layer_norm(x, state, f"{stem}.ln2", 1e-6)
    normalized = torch_functional.gelu(_linear(normalized, state, f"{stem}.ffn_up", bias=True))
    x = x + _linear(normalized, state, f"{stem}.ffn_down", bias=True)

    # Three (not hard-coded eight) leading rows are removed, then the 2x2
    # block is flattened in NHWC pixel-unshuffle order.
    x = x[:, 3:].reshape(1, 1, 2, 1, 2, _HIDDEN)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(1, 1, 4 * _HIDDEN)
    rms_weight = torch.from_numpy(state["mm.model.mlp.0.weight"])
    x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6) * rms_weight
    x = torch_functional.relu(
        torch_functional.linear(x, torch.from_numpy(state["mm.model.mlp.1.weight"]))
    ).square()
    expected = torch_functional.linear(
        x,
        torch.from_numpy(state["mm.model.mlp.3.weight"]),
    ).numpy()

    actual = _run(model, pixels)
    assert {
        "mm.model.mlp.0.weight",
        "mm.model.mlp.1.weight",
        "mm.model.mlp.3.weight",
    } <= parameter_names
    assert model.graph.num_nodes() > 0
    assert actual.shape == (1, 1, 5)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
