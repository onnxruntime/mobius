# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch
import torch.nn.functional as torch_functional

from mobius._testing import create_test_builder, create_test_input
from mobius.components._clip_sidecars import (
    MeralionAudioSidecar,
    MeralionProjector,
    Yasa2GlobalResponseNorm,
    Yasa2VisionSidecar,
)


def _state_and_run(module, inputs: dict[str, np.ndarray], *, seed: int = 0):
    builder, op, graph = create_test_builder()
    values = [
        create_test_input(builder, name, list(value.shape), dtype=ir.DataType.FLOAT)
        for name, value in inputs.items()
    ]
    output = module(op, *values)
    graph.outputs.append(output)

    rng = np.random.default_rng(seed)
    state: dict[str, np.ndarray] = {}
    for name, parameter in module.named_parameters():
        shape = tuple(int(dim) for dim in parameter.shape)
        if name.endswith(("norm_weight", "layer_norm.weight")):
            value = rng.uniform(0.7, 1.3, shape).astype(np.float32)
        else:
            value = rng.normal(0.0, 0.08, shape).astype(np.float32)
        parameter.const_value = ir.tensor(value)
        state[name] = value

    proto = ir.serde.serialize_model(ir.Model(graph, ir_version=11))
    session = ort.InferenceSession(
        proto.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (actual,) = session.run(None, inputs)
    return actual, state, graph


def _linear(x: torch.Tensor, state: dict[str, np.ndarray], stem: str) -> torch.Tensor:
    return torch_functional.linear(
        x,
        torch.from_numpy(state[f"{stem}.weight"]),
        torch.from_numpy(state[f"{stem}.bias"]),
    )


def _channels_first_norm(
    x: torch.Tensor, state: dict[str, np.ndarray], stem: str, eps: float
) -> torch.Tensor:
    mean = x.float().mean(dim=1, keepdim=True)
    centered = x.float() - mean
    variance = centered.square().mean(dim=1, keepdim=True)
    x = centered / variance.clamp(eps, 1e30).sqrt()
    return (
        x * torch.from_numpy(state[f"{stem}.weight"])[None, :, None, None]
        + torch.from_numpy(state[f"{stem}.bias"])[None, :, None, None]
    )


def _grn(x: torch.Tensor, state: dict[str, np.ndarray], stem: str) -> torch.Tensor:
    gx = x.float().square().sum(dim=(2, 3), keepdim=True).sqrt()
    nx = gx / gx.mean(dim=1, keepdim=True).clamp(1e-6, 1e30)
    prefix = f"{stem}." if stem else ""
    weight = torch.from_numpy(state[f"{prefix}weight"])[None, :, None, None]
    bias = torch.from_numpy(state[f"{prefix}bias"])[None, :, None, None]
    return x + weight * (x * nx) + bias


def _yasa_reference(x: torch.Tensor, state: dict[str, np.ndarray]) -> torch.Tensor:
    x = torch_functional.conv2d(
        x,
        torch.from_numpy(state["patch_embedding.weight"]),
        torch.from_numpy(state["patch_embedding.bias"]),
        stride=4,
    )
    x = _channels_first_norm(x, state, "patch_layer_norm", 1e-12)
    residual = x
    x = torch_functional.conv2d(
        x,
        torch.from_numpy(state["stages.0.blocks.0.depthwise.weight"]),
        torch.from_numpy(state["stages.0.blocks.0.depthwise.bias"]),
        padding=3,
        groups=4,
    )
    x = _channels_first_norm(x, state, "stages.0.blocks.0.layer_norm", 1e-12)
    x = _linear(x.permute(0, 2, 3, 1), state, "stages.0.blocks.0.pointwise_up")
    x = torch_functional.gelu(x, approximate="none").permute(0, 3, 1, 2)
    x = _grn(x, state, "stages.0.blocks.0.grn")
    x = _linear(x.permute(0, 2, 3, 1), state, "stages.0.blocks.0.pointwise_down")
    x = residual + x.permute(0, 3, 1, 2)

    x = x.permute(0, 2, 3, 1).reshape(1, 256, 4)
    x = x + torch.from_numpy(state["vision_position_embedding"])
    x = x.reshape(1, 16, 16, 4).permute(0, 3, 1, 2)
    x = torch_functional.avg_pool2d(x, kernel_size=2, stride=2)
    x = x.permute(0, 2, 3, 1).reshape(1, 64, 4)
    return _linear(
        torch_functional.gelu(_linear(x, state, "projector_up"), approximate="none"),
        state,
        "projector_down",
    )


def test_yasa2_nonzero_convnext_grn_pool_and_projector_parity():
    model = Yasa2VisionSidecar(
        depths=[1],
        hidden_sizes=[4],
        projector_hidden_size=7,
        output_size=5,
        image_size=64,
    )
    pixels = np.random.default_rng(11).normal(size=(1, 3, 64, 64)).astype(np.float32)
    actual, state, graph = _state_and_run(model, {"pixel_values": pixels}, seed=12)
    expected = _yasa_reference(torch.from_numpy(pixels), state).numpy()

    assert len(graph) > 0
    assert actual.shape == (1, 64, 5)
    assert np.count_nonzero(actual) > 0
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("scale", [1.0, 1e3])
def test_yasa2_grn_is_finite_and_matches_float32_reference(scale: float):
    grn = Yasa2GlobalResponseNorm(6)
    features = np.random.default_rng(21).normal(size=(2, 6, 3, 5)).astype(np.float32) * scale
    actual, state, _ = _state_and_run(grn, {"features": features}, seed=22)
    expected = _grn(torch.from_numpy(features), state, "").numpy()
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def _meralion_reference(
    x: torch.Tensor, state: dict[str, np.ndarray], *, norm_before_stack: bool = False
) -> torch.Tensor:
    weight = torch.from_numpy(state["norm_weight"])
    bias = torch.from_numpy(state["norm_bias"])
    if norm_before_stack:
        x = torch_functional.layer_norm(x, (4,), weight, bias, 1e-5).reshape(2, 2, 12)
    else:
        x = x.reshape(2, 2, 3, 4)
        mean = x.float().mean(dim=(-2, -1), keepdim=True)
        variance = (x.float() - mean).square().mean(dim=(-2, -1), keepdim=True)
        x = (x.float() - mean) / torch.sqrt(variance + 1e-5)
        x = (x * weight + bias).reshape(2, 2, 12)
    hidden = torch_functional.silu(_linear(x, state, "linear0"))
    gate = torch_functional.silu(_linear(hidden, state, "linear1"))
    pool = _linear(hidden, state, "linear2")
    return _linear(gate * pool, state, "linear3")


def test_meralion_stacks_then_normalizes_and_gates():
    projector = MeralionProjector(
        d_model=4,
        projector_hidden_size=9,
        output_size=6,
        stack_factor=3,
    )
    frames = np.random.default_rng(31).normal(size=(2, 6, 4)).astype(np.float32)
    # Make frame statistics deliberately distinct so ordering cannot pass accidentally.
    frames[:, 1::3] += 5.0
    frames[:, 2::3] *= 4.0
    actual, state, graph = _state_and_run(projector, {"encoder_output": frames}, seed=32)
    expected = _meralion_reference(torch.from_numpy(frames), state).numpy()
    hf_order = _meralion_reference(
        torch.from_numpy(frames), state, norm_before_stack=True
    ).numpy()

    assert len(graph) > 0
    assert actual.shape == (2, 2, 6)
    assert np.count_nonzero(actual) > 0
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    assert not np.allclose(actual, hf_order, rtol=1e-3, atol=1e-3)


def test_meralion_audio_sidecar_builds_exact_mel_to_tokens_shape():
    sidecar = MeralionAudioSidecar(
        num_mel_bins=8,
        d_model=4,
        encoder_layers=1,
        encoder_heads=1,
        encoder_ffn_dim=8,
        max_source_positions=6,
        projector_hidden_size=9,
        output_size=5,
        stack_factor=3,
    )
    builder, op, graph = create_test_builder()
    input_features = create_test_input(
        builder,
        "input_features",
        [12, 8],
        dtype=ir.DataType.FLOAT,
    )
    output = sidecar(op, input_features)
    builder._adapt_outputs([output], "")

    assert len(graph) > 0
    assert list(output.shape) == [2, 5]
    assert sidecar.input_schema[0][0] == "input_features"
    names = {name for name, _ in sidecar.named_parameters()}
    assert "conv1.weight" in names
    assert "layers.0.self_attn.q_proj.weight" in names
    assert "projector.linear3.weight" in names


def test_sidecar_contract_validation():
    with pytest.raises(ValueError, match="final spatial grid"):
        Yasa2VisionSidecar([1, 1], [4, 8], 8, 8, image_size=48)
    with pytest.raises(ValueError, match="divisible by stack_factor"):
        MeralionAudioSidecar(
            num_mel_bins=8,
            d_model=4,
            encoder_layers=0,
            encoder_heads=1,
            encoder_ffn_dim=8,
            max_source_positions=7,
            projector_hidden_size=8,
            output_size=5,
            stack_factor=3,
        )
