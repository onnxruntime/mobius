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
from mobius.components._ocr_projectors import (
    DeepSeekOCRProjector,
    Dots3NoteAudioProjector,
    DotsOCRProjector,
    LightOnOCRProjector,
    PaddleOCRProjector,
    YouTuVLProjector,
)


def _run(module, inputs: dict[str, np.ndarray], *args):
    builder, op, graph = create_test_builder()
    values = [
        create_test_input(
            builder,
            name,
            list(value.shape),
            dtype=ir.DataType.INT64 if value.dtype == np.int64 else ir.DataType.FLOAT,
        )
        for name, value in inputs.items()
    ]
    output = module(op, *values, *args)
    graph.outputs.append(output)

    rng = np.random.default_rng(0)
    state = {}
    for name, parameter in module.named_parameters():
        shape = tuple(int(dim) for dim in parameter.shape)
        if name.endswith(".weight") and ("norm" in name or "input_norm" in name):
            array = (1.0 + rng.normal(0.0, 0.05, shape)).astype(np.float32)
        else:
            array = rng.normal(0.0, 0.05, shape).astype(np.float32)
        parameter.const_value = ir.tensor(array)
        state[name] = torch.from_numpy(array)

    session = ort.InferenceSession(
        ir.serde.serialize_model(ir.Model(graph, ir_version=11)).SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    return session.run(None, inputs)[0], state


def _linear(x: torch.Tensor, state: dict[str, torch.Tensor], stem: str) -> torch.Tensor:
    return torch_functional.linear(
        x,
        state[f"{stem}.weight"],
        state.get(f"{stem}.bias"),
    )


def _layer_norm(
    x: torch.Tensor,
    state: dict[str, torch.Tensor],
    stem: str,
    eps: float,
) -> torch.Tensor:
    return torch_functional.layer_norm(
        x,
        (x.shape[-1],),
        state[f"{stem}.weight"],
        state[f"{stem}.bias"],
        eps,
    )


def _rms_norm(
    x: torch.Tensor,
    state: dict[str, torch.Tensor],
    stem: str,
    eps: float,
) -> torch.Tensor:
    normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return normalized.to(x.dtype) * state[f"{stem}.weight"]


def test_deepseek_projector_matches_linear_concatenation():
    module = DeepSeekOCRProjector(4, 3, 5)
    clip = np.random.default_rng(1).normal(size=(1, 6, 4)).astype(np.float32)
    sam = np.random.default_rng(2).normal(size=(1, 6, 3)).astype(np.float32)

    actual, state = _run(module, {"clip": clip, "sam": sam})
    expected = _linear(
        torch.cat((torch.from_numpy(clip), torch.from_numpy(sam)), dim=-1),
        state,
        "linear",
    )

    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-5, atol=1e-5)


def test_dots_projector_matches_merge_ordered_reference():
    module = DotsOCRProjector(4, 12, 7)
    features = np.random.default_rng(3).normal(size=(16, 4)).astype(np.float32)

    actual, state = _run(module, {"features": features})
    expected = _layer_norm(torch.from_numpy(features), state, "input_norm", 1e-6)
    expected = expected.reshape(4, 16)
    expected = torch_functional.gelu(_linear(expected, state, "linear_0"))
    expected = _linear(expected, state, "linear_2")

    assert actual.shape == (4, 7)
    np.testing.assert_allclose(actual, expected.numpy(), rtol=2e-5, atol=2e-5)


def test_paddle_projector_groups_raster_neighbors():
    module = PaddleOCRProjector(2, 10, 3)
    features = np.arange(4 * 6 * 2, dtype=np.float32).reshape(24, 2) / 10
    grid_h = np.array(4, dtype=np.int64)
    grid_w = np.array(6, dtype=np.int64)

    actual, state = _run(
        module,
        {"features": features, "grid_h": grid_h, "grid_w": grid_w},
    )
    expected = _layer_norm(torch.from_numpy(features), state, "input_norm", 1e-5)
    expected = expected.reshape(2, 2, 3, 2, 2).permute(0, 2, 1, 3, 4).reshape(6, 8)
    expected = torch_functional.gelu(
        _linear(expected, state, "linear_1"),
        approximate="tanh",
    )
    expected = _linear(expected, state, "linear_2")

    assert actual.shape == (6, 3)
    np.testing.assert_allclose(actual, expected.numpy(), rtol=2e-5, atol=2e-5)


def test_youtu_projector_matches_rms_merge_reference():
    module = YouTuVLProjector(4, 16, 6)
    features = np.random.default_rng(4).normal(size=(12, 4)).astype(np.float32)

    actual, state = _run(module, {"features": features})
    expected = _rms_norm(torch.from_numpy(features), state, "input_norm", 1e-6)
    expected = expected.reshape(3, 16)
    expected = torch_functional.gelu(
        _linear(expected, state, "linear_0"),
        approximate="tanh",
    )
    expected = _linear(expected, state, "linear_2")

    np.testing.assert_allclose(actual, expected.numpy(), rtol=2e-5, atol=2e-5)


def test_lighton_projector_matches_unfold_reference():
    module = LightOnOCRProjector(4, 6)
    features = np.random.default_rng(5).normal(size=(1, 16, 4)).astype(np.float32)
    grid_h = np.array(4, dtype=np.int64)
    grid_w = np.array(4, dtype=np.int64)

    actual, state = _run(
        module,
        {"features": features, "grid_h": grid_h, "grid_w": grid_w},
    )
    expected = _rms_norm(torch.from_numpy(features), state, "input_norm", 1e-5)
    expected = expected.transpose(1, 2).reshape(1, 4, 4, 4)
    expected = torch_functional.unfold(expected, kernel_size=2, stride=2).transpose(1, 2)
    expected = _linear(expected, state, "patch_merger.merging_layer")
    expected = torch_functional.gelu(
        _linear(expected, state, "linear_1"),
        approximate="tanh",
    )
    expected = _linear(expected, state, "linear_2")

    assert actual.shape == (1, 4, 6)
    np.testing.assert_allclose(actual, expected.numpy(), rtol=2e-5, atol=2e-5)


def test_dots_audio_projector_matches_reference():
    module = Dots3NoteAudioProjector(4, 9, 7)
    features = np.random.default_rng(6).normal(size=(1, 5, 4)).astype(np.float32)

    actual, state = _run(module, {"features": features})
    expected = _layer_norm(torch.from_numpy(features), state, "norm_pre", 1e-5)
    expected = torch_functional.gelu(_linear(expected, state, "linear_1"))
    expected = _linear(expected, state, "linear_3")

    np.testing.assert_allclose(actual, expected.numpy(), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize(
    "module",
    [
        DeepSeekOCRProjector(4, 3, 5),
        DotsOCRProjector(4, 12, 7),
        PaddleOCRProjector(4, 16, 7),
        YouTuVLProjector(4, 16, 7),
        LightOnOCRProjector(4, 7),
        Dots3NoteAudioProjector(4, 9, 7),
    ],
)
def test_projector_parameters_are_nonempty_and_unique(module):
    names = [name for name, _ in module.named_parameters()]
    assert names
    assert len(names) == len(set(names))
