# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import torch
import torch.nn.functional as torch_functional

from mobius._testing import create_test_builder, create_test_input
from mobius.components._core_vlm_projector import (
    Idefics3Projector,
    InternVLProjector,
    Llama4Projector,
    PixtralProjector,
)


def _pixel_unshuffle(features: torch.Tensor, grid: int, scale: int) -> torch.Tensor:
    batch, _, channels = features.shape
    features = features.reshape(batch, grid, grid, channels)
    features = features.reshape(batch, grid, grid // scale, channels * scale)
    features = features.permute(0, 2, 1, 3).contiguous()
    features = features.reshape(
        batch,
        grid // scale,
        grid // scale,
        channels * scale * scale,
    )
    features = features.permute(0, 2, 1, 3).contiguous()
    return features.reshape(batch, grid * grid // (scale * scale), -1)


def _run(projector, features: np.ndarray, *extra_inputs: np.ndarray):
    builder, op, graph = create_test_builder()
    feature_value = create_test_input(builder, "features", list(features.shape))
    values = [feature_value]
    feeds = {"features": features}
    for index, array in enumerate(extra_inputs):
        name = f"extra_{index}"
        values.append(
            builder.input(
                name,
                dtype=ir.DataType.INT64,
                shape=list(array.shape),
            )
        )
        feeds[name] = array
    output = projector(op, *values)
    output.name = "output"
    graph.outputs.append(output)

    rng = np.random.default_rng(123)
    state: dict[str, np.ndarray] = {}
    for name, parameter in projector.named_parameters():
        shape = tuple(int(dim) for dim in parameter.shape)
        if "norm" in name and name.endswith(".weight"):
            data = (1.0 + rng.normal(0.0, 0.1, shape)).astype(np.float32)
        else:
            data = rng.normal(0.0, 0.1, shape).astype(np.float32)
        parameter.const_value = ir.tensor(data)
        state[name] = data

    session = ort.InferenceSession(
        ir.serde.serialize_model(ir.Model(graph, ir_version=11)).SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    return session.run(None, feeds)[0], state


def _linear(x: torch.Tensor, state: dict[str, np.ndarray], stem: str) -> torch.Tensor:
    bias = state.get(f"{stem}.bias")
    return torch_functional.linear(
        x,
        torch.from_numpy(state[f"{stem}.weight"]),
        None if bias is None else torch.from_numpy(bias),
    )


def test_idefics3_projector_matches_independent_reference() -> None:
    features = np.arange(1 * 16 * 3, dtype=np.float32).reshape(1, 16, 3) / 20
    projector = Idefics3Projector(3, 5, grid_size=4, scale_factor=2)

    actual, state = _run(projector, features)
    expected = _linear(
        _pixel_unshuffle(torch.from_numpy(features), 4, 2),
        state,
        "model_fc",
    )

    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-5, atol=1e-5)
    assert actual.shape == (1, 4, 5)


def test_internvl_projector_matches_independent_reference() -> None:
    features = np.arange(1 * 16 * 3, dtype=np.float32).reshape(1, 16, 3) / 20
    projector = InternVLProjector(3, 5, grid_size=4, scale_factor=2)

    actual, state = _run(projector, features)
    expected = _pixel_unshuffle(torch.from_numpy(features), 4, 2)
    expected = torch_functional.layer_norm(
        expected,
        (12,),
        torch.from_numpy(state["mlp.0.weight"]),
        torch.from_numpy(state["mlp.0.bias"]),
        1e-5,
    )
    expected = torch_functional.gelu(_linear(expected, state, "mlp.1"))
    expected = _linear(expected, state, "mlp.3")

    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-5, atol=1e-5)
    assert actual.shape == (1, 4, 5)


def test_llama4_projector_matches_independent_reference() -> None:
    features = np.arange(1 * 16 * 3, dtype=np.float32).reshape(1, 16, 3) / 20
    projector = Llama4Projector(3, 7, 5, grid_size=4, scale_factor=2)

    actual, state = _run(projector, features)
    expected = _pixel_unshuffle(torch.from_numpy(features), 4, 2)
    expected = torch_functional.gelu(_linear(expected, state, "model_mlp_1"))
    expected = torch_functional.gelu(_linear(expected, state, "model_mlp_2"))
    expected = _linear(expected, state, "model_fc")

    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-5, atol=1e-5)
    assert actual.shape == (1, 4, 5)


def test_pixtral_projector_inserts_distinct_row_breaks() -> None:
    features = np.arange(1 * 6 * 3, dtype=np.float32).reshape(1, 6, 3) / 20
    projector = PixtralProjector(3, 5, with_image_break=True)

    actual, state = _run(
        projector,
        features,
        np.array(2, dtype=np.int64),
        np.array(3, dtype=np.int64),
    )
    expected = _linear(
        torch_functional.gelu(_linear(torch.from_numpy(features), state, "linear_1")),
        state,
        "linear_2",
    ).reshape(2, 3, 5)
    expected = torch.cat(
        (
            expected[0],
            torch.from_numpy(state["image_break"])[None],
            expected[1],
        ),
        dim=0,
    )

    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-5, atol=1e-5)
    assert actual.shape == (7, 5)
