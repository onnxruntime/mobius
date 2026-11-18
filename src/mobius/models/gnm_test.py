# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import dataclasses

import numpy as np
import onnx_ir as ir
import pytest
import torch

from mobius import build_from_module
from mobius.models.gnm import GNMConfig, GNMHeadModel
from mobius.tasks import MeshGenerationTask


def test_gnm_build_and_bind_identity_skinning(tmp_path):
    """Zero rotations recover the full identity/expression affine mesh."""
    import onnxruntime as ort

    from mobius import ModelPackage

    config = GNMConfig(num_vertices=7, identity_dim=2, expression_dim=3)
    package = build_from_module(GNMHeadModel(config), config, MeshGenerationTask())
    rng = np.random.default_rng(42)
    arrays = {}
    for name, value in package["model"].graph.initializers.items():
        if value.const_value is None:
            arrays[name] = rng.normal(0, 0.1, tuple(value.shape)).astype(np.float32)
    weights = rng.uniform(0, 1, (4, 7)).astype(np.float32)
    arrays["skinning_weights"] = weights / weights.sum(axis=0, keepdims=True)
    package.apply_weights({name: torch.from_numpy(value) for name, value in arrays.items()})
    package.save(tmp_path, max_workers=1)
    restored = ModelPackage.load(tmp_path)
    assert restored["model"].graph.outputs[0].name == "vertices"
    session = ort.InferenceSession(str(tmp_path / "model.onnx"))
    identity = rng.normal(size=(2, 2)).astype(np.float32)
    expression = rng.normal(size=(2, 3)).astype(np.float32)
    translation = rng.normal(size=(2, 3)).astype(np.float32)
    actual = session.run(
        None,
        {
            "identity": identity,
            "expression": expression,
            "rotations": np.zeros((2, 4, 3), np.float32),
            "translation": translation,
        },
    )[0]
    expected = (
        arrays["template_vertex_positions"]
        + np.einsum("bi,ivc->bvc", identity, arrays["vertex_identity_basis"])
        + np.einsum("be,evc->bvc", expression, arrays["expression_basis"])
        + translation[:, None, :]
    )
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
def test_gnm_rejects_unqualified_dtype_at_construction_and_build(dtype):
    config = GNMConfig(num_vertices=7, identity_dim=2, expression_dim=3)
    with pytest.raises(ValueError, match="float32 only"):
        GNMHeadModel(dataclasses.replace(config, dtype=dtype))
    module = GNMHeadModel(config)
    config.dtype = dtype
    with pytest.raises(ValueError, match="float32 only"):
        MeshGenerationTask().build(module, config)


@pytest.mark.parametrize("parents", [(), (2, 0), (-1, 1), (-1, 0, 3)])
def test_gnm_rejects_invalid_skeleton(parents):
    with pytest.raises(ValueError, match=r"root joint|topological order"):
        GNMHeadModel(GNMConfig(joint_parent_indices=parents))
