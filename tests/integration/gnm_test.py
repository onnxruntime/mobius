# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Synthetic GNM parity against a separately installed pinned upstream package."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from mobius import build_from_module
from mobius._testing.gnm import GNM_REFERENCE_REVISION
from mobius.integrations._gnm import gnm_archive_path
from mobius.models.gnm import GNM_MODEL_ID, GNM_REVISION, GNMConfig, GNMHeadModel
from mobius.tasks import MeshGenerationTask


@pytest.mark.integration
def test_gnm_synthetic_gpu_parity(tmp_path):
    """All nonzero synthetic tensors exercise rotations, hierarchy and correctives."""
    import onnxruntime as ort
    from gnm.shape import gnm_pytorch

    reference_file = Path(gnm_pytorch.__file__).resolve()
    root = next(parent for parent in reference_file.parents if (parent / ".git").exists())
    assert (
        subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        == GNM_REFERENCE_REVISION
    )
    assert not subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    assert torch.cuda.is_available()
    assert "CUDAExecutionProvider" in ort.get_available_providers()
    assert os.environ.get("NVIDIA_TF32_OVERRIDE") == "0"
    torch.backends.cuda.matmul.allow_tf32 = False
    reference = gnm_pytorch.GNM.from_custom_file(gnm_archive_path(GNM_MODEL_ID, GNM_REVISION))
    data = reference.to_numpy_data_dict()
    rng = np.random.default_rng(42)
    config = GNMConfig(num_vertices=7, identity_dim=2, expression_dim=3)
    module = GNMHeadModel(config)
    package = build_from_module(module, config, MeshGenerationTask())
    weights = {}
    for name, value in package["model"].graph.initializers.items():
        if value.const_value is None:
            array = rng.normal(0, 0.1, tuple(value.shape)).astype(np.float32)
            if name == "skinning_weights":
                array = np.abs(array)
                array /= array.sum(axis=0, keepdims=True)
            data[name] = array
            weights[name] = torch.from_numpy(array)
    data["identity_names"] = data["identity_names"][:2]
    data["expression_names"] = data["expression_names"][:3]
    data["mirror_indices"] = np.arange(7, dtype=np.int32)
    data["vertex_groups"] = data["vertex_groups"][:, :7]
    data["joint_regressor"] = data["joint_regressor"][:, :7]
    data["quads"] = np.array([[0, 1, 2, 3]], dtype=np.int32)
    data["triangles"] = np.array([[0, 1, 2]], dtype=np.int32)
    data["quad_uvs"] = data["quad_uvs"][:1]
    data["triangle_uvs"] = data["triangle_uvs"][:1]
    reference = gnm_pytorch.GNM._from_model_data(data).to("cuda")
    package.apply_weights(weights)
    package.save(tmp_path, max_workers=1)
    session = ort.InferenceSession(
        str(tmp_path / "model.onnx"), providers=["CUDAExecutionProvider"]
    )
    assert session.get_providers()[0] == "CUDAExecutionProvider"
    for batch in (1, 3):
        feeds = {
            "identity": rng.normal(size=(batch, 2)).astype(np.float32),
            "expression": rng.normal(size=(batch, 3)).astype(np.float32),
            "rotations": rng.uniform(-0.5, 0.5, (batch, 4, 3)).astype(np.float32),
            "translation": rng.normal(size=(batch, 3)).astype(np.float32),
        }
        with torch.no_grad(), torch.device("cuda"):
            expected = (
                reference(
                    **{name: torch.from_numpy(value).cuda() for name, value in feeds.items()}
                )
                .cpu()
                .numpy()
            )
        actual = session.run(["vertices"], feeds)[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)
