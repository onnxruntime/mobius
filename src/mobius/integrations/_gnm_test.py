# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from mobius import build, build_from_module
from mobius.integrations._gnm import gnm_archive_path, read_gnm_config
from mobius.models.gnm import GNM_REVISION, GNMConfig, GNMHeadModel
from mobius.tasks import MeshGenerationTask


@pytest.fixture
def gnm_archive(tmp_path):
    config = GNMConfig(num_vertices=7, identity_dim=2, expression_dim=3)
    package = build_from_module(GNMHeadModel(config), config, MeshGenerationTask())
    arrays = {
        name: np.ones(tuple(value.shape), dtype=np.float32)
        for name, value in package["model"].graph.initializers.items()
        if value.const_value is None
    }
    arrays["joint_parent_indices"] = np.array([-1, 0, 1, 1], dtype=np.int32)
    arrays["version"] = np.array("3.0")
    arrays["variant"] = np.array("head")
    path = tmp_path / "gnm_head.npz"
    np.savez(path, **arrays)
    return path


def test_gnm_public_build_accepts_task_object(gnm_archive):
    package = build(str(gnm_archive), task=MeshGenerationTask())
    assert package["model"].graph.outputs[0].name == "vertices"
    assert all(
        value.const_value is not None for value in package["model"].graph.initializers.values()
    )


def test_gnm_native_revision_and_checksum_are_enforced(gnm_archive, monkeypatch):
    calls = []

    def download(model_id, filename, *, revision):
        calls.append((model_id, filename, revision))
        return str(gnm_archive)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    monkeypatch.setattr(
        "mobius.integrations._gnm.GNM_SHA256",
        hashlib.sha256(gnm_archive.read_bytes()).hexdigest(),
    )
    assert gnm_archive_path("google/gnm-v3") == gnm_archive
    assert calls == [("google/gnm-v3", "v3_0/gnm_head.npz", GNM_REVISION)]
    with pytest.raises(ValueError, match="immutable release SHA"):
        gnm_archive_path("google/gnm-v3", "main")
    assert len(calls) == 1
    monkeypatch.setattr("mobius.integrations._gnm.GNM_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA-256 differs"):
        gnm_archive_path("google/gnm-v3")


def test_gnm_native_config_reports_missing_arrays():
    with pytest.raises(ValueError, match="lacks required arrays"):
        read_gnm_config({})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dtype": "f16"},
        {"text_only": True},
        {"task": "text-generation"},
        {"fp8_kv_cache": True},
    ],
)
def test_gnm_public_build_rejects_unsupported_options_before_download(monkeypatch, kwargs):
    def download(*args, **kwargs):
        pytest.fail("Unsupported GNM options must fail before network I/O")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    with pytest.raises(ValueError, match=r"float32 only|unsupported build options"):
        build("google/gnm-v3", **kwargs)
