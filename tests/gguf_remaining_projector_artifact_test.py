# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Bounded real-header evidence for the remaining projector cohort."""

from __future__ import annotations

import hashlib
import urllib.request
from collections import Counter

import pytest
from huggingface_hub import HfApi, hf_hub_url

from mobius.integrations.gguf._mmproj import _preflight_standalone_mmproj
from mobius.integrations.gguf._mmproj_registry import (
    MMPROJ_ARTIFACT_PINS,
    MMProjArtifactPin,
)
from mobius.integrations.gguf._reader import GGUFModel

_HEADER_PINS = tuple(
    pin for pin in MMPROJ_ARTIFACT_PINS if pin.bounded_header_bytes is not None
)


@pytest.mark.integration
@pytest.mark.parametrize("pin", _HEADER_PINS, ids=lambda pin: pin.projector_types[0])
def test_projector_header_identity_and_tensor_inventory(
    pin: MMProjArtifactPin, tmp_path
) -> None:
    """Verify one immutable sidecar without retaining or downloading its payload."""
    (record,) = HfApi().get_paths_info(
        pin.repository,
        [pin.filename],
        revision=pin.revision,
    )
    assert record.size == pin.size
    assert record.lfs is not None
    assert record.lfs.sha256 == pin.lfs_sha256

    assert pin.bounded_header_bytes is not None
    assert pin.bounded_header_sha256 is not None
    request = urllib.request.Request(
        hf_hub_url(pin.repository, pin.filename, revision=pin.revision),
        headers={"Range": f"bytes=0-{pin.bounded_header_bytes - 1}"},
    )
    header_path = tmp_path / "header.bin"
    sparse_path = tmp_path / "sparse.gguf"
    try:
        with urllib.request.urlopen(request) as response:
            header = response.read(pin.bounded_header_bytes + 1)
        assert len(header) == pin.bounded_header_bytes
        assert hashlib.sha256(header).hexdigest() == pin.bounded_header_sha256

        header_path.write_bytes(header)
        sparse_path.write_bytes(header)
        with sparse_path.open("r+b") as stream:
            stream.truncate(pin.size)

        model = GGUFModel(sparse_path)
        try:
            assert model.architecture == "clip"
            assert model.metadata["clip.projector_type"] == pin.projector_types[0]
            assert len(model.tensor_names) == pin.tensor_count
            assert Counter(
                model.get_tensor_type(name).name for name in model.tensor_names
            ) == dict(pin.tensor_qtypes)
            for key, expected in pin.metadata:
                assert model.metadata[key] == expected
            spec = _preflight_standalone_mmproj(
                model,
                projector_type=pin.projector_types[0],
                target_architecture=pin.paired_text_architecture,
            )
            assert spec.is_importable
        finally:
            model.close()
    finally:
        sparse_path.unlink(missing_ok=True)
        header_path.unlink(missing_ok=True)
