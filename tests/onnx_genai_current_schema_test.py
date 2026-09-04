# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Validate every generated signal package against the current tensor contract."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import yaml

_SCHEMA = (
    Path(__file__).parents[1]
    / "src/mobius/integrations/onnx_genai/_schema/inference_metadata.schema.json"
)
_EXPECTED_PACKAGES = {
    "adapter",
    "codec",
    "decoder",
    "diffusion",
    "diffusion_guided",
    "esm2_protein_embeddings",
    "hierarchical_audio",
    "masked",
    "protbert_protein_embeddings",
    "shared_state_pixel_flow",
    "speculative",
    "static_cache",
    "tts",
    "video",
    "vlm",
}


def test_all_signal_packages_match_current_onnx_genai_schema(
    materialized_workflow_packages: str,
) -> None:
    root = Path(materialized_workflow_packages)
    metadata_paths = sorted(root.glob("*/inference_metadata.yaml"))
    assert {path.parent.name for path in metadata_paths} == _EXPECTED_PACKAGES
    schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))

    for path in metadata_paths:
        metadata = yaml.safe_load(path.read_text(encoding="utf-8"))
        jsonschema.validate(metadata, schema)
