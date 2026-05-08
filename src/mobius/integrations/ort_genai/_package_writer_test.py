# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the ORT Model Package metadata writers."""

from __future__ import annotations

import json
import os

from mobius.integrations.ort_genai._package_writer import (
    BASE_VARIANT_NAME,
    DEFAULT_BASE_EP_COMPATIBILITY,
    SCHEMA_VERSION,
    write_component_metadata,
    write_manifest,
    write_variant_json,
)


class TestWriteManifest:
    """Tests for the package-root manifest.json writer."""

    def test_writes_schema_version_and_components(self, tmp_path):
        path = write_manifest(str(tmp_path), ["decoder"])
        assert os.path.isfile(path)
        with open(path, encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["schema_version"] == SCHEMA_VERSION
        assert manifest["components"] == ["decoder"]

    def test_preserves_component_order(self, tmp_path):
        path = write_manifest(str(tmp_path), ["decoder", "vision_encoder", "embedding"])
        with open(path, encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["components"] == ["decoder", "vision_encoder", "embedding"]

    def test_creates_directory_when_missing(self, tmp_path):
        new_dir = tmp_path / "nested" / "package"
        path = write_manifest(str(new_dir), ["decoder"])
        assert os.path.isfile(path)


class TestWriteComponentMetadata:
    """Tests for per-component metadata.json writer."""

    def test_default_base_variant(self, tmp_path):
        component_dir = tmp_path / "decoder"
        path = write_component_metadata(str(component_dir))
        assert os.path.isfile(path)
        with open(path, encoding="utf-8") as f:
            metadata = json.load(f)
        assert "variants" in metadata
        # Schema: variants is a map keyed by variant name.
        assert isinstance(metadata["variants"], dict)
        assert list(metadata["variants"].keys()) == [BASE_VARIANT_NAME]
        assert (
            metadata["variants"][BASE_VARIANT_NAME]["ep_compatibility"]
            == DEFAULT_BASE_EP_COMPATIBILITY
        )

    def test_default_advertises_all_major_eps(self, tmp_path):
        component_dir = tmp_path / "decoder"
        path = write_component_metadata(str(component_dir))
        with open(path, encoding="utf-8") as f:
            metadata = json.load(f)
        ep_names = {
            entry["ep"]
            for entry in metadata["variants"][BASE_VARIANT_NAME]["ep_compatibility"]
        }
        # The base variant must declare compatibility with the major EPs
        # so that today's og.Model(path, ep="...") UX keeps working.
        assert "CPUExecutionProvider" in ep_names
        assert "CUDAExecutionProvider" in ep_names
        assert "DmlExecutionProvider" in ep_names
        assert "WebGpuExecutionProvider" in ep_names
        assert "NvTensorRTRTXExecutionProvider" in ep_names

    def test_default_uses_spec_field_names(self, tmp_path):
        # The ORT-GenAI loader uses streaming JSON and rejects unknown
        # keys: ep_compatibility entries must use 'ep' / 'device' /
        # 'compatibility' (not legacy aliases like 'ep_name' or
        # 'compatibility_strings').
        component_dir = tmp_path / "decoder"
        path = write_component_metadata(str(component_dir))
        with open(path, encoding="utf-8") as f:
            metadata = json.load(f)
        for entry in metadata["variants"][BASE_VARIANT_NAME]["ep_compatibility"]:
            assert set(entry.keys()) <= {"ep", "device", "compatibility"}
            assert "ep" in entry

    def test_custom_variants_override_default(self, tmp_path):
        component_dir = tmp_path / "decoder"
        custom = {
            "cuda-ada-fp16": {
                "ep_compatibility": [
                    {"ep": "CUDAExecutionProvider", "compatibility": ["sm_89"]},
                ],
            },
        }
        path = write_component_metadata(str(component_dir), variants=custom)
        with open(path, encoding="utf-8") as f:
            metadata = json.load(f)
        assert metadata["variants"] == custom


class TestWriteVariantJson:
    """Tests for per-variant variant.json writer."""

    def test_default_single_model_file(self, tmp_path):
        variant_dir = tmp_path / "decoder" / "base"
        path = write_variant_json(str(variant_dir))
        assert os.path.isfile(path)
        with open(path, encoding="utf-8") as f:
            variant = json.load(f)
        assert len(variant["files"]) == 1
        entry = variant["files"][0]
        assert entry["filename"] == "model.onnx"
        # Base variant ships EP-agnostic — optional fields are omitted
        # entirely (the loader treats absent as default and rejects
        # list-form values).
        assert "session_options" not in entry
        assert "provider_options" not in entry
        assert "shared_files" not in entry

    def test_default_consumer_metadata_has_empty_overlay(self, tmp_path):
        variant_dir = tmp_path / "decoder" / "base"
        path = write_variant_json(str(variant_dir))
        with open(path, encoding="utf-8") as f:
            variant = json.load(f)
        assert "consumer_metadata" in variant
        assert variant["consumer_metadata"] == {"genai_config_overlay": {}}

    def test_explicit_consumer_metadata_overlay(self, tmp_path):
        variant_dir = tmp_path / "decoder" / "cuda"
        overlay = {
            "genai_config_overlay": {
                "search": {"past_present_share_buffer": True},
            }
        }
        path = write_variant_json(str(variant_dir), consumer_metadata=overlay)
        with open(path, encoding="utf-8") as f:
            variant = json.load(f)
        assert variant["consumer_metadata"] == overlay

    def test_normalizes_partial_file_entries(self, tmp_path):
        variant_dir = tmp_path / "decoder" / "base"
        path = write_variant_json(
            str(variant_dir),
            files=[{"filename": "model.onnx"}],  # missing all optional fields
        )
        with open(path, encoding="utf-8") as f:
            variant = json.load(f)
        entry = variant["files"][0]
        assert entry["filename"] == "model.onnx"
        # Optional fields are absent (not empty list / empty dict) —
        # matches what the loader expects to round-trip cleanly.
        assert set(entry.keys()) == {"filename"}

    def test_keeps_non_empty_optional_object_fields(self, tmp_path):
        variant_dir = tmp_path / "decoder" / "qnn-htp-v75"
        path = write_variant_json(
            str(variant_dir),
            files=[
                {
                    "filename": "model.onnx",
                    "session_options": {"graph_optimization_level": "all"},
                    "provider_options": {"backend_path": "QnnHtp.dll"},
                    "shared_files": {"weights.data": "deadbeef"},
                }
            ],
        )
        with open(path, encoding="utf-8") as f:
            variant = json.load(f)
        entry = variant["files"][0]
        assert entry["session_options"] == {"graph_optimization_level": "all"}
        assert entry["provider_options"] == {"backend_path": "QnnHtp.dll"}
        assert entry["shared_files"] == {"weights.data": "deadbeef"}

    def test_drops_empty_optional_object_fields(self, tmp_path):
        variant_dir = tmp_path / "decoder" / "base"
        path = write_variant_json(
            str(variant_dir),
            files=[
                {
                    "filename": "model.onnx",
                    "session_options": {},
                    "provider_options": {},
                    "shared_files": {},
                }
            ],
        )
        with open(path, encoding="utf-8") as f:
            variant = json.load(f)
        entry = variant["files"][0]
        assert set(entry.keys()) == {"filename"}

    def test_rejects_list_provider_options(self, tmp_path):
        # The ORT-GenAI loader requires provider_options / shared_files
        # to be JSON objects; lists trigger a runtime error. Catch this
        # at write time so we never produce an unloadable variant.json.
        import pytest

        variant_dir = tmp_path / "decoder" / "base"
        with pytest.raises(ValueError, match="must be an object"):
            write_variant_json(
                str(variant_dir),
                files=[{"filename": "model.onnx", "provider_options": []}],
            )

    def test_rejects_list_shared_files(self, tmp_path):
        import pytest

        variant_dir = tmp_path / "decoder" / "base"
        with pytest.raises(ValueError, match="must be an object"):
            write_variant_json(
                str(variant_dir),
                files=[{"filename": "model.onnx", "shared_files": []}],
            )

    def test_rejects_file_entry_without_filename(self, tmp_path):
        import pytest

        variant_dir = tmp_path / "decoder" / "base"
        with pytest.raises(ValueError, match="filename"):
            write_variant_json(
                str(variant_dir),
                files=[{"session_options": {}}],
            )
