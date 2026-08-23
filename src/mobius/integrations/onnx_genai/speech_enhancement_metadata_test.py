# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for spectral speech-enhancement onnx-genai metadata."""

from __future__ import annotations

import json
import os

import pytest

from mobius import build_from_module
from mobius.integrations.onnx_genai import (
    build_speech_enhancement_workflow_metadata,
    write_onnx_genai_config,
    write_speech_enhancement_workflow_metadata,
)
from mobius.integrations.onnx_genai.auto_export import _looks_like_speech_enhancement
from mobius.integrations.onnx_genai.inference_metadata_test import (
    _onnx_genai_schema_path,
)
from mobius.models.reuse import ReUseConfig, SEMambaSpeechEnhancementModel

_TINY = {
    "model_cfg": {
        "hid_feature": 8,
        "num_tfmamba": 1,
        "d_state": 4,
        "expand": 2,
        "compress_factor": "relu_log1p",
    },
    "stft_cfg": {"n_fft": 32, "hop_size": 4, "win_size": 32, "sampling_rate": 8000},
}


def _package(**config_overrides):
    config = ReUseConfig.from_json(_TINY)
    for key, value in config_overrides.items():
        setattr(config, key, value)
    module = SEMambaSpeechEnhancementModel(config)
    return build_from_module(module, config, task="speech-enhancement"), config


class TestDetection:
    """The dispatcher must recognise the package structurally."""

    def test_detects_enhancement_package(self):
        pkg, _config = _package()
        assert _looks_like_speech_enhancement(pkg) is True

    def test_rejects_a_decoder_package(self):
        from mobius._testing import make_config
        from mobius.models import CausalLMModel

        config = make_config()
        pkg = build_from_module(CausalLMModel(config), config)

        assert _looks_like_speech_enhancement(pkg) is False


class TestMetadata:
    """The emitted document must describe the graph truthfully."""

    def test_declares_a_single_pure_invocation(self):
        pkg, config = _package()

        metadata = build_speech_enhancement_workflow_metadata(pkg, config)

        workflow = metadata["pipeline"]["workflow"]
        # No generation loop: nothing to sample, nothing carried between calls.
        invokes = [s for s in workflow["steps"] if s["kind"] == "invoke"]
        assert [s["component"] for s in invokes] == ["audio_preprocess", "enhancer"]
        emits = [s for s in workflow["steps"] if s["kind"] == "emit"]
        assert len(emits) == 3
        assert not any(s["kind"] == "loop" for s in workflow["steps"])
        for effect in workflow["effects"].values():
            assert effect["retry"] == "pure"

    def test_publishes_every_graph_output(self):
        pkg, config = _package()

        metadata = build_speech_enhancement_workflow_metadata(pkg, config)

        outputs = metadata["pipeline"]["workflow"]["outputs"]
        assert set(outputs) == {"denoised_mag", "denoised_pha", "denoised_com"}
        # The complex spectrogram carries a trailing (real, imag) pair.
        assert outputs["denoised_com"]["contract"]["rank"] == 4
        assert outputs["denoised_com"]["contract"]["shape"][-1] == 2
        assert outputs["denoised_mag"]["contract"]["rank"] == 3
        profile = metadata["profiles"]["speech_enhancement"]
        assert set(profile["outputs"]) == set(outputs)

    def test_declares_the_stft_front_end(self):
        pkg, config = _package()

        metadata = build_speech_enhancement_workflow_metadata(pkg, config)

        transforms = metadata["preprocessing"]["audio"]["transforms"]
        by_op = {t["op"]: t for t in transforms}
        assert by_op["resample"]["sample_rate"] == config.sampling_rate
        spectrogram = by_op["spectrogram"]
        assert spectrogram["n_fft"] == config.n_fft
        assert spectrogram["hop_length"] == config.hop_size
        assert spectrogram["win_length"] == config.win_size
        # The model is trained on log1p-compressed magnitudes.
        assert "log1p" in by_op
        assert by_op["log1p"]["inputs"] == ["magnitude"]

    def test_audio_outputs_bind_to_the_graph_inputs(self):
        pkg, config = _package()

        metadata = build_speech_enhancement_workflow_metadata(pkg, config)

        bindings = {b["name"]: b for b in metadata["preprocessing"]["audio"]["outputs"]}
        assert set(bindings) == {"noisy_mag", "noisy_pha"}
        assert bindings["noisy_mag"]["source"] == "magnitude"
        assert bindings["noisy_pha"]["source"] == "phase"
        for binding in bindings.values():
            # Required whenever the package declares a pipeline.workflow.
            assert binding["contract"]["rank"] == 3
            assert binding["dtype"] == "float32"

    def test_declares_the_adapter_abi_when_the_front_end_ships(self):
        """A runtime must be able to version-check the STFT adapter it has to supply.

        The component already names the ABI, but a consumer reads the manifest to
        decide whether it can run the package at all — the other audio workflows
        declare it there, and omitting it hides the requirement until binding time.
        """
        pkg, config = _package()

        metadata = build_speech_enhancement_workflow_metadata(pkg, config)

        manifest = metadata["pipeline"]["workflow"]["manifest"]
        component = metadata["pipeline"]["workflow"]["components"]["audio_preprocess"]
        abi = component["implementation"]["abi"]
        assert manifest["adapter_abis"][abi] == component["implementation"]["version"]

    def test_omits_the_adapter_abi_when_no_front_end_ships(self):
        """No adapter in the package means nothing for a runtime to version-check.

        Declaring the ABI unconditionally would advertise a requirement the caller
        does not have to satisfy, since it supplies the spectra itself.
        """
        pkg, config = _package()
        config.sampling_rate = None  # type: ignore[assignment]

        metadata = build_speech_enhancement_workflow_metadata(pkg, config)

        manifest = metadata["pipeline"]["workflow"]["manifest"]
        assert "adapter_abis" not in manifest
        assert "audio_preprocess" not in metadata["pipeline"]["workflow"]["components"]

    def test_omits_the_program_when_geometry_is_unknown(self):
        """Never invent a transform program the config cannot justify."""
        pkg, config = _package()
        config.sampling_rate = None  # type: ignore[assignment]

        metadata = build_speech_enhancement_workflow_metadata(pkg, config)

        assert "preprocessing" not in metadata
        # The caller then supplies the spectra directly.
        inputs = metadata["pipeline"]["workflow"]["inputs"]
        assert set(inputs) == {"request.noisy_mag", "request.noisy_pha"}

    def test_rejects_a_graph_without_the_expected_ports(self):
        pkg, config = _package()
        pkg["model"].graph.inputs[0].name = "something_else"

        with pytest.raises(ValueError, match="noisy_mag"):
            build_speech_enhancement_workflow_metadata(pkg, config)


class TestAutoExport:
    """`write_onnx_genai_config` must route the package here."""

    def test_dispatches_to_the_enhancement_writer(self, tmp_path):
        pkg, config = _package()

        artifacts = write_onnx_genai_config(pkg, str(tmp_path), config=config)

        assert os.path.isfile(artifacts["inference_metadata"])

    def test_written_document_round_trips(self, tmp_path):
        pkg, config = _package()
        yaml = pytest.importorskip("yaml")

        path = write_speech_enhancement_workflow_metadata(pkg, str(tmp_path), config)

        with open(path, encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
        assert document["schema_version"] == "v1"
        assert "speech_enhancement" in document["profiles"]


class TestSchema:
    """The document must validate against onnx-genai's committed schema."""

    def test_matches_onnx_genai_json_schema(self):
        jsonschema = pytest.importorskip("jsonschema")
        schema_path = _onnx_genai_schema_path()
        if schema_path is None:
            pytest.skip("onnx-genai schema not found (set ONNX_GENAI_SCHEMA)")
        pkg, config = _package()

        metadata = build_speech_enhancement_workflow_metadata(pkg, config)

        with open(schema_path, encoding="utf-8") as handle:
            schema = json.load(handle)
        jsonschema.validate(metadata, schema)

    def test_matches_schema_without_preprocessing(self):
        jsonschema = pytest.importorskip("jsonschema")
        schema_path = _onnx_genai_schema_path()
        if schema_path is None:
            pytest.skip("onnx-genai schema not found (set ONNX_GENAI_SCHEMA)")
        pkg, config = _package()
        config.n_fft = None  # type: ignore[assignment]

        metadata = build_speech_enhancement_workflow_metadata(pkg, config)

        with open(schema_path, encoding="utf-8") as handle:
            schema = json.load(handle)
        jsonschema.validate(metadata, schema)
