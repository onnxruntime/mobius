# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Faithful three-model metadata, including stateless trailing Puzzle layers."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import onnx_ir as ir
import pytest

from mobius._configs import NemotronHOmniConfig
from mobius._configs.per_model._nemotron_h_omni_test import _omni_config
from mobius._model_package import ModelPackage
from mobius.integrations.ort_genai.auto_export import (
    _inspect_hybrid_multimodal_decoder_abi,
    _write_genai_config,
    _write_vision_processor_config,
    write_ort_genai_config,
)
from mobius.integrations.transformers._nemotron_h_omni_weights_test import _fixture


def _model(inputs, outputs):
    return ir.Model(
        ir.Graph(
            [ir.Value(name=name) for name in inputs],
            [ir.Value(name=name) for name in outputs],
            nodes=[],
        ),
        ir_version=11,
    )


def _package():
    config = NemotronHOmniConfig.from_transformers(_omni_config())
    decoder = _model(
        [
            "inputs_embeds",
            "attention_mask",
            "past_key_values.0.conv_state",
            "past_key_values.0.ssm_state",
            "past_key_values.2.key",
            "past_key_values.2.value",
        ],
        [
            "logits",
            "present.0.conv_state",
            "present.0.ssm_state",
            "present.2.key",
            "present.2.value",
        ],
    )
    return ModelPackage(
        {
            "decoder": decoder,
            "vision_encoder": _model(
                ["pixel_values", "is_video"],
                ["image_features", "retention_mask", "kept_indices"],
            ),
            "embedding": _model(["input_ids", "image_features"], ["inputs_embeds"]),
        },
        config=config,
    )


def test_omni_metadata_preserves_state_names_and_trailing_stateless_layer(tmp_path):
    package = _package()
    path = _write_genai_config(
        package.config,
        str(tmp_path),
        pkg=package,
        ort_model_type="nemotron_h_omni",
        ep="cuda",
        context_length=128,
        bos_token_id=1,
        eos_token_id=[2, 11],
        pad_token_id=0,
        is_vlm=True,
        has_speech=False,
    )
    data = json.loads(Path(path).read_text())
    model = data["model"]
    decoder = model["decoder"]
    assert model["type"] == "nemotron_h_omni"
    assert model["eos_token_id"] == [2, 11]
    assert model["image_token_id"] == 18
    assert decoder["num_hidden_layers"] == 4
    assert decoder["layer_types"] == ["mamba", "moe", "attention", "moe"]
    assert "conv_cache_size" not in decoder
    assert decoder["inputs"] == {
        "inputs_embeds": "inputs_embeds",
        "attention_mask": "attention_mask",
        "past_key_names": "past_key_values.%d.key",
        "past_value_names": "past_key_values.%d.value",
        "past_conv_names": "past_key_values.%d.conv_state",
        "past_recurrent_names": "past_key_values.%d.ssm_state",
    }
    assert decoder["outputs"] == {
        "logits": "logits",
        "present_key_names": "present.%d.key",
        "present_value_names": "present.%d.value",
        "present_conv_names": "present.%d.conv_state",
        "present_recurrent_names": "present.%d.ssm_state",
    }
    assert data["search"]["past_present_share_buffer"] is False
    assert (
        decoder["session_options"]["provider_options"][0]["cuda"]["enable_cuda_graph"] == "0"
    )
    assert model["vision"]["inputs"] == {
        "pixel_values": "pixel_values",
        "is_video": "is_video",
    }
    assert model["vision"]["outputs"] == {
        "image_features": "image_features",
        "retention_mask": "retention_mask",
        "kept_indices": "kept_indices",
    }
    assert "config_filename" not in model["vision"]
    assert "spatial_merge_size" not in model["vision"]
    assert model["embedding"]["inputs"] == {
        "input_ids": "input_ids",
        "image_features": "image_features",
    }
    assert model["embedding"]["outputs"] == {"inputs_embeds": "inputs_embeds"}


def test_sparse_omni_actual_graph_keeps_multimodal_cache_metadata_and_capture_off(tmp_path):
    config, package, _ = _fixture(sparse=True)
    path = _write_genai_config(
        config,
        str(tmp_path),
        pkg=package,
        ort_model_type="nemotron_h_omni",
        ep="cuda",
        context_length=128,
        bos_token_id=1,
        eos_token_id=[2, 11],
        pad_token_id=0,
        is_vlm=True,
        has_speech=False,
    )
    data = json.loads(Path(path).read_text())
    model = data["model"]
    decoder = model["decoder"]
    assert decoder["layer_types"] == ["mamba", "moe", "attention", "moe"]
    assert decoder["num_hidden_layers"] == 4
    assert decoder["inputs"]["inputs_embeds"] == "inputs_embeds"
    assert decoder["inputs"]["past_conv_names"] == "past_key_values.%d.conv_state"
    assert decoder["inputs"]["past_recurrent_names"] == "past_key_values.%d.ssm_state"
    assert decoder["inputs"]["past_key_names"] == "past_key_values.%d.key"
    assert decoder["outputs"]["present_key_names"] == "present.%d.key"
    assert {"embedding", "vision", "decoder"} <= set(model)
    assert data["search"]["past_present_share_buffer"] is False
    options = decoder["session_options"]["provider_options"][0]["cuda"]
    assert options["enable_cuda_graph"] == "0"


@pytest.mark.parametrize(
    "missing",
    ["present.0.ssm_state", "present.0.conv_state", "present.2.key", "present.2.value"],
)
def test_omni_metadata_rejects_unpaired_cache_outputs(missing):
    model = _package()["decoder"]
    model.graph.outputs.remove(
        next(value for value in model.graph.outputs if value.name == missing)
    )
    with pytest.raises(ValueError, match="must match"):
        _inspect_hybrid_multimodal_decoder_abi(model)


def test_omni_does_not_emit_fake_ort_extensions_image_processor(tmp_path):
    config = _package().config
    assert _write_vision_processor_config(config, str(tmp_path)) is None
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("revision", [None, "explicit-revision"])
def test_omni_writer_pins_config_generation_and_tokenizer_assets(
    monkeypatch, tmp_path, revision
):
    import transformers

    from mobius.integrations.transformers import _builder
    from mobius.models.nemotron_h_omni import BOLT_REVISION

    auto_export = importlib.import_module("mobius.integrations.ort_genai.auto_export")
    model_id = "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB"
    expected_revision = revision or BOLT_REVISION
    probe = mock.Mock(
        return_value=(
            SimpleNamespace(
                model_type="nemotron_h_omni",
                bos_token_id=1,
                eos_token_id=11,
                pad_token_id=0,
            ),
            True,
        )
    )
    generation = mock.Mock(
        return_value=SimpleNamespace(
            bos_token_id=1,
            eos_token_id=[2, 11],
            pad_token_id=0,
        )
    )
    copy = mock.Mock(return_value=[])
    monkeypatch.setattr(_builder, "_load_transformers_config", probe)
    monkeypatch.setattr(transformers.GenerationConfig, "from_pretrained", generation)
    monkeypatch.setattr(auto_export, "_copy_tokenizer_files", copy)
    result = write_ort_genai_config(
        _package(), str(tmp_path), hf_model_id=model_id, revision=revision, ep="cuda"
    )
    probe.assert_called_once_with(
        model_id, revision=expected_revision, trust_remote_code=False
    )
    generation.assert_called_once_with(model_id, revision=expected_revision)
    copy.assert_called_once_with(model_id, str(tmp_path), revision=expected_revision)
    data = json.loads(Path(result["genai_config"]).read_text())
    assert data["model"]["eos_token_id"] == [2, 11]
    assert "processor_config" not in result
