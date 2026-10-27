# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import copy
import dataclasses
import json

import onnx_ir as ir
import pytest
from _test_configs import FISH_OMNI_CONFIG

from mobius import build_from_module
from mobius._configs.fish_omni import FishOmniConfig
from mobius._registry import registry
from mobius.integrations.transformers._config_resolver import (
    _config_from_hf,
    _dict_to_pretrained_config,
)
from mobius.models import FishOmniModel


def tiny_config():
    return FishOmniConfig.from_transformers(
        _dict_to_pretrained_config(copy.deepcopy(FISH_OMNI_CONFIG))
    )


def test_fish_omni_build():
    dtype = ir.DataType.FLOAT
    config = tiny_config()
    package = build_from_module(FishOmniModel(config), config, task="fish-omni")
    assert set(package) == {"decoder", "embedding", "audio_decoder"}
    assert {v.name for v in package["embedding"].graph.outputs} == {
        "inputs_embeds",
        "shared_input",
    }
    decoder = package["decoder"]
    assert "past_key_values.0.recurrent_state" in {v.name for v in decoder.graph.inputs}
    recurrent = next(v for v in decoder.graph.inputs if v.name.endswith("recurrent_state"))
    assert recurrent.dtype == ir.DataType.FLOAT
    assert {v.name for v in decoder.graph.outputs} >= {"logits", "hidden_states"}
    tower = package["audio_decoder"]
    assert tuple(tower.graph.outputs[0].shape)[-2:] == (2, 16)
    assert sum(v.name.startswith("past_key_values.") for v in tower.graph.inputs) == 6
    assert any("q_norm.weight" in key for key in tower.graph.initializers)
    assert all(
        v.const_value is not None or v.dtype == dtype
        for m in package.values()
        for v in m.graph.initializers.values()
    )


def test_fish_omni_registry_config_resolution():
    parent = _dict_to_pretrained_config(copy.deepcopy(FISH_OMNI_CONFIG))
    resolved = _config_from_hf(
        parent.text_config, parent_config=parent, module_class=registry.get("fish_omni")
    )
    assert isinstance(resolved, FishOmniConfig)
    assert resolved.partial_rotary_factor == resolved.tower.partial_rotary_factor
    assert resolved.tower.partial_rotary_factor == pytest.approx(0.25)
    assert resolved.tower.rope_interleave
    assert resolved.tower.max_position_embeddings == resolved.max_position_embeddings
    assert resolved.quantization is None


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
def test_fish_omni_rejects_unqualified_precision(dtype):
    with pytest.raises(ValueError, match="requires dtype='f32'"):
        FishOmniModel(dataclasses.replace(tiny_config(), dtype=dtype))


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("text_config", "attn_output_gate", False),
        ("text_config", "layer_types", ["full_attention"]),
        ("text_config", "tie_word_embeddings", True),
        ("audio_decoder_config", "vq_feature_placement", "other"),
        ("audio_decoder_config", "dim", 64),
        ("audio_decoder_config", "rotary_interleaved", False),
        ("audio_decoder_config", "vocab_size", 17),
    ],
)
def test_fish_omni_rejects_incompatible_configs(section, key, value):
    raw = copy.deepcopy(FISH_OMNI_CONFIG)
    raw[section][key] = value
    with pytest.raises(ValueError):
        FishOmniConfig.from_transformers(_dict_to_pretrained_config(raw))


def test_fish_omni_ort_genai_metadata_rejects_incomplete_runtime(tmp_path):
    from mobius.integrations.ort_genai import write_ort_genai_config

    config = tiny_config()
    package = build_from_module(FishOmniModel(config), config, task="fish-omni")
    output = tmp_path / "ort-genai"
    with pytest.raises(ValueError, match="next-frame codebook sampling"):
        write_ort_genai_config(package, str(output), hf_model_id="must-not-download")
    assert not output.exists()


def test_fish_omni_onnx_genai_metadata_is_graph_derived_and_advisory(tmp_path):
    from mobius.integrations.onnx_genai import write_onnx_genai_config

    config = tiny_config()
    package = build_from_module(FishOmniModel(config), config, task="fish-omni")
    artifacts = write_onnx_genai_config(package, str(tmp_path))
    metadata = json.loads((tmp_path / "inference_metadata.yaml").read_text())
    assert set(artifacts) == {"inference_metadata", "runtime_compatibility"}
    assert metadata["runtime_validation_status"] == "unsupported-by-tested-runtime"
    assert set(metadata["components"]) == set(package)
    for name, model in package.items():
        component = metadata["components"][name]
        assert component["filename"] == f"{name}/model.onnx"
        assert component["inputs"] == [value.name for value in model.graph.inputs]
        assert component["outputs"] == [value.name for value in model.graph.outputs]
