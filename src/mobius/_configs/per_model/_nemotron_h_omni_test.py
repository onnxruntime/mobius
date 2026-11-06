# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Omni extraction must preserve the nested decoder and packed source formats."""

from __future__ import annotations

import copy
import types

import pytest

from mobius._configs import NemotronHOmniConfig
from mobius._configs.per_model._nemotron_h_puzzle_test import _source_config
from mobius.integrations.transformers._config_resolver import _config_from_hf


def _omni_config():
    return {
        "model_type": "nemotron_h_omni",
        "llm_config": _source_config(),
        "vision_config": {
            "model_type": "radio",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "mlp_ratio": 4,
            "patch_size": 4,
            "image_size": 8,
            "max_img_size": 16,
            "num_cls_tokens": 3,
            "num_registers": 7,
            "summary_idxs": [0, 1],
        },
        "img_context_token_id": 18,
        "eos_token_id": 11,
        "video_temporal_patch_size": 2,
        "projector_hidden_size": 24,
        "video_pruning_rate": 0.7,
        "ps_version": "v2",
        "downsample_ratio": 0.5,
        "vision_final_layernorm": False,
        "quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                "language_model.backbone.layers.1.mixer.experts.0.up_proj": {
                    "quant_algo": "IQ2_XXS",
                    "group_size": 256,
                    "block_payload_bytes": 66,
                    "packing": "ggml",
                },
                "language_model.lm_head": {
                    "quant_algo": "Q8_0",
                    "group_size": 32,
                    "block_payload_bytes": 34,
                    "packing": "ggml",
                },
                "language_model.backbone.layers.0.mixer.in_proj": {
                    "quant_algo": "W4A16_NVFP4",
                    "group_size": 16,
                },
            },
        },
    }


@pytest.mark.parametrize("native", [False, True])
def test_omni_nested_topology_and_packed_formats(native):
    source = _omni_config()
    original = copy.deepcopy(source)
    if native:
        source = types.SimpleNamespace(**source)
    config = _config_from_hf(
        source,
        module_class=types.SimpleNamespace(config_class=NemotronHOmniConfig),
    )
    assert isinstance(config, NemotronHOmniConfig)
    assert config.model_type == "nemotron_h_omni"
    assert config.num_hidden_layers == 4
    assert config.for_layer(3).num_local_experts == 3
    assert config.for_layer(3).moe_intermediate_size == 12
    assert config.vision.hidden_size == 16
    assert config.vision.intermediate_size == 64
    assert config.radio_max_img_size == 16
    assert config.radio_num_cls_tokens == 3
    assert config.radio_num_registers == 7
    assert config.video_temporal_patch_size == 2
    assert config.projector_hidden_size == 24
    assert config.image_token_id == 18
    assert config.eos_token_id == 11
    assert config.quantization is None
    assert len(config.quantized_projections) == 3
    assert {v.quant_algo for v in config.quantized_projections.values()} == {
        "IQ2_XXS",
        "Q8_0",
        "W4A16_NVFP4",
    }
    if not native:
        assert source == original


@pytest.mark.parametrize(
    ("field", "value"),
    [("group_size", 32), ("packing", "affine"), ("block_payload_bytes", 65)],
)
def test_omni_rejects_incompatible_iq2_packing(field, value):
    source = _omni_config()
    projection = next(iter(source["quantization_config"]["quantized_layers"].values()))
    projection[field] = value
    with pytest.raises(ValueError, match=r"group size|block packing"):
        NemotronHOmniConfig.from_transformers(source)


def test_omni_rejects_unknown_format_without_dense_fallback():
    source = _omni_config()
    projection = next(iter(source["quantization_config"]["quantized_layers"].values()))
    projection["quant_algo"] = "IQ2_XS"
    with pytest.raises(NotImplementedError, match="projection format"):
        NemotronHOmniConfig.from_transformers(source)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("llm_config", None, "nested"),
        ("vision_config", None, "RADIO"),
        ("sound_config", {}, "sound"),
        ("downsample_ratio", 0.25, "downsampling"),
        ("ps_version", "v1", "downsampling"),
        ("vision_final_layernorm", True, "final vision"),
        ("img_context_token_id", None, "img_context"),
    ],
)
def test_omni_rejects_unsupported_or_missing_components(field, value, message):
    source = _omni_config()
    source[field] = value
    with pytest.raises((ValueError, NotImplementedError), match=message):
        NemotronHOmniConfig.from_transformers(source)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hidden_act", "silu"),
        ("use_swiglu_ffn", True),
        ("qkv_bias", False),
        ("layerscale_value", 0.5),
        ("num_channels", 1),
    ],
)
def test_omni_rejects_incompatible_radio_flags(field, value):
    source = _omni_config()
    source["vision_config"][field] = value
    with pytest.raises(NotImplementedError, match=field):
        NemotronHOmniConfig.from_transformers(source)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("norm_mean", [0.0, 0.0]),
        ("norm_mean", [0.0, float("nan"), 0.0]),
        ("norm_std", [1.0, 0.0, 1.0]),
        ("norm_std", [1.0, -1.0, 1.0]),
        ("norm_std", [1.0, float("inf"), 1.0]),
        ("norm_std", [1.0, 1.0, 1.0, 1.0]),
    ],
)
def test_omni_rejects_invalid_processor_normalization(field, value):
    source = _omni_config()
    source["vision_config"][field] = value
    with pytest.raises(ValueError, match="RGB"):
        NemotronHOmniConfig.from_transformers(source)


def test_omni_preserves_custom_processor_normalization():
    source = _omni_config()
    source["vision_config"].update(norm_mean=[0.1, 0.2, 0.3], norm_std=[0.4, 0.5, 0.6])
    config = NemotronHOmniConfig.from_transformers(source)
    assert config.radio_norm_mean == (0.1, 0.2, 0.3)
    assert config.radio_norm_std == (0.4, 0.5, 0.6)
