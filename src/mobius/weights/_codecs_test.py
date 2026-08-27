# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for typed quantization format codecs."""

from __future__ import annotations

import pytest
import torch

from mobius._component_manifest import ComponentDescriptor
from mobius._configs import QuantizationConfig
from mobius.weights import PackedWeight, QuantizationCodecRegistry, codec_registry


def _component() -> ComponentDescriptor:
    return ComponentDescriptor(
        name="decoder",
        module_path="decoder",
        role="decoder",
        source_paths=("model.layers",),
    )


def _config(method: str = "olive", *, sym: bool = True) -> QuantizationConfig:
    return QuantizationConfig(
        bits=4,
        group_size=16,
        quant_method=method,
        sym=sym,
    )


def test_groups_olive_sidecars_into_one_record():
    state_dict = {
        "model.q_proj.weight_qweight": torch.zeros(32, 32, dtype=torch.uint8),
        "model.q_proj.weight_scales": torch.ones(32, 4),
        "model.norm.weight": torch.ones(64),
    }

    bundle = codec_registry.get("olive").group(
        _component(),
        state_dict,
        _config(),
    )

    record = bundle["model.q_proj.weight"]
    assert isinstance(record.storage, PackedWeight)
    assert record.source_keys == (
        "model.q_proj.weight_qweight",
        "model.q_proj.weight_scales",
    )
    assert bundle["model.norm.weight"].is_quantized is False


def test_groups_gptq_dotted_sidecars():
    state_dict = {
        "model.q_proj.qweight": torch.zeros(8, 32, dtype=torch.int32),
        "model.q_proj.scales": torch.ones(4, 32),
    }

    bundle = codec_registry.get("gptq").group(
        _component(),
        state_dict,
        _config("gptq"),
    )

    assert bundle["model.q_proj.weight"].is_quantized is True


def test_rejects_missing_scales():
    state_dict = {
        "model.q_proj.weight_qweight": torch.zeros(32, 32, dtype=torch.uint8),
    }

    with pytest.raises(ValueError, match="missing scales"):
        codec_registry.get("olive").group(
            _component(),
            state_dict,
            _config(),
        )


def test_rejects_missing_asymmetric_zero_points():
    state_dict = {
        "model.q_proj.weight_qweight": torch.zeros(32, 32, dtype=torch.uint8),
        "model.q_proj.weight_scales": torch.ones(32, 4),
    }

    with pytest.raises(ValueError, match="missing zero points"):
        codec_registry.get("olive").group(
            _component(),
            state_dict,
            _config(sym=False),
        )


def test_rejects_orphan_sidecars():
    with pytest.raises(ValueError, match="no matching qweight"):
        codec_registry.get("olive").group(
            _component(),
            {"model.q_proj.weight_scales": torch.ones(32, 4)},
            _config(),
        )


def test_compatibility_normalizer_uses_existing_packer():
    state_dict = {
        "model.q_proj.weight_qweight": torch.zeros(32, 32, dtype=torch.uint8),
        "model.q_proj.weight_scales": torch.ones(32, 4),
    }
    codec = codec_registry.get("olive")
    record = codec.group(_component(), state_dict, _config())["model.q_proj.weight"]

    normalized = codec.normalize(record, _config())

    assert normalized["model.q_proj.weight"].shape == (32, 4, 8)
    assert normalized["model.q_proj.scales"].shape == (32, 4)


def test_registry_rejects_duplicate_method():
    registry = QuantizationCodecRegistry()
    codec = codec_registry.get("olive")
    registry.register(codec)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(codec)


def test_registry_reports_unknown_method():
    with pytest.raises(KeyError, match="Available methods"):
        QuantizationCodecRegistry().get("unknown")
