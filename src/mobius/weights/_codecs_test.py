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
        module_attribute_path="decoder",
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

    codec = codec_registry.get("olive")
    record = codec.group(_component(), state_dict, _config(sym=False))["model.q_proj.weight"]

    with pytest.raises(ValueError, match="missing zero points"):
        codec.normalize(record, _config(sym=False))

    # A symmetric module override does not require the component's zero points.
    assert codec.normalize(record, _config(sym=True))["model.q_proj.weight"].shape == (
        32,
        4,
        8,
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


@pytest.mark.parametrize(("method", "bits"), [("gptq", 4), ("gptq", 8), ("awq", 4)])
def test_dotted_affine_embedding_normalizes_to_gather_bytes(method, bits):
    component = ComponentDescriptor("embedding", "embedding", "embedding")
    config = QuantizationConfig(
        bits=bits,
        group_size=16,
        quant_method=method,
        sym=True,
        quantize_embeddings=True,
    )
    qweight = (
        torch.arange(bits * 8, dtype=torch.int32).reshape(bits, 8) * 0x010203 + 0x12345678
    )
    scales = torch.linspace(0.125, 1.125, 16).reshape(2, 8)
    codec = codec_registry.get(method)
    record = codec.group(
        component,
        {"embedding.tokens.qweight": qweight, "embedding.tokens.scales": scales},
        config,
    )["embedding.tokens.weight"]

    normalized = codec.normalize(record, config, kind="embedding")

    expected = (
        torch.stack([(qweight >> shift) & 255 for shift in (0, 8, 16, 24)], dim=-1)
        .transpose(0, 1)
        .reshape(8, 32 * bits // 8)
        .to(torch.uint8)
    )
    assert set(normalized) == {"embedding.tokens.qweight", "embedding.tokens.scales"}
    assert normalized["embedding.tokens.qweight"].dtype == torch.uint8
    torch.testing.assert_close(normalized["embedding.tokens.qweight"], expected)
    torch.testing.assert_close(normalized["embedding.tokens.scales"], scales.T)


def test_registry_rejects_duplicate_method():
    registry = QuantizationCodecRegistry()
    codec = codec_registry.get("olive")
    registry.register(codec)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(codec)


def test_registry_reports_unknown_method():
    with pytest.raises(KeyError, match="Available methods"):
        QuantizationCodecRegistry().get("unknown")
