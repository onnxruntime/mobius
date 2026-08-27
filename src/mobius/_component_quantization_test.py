# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for authoritative component quantization plans."""

from __future__ import annotations

from typing import ClassVar

import pytest
import torch
from onnxscript import nn

from mobius._component_quantization import (
    configure_component_quantization,
    normalize_component_quantized_weights,
    validate_quantized_component_bindings,
)
from mobius._configs import ArchitectureConfig, QuantizationConfig
from mobius._model_package import ModelPackage
from mobius.components import (
    Linear,
    QuantizedEmbedding,
    QuantizedLinear,
    make_quantized_linear_factory,
)
from mobius.tasks import ComponentSpec, ModelTask


class _DecoderLayer(nn.Module):
    def __init__(self, linear_class: type[nn.Module]):
        super().__init__()
        self.q_proj = linear_class(64, 64, bias=False)
        self.per_layer_input_gate = linear_class(64, 48, bias=False)
        self.per_layer_projection = linear_class(64, 48, bias=False)


class _Backbone(nn.Module):
    def __init__(self, linear_class: type[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList([_DecoderLayer(linear_class)])


class _Decoder(nn.Module):
    def __init__(self, linear_class: type[nn.Module]):
        super().__init__()
        self.model = _Backbone(linear_class)


class _Projection(nn.Module):
    def __init__(self, linear_class: type[nn.Module] = Linear):
        super().__init__()
        self.proj = linear_class(64, 32, bias=False)


class _Composite(nn.Module):
    HF_COMPONENT_SOURCES: ClassVar[dict[str, tuple[str, ...]]] = {
        "decoder": ("model.language_model.layers", "lm_head"),
        "audio_encoder": ("model.audio_tower",),
        "embedding": ("model.language_model.embed_tokens",),
    }

    def __init__(self):
        super().__init__()
        root_quantized = make_quantized_linear_factory(bits=4, block_size=16)
        self.decoder = _Decoder(root_quantized)
        self.audio_tower = _Projection()
        self.embedding = _Projection(root_quantized)


class _CompositeTask(ModelTask):
    model_roles: ClassVar[dict[str, str]] = {
        "decoder": "decoder",
        "audio_encoder": "encoder",
        "embedding": "embedding",
    }
    components = ComponentSpec(
        decoder="decoder",
        audio_encoder="audio_tower",
        embedding="embedding",
    )

    def build(self, module, config) -> ModelPackage:
        raise NotImplementedError


def _config() -> ArchitectureConfig:
    decoder = QuantizationConfig(
        bits=4,
        group_size=16,
        quant_method="olive",
        sym=True,
        modules_to_not_convert=(
            "lm_head",
            r"re:.*\.per_layer_input_gate",
            r"re:.*\.per_layer_projection",
        ),
    )
    return ArchitectureConfig(
        quantization=decoder,
        component_quantization={
            "decoder": decoder,
            "audio_encoder": QuantizationConfig(
                bits=8,
                group_size=32,
                quant_method="olive",
                sym=True,
            ),
        },
    )


def test_component_plan_applies_regex_exclusions_per_linear():
    module = _Composite()

    configure_component_quantization(module, _config(), _CompositeTask())

    layer = module.decoder.model.layers[0]
    assert isinstance(layer.q_proj, QuantizedLinear)
    assert (layer.q_proj._bits, layer.q_proj._block_size) == (4, 16)
    assert type(layer.per_layer_input_gate) is Linear
    assert type(layer.per_layer_projection) is Linear
    assert isinstance(module.audio_tower.proj, QuantizedLinear)
    assert (module.audio_tower.proj._bits, module.audio_tower.proj._block_size) == (
        8,
        32,
    )
    # The mapping is authoritative: an omitted component stays float even
    # though the top-level module was initially built from the decoder config.
    assert type(module.embedding.proj) is Linear


def test_specialized_quantized_subclass_fails_instead_of_losing_semantics():
    class _SpecialQuantizedLinear(QuantizedLinear):
        def forward(self, op, x):
            return super().forward(op, x)

    module = _Projection(_SpecialQuantizedLinear)
    config = ArchitectureConfig(
        component_quantization={
            "model": QuantizationConfig(
                bits=8,
                group_size=32,
                quant_method="olive",
            )
        }
    )

    with pytest.raises(TypeError, match="specialized quantized module"):
        configure_component_quantization(module, config, _SingleTask())


def test_normalizes_weights_with_component_module_path_routing():
    module = _Composite()
    config = _config()
    task = _CompositeTask()
    manifest = configure_component_quantization(module, config, task)
    state_dict = {
        "decoder.model.layers.0.q_proj.weight_qweight": torch.zeros(64, 32, dtype=torch.uint8),
        "decoder.model.layers.0.q_proj.weight_scales": torch.ones(64, 4),
        "decoder.model.layers.0.per_layer_input_gate.weight": torch.ones(48, 64),
        "decoder.model.layers.0.per_layer_projection.weight": torch.ones(48, 64),
        "audio_tower.proj.weight_qweight": torch.zeros(32, 64, dtype=torch.uint8),
        "audio_tower.proj.weight_scales": torch.ones(32, 2),
        "embedding.proj.weight": torch.ones(32, 64),
    }

    result = normalize_component_quantized_weights(
        state_dict,
        module,
        config,
        ("decoder", "audio_encoder", "embedding"),
        manifest=manifest,
        task=task,
    )

    assert result["decoder.model.layers.0.q_proj.weight"].shape == (64, 4, 8)
    assert result["audio_tower.proj.weight"].shape == (32, 2, 32)
    assert result["decoder.model.layers.0.per_layer_input_gate.weight"].shape == (
        48,
        64,
    )


def test_rejects_packed_weight_for_excluded_module():
    module = _Composite()
    config = _config()
    task = _CompositeTask()
    manifest = configure_component_quantization(module, config, task)
    state_dict = {
        "decoder.model.layers.0.per_layer_input_gate.weight_qweight": torch.zeros(
            48, 32, dtype=torch.uint8
        ),
        "decoder.model.layers.0.per_layer_input_gate.weight_scales": torch.ones(48, 4),
    }

    with pytest.raises(ValueError, match="excluded"):
        normalize_component_quantized_weights(
            state_dict,
            module,
            config,
            ("decoder", "audio_encoder", "embedding"),
            manifest=manifest,
            task=task,
        )


class _SingleTask(ModelTask):
    model_roles: ClassVar[dict[str, str]] = {"model": "decoder"}

    def build(self, module, config) -> ModelPackage:
        raise NotImplementedError


class _QuantizedEmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = QuantizedEmbedding(
            32,
            64,
            bits=4,
            block_size=16,
            has_zero_point=False,
        )
        self.proj = QuantizedLinear(
            64,
            32,
            bits=4,
            block_size=16,
            has_zero_point=False,
        )


def test_canonical_quantized_embedding_is_not_treated_as_raw_sidecars():
    module = _QuantizedEmbeddingModel()
    quantization = QuantizationConfig(
        bits=4,
        group_size=16,
        quant_method="olive",
        sym=True,
        quantize_embeddings=True,
    )
    config = ArchitectureConfig(
        quantization=quantization,
        component_quantization={"model": quantization},
    )
    task = _SingleTask()
    manifest = configure_component_quantization(module, config, task)
    state_dict = {
        "embed_tokens.qweight": torch.zeros(32, 32, dtype=torch.uint8),
        "embed_tokens.scales": torch.ones(32, 4),
        "proj.weight": torch.zeros(32, 4, 8, dtype=torch.uint8),
        "proj.scales": torch.ones(32, 4),
    }

    result = normalize_component_quantized_weights(
        state_dict,
        module,
        config,
        ("model",),
        manifest=manifest,
        task=task,
    )

    assert set(result) == set(state_dict)
    assert result["embed_tokens.qweight"] is state_dict["embed_tokens.qweight"]
    assert result["embed_tokens.scales"] is state_dict["embed_tokens.scales"]
    assert result["proj.weight"] is state_dict["proj.weight"]
    assert result["proj.scales"] is state_dict["proj.scales"]


def test_binding_validator_rejects_unfilled_quantized_parameters():
    from mobius._testing import create_test_builder, create_test_input
    from mobius.tasks._base import _make_model

    linear = QuantizedLinear(
        64,
        32,
        bits=4,
        block_size=16,
        has_zero_point=False,
    )
    builder, op, graph = create_test_builder()
    x = create_test_input(builder, "x", [1, 64])
    output = linear(op, x)
    builder._adapt_outputs([output], "")
    quantization = QuantizationConfig(
        bits=4,
        group_size=16,
        quant_method="olive",
    )

    with pytest.raises(ValueError, match="unbound MatMulNBits parameter"):
        validate_quantized_component_bindings(
            {"model": _make_model(graph)},
            ArchitectureConfig(
                quantization=quantization,
                component_quantization={"model": quantization},
            ),
        )
