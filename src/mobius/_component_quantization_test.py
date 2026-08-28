# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for generic per-component quantization wiring."""

from __future__ import annotations

from typing import ClassVar

import torch
from onnxscript import OpBuilder, nn

from mobius._component_quantization import (
    configure_component_quantization,
    preprocess_component_quantized_state_dict,
)
from mobius._configs import ArchitectureConfig, QuantizationConfig
from mobius._model_package import ModelPackage
from mobius.components import (
    Embedding,
    Linear,
    QuantizedEmbedding,
    QuantizedLinear,
    make_quantized_linear_factory,
)
from mobius.tasks import ComponentSpec, ModelTask


class _Projection(nn.Module):
    def __init__(self, linear_class: type = Linear):
        super().__init__()
        self.proj = linear_class(64, 32, bias=False)

    def forward(self, op: OpBuilder, x):
        return self.proj(op, x)


class _Composite(nn.Module):
    HF_COMPONENT_SOURCES: ClassVar[dict[str, tuple[str, ...]]] = {
        "decoder": ("model.layers",),
        "vision_encoder": ("model.visual",),
        "audio_encoder": ("model.audio",),
        "embedding": ("model.embed_tokens",),
    }

    def __init__(self):
        super().__init__()
        quantized = make_quantized_linear_factory(bits=4, block_size=16)
        self.decoder = _Projection(quantized)
        self.vision_encoder = _Projection()
        self.audio_tower = _Projection()
        self.embedding = _Projection(quantized)


class _CompositeTask(ModelTask):
    model_roles: ClassVar[dict[str, str]] = {
        "decoder": "decoder",
        "vision_encoder": "encoder",
        "audio_encoder": "encoder",
        "embedding": "embedding",
    }
    components = ComponentSpec(
        decoder="decoder",
        vision_encoder="vision_encoder",
        audio_encoder="audio_tower",
        embedding="embedding",
    )

    def build(self, module, config) -> ModelPackage:
        raise NotImplementedError


class _SingleTask(ModelTask):
    model_roles: ClassVar[dict[str, str]] = {"model": "decoder"}

    def build(self, module, config) -> ModelPackage:
        raise NotImplementedError


def _config() -> ArchitectureConfig:
    decoder = QuantizationConfig(
        bits=4,
        group_size=16,
        quant_method="olive",
        sym=True,
    )
    return ArchitectureConfig(
        quantization=decoder,
        component_quantization={
            "decoder": decoder,
            "vision_encoder": QuantizationConfig(
                bits=8,
                group_size=32,
                quant_method="olive",
                sym=True,
            ),
            "audio_encoder": QuantizationConfig(
                bits=2,
                group_size=16,
                quant_method="olive",
                sym=True,
            ),
        },
    )


def test_configures_quantized_and_float_components_independently():
    module = _Composite()

    configure_component_quantization(module, _config(), _CompositeTask())

    assert isinstance(module.decoder.proj, QuantizedLinear)
    assert (module.decoder.proj._bits, module.decoder.proj._block_size) == (4, 16)
    assert isinstance(module.vision_encoder.proj, QuantizedLinear)
    assert (module.vision_encoder.proj._bits, module.vision_encoder.proj._block_size) == (
        8,
        32,
    )
    assert isinstance(module.audio_tower.proj, QuantizedLinear)
    assert (module.audio_tower.proj._bits, module.audio_tower.proj._block_size) == (
        2,
        16,
    )
    assert type(module.embedding.proj) is Linear


def test_preprocesses_raw_weights_with_component_layouts():
    module = _Composite()
    config = _config()
    state_dict = {
        "decoder.proj.weight_qweight": torch.zeros(32, 32, dtype=torch.uint8),
        "decoder.proj.weight_scales": torch.ones(32, 4),
        "vision_encoder.proj.weight_qweight": torch.zeros(32, 64, dtype=torch.uint8),
        "vision_encoder.proj.weight_scales": torch.ones(32, 2),
        "audio_tower.proj.weight_qweight": torch.zeros(32, 16, dtype=torch.uint8),
        "audio_tower.proj.weight_scales": torch.ones(32, 4),
        "embedding.proj.weight": torch.ones(32, 64),
    }

    result = preprocess_component_quantized_state_dict(
        state_dict,
        module,
        config,
        _CompositeTask(),
        ("decoder", "vision_encoder", "audio_encoder", "embedding"),
    )

    assert result["decoder.proj.weight"].shape == (32, 4, 8)
    assert result["vision_encoder.proj.weight"].shape == (32, 2, 32)
    assert result["audio_tower.proj.weight"].shape == (32, 4, 4)
    assert result["embedding.proj.weight"].shape == (32, 64)


def test_single_graph_uses_decoder_layout_and_ignores_split_metadata():
    decoder = QuantizationConfig(
        bits=4,
        group_size=16,
        quant_method="olive",
        sym=True,
    )
    config = ArchitectureConfig(
        quantization=decoder,
        component_quantization={
            "decoder": decoder,
            "vision_encoder": QuantizationConfig(
                bits=8,
                group_size=32,
                quant_method="olive",
            ),
        },
    )
    module = _Projection()

    configure_component_quantization(module, config, _SingleTask())

    assert isinstance(module.proj, QuantizedLinear)
    assert (module.proj._bits, module.proj._block_size) == (4, 16)


def test_quantize_embeddings_only_rewrites_input_token_table():
    class _EmbeddingModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = Embedding(256, 64)
            self.embed_positions = Embedding(128, 64)

    quantization = QuantizationConfig(
        bits=4,
        group_size=16,
        quant_method="olive",
        quantize_embeddings=True,
    )
    config = ArchitectureConfig(
        quantization=quantization,
        component_quantization={"model": quantization},
    )
    module = _EmbeddingModule()

    configure_component_quantization(module, config, _SingleTask())

    assert isinstance(module.embed_tokens, QuantizedEmbedding)
    assert type(module.embed_positions) is Embedding
