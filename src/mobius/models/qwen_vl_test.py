# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Qwen-VL weight preprocessing with tie_word_embeddings."""

from __future__ import annotations

import dataclasses

import torch

from mobius._builder import build_from_module
from mobius._component_quantization import (
    configure_component_quantization,
    preprocess_component_quantized_state_dict,
)
from mobius._configs import ArchitectureConfig, QuantizationConfig, VisionConfig
from mobius.models.qwen_vl import (
    Qwen3VL3ModelCausalLMModel,
    Qwen3VLDecoderModel,
    Qwen25VLCausalLMModel,
    Qwen25VLDecoderModel,
)

# Tiny config for weight preprocessing tests (no graph build needed)
_BASE_CONFIG = ArchitectureConfig(
    model_type="qwen2_5_vl",
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    vocab_size=100,
    rms_norm_eps=1e-6,
    tie_word_embeddings=True,
    hidden_act="silu",
    vision=VisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=28,
        patch_size=14,
    ),
)


def _fake_state_dict_qwen25vl() -> dict[str, torch.Tensor]:
    """State dict mimicking HF Qwen2.5-VL with tie_word_embeddings=True.

    HF keys: model.embed_tokens.weight, model.layers.*, lm_head.weight (absent).
    """
    embed = torch.randn(100, 64)
    return {
        "model.embed_tokens.weight": embed,
        "model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
        "model.norm.weight": torch.randn(64),
    }


def _fake_state_dict_qwen3vl() -> dict[str, torch.Tensor]:
    """State dict mimicking HF Qwen3-VL with tie_word_embeddings=True.

    HF keys: model.visual.*, model.language_model.embed_tokens.weight,
    model.language_model.layers.*, model.language_model.lm_head.weight (absent).
    """
    embed = torch.randn(100, 64)
    return {
        "model.visual.patch_embed.proj.weight": torch.randn(64, 3, 14, 14),
        "model.language_model.embed_tokens.weight": embed,
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
        "model.language_model.model.norm.weight": torch.randn(64),
    }


class TestQwen25VLCausalLMModelTiedWeights:
    """Composite 3-model CausalLM: decoder + vision + embedding."""

    def test_lm_head_present_when_tied(self):
        config = dataclasses.replace(_BASE_CONFIG, model_type="qwen2_5_vl")
        model = Qwen25VLCausalLMModel(config)
        sd = _fake_state_dict_qwen25vl()
        result = model.preprocess_weights(sd)

        assert "decoder.lm_head.weight" in result, (
            "lm_head.weight must be present for tied composite models"
        )

    def test_lm_head_shares_data_ptr_with_embed(self):
        config = dataclasses.replace(_BASE_CONFIG, model_type="qwen2_5_vl")
        model = Qwen25VLCausalLMModel(config)
        sd = _fake_state_dict_qwen25vl()
        result = model.preprocess_weights(sd)

        embed = result["decoder.model.embed_tokens.weight"]
        head = result["decoder.lm_head.weight"]
        assert embed.data_ptr() == head.data_ptr(), (
            "Tied weights must share the same data_ptr() for ONNX dedup"
        )

    def test_builds_different_decoder_vision_and_embedding_layouts(self):
        decoder = QuantizationConfig(
            bits=4,
            group_size=16,
            quant_method="olive",
            sym=True,
        )
        config = dataclasses.replace(
            _BASE_CONFIG,
            tie_word_embeddings=False,
            quantization=decoder,
            component_quantization={
                "decoder": decoder,
                "vision_encoder": QuantizationConfig(
                    bits=8,
                    group_size=32,
                    quant_method="olive",
                    sym=True,
                ),
                "embedding": QuantizationConfig(
                    bits=2,
                    group_size=16,
                    quant_method="olive",
                    sym=True,
                    quantize_embeddings=True,
                ),
            },
        )

        package = build_from_module(
            Qwen25VLCausalLMModel(config),
            config,
            task="qwen-vl",
        )

        decoder_layouts = {
            (
                node.attributes["bits"].as_int(),
                node.attributes["block_size"].as_int(),
            )
            for node in package["decoder"].graph
            if node.op_type == "MatMulNBits"
        }
        vision_layouts = {
            (
                node.attributes["bits"].as_int(),
                node.attributes["block_size"].as_int(),
            )
            for node in package["vision_encoder"].graph
            if node.op_type == "MatMulNBits"
        }

        assert decoder_layouts == {(4, 16)}
        assert vision_layouts == {(8, 32)}
        assert any(
            node.op_type == "GatherBlockQuantized"
            and node.attributes["bits"].as_int() == 2
            and node.attributes["block_size"].as_int() == 16
            for node in package["embedding"].graph
        )

    def test_packed_embedding_routes_only_to_embedding_component(self):
        decoder = QuantizationConfig(
            bits=4,
            group_size=16,
            quant_method="olive",
            sym=True,
        )
        config = dataclasses.replace(
            _BASE_CONFIG,
            tie_word_embeddings=False,
            quantization=decoder,
            component_quantization={
                "decoder": decoder,
                "embedding": QuantizationConfig(
                    bits=2,
                    group_size=16,
                    quant_method="olive",
                    sym=True,
                    quantize_embeddings=True,
                ),
            },
        )
        model = Qwen25VLCausalLMModel(config)
        configure_component_quantization(model, config, "qwen-vl")
        renamed = model.preprocess_weights(
            {
                "model.embed_tokens.weight_qweight": torch.zeros(100, 16, dtype=torch.uint8),
                "model.embed_tokens.weight_scales": torch.ones(100, 4),
            }
        )

        assert not any(key.startswith("decoder.") for key in renamed)
        result = preprocess_component_quantized_state_dict(
            renamed,
            model,
            config,
            "qwen-vl",
            ("decoder", "vision_encoder", "embedding"),
        )

        assert result["embedding.embed_tokens.qweight"].shape == (100, 16)
        assert result["embedding.embed_tokens.scales"].shape == (100, 4)


class TestQwen25VLDecoderModelTiedWeights:
    """Standalone decoder (no composite prefix)."""

    def test_lm_head_present_when_tied(self):
        config = dataclasses.replace(_BASE_CONFIG, model_type="qwen2_5_vl")
        model = Qwen25VLDecoderModel(config)
        sd = _fake_state_dict_qwen25vl()
        result = model.preprocess_weights(sd)

        assert "lm_head.weight" in result

    def test_lm_head_shares_data_ptr_with_embed(self):
        config = dataclasses.replace(_BASE_CONFIG, model_type="qwen2_5_vl")
        model = Qwen25VLDecoderModel(config)
        sd = _fake_state_dict_qwen25vl()
        result = model.preprocess_weights(sd)

        embed = result["model.embed_tokens.weight"]
        head = result["lm_head.weight"]
        assert embed.data_ptr() == head.data_ptr()


class TestQwen3VL3ModelCausalLMModelTiedWeights:
    """Composite 3-model CausalLM for Qwen3-VL."""

    def test_lm_head_present_when_tied(self):
        config = dataclasses.replace(
            _BASE_CONFIG,
            model_type="qwen3_vl",
        )
        model = Qwen3VL3ModelCausalLMModel(config)
        sd = _fake_state_dict_qwen3vl()
        result = model.preprocess_weights(sd)

        assert "decoder.lm_head.weight" in result

    def test_lm_head_shares_data_ptr_with_embed(self):
        config = dataclasses.replace(
            _BASE_CONFIG,
            model_type="qwen3_vl",
        )
        model = Qwen3VL3ModelCausalLMModel(config)
        sd = _fake_state_dict_qwen3vl()
        result = model.preprocess_weights(sd)

        embed = result["decoder.model.embed_tokens.weight"]
        head = result["decoder.lm_head.weight"]
        assert embed.data_ptr() == head.data_ptr()


class TestQwen3VLDecoderModelTiedWeights:
    """Standalone Qwen3-VL decoder."""

    def test_lm_head_present_when_tied(self):
        config = dataclasses.replace(
            _BASE_CONFIG,
            model_type="qwen3_vl",
        )
        model = Qwen3VLDecoderModel(config)
        sd = _fake_state_dict_qwen3vl()
        result = model.preprocess_weights(sd)

        assert "lm_head.weight" in result

    def test_lm_head_shares_data_ptr_with_embed(self):
        config = dataclasses.replace(
            _BASE_CONFIG,
            model_type="qwen3_vl",
        )
        model = Qwen3VLDecoderModel(config)
        sd = _fake_state_dict_qwen3vl()
        result = model.preprocess_weights(sd)

        embed = result["embed_tokens.weight"]
        head = result["lm_head.weight"]
        assert embed.data_ptr() == head.data_ptr()
