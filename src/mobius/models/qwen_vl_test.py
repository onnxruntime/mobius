# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Qwen-VL weight preprocessing with tie_word_embeddings."""

from __future__ import annotations

import dataclasses

import torch

from mobius._configs import ArchitectureConfig, VisionConfig
from mobius.models.qwen_vl import (
    Qwen25VLCausalLMModel,
    Qwen25VLDecoderModel,
    Qwen3VL3ModelCausalLMModel,
    Qwen3VLDecoderModel,
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
            _BASE_CONFIG, model_type="qwen3_vl",
        )
        model = Qwen3VL3ModelCausalLMModel(config)
        sd = _fake_state_dict_qwen3vl()
        result = model.preprocess_weights(sd)

        assert "decoder.lm_head.weight" in result

    def test_lm_head_shares_data_ptr_with_embed(self):
        config = dataclasses.replace(
            _BASE_CONFIG, model_type="qwen3_vl",
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
            _BASE_CONFIG, model_type="qwen3_vl",
        )
        model = Qwen3VLDecoderModel(config)
        sd = _fake_state_dict_qwen3vl()
        result = model.preprocess_weights(sd)

        assert "lm_head.weight" in result

    def test_lm_head_shares_data_ptr_with_embed(self):
        config = dataclasses.replace(
            _BASE_CONFIG, model_type="qwen3_vl",
        )
        model = Qwen3VLDecoderModel(config)
        sd = _fake_state_dict_qwen3vl()
        result = model.preprocess_weights(sd)

        embed = result["embed_tokens.weight"]
        head = result["lm_head.weight"]
        assert embed.data_ptr() == head.data_ptr()
