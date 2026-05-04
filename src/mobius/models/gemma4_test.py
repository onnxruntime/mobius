# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Gemma4 preprocess_weights — MoE expert rename and router scale."""

from __future__ import annotations

import torch

from mobius._configs import Gemma4Config
from mobius.models.gemma4 import Gemma4CausalLMModel, Gemma4Model


def _tiny_gemma4_config(**overrides) -> Gemma4Config:
    """Create a minimal Gemma4Config for preprocess_weights tests."""
    from mobius._configs import VisionConfig

    defaults = dict(
        model_type="gemma4",
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        head_dim=16,
        global_head_dim=32,
        hidden_act="gelu",
        enable_moe_block=True,
        num_local_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        layer_types=["sliding_attention", "full_attention"],
        attention_k_eq_v=True,
        num_global_key_value_heads=1,
        vision=VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=16,
            patch_size=4,
        ),
    )
    defaults.update(overrides)
    return Gemma4Config(**defaults)


class TestGemma4CausalLMPreprocessWeights:
    """Test Gemma4CausalLMModel.preprocess_weights."""

    def test_expert_weight_rename(self):
        config = _tiny_gemma4_config()
        model = Gemma4CausalLMModel(config)

        fake_sd = {
            "model.layers.0.experts.gate_up_proj": torch.zeros(4, 64, 64),
            "model.layers.0.experts.down_proj": torch.zeros(4, 64, 32),
        }
        result = model.preprocess_weights(fake_sd)

        assert "model.layers.0.fc1_experts_weights" in result
        assert "model.layers.0.fc2_experts_weights" in result
        assert "model.layers.0.experts.gate_up_proj" not in result

    def test_router_scale_folding(self):
        config = _tiny_gemma4_config()
        model = Gemma4CausalLMModel(config)

        scale_val = torch.tensor([2.0])
        fake_sd = {"model.layers.0.router.scale": scale_val.clone()}
        result = model.preprocess_weights(fake_sd)

        expected = 2.0 * (64**-0.5)  # hidden_size=64
        assert abs(result["model.layers.0.router.scale"].item() - expected) < 1e-6


class TestGemma4ModelPreprocessWeights:
    """Test Gemma4Model.preprocess_weights (multimodal path)."""

    def test_expert_weight_rename(self):
        config = _tiny_gemma4_config()
        model = Gemma4Model(config)

        fake_sd = {
            "model.language_model.layers.0.experts.gate_up_proj": torch.zeros(4, 64, 64),
            "model.language_model.layers.0.experts.down_proj": torch.zeros(4, 64, 32),
        }
        result = model.preprocess_weights(fake_sd)

        assert "decoder.model.layers.0.fc1_experts_weights" in result
        assert "decoder.model.layers.0.fc2_experts_weights" in result

    def test_router_scale_folding(self):
        config = _tiny_gemma4_config()
        model = Gemma4Model(config)

        scale_val = torch.tensor([2.0])
        fake_sd = {"model.language_model.layers.0.router.scale": scale_val.clone()}
        result = model.preprocess_weights(fake_sd)

        expected = 2.0 * (64**-0.5)
        key = "decoder.model.layers.0.router.scale"
        assert key in result
        assert abs(result[key].item() - expected) < 1e-6

    def test_per_expert_scale_not_folded(self):
        """router.per_expert_scale should NOT be multiplied by scale_factor."""
        config = _tiny_gemma4_config()
        model = Gemma4Model(config)

        fake_sd = {
            "model.language_model.layers.0.router.per_expert_scale": torch.ones(4),
        }
        result = model.preprocess_weights(fake_sd)

        key = "decoder.model.layers.0.router.per_expert_scale"
        assert key in result
        assert torch.allclose(result[key], torch.ones(4))
