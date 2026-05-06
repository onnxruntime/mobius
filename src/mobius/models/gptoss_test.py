# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for GPT-OSS preprocess_weights MXFP4 dequantization."""

from __future__ import annotations

from unittest.mock import patch

import torch

from mobius._testing import make_config
from mobius.models.gptoss import GPTOSSCausalLMModel


class TestGPTOSSPreprocessWeightsMXFP4:
    """Test MXFP4 _blocks/_scales dequantization in preprocess_weights."""

    @staticmethod
    def _make_gptoss_config():
        return make_config(
            num_local_experts=2,
            num_experts_per_tok=1,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=256,
            partial_rotary_factor=1.0,
            rope_interleave=False,
            attn_qkv_bias=True,
            attn_o_bias=True,
        )

    def test_mxfp4_blocks_scales_dequantized(self):
        """Blocks/scales are dequantized then split into per-expert keys."""
        config = self._make_gptoss_config()
        model = GPTOSSCausalLMModel(config)

        hidden = config.hidden_size  # 64
        inter = config.intermediate_size  # 128
        n_exp = config.num_local_experts  # 2

        # The dequantized tensor Phase 2 expects: [N, hidden, 2*inter]
        fake_dequantized = torch.randn(n_exp, hidden, 2 * inter)

        # Small dummy uint8 tensors for _blocks and _scales
        blocks = torch.randint(0, 255, (4, 4), dtype=torch.uint8)
        scales = torch.randint(0, 255, (4, 4), dtype=torch.uint8)

        state_dict = {
            "model.layers.0.mlp.experts.gate_up_proj_blocks": blocks,
            "model.layers.0.mlp.experts.gate_up_proj_scales": scales,
            # A normal key that should pass through
            "model.layers.0.self_attn.q_proj.weight": torch.randn(
                config.num_attention_heads * config.head_dim,
                hidden,
            ),
        }

        mock_path = "transformers.integrations.mxfp4._convert_moe_packed_tensors"
        with patch(mock_path, return_value=fake_dequantized) as mock_fn:
            result = model.preprocess_weights(state_dict)

        # The mock was called with the blocks/scales tensors
        mock_fn.assert_called_once()

        # _blocks and _scales keys must be gone
        assert not any(k.endswith("_blocks") for k in result)
        assert not any(k.endswith("_scales") for k in result)

        # Phase 2 should have split gate_up_proj into per-expert keys
        for i in range(n_exp):
            gate_key = f"model.layers.0.mlp.experts.{i}.gate_proj.weight"
            up_key = f"model.layers.0.mlp.experts.{i}.up_proj.weight"
            assert gate_key in result, f"Missing {gate_key}"
            assert up_key in result, f"Missing {up_key}"
            # Shape: [inter, hidden] after de-interleave + transpose
            assert result[gate_key].shape == (inter, hidden)
            assert result[up_key].shape == (inter, hidden)

    def test_no_blocks_passes_through(self):
        """Weights pass through unchanged when no _blocks/_scales exist."""
        config = self._make_gptoss_config()
        model = GPTOSSCausalLMModel(config)

        state_dict = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(
                config.num_attention_heads * config.head_dim,
                config.hidden_size,
            ),
        }

        result = model.preprocess_weights(state_dict)
        assert "model.layers.0.self_attn.q_proj.weight" in result
