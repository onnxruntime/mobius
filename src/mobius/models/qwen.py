# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import torch

from mobius._weight_utils import split_fused_qkv
from mobius.models.base import CausalLMModel


class QwenCausalLMModel(CausalLMModel):
    """Qwen v1 model, including its canonical fused GGUF QKV input."""

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        state_dict = dict(state_dict)
        for name in tuple(state_dict):
            if ".self_attn.qkv_proj." not in name:
                continue
            q, k, v = split_fused_qkv(
                state_dict.pop(name),
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim,
            )
            state_dict[name.replace("qkv_proj", "q_proj")] = q
            state_dict[name.replace("qkv_proj", "k_proj")] = k
            state_dict[name.replace("qkv_proj", "v_proj")] = v
        return super().preprocess_weights(state_dict)


class Qwen3CausalLMModel(CausalLMModel):
    """Qwen3 model with Q/K normalization.

    Uses RMSNorm on query and key projections before attention,
    configured via attn_qk_norm=True in ArchitectureConfig.
    """
