# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import torch

from mobius._weight_utils import rename_weight_keys
from mobius.models.base import CausalLMModel


class ChatGLMCausalLMModel(CausalLMModel):
    """ChatGLM model with partial rotary (0.5 factor) and MLP name remapping.

    ChatGLM uses interleaved RoPE and has a different MLP attribute naming
    convention (dense_4h_to_h instead of down_proj).
    """

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        state_dict = super().preprocess_weights(state_dict)
        # ChatGLM uses HF-specific attribute names:
        #   dense_h_to_4h / dense_4h_to_h → up_proj / down_proj (FCMLP naming)
        #   self_attention → self_attn
        return rename_weight_keys(
            state_dict,
            [
                ("dense_4h_to_h", "down_proj"),
                ("dense_h_to_4h", "up_proj"),
                ("self_attention", "self_attn"),
            ],
        )
