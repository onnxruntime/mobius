# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""NanoChat causal language model.

Standard Llama-family decoder with a non-gated FCMLP (``fc1 → relu² → fc2``)
and parameter-free RMSNorm (no learnable weight).  QK-norm is also enabled.

HuggingFace uses ``fc1`` / ``fc2`` for the MLP projections; our FCMLP
uses ``up_proj`` / ``down_proj``, so ``preprocess_weights`` renames them.
All norm weights are initialised to ones because HF NanoChatRMSNorm has
no learnable parameters.

Replicates HuggingFace ``NanoChatForCausalLM``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

import onnx_ir as ir
from mobius._configs import ArchitectureConfig
from mobius.components import FCMLP
from mobius.models.base import CausalLMModel


class NanoChatCausalLMModel(CausalLMModel):
    """NanoChat model: FCMLP with relu2 + parameter-free RMSNorm + QK-norm."""

    def __init__(self, config: ArchitectureConfig):
        # Enable QK-norm
        config = dataclasses.replace(config, attn_qk_norm=True)
        super().__init__(config)
        # Replace gated MLP → non-gated FCMLP in every decoder layer
        for layer in self.model.layers:
            layer.mlp = FCMLP(
                config.hidden_size,
                config.intermediate_size,
                activation=config.hidden_act,
                bias=config.mlp_bias,
            )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Rename HF fc1/fc2 → FCMLP up_proj/down_proj.

        Also injects ones for all norm weights since HF NanoChatRMSNorm
        is parameter-free while our ONNX RMSNorm expects a weight tensor.
        """
        new_state_dict = {}
        for name, tensor in state_dict.items():
            # fc1 → up_proj, fc2 → down_proj
            name = name.replace(".mlp.fc1.", ".mlp.up_proj.")
            name = name.replace(".mlp.fc2.", ".mlp.down_proj.")
            new_state_dict[name] = tensor

        # Inject ones for all norm weights (parameter-free in HF)
        hidden_size = next(iter(state_dict.values())).shape[-1]
        head_dim = self.config.head_dim
        ones_hidden = torch.ones(hidden_size)
        ones_head = torch.ones(head_dim)
        for i in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{i}"
            new_state_dict.setdefault(
                f"{prefix}.input_layernorm.weight", ones_hidden
            )
            new_state_dict.setdefault(
                f"{prefix}.post_attention_layernorm.weight", ones_hidden
            )
            new_state_dict.setdefault(
                f"{prefix}.self_attn.q_norm.weight", ones_head
            )
            new_state_dict.setdefault(
                f"{prefix}.self_attn.k_norm.weight", ones_head
            )
        new_state_dict.setdefault("model.norm.weight", ones_hidden)
        return super().preprocess_weights(new_state_dict)
