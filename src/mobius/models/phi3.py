# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Phi-3 / Phi-3.5 causal language model.

Replicates HuggingFace's ``Phi3ForCausalLM``. Phi-3/3.5 fuse QKV and
gate-up projections into single tensors. The QKV fusion is resolved by
splitting weights in ``preprocess_weights``; the gate-up fusion is handled
natively by inheriting from ``FusedGateUpCausalLMModel``, which keeps
``gate_up_proj`` fused and splits activations in the MLP forward pass.
This approach works for both fp16/bf16 and GPTQ int32-packed checkpoints.
"""

from __future__ import annotations

import torch

from mobius._weight_utils import split_fused_qkv
from mobius.models.base import FusedGateUpCausalLMModel


class Phi3CausalLMModel(FusedGateUpCausalLMModel):
    """Phi-3 / Phi-3.5 model with SuRoPE and fused QKV weight splitting.

    Replicates HuggingFace's ``Phi3ForCausalLM``.

    ``gate_up_proj`` is kept fused (see :class:`~mobius.models.base.FusedGateUpCausalLMModel`).
    Only the QKV fusion is resolved via ``preprocess_weights``.
    """

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        state_dict = super().preprocess_weights(state_dict)
        for key in list(state_dict.keys()):
            if "qkv_proj" in key:
                q, k, v = split_fused_qkv(
                    state_dict.pop(key),
                    self.config.num_attention_heads,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                )
                state_dict[key.replace("qkv_proj", "q_proj")] = q
                state_dict[key.replace("qkv_proj", "k_proj")] = k
                state_dict[key.replace("qkv_proj", "v_proj")] = v
        return state_dict
