# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import torch

from mobius._weight_utils import split_fused_qkv, split_gate_up_proj
from mobius.models.base import CausalLMModel


class ChatGLMCausalLMModel(CausalLMModel):
    """ChatGLM model with partial rotary and fused projection imports.

    GLM-4's original ChatGLM checkpoint layout nests decoder weights below
    ``transformer.encoder`` and stores QKV and gate/up projections fused.
    Canonicalize those names before generic GPTQ preprocessing, then split the
    converted QKV and gate/up tensors along their output dimension. Splitting
    the repacked gate/up tensors is lossless and avoids a runtime activation
    Split between captured CUDA graph segments.
    """

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            key = key.replace("transformer.embedding.word_embeddings.", "model.embed_tokens.")
            key = key.replace("transformer.encoder.final_layernorm.", "model.norm.")
            key = key.replace("transformer.output_layer.", "lm_head.")
            key = key.replace("transformer.encoder.layers.", "model.layers.")
            key = key.replace(".self_attention.", ".self_attn.")
            key = key.replace(".self_attn.dense.", ".self_attn.o_proj.")
            key = key.replace(".mlp.dense_h_to_4h.", ".mlp.gate_up_proj.")
            key = key.replace(".mlp.dense_4h_to_h.", ".mlp.down_proj.")
            key = key.replace(".self_attn.query_key_value.", ".self_attn.qkv_proj.")
            renamed[key] = value

        state_dict = super().preprocess_weights(renamed)

        quantization = getattr(self.config, "quantization", None)
        if quantization is not None and quantization.sym:
            state_dict = {
                key: value
                for key, value in state_dict.items()
                if not key.endswith(".zero_points")
            }

        for key in list(state_dict):
            if ".self_attn.qkv_proj." not in key:
                continue
            q, k, v = split_fused_qkv(
                state_dict.pop(key),
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim,
            )
            state_dict[key.replace("qkv_proj", "q_proj")] = q
            state_dict[key.replace("qkv_proj", "k_proj")] = k
            state_dict[key.replace("qkv_proj", "v_proj")] = v

        for key in list(state_dict):
            if ".mlp.gate_up_proj." not in key:
                continue
            gate, up = split_gate_up_proj(
                state_dict.pop(key),
                self.config.intermediate_size,
            )
            state_dict[key.replace("gate_up_proj", "gate_proj")] = gate
            state_dict[key.replace("gate_up_proj", "up_proj")] = up

        return state_dict
