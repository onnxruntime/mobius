# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Conventional legacy decoders used by canonical GGUF architectures."""

from __future__ import annotations

import torch

from mobius._configs import (
    ArchitectureConfig,
    CodeShellConfig,
    Jais2Config,
    XverseConfig,
)
from mobius._weight_utils import split_fused_qkv
from mobius.components import FCMLP
from mobius.models.base import (
    CausalLMModel,
    LayerNormCausalLMModel,
    linear_class_for_config,
)


class LegacyLayerNormCausalLMModel(LayerNormCausalLMModel):
    """LayerNorm decoder with a non-gated FFN and optional fused GGUF QKV input."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        linear_class = linear_class_for_config(config)
        for layer in self.model.layers:
            layer.mlp = FCMLP(
                config.hidden_size,
                config.intermediate_size,
                activation=config.hidden_act,
                bias=config.mlp_bias,
                linear_class=linear_class,
            )

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


class Jais2CausalLMModel(LegacyLayerNormCausalLMModel):
    """Jais2 decoder with its published bias and normalization configuration."""

    config_class = Jais2Config


class CodeShellCausalLMModel(LegacyLayerNormCausalLMModel):
    """CodeShell decoder accepting both published HF and canonical GGUF names."""

    config_class = CodeShellConfig

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        normalized: dict[str, torch.Tensor] = {}
        for original_name, tensor in state_dict.items():
            if original_name.endswith(".attn.rotary_emb.inv_freq"):
                continue

            name = original_name
            name = name.replace("transformer.wte.", "model.embed_tokens.")
            name = name.replace("transformer.ln_f.", "model.norm.")
            name = name.replace("transformer.h.", "model.layers.")
            name = name.replace(".ln_1.", ".input_layernorm.")
            name = name.replace(".ln_2.", ".post_attention_layernorm.")
            name = name.replace(".attn.c_attn.", ".self_attn.qkv_proj.")
            name = name.replace(".attn.c_proj.", ".self_attn.o_proj.")
            name = name.replace(".mlp.c_fc.", ".mlp.up_proj.")
            name = name.replace(".mlp.c_proj.", ".mlp.down_proj.")
            if name in normalized:
                raise ValueError(
                    f"CodeShell weight normalization collision for {original_name!r}: {name!r}"
                )
            normalized[name] = tensor
        normalized = super().preprocess_weights(normalized)
        if self.config.tie_word_embeddings:
            # Tied CodeShell graphs have one initializer owned by embed_tokens.
            normalized.pop("lm_head.weight", None)
        return normalized


class XverseCausalLMModel(CausalLMModel):
    """Xverse decoder with source-compatible RoPE configuration defaults."""

    config_class = XverseConfig
