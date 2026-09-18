# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components import (
    DecoderLayer,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
)
from mobius.models.base import CausalLMModel, embedding_for_config, linear_class_for_config

if TYPE_CHECKING:
    import onnx_ir as ir


class SmolLM3TextModel(nn.Module):
    """SmolLM3 text model with per-layer RoPE control and sliding window attention.

    SmolLM3 features:
    - Per-layer conditional RoPE (some layers have no RoPE)
    - Per-layer attention type (sliding_attention vs full_attention)
    - Dynamic window sizing per layer
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype
        linear_class = linear_class_for_config(config)
        self.embed_tokens = embedding_for_config(config)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(config, linear_class=linear_class)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)
        self.layer_types = config.layer_types
        self.no_rope_layers = config.no_rope_layers
        self.sliding_window = config.sliding_window

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states = self.embed_tokens(op, input_ids)
        position_embeddings = self.rotary_emb(op, position_ids)

        full_attn_bias = create_attention_bias(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )
        sliding_attn_bias = create_attention_bias(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            sliding_window=self.sliding_window,
            dtype=self._dtype,
        )

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for i, (layer, past_kv) in enumerate(zip(self.layers, past_kvs)):
            layer_type = self.layer_types[i] if self.layer_types else "full_attention"
            attn_bias = (
                sliding_attn_bias if layer_type == "sliding_attention" else full_attn_bias
            )
            # SmolLM3 uses no_rope_layers to gate RoPE per layer.
            # Despite the name, the HF convention is:
            #   no_rope_layers[i] == 1 → USE RoPE
            #   no_rope_layers[i] == 0 → skip RoPE
            # (HF assigns self.use_rope = config.no_rope_layers[layer_idx])
            use_rope = (
                self.no_rope_layers is None
                or i >= len(self.no_rope_layers)
                or self.no_rope_layers[i] == 1
            )
            rope = position_embeddings if use_rope else None

            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attn_bias,
                position_embeddings=rope,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values


class SmolLM3CausalLMModel(CausalLMModel):
    """SmolLM3 causal language model with per-layer attention control."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        self._replace_text_model(SmolLM3TextModel(config))
