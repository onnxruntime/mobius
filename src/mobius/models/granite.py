# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components import (
    Embedding,
    RMSNorm,
    create_attention_bias,
    create_decoder_layer,
    initialize_rope,
)
from mobius.models.base import CausalLMModel, Linear

if TYPE_CHECKING:
    import torch

    import onnx_ir as ir


class GraniteTextModel(nn.Module):
    """Granite text model with embedding multiplier."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [create_decoder_layer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)
        self.embedding_multiplier = config.embedding_multiplier

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        inputs_embeds: ir.Value | None = None,
    ):
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(op, input_ids)
        # Apply embedding multiplier
        hidden_states = op.Mul(hidden_states, self.embedding_multiplier)

        position_embeddings = self.rotary_emb(op, position_ids)
        attention_bias = create_attention_bias(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, past_kv in zip(self.layers, past_kvs):
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values


class GraniteCausalLMModel(CausalLMModel):
    """Granite model with embedding/attention/logits/residual scaling multipliers.

    Granite uses scaling factors on embeddings, attention, logits, and residual
    connections. These are configured via the HF config:
    - embedding_multiplier: scales embeddings after lookup
    - attention_multiplier: replaces 1/sqrt(head_dim) as attention scale
    - logits_scaling: divides final logits
    - residual_multiplier: scales attention/MLP outputs before residual add
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        self.model = GraniteTextModel(config)
        self.logits_scaling = config.logits_scaling

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        logits, present_key_values = super().forward(
            op, input_ids, attention_mask, position_ids, past_key_values
        )
        # Apply logits scaling
        logits = op.Div(logits, self.logits_scaling)
        return logits, present_key_values


class GraniteMoETextModel(nn.Module):
    """MoE text model with Granite embedding multiplier.

    Identical to ``MoETextModel`` but applies ``embedding_multiplier``
    after the embedding lookup and uses TopK-then-softmax gating.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        from mobius.components._moe import TopKGate
        from mobius.models.moe import MoEDecoderLayer

        self._dtype = config.dtype
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.embedding_multiplier = config.embedding_multiplier

        self.layers = nn.ModuleList(
            [
                MoEDecoderLayer(
                    config,
                    gate=TopKGate(
                        config.hidden_size,
                        config.num_local_experts,
                        config.num_experts_per_tok,
                    ),
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value | None,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states = self.embed_tokens(op, input_ids)
        # Apply Granite embedding multiplier
        hidden_states = op.Mul(hidden_states, self.embedding_multiplier)

        position_embeddings = self.rotary_emb(op, position_ids)
        if attention_mask is not None:
            attention_bias = create_attention_bias(
                op,
                input_ids=input_ids,
                attention_mask=attention_mask,
                dtype=self._dtype,
            )
        else:
            attention_bias = None

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, past_kv in zip(self.layers, past_kvs):
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values


class GraniteMoECausalLMModel(CausalLMModel):
    """GraniteMoE model: MoE with Granite scaling multipliers.

    Combines MoE routing (TopK-then-softmax gating, per HuggingFace
    ``GraniteMoeTopKGating``) with Granite-specific scaling factors:

    - ``embedding_multiplier``: scales embeddings after lookup
    - ``attention_multiplier``: replaces 1/sqrt(head_dim) attention scale
    - ``logits_scaling``: divides final logits
    - ``residual_multiplier``: scales attention/MLP outputs before residual add

    HuggingFace class: ``GraniteMoeForCausalLM``.
    """

    category: str = "Mixture of Experts"

    def __init__(self, config: ArchitectureConfig):
        nn.Module.__init__(self)
        self.config = config
        self.model = GraniteMoETextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.logits_scaling = config.logits_scaling

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value | None,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        logits, present_key_values = super().forward(
            op, input_ids, attention_mask, position_ids, past_key_values
        )
        # Apply Granite logits scaling
        logits = op.Div(logits, self.logits_scaling)
        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        from mobius.models.moe import _rename_moe_expert_weights

        state_dict = _rename_moe_expert_weights(state_dict)
        return super().preprocess_weights(state_dict)
