# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""EAGLE-3 speculative-decoding draft model.

The drafter emits logits over a compressed *draft* vocabulary. HuggingFace
checkpoints store ``d2t`` offsets; the llama.cpp GGUF converter resolves those
to absolute target token IDs. Mobius records the absolute map in
``draft_manifest.json`` and the direct ONNX Runtime coordinator applies it
outside the neural graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import OpBuilder, nn

from mobius._configs import Eagle3Config
from mobius.components import (
    MLP,
    Embedding,
    Linear,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
)
from mobius.components._attention import Attention
from mobius.models.base import linear_class_for_config

if TYPE_CHECKING:
    import onnx_ir as ir


class Eagle3Attention(Attention):
    """Llama attention with Q/K/V projected from concatenated 2H input."""

    def __init__(self, config: Eagle3Config, linear_class: type | None = None):
        if linear_class is None:
            linear_class = Linear
        super().__init__(config, linear_class=linear_class)
        self.q_proj = linear_class(
            2 * config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=config.attn_qkv_bias,
        )
        self.k_proj = linear_class(
            2 * config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attn_qkv_bias,
        )
        self.v_proj = linear_class(
            2 * config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attn_qkv_bias,
        )


class Eagle3DraftModel(nn.Module):
    """Single-layer EAGLE-3 chain drafter."""

    config_class: type = Eagle3Config
    default_task: str = "eagle3-draft"
    category: str = "Text Generation"

    def __init__(self, config: Eagle3Config):
        super().__init__()
        self.config = config
        self._dtype = config.dtype
        self._norm_before_residual = bool(getattr(config, "norm_before_residual", False))
        self._norm_before_fc = bool(getattr(config, "norm_before_fc", False))
        self._fc_norm = bool(getattr(config, "fc_norm", False))
        self.embed_tokens = (
            Embedding(config.vocab_size, config.hidden_size)
            if config.use_draft_token_embedding
            else None
        )
        draft_vocab_size = config.draft_vocab_size
        if draft_vocab_size is None and not config.use_target_lm_head:
            raise ValueError("Eagle3Config.draft_vocab_size must be set")
        target_hidden = config.target_hidden_size or config.hidden_size
        linear_class = linear_class_for_config(config) or Linear
        # Number of target aux hidden states fused by fc (3 in every EAGLE-3
        # checkpoint; the fc input is target_hidden_size * num_aux = 3 * hidden).
        self._num_aux = 3

        self.fc = linear_class(3 * target_hidden, config.hidden_size, bias=False)
        # Optional pre-fc norms (vLLM EAGLE-3): a single RMSNorm over the full
        # fused 3H input (norm_before_fc) and/or a per-aux RMSNorm on each H chunk
        # (fc_norm). Both are absent in the AngelSlim / speculators checkpoints.
        if self._norm_before_fc:
            self.input_norm = RMSNorm(3 * target_hidden, eps=config.rms_norm_eps)
        if self._fc_norm:
            self.fc_norm = nn.ModuleList(
                [
                    RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                    for _ in range(self._num_aux)
                ]
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Eagle3Attention(config, linear_class=linear_class)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MLP(config, linear_class=linear_class)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = (
            None
            if config.use_target_lm_head
            else linear_class(config.hidden_size, draft_vocab_size, bias=False)
        )
        self.rotary_emb = initialize_rope(config)

    def _project_fused(self, op: OpBuilder, fused_hidden: ir.Value) -> ir.Value:
        """Apply the optional pre-fc norms then fc to the fused 3H aux input.

        norm(zeros) == zeros and fc has no bias, so on chain steps (fused = 0)
        this stays zero and the recycled hidden passes through unchanged.
        """
        x = fused_hidden
        if self._norm_before_fc:
            x = self.input_norm(op, x)
        if self._fc_norm:
            chunks = op.Split(x, num_outputs=self._num_aux, axis=-1, _outputs=self._num_aux)
            x = op.Concat(
                *(self.fc_norm[i](op, chunks[i]) for i in range(self._num_aux)), axis=-1
            )
        return self.fc(op, x)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Strip the layer prefix and drop borrowed / orchestrator-side weights.

        Handles both checkpoint layouts: AngelSlim names the single layer
        ``midlayer.*``; speculators (RedHat) names it ``layers.0.*`` and may
        ship a draft-owned ``embed_tokens`` table. ``d2t`` / ``t2d`` are
        orchestrator-side remap buffers and are dropped from the graph.
        """
        drop = {"d2t", "t2d"}
        if self.embed_tokens is None:
            drop.add("embed_tokens.weight")
        out = {}
        for key, value in state_dict.items():
            if key in drop:
                continue
            if key.startswith("midlayer."):
                key = key[len("midlayer.") :]
            elif key.startswith("layers.0."):
                key = key[len("layers.0.") :]
            out[key] = value
        return out

    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: ir.Value,
        fused_hidden: ir.Value,
        recycled_hidden: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        input_ids: ir.Value | None = None,
    ):
        embedding = self.embed_tokens
        if embedding is not None:
            if input_ids is None:
                raise ValueError("draft-owned EAGLE3 embeddings require input_ids")
            inputs_embeds = embedding(op, input_ids)
        elif inputs_embeds is None:
            raise ValueError("target-shared EAGLE3 embeddings require inputs_embeds")
        combined = op.Add(self._project_fused(op, fused_hidden), recycled_hidden)
        embeds = self.input_layernorm(op, inputs_embeds)
        hidden = self.hidden_norm(op, combined)
        # norm_before_residual: residual is the normalized hidden; otherwise it is
        # the raw fused feature (combined). Matches vllm llama_eagle3.py.
        residual_base = hidden if self._norm_before_residual else combined
        attn_input = op.Concat(embeds, hidden, axis=-1)

        position_embeddings = self.rotary_emb(op, position_ids)
        attention_bias = create_attention_bias(
            op,
            input_ids=inputs_embeds,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )

        past_kv = past_key_values[0] if past_key_values else None
        attn_output, present_kv = self.self_attn(
            op,
            hidden_states=attn_input,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_kv,
        )
        residual = op.Add(attn_output, residual_base)
        hidden = self.post_attention_layernorm(op, residual)
        hidden = self.mlp(op, hidden)
        h_prenorm = op.Add(hidden, residual)
        h_post = self.norm(op, h_prenorm)
        draft_output = h_post if self.lm_head is None else self.lm_head(op, h_post)
        return draft_output, h_prenorm, [present_kv]
