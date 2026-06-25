# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""EAGLE-3 speculative-decoding draft model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import OpBuilder, nn

from mobius._configs import Eagle3Config
from mobius.components import MLP, Linear, RMSNorm, create_attention_bias, initialize_rope
from mobius.components._attention import Attention

if TYPE_CHECKING:
    import onnx_ir as ir


class Eagle3Attention(Attention):
    """Llama attention with Q/K/V projected from concatenated 2H input."""

    def __init__(self, config: Eagle3Config):
        super().__init__(config)
        self.q_proj = Linear(
            2 * config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=config.attn_qkv_bias,
        )
        self.k_proj = Linear(
            2 * config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attn_qkv_bias,
        )
        self.v_proj = Linear(
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
        if config.draft_vocab_size is None:
            raise ValueError("Eagle3Config.draft_vocab_size must be set")
        # Guard vLLM EAGLE-3 options we do not model. They are false/absent in the
        # AngelSlim and speculators (RedHat) checkpoints; fail loudly rather than
        # silently produce wrong outputs if a future checkpoint enables them.
        if getattr(config, "norm_before_fc", False):
            raise NotImplementedError("Eagle3: norm_before_fc is not supported")
        if getattr(config, "fc_norm", False):
            raise NotImplementedError("Eagle3: fc_norm is not supported")
        target_hidden = getattr(config, "target_hidden_size", None)
        if target_hidden is not None and target_hidden != config.hidden_size:
            raise NotImplementedError(
                f"Eagle3: target_hidden_size ({target_hidden}) != hidden_size "
                f"({config.hidden_size}) is not supported"
            )

        self.fc = Linear(3 * config.hidden_size, config.hidden_size, bias=False)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Eagle3Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MLP(config)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = Linear(config.hidden_size, config.draft_vocab_size, bias=False)
        self.rotary_emb = initialize_rope(config)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Strip the layer prefix and drop borrowed / orchestrator-side weights.

        Handles both checkpoint layouts: AngelSlim names the single layer
        ``midlayer.*``; speculators (RedHat) names it ``layers.0.*`` and also
        ships an ``embed_tokens`` copy of the target's embedding (we borrow the
        target's, so drop it). ``d2t`` / ``t2d`` are orchestrator-side remap
        buffers and are dropped from the graph.
        """
        drop = {"d2t", "t2d", "embed_tokens.weight"}
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
    ):
        combined = op.Add(self.fc(op, fused_hidden), recycled_hidden)
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
        draft_logits = self.lm_head(op, h_post)
        return draft_logits, h_prenorm, [present_kv]
