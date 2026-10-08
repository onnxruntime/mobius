# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""K2 Horizon model (dense and MoVA variants).

Reference: llama.cpp "model : add K2 Horizon dense and MoVA support"
(HF class ``K2HorizonForCausalLM``). K2 Horizon is a GQA decoder-only
transformer with:

- A dense SwiGLU FFN prefix (``first_k_dense_replace`` layers), followed by
  DeepSeek-V3-style routed + shared-expert MoE FFN layers.
- "Group RMS norm" pre-attention/pre-FFN/final norms: the hidden dimension is
  split into ``n_norm_groups`` equal slices, each RMS-normalized
  independently, then scaled by one full-width learned weight
  (:class:`~mobius.components.GroupRMSNorm`; ``n_norm_groups=1`` degenerates
  to a standard RMSNorm).
- Per-head Q/K RMSNorm with a weight distinct per head
  (:class:`~mobius.components.PerHeadRMSNorm`), unlike the single
  head_dim-only weight shared across all heads used by most QK-norm models.
- MoVA ("mixture of value attention"): on MoE layers, ``v_proj`` is replaced
  by a router over ``n_value_expert`` single-linear "value experts" (no
  gate/up/down split): ``V = sum_k weight_k * silu(W_k @ x)``. Routing reuses
  :class:`~mobius.models.deepseek.DeepSeekMoEGate`.
- An optional softplus output gate, computed from the attention block's
  pre-QKV normalized input and applied to the attention output *before*
  ``o_proj``.

No real HF checkpoint exists at the time this was added (see
``K2HorizonConfig`` and the "Caveats" note in this module's PR description);
``preprocess_weights`` follows the DeepSeek-V3/Qwen2-MoE HF naming
conventions as the most probable real-world layout.
"""

from __future__ import annotations

import dataclasses
import math

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components import (
    MLP,
    Embedding,
    GroupRMSNorm,
    Linear,
    PerHeadRMSNorm,
    create_attention_bias,
    initialize_rope,
)
from mobius.components._attention import StaticCacheState, _apply_attention
from mobius.components._rotary_embedding import apply_rotary_pos_emb
from mobius.models.base import CausalLMModel
from mobius.models.deepseek import DeepSeekMoEGate, _DeepSeekMoEFFN


class K2HorizonValueRouter(nn.Module):
    """MoVA router: replaces ``v_proj`` on MoE layers with a mixture of value experts.

    Each value expert is a single-linear "value expert" (K2 Horizon).

    Each value expert is ``Linear(hidden_size, n_embd_v_gqa)`` with no
    gate/up/down split, unlike a regular MoE FFN expert. The routed V is the
    weighted sum of the selected experts' SiLU-activated outputs:
    ``V = sum_k weight_k * silu(W_k @ x)``. Routing (sigmoid scoring,
    optional selection-only bias, optional top-k renormalization/scaling)
    reuses :class:`DeepSeekMoEGate` against a value-expert-sized config.
    """

    def __init__(
        self,
        config: ArchitectureConfig,
        n_embd_v_gqa: int,
        linear_class: type | None = None,
    ):
        super().__init__()
        if linear_class is None:
            linear_class = Linear
        assert config.n_value_expert > 0
        assert config.n_value_expert_used > 0
        value_config = dataclasses.replace(
            config,
            num_local_experts=config.n_value_expert,
            num_experts_per_tok=config.n_value_expert_used,
            n_group=1,
            topk_group=1,
        )
        self.gate = DeepSeekMoEGate(value_config)
        self.experts = nn.ModuleList(
            [
                linear_class(config.hidden_size, n_embd_v_gqa, bias=False)
                for _ in range(config.n_value_expert)
            ]
        )

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        routing_weights, selected_experts = self.gate(op, hidden_states)

        result = None
        for expert_idx, expert in enumerate(self.experts):
            expert_output = op.Swish(expert(op, hidden_states))
            expert_id = op.Constant(value_int=expert_idx)
            match = op.Equal(selected_experts, expert_id)
            match_float = op.CastLike(match, routing_weights)
            weight = op.ReduceSum(op.Mul(routing_weights, match_float), [-1], keepdims=True)
            contribution = op.Mul(expert_output, weight)
            result = contribution if result is None else op.Add(result, contribution)
        return result


class K2HorizonAttention(nn.Module):
    """GQA attention with per-head QK-norm, optional MoVA routing, and output gate.

    Adds an optional softplus output gate (K2 Horizon).

    Differs from the generic :class:`~mobius.components.Attention`:
    - Q/K RMSNorm uses :class:`PerHeadRMSNorm` (distinct weight per head),
      not the single weight shared across heads used by ``attn_qk_norm``.
    - MoE layers route V through :class:`K2HorizonValueRouter` instead of a
      plain ``v_proj`` linear.
    - When ``config.has_attn_output_gate``, the attention output is
      multiplied by ``softplus(x * ln2) / ln2`` (computed from the same
      pre-QKV normalized input as Q/K/V) *before* ``o_proj`` is applied.
    """

    _LN2 = math.log(2.0)

    def __init__(
        self,
        config: ArchitectureConfig,
        is_mova: bool,
        linear_class: type | None = None,
    ):
        super().__init__()
        if linear_class is None:
            linear_class = Linear
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        prf = config.partial_rotary_factor if config.partial_rotary_factor is not None else 1.0
        self.rotary_embedding_dim = 0 if math.isclose(prf, 1.0) else int(self.head_dim * prf)
        self._rope_interleave = config.rope_interleave
        self._has_output_gate = config.has_attn_output_gate

        q_dim = self.num_attention_heads * self.head_dim
        kv_dim = self.num_key_value_heads * self.head_dim
        self.q_proj = linear_class(self.hidden_size, q_dim, bias=config.attn_qkv_bias)
        self.k_proj = linear_class(self.hidden_size, kv_dim, bias=config.attn_qkv_bias)
        self.is_mova = is_mova
        if is_mova:
            self.v_router = K2HorizonValueRouter(config, kv_dim, linear_class=linear_class)
        else:
            self.v_proj = linear_class(self.hidden_size, kv_dim, bias=config.attn_qkv_bias)
        self.o_proj = linear_class(q_dim, self.hidden_size, bias=config.attn_o_bias)

        self.q_norm = PerHeadRMSNorm(
            self.num_attention_heads, self.head_dim, eps=config.rms_norm_eps
        )
        self.k_norm = PerHeadRMSNorm(
            self.num_key_value_heads, self.head_dim, eps=config.rms_norm_eps
        )
        if self._has_output_gate:
            self.attn_gate = linear_class(self.hidden_size, q_dim, bias=False)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple,
        past_key_value: tuple | None = None,
        static_cache: StaticCacheState | None = None,
    ):
        query_states = self.q_proj(op, hidden_states)
        key_states = self.k_proj(op, hidden_states)

        # Per-head RMSNorm with a weight distinct per head, on the 4D per-head view.
        query_states = op.Reshape(query_states, [0, 0, -1, self.head_dim])
        key_states = op.Reshape(key_states, [0, 0, -1, self.head_dim])
        query_states = self.q_norm(op, query_states)
        key_states = self.k_norm(op, key_states)
        query_states = op.Reshape(query_states, [0, 0, -1])
        key_states = op.Reshape(key_states, [0, 0, -1])

        query_states = apply_rotary_pos_emb(
            op,
            x=query_states,
            position_embeddings=position_embeddings,
            num_heads=self.num_attention_heads,
            rotary_embedding_dim=self.rotary_embedding_dim,
            interleaved=self._rope_interleave,
        )
        key_states = apply_rotary_pos_emb(
            op,
            x=key_states,
            position_embeddings=position_embeddings,
            num_heads=self.num_key_value_heads,
            rotary_embedding_dim=self.rotary_embedding_dim,
            interleaved=self._rope_interleave,
        )

        if self.is_mova:
            value_states = self.v_router(op, hidden_states)
        else:
            value_states = self.v_proj(op, hidden_states)

        attn_output, present_key, present_value = _apply_attention(
            op,
            query_states,
            key_states,
            value_states,
            attention_bias,
            past_key_value[0] if past_key_value is not None else None,
            past_key_value[1] if past_key_value is not None else None,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            scale=self.scaling,
            static_cache=static_cache,
        )

        if self._has_output_gate:
            # Smoothed gate computed from the SAME pre-QKV normalized input
            # as Q/K/V (``hidden_states``), applied before o_proj — matches
            # llama.cpp's k2_horizon ggml graph ordering.
            gate_inp = self.attn_gate(op, hidden_states)
            gate = op.Div(op.Softplus(op.Mul(gate_inp, self._LN2)), self._LN2)
            attn_output = op.Mul(attn_output, gate)

        attn_output = self.o_proj(op, attn_output)
        return attn_output, (present_key, present_value)


class K2HorizonDecoderLayer(nn.Module):
    """Decoder layer: group-RMS-normed attention + dense-or-MoE FFN."""

    def __init__(self, config: ArchitectureConfig, is_moe: bool, is_mova: bool):
        super().__init__()
        self.self_attn = K2HorizonAttention(config, is_mova=is_mova)
        if is_moe:
            gate = DeepSeekMoEGate(config)
            self.mlp: nn.Module = _DeepSeekMoEFFN(config, gate)
        else:
            self.mlp = MLP(config)
        self.input_layernorm = GroupRMSNorm(
            config.hidden_size, n_groups=config.n_norm_groups, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = GroupRMSNorm(
            config.hidden_size, n_groups=config.n_norm_groups, eps=config.rms_norm_eps
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None = None,
    ):
        if isinstance(past_key_value, StaticCacheState):
            static_cache = past_key_value
            past_key_value = None
        else:
            static_cache = None

        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        hidden_states, present_kv = self.self_attn(
            op,
            hidden_states=hidden_states,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            static_cache=static_cache,
        )
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, present_kv


class K2HorizonTextModel(nn.Module):
    """K2 Horizon text backbone: embed -> N x decoder layer -> final norm.

    The first ``first_k_dense_replace`` layers use a dense SwiGLU FFN; the
    rest use routed + shared-expert MoE. MoVA value routing (replacing
    ``v_proj``) is active on every MoE layer whenever ``n_value_expert`` > 0.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self._dtype = config.dtype
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)

        first_k = config.first_k_dense_replace
        if not config.num_local_experts:
            first_k = config.num_hidden_layers
        use_mova = config.n_value_expert > 0 and config.n_value_expert_used > 0
        self.layers = nn.ModuleList(
            [
                K2HorizonDecoderLayer(
                    config,
                    is_moe=(i >= first_k),
                    is_mova=use_mova and i >= first_k,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = GroupRMSNorm(
            config.hidden_size, n_groups=config.n_norm_groups, eps=config.rms_norm_eps
        )
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value | None,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states = self.embed_tokens(op, input_ids)
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


class K2HorizonCausalLMModel(CausalLMModel):
    """K2 Horizon Causal LM: dense/MoE GQA decoder with MoVA value routing.

    model_type: k2_horizon
    """

    default_task: str = "text-generation"
    category: str = "Mixture of Experts"

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        self._replace_text_model(K2HorizonTextModel(config))

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Remap HuggingFace weight names to ONNX parameter names.

        Key mappings (standard DeepSeek-V3/Qwen2-MoE-style HF naming, the
        most probable layout absent a real checkpoint to calibrate against):
        - FFN MoE gate: ``mlp.gate.weight`` -> ``mlp.moe.gate.weight``;
          ``mlp.gate.bias`` (selection-only correction bias) ->
          ``mlp.moe.gate.e_score_correction_bias``.
        - FFN MoE experts: ``mlp.experts.{i}.*`` -> ``mlp.moe.experts.{i}.*``.
          Shared expert (``mlp.shared_experts.*``) names already align.
        - MoVA value routing: ``self_attn.v_gate.weight`` ->
          ``self_attn.v_router.gate.weight``; ``self_attn.v_gate.bias`` ->
          ``self_attn.v_router.gate.e_score_correction_bias``;
          ``self_attn.v_experts.{i}.weight`` ->
          ``self_attn.v_router.experts.{i}.weight``.
        - Attention/FFN norms (``attn_norm``/``ffn_norm`` in llama.cpp) and
          Q/K norms already align with standard HF
          ``input_layernorm``/``post_attention_layernorm``/``q_norm``/
          ``k_norm`` names.
        """
        renamed = {}
        for key, value in state_dict.items():
            new_key = key
            new_key = new_key.replace(".mlp.gate.weight", ".mlp.moe.gate.weight")
            new_key = new_key.replace(
                ".mlp.gate.bias", ".mlp.moe.gate.e_score_correction_bias"
            )
            if ".mlp.experts." in new_key:
                expert_suffix = new_key.split(".mlp.experts.", 1)[1]
                if expert_suffix.split(".", 1)[0].isdigit():
                    new_key = new_key.replace(".mlp.experts.", ".mlp.moe.experts.")

            new_key = new_key.replace(
                ".self_attn.v_gate.weight", ".self_attn.v_router.gate.weight"
            )
            new_key = new_key.replace(
                ".self_attn.v_gate.bias",
                ".self_attn.v_router.gate.e_score_correction_bias",
            )
            if ".self_attn.v_experts." in new_key:
                expert_suffix = new_key.split(".self_attn.v_experts.", 1)[1]
                if expert_suffix.split(".", 1)[0].isdigit():
                    new_key = new_key.replace(
                        ".self_attn.v_experts.", ".self_attn.v_router.experts."
                    )

            renamed[new_key] = value

        return super().preprocess_weights(renamed)
