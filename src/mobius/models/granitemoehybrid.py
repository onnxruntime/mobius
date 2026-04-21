# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""GraniteMoeHybrid: Mamba2/SSD + Attention hybrid with MoE FFN on all layers.

Every layer has both a routed MoE block (``block_sparse_moe``) and a dense
shared MLP (``shared_mlp``). The layer type ("mamba2" or "full_attention")
controls whether the attention sub-block is a Mamba2/SSD or standard GQA.
Attention layers use NoPE (no rotary position embeddings).

Forward pass per layer::

    residual = x
    x = input_layernorm(x)
    x = mamba(x)  OR  self_attn(x, position_embeddings=None)
    x = residual + x * residual_multiplier
    residual = x
    x = post_attention_layernorm(x)
    x = block_sparse_moe(x) + shared_mlp(x)
    x = residual + x * residual_multiplier

HuggingFace reference: ``GraniteMoeHybridForCausalLM``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import GraniteMoeHybridConfig
from mobius._weight_utils import tie_word_embeddings
from mobius.components import (
    Attention,
    Embedding,
    Linear,
    Mamba2Block,
    RMSNorm,
    TopKGate,
    create_attention_bias,
    get_activation,
)

if TYPE_CHECKING:
    import onnx_ir as ir

# ---------------------------------------------------------------------------
# Fused MoE and shared-MLP blocks
# ---------------------------------------------------------------------------


class _Linear3D(nn.Module):
    """3D stacked-expert linear layer.

    Stores a single weight tensor ``[n_experts, out_features, in_features]``.
    ``forward`` selects one expert slice by index and applies ``x @ W[e].T``,
    equivalent to a per-expert :class:`Linear` without bias.
    """

    def __init__(self, n_experts: int, out_features: int, in_features: int):
        super().__init__()
        self.weight = nn.Parameter([n_experts, out_features, in_features])

    def forward(self, op: builder.OpBuilder, x: ir.Value, expert_index: int) -> ir.Value:
        """Select expert *expert_index* and compute ``x @ W[expert_index].T``."""
        # W[e]: [out_features, in_features]
        w_e = op.Squeeze(op.Gather(self.weight, [expert_index], axis=0), [0])
        return op.MatMul(x, op.Transpose(w_e))


class _FusedMoEBlock(nn.Module):
    """MoE block with fused 3D expert weights matching HuggingFace naming.

    HF stores expert weights as fused 3D tensors:

    * ``input_linear.weight``  — shape ``[n_experts, 2*intermediate, hidden]``
      (gate + up projections concatenated along dim-1)
    * ``output_linear.weight`` — shape ``[n_experts, hidden, intermediate]``
      (down projection per expert)

    Dispatch: loop over static expert indices, Gather from 3D tensors,
    apply gated SiLU MLP, accumulate routing-weighted results.

    This avoids splitting fused weights in ``preprocess_weights``.
    """

    def __init__(self, config: GraniteMoeHybridConfig):
        super().__init__()
        assert config.num_local_experts is not None
        assert config.num_experts_per_tok is not None
        self._num_experts = config.num_local_experts
        self._intermediate_size = config.intermediate_size
        self._top_k = config.num_experts_per_tok
        self._act_fn = get_activation(config.hidden_act)

        # Routing gate: HF name is router.layer, renamed to gate in preprocess_weights
        self.gate = TopKGate(
            config.hidden_size, config.num_local_experts, config.num_experts_per_tok
        )

        # Fused 3D expert weights — names match HF directly
        # input_linear: [n_experts, 2*intermediate, hidden] (gate+up fused)
        self.input_linear = _Linear3D(
            config.num_local_experts,
            2 * config.intermediate_size,
            config.hidden_size,
        )
        # output_linear: [n_experts, hidden, intermediate] (down projection)
        self.output_linear = _Linear3D(
            config.num_local_experts,
            config.hidden_size,
            config.intermediate_size,
        )

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        """Route tokens to experts and accumulate weighted outputs."""
        # routing_weights: [batch*seq, top_k], selected_experts: [batch*seq, top_k]
        routing_weights, selected_experts = self.gate(op, hidden_states)

        result = None
        for e_idx in range(self._num_experts):
            # Gated MLP per expert:
            # input_linear selects expert e → x @ W[e].T → [T, 2*inter]
            proj = self.input_linear(op, hidden_states, e_idx)
            gate, up = op.Split(proj, axis=-1, num_outputs=2, _outputs=2)
            activated = op.Mul(self._act_fn(op, gate), up)  # [T, inter]
            # output_linear selects expert e → activated @ W[e].T → [T, hidden]
            expert_output = self.output_linear(op, activated, e_idx)

            # Mask and weight: accumulate only for tokens routed to this expert
            expert_id = op.Constant(value_int=e_idx)
            match = op.Equal(selected_experts, expert_id)
            match_float = op.CastLike(match, routing_weights)
            weighted = op.Mul(routing_weights, match_float)
            weight = op.ReduceSum(weighted, [-1], keepdims=True)
            contribution = op.Mul(expert_output, weight)

            if result is None:
                result = contribution
            else:
                result = op.Add(result, contribution)

        return result


class _FusedSharedMLP(nn.Module):
    """Shared MLP with fused gate+up matching HuggingFace naming.

    HF stores shared-MLP weights as:

    * ``input_linear.weight``  — shape ``[2*shared_intermediate, hidden]``
    * ``output_linear.weight`` — shape ``[hidden, shared_intermediate]``

    Forward::

        gate_up = x @ input_linear.T       # [*, 2*shared_intermediate]
        gate, up = split(gate_up, axis=-1)
        return act(gate) * up @ output_linear.T  # [*, hidden]
    """

    def __init__(self, config: GraniteMoeHybridConfig):
        super().__init__()
        self.input_linear = Linear(
            config.hidden_size,
            2 * config.shared_intermediate_size,
            bias=config.mlp_bias,
        )
        self.output_linear = Linear(
            config.shared_intermediate_size,
            config.hidden_size,
            bias=config.mlp_bias,
        )
        self.act_fn = get_activation(config.hidden_act)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        gate_up = self.input_linear(op, x)  # [*, 2*shared_intermediate]
        gate, up = op.Split(gate_up, axis=-1, num_outputs=2, _outputs=2)
        return self.output_linear(op, op.Mul(self.act_fn(op, gate), up))


# ---------------------------------------------------------------------------
# Decoder layers
# ---------------------------------------------------------------------------


class _GraniteMoeHybridMambaDecoderLayer(nn.Module):
    """GraniteMoeHybrid Mamba2 layer.

    input_layernorm → Mamba2Block → residual+scale →
    post_attention_layernorm → block_sparse_moe + shared_mlp → residual+scale.

    Args:
        config: GraniteMoeHybrid architecture config.
    """

    def __init__(self, config: GraniteMoeHybridConfig):
        super().__init__()
        d_inner = config.hidden_size * config.mamba_expand

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mamba = Mamba2Block(
            d_model=config.hidden_size,
            d_inner=d_inner,
            num_heads=config.mamba_n_heads,
            d_head=config.mamba_d_head,
            d_state=config.mamba_d_state,
            n_groups=config.mamba_n_groups,
            conv_kernel=config.mamba_d_conv,
            conv_bias=config.mamba_conv_bias,
            proj_bias=config.mamba_proj_bias,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Routed MoE with fused 3D expert weights
        self.block_sparse_moe = _FusedMoEBlock(config)

        # Dense shared MLP with fused gate+up weight
        self.shared_mlp = _FusedSharedMLP(config)

        self._residual_multiplier = config.residual_multiplier

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None,
    ):
        """Forward pass. Returns (hidden_states, (conv_state, ssm_state)).

        attention_bias and position_embeddings are unused by Mamba layers
        but accepted for a uniform interface with attention layers.
        """
        del attention_bias, position_embeddings  # unused by mamba layers

        # Mamba2/SSD path with pre-norm
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)

        conv_state, ssm_state = past_key_value if past_key_value is not None else (None, None)
        mamba_out, new_conv_state, new_ssm_state = self.mamba(
            op, hidden_states, conv_state, ssm_state
        )
        # residual + output * residual_multiplier
        rm = op.CastLike(op.Constant(value_float=self._residual_multiplier), mamba_out)
        hidden_states = op.Add(residual, op.Mul(mamba_out, rm))

        # MoE + shared-MLP path with pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        # Both routed MoE and shared MLP run on every layer; outputs are summed
        hidden_states = op.Add(
            self.block_sparse_moe(op, hidden_states),
            self.shared_mlp(op, hidden_states),
        )
        rm = op.CastLike(op.Constant(value_float=self._residual_multiplier), hidden_states)
        hidden_states = op.Add(residual, op.Mul(hidden_states, rm))

        return hidden_states, (new_conv_state, new_ssm_state)


class _GraniteMoeHybridAttentionDecoderLayer(nn.Module):
    """GraniteMoeHybrid attention layer.

    input_layernorm → GQA (NoPE) → residual+scale →
    post_attention_layernorm → block_sparse_moe + shared_mlp → residual+scale.

    Args:
        config: GraniteMoeHybrid architecture config.
    """

    def __init__(self, config: GraniteMoeHybridConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # GraniteMoeHybrid attention uses NoPE (no position embeddings)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Routed MoE with fused 3D expert weights
        self.block_sparse_moe = _FusedMoEBlock(config)

        # Dense shared MLP with fused gate+up weight
        self.shared_mlp = _FusedSharedMLP(config)

        self._residual_multiplier = config.residual_multiplier

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None,
    ):
        """Forward pass. Returns (hidden_states, (key, value))."""
        del position_embeddings  # GraniteMoeHybrid uses NoPE: no rotary embeddings

        # GQA attention path (no RoPE)
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        attn_out, present_kv = self.self_attn(
            op,
            hidden_states=hidden_states,
            attention_bias=attention_bias,
            position_embeddings=None,  # NoPE: skip rotary embedding application
            past_key_value=past_key_value,
        )
        rm = op.CastLike(op.Constant(value_float=self._residual_multiplier), attn_out)
        hidden_states = op.Add(residual, op.Mul(attn_out, rm))

        # MoE + shared-MLP path with pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = op.Add(
            self.block_sparse_moe(op, hidden_states),
            self.shared_mlp(op, hidden_states),
        )
        rm = op.CastLike(op.Constant(value_float=self._residual_multiplier), hidden_states)
        hidden_states = op.Add(residual, op.Mul(hidden_states, rm))

        return hidden_states, present_kv


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class _GraniteMoeHybridTextModel(nn.Module):
    """GraniteMoeHybrid text backbone: embedding -> N x (Mamba2|Attention) layers -> norm.

    Layer type ("mamba2" or "full_attention") is read from ``config.layer_types``.
    No rotary embeddings are used (NoPE for attention layers).
    """

    def __init__(self, config: GraniteMoeHybridConfig):
        super().__init__()
        self._dtype = config.dtype
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )

        layer_types = config.layer_types or []
        self.layers = nn.ModuleList([])
        for i in range(config.num_hidden_layers):
            ltype = layer_types[i] if i < len(layer_types) else "mamba2"
            if ltype == "mamba2":
                self.layers.append(_GraniteMoeHybridMambaDecoderLayer(config))
            else:
                self.layers.append(_GraniteMoeHybridAttentionDecoderLayer(config))

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # No rotary_emb: GraniteMoeHybrid uses NoPE (no positional encodings)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        del position_ids  # unused: NoPE architecture has no positional embeddings

        # (batch, seq, hidden)
        hidden_states = self.embed_tokens(op, input_ids)
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
                position_embeddings=None,  # NoPE: no RoPE
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values


class GraniteMoeHybridCausalLMModel(nn.Module):
    """GraniteMoeHybrid hybrid Mamba2+Attention causal language model with MoE FFN.

    Every layer has both a routed MoE block (``block_sparse_moe``) and a dense
    shared MLP (``shared_mlp``). Mamba2 layers use the SSD selective scan;
    attention layers use standard GQA without rotary position embeddings (NoPE).

    Uses ``HybridCausalLMTask`` with mixed ``"mamba2"`` and ``"full_attention"``
    layer types for the KV/SSM cache.

    HuggingFace reference: ``GraniteMoeHybridForCausalLM``.
    """

    default_task: str = "hybrid-text-generation"
    category: str = "Hybrid SSM+Attention"
    config_class: type = GraniteMoeHybridConfig

    def __init__(self, config: GraniteMoeHybridConfig):
        super().__init__()
        self.config = config
        self.model = _GraniteMoeHybridTextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states, present_key_values = self.model(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HuggingFace GraniteMoeHybridForCausalLM weights to ONNX parameters.

        Handles:
        1. Weight tying (embed_tokens ↔ lm_head)
        2. MoE gate: block_sparse_moe.router.layer.weight → block_sparse_moe.gate.weight

        Fused expert weights (``input_linear``, ``output_linear``) and shared-MLP
        weights pass through directly — the ONNX model stores them in the same
        fused layout as HuggingFace.
        """
        if self.config.tie_word_embeddings:
            tie_word_embeddings(state_dict)

        new_state_dict: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            # MoE gate: router.layer.weight → gate.weight
            new_key = key.replace(
                ".block_sparse_moe.router.layer.",
                ".block_sparse_moe.gate.",
            )
            new_state_dict[new_key] = value

        return new_state_dict


# ---------------------------------------------------------------------------
# Weight name mapping
# ---------------------------------------------------------------------------
