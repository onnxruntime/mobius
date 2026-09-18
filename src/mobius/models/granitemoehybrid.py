# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""GraniteMoeHybrid: Mamba2/SSD + Attention hybrid with optional MoE FFN.

Each layer has a dense shared MLP (``shared_mlp``) and, when the config has
routed experts (``num_local_experts > 0``), a routed MoE block
(``block_sparse_moe``); variants with no experts (e.g. granite-4.0-1b) run
only the shared MLP. The layer type ("mamba2" or "full_attention") controls
whether the token-mixing sub-block is a Mamba2/SSD or standard GQA. Attention
layers use RoPE when ``position_embedding_type == 'rope'`` (granite-4.0-1b)
and NoPE otherwise (granite-4.0-tiny-preview). Granite scaling multipliers
(embedding/attention/residual/logits) are applied throughout.

Forward pass per layer::

    residual = x
    x = input_layernorm(x)
    x = mamba(x)  OR  self_attn(x, position_embeddings)
    x = residual + x * residual_multiplier
    residual = x
    x = post_attention_layernorm(x)
    x = shared_mlp(x)  [+ block_sparse_moe(x) if experts]
    x = residual + x * residual_multiplier

HuggingFace reference: ``GraniteMoeHybridForCausalLM``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import OpBuilder, nn

from mobius._configs import GraniteMoeHybridConfig
from mobius.components import (
    Attention,
    Embedding,
    Linear,
    Mamba2Block,
    RMSNorm,
    TopKGate,
    create_attention_bias,
    get_activation,
    initialize_rope,
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

    def forward(self, op: OpBuilder, x: ir.Value, expert_index: int) -> ir.Value:
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
            config.hidden_size,
            config.num_local_experts,
            config.num_experts_per_tok,
            routed_scaling_factor=config.routed_scaling_factor,
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

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
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

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        gate_up = self.input_linear(op, x)  # [*, 2*shared_intermediate]
        gate, up = op.Split(gate_up, axis=-1, num_outputs=2, _outputs=2)
        return self.output_linear(op, op.Mul(self.act_fn(op, gate), up))


# ---------------------------------------------------------------------------
# Decoder layers
# ---------------------------------------------------------------------------


def _feedforward(
    op: OpBuilder,
    hidden_states: ir.Value,
    block_sparse_moe: nn.Module | None,
    shared_mlp: nn.Module | None,
) -> ir.Value:
    """Combined routed-MoE + shared-MLP feedforward.

    When routed experts exist, the output is ``moe(x) + shared_mlp(x)``; for
    variants with ``num_local_experts == 0`` (e.g. granite-4.0-1b) only the
    dense shared MLP runs. Mirrors HF ``GraniteMoeHybridDecoderLayer``.
    """
    shared_out = shared_mlp(op, hidden_states) if shared_mlp is not None else None
    if block_sparse_moe is None:
        assert shared_out is not None
        return shared_out
    routed_out = block_sparse_moe(op, hidden_states)
    return op.Add(routed_out, shared_out) if shared_out is not None else routed_out


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

        # Routed MoE with fused 3D expert weights. GraniteMoeHybrid variants with
        # ``num_local_experts == 0`` (e.g. granite-4.0-1b) have no routed experts;
        # only the dense shared MLP runs. Mirrors HF ``block_sparse_moe = MoE(...)
        # if num_local_experts > 0 else None``.
        self._has_experts = bool(config.num_local_experts)
        self.block_sparse_moe = _FusedMoEBlock(config) if self._has_experts else None

        # Dense shared MLP with fused gate+up weight
        self.shared_mlp = (
            _FusedSharedMLP(config) if config.shared_intermediate_size > 0 else None
        )

        self._residual_multiplier = config.residual_multiplier

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple | None,
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
        hidden_states = _feedforward(op, hidden_states, self.block_sparse_moe, self.shared_mlp)
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
        # Attention scale is Granite's ``attention_multiplier`` (a fixed value,
        # not the default 1/sqrt(head_dim)); RoPE is applied only when the text
        # model supplies ``position_embeddings`` (position_embedding_type='rope').
        self.self_attn = Attention(config, scale=config.attention_multiplier)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Routed MoE with fused 3D expert weights. Absent when there are no
        # routed experts (num_local_experts == 0), e.g. granite-4.0-1b.
        self._has_experts = bool(config.num_local_experts)
        self.block_sparse_moe = _FusedMoEBlock(config) if self._has_experts else None

        # Dense shared MLP with fused gate+up weight
        self.shared_mlp = (
            _FusedSharedMLP(config) if config.shared_intermediate_size > 0 else None
        )

        self._residual_multiplier = config.residual_multiplier

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple | None,
        past_key_value: tuple | None,
    ):
        """Forward pass. Returns (hidden_states, (key, value)).

        ``position_embeddings`` is ``None`` for NoPE variants and a
        ``(cos, sin)`` tuple when ``position_embedding_type == 'rope'``.
        """
        # GQA attention path (RoPE applied iff position_embeddings is not None)
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        attn_out, present_kv = self.self_attn(
            op,
            hidden_states=hidden_states,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
        )
        rm = op.CastLike(op.Constant(value_float=self._residual_multiplier), attn_out)
        hidden_states = op.Add(residual, op.Mul(attn_out, rm))

        # MoE + shared-MLP path with pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = _feedforward(op, hidden_states, self.block_sparse_moe, self.shared_mlp)
        rm = op.CastLike(op.Constant(value_float=self._residual_multiplier), hidden_states)
        hidden_states = op.Add(residual, op.Mul(hidden_states, rm))

        return hidden_states, present_kv


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class _GraniteMoeHybridTextModel(nn.Module):
    """GraniteMoeHybrid text backbone: embedding -> N x (Mamba2|Attention) layers -> norm.

    Layer type ("mamba2" or "full_attention") is read from ``config.layer_types``.
    Attention layers use RoPE when ``config`` declares rotary parameters
    (``position_embedding_type == 'rope'``, e.g. granite-4.0-1b) and NoPE
    otherwise (e.g. granite-4.0-tiny-preview).
    """

    def __init__(self, config: GraniteMoeHybridConfig):
        super().__init__()
        self._dtype = config.dtype
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )

        layer_types = config.layer_types or []
        if len(layer_types) != config.num_hidden_layers:
            raise ValueError(
                "GraniteHybrid layer_types must contain exactly num_hidden_layers entries"
            )
        if any(layer_type not in {"mamba2", "full_attention"} for layer_type in layer_types):
            raise ValueError(f"Unknown GraniteHybrid layer type in {layer_types!r}")
        self.layers = nn.ModuleList([])
        for i in range(config.num_hidden_layers):
            ltype = layer_types[i]
            if ltype == "mamba2":
                self.layers.append(_GraniteMoeHybridMambaDecoderLayer(config))
            else:
                self.layers.append(_GraniteMoeHybridAttentionDecoderLayer(config))

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # RoPE when the config declares rotary params; None => NoPE (attention
        # layers then receive position_embeddings=None).
        self.rotary_emb = initialize_rope(config)
        self.embedding_multiplier = config.embedding_multiplier

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        # (batch, seq, hidden)
        hidden_states = self.embed_tokens(op, input_ids)
        # Granite scales embeddings by embedding_multiplier after lookup.
        hidden_states = op.Mul(hidden_states, self.embedding_multiplier)

        # (cos, sin) tuple for RoPE, or None for NoPE variants.
        position_embeddings = (
            self.rotary_emb(op, position_ids) if self.rotary_emb is not None else None
        )
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


class GraniteMoeHybridCausalLMModel(nn.Module):
    """GraniteMoeHybrid hybrid Mamba2+Attention causal language model.

    Routed-MoE checkpoints run ``block_sparse_moe`` plus ``shared_mlp``; dense
    checkpoints run ``shared_mlp`` alone. Mamba2 layers use the SSD selective
    scan, while attention layers use RoPE or NoPE according to the config.

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
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_scaling = config.logits_scaling

    def forward(
        self,
        op: OpBuilder,
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
        # Granite divides final logits by logits_scaling.
        logits = op.Div(logits, self.logits_scaling)
        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HuggingFace GraniteMoeHybridForCausalLM weights to ONNX parameters.

        Handles:
        1. Weight tying (embed_tokens ↔ lm_head)
        2. MoE gate: block_sparse_moe.router[.layer].weight → block_sparse_moe.gate.weight
        3. Fused expert weights: HF renamed the routed-expert tensors from
           ``block_sparse_moe.{input,output}_linear.weight`` to
           ``block_sparse_moe.experts.{gate_up,down}_proj`` (transformers >=5.x).
           The layouts are identical, so only the names are remapped.

        Shared-MLP weights (``shared_mlp.{input,output}_linear``) pass through
        directly — the ONNX model stores them in the same fused layout as HF.
        """
        if self.config.tie_word_embeddings:
            if "model.embed_tokens.weight" not in state_dict:
                state_dict["model.embed_tokens.weight"] = state_dict["lm_head.weight"]
            state_dict.pop("lm_head.weight", None)

        new_state_dict: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = key
            # MoE gate: router.layer.weight (legacy) / router.weight (current) → gate.weight
            new_key = new_key.replace(
                ".block_sparse_moe.router.layer.",
                ".block_sparse_moe.gate.",
            )
            new_key = new_key.replace(
                ".block_sparse_moe.router.weight",
                ".block_sparse_moe.gate.weight",
            )
            # Routed-expert weights: HF renamed the fused 3D tensors.
            new_key = new_key.replace(
                ".block_sparse_moe.experts.gate_up_proj",
                ".block_sparse_moe.input_linear.weight",
            )
            new_key = new_key.replace(
                ".block_sparse_moe.experts.down_proj",
                ".block_sparse_moe.output_linear.weight",
            )
            new_state_dict[new_key] = value

        return new_state_dict


# ---------------------------------------------------------------------------
# Weight name mapping
# ---------------------------------------------------------------------------
