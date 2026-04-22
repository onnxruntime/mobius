# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""NemotronH hybrid Mamba2 + Attention + MLP + MoE causal language model.

NemotronH interleaves up to four layer types in a configurable pattern:
- Mamba2/SSD layers for efficient recurrent processing
- Transformer attention layers for global context
- Dense MLP layers for feedforward computation
- MoE layers for sparse expert routing (Nemotron-3 30B/120B)

Each layer is a single-mixer block: RMSNorm → mixer → residual.
Unlike Jamba/Bamba where every layer has a mixer AND MLP, NemotronH
treats MLP and MoE as standalone layer types.

Layer types are specified via ``layers_block_type`` in the HF config:
``M`` = mamba, ``*`` = attention, ``-`` = mlp (dense feedforward),
``E`` = moe (sparse mixture of experts).

State per layer:
    Mamba2: conv_state (batch, conv_dim, d_conv-1)
            ssm_state (batch, num_heads, d_head, d_state)
    Attention: standard KV cache (key + value)
    MLP: stateless — no cache
    MoE: stateless — no cache

HuggingFace reference: ``NemotronHForCausalLM``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import NemotronHConfig
from mobius._weight_utils import tie_word_embeddings
from mobius.components import (
    FCMLP,
    Attention,
    Embedding,
    Linear,
    Mamba2Block,
    RMSNorm,
    create_padding_mask,
)

if TYPE_CHECKING:
    import onnx_ir as ir

# ---------------------------------------------------------------------------
# Decoder layers
# ---------------------------------------------------------------------------


class NemotronHMambaLayer(nn.Module):
    """NemotronH Mamba2 layer: RMSNorm → Mamba2Block → residual.

    Single-mixer block — no MLP path.

    Args:
        config: NemotronH architecture config.
    """

    def __init__(self, config: NemotronHConfig):
        super().__init__()
        # d_inner = num_heads * head_dim (not hidden_size * expand)
        d_inner = config.mamba_n_heads * config.mamba_d_head

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
            # NemotronH uses grouped RMSNorm: normalize within each
            # group of heads_per_group * head_dim dimensions.
            norm_group_size=d_inner // config.mamba_n_groups,
            time_step_min=config.mamba_time_step_min,
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        but accepted for uniform interface with attention layers.
        """
        del attention_bias, position_embeddings  # unused

        # Pre-norm → Mamba2 → residual
        residual = hidden_states
        hidden_states = self.norm(op, hidden_states)

        conv_state, ssm_state = past_key_value if past_key_value is not None else (None, None)
        mamba_out, new_conv_state, new_ssm_state = self.mamba(
            op, hidden_states, conv_state, ssm_state
        )
        hidden_states = op.Add(residual, mamba_out)

        return hidden_states, (new_conv_state, new_ssm_state)


class NemotronHAttentionLayer(nn.Module):
    """NemotronH attention layer: RMSNorm → Attention → residual.

    Single-mixer block — no MLP path.

    Args:
        config: NemotronH architecture config.
    """

    def __init__(self, config: NemotronHConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None,
    ):
        """Forward pass. Returns (hidden_states, (key, value))."""
        residual = hidden_states
        hidden_states = self.norm(op, hidden_states)

        attn_output, present_kv = self.self_attn(
            op,
            hidden_states=hidden_states,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
        )
        hidden_states = op.Add(residual, attn_output)

        return hidden_states, present_kv


class NemotronHMLPLayer(nn.Module):
    """NemotronH MLP layer: RMSNorm → FCMLP → residual.

    Single-mixer block — stateless, no cache.

    Args:
        config: NemotronH architecture config.
    """

    def __init__(self, config: NemotronHConfig):
        super().__init__()
        self.mlp = FCMLP(
            config.hidden_size,
            config.intermediate_size,
            activation=config.hidden_act,
            bias=config.mlp_bias,
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None,
    ):
        """Forward pass. Returns (hidden_states, (None, None)).

        MLP layers are stateless — the None pair keeps the cache
        list aligned with all layers.
        """
        del attention_bias, position_embeddings, past_key_value  # unused

        # Pre-norm → MLP → residual
        residual = hidden_states
        hidden_states = self.norm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, (None, None)


# ---------------------------------------------------------------------------
# MoE components (Nemotron-3 30B/120B)
# ---------------------------------------------------------------------------


class NemotronHMoEGate(nn.Module):
    """Sigmoid top-k gate with score correction bias (NemotronH style).

    Routing:
        1. router_logits = linear(hidden_states)  [in float32]
        2. probs = sigmoid(router_logits)
        3. choice_scores = probs + e_score_correction_bias
        4. selected_experts = topk(choice_scores)
        5. routing_weights = gather(probs, selected_experts)
        6. normalize + scale

    The correction bias shifts expert selection but does NOT affect the
    final routing weights (which come from the original sigmoid probs).
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        *,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.weight = nn.Parameter([num_experts, hidden_size])
        # Correction bias for expert selection (loaded from checkpoint)
        self.e_score_correction_bias = nn.Parameter([num_experts])

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        # Route in float32 for numerical stability
        weight_t = op.Transpose(self.weight, perm=[1, 0])
        router_logits = op.MatMul(hidden_states, weight_t)

        # Sigmoid probabilities (these become the final routing weights)
        probs = op.Sigmoid(router_logits)

        # Add correction bias for expert selection only
        choice_scores = op.Add(probs, self.e_score_correction_bias)

        # Select top-k experts based on biased scores
        k = op.Constant(value_ints=[self.top_k])
        _top_vals, selected_experts = op.TopK(choice_scores, k, axis=-1, _outputs=2)

        # Gather actual routing weights from unbiased probs
        routing_weights = op.GatherElements(probs, selected_experts, axis=-1)

        if self.norm_topk_prob:
            weight_sum = op.ReduceSum(routing_weights, [-1], keepdims=True)
            routing_weights = op.Div(routing_weights, op.Add(weight_sum, 1e-20))
        if self.routed_scaling_factor != 1.0:  # noqa: RUF069
            routing_weights = op.Mul(routing_weights, self.routed_scaling_factor)

        return routing_weights, selected_experts


class NemotronHMoEBlock(nn.Module):
    """NemotronH MoE block with non-gated experts and shared expert.

    Unlike standard MoE (gated MLP experts), NemotronH uses:
    - Non-gated FCMLP experts: up_proj → act → down_proj
    - Optional latent projection wrapping the routed experts
    - Shared expert (FCMLP) added as residual

    Architecture::

        [optional] hidden → fc1_latent_proj → latent
        latent → routed experts (FCMLP) → expert_output
        [optional] expert_output → fc2_latent_proj → hidden
        output = expert_output + shared_experts(original_hidden)

    HuggingFace reference: ``NemotronHMoE``.
    """

    def __init__(self, config: NemotronHConfig):
        super().__init__()
        assert config.num_local_experts is not None
        assert config.num_experts_per_tok is not None
        num_experts = config.num_local_experts
        top_k = config.num_experts_per_tok

        self.gate = NemotronHMoEGate(
            config.hidden_size,
            num_experts,
            top_k,
            norm_topk_prob=config.norm_topk_prob,
            routed_scaling_factor=config.routed_scaling_factor,
        )

        # Expert input/output dimension depends on latent projection
        expert_input_dim = (
            config.moe_latent_size
            if config.moe_latent_size is not None
            else config.hidden_size
        )
        assert config.moe_intermediate_size is not None
        self.experts = nn.ModuleList(
            [
                FCMLP(
                    expert_input_dim,
                    config.moe_intermediate_size,
                    activation=config.hidden_act,
                    bias=config.mlp_bias,
                )
                for _ in range(num_experts)
            ]
        )

        # Shared expert processes all tokens (not routed)
        shared_intermediate = (
            config.shared_expert_intermediate_size or config.moe_intermediate_size
        )
        self.shared_experts = FCMLP(
            config.hidden_size,
            shared_intermediate,
            activation=config.hidden_act,
            bias=config.mlp_bias,
        )

        # Optional latent projection (e.g. 120B: 4096 → 1024 → experts
        # → 1024 → 4096)
        self._has_latent = config.moe_latent_size is not None
        if self._has_latent:
            self.fc1_latent_proj = Linear(
                config.hidden_size,
                config.moe_latent_size,
                bias=config.mlp_bias,
            )
            self.fc2_latent_proj = Linear(
                config.moe_latent_size,
                config.hidden_size,
                bias=config.mlp_bias,
            )

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        residual = hidden_states

        # Gate routes on original hidden states
        routing_weights, selected_experts = self.gate(op, hidden_states)

        # Optional latent projection before expert dispatch
        if self._has_latent:
            hidden_states = self.fc1_latent_proj(op, hidden_states)

        # Loop-over-experts dispatch: each expert processes all tokens,
        # then results are masked and weighted by routing weights
        result = None
        for expert_idx, expert in enumerate(self.experts):
            expert_output = expert(op, hidden_states)
            expert_id = op.Constant(value_int=expert_idx)
            # match: True where this expert was selected
            match = op.Equal(selected_experts, expert_id)
            match_float = op.CastLike(match, routing_weights)
            weighted = op.Mul(routing_weights, match_float)
            # Sum matched routing weights across top_k dim → per-token weight
            weight = op.ReduceSum(weighted, [-1], keepdims=True)
            contribution = op.Mul(expert_output, weight)
            if result is None:
                result = contribution
            else:
                result = op.Add(result, contribution)

        # Optional latent projection back to hidden_size
        if self._has_latent:
            result = self.fc2_latent_proj(op, result)

        # Add shared expert output (operates on original hidden states)
        shared_output = self.shared_experts(op, residual)
        result = op.Add(result, shared_output)
        return result


class NemotronHMoELayer(nn.Module):
    """NemotronH MoE layer: RMSNorm → MoE block → residual.

    Single-mixer block — stateless, no cache.

    Args:
        config: NemotronH architecture config.
    """

    def __init__(self, config: NemotronHConfig):
        super().__init__()
        self.moe = NemotronHMoEBlock(config)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None,
    ):
        """Forward pass. Returns (hidden_states, (None, None)).

        MoE layers are stateless — the None pair keeps the cache
        list aligned with all layers.
        """
        del attention_bias, position_embeddings, past_key_value  # unused

        # Pre-norm → MoE → residual
        residual = hidden_states
        hidden_states = self.norm(op, hidden_states)
        hidden_states = self.moe(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, (None, None)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class _NemotronHTextModel(nn.Module):
    """NemotronH text backbone: embedding → N x (Mamba2|Attention|MLP) → norm.

    Layer types are selected based on ``config.layer_types``:
        ``"mamba2"`` → NemotronHMambaLayer
        ``"full_attention"`` → NemotronHAttentionLayer
        ``"mlp"`` → NemotronHMLPLayer
    """

    def __init__(self, config: NemotronHConfig):
        super().__init__()
        self._dtype = config.dtype
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )

        layer_types = config.layer_types or []
        self.layers = nn.ModuleList([])
        for i in range(config.num_hidden_layers):
            ltype = layer_types[i] if i < len(layer_types) else "full_attention"
            if ltype == "mamba2":
                self.layers.append(NemotronHMambaLayer(config))
            elif ltype == "mlp":
                self.layers.append(NemotronHMLPLayer(config))
            elif ltype == "moe":
                self.layers.append(NemotronHMoELayer(config))
            else:
                self.layers.append(NemotronHAttentionLayer(config))

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states = self.embed_tokens(op, input_ids)

        # NemotronH does NOT use positional embeddings.  The HF reference
        # (NemotronHAttention.forward) applies no rotary encoding — the
        # model relies on Mamba layers' inherent position-awareness.
        attention_bias = create_padding_mask(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, past_kv in zip(self.layers, past_kvs):
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                position_embeddings=None,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values


class NemotronHCausalLMModel(nn.Module):
    """NemotronH hybrid Mamba2+Attention+MLP causal language model.

    Uses ``HybridCausalLMTask`` with mixed ``"mamba2"``,
    ``"full_attention"``, and ``"mlp"`` layer types for the cache.

    HuggingFace reference: ``NemotronHForCausalLM``.
    """

    default_task: str = "hybrid-text-generation"
    category: str = "Hybrid SSM+Attention"
    config_class: type = NemotronHConfig

    def __init__(self, config: NemotronHConfig):
        super().__init__()
        self.config = config
        self.model = _NemotronHTextModel(config)
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
        """Map HuggingFace NemotronHForCausalLM weights to ONNX parameters.

        Handles:
        1. Weight tying (embed_tokens ↔ lm_head)
        2. ``backbone.`` → ``model.`` prefix rename
        3. ``backbone.embeddings.`` → ``model.embed_tokens.``
        4. ``backbone.norm_f.`` → ``model.norm.``
        5. Per-layer ``mixer.`` rename based on layer type:
           - mamba: ``mixer.`` → ``mamba.``
           - attention: ``mixer.`` → ``self_attn.``
           - mlp: ``mixer.`` → ``mlp.``
           - moe: ``mixer.`` → ``moe.``
        6. MoE stacked 3D expert tensors split into per-expert 2D weights
        """
        layer_types = self.config.layer_types or []

        if self.config.tie_word_embeddings:
            # Detect old "backbone.*" vs new "model.*" prefix
            embed_key = (
                "backbone.embeddings.weight"
                if "backbone.embeddings.weight" in state_dict
                else "model.embeddings.weight"
            )
            tie_word_embeddings(
                state_dict,
                embed_key=embed_key,
                head_key="lm_head.weight",
            )

        new_state_dict: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = _rename_nemotron_h_weight(key, layer_types)
            # Split stacked 3D expert tensors into per-expert 2D weights.
            # HF stores experts.up_proj as (num_experts, inter, input) and
            # experts.down_proj as (num_experts, input, inter). We need
            # individual experts.{i}.up_proj.weight / down_proj.weight.
            if _is_stacked_expert_tensor(new_key, value):
                for i, expert_weight in enumerate(value):
                    # value[i] is (out_dim, in_dim) — standard 2D weight
                    suffix = new_key.rsplit(".", 1)[-1]  # "up_proj"/"down_proj"
                    expert_key = (
                        new_key.rsplit("experts.", 1)[0] + f"experts.{i}.{suffix}.weight"
                    )
                    new_state_dict[expert_key] = expert_weight
            else:
                new_state_dict[new_key] = value

        return new_state_dict


# Layer index regex: {backbone|model}.layers.<N>.<rest>
# Older HF checkpoints use "backbone.*"; newer ones use "model.*".
_LAYER_RE = re.compile(r"^(?:backbone|model)\.layers\.(\d+)\.(.+)$")


def _rename_nemotron_h_weight(key: str, layer_types: list[str]) -> str:
    """Rename a single HF weight key to match ONNX module structure.

    HF NemotronHForCausalLM weight naming (old ``backbone.*`` / new ``model.*``):
        {backbone|model}.embeddings.weight
        {backbone|model}.norm_f.weight
        {backbone|model}.layers.N.norm.weight
        {backbone|model}.layers.N.mixer.{in_proj, conv1d, out_proj, norm, A_log, D, dt_bias}  (mamba)
        {backbone|model}.layers.N.mixer.{q_proj, k_proj, v_proj, o_proj}.weight              (attention)
        {backbone|model}.layers.N.mixer.{up_proj, down_proj}.weight                          (mlp)
        {backbone|model}.layers.N.mixer.gate.{weight, e_score_correction_bias}               (moe gate)
        {backbone|model}.layers.N.mixer.experts.{up_proj, down_proj}                         (moe experts, 3D stacked)
        {backbone|model}.layers.N.mixer.shared_experts.{up_proj, down_proj}.weight           (moe shared expert)
        {backbone|model}.layers.N.mixer.{fc1_latent_proj, fc2_latent_proj}.weight             (moe latent)
        lm_head.weight

    ONNX parameter naming:
        model.embed_tokens.weight
        model.norm.weight
        model.layers.N.norm.weight
        model.layers.N.mamba.{in_proj, conv1d, out_proj, norm, A_log, D, dt_bias}
        model.layers.N.self_attn.{q_proj, k_proj, v_proj, o_proj}.weight
        model.layers.N.mlp.{up_proj, down_proj}.weight
        model.layers.N.moe.gate.{weight, e_score_correction_bias}
        model.layers.N.moe.experts.{i}.{up_proj, down_proj}.weight  (split from 3D)
        model.layers.N.moe.shared_experts.{up_proj, down_proj}.weight
        model.layers.N.moe.{fc1_latent_proj, fc2_latent_proj}.weight
        lm_head.weight
    """
    # Global prefix renames (handle both backbone.* and model.* HF names)
    for prefix in ("backbone.", "model."):
        if key.startswith(f"{prefix}embeddings."):
            return key.replace(f"{prefix}embeddings.", "model.embed_tokens.", 1)
        if key.startswith(f"{prefix}norm_f."):
            return key.replace(f"{prefix}norm_f.", "model.norm.", 1)

    # Per-layer renames
    m = _LAYER_RE.match(key)
    if m:
        layer_idx = int(m.group(1))
        rest = m.group(2)
        ltype = layer_types[layer_idx] if layer_idx < len(layer_types) else "full_attention"

        if rest.startswith("mixer."):
            mixer_rest = rest[len("mixer.") :]
            if ltype == "mamba2":
                return f"model.layers.{layer_idx}.mamba.{mixer_rest}"
            elif ltype == "full_attention":
                return f"model.layers.{layer_idx}.self_attn.{mixer_rest}"
            elif ltype == "moe":
                return f"model.layers.{layer_idx}.moe.{mixer_rest}"
            else:  # mlp
                return f"model.layers.{layer_idx}.mlp.{mixer_rest}"

        # norm.weight stays as norm.weight (already matching)
        return f"model.layers.{layer_idx}.{rest}"

    # Catch-all for backbone.X → model.X (shouldn't normally hit)
    if key.startswith("backbone."):
        return key.replace("backbone.", "model.", 1)

    return key


def _is_stacked_expert_tensor(key: str, value: torch.Tensor) -> bool:
    """Check if a weight is a stacked 3D expert tensor that needs splitting.

    HF NemotronH stores expert weights as (num_experts, out_dim, in_dim).
    These need to be split into per-expert 2D weights.
    """
    return (
        value.ndim == 3 and ".moe.experts." in key and key.endswith((".up_proj", ".down_proj"))
    )
