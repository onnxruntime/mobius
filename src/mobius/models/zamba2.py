# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Zamba2 hybrid Mamba2/SSD + Attention causal language model.

Zamba2 interleaves Mamba2 (SSD) layers with "hybrid" layers that contain
BOTH a shared attention transformer block AND a Mamba2 block.  The attention
block processes a concatenation of the current hidden states with the original
embedding output (2 * hidden_size → hidden_size), then its output is injected
into the Mamba block as an additive residual.

Architecture (per the Zamba2 paper, Eq. 6):
    For pure Mamba layers:
        hidden = residual + Mamba(LayerNorm(hidden))

    For hybrid layers:
        transformer_out = Linear(SharedTransformer(concat(hidden, embed_out)))
        hidden = residual + Mamba(LayerNorm(hidden + transformer_out))

Weight sharing:
    The shared transformer block (attention + MLP) is tied across all hybrid
    layers.  In the ONNX model, weights are duplicated to each logical
    attention layer by ``preprocess_weights``.

Logical layer expansion:
    Each physical "hybrid" layer is represented as two logical layers in the
    ONNX model (for cache alignment):
    - ``"full_attention"`` layer: shared transformer (concat → norm → attention → norm → MLP → linear)
    - ``"mamba2"`` layer: Mamba with transformer injection

State per logical layer:
    Mamba2: conv_state (batch, conv_dim, d_conv-1)
            ssm_state (batch, num_heads, d_state, d_head)
    Attention: standard KV cache (key + value)

HuggingFace reference: ``Zamba2ForCausalLM``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import Zamba2Config
from mobius._weight_utils import tie_word_embeddings
from mobius.components import (
    Embedding,
    Linear,
    Mamba2Block,
    RMSNorm,
    create_attention_bias,
    get_activation,
    initialize_rope,
)

if TYPE_CHECKING:
    import onnx_ir as ir

# ---------------------------------------------------------------------------
# Decoder layers
# ---------------------------------------------------------------------------


class Zamba2MambaDecoderLayer(nn.Module):
    """Pure Mamba2 layer: RMSNorm → Mamba2Block → residual.

    Does NOT receive transformer injection.  Used for mamba-only physical layers.

    Args:
        config: Zamba2 architecture config.
    """

    def __init__(self, config: Zamba2Config):
        super().__init__()
        d_inner = config.hidden_size * config.mamba_expand

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
            # HF Zamba2 hardcodes eps=1e-5 for the mamba GatedRMSNorm
            # (Zamba2RMSNormGated), distinct from config.rms_norm_eps.
            eps=1e-5,
            time_step_min=config.mamba_time_step_min,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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

        # Mamba2 path with pre-norm and residual
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)

        conv_state, ssm_state = past_key_value if past_key_value is not None else (None, None)
        mamba_out, new_conv_state, new_ssm_state = self.mamba(
            op, hidden_states, conv_state, ssm_state
        )
        hidden_states = op.Add(residual, mamba_out)

        return hidden_states, (new_conv_state, new_ssm_state)


class _Adapter(nn.Module):
    """Low-rank adapter: down_proj → up_proj (no bias, no activation).

    Computes: adapter(x) = up(down(x))
    where down: (in_features → rank) and up: (rank → out_features).
    """

    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.down = Linear(in_features, rank, bias=False)
        self.up = Linear(rank, out_features, bias=False)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        return self.up(op, self.down(op, x))


class _Zamba2QKVAdapters(nn.Module):
    """Per-hybrid-layer Q/K/V adapters.

    Computes low-rank adapter contributions for query, key, and value
    projections. Must be called via __call__ so onnxscript correctly
    pushes this module's scope prefix for initializer naming.
    """

    def __init__(self, config: Zamba2Config):
        super().__init__()
        attn_hidden = config.attention_hidden_size
        head_dim = config.head_dim
        rank = config.adapter_rank

        self.q_adapter = _Adapter(
            attn_hidden, config.num_attention_heads * head_dim, rank
        )
        self.k_adapter = _Adapter(
            attn_hidden, config.num_key_value_heads * head_dim, rank
        )
        self.v_adapter = _Adapter(
            attn_hidden, config.num_key_value_heads * head_dim, rank
        )

    def forward(
        self, op: builder.OpBuilder, attn_input: ir.Value
    ) -> tuple:
        """Compute Q/K/V adapter outputs from layer-normed concat hidden."""
        return (
            self.q_adapter(op, attn_input),
            self.k_adapter(op, attn_input),
            self.v_adapter(op, attn_input),
        )


class _Zamba2MLPAdapter(nn.Module):
    """Per-hybrid-layer MLP adapter.

    Computes low-rank adapter contribution for the fused gate+up MLP
    projection. Must be called via __call__ for correct scope naming.
    """

    def __init__(self, config: Zamba2Config):
        super().__init__()
        rank = config.adapter_rank
        self.mlp_adapter = _Adapter(
            config.hidden_size, 2 * config.intermediate_size, rank
        )

    def forward(
        self, op: builder.OpBuilder, mlp_input: ir.Value
    ) -> ir.Value:
        """Compute MLP adapter output from pre-FFN hidden states."""
        return self.mlp_adapter(op, mlp_input)


class _Zamba2AttentionProjections(nn.Module):
    """Q/K/V/O projections for Zamba2 attention (grouped under self_attn).

    Produces query, key, value projections from the concatenated input and
    an output projection from the attention output.
    """

    def __init__(self, config: Zamba2Config):
        super().__init__()
        attn_hidden = config.attention_hidden_size
        head_dim = config.head_dim
        self.q_proj = Linear(attn_hidden, config.num_attention_heads * head_dim, bias=False)
        self.k_proj = Linear(attn_hidden, config.num_key_value_heads * head_dim, bias=False)
        self.v_proj = Linear(attn_hidden, config.num_key_value_heads * head_dim, bias=False)
        self.o_proj = Linear(
            config.num_attention_heads * head_dim, config.hidden_size, bias=False
        )

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attn_output: ir.Value | None = None,
    ):
        """Project input to Q/K/V or project attention output.

        When attn_output is None: returns (query, key, value) projections.
        When attn_output is given: returns o_proj(attn_output).
        """
        if attn_output is not None:
            return self.o_proj(op, attn_output)
        query = self.q_proj(op, hidden_states)
        key = self.k_proj(op, hidden_states)
        value = self.v_proj(op, hidden_states)
        return query, key, value


class Zamba2SharedTransformerLayer(nn.Module):
    """Shared attention transformer layer for Zamba2 hybrid layers.

    Concatenates hidden_states with original_hidden_states to form
    attention_hidden_size (2 * hidden_size) input, then:
        LayerNorm → Attention (+ adapters) → LayerNorm → MLP (+ adapter)

    Weights in this module are SHARED across all hybrid layers.
    Per-layer differentiation is achieved via adapters passed to forward().

    This layer produces the attention KV cache.

    Args:
        config: Zamba2 architecture config.
    """

    def __init__(self, config: Zamba2Config):
        super().__init__()
        head_dim = config.head_dim

        # Attention projections grouped under self_attn for naming
        self.self_attn = _Zamba2AttentionProjections(config)

        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = head_dim
        # Zamba2 uses scaling = (head_dim / 2) ** -0.5
        self.scaling = (head_dim / 2) ** -0.5

        # Pre-FFN norm (stays inside shared_transformer for correct scope)
        self.pre_ff_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        concat_hidden: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple | None,
        past_key_value: tuple | None,
        q_adapter_out: ir.Value | None = None,
        k_adapter_out: ir.Value | None = None,
        v_adapter_out: ir.Value | None = None,
    ):
        """Attention phase only. Returns (mlp_input, (key, value)).

        Handles: Q/K/V projections + adapters → RoPE → Attention → O proj
        → pre_ff_layernorm. Returns the pre-MLP hidden states so the caller
        can apply the MLP with per-layer adapter at the correct scope.

        Args:
            concat_hidden: Layer-normed concat input (batch, seq, attn_hidden)
            attention_bias: Causal attention mask
            position_embeddings: RoPE (cos, sin) or None
            past_key_value: (past_key, past_value) or None
            q_adapter_out: Q adapter contribution (or None)
            k_adapter_out: K adapter contribution (or None)
            v_adapter_out: V adapter contribution (or None)

        Returns:
            (mlp_input, (present_key, present_value))
        """
        # Q/K/V projections from attention_hidden_size
        query_states, key_states, value_states = self.self_attn(op, concat_hidden)

        # Add adapter contributions to Q/K/V
        if q_adapter_out is not None:
            query_states = op.Add(query_states, q_adapter_out)
        if k_adapter_out is not None:
            key_states = op.Add(key_states, k_adapter_out)
        if v_adapter_out is not None:
            value_states = op.Add(value_states, v_adapter_out)

        # Apply RoPE if position_embeddings provided
        if position_embeddings is not None:
            from mobius.components._rotary_embedding import apply_rotary_pos_emb

            query_states = apply_rotary_pos_emb(
                op,
                x=query_states,
                position_embeddings=position_embeddings,
                num_heads=self.num_attention_heads,
                rotary_embedding_dim=0,
                interleaved=False,
            )
            key_states = apply_rotary_pos_emb(
                op,
                x=key_states,
                position_embeddings=position_embeddings,
                num_heads=self.num_key_value_heads,
                rotary_embedding_dim=0,
                interleaved=False,
            )

        # Attention with KV cache
        from mobius.components._attention import _apply_attention

        past_key = past_key_value[0] if past_key_value is not None else None
        past_value = past_key_value[1] if past_key_value is not None else None

        attn_output, present_key, present_value = _apply_attention(
            op,
            query_states,
            key_states,
            value_states,
            attention_bias,
            past_key,
            past_value,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            scale=self.scaling,
        )

        # Output projection: num_heads * head_dim → hidden_size
        attn_output = self.self_attn(op, concat_hidden, attn_output=attn_output)

        # Pre-FFN layer norm — returns hidden states ready for MLP
        mlp_input = self.pre_ff_layernorm(op, attn_output)

        return mlp_input, (present_key, present_value)


class Zamba2InjectedMambaLayer(nn.Module):
    """Mamba2 layer with transformer injection (hybrid layer's Mamba part).

    Receives transformer_hidden_states from the preceding shared transformer
    layer and adds it to hidden_states before applying LayerNorm → Mamba2.

    Implements Eq. 6 from the Zamba2 paper:
        hidden = residual + Mamba(LayerNorm(hidden + transformer_out))

    Args:
        config: Zamba2 architecture config.
    """

    def __init__(self, config: Zamba2Config):
        super().__init__()
        d_inner = config.hidden_size * config.mamba_expand

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
            # HF Zamba2 hardcodes eps=1e-5 for the mamba GatedRMSNorm
            # (Zamba2RMSNormGated), distinct from config.rms_norm_eps.
            eps=1e-5,
            time_step_min=config.mamba_time_step_min,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        transformer_hidden_states: ir.Value,
        past_key_value: tuple | None,
    ):
        """Forward pass with transformer injection.

        Returns (hidden_states, (conv_state, ssm_state)).
        """
        residual = hidden_states

        # Inject transformer output (Eq. 6: hidden + transformer_out)
        hidden_states = op.Add(hidden_states, transformer_hidden_states)

        # Pre-norm → Mamba2 → residual
        hidden_states = self.input_layernorm(op, hidden_states)

        conv_state, ssm_state = past_key_value if past_key_value is not None else (None, None)
        mamba_out, new_conv_state, new_ssm_state = self.mamba(
            op, hidden_states, conv_state, ssm_state
        )
        hidden_states = op.Add(residual, mamba_out)

        return hidden_states, (new_conv_state, new_ssm_state)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class _Zamba2TextModel(nn.Module):
    """Zamba2 text backbone: embedding → N x (Mamba2|SharedTransformer+Mamba2) → norm.

    Architecture:
    - ONE shared transformer (attention + norms + MLP) — weights shared across
      all hybrid layers
    - Per-hybrid-layer adapters (Q/K/V/MLP low-rank) that differentiate each
      usage of the shared transformer
    - Per-hybrid-layer linear projection (NOT shared)
    - Per-layer Mamba2 blocks (pure or injected)

    The ``layers`` ModuleList contains only Mamba layers (pure and injected).
    The shared transformer and adapters are separate attributes.
    """

    def __init__(self, config: Zamba2Config):
        super().__init__()
        self._dtype = config.dtype
        self._use_rope = config.attention_hidden_size != 2 * config.hidden_size
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )

        layer_types = config.layer_types or []

        # ONE shared transformer for all hybrid layers
        self.shared_transformer = Zamba2SharedTransformerLayer(config)

        # Input layernorm for hybrid attention (operates on concat_hidden
        # which has attention_hidden_size). Lives at TextModel scope because
        # it's called from the forward loop (ONNX name: model.input_layernorm)
        attn_hidden = config.attention_hidden_size or 2 * config.hidden_size
        self.input_layernorm = RMSNorm(attn_hidden, eps=config.rms_norm_eps)

        # Per-hybrid-layer adapters and linear projections
        num_hybrid = sum(1 for t in layer_types if t == "full_attention")
        self.qkv_adapters = nn.ModuleList(
            [_Zamba2QKVAdapters(config) for _ in range(num_hybrid)]
        )
        self.mlp_adapters = nn.ModuleList(
            [_Zamba2MLPAdapter(config) for _ in range(num_hybrid)]
        )
        self.linears = nn.ModuleList(
            [Linear(config.hidden_size, config.hidden_size, bias=False)
             for _ in range(num_hybrid)]
        )

        # Shared MLP projections (used by all hybrid layers, lives at model
        # scope for correct ONNX naming: model.gate_proj, model.up_proj, etc.)
        self.gate_proj = Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = Linear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias
        )
        self._act_fn = get_activation(config.hidden_act)

        # Mamba layers only — attention is handled by shared_transformer.
        # We use _layer_types to drive forward iteration and cache alignment.
        self.mamba_layers = nn.ModuleList([])
        i = 0
        while i < len(layer_types):
            if layer_types[i] == "full_attention":
                # Skip attention slot; next must be injected mamba2
                i += 1
                assert i < len(layer_types) and layer_types[i] == "mamba2"
                self.mamba_layers.append(Zamba2InjectedMambaLayer(config))
                i += 1
            else:
                # Pure mamba layer
                self.mamba_layers.append(Zamba2MambaDecoderLayer(config))
                i += 1

        self._layer_types = layer_types
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if self._use_rope:
            self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states = self.embed_tokens(op, input_ids)

        # Save original embedding output for hybrid layer concatenation
        original_hidden_states = hidden_states

        # Position embeddings for attention layers (only if model uses RoPE)
        position_embeddings = None
        if self._use_rope:
            position_embeddings = self.rotary_emb(op, position_ids)

        attention_bias = create_attention_bias(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )

        num_logical_layers = len(self._layer_types)
        present_key_values = []
        past_kvs = past_key_values or [None] * num_logical_layers

        # Track transformer output for injection into mamba layers
        transformer_hidden_states = None
        hybrid_idx = 0  # Counter for adapters/linears
        mamba_idx = 0  # Counter into self.mamba_layers

        for logical_idx in range(num_logical_layers):
            ltype = self._layer_types[logical_idx]
            past_kv = past_kvs[logical_idx]

            if ltype == "full_attention":
                # Compute concat + layer_norm at this scope level so that
                # adapter modules (registered here) get correct ONNX names
                concat_hidden = op.Concat(
                    hidden_states, original_hidden_states, axis=-1
                )
                concat_hidden = self.input_layernorm(
                    op, concat_hidden
                )

                # Compute per-layer QKV adapter outputs (correct scope via
                # __call__ which pushes "qkv_adapters.N" prefix)
                q_out, k_out, v_out = self.qkv_adapters[hybrid_idx](
                    op, concat_hidden
                )

                # Shared transformer attention phase → returns pre-MLP hidden
                mlp_input, present_kv = self.shared_transformer(
                    op,
                    concat_hidden=concat_hidden,
                    attention_bias=attention_bias,
                    position_embeddings=position_embeddings,
                    past_key_value=past_kv,
                    q_adapter_out=q_out,
                    k_adapter_out=k_out,
                    v_adapter_out=v_out,
                )

                # Compute per-layer MLP adapter (correct scope via __call__)
                mlp_adapter_out = self.mlp_adapters[hybrid_idx](
                    op, mlp_input
                )

                # Apply shared MLP with per-layer adapter
                gate_adapter, up_adapter = op.Split(
                    mlp_adapter_out, num_outputs=2, axis=-1, _outputs=2
                )
                gate = op.Add(self.gate_proj(op, mlp_input), gate_adapter)
                gate = self._act_fn(op, gate)
                up = op.Add(self.up_proj(op, mlp_input), up_adapter)
                transformer_out = self.down_proj(op, op.Mul(gate, up))

                # Per-layer linear projection for injection into Mamba
                transformer_hidden_states = self.linears[hybrid_idx](
                    op, transformer_out
                )
                present_key_values.append(present_kv)
                hybrid_idx += 1
            elif ltype == "mamba2" and transformer_hidden_states is not None:
                # Injected Mamba: follows a full_attention layer
                layer = self.mamba_layers[mamba_idx]
                hidden_states, present_kv = layer(
                    op,
                    hidden_states=hidden_states,
                    transformer_hidden_states=transformer_hidden_states,
                    past_key_value=past_kv,
                )
                present_key_values.append(present_kv)
                transformer_hidden_states = None
                mamba_idx += 1
            else:
                # Pure Mamba layer
                layer = self.mamba_layers[mamba_idx]
                hidden_states, present_kv = layer(
                    op,
                    hidden_states=hidden_states,
                    attention_bias=attention_bias,
                    position_embeddings=position_embeddings,
                    past_key_value=past_kv,
                )
                present_key_values.append(present_kv)
                mamba_idx += 1

        hidden_states = self.final_layernorm(op, hidden_states)
        return hidden_states, present_key_values


class Zamba2CausalLMModel(nn.Module):
    """Zamba2 hybrid Mamba2+Attention causal language model.

    Uses ``HybridCausalLMTask`` with mixed ``"mamba2"`` and
    ``"full_attention"`` layer types for the cache.

    HuggingFace reference: ``Zamba2ForCausalLM``.
    """

    default_task: str = "hybrid-text-generation"
    category: str = "Hybrid SSM+Attention"
    config_class: type = Zamba2Config

    def __init__(self, config: Zamba2Config):
        super().__init__()
        self.config = config
        self.model = _Zamba2TextModel(config)
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
        """Map HuggingFace Zamba2ForCausalLM weights to ONNX parameters.

        New structure (shared weights + unfused adapters):
        1. Weight tying (embed_tokens ↔ lm_head)
        2. Shared transformer attn → model.shared_transformer.self_attn.*
        3. Shared norms → model.input_layernorm, model.shared_transformer.pre_ff_layernorm
        4. Shared MLP → model.gate_proj, model.up_proj, model.down_proj
        5. Per-hybrid QKV adapters → model.qkv_adapters.{idx}.*
        6. Per-hybrid MLP adapters → model.mlp_adapters.{idx}.*
        7. Per-hybrid-layer linear → model.linears.{idx}.weight
        8. Mamba layers → model.mamba_layers.{idx}.*
        9. Split fused gate_up_proj into gate_proj + up_proj
        """
        if self.config.tie_word_embeddings:
            tie_word_embeddings(state_dict)

        hybrid_indices = self.config.hybrid_layer_indices or []
        layer_types = self.config.layer_types or []

        # Build HF physical index → mamba_layers index mapping.
        # mamba_layers is a dense list of ALL mamba layers (pure + injected).
        hf_to_mamba = _build_hf_to_mamba_map(layer_types)

        # Canonical source: first hybrid physical layer for shared weights
        first_hybrid_physical = hybrid_indices[0] if hybrid_indices else -1

        new_state_dict: dict[str, torch.Tensor] = {}

        # Process all keys
        for key, value in state_dict.items():
            new_keys = _remap_weight_key(
                key,
                hf_to_mamba=hf_to_mamba,
                hybrid_indices=hybrid_indices,
                first_hybrid_physical=first_hybrid_physical,
            )
            for new_key in new_keys:
                new_state_dict[new_key] = value

        # Split fused gate_up_proj into gate_proj + up_proj for shared transformer
        keys_to_remove = []
        keys_to_add = {}
        for key, value in list(new_state_dict.items()):
            if "gate_up_proj.weight" in key:
                gate, up = value.chunk(2, dim=0)
                gate_key = key.replace("gate_up_proj", "gate_proj")
                up_key = key.replace("gate_up_proj", "up_proj")
                keys_to_add[gate_key] = gate
                keys_to_add[up_key] = up
                keys_to_remove.append(key)

        for k in keys_to_remove:
            del new_state_dict[k]
        new_state_dict.update(keys_to_add)

        return new_state_dict


# ---------------------------------------------------------------------------
# Weight mapping helpers
# ---------------------------------------------------------------------------

# Regex for HF layer keys: model.layers.<N>.<rest>
_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")

# Regex for adapter keys within shared_transformer
_ADAPTER_RE = re.compile(
    r"^self_attn\.linear_(q|k|v)_adapter_list\.(\d+)\.(0|1)\.weight$"
)
_MLP_ADAPTER_RE = re.compile(
    r"^feed_forward\.gate_up_proj_adapter_list\.(\d+)\.(0|1)\.weight$"
)


def _build_hf_to_mamba_map(layer_types: list[str]) -> dict[int, int]:
    """Build mapping from HF physical layer index to mamba_layers index.

    The mamba_layers ModuleList contains ALL mamba layers in order:
    pure mamba layers AND injected mamba layers from hybrid pairs.

    HF physical layer indices are derived from layer_types by iterating
    and grouping [full_attention, mamba2] pairs as one physical layer.

    Returns: {hf_physical_idx: mamba_layers_idx}
    """
    mapping: dict[int, int] = {}
    mamba_idx = 0
    hf_idx = 0
    i = 0
    while i < len(layer_types):
        if layer_types[i] == "full_attention":
            # Hybrid pair: full_attention + mamba2 = one HF physical layer
            # The mamba2 part is an injected mamba layer
            i += 1  # skip to the mamba2 entry
            assert i < len(layer_types) and layer_types[i] == "mamba2"
            mapping[hf_idx] = mamba_idx
            mamba_idx += 1
            i += 1
        else:
            # Pure mamba layer
            mapping[hf_idx] = mamba_idx
            mamba_idx += 1
            i += 1
        hf_idx += 1
    return mapping


def _remap_weight_key(
    key: str,
    hf_to_mamba: dict[int, int],
    hybrid_indices: list[int],
    first_hybrid_physical: int,
) -> list[str]:
    """Remap a single HF weight key to ONNX parameter name(s).

    Naming scheme:
    - Shared transformer attn: model.shared_transformer.self_attn.q_proj.weight
    - Shared transformer norms: model.shared_transformer.pre_ff_layernorm.weight
    - Shared input_layernorm: model.input_layernorm.weight
    - Shared MLP: model.gate_proj.weight, model.up_proj.weight, model.down_proj.weight
    - QKV adapters: model.qkv_adapters.{hybrid_idx}.{q|k|v}_adapter.{down|up}.weight
    - MLP adapters: model.mlp_adapters.{hybrid_idx}.mlp_adapter.{down|up}.weight
    - Linears: model.linears.{hybrid_idx}.weight
    - Mamba layers: model.mamba_layers.{mamba_idx}.mamba.* / input_layernorm.*

    Shared weights are only taken from the first hybrid layer (canonical
    source). Duplicate copies on other hybrid layers are dropped.
    """
    # Non-layer keys (model.embed_tokens, lm_head, model.final_layernorm)
    m = _LAYER_RE.match(key)
    if not m:
        return [key]

    phys_idx = int(m.group(1))
    rest = m.group(2)

    if phys_idx not in hf_to_mamba:
        return [key]

    mamba_idx = hf_to_mamba[phys_idx]

    if phys_idx in hybrid_indices:
        # Determine this hybrid's adapter index (0-based among hybrid layers)
        hybrid_idx = hybrid_indices.index(phys_idx)

        if rest.startswith("shared_transformer."):
            sub = rest[len("shared_transformer."):]

            # Check if this is a QKV adapter key
            adapter_m = _ADAPTER_RE.match(sub)
            if adapter_m:
                proj_type = adapter_m.group(1)  # q, k, or v
                adapter_idx = int(adapter_m.group(2))
                down_or_up = (
                    "down" if adapter_m.group(3) == "0" else "up"
                )
                return [
                    f"model.qkv_adapters.{adapter_idx}.{proj_type}_adapter.{down_or_up}.weight"
                ]

            # Check if this is a MLP adapter key
            mlp_adapter_m = _MLP_ADAPTER_RE.match(sub)
            if mlp_adapter_m:
                adapter_idx = int(mlp_adapter_m.group(1))
                down_or_up = (
                    "down" if mlp_adapter_m.group(2) == "0" else "up"
                )
                return [
                    f"model.mlp_adapters.{adapter_idx}.mlp_adapter.{down_or_up}.weight"
                ]

            # Shared transformer weight — only take from canonical source
            if phys_idx == first_hybrid_physical:
                # Input layernorm is at model level (called from TextModel)
                if sub.startswith("input_layernorm."):
                    return [f"model.{sub}"]
                # MLP projections are at model level
                if sub.startswith("feed_forward."):
                    mlp_sub = sub[len("feed_forward."):]
                    return [f"model.{mlp_sub}"]
                # Everything else stays under shared_transformer
                return [f"model.shared_transformer.{sub}"]
            else:
                # Drop duplicates from non-canonical hybrid layers
                return []

        elif rest.startswith("linear."):
            # Per-hybrid-layer linear projection
            sub = rest[len("linear."):]
            return [f"model.linears.{hybrid_idx}.{sub}"]

        elif rest.startswith("mamba_decoder."):
            # Mamba decoder in hybrid layer → mamba_layers
            sub = rest[len("mamba_decoder."):]
            return [f"model.mamba_layers.{mamba_idx}.{sub}"]

        else:
            # Fallback for unexpected hybrid layer keys
            return [f"model.mamba_layers.{mamba_idx}.{rest}"]
    else:
        # Pure mamba layer → mamba_layers.{mamba_idx}.*
        return [f"model.mamba_layers.{mamba_idx}.{rest}"]
