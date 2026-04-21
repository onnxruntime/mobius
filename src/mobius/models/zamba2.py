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
    MLP,
    Embedding,
    Linear,
    Mamba2Block,
    RMSNorm,
    create_attention_bias,
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
        LayerNorm → Attention → LayerNorm → MLP → Linear projection

    The output is a transformer_hidden_states tensor of shape
    (batch, seq_len, hidden_size) that will be injected into the
    subsequent Mamba layer.

    This layer produces the attention KV cache.

    Args:
        config: Zamba2 architecture config.
    """

    def __init__(self, config: Zamba2Config):
        super().__init__()
        attn_hidden = config.attention_hidden_size
        head_dim = config.head_dim

        # Attention projections grouped under self_attn for naming
        self.self_attn = _Zamba2AttentionProjections(config)

        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = head_dim
        # Zamba2 uses scaling = (head_dim / 2) ** -0.5
        self.scaling = (head_dim / 2) ** -0.5

        # Layer norms
        self.input_layernorm = RMSNorm(attn_hidden, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # MLP (gate_up_proj fused → split in forward)
        self.feed_forward = MLP(config)

        # Linear projection after MLP (projects transformer output for injection)
        self.linear = Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        original_hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple | None,
        past_key_value: tuple | None,
    ):
        """Forward pass. Returns (transformer_hidden_states, (key, value)).

        Args:
            hidden_states: Current hidden states (batch, seq, hidden_size)
            original_hidden_states: Embedding output (batch, seq, hidden_size)
            attention_bias: Causal attention mask
            position_embeddings: RoPE embeddings (unused when use_mem_rope=False)
            past_key_value: (past_key, past_value) KV cache
        """
        # Concatenate hidden_states with original embedding output
        # Result: (batch, seq, 2 * hidden_size) = (batch, seq, attention_hidden_size)
        concat_hidden = op.Concat(hidden_states, original_hidden_states, axis=-1)

        # Pre-attention layer norm on concatenated input
        concat_hidden = self.input_layernorm(op, concat_hidden)

        # Q/K/V projections from attention_hidden_size via self_attn module
        query_states, key_states, value_states = self.self_attn(op, concat_hidden)

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

        # Pre-FFN layer norm and MLP
        attn_output = self.pre_ff_layernorm(op, attn_output)
        attn_output = self.feed_forward(op, attn_output)

        # Linear projection for injection into Mamba
        transformer_hidden_states = self.linear(op, attn_output)

        return transformer_hidden_states, (present_key, present_value)


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

    Mamba2 and hybrid layers are selected based on ``layer_types``.
    Hybrid layers are expanded into (SharedTransformer, InjectedMamba) pairs.
    """

    def __init__(self, config: Zamba2Config):
        super().__init__()
        self._dtype = config.dtype
        self._use_rope = config.attention_hidden_size != 2 * config.hidden_size  # proxy
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )

        layer_types = config.layer_types or []
        self.layers = nn.ModuleList([])
        i = 0
        while i < len(layer_types):
            ltype = layer_types[i]
            if ltype == "full_attention":
                # This is the attention part of a hybrid layer;
                # next layer must be mamba2 (the injected mamba part)
                self.layers.append(Zamba2SharedTransformerLayer(config))
                i += 1
                # The mamba2 part immediately follows
                assert i < len(layer_types) and layer_types[i] == "mamba2"
                self.layers.append(Zamba2InjectedMambaLayer(config))
                i += 1
            else:
                # Pure mamba layer
                self.layers.append(Zamba2MambaDecoderLayer(config))
                i += 1

        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # RoPE only when use_mem_rope=True (default: False in Zamba2)
        self._use_rope = config.attention_hidden_size != 2 * config.hidden_size
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

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)

        # Track transformer output for injection into mamba layers
        transformer_hidden_states = None

        for layer, past_kv in zip(self.layers, past_kvs):
            if isinstance(layer, Zamba2SharedTransformerLayer):
                # Shared transformer: produces transformer_hidden_states + KV cache
                transformer_hidden_states, present_kv = layer(
                    op,
                    hidden_states=hidden_states,
                    original_hidden_states=original_hidden_states,
                    attention_bias=attention_bias,
                    position_embeddings=position_embeddings,
                    past_key_value=past_kv,
                )
                present_key_values.append(present_kv)
            elif isinstance(layer, Zamba2InjectedMambaLayer):
                # Injected Mamba: uses transformer_hidden_states from previous layer
                assert transformer_hidden_states is not None
                hidden_states, present_kv = layer(
                    op,
                    hidden_states=hidden_states,
                    transformer_hidden_states=transformer_hidden_states,
                    past_key_value=past_kv,
                )
                present_key_values.append(present_kv)
                transformer_hidden_states = None
            else:
                # Pure Mamba layer
                hidden_states, present_kv = layer(
                    op,
                    hidden_states=hidden_states,
                    attention_bias=attention_bias,
                    position_embeddings=position_embeddings,
                    past_key_value=past_kv,
                )
                present_key_values.append(present_kv)

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

        Handles:
        1. Weight tying (embed_tokens ↔ lm_head)
        2. MLP adapter fusion (gate_up_proj += adapter[1] @ adapter[0])
        3. Physical→logical layer index remapping
        4. Shared transformer weight duplication to all hybrid attention layers
        5. Fused gate_up_proj splitting into gate_proj + up_proj
        """
        if self.config.tie_word_embeddings:
            tie_word_embeddings(state_dict)

        # Fuse MLP adapters into gate_up_proj before any renaming.
        # HF forward: gate_up = gate_up_proj(x) + adapter(x)
        # Since both operate on same input, we can fuse:
        #   combined_weight = gate_up_proj.weight + adapter[1].weight @ adapter[0].weight
        _fuse_mlp_adapters(state_dict)

        hybrid_indices = self.config.hybrid_layer_indices or []
        layer_types = self.config.layer_types or []

        # Build physical→logical index mapping
        physical_to_logical = _build_physical_to_logical_map(layer_types)

        # Find the first hybrid layer index (canonical shared weights source)
        first_hybrid_physical = hybrid_indices[0] if hybrid_indices else -1

        # Collect shared transformer weights from the first hybrid layer
        shared_prefix = f"model.layers.{first_hybrid_physical}.shared_transformer."
        shared_weights: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith(shared_prefix):
                rest = key[len(shared_prefix) :]
                shared_weights[rest] = value

        new_state_dict: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_keys = _rename_zamba2_weight(
                key, physical_to_logical, hybrid_indices, layer_types
            )
            for new_key in new_keys:
                new_state_dict[new_key] = value

        # Duplicate shared transformer weights to all hybrid attention layers
        for phys_idx in hybrid_indices:
            logical_attn_idx = physical_to_logical[phys_idx][0]  # attention layer
            for rest, value in shared_weights.items():
                new_key = _map_shared_transformer_key(rest, logical_attn_idx)
                if new_key and new_key not in new_state_dict:
                    new_state_dict[new_key] = value

        # Split fused gate_up_proj into gate_proj + up_proj for all attention layers
        keys_to_remove = []
        keys_to_add = {}
        for key, value in list(new_state_dict.items()):
            if "feed_forward.gate_up_proj.weight" in key:
                # Split along dim 0: first half = gate, second half = up
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


def _fuse_mlp_adapters(state_dict: dict[str, torch.Tensor]) -> None:
    """Fuse MLP adapter weights into gate_up_proj in-place.

    HF Zamba2 MLP forward:
        gate_up = gate_up_proj(x) + adapter(x)
    where adapter is Sequential(Linear_down, Linear_up):
        adapter(x) = up_weight @ down_weight @ x

    Since both operate on the same input:
        combined = gate_up_proj.weight + up_weight @ down_weight

    After fusion, adapter keys are removed from state_dict.
    """
    import re as _re

    # Pattern: model.layers.N.shared_transformer.feed_forward.gate_up_proj_adapter_list.M.0.weight
    adapter_pattern = _re.compile(
        r"^(model\.layers\.\d+\.shared_transformer\.feed_forward\.)"
        r"gate_up_proj_adapter_list\.(\d+)\.0\.weight$"
    )
    # Group adapters by their gate_up_proj key
    adapters: dict[str, list[tuple[str, str]]] = {}  # gate_up_key -> [(down_key, up_key)]
    for key in list(state_dict.keys()):
        m = adapter_pattern.match(key)
        if m:
            prefix = m.group(1)
            idx = m.group(2)
            gate_up_key = f"{prefix}gate_up_proj.weight"
            down_key = f"{prefix}gate_up_proj_adapter_list.{idx}.0.weight"
            up_key = f"{prefix}gate_up_proj_adapter_list.{idx}.1.weight"
            if gate_up_key not in adapters:
                adapters[gate_up_key] = []
            adapters[gate_up_key].append((down_key, up_key))

    # Fuse each adapter into its corresponding gate_up_proj
    keys_to_remove = []
    for gate_up_key, adapter_pairs in adapters.items():
        if gate_up_key not in state_dict:
            continue
        gate_up_weight = state_dict[gate_up_key]
        for down_key, up_key in adapter_pairs:
            if down_key in state_dict and up_key in state_dict:
                down_w = state_dict[down_key]  # (adapter_rank, hidden_size)
                up_w = state_dict[up_key]  # (2*intermediate, adapter_rank)
                # Fuse: gate_up_proj.weight += up_w @ down_w
                gate_up_weight = gate_up_weight + up_w @ down_w
                keys_to_remove.extend([down_key, up_key])
        state_dict[gate_up_key] = gate_up_weight

    for k in keys_to_remove:
        state_dict.pop(k, None)


# Regex for HF layer keys: model.layers.<N>.<rest>
_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def _build_physical_to_logical_map(
    layer_types: list[str],
) -> dict[int, tuple[int, ...]]:
    """Build mapping from physical layer index to logical layer index(es).

    Physical layers that are hybrid expand into 2 logical layers.
    Returns a dict: physical_idx → tuple of logical indices.
    """
    mapping: dict[int, tuple[int, ...]] = {}
    logical_idx = 0
    physical_idx = 0
    i = 0
    while i < len(layer_types):
        if layer_types[i] == "full_attention":
            # This is the attention part of a hybrid; next is mamba2
            mapping[physical_idx] = (logical_idx, logical_idx + 1)
            logical_idx += 2
            i += 2  # skip the mamba2 entry
        else:
            mapping[physical_idx] = (logical_idx,)
            logical_idx += 1
            i += 1
        physical_idx += 1
    return mapping


def _rename_zamba2_weight(
    key: str,
    physical_to_logical: dict[int, tuple[int, ...]],
    hybrid_indices: list[int],
    layer_types: list[str],
) -> list[str]:
    """Rename a single HF weight key to ONNX parameter name(s).

    Returns a list of new keys (usually 1, but shared weights may map to multiple).
    """
    # Non-layer keys pass through unchanged
    m = _LAYER_RE.match(key)
    if not m:
        return [key]

    phys_idx = int(m.group(1))
    rest = m.group(2)

    if phys_idx not in physical_to_logical:
        return [key]

    logical_indices = physical_to_logical[phys_idx]

    if phys_idx in hybrid_indices:
        # Hybrid layer: split into attention (logical[0]) and mamba (logical[1])
        logical_attn_idx = logical_indices[0]
        logical_mamba_idx = logical_indices[1]

        if rest.startswith("shared_transformer."):
            # Shared transformer → attention logical layer
            sub = rest[len("shared_transformer.") :]
            new_key = _map_shared_transformer_key(sub, logical_attn_idx)
            return [new_key] if new_key else []
        elif rest.startswith("linear."):
            # Linear projection → part of the attention logical layer
            sub = rest[len("linear.") :]
            return [f"model.layers.{logical_attn_idx}.linear.{sub}"]
        elif rest.startswith("mamba_decoder."):
            # Mamba decoder → mamba logical layer
            sub = rest[len("mamba_decoder.") :]
            return [f"model.layers.{logical_mamba_idx}.{sub}"]
        else:
            # Fallback
            return [f"model.layers.{logical_attn_idx}.{rest}"]
    else:
        # Pure mamba layer
        logical_idx = logical_indices[0]
        return [f"model.layers.{logical_idx}.{rest}"]


def _map_shared_transformer_key(rest: str, logical_idx: int) -> str | None:
    """Map a shared_transformer sub-key to the ONNX attention layer.

    HF: shared_transformer.self_attn.{q,k,v,o}_proj.weight
    HF: shared_transformer.feed_forward.gate_up_proj.weight
    HF: shared_transformer.feed_forward.down_proj.weight
    HF: shared_transformer.input_layernorm.weight
    HF: shared_transformer.pre_ff_layernorm.weight

    ONNX: model.layers.{idx}.self_attn.{q,k,v,o}_proj.weight
    ONNX: model.layers.{idx}.feed_forward.{gate_proj,up_proj,down_proj}.weight
    ONNX: model.layers.{idx}.input_layernorm.weight
    ONNX: model.layers.{idx}.pre_ff_layernorm.weight
    """
    # All sub-keys map directly (self_attn.*, feed_forward.*, norms)
    return f"model.layers.{logical_idx}.{rest}"
