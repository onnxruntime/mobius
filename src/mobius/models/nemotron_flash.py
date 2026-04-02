# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""NemotronFlash hybrid GLA+Mamba2+Attention+FFN causal language model.

NemotronFlash interleaves four layer types in a configurable pattern:
- GLA (Gated Linear Attention / DeltaNet) layers for efficient linear recurrence
- Mamba2/SSD layers for efficient recurrent processing
- Standard GQA attention layers for global context
- Dense FFN-only layers for feedforward computation

Each layer is a single-mixer block: norm → mixer → residual.

Additionally, NemotronFlash prepends ``num_memory_tokens`` learnable memory
tokens to every forward call, extending the effective sequence length.  These
tokens are sliced off after the transformer layers so only the original token
hidden states are returned.

Layer types (HF → mobius canonical):
    ``"deltanet"`` → ``"deltanet"``  (treated as full KV-cache in task)
    ``"m2"`` → ``"mamba2"``
    ``"a"`` → ``"full_attention"``
    ``"f"`` → ``"mlp"``

Weight naming differences from NemotronH:
    - Pre-norm is ``input_layernorm`` (not ``norm``)
    - Final norm is ``final_layernorm`` (not ``norm``)
    - GLA layers use ``gla.*`` weight prefix (not ``self_attn.*``)
    - FFN layers use ``ffn.*`` prefix and ``pre_ffn_layernorm`` (not ``norm``)
    - Memory tokens: ``model.memory_tokens`` parameter

HuggingFace reference: ``NemotronFlashForCausalLM``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import NemotronFlashConfig
from mobius._weight_utils import tie_word_embeddings
from mobius.components import (
    Attention,
    Embedding,
    Linear,
    MLP,
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


class NemotronFlashGLALayer(nn.Module):
    """GLA (Gated Linear Attention / DeltaNet) layer.

    Weight prefix: ``gla.`` (q_proj, k_proj, v_proj, o_proj).
    For graph construction, uses standard GQA Attention as a placeholder.
    The actual GLA recurrence will be implemented for parity later.

    Layer type: ``"deltanet"`` — treated as full KV cache in the task.

    Args:
        config: NemotronFlash architecture config.
    """

    def __init__(self, config: NemotronFlashConfig):
        super().__init__()
        # 'gla' attribute → weight paths: gla.q_proj.weight, gla.k_proj.weight, etc.
        self.gla = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None,
    ):
        """Forward pass. Returns (hidden_states, (key, value)).

        Uses GQA attention as a placeholder for the GLA computation.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)

        attn_output, present_kv = self.gla(
            op,
            hidden_states=hidden_states,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
        )
        hidden_states = op.Add(residual, attn_output)

        return hidden_states, present_kv


class NemotronFlashFFNLayer(nn.Module):
    """FFN-only layer (HF type ``"f"``).

    Weight prefix: ``ffn.`` (gate_proj, up_proj, down_proj).
    Pre-norm attribute: ``pre_ffn_layernorm``.
    Stateless — returns ``(None, None)`` as cache.

    Args:
        config: NemotronFlash architecture config.
    """

    def __init__(self, config: NemotronFlashConfig):
        super().__init__()
        # 'mlp' attribute → weight paths: mlp.gate_proj.weight, etc.
        # HF uses 'ffn.*' prefix; preprocess_weights renames ffn.* → mlp.*
        self.mlp = MLP(config)
        # 'pre_ffn_layernorm' matches HF weight name directly
        self.pre_ffn_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None,
    ):
        """Forward pass. Returns (hidden_states, (None, None)).

        FFN layers are stateless — the None pair keeps the cache list aligned.
        """
        del attention_bias, position_embeddings, past_key_value  # unused

        residual = hidden_states
        hidden_states = self.pre_ffn_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, (None, None)


class NemotronFlashMambaLayer(nn.Module):
    """Mamba2 layer (HF type ``"m2"``).

    Pre-norm attribute: ``input_layernorm`` (unlike NemotronH which uses ``norm``).
    Returns ``(conv_state, ssm_state)`` matching the mamba2 cache type.

    Args:
        config: NemotronFlash architecture config.
    """

    def __init__(self, config: NemotronFlashConfig):
        super().__init__()
        # d_inner = num_heads * head_dim
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
            # NemotronFlash uses grouped RMSNorm: normalize within each group
            norm_group_size=d_inner // config.mamba_n_groups,
        )
        # 'input_layernorm' matches HF weight name (NemotronH uses 'norm')
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
        but accepted for a uniform interface with attention layers.
        """
        del attention_bias, position_embeddings  # unused

        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)

        conv_state, ssm_state = past_key_value if past_key_value is not None else (None, None)
        mamba_out, new_conv_state, new_ssm_state = self.mamba(
            op, hidden_states, conv_state, ssm_state
        )
        hidden_states = op.Add(residual, mamba_out)

        return hidden_states, (new_conv_state, new_ssm_state)


class NemotronFlashAttentionLayer(nn.Module):
    """Standard GQA attention layer (HF type ``"a"``).

    Pre-norm attribute: ``input_layernorm`` (unlike NemotronH which uses ``norm``).

    Args:
        config: NemotronFlash architecture config.
    """

    def __init__(self, config: NemotronFlashConfig):
        super().__init__()
        self.self_attn = Attention(config)
        # 'input_layernorm' matches HF weight name (NemotronH uses 'norm')
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        hidden_states = self.input_layernorm(op, hidden_states)

        attn_output, present_kv = self.self_attn(
            op,
            hidden_states=hidden_states,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
        )
        hidden_states = op.Add(residual, attn_output)

        return hidden_states, present_kv


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class _NemotronFlashTextModel(nn.Module):
    """NemotronFlash text backbone with memory tokens.

    Prepends ``num_memory_tokens`` learnable memory tokens to the input
    embeddings, runs ``N x (GLA | Mamba2 | Attention | FFN)`` layers,
    then slices the memory tokens off before the final norm.

    Layer types are selected based on ``config.layer_types``:
        ``"deltanet"`` → NemotronFlashGLALayer
        ``"mamba2"`` → NemotronFlashMambaLayer
        ``"mlp"`` → NemotronFlashFFNLayer
        ``"full_attention"`` → NemotronFlashAttentionLayer
    """

    def __init__(self, config: NemotronFlashConfig):
        super().__init__()
        self._config = config
        self._dtype = config.dtype

        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )

        # Learnable memory tokens prepended to input embeddings every forward call
        if config.num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(
                [config.num_memory_tokens, config.hidden_size]
            )

        layer_types = config.layer_types or []
        self.layers = nn.ModuleList([])
        for i in range(config.num_hidden_layers):
            ltype = layer_types[i] if i < len(layer_types) else "full_attention"
            if ltype == "deltanet":
                self.layers.append(NemotronFlashGLALayer(config))
            elif ltype == "mamba2":
                self.layers.append(NemotronFlashMambaLayer(config))
            elif ltype == "mlp":
                self.layers.append(NemotronFlashFFNLayer(config))
            else:  # "full_attention"
                self.layers.append(NemotronFlashAttentionLayer(config))

        # 'final_layernorm' matches HF weight name (NemotronH uses 'norm')
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states = self.embed_tokens(op, input_ids)  # (batch, seq, H)

        num_mem = self._config.num_memory_tokens
        if num_mem > 0:
            hidden_states, position_ids, attention_mask = self._prepend_memory_tokens(
                op, hidden_states, position_ids, attention_mask, input_ids
            )
            # Use extended input to drive attention bias shape
            attention_input = self._make_extended_input_ids(op, input_ids)
        else:
            attention_input = input_ids

        position_embeddings = self.rotary_emb(op, position_ids)
        attention_bias = create_attention_bias(
            op,
            input_ids=attention_input,
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

        if num_mem > 0:
            # Slice off the leading memory token outputs; keep only the original seq
            # Slice(data, starts, ends, axes): axis=1, start=num_mem, end=max_int
            hidden_states = op.Slice(
                hidden_states,
                op.Constant(value_ints=[num_mem]),  # starts
                op.Constant(value_ints=[2147483647]),  # ends (INT_MAX)
                op.Constant(value_ints=[1]),  # axes = [1]
            )

        hidden_states = self.final_layernorm(op, hidden_states)
        return hidden_states, present_key_values

    def _prepend_memory_tokens(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        position_ids: ir.Value,
        attention_mask: ir.Value,
        input_ids: ir.Value,
    ) -> tuple[ir.Value, ir.Value, ir.Value]:
        """Prepend memory tokens to hidden_states, and extend position_ids / attention_mask.

        Returns:
            (extended_hidden_states, extended_position_ids, extended_attention_mask)
        """
        num_mem = self._config.num_memory_tokens
        batch_size = op.Shape(hidden_states, start=0, end=1)  # 1D tensor [B]
        num_mem_tensor = op.Constant(value_ints=[num_mem])  # 1D tensor [num_mem]

        # --- Expand memory tokens: (num_mem, H) → (batch, num_mem, H) ---
        mem = op.Unsqueeze(self.memory_tokens, [0])  # (1, num_mem, H)
        expand_shape = op.Concat(
            batch_size,
            op.Constant(value_ints=[num_mem, self._config.hidden_size]),
            axis=0,
        )  # 1D tensor [B, num_mem, H]
        mem = op.Expand(mem, expand_shape)  # (batch, num_mem, H)
        hidden_states = op.Concat(mem, hidden_states, axis=1)  # (batch, num_mem+seq, H)

        # --- Extend position IDs: prepend [0, 1, ..., num_mem-1] ---
        mem_pos = op.Unsqueeze(
            op.Range(
                op.Constant(value_int=0),
                op.Constant(value_int=num_mem),
                op.Constant(value_int=1),
            ),
            [0],
        )  # (1, num_mem) - int64
        mem_pos = op.Expand(
            mem_pos, op.Concat(batch_size, num_mem_tensor, axis=0)
        )  # (batch, num_mem)
        mem_pos = op.CastLike(mem_pos, position_ids)
        position_ids = op.Concat(mem_pos, position_ids, axis=1)  # (batch, num_mem+seq)

        # --- Extend attention mask: prepend ones for memory token positions ---
        mem_mask = op.Expand(
            op.Constant(value_ints=[1]),  # scalar-like 1D [1]
            op.Concat(batch_size, num_mem_tensor, axis=0),  # [B, num_mem]
        )  # (batch, num_mem)
        mem_mask = op.CastLike(mem_mask, attention_mask)
        attention_mask = op.Concat(mem_mask, attention_mask, axis=1)

        return hidden_states, position_ids, attention_mask

    def _make_extended_input_ids(
        self, op: builder.OpBuilder, input_ids: ir.Value
    ) -> ir.Value:
        """Create a fake input_ids of shape (batch, num_mem+seq) for attention bias shape.

        The values don't matter; only the shape is used by create_attention_bias.
        """
        num_mem = self._config.num_memory_tokens
        batch_size = op.Shape(input_ids, start=0, end=1)  # [B]
        seq_len = op.Shape(input_ids, start=1, end=2)  # [S]
        extended_seq = op.Add(seq_len, op.Constant(value_ints=[num_mem]))  # [num_mem+S]
        # Expand shape-[1] zero tensor to (batch, num_mem+seq)
        return op.Expand(
            op.Constant(value_ints=[0]),
            op.Concat(batch_size, extended_seq, axis=0),
        )


class NemotronFlashCausalLMModel(nn.Module):
    """NemotronFlash hybrid GLA+Mamba2+Attention+FFN causal language model.

    Interleaves four layer types:
    - GLA (Gated Linear Attention / DeltaNet) — linear recurrence with KV cache
    - Mamba2/SSD — efficient recurrent SSM layers
    - Standard GQA attention — global context attention layers
    - Dense FFN — stateless feedforward layers

    Prepends ``num_memory_tokens`` learnable memory tokens to every forward call.

    Uses ``HybridCausalLMTask`` with mixed layer types for cache management.

    HuggingFace reference: ``NemotronFlashForCausalLM`` (trust_remote_code).
    """

    default_task: str = "hybrid-text-generation"
    category: str = "Hybrid SSM+Attention"
    config_class: type = NemotronFlashConfig

    def __init__(self, config: NemotronFlashConfig):
        super().__init__()
        self.config = config
        self.model = _NemotronFlashTextModel(config)
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
        """Map HuggingFace NemotronFlashForCausalLM weights to ONNX parameters.

        HF weight names already match the ONNX module hierarchy because
        attribute names were chosen to align with HF naming:
          - model.embed_tokens.weight  ✓
          - model.memory_tokens        ✓
          - model.layers.N.gla.*       ✓ (deltanet layers)
          - model.layers.N.self_attn.* ✓ (attention layers)
          - model.layers.N.mamba.*     ✓ (mamba2 layers)
          - model.layers.N.ffn.*       → model.layers.N.mlp.* (rename: HF uses 'ffn', ONNX uses 'mlp')
          - model.layers.N.input_layernorm.weight  ✓
          - model.layers.N.pre_ffn_layernorm.weight ✓
          - model.final_layernorm.weight  ✓
          - lm_head.weight             ✓

        Handles:
        - ``model.layers.N.ffn.*`` → ``model.layers.N.mlp.*``: ONNX uses the standard
          ``mlp`` attribute name for FFN-only layers; HF uses ``ffn.*`` prefix.
        - Weight tying (embed_tokens ↔ lm_head).
        """
        # Rename HF 'ffn.*' to ONNX 'mlp.*' for FFN-only layers
        state_dict = {key.replace(".ffn.", ".mlp."): value for key, value in state_dict.items()}
        if self.config.tie_word_embeddings:
            tie_word_embeddings(state_dict)
        return state_dict
