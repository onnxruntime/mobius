# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""RecurrentGemma (Griffin) causal language model.

RecurrentGemma uses a hybrid architecture: layers alternate between
Real-Gated Linear Recurrent Units (RG-LRU) and local sliding-window
attention (Griffin paper, Google DeepMind 2024).

Each decoder layer is one of:
    "recurrent" — RG-LRU with depthwise conv prefix + SwiGLU MLP
    "attention"  — Local sliding-window GQA with partial RoPE + MLP

Single-token decode carries per-layer state:
    Recurrent layers: (conv_state, rg_lru_state)
        conv_state:    (batch, lru_width, conv1d_width - 1) FLOAT
        rg_lru_state:  (batch, lru_width)                   FLOAT (fp32)
    Attention layers: (key_cache, value_cache)
        key_cache:   (batch, num_kv_heads, past_len, head_dim) FLOAT
        value_cache: (batch, num_kv_heads, past_len, head_dim) FLOAT

HuggingFace model type: ``recurrent_gemma``
Reference: ``google/recurrentgemma-2b``, ``google/recurrentgemma-9b``
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import RecurrentGemmaConfig
from mobius.components import (
    MLP,
    Attention,
    Embedding,
    Linear,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
)

# ---------------------------------------------------------------------------
# Scaled embedding (sqrt(hidden_size) multiplier, same as Gemma)
# ---------------------------------------------------------------------------


class _ScaledEmbedding(Embedding):
    """Token embedding scaled by ``sqrt(hidden_size)`` per RecurrentGemma normalizer."""

    def __init__(
        self, num_embeddings: int, embedding_dim: int, padding_idx: int, scale: float
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self._scale = scale

    def forward(self, op: builder.OpBuilder, input_ids: ir.Value) -> ir.Value:
        embeddings = super().forward(op, input_ids)
        return op.Mul(embeddings, self._scale)


# ---------------------------------------------------------------------------
# RG-LRU: Real-Gated Linear Recurrent Unit
# ---------------------------------------------------------------------------


class _RgLru(nn.Module):
    """Real-Gated Linear Recurrent Unit (single-token decode).

    Each head computes an input gate and a recurrent gate independently.
    The recurrent gate controls the decay rate A of the state transition.

    State: (batch, lru_width) in fp32 for numerical stability.

    HuggingFace reference: ``RecurrentGemmaRglru``.
    """

    def __init__(self, config: RecurrentGemmaConfig):
        super().__init__()
        num_heads = config.num_attention_heads
        lru_width = config.lru_width
        block_width = lru_width // num_heads

        self._num_heads = num_heads
        self._lru_width = lru_width
        self._block_width = block_width

        # Scalar decay parameter per channel (trained, then softplus'd)
        self.recurrent_param = nn.Parameter([lru_width])

        # Per-head input gate: (num_heads, block_width, block_width)
        self.input_gate_weight = nn.Parameter([num_heads, block_width, block_width])
        self.input_gate_bias = nn.Parameter([num_heads, block_width])

        # Per-head recurrent gate: (num_heads, block_width, block_width)
        self.recurrent_gate_weight = nn.Parameter([num_heads, block_width, block_width])
        self.recurrent_gate_bias = nn.Parameter([num_heads, block_width])

    def _head_gate(
        self,
        op: builder.OpBuilder,
        x_4d: ir.Value,  # (B, num_heads, 1, block_width)
        weight: ir.Value,  # (num_heads, block_width, block_width)
        bias: ir.Value,  # (num_heads, block_width)
    ) -> ir.Value:
        """Compute per-head linear gate → sigmoid → flatten to (B, lru_width)."""
        # Broadcast weight: (1, num_heads, block_width, block_width)
        w_4d = op.Unsqueeze(weight, [0])
        # BatchMatMul: (B, num_heads, 1, block_width)
        result = op.MatMul(x_4d, w_4d)
        # Squeeze seq dim → (B, num_heads, block_width)
        result = op.Squeeze(result, [2])
        # Add per-head bias
        result = op.Add(result, bias)
        # Reshape to flat: (B, lru_width)
        result = op.Reshape(result, op.Constant(value_ints=[0, self._lru_width]))
        return op.Sigmoid(result)

    def forward(
        self,
        op: builder.OpBuilder,
        x: ir.Value,  # (B, 1, lru_width) — single token after conv
        rg_lru_state: ir.Value,  # (B, lru_width) FLOAT fp32
    ) -> tuple[ir.Value, ir.Value]:
        """Single-token RG-LRU step.

        Returns:
            output: (B, 1, lru_width) — updated activations.
            new_state: (B, lru_width) FLOAT fp32 — updated carry state.
        """
        # Squeeze seq dim for gate computation: (B, lru_width)
        x_2d = op.Squeeze(x, [1])

        # Reshape to per-head layout: (B, num_heads, block_width)
        x_heads = op.Reshape(
            x_2d, op.Constant(value_ints=[0, self._num_heads, self._block_width])
        )
        # Add seq dim for batched matmul: (B, num_heads, 1, block_width)
        x_4d = op.Unsqueeze(x_heads, [2])

        # Compute gates (both → sigmoid → (B, lru_width))
        input_gate = self._head_gate(op, x_4d, self.input_gate_weight, self.input_gate_bias)
        recurrent_gate = self._head_gate(
            op, x_4d, self.recurrent_gate_weight, self.recurrent_gate_bias
        )

        # Compute decay A = exp(-8 * recurrent_gate * softplus(recurrent_param))
        # Compute in fp32 for numerical stability (exp of negative values)
        rp_f32 = op.Softplus(op.Cast(self.recurrent_param, to=ir.DataType.FLOAT))
        rg_f32 = op.Cast(recurrent_gate, to=ir.DataType.FLOAT)
        log_a = op.Neg(
            op.Mul(op.Mul(op.Constant(value_float=8.0), rg_f32), rp_f32)
        )  # log(A) = -8 * recurrent_gate * softplus(recurrent_param)
        a = op.Exp(log_a)  # A: (B, lru_width) fp32
        a_sq = op.Exp(op.Mul(log_a, op.Constant(value_float=2.0)))  # A^2

        # Normalization multiplier: sqrt(1 - A^2)
        one = op.Constant(value_float=1.0)
        multiplier = op.Sqrt(op.Sub(one, a_sq))  # (B, lru_width) fp32

        # Gated + normalized input: input_gate * x * sqrt(1 - A^2)
        x_f32 = op.Cast(x_2d, to=ir.DataType.FLOAT)
        ig_f32 = op.Cast(input_gate, to=ir.DataType.FLOAT)
        normalized_x = op.Mul(op.Mul(ig_f32, x_f32), multiplier)

        # State update: h_t = A * h_{t-1} + normalized_x (all in fp32)
        rg_lru_f32 = op.Cast(rg_lru_state, to=ir.DataType.FLOAT)
        new_state = op.Add(op.Mul(a, rg_lru_f32), normalized_x)  # (B, lru_width) fp32

        # Cast output back to input dtype and restore seq dim: (B, 1, lru_width)
        output = op.CastLike(op.Unsqueeze(new_state, [1]), x)

        return output, new_state


# ---------------------------------------------------------------------------
# Recurrent block
# ---------------------------------------------------------------------------


class _RecurrentBlock(nn.Module):
    """Griffin recurrent block: Conv1d + RG-LRU + gated linear output.

    Structure:
        y = gelu(linear_y(x))       # gate branch
        x = linear_x(x)             # recurrent branch
        x = conv1d_state_update(x)  # causal prefix conv
        x = rg_lru(x)               # linear recurrence
        out = linear_out(x * y)     # gated output

    Carry state:
        conv_state:   (batch, lru_width, conv1d_width - 1) FLOAT
        rg_lru_state: (batch, lru_width)                   FLOAT (fp32)

    HuggingFace reference: ``RecurrentGemmaRecurrentBlock``.
    """

    def __init__(self, config: RecurrentGemmaConfig):
        super().__init__()
        hidden_size = config.hidden_size
        lru_width = config.lru_width
        self._lru_width = lru_width
        self._conv1d_width = config.conv1d_width
        self._hidden_act = config.hidden_act or "gelu_pytorch_tanh"

        self.linear_y = Linear(hidden_size, lru_width, bias=True)
        self.linear_x = Linear(hidden_size, lru_width, bias=True)
        self.linear_out = Linear(lru_width, hidden_size, bias=True)

        # Depthwise Conv1d weights: (lru_width, conv1d_width) after HF squeeze
        # HF stores as (lru_width, 1, conv1d_width); we squeeze in preprocess_weights
        self.conv_1d_weight = nn.Parameter([lru_width, config.conv1d_width])
        self.conv_1d_bias = nn.Parameter([lru_width])

        self.rg_lru = _RgLru(config)

    def _gelu_tanh(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        """GELU with tanh approximation (gelu_pytorch_tanh)."""
        return op.Gelu(x)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,  # (B, 1, hidden_size)
        conv_state: ir.Value,  # (B, lru_width, conv1d_width - 1)
        rg_lru_state: ir.Value,  # (B, lru_width) fp32
    ) -> tuple[ir.Value, ir.Value, ir.Value]:
        # Gate branch: GELU(linear_y(x))
        y = self.linear_y(op, hidden_states)  # (B, 1, lru_width)
        y = op.Gelu(y)

        # Recurrent branch: linear_x(x) → (B, 1, lru_width) → transpose
        x = self.linear_x(op, hidden_states)  # (B, 1, lru_width)
        x_t = op.Transpose(x, perm=[0, 2, 1])  # (B, lru_width, 1)

        # --- Causal conv1d state update ---
        # Concatenate conv state with new token: (B, lru_width, conv1d_width)
        conv_input = op.Concat(conv_state, x_t, axis=2)
        # New state: drop oldest token → (B, lru_width, conv1d_width - 1)
        new_conv_state = op.Slice(
            conv_input,
            op.Constant(value_ints=[1]),
            op.Constant(value_ints=[2**31 - 1]),
            op.Constant(value_ints=[2]),  # axis=2
        )
        # Depthwise conv output: sum(conv_input * weight, axis=-1) + bias
        # weight: (lru_width, conv1d_width), conv_input: (B, lru_width, conv1d_width)
        w_expanded = op.Unsqueeze(self.conv_1d_weight, [0])  # (1, lru_width, conv1d_width)
        conv_out = op.ReduceSum(
            op.Mul(conv_input, w_expanded), [-1], keepdims=False
        )  # (B, lru_width)
        conv_out = op.Add(conv_out, self.conv_1d_bias)  # + (lru_width,)
        # Transpose to (B, 1, lru_width) for RG-LRU
        x_for_lru = op.Unsqueeze(conv_out, [1])  # (B, 1, lru_width)

        # RG-LRU state update
        x_out, new_rg_lru_state = self.rg_lru(op, x_for_lru, rg_lru_state)
        # x_out: (B, 1, lru_width)

        # Gated output: linear_out(x * y)
        output = self.linear_out(op, op.Mul(x_out, y))  # (B, 1, hidden_size)

        return output, new_conv_state, new_rg_lru_state


# ---------------------------------------------------------------------------
# Attention block (sliding-window GQA with partial RoPE)
# ---------------------------------------------------------------------------


class _AttentionBlock(nn.Module):
    """Local sliding-window GQA attention block.

    Uses the standard Attention component with partial RoPE.
    HuggingFace reference: ``RecurrentGemmaSdpaAttention``.
    """

    def __init__(self, config: RecurrentGemmaConfig):
        super().__init__()
        # Named "attention" so initializer paths contain the word "attention"
        # (e.g. temporal_block.attention.q_proj.weight)
        self.attention = Attention(config)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple,
        past_key_value: tuple | None = None,
    ) -> tuple[ir.Value, tuple[ir.Value, ir.Value]]:
        # Attention.forward returns (output, (key, value))
        output, present_kv = self.attention(
            op,
            hidden_states,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
        )
        return output, present_kv


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


class _RecurrentGemmaDecoderLayer(nn.Module):
    """One RecurrentGemma decoder layer.

    Structure: temporal_pre_norm → temporal_block → residual →
               channel_pre_norm → mlp_block → residual

    HuggingFace reference: ``RecurrentGemmaDecoderLayer``.
    """

    def __init__(self, config: RecurrentGemmaConfig, block_type: str):
        super().__init__()
        self._block_type = block_type

        self.temporal_pre_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if block_type == "recurrent":
            self.temporal_block = _RecurrentBlock(config)
        else:  # "attention"
            self.temporal_block = _AttentionBlock(config)
        self.channel_pre_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp_block = MLP(config)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple,
        past_state: tuple | None,
    ) -> tuple[ir.Value, tuple]:
        # --- Temporal block (recurrent or attention) ---
        residual = hidden_states
        normed = self.temporal_pre_norm(op, hidden_states)

        if self._block_type == "recurrent":
            conv_state, rg_lru_state = past_state if past_state is not None else (None, None)
            temporal_out, new_conv_state, new_rg_lru_state = self.temporal_block(
                op, normed, conv_state, rg_lru_state
            )
            present_state = (new_conv_state, new_rg_lru_state)
        else:
            temporal_out, present_state = self.temporal_block(
                op, normed, attention_bias, position_embeddings, past_state
            )

        hidden_states = op.Add(residual, temporal_out)

        # --- MLP block ---
        residual = hidden_states
        normed = self.channel_pre_norm(op, hidden_states)
        mlp_out = self.mlp_block(op, normed)
        hidden_states = op.Add(residual, mlp_out)

        return hidden_states, present_state


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class RecurrentGemmaCausalLMModel(nn.Module):
    """RecurrentGemma (Griffin) causal language model.

    Hybrid model alternating Real-Gated LRU recurrent blocks and local
    sliding-window attention blocks (2 recurrent : 1 attention by default).
    Embedding is scaled by ``sqrt(hidden_size)`` à la Gemma.
    Logits are soft-capped: ``tanh(logits / cap) * cap``.

    Uses ``hybrid-text-generation`` task with mixed per-layer state.

    HuggingFace model type: ``recurrent_gemma``
    """

    default_task: str = "hybrid-text-generation"
    category: str = "Text Generation"
    config_class: type = RecurrentGemmaConfig

    def __init__(self, config: RecurrentGemmaConfig):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        self._logits_soft_cap = config.logits_soft_cap

        # Embedding scaled by sqrt(hidden_size) — matches RecurrentGemma normalizer
        embed_scale = float(np.round(np.sqrt(hidden_size), decimals=2))
        self.model = _RecurrentGemmaModel(config, embed_scale)
        self.lm_head = Linear(hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ) -> tuple[ir.Value, list]:
        hidden_states, present_key_values = self.model(
            op, input_ids, attention_mask, position_ids, past_key_values
        )
        logits = self.lm_head(op, hidden_states)

        # Soft cap: tanh(logits / cap) * cap
        if self._logits_soft_cap and self._logits_soft_cap > 0:
            cap = op.Constant(value_float=float(self._logits_soft_cap))
            logits = op.Mul(op.Tanh(op.Div(logits, cap)), cap)

        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HuggingFace RecurrentGemma weights to our naming convention.

        Most weights align directly. The only difference is:
          - conv_1d.weight: (lru_width, 1, conv1d_width) → (lru_width, conv1d_width)
          - Attention weights are on temporal_block directly (HF) vs
            temporal_block.temporal_block (ours, since _AttentionBlock wraps Attention).
        """
        result = {}
        for name, tensor in state_dict.items():
            # Squeeze conv1d weight: (L, 1, K) → (L, K)
            if name.endswith(".conv_1d.weight"):
                tensor = tensor.squeeze(1)
                name = name.replace(".conv_1d.weight", ".conv_1d_weight")
            elif name.endswith(".conv_1d.bias"):
                name = name.replace(".conv_1d.bias", ".conv_1d_bias")
            # Attention block: temporal_block.q_proj → temporal_block.temporal_block.q_proj
            elif _is_attention_weight(name):
                name = _remap_attention_weight(name)
            result[name] = tensor

        return result


# ---------------------------------------------------------------------------
# Internal model backbone
# ---------------------------------------------------------------------------


class _RecurrentGemmaModel(nn.Module):
    """RecurrentGemma backbone: embed → N layers → final norm."""

    def __init__(self, config: RecurrentGemmaConfig, embed_scale: float):
        super().__init__()
        # Token embeddings scaled by sqrt(hidden_size) — weight tied with lm_head
        self.embed_tokens = _ScaledEmbedding(
            config.vocab_size, config.hidden_size, config.pad_token_id, embed_scale
        )
        block_types = config.layer_types or []
        self.layers = nn.ModuleList(
            [
                _RecurrentGemmaDecoderLayer(
                    config,
                    # config.layer_types uses "full_attention"; map back to "attention"
                    "attention"
                    if (block_types[i] if i < len(block_types) else "") == "full_attention"
                    else "recurrent",
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ) -> tuple[ir.Value, list]:
        hidden_states = self.embed_tokens(op, input_ids)

        position_embeddings = self.rotary_emb(op, position_ids)
        attention_bias = create_attention_bias(
            op, input_ids=input_ids, attention_mask=attention_mask
        )

        present_key_values = []
        past = past_key_values or [None] * len(self.layers)
        for layer, past_state in zip(self.layers, past):
            hidden_states, present_state = layer(
                op, hidden_states, attention_bias, position_embeddings, past_state
            )
            present_key_values.append(present_state)

        hidden_states = self.final_norm(op, hidden_states)
        return hidden_states, present_key_values


# ---------------------------------------------------------------------------
# Weight name helpers
# ---------------------------------------------------------------------------

_ATTENTION_SUFFIXES = (
    ".q_proj.",
    ".k_proj.",
    ".v_proj.",
    ".o_proj.",
    ".rotary_emb.",
)


def _is_attention_weight(name: str) -> bool:
    """True if ``name`` is an attention-layer weight in an attention block."""
    # Match: model.layers.N.temporal_block.<attn_suffix>
    if ".temporal_block." not in name:
        return False
    rest = name.split(".temporal_block.", 1)[1]
    return any(rest.startswith(s.lstrip(".")) for s in _ATTENTION_SUFFIXES)


def _remap_attention_weight(name: str) -> str:
    """Remap attention weights: temporal_block.X → temporal_block.attention.X.

    Our _AttentionBlock wraps Attention under the attribute 'attention',
    so the full path becomes: model.layers.N.temporal_block.attention.q_proj.*
    """
    return name.replace(".temporal_block.", ".temporal_block.attention.", 1)
