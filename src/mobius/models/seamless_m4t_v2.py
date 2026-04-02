# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""SeamlessM4T v2 text-to-text encoder-decoder model.

Architecture:
    encoder: _SeamlessM4Tv2TextEncoder
        Scaled word embedding (x sqrt(hidden_size)) + sinusoidal positional
        embeddings + N x EncoderBlock (pre-norm, ReLU FFN) + LayerNorm
    decoder: _SeamlessM4Tv2TextDecoder
        Scaled word embedding + sinusoidal positional embeddings + N x
        DecoderBlock (self-attn + cross-attn + FFN, all pre-norm) +
        LayerNorm + lm_head (tied to encoder embed_tokens)

Differs from BART in:
  - Embeddings are scaled by sqrt(hidden_size) before adding positions
  - No layernorm_embedding before the transformer blocks
  - A final layer_norm is applied after all blocks
  - Positional embeddings are sinusoidal (pre-computed buffer), not learned
  - Separate FFN dims per encoder/decoder (encoder_ffn_dim, decoder_ffn_dim)
  - ReLU activation (BART defaults to GELU)

HuggingFace reference: SeamlessM4Tv2ForTextToText (model_type='seamless_m4t_v2')

Weight prefixes:
    model.shared.weight                          → shared embedding
    model.text_encoder.*                         → encoder.*
    model.text_decoder.*                         → decoder.*
    model.text_encoder.embed_positions.weights   → encoder.embed_positions.weight
    model.text_decoder.embed_positions.weights   → decoder.embed_positions.weight
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import SeamlessM4Tv2Config
from mobius.components._activations import ACT2FN
from mobius.components._common import Embedding, LayerNorm, Linear
from mobius.components._encoder_decoder_attention import EncoderDecoderAttention

if TYPE_CHECKING:
    import onnx_ir as ir


# ---------------------------------------------------------------------------
# Encoder and Decoder Blocks
# ---------------------------------------------------------------------------


class _SeamlessM4Tv2EncoderBlock(nn.Module):
    """Pre-norm encoder block using encoder_ffn_dim for the feed-forward layer.

    Matches HuggingFace SeamlessM4Tv2EncoderLayer: LayerNorm is applied BEFORE
    attention/FFN (pre-norm), and the residual is added AFTER.  Weight names
    follow HF exactly: self_attn_layer_norm, ffn_layer_norm.
    """

    def __init__(self, config: SeamlessM4Tv2Config):
        super().__init__()
        self.self_attn = EncoderDecoderAttention(config)
        self.self_attn_layer_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fc1 = Linear(config.hidden_size, config.encoder_ffn_dim)
        self.fc2 = Linear(config.encoder_ffn_dim, config.hidden_size)
        self.ffn_layer_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._act_fn = ACT2FN[config.hidden_act]

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value) -> ir.Value:
        # Pre-norm self-attention: norm → attn → add residual
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(op, hidden_states)
        hidden_states, _ = self.self_attn(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        # Pre-norm FFN: norm → ffn → add residual
        residual = hidden_states
        hidden_states = self.ffn_layer_norm(op, hidden_states)
        hidden_states = self.fc1(op, hidden_states)
        hidden_states = self._act_fn(op, hidden_states)
        hidden_states = self.fc2(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states


class _SeamlessM4Tv2DecoderBlock(nn.Module):
    """Pre-norm decoder block (self-attn + cross-attn + FFN) using decoder_ffn_dim.

    Matches HuggingFace SeamlessM4Tv2DecoderLayer: LayerNorm applied BEFORE each
    sub-layer (pre-norm).  Weight names follow HF exactly: self_attn_layer_norm,
    cross_attention, cross_attention_layer_norm, ffn_layer_norm.
    """

    def __init__(self, config: SeamlessM4Tv2Config):
        super().__init__()
        self.self_attn = EncoderDecoderAttention(config, is_causal=True)
        self.self_attn_layer_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_attention = EncoderDecoderAttention(config)
        self.cross_attention_layer_norm = LayerNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.fc1 = Linear(config.hidden_size, config.decoder_ffn_dim)
        self.fc2 = Linear(config.decoder_ffn_dim, config.hidden_size)
        self.ffn_layer_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        encoder_hidden_states: ir.Value,
        past_key_value: tuple | None = None,
        cross_past_key_value: ir.Value | None = None,
    ):
        # Pre-norm causal self-attention (with KV cache)
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(op, hidden_states)
        hidden_states, self_kv = self.self_attn(
            op, hidden_states, past_key_value=past_key_value
        )
        hidden_states = op.Add(residual, hidden_states)

        # Pre-norm cross-attention to encoder output
        residual = hidden_states
        hidden_states = self.cross_attention_layer_norm(op, hidden_states)
        hidden_states, cross_kv = self.cross_attention(
            op,
            hidden_states,
            key_value_states=encoder_hidden_states,
            past_key_value=cross_past_key_value,
        )
        hidden_states = op.Add(residual, hidden_states)

        # Pre-norm FFN
        residual = hidden_states
        hidden_states = self.ffn_layer_norm(op, hidden_states)
        hidden_states = self.fc1(op, hidden_states)
        hidden_states = self._act_fn(op, hidden_states)
        hidden_states = self.fc2(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, self_kv, cross_kv


# ---------------------------------------------------------------------------
# Text Encoder and Decoder
# ---------------------------------------------------------------------------


class _SeamlessM4Tv2TextEncoder(nn.Module):
    """SeamlessM4T v2 text encoder.

    Applies scaled token embeddings + sinusoidal positional embeddings, passes
    through N encoder blocks, then applies a final LayerNorm.  Unlike BART,
    there is no layernorm_embedding before the blocks; instead the layer_norm
    at the end matches HF's text_encoder.layer_norm.
    """

    def __init__(self, config: SeamlessM4Tv2Config):
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        # Sinusoidal positional embeddings: size is (max_position_embeddings + 2)
        # to accommodate the HF offset of 2 (positions start at index 2).
        self.embed_positions = Embedding(
            config.max_position_embeddings + 2, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [_SeamlessM4Tv2EncoderBlock(config) for _ in range(config.num_hidden_layers)]
        )
        # Applied after all transformer blocks (not before, unlike BART)
        self.layer_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Pre-compute scale: sqrt(hidden_size) when scale_embedding=True
        self._embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value | None = None,
    ) -> ir.Value:
        # (batch, seq_len, hidden_size) — scaled by sqrt(hidden_size)
        inputs_embeds = self.embed_tokens(op, input_ids)
        inputs_embeds = op.Mul(inputs_embeds, self._embed_scale)

        # Sinusoidal position IDs with offset 2: [2, 3, ..., seq_len + 1]
        seq_len = op.Shape(input_ids, start=1, end=2)
        position_ids = op.Range(
            op.Constant(value_int=2),
            op.Add(seq_len, op.Constant(value_int=2)),
            op.Constant(value_int=1),
        )
        position_ids = op.Cast(position_ids, to=7)  # INT64
        position_ids = op.Unsqueeze(position_ids, [0])
        position_embeds = self.embed_positions(op, position_ids)  # (1, seq_len, hidden)

        # (batch, seq_len, hidden_size)
        hidden_states = op.Add(inputs_embeds, position_embeds)

        for layer in self.layers:
            hidden_states = layer(op, hidden_states)

        # Final layer norm (matches HF text_encoder.layer_norm)
        hidden_states = self.layer_norm(op, hidden_states)
        return hidden_states


class _SeamlessM4Tv2TextDecoder(nn.Module):
    """SeamlessM4T v2 text decoder.

    Scaled token embeddings + sinusoidal positional embeddings + N decoder
    blocks (self-attn + cross-attn + FFN) + final LayerNorm + lm_head.
    """

    def __init__(self, config: SeamlessM4Tv2Config):
        super().__init__()
        num_decoder_layers = config.num_decoder_layers or config.num_hidden_layers
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.embed_positions = Embedding(
            config.max_position_embeddings + 2, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [_SeamlessM4Tv2DecoderBlock(config) for _ in range(num_decoder_layers)]
        )
        # Applied after all decoder blocks (matches HF text_decoder.layer_norm)
        self.layer_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self._embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0

    def forward(
        self,
        op: builder.OpBuilder,
        input_ids: ir.Value,
        encoder_hidden_states: ir.Value,
        position_ids: ir.Value | None = None,
        attention_mask: ir.Value | None = None,
        past_key_values: list | None = None,
        cross_past_key_values: ir.Value | None = None,
    ):
        # (batch, seq_len, hidden_size) — scaled
        inputs_embeds = self.embed_tokens(op, input_ids)
        inputs_embeds = op.Mul(inputs_embeds, self._embed_scale)

        # Position IDs with offset 2, accounting for past KV cache length
        if position_ids is None:
            seq_len = op.Shape(input_ids, start=1, end=2)
            if past_key_values is not None:
                # past_key shape: (batch, num_heads, past_seq_len, head_dim)
                past_len = op.Shape(past_key_values[0][0], start=2, end=3)
            else:
                past_len = op.Constant(value_int=0)
            start = op.Add(past_len, op.Constant(value_int=2))
            end = op.Add(start, seq_len)
            position_ids = op.Range(start, end, op.Constant(value_int=1))
            position_ids = op.Cast(position_ids, to=7)  # INT64
            position_ids = op.Unsqueeze(position_ids, [0])

        position_embeds = self.embed_positions(op, position_ids)
        hidden_states = op.Add(inputs_embeds, position_embeds)

        past_kvs = past_key_values or [None] * len(self.layers)
        cross_past_kvs = cross_past_key_values or [None] * len(self.layers)
        present_self_kvs = []
        present_cross_kvs = []

        for layer, past_kv, cross_kv in zip(self.layers, past_kvs, cross_past_kvs):
            hidden_states, self_kv, cross_kv_out = layer(
                op,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                past_key_value=past_kv,
                cross_past_key_value=cross_kv,
            )
            present_self_kvs.append(self_kv)
            present_cross_kvs.append(cross_kv_out)

        # Final layer norm + lm_head
        hidden_states = self.layer_norm(op, hidden_states)
        logits = self.lm_head(op, hidden_states)
        return logits, present_self_kvs, present_cross_kvs


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class SeamlessM4Tv2Model(nn.Module):
    """SeamlessM4T v2 text-to-text encoder-decoder for multilingual translation.

    Produces a ModelPackage with separate encoder and decoder ONNX graphs via
    Seq2SeqTask.  Uses a BART-like architecture but with scaled embeddings,
    sinusoidal positional embeddings, ReLU FFN activations, and separate
    encoder/decoder FFN dimensions.

    HuggingFace: SeamlessM4Tv2ForTextToText (model_type='seamless_m4t_v2')
    """

    default_task = "seq2seq"
    category = "encoder-decoder"

    def __init__(self, config: SeamlessM4Tv2Config):
        super().__init__()
        self.config = config
        self.encoder = _SeamlessM4Tv2TextEncoder(config)
        self.decoder = _SeamlessM4Tv2TextDecoder(config)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        new_dict: dict[str, torch.Tensor] = {}
        shared: torch.Tensor | None = None

        for name, tensor in state_dict.items():
            # Strip "model." prefix present in SeamlessM4Tv2ForTextToText state dict
            if name.startswith("model."):
                name = name[len("model.") :]

            if name == "shared.weight":
                shared = tensor
                continue

            # Remap text_encoder.* → encoder.* and text_decoder.* → decoder.*
            if name.startswith("text_encoder."):
                name = "encoder." + name[len("text_encoder.") :]
            elif name.startswith("text_decoder."):
                name = "decoder." + name[len("text_decoder.") :]

            # Sinusoidal positional embedding buffer is named "weights" in HF
            # (SeamlessM4Tv2SinusoidalPositionalEmbedding registers a buffer called
            # "weights"), but our Embedding stores it as "weight" (nn.Parameter).
            name = name.replace(".embed_positions.weights", ".embed_positions.weight")

            # lm_head is tied to shared.weight in HF; we handle it below
            if name == "lm_head.weight":
                continue

            new_dict[name] = tensor

        # Shared embedding → encoder and decoder embed_tokens
        if shared is not None:
            new_dict.setdefault("encoder.embed_tokens.weight", shared)
            new_dict.setdefault("decoder.embed_tokens.weight", shared)

        # Tie lm_head to encoder embed_tokens (weight tying)
        embed = new_dict.get("encoder.embed_tokens.weight")
        if embed is not None:
            new_dict.setdefault("decoder.lm_head.weight", embed)

        return new_dict


# ---------------------------------------------------------------------------
# Speech encoder (Conformer)
# ---------------------------------------------------------------------------


class _ConvNoBias(nn.Module):
    """Conv1d without bias (weight-only). Used in conformer conv module."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int = 1, groups: int = 1):
        super().__init__()
        self.weight = nn.Parameter([out_ch, in_ch // groups, kernel])
        self._kernel = kernel
        self._stride = stride
        self._groups = groups

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        # x: (B, in_ch, T)
        return op.Conv(
            x,
            self.weight,
            kernel_shape=[self._kernel],
            strides=[self._stride],
            pads=[0, 0],
            group=self._groups,
        )


class _ConvWithBias(nn.Module):
    """Conv1d with bias (no padding). Used in conformer adapter."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int = 1):
        super().__init__()
        self.weight = nn.Parameter([out_ch, in_ch, kernel])
        self.bias = nn.Parameter([out_ch])
        self._kernel = kernel
        self._stride = stride

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        return op.Conv(
            x,
            self.weight,
            self.bias,
            kernel_shape=[self._kernel],
            strides=[self._stride],
            pads=[0, 0],
        )


class _ConvWithBiasAndPad(nn.Module):
    """Conv1d with bias and symmetric padding."""

    def __init__(
        self, in_ch: int, out_ch: int, kernel: int, stride: int = 1, padding: int = 0
    ):
        super().__init__()
        self.weight = nn.Parameter([out_ch, in_ch, kernel])
        self.bias = nn.Parameter([out_ch])
        self._kernel = kernel
        self._stride = stride
        self._padding = padding

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        return op.Conv(
            x,
            self.weight,
            self.bias,
            kernel_shape=[self._kernel],
            strides=[self._stride],
            pads=[self._padding, self._padding],
        )


class _ConformerFFN(nn.Module):
    """Macaron-style feed-forward: Swish(fc1(x)) → fc2.

    No norm or residual — those are applied by the caller.
    Attribute names match HF SeamlessM4Tv2ConformerFeedForward:
    ``intermediate_dense`` and ``output_dense``.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.intermediate_dense = Linear(hidden_size, intermediate_size, bias=True)
        self.output_dense = Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        x = self.intermediate_dense(op, x)
        x = op.Mul(x, op.Sigmoid(x))  # Swish
        return self.output_dense(op, x)


class _ConformerSelfAttention(nn.Module):
    """Conformer self-attention with optional relative-key position embeddings.

    Uses ``linear_q/k/v/out`` naming to match HuggingFace
    ``SeamlessM4Tv2ConformerSelfAttention``.  When
    ``use_position_embeddings=True``, adds a relative position bias via
    ``distance_embedding``.

    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of attention heads.
        left_max: Maximum left context for relative position buckets.
        right_max: Maximum right context for relative position buckets.
        use_position_embeddings: Whether to compute relative position bias.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        left_max: int = 64,
        right_max: int = 8,
        use_position_embeddings: bool = True,
    ):
        super().__init__()
        self._num_heads = num_heads
        self._head_dim = hidden_size // num_heads
        self._left_max = left_max
        self._right_max = right_max
        self._use_pos = use_position_embeddings
        self.linear_q = Linear(hidden_size, hidden_size, bias=True)
        self.linear_k = Linear(hidden_size, hidden_size, bias=True)
        self.linear_v = Linear(hidden_size, hidden_size, bias=True)
        self.linear_out = Linear(hidden_size, hidden_size, bias=True)
        if use_position_embeddings:
            self.distance_embedding = Embedding(left_max + right_max + 1, self._head_dim)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        # x: (B, T, D)
        B = op.Shape(x, start=0, end=1)  # noqa: N806  # (1,)
        T = op.Shape(x, start=1, end=2)  # noqa: N806  # (1,)
        H = self._num_heads  # noqa: N806
        D = self._head_dim  # noqa: N806
        scale = float(D) ** -0.5

        Q = self.linear_q(op, x)  # noqa: N806  # (B, T, H*D)
        K = self.linear_k(op, x)  # noqa: N806
        V = self.linear_v(op, x)  # noqa: N806

        # Reshape: (B, T, H*D) → (B, T, H, D) → transpose → (B, H, T, D)
        shape_4d = op.Concat(B, T, [H], [D], axis=0)
        Q = op.Reshape(Q, shape_4d)  # noqa: N806
        K = op.Reshape(K, shape_4d)  # noqa: N806
        V = op.Reshape(V, shape_4d)  # noqa: N806
        Q = op.Transpose(Q, perm=[0, 2, 1, 3])  # noqa: N806  # (B, H, T, D)
        K = op.Transpose(K, perm=[0, 2, 1, 3])  # noqa: N806
        V = op.Transpose(V, perm=[0, 2, 1, 3])  # noqa: N806

        # Content attention: Q @ K^T → (B, H, T, T)
        K_t = op.Transpose(K, perm=[0, 1, 3, 2])  # noqa: N806  # (B, H, D, T)
        attn = op.MatMul(Q, K_t)  # (B, H, T, T)
        attn = op.Mul(attn, op.Constant(value_float=scale))

        if self._use_pos:
            # Build (T, T) relative position distance matrix, clipped to
            # [-left_max, right_max] then shifted so index 0 = left_max.
            zero = op.Constant(value_int=0)
            one = op.Constant(value_int=1)
            pos = op.Range(zero, T, one)  # (T,) positions [0 .. T-1]
            pos_l = op.Unsqueeze(pos, [1])  # (T, 1)
            pos_r = op.Unsqueeze(pos, [0])  # (1, T)
            dist = op.Sub(pos_r, pos_l)  # (T, T) relative offsets
            dist = op.Clip(
                dist,
                op.Constant(value_int=-self._left_max),
                op.Constant(value_int=self._right_max),
            )
            dist = op.Add(dist, op.Constant(value_int=self._left_max))  # shift ≥ 0
            dist = op.Cast(dist, to=7)  # INT64 for Gather

            # Gather position embeddings: (T, T, D)
            pos_emb = self.distance_embedding(op, dist)

            # Relative attention: einsum("bhld,lrd->bhlr", Q, pos_emb)
            # Q: (B, H, T, D), pos_emb: (T, T, D) → (B, H, T, T)
            rel_attn = op.Einsum(Q, pos_emb, equation="bhld,lrd->bhlr")
            rel_attn = op.Mul(rel_attn, op.Constant(value_float=scale))
            attn = op.Add(attn, rel_attn)

        attn = op.Softmax(attn, axis=-1)  # (B, H, T, T)

        # Weighted sum over values: (B, H, T, T) @ (B, H, T, D) → (B, H, T, D)
        out = op.MatMul(attn, V)
        out = op.Transpose(out, perm=[0, 2, 1, 3])  # (B, T, H, D)

        # Merge heads: (B, T, H*D)
        shape_3d = op.Concat(B, T, [H * D], axis=0)
        out = op.Reshape(out, shape_3d)

        return self.linear_out(op, out)


class _ConformerConvModule(nn.Module):
    """Conformer causal depthwise-conv module.

    Structure (all in (B, C, T) layout internally)::

        LayerNorm → PointwiseConv(GLU) → CausalDepthwiseConv
            → LayerNorm → Swish → PointwiseConv

    Attribute names match HF ``SeamlessM4Tv2ConformerConvolutionModule``.
    """

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self._channels = channels
        self._kernel_size = kernel_size
        self.layer_norm = LayerNorm(channels)
        # Pointwise conv expands to 2C for GLU gating
        self.pointwise_conv1 = _ConvNoBias(channels, channels * 2, kernel=1)
        # Causal depthwise conv (no bias, padding added manually)
        self.depthwise_conv = _ConvNoBias(
            channels, channels, kernel=kernel_size, groups=channels
        )
        self.depthwise_layer_norm = LayerNorm(channels)
        self.pointwise_conv2 = _ConvNoBias(channels, channels, kernel=1)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        # x: (B, T, C)
        x = self.layer_norm(op, x)
        x = op.Transpose(x, perm=[0, 2, 1])  # (B, C, T)

        # Pointwise conv + GLU: (B, 2C, T) → (B, C, T)
        x = self.pointwise_conv1(op, x)  # (B, 2C, T)
        x_a, x_b = op.Split(x, num_outputs=2, axis=1, _outputs=2)
        x = op.Mul(x_a, op.Sigmoid(x_b))  # GLU → (B, C, T)

        # Causal padding: pad (kernel_size - 1) on the left of T dimension.
        # ONNX Pad format for (B, C, T): [B_beg, C_beg, T_beg, B_end, C_end, T_end]
        pad_size = self._kernel_size - 1
        pads = op.Constant(value_ints=[0, 0, pad_size, 0, 0, 0])
        x = op.Pad(x, pads)  # (B, C, T + pad_size)
        x = self.depthwise_conv(op, x)  # (B, C, T)

        x = op.Transpose(x, perm=[0, 2, 1])  # (B, T, C)
        x = self.depthwise_layer_norm(op, x)
        x = op.Mul(x, op.Sigmoid(x))  # Swish

        x = op.Transpose(x, perm=[0, 2, 1])  # (B, C, T)
        x = self.pointwise_conv2(op, x)  # (B, C, T)
        return op.Transpose(x, perm=[0, 2, 1])  # (B, T, C)


class _ConformerLayer(nn.Module):
    """Single Conformer encoder layer (Macaron structure).

    Forward pass::

        x = x + 0.5 * FFN1(LN(x))
        x = x + SelfAttn(LN(x))
        x = x + ConvModule(x)          # LN is inside ConvModule
        x = x + 0.5 * FFN2(LN(x))
        x = FinalLN(x)

    Attribute names match HF ``SeamlessM4Tv2ConformerEncoderLayer``.
    """

    def __init__(self, config: SeamlessM4Tv2Config):
        super().__init__()
        self.ffn1_layer_norm = LayerNorm(config.hidden_size)
        self.ffn1 = _ConformerFFN(config.hidden_size, config.speech_encoder_intermediate_size)
        self.self_attn_layer_norm = LayerNorm(config.hidden_size)
        self.self_attn = _ConformerSelfAttention(
            config.hidden_size,
            config.speech_encoder_attention_heads,
            left_max=config.left_max_position_embeddings,
            right_max=config.right_max_position_embeddings,
        )
        self.conv_module = _ConformerConvModule(
            config.hidden_size, config.conv_depthwise_kernel_size
        )
        self.ffn2_layer_norm = LayerNorm(config.hidden_size)
        self.ffn2 = _ConformerFFN(config.hidden_size, config.speech_encoder_intermediate_size)
        self.final_layer_norm = LayerNorm(config.hidden_size)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        half = op.Constant(value_float=0.5)
        # Macaron FFN 1 with half-scale residual
        x = op.Add(x, op.Mul(half, self.ffn1(op, self.ffn1_layer_norm(op, x))))
        # Self-attention residual
        x = op.Add(x, self.self_attn(op, self.self_attn_layer_norm(op, x)))
        # Conv module residual (LayerNorm is inside conv_module)
        x = op.Add(x, self.conv_module(op, x))
        # Macaron FFN 2 with half-scale residual
        x = op.Add(x, op.Mul(half, self.ffn2(op, self.ffn2_layer_norm(op, x))))
        return self.final_layer_norm(op, x)


class _ConformerAdapterLayer(nn.Module):
    """Conformer adapter layer with strided subsampling.

    Two parallel strided Conv1d+GLU paths produce the downsampled residual
    and the self-attention query sequence.  Self-attention (no relative pos)
    is then applied, followed by a macaron FFN.

    Attribute names match HF ``SeamlessM4Tv2ConformerAdapterLayer``.
    """

    def __init__(self, config: SeamlessM4Tv2Config):
        super().__init__()
        D = config.hidden_size  # noqa: N806
        k = config.adaptor_kernel_size
        s = config.adaptor_stride
        self.residual_layer_norm = LayerNorm(D)
        self.residual_conv = _ConvWithBias(D, D * 2, kernel=k, stride=s)
        self.self_attn_layer_norm = LayerNorm(D)
        self.self_attn_conv = _ConvWithBias(D, D * 2, kernel=k, stride=s)
        self.self_attn = _ConformerSelfAttention(
            D,
            config.speech_encoder_attention_heads,
            use_position_embeddings=False,
        )
        self.ffn_layer_norm = LayerNorm(D)
        self.ffn = _ConformerFFN(D, config.speech_encoder_intermediate_size)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        # x: (B, T, D)
        # Residual path: norm → stride-s conv → GLU → (B, T//s, D)
        res = self.residual_layer_norm(op, x)
        res = op.Transpose(res, perm=[0, 2, 1])  # (B, D, T)
        res = self.residual_conv(op, res)  # (B, 2D, T//s)
        res_a, res_b = op.Split(res, num_outputs=2, axis=1, _outputs=2)
        res = op.Mul(res_a, op.Sigmoid(res_b))  # GLU → (B, D, T//s)
        res = op.Transpose(res, perm=[0, 2, 1])  # (B, T//s, D)

        # Self-attn path: norm → stride-s conv → GLU → (B, T//s, D)
        h = self.self_attn_layer_norm(op, x)
        h = op.Transpose(h, perm=[0, 2, 1])
        h = self.self_attn_conv(op, h)
        h_a, h_b = op.Split(h, num_outputs=2, axis=1, _outputs=2)
        h = op.Mul(h_a, op.Sigmoid(h_b))
        h = op.Transpose(h, perm=[0, 2, 1])  # (B, T//s, D)

        # Self-attention (no relative pos) + residual
        h = op.Add(h, self.self_attn(op, h))
        # Merge with strided residual
        h = op.Add(res, h)
        # FFN + residual
        h = op.Add(h, self.ffn(op, self.ffn_layer_norm(op, h)))
        return h


class _FeatureProjection(nn.Module):
    """Feature projection for speech encoder input.

    Applies LayerNorm then Linear to project fbank features into the
    hidden space.  Attribute names match HF
    ``SeamlessM4Tv2ConformerFeatureProjection``.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.layer_norm = LayerNorm(input_dim, eps=1e-5)
        self.projection = Linear(input_dim, output_dim, bias=True)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        x = self.layer_norm(op, x)
        return self.projection(op, x)


class _ConformerEncoder(nn.Module):
    """Wrapper holding the stack of Conformer layers and their final LayerNorm.

    Named ``encoder`` on ``SeamlessM4Tv2SpeechEncoderModel`` so that
    weight paths ``encoder.layers.N.*`` and ``encoder.layer_norm.*``
    align with HuggingFace ``SeamlessM4Tv2ConformerEncoder``.
    """

    def __init__(self, config: SeamlessM4Tv2Config):
        super().__init__()
        self.layers = nn.ModuleList(
            [_ConformerLayer(config) for _ in range(config.speech_encoder_layers)]
        )
        self.layer_norm = LayerNorm(config.hidden_size)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        for layer in self.layers:
            x = layer(op, x)
        return self.layer_norm(op, x)


class _ConformerAdapter(nn.Module):
    """Wrapper holding the adapter layers.

    Named ``adapter`` on ``SeamlessM4Tv2SpeechEncoderModel`` so that
    weight paths ``adapter.layers.N.*`` align with HuggingFace
    ``SeamlessM4Tv2ConformerAdapter``.
    """

    def __init__(self, config: SeamlessM4Tv2Config):
        super().__init__()
        self.layers = nn.ModuleList(
            [_ConformerAdapterLayer(config) for _ in range(config.num_adapter_layers)]
        )

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        for layer in self.layers:
            x = layer(op, x)
        return x


class SeamlessM4Tv2SpeechEncoderModel(nn.Module):
    """SeamlessM4T v2 Conformer-based speech encoder.

    Encodes log-mel filterbank features ``(B, T, feature_projection_input_dim)``
    -> hidden states ``(B, T // adaptor_stride, hidden_size)``.

    Architecture::

        feature_projection: LayerNorm(F) + Linear(F -> D)
        encoder: N x _ConformerLayer (Macaron Conformer) + LayerNorm
        intermediate_ffn: _ConformerFFN (applied with residual)
        inner_layer_norm: LayerNorm
        adapter: M x _ConformerAdapterLayer (stride-8 downsampling)

    HuggingFace: ``SeamlessM4Tv2SpeechEncoder``
    (``speech_encoder`` in ``SeamlessM4Tv2ForSpeechToSpeech``)
    """

    def __init__(self, config: SeamlessM4Tv2Config):
        super().__init__()
        self.config = config
        D = config.hidden_size  # noqa: N806
        F = config.feature_projection_input_dim  # noqa: N806

        self.feature_projection = _FeatureProjection(F, D)
        # Conformer encoder stack; stored as 'encoder' to align with HF paths
        # speech_encoder.encoder.layers.N.* and speech_encoder.encoder.layer_norm.*
        self.encoder = _ConformerEncoder(config)
        # Intermediate FFN applied with pre-norm and residual after all conformer layers
        self.intermediate_ffn = _ConformerFFN(D, config.speech_encoder_intermediate_size)
        self.inner_layer_norm = LayerNorm(D)
        # Strided adapter; stored as 'adapter' to align with HF paths
        # speech_encoder.adapter.layers.N.*
        self.adapter = _ConformerAdapter(config)

    def forward(self, op: builder.OpBuilder, input_features: ir.Value) -> ir.Value:
        # input_features: (B, T, F)
        x = self.feature_projection(op, input_features)  # (B, T, D)
        x = self.encoder(op, x)  # (B, T, D)

        # Intermediate FFN: save residual before norm, apply norm+FFN, add residual
        residual = x
        x = self.inner_layer_norm(op, x)
        x = self.intermediate_ffn(op, x)
        x = op.Add(x, residual)

        return self.adapter(op, x)  # (B, T//stride, D)


# ---------------------------------------------------------------------------
# T2U (text-to-unit) model
# ---------------------------------------------------------------------------


class _T2UConv1d(nn.Module):
    """Conv1d with bias and symmetric padding, used in T2U decoder layers."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, padding: int):
        super().__init__()
        self.weight = nn.Parameter([out_ch, in_ch, kernel])
        self.bias = nn.Parameter([out_ch])
        self._kernel = kernel
        self._padding = padding

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        return op.Conv(
            x,
            self.weight,
            self.bias,
            kernel_shape=[self._kernel],
            pads=[self._padding, self._padding],
        )


class _T2UFFN(nn.Module):
    """T2U feed-forward network with ReLU activation.

    Attribute names ``fc1`` / ``fc2`` match HF
    ``SeamlessM4Tv2TextToUnitEncoderLayer`` which wraps the FFN in a
    ``SeamlessM4Tv2FeedForwardNetwork`` sub-module named ``ffn``.
    """

    def __init__(self, hidden_size: int, ffn_dim: int):
        super().__init__()
        self.fc1 = Linear(hidden_size, ffn_dim, bias=True)
        self.fc2 = Linear(ffn_dim, hidden_size, bias=True)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        x = self.fc1(op, x)
        x = op.Relu(x)
        return self.fc2(op, x)


class _T2UEncoderLayer(nn.Module):
    """T2U encoder layer: POST-norm self-attention + nested FFN.

    Attribute names match HF ``SeamlessM4Tv2EncoderLayer`` (used with
    ``is_t2u_encoder=True``).  The FFN is wrapped inside a ``ffn`` sub-module
    to produce weight paths ``ffn.fc1.*`` / ``ffn.fc2.*``.
    """

    def __init__(self, config: SeamlessM4Tv2Config):
        super().__init__()
        self.self_attn = EncoderDecoderAttention(config)
        self.self_attn_layer_norm = LayerNorm(config.hidden_size)
        self.ffn = _T2UFFN(config.hidden_size, config.t2u_encoder_ffn_dim)
        self.ffn_layer_norm = LayerNorm(config.hidden_size)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        attn_out, _ = self.self_attn(op, x)
        x = op.Add(x, attn_out)
        x = self.self_attn_layer_norm(op, x)  # POST-norm after attention
        x = op.Add(x, self.ffn(op, x))
        x = self.ffn_layer_norm(op, x)  # POST-norm after FFN
        return x


class _T2UDecoderLayer(nn.Module):
    """T2U decoder layer: POST-norm self-attention + Conv1d residual block.

    Two Conv1d layers with same-padding (kernel=7, padding=3) replace the
    cross-attention + FFN of a standard decoder layer.  Attribute names match
    HF ``SeamlessM4Tv2TextToUnitDecoderLayer``.
    """

    def __init__(self, config: SeamlessM4Tv2Config):
        super().__init__()
        D = config.hidden_size  # noqa: N806
        self.self_attn = EncoderDecoderAttention(config)
        self.self_attn_layer_norm = LayerNorm(D)
        # kernel=7, same padding=3  (matches HF Conv1d(padding="same"))
        self.conv1 = _T2UConv1d(D, D, kernel=7, padding=3)
        self.conv2 = _T2UConv1d(D, D, kernel=7, padding=3)
        self.conv_layer_norm = LayerNorm(D)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        attn_out, _ = self.self_attn(op, x)
        x = op.Add(x, attn_out)
        x = self.self_attn_layer_norm(op, x)  # POST-norm

        residual = x
        # Conv branch operates in (B, D, T) layout
        x_t = op.Transpose(x, perm=[0, 2, 1])  # (B, D, T)
        x_t = op.Relu(self.conv1(op, x_t))
        x_t = self.conv2(op, x_t)
        x = op.Add(residual, op.Transpose(x_t, perm=[0, 2, 1]))
        x = self.conv_layer_norm(op, x)  # POST-norm
        return x


class SeamlessM4Tv2T2UModel(nn.Module):
    """SeamlessM4T v2 text-to-unit non-autoregressive converter.

    Converts text token IDs to acoustic unit logits via 6 encoder layers
    and 6 decoder layers.  There is no cross-attention; both stacks process
    the same sequence.

    Architecture::

        t2u_embed_tokens: Embedding(t2u_vocab_size, D)
        t2u_embed_positions: Embedding(max_position_embeddings + 2, D)
        t2u_encoder: N x _T2UEncoderLayer + t2u_encoder_layer_norm
        t2u_decoder: M x _T2UDecoderLayer + t2u_decoder_layer_norm
        t2u_lm_head: Linear(D, t2u_vocab_size)

    HuggingFace: ``SeamlessM4Tv2TextToUnitForConditionalGeneration``
    (``t2u_model`` in ``SeamlessM4Tv2ForSpeechToSpeech``)
    """

    def __init__(self, config: SeamlessM4Tv2Config):
        super().__init__()
        self.config = config
        D = config.hidden_size  # noqa: N806

        self.t2u_embed_tokens = Embedding(config.t2u_vocab_size, D)
        # Positional embeddings with HF-style offset of 2
        self.t2u_embed_positions = Embedding(config.t2u_max_position_embeddings + 2, D)

        self.t2u_encoder = nn.ModuleList(
            [_T2UEncoderLayer(config) for _ in range(config.t2u_encoder_layers)]
        )
        self.t2u_encoder_layer_norm = LayerNorm(D)

        self.t2u_decoder = nn.ModuleList(
            [_T2UDecoderLayer(config) for _ in range(config.t2u_decoder_layers)]
        )
        self.t2u_decoder_layer_norm = LayerNorm(D)

        self.t2u_lm_head = Linear(D, config.t2u_vocab_size, bias=False)

    def forward(self, op: builder.OpBuilder, input_ids: ir.Value) -> ir.Value:
        # input_ids: (B, T) int64
        T = op.Shape(input_ids, start=1, end=2)  # noqa: N806  # (1,) sequence length

        # Token embeddings + sinusoidal positional embeddings (offset 2)
        x = self.t2u_embed_tokens(op, input_ids)  # (B, T, D)
        start = op.Constant(value_int=2)
        end = op.Add(start, T)
        pos_ids = op.Range(start, end, op.Constant(value_int=1))  # [2 .. T+1]
        pos_ids = op.Cast(pos_ids, to=7)  # INT64
        pos_ids = op.Unsqueeze(pos_ids, [0])  # (1, T)
        x = op.Add(x, self.t2u_embed_positions(op, pos_ids))

        # Encoder layers
        for layer in self.t2u_encoder:
            x = layer(op, x)
        x = self.t2u_encoder_layer_norm(op, x)

        # Decoder layers (same sequence — no cross-attention)
        for layer in self.t2u_decoder:
            x = layer(op, x)
        x = self.t2u_decoder_layer_norm(op, x)

        return self.t2u_lm_head(op, x)  # (B, T, t2u_vocab_size)


# ---------------------------------------------------------------------------
# Vocoder (HiFi-GAN based)
# ---------------------------------------------------------------------------


class _DilatedConv1d(nn.Module):
    """Dilated Conv1d with bias and symmetric padding.  Used in HiFi-GAN residual blocks."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, padding: int):
        super().__init__()
        self.weight = nn.Parameter([channels, channels, kernel_size])
        self.bias = nn.Parameter([channels])
        self._kernel = kernel_size
        self._dilation = dilation
        self._padding = padding

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        return op.Conv(
            x,
            self.weight,
            self.bias,
            kernel_shape=[self._kernel],
            dilations=[self._dilation],
            pads=[self._padding, self._padding],
        )


class _HifiGanResidualBlock(nn.Module):
    """HiFi-GAN residual block with multiple dilation rates.

    Two parallel convolution stacks (``convs1`` dilated, ``convs2`` undilated)
    are summed for each dilation, with LeakyReLU activations.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation_sizes: list[int],
        leaky_slope: float,
    ):
        super().__init__()
        self._slope = leaky_slope
        self.convs1 = nn.ModuleList(
            [
                _DilatedConv1d(channels, kernel_size, d, d * (kernel_size - 1) // 2)
                for d in dilation_sizes
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                _DilatedConv1d(channels, kernel_size, 1, (kernel_size - 1) // 2)
                for _ in dilation_sizes
            ]
        )

    def forward(self, op: builder.OpBuilder, h: ir.Value) -> ir.Value:
        for c1, c2 in zip(self.convs1, self.convs2):
            x = op.LeakyRelu(h, alpha=self._slope)
            x = c1(op, x)
            x = op.LeakyRelu(x, alpha=self._slope)
            x = c2(op, x)
            h = op.Add(h, x)
        return h


class _ConvTranspose1d(nn.Module):
    """ConvTranspose1d with bias and symmetric output padding crop."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int,
        padding: int,
    ):
        super().__init__()
        # ONNX ConvTranspose weight shape: (in_ch, out_ch/groups, kernel)
        self.weight = nn.Parameter([in_ch, out_ch, kernel])
        self.bias = nn.Parameter([out_ch])
        self._kernel = kernel
        self._stride = stride
        self._padding = padding

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        return op.ConvTranspose(
            x,
            self.weight,
            self.bias,
            kernel_shape=[self._kernel],
            strides=[self._stride],
            pads=[self._padding, self._padding],
        )


class _HifiGan(nn.Module):
    """HiFi-GAN generator for unit-to-waveform synthesis.

    Processes ``(B, model_in, T)`` channel-first input through a series of
    upsampling ConvTranspose layers and multi-dilation residual blocks.

    Attribute names match HF ``SeamlessM4Tv2HifiGan``.
    """

    def __init__(self, config: SeamlessM4Tv2Config):
        super().__init__()
        model_in = config.unit_embed_dim + config.lang_embed_dim + config.spkr_embed_dim
        ch = config.upsample_initial_channel
        rates = config.upsample_rates
        kernels = config.upsample_kernel_sizes
        rb_kernels = config.resblock_kernel_sizes
        rb_dilations = config.resblock_dilation_sizes
        slope = config.leaky_relu_slope

        self.conv_pre = _ConvWithBiasAndPad(model_in, ch, kernel=7, padding=3)

        # One ConvTranspose upsampler per rate
        self.upsampler = nn.ModuleList()
        for i, (rate, kernel) in enumerate(zip(rates, kernels)):
            in_ch = ch // (2**i)
            out_ch = ch // (2 ** (i + 1))
            pad = (kernel - rate) // 2
            self.upsampler.append(_ConvTranspose1d(in_ch, out_ch, kernel, rate, pad))

        # Residual blocks - one set per upsampler stage x kernel size
        self.resblocks = nn.ModuleList()
        for i in range(len(rates)):
            block_ch = ch // (2 ** (i + 1))
            for j in range(len(rb_kernels)):
                self.resblocks.append(
                    _HifiGanResidualBlock(block_ch, rb_kernels[j], rb_dilations[j], slope)
                )

        final_ch = ch // (2 ** len(rates))
        self.conv_post = _ConvWithBiasAndPad(final_ch, 1, kernel=7, padding=3)
        self._num_resblocks_per_up = len(rb_kernels)
        self._slope = slope

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        # x: (B, model_in, T)
        h = self.conv_pre(op, x)
        n = self._num_resblocks_per_up
        for i, up in enumerate(self.upsampler):
            h = op.LeakyRelu(h, alpha=self._slope)
            h = up(op, h)
            # Average over the n residual block outputs for this stage
            blocks = self.resblocks[i * n : (i + 1) * n]
            hs = blocks[0](op, h)
            for blk in blocks[1:]:
                hs = op.Add(hs, blk(op, h))
            h = op.Mul(hs, op.Constant(value_float=1.0 / float(n)))
        h = op.LeakyRelu(h, alpha=self._slope)
        h = self.conv_post(op, h)
        h = op.Tanh(h)
        return op.Squeeze(h, [1])  # (B, T_audio)


class _VariancePredictor(nn.Module):
    """Duration predictor for log-duration prediction per unit token.

    Two dilated conv layers with LayerNorm and ReLU, followed by a linear
    projection to a scalar.  Attribute names match HF
    ``SeamlessM4Tv2VariancePredictor``.
    """

    def __init__(self, config: SeamlessM4Tv2Config):
        super().__init__()
        in_dim = config.unit_embed_dim
        hidden = config.t2u_variance_predictor_hidden_dim
        k = config.t2u_variance_predictor_kernel_size
        pad = k // 2
        self.conv1 = _ConvWithBiasAndPad(in_dim, hidden, kernel=k, padding=pad)
        self.ln1 = LayerNorm(hidden)
        self.conv2 = _ConvWithBiasAndPad(hidden, hidden, kernel=k, padding=pad)
        self.ln2 = LayerNorm(hidden)
        self.proj = Linear(hidden, 1, bias=True)

    def forward(self, op: builder.OpBuilder, hidden: ir.Value) -> ir.Value:
        # hidden: (B, T, in_dim) — operate in (B, in_dim, T) for Conv
        h = op.Transpose(hidden, perm=[0, 2, 1])  # (B, in_dim, T)
        h = op.Relu(self.conv1(op, h))  # (B, hidden, T)
        h = op.Transpose(h, perm=[0, 2, 1])  # (B, T, hidden)
        h = self.ln1(op, h)
        h = op.Transpose(h, perm=[0, 2, 1])  # (B, hidden, T)
        h = op.Relu(self.conv2(op, h))  # (B, hidden, T)
        h = op.Transpose(h, perm=[0, 2, 1])  # (B, T, hidden)
        h = self.ln2(op, h)
        out = self.proj(op, h)  # (B, T, 1)
        return op.Squeeze(out, [-1])  # (B, T)


class _DurHead(nn.Module):
    """Duration-prediction head: unit_embedding + dur_predictor.

    Exposed as ``SeamlessM4Tv2VocoderModel.dur_head`` to give this path
    its own parameter scope, preventing parameter sharing conflicts when
    Speech2SpeechTask builds the ``vocoder_dur`` and ``vocoder_hifigan``
    ONNX graphs from the same module instance.
    """

    def __init__(self, config: SeamlessM4Tv2Config):
        super().__init__()
        self.unit_embedding = Embedding(config.unit_hifi_gan_vocab_size, config.unit_embed_dim)
        self.dur_predictor = _VariancePredictor(config)

    def forward(self, op: builder.OpBuilder, unit_ids: ir.Value) -> ir.Value:
        # unit_ids: (B, T) int64
        hidden = self.unit_embedding(op, unit_ids)  # (B, T, unit_embed_dim)
        return self.dur_predictor(op, hidden)  # (B, T)


class _HifiGanHead(nn.Module):
    """HiFi-GAN synthesis head: unit_embedding + conditioning embeddings + generator.

    Exposed as ``SeamlessM4Tv2VocoderModel.hifigan_head`` so this path
    has its own parameter scope independent from :class:`_DurHead`.
    """

    def __init__(self, config: SeamlessM4Tv2Config):
        super().__init__()
        self._spkr_embed_dim = config.spkr_embed_dim
        self._lang_embed_dim = config.lang_embed_dim
        self.unit_embedding = Embedding(config.unit_hifi_gan_vocab_size, config.unit_embed_dim)
        self.speaker_embedding = Embedding(config.vocoder_num_spkrs, config.spkr_embed_dim)
        self.language_embedding = Embedding(config.vocoder_num_langs, config.lang_embed_dim)
        self.hifi_gan = _HifiGan(config)

    def forward(
        self,
        op: builder.OpBuilder,
        unit_ids: ir.Value,
        speaker_id: ir.Value,
        lang_id: ir.Value,
    ) -> ir.Value:
        hidden = self.unit_embedding(op, unit_ids)  # (B, T, unit_embed_dim)
        spkr = self.speaker_embedding(op, speaker_id)  # (B, 1, spkr_embed_dim)
        lang = self.language_embedding(op, lang_id)  # (B, 1, lang_embed_dim)

        hidden_t = op.Transpose(hidden, perm=[0, 2, 1])  # (B, unit_embed_dim, T)
        spkr_t = op.Transpose(spkr, perm=[0, 2, 1])  # (B, spkr_embed_dim, 1)
        lang_t = op.Transpose(lang, perm=[0, 2, 1])  # (B, lang_embed_dim, 1)

        B = op.Shape(hidden_t, start=0, end=1)  # noqa: N806
        T = op.Shape(hidden_t, start=2, end=3)  # noqa: N806
        spkr_exp = op.Expand(
            spkr_t, op.Concat(B, [self._spkr_embed_dim], T, axis=0)
        )  # (B, spkr_embed_dim, T)
        lang_exp = op.Expand(
            lang_t, op.Concat(B, [self._lang_embed_dim], T, axis=0)
        )  # (B, lang_embed_dim, T)

        combined = op.Concat(lang_exp, hidden_t, spkr_exp, axis=1)  # (B, model_in, T)
        return self.hifi_gan(op, combined)  # (B, T_audio)


class SeamlessM4Tv2VocoderModel(nn.Module):
    """SeamlessM4T v2 vocoder: acoustic unit tokens → waveform.

    Wraps two independently-scoped sub-modules used by Speech2SpeechTask:

    * ``dur_head``    — Duration predictor path (``unit_embedding`` + ``dur_predictor``)
    * ``hifigan_head``— HiFi-GAN synthesis path (``unit_embedding`` + conditioning + generator)

    Each path carries its own ``unit_embedding`` copy so that
    Speech2SpeechTask can build ``vocoder_dur`` and ``vocoder_hifigan``
    as separate ONNX models without cross-graph parameter conflicts.

    HuggingFace: ``SeamlessM4Tv2CodeHifiGan``
    (``vocoder`` in ``SeamlessM4Tv2ForSpeechToSpeech``)
    """

    def __init__(self, config: SeamlessM4Tv2Config):
        super().__init__()
        self.dur_head = _DurHead(config)
        self.hifigan_head = _HifiGanHead(config)

    def forward(self, op: builder.OpBuilder, unit_ids: ir.Value) -> ir.Value:
        """Default forward: duration prediction path."""
        return self.dur_head(op, unit_ids)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sinusoidal_embeddings(
    num_embeddings: int, embedding_dim: int, padding_idx: int | None = None
) -> torch.Tensor:
    """Compute sinusoidal positional embeddings (same formula as HF SeamlessM4Tv2).

    HF stores these as ``persistent=False`` buffers, so they are absent from
    saved checkpoints and must be re-computed at weight-loading time.
    """
    half_dim = embedding_dim // 2
    scale = math.log(10000) / (half_dim - 1)
    freqs = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -scale)
    positions = torch.arange(num_embeddings, dtype=torch.float32)
    emb = positions.unsqueeze(1) * freqs.unsqueeze(0)  # (N, half_dim)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # (N, embedding_dim)
    if embedding_dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
    if padding_idx is not None:
        emb[padding_idx, :] = 0.0
    return emb


# T2U weight name remapper (module-level helper)
# ---------------------------------------------------------------------------


def _remap_t2u_name(name: str) -> str | None:
    """Remap a HuggingFace T2U weight name to its ONNX module path.

    Handles the translation from ``t2u_model.*`` (HF
    ``SeamlessM4Tv2ForSpeechToSpeech``) to ``t2u.*``
    (``SeamlessM4Tv2T2UModel`` inside the ONNX wrapper).

    Returns ``None`` to drop a weight (e.g. duration predictor, char
    embeddings, and positional-scaling scalars that have no ONNX equivalent).
    """
    # Encoder token embedding lives on model.decoder.embed_tokens in HF
    if name == "t2u_model.model.decoder.embed_tokens.weight":
        return "t2u.t2u_embed_tokens.weight"

    # Encoder layers: t2u_model.model.encoder.layers.N.* → t2u.t2u_encoder.N.*
    pfx = "t2u_model.model.encoder.layers."
    if name.startswith(pfx):
        return "t2u.t2u_encoder." + name[len(pfx) :]

    # Encoder final layer norm
    if name == "t2u_model.model.encoder.layer_norm.weight":
        return "t2u.t2u_encoder_layer_norm.weight"
    if name == "t2u_model.model.encoder.layer_norm.bias":
        return "t2u.t2u_encoder_layer_norm.bias"

    # Decoder layers: t2u_model.model.decoder.layers.N.* → t2u.t2u_decoder.N.*
    pfx = "t2u_model.model.decoder.layers."
    if name.startswith(pfx):
        return "t2u.t2u_decoder." + name[len(pfx) :]

    # Decoder final layer norm
    if name == "t2u_model.model.decoder.layer_norm.weight":
        return "t2u.t2u_decoder_layer_norm.weight"
    if name == "t2u_model.model.decoder.layer_norm.bias":
        return "t2u.t2u_decoder_layer_norm.bias"

    # LM head
    if name.startswith("t2u_model.lm_head."):
        return "t2u.t2u_lm_head." + name[len("t2u_model.lm_head.") :]

    # Drop: duration predictor, char embedding, positional scaling scalars
    return None


# ---------------------------------------------------------------------------
# Speech-to-Speech wrapper
# ---------------------------------------------------------------------------


class SeamlessM4Tv2SpeechToSpeechModel(nn.Module):
    """SeamlessM4T v2 speech-to-speech pipeline wrapper.

    Holds all four sub-models and provides ``preprocess_weights`` for
    loading a ``SeamlessM4Tv2ForSpeechToSpeech`` HuggingFace checkpoint.

    Sub-models:

    * ``speech_encoder`` — Conformer encoder (fbank → encoder hidden states)
    * ``decoder``        — Text decoder (encoder hidden states → text logits)
    * ``t2u``            — T2U NAR model (text tokens → acoustic unit logits)
    * ``vocoder``        — HiFi-GAN vocoder (acoustic units → waveform)

    HuggingFace: ``SeamlessM4Tv2ForSpeechToSpeech``
    """

    default_task = "speech2speech"
    category = "Speech-to-Text"

    def __init__(self, config: SeamlessM4Tv2Config):
        super().__init__()
        self.config = config
        self.speech_encoder = SeamlessM4Tv2SpeechEncoderModel(config)
        self.decoder = _SeamlessM4Tv2TextDecoder(config)
        self.t2u = SeamlessM4Tv2T2UModel(config)
        self.vocoder = SeamlessM4Tv2VocoderModel(config)

    def forward(self, op: builder.OpBuilder, input_features: ir.Value) -> ir.Value:
        """Stub forward — Speech2SpeechTask builds each sub-model separately."""
        return self.speech_encoder(op, input_features)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        new_dict: dict[str, torch.Tensor] = {}
        shared: torch.Tensor | None = None

        for name, tensor in state_dict.items():
            # Speech encoder: speech_encoder.* → speech_encoder.*
            # (no renaming needed — HF and ONNX paths align)
            if name.startswith("speech_encoder."):
                new_dict[name] = tensor
                continue

            # Shared text embedding
            if name == "shared.weight":
                shared = tensor
                continue

            # Top-level lm_head is tied to shared embedding; handled below
            if name == "lm_head.weight":
                continue

            # Text decoder: text_decoder.* → decoder.*
            # Also strip the nested 'ffn.' wrapper around fc1/fc2 to align
            # with _SeamlessM4Tv2DecoderBlock which exposes fc1/fc2 directly.
            if name.startswith("text_decoder."):
                sub = name[len("text_decoder.") :]
                # sub starts at the segment *after* "text_decoder." so there's no
                # leading dot — use bare "embed_positions.weights" pattern.
                sub = sub.replace("embed_positions.weights", "embed_positions.weight")
                sub = sub.replace(".ffn.fc1.", ".fc1.")
                sub = sub.replace(".ffn.fc2.", ".fc2.")
                new_dict["decoder." + sub] = tensor
                continue

            # T2U model
            if name.startswith("t2u_model."):
                new_name = _remap_t2u_name(name)
                if new_name is not None:
                    new_dict[new_name] = tensor
                continue

            # Vocoder weights: remap to the split dur_head / hifigan_head sub-modules.
            # The task builds vocoder_dur and vocoder_hifigan models by calling
            # module.vocoder.dur_head.forward() and module.vocoder.hifigan_head.forward().
            # onnxscript resolves initializer names relative to the sub-module, so
            # ONNX params are "dur_head.*" and "hifigan_head.*" (no "vocoder." prefix).
            if name.startswith("vocoder."):
                sub = name[len("vocoder.") :]
                if sub.startswith("unit_embedding."):
                    new_dict["dur_head." + sub] = tensor
                    new_dict["hifigan_head." + sub] = tensor
                elif sub.startswith("speaker_embedding."):
                    new_dict["hifigan_head." + sub] = tensor
                elif sub.startswith("language_embedding."):
                    new_dict["hifigan_head." + sub] = tensor
                elif sub.startswith("dur_predictor."):
                    new_dict["dur_head." + sub] = tensor
                elif sub.startswith("hifi_gan."):
                    new_dict["hifigan_head." + sub] = tensor
                continue

        # Populate decoder token embedding and tied lm_head from shared weight
        if shared is not None:
            new_dict.setdefault("decoder.embed_tokens.weight", shared)
        embed = new_dict.get("decoder.embed_tokens.weight")
        if embed is not None:
            new_dict.setdefault("decoder.lm_head.weight", embed)

        # Sinusoidal positional embeddings are stored as persistent=False buffers
        # in HF, so they are absent from the checkpoint.  Compute and inject them.
        cfg = self.config
        # Text decoder: max_position_embeddings + 2 (SeamlessM4T uses offset=2)
        dec_sin = _make_sinusoidal_embeddings(
            cfg.max_position_embeddings + 2, cfg.hidden_size, padding_idx=cfg.pad_token_id
        )
        new_dict.setdefault("decoder.embed_positions.weight", dec_sin)

        # T2U encoder embed_positions (same formula, own max/pad config)
        t2u_sin = _make_sinusoidal_embeddings(
            cfg.t2u_max_position_embeddings + 2,
            cfg.hidden_size,
            padding_idx=cfg.t2u_pad_token_id,
        )
        new_dict.setdefault("t2u.t2u_embed_positions.weight", t2u_sin)

        return new_dict
