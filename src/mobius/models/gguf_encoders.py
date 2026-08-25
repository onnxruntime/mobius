# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Specialized stateless encoder graphs matching llama.cpp GGUF architectures.

The models in this module consume the exact float tensor layouts emitted for
EuroBERT, NeoBERT, dense NomicBERT, and JinaBERT-v2. They expose token-level
hidden states through the standard feature-extraction ABI.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components import Embedding, LayerNorm, Linear, RMSNorm, create_padding_mask
from mobius.components._activations import ACT2FN
from mobius.components._rotary_embedding import apply_rotary_pos_emb, initialize_rope

if TYPE_CHECKING:
    import onnx_ir as ir


def _position_embeddings(op: OpBuilder, input_ids: ir.Value, rotary_emb):
    """Build full-sequence RoPE coordinates for a stateless encoder."""
    seq_len = op.Shape(input_ids, start=1, end=2)
    position_ids = op.Range(
        op.Constant(value_int=0),
        op.Squeeze(seq_len),
        op.Constant(value_int=1),
    )
    return rotary_emb(op, op.Unsqueeze(op.Cast(position_ids, to=7), [0]))


def _bidirectional_alibi_bias(
    op: OpBuilder,
    input_ids: ir.Value,
    padding_mask: ir.Value,
    num_heads: int,
):
    """Create llama.cpp-compatible non-causal ALiBi plus the request padding mask."""
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))
    slopes = [base**i for i in range(1, closest_power_of_2 + 1)]
    if closest_power_of_2 != num_heads:
        extra_base = 2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3)))
        remaining = min(closest_power_of_2, num_heads - closest_power_of_2)
        slopes.extend(extra_base**i for i in range(1, 2 * remaining + 1, 2))

    seq_len = op.Squeeze(op.Shape(input_ids, start=1, end=2))
    positions = op.Cast(
        op.Range(op.Constant(value_int=0), seq_len, op.Constant(value_int=1)),
        to=1,
    )
    distance = op.Abs(op.Sub(op.Unsqueeze(positions, [1]), op.Unsqueeze(positions, [0])))
    slopes_const = op.Constant(
        value_floats=np.asarray(slopes[:num_heads], dtype=np.float32).tolist()
    )
    alibi = op.Neg(
        op.Mul(
            op.Unsqueeze(slopes_const, [0, 2, 3]),
            op.Unsqueeze(distance, [0, 1]),
        )
    )  # (1, heads, sequence, sequence)
    return op.Where(padding_mask, alibi, op.CastLike(-10000.0, alibi))


class _SplitAttention(nn.Module):
    """Bias-configurable split Q/K/V bidirectional attention."""

    def __init__(
        self,
        config: ArchitectureConfig,
        *,
        rope: bool,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope = rope
        self.q = Linear(config.hidden_size, config.hidden_size, bias=config.encoder_q_bias)
        self.k = Linear(config.hidden_size, config.hidden_size, bias=config.encoder_k_bias)
        self.v = Linear(config.hidden_size, config.hidden_size, bias=config.encoder_v_bias)
        self.output = Linear(config.hidden_size, config.hidden_size, bias=config.attn_o_bias)
        if qk_norm:
            # Jina normalizes the complete projected width before reshaping to heads.
            self.q_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.k_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._qk_norm = qk_norm

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value,
        position_embeddings=None,
    ):
        query = self.q(op, hidden_states)
        key = self.k(op, hidden_states)
        value = self.v(op, hidden_states)
        if self._qk_norm:
            query = self.q_norm(op, query)
            key = self.k_norm(op, key)
        if self.rope:
            query = apply_rotary_pos_emb(
                op, query, position_embeddings, num_heads=self.num_heads
            )
            key = apply_rotary_pos_emb(op, key, position_embeddings, num_heads=self.num_heads)
        attn_output = op.Attention(
            query,
            key,
            value,
            attention_mask,
            q_num_heads=self.num_heads,
            kv_num_heads=self.num_heads,
            scale=float(self.head_dim**-0.5),
        )
        return self.output(op, attn_output)


class _PackedAttention(nn.Module):
    """Bias-free packed-QKV bidirectional RoPE attention used by NeoBERT."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.qkv = Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.output = Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, op, hidden_states, attention_mask, position_embeddings):
        query, key, value = op.Split(
            self.qkv(op, hidden_states), axis=-1, num_outputs=3, _outputs=3
        )
        query = apply_rotary_pos_emb(op, query, position_embeddings, num_heads=self.num_heads)
        key = apply_rotary_pos_emb(op, key, position_embeddings, num_heads=self.num_heads)
        attended = op.Attention(
            query,
            key,
            value,
            attention_mask,
            q_num_heads=self.num_heads,
            kv_num_heads=self.num_heads,
            scale=float(self.head_dim**-0.5),
        )
        return self.output(op, attended)


class _ParallelGatedMLP(nn.Module):
    """Parallel gate/up MLP with an architecture-selected activation."""

    def __init__(self, config: ArchitectureConfig, activation: str):
        super().__init__()
        self.gate = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up = Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.encoder_ffn_up_bias,
        )
        self.down = Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.encoder_ffn_down_bias,
        )
        self._act = ACT2FN[activation]

    def forward(self, op, hidden_states):
        return self.down(
            op, op.Mul(self._act(op, self.gate(op, hidden_states)), self.up(op, hidden_states))
        )


class _FusedGatedMLP(nn.Module):
    """Fused gate/up input projection used by NeoBERT and some JinaBERT-v2 files."""

    def __init__(
        self,
        config: ArchitectureConfig,
        activation: str,
        *,
        up_bias: bool = False,
        down_bias: bool = False,
    ):
        super().__init__()
        self.up = Linear(config.hidden_size, 2 * config.intermediate_size, bias=up_bias)
        self.down = Linear(config.intermediate_size, config.hidden_size, bias=down_bias)
        self._act = ACT2FN[activation]

    def forward(self, op, hidden_states):
        gate, up = op.Split(self.up(op, hidden_states), axis=-1, num_outputs=2, _outputs=2)
        return self.down(op, op.Mul(self._act(op, gate), up))


class _PreNormLayer(nn.Module):
    """RMS pre-norm attention and FFN layer shared by EuroBERT and NeoBERT."""

    def __init__(self, config: ArchitectureConfig, *, packed: bool):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = (
            _PackedAttention(config) if packed else _SplitAttention(config, rope=True)
        )
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = (
            _FusedGatedMLP(config, "silu") if packed else _ParallelGatedMLP(config, "silu")
        )

    def forward(self, op, hidden_states, attention_mask, position_embeddings):
        attention_output = self.attention(
            op, self.attn_norm(op, hidden_states), attention_mask, position_embeddings
        )
        hidden_states = op.Add(hidden_states, attention_output)
        return op.Add(hidden_states, self.mlp(op, self.ffn_norm(op, hidden_states)))


class _PreNormEncoder(nn.Module):
    """Shared pre-RMSNorm RoPE encoder body."""

    default_task = "feature-extraction"
    category = "encoder"

    def __init__(self, config: ArchitectureConfig, *, packed: bool):
        super().__init__()
        self.token_embeddings = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [_PreNormLayer(config, packed=packed) for _ in range(config.num_hidden_layers)]
        )
        self.output_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(self, op, input_ids, attention_mask, token_type_ids):
        del token_type_ids
        hidden_states = self.token_embeddings(op, input_ids)
        padding_mask = create_padding_mask(op, input_ids, attention_mask)
        positions = _position_embeddings(op, input_ids, self.rotary_emb)
        for layer in self.layers:
            hidden_states = layer(op, hidden_states, padding_mask, positions)
        return self.output_norm(op, hidden_states)

    def preprocess_weights(self, state_dict: dict[str, torch.Tensor]):
        return state_dict


class EuroBertGGUFModel(_PreNormEncoder):
    """EuroBERT's bias-free split-QKV pre-RMSNorm RoPE encoder."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config, packed=False)


class NeoBertGGUFModel(_PreNormEncoder):
    """NeoBERT's packed-QKV and fused-SwiGLU pre-RMSNorm RoPE encoder."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config, packed=True)


class _PostNormEncoderLayer(nn.Module):
    """BERT-style post-norm layer with a parallel gated feed-forward branch."""

    def __init__(self, config: ArchitectureConfig, *, jina: bool):
        super().__init__()
        self.attention = _SplitAttention(
            config,
            rope=not jina,
            qk_norm=jina and config.encoder_qk_norm,
        )
        self.attention_output_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._extra_attention_norm = jina and config.encoder_extra_attention_norm
        if self._extra_attention_norm:
            self.extra_attention_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        if jina and config.encoder_fused_geglu:
            self.mlp = _FusedGatedMLP(
                config,
                "gelu",
                up_bias=config.encoder_ffn_up_bias,
                down_bias=config.encoder_ffn_down_bias,
            )
        else:
            self.mlp = _ParallelGatedMLP(config, "gelu" if jina else "silu")
        self.layer_output_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, op, hidden_states, attention_mask, position_embeddings=None):
        residual = hidden_states
        attention_output = self.attention(
            op, hidden_states, attention_mask, position_embeddings
        )
        hidden_states = self.attention_output_norm(op, op.Add(residual, attention_output))
        if self._extra_attention_norm:
            # llama.cpp's optional attn_norm_2 path re-adds the original layer input.
            hidden_states = self.extra_attention_norm(op, op.Add(hidden_states, residual))
        return self.layer_output_norm(op, op.Add(hidden_states, self.mlp(op, hidden_states)))


class _PostNormEncoder(nn.Module):
    """Shared embedding/post-norm body for dense NomicBERT and JinaBERT-v2."""

    default_task = "feature-extraction"
    category = "encoder"

    def __init__(self, config: ArchitectureConfig, *, jina: bool):
        super().__init__()
        self._jina = jina
        self.token_embeddings = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self._use_token_types = config.encoder_use_token_type_embeddings
        if self._use_token_types:
            self.token_type_embeddings = Embedding(config.type_vocab_size, config.hidden_size)
        self.token_embeddings_norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList(
            [_PostNormEncoderLayer(config, jina=jina) for _ in range(config.num_hidden_layers)]
        )
        if not jina:
            self.rotary_emb = initialize_rope(config)

    def forward(self, op, input_ids, attention_mask, token_type_ids):
        hidden_states = self.token_embeddings(op, input_ids)
        if self._use_token_types:
            # llama.cpp currently fixes every request to token type zero ("Sentence A").
            zero_types = op.Mul(token_type_ids, op.Constant(value_int=0))
            hidden_states = op.Add(hidden_states, self.token_type_embeddings(op, zero_types))
        hidden_states = self.token_embeddings_norm(op, hidden_states)
        padding_mask = create_padding_mask(op, input_ids, attention_mask)
        if self._jina:
            attention_bias = _bidirectional_alibi_bias(
                op, input_ids, padding_mask, self.layers[0].attention.num_heads
            )
            positions = None
        else:
            attention_bias = padding_mask
            positions = _position_embeddings(op, input_ids, self.rotary_emb)
        for layer in self.layers:
            hidden_states = layer(op, hidden_states, attention_bias, positions)
        return hidden_states

    def preprocess_weights(self, state_dict: dict[str, torch.Tensor]):
        return state_dict


class NomicBertGGUFModel(_PostNormEncoder):
    """Dense NomicBERT with embedding/post LayerNorm, RoPE, and gated FFNs."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config, jina=False)


class JinaBertV2GGUFModel(_PostNormEncoder):
    """JinaBERT-v2 ALiBi encoder for the strictly validated dense tensor variants."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config, jina=True)
