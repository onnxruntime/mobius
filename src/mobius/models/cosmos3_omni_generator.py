# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""NVIDIA Cosmos3-Omni unified MoT transformer (Reasoner + Generator).

This module builds the **complete neural** ``Cosmos3OmniTransformer`` graph —
the unified Mixture-of-Transformers (MoT) backbone that carries the
understanding ("und" / Reasoner) expert and the rectified-flow diffusion
generation ("gen" / Generator) expert in a *single* stack of layers, plus the
vision, optional Sound and optional Action projection heads.

It is deliberately **not** the reasoner-only Qwen3-VL wrapper in
:mod:`mobius.models.cosmos3_omni`: that module exports the understanding tower
alone (``input_ids -> logits + KV cache``) and drops every generator weight.
Here the generator is the point — a single denoising step over a packed joint
token sequence.

Architecture reference
----------------------
Translated from ``huggingface/diffusers``
``src/diffusers/models/transformers/transformer_cosmos3.py``
(Apache License 2.0, Copyright 2025 The NVIDIA Team and The HuggingFace Team)
and the public ``nvidia/Cosmos3-Nano`` ``transformer/config.json``.  The
architecture (module structure, parameter names, dual-pathway attention,
interleaved 3-axis mRoPE, timestep handling) is derived from that source; the
ONNX graph construction below is an independent implementation using
``onnxscript.nn``.

Per-layer dataflow
------------------
Every layer holds two complete expert weight sets and mixes them only inside
attention::

    und_norm = input_layernorm(und)               gen_norm = input_layernorm_moe_gen(gen)
    q,k,v    = to_q/to_k/to_v(und_norm)           q',k',v' = add_{q,k,v}_proj(gen_norm)
    q,k      = norm_q/norm_k                      q',k'    = norm_added_q/norm_added_k
    k_ufg    = k_norm_und_for_gen(k)  (optional)
    <mRoPE on q, k, k_ufg with und positions>     <mRoPE on q', k' with gen positions>
    und_attn = Attention(q, k, v, causal)         gen_attn = Attention(q', [k_ufg;k'], [v;v'])
    und     += to_out(und_attn)                   gen    += to_add_out(gen_attn)
    und     += mlp(post_attention_layernorm)      gen   += mlp_moe_gen(post_attention_layernorm_moe_gen)

The understanding pathway is strictly causal and self-contained; the
generation pathway is non-causal and cross-attends over the concatenated
understanding + generation keys/values.

Packed ONNX contract
--------------------
Upstream's ``forward`` takes Python lists of ragged per-item tensors
(``vision_tokens: list[torch.Tensor]``, ``vision_token_shapes: list[tuple]``,
``vision_noisy_frame_indexes: list[torch.Tensor]``, ...) and performs
patchify/unpatchify using host-side shape arithmetic.  Python lists and
host-resident shape values cannot appear in an ONNX signature, so this graph
is defined at the **packed-token boundary**:

* everything upstream does *before* ``proj_in`` / ``audio_proj_in`` /
  ``action_proj_in`` (per-item patchify, padding, packing, and the flattening
  of ``noisy_frame_indexes`` into row offsets) is host preprocessing;
* everything upstream does *after* ``proj_out`` / ``audio_proj_out`` /
  ``action_proj_out`` (unpatchify, per-item scatter back into ``[C, T, H, W]``
  buffers) is host postprocessing;
* the entire neural computation in between is in the graph, with no semantics
  dropped: the flattened noisy row offsets that ``_apply_timestep_embeds_to_
  noisy_tokens`` computes from ``token_shapes`` become an explicit
  ``*_timestep_token_indexes`` int64 input, and ``und_len`` becomes an
  explicit int64 input instead of a Python ``int``.

See :class:`~mobius.tasks._cosmos3_omni_generator.Cosmos3OmniGeneratorTask`
for the exact input/output names, dtypes and shapes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius._configs._cosmos3_omni_generator import Cosmos3OmniGeneratorConfig
from mobius.components import (
    FCMLP,
    INT64_MAX,
    Embedding,
    GatedMLP,
    Linear,
    RMSNorm,
    TimestepEmbedding,
)

if TYPE_CHECKING:
    import torch


def _const_ints(op: OpBuilder, values: list[int]) -> ir.Value:
    """Materialize a 1-D int64 constant."""
    return op.Constant(value_ints=values)


def _apply_rope(
    op: OpBuilder, x: ir.Value, cos: ir.Value, sin: ir.Value, num_heads: int
) -> ir.Value:
    """Apply half-split rotary embedding to a flat multi-head tensor.

    The opset-24 ``RotaryEmbedding`` op with the half-width cos/sin cache and
    ``interleaved=0`` computes exactly upstream's
    ``x * cos + rotate_half(x) * sin`` (upstream duplicates the frequency table
    with ``cat((freqs, freqs), -1)``; the op consumes the un-duplicated half).

    Args:
        op: ONNX op builder.
        x: ``[1, seq, num_heads * head_dim]``.
        cos: ``[1, seq, head_dim // 2]``.
        sin: ``[1, seq, head_dim // 2]``.
        num_heads: Heads packed into the last axis of *x*.

    Returns:
        ``[1, seq, num_heads * head_dim]`` with rotary applied.
    """
    return op.RotaryEmbedding(x, cos, sin, num_heads=num_heads, interleaved=0)


def _row_indices(op: OpBuilder, indexes: ir.Value) -> ir.Value:
    """Turn a 1-D index vector into ScatterND row indices.

    Args:
        op: ONNX op builder.
        indexes: ``[N]`` int64 row indices.

    Returns:
        ``[N, 1]`` int64 indices addressing whole rows of a 2-D tensor.
    """
    return op.Unsqueeze(indexes, [-1])


class Cosmos3OmniDomainAwareLinear(nn.Module):
    """Per-embodiment-domain linear projection (Action head).

    Replicates upstream ``DomainAwareLinear``: instead of one shared weight,
    an ``nn.Embedding`` table stores one flattened ``[in, out]`` weight matrix
    and one ``[out]`` bias per embodiment domain, and each *token* selects its
    matrix by ``domain_id``.  This is a Gather + batched MatMul, **not** a
    plain Linear — a plain Linear would silently collapse all domains onto one
    weight set.

    Weight layout (matching the published checkpoint exactly)::

        action_proj_in.fc.weight    [num_domains, out_features * in_features]
        action_proj_in.bias.weight  [num_domains, out_features]

    The flattened row is interpreted as ``[in_features, out_features]``
    (upstream does ``self.fc(domain_id).view(D, input_size, output_size)``),
    i.e. it is already transposed relative to ``torch.nn.Linear``.

    Args:
        in_features: Input width.
        out_features: Output width.
        num_domains: Number of embodiment domains in the table.
    """

    def __init__(self, in_features: int, out_features: int, num_domains: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_domains = num_domains
        # Named ``fc`` / ``bias`` so parameters land on ``<name>.fc.weight``
        # and ``<name>.bias.weight`` — the published flat names.
        self.fc = Embedding(num_domains, out_features * in_features)
        self.bias = Embedding(num_domains, out_features)

    def forward(self, op: OpBuilder, x: ir.Value, domain_ids: ir.Value) -> ir.Value:
        """Apply the per-token domain weight.

        Args:
            op: ONNX op builder.
            x: ``[N, in_features]`` packed tokens.
            domain_ids: ``[N]`` int64 embodiment domain id per token.

        Returns:
            ``[N, out_features]`` projected tokens.
        """
        # [N, out*in] -> [N, in, out]; one weight matrix per token.
        weight = self.fc(op, domain_ids)
        weight = op.Reshape(weight, _const_ints(op, [-1, self.in_features, self.out_features]))
        # [N, out]
        bias = self.bias(op, domain_ids)
        # [N, 1, in] @ [N, in, out] -> [N, 1, out] -> [N, out]
        projected = op.MatMul(op.Unsqueeze(x, [1]), weight)
        projected = op.Squeeze(projected, [1])
        return op.Add(projected, bias)


class Cosmos3OmniTimesteps(nn.Module):
    """Sinusoidal timestep projection (diffusers ``Timesteps``).

    Upstream instantiates ``Timesteps(num_channels=256, flip_sin_to_cos=True,
    downscale_freq_shift=0)``, i.e. ``concat(cos(t * inv_freq), sin(t * inv_freq))``.
    ``inv_freq`` is a derived constant, not a checkpoint weight, and is kept in
    float32 (``_keep_float32``) because the timestep path must stay fp32.

    Args:
        num_channels: Output width (must be even).
        max_period: Sinusoid base period.
        downscale_freq_shift: Upstream's exponent shift (``0`` for Cosmos3).
    """

    def __init__(
        self,
        num_channels: int,
        max_period: float = 10_000.0,
        downscale_freq_shift: float = 0.0,
    ):
        super().__init__()
        half_dim = num_channels // 2
        exponent = -np.log(max_period) * np.arange(half_dim, dtype=np.float32)
        exponent = exponent / (half_dim - downscale_freq_shift)
        self.inv_freq = nn.Parameter(
            [half_dim],
            name="inv_freq",
            data=ir.tensor(np.exp(exponent).astype(np.float32)),
        )
        self.inv_freq._keep_float32 = True

    def forward(self, op: OpBuilder, timesteps: ir.Value) -> ir.Value:
        """Project ``[N]`` float32 timesteps to ``[N, num_channels]`` float32."""
        # [N, 1] * [half_dim] -> [N, half_dim]
        freqs = op.Mul(op.Unsqueeze(timesteps, [-1]), self.inv_freq)
        # flip_sin_to_cos=True puts the cosine half first.
        return op.Concat(op.Cos(freqs), op.Sin(freqs), axis=-1)


class Cosmos3OmniRotaryEmbedding(nn.Module):
    """Interleaved 3-axis mRoPE over a packed joint sequence.

    Reproduces upstream ``Cosmos3VLTextRotaryEmbedding``.  Frequencies for the
    three position axes (T, H, W) are computed independently in **float32**
    (upstream disables autocast here precisely because bf16 cannot represent
    consecutive integers past 256), then merged channel-wise using the
    *interleaved* layout::

        channel c takes the H axis  if c in range(1, rope_axes_dim[1] * 3, 3)
        channel c takes the W axis  if c in range(2, rope_axes_dim[2] * 3, 3)
        channel c takes the T axis  otherwise

    For ``head_dim=128`` / ``rope_axes_dim=(24, 20, 20)`` this yields the
    ``[T H W T H W ... T T]`` pattern with exactly 24/20/20 channels.

    The merged ``[seq, head_dim // 2]`` frequency table is turned into cos/sin
    and cast to the model dtype; the ONNX ``RotaryEmbedding`` op consumes the
    half-width cache directly (upstream's ``cat((freqs, freqs), -1)`` is the
    same rotation in the ``rotate_half`` convention).

    Args:
        config: Cosmos3-Omni generator configuration.
    """

    def __init__(self, config: Cosmos3OmniGeneratorConfig):
        super().__init__()
        self._dtype = config.dtype
        rotary_dim = config.rotary_dim
        head_dim = config.head_dim

        inv_freq = 1.0 / (
            config.rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
        )
        self.inv_freq = nn.Parameter(
            [rotary_dim], name="inv_freq", data=ir.tensor(inv_freq.astype(np.float32))
        )
        # Rotary frequencies must stay fp32 even for a bf16 model.
        self.inv_freq._keep_float32 = True

        h_mask = np.zeros(rotary_dim, dtype=np.bool_)
        w_mask = np.zeros(rotary_dim, dtype=np.bool_)
        for channel in range(1, config.rope_axes_dim[1] * 3, 3):
            if channel < rotary_dim:
                h_mask[channel] = True
        for channel in range(2, config.rope_axes_dim[2] * 3, 3):
            if channel < rotary_dim:
                w_mask[channel] = True
        self.h_mask = nn.Parameter([rotary_dim], name="h_mask", data=ir.tensor(h_mask))
        self.w_mask = nn.Parameter([rotary_dim], name="w_mask", data=ir.tensor(w_mask))

    def forward(self, op: OpBuilder, position_ids: ir.Value) -> tuple[ir.Value, ir.Value]:
        """Compute cos/sin for every joint-sequence row.

        Args:
            op: ONNX op builder.
            position_ids: ``[3, sequence_length]`` int64 (T, H, W) positions.

        Returns:
            ``(cos, sin)``, each ``[1, sequence_length, head_dim // 2]`` in the
            model dtype — the layout the ONNX ``RotaryEmbedding`` op expects
            when no ``position_ids`` input is supplied.
        """
        # [3, seq] -> [3, seq, 1] * [rotary_dim] -> [3, seq, rotary_dim], fp32.
        positions = op.Cast(position_ids, to=ir.DataType.FLOAT)
        freqs = op.Mul(op.Unsqueeze(positions, [-1]), self.inv_freq)

        freqs_t = op.Squeeze(op.Gather(freqs, [0], axis=0), [0])  # (seq, rotary_dim)
        freqs_h = op.Squeeze(op.Gather(freqs, [1], axis=0), [0])
        freqs_w = op.Squeeze(op.Gather(freqs, [2], axis=0), [0])

        # Interleaved channel selection: H and W overwrite disjoint channel
        # subsets of the T table, exactly like upstream's in-place slice write.
        merged = op.Where(self.h_mask, freqs_h, freqs_t)
        merged = op.Where(self.w_mask, freqs_w, merged)

        cos = op.Unsqueeze(op.Cos(merged), [0])  # (1, seq, rotary_dim)
        sin = op.Unsqueeze(op.Sin(merged), [0])
        if self._dtype != ir.DataType.FLOAT:
            cos = op.Cast(cos, to=self._dtype)
            sin = op.Cast(sin, to=self._dtype)
        return cos, sin


class Cosmos3OmniMoTAttention(nn.Module):
    """Dual-pathway packed MoT attention (upstream ``Cosmos3PackedMoTAttention``).

    Holds two complete projection sets.  The understanding set
    (``to_q``/``to_k``/``to_v``/``to_out`` + ``norm_q``/``norm_k``) runs a
    causal self-attention over the understanding tokens only.  The generation
    set (``add_q_proj``/``add_k_proj``/``add_v_proj``/``to_add_out`` +
    ``norm_added_q``/``norm_added_k``) runs a *non-causal* attention whose
    keys/values are ``concat(understanding, generation)``.

    Args:
        config: Cosmos3-Omni generator configuration.
    """

    def __init__(self, config: Cosmos3OmniGeneratorConfig):
        super().__init__()
        self.config = config
        self._num_heads = config.num_attention_heads
        self._num_kv_heads = config.num_key_value_heads
        self._head_dim = config.head_dim
        self._scale = float(config.head_dim**-0.5)
        hidden_size = config.hidden_size
        q_size = config.attention_out_size
        kv_size = config.key_value_size
        bias = config.attention_bias
        eps = config.rms_norm_eps

        # Understanding ("und") pathway.
        self.to_q = Linear(hidden_size, q_size, bias=bias)
        self.to_k = Linear(hidden_size, kv_size, bias=bias)
        self.to_v = Linear(hidden_size, kv_size, bias=bias)
        self.to_out = Linear(q_size, hidden_size, bias=bias)
        # Per-head QK RMSNorm over head_dim only (no reshape needed after it).
        self.norm_q = RMSNorm(config.head_dim, eps=eps) if config.qk_norm_for_text else None
        self.norm_k = RMSNorm(config.head_dim, eps=eps) if config.qk_norm_for_text else None
        # Separate norm for the understanding keys consumed by the generator.
        self.k_norm_und_for_gen = (
            RMSNorm(config.head_dim, eps=eps) if config.has_und_k_norm_for_gen else None
        )

        # Generation ("gen") pathway.
        self.add_q_proj = Linear(hidden_size, q_size, bias=bias)
        self.add_k_proj = Linear(hidden_size, kv_size, bias=bias)
        self.add_v_proj = Linear(hidden_size, kv_size, bias=bias)
        self.to_add_out = Linear(q_size, hidden_size, bias=bias)
        self.norm_added_q = RMSNorm(config.head_dim, eps=eps)
        self.norm_added_k = RMSNorm(config.head_dim, eps=eps)

    def _norm_per_head(
        self, op: OpBuilder, projected: ir.Value, norm: RMSNorm | None, num_heads: int
    ) -> ir.Value:
        """RMS-normalize each head slice of a flat ``[N, num_heads * head_dim]``."""
        if norm is None:
            return projected
        # (N, num_heads * head_dim) -> (N, num_heads, head_dim)
        heads = op.Reshape(projected, _const_ints(op, [-1, num_heads, self._head_dim]))
        heads = norm(op, heads)
        # back to (N, num_heads * head_dim)
        return op.Reshape(heads, _const_ints(op, [-1, num_heads * self._head_dim]))

    def forward(
        self,
        op: OpBuilder,
        und_seq: ir.Value,
        gen_seq: ir.Value,
        rotary_emb: tuple[ir.Value, ir.Value, ir.Value, ir.Value],
    ) -> tuple[ir.Value, ir.Value]:
        """Run both attention pathways.

        Args:
            op: ONNX op builder.
            und_seq: ``[und_len, hidden_size]`` pre-normalized understanding tokens.
            gen_seq: ``[gen_len, hidden_size]`` pre-normalized generation tokens.
            rotary_emb: ``(cos_und, sin_und, cos_gen, sin_gen)``, each
                ``[1, *, head_dim // 2]``.

        Returns:
            ``(und_out, gen_out)``, each ``[*, hidden_size]``.
        """
        cos_und, sin_und, cos_gen, sin_gen = rotary_emb

        # --- projections: (seq, hidden) -> (seq, heads * head_dim) ---
        q_und = self._norm_per_head(op, self.to_q(op, und_seq), self.norm_q, self._num_heads)
        k_und = self._norm_per_head(
            op, self.to_k(op, und_seq), self.norm_k, self._num_kv_heads
        )
        v_und = self.to_v(op, und_seq)
        q_gen = self._norm_per_head(
            op, self.add_q_proj(op, gen_seq), self.norm_added_q, self._num_heads
        )
        k_gen = self._norm_per_head(
            op, self.add_k_proj(op, gen_seq), self.norm_added_k, self._num_kv_heads
        )
        v_gen = self.add_v_proj(op, gen_seq)

        # The generator may consume a separately normalized copy of the
        # understanding keys (upstream ``k_norm_und_for_gen``).
        k_und_for_gen = self._norm_per_head(
            op, k_und, self.k_norm_und_for_gen, self._num_kv_heads
        )

        # --- rotary: the op wants (batch, seq, heads * head_dim) ---
        q_und = op.Unsqueeze(q_und, [0])
        k_und_b = op.Unsqueeze(k_und, [0])
        v_und = op.Unsqueeze(v_und, [0])
        q_gen = op.Unsqueeze(q_gen, [0])
        k_gen = op.Unsqueeze(k_gen, [0])
        v_gen = op.Unsqueeze(v_gen, [0])

        und_pos = (cos_und, sin_und)
        gen_pos = (cos_gen, sin_gen)
        q_und = _apply_rope(op, q_und, *und_pos, self._num_heads)
        k_und_roped = _apply_rope(op, k_und_b, *und_pos, self._num_kv_heads)
        if self.k_norm_und_for_gen is None:
            # Same tensor upstream — reuse the roped keys instead of recomputing.
            k_und_for_gen_roped = k_und_roped
        else:
            k_und_for_gen_roped = _apply_rope(
                op, op.Unsqueeze(k_und_for_gen, [0]), *und_pos, self._num_kv_heads
            )
        q_gen = _apply_rope(op, q_gen, *gen_pos, self._num_heads)
        k_gen = _apply_rope(op, k_gen, *gen_pos, self._num_kv_heads)

        # --- causal understanding self-attention (GQA) ---
        und_attn = op.Attention(
            q_und,
            k_und_roped,
            v_und,
            q_num_heads=self._num_heads,
            kv_num_heads=self._num_kv_heads,
            is_causal=1,
            scale=self._scale,
        )

        # --- non-causal generation attention over [und ; gen] K/V (GQA) ---
        all_k = op.Concat(k_und_for_gen_roped, k_gen, axis=1)
        all_v = op.Concat(v_und, v_gen, axis=1)
        gen_attn = op.Attention(
            q_gen,
            all_k,
            all_v,
            q_num_heads=self._num_heads,
            kv_num_heads=self._num_kv_heads,
            is_causal=0,
            scale=self._scale,
        )

        # (1, seq, heads * head_dim) -> (seq, hidden_size)
        und_out = self.to_out(op, op.Squeeze(und_attn, [0]))
        gen_out = self.to_add_out(op, op.Squeeze(gen_attn, [0]))
        return und_out, gen_out


class Cosmos3OmniMoTDecoderLayer(nn.Module):
    """One MoT decoder layer holding both expert weight sets.

    Args:
        config: Cosmos3-Omni generator configuration.
    """

    def __init__(self, config: Cosmos3OmniGeneratorConfig):
        super().__init__()
        eps = config.rms_norm_eps
        self.self_attn = Cosmos3OmniMoTAttention(config)
        self.mlp = _build_mlp(config)
        self.mlp_moe_gen = _build_mlp(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.input_layernorm_moe_gen = RMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm_moe_gen = RMSNorm(config.hidden_size, eps=eps)

    def forward(
        self,
        op: OpBuilder,
        und_seq: ir.Value,
        gen_seq: ir.Value,
        rotary_emb: tuple[ir.Value, ir.Value, ir.Value, ir.Value],
    ) -> tuple[ir.Value, ir.Value]:
        """Apply pre-norm attention and per-expert feed-forward with residuals.

        Args:
            op: ONNX op builder.
            und_seq: ``[und_len, hidden_size]`` understanding hidden states.
            gen_seq: ``[gen_len, hidden_size]`` generation hidden states.
            rotary_emb: ``(cos_und, sin_und, cos_gen, sin_gen)``.

        Returns:
            Updated ``(und_seq, gen_seq)``.
        """
        und_attn, gen_attn = self.self_attn(
            op,
            self.input_layernorm(op, und_seq),
            self.input_layernorm_moe_gen(op, gen_seq),
            rotary_emb,
        )
        und_residual = op.Add(und_seq, und_attn)
        gen_residual = op.Add(gen_seq, gen_attn)

        und_mlp = self.mlp(op, self.post_attention_layernorm(op, und_residual))
        gen_mlp = self.mlp_moe_gen(op, self.post_attention_layernorm_moe_gen(op, gen_residual))
        return op.Add(und_residual, und_mlp), op.Add(gen_residual, gen_mlp)


def _build_mlp(config: Cosmos3OmniGeneratorConfig) -> nn.Module:
    """Create the per-expert feed-forward matching ``hidden_act``.

    ``silu`` -> gated SwiGLU (``gate_proj``/``up_proj``/``down_proj``);
    ``relu2`` -> non-gated squared ReLU (``up_proj``/``down_proj``).  Both
    parameter layouts match the published checkpoint names.
    """
    if config.is_gated_mlp:
        return GatedMLP(
            config.hidden_size, config.intermediate_size, activation="silu", bias=False
        )
    return FCMLP(config.hidden_size, config.intermediate_size, activation="relu2", bias=False)


# Keys present in the published ``transformer/`` checkpoint that carry no
# neural computation in this graph.  ``lm_head`` is constructed by the
# upstream module but never called in ``Cosmos3OmniTransformer.forward`` — the
# understanding logits come from the separately exported Reasoner
# (:mod:`mobius.models.cosmos3_omni`), so it is dead weight here.
_UNUSED_PUBLISHED_KEYS: frozenset[str] = frozenset({"lm_head.weight"})

# Non-persistent buffers that some export paths materialize into the state
# dict.  They are recomputed as graph constants, so a checkpoint copy is
# redundant rather than unexpected.
_RECOMPUTED_BUFFER_SUFFIXES: tuple[str, ...] = (".inv_freq", ".h_mask", ".w_mask")

# Reasoner (understanding-tower) vision-encoder prefixes.  A *unified*
# Cosmos3-Omni checkpoint carries them alongside the transformer; they belong
# to the separately exported vision tower, never to this graph.
_REASONER_VISION_PREFIXES: tuple[str, ...] = (
    "visual.",
    "projector.",
    "blocks.",
    "patch_embed.",
    "merger.",
    "deepstack_merger_list.",
    "pos_embed",
)

# Cosmos3-Edge Policy checkpoints carry a tiny framework sidecar with duplicate
# generator-facing key-norm tensors under nested training-framework paths.
_EDGE_FRAMEWORK_K_NORM_PREFIXES: tuple[str, ...] = (
    "net.language_model.model.layers.",
    "layers.layers.",
)

# Accepted container prefixes on a flat published state dict.
_STRIPPABLE_PREFIXES: tuple[str, ...] = ("transformer.", "model.")


class Cosmos3OmniGeneratorModel(nn.Module):
    """Unified Cosmos3-Omni MoT transformer — one denoising step.

    Builds the whole neural graph: token embedding, per-modality input
    projections with timestep conditioning, interleaved 3-axis mRoPE, the
    dual-expert layer stack, the two final norms, and the per-modality output
    projections.

    Parameter names match the published flat ``transformer/`` checkpoint
    one-for-one (``embed_tokens.weight``, ``layers.N.self_attn.to_q.weight``,
    ``layers.N.mlp_moe_gen.down_proj.weight``, ``norm_moe_gen.weight``,
    ``proj_in.bias``, ``time_embedder.linear_1.weight``,
    ``audio_proj_in.weight``, ``action_proj_in.fc.weight``, ...), so
    :meth:`preprocess_weights` is a validation + normalization pass rather
    than a rename table.

    Args:
        config: Cosmos3-Omni generator configuration.
    """

    default_task: str = "cosmos3-omni-generator"
    category: str = "Multimodal"

    def __init__(self, config: Cosmos3OmniGeneratorConfig):
        super().__init__()
        config.validate()
        self.config = config
        self._dtype = config.dtype

        # Shared backbone.
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Cosmos3OmniMoTDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_moe_gen = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Cosmos3OmniRotaryEmbedding(config)

        # Vision (diffusion) head + timestep conditioning.
        self.proj_in = Linear(config.patch_latent_dim, config.hidden_size, bias=True)
        self.proj_out = Linear(config.hidden_size, config.patch_latent_dim, bias=True)
        self.time_proj = Cosmos3OmniTimesteps(config.time_proj_channels)
        self.time_embedder = TimestepEmbedding(config.time_proj_channels, config.hidden_size)
        # Upstream keeps ``time_embedder`` in fp32 (``_keep_in_fp32_modules``);
        # the sinusoid -> MLP path must not run in bf16.
        for parameter in self.time_embedder.parameters():
            parameter._keep_float32 = True

        # Optional Sound head.
        if config.sound_gen:
            assert config.sound_dim is not None  # guaranteed by config.validate()
            self.audio_proj_in = Linear(config.sound_dim, config.hidden_size, bias=True)
            self.audio_proj_out = Linear(config.hidden_size, config.sound_dim, bias=True)
            self.audio_modality_embed = nn.Parameter(
                [config.hidden_size], name="audio_modality_embed"
            )

        # Optional Action head (per-embodiment-domain weights).
        if config.action_gen:
            assert config.action_dim is not None  # guaranteed by config.validate()
            self.action_proj_in = Cosmos3OmniDomainAwareLinear(
                config.action_dim, config.hidden_size, config.num_embodiment_domains
            )
            self.action_proj_out = Cosmos3OmniDomainAwareLinear(
                config.hidden_size, config.action_dim, config.num_embodiment_domains
            )
            self.action_modality_embed = nn.Parameter(
                [config.hidden_size], name="action_modality_embed"
            )

    # ------------------------------------------------------------------
    # Graph construction helpers
    # ------------------------------------------------------------------

    def _timestep_embedding(self, op: OpBuilder, timesteps: ir.Value) -> ir.Value:
        """Embed diffusion timesteps, keeping the whole path in float32.

        Args:
            op: ONNX op builder.
            timesteps: ``[N]`` **float32** timesteps (pre-``timestep_scale``).

        Returns:
            ``[N, hidden_size]`` embeddings cast to the model dtype — the cast
            happens only at the boundary where the result is added to the
            model-dtype token stream.
        """
        scaled = op.Mul(timesteps, float(self.config.timestep_scale))
        projected = self.time_proj(op, scaled)
        embedded = self.time_embedder(op, projected)
        if self._dtype != ir.DataType.FLOAT:
            embedded = op.Cast(embedded, to=self._dtype)
        return embedded

    def _add_timestep_embeds(
        self,
        op: OpBuilder,
        tokens: ir.Value,
        timesteps: ir.Value,
        token_indexes: ir.Value,
    ) -> ir.Value:
        """Scatter-add timestep embeddings onto the noisy rows of a token block.

        Tensorizes upstream ``_apply_timestep_embeds_to_noisy_tokens``: the
        host precomputes the flattened noisy row offsets (which upstream
        derives from ``noisy_frame_indexes`` + ``token_shapes``) and passes
        them as ``token_indexes``.  An empty ``token_indexes`` is a no-op,
        matching upstream's ``if action_mse_loss_indexes.numel() > 0`` guard.

        Args:
            op: ONNX op builder.
            tokens: ``[N_tokens, hidden_size]`` projected modality tokens.
            timesteps: ``[N_noisy]`` float32 timesteps.
            token_indexes: ``[N_noisy]`` int64 rows of *tokens* to add into.

        Returns:
            ``[N_tokens, hidden_size]`` tokens with timestep conditioning added.
        """
        embeds = self._timestep_embedding(op, timesteps)
        return op.ScatterND(tokens, _row_indices(op, token_indexes), embeds, reduction="add")

    def _scatter_into_joint(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        tokens: ir.Value,
        sequence_indexes: ir.Value,
    ) -> ir.Value:
        """Write a packed modality block into its joint-sequence rows."""
        return op.ScatterND(hidden_states, _row_indices(op, sequence_indexes), tokens)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        op: OpBuilder,
        *,
        input_ids: ir.Value,
        text_indexes: ir.Value,
        position_ids: ir.Value,
        und_len: ir.Value,
        vision_tokens: ir.Value,
        vision_sequence_indexes: ir.Value,
        vision_timesteps: ir.Value,
        vision_timestep_token_indexes: ir.Value,
        vision_mse_loss_indexes: ir.Value,
        sound_tokens: ir.Value | None = None,
        sound_sequence_indexes: ir.Value | None = None,
        sound_timesteps: ir.Value | None = None,
        sound_timestep_token_indexes: ir.Value | None = None,
        sound_mse_loss_indexes: ir.Value | None = None,
        action_tokens: ir.Value | None = None,
        action_domain_ids: ir.Value | None = None,
        action_sequence_indexes: ir.Value | None = None,
        action_timesteps: ir.Value | None = None,
        action_timestep_token_indexes: ir.Value | None = None,
        action_mse_loss_indexes: ir.Value | None = None,
        action_pred_domain_ids: ir.Value | None = None,
    ) -> tuple[ir.Value, ir.Value | None, ir.Value | None]:
        """Run one unified denoising step over the packed joint sequence.

        Args:
            op: ONNX op builder.
            input_ids: ``[num_text_tokens]`` int64 text token ids.
            text_indexes: ``[num_text_tokens]`` int64 joint-sequence rows for
                the text tokens.
            position_ids: ``[3, sequence_length]`` int64 mRoPE (T, H, W)
                positions covering the *whole* joint sequence.  Its second
                dimension defines ``sequence_length``.
            und_len: ``[1]`` int64 — number of leading joint-sequence rows
                routed through the understanding expert.  The remainder goes
                through the generation expert.
            vision_tokens: ``[num_vision_tokens, patch_latent_dim]`` packed,
                host-patchified vision latents in the model dtype.
            vision_sequence_indexes: ``[num_vision_tokens]`` int64 joint rows.
            vision_timesteps: ``[num_vision_noisy_tokens]`` float32 timesteps.
            vision_timestep_token_indexes: ``[num_vision_noisy_tokens]`` int64
                rows of ``vision_tokens`` receiving each timestep embedding.
            vision_mse_loss_indexes: ``[num_vision_noisy_tokens]`` int64 joint
                rows to decode into velocity predictions.
            sound_tokens: ``[num_sound_tokens, sound_dim]`` packed sound latents.
            sound_sequence_indexes: ``[num_sound_tokens]`` int64 joint rows.
            sound_timesteps: ``[num_sound_noisy_tokens]`` float32 timesteps.
            sound_timestep_token_indexes: ``[num_sound_noisy_tokens]`` int64
                rows of ``sound_tokens``.
            sound_mse_loss_indexes: ``[num_sound_noisy_tokens]`` int64 joint rows.
            action_tokens: ``[num_action_tokens, action_dim]`` packed actions.
            action_domain_ids: ``[num_action_tokens]`` int64 embodiment domain
                per action token (input projection).
            action_sequence_indexes: ``[num_action_tokens]`` int64 joint rows.
            action_timesteps: ``[num_action_noisy_tokens]`` float32 timesteps.
            action_timestep_token_indexes: ``[num_action_noisy_tokens]`` int64
                rows of ``action_tokens``.
            action_mse_loss_indexes: ``[num_action_noisy_tokens]`` int64 joint rows.
            action_pred_domain_ids: ``[num_action_noisy_tokens]`` int64
                embodiment domain per predicted action token (output projection).

        Returns:
            ``(vision_pred, sound_pred, action_pred)`` where ``vision_pred`` is
            ``[num_vision_noisy_tokens, patch_latent_dim]`` and the optional
            entries are ``[num_*_noisy_tokens, sound_dim | action_dim]`` or
            ``None`` when the corresponding head is disabled in the config.

        Raises:
            ValueError: If a configured head is missing any of its inputs.
        """
        self._require_head_inputs(
            "sound",
            self.config.sound_gen,
            {
                "sound_tokens": sound_tokens,
                "sound_sequence_indexes": sound_sequence_indexes,
                "sound_timesteps": sound_timesteps,
                "sound_timestep_token_indexes": sound_timestep_token_indexes,
                "sound_mse_loss_indexes": sound_mse_loss_indexes,
            },
        )
        self._require_head_inputs(
            "action",
            self.config.action_gen,
            {
                "action_tokens": action_tokens,
                "action_domain_ids": action_domain_ids,
                "action_sequence_indexes": action_sequence_indexes,
                "action_timesteps": action_timesteps,
                "action_timestep_token_indexes": action_timestep_token_indexes,
                "action_mse_loss_indexes": action_mse_loss_indexes,
                "action_pred_domain_ids": action_pred_domain_ids,
            },
        )

        # --- joint hidden-state buffer ------------------------------------
        # sequence_length is read off position_ids so no host int is needed.
        sequence_length = op.Shape(position_ids, start=1, end=2)  # (1,)
        buffer_shape = op.Concat(
            sequence_length, _const_ints(op, [self.config.hidden_size]), axis=0
        )
        hidden_states = op.ConstantOfShape(
            buffer_shape, value=ir.tensor(np.zeros(1, dtype=np.float32))
        )
        if self._dtype != ir.DataType.FLOAT:
            hidden_states = op.Cast(hidden_states, to=self._dtype)

        # --- text tokens ---------------------------------------------------
        # (num_text_tokens, hidden_size) written at their joint rows.
        text_embeds = self.embed_tokens(op, input_ids)
        hidden_states = self._scatter_into_joint(op, hidden_states, text_embeds, text_indexes)

        # --- vision latents ------------------------------------------------
        # (num_vision_tokens, patch_latent_dim) -> (num_vision_tokens, hidden_size)
        vision_hidden = self.proj_in(op, vision_tokens)
        vision_hidden = self._add_timestep_embeds(
            op, vision_hidden, vision_timesteps, vision_timestep_token_indexes
        )
        hidden_states = self._scatter_into_joint(
            op, hidden_states, vision_hidden, vision_sequence_indexes
        )

        # --- sound latents (optional) --------------------------------------
        if self.config.sound_gen:
            # (num_sound_tokens, sound_dim) -> (num_sound_tokens, hidden_size)
            sound_hidden = self.audio_proj_in(op, sound_tokens)
            sound_hidden = op.Add(sound_hidden, self.audio_modality_embed)
            sound_hidden = self._add_timestep_embeds(
                op, sound_hidden, sound_timesteps, sound_timestep_token_indexes
            )
            hidden_states = self._scatter_into_joint(
                op, hidden_states, sound_hidden, sound_sequence_indexes
            )

        # --- action latents (optional) -------------------------------------
        if self.config.action_gen:
            # (num_action_tokens, action_dim) -> (num_action_tokens, hidden_size)
            # via the per-embodiment-domain weight table.
            action_hidden = self.action_proj_in(op, action_tokens, action_domain_ids)
            action_hidden = op.Add(action_hidden, self.action_modality_embed)
            action_hidden = self._add_timestep_embeds(
                op, action_hidden, action_timesteps, action_timestep_token_indexes
            )
            hidden_states = self._scatter_into_joint(
                op, hidden_states, action_hidden, action_sequence_indexes
            )

        # --- mRoPE, then split the joint sequence into the two experts -----
        cos, sin = self.rotary_emb(op, position_ids)  # (1, seq, head_dim // 2)
        zero = _const_ints(op, [0])
        end = _const_ints(op, [INT64_MAX])
        cos_und = op.Slice(cos, zero, und_len, _const_ints(op, [1]))
        sin_und = op.Slice(sin, zero, und_len, _const_ints(op, [1]))
        cos_gen = op.Slice(cos, und_len, end, _const_ints(op, [1]))
        sin_gen = op.Slice(sin, und_len, end, _const_ints(op, [1]))
        rotary_emb = (cos_und, sin_und, cos_gen, sin_gen)

        und_seq = op.Slice(hidden_states, zero, und_len, zero)  # (und_len, hidden)
        gen_seq = op.Slice(hidden_states, und_len, end, zero)  # (gen_len, hidden)

        # --- MoT layer stack ------------------------------------------------
        for layer in self.layers:
            und_seq, gen_seq = layer(op, und_seq, gen_seq, rotary_emb)
        und_out = self.norm(op, und_seq)
        gen_out = self.norm_moe_gen(op, gen_seq)
        # Re-joined so the *_mse_loss_indexes address joint-sequence rows.
        last_hidden_state = op.Concat(und_out, gen_out, axis=0)

        # --- per-modality velocity predictions ------------------------------
        vision_pred = self.proj_out(
            op, op.Gather(last_hidden_state, vision_mse_loss_indexes, axis=0)
        )

        sound_pred = None
        if self.config.sound_gen:
            sound_pred = self.audio_proj_out(
                op, op.Gather(last_hidden_state, sound_mse_loss_indexes, axis=0)
            )

        action_pred = None
        if self.config.action_gen:
            action_pred = self.action_proj_out(
                op,
                op.Gather(last_hidden_state, action_mse_loss_indexes, axis=0),
                action_pred_domain_ids,
            )

        return vision_pred, sound_pred, action_pred

    @staticmethod
    def _require_head_inputs(
        head: str, enabled: bool, inputs: dict[str, ir.Value | None]
    ) -> None:
        """Fail loudly instead of silently dropping a configured head.

        Args:
            head: Head name (for the error message).
            enabled: Whether the head is enabled in the config.
            inputs: Mapping of input name to the supplied value.

        Raises:
            ValueError: If the head is enabled and any input is missing, or if
                the head is disabled but inputs were supplied anyway.
        """
        missing = sorted(name for name, value in inputs.items() if value is None)
        if enabled and missing:
            raise ValueError(
                f"Cosmos3-Omni {head}_gen=True requires all {head} inputs; missing: {missing}"
            )
        if not enabled and len(missing) != len(inputs):
            supplied = sorted(name for name, value in inputs.items() if value is not None)
            raise ValueError(
                f"Cosmos3-Omni {head}_gen=False but {head} inputs were supplied: {supplied}"
            )

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------

    def expected_checkpoint_keys(self) -> set[str]:
        """Return the checkpoint keys this graph consumes.

        Derived from the module tree (so it tracks the ``sound_gen`` /
        ``action_gen`` / ``hidden_act`` / ``qk_norm_for_text`` gating
        automatically), minus the parameters that are recomputed graph
        constants rather than checkpoint weights (``rotary_emb.inv_freq``,
        ``rotary_emb.h_mask``, ``rotary_emb.w_mask``, ``time_proj.inv_freq``).
        """
        return {
            name
            for name, _ in self.named_parameters()
            if not name.endswith(_RECOMPUTED_BUFFER_SUFFIXES)
        }

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Normalize and validate a flat published ``transformer/`` state dict.

        The module tree already mirrors the published names, so this performs
        no renames beyond two structural normalizations:

        1. strips an optional ``transformer.`` / ``model.`` container prefix;
        2. collapses ``self_attn.to_out.0.*`` (checkpoints where the output
           projection was wrapped in an ``nn.Sequential``) onto
           ``self_attn.to_out.*``.

        Keys are then partitioned explicitly — no broad pattern filtering:

        * ``lm_head.weight`` is dropped because upstream constructs it but
          never calls it in ``Cosmos3OmniTransformer.forward``; understanding
          logits come from the separately exported Reasoner;
        * Reasoner vision-tower keys (present only in a *unified* checkpoint)
          are dropped because they belong to that separate export;
        * recomputed buffers (``*.inv_freq`` / ``*.h_mask`` / ``*.w_mask``)
          are dropped because the graph derives them as constants.

        Anything else that does not name a parameter of this graph — and any
        graph parameter left without a weight — raises.

        Args:
            state_dict: Flat published transformer weights.

        Returns:
            A state dict keyed by this graph's initializer names.

        Raises:
            ValueError: If the checkpoint carries unexpected keys, or does not
                cover every parameter this graph needs (both indicate an
                architecture mismatch against the config).
        """
        return self._preprocess_weights(state_dict, require_complete=True)

    def preprocess_weight_shard(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Normalize and validate one shard of the unified checkpoint.

        Unlike :meth:`preprocess_weights`, this does not require the shard to
        contain every graph parameter. Composite exporters call it for each
        shard, accumulate the returned names, then compare that union with
        :meth:`expected_checkpoint_keys`.
        """
        return self._preprocess_weights(state_dict, require_complete=False)

    def _preprocess_weights(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        require_complete: bool,
    ) -> dict[str, torch.Tensor]:
        expected = self.expected_checkpoint_keys()
        renamed: dict[str, torch.Tensor] = {}
        unexpected: list[str] = []

        for raw_key, value in state_dict.items():
            key = raw_key
            for prefix in _STRIPPABLE_PREFIXES:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    break
            key = key.replace(".self_attn.to_out.0.", ".self_attn.to_out.")

            if key in _UNUSED_PUBLISHED_KEYS:
                continue
            if key.endswith(_RECOMPUTED_BUFFER_SUFFIXES):
                continue
            if key.startswith(_REASONER_VISION_PREFIXES):
                continue
            if key.startswith(_EDGE_FRAMEWORK_K_NORM_PREFIXES) and key.endswith(
                ".self_attn.k_norm_und_for_gen.weight"
            ):
                continue
            if key not in expected:
                unexpected.append(raw_key)
                continue
            renamed[key] = value

        if unexpected:
            raise ValueError(
                "Unexpected Cosmos3-Omni transformer weights (architecture mismatch "
                f"against the config): {sorted(unexpected)[:16]}"
            )
        missing = sorted(expected - renamed.keys())
        if require_complete and missing:
            raise ValueError(
                "Cosmos3-Omni transformer checkpoint is missing weights required by "
                f"the configured architecture: {missing[:16]}"
            )
        return renamed


__all__ = [
    "Cosmos3OmniDomainAwareLinear",
    "Cosmos3OmniGeneratorModel",
    "Cosmos3OmniMoTAttention",
    "Cosmos3OmniMoTDecoderLayer",
    "Cosmos3OmniRotaryEmbedding",
    "Cosmos3OmniTimesteps",
]
