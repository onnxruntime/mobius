# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SenseNova-U1.5 ``neo_chat`` — NEO-unify native any-to-any model.

Replicates ``sensenova/SenseNova-U1.5-8B-MoT`` (``NEOChatModel``), whose
reference implementation lives in the ``sensenova_u1`` package at
https://github.com/OpenSenseNova/SenseNova-U1 (branch ``feat/u1.5``) —
the HF repo declares ``auto_map`` entries for ``modeling_neo_chat.py`` but
does **not** ship those files, so the GitHub package is the only source.

Architecture (17.5 B parameters total)
--------------------------------------
NEO-unify is a *Mixture of Transformers* (MoT).  A single 42-layer Qwen3
backbone carries **two complete, disjoint sets of transformer weights**:

* the **understanding** branch (``q_proj``, ``mlp``, ``input_layernorm``,
  ``norm``, …) processes text and reference-image tokens, and
* the **generation** branch (every module suffixed ``_mot_gen``) processes
  the noisy image tokens of the flow-matching sampler.

Attention itself is *shared*: the generation pass attends over the KV
cache written by the understanding pass, which is how text conditioning
reaches the image tokens.  Upstream never mixes both branches inside one
forward call — ``Qwen3Attention.forward`` and ``Qwen3DecoderLayer.forward``
raise ``NotImplementedError`` for the mixed path — so the two branches are
exported here as two separate ONNX graphs, exactly mirroring the two
production passes.

Rotary layout
~~~~~~~~~~~~~
``head_dim`` 128 is split into three independently-rotated axes::

    [ 0 : 64) temporal / text   RoPE, theta = rope_theta      (5e6)
    [64 : 96) image height      RoPE, theta = rope_theta_hw   (1e4)
    [96 :128) image width       RoPE, theta = rope_theta_hw   (1e4)

QK-norm is applied per *half*: ``q_norm`` over the 64 temporal dims and
``q_norm_hw`` over the 64 spatial dims (before the h/w split), which is
why the checkpoint's ``q_norm`` tensors have shape ``[64]`` and not
``[128]``.

Vision tower
~~~~~~~~~~~~
The NEO vision "encoder" has **no transformer blocks**.  It is::

    Conv2d(3, 1024, k=16, s=16) → GELU → interleaved 2-D RoPE
    → Conv2d(1024, 4096, k=2, s=2)

so one LLM token covers a 32x32 pixel tile.  The same module is
instantiated twice: ``vision_model`` embeds reference images into the
understanding branch, and ``fm_modules.vision_model_mot_gen`` embeds the
noisy latent into the generation branch.

Flow-matching head
~~~~~~~~~~~~~~~~~~
``use_pixel_head`` is true, so there is **no VAE**.  The generation
branch's hidden states are reshaped to a ``(B, 4096, H/32, W/32)`` feature
map and decoded straight to RGB by a pixel-shuffle ``ConvDecoder``::

    PixelShuffle(2) → Conv2d(1024,1024,k3) → GELU
    → PixelShuffle(2) → Conv2d(256,192,k3) → PixelShuffle(8)

The head predicts ``x0`` (the clean image), and the sampler converts it
to a velocity via ``v = (x0 - z) / max(1 - t, t_eps)``.

HuggingFace weight layout::

    vision_model.embeddings.patch_embedding.{weight,bias}
    vision_model.embeddings.dense_embedding.{weight,bias}
    fm_modules.vision_model_mot_gen.embeddings.patch_embedding.{weight,bias}
    fm_modules.vision_model_mot_gen.embeddings.dense_embedding.{weight,bias}
    fm_modules.timestep_embedder.mlp.{0,2}.{weight,bias}
    fm_modules.noise_scale_embedder.mlp.{0,2}.{weight,bias}
    fm_modules.fm_head.conv{1,2}.{weight,bias}
    language_model.model.embed_tokens.weight
    language_model.model.layers.{i}.self_attn.{q,k,v,o}_proj[_mot_gen].weight
    language_model.model.layers.{i}.self_attn.{q,k}_norm[_hw][_mot_gen].weight
    language_model.model.layers.{i}.mlp[_mot_gen].{gate,up,down}_proj.weight
    language_model.model.layers.{i}.{input,post_attention}_layernorm[_mot_gen].weight
    language_model.model.norm[_mot_gen].weight
    language_model.lm_head.weight
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from onnxscript import OpBuilder, nn

from mobius._configs import SenseNovaU1Config
from mobius.components import Conv2d, Embedding, GatedMLP, Linear, RMSNorm, SiLU
from mobius.components._attention import _apply_attention
from mobius.components._rotary_embedding import BaseRope, _get_cos_sin_cache

if TYPE_CHECKING:
    import onnx_ir as ir
    import torch


def _inv_freq(dim: int, theta: float) -> np.ndarray:
    """Standard RoPE inverse frequencies for a ``dim``-wide rotary axis."""
    return 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))


class _AxisRope(BaseRope):
    """RoPE table for one of the three (t / h / w) rotary axes."""

    def __init__(self, dim: int, theta: float, max_positions: int, dtype):
        cos_cache, sin_cache = _get_cos_sin_cache(max_positions, _inv_freq(dim, theta))
        super().__init__(cos_cache, sin_cache, dtype=dtype)


# ── Vision tower (patchify + 2-D RoPE + merge) ──────────────────────────


class _NEOVisionEmbeddings(nn.Module):
    """Patchify → GELU → interleaved 2-D RoPE → 2x2 merge projection.

    Replicates ``NEOVisionEmbeddings``.  Upstream reshapes a packed patch
    tensor to ``(N, 3, 16, 16)`` and applies a ``kernel=stride=16``
    convolution to every patch; running that same convolution over the
    whole image is numerically identical and keeps the graph free of
    ragged per-image loops.
    """

    def __init__(self, config: SenseNovaU1Config):
        super().__init__()
        vision = config.vision
        self.embed_dim = int(vision.hidden_size or 1024)
        self.llm_embed_dim = int(vision.out_hidden_size or config.hidden_size)
        self.patch_size = int(vision.patch_size or config.patch_size)
        self.merge = int(vision.spatial_merge_size or config.merge_size)
        self._dtype = config.dtype

        self.patch_embedding = Conv2d(
            int(vision.in_channels or 3),
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.dense_embedding = Conv2d(
            self.embed_dim,
            self.llm_embed_dim,
            kernel_size=self.merge,
            stride=self.merge,
        )

        # 2-D RoPE over the (H/patch, W/patch) grid: the first half of the
        # channel axis is rotated by the x (column) index, the second half
        # by the y (row) index.  Rotation is *interleaved* — upstream pairs
        # (0,1), (2,3), … rather than the half-split GPT-NeoX convention.
        rope_dim = self.embed_dim // 2
        max_positions = int(vision.num_position_embeddings or 10_000)
        theta = float(vision.rope_theta or 10_000.0)
        cos_cache, sin_cache = _get_cos_sin_cache(max_positions, _inv_freq(rope_dim, theta))
        self.rope = BaseRope(cos_cache, sin_cache, dtype=config.dtype)

    def forward(self, op: OpBuilder, pixel_values: ir.Value):
        # (B, 3, H, W) -> (B, embed_dim, H/p, W/p)
        patch_embeds = self.patch_embedding(op, pixel_values)
        patch_embeds = op.Gelu(patch_embeds)

        grid_h = op.Shape(patch_embeds, start=2, end=3)
        grid_w = op.Shape(patch_embeds, start=3, end=4)

        # Channel-last (B, H/p, W/p, embed_dim) so RoPE rotates channels.
        hidden = op.Transpose(patch_embeds, perm=[0, 2, 3, 1])
        flat = op.Reshape(hidden, op.Constant(value_ints=[1, -1, self.embed_dim]))

        # Row-major patch order: x = column index, y = row index.
        x_pos, y_pos = _grid_positions(op, grid_h, grid_w)
        cos_x, sin_x = self.rope(op, x_pos)
        cos_y, sin_y = self.rope(op, y_pos)

        half = self.embed_dim // 2
        part_x = op.Slice(
            flat,
            op.Constant(value_ints=[0]),
            op.Constant(value_ints=[half]),
            op.Constant(value_ints=[2]),
        )
        part_y = op.Slice(
            flat,
            op.Constant(value_ints=[half]),
            op.Constant(value_ints=[self.embed_dim]),
            op.Constant(value_ints=[2]),
        )
        # RoPE is evaluated in float32 upstream for numerical stability.
        part_x = _rotate_interleaved(op, part_x, cos_x, sin_x, self._dtype)
        part_y = _rotate_interleaved(op, part_y, cos_y, sin_y, self._dtype)
        rotated = op.Concat(part_x, part_y, axis=-1)

        # Back to NCHW for the 2x2 merge convolution.
        shape_4d = op.Concat(
            op.Constant(value_ints=[1]),
            grid_h,
            grid_w,
            op.Constant(value_ints=[self.embed_dim]),
            axis=0,
        )
        rotated = op.Reshape(rotated, shape_4d)
        rotated = op.Transpose(rotated, perm=[0, 3, 1, 2])
        merged = self.dense_embedding(op, rotated)  # (1, llm_dim, H/32, W/32)

        # (1, llm_dim, h, w) -> (1, h*w, llm_dim)
        merged = op.Transpose(merged, perm=[0, 2, 3, 1])
        return op.Reshape(merged, op.Constant(value_ints=[1, -1, self.llm_embed_dim]))


def _grid_positions(op: OpBuilder, grid_h: ir.Value, grid_w: ir.Value):
    """Row-major (x, y) patch coordinates for an ``h * w`` grid.

    Mirrors ``build_abs_positions_from_grid_hw``: ``x = idx % W`` (column)
    and ``y = idx // W`` (row), both shaped ``(1, h * w)`` so they can be
    consumed directly as position ids.
    """
    total = op.Mul(grid_h, grid_w)
    idx = op.Range(
        op.Constant(value_int=0),
        op.Squeeze(total, op.Constant(value_ints=[0])),
        op.Constant(value_int=1),
    )
    idx = op.Reshape(idx, op.Constant(value_ints=[1, -1]))
    width = op.Cast(op.Reshape(grid_w, op.Constant(value_ints=[1, 1])), to=7)
    x_pos = op.Mod(idx, width)
    y_pos = op.Div(idx, width)
    return x_pos, y_pos


def _rotate_interleaved(
    op: OpBuilder,
    x: ir.Value,
    cos: ir.Value,
    sin: ir.Value,
    dtype,
):
    """Interleaved rotation: pairs ``(0,1), (2,3), …`` share one angle.

    ``x`` has shape ``(1, N, D)``; ``cos``/``sin`` have shape ``(1, N, D/2)``.
    """
    x32 = op.Cast(x, to=1)
    pairs = op.Reshape(x32, op.Constant(value_ints=[1, 0, -1, 2]))
    even = op.Squeeze(
        op.Slice(
            pairs,
            op.Constant(value_ints=[0]),
            op.Constant(value_ints=[1]),
            op.Constant(value_ints=[3]),
        ),
        op.Constant(value_ints=[3]),
    )
    odd = op.Squeeze(
        op.Slice(
            pairs,
            op.Constant(value_ints=[1]),
            op.Constant(value_ints=[2]),
            op.Constant(value_ints=[3]),
        ),
        op.Constant(value_ints=[3]),
    )
    cos32 = op.Cast(cos, to=1)
    sin32 = op.Cast(sin, to=1)
    rot_even = op.Sub(op.Mul(even, cos32), op.Mul(odd, sin32))
    rot_odd = op.Add(op.Mul(even, sin32), op.Mul(odd, cos32))
    stacked = op.Concat(
        op.Unsqueeze(rot_even, op.Constant(value_ints=[3])),
        op.Unsqueeze(rot_odd, op.Constant(value_ints=[3])),
        axis=3,
    )
    out = op.Reshape(stacked, op.Constant(value_ints=[1, 0, -1]))
    return op.Cast(out, to=dtype)


class _SenseNovaU1VisionModel(nn.Module):
    """Understanding-branch image tower (``vision_model``)."""

    def __init__(self, config: SenseNovaU1Config):
        super().__init__()
        self.embeddings = _NEOVisionEmbeddings(config)

    def forward(self, op: OpBuilder, pixel_values: ir.Value):
        return self.embeddings(op, pixel_values)


# ── Timestep / noise-scale embedders ────────────────────────────────────


class _TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding + 2-layer MLP.

    Replicates ``modeling_fm_modules.TimestepEmbedder``.  Note the
    concatenation order is ``[cos, sin]`` (diffusers uses ``[sin, cos]``).
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int, dtype):
        super().__init__()
        self.mlp = nn.ModuleList(
            [
                Linear(frequency_embedding_size, hidden_size),
                SiLU(),
                Linear(hidden_size, hidden_size),
            ]
        )
        self._half = frequency_embedding_size // 2
        self._dtype = dtype
        freqs = np.exp(
            -math.log(10_000.0)
            * np.arange(0, self._half, dtype=np.float32)
            / float(self._half)
        )
        self.freqs = nn.Parameter([self._half], name="freqs", data=_as_tensor(freqs))

    def forward(self, op: OpBuilder, timesteps: ir.Value):
        # timesteps: (N,) -> (N, frequency_embedding_size).  Upstream builds
        # the sinusoidal basis in float32 regardless of model dtype, so the
        # frequency table is cast up explicitly (``_cast_module_dtype`` will
        # otherwise have demoted it alongside the real weights).
        t = op.Cast(op.Reshape(timesteps, op.Constant(value_ints=[-1, 1])), to=1)
        freqs = op.Cast(op.Reshape(self.freqs, op.Constant(value_ints=[1, -1])), to=1)
        args = op.Mul(t, freqs)
        emb = op.Concat(op.Cos(args), op.Sin(args), axis=-1)
        emb = op.Cast(emb, to=self._dtype)
        hidden = self.mlp[0](op, emb)
        hidden = self.mlp[1](op, hidden)
        return self.mlp[2](op, hidden)


def _as_tensor(array: np.ndarray):
    import onnx_ir as ir

    return ir.tensor(array)


# ── MoT attention / decoder layer ───────────────────────────────────────


class _MoTAttention(nn.Module):
    """Qwen3 attention with a per-branch weight set and 3-axis RoPE.

    ``branch`` selects the parameter names: ``""`` for the understanding
    branch and ``"_mot_gen"`` for the generation branch.  Both branches
    read and write the *same* KV cache layout, which is what lets the
    generation pass attend over the understanding prefix.
    """

    def __init__(self, config: SenseNovaU1Config, branch: str = ""):
        super().__init__()
        self.config = config
        self.branch = branch
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5

        suffix = branch
        setattr(
            self,
            f"q_proj{suffix}",
            Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False),
        )
        setattr(
            self,
            f"k_proj{suffix}",
            Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False),
        )
        setattr(
            self,
            f"v_proj{suffix}",
            Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False),
        )
        setattr(
            self,
            f"o_proj{suffix}",
            Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False),
        )
        half = self.head_dim // 2
        setattr(self, f"q_norm{suffix}", RMSNorm(half, eps=config.rms_norm_eps))
        setattr(self, f"q_norm_hw{suffix}", RMSNorm(half, eps=config.rms_norm_eps))
        setattr(self, f"k_norm{suffix}", RMSNorm(half, eps=config.rms_norm_eps))
        setattr(self, f"k_norm_hw{suffix}", RMSNorm(half, eps=config.rms_norm_eps))

    def _p(self, name: str):
        return getattr(self, f"{name}{self.branch}")

    def _split_norm_rope(
        self,
        op: OpBuilder,
        projected: ir.Value,
        num_heads: int,
        norm_t,
        norm_hw,
        position_embeddings,
    ):
        """Chunk into t/h/w, QK-norm each half, rotate, re-concatenate.

        ``projected`` is ``(B, S, num_heads * head_dim)``.  Upstream views
        it as ``(B, S, num_heads, head_dim)`` before chunking, so the
        split is *within* each head, not across the flat projection.
        """
        half = self.head_dim // 2
        quarter = self.head_dim // 4
        heads = op.Reshape(
            projected,
            op.Constant(value_ints=[0, 0, num_heads, self.head_dim]),
        )
        part_t = op.Slice(
            heads,
            op.Constant(value_ints=[0]),
            op.Constant(value_ints=[half]),
            op.Constant(value_ints=[3]),
        )
        part_hw = op.Slice(
            heads,
            op.Constant(value_ints=[half]),
            op.Constant(value_ints=[self.head_dim]),
            op.Constant(value_ints=[3]),
        )
        part_t = norm_t(op, part_t)
        part_hw = norm_hw(op, part_hw)

        cos_t, sin_t, cos_h, sin_h, cos_w, sin_w = position_embeddings
        part_t = _rotate_half_axis(op, part_t, cos_t, sin_t, num_heads, half)
        part_h = op.Slice(
            part_hw,
            op.Constant(value_ints=[0]),
            op.Constant(value_ints=[quarter]),
            op.Constant(value_ints=[3]),
        )
        part_w = op.Slice(
            part_hw,
            op.Constant(value_ints=[quarter]),
            op.Constant(value_ints=[half]),
            op.Constant(value_ints=[3]),
        )
        part_h = _rotate_half_axis(op, part_h, cos_h, sin_h, num_heads, quarter)
        part_w = _rotate_half_axis(op, part_w, cos_w, sin_w, num_heads, quarter)

        merged = op.Concat(part_t, part_h, part_w, axis=-1)
        return op.Reshape(
            merged,
            op.Constant(value_ints=[0, 0, num_heads * self.head_dim]),
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        position_embeddings,
        attention_mask: ir.Value | None = None,
        past_key: ir.Value | None = None,
        past_value: ir.Value | None = None,
        update_cache: bool = True,
    ):
        query = self._p("q_proj")(op, hidden_states)
        key = self._p("k_proj")(op, hidden_states)
        value = self._p("v_proj")(op, hidden_states)

        query = self._split_norm_rope(
            op,
            query,
            self.num_heads,
            self._p("q_norm"),
            self._p("q_norm_hw"),
            position_embeddings,
        )
        key = self._split_norm_rope(
            op,
            key,
            self.num_kv_heads,
            self._p("k_norm"),
            self._p("k_norm_hw"),
            position_embeddings,
        )

        attn_output, present_key, present_value = _apply_attention(
            op,
            query,
            key,
            value,
            attention_mask,
            past_key,
            past_value,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            scale=self.scaling,
            # ``attn_mask`` (when present) already bakes in the full
            # block-causal + padding mask, and the generation pass attends
            # bidirectionally over its image tokens, so causality is never
            # re-applied by the op itself.
            is_causal=0,
        )
        return self._p("o_proj")(op, attn_output), present_key, present_value


def _rotate_half_axis(
    op: OpBuilder,
    x: ir.Value,
    cos: ir.Value,
    sin: ir.Value,
    num_heads: int,
    axis_dim: int,
):
    """Half-split (``rotate_half``) RoPE over one axis of every head.

    ``x`` is ``(B, S, num_heads, axis_dim)``; cos/sin are
    ``(B, S, axis_dim / 2)``.
    """
    half = axis_dim // 2
    first = op.Slice(
        x,
        op.Constant(value_ints=[0]),
        op.Constant(value_ints=[half]),
        op.Constant(value_ints=[3]),
    )
    second = op.Slice(
        x,
        op.Constant(value_ints=[half]),
        op.Constant(value_ints=[axis_dim]),
        op.Constant(value_ints=[3]),
    )
    cos_b = op.Unsqueeze(cos, op.Constant(value_ints=[2]))
    sin_b = op.Unsqueeze(sin, op.Constant(value_ints=[2]))
    del num_heads
    rot_first = op.Sub(op.Mul(first, cos_b), op.Mul(second, sin_b))
    rot_second = op.Add(op.Mul(second, cos_b), op.Mul(first, sin_b))
    return op.Concat(rot_first, rot_second, axis=-1)


class _MoTDecoderLayer(nn.Module):
    """One NEO-unify decoder layer for a single branch."""

    def __init__(self, config: SenseNovaU1Config, branch: str = ""):
        super().__init__()
        self.branch = branch
        self.self_attn = _MoTAttention(config, branch)
        setattr(
            self,
            f"mlp{branch}",
            GatedMLP(config.hidden_size, config.intermediate_size, bias=config.mlp_bias),
        )
        setattr(
            self,
            f"input_layernorm{branch}",
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
        )
        setattr(
            self,
            f"post_attention_layernorm{branch}",
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        position_embeddings,
        attention_mask: ir.Value | None = None,
        past_key: ir.Value | None = None,
        past_value: ir.Value | None = None,
        update_cache: bool = True,
    ):
        residual = hidden_states
        hidden_states = getattr(self, f"input_layernorm{self.branch}")(op, hidden_states)
        hidden_states, present_key, present_value = self.self_attn(
            op,
            hidden_states,
            position_embeddings,
            attention_mask=attention_mask,
            past_key=past_key,
            past_value=past_value,
            update_cache=update_cache,
        )
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = getattr(self, f"post_attention_layernorm{self.branch}")(
            op, hidden_states
        )
        hidden_states = getattr(self, f"mlp{self.branch}")(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)
        return hidden_states, present_key, present_value


class _MoTRotary(nn.Module):
    """The three rotary tables shared by every layer of a branch."""

    def __init__(self, config: SenseNovaU1Config):
        super().__init__()
        half = config.head_dim // 2
        quarter = config.head_dim // 4
        self.rope_t = _AxisRope(
            half,
            float(config.rope_theta or 10_000.0),
            config.max_position_embeddings,
            config.dtype,
        )
        self.rope_hw = _AxisRope(
            quarter,
            config.rope_theta_hw,
            config.max_position_embeddings_hw,
            config.dtype,
        )

    def forward(self, op: OpBuilder, position_ids: ir.Value):
        """``position_ids`` is ``(3, B, S)`` — the t, h and w axes."""
        axes = [
            op.Squeeze(
                op.Slice(
                    position_ids,
                    op.Constant(value_ints=[i]),
                    op.Constant(value_ints=[i + 1]),
                    op.Constant(value_ints=[0]),
                ),
                op.Constant(value_ints=[0]),
            )
            for i in range(3)
        ]
        cos_t, sin_t = self.rope_t(op, axes[0])
        cos_h, sin_h = self.rope_hw(op, axes[1])
        cos_w, sin_w = self.rope_hw(op, axes[2])
        return cos_t, sin_t, cos_h, sin_h, cos_w, sin_w


# ── Understanding decoder (text + VLM) ──────────────────────────────────


class _SenseNovaU1DecoderModel(nn.Module):
    """Understanding branch: ``inputs_embeds`` → ``logits`` + KV cache."""

    def __init__(self, config: SenseNovaU1Config):
        super().__init__()
        self.config = config
        self.model = _SenseNovaU1DecoderBody(config, branch="")
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value | None = None,
        position_ids: ir.Value | None = None,
        past_key_values: list | None = None,
    ):
        hidden_states, present = self.model(
            op,
            inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        return self.lm_head(op, hidden_states), present


def _block_causal_mask(
    op: OpBuilder,
    t_positions: ir.Value,
    attention_mask: ir.Value,
):
    """Block-causal bool mask ``(batch, 1, q_len, total_len)``.

    Replicates ``create_block_causal_mask``::

        allow[i][j] = (t[j] == t[i]) or (j <= i)

    Tokens sharing a temporal index — every patch of one image — therefore
    attend to each other *bidirectionally*, while ordinary causality holds
    across different temporal indices.  Cached prefix positions always
    precede the current chunk, so they are unconditionally visible.
    """
    q_len = op.Shape(t_positions, start=1, end=2)
    total_len = op.Shape(attention_mask, start=1, end=2)
    past_len = op.Sub(total_len, q_len)

    # (batch, q_len, q_len): same temporal block.
    t_i = op.Unsqueeze(t_positions, op.Constant(value_ints=[2]))
    t_j = op.Unsqueeze(t_positions, op.Constant(value_ints=[1]))
    same_block = op.Equal(t_i, t_j)

    # (q_len, q_len): lower-triangular causal part.
    positions = op.Range(
        op.Constant(value_int=0),
        op.Squeeze(q_len, op.Constant(value_ints=[0])),
        op.Constant(value_int=1),
    )
    causal = op.LessOrEqual(
        op.Unsqueeze(positions, op.Constant(value_ints=[0])),
        op.Unsqueeze(positions, op.Constant(value_ints=[1])),
    )
    allow = op.Or(same_block, op.Unsqueeze(causal, op.Constant(value_ints=[0])))

    # Prepend the always-visible cached prefix.
    prefix_shape = op.Concat(op.Shape(t_positions, start=0, end=1), q_len, past_len, axis=0)
    prefix = op.Expand(op.Constant(value_int=1), prefix_shape)
    allow = op.Concat(op.Cast(prefix, to=9), allow, axis=-1)

    # Fold in padding.
    padding = op.Unsqueeze(op.Cast(attention_mask, to=9), op.Constant(value_ints=[1]))
    allow = op.And(allow, padding)
    return op.Unsqueeze(allow, op.Constant(value_ints=[1]))


class _SenseNovaU1DecoderBody(nn.Module):
    """Shared layer stack + final norm for one branch."""

    def __init__(self, config: SenseNovaU1Config, branch: str = ""):
        super().__init__()
        self.config = config
        self.branch = branch
        self.layers = nn.ModuleList(
            [_MoTDecoderLayer(config, branch) for _ in range(config.num_hidden_layers)]
        )
        setattr(self, f"norm{branch}", RMSNorm(config.hidden_size, eps=config.rms_norm_eps))
        self.rotary_emb = _MoTRotary(config)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value | None = None,
        position_ids: ir.Value | None = None,
        past_key_values: list | None = None,
        update_cache: bool = True,
    ):
        position_embeddings = self.rotary_emb(op, position_ids)
        # Block-causal masking is intrinsic to NEO-unify: image patches that
        # share a temporal index attend bidirectionally.  The bool mask
        # therefore carries the FULL mask and ``is_causal`` must be 0.
        attention_bias = None
        if attention_mask is not None:
            t_positions = op.Squeeze(
                op.Slice(
                    position_ids,
                    op.Constant(value_ints=[0]),
                    op.Constant(value_ints=[1]),
                    op.Constant(value_ints=[0]),
                ),
                op.Constant(value_ints=[0]),
            )
            attention_bias = _block_causal_mask(op, t_positions, attention_mask)
        present: list = []
        for index, layer in enumerate(self.layers):
            past_key = past_key_values[index][0] if past_key_values else None
            past_value = past_key_values[index][1] if past_key_values else None
            hidden_states, present_key, present_value = layer(
                op,
                hidden_states,
                position_embeddings,
                attention_mask=attention_bias,
                past_key=past_key,
                past_value=past_value,
                update_cache=update_cache,
            )
            present.append((present_key, present_value))
        hidden_states = getattr(self, f"norm{self.branch}")(op, hidden_states)
        return hidden_states, present


# ── Embedding (token lookup + reference-image scatter) ──────────────────


class _SenseNovaU1EmbeddingModel(nn.Module):
    """Token embedding with reference-image feature scatter."""

    def __init__(self, config: SenseNovaU1Config):
        super().__init__()
        self.config = config
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        image_features: ir.Value | None = None,
        image_mask: ir.Value | None = None,
    ):
        inputs_embeds = self.embed_tokens(op, input_ids)
        if image_features is None:
            return inputs_embeds
        # Scatter image features at ``<IMG_CONTEXT>`` placeholder positions.
        mask = op.Unsqueeze(image_mask, op.Constant(value_ints=[-1]))
        mask = op.Expand(mask, op.Shape(inputs_embeds))
        flat_embeds = op.Reshape(
            inputs_embeds, op.Constant(value_ints=[-1, self.config.hidden_size])
        )
        flat_features = op.Reshape(
            image_features, op.Constant(value_ints=[-1, self.config.hidden_size])
        )
        indices = op.Reshape(
            op.NonZero(op.Reshape(image_mask, op.Constant(value_ints=[-1]))),
            op.Constant(value_ints=[-1, 1]),
        )
        scattered = op.ScatterND(flat_embeds, indices, flat_features)
        del mask
        return op.Reshape(scattered, op.Shape(inputs_embeds))


# ── Generation branch: noisy-latent embedding ───────────────────────────


class _SenseNovaU1ImageGenEmbeddingModel(nn.Module):
    """Noisy image → generation-branch token embeddings.

    Mirrors the per-step head of ``t2i_generate``::

        image_embeds = vision_model_mot_gen(patchify(z_t))
        t_emb       = timestep_embedder(t)
        t_emb      += noise_scale_embedder(noise_scale / noise_scale_max)
        image_embeds = image_embeds + t_emb
    """

    def __init__(self, config: SenseNovaU1Config):
        super().__init__()
        self.config = config
        self.vision_model_mot_gen = _SenseNovaU1VisionModel(config)
        self.timestep_embedder = _TimestepEmbedder(
            config.hidden_size, config.frequency_embedding_size, config.dtype
        )
        self.add_noise_scale_embedding = config.add_noise_scale_embedding
        if config.add_noise_scale_embedding:
            self.noise_scale_embedder = _TimestepEmbedder(
                config.hidden_size, config.frequency_embedding_size, config.dtype
            )

    def forward(
        self,
        op: OpBuilder,
        latent: ir.Value,
        timestep: ir.Value,
        noise_scale: ir.Value | None = None,
    ):
        image_embeds = self.vision_model_mot_gen(op, latent)
        time_embed = self.timestep_embedder(op, timestep)
        if self.add_noise_scale_embedding and noise_scale is not None:
            time_embed = op.Add(time_embed, self.noise_scale_embedder(op, noise_scale))
        # (1, hidden) broadcast over every image token.
        time_embed = op.Reshape(
            time_embed, op.Constant(value_ints=[1, -1, self.config.hidden_size])
        )
        return op.Add(image_embeds, time_embed)


# ── Generation branch: denoiser + pixel head ────────────────────────────


class _ConvDecoder(nn.Module):
    """Pixel-shuffle flow-matching head (``fm_modules.fm_head``).

    ``PixelShuffle`` maps to ONNX ``DepthToSpace`` with ``mode="CRD"``,
    which is exactly PyTorch's channel ordering.

    The head upsamples by ``pixels_per_token`` in total, factored as
    ``2 * 2 * final_block``.  For the released checkpoint
    ``pixels_per_token`` is ``patch_size * merge_size == 32``, giving the
    upstream ``PixelShuffle(2) / PixelShuffle(2) / PixelShuffle(8)``
    sequence and a 192-channel ``conv2`` (``3 * 8 * 8``).
    """

    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 1024,
        pixels_per_token: int = 32,
        out_channels: int = 3,
    ):
        super().__init__()
        if pixels_per_token % 4 != 0:
            raise ValueError(
                "pixels_per_token must be divisible by 4 (two 2x pixel shuffles); "
                f"got {pixels_per_token}"
            )
        self._final_block = pixels_per_token // 4
        self._out_channels = out_channels
        conv2_out = out_channels * self._final_block**2
        self.conv1 = Conv2d(input_dim // 4, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = Conv2d(hidden_dim // 4, conv2_out, kernel_size=3, padding=1)

    def forward(self, op: OpBuilder, x: ir.Value):
        # (B, hidden, h, w) -> (B, hidden/4, 2h, 2w)
        x = op.DepthToSpace(x, blocksize=2, mode="CRD")
        x = op.Gelu(self.conv1(op, x))
        # (B, hidden_dim/4, 4h, 4w)
        x = op.DepthToSpace(x, blocksize=2, mode="CRD")
        x = self.conv2(op, x)
        # (B, 3, pixels_per_token * h, pixels_per_token * w)
        return op.DepthToSpace(x, blocksize=self._final_block, mode="CRD")


class _SenseNovaU1ImageGenDenoiserModel(nn.Module):
    """Generation branch: image embeds + cached prefix → predicted image.

    The output is the head's ``x0`` estimate in **pixel** space.  Upstream
    re-patchifies before computing ``v = (x0 - z) / (1 - t)``; because
    patchify is a pure permutation the sampler can equivalently do that
    arithmetic on the pixel grid, so the graph returns pixels directly.
    """

    def __init__(self, config: SenseNovaU1Config):
        super().__init__()
        self.config = config
        self.model = _SenseNovaU1DecoderBody(config, branch="_mot_gen")
        self.fm_head = _ConvDecoder(
            config.hidden_size,
            pixels_per_token=config.pixels_per_token,
            out_channels=int((config.vision.in_channels if config.vision else 3) or 3),
        )

    def forward(
        self,
        op: OpBuilder,
        image_embeds: ir.Value,
        position_ids: ir.Value | None = None,
        past_key_values: list | None = None,
        token_grid: ir.Value | None = None,
    ):
        hidden_states, present = self.model(
            op,
            image_embeds,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=past_key_values,
            update_cache=False,
        )
        # (B, h*w, hidden) -> (B, hidden, h, w)
        grid_h = op.Slice(
            token_grid,
            op.Constant(value_ints=[0]),
            op.Constant(value_ints=[1]),
            op.Constant(value_ints=[0]),
        )
        grid_w = op.Slice(
            token_grid,
            op.Constant(value_ints=[1]),
            op.Constant(value_ints=[2]),
            op.Constant(value_ints=[0]),
        )
        shape_4d = op.Concat(
            op.Constant(value_ints=[-1]),
            grid_h,
            grid_w,
            op.Constant(value_ints=[self.config.hidden_size]),
            axis=0,
        )
        feature_map = op.Reshape(hidden_states, shape_4d)
        feature_map = op.Transpose(feature_map, perm=[0, 3, 1, 2])
        return self.fm_head(op, feature_map), present


# ── Top-level package ───────────────────────────────────────────────────


class SenseNovaU1Model(nn.Module):
    """SenseNova-U1.5 NEO-unify unified understanding + generation model.

    Builds five ONNX graphs that mirror the upstream inference stages:

    ``model``
        Understanding decoder — text generation and VLM understanding.
    ``vision``
        Reference-image tower feeding the understanding branch.
    ``embedding``
        Token embedding with reference-image feature scatter.
    ``image_gen_embedding``
        Noisy-latent tower + timestep / noise-scale embedders.
    ``image_gen_denoiser``
        Generation-branch transformer + pixel-shuffle flow-matching head.
    """

    default_task: str = "sensenova-u1"
    category: str = "Multimodal"
    config_class: ClassVar[type[SenseNovaU1Config]] = SenseNovaU1Config

    HF_COMPONENT_SOURCES: ClassVar[dict[str, tuple[str, ...]]] = {
        "model": (
            "language_model.model.layers",
            "language_model.model.norm",
            "language_model.lm_head",
        ),
        "vision": ("vision_model",),
        "embedding": ("language_model.model.embed_tokens",),
        "image_gen_embedding": (
            "fm_modules.vision_model_mot_gen",
            "fm_modules.timestep_embedder",
            "fm_modules.noise_scale_embedder",
        ),
        "image_gen_denoiser": (
            "language_model.model.layers",
            "language_model.model.norm_mot_gen",
            "fm_modules.fm_head",
        ),
    }

    def __init__(self, config: SenseNovaU1Config):
        super().__init__()
        self.config = config
        self.model = _SenseNovaU1DecoderModel(config)
        self.vision = _SenseNovaU1VisionModel(config)
        self.embedding = _SenseNovaU1EmbeddingModel(config)
        self.image_gen_embedding = _SenseNovaU1ImageGenEmbeddingModel(config)
        self.image_gen_denoiser = _SenseNovaU1ImageGenDenoiserModel(config)

    def forward(self, op: OpBuilder, **kwargs):
        raise NotImplementedError(
            "SenseNovaU1Model uses SenseNovaU1Task, which builds each "
            "component (model, vision, embedding, image_gen_embedding, "
            "image_gen_denoiser) separately."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Route HF ``NEOChatModel`` weights onto the five ONNX components.

        The understanding and generation branches share the HF layer
        prefix (``language_model.model.layers.{i}.*``) and are told apart
        purely by the ``_mot_gen`` suffix, so each layer tensor is routed
        to exactly one of the two decoder graphs.
        """
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            for target in self._targets_for(key):
                renamed[target] = value
        return renamed

    def _targets_for(self, key: str) -> list[str]:
        """Map one HF key to its ONNX initializer name(s)."""
        if key.startswith("vision_model."):
            # ``vision_model.embeddings.*`` -> ``vision.embeddings.*``
            return [f"vision.{key[len('vision_model.') :]}"]

        if key.startswith("fm_modules."):
            suffix = key[len("fm_modules.") :]
            if suffix.startswith("fm_head."):
                return [f"image_gen_denoiser.{suffix}"]
            return [f"image_gen_embedding.{suffix}"]

        if key.startswith("language_model."):
            suffix = key[len("language_model.") :]
            if suffix.startswith("model.embed_tokens."):
                return [f"embedding.{suffix[len('model.') :]}"]
            if suffix.startswith("lm_head."):
                return [f"model.{suffix}"]
            if suffix == "model.norm.weight":
                return ["model.model.norm.weight"]
            if suffix == "model.norm_mot_gen.weight":
                return ["image_gen_denoiser.model.norm_mot_gen.weight"]
            if suffix.startswith("model.layers."):
                target = "image_gen_denoiser" if "_mot_gen" in suffix else "model"
                return [f"{target}.{suffix}"]
        return []
