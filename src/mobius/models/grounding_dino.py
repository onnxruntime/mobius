"""Grounding DINO: text-guided open-set object detection.

Architecture: Swin Transformer backbone + BERT text encoder +
multi-scale deformable attention encoder/decoder with text-vision fusion.

HuggingFace class: ``GroundingDinoForObjectDetection``.

The encoder fuses text and vision features through bidirectional cross-attention,
text self-attention enhancement, and multi-scale deformable self-attention on
vision features. The decoder uses standard self-attention, text cross-attention,
and deformable cross-attention with iterative bounding box refinement.

Class prediction is contrastive: ``vision_hidden @ text_hidden.T`` instead of
a learned linear classifier, enabling open-vocabulary detection.

Reuses algorithmic building blocks from:
- CLAP (Swin shifted-window attention, cyclic shift, window partition)
- RT-DETR (multi-scale deformable attention with GridSample)
- DETR (MLP prediction head, sine position embeddings)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import onnx_ir as ir
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import GroundingDinoConfig
from mobius.components import Conv2d, GroupNorm, LayerNorm, Linear

# --------------------------------------------------------------------------
# Utility: dict → object adapter for sub-configs (e.g., text_config for BERT)
# --------------------------------------------------------------------------


class _DictConfig:
    """Adapts a dict to an object with attribute access for BertModel."""

    def __init__(self, d: dict[str, Any]):
        for k, v in d.items():
            setattr(self, k, v)
        # Ensure defaults for BertModel
        if not hasattr(self, "rms_norm_eps"):
            self.rms_norm_eps = d.get("layer_norm_eps", 1e-12)
        if not hasattr(self, "pad_token_id"):
            self.pad_token_id = 0
        if not hasattr(self, "hidden_act"):
            self.hidden_act = "gelu"


# --------------------------------------------------------------------------
# Swin Backbone components
# (Adapted from CLAP Swin implementation — same shifted window algorithm)
# --------------------------------------------------------------------------


def _precompute_relative_position_index(window_size: int) -> np.ndarray:
    """Compute (window^2, window^2) relative position index table."""
    wh = ww = window_size
    coords_h = np.arange(wh)
    coords_w = np.arange(ww)
    coords = np.stack(np.meshgrid(coords_h, coords_w, indexing="ij"), axis=0)  # (2, wH, wW)
    coords_flat = coords.reshape(2, -1)  # (2, wH*wW)
    rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, N, N)
    rel = rel.transpose(1, 2, 0)  # (N, N, 2)
    rel[:, :, 0] += wh - 1
    rel[:, :, 1] += ww - 1
    rel[:, :, 0] *= 2 * ww - 1
    return rel.sum(axis=-1).astype(np.int64)  # (N, N)


def _precompute_shift_attn_mask(
    h: int, w: int, window_size: int, shift_size: int
) -> np.ndarray | None:
    """Compute shifted-window attention mask on PADDED spatial dims.

    Pads to nearest multiple of window_size so reshape is valid.
    """
    if shift_size == 0:
        return None
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    hp, wp = h + pad_h, w + pad_w
    img_mask = np.zeros((1, hp, wp, 1), dtype=np.float32)
    h_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    w_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    cnt = 0
    for hs in h_slices:
        for ws in w_slices:
            img_mask[:, hs, ws, :] = cnt
            cnt += 1
    nh, nw = hp // window_size, wp // window_size
    mw = img_mask.reshape(1, nh, window_size, nw, window_size, 1)
    mw = mw.transpose(0, 1, 3, 2, 4, 5).reshape(nh * nw, window_size * window_size)
    attn_mask = mw[:, np.newaxis, :] - mw[:, :, np.newaxis]
    return np.where(attn_mask != 0, -100.0, 0.0).astype(np.float32)


class _SwinSelfAttention(nn.Module):
    """Swin windowed self-attention with relative position bias.

    Weight naming matches HF ``SwinSelfAttention``:
        query, key, value, relative_position_bias_table, relative_position_index
    """

    def __init__(self, dim: int, num_heads: int, window_size: int):
        super().__init__()
        self._num_heads = num_heads
        self._head_dim = dim // num_heads
        self._n_tokens = window_size * window_size
        self._scale = (dim // num_heads) ** -0.5
        self.query = Linear(dim, dim, bias=True)
        self.key = Linear(dim, dim, bias=True)
        self.value = Linear(dim, dim, bias=True)
        n_rel = (2 * window_size - 1) ** 2
        self.relative_position_bias_table = nn.Parameter((n_rel, num_heads))
        rpi = _precompute_relative_position_index(window_size)
        self.relative_position_index = nn.Parameter(rpi.shape, data=ir.tensor(rpi))

    def forward(
        self,
        op: builder.OpBuilder,
        x: ir.Value,
        attn_mask: np.ndarray | None,
    ) -> ir.Value:
        nh = self._num_heads
        hd = self._head_dim
        n = self._n_tokens
        dim = nh * hd

        # x: (batch_wins, n, dim) — use constant shapes with -1 for batch
        q = op.Reshape(self.query(op, x), [-1, n, nh, hd])
        k = op.Reshape(self.key(op, x), [-1, n, nh, hd])
        v = op.Reshape(self.value(op, x), [-1, n, nh, hd])
        q = op.Transpose(q, perm=[0, 2, 1, 3])
        k = op.Transpose(k, perm=[0, 2, 1, 3])
        v = op.Transpose(v, perm=[0, 2, 1, 3])

        attn = op.Mul(
            op.MatMul(q, op.Transpose(k, perm=[0, 1, 3, 2])),
            op.Constant(value_float=self._scale),
        )  # (batch_wins, nh, n, n)

        # Relative position bias: (1, nh, n, n)
        flat_idx = op.Reshape(self.relative_position_index, [-1])
        bias = op.Gather(self.relative_position_bias_table, flat_idx, axis=0)
        bias = op.Reshape(bias, [n, n, nh])
        bias = op.Transpose(bias, perm=[2, 0, 1])
        bias = op.Unsqueeze(bias, [0])
        attn = op.Add(attn, bias)

        if attn_mask is not None:
            mask_const = op.Constant(value=ir.tensor(attn_mask[:, np.newaxis]))
            n_wins = attn_mask.shape[0]
            # Reshape: (batch_wins, nh, n, n) → (batch, n_wins, nh, n, n)
            attn = op.Reshape(attn, [-1, n_wins, nh, n, n])
            attn = op.Add(attn, mask_const)
            attn = op.Reshape(attn, [-1, nh, n, n])

        attn = op.Softmax(attn, axis=-1)
        out = op.MatMul(attn, v)
        out = op.Transpose(out, perm=[0, 2, 1, 3])
        return op.Reshape(out, [-1, n, dim])


class _SwinAttentionOutput(nn.Module):
    """Attention output projection. HF name: ``attention.output.dense``."""

    def __init__(self, dim: int):
        super().__init__()
        self.dense = Linear(dim, dim, bias=True)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        return self.dense(op, x)


class _SwinAttention(nn.Module):
    """Combined self-attention + output. HF name: ``attention``."""

    def __init__(self, dim: int, num_heads: int, window_size: int):
        super().__init__()
        self_attn = _SwinSelfAttention(dim, num_heads, window_size)
        self.self = self_attn
        self.output = _SwinAttentionOutput(dim)

    def forward(
        self,
        op: builder.OpBuilder,
        x: ir.Value,
        attn_mask: np.ndarray | None,
    ) -> ir.Value:
        return self.output(op, self.self(op, x, attn_mask))


class _SwinIntermediate(nn.Module):
    """FFN up-projection with GELU. HF name: ``intermediate.dense``."""

    def __init__(self, dim: int, intermediate_size: int):
        super().__init__()
        self.dense = Linear(dim, intermediate_size, bias=True)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        return op.Gelu(self.dense(op, x))


class _SwinOutput(nn.Module):
    """FFN down-projection. HF name: ``output.dense``."""

    def __init__(self, intermediate_size: int, dim: int):
        super().__init__()
        self.dense = Linear(intermediate_size, dim, bias=True)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        return self.dense(op, x)


class _SwinBlock(nn.Module):
    """One Swin Transformer block with optional cyclic shift.

    HF name: ``encoder.layers.{stage}.blocks.{block}``
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        h: int,
        w: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self._H = h
        self._W = w
        self._dim = dim
        self._window_size = window_size
        self._shift_size = shift_size
        self._n_tokens = window_size * window_size

        # Pad spatial dims to nearest multiple of window_size
        pad_h = (window_size - h % window_size) % window_size
        pad_w = (window_size - w % window_size) % window_size
        self._pad_h = pad_h
        self._pad_w = pad_w
        self._Hp = h + pad_h  # padded height
        self._Wp = w + pad_w  # padded width

        intermediate_size = int(dim * mlp_ratio)
        self.layernorm_before = LayerNorm(dim, eps=1e-5)
        self.attention = _SwinAttention(dim, num_heads, window_size)
        self.layernorm_after = LayerNorm(dim, eps=1e-5)
        self.intermediate = _SwinIntermediate(dim, intermediate_size)
        self.output = _SwinOutput(intermediate_size, dim)

        # Mask uses padded spatial dims
        self._attn_mask = _precompute_shift_attn_mask(h, w, window_size, shift_size)

    def _cyclic_shift(self, op: builder.OpBuilder, x: ir.Value, neg: bool) -> ir.Value:
        """Cyclic shift of (batch, hp, wp, c) along h and w.

        Uses PADDED spatial dims (self._Hp, self._Wp).
        """
        s = self._shift_size
        hp, wp = self._Hp, self._Wp
        if neg:
            a = op.Slice(x, [s], [hp], [1])
            b = op.Slice(x, [0], [s], [1])
            x = op.Concat(a, b, axis=1)
            a = op.Slice(x, [s], [wp], [2])
            b = op.Slice(x, [0], [s], [2])
            return op.Concat(a, b, axis=2)
        else:
            a = op.Slice(x, [hp - s], [hp], [1])
            b = op.Slice(x, [0], [hp - s], [1])
            x = op.Concat(a, b, axis=1)
            a = op.Slice(x, [wp - s], [wp], [2])
            b = op.Slice(x, [0], [wp - s], [2])
            return op.Concat(a, b, axis=2)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value) -> ir.Value:
        # hidden_states: (batch, H*W, C)
        h, w = self._H, self._W
        hp, wp = self._Hp, self._Wp
        ws = self._window_size
        nh, nw = hp // ws, wp // ws
        n_tokens = self._n_tokens
        dim = self._dim

        shortcut = hidden_states

        x = self.layernorm_before(op, hidden_states)

        # Reshape to 2D spatial: (batch, h, w, C)
        x = op.Reshape(x, [-1, h, w, dim])

        # Pad to multiple of window_size if needed
        if self._pad_h > 0 or self._pad_w > 0:
            # Pad format: [begin_d0,...,begin_dN, end_d0,...,end_dN]
            x = op.Pad(
                x,
                op.Constant(value_ints=[0, 0, 0, 0, 0, self._pad_h, self._pad_w, 0]),
                op.Constant(value_float=0.0),
            )

        if self._shift_size > 0:
            x = self._cyclic_shift(op, x, neg=True)

        # Window partition: (batch*nh*nw, ws*ws, C)
        x = op.Reshape(x, [-1, nh, ws, nw, ws, dim])
        x = op.Transpose(x, perm=[0, 1, 3, 2, 4, 5])
        x = op.Reshape(x, [-1, n_tokens, dim])

        x = self.attention(op, x, self._attn_mask)

        # Window unpartition
        x = op.Reshape(x, [-1, nh, nw, ws, ws, dim])
        x = op.Transpose(x, perm=[0, 1, 3, 2, 4, 5])
        x = op.Reshape(x, [-1, hp, wp, dim])

        if self._shift_size > 0:
            x = self._cyclic_shift(op, x, neg=False)

        # Crop back to original spatial size if padded
        if self._pad_h > 0 or self._pad_w > 0:
            x = op.Slice(x, [0], [h], [1])
            x = op.Slice(x, [0], [w], [2])

        x = op.Reshape(x, [-1, h * w, dim])

        hidden_states = op.Add(shortcut, x)

        layer_out = self.layernorm_after(op, hidden_states)
        layer_out = self.intermediate(op, layer_out)
        layer_out = self.output(op, layer_out)

        return op.Add(hidden_states, layer_out)


class _SwinPatchMerging(nn.Module):
    """Patch merging: combine 2x2 patches → double channels, halve spatial.

    HF name: ``encoder.layers.{stage}.downsample``
    """

    def __init__(self, h: int, w: int, in_dim: int):
        super().__init__()
        self._H = h
        self._W = w
        self._in_dim = in_dim
        # Output spatial dims: ceil(h/2), ceil(w/2)
        self._out_h = (h + 1) // 2
        self._out_w = (w + 1) // 2
        self.norm = LayerNorm(4 * in_dim, eps=1e-5)
        self.reduction = Linear(4 * in_dim, 2 * in_dim, bias=False)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        h, w = self._H, self._W
        dim = self._in_dim
        oh, ow = self._out_h, self._out_w
        x = op.Reshape(x, [-1, h, w, dim])

        # Pad to even dims if needed (matches HF SwinPatchMerging)
        if h % 2 != 0:
            # Pad one row at bottom: (B, h+1, w, dim)
            x = op.Pad(
                x,
                op.Constant(value_ints=[0, 0, 0, 0, 0, 1, 0, 0]),
                op.Constant(value_float=0.0),
            )
        if w % 2 != 0:
            # Pad one col at right: (B, h_padded, w+1, dim)
            x = op.Pad(
                x,
                op.Constant(value_ints=[0, 0, 0, 0, 0, 0, 0, 1]),
                op.Constant(value_float=0.0),
            )

        hp = h + (h % 2)  # padded height
        wp = w + (w % 2)  # padded width
        x0 = op.Slice(op.Slice(x, [0], [hp], [1], [2]), [0], [wp], [2], [3])
        x1 = op.Slice(op.Slice(x, [1], [hp], [1], [2]), [0], [wp], [2], [3])
        x2 = op.Slice(op.Slice(x, [0], [hp], [1], [2]), [1], [wp], [2], [3])
        x3 = op.Slice(op.Slice(x, [1], [hp], [1], [2]), [1], [wp], [2], [3])

        x = op.Concat(x0, x1, x2, x3, axis=-1)
        x = op.Reshape(x, [-1, oh * ow, 4 * dim])
        x = self.norm(op, x)
        return self.reduction(op, x)


class _SwinStage(nn.Module):
    """One Swin stage: multiple blocks + optional downsample.

    HF name: ``encoder.layers.{stage}``
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        depth: int,
        h: int,
        w: int,
        window_size: int,
        mlp_ratio: float,
        downsample: bool,
    ):
        super().__init__()
        # Clamp window_size to spatial dims
        ws = min(window_size, h, w)

        blocks = []
        for i in range(depth):
            shift = ws // 2 if (i % 2 == 1) else 0
            blocks.append(_SwinBlock(dim, num_heads, h, w, ws, shift, mlp_ratio))
        self.blocks = nn.ModuleList(blocks)

        if downsample:
            self.downsample = _SwinPatchMerging(h, w, dim)
        else:
            self.downsample = None

    def forward(
        self, op: builder.OpBuilder, hidden_states: ir.Value
    ) -> tuple[ir.Value, ir.Value]:
        """Returns (output_after_downsample, output_before_downsample)."""
        for block in self.blocks:
            hidden_states = block(op, hidden_states)
        before_ds = hidden_states
        if self.downsample is not None:
            hidden_states = self.downsample(op, hidden_states)
        return hidden_states, before_ds


class _SwinEmbeddings(nn.Module):
    """Swin patch embeddings.

    HF name: ``embeddings``
    Weight keys: patch_embeddings.projection.{weight,bias}, norm.{weight,bias}
    """

    def __init__(self, num_channels: int, embed_dim: int, patch_size: int, image_size: int):
        super().__init__()
        self._H = image_size // patch_size
        self._W = image_size // patch_size
        self._embed_dim = embed_dim

        class _PatchEmbed(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = Conv2d(num_channels, embed_dim, patch_size, patch_size)

            def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
                return self.projection(op, x)

        self.patch_embeddings = _PatchEmbed()
        self.norm = LayerNorm(embed_dim, eps=1e-5)

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value) -> ir.Value:
        # pixel_values: (B, C, H, W)
        x = self.patch_embeddings(op, pixel_values)  # (B, embed_dim, pH, pW)
        # Flatten spatial dims: (B, pH*pW, embed_dim) — use constant shape
        hw = self._H * self._W
        x = op.Reshape(x, [-1, hw, self._embed_dim])
        return self.norm(op, x)


class _SwinBackbone(nn.Module):
    """Swin Transformer backbone producing multi-scale features.

    HF name: ``backbone.conv_encoder.model``
    Returns feature maps from specified output stages (e.g. stages 2,3,4).
    """

    def __init__(self, config: GroundingDinoConfig):
        super().__init__()
        bc = config.backbone_config
        embed_dim = bc["embed_dim"]
        depths = bc["depths"]
        num_heads = bc["num_heads"]
        window_size = bc["window_size"]
        patch_size = bc.get("patch_size", 4)
        num_channels = bc.get("num_channels", 3)
        image_size = config.image_size
        mlp_ratio = bc.get("mlp_ratio", 4.0)
        out_indices = bc.get("out_indices", [2, 3, 4])
        # out_indices uses 1-based: stage1=index1, stage2=index2, etc.
        # Convert to 0-based stage indices
        self._out_stage_indices = [i - 1 for i in out_indices]

        self.embeddings = _SwinEmbeddings(num_channels, embed_dim, patch_size, image_size)

        num_stages = len(depths)
        layers = []
        h = image_size // patch_size
        w = image_size // patch_size
        dim = embed_dim
        self._stage_dims = []
        self._stage_spatial = []

        for stage_id in range(num_stages):
            downsample = stage_id < num_stages - 1
            layers.append(
                _SwinStage(
                    dim=dim,
                    num_heads=num_heads[stage_id],
                    depth=depths[stage_id],
                    h=h,
                    w=w,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    downsample=downsample,
                )
            )
            self._stage_dims.append(dim)
            self._stage_spatial.append((h, w))
            if downsample:
                h = (h + 1) // 2  # ceil division for odd spatial dims
                w = (w + 1) // 2
                dim *= 2

        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(layers)

        # Per-stage output norms for multi-scale features
        # HF uses hidden_states_norms with "stage{i+1}" naming
        # We use a dict-like approach matching HF weight keys
        norms = {}
        for stage_id in self._out_stage_indices:
            norms[f"stage{stage_id + 1}"] = LayerNorm(self._stage_dims[stage_id], eps=1e-5)
        self.hidden_states_norms = nn.Module()
        for name, norm in norms.items():
            setattr(self.hidden_states_norms, name, norm)
        self._norm_names = list(norms.keys())

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value) -> list[ir.Value]:
        """Returns list of feature maps from output stages, each as (B, C, H, W)."""
        x = self.embeddings(op, pixel_values)

        feature_maps = []
        for stage_id, stage in enumerate(self.encoder.layers):
            x, before_ds = stage(op, x)
            if stage_id in self._out_stage_indices:
                norm_name = f"stage{stage_id + 1}"
                norm = getattr(self.hidden_states_norms, norm_name)
                normed = norm(op, before_ds)
                # Reshape from (B, H*W, C) to (B, C, H, W)
                h, w = self._stage_spatial[stage_id]
                c = self._stage_dims[stage_id]
                feat = op.Reshape(normed, [-1, h, w, c])
                feat = op.Transpose(feat, perm=[0, 3, 1, 2])  # (B, C, H, W)
                feature_maps.append(feat)

        return feature_maps


# --------------------------------------------------------------------------
# Input projection: Conv2d + GroupNorm per feature level
# --------------------------------------------------------------------------


class _InputProjConvNorm(nn.Module):
    """Conv2d + GroupNorm input projection for one feature level.

    HF weight naming: ``input_proj_vision.{i}.0`` (Conv) + ``.1`` (GroupNorm).
    We use a Sequential-like wrapper with indices matching HF.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
    ):
        super().__init__()
        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2,
        )
        self.norm = GroupNorm(32, out_channels)

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        return self.norm(op, self.conv(op, x))


# --------------------------------------------------------------------------
# Sine position embeddings for 2D feature maps (precomputed as constants)
# --------------------------------------------------------------------------


def _precompute_sine_pos_embed_2d(
    h: int,
    w: int,
    d_model: int,
    temperature: float = 20.0,
) -> np.ndarray:
    """Precompute 2D sine position embeddings as a numpy array.

    Returns: (1, d_model, h, w) float32 array.
    """
    embedding_dim = d_model // 2
    scale = 2 * math.pi

    dim_t = np.arange(embedding_dim, dtype=np.float32)
    dim_t = temperature ** (2 * np.floor(dim_t / 2) / embedding_dim)

    # Normalized coordinate grids
    y_embed = (np.arange(h, dtype=np.float32) + 0.5) / h * scale
    x_embed = (np.arange(w, dtype=np.float32) + 0.5) / w * scale

    # Position encoding for y: (H, embedding_dim)
    pos_y = y_embed[:, np.newaxis] / dim_t[np.newaxis, :]
    pos_y_sin = np.sin(pos_y[:, 0::2])
    pos_y_cos = np.cos(pos_y[:, 1::2])
    # Interleave: (H, embedding_dim)
    pos_y_interleaved = np.zeros((h, embedding_dim), dtype=np.float32)
    pos_y_interleaved[:, 0::2] = pos_y_sin
    pos_y_interleaved[:, 1::2] = pos_y_cos

    # Position encoding for x: (W, embedding_dim)
    pos_x = x_embed[:, np.newaxis] / dim_t[np.newaxis, :]
    pos_x_sin = np.sin(pos_x[:, 0::2])
    pos_x_cos = np.cos(pos_x[:, 1::2])
    pos_x_interleaved = np.zeros((w, embedding_dim), dtype=np.float32)
    pos_x_interleaved[:, 0::2] = pos_x_sin
    pos_x_interleaved[:, 1::2] = pos_x_cos

    # Combine: (H, W, d_model) via broadcast
    pos_y_hw = np.tile(pos_y_interleaved[:, np.newaxis, :], (1, w, 1))
    pos_x_hw = np.tile(pos_x_interleaved[np.newaxis, :, :], (h, 1, 1))
    pos = np.concatenate([pos_y_hw, pos_x_hw], axis=-1)

    # (1, d_model, H, W)
    return pos.transpose(2, 0, 1)[np.newaxis].astype(np.float32)


# --------------------------------------------------------------------------
# Sine position embedding for decoder queries (1D)
# --------------------------------------------------------------------------


def _get_sine_pos_embed(
    op: builder.OpBuilder,
    pos_tensor: ir.Value,
    num_pos_feats: int,
    temperature: int = 10000,
    exchange_xy: bool = True,
) -> ir.Value:
    """Generate sine position embedding from reference points.

    Args:
        pos_tensor: (B, Q, 4) coordinate tensor [x, y, w, h]
        num_pos_feats: projected shape per float
        temperature: frequency temperature
        exchange_xy: swap x and y in output
    Returns: (B, Q, 4 * num_pos_feats) embeddings
    """
    scale = 2 * math.pi

    dim_t_vals = np.arange(num_pos_feats, dtype=np.float32)
    dim_t_vals = temperature ** (2 * np.floor(dim_t_vals / 2) / num_pos_feats)
    dim_t = op.Constant(value=ir.tensor(dim_t_vals))

    # Scale coordinates
    pos_scaled = op.Mul(pos_tensor, op.Constant(value_float=scale))

    # Process each of the 4 coordinates
    coord_embeddings = []
    for i in range(4):
        coord = op.Gather(pos_scaled, op.Constant(value_int=i), axis=-1)
        coord = op.Unsqueeze(coord, [-1])  # (B, Q, 1)
        sin_x = op.Div(coord, dim_t)  # (B, Q, num_pos_feats)

        # Apply sin to even indices, cos to odd indices
        even_idx = list(range(0, num_pos_feats, 2))
        odd_idx = list(range(1, num_pos_feats, 2))

        sin_even = op.Sin(op.Gather(sin_x, op.Constant(value_ints=even_idx), axis=-1))
        cos_odd = op.Cos(op.Gather(sin_x, op.Constant(value_ints=odd_idx), axis=-1))

        # Interleave: concat along a new dim and flatten
        # sin: (B, Q, F/2), cos: (B, Q, F/2) → interleave to (B, Q, F)
        # Use unsqueeze + concat + reshape
        batch_q = op.Shape(sin_even, start=0, end=2)
        interleaved = op.Reshape(
            op.Concat(
                op.Unsqueeze(sin_even, [-1]),
                op.Unsqueeze(cos_odd, [-1]),
                axis=-1,
            ),
            op.Concat(batch_q, op.Constant(value_ints=[num_pos_feats]), axis=0),
        )
        coord_embeddings.append(interleaved)

    # Optionally swap x and y
    if exchange_xy and len(coord_embeddings) >= 2:
        coord_embeddings[0], coord_embeddings[1] = (
            coord_embeddings[1],
            coord_embeddings[0],
        )

    return op.Concat(*coord_embeddings, axis=-1)


# --------------------------------------------------------------------------
# Multi-scale deformable attention
# (Adapted from RT-DETR — supports both 2-coord and 4-coord reference points)
# --------------------------------------------------------------------------


class _DeformableAttention(nn.Module):
    """Multi-scale deformable attention using ONNX GridSample.

    Supports both 2-coordinate (encoder, normalized by spatial shapes)
    and 4-coordinate (decoder, using width/height) reference points.

    Weight naming matches HF ``GroundingDinoMultiscaleDeformableAttention``:
        sampling_offsets, attention_weights, value_proj, output_proj
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_levels: int,
        n_points: int,
    ):
        super().__init__()
        self._d_model = d_model
        self._n_heads = n_heads
        self._n_levels = n_levels
        self._n_points = n_points
        self._d_head = d_model // n_heads

        self.sampling_offsets = Linear(d_model, n_heads * n_levels * n_points * 2, bias=True)
        self.attention_weights = Linear(d_model, n_heads * n_levels * n_points, bias=True)
        self.value_proj = Linear(d_model, d_model, bias=True)
        self.output_proj = Linear(d_model, d_model, bias=True)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        encoder_hidden_states: ir.Value,
        reference_points: ir.Value,
        spatial_shapes: list[tuple[int, int]],
        level_start_index: list[int],
        num_coordinates: int,
        position_embeddings: ir.Value | None = None,
    ) -> ir.Value:
        """Multi-scale deformable attention.

        Args:
            hidden_states: (B, Q, D) query features
            encoder_hidden_states: (B, S, D) flattened multi-scale features
            reference_points: (B, Q, L, 2) or (B, Q, L, 4) normalized coords
            spatial_shapes: [(H, W)] per level
            level_start_index: [start_idx] per level
            num_coordinates: 2 (encoder) or 4 (decoder)
            position_embeddings: optional position to add to hidden_states
        """
        n_heads = self._n_heads
        n_levels = self._n_levels
        n_points = self._n_points
        d_head = self._d_head

        if position_embeddings is not None:
            hidden_states_with_pos = op.Add(hidden_states, position_embeddings)
        else:
            hidden_states_with_pos = hidden_states

        # Project values: (B, S, D) → (B, S, n_heads, d_head)
        value = self.value_proj(op, encoder_hidden_states)
        value = op.Reshape(
            value,
            op.Concat(
                op.Shape(value, start=0, end=2),
                op.Constant(value_ints=[n_heads, d_head]),
                axis=0,
            ),
        )

        # Predict offsets: (B, Q, H*L*P*2) → (B, Q, H, L, P, 2)
        offsets = self.sampling_offsets(op, hidden_states_with_pos)
        offsets = op.Reshape(
            offsets,
            op.Concat(
                op.Shape(offsets, start=0, end=2),
                op.Constant(value_ints=[n_heads, n_levels, n_points, 2]),
                axis=0,
            ),
        )

        # Predict attention weights: softmax over L*P
        attn_w = self.attention_weights(op, hidden_states_with_pos)
        attn_w = op.Reshape(
            attn_w,
            op.Concat(
                op.Shape(attn_w, start=0, end=2),
                op.Constant(value_ints=[n_heads, n_levels * n_points]),
                axis=0,
            ),
        )
        attn_w = op.Softmax(attn_w, axis=-1)
        attn_w = op.Reshape(
            attn_w,
            op.Concat(
                op.Shape(attn_w, start=0, end=2),
                op.Constant(value_ints=[n_heads, n_levels, n_points]),
                axis=0,
            ),
        )

        # Compute sampling locations from reference_points + offsets
        if num_coordinates == 2:
            # Encoder path: reference_points (B, Q, L, 2)
            # offset_normalizer per level: [W, H]
            offset_normalizer = np.array(
                [[w, h] for h, w in spatial_shapes], dtype=np.float32
            )  # (L, 2)
            norm_const = op.Constant(value=ir.tensor(offset_normalizer))
            # offsets: (B, Q, H, L, P, 2) / norm (1, 1, 1, L, 1, 2)
            norm_expanded = op.Reshape(norm_const, [1, 1, 1, n_levels, 1, 2])
            normalized_offsets = op.Div(offsets, norm_expanded)

            # ref: (B, Q, 1, L, 1, 2)
            ref_expanded = op.Unsqueeze(op.Unsqueeze(reference_points, [2]), [4])
            sampling_locs = op.Add(ref_expanded, normalized_offsets)
        else:
            # Decoder path: reference_points (B, Q, L, 4) [x, y, w, h]
            ref_xy = op.Slice(
                reference_points,
                op.Constant(value_ints=[0]),
                op.Constant(value_ints=[2]),
                op.Constant(value_ints=[3]),
            )  # (B, Q, L, 2)
            ref_wh = op.Slice(
                reference_points,
                op.Constant(value_ints=[2]),
                op.Constant(value_ints=[4]),
                op.Constant(value_ints=[3]),
            )  # (B, Q, L, 2)

            ref_xy = op.Unsqueeze(op.Unsqueeze(ref_xy, [2]), [4])
            ref_wh = op.Unsqueeze(op.Unsqueeze(ref_wh, [2]), [4])

            normalized_offsets = op.Mul(
                op.Div(offsets, op.Constant(value_float=float(n_points))),
                op.Mul(ref_wh, op.Constant(value_float=0.5)),
            )
            sampling_locs = op.Add(ref_xy, normalized_offsets)

        # Convert to grid_sample coords: grid = 2 * loc - 1
        sampling_grids = op.Sub(
            op.Mul(sampling_locs, op.Constant(value_float=2.0)),
            op.Constant(value_float=1.0),
        )  # (B, Q, H, L, P, 2)

        # GridSample per level
        sampled_values = []
        batch = op.Shape(value, start=0, end=1)

        for level_id, (h_l, w_l) in enumerate(spatial_shapes):
            start = level_start_index[level_id]
            length = h_l * w_l

            # value_l: (B, h*w, H, d_head) → (B*H, d_head, h, w)
            value_l = op.Slice(
                value,
                op.Constant(value_ints=[start]),
                op.Constant(value_ints=[start + length]),
                op.Constant(value_ints=[1]),
            )
            value_l = op.Transpose(value_l, perm=[0, 2, 3, 1])
            value_l = op.Reshape(
                value_l,
                op.Concat(
                    op.Mul(batch, op.Constant(value_ints=[n_heads])),
                    op.Constant(value_ints=[d_head, h_l, w_l]),
                    axis=0,
                ),
            )

            # grid_l: (B, Q, H, P, 2) → (B*H, Q, P, 2)
            grid_l = op.Gather(sampling_grids, op.Constant(value_int=level_id), axis=3)
            grid_l = op.Transpose(grid_l, perm=[0, 2, 1, 3, 4])
            grid_l = op.Reshape(
                grid_l,
                op.Concat(
                    op.Mul(batch, op.Constant(value_ints=[n_heads])),
                    op.Shape(grid_l, start=2, end=5),
                    axis=0,
                ),
            )

            # GridSample: (B*H, d_head, h, w) + (B*H, Q, P, 2) → (B*H, d_head, Q, P)
            sampled_l = op.GridSample(
                value_l,
                grid_l,
                align_corners=0,
                mode="bilinear",
                padding_mode="zeros",
            )
            sampled_values.append(sampled_l)

        # Stack levels and flatten: (B*H, d_head, Q, L*P)
        output = op.Concat(*sampled_values, axis=-1)

        # Apply attention weights
        # attn_w: (B, Q, H, L, P) → (B*H, 1, Q, L*P)
        attn_w = op.Transpose(attn_w, perm=[0, 2, 1, 3, 4])
        attn_w = op.Reshape(
            attn_w,
            op.Concat(
                op.Mul(batch, op.Constant(value_ints=[n_heads])),
                op.Constant(value_ints=[1, -1, n_levels * n_points]),
                axis=0,
            ),
        )

        # Weighted sum: (B*H, d_head, Q, L*P) * (B*H, 1, Q, L*P) → sum → (B*H, d_head, Q)
        output = op.ReduceSum(
            op.Mul(output, attn_w),
            op.Constant(value_ints=[-1]),
            keepdims=0,
        )

        # Reshape: (B*H, d_head, Q) → (B, D, Q) → (B, Q, D)
        output = op.Reshape(
            output,
            op.Concat(batch, op.Constant(value_ints=[self._d_model, -1]), axis=0),
        )
        output = op.Transpose(output, perm=[0, 2, 1])

        return self.output_proj(op, output)


# --------------------------------------------------------------------------
# Text Enhancer Layer (self-attention on text features in encoder)
# --------------------------------------------------------------------------


class _TextEnhancerSelfAttn(nn.Module):
    """Multi-head attention for text enhancer. Uses q/k/v + out_proj naming.

    HF name: ``text_enhancer_layer.self_attn``
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self._num_heads = num_heads
        self._head_dim = d_model // num_heads
        self.query = Linear(d_model, d_model, bias=True)
        self.key = Linear(d_model, d_model, bias=True)
        self.value = Linear(d_model, d_model, bias=True)
        self.out_proj = Linear(d_model, d_model, bias=True)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        position_embeddings: ir.Value | None = None,
    ) -> ir.Value:
        nh = self._num_heads
        hd = self._head_dim

        if position_embeddings is not None:
            q_input = op.Add(hidden_states, position_embeddings)
            k_input = op.Add(hidden_states, position_embeddings)
        else:
            q_input = hidden_states
            k_input = hidden_states

        batch = op.Shape(hidden_states, start=0, end=1)
        seq = op.Shape(hidden_states, start=1, end=2)

        q = op.Reshape(
            self.query(op, q_input),
            op.Concat(batch, seq, op.Constant(value_ints=[nh, hd]), axis=0),
        )
        k = op.Reshape(
            self.key(op, k_input),
            op.Concat(batch, seq, op.Constant(value_ints=[nh, hd]), axis=0),
        )
        v = op.Reshape(
            self.value(op, hidden_states),
            op.Concat(batch, seq, op.Constant(value_ints=[nh, hd]), axis=0),
        )

        q = op.Transpose(q, perm=[0, 2, 1, 3])  # (B, nh, S, hd)
        k = op.Transpose(k, perm=[0, 2, 1, 3])
        v = op.Transpose(v, perm=[0, 2, 1, 3])

        scale = self._head_dim**-0.5
        attn = op.Mul(
            op.MatMul(q, op.Transpose(k, perm=[0, 1, 3, 2])),
            op.Constant(value_float=scale),
        )
        attn = op.Softmax(attn, axis=-1)

        out = op.MatMul(attn, v)  # (B, nh, S, hd)
        out = op.Transpose(out, perm=[0, 2, 1, 3])
        out = op.Reshape(out, op.Concat(batch, seq, op.Constant(value_ints=[nh * hd]), axis=0))
        return self.out_proj(op, out)


class _TextEnhancerLayer(nn.Module):
    """Self-attention + FFN on text features in the encoder.

    HF name: ``encoder.layers.{i}.text_enhancer_layer``
    Uses half the heads/FFN dim of the main model.
    """

    def __init__(self, config: GroundingDinoConfig):
        super().__init__()
        d = config.d_model
        num_heads = config.encoder_attention_heads // 2
        ffn_dim = config.encoder_ffn_dim // 2

        self.self_attn = _TextEnhancerSelfAttn(d, num_heads)
        self.fc1 = Linear(d, ffn_dim, bias=True)
        self.fc2 = Linear(ffn_dim, d, bias=True)
        self.layer_norm_before = LayerNorm(d, eps=config.layer_norm_eps)
        self.layer_norm_after = LayerNorm(d, eps=config.layer_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        position_embeddings: ir.Value | None = None,
    ) -> ir.Value:
        residual = hidden_states
        hidden_states = self.layer_norm_before(op, hidden_states)
        hidden_states = self.self_attn(op, hidden_states, position_embeddings)
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.layer_norm_after(op, hidden_states)
        hidden_states = op.Relu(self.fc1(op, hidden_states))
        hidden_states = self.fc2(op, hidden_states)
        return op.Add(residual, hidden_states)


# --------------------------------------------------------------------------
# Bidirectional Multi-Head Attention (Vision ↔ Text Fusion)
# --------------------------------------------------------------------------


class _BiMultiHeadAttention(nn.Module):
    """Bidirectional cross-attention between vision and text features.

    HF name: ``encoder.layers.{i}.fusion_layer.attn``
    """

    def __init__(self, config: GroundingDinoConfig):
        super().__init__()
        embed_dim = config.encoder_ffn_dim // 2
        num_heads = config.encoder_attention_heads // 2
        vision_dim = text_dim = config.d_model

        self._num_heads = num_heads
        self._head_dim = embed_dim // num_heads
        self._scale = (embed_dim // num_heads) ** -0.5

        self.vision_proj = Linear(vision_dim, embed_dim)
        self.text_proj = Linear(text_dim, embed_dim)
        self.values_vision_proj = Linear(vision_dim, embed_dim)
        self.values_text_proj = Linear(text_dim, embed_dim)
        self.out_vision_proj = Linear(embed_dim, vision_dim)
        self.out_text_proj = Linear(embed_dim, text_dim)

    def forward(
        self,
        op: builder.OpBuilder,
        vision_features: ir.Value,
        text_features: ir.Value,
    ) -> tuple[ir.Value, ir.Value]:
        """Returns (delta_vision, delta_text) cross-attention outputs."""
        nh = self._num_heads
        hd = self._head_dim
        scale = self._scale

        batch = op.Shape(vision_features, start=0, end=1)

        # Vision queries/values
        v_q = self.vision_proj(op, vision_features)
        v_v = self.values_vision_proj(op, vision_features)
        v_seq = op.Shape(vision_features, start=1, end=2)

        # Text keys/values
        t_k = self.text_proj(op, text_features)
        t_v = self.values_text_proj(op, text_features)
        t_seq = op.Shape(text_features, start=1, end=2)

        # Reshape to multi-head: (B, S, nh, hd) → (B, nh, S, hd)
        def reshape_mh(x: ir.Value, seq_len: ir.Value) -> ir.Value:
            return op.Transpose(
                op.Reshape(
                    x,
                    op.Concat(
                        batch,
                        seq_len,
                        op.Constant(value_ints=[nh, hd]),
                        axis=0,
                    ),
                ),
                perm=[0, 2, 1, 3],
            )

        v_q = reshape_mh(v_q, v_seq)
        v_v = reshape_mh(v_v, v_seq)
        t_k = reshape_mh(t_k, t_seq)
        t_v = reshape_mh(t_v, t_seq)

        # Vision→Text attention: v_q @ t_k.T → softmax → @ t_v
        attn_v2t = op.Mul(
            op.MatMul(v_q, op.Transpose(t_k, perm=[0, 1, 3, 2])),
            op.Constant(value_float=scale),
        )  # (B, nh, v_seq, t_seq)
        attn_v2t = op.Softmax(attn_v2t, axis=-1)
        delta_v = op.MatMul(attn_v2t, t_v)  # (B, nh, v_seq, hd)

        # Text→Vision attention: t_k @ v_q.T → softmax → @ v_v
        # Reuse the transposed attention weights
        attn_t2v = op.Softmax(
            op.Transpose(attn_v2t, perm=[0, 1, 3, 2]), axis=-1
        )  # (B, nh, t_seq, v_seq)
        delta_t = op.MatMul(attn_t2v, v_v)  # (B, nh, t_seq, hd)

        # Reshape back: (B, nh, S, hd) → (B, S, embed_dim)
        delta_v = op.Reshape(
            op.Transpose(delta_v, perm=[0, 2, 1, 3]),
            op.Concat(batch, v_seq, op.Constant(value_ints=[nh * hd]), axis=0),
        )
        delta_t = op.Reshape(
            op.Transpose(delta_t, perm=[0, 2, 1, 3]),
            op.Concat(batch, t_seq, op.Constant(value_ints=[nh * hd]), axis=0),
        )

        return self.out_vision_proj(op, delta_v), self.out_text_proj(op, delta_t)


class _FusionLayer(nn.Module):
    """Vision-text fusion with layer-scale parameters.

    HF name: ``encoder.layers.{i}.fusion_layer``
    """

    def __init__(self, config: GroundingDinoConfig):
        super().__init__()
        d = config.d_model
        self.layer_norm_vision = LayerNorm(d, eps=config.layer_norm_eps)
        self.layer_norm_text = LayerNorm(d, eps=config.layer_norm_eps)
        self.attn = _BiMultiHeadAttention(config)
        # Layer-scale parameters (scalar per channel)
        self.vision_param = nn.Parameter((d,))
        self.text_param = nn.Parameter((d,))

    def forward(
        self,
        op: builder.OpBuilder,
        vision_features: ir.Value,
        text_features: ir.Value,
    ) -> tuple[ir.Value, ir.Value]:
        """Returns (updated_vision, updated_text)."""
        v_normed = self.layer_norm_vision(op, vision_features)
        t_normed = self.layer_norm_text(op, text_features)

        delta_v, delta_t = self.attn(op, v_normed, t_normed)

        vision_features = op.Add(vision_features, op.Mul(self.vision_param, delta_v))
        text_features = op.Add(text_features, op.Mul(self.text_param, delta_t))
        return vision_features, text_features


# --------------------------------------------------------------------------
# Encoder
# --------------------------------------------------------------------------


class _DeformableLayer(nn.Module):
    """Deformable self-attention + FFN in encoder.

    HF name: ``encoder.layers.{i}.deformable_layer``
    """

    def __init__(self, config: GroundingDinoConfig):
        super().__init__()
        d = config.d_model
        self.self_attn = _DeformableAttention(
            d,
            config.encoder_attention_heads,
            config.num_feature_levels,
            config.encoder_n_points,
        )
        self.self_attn_layer_norm = LayerNorm(d, eps=config.layer_norm_eps)
        self.fc1 = Linear(d, config.encoder_ffn_dim, bias=True)
        self.fc2 = Linear(config.encoder_ffn_dim, d, bias=True)
        self.final_layer_norm = LayerNorm(d, eps=config.layer_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        position_embeddings: ir.Value,
        reference_points: ir.Value,
        spatial_shapes: list[tuple[int, int]],
        level_start_index: list[int],
    ) -> ir.Value:
        residual = hidden_states

        hidden_states = self.self_attn(
            op,
            hidden_states,
            hidden_states,
            reference_points,
            spatial_shapes,
            level_start_index,
            num_coordinates=2,
            position_embeddings=position_embeddings,
        )
        hidden_states = op.Add(residual, hidden_states)
        hidden_states = self.self_attn_layer_norm(op, hidden_states)

        residual = hidden_states
        hidden_states = op.Relu(self.fc1(op, hidden_states))
        hidden_states = self.fc2(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)
        return self.final_layer_norm(op, hidden_states)


class _EncoderLayer(nn.Module):
    """One encoder layer: fusion → text_enhancer → deformable.

    HF name: ``encoder.layers.{i}``
    """

    def __init__(self, config: GroundingDinoConfig):
        super().__init__()
        self.text_enhancer_layer = _TextEnhancerLayer(config)
        self.fusion_layer = _FusionLayer(config)
        self.deformable_layer = _DeformableLayer(config)

    def forward(
        self,
        op: builder.OpBuilder,
        vision_features: ir.Value,
        vision_position_embedding: ir.Value,
        text_features: ir.Value,
        reference_points: ir.Value,
        spatial_shapes: list[tuple[int, int]],
        level_start_index: list[int],
    ) -> tuple[ir.Value, ir.Value]:
        """Returns (updated_vision, updated_text)."""
        # Fusion: bidirectional cross-attention
        vision_features, text_features = self.fusion_layer(op, vision_features, text_features)

        # Text enhancement: self-attention on text
        text_features = self.text_enhancer_layer(op, text_features)

        # Deformable self-attention on vision
        vision_features = self.deformable_layer(
            op,
            vision_features,
            vision_position_embedding,
            reference_points,
            spatial_shapes,
            level_start_index,
        )

        return vision_features, text_features


class _Encoder(nn.Module):
    """Grounding DINO encoder stack.

    HF name: ``encoder``
    """

    def __init__(self, config: GroundingDinoConfig):
        super().__init__()
        enc_layers = []
        for _ in range(config.encoder_layers):
            enc_layers.append(_EncoderLayer(config))
        self.layers = nn.ModuleList(enc_layers)

    def forward(
        self,
        op: builder.OpBuilder,
        vision_features: ir.Value,
        vision_position_embedding: ir.Value,
        text_features: ir.Value,
        reference_points: ir.Value,
        spatial_shapes: list[tuple[int, int]],
        level_start_index: list[int],
    ) -> tuple[ir.Value, ir.Value]:
        """Returns (vision_output, text_output)."""
        for layer in self.layers:
            vision_features, text_features = layer(
                op,
                vision_features,
                vision_position_embedding,
                text_features,
                reference_points,
                spatial_shapes,
                level_start_index,
            )
        return vision_features, text_features


# --------------------------------------------------------------------------
# Decoder
# --------------------------------------------------------------------------


class _DecoderSelfAttn(nn.Module):
    """Standard multi-head attention for decoder self-attention.

    HF name: ``decoder.layers.{i}.self_attn``
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self._num_heads = num_heads
        self._head_dim = d_model // num_heads
        self.query = Linear(d_model, d_model, bias=True)
        self.key = Linear(d_model, d_model, bias=True)
        self.value = Linear(d_model, d_model, bias=True)
        self.out_proj = Linear(d_model, d_model, bias=True)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        position_embeddings: ir.Value | None = None,
    ) -> ir.Value:
        nh = self._num_heads
        hd = self._head_dim

        if position_embeddings is not None:
            qk_input = op.Add(hidden_states, position_embeddings)
        else:
            qk_input = hidden_states

        batch = op.Shape(hidden_states, start=0, end=1)
        seq = op.Shape(hidden_states, start=1, end=2)

        q = op.Reshape(
            self.query(op, qk_input),
            op.Concat(batch, seq, op.Constant(value_ints=[nh, hd]), axis=0),
        )
        k = op.Reshape(
            self.key(op, qk_input),
            op.Concat(batch, seq, op.Constant(value_ints=[nh, hd]), axis=0),
        )
        v = op.Reshape(
            self.value(op, hidden_states),
            op.Concat(batch, seq, op.Constant(value_ints=[nh, hd]), axis=0),
        )

        q = op.Transpose(q, perm=[0, 2, 1, 3])
        k = op.Transpose(k, perm=[0, 2, 1, 3])
        v = op.Transpose(v, perm=[0, 2, 1, 3])

        scale = hd**-0.5
        attn = op.Mul(
            op.MatMul(q, op.Transpose(k, perm=[0, 1, 3, 2])),
            op.Constant(value_float=scale),
        )
        attn = op.Softmax(attn, axis=-1)

        out = op.MatMul(attn, v)
        out = op.Transpose(out, perm=[0, 2, 1, 3])
        out = op.Reshape(out, op.Concat(batch, seq, op.Constant(value_ints=[nh * hd]), axis=0))
        return self.out_proj(op, out)


class _DecoderTextCrossAttn(nn.Module):
    """Cross-attention from decoder queries to text features.

    HF name: ``decoder.layers.{i}.encoder_attn_text``
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self._num_heads = num_heads
        self._head_dim = d_model // num_heads
        self.query = Linear(d_model, d_model, bias=True)
        self.key = Linear(d_model, d_model, bias=True)
        self.value = Linear(d_model, d_model, bias=True)
        self.out_proj = Linear(d_model, d_model, bias=True)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        text_features: ir.Value,
        position_embeddings: ir.Value | None = None,
    ) -> ir.Value:
        nh = self._num_heads
        hd = self._head_dim

        if position_embeddings is not None:
            q_input = op.Add(hidden_states, position_embeddings)
        else:
            q_input = hidden_states

        batch = op.Shape(hidden_states, start=0, end=1)
        q_seq = op.Shape(hidden_states, start=1, end=2)
        kv_seq = op.Shape(text_features, start=1, end=2)

        q = op.Reshape(
            self.query(op, q_input),
            op.Concat(batch, q_seq, op.Constant(value_ints=[nh, hd]), axis=0),
        )
        k = op.Reshape(
            self.key(op, text_features),
            op.Concat(batch, kv_seq, op.Constant(value_ints=[nh, hd]), axis=0),
        )
        v = op.Reshape(
            self.value(op, text_features),
            op.Concat(batch, kv_seq, op.Constant(value_ints=[nh, hd]), axis=0),
        )

        q = op.Transpose(q, perm=[0, 2, 1, 3])
        k = op.Transpose(k, perm=[0, 2, 1, 3])
        v = op.Transpose(v, perm=[0, 2, 1, 3])

        scale = hd**-0.5
        attn = op.Mul(
            op.MatMul(q, op.Transpose(k, perm=[0, 1, 3, 2])),
            op.Constant(value_float=scale),
        )
        attn = op.Softmax(attn, axis=-1)

        out = op.MatMul(attn, v)
        out = op.Transpose(out, perm=[0, 2, 1, 3])
        out = op.Reshape(
            out, op.Concat(batch, q_seq, op.Constant(value_ints=[nh * hd]), axis=0)
        )
        return self.out_proj(op, out)


class _DecoderLayer(nn.Module):
    """Grounding DINO decoder layer.

    self_attn → text_cross_attn → deformable_cross_attn → FFN
    HF name: ``decoder.layers.{i}``
    """

    def __init__(self, config: GroundingDinoConfig):
        super().__init__()
        d = config.d_model

        self.self_attn = _DecoderSelfAttn(d, config.decoder_attention_heads)
        self.self_attn_layer_norm = LayerNorm(d, eps=config.layer_norm_eps)

        self.encoder_attn_text = _DecoderTextCrossAttn(d, config.decoder_attention_heads)
        self.encoder_attn_text_layer_norm = LayerNorm(d, eps=config.layer_norm_eps)

        self.encoder_attn = _DeformableAttention(
            d,
            config.decoder_attention_heads,
            config.num_feature_levels,
            config.decoder_n_points,
        )
        self.encoder_attn_layer_norm = LayerNorm(d, eps=config.layer_norm_eps)

        self.fc1 = Linear(d, config.decoder_ffn_dim, bias=True)
        self.fc2 = Linear(config.decoder_ffn_dim, d, bias=True)
        self.final_layer_norm = LayerNorm(d, eps=config.layer_norm_eps)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        position_embeddings: ir.Value,
        text_features: ir.Value,
        vision_features: ir.Value,
        reference_points: ir.Value,
        spatial_shapes: list[tuple[int, int]],
        level_start_index: list[int],
    ) -> ir.Value:
        # Self-attention
        residual = hidden_states
        hidden_states = self.self_attn(op, hidden_states, position_embeddings)
        hidden_states = op.Add(residual, hidden_states)
        hidden_states = self.self_attn_layer_norm(op, hidden_states)

        # Text cross-attention
        residual = hidden_states
        hidden_states = self.encoder_attn_text(
            op, hidden_states, text_features, position_embeddings
        )
        hidden_states = op.Add(residual, hidden_states)
        hidden_states = self.encoder_attn_text_layer_norm(op, hidden_states)

        # Deformable cross-attention to vision
        residual = hidden_states
        hidden_states = self.encoder_attn(
            op,
            hidden_states,
            vision_features,
            reference_points,
            spatial_shapes,
            level_start_index,
            num_coordinates=4,
            position_embeddings=position_embeddings,
        )
        hidden_states = op.Add(residual, hidden_states)
        hidden_states = self.encoder_attn_layer_norm(op, hidden_states)

        # FFN
        residual = hidden_states
        hidden_states = op.Relu(self.fc1(op, hidden_states))
        hidden_states = self.fc2(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)
        return self.final_layer_norm(op, hidden_states)


class _ReferencePointsHead(nn.Module):
    """2-layer MLP projecting sine embeddings of reference points to query pos.

    HF name: ``decoder.reference_points_head``
    """

    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                Linear(input_dim, d_model, bias=True),
                Linear(d_model, d_model, bias=True),
            ]
        )

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        x = op.Relu(self.layers[0](op, x))
        return self.layers[1](op, x)


class _MLPPredictionHead(nn.Module):
    """MLP for bounding box regression (3 layers).

    HF name: ``bbox_embed.{i}`` or ``decoder.bbox_embed.{i}``
    """

    def __init__(self, d_model: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                Linear(d_model, hidden_dim, bias=True),
                Linear(hidden_dim, hidden_dim, bias=True),
                Linear(hidden_dim, output_dim, bias=True),
            ]
        )

    def forward(self, op: builder.OpBuilder, x: ir.Value) -> ir.Value:
        for i, layer in enumerate(self.layers):
            x = layer(op, x)
            if i < len(self.layers) - 1:
                x = op.Relu(x)
        return x


class _Decoder(nn.Module):
    """Grounding DINO decoder with iterative bbox refinement.

    HF name: ``decoder``
    """

    def __init__(self, config: GroundingDinoConfig):
        super().__init__()
        self._d_model = config.d_model

        dec_layers = []
        for _ in range(config.decoder_layers):
            dec_layers.append(_DecoderLayer(config))
        self.layers = nn.ModuleList(dec_layers)

        self.layer_norm = LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # MLP to project sine embeddings of reference points to query pos.
        # Input dim is d_model*2 (x and y sine embeddings concatenated).
        self.reference_points_head = _ReferencePointsHead(config.d_model * 2, config.d_model)

        # Per-layer bbox prediction for iterative refinement
        bbox_embeds = []
        for _ in range(config.decoder_layers):
            bbox_embeds.append(_MLPPredictionHead(config.d_model, config.d_model, 4))
        self.bbox_embed = nn.ModuleList(bbox_embeds)

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        vision_features: ir.Value,
        text_features: ir.Value,
        reference_points: ir.Value,
        spatial_shapes: list[tuple[int, int]],
        level_start_index: list[int],
        valid_ratios: ir.Value,
    ) -> tuple[ir.Value, ir.Value, ir.Value]:
        """Returns (last_hidden, intermediate_hidden, intermediate_refs)."""
        intermediate_hidden = []
        intermediate_refs = []

        for idx, decoder_layer in enumerate(self.layers):
            # Scale reference points by valid_ratios
            # reference_points: (B, Q, 4) [x,y,w,h]
            # valid_ratios: (B, L, 2) → expand to (B, 1, L, 4)
            # ref_input = ref[:,:,None,:] * cat([valid_ratios, valid_ratios], -1)[:,None,:]
            ref_input = op.Mul(
                op.Unsqueeze(reference_points, [2]),  # (B, Q, 1, 4)
                op.Concat(valid_ratios, valid_ratios, axis=-1),
            )
            # ref_input[:,None,:] → expand broadcast handles this

            # Sine position embeddings from reference points
            # Use first level's reference for query_pos
            ref_first = op.Gather(ref_input, op.Constant(value_int=0), axis=2)
            # ref_first: (B, Q, 4)
            query_pos = _get_sine_pos_embed(
                op, ref_first, self._d_model // 2
            )  # (B, Q, d_model*2) — but we want d_model
            query_pos = self.reference_points_head(op, query_pos)

            hidden_states = decoder_layer(
                op,
                hidden_states,
                query_pos,
                text_features,
                vision_features,
                ref_input,
                spatial_shapes,
                level_start_index,
            )

            # Iterative bbox refinement
            delta = self.bbox_embed[idx](op, hidden_states)
            # ref_logit = inverse_sigmoid(reference_points)
            ref_logit = op.Log(
                op.Div(
                    op.Clip(reference_points, min=1e-5, max=1.0 - 1e-5),
                    op.Sub(
                        op.Constant(value_float=1.0),
                        op.Clip(reference_points, min=1e-5, max=1.0 - 1e-5),
                    ),
                )
            )
            new_ref = op.Sigmoid(op.Add(delta, ref_logit))
            reference_points = new_ref

            intermediate_hidden.append(self.layer_norm(op, hidden_states))
            intermediate_refs.append(reference_points)

        # Stack intermediates: (B, num_layers, Q, D) and (B, num_layers, Q, 4)
        stacked_hidden = op.Unsqueeze(intermediate_hidden[0], [1])
        for ih in intermediate_hidden[1:]:
            stacked_hidden = op.Concat(stacked_hidden, op.Unsqueeze(ih, [1]), axis=1)

        stacked_refs = op.Unsqueeze(intermediate_refs[0], [1])
        for ir_ref in intermediate_refs[1:]:
            stacked_refs = op.Concat(stacked_refs, op.Unsqueeze(ir_ref, [1]), axis=1)

        return (
            self.layer_norm(op, hidden_states),
            stacked_hidden,
            stacked_refs,
        )


# --------------------------------------------------------------------------
# Top-level model
# --------------------------------------------------------------------------


class GroundingDinoForObjectDetection(nn.Module):
    """Grounding DINO for text-guided open-set object detection.

    Produces per-layer class logits (contrastive) and bounding box predictions.

    Inputs: pixel_values (B, 3, H, W), input_ids (B, T)
    Outputs: logits (B, num_queries, max_text_len), pred_boxes (B, num_queries, 4)
    """

    def __init__(self, config: GroundingDinoConfig):
        super().__init__()
        self._config = config
        d = config.d_model
        bc = config.backbone_config

        # Backbone
        self.backbone = nn.Module()
        self.backbone.conv_encoder = nn.Module()
        self.backbone.conv_encoder.model = _SwinBackbone(config)

        # Input projections: one per backbone output + extra stride-2 levels
        # Backbone outputs stages 2,3,4 with dims embed_dim*{4,8,16}
        embed_dim = bc["embed_dim"]
        backbone_out_dims = [
            embed_dim * (2 ** (i - 1)) for i in bc.get("out_indices", [2, 3, 4])
        ]
        num_backbone_outs = len(backbone_out_dims)

        input_projs = []
        for i in range(num_backbone_outs):
            input_projs.append(_InputProjConvNorm(backbone_out_dims[i], d, 1, 1))
        # Extra levels via 3x3 stride-2 conv on last backbone output
        for _ in range(config.num_feature_levels - num_backbone_outs):
            input_projs.append(_InputProjConvNorm(backbone_out_dims[-1], d, 3, 2))
        self.input_proj_vision = nn.ModuleList(input_projs)

        # Level embedding
        self.level_embed = nn.Parameter((config.num_feature_levels, d))

        # Text backbone (BERT)
        from mobius.models.bert import BertModel

        text_config = config.text_config
        # Wrap dict as object for BertModel (expects config.vocab_size, etc.)
        text_cfg_obj = _DictConfig(text_config)
        self.text_backbone = BertModel(text_cfg_obj)
        self.text_projection = Linear(text_config["hidden_size"], d)

        # Query embeddings — stored as bare parameter (like DETR).
        # HF stores this as nn.Embedding, so preprocess_weights strips
        # the trailing ".weight" suffix.
        self.query_position_embeddings = nn.Parameter([config.num_queries, d])

        # Encoder + Decoder
        self.encoder = _Encoder(config)
        self.decoder = _Decoder(config)

        # Two-stage encoder output processing
        self.enc_output = Linear(d, d, bias=True)
        self.enc_output_norm = LayerNorm(d, eps=config.layer_norm_eps)
        self.encoder_output_bbox_embed = _MLPPredictionHead(d, d, 4)

        # Per-layer bbox prediction heads (top-level, shared with decoder)
        bbox_embeds = []
        for _ in range(config.decoder_layers):
            bbox_embeds.append(_MLPPredictionHead(d, d, 4))
        self.bbox_embed = nn.ModuleList(bbox_embeds)

        # Precompute spatial info
        self._compute_spatial_info(config)

    def _compute_spatial_info(self, config: GroundingDinoConfig) -> None:
        """Precompute spatial shapes and level start indices."""
        bc = config.backbone_config
        image_size = config.image_size
        patch_size = bc.get("patch_size", 4)
        depths = bc["depths"]
        out_indices = bc.get("out_indices", [2, 3, 4])

        # Compute spatial dimensions for each output stage
        h = image_size // patch_size
        w = image_size // patch_size

        stage_spatial = []
        for stage_id in range(len(depths)):
            stage_spatial.append((h, w))
            if stage_id < len(depths) - 1:
                h = (h + 1) // 2  # ceil division for odd spatial dims
                w = (w + 1) // 2

        self._spatial_shapes = []
        for idx in out_indices:
            stage_id = idx - 1  # Convert 1-based to 0-based
            self._spatial_shapes.append(stage_spatial[stage_id])

        # Extra levels (stride-2 from last backbone output)
        last_h, last_w = stage_spatial[out_indices[-1] - 1]
        for _ in range(config.num_feature_levels - len(out_indices)):
            last_h = (last_h + 1) // 2
            last_w = (last_w + 1) // 2
            self._spatial_shapes.append((last_h, last_w))

        # Level start indices
        self._level_start_index = []
        idx = 0
        for h_l, w_l in self._spatial_shapes:
            self._level_start_index.append(idx)
            idx += h_l * w_l

    def _compute_encoder_reference_points(self, op: builder.OpBuilder) -> ir.Value:
        """Compute reference points for encoder deformable attention.

        Returns: (1, total_tokens, n_levels, 2) normalized reference points.
        """
        ref_points_list = []
        n_levels = len(self._spatial_shapes)

        for _level_id, (h_l, w_l) in enumerate(self._spatial_shapes):
            # Create meshgrid of normalized coordinates
            # y: [0.5/H, 1.5/H, ..., (H-0.5)/H]
            # x: [0.5/W, 1.5/W, ..., (W-0.5)/W]
            ys = np.linspace(0.5, h_l - 0.5, h_l, dtype=np.float32) / h_l
            xs = np.linspace(0.5, w_l - 0.5, w_l, dtype=np.float32) / w_l

            grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
            ref = np.stack([grid_x.flatten(), grid_y.flatten()], axis=-1)
            # ref: (h*w, 2) — tile across levels
            ref = np.tile(ref[:, np.newaxis, :], (1, n_levels, 1))
            ref_points_list.append(ref)

        # Concatenate all levels: (total_tokens, n_levels, 2)
        all_refs = np.concatenate(ref_points_list, axis=0)
        # Add batch dim: (1, total_tokens, n_levels, 2)
        return op.Constant(value=ir.tensor(all_refs[np.newaxis]))

    def _compute_valid_ratios(self, op: builder.OpBuilder) -> ir.Value:
        """Compute valid ratios (all 1.0 since we don't use pixel masks).

        Returns: (1, n_levels, 2) tensor of ones.
        """
        n_levels = len(self._spatial_shapes)
        return op.Constant(value=ir.tensor(np.ones((1, n_levels, 2), dtype=np.float32)))

    def forward(
        self,
        op: builder.OpBuilder,
        pixel_values: ir.Value,
        input_ids: ir.Value,
    ) -> tuple[ir.Value, ir.Value]:
        """Forward pass.

        Args:
            pixel_values: (B, 3, H, W) image tensor
            input_ids: (B, T) text token ids

        Returns:
            logits: (B, num_queries, max_text_len) contrastive class logits
            pred_boxes: (B, num_queries, 4) predicted bounding boxes [x,y,w,h]
        """
        config = self._config
        d = config.d_model
        n_levels = config.num_feature_levels

        # === Text backbone ===
        # Create attention_mask (all ones) and token_type_ids (all zeros)
        attn_mask = op.Expand(
            op.Constant(value_int=1),
            op.Shape(input_ids),
        )
        token_type_ids = op.Expand(
            op.Constant(value_int=0),
            op.Shape(input_ids),
        )
        text_features = self.text_backbone(op, input_ids, attn_mask, token_type_ids)
        text_features = self.text_projection(op, text_features)

        # === Vision backbone ===
        backbone_features = self.backbone.conv_encoder.model(op, pixel_values)

        # === Input projections ===
        feature_maps = []
        for i, feat in enumerate(backbone_features):
            feature_maps.append(self.input_proj_vision[i](op, feat))

        # Extra levels from stride-2 conv on last backbone output
        for i in range(len(backbone_features), n_levels):
            if i == len(backbone_features):
                feature_maps.append(self.input_proj_vision[i](op, backbone_features[-1]))
            else:
                feature_maps.append(self.input_proj_vision[i](op, feature_maps[-1]))

        # === Position embeddings per level (precomputed as constants) ===
        # Position embeddings are static since spatial shapes are known at build time
        position_embeddings = []
        for _level, (h_l, w_l) in enumerate(self._spatial_shapes):
            pos_np = _precompute_sine_pos_embed_2d(
                h_l, w_l, d, config.positional_embedding_temperature
            )
            position_embeddings.append(pos_np)

        # === Flatten multi-scale features ===
        source_flatten_list = []
        pos_flatten_list = []
        batch = op.Shape(feature_maps[0], start=0, end=1)

        for level, feat in enumerate(feature_maps):
            h_l, w_l = self._spatial_shapes[level]
            # feat: (B, D, H, W) → (B, H*W, D)
            feat_flat = op.Transpose(
                op.Reshape(feat, op.Concat(batch, [d, h_l * w_l], axis=0)),
                perm=[0, 2, 1],
            )
            # pos: (1, D, H, W) → (1, H*W, D) — precomputed constant
            pos_np = position_embeddings[level]
            pos_flat_np = pos_np.reshape(1, d, h_l * w_l).transpose(0, 2, 1)
            pos_flat = op.Constant(value=ir.tensor(pos_flat_np))
            # Add level embedding: (1, 1, D)
            level_emb = op.Unsqueeze(
                op.Gather(self.level_embed, op.Constant(value_int=level), axis=0),
                [0, 1],
            )
            pos_flat = op.Add(pos_flat, level_emb)

            source_flatten_list.append(feat_flat)
            pos_flatten_list.append(pos_flat)

        source_flatten = op.Concat(*source_flatten_list, axis=1)
        pos_flatten = op.Concat(*pos_flatten_list, axis=1)

        # === Encoder reference points ===
        encoder_ref_points = self._compute_encoder_reference_points(op)

        # === Encoder ===
        vision_output, text_output = self.encoder(
            op,
            source_flatten,
            pos_flatten,
            text_features,
            encoder_ref_points,
            self._spatial_shapes,
            self._level_start_index,
        )

        # === Two-stage query selection ===
        # Process encoder output for proposals
        enc_processed = self.enc_output_norm(op, self.enc_output(op, vision_output))

        # Contrastive class scores for encoder output
        # enc_class: (B, total_tokens, text_seq) = enc_processed @ text_output.T
        enc_class = op.MatMul(
            enc_processed,
            op.Transpose(text_output, perm=[0, 2, 1]),
        )

        # Bbox proposals from encoder
        enc_bbox_delta = self.encoder_output_bbox_embed(op, enc_processed)

        # Generate output proposals (grid-based)
        proposals = self._generate_proposals(op, batch)
        enc_bbox = op.Add(enc_bbox_delta, proposals)

        # Top-K selection: pick top num_queries proposals by max class score
        topk_logits = op.ReduceMax(enc_class, op.Constant(value_ints=[-1]), keepdims=0)
        # topk_logits: (B, total_tokens)
        _topk_vals, topk_idx = op.TopK(
            topk_logits,
            op.Constant(value_int=config.num_queries),
            axis=-1,
            _outputs=2,
        )  # topk_idx: (B, num_queries)

        # Gather top-K reference points and target embeddings
        topk_coords = op.GatherElements(
            enc_bbox,
            op.Unsqueeze(
                op.Expand(topk_idx, op.Concat(batch, [config.num_queries], axis=0)), [-1]
            ),
            axis=1,
        )
        # Need to repeat topk_idx for the 4 coordinate dims
        topk_idx_4d = op.Expand(
            op.Unsqueeze(topk_idx, [-1]),
            op.Concat(batch, [config.num_queries, 4], axis=0),
        )
        topk_coords = op.GatherElements(enc_bbox, topk_idx_4d, axis=1)
        reference_points = op.Sigmoid(topk_coords)

        # Target: use learned query embeddings (embedding_init_target=True)
        target = op.Expand(
            op.Unsqueeze(self.query_position_embeddings, [0]),
            op.Concat(batch, [config.num_queries, d], axis=0),
        )

        # === Valid ratios ===
        valid_ratios = self._compute_valid_ratios(op)

        # === Decoder ===
        _last_hidden, intermediate_hidden, intermediate_refs = self.decoder(
            op,
            target,
            vision_output,
            text_output,
            reference_points,
            self._spatial_shapes,
            self._level_start_index,
            valid_ratios,
        )

        # === Per-layer predictions ===
        # Use last layer's output for final prediction
        num_layers = config.decoder_layers
        last_layer_hidden = op.Gather(
            intermediate_hidden,
            op.Constant(value_int=num_layers - 1),
            axis=1,
        )  # (B, Q, D)

        # Get last layer's reference for bbox
        last_ref = op.Gather(
            intermediate_refs,
            op.Constant(value_int=num_layers - 1),
            axis=1,
        )  # (B, Q, 4)
        last_ref_logit = op.Log(
            op.Div(
                op.Clip(last_ref, min=1e-5, max=1.0 - 1e-5),
                op.Sub(
                    op.Constant(value_float=1.0),
                    op.Clip(last_ref, min=1e-5, max=1.0 - 1e-5),
                ),
            )
        )

        # Bbox: delta + inverse_sigmoid(reference)
        delta_bbox = self.bbox_embed[num_layers - 1](op, last_layer_hidden)
        pred_boxes = op.Sigmoid(op.Add(delta_bbox, last_ref_logit))

        # Contrastive class logits: hidden @ text.T
        logits = op.MatMul(
            last_layer_hidden,
            op.Transpose(text_output, perm=[0, 2, 1]),
        )  # (B, Q, text_seq)

        return logits, pred_boxes

    def _generate_proposals(self, op: builder.OpBuilder, batch: ir.Value) -> ir.Value:
        """Generate grid-based encoder output proposals.

        Returns: (1, total_tokens, 4) proposals in logit space.
        """
        proposals_list = []
        for level_id, (h_l, w_l) in enumerate(self._spatial_shapes):
            ys = np.linspace(0, h_l - 1, h_l, dtype=np.float32)
            xs = np.linspace(0, w_l - 1, w_l, dtype=np.float32)
            grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
            grid = np.stack([grid_x.flatten(), grid_y.flatten()], axis=-1)  # (h*w, 2)
            # Normalize: (grid + 0.5) / [w, h]
            scale = np.array([w_l, h_l], dtype=np.float32)
            grid = (grid + 0.5) / scale
            # Width/height: 0.05 * 2^level
            wh = np.ones_like(grid) * 0.05 * (2.0**level_id)
            proposal = np.concatenate([grid, wh], axis=-1)  # (h*w, 4)
            proposals_list.append(proposal)

        all_proposals = np.concatenate(proposals_list, axis=0)
        # Inverse sigmoid
        all_proposals = np.clip(all_proposals, 0.01, 0.99)
        all_proposals = np.log(all_proposals / (1 - all_proposals))
        return op.Constant(value=ir.tensor(all_proposals[np.newaxis].astype(np.float32)))

    @staticmethod
    def preprocess_weights(
        state_dict: dict[str, Any],
        config: GroundingDinoConfig,
    ) -> dict[str, Any]:
        """Map HuggingFace weight names to mobius ONNX parameter names.

        Key renames:
        - Strip ``model.`` prefix (ForObjectDetection → inner model)
        - Swin backbone: strip ``backbone.conv_encoder.`` (onnxscript
          skips intermediate plain nn.Module containers), also strip
          ``encoder.`` and ``hidden_states_norms.`` within the backbone
        - ``query_position_embeddings.weight`` → bare parameter
        - BERT text backbone: collapse ``attention.self.`` →
          ``attention.``, etc. (standard BERT renames)
        - Input projections: ``.0.`` → ``.conv.``, ``.1.`` → ``.norm.``
        - Skip ``relative_position_index`` (precomputed constant)
        - Skip ``decoder.bbox_embed.*`` (shared with top-level bbox_embed)
        """
        new_dict: dict[str, Any] = {}

        for key, value in state_dict.items():
            new_key = key

            # Strip "model." prefix from ForObjectDetection
            if new_key.startswith("model."):
                new_key = new_key[6:]

            # Skip decoder.bbox_embed (shared with top-level bbox_embed)
            if new_key.startswith("decoder.bbox_embed."):
                continue

            # Skip relative_position_index (precomputed constant)
            if "relative_position_index" in new_key:
                continue

            # --- Swin backbone renames ---
            # onnxscript skips plain nn.Module() containers, so:
            # backbone.conv_encoder.model.X → model.X
            if new_key.startswith("backbone.conv_encoder."):
                new_key = new_key[len("backbone.conv_encoder.") :]
            # Within Swin: model.encoder.layers.N → model.layers.N
            if new_key.startswith("model.encoder."):
                new_key = "model." + new_key[len("model.encoder.") :]
            # Within Swin: model.hidden_states_norms.stageN → model.stageN
            if "hidden_states_norms." in new_key:
                new_key = new_key.replace("hidden_states_norms.", "")

            # --- query_position_embeddings: Embedding → bare parameter ---
            if new_key == "query_position_embeddings.weight":
                new_key = "query_position_embeddings"

            # --- Input projection renames ---
            # HF: input_proj_vision.{i}.0.{w/b} → .{i}.conv.{w/b}
            # HF: input_proj_vision.{i}.1.{w/b} → .{i}.norm.{w/b}
            if "input_proj_vision" in new_key:
                new_key = new_key.replace(".0.weight", ".conv.weight")
                new_key = new_key.replace(".0.bias", ".conv.bias")
                new_key = new_key.replace(".1.weight", ".norm.weight")
                new_key = new_key.replace(".1.bias", ".norm.bias")

            # --- BERT text backbone renames ---
            if new_key.startswith("text_backbone."):
                new_key = new_key.replace(".attention.self.", ".attention.")
                new_key = new_key.replace(".attention.output.", ".attention.")
                new_key = new_key.replace(".output.dense.", ".dense.")
                new_key = new_key.replace(".output.LayerNorm.", ".LayerNorm.")

            new_dict[new_key] = value

        return new_dict
