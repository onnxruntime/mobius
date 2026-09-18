# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Llama4 vision tower with learned positions and split-axis 2D RoPE."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius.components._common import LayerNorm, Linear
from mobius.components._mlp import FCMLP

if TYPE_CHECKING:
    from mobius._configs._sub_configs import VisionConfig


class _Llama4VisionEmbeddings(nn.Module):
    """Patch projection followed by an appended CLS token and learned positions."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        image_size = int(config.image_size or 0)
        patch_size = int(config.patch_size or 0)
        hidden_size = int(config.hidden_size or 0)
        if image_size <= 0 or patch_size <= 0 or image_size % patch_size:
            raise ValueError("Llama4 vision image size must be divisible by its patch size")
        self.patch_embedding = nn.Parameter([hidden_size, 3 * patch_size * patch_size])
        self.class_embedding = nn.Parameter([hidden_size])
        self.position_embedding = nn.Parameter(
            [(image_size // patch_size) ** 2 + 1, hidden_size]
        )
        self._patch_size = patch_size
        self._hidden_size = hidden_size

    def forward(self, op: OpBuilder, pixel_values: ir.Value) -> ir.Value:
        # Preserve the serialized flattened [out, C*P*P] weight while using Conv
        # to patchify NCHW pixels in the same C/H/W order as llama.cpp im2col.
        kernel = op.Reshape(
            self.patch_embedding,
            [self._hidden_size, 3, self._patch_size, self._patch_size],
        )
        patches = op.Conv(
            pixel_values,
            kernel,
            strides=[self._patch_size, self._patch_size],
        )
        batch = op.Shape(patches, start=0, end=1)
        patches = op.Reshape(
            patches,
            op.Concat(batch, [self._hidden_size, -1], axis=0),
        )
        patches = op.Transpose(patches, perm=[0, 2, 1])

        cls = op.Expand(
            op.Reshape(self.class_embedding, [1, 1, self._hidden_size]),
            op.Concat(batch, [1, self._hidden_size], axis=0),
        )
        hidden_states = op.Concat(patches, cls, axis=1)
        return op.Add(hidden_states, op.Unsqueeze(self.position_embedding, [0]))


class _Llama4VisionRoPE2D(nn.Module):
    """Precomputed split-axis RoPE tables for a fixed square patch grid."""

    def __init__(self, head_dim: int, grid_size: int, rope_theta: float):
        super().__init__()
        if head_dim % 4:
            raise ValueError(f"Llama4 vision head_dim must be divisible by 4, got {head_dim}")
        spatial_dim = head_dim // 2
        inv_freq = 1.0 / (
            rope_theta ** (np.arange(0, spatial_dim, 2, dtype=np.float32) / float(spatial_dim))
        )
        positions: np.ndarray = np.arange(grid_size + 1, dtype=np.float32)
        angles = np.outer(positions, inv_freq)
        # llama.cpp rotates adjacent complex pairs within each spatial-axis
        # half, so each frequency is repeated for its even/odd pair.
        angles = np.repeat(angles, 2, axis=-1).astype(np.float32)
        self.cos_cache = nn.Parameter(
            list(angles.shape),
            name="cos_cache",
            data=ir.tensor(np.cos(angles)),
        )
        self.sin_cache = nn.Parameter(
            list(angles.shape),
            name="sin_cache",
            data=ir.tensor(np.sin(angles)),
        )

        patch_h: np.ndarray = np.repeat(
            np.arange(1, grid_size + 1, dtype=np.int64),
            grid_size,
        )
        patch_w: np.ndarray = np.tile(
            np.arange(1, grid_size + 1, dtype=np.int64),
            grid_size,
        )
        self._position_ids = np.stack(
            (
                np.concatenate((patch_h, np.array([0], dtype=np.int64))),
                np.concatenate((patch_w, np.array([0], dtype=np.int64))),
            ),
            axis=-1,
        )

    def forward(
        self, op: OpBuilder
    ) -> tuple[tuple[ir.Value, ir.Value], tuple[ir.Value, ir.Value]]:
        positions = op.Constant(value=ir.tensor(self._position_ids))
        pos_h = op.Gather(positions, op.Constant(value_int=0), axis=1)
        pos_w = op.Gather(positions, op.Constant(value_int=1), axis=1)
        return (
            (
                op.Unsqueeze(op.Gather(self.cos_cache, pos_w, axis=0), [0, 2]),
                op.Unsqueeze(op.Gather(self.sin_cache, pos_w, axis=0), [0, 2]),
            ),
            (
                op.Unsqueeze(op.Gather(self.cos_cache, pos_h, axis=0), [0, 2]),
                op.Unsqueeze(op.Gather(self.sin_cache, pos_h, axis=0), [0, 2]),
            ),
        )


class _Llama4VisionAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("Llama4 vision hidden size must divide by its head count")
        self.q_proj = Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = Linear(hidden_size, hidden_size, bias=True)
        self._num_heads = num_heads
        self._head_dim = hidden_size // num_heads

    @staticmethod
    def _apply_half_rope(
        op: OpBuilder,
        hidden_states: ir.Value,
        position: tuple[ir.Value, ir.Value],
        *,
        start: int,
        end: int,
    ) -> ir.Value:
        half = end - start
        part = op.Slice(hidden_states, [start], [end], [3])
        pairs = op.Reshape(part, [0, 0, 0, half // 2, 2])
        even = op.Slice(pairs, [0], [1], [4])
        odd = op.Slice(pairs, [1], [2], [4])
        rotated = op.Reshape(op.Concat(op.Neg(odd), even, axis=4), [0, 0, 0, half])
        cos, sin = position
        return op.Add(op.Mul(part, cos), op.Mul(rotated, sin))

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        positions: tuple[tuple[ir.Value, ir.Value], tuple[ir.Value, ir.Value]],
    ) -> ir.Value:
        q = op.Reshape(self.q_proj(op, hidden_states), [0, 0, self._num_heads, -1])
        k = op.Reshape(self.k_proj(op, hidden_states), [0, 0, self._num_heads, -1])
        half = self._head_dim // 2
        q = op.Concat(
            self._apply_half_rope(op, q, positions[0], start=0, end=half),
            self._apply_half_rope(
                op,
                q,
                positions[1],
                start=half,
                end=self._head_dim,
            ),
            axis=3,
        )
        k = op.Concat(
            self._apply_half_rope(op, k, positions[0], start=0, end=half),
            self._apply_half_rope(
                op,
                k,
                positions[1],
                start=half,
                end=self._head_dim,
            ),
            axis=3,
        )
        q = op.Reshape(q, [0, 0, self._num_heads * self._head_dim])
        k = op.Reshape(k, [0, 0, self._num_heads * self._head_dim])
        value = self.v_proj(op, hidden_states)
        attended = op.Attention(
            q,
            k,
            value,
            q_num_heads=self._num_heads,
            kv_num_heads=self._num_heads,
            is_causal=0,
        )
        return self.out_proj(op, attended)


class _Llama4VisionEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        eps: float,
    ):
        super().__init__()
        self.ln1 = LayerNorm(hidden_size, eps=eps)
        self.attn = _Llama4VisionAttention(hidden_size, num_heads)
        self.ln2 = LayerNorm(hidden_size, eps=eps)
        self.mlp = FCMLP(hidden_size, intermediate_size, activation="gelu", bias=True)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        positions: tuple[tuple[ir.Value, ir.Value], tuple[ir.Value, ir.Value]],
    ) -> ir.Value:
        hidden_states = op.Add(
            hidden_states,
            self.attn(op, self.ln1(op, hidden_states), positions),
        )
        return op.Add(hidden_states, self.mlp(op, self.ln2(op, hidden_states)))


class Llama4VisionTower(nn.Module):
    """Llama4 fixed-tile vision transformer used by GGUF ``llama4`` sidecars."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        hidden_size = int(config.hidden_size or 0)
        intermediate_size = int(config.intermediate_size or 0)
        num_heads = int(config.num_attention_heads or 0)
        num_layers = int(config.num_hidden_layers or 0)
        image_size = int(config.image_size or 0)
        patch_size = int(config.patch_size or 0)
        if min(hidden_size, intermediate_size, num_heads, num_layers) <= 0:
            raise ValueError("Llama4 vision transformer dimensions must be positive")
        self.embeddings = _Llama4VisionEmbeddings(config)
        self.pre_layernorm = LayerNorm(hidden_size, eps=config.norm_eps)
        self.encoder = nn.ModuleList(
            [
                _Llama4VisionEncoderLayer(
                    hidden_size,
                    intermediate_size,
                    num_heads,
                    config.norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.post_layernorm = LayerNorm(hidden_size, eps=config.norm_eps)
        self.rope = _Llama4VisionRoPE2D(
            hidden_size // num_heads,
            image_size // patch_size,
            config.rope_theta or 10_000.0,
        )
        self._num_patches = (image_size // patch_size) ** 2

    def forward(self, op: OpBuilder, pixel_values: ir.Value) -> ir.Value:
        hidden_states = self.pre_layernorm(op, self.embeddings(op, pixel_values))
        positions = self.rope(op)
        for layer in self.encoder:
            hidden_states = layer(op, hidden_states, positions)
        hidden_states = self.post_layernorm(op, hidden_states)
        # llama.cpp appends CLS and drops the last row after the transformer.
        return op.Slice(hidden_states, [0], [self._num_patches], [1])
