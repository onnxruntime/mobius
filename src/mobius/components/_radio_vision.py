# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""C-RADIO ViT vision encoder components."""

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import OpBuilder, nn

from mobius.components._common import INT64_MAX, LayerNorm, Linear

if TYPE_CHECKING:
    import onnx_ir as ir


class _RadioClsToken(nn.Module):
    def __init__(self, num_tokens: int, hidden_size: int):
        super().__init__()
        self.token = nn.Parameter([num_tokens, hidden_size])
        self._num_tokens = num_tokens
        self._hidden_size = hidden_size

    def forward(self, op: OpBuilder, patches: ir.Value) -> ir.Value:
        batch = op.Shape(patches, start=0, end=1)
        shape = op.Concat(
            batch,
            op.Constant(value_ints=[self._num_tokens, self._hidden_size]),
            axis=0,
        )
        tokens = op.Expand(op.Unsqueeze(self.token, [0]), shape)
        return op.Concat(tokens, patches, axis=1)


class _RadioPatchEmbedder(nn.Module):
    """Checkpoint-aligned linear patch projector emitted as an ONNX Conv."""

    def __init__(self, patch_size: int, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter([hidden_size, 3 * patch_size * patch_size])
        self._patch_size = patch_size
        self._hidden_size = hidden_size

    def forward(self, op: OpBuilder, pixel_values: ir.Value) -> ir.Value:
        patch_weight = op.Reshape(
            self.weight,
            [self._hidden_size, 3, self._patch_size, self._patch_size],
        )
        return op.Conv(
            pixel_values,
            patch_weight,
            kernel_shape=[self._patch_size, self._patch_size],
            strides=[self._patch_size, self._patch_size],
        )


class RadioPatchGenerator(nn.Module):
    """Linear patchification plus CPE positional embeddings and register tokens."""

    def __init__(
        self,
        *,
        image_height: int,
        image_width: int,
        patch_size: int,
        max_grid_size: int,
        hidden_size: int,
        num_register_tokens: int = 8,
    ):
        super().__init__()
        if image_height % patch_size or image_width % patch_size:
            raise ValueError("RADIO image dimensions must be divisible by patch_size")
        self.embedder = _RadioPatchEmbedder(patch_size, hidden_size)
        self.pos_embed = nn.Parameter([1, max_grid_size * max_grid_size, hidden_size])
        self.cls_token = _RadioClsToken(num_register_tokens, hidden_size)
        self._image_height = image_height
        self._image_width = image_width
        self._patch_size = patch_size
        self._max_grid_size = max_grid_size
        self._hidden_size = hidden_size

        grid_h = image_height // patch_size
        grid_w = image_width // patch_size
        if max(grid_h, grid_w) != max_grid_size:
            raise ValueError(
                "RADIO export currently requires a canvas whose longest patch-grid "
                "dimension equals max_grid_size"
            )

    def forward(self, op: OpBuilder, pixel_values: ir.Value) -> ir.Value:
        grid_h = self._image_height // self._patch_size
        grid_w = self._image_width // self._patch_size

        # The checkpoint stores a linear patchification matrix; reshaping it
        # inside the embedder gives the equivalent strided Conv2d.
        patches = self.embedder(op, pixel_values)  # (B, hidden, grid_h, grid_w)
        patches = op.Reshape(patches, [0, self._hidden_size, -1])
        patches = op.Transpose(patches, perm=[0, 2, 1])  # (B, grid_h*grid_w, hidden)

        # At the checkpoint's 2048x1664 canvas the CPE algorithm is exactly a
        # top-left crop from the learned 128x128 grid (no interpolation).
        pos = op.Reshape(
            self.pos_embed,
            [1, self._max_grid_size, self._max_grid_size, self._hidden_size],
        )
        pos = op.Slice(pos, [0, 0], [grid_h, grid_w], [1, 2])
        pos = op.Reshape(pos, [1, grid_h * grid_w, self._hidden_size])
        patches = op.Add(patches, pos)

        # Four teacher CLS tokens plus four padding registers are prepended.
        return self.cls_token(op, patches)


class RadioAttention(nn.Module):
    """Fused-QKV bidirectional attention used by timm ViT-Huge."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.qkv = Linear(hidden_size, 3 * hidden_size)
        self.proj = Linear(hidden_size, hidden_size)
        self._num_heads = num_heads
        self._scale = float((hidden_size // num_heads) ** -0.5)

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        qkv = self.qkv(op, hidden_states)
        query, key, value = op.Split(qkv, axis=-1, num_outputs=3, _outputs=3)
        hidden_states = op.Attention(
            query,
            key,
            value,
            q_num_heads=self._num_heads,
            kv_num_heads=self._num_heads,
            scale=self._scale,
        )
        return self.proj(op, hidden_states)


class RadioMLP(nn.Module):
    """Exact-GELU ViT feed-forward network with checkpoint-aligned names."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = Linear(hidden_size, intermediate_size)
        self.fc2 = Linear(intermediate_size, hidden_size)

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        return self.fc2(op, op.Gelu(self.fc1(op, hidden_states)))


class RadioBlock(nn.Module):
    """Pre-norm C-RADIO transformer block."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        norm_eps: float,
    ):
        super().__init__()
        self.norm1 = LayerNorm(hidden_size, eps=norm_eps)
        self.attn = RadioAttention(hidden_size, num_heads)
        self.norm2 = LayerNorm(hidden_size, eps=norm_eps)
        self.mlp = RadioMLP(hidden_size, intermediate_size)

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        hidden_states = op.Add(
            hidden_states,
            self.attn(op, self.norm1(op, hidden_states)),
        )
        return op.Add(
            hidden_states,
            self.mlp(op, self.norm2(op, hidden_states)),
        )


class RadioVisionModel(nn.Module):
    """C-RADIOv2-H ViT backbone returning summary and spatial features."""

    def __init__(
        self,
        *,
        image_height: int,
        image_width: int,
        patch_size: int,
        max_grid_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_layers: int,
        num_heads: int,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.patch_generator = RadioPatchGenerator(
            image_height=image_height,
            image_width=image_width,
            patch_size=patch_size,
            max_grid_size=max_grid_size,
            hidden_size=hidden_size,
        )
        self.blocks = nn.ModuleList(
            [
                RadioBlock(
                    hidden_size,
                    intermediate_size,
                    num_heads,
                    norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, op: OpBuilder, pixel_values: ir.Value) -> tuple[ir.Value, ir.Value]:
        hidden_states = self.patch_generator(op, pixel_values)
        for block in self.blocks:
            hidden_states = block(op, hidden_states)

        # The first four outputs are teacher CLS tokens; summaries [0,1,2]
        # are flattened, while all eight CLS/register tokens are discarded
        # from the spatial feature sequence.
        all_summary = op.Slice(hidden_states, [0], [4], [1])
        summary = op.Gather(all_summary, [0, 1, 2], axis=1)
        summary = op.Reshape(summary, [0, -1])
        features = op.Slice(hidden_states, [8], [INT64_MAX], [1])
        return summary, features
