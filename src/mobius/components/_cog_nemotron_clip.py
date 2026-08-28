# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Standalone CogVLM and Nemotron V2 VL clip-sidecar vision graphs.

The implementations follow llama.cpp commit
``8d9af256337d1a501250f9bbf4c0859a654bddd6``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import OpBuilder, nn

from mobius.components._common import LayerNorm, Linear
from mobius.components._rms_norm import apply_rms_norm

if TYPE_CHECKING:
    import onnx_ir as ir


class _NamedLinear(Linear):
    def __init__(self, in_features: int, out_features: int, stem: str):
        super().__init__(in_features, out_features)
        self.weight = nn.Parameter([out_features, in_features], name=f"{stem}.weight")
        self.bias = nn.Parameter([out_features], name=f"{stem}.bias")


class _NamedLinearNoBias(Linear):
    def __init__(self, in_features: int, out_features: int, stem: str):
        super().__init__(in_features, out_features, bias=False)
        self.weight = nn.Parameter([out_features, in_features], name=f"{stem}.weight")


class _NamedLayerNorm(LayerNorm):
    def __init__(self, hidden_size: int, stem: str, eps: float):
        super().__init__(hidden_size, eps)
        self.weight = nn.Parameter([hidden_size], name=f"{stem}.weight")
        self.bias = nn.Parameter([hidden_size], name=f"{stem}.bias")


class _PatchEmbeddingBase(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, grid_size: int):
        super().__init__()
        self.weight = nn.Parameter(
            [hidden_size, 3, patch_size, patch_size],
            name="v.patch_embd.weight",
        )
        self._patch_size = patch_size
        self._hidden_size = hidden_size
        self._num_patches = grid_size * grid_size

    def forward(self, op: OpBuilder, pixel_values: ir.Value) -> ir.Value:
        patches = self._conv(op, pixel_values)
        shape = op.Concat(
            op.Shape(patches, start=0, end=1),
            op.Constant(value_ints=[self._hidden_size, self._num_patches]),
            axis=0,
        )
        patches = op.Reshape(patches, shape)
        return op.Transpose(patches, perm=[0, 2, 1])  # (B, patches, hidden)

    def _conv(self, op: OpBuilder, pixel_values: ir.Value) -> ir.Value:
        raise NotImplementedError


class _ClipPatchEmbedding(_PatchEmbeddingBase):
    def __init__(self, hidden_size: int, patch_size: int, grid_size: int):
        super().__init__(hidden_size, patch_size, grid_size)
        self.bias = nn.Parameter([hidden_size], name="v.patch_embd.bias")

    def _conv(self, op: OpBuilder, pixel_values: ir.Value) -> ir.Value:
        return op.Conv(
            pixel_values,
            self.weight,
            self.bias,
            kernel_shape=[self._patch_size, self._patch_size],
            strides=[self._patch_size, self._patch_size],
        )


class _RadioPatchEmbedding(_PatchEmbeddingBase):
    def _conv(self, op: OpBuilder, pixel_values: ir.Value) -> ir.Value:
        return op.Conv(
            pixel_values,
            self.weight,
            kernel_shape=[self._patch_size, self._patch_size],
            strides=[self._patch_size, self._patch_size],
        )


class _FusedQKVAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, stem: str):
        super().__init__()
        self.qkv = _NamedLinear(hidden_size, 3 * hidden_size, f"{stem}.attn_qkv")
        self.proj = _NamedLinear(hidden_size, hidden_size, f"{stem}.attn_out")
        self._num_heads = num_heads
        self._head_size = hidden_size // num_heads
        self._hidden_size = hidden_size
        self._scale = float(self._head_size**-0.5)

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        # llama.cpp stores Q, K, and V as three contiguous hidden-sized rows.
        qkv = self.qkv(op, hidden_states)
        query, key, value = op.Split(qkv, axis=-1, num_outputs=3, _outputs=3)
        batch = op.Shape(hidden_states, start=0, end=1)
        sequence = op.Shape(hidden_states, start=1, end=2)
        head_shape = op.Concat(
            batch,
            sequence,
            op.Constant(value_ints=[self._num_heads, self._head_size]),
            axis=0,
        )
        query = op.Transpose(op.Reshape(query, head_shape), perm=[0, 2, 1, 3])
        key = op.Transpose(op.Reshape(key, head_shape), perm=[0, 2, 1, 3])
        value = op.Transpose(op.Reshape(value, head_shape), perm=[0, 2, 1, 3])
        # Decomposed bidirectional attention preserves the pinned clip graph's
        # exact QKV split while remaining executable on ORT builds predating
        # the opset-23 Attention kernel.
        scores = op.Mul(
            op.MatMul(query, op.Transpose(key, perm=[0, 1, 3, 2])),
            self._scale,
        )
        attended = op.MatMul(op.Softmax(scores, axis=-1), value)
        attended = op.Transpose(attended, perm=[0, 2, 1, 3])
        attended = op.Reshape(
            attended,
            op.Concat(
                batch,
                sequence,
                op.Constant(value_ints=[self._hidden_size]),
                axis=0,
            ),
        )
        return self.proj(op, attended)


class _CogVLMFeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, stem: str):
        super().__init__()
        self.up = _NamedLinear(hidden_size, intermediate_size, f"{stem}.ffn_up")
        self.gate = _NamedLinear(hidden_size, intermediate_size, f"{stem}.ffn_gate")
        self.down = _NamedLinear(intermediate_size, hidden_size, f"{stem}.ffn_down")

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        gate = self.gate(op, hidden_states)
        return self.down(
            op,
            op.Mul(self.up(op, hidden_states), op.Mul(gate, op.Sigmoid(gate))),
        )


class CogVLMClipBlock(nn.Module):
    """CogVLM's post-attention/post-FFN normal-norm transformer block."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        layer_index: int,
        norm_eps: float,
    ):
        super().__init__()
        stem = f"v.blk.{layer_index}"
        self.attention = _FusedQKVAttention(hidden_size, num_heads, stem)
        self.input_layernorm = _NamedLayerNorm(hidden_size, f"{stem}.ln1", norm_eps)
        self.mlp = _CogVLMFeedForward(hidden_size, intermediate_size, stem)
        self.post_attention_layernorm = _NamedLayerNorm(hidden_size, f"{stem}.ln2", norm_eps)

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        # CogVLM is intentionally post-norm, unlike the generic CLIP ViT path.
        attended = self.input_layernorm(op, self.attention(op, hidden_states))
        hidden_states = op.Add(hidden_states, attended)
        fed_forward = self.post_attention_layernorm(op, self.mlp(op, hidden_states))
        return op.Add(hidden_states, fed_forward)


class CogVLMClipSidecar(nn.Module):
    """CogVLM CLIP tower and gated image projector.

    Input is ``pixel_values`` in NCHW layout. Output is the projected patch
    sequence bracketed by the learned BOI and EOI rows.
    """

    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_layers: int,
        num_heads: int,
        projector_hidden_size: int,
        projector_intermediate_size: int,
        output_size: int,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        if image_size % patch_size:
            raise ValueError("CogVLM image_size must be divisible by patch_size")
        num_patches = (image_size // patch_size) ** 2
        self.patch_embedding = _ClipPatchEmbedding(
            hidden_size, patch_size, image_size // patch_size
        )
        self.class_embedding = nn.Parameter([1, hidden_size], name="v.class_embd")
        self.position_embedding = nn.Parameter(
            [num_patches + 1, hidden_size], name="v.position_embd.weight"
        )
        self.blocks = nn.ModuleList(
            [
                CogVLMClipBlock(
                    hidden_size,
                    intermediate_size,
                    num_heads,
                    layer_index,
                    norm_eps,
                )
                for layer_index in range(num_layers)
            ]
        )
        self.projection = _NamedLinearNoBias(hidden_size, projector_hidden_size, "mm.model.fc")
        self.projector_norm = _NamedLayerNorm(projector_hidden_size, "mm.post_fc_norm", 1e-5)
        self.projector_up = _NamedLinearNoBias(
            projector_hidden_size, projector_intermediate_size, "mm.up"
        )
        self.projector_gate = _NamedLinearNoBias(
            projector_hidden_size, projector_intermediate_size, "mm.gate"
        )
        self.projector_down = _NamedLinearNoBias(
            projector_intermediate_size,
            output_size,
            "mm.down",
        )
        self.boi = nn.Parameter([1, 1, output_size], name="v.boi")
        self.eoi = nn.Parameter([1, 1, output_size], name="v.eoi")
        self._num_patches = num_patches

    def forward(self, op: OpBuilder, pixel_values: ir.Value) -> ir.Value:
        patches = self.patch_embedding(op, pixel_values)
        batch = op.Shape(patches, start=0, end=1)
        cls = op.Expand(
            op.Unsqueeze(self.class_embedding, [0]),
            op.Concat(batch, op.Shape(self.class_embedding), axis=0),
        )
        hidden_states = op.Add(
            op.Concat(patches, cls, axis=1),
            op.Unsqueeze(self.position_embedding, [0]),
        )
        for block in self.blocks:
            hidden_states = block(op, hidden_states)

        # The CLS row participates in the ViT but not in the projected image sequence.
        hidden_states = op.Gather(hidden_states, list(range(self._num_patches)), axis=1)
        hidden_states = self.projection(op, hidden_states)
        hidden_states = self.projector_norm(op, hidden_states)
        hidden_states = op.Gelu(hidden_states, approximate="tanh")
        gate = self.projector_gate(op, hidden_states)
        hidden_states = op.Mul(
            self.projector_up(op, hidden_states),
            op.Mul(gate, op.Sigmoid(gate)),
        )
        hidden_states = self.projector_down(op, hidden_states)

        boi = op.Expand(self.boi, op.Concat(batch, op.Shape(self.boi, start=1), axis=0))
        eoi = op.Expand(self.eoi, op.Concat(batch, op.Shape(self.eoi, start=1), axis=0))
        return op.Concat(boi, hidden_states, eoi, axis=1)


class _GeluFeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, stem: str):
        super().__init__()
        self.fc1 = _NamedLinear(hidden_size, intermediate_size, f"{stem}.ffn_up")
        self.fc2 = _NamedLinear(intermediate_size, hidden_size, f"{stem}.ffn_down")

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        return self.fc2(op, op.Gelu(self.fc1(op, hidden_states)))


class NemotronV2VLClipBlock(nn.Module):
    """Pre-normalized, fused-QKV RADIO ViT block."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        layer_index: int,
        norm_eps: float,
    ):
        super().__init__()
        stem = f"v.blk.{layer_index}"
        self.norm1 = _NamedLayerNorm(hidden_size, f"{stem}.ln1", norm_eps)
        self.attention = _FusedQKVAttention(hidden_size, num_heads, stem)
        self.norm2 = _NamedLayerNorm(hidden_size, f"{stem}.ln2", norm_eps)
        self.mlp = _GeluFeedForward(hidden_size, intermediate_size, stem)

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        hidden_states = op.Add(
            hidden_states, self.attention(op, self.norm1(op, hidden_states))
        )
        return op.Add(hidden_states, self.mlp(op, self.norm2(op, hidden_states)))


class NemotronV2VLClipSidecar(nn.Module):
    """Fixed-resolution RADIO vision sidecar from Nemotron Nano V2 VL.

    Nemotron's GGUF sidecar can also contain Parakeet audio tensors. This
    vision-only component deliberately neither models nor drops them; callers
    rewriting a combined sidecar must preserve the companion audio tensors.
    """

    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_layers: int,
        num_heads: int,
        num_register_tokens: int,
        projector_hidden_size: int,
        output_size: int,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        if image_size % patch_size:
            raise ValueError("Nemotron image_size must be divisible by patch_size")
        grid_size = image_size // patch_size
        if grid_size % 2:
            raise ValueError("Nemotron patch grid must be divisible by the 2x2 merge")
        self.patch_embedding = _RadioPatchEmbedding(hidden_size, patch_size, grid_size)
        self.class_embedding = nn.Parameter(
            [num_register_tokens, hidden_size], name="v.class_embd"
        )
        self.position_embedding = nn.Parameter(
            [1, grid_size * grid_size, hidden_size], name="v.position_embd.weight"
        )
        self.blocks = nn.ModuleList(
            [
                NemotronV2VLClipBlock(
                    hidden_size,
                    intermediate_size,
                    num_heads,
                    layer_index,
                    norm_eps,
                )
                for layer_index in range(num_layers)
            ]
        )
        merged_size = hidden_size * 4
        self.projector_norm_weight = nn.Parameter(
            [merged_size],
            name="mm.model.mlp.0.weight",
        )
        self.projector_up_weight = nn.Parameter(
            [projector_hidden_size, merged_size],
            name="mm.model.mlp.1.weight",
        )
        self.projector_down_weight = nn.Parameter(
            [output_size, projector_hidden_size],
            name="mm.model.mlp.3.weight",
        )
        self._grid_size = grid_size
        self._hidden_size = hidden_size
        self._merge_size = 2

    def forward(self, op: OpBuilder, pixel_values: ir.Value) -> ir.Value:
        patches = op.Add(self.patch_embedding(op, pixel_values), self.position_embedding)
        batch = op.Shape(patches, start=0, end=1)
        register_shape = op.Concat(batch, op.Shape(self.class_embedding), axis=0)
        hidden_states = op.Concat(
            op.Expand(op.Unsqueeze(self.class_embedding, [0]), register_shape),
            patches,
            axis=1,
        )
        for block in self.blocks:
            hidden_states = block(op, hidden_states)

        # Derive the register count from the sidecar tensor instead of baking in eight.
        register_count = op.Shape(self.class_embedding, start=0, end=1)
        sequence_length = op.Shape(hidden_states, start=1, end=2)
        patch_indices = op.Range(
            op.Squeeze(register_count, [0]),
            op.Squeeze(sequence_length, [0]),
            op.Constant(value_int=1),
        )
        patches = op.Gather(hidden_states, patch_indices, axis=1)

        # NHWC pixel-unshuffle order: each output row contains its m x m patch block.
        m = self._merge_size
        g = self._grid_size
        merged_grid = g // m
        merge_shape = op.Concat(
            batch,
            op.Constant(value_ints=[merged_grid, m, merged_grid, m, self._hidden_size]),
            axis=0,
        )
        patches = op.Reshape(patches, merge_shape)
        patches = op.Transpose(patches, perm=[0, 1, 3, 2, 4, 5])
        output_shape = op.Concat(
            batch,
            op.Constant(
                value_ints=[
                    merged_grid * merged_grid,
                    self._hidden_size * m * m,
                ]
            ),
            axis=0,
        )
        patches = op.Reshape(patches, output_shape)

        patches = apply_rms_norm(op, patches, self.projector_norm_weight, 1e-6)
        patches = op.Relu(
            op.MatMul(patches, op.Transpose(self.projector_up_weight, perm=[1, 0]))
        )
        patches = op.Mul(patches, patches)
        return op.MatMul(patches, op.Transpose(self.projector_down_weight, perm=[1, 0]))
