# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Pixtral vision encoder components for Mistral-3 multimodal models.

Implements the Pixtral vision architecture: 2D RoPE, bidirectional
attention, gated MLP, and spatial patch merging. Used by Mistral-3
and Pixtral model families via LLaVAModel dispatch.

HuggingFace reference: PixtralVisionModel, Mistral3MultiModalProjector

Weight name alignment (HF → ONNX):
- ``vision_tower.patch_conv.weight``
- ``vision_tower.ln_pre.weight``
- ``vision_tower.transformer.layers.N.{attention_norm,attention,
  ffn_norm,feed_forward}.*``
- ``multi_modal_projector.{patch_merger,norm,linear_1,linear_2}.*``
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
from onnxscript import nn
from onnxscript._internal import builder

from mobius._build_context import ep_capabilities
from mobius._configs import ArchitectureConfig
from mobius.components._common import Linear
from mobius.components._conv import Conv2dNoBias
from mobius.components._mlp import GatedMLP
from mobius.components._rms_norm import RMSNorm
from mobius.components._rotary_embedding import (
    apply_rotary_pos_emb,
    get_rotary_pos_emb,
)


class PixtralRoPE2D(nn.Module):
    """2D rotary position embeddings for Pixtral vision encoder.

    Precomputes cos/sin caches over a 2D grid, indexed by flattened
    position ID ``h * max_grid_size + w``. Height maps to even
    frequency indices, width maps to odd indices.

    Reference: HF ``PixtralRotaryEmbedding``
    """

    def __init__(
        self,
        head_dim: int,
        max_grid_size: int,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        assert head_dim % 4 == 0, (
            f"Pixtral 2D RoPE requires head_dim divisible by 4, got {head_dim}"
        )
        inv_freq = 1.0 / (
            rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
        )
        # Split into height (even indices) and width (odd indices)
        freq_h = inv_freq[0::2]  # [head_dim/4]
        freq_w = inv_freq[1::2]  # [head_dim/4]

        # Precompute per-position frequencies
        positions = np.arange(max_grid_size, dtype=np.float32)
        freqs_h = np.outer(positions, freq_h)  # [max_grid, head_dim/4]
        freqs_w = np.outer(positions, freq_w)  # [max_grid, head_dim/4]

        # Build cache: for (h, w), freqs = concat(h*freq_h, w*freq_w)
        total_positions = max_grid_size * max_grid_size
        cos_cache = np.zeros((total_positions, head_dim // 2), dtype=np.float32)
        sin_cache = np.zeros_like(cos_cache)

        # Row-major: idx = h * max_grid_size + w, matching forward() meshgrid
        for h in range(max_grid_size):
            for w in range(max_grid_size):
                freqs = np.concatenate([freqs_h[h], freqs_w[w]])
                idx = h * max_grid_size + w
                cos_cache[idx] = np.cos(freqs)
                sin_cache[idx] = np.sin(freqs)

        self.cos_cache = nn.Parameter(
            list(cos_cache.shape),
            name="cos_cache",
            data=ir.tensor(cos_cache),
        )
        self.sin_cache = nn.Parameter(
            list(sin_cache.shape),
            name="sin_cache",
            data=ir.tensor(sin_cache),
        )

    def forward(
        self,
        op: builder.OpBuilder,
        position_ids: ir.Value,
    ) -> tuple[ir.Value, ir.Value]:
        """Look up 2D RoPE cos/sin by flattened grid position IDs.

        Args:
            position_ids: [batch, seq_len] flattened 2D grid indices.

        Returns:
            (cos_emb, sin_emb) each [batch, seq_len, head_dim/2].
        """
        return get_rotary_pos_emb(op, position_ids, self.cos_cache, self.sin_cache)


class PixtralAttention(nn.Module):
    """Bidirectional multi-head attention with 2D RoPE.

    No causal mask, no KV cache — vision encoder only.
    HF weight names: ``attention.{q,k,v,o}_proj.weight``
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self._num_heads = num_heads
        self._head_dim = head_dim
        self.q_proj = Linear(
            hidden_size,
            num_heads * head_dim,
            bias=False,
        )
        self.k_proj = Linear(
            hidden_size,
            num_heads * head_dim,
            bias=False,
        )
        self.v_proj = Linear(
            hidden_size,
            num_heads * head_dim,
            bias=False,
        )
        self.o_proj = Linear(
            num_heads * head_dim,
            hidden_size,
            bias=False,
        )

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        position_embeddings: tuple[ir.Value, ir.Value],
    ) -> ir.Value:
        """Bidirectional self-attention with 2D RoPE.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            position_embeddings: (cos, sin) from PixtralRoPE2D.
        """
        q = self.q_proj(op, hidden_states)
        k = self.k_proj(op, hidden_states)
        v = self.v_proj(op, hidden_states)

        # Apply 2D RoPE to Q and K
        q = apply_rotary_pos_emb(
            op,
            q,
            position_embeddings,
            num_heads=self._num_heads,
        )
        k = apply_rotary_pos_emb(
            op,
            k,
            position_embeddings,
            num_heads=self._num_heads,
        )

        # Bidirectional attention (no causal mask, no KV cache).
        # Use com.microsoft.MultiHeadAttention for all EPs that support
        # custom-domain ops (it has fused kernels on CUDA/DML and runs
        # correctly on CPU). Fall back to standard opset-24 Attention
        # for onnx-standard EP which prohibits custom-domain ops.
        scale = float(1.0 / (self._head_dim**0.5))
        if ep_capabilities().name == "onnx-standard":
            attn_output = op.Attention(
                q,
                k,
                v,
                q_num_heads=self._num_heads,
                kv_num_heads=self._num_heads,
                scale=scale,
                is_causal=0,
            )
        else:
            attn_output = op.MultiHeadAttention(
                q,
                k,
                v,
                num_heads=self._num_heads,
                scale=scale,
                _domain="com.microsoft",
            )
        return self.o_proj(op, attn_output)


class PixtralTransformerLayer(nn.Module):
    """Single Pixtral transformer layer: pre-norm attn + MLP.

    ``attention_norm → attention → residual → ffn_norm → mlp → residual``
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.attention_norm = RMSNorm(hidden_size, eps)
        self.attention = PixtralAttention(
            hidden_size,
            num_heads,
            head_dim,
        )
        self.ffn_norm = RMSNorm(hidden_size, eps)
        # SiLU gated MLP (no bias, matches HF PixtralAttentionMLP)
        self.feed_forward = GatedMLP(
            hidden_size, intermediate_size, activation="silu", bias=False
        )

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        position_embeddings: tuple[ir.Value, ir.Value],
    ) -> ir.Value:
        residual = hidden_states
        hidden_states = self.attention_norm(op, hidden_states)
        hidden_states = self.attention(
            op,
            hidden_states,
            position_embeddings,
        )
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.ffn_norm(op, hidden_states)
        hidden_states = self.feed_forward(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)
        return hidden_states


class PixtralTransformerEncoder(nn.Module):
    """Stack of Pixtral transformer layers.

    Processes [batch, seq_len, hidden_size] through N identical
    pre-norm attention + MLP layers with 2D RoPE position embeddings.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                PixtralTransformerLayer(
                    hidden_size,
                    intermediate_size,
                    num_heads,
                    head_dim,
                    eps,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        position_embeddings: tuple[ir.Value, ir.Value],
    ) -> ir.Value:
        for layer in self.layers:
            hidden_states = layer(
                op,
                hidden_states,
                position_embeddings,
            )
        return hidden_states


# ── Mistral-3 patch merging and projection ────────────────────


class Mistral3PatchMerger(nn.Module):
    """Merge spatial_merge_size x spatial_merge_size adjacent patches.

    Reshapes [batch, H*W, D] → groups spatially adjacent patches →
    [batch, H/ms * W/ms, ms^2*D] → linear projection.

    HF weight: ``patch_merger.merging_layer.weight``
    """

    def __init__(
        self,
        hidden_size: int,
        spatial_merge_size: int = 2,
    ):
        super().__init__()
        self._merge_size = spatial_merge_size
        merged_dim = hidden_size * spatial_merge_size * spatial_merge_size
        # Reduces merged patches back to vision_hidden_size
        self.merging_layer = Linear(
            merged_dim,
            hidden_size,
            bias=False,
        )

    def forward(
        self,
        op: builder.OpBuilder,
        hidden_states: ir.Value,
        grid_h: ir.Value,
        grid_w: ir.Value,
    ) -> ir.Value:
        """Spatial merge of adjacent patches.

        Args:
            hidden_states: [batch, H*W, D]
            grid_h: scalar — patch grid height.
            grid_w: scalar — patch grid width.

        Returns:
            [batch, (H/ms)*(W/ms), hidden_size] after merge + linear.
        """
        ms = self._merge_size
        batch = op.Shape(hidden_states, start=0, end=1)
        d = op.Shape(hidden_states, start=2, end=3)

        ms_scalar = op.Constant(value_int=ms)
        h_m = op.Div(grid_h, ms_scalar)
        w_m = op.Div(grid_w, ms_scalar)

        # [batch, H*W, D] → [batch, H/ms, ms, W/ms, ms, D]
        ms_1d = op.Constant(value_ints=[ms])
        h_m_1d = op.Reshape(h_m, op.Constant(value_ints=[1]))
        w_m_1d = op.Reshape(w_m, op.Constant(value_ints=[1]))
        shape_6d = op.Concat(
            batch,
            h_m_1d,
            ms_1d,
            w_m_1d,
            ms_1d,
            d,
            axis=0,
        )
        x = op.Reshape(hidden_states, shape_6d)

        # Transpose to match HuggingFace F.unfold ordering (dim-major).
        # F.unfold groups elements as [D, ms_h, ms_w] per spatial position,
        # meaning the hidden dim D is the outermost loop. To reproduce
        # this with reshape + transpose:
        #   from: [batch, H/ms, ms, W/ms, ms, D]
        #     to: [batch, H/ms, W/ms, D, ms, ms]
        x = op.Transpose(x, perm=[0, 1, 3, 5, 2, 4])

        # Flatten: [batch, (H/ms)*(W/ms), D*ms*ms]
        merged_count = op.Mul(h_m_1d, w_m_1d)
        shape_3d = op.Concat(
            batch,
            merged_count,
            op.Constant(value_ints=[-1]),
            axis=0,
        )
        x = op.Reshape(x, shape_3d)

        return self.merging_layer(op, x)


class Mistral3MultiModalProjector(nn.Module):
    """Mistral-3 multimodal projector: norm → merge → MLP.

    ``norm → patch_merger → linear_1 → GELU → linear_2``

    HF weights:
    - ``multi_modal_projector.patch_merger.merging_layer.weight``
    - ``multi_modal_projector.norm.weight``
    - ``multi_modal_projector.linear_1.weight``
    - ``multi_modal_projector.linear_2.weight``
    """

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        spatial_merge_size: int = 2,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        # HF applies norm BEFORE spatial merging
        self.norm = RMSNorm(vision_hidden_size, eps=norm_eps)
        self.patch_merger = Mistral3PatchMerger(
            vision_hidden_size,
            spatial_merge_size,
        )
        # After merging, dims are back to vision_hidden_size
        self.linear_1 = Linear(
            vision_hidden_size,
            text_hidden_size,
            bias=False,
        )
        self.linear_2 = Linear(
            text_hidden_size,
            text_hidden_size,
            bias=False,
        )

    def forward(
        self,
        op: builder.OpBuilder,
        vision_features: ir.Value,
        grid_h: ir.Value,
        grid_w: ir.Value,
    ) -> ir.Value:
        """Project vision features to text embedding space.

        Args:
            vision_features: [batch, num_patches, vision_hidden]
            grid_h: scalar — patch grid height.
            grid_w: scalar — patch grid width.

        Returns:
            [batch, num_merged_patches, text_hidden_size]
        """
        # Step 1: Norm before spatial merge (HF order)
        normed = self.norm(op, vision_features)
        # Step 2: Spatial merge
        merged = self.patch_merger(
            op,
            normed,
            grid_h,
            grid_w,
        )
        # Step 3: MLP projection
        hidden = op.Gelu(self.linear_1(op, merged))
        return self.linear_2(op, hidden)


class PixtralVisionTower(nn.Module):
    """Pixtral vision tower: patch_conv → ln_pre → transformer.

    Matches the HF ``vision_tower.*`` weight namespace.  Does NOT
    include the multi-modal projector — that is a separate module
    in the LLaVA-style split.

    Forward returns ``(hidden_states, grid_h, grid_w)`` so the
    projector can perform spatial merging.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        vc = config.vision
        assert vc is not None, "VisionConfig is required"
        assert vc.hidden_size is not None
        assert vc.intermediate_size is not None
        assert vc.num_hidden_layers is not None
        assert vc.num_attention_heads is not None
        assert vc.image_size is not None
        assert vc.patch_size is not None

        head_dim = vc.head_dim or vc.hidden_size // vc.num_attention_heads
        rope_theta = vc.rope_theta or 10000.0
        max_grid_size = vc.image_size // vc.patch_size

        self.patch_conv = Conv2dNoBias(
            in_channels=vc.in_channels,
            out_channels=vc.hidden_size,
            kernel_size=vc.patch_size,
            stride=vc.patch_size,
        )
        # HuggingFace PixtralVisionModel hardcodes eps=1e-5 (no config field)
        self.ln_pre = RMSNorm(vc.hidden_size, eps=1e-5)
        self.transformer = PixtralTransformerEncoder(
            num_layers=vc.num_hidden_layers,
            hidden_size=vc.hidden_size,
            intermediate_size=vc.intermediate_size,
            num_heads=vc.num_attention_heads,
            head_dim=head_dim,
            eps=1e-5,
        )
        self.rope = PixtralRoPE2D(
            head_dim=head_dim,
            max_grid_size=max_grid_size,
            rope_theta=rope_theta,
        )
        self._max_grid_size = max_grid_size

    def forward(
        self,
        op: builder.OpBuilder,
        pixel_values: ir.Value,
    ) -> tuple[ir.Value, ir.Value, ir.Value]:
        """Encode pixel values to patch features.

        Args:
            pixel_values: [batch, channels, height, width]

        Returns:
            (hidden_states, grid_h, grid_w) where
            hidden_states is [batch, num_patches, hidden_size].
        """
        # Patch embedding: [batch, hidden, grid_h, grid_w]
        patches_4d = self.patch_conv(op, pixel_values)

        # Extract grid dims for position IDs and merging
        grid_h = op.Squeeze(
            op.Shape(patches_4d, start=2, end=3),
        )
        grid_w = op.Squeeze(
            op.Shape(patches_4d, start=3, end=4),
        )

        # Reshape to [batch, num_patches, hidden_size]
        batch_size = op.Shape(patches_4d, start=0, end=1)
        hidden_dim = op.Shape(patches_4d, start=1, end=2)
        new_shape = op.Concat(
            batch_size,
            hidden_dim,
            op.Constant(value_ints=[-1]),
            axis=0,
        )
        patches = op.Reshape(patches_4d, new_shape)
        patches = op.Transpose(patches, perm=[0, 2, 1])

        # Pre-norm
        hidden_states = self.ln_pre(op, patches)

        # 2D meshgrid position IDs for RoPE cache lookup
        zero = op.Constant(value_int=0)
        one = op.Constant(value_int=1)
        h_ids = op.Range(zero, grid_h, one)
        w_ids = op.Range(zero, grid_w, one)
        max_grid = op.Constant(value_int=self._max_grid_size)
        h_scaled = op.Mul(h_ids, max_grid)
        # Meshgrid: pos[i,j] = h_scaled[i] + w_ids[j]
        h_2d = op.Unsqueeze(h_scaled, [1])
        w_2d = op.Unsqueeze(w_ids, [0])
        pos_grid = op.Add(h_2d, w_2d)
        # Reshape to [1, num_patches] — batch=1 is correct here because
        # the vision encoder processes one image at a time; cos/sin from
        # RoPE broadcast across the num_heads dimension, not batch.
        position_ids = op.Reshape(
            pos_grid,
            op.Constant(value_ints=[1, -1]),
        )

        # RoPE + transformer
        pos_emb = self.rope(op, position_ids)
        hidden_states = self.transformer(
            op,
            hidden_states,
            pos_emb,
        )

        return hidden_states, grid_h, grid_w
