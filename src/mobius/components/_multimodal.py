# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Multimodal components for bridging vision and text models.

Provides modules for connecting vision encoder outputs to text model inputs:

Projectors (vision → text embedding space):
- ``Gemma3MultiModalProjector``: AvgPool2d → RMSNorm → MatMul (Gemma3)
- ``MLPMultiModalProjector``: Linear → Act → Linear (LLaVA, Phi4MM)
- ``LinearMultiModalProjector``: Single Linear (PaliGemma)
- ``GGUFMLPProjector``: llama.cpp's one/two-layer LLaVA MLP
- ``MobileLDPProjector`` / ``MobileLDPV2Projector``: MobileVLM token downsamplers
- ``GLMEdgeAdapterProjector``: GLM-Edge spatial adapter with BOI/EOI tokens
- ``MiniCPMResamplerProjector``: MiniCPM-V learned-query cross-attention resampler

Mixer:
- ``InputMixer``: Merges vision embeddings into text embeddings at
  placeholder token positions
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import OpBuilder, nn

from mobius.components._common import LayerNorm, Linear
from mobius.components._conv import Conv2d, Conv2dNoBias
from mobius.components._rms_norm import RMSNorm
from mobius.components._vision import VisionLayerNorm

if TYPE_CHECKING:
    import onnx_ir as ir


class Gemma3MultiModalProjector(nn.Module):
    """AvgPool2d → RMSNorm → MatMul projector (Gemma3).

    Reshapes vision features into a 2-D spatial grid, applies 2-D average
    pooling to reduce the number of tokens, normalises with RMSNorm, then
    projects via a learnable matrix multiplication.

    HF reference: ``Gemma3MultiModalProjector`` in
    ``transformers.models.gemma3.modeling_gemma3``.
    """

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        patches_per_image: int,
        tokens_per_image: int,
        norm: RMSNorm | None = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.mm_soft_emb_norm = norm or RMSNorm(vision_hidden_size, eps=eps)
        self.patches_per_image = patches_per_image
        tokens_per_side = int(tokens_per_image**0.5)
        self.pool_kernel = patches_per_image // tokens_per_side
        self.mm_input_projection_weight = nn.Parameter([vision_hidden_size, text_hidden_size])

    def forward(self, op: OpBuilder, vision_features: ir.Value):
        # vision_features: [batch, num_patches, vision_hidden_size]
        batch_size = op.Shape(vision_features, start=0, end=1)
        hidden_size = op.Shape(vision_features, start=2, end=3)
        patches = op.Constant(value_ints=[self.patches_per_image])

        # Transpose to [batch, hidden, num_patches]
        hidden = op.Transpose(vision_features, perm=[0, 2, 1])
        # Reshape to [batch, hidden, patches_per_image, patches_per_image]
        new_shape = op.Concat(batch_size, hidden_size, patches, patches, axis=0)
        hidden = op.Reshape(hidden, new_shape)

        # 2D average pooling
        hidden = op.AveragePool(
            hidden,
            kernel_shape=[self.pool_kernel, self.pool_kernel],
            strides=[self.pool_kernel, self.pool_kernel],
        )

        # Flatten spatial dims: [batch, hidden, h, w] → [batch, hidden, h*w]
        minus_one = op.Constant(value_ints=[-1])
        flat_shape = op.Concat(batch_size, hidden_size, minus_one, axis=0)
        hidden = op.Reshape(hidden, flat_shape)
        # Transpose to [batch, tokens, hidden]
        hidden = op.Transpose(hidden, perm=[0, 2, 1])

        # RMSNorm after pooling
        hidden = self.mm_soft_emb_norm(op, hidden)

        # Linear projection: vision_hidden → text_hidden
        projected = op.MatMul(hidden, self.mm_input_projection_weight)
        return projected


class MLPMultiModalProjector(nn.Module):
    """Two-layer MLP projector: Linear → GELU → Linear.

    The most common projector pattern, used by LLaVA, LLaVA-NeXT, VipLLaVA,
    Phi-4-multimodal and others.

    HF reference: ``LlavaMultiModalProjector`` in
    ``transformers.models.llava.modeling_llava``.
    """

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        bias: bool = True,
    ):
        super().__init__()
        self.linear_1 = Linear(vision_hidden_size, text_hidden_size, bias=bias)
        self.linear_2 = Linear(text_hidden_size, text_hidden_size, bias=bias)

    def forward(self, op: OpBuilder, vision_features: ir.Value):
        # vision_features: [batch, num_patches, vision_hidden_size]
        hidden = self.linear_1(op, vision_features)
        hidden = op.Gelu(hidden)
        hidden = self.linear_2(op, hidden)
        return hidden


class GGUFMLPProjector(nn.Module):
    """llama.cpp LLaVA MLP projector with an optional second affine layer.

    The serialized ``projector:mlp`` closure is ``mm.0`` followed by GELU and,
    when present, ``mm.2``. The distinct Yi ``MLP_NORM`` compatibility topology
    is intentionally not represented by this component.
    """

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        *,
        has_second_layer: bool = True,
    ):
        super().__init__()
        if vision_hidden_size <= 0 or text_hidden_size <= 0:
            raise ValueError("MLP projector dimensions must be positive")
        self.linear_0 = Linear(vision_hidden_size, text_hidden_size, bias=True)
        self.linear_2 = (
            Linear(text_hidden_size, text_hidden_size, bias=True) if has_second_layer else None
        )

    def forward(self, op: OpBuilder, vision_features: ir.Value):
        # [B, patches, vision_hidden] -> [B, patches, text_hidden]
        hidden = op.Gelu(self.linear_0(op, vision_features), approximate="tanh")
        if self.linear_2 is not None:
            hidden = self.linear_2(op, hidden)
        return hidden


class _MobileLDPBlock(nn.Module):
    """MobileVLM inverted residual block with depthwise SE gating."""

    def __init__(self, hidden_size: int, *, stride: int, eps: float):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"LDP stride must be 1 or 2, got {stride}")
        self.depthwise = Conv2dNoBias(
            hidden_size,
            hidden_size,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=hidden_size,
        )
        self.depthwise_norm = LayerNorm(hidden_size, eps=eps)
        squeeze = hidden_size // 4
        if squeeze <= 0:
            raise ValueError(f"LDP hidden size must be at least 4, got {hidden_size}")
        self.se_fc1 = Linear(hidden_size, squeeze, bias=True)
        self.se_fc2 = Linear(squeeze, hidden_size, bias=True)
        self.pointwise = Conv2dNoBias(hidden_size, hidden_size, kernel_size=1)
        self.pointwise_norm = LayerNorm(hidden_size, eps=eps)
        self._stride = stride

    @staticmethod
    def _layer_norm_nchw(op: OpBuilder, norm: LayerNorm, hidden_states: ir.Value) -> ir.Value:
        # LayerNorm is channel-last: [B, C, H, W] -> [B, H, W, C] -> NCHW.
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 3, 1])
        hidden_states = norm(op, hidden_states)
        return op.Transpose(hidden_states, perm=[0, 3, 1, 2])

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        residual = hidden_states
        hidden_states = self.depthwise(op, hidden_states)
        hidden_states = self._layer_norm_nchw(op, self.depthwise_norm, hidden_states)
        hidden_states = op.HardSwish(hidden_states)

        # Squeeze-and-excitation: [B, C, H, W] -> [B, C] -> [B, C, 1, 1].
        gate = op.ReduceMean(hidden_states, [2, 3], keepdims=0)
        gate = op.Relu(self.se_fc1(op, gate))
        gate = op.HardSigmoid(self.se_fc2(op, gate), alpha=1.0 / 6.0, beta=0.5)
        gate = op.Unsqueeze(gate, [2, 3])
        hidden_states = op.Mul(hidden_states, gate)

        hidden_states = self.pointwise(op, hidden_states)
        hidden_states = self._layer_norm_nchw(op, self.pointwise_norm, hidden_states)
        if self._stride == 1:
            hidden_states = op.Add(residual, hidden_states)
        return hidden_states


class MobileLDPProjector(nn.Module):
    """MobileVLM LDP projector for a fixed 24x24 CLIP patch grid.

    The two affine layers preserve 576 tokens, then an inverted-residual block
    keeps the 24x24 grid and a second block downsamples it to 12x12 (144 tokens).
    """

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        *,
        grid_size: int = 24,
        eps: float = 1e-5,
    ):
        super().__init__()
        if grid_size != 24:
            raise ValueError(f"LDP requires the pinned 24x24 patch grid, got {grid_size}")
        self.mlp_1 = Linear(vision_hidden_size, text_hidden_size, bias=True)
        self.mlp_3 = Linear(text_hidden_size, text_hidden_size, bias=True)
        self.block_1 = _MobileLDPBlock(text_hidden_size, stride=1, eps=eps)
        self.block_2 = _MobileLDPBlock(text_hidden_size, stride=2, eps=eps)
        self._grid = grid_size
        self._hidden = text_hidden_size

    def forward(self, op: OpBuilder, vision_features: ir.Value):
        # [B, 576, vision_hidden] -> [B, 576, text_hidden].
        hidden = op.Gelu(self.mlp_1(op, vision_features), approximate="tanh")
        hidden = self.mlp_3(op, hidden)
        batch = op.Shape(hidden, start=0, end=1)
        hidden = op.Transpose(hidden, perm=[0, 2, 1])
        hidden = op.Reshape(
            hidden,
            op.Concat(batch, [self._hidden, self._grid, self._grid], axis=0),
        )
        # [B, text_hidden, 24, 24] -> [B, text_hidden, 12, 12].
        hidden = self.block_2(op, self.block_1(op, hidden))
        hidden = op.Reshape(
            hidden,
            op.Concat(batch, [self._hidden, (self._grid // 2) ** 2], axis=0),
        )
        return op.Transpose(hidden, perm=[0, 2, 1])


class MobileLDPV2Projector(nn.Module):
    """MobileVLM-V2 LDP projector: MLP, 2x2 pool, residual depthwise PEG."""

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        *,
        grid_size: int = 24,
    ):
        super().__init__()
        if grid_size != 24:
            raise ValueError(f"LDPv2 requires the pinned 24x24 patch grid, got {grid_size}")
        self.mlp_0 = Linear(vision_hidden_size, text_hidden_size, bias=True)
        self.mlp_2 = Linear(text_hidden_size, text_hidden_size, bias=True)
        self.peg_0 = Conv2d(
            text_hidden_size,
            text_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=text_hidden_size,
        )
        self._grid = grid_size
        self._hidden = text_hidden_size

    def forward(self, op: OpBuilder, vision_features: ir.Value):
        hidden = self.mlp_2(
            op,
            op.Gelu(self.mlp_0(op, vision_features), approximate="tanh"),
        )
        batch = op.Shape(hidden, start=0, end=1)
        hidden = op.Transpose(hidden, perm=[0, 2, 1])
        hidden = op.Reshape(
            hidden,
            op.Concat(batch, [self._hidden, self._grid, self._grid], axis=0),
        )
        # [B, C, 24, 24] -> [B, C, 12, 12].
        hidden = op.AveragePool(hidden, kernel_shape=[2, 2], strides=[2, 2])
        hidden = op.Add(hidden, self.peg_0(op, hidden))
        hidden = op.Reshape(
            hidden,
            op.Concat(batch, [self._hidden, (self._grid // 2) ** 2], axis=0),
        )
        return op.Transpose(hidden, perm=[0, 2, 1])


class GLMEdgeAdapterProjector(nn.Module):
    """GLM-Edge spatial adapter with gated MLP and boundary embeddings."""

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        intermediate_size: int,
        *,
        grid_size: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        if grid_size <= 0 or grid_size % 2:
            raise ValueError(f"GLM-Edge grid size must be positive and even, got {grid_size}")
        if intermediate_size <= 0:
            raise ValueError("GLM-Edge intermediate size must be positive")
        self.conv = Conv2d(
            vision_hidden_size,
            text_hidden_size,
            kernel_size=2,
            stride=2,
        )
        self.linear = Linear(text_hidden_size, text_hidden_size, bias=False)
        self.norm1 = LayerNorm(text_hidden_size, eps=eps)
        self.dense_h_to_4h = Linear(text_hidden_size, intermediate_size, bias=False)
        self.gate = Linear(text_hidden_size, intermediate_size, bias=False)
        self.dense_4h_to_h = Linear(intermediate_size, text_hidden_size, bias=False)
        self.boi = nn.Parameter([text_hidden_size])
        self.eoi = nn.Parameter([text_hidden_size])
        self._grid = grid_size
        self._vision_hidden = vision_hidden_size
        self._text_hidden = text_hidden_size

    def forward(self, op: OpBuilder, vision_features: ir.Value):
        batch = op.Shape(vision_features, start=0, end=1)
        hidden = op.Reshape(
            vision_features,
            op.Concat(
                batch,
                [self._grid, self._grid, self._vision_hidden],
                axis=0,
            ),
        )
        hidden = self.conv(op, op.Transpose(hidden, perm=[0, 3, 1, 2]))
        # [B, text_hidden, grid/2, grid/2] -> [B, grid^2/4, text_hidden].
        hidden = op.Reshape(
            hidden,
            op.Concat(batch, [self._text_hidden, -1], axis=0),
        )
        hidden = op.Transpose(hidden, perm=[0, 2, 1])
        hidden = op.Gelu(
            self.norm1(op, self.linear(op, hidden)),
            approximate="tanh",
        )

        gate = self.gate(op, hidden)
        up = self.dense_h_to_4h(op, hidden)
        hidden = self.dense_4h_to_h(op, op.Mul(op.Mul(gate, op.Sigmoid(gate)), up))

        boundary_shape = op.Concat(batch, [1, self._text_hidden], axis=0)
        boi = op.Expand(op.Reshape(self.boi, [1, 1, self._text_hidden]), boundary_shape)
        eoi = op.Expand(op.Reshape(self.eoi, [1, 1, self._text_hidden]), boundary_shape)
        return op.Concat(boi, hidden, eoi, axis=1)


class MiniCPMResamplerProjector(nn.Module):
    """MiniCPM-V learned-query resampler with 2D sinusoidal key positions."""

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        *,
        num_queries: int,
        grid_size: int,
        head_dim: int = 128,
        eps: float = 1e-6,
    ):
        super().__init__()
        if num_queries <= 0 or grid_size <= 0:
            raise ValueError("MiniCPM resampler query and grid sizes must be positive")
        if text_hidden_size % head_dim:
            raise ValueError(
                f"MiniCPM resampler width {text_hidden_size} must be divisible by "
                f"head_dim {head_dim}"
            )
        if text_hidden_size % 4:
            raise ValueError("MiniCPM resampler width must be divisible by four")
        self.query = nn.Parameter([num_queries, text_hidden_size])
        self.pos_embed = nn.Parameter([num_queries, text_hidden_size])
        self.kv = Linear(vision_hidden_size, text_hidden_size, bias=False)
        self.ln_q = LayerNorm(text_hidden_size, eps=eps)
        self.ln_kv = LayerNorm(text_hidden_size, eps=eps)
        self.attn_q = Linear(text_hidden_size, text_hidden_size, bias=True)
        self.attn_k = Linear(text_hidden_size, text_hidden_size, bias=True)
        self.attn_v = Linear(text_hidden_size, text_hidden_size, bias=True)
        self.attn_out = Linear(text_hidden_size, text_hidden_size, bias=True)
        self.ln_post = LayerNorm(text_hidden_size, eps=eps)
        self.proj = Linear(text_hidden_size, text_hidden_size, bias=False)
        self._num_queries = num_queries
        self._grid = grid_size
        self._hidden = text_hidden_size
        self._head_dim = head_dim
        self._heads = text_hidden_size // head_dim

    def _position_embedding(self, op: OpBuilder, like: ir.Value) -> ir.Value:
        num_patches = self._grid * self._grid
        indices = op.Range(
            op.Constant(value_int=0),
            op.Constant(value_int=num_patches),
            op.Constant(value_int=1),
        )
        pos_h = op.Cast(op.Div(indices, op.Constant(value_int=self._grid)), to=1)
        pos_w = op.Cast(op.Mod(indices, op.Constant(value_int=self._grid)), to=1)
        omega_indices = op.Cast(
            op.Range(
                op.Constant(value_int=0),
                op.Constant(value_int=self._hidden // 4),
                op.Constant(value_int=1),
            ),
            to=1,
        )
        exponent = op.Div(omega_indices, op.Constant(value_float=self._hidden / 4))
        omega = op.Reciprocal(op.Pow(op.Constant(value_float=10_000.0), exponent))
        theta_x = op.Mul(op.Unsqueeze(pos_w, [1]), op.Unsqueeze(omega, [0]))
        theta_y = op.Mul(op.Unsqueeze(pos_h, [1]), op.Unsqueeze(omega, [0]))
        position = op.Concat(
            op.Sin(theta_x),
            op.Cos(theta_x),
            op.Sin(theta_y),
            op.Cos(theta_y),
            axis=1,
        )
        return op.CastLike(position, like)

    def forward(self, op: OpBuilder, vision_features: ir.Value):
        batch = op.Shape(vision_features, start=0, end=1)
        query = op.Expand(
            op.Unsqueeze(self.query, [0]),
            op.Concat(batch, [self._num_queries, self._hidden], axis=0),
        )
        query = op.Add(self.ln_q(op, query), op.Unsqueeze(self.pos_embed, [0]))
        value = self.ln_kv(op, self.kv(op, vision_features))
        key = op.Add(value, op.Unsqueeze(self._position_embedding(op, value), [0]))

        q = self.attn_q(op, query)
        k = self.attn_k(op, key)
        v = self.attn_v(op, value)
        # [B, tokens, hidden] -> [B, heads, tokens, 128].
        q = op.Transpose(
            op.Reshape(
                q,
                op.Concat(
                    batch,
                    [self._num_queries, self._heads, self._head_dim],
                    axis=0,
                ),
            ),
            perm=[0, 2, 1, 3],
        )
        k = op.Transpose(
            op.Reshape(
                k,
                op.Concat(
                    batch,
                    [self._grid**2, self._heads, self._head_dim],
                    axis=0,
                ),
            ),
            perm=[0, 2, 1, 3],
        )
        v = op.Transpose(
            op.Reshape(
                v,
                op.Concat(
                    batch,
                    [self._grid**2, self._heads, self._head_dim],
                    axis=0,
                ),
            ),
            perm=[0, 2, 1, 3],
        )
        scores = op.MatMul(q, op.Transpose(k, perm=[0, 1, 3, 2]))
        scores = op.Mul(scores, op.CastLike(self._head_dim**-0.5, scores))
        hidden = op.MatMul(op.Softmax(scores, axis=-1), v)
        hidden = op.Reshape(
            op.Transpose(hidden, perm=[0, 2, 1, 3]),
            op.Concat(batch, [self._num_queries, self._hidden], axis=0),
        )
        hidden = self.attn_out(op, hidden)
        return self.proj(op, self.ln_post(op, hidden))


class Cosmos3EdgeMultiModalProjector(nn.Module):
    """Cosmos3-Edge pixel-shuffle merger projector.

    ``LayerNorm → spatial 2x2 pixel-shuffle → Linear(fc1) → GELU → Linear(fc2)``

    The SigLIP vision encoder emits a fixed ``grid x grid`` patch grid
    (``num_patches`` patches, e.g. 16x16 = 256). ``use_postshuffle_norm=false``
    means the ``LayerNorm`` is applied on the raw ``vision_hidden_size`` (1152)
    features **before** the spatial merge. The merge concatenates each
    ``spatial_merge_size x spatial_merge_size`` block of adjacent patches into a
    single ``spatial_merge_size**2 * vision_hidden_size`` (4608) vector, which
    ``linear_fc1`` maps to ``intermediate_size`` (11520) and ``linear_fc2`` maps
    to ``text_hidden_size`` (2048).

    HF weights (``model.projector.*``):
    - ``norm.{weight,bias}`` (pre-shuffle LayerNorm)
    - ``linear_fc1.{weight,bias}``
    - ``linear_fc2.{weight,bias}``
    """

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        intermediate_size: int,
        grid_size: int,
        spatial_merge_size: int = 2,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        if grid_size <= 0:
            raise ValueError(f"grid_size must be positive, got {grid_size}")
        if spatial_merge_size <= 0:
            raise ValueError(f"spatial_merge_size must be positive, got {spatial_merge_size}")
        if grid_size % spatial_merge_size != 0:
            raise ValueError(
                f"grid_size ({grid_size}) must be divisible by "
                f"spatial_merge_size ({spatial_merge_size})"
            )
        self._grid = grid_size
        self._ms = spatial_merge_size
        self._vision_hidden = vision_hidden_size
        merged_dim = vision_hidden_size * spatial_merge_size * spatial_merge_size
        # Pre-shuffle LayerNorm over the raw vision hidden size.
        self.norm = VisionLayerNorm(vision_hidden_size, eps=norm_eps)
        self.linear_fc1 = Linear(merged_dim, intermediate_size, bias=True)
        self.linear_fc2 = Linear(intermediate_size, text_hidden_size, bias=True)

    def forward(self, op: OpBuilder, vision_features: ir.Value):
        # vision_features: [batch, grid*grid, vision_hidden]
        ms = self._ms
        g = self._grid
        gm = g // ms
        d = self._vision_hidden

        # Pre-shuffle LayerNorm (use_postshuffle_norm=false).
        x = self.norm(op, vision_features)

        batch = op.Shape(vision_features, start=0, end=1)  # dynamic [1]

        # [B, g*g, D] -> [B, g/ms, ms, g/ms, ms, D]
        shape_6d = op.Concat(
            batch,
            op.Constant(value_ints=[gm, ms, gm, ms, d]),
            axis=0,
        )
        x = op.Reshape(x, shape_6d)
        # Group hidden dim outermost per merged block (HF F.unfold ordering):
        # [B, g/ms, ms, g/ms, ms, D] -> [B, g/ms, g/ms, D, ms, ms]
        x = op.Transpose(x, perm=[0, 1, 3, 5, 2, 4])
        # Flatten to [B, (g/ms)^2, D*ms*ms]
        shape_3d = op.Concat(
            batch,
            op.Constant(value_ints=[gm * gm, d * ms * ms]),
            axis=0,
        )
        x = op.Reshape(x, shape_3d)

        x = self.linear_fc1(op, x)
        x = op.Gelu(x)
        x = self.linear_fc2(op, x)
        return x


class LinearMultiModalProjector(nn.Module):
    """Single linear projection (PaliGemma, Qwen2-Audio).

    The simplest projector — a single ``nn.Linear`` mapping vision features
    directly to the text hidden dimension.

    HF reference: ``PaliGemmaMultiModalProjector`` in
    ``transformers.models.paligemma.modeling_paligemma``.
    """

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        bias: bool = True,
    ):
        super().__init__()
        self.linear = Linear(vision_hidden_size, text_hidden_size, bias=bias)

    def forward(self, op: OpBuilder, vision_features: ir.Value):
        return self.linear(op, vision_features)


class InputMixer(nn.Module):
    """Merges vision embeddings into text embeddings at placeholder positions.

    Replaces image_token_id positions in text embeddings with projected
    vision embeddings using scatter-like operations.
    """

    def __init__(self, image_token_id: int):
        super().__init__()
        self.image_token_id = image_token_id

    def forward(
        self,
        op: OpBuilder,
        text_embeddings: ir.Value,
        vision_embeddings: ir.Value,
        input_ids: ir.Value,
    ):
        # text_embeddings: [batch, text_seq, hidden]
        # vision_embeddings: [batch, vision_seq, hidden]
        # input_ids: [batch, text_seq]

        # Create mask where input_ids == image_token_id
        token_id = op.Constant(value_int=self.image_token_id)
        mask = op.Equal(input_ids, token_id)  # [batch, text_seq]
        # Expand mask to [batch, text_seq, 1] for broadcasting
        mask_expanded = op.Unsqueeze(mask, [-1])

        # Pad vision_embeddings with a single zero row so GatherElements
        # always has at least one row to index into.  When the modality
        # is absent (vision_seq == 0), all indices clamp to 0 and the
        # mask ensures the padding is never selected.
        batch_dim = op.Shape(vision_embeddings, start=0, end=1)
        hidden_dim = op.Shape(vision_embeddings, start=2, end=3)
        pad_shape = op.Concat(
            batch_dim,
            op.Constant(value_ints=[1]),
            hidden_dim,
            axis=0,
        )
        zero_pad = op.Expand(
            op.CastLike(0.0, vision_embeddings),
            pad_shape,
        )
        # [batch, vision_seq + 1, hidden]
        vision_padded = op.Concat(vision_embeddings, zero_pad, axis=1)

        # Create full-size vision tensor at text positions
        # Use cumulative sum of mask to index into vision_embeddings
        mask_int = op.Cast(mask, to=7)  # INT64
        # cumsum along seq dim gives position indices into vision embeddings
        cumsum = op.CumSum(mask_int, op.Constant(value_int=1))
        # Subtract 1 for 0-based indexing, clamp to 0
        indices = op.Sub(cumsum, op.Constant(value_int=1))
        indices = op.Clip(indices, op.Constant(value_int=0))

        # Gather vision embeddings at computed indices
        # indices: [batch, text_seq], vision_padded: [batch, vision_seq+1, hidden]
        indices_3d = op.Unsqueeze(indices, [-1])
        expand_shape = op.Concat(
            op.Constant(value_ints=[1, 1]),
            hidden_dim,
            axis=0,
        )
        indices_expanded = op.Expand(indices_3d, expand_shape)
        scattered_vision = op.GatherElements(vision_padded, indices_expanded, axis=1)

        # Mix: where mask, use vision; else use text
        mixed = op.Where(mask_expanded, scattered_vision, text_embeddings)
        return mixed
