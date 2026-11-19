# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""LiquidAI LFM3-VL vision-language model — 3-model split.

Replicates the checkpoint's remote-code ``Lfm3VlForConditionalGeneration``.
Builds three ONNX models:

- **decoder**: LFM3 hybrid decoder taking ``inputs_embeds``
- **vision_encoder**: gated-conv / GQA vision tower + MLP projector, one image
  per call (the ``t`` frames of its grid row run as a batch)
- **embedding**: token embedding + image-feature fusion; video tokens are not
  scattered, so video input is not supported

Pipeline for ``LiquidAI/LFM3-VL-2B``::

    pixel_values (t*h*w, 3*2*16*16), image_grid_thw [[t, h, w]]
        -> patchifier Linear(1536 -> 1024) -> RMSNorm   -> (t, h*w, 1024)
        -> 14 layers: gated depthwise 3x3 conv or GQA 16/8 with 2D RoPE
        -> LayerNorm, 4 consecutive tokens -> 4096
        -> Linear(4096 -> 4096) -> GELU -> Linear(4096 -> 2048)
                                                        -> (t*h*w/4, 2048)
        -> scattered onto ``image_token_id`` positions of inputs_embeds
        -> LFM3 decoder (24 layers: conv, full and sliding attention)

The processor emits patches in 2x2 merge-block order (block row, block column,
then the 2x2 block); only the convolution works on the raster grid.

HuggingFace weight prefixes -> ONNX initializer names::

    model.vision_tower.*            -> vision_encoder.vision_tower.*
    model.multi_modal_projector.*   -> vision_encoder.multi_modal_projector.*
    model.language_model.*          -> decoder.model.*
    model.language_model.embed_tokens.weight
                                    -> embedding.embed_tokens.weight
                                    -> decoder.lm_head.weight (tie_word_embeddings)
"""

from __future__ import annotations

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._build_context import ep_capabilities
from mobius._configs import ArchitectureConfig, Lfm3VlConfig
from mobius.components import Conv2dNoBias, LayerNorm, Linear, get_activation
from mobius.models.lfm2_vl import _Lfm2VlEmbeddingModel
from mobius.models.lfm3 import _FLOAT32_ONLY, Lfm3MLP, Lfm3RMSNorm, Lfm3TextModel


class Lfm3VlVisionAttention(nn.Module):
    """Bidirectional GQA within each frame, with per-head RMSNorm on queries and keys."""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, eps: float):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = Linear(num_heads * self.head_dim, hidden_size, bias=False)
        self.q_layernorm = Lfm3RMSNorm(self.head_dim, eps=eps)
        self.k_layernorm = Lfm3RMSNorm(self.head_dim, eps=eps)

    def forward(
        self, op: OpBuilder, hidden_states: ir.Value, cos: ir.Value, sin: ir.Value
    ) -> ir.Value:
        query = self._normalize_and_rotate(
            op, self.q_layernorm, self.q_proj(op, hidden_states), cos, sin
        )
        key = self._normalize_and_rotate(
            op, self.k_layernorm, self.k_proj(op, hidden_states), cos, sin
        )
        # One K/V head per query head: MultiHeadAttention needs it, and ORT runs
        # grouped float32 Attention unfused, holding heads x S x S scores per frame.
        key, value = (
            op.Reshape(
                op.Expand(
                    op.Reshape(x, [0, 0, self.num_kv_heads, 1, -1]),
                    [1, 1, 1, self.num_heads // self.num_kv_heads, 1],
                ),
                [0, 0, -1],
            )
            for x in (key, self.v_proj(op, hidden_states))
        )
        # Frames are the batch axis, so no mask is needed to keep them apart.
        caps = ep_capabilities()
        # Without an Attention kernel (QNN) the build lowers this op to MatMul/Softmax.
        if caps.name == "onnx-standard" or not caps.supports_attention:
            output = op.Attention(
                query,
                key,
                value,
                q_num_heads=self.num_heads,
                kv_num_heads=self.num_heads,
                scale=self.head_dim**-0.5,
                _outputs=1,
            )
        else:
            # ORT's CPU Attention holds every score even without groups; this op tiles them.
            output = op.MultiHeadAttention(
                query,
                key,
                value,
                num_heads=self.num_heads,
                scale=self.head_dim**-0.5,
                _domain="com.microsoft",
            )
        return self.out_proj(op, output)  # output: (t, h*w, heads * head_dim)

    def _normalize_and_rotate(
        self, op: OpBuilder, norm: nn.Module, x: ir.Value, cos: ir.Value, sin: ir.Value
    ) -> ir.Value:
        """Per-head RMSNorm, then rotate-half RoPE over the full head in float32."""
        x = norm(op, op.Reshape(x, [0, 0, -1, self.head_dim]))  # (t, h*w, heads, head_dim)
        rotated = op.Cast(x, to=ir.DataType.FLOAT)
        first, second = op.Split(rotated, num_outputs=2, axis=-1, _outputs=2)
        rotated = op.Add(
            op.Mul(rotated, op.Unsqueeze(cos, [1])),
            op.Mul(op.Concat(op.Neg(second), first, axis=-1), op.Unsqueeze(sin, [1])),
        )
        return op.Reshape(op.CastLike(rotated, x), [0, 0, -1])


class Lfm3VlVisionConv2d(nn.Module):
    """Gated depthwise 2D convolution over the raster patch grid of each frame."""

    def __init__(self, hidden_size: int, kernel_size: int, merge_size: int):
        super().__init__()
        self.merge_size = merge_size
        self.in_proj = Linear(hidden_size, 3 * hidden_size, bias=False)
        self.spatial_conv = Conv2dNoBias(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_size,
        )
        self.out_proj = Linear(hidden_size, hidden_size, bias=False)

    def forward(self, op: OpBuilder, hidden_states: ir.Value, frame_shape: tuple) -> ir.Value:
        frames, height, width = frame_shape
        merge = self.merge_size
        block_rows, block_cols = op.Div(height, merge), op.Div(width, merge)
        gate_b, gate_c, values = op.Split(
            self.in_proj(op, hidden_states), num_outputs=3, axis=-1, _outputs=3
        )
        grid = op.Reshape(
            op.Mul(gate_b, values),
            op.Concat(frames, block_rows, block_cols, [merge, merge, -1], axis=0),
        )
        grid = op.Reshape(
            op.Transpose(grid, perm=[0, 5, 1, 3, 2, 4]),
            op.Concat(frames, [-1], height, width, axis=0),
        )  # (t, C, h, w)
        grid = op.Reshape(
            self.spatial_conv(op, grid),
            op.Concat(frames, [-1], block_rows, [merge], block_cols, [merge], axis=0),
        )
        conv = op.Reshape(
            op.Transpose(grid, perm=[0, 2, 4, 3, 5, 1]), op.Shape(hidden_states)
        )  # (t, h*w, C) in merge-block order
        return self.out_proj(op, op.Mul(gate_c, conv))


class Lfm3VlVisionLayer(nn.Module):
    """Pre-norm conv or attention operator, then a SwiGLU MLP, each with a residual."""

    def __init__(self, config: Lfm3VlConfig, layer_type: str):
        super().__init__()
        vision = config.vision
        self.layer_type = layer_type
        if layer_type == "conv":
            self.conv = Lfm3VlVisionConv2d(
                vision.hidden_size, config.vision_conv_kernel_size, vision.spatial_merge_size
            )
        else:
            self.self_attn = Lfm3VlVisionAttention(
                vision.hidden_size,
                vision.num_attention_heads,
                config.vision_num_key_value_heads,
                vision.norm_eps,
            )
        self.feed_forward = Lfm3MLP(vision)
        self.operator_norm = Lfm3RMSNorm(vision.hidden_size, eps=vision.norm_eps)
        self.ffn_norm = Lfm3RMSNorm(vision.hidden_size, eps=vision.norm_eps)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        frame_shape: tuple,
        cos: ir.Value,
        sin: ir.Value,
    ) -> ir.Value:
        operator_input = self.operator_norm(op, hidden_states)
        if self.layer_type == "conv":
            output = self.conv(op, operator_input, frame_shape)
        else:
            output = self.self_attn(op, operator_input, cos, sin)
        hidden_states = op.Add(hidden_states, output)
        return op.Add(hidden_states, self.feed_forward(op, self.ffn_norm(op, hidden_states)))


class Lfm3VlVisionModel(nn.Module):
    """Linear patchifier and RMSNorm, then the conv/attention stack; no final norm."""

    def __init__(self, config: Lfm3VlConfig):
        super().__init__()
        vision = config.vision
        assert vision is not None and config.vision_layer_types is not None
        self.hidden_size = vision.hidden_size
        self.merge_size = vision.spatial_merge_size
        patch_dim = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
        self.patchifier = Linear(patch_dim, vision.hidden_size, bias=False)
        self.embedding_norm = Lfm3RMSNorm(vision.hidden_size, eps=vision.norm_eps)
        self.layers = nn.ModuleList(
            [Lfm3VlVisionLayer(config, kind) for kind in config.vision_layer_types]
        )
        # Axial RoPE: half the head rotates with the row, half with the column.
        # Upstream's float32 torch expression; NumPy's pow differs by up to 2 ULP.
        dim = vision.hidden_size // vision.num_attention_heads // 2
        self.inv_freq = (
            1.0 / (vision.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        ).numpy()

    def forward(
        self, op: OpBuilder, pixel_values: ir.Value, image_grid_thw: ir.Value
    ) -> ir.Value:
        grid = op.Reshape(image_grid_thw, [3])
        frame_shape = tuple(op.Slice(grid, [i], [i + 1], [0]) for i in range(3))
        hidden_states = self.patchifier(op, pixel_values)
        hidden_states = op.Reshape(
            hidden_states, op.Concat(frame_shape[0], [-1, self.hidden_size], axis=0)
        )  # (t, h*w, C)
        hidden_states = self.embedding_norm(op, hidden_states)
        cos, sin = self._rotary(op, frame_shape[1], frame_shape[2])
        for layer in self.layers:
            hidden_states = layer(op, hidden_states, frame_shape, cos, sin)
        return hidden_states

    def _rotary(self, op: OpBuilder, height: ir.Value, width: ir.Value):
        """Float32 ``(cos, sin)`` of shape ``(h*w, head_dim)`` for one frame's patches."""
        merge = self.merge_size
        index = op.Range(0, op.Squeeze(op.Mul(height, width), [0]), 1)
        block = op.Div(index, merge * merge)
        blocks_per_row = op.Div(width, merge)
        rows = op.Add(
            op.Mul(op.Div(block, blocks_per_row), merge),
            op.Div(op.Mod(index, merge * merge), merge),
        )
        cols = op.Add(op.Mul(op.Mod(block, blocks_per_row), merge), op.Mod(index, merge))
        inv_freq = op.Constant(value=ir.tensor(self.inv_freq))
        row_angles, col_angles = (
            op.Mul(op.Unsqueeze(op.Cast(position, to=ir.DataType.FLOAT), [1]), inv_freq)
            for position in (rows, cols)
        )
        angles = op.Concat(row_angles, col_angles, row_angles, col_angles, axis=-1)
        return op.Cos(angles), op.Sin(angles)


class Lfm3VlMultiModalProjector(nn.Module):
    """Per-token LayerNorm, then each block of merged tokens into one text token."""

    def __init__(self, config: Lfm3VlConfig):
        super().__init__()
        vision_hidden_size = config.vision.hidden_size
        self.merged_size = vision_hidden_size * config.downsample_factor**2
        self.layer_norm = LayerNorm(vision_hidden_size, eps=config.projector_layer_norm_eps)
        self.linear_1 = Linear(
            self.merged_size, config.projector_hidden_size, bias=config.projector_bias
        )
        self.act = get_activation(config.projector_hidden_act)
        self.linear_2 = Linear(
            config.projector_hidden_size, config.hidden_size, bias=config.projector_bias
        )

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        merged = op.Reshape(self.layer_norm(op, hidden_states), [-1, self.merged_size])
        return self.linear_2(op, self.act(op, self.linear_1(op, merged)))


class _Lfm3VlVisionEncoderModel(nn.Module):
    """Vision tower plus projector for one image per call."""

    def __init__(self, config: Lfm3VlConfig):
        super().__init__()
        self.vision_tower = Lfm3VlVisionModel(config)
        self.multi_modal_projector = Lfm3VlMultiModalProjector(config)

    def forward(
        self, op: OpBuilder, pixel_values: ir.Value, image_grid_thw: ir.Value
    ) -> ir.Value:
        hidden_states = self.vision_tower(op, pixel_values, image_grid_thw)
        return self.multi_modal_projector(op, hidden_states)  # (t*h*w/4, text_hidden)


class _Lfm3VlDecoderModel(nn.Module):
    """LFM3 hybrid decoder taking pre-fused ``inputs_embeds``."""

    def __init__(self, config: Lfm3VlConfig):
        super().__init__()
        self.model = Lfm3TextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list,
    ):
        hidden_states, present_key_values = self.model(
            op,
            None,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds=inputs_embeds,
        )
        return self.lm_head(op, hidden_states), present_key_values


class Lfm3VlForConditionalGeneration(nn.Module):
    """LiquidAI LFM3-VL image-conditioned generation model (3-model split).

    Replicates the remote-code ``Lfm3VlForConditionalGeneration`` of
    ``LiquidAI/LFM3-VL-2B``: a hybrid gated-convolution/GQA vision tower on
    native-resolution 16x16 patches, a LayerNorm + 2x2-merge MLP projector,
    and the float32 LFM3 decoder with plain 1D positions.
    """

    default_task: str = "lfm3-vl"
    category: str = "Multimodal"
    config_class: type = Lfm3VlConfig

    def __init__(self, config: ArchitectureConfig):
        assert isinstance(config, Lfm3VlConfig), (
            "Lfm3VlForConditionalGeneration requires an Lfm3VlConfig, "
            f"got {type(config).__name__}"
        )
        if config.dtype != ir.DataType.FLOAT:
            raise ValueError(_FLOAT32_ONLY)
        super().__init__()
        self.config = config
        self.decoder = _Lfm3VlDecoderModel(config)
        self.vision_encoder = _Lfm3VlVisionEncoderModel(config)
        self.embedding = _Lfm2VlEmbeddingModel(config)

    def forward(self, op: OpBuilder, **kwargs):
        raise NotImplementedError(
            "Lfm3VlForConditionalGeneration uses Lfm3VlTask, which builds "
            "decoder, vision_encoder, and embedding graphs separately."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Route the HuggingFace checkpoint into the three ONNX sub-models by prefix."""
        prefixes = {
            "model.vision_tower.": "vision_encoder.vision_tower.",
            "model.multi_modal_projector.": "vision_encoder.multi_modal_projector.",
            "model.language_model.": "decoder.model.",
            "lm_head.": "decoder.lm_head.",
        }
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            for source, target in prefixes.items():
                if key.startswith(source):
                    renamed[target + key[len(source) :]] = value
            if key == "model.language_model.embed_tokens.weight":
                renamed["embedding.embed_tokens.weight"] = value
                if self.config.tie_word_embeddings:
                    renamed["decoder.lm_head.weight"] = value
        return renamed
