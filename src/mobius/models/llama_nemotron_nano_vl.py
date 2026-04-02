# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Llama_Nemotron_Nano_VL multimodal model — RADIO vision + Llama 3.1 text decoder.

Splits the Llama_Nemotron_Nano_VL architecture into three ONNX models for
onnxruntime-genai:

- **decoder**: Llama 3.1 text decoder taking ``inputs_embeds``
- **vision**: RADIO ViT-H/16 encoder + pixel shuffle + MLP projector
- **embedding**: token embedding + image feature fusion

Architecture differences from InternVL2:
- RADIO ViT-H/16 vision encoder with CPE-style conditional positional encoding
- CPE = ViTPatchGenerator: learned pos_embed stored at max resolution (2048×2048),
  window-selected to actual input resolution at inference (512×512 → 32×32 grid)
- pos_embed added to patch tokens BEFORE prepending CLS + register tokens
- Patch embedding uses Conv2d without bias (``Conv2dNoBias``)
- 8 special summary tokens total (4 CLS per teacher + 4 registers) stripped after ViT
- No final LayerNorm (RADIO sets ``model.norm = nn.Identity()``)
- Llama 3.1 text decoder (GQA) instead of InternLM/Qwen2

HuggingFace reference: ``nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1``
(model_type ``Llama_Nemotron_Nano_VL``).

HuggingFace weight names:
- ``vision_model.radio_model.model.patch_generator.{embedder.weight, cls_token.token, pos_embed}``
- ``vision_model.radio_model.model.blocks.N.*``
- ``vision_model.radio_model.input_conditioner.*`` (skipped — normalization stats)
- ``mlp1.{0,1,3}.*``
- ``language_model.model.* / language_model.lm_head.*``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
from mobius.components import (
    Conv2dNoBias,
    Linear,
)
from mobius.components._vision import VisionLayerNorm
from mobius.models.internvl import (
    _GELUPlaceholder,
    _InternVisionAttention,
    _InternVisionMLP,
    _InternVL2DecoderModel,
    _InternVL2EmbeddingModel,
)

if TYPE_CHECKING:
    import onnx_ir as ir


# ---------------------------------------------------------------------------
# RADIO vision encoder components
# ---------------------------------------------------------------------------


class _RADIOBlock(nn.Module):
    """RADIO transformer block: pre-norm attention + pre-norm FFN.

    Unlike InternViT, RADIO omits layer scale parameters (``ls1``, ``ls2``).
    Attention and MLP sub-modules reuse InternViT implementations since the
    QKV-fused attention and fc1→GELU→fc2 MLP are identical.

    HF reference: ViT block in ``vision_model.radio_model.model.blocks.N``.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.attn = _InternVisionAttention(hidden_size, num_heads)
        self.mlp = _InternVisionMLP(hidden_size, intermediate_size)
        self.norm1 = VisionLayerNorm(hidden_size, eps=norm_eps)
        self.norm2 = VisionLayerNorm(hidden_size, eps=norm_eps)

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value):
        # Pre-norm attention with residual (no layer scale)
        residual = hidden_states
        hidden_states = self.norm1(op, hidden_states)
        hidden_states = self.attn(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        # Pre-norm FFN with residual (no layer scale)
        residual = hidden_states
        hidden_states = self.norm2(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)
        return hidden_states


class _RADIOVisionModel(nn.Module):
    """RADIO ViT-H/16 vision encoder with CPE-style variable-resolution position encoding.

    Uses RADIO's ``ViTPatchGenerator`` architecture:
    1. Patch embedding (Conv2d, no bias)
    2. Add position embeddings to patch tokens only (BEFORE prepending CLS)
    3. Prepend CLS + register tokens (``num_summary_tokens`` total)
    4. 32 transformer blocks (pre-norm, no layer scale)
    5. No final LayerNorm (RADIO sets ``model.norm = nn.Identity()``)

    RADIO's position embedding table (``pos_embed``) is stored at max-resolution
    ``(cpe_max_size // patch_size)²``.  For the fixed 512×512 inference resolution
    used here, a ``(32×32)`` window is sliced from the ``(128×128)`` table during
    weight loading in ``preprocess_weights``.

    Parameter shapes:
    - ``patch_embed.weight``: ``[hidden, 3, patch, patch]``
    - ``cls_token``: ``[num_summary_tokens, hidden]`` (CLS + register tokens, no batch dim)
    - ``pos_embed``: ``[1, num_patches, hidden]`` (patches only, no CLS position)

    HF reference: ``vision_model.radio_model.model`` in ``Llama_Nemotron_Nano_VL``.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        hidden_size: int,
        num_layers: int,
        intermediate_size: int,
        num_heads: int,
        num_summary_tokens: int = 8,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.num_summary_tokens = num_summary_tokens
        # RADIO patch embedding has no bias term
        self.patch_embed = Conv2dNoBias(
            3, hidden_size, kernel_size=patch_size, stride=patch_size
        )
        # CLS + register tokens — shape [num_summary_tokens, hidden_size] (no batch dim)
        # HF: cls_token.token  shape [num_cls + num_registers, hidden]
        self.cls_token = nn.Parameter([num_summary_tokens, hidden_size])
        # Position embedding table for patch positions only (no CLS in pos table)
        # At runtime (fixed 512×512 input), shape [1, num_patches, hidden_size]
        # preprocess_weights windows this from the large [1, max_patches, hidden] table
        self.pos_embed = nn.Parameter([1, self.num_patches, hidden_size])
        self.blocks = nn.ModuleList(
            [
                _RADIOBlock(hidden_size, intermediate_size, num_heads, norm_eps)
                for _ in range(num_layers)
            ]
        )

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        # pixel_values: [batch, 3, image_size, image_size]
        patch_embeds = self.patch_embed(op, pixel_values)
        # patch_embeds: [batch, hidden_size, grid_h, grid_w]

        # Flatten spatial dims: [batch, hidden_size, num_patches]
        batch_size = op.Shape(patch_embeds, start=0, end=1)
        hidden_dim = op.Shape(patch_embeds, start=1, end=2)
        flat_shape = op.Concat(batch_size, hidden_dim, op.Constant(value_ints=[-1]), axis=0)
        patch_embeds = op.Reshape(patch_embeds, flat_shape)
        # Transpose to [batch, num_patches, hidden_size]
        patch_embeds = op.Transpose(patch_embeds, perm=[0, 2, 1])

        # Add position embeddings to patch tokens ONLY (RADIO style — before CLS prepend)
        # pos_embed: [1, num_patches, hidden_size] → broadcast over batch
        patch_embeds = op.Add(patch_embeds, self.pos_embed)

        # Expand and prepend CLS + register tokens: [num_summary_tokens, hidden] → [batch, N, hidden]
        cls = op.Expand(
            op.Unsqueeze(self.cls_token, [0]),  # [1, num_summary_tokens, hidden]
            op.Concat(
                batch_size,
                op.Constant(value_ints=[self.num_summary_tokens]),
                hidden_dim,
                axis=0,
            ),
        )
        # [batch, num_summary_tokens + num_patches, hidden_size]
        embeddings = op.Concat(cls, patch_embeds, axis=1)

        # Pass through transformer blocks (no final LayerNorm — RADIO uses nn.Identity)
        for block in self.blocks:
            embeddings = block(op, embeddings)
        return embeddings


# ---------------------------------------------------------------------------
# Vision encoder with pixel shuffle + MLP projector
# ---------------------------------------------------------------------------


class _LlamaNemotronNanoVLVisionEncoderModel(nn.Module):
    """RADIO vision encoder + pixel shuffle + mlp1 projector.

    Pipeline: pixel_values → RADIO ViT → strip CLS → pixel_shuffle(0.5)
    → mlp1(LayerNorm → Linear → GELU → Linear) → image_features.

    The pixel shuffle halves spatial dimensions and quadruples channels:
    ``(N, H*W, C) → (N, H/2 * W/2, C*4)``

    HF reference: ``Llama_Nemotron_Nano_VL.extract_feature`` +
    ``Llama_Nemotron_Nano_VL.pixel_shuffle``.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        vc = config.vision
        assert vc is not None, "VisionConfig is required"
        self._downsample_ratio = 0.5
        vit_hidden = vc.hidden_size
        llm_hidden = config.hidden_size
        # After pixel shuffle: vit_hidden * (1 / ratio)^2 = vit_hidden * 4
        proj_input_dim = vit_hidden * int((1 / self._downsample_ratio) ** 2)
        # Number of special tokens (CLS + registers) prepended by RADIO's ViTPatchGenerator
        self._num_summary_tokens = vc.num_summary_tokens  # 8 for C-RADIOv2-H

        self.vision_model = _RADIOVisionModel(
            image_size=vc.image_size,
            patch_size=vc.patch_size,
            hidden_size=vit_hidden,
            num_layers=vc.num_hidden_layers,
            intermediate_size=vc.intermediate_size,
            num_heads=vc.num_attention_heads,
            num_summary_tokens=self._num_summary_tokens,
            norm_eps=vc.norm_eps,
        )
        # mlp1: Sequential(LayerNorm, Linear, GELU, Linear)
        # Indices: 0=LayerNorm, 1=Linear, 2=GELU (no params), 3=Linear
        self.mlp1 = nn.Sequential(
            VisionLayerNorm(proj_input_dim),
            Linear(proj_input_dim, llm_hidden, bias=True),
            _GELUPlaceholder(),
            Linear(llm_hidden, llm_hidden, bias=True),
        )

    def forward(self, op: builder.OpBuilder, pixel_values: ir.Value):
        # Run RADIO ViT encoder
        vit_embeds = self.vision_model(op, pixel_values)
        # vit_embeds: [batch, num_summary_tokens + num_patches, hidden_size]

        # Strip all special tokens (CLS + registers) — RADIO uses num_summary_tokens=8
        # for C-RADIOv2-H (4 CLS per teacher + 4 registers).
        # Result: [batch, num_patches, hidden_size] (spatial patch features only)
        vit_embeds = op.Slice(
            vit_embeds,
            op.Constant(value_ints=[self._num_summary_tokens]),  # start
            op.Constant(value_ints=[2147483647]),  # end=INT_MAX
            op.Constant(value_ints=[1]),  # axes=[1] (sequence dim)
        )

        # Pixel shuffle: (batch, num_patches, C) → (batch, num_patches/4, C*4)
        vit_embeds = self._pixel_shuffle(op, vit_embeds)

        # MLP projector: LayerNorm → Linear → GELU → Linear
        image_features = self.mlp1(op, vit_embeds)
        return image_features

    def _pixel_shuffle(self, op, x):
        """Pixel shuffle downsampling (ps_version='v2').

        Reshapes (N, H*W, C) → spatial grid → interleave by scale factor
        → flatten back to (N, H'*W', C').

        With downsample_ratio=0.5:
          (N, H*W, C) → (N, H, W, C) → (N, W, H/2, C*2)
          → (N, H/2, W, C*2) → (N, H/2, W/2, C*4)
          → (N, W/2, H/2, C*4) → (N, H/2*W/2, C*4)

        HF reference: ``Llama_Nemotron_Nano_VL.pixel_shuffle``.
        """
        scale = self._downsample_ratio  # 0.5
        batch = op.Shape(x, start=0, end=1)
        seq_len = op.Shape(x, start=1, end=2)
        channels = op.Shape(x, start=2, end=3)

        # Compute H = W = sqrt(num_patches). Tiles are always square.
        h = op.Cast(op.Sqrt(op.Cast(seq_len, to=1)), to=7)  # float sqrt → int64
        w = h

        # Reshape to (N, H, W, C) spatial grid
        x = op.Reshape(x, op.Concat(batch, h, w, channels, axis=0))

        # Step 1: view(N, W, H*scale, C/scale)
        h_scaled = op.Cast(op.Mul(op.Cast(h, to=1), op.Constant(value_float=scale)), to=7)
        c_over_scale = op.Cast(
            op.Div(op.Cast(channels, to=1), op.Constant(value_float=scale)), to=7
        )
        x = op.Reshape(x, op.Concat(batch, w, h_scaled, c_over_scale, axis=0))

        # Step 2: permute(0, 2, 1, 3) → (N, H*scale, W, C/scale)
        x = op.Transpose(x, perm=[0, 2, 1, 3])

        # Step 3: view(N, H*scale, W*scale, C/(scale^2))
        w_scaled = op.Cast(op.Mul(op.Cast(w, to=1), op.Constant(value_float=scale)), to=7)
        c_over_scale2 = op.Cast(
            op.Div(op.Cast(channels, to=1), op.Constant(value_float=scale * scale)), to=7
        )
        x = op.Reshape(x, op.Concat(batch, h_scaled, w_scaled, c_over_scale2, axis=0))

        # Step 4 (ps_version='v2'): permute(0, 2, 1, 3)
        x = op.Transpose(x, perm=[0, 2, 1, 3])

        # Flatten spatial dims → (N, H'*W', C*4)
        x = op.Reshape(
            x, op.Concat(batch, op.Constant(value_ints=[-1]), c_over_scale2, axis=0)
        )
        return x

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {
            key: value
            for key, value in state_dict.items()
            if key.startswith(("vision_model.", "mlp1."))
        }


# ---------------------------------------------------------------------------
# Three-model split
# ---------------------------------------------------------------------------


class LlamaNemotronNanoVLModel(nn.Module):
    """Llama_Nemotron_Nano_VL vision-language model (3-model split).

    Builds three separate ONNX models:
    - decoder: Llama 3.1 text decoder taking inputs_embeds
    - vision: RADIO ViT-H/16 + pixel shuffle + MLP projector
    - embedding: token embedding + image feature fusion

    HF reference: ``nvidia/Llama-Nemotron-Nano-VL-8B-v1``
    (model_type ``Llama_Nemotron_Nano_VL``).
    """

    default_task: str = "vision-language"
    category: str = "Multimodal"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.decoder = _InternVL2DecoderModel(config)
        self.vision_encoder = _LlamaNemotronNanoVLVisionEncoderModel(config)
        self.embedding = _InternVL2EmbeddingModel(config)

    def forward(self, op: builder.OpBuilder, **kwargs):
        raise NotImplementedError(
            "LlamaNemotronNanoVLModel uses VisionLanguageTask which calls "
            "each sub-module (decoder, vision_encoder, embedding) separately."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Route HF weights to the correct ONNX sub-model initializer names.

        HF prefixes → ONNX prefixes:
        - ``vision_model.radio_model.model.patch_generator.embedder.weight``
            → ``vision_encoder.vision_model.patch_embed.weight``
        - ``vision_model.radio_model.model.patch_generator.cls_token.token``
            → ``vision_encoder.vision_model.cls_token``
            (shape [num_summary_tokens, hidden] — CLS + registers, no batch dim)
        - ``vision_model.radio_model.model.patch_generator.pos_embed``
            → ``vision_encoder.vision_model.pos_embed``  (window-selected from max resolution)
            The stored pos_embed is [1, (cpe_max_size/patch)², hidden].
            For fixed 512×512 inference we window-select the top-left 32×32 tile:
            reshape [1, 128, 128, hidden] → take [:, :32, :32, :] → [1, 1024, hidden].
        - ``vision_model.radio_model.model.blocks.N.*``
            → ``vision_encoder.vision_model.blocks.N.*``
        - ``vision_model.radio_model.input_conditioner.*`` → SKIP (image normalization stats)
        - ``mlp1.*`` → ``vision_encoder.mlp1.*``
        - ``language_model.model.*`` → ``decoder.model.*``
        - ``language_model.lm_head.*`` → ``decoder.lm_head.*``
        - ``language_model.model.embed_tokens.*``
            → ``embedding.embed_tokens.*`` (dual copy)
        """
        renamed: dict[str, torch.Tensor] = {}

        vc = self.config.vision
        assert vc is not None
        # Window-select parameters: pos_embed is stored at max resolution (128×128 grid)
        # but we use a fixed 32×32 grid for 512×512 images.
        image_size = vc.image_size
        patch_size = vc.patch_size
        h_in = w_in = image_size // patch_size  # 32 for 512px

        pg = "vision_model.radio_model.model.patch_generator."
        blocks_prefix = "vision_model.radio_model.model.blocks."
        ic = "vision_model.radio_model.input_conditioner."

        for key, value in state_dict.items():
            if key.startswith(ic):
                # Skip image normalization stats — not used for ONNX inference
                continue
            elif key == pg + "embedder.weight":
                renamed["vision_encoder.vision_model.patch_embed.weight"] = value
            elif key == pg + "cls_token.token":
                # shape [num_summary_tokens, hidden] — no batch dim (ClsToken.token)
                renamed["vision_encoder.vision_model.cls_token"] = value
            elif key == pg + "pos_embed":
                # Window-select from max-resolution table → target inference resolution.
                # HF stores [1, (cpe_max/patch)², hidden]; we slice [1, h_in*w_in, hidden].
                # The table is a 2-D grid; top-left (h_in × w_in) window matches the
                # upper-left patch positions at the target resolution.
                n, max_patches, c = value.shape
                h_max = w_max = int(max_patches**0.5)
                if h_max == h_in:
                    # Already the right size (e.g. model trained at exactly image_size)
                    renamed["vision_encoder.vision_model.pos_embed"] = value
                else:
                    # Reshape to 2-D grid, take top-left window, flatten
                    grid = value.reshape(n, h_max, w_max, c)
                    sliced = grid[:, :h_in, :w_in, :].reshape(n, h_in * w_in, c)
                    renamed["vision_encoder.vision_model.pos_embed"] = sliced.contiguous()
            elif key.startswith(blocks_prefix):
                # blocks.N.* → vision_encoder.vision_model.blocks.N.*
                suffix = key[len("vision_model.radio_model.model.") :]
                renamed[f"vision_encoder.vision_model.{suffix}"] = value
            elif key.startswith("mlp1."):
                renamed[f"vision_encoder.{key}"] = value
            elif key.startswith("language_model."):
                suffix = key[len("language_model.") :]
                renamed[f"decoder.{suffix}"] = value
                # Embedding model also needs embed_tokens
                if suffix.startswith("model.embed_tokens."):
                    embed_suffix = suffix[len("model.") :]
                    renamed[f"embedding.{embed_suffix}"] = value

        # Weight tying: copy embed_tokens → lm_head when lm_head is absent
        if self.config.tie_word_embeddings:
            embed_key = "embedding.embed_tokens.weight"
            head_key = "decoder.lm_head.weight"
            if head_key not in renamed and embed_key in renamed:
                renamed[head_key] = renamed[embed_key]

        return renamed
