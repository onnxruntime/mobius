# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""VideoChat-Flash-Qwen VLM model — 3-model split.

Replicates ``OpenGVLab/VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B``
(VideoChatFlashQwenForCausalLM).

Architecture:
- **Vision encoder**: InternVideo2 (24-block ViT with LayerScale,
  QK-norm, fused QKV) + MLP projector (Linear → GELU → Linear).
  The HF model includes a Token Merging (ToMe) step before the MLP
  that reduces the token count; this is not included in the ONNX graph
  as it has no learned parameters and is purely algorithmic.
- **Embedding**: Token lookup + image feature scatter at placeholder
  positions
- **Decoder**: Standard Qwen2.5 (28 layers, GQA 28Q/4KV)

HuggingFace weight layout::

    model.vision_tower.vision_tower.patch_embed.proj.{weight,bias}
    model.vision_tower.vision_tower.pos_embed
    model.vision_tower.vision_tower.img_pos_embed
    model.vision_tower.vision_tower.cls_token
    model.vision_tower.vision_tower.blocks.{i}.attn.qkv.weight
    model.vision_tower.vision_tower.blocks.{i}.attn.proj.{weight,bias}
    model.vision_tower.vision_tower.blocks.{i}.attn.q_norm.weight
    model.vision_tower.vision_tower.blocks.{i}.attn.k_norm.weight
    model.vision_tower.vision_tower.blocks.{i}.ls1.weight
    model.vision_tower.vision_tower.blocks.{i}.ls2.weight
    model.vision_tower.vision_tower.blocks.{i}.mlp.fc1/fc2.{weight,bias}
    model.vision_tower.vision_tower.blocks.{i}.norm1/norm2.weight
    model.mm_projector.mlp.{0,2}.{weight,bias}
    model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
    model.layers.{i}.mlp.{gate,up,down}_proj.weight
    model.embed_tokens.weight
    model.norm.weight
    lm_head.weight
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components import (
    Embedding,
    Linear,
    MLPMultiModalProjector,
)
from mobius.models.base import TextModel

if TYPE_CHECKING:
    import onnx_ir as ir


# ── InternVideo2 Vision Encoder ──────────────────────────────────────────


class _IV2Linear(nn.Module):
    """Linear with bias for InternVideo2 vision encoder."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter([out_features, in_features])
        self.bias = nn.Parameter([out_features])

    def forward(self, op: OpBuilder, x: ir.Value):
        weight_t = op.Transpose(self.weight, perm=[1, 0])
        return op.Add(op.MatMul(x, weight_t), self.bias)


class _IV2Attention(nn.Module):
    """InternVideo2 attention: fused QKV + QK norms.

    Key differences from InternViT:
    - QKV has no bias (``qkv.weight`` only)
    - Per-head QK normalization via RMSNorm (``q_norm``, ``k_norm``)

    HF weight names::

        blocks.{i}.attn.qkv.weight       [3*H, H]
        blocks.{i}.attn.proj.weight       [H, H]
        blocks.{i}.attn.proj.bias         [H]
        blocks.{i}.attn.q_norm.weight     [head_dim]
        blocks.{i}.attn.k_norm.weight     [head_dim]
    """

    def __init__(self, hidden_size: int, num_heads: int, norm_eps: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        # Fused QKV without bias
        self.qkv = nn.Parameter([3 * hidden_size, hidden_size], name="qkv.weight")
        self.proj = _IV2Linear(hidden_size, hidden_size)
        # Full-dimension QK norms (RMSNorm on hidden_size, not per-head)
        from mobius.components._rms_norm import RMSNorm

        self.q_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.k_norm = RMSNorm(hidden_size, eps=norm_eps)

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        # Fused QKV projection (no bias)
        qkv_weight_t = op.Transpose(self.qkv, perm=[1, 0])
        qkv = op.MatMul(hidden_states, qkv_weight_t)  # [B, S, 3*H]
        q, k, v = op.Split(qkv, num_outputs=3, axis=-1, _outputs=3)

        # Apply full-dimension QK norms (on 3D tensor)
        q = self.q_norm(op, q)
        k = self.k_norm(op, k)

        # Bidirectional attention (no causal mask, no KV cache)
        attn_output = op.Attention(
            q,
            k,
            v,
            kv_num_heads=self.num_heads,
            q_num_heads=self.num_heads,
            scale=self.scale,
            _outputs=1,
        )
        return self.proj(op, attn_output)


class _IV2MLP(nn.Module):
    """InternVideo2 MLP: fc1 (with bias) → GELU → fc2 (with bias)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = _IV2Linear(hidden_size, intermediate_size)
        self.fc2 = _IV2Linear(intermediate_size, hidden_size)

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        return self.fc2(op, op.Gelu(self.fc1(op, hidden_states)))


class _IV2EncoderLayer(nn.Module):
    """InternVideo2 encoder layer: RMSNorm + Attention + LayerScale.

    Structure: RMSNorm → Attention → QK-norm → LayerScale → Residual
              → RMSNorm → MLP → LayerScale → Residual

    HF weight names::

        blocks.{i}.norm1.weight           [H]  (RMSNorm, no bias)
        blocks.{i}.norm2.weight           [H]
        blocks.{i}.ls1.weight             [H]  (LayerScale)
        blocks.{i}.ls2.weight             [H]
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        from mobius.components._rms_norm import RMSNorm

        self.attn = _IV2Attention(hidden_size, num_heads, norm_eps)
        self.mlp = _IV2MLP(hidden_size, intermediate_size)
        # RMSNorm (no bias, unlike InternViT's LayerNorm)
        self.norm1 = RMSNorm(hidden_size, eps=norm_eps)
        self.norm2 = RMSNorm(hidden_size, eps=norm_eps)
        # LayerScale — learned per-channel scaling
        self.ls1 = nn.Parameter([hidden_size], name="ls1.weight")
        self.ls2 = nn.Parameter([hidden_size], name="ls2.weight")

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        # Pre-norm attention with layer scale
        residual = hidden_states
        hidden_states = self.norm1(op, hidden_states)
        hidden_states = self.attn(op, hidden_states)
        hidden_states = op.Mul(hidden_states, self.ls1)
        hidden_states = op.Add(residual, hidden_states)

        # Pre-norm MLP with layer scale
        residual = hidden_states
        hidden_states = self.norm2(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = op.Mul(hidden_states, self.ls2)
        hidden_states = op.Add(residual, hidden_states)
        return hidden_states


class _IV2Encoder(nn.Module):
    """Stack of InternVideo2 encoder layers."""

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _IV2EncoderLayer(hidden_size, intermediate_size, num_heads, norm_eps)
                for _ in range(num_layers)
            ]
        )

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        for layer in self.layers:
            hidden_states = layer(op, hidden_states)
        return hidden_states


class _IV2Embeddings(nn.Module):
    """InternVideo2 patch embedding with CLS token + position embedding.

    HF weight names::

        patch_embed.proj.weight    [H, 3, 1, P, P]  (Conv3d → squeeze to Conv2d)
        patch_embed.proj.bias      [H]
        cls_token                  [1, 1, H]
        pos_embed                  [1, num_patches+1, H]
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        hidden_size: int,
    ):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        from mobius.components import Conv2d

        self.patch_embedding = Conv2d(
            3,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        # CLS token
        self.cls_token = nn.Parameter([1, 1, hidden_size])
        # Position embedding (includes CLS position)
        self.position_embedding = nn.Parameter(
            [1, self.num_patches + 1, hidden_size],
            name="position_embedding.weight",
        )

    def forward(self, op: OpBuilder, pixel_values: ir.Value):
        patch_embeds = self.patch_embedding(op, pixel_values)
        # [B, H, grid_h, grid_w] → [B, num_patches, H]
        batch = op.Shape(patch_embeds, start=0, end=1)
        hidden = op.Shape(patch_embeds, start=1, end=2)
        flat_shape = op.Concat(batch, hidden, op.Constant(value_ints=[-1]), axis=0)
        patch_embeds = op.Reshape(patch_embeds, flat_shape)
        patch_embeds = op.Transpose(patch_embeds, perm=[0, 2, 1])

        # Prepend CLS token
        cls = op.Expand(
            self.cls_token,
            op.Concat(batch, op.Constant(value_ints=[1]), hidden, axis=0),
        )
        embeddings = op.Concat(cls, patch_embeds, axis=1)
        return op.Add(embeddings, self.position_embedding)


class _InternVideo2VisionModel(nn.Module):
    """InternVideo2 vision model: embeddings + encoder (no post-norm).

    Outputs include CLS token — caller strips it if needed.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        vc = config.vision
        assert vc is not None
        self.embeddings = _IV2Embeddings(
            image_size=vc.image_size or 448,
            patch_size=vc.patch_size or 14,
            hidden_size=vc.hidden_size or 1408,
        )
        self.encoder = _IV2Encoder(
            num_layers=vc.num_hidden_layers or 24,
            hidden_size=vc.hidden_size or 1408,
            intermediate_size=vc.intermediate_size or 6144,
            num_heads=vc.num_attention_heads or 16,
            norm_eps=vc.norm_eps,
        )

    def forward(self, op: OpBuilder, pixel_values: ir.Value):
        hidden_states = self.embeddings(op, pixel_values)
        hidden_states = self.encoder(op, hidden_states)
        # Strip CLS token → [B, num_patches, H]
        return op.Slice(
            hidden_states,
            op.Constant(value_ints=[1]),
            op.Constant(value_ints=[2147483647]),
            op.Constant(value_ints=[1]),
        )


# ── Vision encoder (top-level sub-model) ────────────────────────────────


class _VideoChatFlashVisionEncoderModel(nn.Module):
    """InternVideo2 vision tower + MLP projector.

    The HF model's Token Merging (ToMe) bipartite soft matching step
    reduces token count before projection; this is not included in the
    ONNX graph since it has no learned parameters.

    Weight renaming is handled by
    :meth:`VideoChatFlashModel.preprocess_weights`.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        vc = config.vision
        assert vc is not None, "VisionConfig is required"
        self.vision_tower = _InternVideo2VisionModel(config)
        self.multi_modal_projector = MLPMultiModalProjector(
            vision_hidden_size=vc.hidden_size or config.hidden_size,
            text_hidden_size=config.hidden_size,
        )

    def forward(self, op: OpBuilder, pixel_values: ir.Value):
        vision_features = self.vision_tower(op, pixel_values)
        return self.multi_modal_projector(op, vision_features)


# ── Embedding ───────────────────────────────────────────────────────────


class _VideoChatFlashEmbeddingModel(nn.Module):
    """Token embedding + image feature scatter."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.image_token_id = (config.vision.image_token_id if config.vision else 0) or 0

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        image_features: ir.Value,
    ):
        text_embeds = self.embed_tokens(op, input_ids)

        image_mask = op.Equal(input_ids, op.Constant(value_int=self.image_token_id))
        image_mask_3d = op.Unsqueeze(image_mask, [-1])

        mask_int = op.Cast(image_mask, to=7)
        cumsum = op.CumSum(mask_int, 1)
        indices = op.Sub(cumsum, op.Constant(value_int=1))
        indices = op.Clip(indices, op.Constant(value_int=0))

        pad_row = op.Expand(
            op.CastLike(0.0, image_features),
            op.Concat(
                op.Constant(value_ints=[1]),
                op.Shape(image_features, start=1, end=2),
                axis=0,
            ),
        )
        padded_features = op.Concat(image_features, pad_row, axis=0)

        gathered = op.Gather(padded_features, indices, axis=0)
        return op.Where(image_mask_3d, gathered, text_embeds)


# ── Decoder ─────────────────────────────────────────────────────────────


class _VideoChatFlashDecoderModel(nn.Module):
    """Qwen2.5 text decoder taking inputs_embeds.

    Weight renaming is handled by
    :meth:`VideoChatFlashModel.preprocess_weights`.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.model = TextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states, present_key_values = self.model(
            op,
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values


# ── Top-level model ─────────────────────────────────────────────────────


class VideoChatFlashModel(nn.Module):
    """VideoChat-Flash-Qwen VLM (3-model split).

    Builds three ONNX models for ORT GenAI deployment:

    - **decoder**: Qwen2.5 text decoder taking ``inputs_embeds``
    - **vision_encoder**: InternVideo2 ViT + MLP projector
    - **embedding**: token embedding + image feature fusion
    """

    default_task: str = "vision-language"
    category: str = "Multimodal"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.decoder = _VideoChatFlashDecoderModel(config)
        self.vision_encoder = _VideoChatFlashVisionEncoderModel(config)
        self.embedding = _VideoChatFlashEmbeddingModel(config)

    def forward(self, op: OpBuilder, **kwargs):
        raise NotImplementedError(
            "VideoChatFlashModel uses VisionLanguageTask which calls "
            "each sub-module separately."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # HF uses flat model.* prefix (no language_model wrapper)
        # Route by prefix to sub-models
        result: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            # Strip outer "model." prefix if present
            k = key[len("model.") :] if key.startswith("model.") else key

            if k.startswith("vision_tower.vision_tower."):
                self._route_vision_weight(k, value, result)
            elif k.startswith("mm_projector."):
                self._route_projector_weight(k, value, result)
            elif k.startswith(("layers.", "embed_tokens.", "norm.")):
                self._route_decoder_weight(k, value, result)
            elif k == "lm_head.weight" or key == "lm_head.weight":
                result["decoder.lm_head.weight"] = value

        # Weight tying: embed_tokens → lm_head
        if self.config.tie_word_embeddings:
            embed_key = "decoder.model.embed_tokens.weight"
            head_key = "decoder.lm_head.weight"
            if head_key not in result and embed_key in result:
                result[head_key] = result[embed_key]

        return result

    def _route_vision_weight(
        self,
        key: str,
        value: torch.Tensor,
        result: dict[str, torch.Tensor],
    ) -> None:
        """Route vision_tower.vision_tower.* to vision_encoder.vision_tower.

        The new InternVideo2 model structure matches HF names closely:
        - blocks.N.{attn,mlp,norm1,norm2,ls1,ls2} → encoder.layers.N.*
        - patch_embed.proj → embeddings.patch_embedding (Conv3d→Conv2d)
        - pos_embed → embeddings.position_embedding.weight
        - cls_token → embeddings.cls_token
        """
        suffix = key[len("vision_tower.vision_tower.") :]

        # patch_embed.proj: Conv3d [out, in, 1, H, W] → Conv2d
        if suffix.startswith("patch_embed.proj."):
            param = suffix[len("patch_embed.proj.") :]
            result[f"vision_encoder.vision_tower.embeddings.patch_embedding.{param}"] = (
                value.squeeze(2) if value.dim() == 5 else value
            )
            return

        # pos_embed: [1, num_patches+1, H] → keep as-is (our model has CLS)
        if suffix == "pos_embed":
            result["vision_encoder.vision_tower.embeddings.position_embedding.weight"] = value
            return

        # cls_token: [1, 1, H]
        if suffix == "cls_token":
            result["vision_encoder.vision_tower.embeddings.cls_token"] = value
            return

        # img_pos_embed: extra temporal pos embed — skip
        if suffix == "img_pos_embed":
            return

        # Encoder blocks: blocks.N.* → encoder.layers.N.*
        new_key = suffix.replace("blocks.", "encoder.layers.")
        # LayerScale: HF safetensors uses ls1.gamma/ls2.gamma → ls1.weight/ls2.weight
        new_key = new_key.replace(".ls1.gamma", ".ls1.weight")
        new_key = new_key.replace(".ls2.gamma", ".ls2.weight")
        # mlp.fc1/fc2 stays (our _IV2MLP has these)

        result[f"vision_encoder.vision_tower.{new_key}"] = value

    def _route_projector_weight(
        self,
        key: str,
        value: torch.Tensor,
        result: dict[str, torch.Tensor],
    ) -> None:
        """Route mm_projector.* to vision_encoder.multi_modal_projector."""
        # mm_projector.mlp.0 → multi_modal_projector.linear_1
        # mm_projector.mlp.2 → multi_modal_projector.linear_2
        suffix = key[len("mm_projector.") :]
        suffix = suffix.replace("mlp.0.", "linear_1.")
        suffix = suffix.replace("mlp.2.", "linear_2.")
        result[f"vision_encoder.multi_modal_projector.{suffix}"] = value

    def _route_decoder_weight(
        self,
        key: str,
        value: torch.Tensor,
        result: dict[str, torch.Tensor],
    ) -> None:
        """Route decoder weights (flat prefix, no language_model wrapper)."""
        # Duplicate embed_tokens to embedding sub-model
        if "embed_tokens" in key:
            result[f"embedding.{key}"] = value

        # Decoder: model.layers.* → decoder.model.layers.*
        result[f"decoder.model.{key}"] = value
