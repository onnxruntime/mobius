# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""NVIDIA Nemotron Parse vision-language encoder-decoder model.

Replicates HuggingFace's ``NemotronParseForConditionalGeneration`` with a
C-RADIOv2-H vision encoder, convolutional feature neck, and a cross-attentive
mBART text decoder. The architecture is exported as separate
``vision_encoder`` and ``decoder`` ONNX models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components import (
    Conv2dNoBias,
    Embedding,
    EncoderDecoderAttention,
    LayerNorm,
    Linear,
    RadioVisionModel,
    create_padding_mask,
)

if TYPE_CHECKING:
    import onnx_ir as ir


class _RadioModel(nn.Module):
    """Wrapper matching ``radio_model.model`` in the HuggingFace checkpoint."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.model = RadioVisionModel(
            image_height=config.image_height,
            image_width=config.image_width,
            patch_size=config.vision.patch_size,
            max_grid_size=config.vision_max_grid_size,
            hidden_size=config.vision.hidden_size,
            intermediate_size=config.vision.intermediate_size,
            num_layers=config.vision.num_hidden_layers,
            num_heads=config.vision.num_attention_heads,
            norm_eps=config.vision.norm_eps,
        )

    def forward(self, op: OpBuilder, pixel_values: ir.Value):
        return self.model(op, pixel_values)


class _RadioEncoder(nn.Module):
    """Wrapper matching the checkpoint's ``model_encoder`` module."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.radio_model = _RadioModel(config)

    def forward(self, op: OpBuilder, pixel_values: ir.Value):
        return self.radio_model(op, pixel_values)


class NemotronParseVisionEncoder(nn.Module):
    """C-RADIO vision encoder and Nemotron Parse feature-compression neck."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.model_encoder = _RadioEncoder(config)
        self.conv1 = Linear(config.vision.hidden_size, config.hidden_size)
        self.layer_norm1 = LayerNorm(config.hidden_size, eps=1e-6)
        self.conv2 = Conv2dNoBias(
            config.hidden_size,
            config.hidden_size,
            kernel_size=(1, 4),
            stride=(1, 4),
        )
        self.layer_norm2 = LayerNorm(config.hidden_size, eps=1e-6)
        self.sum_proj = Linear(
            config.num_summary_tokens * config.vision.hidden_size,
            config.hidden_size,
        )
        self.layer_norm3 = LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, op: OpBuilder, pixel_values: ir.Value):
        # C-RADIO returns three flattened teacher summaries plus the spatial
        # patch sequence: (B, 3*1280), (B, H/16*W/16, 1280).
        summary, features = self.model_encoder(op, pixel_values)

        # Pointwise projection and normalization: (B, H/16*W/16, 1024).
        features = self.conv1(op, features)
        features = self.layer_norm1(op, features)

        # Restore the patch grid, compress every four horizontal patches, then
        # flatten back to a sequence: (B, H/16*W/64, 1024).
        batch = op.Shape(features, start=0, end=1)
        grid_height = self.config.image_height // self.config.vision.patch_size
        grid_width = self.config.image_width // self.config.vision.patch_size
        grid_shape = op.Concat(
            batch,
            op.Constant(value_ints=[grid_height, grid_width, self.config.hidden_size]),
            axis=0,
        )
        features = op.Reshape(features, grid_shape)
        features = op.Transpose(features, perm=[0, 3, 1, 2])
        features = self.conv2(op, features)
        features = op.Transpose(features, perm=[0, 2, 3, 1])
        sequence_shape = op.Concat(
            batch,
            op.Constant(value_ints=[
                grid_height * (grid_width // 4),
                self.config.hidden_size,
            ]),
            axis=0,
        )
        features = op.Reshape(features, sequence_shape)
        features = self.layer_norm2(op, features)

        # Project the concatenated teacher summaries to one final visual token.
        summary = self.sum_proj(op, summary)  # (B, 1024)
        summary = self.layer_norm3(op, summary)
        summary = op.Unsqueeze(summary, [1])  # (B, 1, 1024)
        return op.Concat(features, summary, axis=1)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Select vision weights and reshape the checkpoint's Conv1d kernel."""
        weights: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if not key.startswith("encoder."):
                continue
            key = key.removeprefix("encoder.")
            if ".input_conditioner." in key or key.endswith(".summary_idxs"):
                continue
            if key == "conv1.weight":
                value = value.squeeze(-1)
            weights[key] = value
        return weights


class _NemotronParseDecoderLayer(nn.Module):
    """Pre-norm mBART decoder layer with self- and cross-attention."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.self_attn = EncoderDecoderAttention(config, is_causal=True)
        self.self_attn_layer_norm = LayerNorm(config.hidden_size, eps=1e-5)
        self.encoder_attn = EncoderDecoderAttention(config)
        self.encoder_attn_layer_norm = LayerNorm(config.hidden_size, eps=1e-5)
        self.fc1 = Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = Linear(config.intermediate_size, config.hidden_size)
        self.final_layer_norm = LayerNorm(config.hidden_size, eps=1e-5)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        encoder_hidden_states: ir.Value,
        self_attention_bias: ir.Value,
        past_key_value: tuple[ir.Value, ir.Value] | None = None,
    ):
        # Pre-norm causal self-attention.
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(op, hidden_states)
        hidden_states, self_kv = self.self_attn(
            op,
            hidden_states,
            attention_bias=self_attention_bias,
            past_key_value=past_key_value,
        )
        hidden_states = op.Add(residual, hidden_states)

        # Pre-norm cross-attention over the compressed C-RADIO sequence.
        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(op, hidden_states)
        hidden_states, _ = self.encoder_attn(
            op, hidden_states, key_value_states=encoder_hidden_states
        )
        hidden_states = op.Add(residual, hidden_states)

        # Pre-norm exact-GELU feed-forward block.
        residual = hidden_states
        hidden_states = self.final_layer_norm(op, hidden_states)
        hidden_states = self.fc1(op, hidden_states)
        hidden_states = op.Gelu(hidden_states)
        hidden_states = self.fc2(op, hidden_states)
        return op.Add(residual, hidden_states), self_kv


class NemotronParseDecoder(nn.Module):
    """Position-free scaled-embedding mBART decoder used by Nemotron Parse."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layernorm_embedding = LayerNorm(config.hidden_size, eps=1e-5)
        self.layers = nn.ModuleList(
            [_NemotronParseDecoderLayer(config) for _ in range(config.num_decoder_layers)]
        )
        self.layer_norm = LayerNorm(config.hidden_size, eps=1e-5)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        self._embedding_scale = float(config.hidden_size**0.5)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        encoder_hidden_states: ir.Value,
        past_key_values: list[tuple[ir.Value, ir.Value]] | None = None,
    ):
        hidden_states = self.embed_tokens(op, input_ids)
        embedding_scale = op.CastLike(
            op.Constant(value_float=self._embedding_scale), hidden_states
        )
        hidden_states = op.Mul(
            hidden_states,
            embedding_scale,
        )
        hidden_states = self.layernorm_embedding(op, hidden_states)
        self_attention_bias = create_padding_mask(op, input_ids, attention_mask)

        present_self_kvs = []
        layer_past = past_key_values or [None] * len(self.layers)
        for layer, past_key_value in zip(self.layers, layer_past):
            hidden_states, self_kv = layer(
                op,
                hidden_states,
                encoder_hidden_states,
                self_attention_bias,
                past_key_value=past_key_value,
            )
            present_self_kvs.append(self_kv)

        hidden_states = self.layer_norm(op, hidden_states)
        logits = self.lm_head(op, hidden_states)
        return logits, present_self_kvs

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Select decoder weights and materialize the tied language-model head."""
        weights = {
            key.removeprefix("decoder."): value
            for key, value in state_dict.items()
            if key.startswith("decoder.")
            and ".extra_heads." not in key
            and ".extra_proj." not in key
        }
        embed_weight = weights.get("embed_tokens.weight")
        if embed_weight is not None:
            weights["lm_head.weight"] = embed_weight
        return weights


class NemotronParseForConditionalGeneration(nn.Module):
    """NVIDIA Nemotron Parse OCR/document parser with C-RADIO and mBART."""

    default_task = "vision-encoder-decoder"
    category: str = "Multimodal"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.vision_encoder = NemotronParseVisionEncoder(config)
        self.decoder = NemotronParseDecoder(config)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map the HuggingFace checkpoint to the two exported ONNX models."""
        weights = {
            key: value
            for key, value in state_dict.items()
            if key.startswith("vision_encoder.")
        }
        weights.update(
            {
                f"vision_encoder.{key}": value
                for key, value in self.vision_encoder.preprocess_weights(state_dict).items()
            }
        )
        weights.update(
            {
                f"decoder.{key}": value
                for key, value in self.decoder.preprocess_weights(state_dict).items()
            }
        )
        return weights
