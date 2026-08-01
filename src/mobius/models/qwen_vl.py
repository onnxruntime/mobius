# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius._weight_utils import preprocess_olive_weights, tie_word_embeddings
from mobius.components import (
    InputMixer,
    Qwen2VLVisionModel,
    Qwen3VLVisionModel,
    Qwen25VLVisionModel,
)
from mobius.components._common import (
    Embedding,
    Linear,
    create_attention_bias,
)
from mobius.components._decoder import DecoderLayer
from mobius.components._rms_norm import RMSNorm
from mobius.components._rotary_embedding import initialize_rope
from mobius.models.base import CausalLMModel, TextModel

if TYPE_CHECKING:
    import onnx_ir as ir

# Text-only decoders — extract the language model from multimodal weights.
# These strip ``language_model.`` prefixes and drop ``visual.`` keys.


class _QwenVLTextMixin:
    """Shared weight preprocessing for Qwen VL text decoders."""

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        for key in list(state_dict.keys()):
            if "language_model." in key:
                new_key = key.replace("language_model.", "")
                state_dict[new_key] = state_dict.pop(key)
            elif "visual." in key:
                state_dict.pop(key)
        return super().preprocess_weights(state_dict)


class Qwen25VLTextModel(_QwenVLTextMixin, CausalLMModel):
    """Qwen2.5-VL text-only decoder.

    Extracts the text backbone from the Qwen2.5-VL multimodal model.
    Strips ``language_model.`` weight prefixes and drops ``visual.`` keys.
    For text-only inference the standard 1D RoPE is equivalent to MRoPE
    (all three dimensions are identical for text tokens).
    """


class Qwen3VLTextModel(_QwenVLTextMixin, CausalLMModel):
    """Qwen3-VL text-only decoder.

    Extends Qwen2.5-VL text decoder with Q/K normalization (RMSNorm on
    query and key projections), configured via ``attn_qk_norm=True``.
    """


# Full VL models — multimodal (text + vision).


class Qwen25VLCausalLMModel(nn.Module):
    """Qwen2.5-VL vision-language model (3-model split).

    Builds three separate ONNX models for onnxruntime-genai:

    - ``decoder``: text decoder taking ``inputs_embeds`` (MRoPE position_ids)
    - ``vision_encoder``: vision ViT with windowed/full attention
    - ``embedding``: token embedding + image feature fusion

    The :class:`~mobius.tasks.Qwen25VL3ModelTask` calls each
    sub-module separately to produce 3 ONNX graphs.
    """

    default_task: str = "qwen-vl"
    category: str = "Multimodal"
    config_class: type = ArchitectureConfig

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.decoder = Qwen25VLDecoderModel(config)
        self.vision_encoder = Qwen25VLVisionEncoderModel(config)
        self.embedding = Qwen25VLEmbeddingModel(config)

    def forward(self, op: OpBuilder, **kwargs):
        raise NotImplementedError(
            "Qwen25VLCausalLMModel uses Qwen25VL3ModelTask which calls "
            "each sub-module (decoder, vision_encoder, embedding) separately."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Route HF weights to the correct sub-model ONNX initializer names.

        ONNX initializer names include the composite module's attribute
        prefixes (``decoder.``, ``vision_encoder.``, ``embedding.``) because
        onnxscript qualifies parameter names via the parent-child hierarchy.
        """
        quantization = self.config.quantization
        if quantization is not None and quantization.quant_method == "olive":
            state_dict = preprocess_olive_weights(
                state_dict,
                bits=quantization.bits,
                group_size=quantization.group_size,
                quantize_embeddings=quantization.quantize_embeddings,
                quantize_lm_head=quantization.quantize_lm_head,
                tie_word_embeddings=self.config.tie_word_embeddings
                or quantization.tie_word_embeddings,
            )

        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("visual."):
                new_key = f"vision_encoder.{key}"
                # Merger uses mlp_0/mlp_2 attributes; HF uses mlp.0/mlp.2
                new_key = new_key.replace(".merger.mlp.0.", ".merger.mlp_0.")
                new_key = new_key.replace(".merger.mlp.2.", ".merger.mlp_2.")
                renamed[new_key] = value
            elif key.startswith("model.embed_tokens."):
                # Shared embedding → both decoder (TextModel) and embedding model
                renamed[f"decoder.{key}"] = value
                stripped = key[len("model.") :]
                renamed[f"embedding.{stripped}"] = value
            elif key.startswith("model."):
                renamed[f"decoder.{key}"] = value
            elif key.startswith("lm_head."):
                if not self.config.tie_word_embeddings:
                    renamed[f"decoder.{key}"] = value
        if self.config.tie_word_embeddings:
            # onnxscript qualifies params by module path, so the in-tree
            # alias set in __init__ does not cross composite module
            # boundaries.  Establish tensor identity here so apply_weights
            # sees a single data_ptr() across both initializers.
            tie_word_embeddings(
                renamed,
                embed_key="decoder.model.embed_tokens.weight",
                head_key="decoder.lm_head.weight",
            )
        return renamed


class Qwen25VLDecoderModel(nn.Module):
    """Qwen2.5-VL text decoder that takes ``inputs_embeds`` instead of ``input_ids``.

    This is the decoder component of the 3-model split for onnxruntime-genai.
    It receives pre-computed embeddings (from the embedding model) and uses
    MRoPE with 3D ``position_ids`` of shape ``(3, batch, seq_len)``.

    Weight prefix: ``language_model.``
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.model = TextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

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

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Route language_model weights for standalone decoder build.

        When built standalone (not via composite), the decoder ONNX
        initializers are ``model.layers.*``, ``model.norm.*``,
        ``model.embed_tokens.*``, ``lm_head.*``.  HF keys already match.
        """
        quantization = self.config.quantization
        if quantization is not None and quantization.quant_method == "olive":
            state_dict = preprocess_olive_weights(
                state_dict,
                bits=quantization.bits,
                group_size=quantization.group_size,
                quantize_embeddings=quantization.quantize_embeddings,
                quantize_lm_head=quantization.quantize_lm_head,
                tie_word_embeddings=self.config.tie_word_embeddings
                or quantization.tie_word_embeddings,
            )

        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            # Drop vision weights
            if key.startswith("visual."):
                continue
            renamed[key] = value

        # Establish tensor identity for tied weights so apply_weights
        # detects shared data_ptr() and emits a single ONNX initializer.
        if self.config.tie_word_embeddings:
            tie_word_embeddings(renamed)
        return renamed


class Qwen25VLVisionEncoderModel(nn.Module):
    """Qwen2.5-VL vision encoder for the 3-model split.

    Processes image patches through ViT with windowed/full attention,
    spatial merge, and outputs image features.

    Inputs:
        - pixel_values: (total_patches, C*T_p*P*P) — flattened patches
        - image_grid_thw: (num_images, 3) INT64 — [T, H, W] per image
    Output:
        - image_features: (num_merged_patches, out_hidden_size)

    Weight prefix: ``visual.``
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        vc = config.vision
        assert vc is not None and vc.hidden_size is not None
        assert vc.num_hidden_layers is not None
        assert vc.num_attention_heads is not None
        self.visual = Qwen25VLVisionModel(
            depth=vc.num_hidden_layers,
            hidden_size=vc.hidden_size,
            intermediate_size=vc.intermediate_size or vc.hidden_size * 4,
            num_heads=vc.num_attention_heads,
            patch_size=vc.patch_size or 14,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=vc.in_channels,
            out_hidden_size=vc.out_hidden_size or config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
            fullatt_block_indexes=config.fullatt_block_indexes,
            window_size=config.window_size,
        )

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        image_grid_thw: ir.Value,
    ):
        image_features = self.visual(
            op,
            pixel_values,
            image_grid_thw,
        )
        return image_features

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Keep only visual.* weights for standalone vision encoder build.

        Also maps merger ``mlp.0``/``mlp.2`` → ``mlp_0``/``mlp_2``.
        """
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("visual."):
                new_key = key.replace(".merger.mlp.0.", ".merger.mlp_0.")
                new_key = new_key.replace(".merger.mlp.2.", ".merger.mlp_2.")
                renamed[new_key] = value
        return renamed


class Qwen25VLEmbeddingModel(nn.Module):
    """Qwen2.5-VL embedding model for the 3-model split.

    Fuses token embeddings with image features at image token positions.

    Inputs:
        - input_ids: (batch, seq_len) INT64
        - image_features: (num_image_tokens, hidden_size) FLOAT
    Output:
        - inputs_embeds: (batch, seq_len, hidden_size) FLOAT
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.image_token_id = config.image_token_id or 151655

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        image_features: ir.Value,
    ):
        # Token embedding lookup
        text_embeds = self.embed_tokens(op, input_ids)

        # Create mask for image token positions
        image_mask = op.Equal(
            input_ids,
            op.Constant(value_int=self.image_token_id),
        )
        # Expand mask to 3D for broadcasting: (batch, seq, 1)
        image_mask_3d = op.Unsqueeze(image_mask, [-1])

        # Cumulative sum to map flat image_features indices
        # image_mask is (batch, seq), cast to int
        mask_int = op.Cast(image_mask, to=7)  # INT64
        cumsum = op.CumSum(mask_int, op.Constant(value_int=1))
        # Zero-based index: subtract 1, clip to 0
        indices = op.Sub(cumsum, op.Constant(value_int=1))
        indices = op.Clip(indices, op.Constant(value_int=0))

        # Pad image_features with one zero row so Gather is valid even when
        # image_features is empty (text-only input: num_image_tokens == 0).
        # The Where mask ensures the padding row is never used in the output.
        pad_row = op.Expand(
            op.CastLike(0.0, image_features),
            op.Concat(
                op.Constant(value_ints=[1]),
                op.Shape(image_features, start=1, end=2),
                axis=0,
            ),
        )
        padded_features = op.Concat(image_features, pad_row, axis=0)

        # Gather from padded_features using indices
        # padded_features: (num_image_tokens + 1, hidden)
        # indices: (batch, seq) → gather → (batch, seq, hidden)
        gathered = op.Gather(padded_features, indices, axis=0)

        # Where image_mask → use gathered features, else text_embeds
        inputs_embeds = op.Where(image_mask_3d, gathered, text_embeds)

        return inputs_embeds

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Keep only embed_tokens weights for standalone embedding build.

        HF key ``model.embed_tokens.weight`` → ``embed_tokens.weight``.
        """
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if "embed_tokens" in key:
                new_key = key
                if new_key.startswith("model."):
                    new_key = new_key[len("model.") :]
                renamed[new_key] = value
        return renamed


class Qwen2VLVisionEncoderModel(Qwen25VLVisionEncoderModel):
    """Qwen2-VL vision encoder for the 3-model split.

    Uses Qwen2VLVisionModel (LayerNorm + FCMLP) instead of Qwen2.5-VL's
    RMSNorm + GatedMLP.  All attention blocks are full (no windowing).
    """

    def __init__(self, config: ArchitectureConfig):
        # Skip Qwen25VLVisionEncoderModel.__init__ to use our own model
        nn.Module.__init__(self)
        vc = config.vision
        assert vc is not None and vc.hidden_size is not None
        assert vc.num_hidden_layers is not None
        assert vc.num_attention_heads is not None
        self.visual = Qwen2VLVisionModel(
            depth=vc.num_hidden_layers,
            hidden_size=vc.hidden_size,
            intermediate_size=vc.intermediate_size or vc.hidden_size * 4,
            num_heads=vc.num_attention_heads,
            patch_size=vc.patch_size or 14,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=vc.in_channels,
            out_hidden_size=vc.out_hidden_size or config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Keep only visual.* weights and rename fc1/fc2 → up_proj/down_proj."""
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("visual."):
                new_key = key
                new_key = new_key.replace(".mlp.fc1.", ".mlp.up_proj.")
                new_key = new_key.replace(".mlp.fc2.", ".mlp.down_proj.")
                new_key = new_key.replace(".merger.mlp.0.", ".merger.mlp_0.")
                new_key = new_key.replace(".merger.mlp.2.", ".merger.mlp_2.")
                renamed[new_key] = value
        return renamed


class Qwen2VLCausalLMModel(nn.Module):
    """Qwen2-VL vision-language model (3-model split).

    Same 3-model architecture as Qwen2.5-VL but with:
    - Qwen2VLVisionModel (LayerNorm + FCMLP, no windowed attention)
    - Same text decoder and embedding model
    """

    default_task: str = "qwen-vl"
    category: str = "Multimodal"
    config_class: type = ArchitectureConfig

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.decoder = Qwen25VLDecoderModel(config)
        self.vision_encoder = Qwen2VLVisionEncoderModel(config)
        self.embedding = Qwen25VLEmbeddingModel(config)

    def forward(self, op: OpBuilder, **kwargs):
        raise NotImplementedError(
            "Qwen2VLCausalLMModel uses Qwen25VL3ModelTask which calls "
            "each sub-module (decoder, vision_encoder, embedding) "
            "separately."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Route HF weights to the correct sub-model ONNX initializer names.

        Vision weights get fc1→up_proj, fc2→down_proj renames for FCMLP.
        """
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("visual."):
                new_key = f"vision_encoder.{key}"
                new_key = new_key.replace(".mlp.fc1.", ".mlp.up_proj.")
                new_key = new_key.replace(".mlp.fc2.", ".mlp.down_proj.")
                new_key = new_key.replace(".merger.mlp.0.", ".merger.mlp_0.")
                new_key = new_key.replace(".merger.mlp.2.", ".merger.mlp_2.")
                renamed[new_key] = value
            elif key.startswith("model.embed_tokens."):
                renamed[f"decoder.{key}"] = value
                stripped = key[len("model.") :]
                renamed[f"embedding.{stripped}"] = value
                if self.config.tie_word_embeddings and key == "model.embed_tokens.weight":
                    renamed["decoder.lm_head.weight"] = value
            elif key.startswith("model."):
                renamed[f"decoder.{key}"] = value
            elif key.startswith("lm_head."):
                renamed[f"decoder.{key}"] = value
        return renamed


class _Qwen3VLTextModel(nn.Module):
    """Qwen3-VL text model with DeepStack injection and MRoPE.

    After specific decoder layers, adds visual features from DeepStack at
    positions corresponding to image tokens.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype
        self.image_token_id = config.image_token_id or 0
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)
        self.input_mixer = InputMixer(
            image_token_id=config.image_token_id or 0,
        )

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        inputs_embeds: ir.Value | None = None,
        vision_embeddings: ir.Value | None = None,
        deepstack_visual_embeds: list | None = None,
    ):
        if vision_embeddings is not None and inputs_embeds is None:
            text_embeddings = self.embed_tokens(op, input_ids)
            hidden_states = self.input_mixer(
                op,
                text_embeddings,
                vision_embeddings,
                input_ids,
            )
        elif inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(op, input_ids)
        position_embeddings = self.rotary_emb(op, position_ids)

        attention_bias = create_attention_bias(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )

        # Visual position mask for DeepStack injection
        if deepstack_visual_embeds is not None:
            visual_mask = op.Equal(
                input_ids,
                op.Constant(value_int=self.image_token_id),
            )
            # Expand to [batch, seq, 1] for broadcasting
            visual_mask_3d = op.Unsqueeze(visual_mask, [-1])

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer_idx, (layer, past_kv) in enumerate(zip(self.layers, past_kvs)):
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

            # DeepStack: add intermediate vision features at image positions
            if deepstack_visual_embeds is not None and layer_idx < len(
                deepstack_visual_embeds
            ):
                ds_embeds = deepstack_visual_embeds[layer_idx]
                # Scatter vision features at visual token positions
                # ds_embeds: (num_visual_tokens, hidden_size)
                # Use cumsum of mask to index into ds_embeds
                mask_int = op.Cast(visual_mask, to=7)  # INT64
                cumsum = op.CumSum(mask_int, op.Constant(value_int=1))
                indices = op.Sub(cumsum, op.Constant(value_int=1))
                indices = op.Clip(indices, op.Constant(value_int=0))
                # Expand ds_embeds with batch dim: (1, num_visual_tokens, hidden_size)
                ds_embeds_3d = op.Unsqueeze(ds_embeds, [0])
                # Gather at computed indices
                indices_3d = op.Unsqueeze(indices, [-1])
                hidden_dim = op.Shape(ds_embeds_3d, start=2, end=3)
                ones_shape = op.Concat(
                    op.Constant(value_ints=[1, 1]),
                    hidden_dim,
                    axis=0,
                )
                gather_idx = op.Expand(indices_3d, ones_shape)
                scattered_ds = op.GatherElements(ds_embeds_3d, gather_idx, axis=1)
                # Add at visual positions only
                hidden_states = op.Add(
                    hidden_states,
                    op.Where(
                        visual_mask_3d,
                        scattered_ds,
                        op.CastLike(0.0, hidden_states),
                    ),
                )

        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values


class _Qwen3VLForMultimodalLM(nn.Module):
    """Qwen3-VL causal LM with DeepStack-aware text decoder.

    Accepts ``vision_embeddings`` for input mixing and
    ``deepstack_visual_embeds`` for intermediate layer injection.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.model = _Qwen3VLTextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        vision_embeddings: ir.Value | None = None,
        deepstack_visual_embeds: list | None = None,
    ):
        hidden_states, present_key_values = self.model(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            vision_embeddings=vision_embeddings,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values


class Qwen3VLCausalLMModel(nn.Module):
    """Qwen3-VL vision-language model with packed-attention vision encoder.

    Combines a Qwen3-VL ViT vision encoder with a Qwen3-VL text decoder.
    The vision encoder processes packed image/video patches through Conv3d
    embedding, rotary-embedded transformer blocks with packed attention,
    spatial merge, and DeepStack intermediate feature extraction.

    The text decoder uses interleaved MRoPE for 3D positional encoding
    (temporal, height, width) and injects DeepStack vision features into
    early decoder layers at image token positions.

    Weight names match HuggingFace convention:
    ``visual.*`` for vision encoder, ``language_model.*`` for text decoder.
    """

    default_task: str = "qwen3-vl-vision-language"
    category: str = "Multimodal"

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config

        # Vision encoder — uses ``visual.*`` weight prefix
        vc = config.vision
        assert vc is not None and vc.hidden_size is not None
        assert vc.num_hidden_layers is not None
        assert vc.num_attention_heads is not None
        self.visual = Qwen3VLVisionModel(
            depth=vc.num_hidden_layers,
            hidden_size=vc.hidden_size,
            intermediate_size=vc.intermediate_size or vc.hidden_size * 4,
            num_heads=vc.num_attention_heads,
            patch_size=vc.patch_size or 16,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=vc.in_channels,
            out_hidden_size=vc.out_hidden_size or config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
            num_position_embeddings=vc.num_position_embeddings or 2304,
            deepstack_visual_indexes=config.deepstack_visual_indexes or [],
        )

        # Text decoder — uses ``language_model.*`` weight prefix
        self.language_model = _Qwen3VLForMultimodalLM(config)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        pixel_values: ir.Value,
        grid_thw: ir.Value,
        past_key_values: list | None = None,
    ):
        """Full vision-language forward pass.

        Args:
            input_ids: Text input token IDs ``(batch, seq_len)``.
            attention_mask: ``(batch, past_seq_len + seq_len)``.
            position_ids: MRoPE 3D positions ``(3, batch, seq_len)``.
            pixel_values: Flattened image patches
                ``(total_patches, C * T_p * P * P)``.
            grid_thw: ``(num_images, 3)`` INT64 with ``[T, H, W]`` per
                image, used for position embedding interpolation and
                computing cu_seqlens and rotary position IDs.
            past_key_values: Optional KV cache.

        Returns:
            Tuple of ``(logits, present_key_values)``.
        """
        # Vision encoding
        vision_outputs = self.visual(
            op,
            hidden_states=pixel_values,
            grid_thw=grid_thw,
        )

        # Separate merged features from deepstack features
        if isinstance(vision_outputs, tuple):
            vision_embeddings = vision_outputs[0]
            deepstack_features = list(vision_outputs[1:]) if len(vision_outputs) > 1 else None
        else:
            vision_embeddings = vision_outputs
            deepstack_features = None

        # Add batch dim to vision embeddings for InputMixer:
        # (num_merged_tokens, hidden) → (1, num_merged_tokens, hidden)
        vision_embeddings = op.Unsqueeze(vision_embeddings, [0])

        logits, present_key_values = self.language_model(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            vision_embeddings=vision_embeddings,
            deepstack_visual_embeds=deepstack_features,
        )
        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HuggingFace weight names to the ONNX model's parameter names.

        HF keys use ``model.`` prefix and flatten the language model:
        ``model.language_model.layers.*``, ``model.visual.*``.
        Our model uses ``language_model.model.layers.*``, ``visual.*``.
        """
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = key
            # Strip outer ``model.`` prefix
            if new_key.startswith("model."):
                new_key = new_key[len("model.") :]
            # Vision transformer blocks expose Qwen3's ``linear_fc1/fc2``
            # weights through the generic MLP's ``up_proj/down_proj`` names.
            # Merger modules deliberately keep their native ``linear_fc*``
            # names and do not contain the ``.mlp.`` segment.
            if new_key.startswith("visual."):
                new_key = new_key.replace(".mlp.linear_fc1.", ".mlp.up_proj.")
                new_key = new_key.replace(".mlp.linear_fc2.", ".mlp.down_proj.")
            # HF flattens language_model.model → language_model; restore it
            if new_key.startswith("language_model.") and not new_key.startswith(
                "language_model.lm_head"
            ):
                new_key = new_key.replace("language_model.", "language_model.model.", 1)
            renamed[new_key] = value

        # Single-model composite: lm_head shares the embed_tokens initializer
        # at graph level (onnxscript ties them in __init__). Discard the
        # checkpoint's separate lm_head entry to avoid a duplicate.
        config = self.config
        if config.tie_word_embeddings:
            renamed.pop("language_model.lm_head.weight", None)
        return renamed


# ---------------------------------------------------------------------------
# Qwen3-VL 3-model split for onnxruntime-genai
# ---------------------------------------------------------------------------


class Qwen3VL3ModelCausalLMModel(nn.Module):
    """Qwen3-VL vision-language model (3-model split), with DeepStack.

    Builds three separate ONNX models for onnxruntime-genai:

    - ``decoder``: text decoder taking ``inputs_embeds`` (interleaved MRoPE)
      plus an optional packed ``per_layer_inputs`` DeepStack tensor
    - ``vision_encoder``: packed-attention ViT outputting merged features
      plus ``D`` DeepStack intermediate features
    - ``embedding``: token/image/video feature fusion, producing
      ``inputs_embeds`` and the packed ``per_layer_inputs`` DeepStack tensor

    ``D`` (the number of DeepStack features) is derived from
    ``len(config.deepstack_visual_indexes)`` and is 0 for configs that don't
    set it, in which case this behaves exactly like the pre-DeepStack
    3-model split. Built via :class:`~mobius.tasks.Qwen3VLDeepStackTask`.
    """

    default_task: str = "qwen3-vl-deepstack"
    category: str = "Multimodal"
    config_class: type = ArchitectureConfig

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.decoder = Qwen3VLDecoderModel(config)
        self.vision_encoder = Qwen3VLDeepStackVisionEncoderModel(config)
        self.embedding = Qwen3VLDeepStackEmbeddingModel(config)

    def forward(self, op: OpBuilder, **kwargs):
        raise NotImplementedError(
            "Qwen3VL3ModelCausalLMModel uses QwenVLTask "
            "which calls each sub-module separately."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Route HF weights to the correct sub-model ONNX initializer names.

        ONNX initializer names include composite attribute prefixes
        (``decoder.``, ``vision_encoder.``, ``embedding.``).

        HF keys: ``model.visual.*``, ``model.language_model.*``.
        """
        quantization = self.config.quantization
        if quantization is not None and quantization.quant_method == "olive":
            state_dict = preprocess_olive_weights(
                state_dict,
                bits=quantization.bits,
                group_size=quantization.group_size,
                quantize_embeddings=quantization.quantize_embeddings,
                quantize_lm_head=quantization.quantize_lm_head,
                tie_word_embeddings=self.config.tie_word_embeddings
                or quantization.tie_word_embeddings,
            )

        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            stripped = key
            if stripped.startswith("model."):
                stripped = stripped[len("model.") :]

            if stripped.startswith("visual."):
                # Qwen3-VL uses linear_fc1/fc2; ONNX uses up_proj/down_proj
                stripped = stripped.replace(".mlp.linear_fc1.", ".mlp.up_proj.")
                stripped = stripped.replace(".mlp.linear_fc2.", ".mlp.down_proj.")
                renamed[f"vision_encoder.{stripped}"] = value
            elif stripped.startswith("language_model.embed_tokens."):
                # Shared embedding → both decoder and embedding model
                suffix = stripped[len("language_model.") :]
                renamed[f"decoder.model.{suffix}"] = value
                renamed[f"embedding.{suffix}"] = value
            elif stripped.startswith("language_model.lm_head."):
                if not self.config.tie_word_embeddings:
                    renamed[f"decoder.{stripped[len('language_model.') :]}"] = value
            elif stripped.startswith("language_model."):
                # language_model.layers.* → decoder.model.layers.*
                suffix = stripped[len("language_model.") :]
                renamed[f"decoder.model.{suffix}"] = value
        if self.config.tie_word_embeddings:
            # onnxscript qualifies params by module path, so the in-tree
            # alias set in __init__ does not cross composite module
            # boundaries.  Establish tensor identity here so apply_weights
            # sees a single data_ptr() across both initializers.
            tie_word_embeddings(
                renamed,
                embed_key="decoder.model.embed_tokens.weight",
                head_key="decoder.lm_head.weight",
            )
        return renamed


class Qwen3VLDecoderModel(nn.Module):
    """Qwen3-VL text decoder taking ``inputs_embeds`` (3-model split).

    Uses interleaved MRoPE with 3D ``position_ids`` of shape
    ``(3, batch, seq_len)``. QK normalization is enabled.

    When ``config.deepstack_visual_indexes`` is non-empty, also accepts an
    optional ``per_layer_inputs [B, S, D*H]`` tensor (produced by
    :class:`Qwen3VLDeepStackEmbeddingModel`) that is unpacked into ``D``
    ``[B, S, H]`` slices and added to the hidden state after each of the
    first ``D`` decoder layers — equivalent to HF's
    ``hidden_states[visual_pos_masks] += deepstack_visual_embeds[layer_idx]``.
    ``D`` is derived from ``len(config.deepstack_visual_indexes)``, never
    hardcoded.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.model = TextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self._num_deepstack_layers = len(config.deepstack_visual_indexes or [])

    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        per_layer_inputs: ir.Value | None = None,
    ):
        deepstack_inputs: list[ir.Value] | None = None
        if self._num_deepstack_layers and per_layer_inputs is not None:
            num_layers = self._num_deepstack_layers
            hidden_size = self.config.hidden_size
            # [B, S, D*H] → [B, S, D, H]
            per_layer_4d = op.Reshape(
                per_layer_inputs,
                op.Constant(value_ints=[0, 0, num_layers, hidden_size]),
            )
            deepstack_inputs = [
                op.Squeeze(op.Slice(per_layer_4d, starts=[i], ends=[i + 1], axes=[2]), [2])
                for i in range(num_layers)
            ]

        hidden_states, present_key_values = self.model(
            op,
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            deepstack_inputs=deepstack_inputs,
        )
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Route language_model weights for standalone decoder build."""
        quantization = self.config.quantization
        if quantization is not None and quantization.quant_method == "olive":
            state_dict = preprocess_olive_weights(
                state_dict,
                bits=quantization.bits,
                group_size=quantization.group_size,
                quantize_embeddings=quantization.quantize_embeddings,
                quantize_lm_head=quantization.quantize_lm_head,
                tie_word_embeddings=self.config.tie_word_embeddings
                or quantization.tie_word_embeddings,
            )

        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            stripped = key
            if stripped.startswith("model."):
                stripped = stripped[len("model.") :]
            if stripped.startswith("visual."):
                continue
            # language_model.layers.* → model.layers.*
            if stripped.startswith("language_model."):
                stripped = stripped[len("language_model.") :]
            renamed[stripped] = value

        if self.config.tie_word_embeddings:
            # After stripping language_model., keys are embed_tokens.weight
            # and lm_head.weight (no model. prefix).
            tie_word_embeddings(
                renamed,
                embed_key="embed_tokens.weight",
                head_key="lm_head.weight",
            )
        return renamed


class Qwen3VLVisionEncoderModel(nn.Module):
    """Qwen3-VL vision encoder for the 3-model split.

    Processes packed image patches through the Qwen3-VL ViT and outputs
    merged features (DeepStack intermediate features are not exported).

    Inputs:
        - pixel_values: (total_patches, C*T_p*P*P)
        - image_grid_thw: (num_images, 3) INT64
    Output:
        - image_features: (num_merged_patches, out_hidden_size)
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        vc = config.vision
        assert vc is not None and vc.hidden_size is not None
        assert vc.num_hidden_layers is not None
        assert vc.num_attention_heads is not None
        self.visual = Qwen3VLVisionModel(
            depth=vc.num_hidden_layers,
            hidden_size=vc.hidden_size,
            intermediate_size=vc.intermediate_size or vc.hidden_size * 4,
            num_heads=vc.num_attention_heads,
            patch_size=vc.patch_size or 16,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=vc.in_channels,
            out_hidden_size=vc.out_hidden_size or config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
            num_position_embeddings=vc.num_position_embeddings or 2304,
            deepstack_visual_indexes=config.deepstack_visual_indexes or [],
        )

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        image_grid_thw: ir.Value,
    ):
        outputs = self.visual(
            op,
            hidden_states=pixel_values,
            grid_thw=image_grid_thw,
        )
        # Only return merged features (first element), drop DeepStack
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Keep only visual.* weights for standalone vision encoder build."""
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            stripped = key
            if stripped.startswith("model."):
                stripped = stripped[len("model.") :]
            if stripped.startswith("visual."):
                # Qwen3-VL uses linear_fc1/fc2; ONNX uses up_proj/down_proj
                stripped = stripped.replace(".mlp.linear_fc1.", ".mlp.up_proj.")
                stripped = stripped.replace(".mlp.linear_fc2.", ".mlp.down_proj.")
                renamed[stripped] = value
        return renamed


class Qwen3VLEmbeddingModel(Qwen25VLEmbeddingModel):
    """Qwen3-VL embedding model for the 3-model split.

    Identical to Qwen2.5-VL embedding — scatters image features at
    image token positions using cumsum + Gather + Where.
    """

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Keep only embed_tokens weights.

        HF key: ``model.language_model.embed_tokens.weight`` → ``embed_tokens.weight``.
        """
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if "embed_tokens" in key:
                new_key = key
                for prefix in ("model.", "language_model."):
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix) :]
                renamed[new_key] = value
        return renamed


# ---------------------------------------------------------------------------
# Qwen3-VL DeepStack 3-model split (separate from the plain Qwen3-VL /
# Qwen3.5-VL 3-model classes above, which stay untouched so their exported
# contract does not change).
# ---------------------------------------------------------------------------


class Qwen3VLDeepStackVisionEncoderModel(Qwen3VLVisionEncoderModel):
    """Qwen3-VL vision encoder for the 3-model split, with DeepStack outputs.

    Identical to :class:`Qwen3VLVisionEncoderModel` except ``forward`` keeps
    (rather than drops) the ``D`` intermediate DeepStack features, so the
    ONNX graph exposes ``image_features`` plus ``D`` ``deepstack_features_i``
    outputs (``D == len(config.deepstack_visual_indexes)``, never hardcoded).

    Weight preprocessing is unchanged (inherited): the ``deepstack_merger_list``
    weights are already covered by the parent's blanket ``visual.*`` copy,
    since :class:`~mobius.components._qwen3_vl_vision.Qwen3VLPatchMerger`
    (used for both the final merger and each ``deepstack_merger_list[i]``)
    already uses HF-native ``linear_fc1``/``linear_fc2`` attribute names.
    """

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        image_grid_thw: ir.Value,
    ):
        outputs = self.visual(
            op,
            hidden_states=pixel_values,
            grid_thw=image_grid_thw,
        )
        # Keep the merged features plus every DeepStack feature (in
        # deepstack_visual_indexes order); the vision component already
        # returns them as (merged, *deepstack_features).
        if isinstance(outputs, tuple):
            return outputs
        return (outputs,)


class Qwen3VLDeepStackEmbeddingModel(Qwen3VLEmbeddingModel):
    """Qwen3-VL embedding model for the 3-model split, with DeepStack support.

    In addition to the inherited image/video-feature fusion into
    ``inputs_embeds``, this variant accepts ``D`` ``deepstack_features_i``
    inputs and scatters each of them (using the same visual-token positions
    and ordering as ``image_features``) into a packed, zero-elsewhere
    ``per_layer_inputs [B, S, D*H]`` output consumed by
    :class:`Qwen3VLDecoderModel`. ``D`` is derived from
    ``len(config.deepstack_visual_indexes)``, never hardcoded.

    The visual-token mask combines image AND video placeholder tokens
    (``input_ids == image_token_id) | (input_ids == video_token_id)``),
    matching HF's ``image_mask | video_mask`` since Mobius's 3-model split
    already merges all visual patches (image and video frames alike) into a
    single flat ``image_features``/``deepstack_features_i`` stream.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        self.video_token_id: int | None = config.video_token_id
        self._num_deepstack_layers = len(config.deepstack_visual_indexes or [])

    def _visual_mask_and_indices(
        self, op: OpBuilder, input_ids: ir.Value
    ) -> tuple[ir.Value, ir.Value]:
        """Combined image-or-video mask (3D) and per-token gather indices.

        Both ``image_features`` and every ``deepstack_features_i`` share the
        same flat visual-token ordering, so the mask and cumsum-derived
        indices computed here are reused for all of them.
        """
        visual_mask = op.Equal(input_ids, op.Constant(value_int=self.image_token_id))
        if self.video_token_id is not None:
            video_mask = op.Equal(input_ids, op.Constant(value_int=self.video_token_id))
            visual_mask = op.Or(visual_mask, video_mask)
        visual_mask_3d = op.Unsqueeze(visual_mask, [-1])

        mask_int = op.Cast(visual_mask, to=7)  # INT64
        cumsum = op.CumSum(mask_int, op.Constant(value_int=1))
        indices = op.Sub(cumsum, op.Constant(value_int=1))
        indices = op.Clip(indices, op.Constant(value_int=0))
        return visual_mask_3d, indices

    def _scatter(
        self,
        op: OpBuilder,
        mask_3d: ir.Value,
        indices: ir.Value,
        features: ir.Value,
        fallback: ir.Value,
    ) -> ir.Value:
        """Gather ``features`` at ``indices`` and select via ``mask_3d``.

        Pads ``features`` with one zero row before ``Gather`` so the op is
        safe even when ``features`` is empty (text-only input, N == 0); the
        ``Where`` mask guarantees the padding row is never selected.
        """
        pad_row = op.Expand(
            op.CastLike(0.0, features),
            op.Concat(
                op.Constant(value_ints=[1]),
                op.Shape(features, start=1, end=2),
                axis=0,
            ),
        )
        padded = op.Concat(features, pad_row, axis=0)
        gathered = op.Gather(padded, indices, axis=0)
        return op.Where(mask_3d, gathered, fallback)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        image_features: ir.Value,
        deepstack_features: list[ir.Value] | None = None,
    ):
        if deepstack_features:
            assert len(deepstack_features) == self._num_deepstack_layers, (
                f"Expected {self._num_deepstack_layers} deepstack_features "
                f"(len(config.deepstack_visual_indexes)), got "
                f"{len(deepstack_features)}"
            )
        text_embeds = self.embed_tokens(op, input_ids)
        mask_3d, indices = self._visual_mask_and_indices(op, input_ids)

        inputs_embeds = self._scatter(op, mask_3d, indices, image_features, text_embeds)
        outputs: dict[str, ir.Value] = {"inputs_embeds": inputs_embeds}

        if deepstack_features:
            zero = op.CastLike(0.0, inputs_embeds)
            per_layer_slices = [
                op.Unsqueeze(
                    self._scatter(op, mask_3d, indices, feature, zero),
                    [2],
                )
                for feature in deepstack_features
            ]
            # [B, S, 1, H] * D → [B, S, D, H] → [B, S, D*H]
            stacked = op.Concat(*per_layer_slices, axis=2)
            per_layer_inputs = op.Reshape(stacked, op.Constant(value_ints=[0, 0, -1]))
            outputs["per_layer_inputs"] = per_layer_inputs

        return outputs
