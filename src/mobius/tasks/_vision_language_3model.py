# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Vision-language 3-model split tasks.

Builds three separate ONNX models:
1. **decoder** (text decoder): inputs_embeds → logits + KV cache
2. **vision** (vision encoder): pixel_values → image_features
3. **embedding**: input_ids + image_features → inputs_embeds
"""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import ArchitectureConfig, MllamaConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import (
    ComponentSpec,
    ModelTask,
    _make_graph,
    _make_model,
    build_decoder_from_embeds,
    build_embedding_from_features,
)
from mobius.tasks._cache_utils import (
    _register_kv_cache_outputs,
)


class VisionLanguageTask(ModelTask):
    """3-model split vision-language task.

    Produces three ONNX models (decoder, vision, embedding). The module must
    provide three sub-modules as attributes:

    - ``decoder``: text decoder taking ``inputs_embeds``
    - ``vision_encoder``: vision encoder taking ``pixel_values``
    - ``embedding``: embedding model fusing text + image features

    Subclass and override ``_build_vision`` for non-standard vision I/O
    (e.g. Qwen2.5-VL packed attention with cu_seqlens).
    """

    model_roles: ClassVar[dict[str, str]] = {
        "decoder": "decoder",
        "vision_encoder": "encoder",
        "embedding": "embedding",
    }
    components = ComponentSpec(
        decoder="decoder",
        vision_encoder="vision_encoder",
        embedding="embedding",
    )

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        self._validate_components(module)
        models: dict[str, ir.Model] = {}
        models["decoder"] = build_decoder_from_embeds(module.decoder, config)
        models["vision_encoder"] = self._build_vision(module.vision_encoder, config)
        models["embedding"] = build_embedding_from_features(
            module.embedding,
            config,
            feature_name="image_features",
            feature_dim=config.hidden_size,
        )
        return ModelPackage(models, config=config)

    def _build_vision(
        self,
        vision: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build vision encoder: pixel_values [batch, C, H, W] -> image_features."""
        batch = ir.SymbolicDim("batch")
        image_size = (config.vision.image_size if config.vision else None) or 224

        graph, builder = _make_graph(name="vision_encoder")
        op = builder.op
        pixel_values = builder.input(
            "pixel_values",
            dtype=config.dtype,
            shape=[batch, 3, image_size, image_size],
        )
        image_features = vision(op, pixel_values=pixel_values)

        builder.add_output(image_features, "image_features")
        return _make_model(graph)


class GGUFProjectorVisionLanguageTask(VisionLanguageTask):
    """Generic sidecar split with the processor-native float image boundary."""

    def _build_vision(
        self,
        vision: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build one image per call and expose rank-2 projected feature rows."""
        image_size = (config.vision.image_size if config.vision else None) or 224
        graph, builder = _make_graph(name="vision_encoder")
        pixel_values = builder.input(
            "pixel_values",
            dtype=ir.DataType.FLOAT,
            shape=[1, 3, image_size, image_size],
        )
        image_features = vision(builder.op, pixel_values=pixel_values)
        image_features = builder.op.Squeeze(image_features, [0])
        builder.add_output(image_features, "image_features")
        return _make_model(graph)


class Gemma3VisionLanguageTask(VisionLanguageTask):
    """Gemma 3 split with the processor-native, single-image vision boundary."""

    def _build_vision(
        self,
        vision: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build one Gemma 3 image per invocation; callers split processor image rows."""
        image_size = (config.vision.image_size if config.vision else None) or 896

        graph, builder = _make_graph(name="vision_encoder")
        op = builder.op
        pixel_values = builder.input(
            "pixel_values",
            dtype=ir.DataType.FLOAT,
            shape=[1, 3, image_size, image_size],
        )
        image_features = vision(op, pixel_values=pixel_values)

        builder.add_output(image_features, "image_features")
        return _make_model(graph)


class Cosmos3EdgeVLTask(VisionLanguageTask):
    """NVIDIA Cosmos3-Edge VL 3-model split.

    Builds the text decoder with 3D multimodal RoPE
    (``mrope_section=[24, 20, 20]``), so ``position_ids`` has shape
    ``[3, batch, seq]``. The vision runtime processes one image at a time and
    removes the image batch dimension so its output matches the embedding
    model's rank-2 ``[num_image_tokens, hidden]`` input contract.
    """

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        self._validate_components(module)
        models: dict[str, ir.Model] = {}
        models["decoder"] = build_decoder_from_embeds(module.decoder, config, mrope=True)
        models["vision_encoder"] = self._build_vision(module.vision_encoder, config)
        models["embedding"] = build_embedding_from_features(
            module.embedding,
            config,
            feature_name="image_features",
            feature_dim=config.hidden_size,
        )
        return ModelPackage(models, config=config)

    def _build_vision(
        self,
        vision: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build the single-image vision encoder with rank-2 feature output."""
        image_size = (config.vision.image_size if config.vision else None) or 224

        graph, builder = _make_graph(name="vision_encoder")
        pixel_values = builder.input(
            "pixel_values",
            dtype=config.dtype,
            shape=[1, 3, image_size, image_size],
        )
        image_features = vision(builder.op, pixel_values=pixel_values)
        image_features = builder.op.Squeeze(image_features, [0])

        builder.add_output(image_features, "image_features")
        return _make_model(graph)


class QwenVLTask(VisionLanguageTask):
    """Qwen-family VL 3-model split with packed-attention vision and MRoPE.

    Used by Qwen2.5-VL, Qwen3-VL, and Qwen3.5-VL.  Overrides
    ``_build_vision`` for the packed-attention I/O contract and uses
    MRoPE 3D position_ids for the decoder.
    """

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        self._validate_components(module)
        deepstack = bool(getattr(config, "deepstack_visual_indexes", None))
        models: dict[str, ir.Model] = {}
        models["decoder"] = build_decoder_from_embeds(
            module.decoder, config, mrope=True, deepstack=deepstack
        )
        models["vision_encoder"] = self._build_vision(module.vision_encoder, config)
        models["embedding"] = build_embedding_from_features(
            module.embedding,
            config,
            feature_name="image_features",
            feature_dim=config.hidden_size,
            deepstack=deepstack,
        )
        return ModelPackage(models, config=config)

    def _build_vision(
        self,
        vision: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build Qwen VL vision encoder with packed patches and grid_thw.

        When ``deepstack_visual_indexes`` is set (Qwen3-VL family), the final
        and intermediate maps are packed into the single ``image_features``
        output as ``[num_merged_patches, (D + 1) * out_hidden]``.
        """
        total_patches = ir.SymbolicDim("total_patches")
        num_images = ir.SymbolicDim("num_images")

        patch_size = (config.vision.patch_size if config.vision else None) or 14
        temporal_patch_size = config.temporal_patch_size
        in_channels = config.vision.in_channels if config.vision else 3
        pixel_dim = in_channels * temporal_patch_size * patch_size * patch_size

        graph, builder = _make_graph(name="vision_encoder")
        op = builder.op
        pixel_values = builder.input(
            "pixel_values",
            dtype=ir.DataType.FLOAT,
            shape=[total_patches, pixel_dim],
        )
        image_grid_thw = builder.input(
            "image_grid_thw",
            dtype=ir.DataType.INT64,
            shape=[num_images, 3],
        )

        outputs = vision(
            op,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

        if isinstance(outputs, tuple):
            image_features, deepstack_features = outputs
            num_deepstack = len(config.deepstack_visual_indexes or [])
            deepstack_flat = op.Reshape(
                op.Transpose(deepstack_features, perm=[1, 0, 2]),
                op.Constant(value_ints=[0, num_deepstack * config.hidden_size]),
            )
            packed_features = op.Concat(image_features, deepstack_flat, axis=1)
            builder.add_output(packed_features, "image_features")
        else:
            builder.add_output(outputs, "image_features")
        return _make_model(graph)


class Qwen2VLMultimediaTask(QwenVLTask):
    """Qwen2/Qwen2.5 task preserving independent image and video streams."""

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        self._validate_components(module)
        models = {
            "decoder": build_decoder_from_embeds(module.decoder, config, mrope=True),
            "vision_encoder": self._build_vision(module.vision_encoder, config),
            "embedding": self._build_multimedia_embedding(module.embedding, config),
        }
        return ModelPackage(models, config=config)

    @staticmethod
    def _build_multimedia_embedding(
        embedding: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        num_image_tokens = ir.SymbolicDim("num_image_tokens")
        num_video_tokens = ir.SymbolicDim("num_video_tokens")

        graph, builder = _make_graph(name="embedding")
        input_ids = builder.input(
            "input_ids",
            dtype=ir.DataType.INT64,
            shape=[batch, seq_len],
        )
        image_features = builder.input(
            "image_features",
            dtype=config.dtype,
            shape=[num_image_tokens, config.hidden_size],
        )
        video_features = builder.input(
            "video_features",
            dtype=config.dtype,
            shape=[num_video_tokens, config.hidden_size],
        )
        inputs_embeds = embedding(
            builder.op,
            input_ids=input_ids,
            image_features=image_features,
            video_features=video_features,
        )
        builder.add_output(inputs_embeds, "inputs_embeds")
        return _make_model(graph)


class GlmOcrVLTask(QwenVLTask):
    """GLM-OCR packed vision task with a float32 processor boundary."""

    def _build_vision(
        self,
        vision: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        total_patches = ir.SymbolicDim("total_patches")
        num_images = ir.SymbolicDim("num_images")
        vision_config = config.vision
        assert vision_config is not None
        patch_size = vision_config.patch_size or 14
        pixel_dim = (
            vision_config.in_channels
            * vision_config.temporal_patch_size
            * patch_size
            * patch_size
        )

        graph, builder = _make_graph(name="vision_encoder")
        pixel_values = builder.input(
            "pixel_values",
            dtype=ir.DataType.FLOAT,
            shape=[total_patches, pixel_dim],
        )
        image_grid_thw = builder.input(
            "image_grid_thw",
            dtype=ir.DataType.INT64,
            shape=[num_images, 3],
        )
        model_pixels = builder.op.Cast(pixel_values, to=config.dtype)
        image_features = vision(
            builder.op,
            pixel_values=model_pixels,
            image_grid_thw=image_grid_thw,
        )
        builder.add_output(image_features, "image_features")
        return _make_model(graph)


class MuseGlimmerVLTask(QwenVLTask):
    """Muse Glimmer packed vision pipeline with standard 1D text RoPE."""

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        self._validate_components(module)
        models: dict[str, ir.Model] = {}
        models["decoder"] = build_decoder_from_embeds(module.decoder, config)
        models["vision_encoder"] = self._build_vision(module.vision_encoder, config)
        models["embedding"] = build_embedding_from_features(
            module.embedding,
            config,
            feature_name="image_features",
            feature_dim=config.hidden_size,
        )
        return ModelPackage(models, config=config)


class MageVLTask(VisionLanguageTask):
    """Mage-VL split with packed patches and explicit sampled-frame positions."""

    def _build_vision(
        self,
        vision: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        total_patches = ir.SymbolicDim("total_patches")
        num_visuals = ir.SymbolicDim("num_visuals")
        patch_size = (config.vision.patch_size if config.vision else None) or 16
        in_channels = config.vision.in_channels if config.vision else 3
        pixel_dim = in_channels * patch_size * patch_size

        graph, builder = _make_graph(name="vision_encoder")
        pixel_values = builder.input(
            "pixel_values",
            dtype=ir.DataType.FLOAT,
            shape=[total_patches, pixel_dim],
        )
        image_grid_thw = builder.input(
            "image_grid_thw",
            dtype=ir.DataType.INT64,
            shape=[num_visuals, 3],
        )
        patch_positions = builder.input(
            "patch_positions",
            dtype=ir.DataType.INT64,
            shape=[total_patches, 3],
        )
        image_features = vision(
            builder.op,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            patch_positions=patch_positions,
        )
        builder.add_output(image_features, "image_features")
        return _make_model(graph)


class HybridQwenVLTask(QwenVLTask):
    """Qwen VL 3-model split with hybrid KV + DeltaNet cache.

    Used by Qwen3.5-VL which has mixed ``"full_attention"`` and
    ``"linear_attention"`` (DeltaNet) layers.  Vision and embedding
    models are identical to :class:`QwenVLTask`; only the decoder
    uses hybrid cache inputs/outputs.
    """

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        self._validate_components(module)
        deepstack = bool(getattr(config, "deepstack_visual_indexes", None))
        models: dict[str, ir.Model] = {}
        models["decoder"] = build_decoder_from_embeds(
            module.decoder, config, mrope=True, hybrid=True, deepstack=deepstack
        )
        models["vision_encoder"] = self._build_vision(module.vision_encoder, config)
        models["embedding"] = build_embedding_from_features(
            module.embedding,
            config,
            feature_name="image_features",
            feature_dim=config.hidden_size,
            deepstack=deepstack,
        )
        return ModelPackage(models, config=config)


class Lfm2VlTask(VisionLanguageTask):
    """LFM2-VL split: SigLIP2 NaFlex vision with an LFM2 hybrid decoder.

    The decoder mixes ``"conv"`` (gated short convolution) and
    ``"full_attention"`` layers, so it uses the hybrid cache contract.  The
    vision encoder takes the NaFlex triple emitted by the image processor:
    pre-patchified pixels, a per-patch padding mask, and the per-image patch
    grid, and returns the flat ``[num_image_tokens, text_hidden]`` stream that
    the embedding model scatters onto the image placeholder tokens.
    """

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        self._validate_components(module)
        models: dict[str, ir.Model] = {}
        models["decoder"] = build_decoder_from_embeds(
            module.decoder,
            config,
            hybrid=True,
        )
        models["vision_encoder"] = self._build_vision(module.vision_encoder, config)
        models["embedding"] = build_embedding_from_features(
            module.embedding,
            config,
            feature_name="image_features",
            feature_dim=config.hidden_size,
        )
        return ModelPackage(models, config=config)

    def _build_vision(
        self,
        vision: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build the NaFlex vision encoder: patchified pixels -> image features."""
        num_images = ir.SymbolicDim("num_images")
        max_patches = ir.SymbolicDim("max_num_patches")
        vision_config = config.vision
        assert vision_config is not None
        patch_size = vision_config.patch_size or 16
        patch_dim = vision_config.in_channels * patch_size * patch_size

        graph, builder = _make_graph(name="vision_encoder")
        pixel_values = builder.input(
            "pixel_values",
            dtype=config.dtype,
            shape=[num_images, max_patches, patch_dim],
        )
        pixel_attention_mask = builder.input(
            "pixel_attention_mask",
            dtype=ir.DataType.INT64,
            shape=[num_images, max_patches],
        )
        spatial_shapes = builder.input(
            "spatial_shapes",
            dtype=ir.DataType.INT64,
            shape=[num_images, 2],
        )
        image_features = vision(
            builder.op,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        )
        builder.add_output(image_features, "image_features")
        return _make_model(graph)


class MiniCPMVLTask(VisionLanguageTask):
    """MiniCPM-V packed-NaViT vision with a Qwen3.5 hybrid decoder."""

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        self._validate_components(module)
        models: dict[str, ir.Model] = {}
        models["decoder"] = build_decoder_from_embeds(
            module.decoder,
            config,
            hybrid=True,
        )
        models["vision_encoder"] = self._build_vision(module.vision_encoder, config)
        models["embedding"] = build_embedding_from_features(
            module.embedding,
            config,
            feature_name="image_features",
            feature_dim=config.hidden_size,
        )
        return ModelPackage(models, config=config)

    def _build_vision(
        self,
        vision: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build packed pixels + patch-grid sizes -> compressed visual tokens."""
        packed_batch = ir.SymbolicDim("packed_batch")
        packed_width = ir.SymbolicDim("packed_width")
        num_visual_units = ir.SymbolicDim("num_visual_units")
        vision_config = config.vision
        assert vision_config is not None
        patch_size = vision_config.patch_size or 14

        graph, builder = _make_graph(name="vision_encoder")
        pixel_values = builder.input(
            "pixel_values",
            dtype=config.dtype,
            shape=[
                packed_batch,
                vision_config.in_channels,
                patch_size,
                packed_width,
            ],
        )
        target_sizes = builder.input(
            "target_sizes",
            dtype=ir.DataType.INT32,
            shape=[num_visual_units, 2],
        )
        image_features = vision(
            builder.op,
            pixel_values=pixel_values,
            target_sizes=target_sizes,
        )
        builder.add_output(image_features, "image_features")
        return _make_model(graph)


class PixtralVLTask(VisionLanguageTask):
    """Vision-language task with dynamic-resolution Pixtral vision encoder.

    Pixtral's internal computations (patch embedding, 2D RoPE, spatial merge)
    are fully dynamic — grid_h and grid_w are derived from ``op.Shape()`` at
    runtime.  This subclass replaces the static ``image_size`` input shape
    with symbolic ``height`` / ``width`` dimensions so the exported ONNX
    model accepts variable-resolution images.

    Constraints (enforced at runtime, not in the graph):
      - H, W ≥ ``patch_size * spatial_merge_size`` (28 for Pixtral)
      - H, W ≤ ``image_size`` (1540 for Pixtral) due to RoPE cache limits
    """

    def _build_vision(
        self,
        vision: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build Pixtral vision encoder with dynamic HxW input."""
        batch = ir.SymbolicDim("batch")
        height = ir.SymbolicDim("height")
        width = ir.SymbolicDim("width")

        graph, builder = _make_graph(name="vision_encoder")
        op = builder.op
        pixel_values = builder.input(
            "pixel_values",
            dtype=config.dtype,
            shape=[batch, 3, height, width],
        )

        image_features = vision(
            op,
            pixel_values=pixel_values,
        )

        # Squeeze batch dim: [batch, num_patches, hidden] → [num_patches, hidden]
        # The runtime (ort-genai) expects rank-2 vision features because the
        # vision encoder always processes one image at a time.
        image_features = op.Squeeze(image_features, [0])

        builder.add_output(image_features, "image_features")

        return _make_model(graph)


class MllamaVisionLanguageTask(VisionLanguageTask):
    """Mllama VL task with cross-attention KV caching.

    Mllama's text decoder has interleaved self-attention and cross-attention
    layers.  Cross-attention layers attend to vision encoder output, which
    is constant during generation.  This task:

    1. Adds ``cross_attention_states`` as a decoder graph input.
    2. Uses separate KV cache symbolic dims for self-attention
       (``past_sequence_len``) and cross-attention (``cross_past_seq_len``).

    At runtime the host passes the full vision features on prefill
    (``cross_attention_states`` has shape ``[B, N, H]``), then an empty
    tensor on decode (``[B, 0, H]``).  The cross-attention KV cache is
    populated during prefill and reused unchanged on every decode step.
    """

    components = ComponentSpec(
        decoder="decoder",
        vision_encoder="vision_encoder",
        embedding="embedding",
    )

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        self._validate_components(module)
        models: dict[str, ir.Model] = {}
        models["decoder"] = self._build_decoder(module.decoder, config)
        models["vision_encoder"] = self._build_vision(module.vision_encoder, config)
        models["embedding"] = build_embedding_from_features(
            module.embedding,
            config,
            feature_name="image_features",
            feature_dim=config.hidden_size,
        )
        return ModelPackage(models, config=config)

    def _build_decoder(
        self,
        decoder: nn.Module,
        config: ArchitectureConfig,
    ) -> ir.Model:
        """Build decoder with cross_attention_states input and split KV cache."""
        assert isinstance(config, MllamaConfig)

        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        past_seq_len = ir.SymbolicDim("past_sequence_len")
        cross_seq_len = ir.SymbolicDim("cross_sequence_len")
        cross_past_seq_len = ir.SymbolicDim("cross_past_seq_len")

        graph, builder = _make_graph()
        inputs_embeds = builder.input(
            "inputs_embeds",
            dtype=config.dtype,
            shape=[batch, seq_len, config.hidden_size],
        )
        attention_mask = builder.input(
            "attention_mask",
            dtype=ir.DataType.INT64,
            shape=[batch, "past_seq_len + seq_len"],
        )
        position_ids = builder.input(
            "position_ids",
            dtype=ir.DataType.INT64,
            shape=[batch, seq_len],
        )
        # Vision features: full on prefill, empty (0-length) on decode
        cross_attention_states = builder.input(
            "cross_attention_states",
            dtype=config.dtype,
            shape=[batch, cross_seq_len, config.hidden_size],
        )

        # Per-layer KV cache with separate dims for self-attention
        # (past_seq_len) and cross-attention (cross_past_seq_len)
        cross_attention_layers = set(config.cross_attention_layers or [])
        past_key_values: list[tuple[ir.Value, ir.Value]] = []

        for i in range(config.num_hidden_layers):
            psl = cross_past_seq_len if i in cross_attention_layers else past_seq_len
            past_key = builder.input(
                f"past_key_values.{i}.key",
                dtype=config.dtype,
                shape=[batch, config.num_key_value_heads, psl, config.head_dim],
            )
            past_value = builder.input(
                f"past_key_values.{i}.value",
                dtype=config.dtype,
                shape=[batch, config.num_key_value_heads, psl, config.head_dim],
            )
            past_key_values.append((past_key, past_value))

        op = builder.op

        logits, present_key_values = decoder(
            op,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cross_attention_states=cross_attention_states,
            past_key_values=past_key_values,
        )

        builder.add_output(logits, "logits")
        _register_kv_cache_outputs(builder, present_key_values)

        return _make_model(graph)
