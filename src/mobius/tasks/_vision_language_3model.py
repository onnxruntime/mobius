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

        pixel_values = ir.Value(
            name="pixel_values",
            shape=ir.Shape([batch, 3, image_size, image_size]),
            type=ir.TensorType(config.dtype),
        )

        graph, graph_builder = _make_graph([pixel_values], name="vision_encoder")
        image_features = vision(graph_builder.op, pixel_values=pixel_values)

        image_features.name = "image_features"
        graph.outputs.append(image_features)
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
        """Build Qwen VL vision encoder with packed patches and grid_thw."""
        total_patches = ir.SymbolicDim("total_patches")
        num_images = ir.SymbolicDim("num_images")

        patch_size = (config.vision.patch_size if config.vision else None) or 14
        temporal_patch_size = config.temporal_patch_size
        in_channels = config.vision.in_channels if config.vision else 3
        pixel_dim = in_channels * temporal_patch_size * patch_size * patch_size

        pixel_values = ir.Value(
            name="pixel_values",
            shape=ir.Shape([total_patches, pixel_dim]),
            type=ir.TensorType(config.dtype),
        )
        image_grid_thw = ir.Value(
            name="image_grid_thw",
            shape=ir.Shape([num_images, 3]),
            type=ir.TensorType(ir.DataType.INT64),
        )

        graph, graph_builder = _make_graph(
            [pixel_values, image_grid_thw], name="vision_encoder"
        )
        image_features = vision(
            graph_builder.op,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

        image_features.name = "image_features"
        graph.outputs.append(image_features)
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
        models: dict[str, ir.Model] = {}
        models["decoder"] = build_decoder_from_embeds(
            module.decoder, config, mrope=True, hybrid=True
        )
        models["vision_encoder"] = self._build_vision(module.vision_encoder, config)
        models["embedding"] = build_embedding_from_features(
            module.embedding,
            config,
            feature_name="image_features",
            feature_dim=config.hidden_size,
        )
        return ModelPackage(models, config=config)


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

        pixel_values = ir.Value(
            name="pixel_values",
            shape=ir.Shape([batch, 3, height, width]),
            type=ir.TensorType(config.dtype),
        )

        graph_inputs = [pixel_values]

        graph, graph_builder = _make_graph(graph_inputs, name="vision_encoder")
        op = graph_builder.op

        image_features = vision(
            op,
            pixel_values=pixel_values,
        )

        # Squeeze batch dim: [batch, num_patches, hidden] → [num_patches, hidden]
        # The runtime (ort-genai) expects rank-2 vision features because the
        # vision encoder always processes one image at a time.
        image_features = op.Squeeze(image_features, [0])

        image_features.name = "image_features"
        graph.outputs.append(image_features)

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

        inputs_embeds = ir.Value(
            name="inputs_embeds",
            shape=ir.Shape([batch, seq_len, config.hidden_size]),
            type=ir.TensorType(config.dtype),
        )
        attention_mask = ir.Value(
            name="attention_mask",
            shape=ir.Shape([batch, "past_seq_len + seq_len"]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        position_ids = ir.Value(
            name="position_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        # Vision features: full on prefill, empty (0-length) on decode
        cross_attention_states = ir.Value(
            name="cross_attention_states",
            shape=ir.Shape([batch, cross_seq_len, config.hidden_size]),
            type=ir.TensorType(config.dtype),
        )

        graph_inputs = [
            inputs_embeds,
            attention_mask,
            position_ids,
            cross_attention_states,
        ]

        # Per-layer KV cache with separate dims for self-attention
        # (past_seq_len) and cross-attention (cross_past_seq_len)
        cross_attention_layers = set(config.cross_attention_layers or [])
        flat_kv: list[ir.Value] = []
        past_key_values: list[tuple[ir.Value, ir.Value]] = []

        for i in range(config.num_hidden_layers):
            psl = cross_past_seq_len if i in cross_attention_layers else past_seq_len
            past_key = ir.Value(
                name=f"past_key_values.{i}.key",
                shape=ir.Shape([batch, config.num_key_value_heads, psl, config.head_dim]),
                type=ir.TensorType(config.dtype),
            )
            past_value = ir.Value(
                name=f"past_key_values.{i}.value",
                shape=ir.Shape([batch, config.num_key_value_heads, psl, config.head_dim]),
                type=ir.TensorType(config.dtype),
            )
            flat_kv.extend([past_key, past_value])
            past_key_values.append((past_key, past_value))

        graph_inputs.extend(flat_kv)

        graph, graph_builder = _make_graph(graph_inputs)
        op = graph_builder.op

        logits, present_key_values = decoder(
            op,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cross_attention_states=cross_attention_states,
            past_key_values=past_key_values,
        )

        logits.name = "logits"
        graph.outputs.append(logits)
        _register_kv_cache_outputs(graph, present_key_values)

        return _make_model(graph)
