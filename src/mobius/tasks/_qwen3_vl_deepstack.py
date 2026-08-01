# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Qwen3-VL 3-model split task with DeepStack intermediate feature injection.

Separate from :class:`~mobius.tasks._vision_language_3model.QwenVLTask` /
:class:`~mobius.tasks._vision_language_3model.HybridQwenVLTask` (still used
unchanged by Qwen2.5-VL and Qwen3.5-VL) so those contracts do not change.

Builds three ONNX models:

1. **vision_encoder**: ``pixel_values`` + ``image_grid_thw`` →
   ``image_features`` plus ``D`` ``deepstack_features_i`` outputs
   (in ``deepstack_visual_indexes`` order).
2. **embedding**: ``input_ids`` + ``image_features`` + ``D``
   ``deepstack_features_i`` inputs → ``inputs_embeds`` plus (when ``D > 0``)
   a packed ``per_layer_inputs [B, S, D*H]`` output. Each flat feature is
   scattered to visual-token positions (``image_token_id`` OR
   ``video_token_id``), safely handling empty (text-only) tensors, and is
   exactly zero at non-visual positions in ``per_layer_inputs``.
3. **decoder**: ``inputs_embeds`` + optional ``per_layer_inputs`` →
   ``logits`` + KV cache. Slices ``per_layer_inputs`` into ``D`` per-layer
   ``[B, S, H]`` chunks and adds each one after the corresponding decoder
   layer (0..D-1), equivalent to HuggingFace's
   ``hidden_states[visual_pos_masks] += deepstack_visual_embeds[layer_idx]``.

``D`` is always derived from ``len(config.deepstack_visual_indexes)`` and is
never hardcoded; ``D == 0`` degrades to the same I/O contract as
:class:`QwenVLTask` (no ``deepstack_features_i``/``per_layer_inputs`` ports).
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import _make_graph, _make_model
from mobius.tasks._cache_utils import _make_kv_cache_inputs, _register_kv_cache_outputs
from mobius.tasks._vision_language_3model import QwenVLTask


class Qwen3VLDeepStackTask(QwenVLTask):
    """Qwen3-VL 3-model split with DeepStack support.

    Used only by Qwen3-VL (``model_type == "qwen3_vl"``); registered
    separately from ``qwen-vl``/``hybrid-qwen-vl`` so Qwen2.5-VL and
    Qwen3.5-VL keep their existing (non-DeepStack) 3-model contract.
    """

    def build(
        self,
        module: nn.Module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        self._validate_components(module)
        num_deepstack = len(config.deepstack_visual_indexes or [])

        models: dict[str, ir.Model] = {}
        models["vision_encoder"] = self._build_vision(
            module.vision_encoder, config, num_deepstack
        )
        models["embedding"] = self._build_embedding(module.embedding, config, num_deepstack)
        models["decoder"] = self._build_decoder(module.decoder, config, num_deepstack)
        return ModelPackage(models, config=config)

    def _build_vision(
        self,
        vision: nn.Module,
        config: ArchitectureConfig,
        num_deepstack: int,
    ) -> ir.Model:
        """Build vision encoder: packed patches -> image_features + deepstack_features_i."""
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
            dtype=config.dtype,
            shape=[total_patches, pixel_dim],
        )
        image_grid_thw = builder.input(
            "image_grid_thw",
            dtype=ir.DataType.INT64,
            shape=[num_images, 3],
        )

        # Qwen3VLDeepStackVisionEncoderModel.forward always returns
        # (image_features, *deepstack_features), even when D == 0.
        outputs = vision(
            op,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        image_features, *deepstack_features = outputs
        assert len(deepstack_features) == num_deepstack, (
            f"vision encoder produced {len(deepstack_features)} deepstack "
            f"feature(s), expected {num_deepstack} "
            "(len(config.deepstack_visual_indexes))"
        )

        builder.add_output(image_features, "image_features")
        for i, feature in enumerate(deepstack_features):
            builder.add_output(feature, f"deepstack_features_{i}")

        return _make_model(graph)

    def _build_embedding(
        self,
        embedding: nn.Module,
        config: ArchitectureConfig,
        num_deepstack: int,
    ) -> ir.Model:
        """Build embedding: input_ids + image_features + deepstack_features_i -> inputs_embeds [+ per_layer_inputs]."""
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        num_feature_tokens = ir.SymbolicDim("num_feature_tokens")
        hidden_size = config.hidden_size

        graph, builder = _make_graph(name="embedding")
        input_ids = builder.input(
            "input_ids",
            dtype=ir.DataType.INT64,
            shape=[batch, seq_len],
        )
        image_features = builder.input(
            "image_features",
            dtype=config.dtype,
            shape=[num_feature_tokens, hidden_size],
        )
        deepstack_features = [
            builder.input(
                f"deepstack_features_{i}",
                dtype=config.dtype,
                shape=[num_feature_tokens, hidden_size],
            )
            for i in range(num_deepstack)
        ]

        outputs = embedding(
            builder.op,
            input_ids=input_ids,
            image_features=image_features,
            deepstack_features=deepstack_features or None,
        )

        builder.add_output(outputs["inputs_embeds"], "inputs_embeds")
        if "per_layer_inputs" in outputs:
            builder.add_output(outputs["per_layer_inputs"], "per_layer_inputs")

        return _make_model(graph)

    def _build_decoder(
        self,
        decoder: nn.Module,
        config: ArchitectureConfig,
        num_deepstack: int,
    ) -> ir.Model:
        """Build decoder: inputs_embeds [+ per_layer_inputs] -> logits + KV cache."""
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        past_seq_len = ir.SymbolicDim("past_sequence_len")

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
        # Interleaved MRoPE: 3D position IDs (temporal, height, width)
        position_ids = builder.input(
            "position_ids",
            dtype=ir.DataType.INT64,
            shape=[3, batch, seq_len],
        )

        per_layer_inputs = None
        if num_deepstack:
            per_layer_inputs = builder.input(
                "per_layer_inputs",
                dtype=config.dtype,
                shape=[batch, seq_len, num_deepstack * config.hidden_size],
            )

        past_key_values = _make_kv_cache_inputs(
            builder,
            config.num_hidden_layers,
            config.num_key_value_heads,
            config.head_dim,
            config.dtype,
            batch,
            past_seq_len,
        )

        logits, present_key_values = decoder(
            builder.op,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            per_layer_inputs=per_layer_inputs,
        )

        builder.add_output(logits, "logits")
        _register_kv_cache_outputs(builder, present_key_values)

        return _make_model(graph)
