# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Gemma 3n multimodal task (3- or 4-model split).

Builds:

1. **decoder** — ``inputs_embeds, attention_mask, position_ids,
   per_layer_inputs`` + KV cache -> ``logits`` + updated cache
2. **vision_encoder** — ``pixel_values`` -> ``image_features``
3. **embedding** — ``input_ids, image_features[, audio_features]`` ->
   ``inputs_embeds`` + ``per_layer_inputs``
4. **audio_encoder** (only when ``config.audio is not None``) —
   ``input_features, input_features_mask`` -> ``audio_features``

Two Gemma 3n traits shape the decoder's cache contract:

- ``num_kv_shared_layers`` — the trailing decoder layers borrow K,V from an
  earlier layer of the same attention type, so they own no cache entry.  The
  cache carries ``num_hidden_layers - num_kv_shared_layers`` pairs, which the
  decoder sub-module reports via ``kv_cache_layer_count()``.
- ``hidden_size_per_layer_input`` — the per-layer embedding tables live in the
  embedding sub-model (they are the bulk of the checkpoint), so the decoder
  takes their combined output as the ``per_layer_inputs`` graph input rather
  than re-deriving it from ``input_ids``.
"""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import Gemma3nMultiModalConfig
from mobius._model_package import ModelPackage
from mobius._pipeline_contract import (
    declare_component_presence,
    declare_optional_input,
)
from mobius.tasks._base import (
    ComponentSpec,
    ModelTask,
    _make_graph,
    _make_model,
)
from mobius.tasks._cache_utils import (
    _make_kv_cache_inputs,
    _register_kv_cache_outputs,
)

#: MobileNet-V5 has no dynamic-resolution path, so the vision graph fixes its
#: spatial extent.  Matches ``_GEMMA3N_IMAGE_SIZE`` in the vision extractor
#: hook and the ``size`` of the checkpoint's ``SiglipImageProcessorFast``.
_DEFAULT_IMAGE_SIZE = 768


class Gemma3nTask(ModelTask):
    """Task for Gemma 3n multimodal models (3- or 4-model split).

    Always builds ``decoder``, ``vision_encoder``, and ``embedding``; adds
    ``audio_encoder`` when ``config.audio is not None``.  The text-only export
    path uses :class:`~mobius.tasks.CausalLMTask` with
    :class:`~mobius.models.gemma3n.Gemma3nCausalLMModel` instead.

    Batching strategies
    -------------------
    **Vision** — fixed-size images, no mask.
        ``pixel_values [B, 3, S, S]`` with ``S`` from
        ``config.vision.image_size``.  The processor resizes every image to
        that extent, and the tower's output token count (``S/16`` squared) is
        therefore constant, so no padding or mask is needed.

    **Audio** — explicit contiguous bool mask, fixed-length output.
        ``input_features_mask [B, T]`` marks valid mel frames (``True``) vs
        right-padding.  Unlike Gemma 4, padded rows are *not* stripped from
        the output: the processor always splices exactly
        ``audio_soft_tokens_per_image`` placeholders into the prompt, and the
        encoder sub-model pads its output to that count, so the row count is
        fixed per clip.

    **Decoder** — standard ``attention_mask`` over ``past + current``.
    """

    model_roles: ClassVar[dict[str, str]] = {
        "decoder": "decoder",
        "vision_encoder": "encoder",
        "audio_encoder": "encoder",
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
        config: Gemma3nMultiModalConfig,
    ) -> ModelPackage:
        self._validate_components(module)
        models: dict[str, ir.Model] = {}
        models["decoder"] = self._build_decoder(module.decoder, config)
        models["vision_encoder"] = self._build_vision(module.vision_encoder, config)
        if config.audio is not None:
            if module.audio_encoder is None:
                raise ValueError(
                    "Gemma3nTask: config.audio is set but the module has no "
                    "audio_encoder sub-module."
                )
            models["audio_encoder"] = self._build_audio(module.audio_encoder, config)
        models["embedding"] = self._build_embedding(module.embedding, config)
        return ModelPackage(models, config=config)

    def _build_decoder(
        self,
        decoder: nn.Module,
        config: Gemma3nMultiModalConfig,
    ) -> ir.Model:
        """Build ``inputs_embeds + per_layer_inputs`` -> logits + KV cache."""
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        past_seq_len = ir.SymbolicDim("past_sequence_len")

        graph, builder = _make_graph(name="decoder")

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
        per_layer_inputs = builder.input(
            "per_layer_inputs",
            dtype=config.dtype,
            shape=[
                batch,
                seq_len,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
            ],
        )

        # KV-shared layers own no cache entry, so the cache is shorter than the
        # layer count.  The decoder reports the real count.
        count_fn = getattr(decoder, "kv_cache_layer_count", None)
        num_cache_layers = count_fn() if callable(count_fn) else config.num_hidden_layers
        past_key_values = _make_kv_cache_inputs(
            builder,
            num_cache_layers,
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
            per_layer_inputs=per_layer_inputs,
            past_key_values=past_key_values,
        )

        builder.add_output(logits, "logits")
        _register_kv_cache_outputs(builder, present_key_values)
        return _make_model(graph)

    def _build_vision(
        self,
        vision: nn.Module,
        config: Gemma3nMultiModalConfig,
    ) -> ir.Model:
        """Build ``pixel_values [B, 3, S, S]`` -> ``image_features``.

        Output is ``[B * vision_soft_tokens_per_image, hidden_size]``: one row
        per image placeholder token, in prompt order, matching the 2-D
        contract the embedding graph Gathers from.
        """
        batch = ir.SymbolicDim("batch")
        image_size = (config.vision.image_size if config.vision else None) or (
            _DEFAULT_IMAGE_SIZE
        )

        graph, builder = _make_graph(name="vision_encoder")

        pixel_values = builder.input(
            "pixel_values",
            dtype=config.dtype,
            shape=[batch, 3, image_size, image_size],
        )
        image_features = vision(builder.op, pixel_values=pixel_values)

        builder.add_output(image_features, "image_features")
        declare_component_presence(graph, "image")
        return _make_model(graph)

    def _build_audio(
        self,
        audio: nn.Module,
        config: Gemma3nMultiModalConfig,
    ) -> ir.Model:
        """Build ``input_features + input_features_mask`` -> ``audio_features``.

        ``input_features_mask [B, T]`` marks valid mel frames (``True``); it
        must be contiguous (right-padded), which the HuggingFace
        ``Gemma3nAudioFeatureExtractor`` guarantees.  Note the polarity is the
        negation of HF's ``audio_mel_mask``, matching
        :class:`~mobius.components.Gemma3nAudioEncoder`.

        Output is ``[B * audio_soft_tokens_per_image, hidden_size]``: the
        encoder sub-model pads its own output to the fixed placeholder count,
        so — unlike Gemma 4 — no ``Compress`` strip and no companion mask
        output are needed.
        """
        batch = ir.SymbolicDim("batch")
        time = ir.SymbolicDim("time")
        input_size = (config.audio.input_feat_size if config.audio else None) or 128

        graph, builder = _make_graph(name="audio_encoder")

        input_features = builder.input(
            "input_features",
            dtype=config.dtype,
            shape=[batch, time, input_size],
        )
        input_features_mask = builder.input(
            "input_features_mask",
            dtype=ir.DataType.BOOL,
            shape=[batch, time],
        )

        audio_features = audio(
            builder.op,
            input_features=input_features,
            input_features_mask=input_features_mask,
        )

        builder.add_output(audio_features, "audio_features")
        declare_component_presence(graph, "audio")
        return _make_model(graph)

    def _build_embedding(
        self,
        embedding: nn.Module,
        config: Gemma3nMultiModalConfig,
    ) -> ir.Model:
        """Build ``input_ids + features`` -> ``inputs_embeds + per_layer_inputs``.

        The feature inputs are optional: a text-only prompt or any decode step
        passes a zero-row tensor, declared through
        :func:`~mobius._pipeline_contract.declare_optional_input` so the
        runtime can synthesize it.
        """
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        num_image_tokens = ir.SymbolicDim("num_image_tokens")

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
        declare_optional_input(
            image_features,
            presence="image",
            absent_shape=[0, config.hidden_size],
        )

        audio_features: ir.Value | None = None
        if config.audio is not None:
            num_audio_tokens = ir.SymbolicDim("num_audio_tokens")
            audio_features = builder.input(
                "audio_features",
                dtype=config.dtype,
                shape=[num_audio_tokens, config.hidden_size],
            )
            declare_optional_input(
                audio_features,
                presence="audio",
                absent_shape=[0, config.hidden_size],
            )

        result = embedding(
            builder.op,
            input_ids=input_ids,
            image_features=image_features,
            audio_features=audio_features,
        )

        builder.add_output(result["inputs_embeds"], "inputs_embeds")
        builder.add_output(result["per_layer_inputs"], "per_layer_inputs")
        return _make_model(graph)
