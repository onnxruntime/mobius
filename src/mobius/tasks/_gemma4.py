# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Gemma4 task classes.

The unified :class:`Gemma4Task` builds a 3- or 4-model package:

1. **decoder** (text decoder): ``inputs_embeds`` → logits + KV cache
2. **vision** (vision encoder): ``pixel_values, pixel_position_ids`` → ``image_features``
3. **embedding**: ``input_ids, image_features[, audio_features]`` → ``inputs_embeds``
4. **audio** (audio encoder, only when ``config.audio is not None``):
   ``input_features`` → ``audio_features``

The decoder uses per-layer KV cache where local (sliding_attention) layers
use ``config.head_dim`` and global (full_attention) layers use
``config.global_head_dim``.  When ``config.num_kv_shared_layers > 0``, the
last ``num_kv_shared_layers`` decoder layers share K,V from earlier layers
and therefore have NO separate KV cache entries — the cache only contains
entries for the first ``num_hidden_layers - num_kv_shared_layers`` layers.
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._build_context import ep_capabilities, get_build_dtype
from mobius._configs import Gemma4Config
from mobius._flags import flags
from mobius._model_package import ModelPackage
from mobius.tasks._base import (
    ModelTask,
    _make_graph,
    _make_model,
)
from mobius.tasks._cache_utils import (
    _register_kv_cache_outputs,
)


def _will_use_gqa() -> bool:
    """Return True when the current build context will use GQA rewrite rules."""
    caps = ep_capabilities()
    dtype = get_build_dtype()
    return dtype in caps.gqa_dtypes and caps.supports_fused_rope


def _make_gemma4_kv_cache_inputs(
    config: Gemma4Config,
    batch: ir.SymbolicDim,
    past_seq_len: ir.SymbolicDim,
    *,
    use_gqa: bool = False,
) -> tuple[list[ir.Value], list[tuple[ir.Value, ir.Value]]]:
    """Create per-layer KV cache inputs accounting for dual head_dim and KV sharing.

    Local (sliding_attention) layers use ``config.head_dim``;
    global (full_attention) layers use ``config.global_head_dim``.

    The last ``config.num_kv_shared_layers`` layers share K,V from earlier
    layers and do NOT have their own cache entries — only
    ``num_hidden_layers - num_kv_shared_layers`` entries are created.

    When *use_gqa* is True (CUDA/DML EP with supported dtype), sliding
    layers use GroupQueryAttention ops with native GQA support, so their
    cache keeps the original ``num_key_value_heads``.  Full_attention layers
    always expand because head_dim exceeds the GQA MAX_HEAD_SIZE limit.

    When *use_gqa* is False (CPU EP), ALL layers use the standard Attention
    op which requires KV expansion to avoid GQA dispatch.
    """
    local_head_dim = config.head_dim
    global_head_dim = config.global_head_dim or config.head_dim
    # Number of layers with independent KV projections
    num_kv_shared = config.num_kv_shared_layers or 0
    num_kv_layers = config.num_hidden_layers - num_kv_shared
    layer_types = config.layer_types or ["sliding_attention"] * config.num_hidden_layers
    if len(layer_types) != config.num_hidden_layers:
        raise ValueError(
            f"Gemma4Config.layer_types length ({len(layer_types)}) must match "
            f"num_hidden_layers ({config.num_hidden_layers})"
        )

    flat: list[ir.Value] = []
    pairs: list[tuple[ir.Value, ir.Value]] = []
    expand_kv = (
        flags.expand_kv_heads_for_attention
        and config.num_key_value_heads < config.num_attention_heads
    )
    for i in range(num_kv_layers):
        layer_type = layer_types[i] if i < len(layer_types) else "sliding_attention"
        hd = global_head_dim if layer_type == "full_attention" else local_head_dim
        # Expand KV heads when the layer uses the standard Attention path
        # (which can't handle GQA + attn_mask).  With GQA rewrite rules
        # active, only full_attention layers need expansion (sliding layers
        # go through the GQA op).  Without GQA, all layers need expansion.
        needs_expand = expand_kv and (layer_type == "full_attention" or not use_gqa)
        kv_heads = config.num_attention_heads if needs_expand else config.num_key_value_heads
        past_key = ir.Value(
            name=f"past_key_values.{i}.key",
            shape=ir.Shape([batch, kv_heads, past_seq_len, hd]),
            type=ir.TensorType(config.dtype),
        )
        past_value = ir.Value(
            name=f"past_key_values.{i}.value",
            shape=ir.Shape([batch, kv_heads, past_seq_len, hd]),
            type=ir.TensorType(config.dtype),
        )
        flat.extend([past_key, past_value])
        pairs.append((past_key, past_value))
    return flat, pairs


class Gemma4TextCausalLMTask(ModelTask):
    """Text-only causal LM task for Gemma4 with correct dual head_dim KV cache.

    Identical to :class:`~mobius.tasks.CausalLMTask` but uses
    :func:`_make_gemma4_kv_cache_inputs` for the KV cache so that:

    - global (full_attention) layers use ``config.global_head_dim`` (512)
    - local (sliding_attention) layers use ``config.head_dim`` (256)
    - the last ``config.num_kv_shared_layers`` layers share K,V and have no
      independent cache entries

    Inputs:
        - input_ids: [batch, sequence_len] INT64
        - attention_mask: [batch, past_seq_len + seq_len] INT64
        - position_ids: [batch, sequence_len] INT64
        - past_key_values.{i}.key / .value for i in 0..num_kv_layers-1
    Outputs:
        - logits: FLOAT
        - present.{i}.key / present.{i}.value for i in 0..num_kv_layers-1
    """

    def build(
        self,
        module: nn.Module,
        config: Gemma4Config,
    ) -> ModelPackage:
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        past_seq_len = ir.SymbolicDim("past_sequence_len")

        input_ids = ir.Value(
            name="input_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
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

        graph_inputs = [input_ids, attention_mask, position_ids]
        kv_inputs, past_key_values = _make_gemma4_kv_cache_inputs(
            config, batch, past_seq_len, use_gqa=_will_use_gqa()
        )
        graph_inputs.extend(kv_inputs)

        graph, graph_builder = _make_graph(graph_inputs)
        op = graph_builder.op

        logits, present_key_values = module(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        logits.name = "logits"
        graph.outputs.append(logits)
        _register_kv_cache_outputs(graph, present_key_values)

        return ModelPackage({"model": _make_model(graph)}, config=config)


class Gemma4Task(ModelTask):
    """Unified task for Gemma4 multimodal models (3- or 4-model split).

    Always builds:
    - ``decoder``: text decoder accepting ``inputs_embeds``
    - ``vision``: vision encoder accepting ``pixel_values, pixel_position_ids``
    - ``embedding``: embedding model fusing ``input_ids`` and multimodal features

    When ``config.audio is not None``, also builds:
    - ``audio``: Conformer audio encoder accepting ``input_features``
      (and adds ``audio_features`` as a third input to ``embedding``)

    Decoder KV cache is per-layer with the correct head_dim for each layer type
    (local vs global), unlike the uniform head_dim in :class:`VisionLanguageTask`.

    Vision input is pre-patchified: ``pixel_values [B, N, 3*P^2]`` and
    ``pixel_position_ids [B, N, 2]`` with (x, y) patch coordinates.
    """

    def build(
        self,
        module: nn.Module,
        config: Gemma4Config,
    ) -> ModelPackage:
        models: dict[str, ir.Model] = {}
        models["decoder"] = self._build_decoder(module.decoder, config)
        models["vision"] = self._build_vision(module.vision_encoder, config)
        if config.audio is not None:
            models["audio"] = self._build_audio(module.audio_encoder, config)
        models["embedding"] = self._build_embedding(module.embedding, config)
        return ModelPackage(models, config=config)

    def _build_decoder(
        self,
        decoder: nn.Module,
        config: Gemma4Config,
    ) -> ir.Model:
        """Build text decoder: inputs_embeds + input_ids -> logits + per-layer KV cache.

        ``input_ids`` is included alongside ``inputs_embeds`` because models with
        ``hidden_size_per_layer_input > 0`` (e.g. Gemma4 E2B) need the original token
        IDs to compute per-layer token embeddings that condition each decoder layer.
        When ``hidden_size_per_layer_input == 0`` the tensor is passed through but has
        no effect (``_compute_per_layer_inputs`` short-circuits to ``None``).
        """
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        past_seq_len = ir.SymbolicDim("past_sequence_len")

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
        input_ids = ir.Value(
            name="input_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )

        graph_inputs = [inputs_embeds, attention_mask, position_ids, input_ids]

        kv_inputs, past_key_values = _make_gemma4_kv_cache_inputs(
            config, batch, past_seq_len, use_gqa=_will_use_gqa()
        )
        graph_inputs.extend(kv_inputs)

        graph, graph_builder = _make_graph(graph_inputs, name="decoder")
        op = graph_builder.op

        logits, present_key_values = decoder(
            op,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_ids=input_ids,
            past_key_values=past_key_values,
        )

        logits.name = "logits"
        graph.outputs.append(logits)
        _register_kv_cache_outputs(graph, present_key_values)

        return _make_model(graph)

    def _build_vision(
        self,
        vision: nn.Module,
        config: Gemma4Config,
    ) -> ir.Model:
        """Build vision encoder: pixel_values + pixel_position_ids -> image_features.

        Inputs:
        - ``pixel_values [B, N, 3*P^2]``: pre-patchified image data
        - ``pixel_position_ids [B, N, 2]``: (x, y) patch coordinates

        Output:
        - ``image_features [B*N, text_hidden_size]``: projected vision features
        """
        batch = ir.SymbolicDim("batch")
        num_patches = ir.SymbolicDim("num_patches")
        patch_size = config.vision.patch_size or 16 if config.vision else 16
        pixel_dim = 3 * patch_size * patch_size

        pixel_values = ir.Value(
            name="pixel_values",
            shape=ir.Shape([batch, num_patches, pixel_dim]),
            type=ir.TensorType(config.dtype),
        )
        pixel_position_ids = ir.Value(
            name="pixel_position_ids",
            shape=ir.Shape([batch, num_patches, 2]),
            type=ir.TensorType(ir.DataType.INT64),
        )

        graph_inputs = [pixel_values, pixel_position_ids]

        graph, graph_builder = _make_graph(graph_inputs, name="vision")
        op = graph_builder.op

        image_features = vision(
            op,
            pixel_values=pixel_values,
            pixel_position_ids=pixel_position_ids,
        )

        image_features.name = "image_features"
        graph.outputs.append(image_features)

        return _make_model(graph)

    def _build_audio(
        self,
        audio: nn.Module,
        config: Gemma4Config,
    ) -> ir.Model:
        """Build audio encoder: input_features -> audio_features.

        Input:
        - ``input_features [batch, time, input_size]``: mel-spectrogram

        Output:
        - ``audio_features [batch, time//4, text_hidden_size]``: encoded tokens
        """
        batch = ir.SymbolicDim("batch")
        time = ir.SymbolicDim("time")
        input_size = (config.audio.input_size if config.audio else None) or 128

        input_features = ir.Value(
            name="input_features",
            shape=ir.Shape([batch, time, input_size]),
            type=ir.TensorType(config.dtype),
        )

        graph, graph_builder = _make_graph([input_features], name="audio")
        op = graph_builder.op

        audio_features = audio(op, input_features)
        audio_features.name = "audio_features"
        graph.outputs.append(audio_features)
        return _make_model(graph)

    def _build_embedding(
        self,
        embedding: nn.Module,
        config: Gemma4Config,
    ) -> ir.Model:
        """Build embedding: input_ids + image_features [+ audio_features] -> inputs_embeds."""
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")
        num_image_tokens = ir.SymbolicDim("num_image_tokens")

        input_ids = ir.Value(
            name="input_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        image_features = ir.Value(
            name="image_features",
            shape=ir.Shape([num_image_tokens, config.hidden_size]),
            type=ir.TensorType(config.dtype),
        )

        graph_inputs = [input_ids, image_features]
        audio_features_val: ir.Value | None = None

        if config.audio is not None:
            num_audio_tokens = ir.SymbolicDim("num_audio_tokens")
            audio_features_val = ir.Value(
                name="audio_features",
                shape=ir.Shape([num_audio_tokens, config.hidden_size]),
                type=ir.TensorType(config.dtype),
            )
            graph_inputs.append(audio_features_val)

        graph, graph_builder = _make_graph(graph_inputs, name="embedding")
        op = graph_builder.op

        inputs_embeds = embedding(
            op,
            input_ids=input_ids,
            image_features=image_features,
            audio_features=audio_features_val,
        )
        inputs_embeds.name = "inputs_embeds"
        graph.outputs.append(inputs_embeds)
        return _make_model(graph)
