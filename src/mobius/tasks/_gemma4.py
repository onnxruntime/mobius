# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Gemma4 vision-language task (3-model split).

Builds three separate ONNX models:
1. **decoder** (text decoder): ``inputs_embeds`` -> logits + KV cache
2. **vision** (vision encoder): ``pixel_values, pixel_position_ids`` -> ``image_features``
3. **embedding**: ``input_ids, image_features`` -> ``inputs_embeds``

The decoder uses per-layer KV cache where local (sliding_attention) layers
use ``config.head_dim`` and global (full_attention) layers use
``config.global_head_dim``.
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import Gemma4Config
from mobius._model_package import ModelPackage
from mobius.tasks._base import (
    ModelTask,
    _make_graph,
    _make_model,
)
from mobius.tasks._cache_utils import (
    _register_kv_cache_outputs,
)


def _make_gemma4_kv_cache_inputs(
    config: Gemma4Config,
    batch: ir.SymbolicDim,
    past_seq_len: ir.SymbolicDim,
) -> tuple[list[ir.Value], list[tuple[ir.Value, ir.Value]]]:
    """Create per-layer KV cache inputs accounting for dual head_dim.

    Local (sliding_attention) layers use ``config.head_dim``;
    global (full_attention) layers use ``config.global_head_dim``.
    """
    local_head_dim = config.head_dim
    global_head_dim = config.global_head_dim or config.head_dim
    layer_types = config.layer_types or ["sliding_attention"] * config.num_hidden_layers

    flat: list[ir.Value] = []
    pairs: list[tuple[ir.Value, ir.Value]] = []
    for i, layer_type in enumerate(layer_types):
        hd = global_head_dim if layer_type == "full_attention" else local_head_dim
        past_key = ir.Value(
            name=f"past_key_values.{i}.key",
            shape=ir.Shape([batch, config.num_key_value_heads, past_seq_len, hd]),
            type=ir.TensorType(config.dtype),
        )
        past_value = ir.Value(
            name=f"past_key_values.{i}.value",
            shape=ir.Shape([batch, config.num_key_value_heads, past_seq_len, hd]),
            type=ir.TensorType(config.dtype),
        )
        flat.extend([past_key, past_value])
        pairs.append((past_key, past_value))
    return flat, pairs


class Gemma4VisionLanguageTask(ModelTask):
    """3-model split task for Gemma4 vision-language models.

    The module must expose three sub-modules:

    - ``decoder``: text decoder accepting ``inputs_embeds``
    - ``vision_encoder``: vision encoder accepting ``pixel_values, pixel_position_ids``
    - ``embedding``: embedding model fusing ``input_ids`` and ``image_features``

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
        models["embedding"] = self._build_embedding(module.embedding, config)
        return ModelPackage(models, config=config)

    def _build_decoder(
        self,
        decoder: nn.Module,
        config: Gemma4Config,
    ) -> ir.Model:
        """Build text decoder: inputs_embeds -> logits + per-layer KV cache."""
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

        graph_inputs = [inputs_embeds, attention_mask, position_ids]

        kv_inputs, past_key_values = _make_gemma4_kv_cache_inputs(
            config, batch, past_seq_len
        )
        graph_inputs.extend(kv_inputs)

        graph, graph_builder = _make_graph(graph_inputs, name="decoder")
        op = graph_builder.op

        logits, present_key_values = decoder(
            op,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
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

    def _build_embedding(
        self,
        embedding: nn.Module,
        config: Gemma4Config,
    ) -> ir.Model:
        """Build embedding model: input_ids + image_features -> inputs_embeds."""
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

        graph, graph_builder = _make_graph(graph_inputs, name="embedding")
        op = graph_builder.op

        inputs_embeds = embedding(
            op,
            input_ids=input_ids,
            image_features=image_features,
        )

        inputs_embeds.name = "inputs_embeds"
        graph.outputs.append(inputs_embeds)

        return _make_model(graph)
