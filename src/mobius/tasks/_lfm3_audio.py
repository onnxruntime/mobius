# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Four-component offline LFM3 Audio export with hybrid language-model cache."""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir
from onnxscript import nn

from mobius._configs.lfm3_audio import Lfm3AudioConfig
from mobius._model_package import ModelPackage
from mobius.models.lfm3_audio import Lfm3AudioForConditionalGeneration
from mobius.tasks._base import (
    ComponentConfig,
    ComponentSpec,
    ModelTask,
    _make_graph,
    _make_model,
)
from mobius.tasks._cache_utils import (
    _make_hybrid_cache_inputs,
    _register_hybrid_cache_outputs,
    _register_linear_attention_functions,
)


class Lfm3AudioTask(ModelTask):
    """Export causal encoder, media embedding, hybrid decoder and MaskGIT pass."""

    components = ComponentSpec(
        audio_encoder="audio_encoder",
        embedding="embedding",
        decoder="decoder",
        depth_decoder=ComponentConfig("depth_decoder", "head"),
    )
    model_roles: ClassVar[dict[str, str]] = {
        "audio_encoder": "encoder",
        "embedding": "embedding",
        "decoder": "decoder",
        "depth_decoder": "head",
    }

    def build(self, module: nn.Module, config: Lfm3AudioConfig) -> ModelPackage:
        self._validate_components(module)
        if not isinstance(module, Lfm3AudioForConditionalGeneration):
            raise TypeError(
                "Lfm3AudioTask requires Lfm3AudioForConditionalGeneration; "
                "use task='lfm3-audio' with the composite audio model."
            )
        batch, sequence = "batch", "sequence_len"
        graph, builder = _make_graph(name="audio_encoder")
        features = builder.input(
            "input_features",
            dtype=ir.DataType.FLOAT,
            shape=[batch, config.num_mel_bins, "mel_frames"],
        )
        lengths = builder.input("feature_lengths", dtype=ir.DataType.INT64, shape=[batch])
        encoded, output_lengths = module.audio_encoder(builder.op, features, lengths)
        builder.add_output(encoded, "audio_features")
        builder.add_output(output_lengths, "audio_feature_lengths")
        models = {"audio_encoder": _make_model(graph)}

        graph, builder = _make_graph(name="embedding")
        ids = builder.input("input_ids", dtype=ir.DataType.INT64, shape=[batch, sequence])
        audio = builder.input(
            "audio_features", dtype=config.dtype, shape=["audio_tokens", config.hidden_size]
        )
        codes = builder.input(
            "audio_out_codes",
            dtype=ir.DataType.INT64,
            shape=["audio_frames", config.num_codebooks],
        )
        builder.add_output(module.embedding(builder.op, ids, audio, codes), "inputs_embeds")
        models["embedding"] = _make_model(graph)

        graph, builder = _make_graph(name="decoder")
        embeds = builder.input(
            "inputs_embeds", dtype=config.dtype, shape=[batch, sequence, config.hidden_size]
        )
        mask = builder.input(
            "attention_mask", dtype=ir.DataType.INT64, shape=[batch, "total_sequence_len"]
        )
        positions = builder.input(
            "position_ids", dtype=ir.DataType.INT64, shape=[batch, sequence]
        )
        cache = _make_hybrid_cache_inputs(
            builder,
            config,
            config.dtype,
            ir.SymbolicDim(batch),
            ir.SymbolicDim("past_sequence_len"),
        )
        logits, hidden, present = module.decoder(builder.op, embeds, mask, positions, cache)
        builder.add_output(logits, "logits")
        builder.add_output(hidden, "hidden_states")
        _register_hybrid_cache_outputs(builder, present, config.layer_types or [])
        decoder = _make_model(graph)
        _register_linear_attention_functions(decoder, config)
        models["decoder"] = decoder

        graph, builder = _make_graph(name="depth_decoder")
        hidden = builder.input(
            "hidden_states", dtype=config.dtype, shape=["audio_frames", config.hidden_size]
        )
        codes = builder.input(
            "codes", dtype=ir.DataType.INT64, shape=["audio_frames", config.num_codebooks]
        )
        masked = builder.input(
            "masked", dtype=ir.DataType.BOOL, shape=["audio_frames", config.num_codebooks]
        )
        builder.add_output(module.depth_decoder(builder.op, hidden, codes, masked), "logits")
        models["depth_decoder"] = _make_model(graph)
        return ModelPackage(models, config=config)
