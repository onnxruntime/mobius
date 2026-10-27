# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Drama 3 neural graphs and ONNX sampling policies."""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir

from mobius._build_context import ep_capabilities
from mobius._configs.fish_omni import FishOmniConfig
from mobius._model_package import ModelPackage
from mobius.generation._fish_omni import (
    build_fish_codebook_sampler,
    build_fish_main_sampler,
)
from mobius.tasks._base import ComponentSpec, ModelTask, _make_graph, _make_model
from mobius.tasks._cache_utils import (
    _make_hybrid_cache_inputs,
    _make_kv_cache_inputs,
    _register_hybrid_cache_outputs,
    _register_kv_cache_outputs,
    _register_linear_attention_functions,
)


class FishOmniTask(ModelTask):
    """Drama 3 frame embedding, hybrid decoder, and temporal codebook tower."""

    model_roles: ClassVar[dict[str, str]] = {
        "embedding": "embedding",
        "decoder": "decoder",
        "audio_decoder": "decoder",
    }
    components: ClassVar[ComponentSpec] = ComponentSpec(
        embedding="embedding",
        decoder="decoder",
        audio_decoder="audio_decoder",
    )

    def build(self, module, config: FishOmniConfig) -> ModelPackage:
        if config.dtype == ir.DataType.BFLOAT16 and ep_capabilities().name != "onnx-standard":
            raise ValueError(
                "Drama 3 BF16 currently requires execution_provider='onnx-standard' "
                "and a runtime with BF16 CUDA Attention support"
            )
        self._validate_components(module)
        assert config.tower is not None
        batch, seq = "batch", "sequence_len"
        past = ir.SymbolicDim("past_sequence_len")
        models = {}
        graph, builder = _make_graph("embedding")
        frames = builder.input(
            "frames", dtype=ir.DataType.INT64, shape=[batch, seq, config.num_codebooks + 1]
        )
        embeds, unseen = module.embedding(builder.op, frames)
        builder.add_output(embeds, "inputs_embeds")
        builder.add_output(unseen, "shared_input")
        models["embedding"] = _make_model(graph)

        graph, builder = _make_graph("decoder")
        embeds = builder.input(
            "inputs_embeds", dtype=config.dtype, shape=[batch, seq, config.hidden_size]
        )
        mask = builder.input(
            "attention_mask", dtype=ir.DataType.INT64, shape=[batch, "total_sequence_len"]
        )
        positions = builder.input("position_ids", dtype=ir.DataType.INT64, shape=[batch, seq])
        decoder_caches = _make_hybrid_cache_inputs(
            builder, config, config.dtype, ir.SymbolicDim(batch), past
        )
        logits, hidden, present = module.decoder(
            builder.op, embeds, mask, positions, decoder_caches
        )
        builder.add_output(logits, "logits")
        builder.add_output(hidden, "hidden_states")
        _register_hybrid_cache_outputs(builder, present, config.layer_types or [])
        models["decoder"] = _make_model(graph)
        _register_linear_attention_functions(models["decoder"], config)

        graph, builder = _make_graph("audio_decoder")
        hidden = builder.input(
            "hidden_states", dtype=config.dtype, shape=[batch, seq, config.hidden_size]
        )
        codes = builder.input(
            "codes", dtype=ir.DataType.INT64, shape=[batch, seq, config.num_codebooks]
        )
        semantic = builder.input("semantic_mask", dtype=ir.DataType.BOOL, shape=[batch, seq])
        unseen = builder.input(
            "shared_input", dtype=config.dtype, shape=[batch, seq, config.hidden_size]
        )
        mask = builder.input(
            "attention_mask", dtype=ir.DataType.INT64, shape=[batch, "total_sequence_len"]
        )
        positions = builder.input("position_ids", dtype=ir.DataType.INT64, shape=[batch, seq])
        tower = config.tower
        tower_caches = _make_kv_cache_inputs(
            builder,
            tower.num_hidden_layers,
            tower.num_key_value_heads,
            tower.head_dim,
            config.dtype,
            ir.SymbolicDim(batch),
            past,
        )
        logits, present = module.audio_decoder(
            builder.op, hidden, codes, semantic, unseen, mask, positions, tower_caches
        )
        builder.add_output(logits, "codebook_logits")
        _register_kv_cache_outputs(builder, present)
        models["audio_decoder"] = _make_model(graph)
        for model in models.values():
            model.metadata_props["mobius.fish_omni.scope"] = (
                "Neural token graphs with ONNX sampling policies; caller orchestrates "
                "temporal decoding and prompt formatting; waveform decoder exported separately"
            )
        return ModelPackage(
            models,
            config=config,
            policy_components={
                "main_sampler": build_fish_main_sampler(config),
                "codebook_sampler": build_fish_codebook_sampler(config),
            },
        )
