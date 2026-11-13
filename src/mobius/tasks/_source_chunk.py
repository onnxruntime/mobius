# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Manual-only task interfaces for explicit-phase numerical candidate decoders."""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import (
    ModelTask,
    _make_graph,
    _make_model,
    build_embedding_from_features,
)
from mobius.tasks._cache_utils import (
    _make_hybrid_cache_inputs,
    _register_hybrid_cache_outputs,
    _register_linear_attention_functions,
)
from mobius.tasks._nemotron_h_omni import NemotronHOmniTask


def _phase_decoder(module: nn.Module, config: ArchitectureConfig, input_name: str) -> ir.Model:
    if config.dtype != ir.DataType.BFLOAT16:
        raise TypeError("Source-chunk candidate task requires BF16 activations")
    graph, builder = _make_graph()
    shape = ["batch", "sequence_len"]
    tokens = builder.input(
        input_name,
        dtype=ir.DataType.INT64 if input_name == "input_ids" else config.dtype,
        shape=shape if input_name == "input_ids" else [*shape, config.hidden_size],
    )
    mask = builder.input(
        "attention_mask",
        dtype=ir.DataType.INT64,
        shape=["batch", "past_sequence_len + sequence_len"],
    )
    positions = builder.input("position_ids", dtype=ir.DataType.INT64, shape=shape)
    phase = builder.input("has_previous_state", dtype=ir.DataType.BOOL, shape=[])
    caches = _make_hybrid_cache_inputs(
        builder,
        config,
        config.dtype,
        ir.SymbolicDim("batch"),
        ir.SymbolicDim("past_sequence_len"),
    )
    logits, present = module(
        builder.op,
        **{input_name: tokens},
        attention_mask=mask,
        position_ids=positions,
        past_key_values=caches,
        has_previous_state=phase,
    )
    builder.add_output(logits, "logits")
    _register_hybrid_cache_outputs(builder, present, config.layer_types or [])
    model = _make_model(graph)
    phase_inputs = [
        value for value in model.graph.inputs if value.name == "has_previous_state"
    ]
    if (
        len(phase_inputs) != 1
        or phase_inputs[0].dtype != ir.DataType.BOOL
        or phase_inputs[0].shape != ir.Shape([])
        or not phase_inputs[0].uses()
    ):
        raise ValueError(
            "Source-chunk task requires an actual consumed BOOL[] cache-phase input"
        )
    _register_linear_attention_functions(model, config)
    model.metadata_props["mobius.experimental.cache_phase"] = "explicit-bool-v1"
    model.metadata_props["mobius.experimental.source_chunk"] = (
        "numerical-candidate; ordinary fresh caller only; no prompt chunking, "
        "cached S>1, rewind, scheduler, Engine or CUDA Graph capture"
    )
    return model


class SourceChunkHybridCausalLMTask(ModelTask):
    """Explicit opt-in text ABI; no registration or default task replacement."""

    model_roles: ClassVar[dict[str, str]] = {"model": "decoder"}

    def build(self, module: nn.Module, config: ArchitectureConfig) -> ModelPackage:
        return ModelPackage(
            {"model": _phase_decoder(module, config, "input_ids")}, config=config
        )


class SourceChunkNemotronHOmniTask(NemotronHOmniTask):
    """Keep vision/embedding ordinary and add BOOL[] only to the decoder."""

    def build(self, module: nn.Module, config: ArchitectureConfig) -> ModelPackage:
        from mobius.models import NemotronHOmniSourceChunkModel

        if not isinstance(module, NemotronHOmniSourceChunkModel):
            raise TypeError("Source-chunk Omni task requires the explicit candidate model")
        self._validate_components(module)
        return ModelPackage(
            {
                "decoder": _phase_decoder(module.decoder, config, "inputs_embeds"),
                "vision_encoder": self._build_vision(module.vision_encoder, config),
                "embedding": build_embedding_from_features(
                    module.embedding,
                    config,
                    feature_name="image_features",
                    feature_dim=config.hidden_size,
                ),
            },
            config=config,
        )
