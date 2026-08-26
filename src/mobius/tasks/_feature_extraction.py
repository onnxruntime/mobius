# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Feature extraction task for encoder-only models (BERT, RoBERTa, etc.)."""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import BaseModelConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class FeatureExtractionTask(ModelTask):
    """Encoder-only feature extraction (no KV cache, no causal mask).

    Inputs:
        - input_ids: [batch, sequence_len] INT64
        - attention_mask: [batch, sequence_len] INT64
        - token_type_ids: [batch, sequence_len] INT64 (optional, for BERT)

    Outputs:
        - last_hidden_state: [batch, sequence_len, hidden_size] FLOAT
    """

    model_roles: ClassVar[dict[str, str]] = {"model": "encoder"}

    def build(
        self,
        module: nn.Module,
        config: BaseModelConfig,
    ) -> ModelPackage:
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")

        graph, builder = _make_graph()
        op = builder.op

        input_ids = builder.input("input_ids", dtype=ir.DataType.INT64, shape=[batch, seq_len])
        attention_mask = builder.input(
            "attention_mask", dtype=ir.DataType.INT64, shape=[batch, seq_len]
        )
        token_type_ids = builder.input(
            "token_type_ids", dtype=ir.DataType.INT64, shape=[batch, seq_len]
        )

        last_hidden_state = module(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        builder.add_output(last_hidden_state, self.output_name(config))

        return ModelPackage({"model": _make_model(graph)}, config=config)

    def output_name(self, config: BaseModelConfig) -> str:
        """Return the standard token-level feature output name."""
        return "last_hidden_state"


class GGUFEncoderFeatureExtractionTask(FeatureExtractionTask):
    """Feature extraction with the GGUF NONE/MEAN/CLS pooling output ABI."""

    def output_name(self, config: BaseModelConfig) -> str:
        return (
            "sentence_embedding"
            if getattr(config, "pooling_type", 0) in {1, 2}
            else "last_hidden_state"
        )


class GGUFEmbeddingFeatureExtractionTask(ModelTask):
    """Two-input stateless ABI for canonical GGUF embedding architectures."""

    model_roles: ClassVar[dict[str, str]] = {"model": "encoder"}

    def build(self, module: nn.Module, config: BaseModelConfig) -> ModelPackage:
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_length")
        graph, builder = _make_graph()
        input_ids = builder.input("input_ids", dtype=ir.DataType.INT64, shape=[batch, seq_len])
        attention_mask = builder.input(
            "attention_mask", dtype=ir.DataType.INT64, shape=[batch, seq_len]
        )
        output = module(builder.op, input_ids=input_ids, attention_mask=attention_mask)
        name = (
            "sentence_embedding"
            if getattr(config, "pooling_type", 0) != 0
            else "last_hidden_state"
        )
        builder.add_output(output, name)
        return ModelPackage({"model": _make_model(graph)}, config=config)
