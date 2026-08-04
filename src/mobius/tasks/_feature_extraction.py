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

        builder.add_output(last_hidden_state, "last_hidden_state")

        return ModelPackage({"model": _make_model(graph)}, config=config)
