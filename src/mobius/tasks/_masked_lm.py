# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Masked language modeling task for encoder-only models (BERT, ESM, RoBERTa, etc.)."""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import BaseModelConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class MaskedLMTask(ModelTask):
    """Encoder-only masked language modeling (predict masked tokens).

    The module must accept ``(op, input_ids, attention_mask, token_type_ids)``
    and return ``logits`` with shape ``[batch, sequence_len, vocab_size]``.

    Inputs:
        - input_ids: [batch, sequence_len] INT64
        - attention_mask: [batch, sequence_len] INT64
        - token_type_ids: [batch, sequence_len] INT64 (optional, for BERT)

    Outputs:
        - logits: [batch, sequence_len, vocab_size] FLOAT
    """

    def build(
        self,
        module: nn.Module,
        config: BaseModelConfig,
    ) -> ModelPackage:
        batch = ir.SymbolicDim("batch")
        seq_len = ir.SymbolicDim("sequence_len")

        input_ids = ir.Value(
            name="input_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        attention_mask = ir.Value(
            name="attention_mask",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )
        token_type_ids = ir.Value(
            name="token_type_ids",
            shape=ir.Shape([batch, seq_len]),
            type=ir.TensorType(ir.DataType.INT64),
        )

        graph, builder = _make_graph([input_ids, attention_mask, token_type_ids])
        op = builder.op

        logits = module(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        logits.name = "logits"
        graph.outputs.append(logits)

        return ModelPackage({"model": _make_model(graph)}, config=config)
