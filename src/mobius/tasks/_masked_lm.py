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
        - token_type_ids: [batch, sequence_len] INT64 (always emitted; pass
          zeros for models such as RoBERTa/ESM that do not use segment IDs)

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

        graph, builder = _make_graph()
        op = builder.op

        input_ids = builder.input("input_ids", dtype=ir.DataType.INT64, shape=[batch, seq_len])
        attention_mask = builder.input(
            "attention_mask", dtype=ir.DataType.INT64, shape=[batch, seq_len]
        )
        token_type_ids = builder.input(
            "token_type_ids", dtype=ir.DataType.INT64, shape=[batch, seq_len]
        )

        logits = module(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        builder.add_output(logits, "logits")

        return ModelPackage({"model": _make_model(graph)}, config=config)
