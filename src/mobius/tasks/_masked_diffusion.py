# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Masked-diffusion language model task (LLaDA / Dream).

Builds an ONNX graph for a discrete masked-diffusion *mask predictor*: it maps
an int64 token sequence to per-position vocabulary logits in a single
full-sequence pass. There is no KV cache and no attention-mask input — the
model attends bidirectionally over the whole sequence. The graph also exposes
the greedy full-sequence proposal consumed by the generic workflow's
``masked_update`` policy component.
"""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import BaseModelConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class MaskedDiffusionTask(ModelTask):
    """Masked-diffusion mask predictor (no KV cache, bidirectional attention).

    Inputs:
        - input_ids: [batch, sequence_len] INT64

    Outputs:
        - logits: [batch, sequence_len, vocab_size] FLOAT
        - proposed_tokens: [batch, sequence_len] INT64
    """

    # ``encoder`` role: attention is bidirectional, so the decoder-only GQA /
    # causal fusions must not run (they would re-impose a causal mask).
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

        logits = module(op, input_ids=input_ids)
        proposed_tokens = op.ArgMax(logits, axis=-1, keepdims=0)

        builder.add_output(logits, "logits")
        builder.add_output(proposed_tokens, "proposed_tokens")

        return ModelPackage({"model": _make_model(graph)}, config=config)
