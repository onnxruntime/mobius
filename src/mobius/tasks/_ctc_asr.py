# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""CTC ASR task — builds a single ONNX graph for CTC-based ASR models.

Produces one model:
  - ``"model"``: raw audio waveform → per-frame vocabulary logits

Supports MMS (facebook/mms-1b-all) and any Wav2Vec2ForCTC checkpoint.
"""

from __future__ import annotations

import onnx_ir as ir

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class CTCAsrTask(ModelTask):
    """Build ONNX graph for CTC-based ASR (raw waveform → frame logits).

    Input:
        ``input_values``  — (batch, num_samples) raw audio at 16 kHz  FLOAT
        ``attention_mask`` — (batch, num_samples) INT64  (optional)

    Output:
        ``logits`` — (batch, num_frames, vocab_size) CTC logit scores  FLOAT
    """

    name = "ctc-asr"

    def build(
        self,
        module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        batch = ir.SymbolicDim("batch")
        time = ir.SymbolicDim("time")

        input_values = ir.Value(
            name="input_values",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([batch, time]),
        )
        attention_mask = ir.Value(
            name="attention_mask",
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape([batch, time]),
        )

        graph, builder = _make_graph([input_values, attention_mask], name="ctc_asr")
        op = builder.op

        logits = module(op, input_values=input_values, attention_mask=attention_mask)

        logits.name = "logits"
        graph.outputs.append(logits)

        return ModelPackage({"model": _make_model(graph)}, config=config)
