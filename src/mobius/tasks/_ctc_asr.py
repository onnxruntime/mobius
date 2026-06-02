# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CTC ASR task — builds a single ONNX graph for CTC-based ASR models.

Produces one model:
  - ``"model"``: raw audio waveform → per-frame vocabulary logits

Supports MMS (facebook/mms-1b-all) and any Wav2Vec2ForCTC checkpoint.
"""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class CTCAsrTask(ModelTask):
    """Build ONNX graph for CTC-based ASR (raw waveform → frame logits).

    Input:
        ``input_values``   — (batch, num_samples) raw audio at 16 kHz  FLOAT
        ``attention_mask`` — (batch, num_samples) INT64 padding mask
                             (1 = valid sample, 0 = padding). Always a
                             required graph input; callers with no
                             padding should pass an all-ones mask.

    Output:
        ``logits`` — (batch, num_frames, vocab_size) CTC logit scores  FLOAT
    """

    name = "ctc-asr"
    model_roles: ClassVar[dict[str, str]] = {"model": "encoder"}

    def build(
        self,
        module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        batch = ir.SymbolicDim("batch")
        time = ir.SymbolicDim("time")

        graph, builder = _make_graph(name="ctc_asr")
        input_values = builder.input(
            "input_values",
            dtype=ir.DataType.FLOAT,
            shape=[batch, time],
        )
        attention_mask = builder.input(
            "attention_mask",
            dtype=ir.DataType.INT64,
            shape=[batch, time],
        )

        logits = module(builder.op, input_values=input_values, attention_mask=attention_mask)
        builder.add_output(logits, "logits")

        return ModelPackage({"model": _make_model(graph)}, config=config)
