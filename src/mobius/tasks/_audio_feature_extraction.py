# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Audio feature extraction task.

Builds a single ONNX graph for encoder-only audio models (Wav2Vec2, HuBERT, etc.)
that take raw waveform input and produce hidden state outputs.
"""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class AudioFeatureExtractionTask(ModelTask):
    """Build ONNX graph for audio feature extraction (encoder-only)."""

    model_roles: ClassVar[dict[str, str]] = {"model": "encoder"}

    def build(
        self,
        module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        graph, builder = _make_graph()
        op = builder.op

        input_values = builder.input(
            "input_values", dtype=ir.DataType.FLOAT, shape=["batch", "time"]
        )

        last_hidden_state = module(op, input_values=input_values)

        builder.add_output(last_hidden_state, "last_hidden_state")

        return ModelPackage({"model": _make_model(graph, builder)}, config=config)
