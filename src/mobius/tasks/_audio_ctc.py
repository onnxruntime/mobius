# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Audio CTC task.

Builds a single ONNX graph for CTC-based audio models (SenseVoiceSmall, etc.)
that take pre-processed audio features and produce CTC logits.

The model performs its own query prepending and encoding internally.
The task wires up the I/O contract:

- Input: ``input_features`` (batch, time, feature_dim)
- Input: ``language_id`` (batch, 1) — integer language ID
- Output: ``logits`` (batch, time + num_query_tokens, vocab_size)
"""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class AudioCTCTask(ModelTask):
    """Build ONNX graph for CTC-based audio recognition (single model)."""

    model_roles: ClassVar[dict[str, str]] = {"model": "encoder"}

    def build(
        self,
        module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        graph, builder = _make_graph()
        op = builder.op

        audio = config.audio
        input_dim = audio.input_size or 560

        input_features = builder.input(
            "input_features",
            dtype=ir.DataType.FLOAT,
            shape=["batch", "time", input_dim],
        )
        # Audio input is always f32 (matching audio processor output).
        # Cast at graph entry for f16/bf16 builds.
        if config.dtype and config.dtype != ir.DataType.FLOAT:
            input_features = op.Cast(input_features, to=config.dtype)
        language_id = builder.input(
            "language_id",
            dtype=ir.DataType.INT64,
            shape=["batch", 1],
        )

        logits = module(op, input_features=input_features, language_id=language_id)

        builder.add_output(logits, "logits")

        return ModelPackage({"model": _make_model(graph)}, config=config)
