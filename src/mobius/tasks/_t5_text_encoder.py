# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""T5 prompt encoder task for diffusion pipelines."""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class T5TextEncoderTask(ModelTask):
    """Build the two-input T5 encoder contract used by diffusers."""

    model_roles: ClassVar[dict[str, str]] = {"model": "encoder"}

    def build(self, module: nn.Module, config: ArchitectureConfig) -> ModelPackage:
        graph, builder = _make_graph(name="t5_text_encoder")
        input_ids = builder.input(
            "input_ids",
            dtype=ir.DataType.INT64,
            shape=["batch", "sequence_length"],
        )
        attention_mask = builder.input(
            "attention_mask",
            dtype=ir.DataType.INT64,
            shape=["batch", "sequence_length"],
        )
        hidden_states = module(
            builder.op,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        builder.add_output(hidden_states, "last_hidden_state")
        return ModelPackage({"model": _make_model(graph)}, config=config)
