# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Single-step latent-dynamics task."""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir
from onnxscript import nn

from mobius._configs import LatentDynamicsConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class LatentDynamicsTask(ModelTask):
    """Build a stateful one-step latent-dynamics graph.

    Inputs:
        - observation: ``[batch, *observation_shape]``
        - action: ``[batch, *action_shape]``
        - state: ``[batch, *state_shape]``

    Outputs:
        - next_state: recurrent state for the next invocation
        - observation_prediction: prediction using ``observation_shape``
        - reward: scalar reward per batch item, shaped ``[batch, 1]``
        - continuation: continuation probability, shaped ``[batch, 1]``
    """

    input_names: ClassVar[tuple[str, ...]] = ("observation", "action", "state")
    output_names: ClassVar[tuple[str, ...]] = (
        "next_state",
        "observation_prediction",
        "reward",
        "continuation",
    )
    model_roles: ClassVar[dict[str, str]] = {"model": "encoder"}

    def build(
        self,
        module: nn.Module,
        config: LatentDynamicsConfig,
    ) -> ModelPackage:
        config.validate()
        batch = ir.SymbolicDim("batch")
        graph, builder = _make_graph(name="world_model_step")

        observation_name, action_name, state_name = self.input_names

        observation = builder.input(
            observation_name,
            dtype=config.dtype,
            shape=[batch, *config.observation_shape],
        )
        action = builder.input(
            action_name,
            dtype=config.dtype,
            shape=[batch, *config.action_shape],
        )
        state = builder.input(
            state_name,
            dtype=config.dtype,
            shape=[batch, *config.state_shape],
        )

        outputs = module(
            builder.op,
            observation=observation,
            action=action,
            state=state,
        )
        if not isinstance(outputs, (tuple, list)) or len(outputs) != len(self.output_names):
            raise TypeError(
                f"{type(module).__name__} must return "
                "(next_state, observation_prediction, reward, continuation)"
            )

        for value, name in zip(outputs, self.output_names, strict=True):
            builder.add_output(value, name)

        return ModelPackage({"model": _make_model(graph)}, config=config)


# Backward-compatible alias for the task name introduced by the initial
# single-step implementation.
WorldModelTask = LatentDynamicsTask
