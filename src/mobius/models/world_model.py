# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Minimal directly declared latent-dynamics implementation."""

from __future__ import annotations

import torch
from onnxscript import OpBuilder, nn

from mobius._configs import LatentDynamicsConfig
from mobius.components import Linear, get_activation


class MLPLatentDynamicsModel(nn.Module):
    """Deterministic MLP reference model for the latent-dynamics contract."""

    default_task = "latent-dynamics"
    config_class = LatentDynamicsConfig
    category = "World Model / Dynamics"

    def __init__(self, config: LatentDynamicsConfig):
        super().__init__()
        config.validate()
        self.config = config
        input_size = config.observation_size + config.action_size + config.state_size
        self.input_layer = Linear(input_size, config.hidden_size)
        self.hidden_layers = nn.ModuleList(
            [
                Linear(config.hidden_size, config.hidden_size)
                for _ in range(config.num_hidden_layers - 1)
            ]
        )
        self.state_head = Linear(config.hidden_size, config.state_size)
        self.observation_head = Linear(config.hidden_size, config.observation_size)
        self.reward_head = Linear(config.hidden_size, 1)
        self.continuation_head = Linear(config.hidden_size, 1)
        self._activation = get_activation(config.hidden_act)

    def forward(self, op: OpBuilder, observation, action, state):
        observation_flat = op.Flatten(observation, axis=1)
        action_flat = op.Flatten(action, axis=1)
        state_flat = op.Flatten(state, axis=1)

        hidden = self._activation(
            op,
            self.input_layer(
                op,
                op.Concat(observation_flat, action_flat, state_flat, axis=1),
            ),
        )
        for layer in self.hidden_layers:
            hidden = self._activation(op, layer(op, hidden))

        next_state_flat = self.state_head(op, hidden)
        if self.config.residual_state:
            next_state_flat = op.Add(state_flat, next_state_flat)

        observation_prediction_flat = self.observation_head(op, hidden)
        reward = self.reward_head(op, hidden)
        continuation = op.Sigmoid(self.continuation_head(op, hidden))

        next_state = self._reshape_batch(
            op,
            next_state_flat,
            state,
            self.config.state_shape,
        )
        observation_prediction = self._reshape_batch(
            op,
            observation_prediction_flat,
            observation,
            self.config.observation_shape,
        )
        return next_state, observation_prediction, reward, continuation

    @staticmethod
    def _reshape_batch(op: OpBuilder, value, batch_source, shape: tuple[int, ...]):
        batch = op.Shape(batch_source, start=0, end=1)
        return op.Reshape(value, op.Concat(batch, list(shape), axis=0))

    def preprocess_weights(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Return weights unchanged; provided for parity with other Mobius models."""
        return state_dict


# Kept for source compatibility. This model is one possible dynamics component,
# not a complete world-model pipeline.
MLPWorldModel = MLPLatentDynamicsModel
