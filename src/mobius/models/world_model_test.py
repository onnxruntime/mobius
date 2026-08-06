# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import pytest
import torch
import torch.nn.functional as functional
from onnxscript import nn

from mobius import (
    MLPWorldModel,
    WorldModelConfig,
    WorldModelTask,
    build_from_module,
)
from mobius.tasks import TASK_REGISTRY, get_task


class _TorchMLPWorldModel(torch.nn.Module):
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        input_size = config.observation_size + config.action_size + config.state_size
        self.config = config
        self.input_layer = torch.nn.Linear(input_size, config.hidden_size)
        self.hidden_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(config.hidden_size, config.hidden_size)
                for _ in range(config.num_hidden_layers - 1)
            ]
        )
        self.state_head = torch.nn.Linear(config.hidden_size, config.state_size)
        self.observation_head = torch.nn.Linear(config.hidden_size, config.observation_size)
        self.reward_head = torch.nn.Linear(config.hidden_size, 1)
        self.continuation_head = torch.nn.Linear(config.hidden_size, 1)

    def forward(self, observation, action, state):
        hidden = functional.silu(
            self.input_layer(
                torch.cat(
                    (
                        observation.flatten(start_dim=1),
                        action.flatten(start_dim=1),
                        state.flatten(start_dim=1),
                    ),
                    dim=1,
                )
            )
        )
        for layer in self.hidden_layers:
            hidden = functional.silu(layer(hidden))

        next_state = self.state_head(hidden)
        if self.config.residual_state:
            next_state = state.flatten(start_dim=1) + next_state
        return (
            next_state.reshape_as(state),
            self.observation_head(hidden).reshape_as(observation),
            self.reward_head(hidden),
            torch.sigmoid(self.continuation_head(hidden)),
        )


def _config() -> WorldModelConfig:
    return WorldModelConfig(
        observation_shape=(2, 2),
        action_shape=(2,),
        state_shape=(3,),
        hidden_size=8,
        num_hidden_layers=2,
    )


class TestWorldModelConfig:
    def test_flattened_sizes(self):
        config = _config()
        assert config.observation_size == 4
        assert config.action_size == 2
        assert config.state_size == 3

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("observation_shape", ()),
            ("action_shape", (0,)),
            ("state_shape", (True,)),
            ("hidden_size", 0),
            ("num_hidden_layers", 0),
            ("hidden_act", None),
        ],
    )
    def test_invalid_config_raises(self, field, value):
        config = _config()
        setattr(config, field, value)
        with pytest.raises(ValueError):
            config.validate()


class TestWorldModelTask:
    def test_registered(self):
        assert TASK_REGISTRY["world-model"] is WorldModelTask
        assert isinstance(get_task("world-model"), WorldModelTask)

    def test_graph_contract(self):
        config = _config()
        package = build_from_module(
            MLPWorldModel(config),
            config,
            task="world-model",
        )
        model = package["model"]

        assert [value.name for value in model.graph.inputs] == list(WorldModelTask.input_names)
        assert [value.name for value in model.graph.outputs] == list(
            WorldModelTask.output_names
        )
        assert list(model.graph.inputs[0].shape)[1:] == [2, 2]
        assert list(model.graph.inputs[1].shape)[1:] == [2]
        assert list(model.graph.inputs[2].shape)[1:] == [3]
        assert model.graph.name == "world_model_step"

    def test_rejects_wrong_module_output_contract(self):
        class InvalidWorldModel(nn.Module):
            def forward(self, op, observation, action, state):
                return op.Identity(state)

        with pytest.raises(TypeError, match="must return"):
            WorldModelTask().build(InvalidWorldModel(), _config())


def test_mlp_world_model_matches_pytorch(tmp_path):
    torch.manual_seed(7)
    config = _config()
    reference = _TorchMLPWorldModel(config).eval()
    package = build_from_module(
        MLPWorldModel(config),
        config,
        task="world-model",
        execution_provider="cpu",
    )
    package.apply_weights(dict(reference.state_dict()))
    package.save(str(tmp_path), progress_bar=False)

    session = ort.InferenceSession(
        str(tmp_path / "model.onnx"),
        providers=["CPUExecutionProvider"],
    )
    rng = np.random.default_rng(11)
    observation = rng.standard_normal((2, 2, 2)).astype(np.float32)
    action = rng.standard_normal((2, 2)).astype(np.float32)
    state = rng.standard_normal((2, 3)).astype(np.float32)

    with torch.no_grad():
        expected = reference(
            torch.from_numpy(observation),
            torch.from_numpy(action),
            torch.from_numpy(state),
        )
    actual = session.run(
        list(WorldModelTask.output_names),
        {
            "observation": observation,
            "action": action,
            "state": state,
        },
    )

    for actual_value, expected_value in zip(
        actual,
        expected,
        strict=True,
    ):
        np.testing.assert_allclose(
            actual_value,
            expected_value.numpy(),
            rtol=1e-5,
            atol=1e-6,
        )

    next_observation = rng.standard_normal((2, 2, 2)).astype(np.float32)
    next_action = rng.standard_normal((2, 2)).astype(np.float32)
    with torch.no_grad():
        expected_next = reference(
            torch.from_numpy(next_observation),
            torch.from_numpy(next_action),
            expected[0],
        )
    actual_next = session.run(
        list(WorldModelTask.output_names),
        {
            "observation": next_observation,
            "action": next_action,
            "state": actual[0],
        },
    )

    np.testing.assert_allclose(
        actual_next[0],
        expected_next[0].numpy(),
        rtol=1e-5,
        atol=1e-6,
    )
