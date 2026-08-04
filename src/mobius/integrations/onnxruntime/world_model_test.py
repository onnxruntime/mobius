# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from mobius.integrations.onnxruntime import WorldModelRunner


@dataclasses.dataclass
class _Node:
    name: str
    shape: list[int | str | None]
    type: str = "tensor(float)"


class _FakeSession:
    def __init__(self):
        self.inputs = [
            _Node("observation", ["batch", 4]),
            _Node("action", ["batch", 2]),
            _Node("state", ["batch", 3]),
        ]
        self.outputs = [
            _Node("next_state", ["batch", 3]),
            _Node("observation_prediction", ["batch", 4]),
            _Node("reward", ["batch", 1]),
            _Node("continuation", ["batch", 1]),
        ]
        self.last_feed = None

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def run(self, output_names, input_feed):
        self.last_feed = input_feed
        batch = input_feed["observation"].shape[0]
        return [
            input_feed["state"] + 1.0,
            input_feed["observation"] * 2.0,
            np.zeros((batch, 1), dtype=np.float32),
            np.ones((batch, 1), dtype=np.float32),
        ]


def test_step_initializes_and_preserves_state():
    session = _FakeSession()
    runner = WorldModelRunner(session)
    observation = np.ones((2, 4), dtype=np.float64)
    action = np.ones((2, 2), dtype=np.float64)

    first = runner.step(observation, action)
    assert session.last_feed["observation"].dtype == np.float32
    np.testing.assert_array_equal(
        session.last_feed["state"],
        np.zeros((2, 3), dtype=np.float32),
    )
    np.testing.assert_array_equal(first.next_state, np.ones((2, 3), dtype=np.float32))

    second = runner.step(observation, action)
    np.testing.assert_array_equal(
        session.last_feed["state"],
        first.next_state,
    )
    np.testing.assert_array_equal(second.next_state, np.full((2, 3), 2.0, np.float32))


def test_rollout_uses_initial_state():
    runner = WorldModelRunner(_FakeSession())
    observations = np.ones((3, 1, 4), dtype=np.float32)
    actions = np.ones((3, 1, 2), dtype=np.float32)
    initial_state = np.full((1, 3), 5.0, dtype=np.float32)

    outputs = runner.rollout(
        observations,
        actions,
        initial_state=initial_state,
    )

    assert len(outputs) == 3
    np.testing.assert_array_equal(outputs[-1].next_state, np.full((1, 3), 8.0))
    assert runner.state is outputs[-1].next_state


def test_rollout_defaults_to_fresh_zero_state():
    runner = WorldModelRunner(_FakeSession())
    runner.reset(np.full((1, 3), 10.0, dtype=np.float32))

    outputs = runner.rollout(
        np.ones((2, 1, 4), dtype=np.float32),
        np.ones((2, 1, 2), dtype=np.float32),
    )

    np.testing.assert_array_equal(outputs[-1].next_state, np.full((1, 3), 2.0))


def test_rejects_contract_mismatch():
    session = _FakeSession()
    session.inputs.pop()

    with pytest.raises(ValueError, match="input contract mismatch"):
        WorldModelRunner(session)


def test_rejects_incompatible_recurrent_state_contract():
    session = _FakeSession()
    session.outputs[0].type = "tensor(double)"

    with pytest.raises(ValueError, match="next_state dtype must match"):
        WorldModelRunner(session)


def test_rejects_mismatched_batches_before_inference():
    session = _FakeSession()
    runner = WorldModelRunner(session)

    with pytest.raises(ValueError, match="batch dimensions differ"):
        runner.step(
            np.zeros((2, 4), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
        )
    assert session.last_feed is None


def test_reset_requires_explicit_dynamic_state_tail():
    session = _FakeSession()
    session.inputs[-1].shape = ["batch", "state_size"]
    runner = WorldModelRunner(session)

    with pytest.raises(ValueError, match="non-batch state dimension is dynamic"):
        runner.reset(batch_size=2)
