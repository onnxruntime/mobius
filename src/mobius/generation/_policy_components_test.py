# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort

from mobius._model_package import ModelPackage
from mobius.generation import (
    PolicyCapabilities,
    PolicyRole,
    attach_policy_components,
    build_eos_termination,
    build_euler_solver_step,
    build_greedy_sampler,
    build_masked_token_update,
    build_seeded_categorical_sampler,
    build_speculative_acceptance,
    build_token_state_update,
)


def _run(component, tmp_path, feeds):
    path = tmp_path / f"{component.model.graph.name}.onnx"
    ir.save(component.model, path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(None, feeds)


def test_greedy_sampler_runtime(tmp_path):
    (tokens,) = _run(
        build_greedy_sampler(),
        tmp_path,
        {"logits": np.array([[0.2, 0.7, 0.1], [2.0, 1.0, 3.0]], np.float32)},
    )
    np.testing.assert_array_equal(tokens, [1, 2])


def test_seeded_sampler_is_counter_based_and_reproducible(tmp_path):
    component = build_seeded_categorical_sampler()
    feeds = {
        "logits": np.array([[0.0, 0.0, 0.0, 0.0]], np.float32),
        "temperature": np.array([1.0], np.float32),
        "seed": np.array([7], np.int64),
        "offset": np.array([11], np.int64),
    }
    first = _run(component, tmp_path, feeds)
    second = _run(component, tmp_path, feeds)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], [12])


def test_eos_termination_runtime(tmp_path):
    (terminated,) = _run(
        build_eos_termination(),
        tmp_path,
        {
            "token_ids": np.array([2, 8, 9], np.int64),
            "eos_ids": np.array([2, 9], np.int64),
            "iteration": np.array([0, 4, 1], np.int64),
            "max_iterations": np.array([5, 5, 2], np.int64),
        },
    )
    np.testing.assert_array_equal(terminated, [True, True, True])


def test_euler_solver_runtime_parity(tmp_path):
    sample = np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2)
    derivative = np.full_like(sample, 0.25)
    (actual,) = _run(
        build_euler_solver_step(),
        tmp_path,
        {
            "sample": sample,
            "derivative": derivative,
            "step": np.array([0], np.int64),
            "schedule": np.array([1.5, 0.5], np.float32),
        },
    )
    np.testing.assert_allclose(actual, sample - derivative)


def test_masked_update_runtime_parity(tmp_path):
    outputs = _run(
        build_masked_token_update(),
        tmp_path,
        {
            "current_tokens": np.array([[1, 99, 99]], np.int64),
            "proposed_tokens": np.array([[4, 5, 6]], np.int64),
            "masked": np.array([[False, True, True]]),
            "step": np.array([0], np.int64),
        },
    )
    np.testing.assert_array_equal(outputs[0], [[1, 5, 6]])
    np.testing.assert_array_equal(outputs[1], [[False, False, False]])


def test_speculative_acceptance_prefix_runtime(tmp_path):
    accepted_tokens, count, done = _run(
        build_speculative_acceptance(),
        tmp_path,
        {
            "target_scores": np.array(
                [[[0, 1], [1, 0], [0, 1], [1, 0]]],
                np.float32,
            ),
            "proposed_tokens": np.array([[1, 0, 0, 0]], np.int64),
        },
    )
    np.testing.assert_array_equal(accepted_tokens, [[1, 0, 0, 0]])
    np.testing.assert_array_equal(count, [2])
    np.testing.assert_array_equal(done, [False])


def test_token_state_update_runtime(tmp_path):
    (next_state,) = _run(
        build_token_state_update(),
        tmp_path,
        {
            "current": np.array([[1], [3]], np.int64),
            "update": np.array([5, 7], np.int64),
        },
    )
    np.testing.assert_array_equal(next_state, [[5], [7]])


def test_capability_driven_attachment_is_model_agnostic():
    package = ModelPackage()
    artifacts = attach_policy_components(
        package,
        PolicyCapabilities(
            sampler="greedy",
            eos_termination=True,
            solver="euler",
            masked_update=True,
            speculative_acceptance=True,
            token_state_update=True,
        ),
    )

    assert artifacts == {
        "token_sampler": "policies/token_sampler.onnx",
        "termination": "policies/termination.onnx",
        "solver_step": "policies/solver_step.onnx",
        "masked_update": "policies/masked_update.onnx",
        "speculative_acceptance": "policies/speculative_acceptance.onnx",
        "token_state_update": "policies/token_state_update.onnx",
    }
    assert {component.role for component in package.policy_components.values()} == {
        PolicyRole.TOKEN_SAMPLER,
        PolicyRole.TERMINATION,
        PolicyRole.SOLVER_STEP,
        PolicyRole.MASKED_UPDATE,
        PolicyRole.SPECULATIVE_ACCEPTANCE,
        PolicyRole.STATE_UPDATE,
    }
