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
    build_boolean_not,
    build_decoder_state_initializer,
    build_decoder_step_update,
    build_eos_termination,
    build_euler_solver_step,
    build_greedy_sampler,
    build_last_token_logits,
    build_masked_token_update,
    build_model_token_cast,
    build_seeded_categorical_sampler,
    build_speculative_acceptance,
    build_token_state_update,
)
from mobius.generation._policy_components import _make_graph


def _run(component, tmp_path, feeds):
    path = tmp_path / f"{component.model.graph.name}.onnx"
    ir.save(component.model, path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(None, feeds)


def _run_model(model, tmp_path, feeds):
    path = tmp_path / f"{model.graph.name}.onnx"
    ir.save(model, path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(None, feeds)


def test_greedy_sampler_runtime(tmp_path):
    (tokens,) = _run(
        build_greedy_sampler(),
        tmp_path,
        {"logits": np.array([[0.2, 0.7, 0.1], [2.0, 1.0, 3.0]], np.float32)},
    )
    np.testing.assert_array_equal(tokens, [1, 2])


def test_last_token_logits_and_continue_predicate_runtime(tmp_path):
    logits = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    (last,) = _run(build_last_token_logits(), tmp_path, {"logits": logits})
    np.testing.assert_array_equal(last, logits[:, -1, :])

    (continued,) = _run(
        build_boolean_not(),
        tmp_path,
        {"done": np.array([True, False])},
    )
    np.testing.assert_array_equal(continued, [False, True])


def test_decoder_state_initializer_and_step_update_runtime(tmp_path):
    inputs = [
        ir.Value(
            name="input_ids",
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape(["batch", "sequence"]),
        ),
        ir.Value(
            name="attention_mask",
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape(["batch", "past_sequence + sequence"]),
        ),
        ir.Value(
            name="position_ids",
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape(["batch", "sequence"]),
        ),
        ir.Value(
            name="past_key_values.0.key",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape(["batch", 2, "past_sequence", 4]),
        ),
    ]
    decoder = ir.Model(ir.Graph(inputs, [], nodes=[], name="decoder"), ir_version=11)
    initializer = build_decoder_state_initializer(
        decoder,
        token_input="input_ids",
        attention_mask_input="attention_mask",
        position_ids_input="position_ids",
        cache_inputs=["past_key_values.0.key"],
    )
    outputs = _run(
        initializer,
        tmp_path,
        {"prompt_tokens": np.array([[3, 4, 5]], np.int64)},
    )
    np.testing.assert_array_equal(outputs[0], [[1, 1, 1]])
    np.testing.assert_array_equal(outputs[1], [[0, 1, 2]])
    np.testing.assert_array_equal(outputs[2], [[1, 1, 1, 1]])
    np.testing.assert_array_equal(outputs[3], [[3]])
    assert outputs[5].shape == (1, 2, 0, 4)

    updated = _run(
        build_decoder_step_update(
            attention_dtype=ir.DataType.INT64,
            position_dtype=ir.DataType.INT64,
        ),
        tmp_path,
        {
            "attention_mask": outputs[2],
            "position_ids": outputs[3],
        },
    )
    np.testing.assert_array_equal(updated[0], [[1, 1, 1, 1, 1]])
    np.testing.assert_array_equal(updated[1], [[4]])
    (cast_token,) = _run(
        build_model_token_cast(ir.DataType.INT32),
        tmp_path,
        {"token": np.array([[8]], np.int64)},
    )
    assert cast_token.dtype == np.int32
    np.testing.assert_array_equal(cast_token, [[8]])


def test_decoder_policy_chain_generates_multiple_tokens_from_prompt_only(tmp_path):
    graph, builder = _make_graph("decoder_stub")
    op = builder.op
    input_ids = builder.input("input_ids", ir.DataType.INT64, ["batch", "sequence"])
    builder.input("attention_mask", ir.DataType.INT64, ["batch", "context"])
    builder.input("position_ids", ir.DataType.INT64, ["batch", "sequence"])
    past = builder.input(
        "past_key_values.0.key",
        ir.DataType.FLOAT,
        ["batch", 2, "past_sequence", 4],
    )
    logits_shape = op.Concat(op.Shape(input_ids), op.Constant(value_ints=[8]), axis=0)
    logits = op.ConstantOfShape(
        logits_shape,
        value=ir.tensor([0.0], dtype=ir.DataType.FLOAT),
    )
    logits.shape = ir.Shape(["batch", "sequence", 8])
    builder.add_output(logits, "logits")
    builder.add_output(op.Identity(past), "present.0.key")
    decoder = ir.Model(graph, ir_version=11)

    prompt = np.array([[4, 5]], np.int64)
    max_output_tokens = np.array([3], np.int64)
    initialized = _run(
        build_decoder_state_initializer(
            decoder,
            token_input="input_ids",
            attention_mask_input="attention_mask",
            position_ids_input="position_ids",
            cache_inputs=["past_key_values.0.key"],
        ),
        tmp_path,
        {"prompt_tokens": prompt},
    )
    attention, positions, body_attention, body_position, token, cache = initialized
    logits, cache = _run_model(
        decoder,
        tmp_path,
        {
            "input_ids": prompt,
            "attention_mask": attention,
            "position_ids": positions,
            "past_key_values.0.key": cache,
        },
    )

    emitted = []
    for iteration in range(3):
        (last,) = _run(build_last_token_logits(), tmp_path, {"logits": logits})
        (sample,) = _run(build_greedy_sampler(), tmp_path, {"logits": last})
        emitted.append(int(sample[0]))
        (done,) = _run(
            build_eos_termination(),
            tmp_path,
            {
                "token_ids": sample,
                "eos_ids": np.array([7], np.int64),
                "iteration": np.array([iteration], np.int64),
                "max_iterations": max_output_tokens,
            },
        )
        (token,) = _run(
            build_token_state_update(),
            tmp_path,
            {"current": token, "update": sample},
        )
        logits, cache = _run_model(
            decoder,
            tmp_path,
            {
                "input_ids": token,
                "attention_mask": body_attention,
                "position_ids": body_position,
                "past_key_values.0.key": cache,
            },
        )
        body_attention, body_position = _run(
            build_decoder_step_update(
                attention_dtype=ir.DataType.INT64,
                position_dtype=ir.DataType.INT64,
            ),
            tmp_path,
            {
                "attention_mask": body_attention,
                "position_ids": body_position,
            },
        )
        if done[0]:
            break

    assert emitted == [0, 0, 0]
    assert done.tolist() == [True]


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
    logits = np.zeros((1, 3, 7), dtype=np.float32)
    logits[0, 1, 5] = 1.0
    logits[0, 2, 6] = 4.0
    outputs = _run(
        build_masked_token_update(),
        tmp_path,
        {
            "current_tokens": np.array([[1, 99, 99]], np.int64),
            "proposed_tokens": np.array([[4, 5, 6]], np.int64),
            "logits": logits,
            "masked": np.array([[False, True, True]]),
            "step": np.array([0], np.int64),
            "total_steps": np.array([2], np.int64),
            "seed": np.array([7], np.int64),
            "offset": np.array([11], np.int64),
        },
    )
    np.testing.assert_array_equal(outputs[0], [[1, 99, 6]])
    np.testing.assert_array_equal(outputs[1], [[False, True, False]])
    np.testing.assert_array_equal(outputs[2], [14])
    np.testing.assert_array_equal(outputs[3], [False])

    final = _run(
        build_masked_token_update(),
        tmp_path,
        {
            "current_tokens": outputs[0],
            "proposed_tokens": np.array([[4, 5, 6]], np.int64),
            "logits": logits,
            "masked": outputs[1],
            "step": np.array([1], np.int64),
            "total_steps": np.array([2], np.int64),
            "seed": np.array([7], np.int64),
            "offset": outputs[2],
        },
    )
    np.testing.assert_array_equal(final[0], [[1, 5, 6]])
    np.testing.assert_array_equal(final[1], [[False, False, False]])
    np.testing.assert_array_equal(final[3], [True])


def test_speculative_acceptance_prefix_runtime(tmp_path):
    accepted_tokens, count, done, next_offset = _run(
        build_speculative_acceptance(),
        tmp_path,
        {
            "target_scores": np.array(
                [[[0, 1], [1, 0], [0, 1], [1, 0]]],
                np.float32,
            ),
            "proposed_tokens": np.array([[1, 0, 0, 0]], np.int64),
            "seed": np.array([3], np.int64),
            "offset": np.array([8], np.int64),
        },
    )
    np.testing.assert_array_equal(accepted_tokens, [[1, 0, 0, 0]])
    np.testing.assert_array_equal(count, [2])
    np.testing.assert_array_equal(done, [False])
    np.testing.assert_array_equal(next_offset, [12])


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
