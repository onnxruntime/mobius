# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort

from mobius._model_package import ModelPackage
from mobius.generation import (
    PolicyCapabilities,
    attach_policy_components,
    build_adaptive_k_policy,
    build_batch_minimum,
    build_boolean_not,
    build_code_frame_update,
    build_decoder_state_initializer,
    build_decoder_step_update,
    build_empty_features,
    build_eos_termination,
    build_euler_model_input,
    build_euler_solver_step,
    build_grammar_logits_processor,
    build_greedy_sampler,
    build_integer_minimum,
    build_last_token_logits,
    build_masked_token_update,
    build_model_token_cast,
    build_proposal_metrics,
    build_seeded_categorical_sampler,
    build_speculative_acceptance,
    build_speculative_state_rollback,
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


def test_grammar_logits_processor_applies_mask_and_forced_tokens(tmp_path):
    logits = np.array([[1.0, 4.0, 3.0], [5.0, 2.0, 1.0]], np.float32)
    (tokens,) = _run(
        build_grammar_logits_processor(),
        tmp_path,
        {
            "logits": logits,
            "logits_mask": np.array([[True, False, True], [True, True, True]], np.bool_),
            "forced_tokens": np.array([[0], [2]], np.int64),
            "forced_length": np.array([0, 1], np.int64),
        },
    )
    np.testing.assert_array_equal(tokens, [[2], [2]])


def test_adaptive_k_policy_probes_and_keeps_faster_adjacent_width(tmp_path):
    max_k = 4
    k_slots = max_k + 1
    current_k = np.array([2], np.int64)
    estimates = np.zeros((1, 4 * k_slots + 4), np.float32)

    def observe(*, evaluated, accepted, committed, draft_ms, target_ms):
        nonlocal current_k, estimates
        current_k, estimates = _run(
            build_adaptive_k_policy(max_k=max_k),
            tmp_path,
            {
                "current_k": current_k,
                "accepted": np.array([accepted], np.int64),
                "evaluated": np.array([evaluated], np.int64),
                "committed_tokens": np.array([committed], np.int64),
                "filled_proposal_budget": np.array([True], np.bool_),
                "draft_ms": np.array([draft_ms], np.float32),
                "target_ms": np.array([target_ms], np.float32),
                "estimates": estimates,
            },
        )

    observe(evaluated=2, accepted=2, committed=3, draft_ms=1.0, target_ms=2.0)
    observe(evaluated=2, accepted=2, committed=3, draft_ms=1.0, target_ms=2.0)
    np.testing.assert_array_equal(current_k, [3])
    np.testing.assert_array_equal(estimates[:, 4 * k_slots], [2.0])

    observe(evaluated=3, accepted=3, committed=4, draft_ms=1.0, target_ms=2.0)
    observe(evaluated=3, accepted=3, committed=4, draft_ms=1.0, target_ms=2.0)
    np.testing.assert_array_equal(current_k, [3])
    np.testing.assert_array_equal(estimates[:, 4 * k_slots], [0.0])
    np.testing.assert_array_equal(estimates[:, 4 * k_slots + 3], [1.0])
    assert estimates[0, 3] / estimates[0, k_slots + 3] > 1.0


def test_speculative_guidance_length_and_budget_math(tmp_path):
    (minimum,) = _run(
        build_integer_minimum(),
        tmp_path,
        {
            "left": np.array([3, 1], np.int64),
            "right": np.array([2, 4], np.int64),
        },
    )
    evaluated, filled = _run(
        build_proposal_metrics(),
        tmp_path,
        {
            "proposed_tokens": np.zeros((2, 3), np.int64),
            "requested_k": np.array([3, 2], np.int64),
        },
    )
    np.testing.assert_array_equal(minimum, [2, 1])
    np.testing.assert_array_equal(evaluated, [3, 3])
    np.testing.assert_array_equal(filled, [True, False])
    (synchronized,) = _run(
        build_batch_minimum(),
        tmp_path,
        {"values": np.array([2, 1, 3], np.int64)},
    )
    np.testing.assert_array_equal(synchronized, [1])


def test_adaptive_k_ignores_invalid_telemetry_per_batch(tmp_path):
    estimates = np.arange(48, dtype=np.float32).reshape(2, 24)
    next_k, next_estimates = _run(
        build_adaptive_k_policy(max_k=4),
        tmp_path,
        {
            "current_k": np.array([2, 3], np.int64),
            "accepted": np.array([2, 2], np.int64),
            "evaluated": np.array([0, 3], np.int64),
            "committed_tokens": np.array([3, 0], np.int64),
            "filled_proposal_budget": np.array([True, True], np.bool_),
            "draft_ms": np.array([1.0, np.nan], np.float32),
            "target_ms": np.array([2.0, 2.0], np.float32),
            "estimates": estimates,
        },
    )
    np.testing.assert_array_equal(next_k, [2, 3])
    np.testing.assert_array_equal(next_estimates, estimates)


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

    half_logits = logits.astype(np.float16)
    (last_float,) = _run(
        build_last_token_logits(ir.DataType.FLOAT16),
        tmp_path,
        {"logits": half_logits},
    )
    assert last_float.dtype == np.float32
    np.testing.assert_array_equal(last_float, logits[:, -1, :])

    (continued,) = _run(
        build_boolean_not(),
        tmp_path,
        {"done": np.array([True, False])},
    )
    np.testing.assert_array_equal(continued, [False])


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


def test_decoder_fixed_capacity_state_matches_native_capture_layout(tmp_path):
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
            name="past_key_values.0.key",
            type=ir.TensorType(ir.DataType.FLOAT16),
            shape=ir.Shape(["batch", 2, "past_sequence", 4]),
        ),
    ]
    decoder = ir.Model(ir.Graph(inputs, [], nodes=[], name="decoder"), ir_version=11)
    outputs = _run(
        build_decoder_state_initializer(
            decoder,
            token_input="input_ids",
            attention_mask_input="attention_mask",
            position_ids_input=None,
            cache_inputs=["past_key_values.0.key"],
            fixed_capacity=True,
        ),
        tmp_path,
        {
            "prompt_tokens": np.array([[3, 4, 5]], np.int64),
            "max_iterations": np.array([2], np.int64),
        },
    )
    attention, body_attention, token, cache_lengths, cache = outputs
    np.testing.assert_array_equal(attention, [[1, 1, 1, 0, 0]])
    np.testing.assert_array_equal(body_attention, [[1, 1, 1, 1, 0]])
    np.testing.assert_array_equal(token, [[0]])
    np.testing.assert_array_equal(cache_lengths, [3])
    assert cache.shape == (1, 2, 5, 4)

    (next_attention,) = _run(
        build_decoder_step_update(
            attention_dtype=ir.DataType.INT64,
            position_dtype=None,
            fixed_capacity=True,
        ),
        tmp_path,
        {
            "attention_mask": body_attention,
            "logical_length": np.array([4], np.int64),
        },
    )
    np.testing.assert_array_equal(next_attention, [[1, 1, 1, 1, 1]])


def test_empty_features_runtime(tmp_path):
    (features,) = _run(
        build_empty_features(ir.DataType.FLOAT16, 64),
        tmp_path,
        {},
    )
    assert features.dtype == np.float16
    assert features.shape == (0, 64)


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
        done, _continue = _run(
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
        "top_k": np.array([0], np.int64),
        "top_p": np.array([1.0], np.float32),
        "min_p": np.array([0.0], np.float32),
        "grammar_mask": np.array([[True, True, True, True]], np.bool_),
        "seed": np.array([7], np.int64),
        "offset": np.array([11], np.int64),
    }
    first = _run(component, tmp_path, feeds)
    second = _run(component, tmp_path, feeds)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], [12])


def test_seeded_sampler_applies_request_top_k_and_grammar_mask(tmp_path):
    (token, _) = _run(
        build_seeded_categorical_sampler(),
        tmp_path,
        {
            "logits": np.array([[10.0, 9.0, 8.0, 7.0]], np.float32),
            "temperature": np.array([0.7], np.float32),
            "top_k": np.array([1], np.int64),
            "top_p": np.array([0.5], np.float32),
            "min_p": np.array([0.0], np.float32),
            "grammar_mask": np.array([[False, True, True, True]], np.bool_),
            "seed": np.array([17], np.int64),
            "offset": np.array([0], np.int64),
        },
    )
    np.testing.assert_array_equal(token, [1])


def test_seeded_sampler_rejects_empty_grammar_vocabulary(tmp_path):
    (token, _) = _run(
        build_seeded_categorical_sampler(),
        tmp_path,
        {
            "logits": np.array([[1.0, 2.0, 3.0]], np.float32),
            "temperature": np.array([1.0], np.float32),
            "top_k": np.array([0], np.int64),
            "top_p": np.array([1.0], np.float32),
            "min_p": np.array([0.0], np.float32),
            "grammar_mask": np.array([[False, False, False]], np.bool_),
            "seed": np.array([1], np.int64),
            "offset": np.array([0], np.int64),
        },
    )
    np.testing.assert_array_equal(token, [-1])


def test_seeded_sampler_applies_request_min_p_in_logit_space(tmp_path):
    (token, next_offset) = _run(
        build_seeded_categorical_sampler(),
        tmp_path,
        {
            "logits": np.array([[0.0, -0.1, -10.0]], np.float32),
            "temperature": np.array([1.0], np.float32),
            "top_k": np.array([0], np.int64),
            "top_p": np.array([1.0], np.float32),
            "min_p": np.array([0.95], np.float32),
            "grammar_mask": np.array([[True, True, True]], np.bool_),
            "seed": np.array([23], np.int64),
            "offset": np.array([4], np.int64),
        },
    )
    np.testing.assert_array_equal(token, [0])
    np.testing.assert_array_equal(next_offset, [5])


def test_eos_termination_runtime(tmp_path):
    terminated, continued = _run(
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
    np.testing.assert_array_equal(continued, [False])


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
    np.testing.assert_array_equal(outputs[4], [True])

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
    np.testing.assert_array_equal(final[4], [False])


def test_speculative_acceptance_prefix_runtime(tmp_path):
    (
        accepted_tokens,
        count,
        done,
        next_offset,
        rollback_len,
        continued,
    ) = _run(
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
    np.testing.assert_array_equal(accepted_tokens, [[1, 0, 1, 0]])
    np.testing.assert_array_equal(count, [3])
    np.testing.assert_array_equal(done, [False])
    np.testing.assert_array_equal(next_offset, [12])
    np.testing.assert_array_equal(rollback_len, [2])
    np.testing.assert_array_equal(continued, [True])
    (corrected_cache,) = _run(
        build_speculative_state_rollback(
            ir.DataType.FLOAT,
            ["batch", 1, "past_sequence", 2],
            sequence_axis=2,
        ),
        tmp_path,
        {
            "past_state": np.zeros((1, 1, 2, 2), np.float32),
            "tentative_state": np.zeros((1, 1, 6, 2), np.float32),
            "accepted_len": rollback_len,
        },
    )
    assert corrected_cache.shape[2] == 4


def test_speculative_acceptance_preserves_per_row_prefixes(tmp_path):
    (
        accepted_tokens,
        count,
        done,
        _,
        rollback_len,
        continued,
    ) = _run(
        build_speculative_acceptance(),
        tmp_path,
        {
            "target_scores": np.array(
                [
                    [[0, 1], [1, 0], [0, 1], [1, 0]],
                    [[0, 1], [1, 0], [0, 1], [1, 0]],
                ],
                np.float32,
            ),
            "proposed_tokens": np.array([[1, 1, 0, 0], [1, 0, 1, 0]], np.int64),
            "seed": np.array([3, 4], np.int64),
            "offset": np.array([0, 0], np.int64),
        },
    )
    np.testing.assert_array_equal(count, [2, 4])
    np.testing.assert_array_equal(
        accepted_tokens,
        [[1, 0, 0, 0], [1, 0, 1, 0]],
    )
    np.testing.assert_array_equal(done, [False, True])
    np.testing.assert_array_equal(rollback_len, [1, 4])
    np.testing.assert_array_equal(continued, [True, False])


def test_speculative_state_rollback_trims_tentative_cache(tmp_path):
    past = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
    tentative = np.arange(12, dtype=np.float32).reshape(1, 1, 6, 2)
    (corrected,) = _run(
        build_speculative_state_rollback(
            ir.DataType.FLOAT,
            ["batch", 1, "past_sequence", 2],
            sequence_axis=2,
        ),
        tmp_path,
        {
            "past_state": past,
            "tentative_state": tentative,
            "accepted_len": np.array([2], np.int64),
        },
    )
    assert corrected.shape == (1, 1, 4, 2)
    np.testing.assert_array_equal(corrected, tentative[:, :, :4, :])


def test_euler_model_input_scales_by_sigma(tmp_path):
    (scaled,) = _run(
        build_euler_model_input(),
        tmp_path,
        {
            "sample": np.full((1, 1, 1, 1), 10.0, np.float32),
            "step": np.array([0], np.int64),
            "schedule": np.array([2.0, 0.0], np.float32),
        },
    )
    np.testing.assert_allclose(scaled, 10.0 / np.sqrt(5.0), rtol=1e-6)


def test_code_frame_update_accepts_scalar_loop_index(tmp_path):
    (updated,) = _run(
        build_code_frame_update(4, scalar_index=True),
        tmp_path,
        {
            "frame_codes": np.zeros((2, 4), np.int64),
            "token": np.array([5, 7], np.int64),
            "index": np.array(2, np.int64),
        },
    )
    np.testing.assert_array_equal(updated, [[0, 0, 5, 0], [0, 0, 7, 0]])


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
    assert {component.contract_id for component in package.policy_components.values()} == {
        "onnx-genai.token-sampler@1",
        "onnx-genai.termination-predicate@1",
        "onnx-genai.solver-step@1",
        "onnx-genai.masked-update@1",
        "onnx-genai.speculative-verifier@1",
        "onnx-genai.state-update@1",
    }
