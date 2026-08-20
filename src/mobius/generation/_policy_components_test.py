# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

from mobius._model_package import ModelPackage
from mobius.generation import (
    PolicyCapabilities,
    attach_policy_components,
    build_adaptive_k_policy,
    build_batch_minimum,
    build_boolean_not,
    build_code_frame_update,
    build_counter_rng_normal,
    build_ddim_solver_step,
    build_decoder_state_initializer,
    build_decoder_step_update,
    build_empty_features,
    build_eos_termination,
    build_euler_model_input,
    build_euler_solver_step,
    build_flow_match_solver_step,
    build_grammar_logits_processor,
    build_greedy_sampler,
    build_guidance_combine,
    build_integer_minimum,
    build_integer_row_broadcast,
    build_last_token_logits,
    build_masked_token_update,
    build_model_token_cast,
    build_multistep_solver_step,
    build_pack_latents_2x2,
    build_proposal_metrics,
    build_scalar_constant,
    build_schedule_history_append,
    build_seeded_categorical_sampler,
    build_sequence_concat,
    build_shape_constant,
    build_speculative_acceptance,
    build_speculative_state_rollback,
    build_tensor_scale,
    build_termination_batch_initializer,
    build_token_state_update,
    build_true_cfg,
    build_unpack_latents_2x2,
    build_video_conv_cache_initializer,
    build_video_decode_chunk,
    build_video_decode_chunk_count,
    build_video_latent_initializer,
    build_zeros_like,
    rotary_axis_count,
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


def test_row_selective_greedy_sampler_suppresses_inactive_rows(tmp_path):
    (tokens,) = _run(
        build_greedy_sampler(row_selective=True),
        tmp_path,
        {
            "logits": np.array([[0.0, 2.0], [3.0, 1.0], [1.0, 4.0]], np.float32),
            "active": np.array([True, False, True], np.bool_),
            "done": np.array([False, False, True], np.bool_),
        },
    )
    np.testing.assert_array_equal(tokens, [1, -1, -1])


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
    prompt = np.arange(68, dtype=np.int64).reshape(1, 68)
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
            "prompt_tokens": prompt,
            "max_iterations": np.array([128], np.int64),
        },
    )
    attention, body_attention, token, cache_lengths, cache = outputs
    expected_body_attention = np.zeros((1, 196), np.int64)
    expected_body_attention[:, :68] = 1
    np.testing.assert_array_equal(attention, expected_body_attention)
    np.testing.assert_array_equal(body_attention, expected_body_attention)
    np.testing.assert_array_equal(token, [[0]])
    np.testing.assert_array_equal(cache_lengths, [68])
    assert cache.shape == (1, 2, 196, 4)

    (next_attention,) = _run(
        build_decoder_step_update(
            attention_dtype=ir.DataType.INT64,
            position_dtype=None,
            fixed_capacity=True,
        ),
        tmp_path,
        {
            "attention_mask": body_attention,
            "logical_length": np.array([68], np.int64),
        },
    )
    expected_body_attention[:, :69] = 1
    np.testing.assert_array_equal(next_attention, expected_body_attention)


def test_decoder_ragged_batch_uses_stable_capacity_and_independent_rows(tmp_path):
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
    component = build_decoder_state_initializer(
        decoder,
        token_input="input_ids",
        attention_mask_input="attention_mask",
        position_ids_input=None,
        cache_inputs=["past_key_values.0.key"],
        fixed_capacity=True,
        ragged=True,
    )
    feeds = {
        "prompt_tokens": np.arange(12, dtype=np.int64).reshape(3, 4),
        "prompt_lengths": np.array([4, 2, 1], np.int64),
        "max_iterations": np.array([5], np.int64),
    }
    attention, body_attention, token, generated, cache_lengths, cache = _run(
        component, tmp_path, feeds
    )
    expected = np.array(
        [
            [1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        np.int64,
    )
    np.testing.assert_array_equal(attention, expected)
    np.testing.assert_array_equal(body_attention, expected)
    np.testing.assert_array_equal(token, np.zeros((3, 1), np.int64))
    np.testing.assert_array_equal(generated, [0, 0, 0])
    np.testing.assert_array_equal(cache_lengths, [4, 2, 1])
    assert cache.shape == (3, 2, 9, 4)

    for row in range(3):
        row_outputs = _run(
            component,
            tmp_path,
            {
                "prompt_tokens": feeds["prompt_tokens"][row : row + 1],
                "prompt_lengths": feeds["prompt_lengths"][row : row + 1],
                "max_iterations": feeds["max_iterations"],
            },
        )
        for batched, independent in zip(
            (attention, body_attention, token, generated, cache_lengths),
            row_outputs[:-1],
            strict=True,
        ):
            np.testing.assert_array_equal(batched[row : row + 1], independent)
        assert row_outputs[-1].shape == (1, 2, 9, 4)


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
                "tokens": sample,
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
        "seed": np.array([7], np.int64),
        "counter": np.array([11], np.int64),
        "active": np.array([True], np.bool_),
        "done": np.array([False], np.bool_),
    }
    first = _run(component, tmp_path, feeds)
    second = _run(component, tmp_path, feeds)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], [12])


def test_batched_policy_v2_ports_use_exact_public_shapes():
    components = {
        "sampler": build_seeded_categorical_sampler(),
        "termination": build_eos_termination(row_selective=True),
        "state": build_token_state_update(row_selective=True),
    }
    expected = {
        "sampler": {
            "logits": ["batch", "vocabulary"],
            "temperature": ["batch"],
            "top_k": ["batch"],
            "top_p": ["batch"],
            "min_p": ["batch"],
            "seed": ["batch"],
            "counter": ["batch"],
            "active": ["batch"],
            "done": ["batch"],
            "token": ["batch"],
            "next_counter": ["batch"],
        },
        "termination": {
            "tokens": ["batch"],
            "eos_ids": ["batch", "num_eos"],
            "eos_lengths": ["batch"],
            "iteration": ["1"],
            "max_iterations": ["batch"],
            "active": ["batch"],
            "done": ["batch"],
            "next_active": ["batch"],
            "continue": ["1"],
        },
        "state": {
            "current": ["batch", "1"],
            "update": ["batch", "1"],
            "active": ["batch"],
            "done": ["batch"],
            "next": ["batch", "1"],
        },
    }
    for name, component in components.items():
        ports = [*component.model.graph.inputs, *component.model.graph.outputs]
        assert {port.name: [str(dim) for dim in port.shape] for port in ports} == expected[
            name
        ]


def test_seeded_sampler_applies_request_top_k(tmp_path):
    (token, _) = _run(
        build_seeded_categorical_sampler(),
        tmp_path,
        {
            "logits": np.array([[10.0, 9.0, 8.0, 7.0]], np.float32),
            "temperature": np.array([0.7], np.float32),
            "top_k": np.array([1], np.int64),
            "top_p": np.array([0.5], np.float32),
            "min_p": np.array([0.0], np.float32),
            "seed": np.array([17], np.int64),
            "counter": np.array([0], np.int64),
            "active": np.array([True], np.bool_),
            "done": np.array([False], np.bool_),
        },
    )
    np.testing.assert_array_equal(token, [0])


def test_seeded_sampler_applies_request_min_p_in_logit_space(tmp_path):
    (token, next_counter) = _run(
        build_seeded_categorical_sampler(),
        tmp_path,
        {
            "logits": np.array([[0.0, -0.1, -10.0]], np.float32),
            "temperature": np.array([1.0], np.float32),
            "top_k": np.array([0], np.int64),
            "top_p": np.array([1.0], np.float32),
            "min_p": np.array([0.95], np.float32),
            "seed": np.array([23], np.int64),
            "counter": np.array([4], np.int64),
            "active": np.array([True], np.bool_),
            "done": np.array([False], np.bool_),
        },
    )
    np.testing.assert_array_equal(token, [0])
    np.testing.assert_array_equal(next_counter, [5])


def test_seeded_sampler_heterogeneous_batch_matches_independent_rows(tmp_path):
    component = build_seeded_categorical_sampler()
    feeds = {
        "logits": np.array(
            [
                [2.0, 1.0, 0.5, -1.0, -2.0],
                [0.0, 0.1, 0.2, 0.3, 0.4],
                [4.0, 3.0, 2.0, 1.0, 0.0],
                [-1.0, 0.0, 1.0, 2.0, 3.0],
            ],
            np.float32,
        ),
        "temperature": np.array([0.5, 1.5, 0.8, 2.0], np.float32),
        "top_k": np.array([1, 3, 0, 2], np.int64),
        "top_p": np.array([1.0, 0.8, 0.6, 0.9], np.float32),
        "min_p": np.array([0.0, 0.05, 0.2, 0.1], np.float32),
        "seed": np.array([3, 7, 11, 13], np.int64),
        "counter": np.array([0, 5, 9, 12], np.int64),
        "active": np.array([True, True, False, True], np.bool_),
        "done": np.array([False, False, False, True], np.bool_),
    }
    batched_token, batched_offset = _run(component, tmp_path, feeds)
    for row in range(4):
        row_feeds = {name: value[row : row + 1] for name, value in feeds.items()}
        row_token, row_offset = _run(component, tmp_path, row_feeds)
        np.testing.assert_array_equal(batched_token[row : row + 1], row_token)
        np.testing.assert_array_equal(batched_offset[row : row + 1], row_offset)
    np.testing.assert_array_equal(batched_token[2:], [-1, -1])
    np.testing.assert_array_equal(batched_offset, [1, 6, 9, 12])


def test_row_selective_state_and_termination_preserve_inactive_rows(tmp_path):
    (next_state,) = _run(
        build_token_state_update(row_selective=True),
        tmp_path,
        {
            "current": np.array([[10], [20], [30]], np.int64),
            "update": np.array([[11], [21], [31]], np.int64),
            "active": np.array([True, False, True], np.bool_),
            "done": np.array([False, False, True], np.bool_),
        },
    )
    np.testing.assert_array_equal(next_state, [[11], [20], [30]])

    done, next_active, continued = _run(
        build_eos_termination(row_selective=True),
        tmp_path,
        {
            "tokens": np.array([2, 8, 9], np.int64),
            "eos_ids": np.array([[2, 9], [2, 9], [2, 9]], np.int64),
            "eos_lengths": np.array([2, 1, 2], np.int64),
            "iteration": np.array([0], np.int64),
            "max_iterations": np.array([5, 5, 5], np.int64),
            "active": np.array([True, False, True], np.bool_),
        },
    )
    np.testing.assert_array_equal(done, [True, True, True])
    np.testing.assert_array_equal(next_active, [False, False, False])
    np.testing.assert_array_equal(continued, [False])


def test_row_selective_termination_heterogeneous_batch_matches_independent_rows(
    tmp_path,
):
    component = build_eos_termination(row_selective=True)
    feeds = {
        "tokens": np.array([2, 9, 5, 7], np.int64),
        "eos_ids": np.array([[2, 99], [8, 9], [5, 6], [1, 2]], np.int64),
        "eos_lengths": np.array([1, 1, 2, 2], np.int64),
        "iteration": np.array([2], np.int64),
        "max_iterations": np.array([5, 5, 10, 3], np.int64),
        "active": np.array([True, True, True, True], np.bool_),
    }
    done, next_active, continued = _run(component, tmp_path, feeds)
    np.testing.assert_array_equal(done, [True, False, True, True])
    np.testing.assert_array_equal(next_active, [False, True, False, False])
    np.testing.assert_array_equal(continued, [True])
    for row in range(4):
        row_outputs = _run(
            component,
            tmp_path,
            {
                name: value if name == "iteration" else value[row : row + 1]
                for name, value in feeds.items()
            },
        )
        np.testing.assert_array_equal(done[row : row + 1], row_outputs[0])
        np.testing.assert_array_equal(next_active[row : row + 1], row_outputs[1])


def test_termination_batch_controls_initialize_dynamic_rows(tmp_path):
    eos_ids, eos_lengths, max_iterations = _run(
        build_termination_batch_initializer(),
        tmp_path,
        {
            "input_eos_ids": np.array([[2, 3], [7, -1]], np.int64),
            "input_eos_lengths": np.array([2, 1], np.int64),
            "input_max_iterations": np.array([4, -1], np.int64),
            "fallback_max_iterations": np.array([8], np.int64),
            "active": np.array([True, False], np.bool_),
        },
    )
    np.testing.assert_array_equal(eos_ids, [[2, 3], [7, -1]])
    np.testing.assert_array_equal(eos_lengths, [2, 1])
    np.testing.assert_array_equal(max_iterations, [4, 8])

    (iteration_rows,) = _run(
        build_integer_row_broadcast(),
        tmp_path,
        {
            "value": np.array([3], np.int64),
            "active": np.array([True, False], np.bool_),
        },
    )
    np.testing.assert_array_equal(iteration_rows, [3, 3])


def test_eos_termination_runtime(tmp_path):
    terminated, continued = _run(
        build_eos_termination(),
        tmp_path,
        {
            "tokens": np.array([2, 8, 9], np.int64),
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


def test_ddim_solver_runtime_parity_on_a_video_latent(tmp_path):
    # [batch, frames, channels, height, width]: the update has to broadcast the
    # per-row alphas over a temporal axis, not just over an image.
    dims = ["batch", "frames", "channels", "height", "width"]
    sample = np.linspace(-2.0, 2.0, 24, dtype=np.float32).reshape(1, 3, 2, 2, 2)
    estimate = np.full_like(sample, 0.25)
    schedule = np.array([0.4, 0.9, 1.0], np.float32)
    (actual,) = _run(
        build_ddim_solver_step(latent_dims=dims),
        tmp_path,
        {
            "sample": sample,
            "derivative": estimate,
            "step": np.array([0], np.int64),
            "schedule": schedule,
        },
    )
    alpha, alpha_prev = schedule[0], schedule[1]
    expected = (
        np.sqrt(alpha_prev) * ((sample - np.sqrt(1 - alpha) * estimate) / np.sqrt(alpha))
        + np.sqrt(1 - alpha_prev) * estimate
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_ddim_solver_clips_the_predicted_clean_latent(tmp_path):
    dims = ["batch", "frames", "channels", "height", "width"]
    sample = np.full((1, 1, 1, 1, 1), 8.0, np.float32)
    estimate = np.zeros_like(sample)
    (actual,) = _run(
        build_ddim_solver_step(latent_dims=dims, clip_sample_range=1.0),
        tmp_path,
        {
            "sample": sample,
            "derivative": estimate,
            "step": np.array([0], np.int64),
            "schedule": np.array([0.25, 0.81], np.float32),
        },
    )
    # pred_x0 = 8 / sqrt(0.25) = 16, clipped to 1, then renoised to alpha_prev.
    np.testing.assert_allclose(actual, np.sqrt(0.81), rtol=1e-6)


def test_video_decode_chunk_walk_matches_the_reference(tmp_path):
    latent = np.arange(5 * 2, dtype=np.float32).reshape(1, 1, 5, 1, 2)
    (count,) = _run(build_video_decode_chunk_count(), tmp_path, {"latent": latent})
    np.testing.assert_array_equal(count, [2])
    chunks = [
        _run(
            build_video_decode_chunk(),
            tmp_path,
            {"latent": latent, "step": np.array([step], np.int64)},
        )[0]
        for step in range(int(count[0]))
    ]
    # Five latent frames split as three then two: the odd frame is folded into
    # the first chunk, and the chunks tile the clip without gaps or overlap.
    assert [chunk.shape[2] for chunk in chunks] == [3, 2]
    np.testing.assert_array_equal(np.concatenate(chunks, axis=2), latent)

    single = np.arange(3 * 2, dtype=np.float32).reshape(1, 1, 3, 1, 2)
    (single_count,) = _run(build_video_decode_chunk_count(), tmp_path, {"latent": single})
    np.testing.assert_array_equal(single_count, [1])


def test_video_conv_cache_initializer_sizes_each_resolution(tmp_path):
    latent = np.zeros((2, 4, 3, 5, 6), np.float32)
    caches = _run(
        build_video_conv_cache_initializer(
            [("conv_cache.conv_in", 4, 1), ("conv_cache.conv_out", 8, 4)]
        ),
        tmp_path,
        {"latent": latent},
    )
    # Zero frames is how "no previous chunk" is expressed; the spatial extents
    # still have to match the resolution each cached convolution runs at.
    assert caches[0].shape == (2, 4, 0, 5, 6)
    assert caches[1].shape == (2, 8, 0, 20, 24)


def test_scheduler_history_starts_empty_and_grows_per_step(tmp_path):
    noise = np.zeros((2, 3, 4, 2, 2), np.float32)
    latent, history = _run(
        build_video_latent_initializer(ir.DataType.FLOAT, 2.0), tmp_path, {"noise": noise}
    )
    assert latent.shape == noise.shape
    assert history.shape == (2, 0)
    assert history.dtype == np.int64
    for step, timestep in enumerate([600, 300, 0]):
        (history,) = _run(
            build_schedule_history_append(ir.DataType.INT64),
            tmp_path,
            {"history": history, "timestep": np.array([timestep, timestep], np.int64)},
        )
        assert history.shape == (2, step + 1)
    np.testing.assert_array_equal(history, [[600, 300, 0], [600, 300, 0]])


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


def test_guidance_combine_extrapolates_per_row(tmp_path):
    # Classifier-free guidance: uncond + scale * (cond - uncond), with the scale
    # supplied per request row rather than baked into the graph.
    unconditional = np.array([[[[1.0]]], [[[2.0]]]], dtype=np.float32)
    conditional = np.array([[[[3.0]]], [[[-2.0]]]], dtype=np.float32)
    (guided,) = _run(
        build_guidance_combine(),
        tmp_path,
        {
            "unconditional": unconditional,
            "conditional": conditional,
            "scale": np.array([7.5, 0.0], np.float32),
        },
    )
    np.testing.assert_allclose(
        guided,
        unconditional
        + np.array([7.5, 0.0], np.float32).reshape(2, 1, 1, 1) * (conditional - unconditional),
        rtol=1e-6,
    )


def test_multistep_solver_matches_dpmsolverpp_second_order(tmp_path):
    # Reproduces diffusers' DPMSolverMultistepScheduler.step for dpmsolver++
    # midpoint updates, including the first-order fallback on the first step.
    schedule = np.array([8.0, 4.0, 1.0, 0.0], np.float32)
    sample = np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(1, 2, 2, 2)
    estimate = np.linspace(0.5, -0.5, 8, dtype=np.float32).reshape(1, 2, 2, 2)
    history = np.full_like(sample, 0.25)

    def reference(sample, estimate, history, step, first_order):
        sigma, sigma_next = schedule[step], schedule[step + 1]
        alpha = 1.0 / np.sqrt(sigma**2 + 1.0)
        alpha_next = 1.0 / np.sqrt(sigma_next**2 + 1.0)
        noise, noise_next = sigma * alpha, sigma_next * alpha_next
        x0 = (sample - noise * estimate) / alpha
        ratio = noise_next / noise
        if first_order:
            return ratio * sample - alpha_next * (sigma_next / sigma - 1.0) * x0
        step_size = np.log(sigma) - np.log(sigma_next)
        previous = np.log(schedule[step - 1]) - np.log(sigma)
        difference = (step_size / previous) * (x0 - history)
        return ratio * sample - alpha_next * (sigma_next / sigma - 1.0) * (
            x0 + 0.5 * difference
        )

    # Step 0 has no usable history, so the solver must fall back to first order.
    next_state, next_history = _run(
        build_multistep_solver_step(),
        tmp_path,
        {
            "sample": sample,
            "estimate": estimate,
            "history": history,
            "step": np.array([0], np.int64),
            "schedule": schedule,
        },
    )
    np.testing.assert_allclose(
        next_state, reference(sample, estimate, history, 0, True), rtol=1e-5, atol=1e-6
    )

    # Step 1 uses the carried estimate; the returned history is the new one.
    second_state, second_history = _run(
        build_multistep_solver_step(),
        tmp_path,
        {
            "sample": sample,
            "estimate": estimate,
            "history": next_history,
            "step": np.array([1], np.int64),
            "schedule": schedule,
        },
    )
    np.testing.assert_allclose(
        second_state,
        reference(sample, estimate, next_history, 1, False),
        rtol=1e-5,
        atol=1e-6,
    )

    # The final step drops back to first order the way lower_order_final does.
    final_state, _ = _run(
        build_multistep_solver_step(),
        tmp_path,
        {
            "sample": sample,
            "estimate": estimate,
            "history": second_history,
            "step": np.array([2], np.int64),
            "schedule": schedule,
        },
    )
    np.testing.assert_allclose(
        final_state,
        reference(sample, estimate, second_history, 2, True),
        rtol=1e-5,
        atol=1e-6,
    )


def test_counter_rng_is_reproducible_and_row_private(tmp_path):
    component = build_counter_rng_normal()
    feeds = {
        "seed": np.array([1234, 4321], np.int64),
        "offset": np.array([0, 0], np.int64),
        "row_shape": np.array([4, 8, 8], np.int64),
    }
    noise, next_offset = _run(component, tmp_path, feeds)
    assert noise.shape == (2, 4, 8, 8)
    np.testing.assert_array_equal(next_offset, [1, 1])

    repeat, _ = _run(component, tmp_path, feeds)
    np.testing.assert_array_equal(noise, repeat)

    # A row's draw depends only on its own seed, not on its batch position.
    swapped, _ = _run(
        component,
        tmp_path,
        {**feeds, "seed": np.array([4321, 1234], np.int64)},
    )
    np.testing.assert_array_equal(swapped[0], noise[1])
    np.testing.assert_array_equal(swapped[1], noise[0])

    # Advancing the counter draws a different, decorrelated block.
    advanced, advanced_offset = _run(
        component,
        tmp_path,
        {**feeds, "offset": np.array([1, 1], np.int64)},
    )
    np.testing.assert_array_equal(advanced_offset, [2, 2])
    assert np.abs(advanced - noise).max() > 1e-3
    assert abs(float(np.corrcoef(advanced.ravel(), noise.ravel())[0, 1])) < 0.1
    # Box-Muller output must look standard normal.
    assert abs(float(noise.mean())) < 0.1
    assert abs(float(noise.std()) - 1.0) < 0.1


def test_tensor_scale_and_zeros_like_shape_from_their_input(tmp_path):
    sample = np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2)
    (scaled,) = _run(
        build_tensor_scale(),
        tmp_path,
        {"tensor": sample, "scale": np.array([0.5], np.float32)},
    )
    np.testing.assert_allclose(scaled, sample * 0.5)
    (zeros,) = _run(build_zeros_like(), tmp_path, {"reference": sample})
    np.testing.assert_array_equal(zeros, np.zeros_like(sample))


def test_scalar_and_shape_constants_publish_their_values(tmp_path):
    (value,) = _run(build_scalar_constant(0.18215), tmp_path, {})
    np.testing.assert_allclose(value, 0.18215, rtol=1e-6)
    (shape,) = _run(build_shape_constant([4, 64, 64]), tmp_path, {})
    np.testing.assert_array_equal(shape, [4, 64, 64])


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


def test_state_initializer_allocates_fp8_cache_through_a_cast(tmp_path):
    """An fp8 KV cache must not be materialized by ``ConstantOfShape`` directly.

    ORT's ``ConstantOfShape`` kernel has no fp8 output implementation, so an
    fp8 cache emitted that way fails at session initialization with
    "Unsupported value attribute datatype: 17". Fill a supported dtype and cast.
    """
    inputs = [
        ir.Value(
            name="input_ids",
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape(["batch", "sequence"]),
        ),
        ir.Value(
            name="attention_mask",
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape(["batch", "sequence"]),
        ),
        ir.Value(
            name="past_key_values.0.key",
            type=ir.TensorType(ir.DataType.FLOAT8E4M3FN),
            shape=ir.Shape(["batch", 2, "past_sequence", 4]),
        ),
    ]
    decoder = ir.Model(ir.Graph(inputs, [], nodes=[], name="decoder"), ir_version=11)
    initializer = build_decoder_state_initializer(
        decoder,
        token_input="input_ids",
        attention_mask_input="attention_mask",
        position_ids_input=None,
        cache_inputs=["past_key_values.0.key"],
    )

    cache = next(
        value
        for value in initializer.model.graph.outputs
        if value.name == "past_key_values.0.key"
    )
    assert cache.dtype == ir.DataType.FLOAT8E4M3FN
    assert cache.producer().op_type == "Cast"
    fill = cache.producer().inputs[0].producer()
    assert fill.op_type == "ConstantOfShape"
    assert fill.attributes["value"].value.dtype == ir.DataType.FLOAT

    outputs = _run(
        initializer,
        tmp_path,
        {"prompt_tokens": np.array([[3, 4, 5]], np.int64)},
    )
    assert outputs[-1].shape == (1, 2, 0, 4)


def _reference_pack(latent: np.ndarray) -> np.ndarray:
    """``QwenImagePipeline._pack_latents`` in numpy: (B,C,T,H,W) -> (B,T*H/2*W/2,C*4)."""
    batch, channels, frames, height, width = latent.shape
    packed = latent.reshape(batch, channels, frames, height // 2, 2, width // 2, 2)
    packed = packed.transpose(0, 2, 3, 5, 1, 4, 6)
    return packed.reshape(batch, frames * (height // 2) * (width // 2), channels * 4)


def test_flow_match_solver_step_runtime_parity(tmp_path):
    """Flow matching integrates ``x + (sigma_next - sigma) * v`` on rank-3 tokens."""
    sample = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    derivative = np.full_like(sample, 0.5)
    (actual,) = _run(
        build_flow_match_solver_step(),
        tmp_path,
        {
            "sample": sample,
            "derivative": derivative,
            "step": np.array([1], np.int64),
            "schedule": np.array([1.0, 0.6, 0.2], np.float32),
        },
    )
    # step 1 moves sigma 0.6 -> 0.2, so the update is -0.4 * derivative.
    np.testing.assert_allclose(actual, sample - 0.4 * derivative, rtol=1e-6, atol=1e-6)


def test_pack_latents_matches_diffusers_patchify(tmp_path):
    latent = np.arange(2 * 4 * 1 * 4 * 6, dtype=np.float32).reshape(2, 4, 1, 4, 6)
    (packed,) = _run(build_pack_latents_2x2(), tmp_path, {"latent_sample": latent})
    assert packed.shape == (2, 6, 16)
    np.testing.assert_array_equal(packed, _reference_pack(latent))


def test_unpack_latents_inverts_pack(tmp_path):
    """Round-tripping must be exact: the loop packs once and unpacks once.

    ``height``/``width`` are the *packed* token grid, i.e. half the latent
    spatial extent, because each token folds a 2x2 patch into its channels.
    """
    latent = np.arange(1 * 4 * 1 * 6 * 4, dtype=np.float32).reshape(1, 4, 1, 6, 4)
    (packed,) = _run(build_pack_latents_2x2(), tmp_path, {"latent_sample": latent})
    (restored,) = _run(
        build_unpack_latents_2x2(),
        tmp_path,
        {
            "packed_latent": packed,
            "height": np.array([3], np.int64),
            "width": np.array([2], np.int64),
        },
    )
    np.testing.assert_array_equal(restored, latent)


def test_sequence_concat_joins_target_then_source(tmp_path):
    """Order matters: the denoiser slices its estimate back off the front."""
    target = np.ones((1, 2, 3), np.float32)
    source = np.full((1, 4, 3), 2.0, np.float32)
    (joined,) = _run(build_sequence_concat(), tmp_path, {"target": target, "source": source})
    assert joined.shape == (1, 6, 3)
    np.testing.assert_array_equal(joined, np.concatenate([target, source], axis=1))


def test_true_cfg_matches_diffusers_norm_rescale(tmp_path):
    """True CFG rescales the guided estimate back to the conditional norm.

    ``QwenImageEditPlusPipeline`` computes ``comb = neg + s * (cond - neg)`` and
    then multiplies by ``||cond||/||comb||`` over the channel axis, so guidance
    changes direction without inflating magnitude.
    """
    rng = np.random.default_rng(3)
    cond = rng.standard_normal((2, 3, 4)).astype(np.float32)
    uncond = rng.standard_normal((2, 3, 4)).astype(np.float32)
    (actual,) = _run(
        build_true_cfg(guidance_scale=4.0),
        tmp_path,
        {"conditional": cond, "unconditional": uncond},
    )
    comb = uncond + 4.0 * (cond - uncond)
    cond_norm = np.linalg.norm(cond, axis=-1, keepdims=True)
    comb_norm = np.linalg.norm(comb, axis=-1, keepdims=True)
    np.testing.assert_allclose(actual, comb * (cond_norm / comb_norm), rtol=1e-5, atol=1e-5)


def test_true_cfg_is_identity_at_unit_guidance(tmp_path):
    rng = np.random.default_rng(5)
    cond = rng.standard_normal((1, 2, 4)).astype(np.float32)
    uncond = rng.standard_normal((1, 2, 4)).astype(np.float32)
    (actual,) = _run(
        build_true_cfg(guidance_scale=1.0),
        tmp_path,
        {"conditional": cond, "unconditional": uncond},
    )
    np.testing.assert_allclose(actual, cond, rtol=1e-5, atol=1e-5)


def _multi_axis_decoder(sections) -> ir.Model:
    """A decoder whose ``position_ids`` carries `sections` rotary axes."""
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
            shape=ir.Shape([sections, "batch", "sequence"]),
        ),
        ir.Value(
            name="past_key_values.0.key",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape(["batch", 2, "past_sequence", 4]),
        ),
    ]
    return ir.Model(ir.Graph(inputs, [], nodes=[], name="decoder"), ir_version=11)


def test_rotary_axis_count_reads_the_declared_leading_dimension():
    decoder = _multi_axis_decoder(3)
    positions = {value.name: value for value in decoder.graph.inputs}["position_ids"]
    assert rotary_axis_count(positions) == 3


def test_rotary_axis_count_is_none_for_plain_rank_2_positions():
    # A plain decoder has no rotary-axis dimension to broadcast over, so the
    # policy graphs keep emitting (batch, sequence) positions.
    positions = ir.Value(
        name="position_ids",
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape(["batch", "sequence"]),
    )
    assert rotary_axis_count(positions) is None


def test_rotary_axis_count_refuses_a_symbolic_axis_count():
    # The number of rotary axes is fixed by the exported graph. A symbolic
    # leading dimension means the export did not state it, and guessing would
    # silently mis-shape every position the decoder reads.
    positions = ir.Value(
        name="position_ids",
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape(["sections", "batch", "sequence"]),
    )
    with pytest.raises(TypeError, match="symbolic"):
        rotary_axis_count(positions)


def test_multi_axis_positions_are_broadcast_to_every_rotary_axis(tmp_path):
    # A 3D-rotary decoder reads (sections, batch, sequence) positions. The
    # prefill range and the per-step increment must be shaped for it, or the
    # decoder silently reads a rank-2 tensor as one axis of three.
    initializer = build_decoder_state_initializer(
        _multi_axis_decoder(3),
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
    prefill_positions, body_positions = outputs[1], outputs[3]
    assert prefill_positions.shape == (3, 1, 3)
    np.testing.assert_array_equal(prefill_positions, np.broadcast_to([[0, 1, 2]], (3, 1, 3)))
    assert body_positions.shape == (3, 1, 1)
    np.testing.assert_array_equal(body_positions, np.full((3, 1, 1), 3))

    updated = _run(
        build_decoder_step_update(
            attention_dtype=ir.DataType.INT64,
            position_dtype=ir.DataType.INT64,
            position_sections=3,
        ),
        tmp_path,
        {
            "attention_mask": outputs[2],
            "position_ids": body_positions,
        },
    )
    np.testing.assert_array_equal(updated[0], [[1, 1, 1, 1, 1]])
    assert updated[1].shape == (3, 1, 1)
    np.testing.assert_array_equal(updated[1], np.full((3, 1, 1), 4))
