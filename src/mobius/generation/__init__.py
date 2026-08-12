# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Reusable ONNX generation-policy components."""

from __future__ import annotations

from mobius.generation._policy_components import (
    PolicyCapabilities,
    PolicyComponent,
    PolicyRole,
    attach_policy_components,
    build_boolean_not,
    build_code_frame_update,
    build_code_history_append,
    build_codec_layout_transpose,
    build_decoder_state_initializer,
    build_decoder_step_update,
    build_eos_termination,
    build_euler_solver_step,
    build_greedy_sampler,
    build_integer_increment,
    build_iteration_cast,
    build_last_token_logits,
    build_masked_token_update,
    build_model_token_cast,
    build_schedule_constant,
    build_seeded_categorical_sampler,
    build_speculative_acceptance,
    build_token_block_identity,
    build_token_state_update,
    build_tts_state_initializer,
)

__all__ = [
    "PolicyComponent",
    "PolicyCapabilities",
    "PolicyRole",
    "attach_policy_components",
    "build_boolean_not",
    "build_code_frame_update",
    "build_code_history_append",
    "build_codec_layout_transpose",
    "build_decoder_state_initializer",
    "build_decoder_step_update",
    "build_eos_termination",
    "build_euler_solver_step",
    "build_greedy_sampler",
    "build_integer_increment",
    "build_iteration_cast",
    "build_last_token_logits",
    "build_masked_token_update",
    "build_model_token_cast",
    "build_schedule_constant",
    "build_seeded_categorical_sampler",
    "build_speculative_acceptance",
    "build_token_state_update",
    "build_token_block_identity",
    "build_tts_state_initializer",
]
