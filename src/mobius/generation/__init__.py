# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Reusable ONNX generation-policy components."""

from __future__ import annotations

from mobius.generation._policy_components import (
    PolicyCapabilities,
    PolicyComponent,
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

__all__ = [
    "PolicyComponent",
    "PolicyCapabilities",
    "PolicyRole",
    "attach_policy_components",
    "build_eos_termination",
    "build_euler_solver_step",
    "build_greedy_sampler",
    "build_masked_token_update",
    "build_seeded_categorical_sampler",
    "build_speculative_acceptance",
    "build_token_state_update",
]
