# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Small, model-agnostic ONNX graphs for generation policy and state math.

These components deliberately receive policy parameters as tensor inputs. They
contain no model-family dispatch and can therefore be invoked from a generic
workflow IR just like neural ONNX components.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import onnx_ir as ir
from onnxscript import GraphBuilder

from mobius._constants import OPSET_VERSION

_POLICY_ROLE_METADATA = "mobius.generation.policy_role"


class PolicyRole(StrEnum):
    """Architecture-neutral role performed by a policy component."""

    TOKEN_SAMPLER = "token_sampler"
    TERMINATION = "termination"
    SOLVER_STEP = "solver_step"
    MASKED_UPDATE = "masked_update"
    SPECULATIVE_ACCEPTANCE = "speculative_acceptance"
    STATE_UPDATE = "state_update"


@dataclass(frozen=True)
class PolicyComponent:
    """A named role and its executable ONNX model."""

    role: PolicyRole
    model: ir.Model

    def __post_init__(self) -> None:
        self.model.graph.metadata_props[_POLICY_ROLE_METADATA] = self.role.value

    @classmethod
    def from_model(cls, model: ir.Model) -> PolicyComponent:
        """Restore a component from role metadata embedded in its ONNX graph."""
        role = model.graph.metadata_props.get(_POLICY_ROLE_METADATA)
        if role is None:
            raise ValueError("ONNX policy component is missing its Mobius policy role")
        return cls(PolicyRole(role), model)


@dataclass(frozen=True)
class PolicyCapabilities:
    """Data-driven declaration of policy math required by a package."""

    sampler: str | None = None
    eos_termination: bool = False
    solver: str | None = None
    masked_update: bool = False
    speculative_acceptance: bool = False
    token_state_update: bool = False


class _PolicyPackage(Protocol):
    def add_policy_component(self, name: str, component: PolicyComponent) -> None: ...


def attach_policy_components(
    pkg: _PolicyPackage,
    capabilities: PolicyCapabilities,
) -> dict[str, str]:
    """Attach exactly the policy artifacts selected by declared capabilities."""
    builders = {
        "greedy": build_greedy_sampler,
        "seeded_categorical": build_seeded_categorical_sampler,
    }
    solvers = {"euler": build_euler_solver_step}
    if capabilities.sampler not in {None, *builders}:
        raise ValueError(f"Unsupported sampler policy {capabilities.sampler!r}")
    if capabilities.solver not in {None, *solvers}:
        raise ValueError(f"Unsupported solver policy {capabilities.solver!r}")

    selected: list[tuple[str, PolicyComponent]] = []
    if capabilities.sampler is not None:
        selected.append(("token_sampler", builders[capabilities.sampler]()))
    if capabilities.eos_termination:
        selected.append(("termination", build_eos_termination()))
    if capabilities.solver is not None:
        selected.append(("solver_step", solvers[capabilities.solver]()))
    if capabilities.masked_update:
        selected.append(("masked_update", build_masked_token_update()))
    if capabilities.speculative_acceptance:
        selected.append(("speculative_acceptance", build_speculative_acceptance()))
    if capabilities.token_state_update:
        selected.append(("token_state_update", build_token_state_update()))

    for name, component in selected:
        pkg.add_policy_component(name, component)
    return {name: f"policies/{name}.onnx" for name, _ in selected}


def _component(role: PolicyRole, graph: ir.Graph) -> PolicyComponent:
    model = ir.Model(graph, ir_version=11)
    model.producer_name = "mobius"
    return PolicyComponent(role, model)


def _make_graph(name: str) -> tuple[ir.Graph, GraphBuilder]:
    graph = ir.Graph(
        [],
        [],
        nodes=[],
        name=name,
        opset_imports={"": OPSET_VERSION},
    )
    return graph, GraphBuilder(graph)


def build_greedy_sampler() -> PolicyComponent:
    """Build ``logits -> token_ids`` greedy sampling over the final axis."""
    graph, builder = _make_graph("greedy_sampler")
    logits = builder.input(
        "logits",
        dtype=ir.DataType.FLOAT,
        shape=["batch", "vocabulary"],
    )
    token_ids = builder.op.ArgMax(logits, axis=-1, keepdims=0)
    builder.add_output(token_ids, "token_ids")
    return _component(PolicyRole.TOKEN_SAMPLER, graph)


def build_seeded_categorical_sampler() -> PolicyComponent:
    """Build deterministic categorical sampling with explicit seed and counter.

    The integer hash is counter based: the same ``(seed, counter, logits,
    temperature)`` inputs always produce the same token. The updated counter is
    an explicit output, so no random or hidden mutable state exists in the graph.
    """
    graph, builder = _make_graph("seeded_categorical_sampler")
    op = builder.op
    logits = builder.input("logits", ir.DataType.FLOAT, ["batch", "vocabulary"])
    temperature = builder.input("temperature", ir.DataType.FLOAT, [])
    seed = builder.input("seed", ir.DataType.INT64, [])
    counter = builder.input("counter", ir.DataType.INT64, [])

    # A compact LCG-style integer hash. Constants remain below signed-int64
    # limits, and the prime modulus keeps the result in a precisely castable range.
    multiplier = op.Constant(value_int=1_103_515_245)
    stream_multiplier = op.Constant(value_int=12_345)
    increment = op.Constant(value_int=1_013_904_223)
    modulus = op.Constant(value_int=2_147_483_647)
    hashed = op.Add(
        op.Add(op.Mul(seed, multiplier), op.Mul(counter, stream_multiplier)),
        increment,
    )
    hashed = op.Mod(hashed, modulus, fmod=0)
    uniform = op.Div(
        op.Add(op.Cast(hashed, to=ir.DataType.FLOAT), op.Constant(value_float=0.5)),
        op.Constant(value_float=2_147_483_647.0),
    )

    scaled_logits = op.Div(logits, temperature)
    probabilities = op.Softmax(scaled_logits, axis=-1)
    axis = op.Constant(value_int=-1)
    cumulative = op.CumSum(probabilities, axis)
    uniform = op.Unsqueeze(uniform, op.Constant(value_ints=[0]))
    candidates = op.GreaterOrEqual(cumulative, uniform)
    token_ids = op.ArgMax(
        op.Cast(candidates, to=ir.DataType.INT64),
        axis=-1,
        keepdims=0,
    )
    next_counter = op.Add(counter, op.Constant(value_int=1))
    builder.add_output(token_ids, "token_ids")
    builder.add_output(next_counter, "next_counter")
    return _component(PolicyRole.TOKEN_SAMPLER, graph)


def build_eos_termination() -> PolicyComponent:
    """Build an EOS predicate for batched current tokens and an EOS-id set."""
    graph, builder = _make_graph("eos_termination")
    op = builder.op
    token_ids = builder.input("token_ids", ir.DataType.INT64, ["batch"])
    eos_token_ids = builder.input("eos_token_ids", ir.DataType.INT64, ["num_eos"])
    tokens = op.Unsqueeze(token_ids, op.Constant(value_ints=[-1]))
    eos = op.Unsqueeze(eos_token_ids, op.Constant(value_ints=[0]))
    matches = op.Equal(tokens, eos)
    match_count = op.ReduceSum(
        op.Cast(matches, to=ir.DataType.INT64),
        axes=[-1],
        keepdims=0,
    )
    terminated = op.Greater(match_count, op.Constant(value_int=0))
    builder.add_output(terminated, "terminated")
    return _component(PolicyRole.TERMINATION, graph)


def build_euler_solver_step() -> PolicyComponent:
    """Build the generic Euler update ``x_next = x + dx * (sigma_next-sigma)``."""
    graph, builder = _make_graph("euler_solver_step")
    op = builder.op
    sample = builder.input(
        "sample",
        ir.DataType.FLOAT,
        ["batch", "channels", "height", "width"],
    )
    derivative = builder.input(
        "derivative",
        ir.DataType.FLOAT,
        ["batch", "channels", "height", "width"],
    )
    sigma = builder.input("sigma", ir.DataType.FLOAT, [])
    sigma_next = builder.input("sigma_next", ir.DataType.FLOAT, [])
    next_sample = op.Add(sample, op.Mul(derivative, op.Sub(sigma_next, sigma)))
    builder.add_output(next_sample, "next_sample")
    return _component(PolicyRole.SOLVER_STEP, graph)


def build_masked_token_update() -> PolicyComponent:
    """Build confidence-thresholded replacement for masked token positions."""
    graph, builder = _make_graph("masked_token_update")
    op = builder.op
    current = builder.input("current_tokens", ir.DataType.INT64, ["batch", "sequence"])
    proposed = builder.input("proposed_tokens", ir.DataType.INT64, ["batch", "sequence"])
    confidence = builder.input("confidence", ir.DataType.FLOAT, ["batch", "sequence"])
    masked = builder.input("masked", ir.DataType.BOOL, ["batch", "sequence"])
    threshold = builder.input("threshold", ir.DataType.FLOAT, [])
    accepted = op.And(masked, op.GreaterOrEqual(confidence, threshold))
    updated = op.Where(accepted, proposed, current)
    remaining = op.And(masked, op.Not(accepted))
    builder.add_output(updated, "updated_tokens")
    builder.add_output(remaining, "remaining_mask")
    return _component(PolicyRole.MASKED_UPDATE, graph)


def build_speculative_acceptance() -> PolicyComponent:
    """Build per-token speculative acceptance and accepted-prefix length."""
    graph, builder = _make_graph("speculative_acceptance")
    op = builder.op
    target_probability = builder.input(
        "target_probability", ir.DataType.FLOAT, ["batch", "draft_sequence"]
    )
    draft_probability = builder.input(
        "draft_probability", ir.DataType.FLOAT, ["batch", "draft_sequence"]
    )
    uniform = builder.input("uniform", ir.DataType.FLOAT, ["batch", "draft_sequence"])
    ratio = op.Div(target_probability, draft_probability)
    probability = op.Min(ratio, op.Constant(value_float=1.0))
    accepted = op.LessOrEqual(uniform, probability)
    rejected = op.Cast(op.Not(accepted), to=ir.DataType.INT64)
    rejection_count = op.CumSum(rejected, op.Constant(value_int=-1))
    prefix = op.Cast(
        op.Equal(rejection_count, op.Constant(value_int=0)),
        to=ir.DataType.INT64,
    )
    accepted_count = op.ReduceSum(prefix, axes=[-1], keepdims=0)
    builder.add_output(accepted, "accepted")
    builder.add_output(accepted_count, "accepted_count")
    return _component(PolicyRole.SPECULATIVE_ACCEPTANCE, graph)


def build_token_state_update() -> PolicyComponent:
    """Build explicit token-history append and sequence-length update math."""
    graph, builder = _make_graph("token_state_update")
    op = builder.op
    tokens = builder.input("tokens", ir.DataType.INT64, ["batch", "sequence"])
    next_token = builder.input("next_token", ir.DataType.INT64, ["batch"])
    sequence_length = builder.input("sequence_length", ir.DataType.INT64, [])
    appended = op.Concat(
        tokens,
        op.Unsqueeze(next_token, op.Constant(value_ints=[-1])),
        axis=-1,
    )
    next_length = op.Add(sequence_length, op.Constant(value_int=1))
    builder.add_output(appended, "updated_tokens")
    builder.add_output(next_length, "updated_sequence_length")
    return _component(PolicyRole.STATE_UPDATE, graph)
