# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Small, model-agnostic ONNX graphs for generation policy and state math.

These components deliberately receive policy parameters as tensor inputs. They
contain no model-family dispatch and can therefore be invoked from a generic
workflow IR just like neural ONNX components.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import onnx_ir as ir
from onnxscript import GraphBuilder

from mobius._constants import OPSET_VERSION

_POLICY_ROLE_METADATA = "mobius.generation.policy_role"
_POLICY_CONTRACT_METADATA = "mobius.generation.policy_contract"
_POLICY_EFFECTS_METADATA = "mobius.generation.policy_effects"


class PolicyRole(StrEnum):
    """Architecture-neutral role performed by a policy component."""

    TOKEN_SAMPLER = "token_sampler"
    TERMINATION = "termination_predicate"
    SOLVER_STEP = "solver_step"
    MASKED_UPDATE = "masked_update"
    SPECULATIVE_ACCEPTANCE = "speculative_verifier"
    STATE_UPDATE = "state_update"


@dataclass(frozen=True)
class PolicyComponent:
    """A named role and its executable ONNX model."""

    role: PolicyRole
    model: ir.Model
    contract: dict[str, object]
    effects: tuple[str, ...]

    def __post_init__(self) -> None:
        self.model.graph.metadata_props[_POLICY_ROLE_METADATA] = self.role.value
        self.model.graph.metadata_props[_POLICY_CONTRACT_METADATA] = json.dumps(self.contract)
        self.model.graph.metadata_props[_POLICY_EFFECTS_METADATA] = json.dumps(self.effects)

    @classmethod
    def from_model(cls, model: ir.Model) -> PolicyComponent:
        """Restore a component from role metadata embedded in its ONNX graph."""
        role = model.graph.metadata_props.get(_POLICY_ROLE_METADATA)
        if role is None:
            raise ValueError("ONNX policy component is missing its Mobius policy role")
        contract = json.loads(model.graph.metadata_props[_POLICY_CONTRACT_METADATA])
        effects = tuple(json.loads(model.graph.metadata_props[_POLICY_EFFECTS_METADATA]))
        return cls(PolicyRole(role), model, contract, effects)


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


def _component(
    role: PolicyRole,
    graph: ir.Graph,
    contract: dict[str, object],
    *effects: str,
) -> PolicyComponent:
    model = ir.Model(graph, ir_version=11)
    model.producer_name = "mobius"
    return PolicyComponent(role, model, contract, effects)


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
    builder.add_output(token_ids, "token")
    return _component(
        PolicyRole.TOKEN_SAMPLER,
        graph,
        {
            "role": "token_sampler",
            "mode": "greedy",
            "logits": "logits",
            "token": "token",
            "effect": "sample",
        },
        "sample",
    )


def build_seeded_categorical_sampler() -> PolicyComponent:
    """Build deterministic categorical sampling with explicit seed and offset.

    Threefry is counter based: the same ``(seed, offset, logits, temperature)``
    inputs always produce the same token. The updated offset is
    an explicit output, so no random or hidden mutable state exists in the graph.
    """
    graph, builder = _make_graph("seeded_categorical_sampler")
    op = builder.op
    logits = builder.input("logits", ir.DataType.FLOAT, ["batch", "vocabulary"])
    temperature = builder.input("temperature", ir.DataType.FLOAT, ["batch"])
    seed = builder.input("seed", ir.DataType.INT64, ["batch"])
    offset = builder.input("offset", ir.DataType.INT64, ["batch"])

    # Threefry2x64: a counter-based Random123 generator with no hidden state.
    # Unsigned arithmetic gives the specified modulo-2^64 round behavior.
    k0 = op.Cast(seed, to=ir.DataType.UINT64)
    k1 = op.Constant(value_int=0)
    k1 = op.Cast(k1, to=ir.DataType.UINT64)
    parity = op.Cast(op.Constant(value_int=0x1BD11BDAA9FC1A22), to=ir.DataType.UINT64)
    k2 = op.BitwiseXor(op.BitwiseXor(k0, k1), parity)
    keys = [k0, k1, k2]
    x0 = op.Add(op.Cast(offset, to=ir.DataType.UINT64), k0)
    x1 = op.Add(k1, op.Cast(op.Constant(value_int=0), to=ir.DataType.UINT64))
    rotations = [16, 42, 12, 31, 16, 32, 24, 21]
    for round_index in range(20):
        x0 = op.Add(x0, x1)
        rotation = rotations[round_index % len(rotations)]
        left = op.BitShift(
            x1,
            op.Cast(op.Constant(value_int=rotation), to=ir.DataType.UINT64),
            direction="LEFT",
        )
        right = op.BitShift(
            x1,
            op.Cast(op.Constant(value_int=64 - rotation), to=ir.DataType.UINT64),
            direction="RIGHT",
        )
        x1 = op.BitwiseXor(op.BitwiseOr(left, right), x0)
        if (round_index + 1) % 4 == 0:
            injection = (round_index + 1) // 4
            x0 = op.Add(x0, keys[injection % 3])
            x1 = op.Add(
                op.Add(x1, keys[(injection + 1) % 3]),
                op.Cast(op.Constant(value_int=injection), to=ir.DataType.UINT64),
            )
    mantissa = op.BitShift(
        x0,
        op.Cast(op.Constant(value_int=11), to=ir.DataType.UINT64),
        direction="RIGHT",
    )
    uniform = op.Div(
        op.Cast(mantissa, to=ir.DataType.DOUBLE),
        op.Cast(
            op.Constant(value_float=9_007_199_254_740_992.0),
            to=ir.DataType.DOUBLE,
        ),
    )
    uniform = op.Cast(uniform, to=ir.DataType.FLOAT)

    scaled_logits = op.Div(
        logits,
        op.Unsqueeze(temperature, op.Constant(value_ints=[-1])),
    )
    probabilities = op.Softmax(scaled_logits, axis=-1)
    axis = op.Constant(value_int=-1)
    cumulative = op.CumSum(probabilities, axis)
    uniform = op.Unsqueeze(uniform, op.Constant(value_ints=[-1]))
    candidates = op.GreaterOrEqual(cumulative, uniform)
    token_ids = op.ArgMax(
        op.Cast(candidates, to=ir.DataType.INT64),
        axis=-1,
        keepdims=0,
    )
    next_offset = op.Add(offset, op.Constant(value_int=1))
    builder.add_output(token_ids, "token")
    builder.add_output(next_offset, "next_offset")
    return _component(
        PolicyRole.TOKEN_SAMPLER,
        graph,
        {
            "role": "token_sampler",
            "mode": "seeded_stochastic",
            "logits": "logits",
            "token": "token",
            "temperature": "temperature",
            "rng": {
                "seed": "seed",
                "offset": "offset",
                "next_offset": "next_offset",
            },
            "effect": "rng",
        },
        "rng",
    )


def build_eos_termination() -> PolicyComponent:
    """Build an EOS predicate for batched current tokens and an EOS-id set."""
    graph, builder = _make_graph("eos_termination")
    op = builder.op
    token_ids = builder.input("token_ids", ir.DataType.INT64, ["batch"])
    eos_ids = builder.input("eos_ids", ir.DataType.INT64, ["num_eos"])
    iteration = builder.input("iteration", ir.DataType.INT64, ["batch"])
    max_iterations = builder.input("max_iterations", ir.DataType.INT64, ["batch"])
    tokens = op.Unsqueeze(token_ids, op.Constant(value_ints=[-1]))
    eos = op.Unsqueeze(eos_ids, op.Constant(value_ints=[0]))
    matches = op.Equal(tokens, eos)
    match_count = op.ReduceSum(
        op.Cast(matches, to=ir.DataType.INT64),
        axes=[-1],
        keepdims=0,
    )
    hit_eos = op.Greater(match_count, op.Constant(value_int=0))
    hit_limit = op.GreaterOrEqual(
        op.Add(iteration, op.Constant(value_int=1)),
        max_iterations,
    )
    done = op.Or(hit_eos, hit_limit)
    builder.add_output(done, "done")
    return _component(
        PolicyRole.TERMINATION,
        graph,
        {
            "role": "termination_predicate",
            "tokens": "token_ids",
            "eos_ids": "eos_ids",
            "iteration": "iteration",
            "max_iterations": "max_iterations",
            "done": "done",
            "effect": "termination",
        },
        "termination",
    )


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
    step = builder.input("step", ir.DataType.INT64, ["batch"])
    schedule = builder.input("schedule", ir.DataType.FLOAT, ["schedule_length"])
    final_index = op.Sub(op.Shape(schedule, start=0, end=1), op.Constant(value_ints=[1]))
    next_step = op.Min(op.Add(step, op.Constant(value_int=1)), final_index)
    sigma = op.Gather(schedule, step, axis=0)
    sigma_next = op.Gather(schedule, next_step, axis=0)
    delta = op.Sub(sigma_next, sigma)
    delta = op.Unsqueeze(delta, op.Constant(value_ints=[1, 2, 3]))
    next_sample = op.Add(sample, op.Mul(derivative, delta))
    builder.add_output(next_sample, "next_state")
    return _component(
        PolicyRole.SOLVER_STEP,
        graph,
        {
            "role": "solver_step",
            "state": "sample",
            "estimate": "derivative",
            "step": "step",
            "schedule": "schedule",
            "next_state": "next_state",
            "effect": "solver",
        },
        "solver",
    )


def build_masked_token_update() -> PolicyComponent:
    """Build replacement of masked positions with explicit RNG-counter threading."""
    graph, builder = _make_graph("masked_token_update")
    op = builder.op
    current = builder.input("current_tokens", ir.DataType.INT64, ["batch", "sequence"])
    proposed = builder.input("proposed_tokens", ir.DataType.INT64, ["batch", "sequence"])
    masked = builder.input("masked", ir.DataType.BOOL, ["batch", "sequence"])
    step = builder.input("step", ir.DataType.INT64, ["batch"])
    seed = builder.input("seed", ir.DataType.INT64, ["batch"])
    offset = builder.input("offset", ir.DataType.INT64, ["batch"])
    updated = op.Where(masked, proposed, current)
    # Consume the declared step without changing values; schedules that remask
    # tokens can be expressed by a richer artifact with the same semantic ports.
    updated = op.Add(updated, op.Unsqueeze(op.Mul(step, 0), op.Constant(value_ints=[-1])))
    updated.shape = ir.Shape(["batch", "sequence"])
    remaining = op.ConstantOfShape(op.Shape(masked), value=ir.tensor([False]))
    remaining.shape = ir.Shape(["batch", "sequence"])
    remaining_count = op.ReduceSum(
        op.Cast(remaining, to=ir.DataType.INT64),
        axes=[-1],
        keepdims=0,
    )
    done = op.Equal(remaining_count, op.Constant(value_int=0))
    done.shape = ir.Shape(["batch"])
    next_offset = op.Add(
        op.Add(offset, op.Constant(value_int=1)),
        op.Mul(seed, op.Constant(value_int=0)),
    )
    next_offset.shape = ir.Shape(["batch"])
    builder.add_output(updated, "next_state")
    builder.add_output(remaining, "next_mask")
    builder.add_output(next_offset, "next_offset")
    builder.add_output(done, "done")
    return _component(
        PolicyRole.MASKED_UPDATE,
        graph,
        {
            "role": "masked_update",
            "state": "current_tokens",
            "proposal": "proposed_tokens",
            "mask": "masked",
            "step": "step",
            "next_state": "next_state",
            "next_mask": "next_mask",
            "rng": {
                "seed": "seed",
                "offset": "offset",
                "next_offset": "next_offset",
            },
            "effect": "update",
        },
        "update",
    )


def build_speculative_acceptance() -> PolicyComponent:
    """Build per-token speculative acceptance and accepted-prefix length."""
    graph, builder = _make_graph("speculative_acceptance")
    op = builder.op
    target_scores = builder.input(
        "target_scores", ir.DataType.FLOAT, ["batch", "draft_sequence", "vocabulary"]
    )
    proposed_tokens = builder.input(
        "proposed_tokens", ir.DataType.INT64, ["batch", "draft_sequence"]
    )
    target_tokens = op.ArgMax(target_scores, axis=-1, keepdims=0)
    accepted = op.Equal(target_tokens, proposed_tokens)
    rejected = op.Cast(op.Not(accepted), to=ir.DataType.INT64)
    rejection_count = op.CumSum(rejected, op.Constant(value_int=-1))
    prefix = op.Cast(
        op.Equal(rejection_count, op.Constant(value_int=0)),
        to=ir.DataType.INT64,
    )
    accepted_count = op.ReduceSum(prefix, axes=[-1], keepdims=0)
    accepted_tokens = op.Where(
        op.Cast(prefix, to=ir.DataType.BOOL),
        proposed_tokens,
        op.ConstantOfShape(
            op.Shape(proposed_tokens),
            value=ir.tensor([0], dtype=ir.DataType.INT64),
        ),
    )
    draft_length = op.Shape(proposed_tokens, start=1, end=2)
    done = op.Equal(accepted_count, draft_length)
    builder.add_output(accepted_tokens, "accepted_tokens")
    builder.add_output(accepted_count, "accepted_len")
    builder.add_output(done, "done")
    return _component(
        PolicyRole.SPECULATIVE_ACCEPTANCE,
        graph,
        {
            "role": "speculative_verifier",
            "target_scores": "target_scores",
            "proposed_tokens": "proposed_tokens",
            "accepted_tokens": "accepted_tokens",
            "accepted_len": "accepted_len",
            "done": "done",
            "effect": "verify",
        },
        "verify",
    )


def build_token_state_update() -> PolicyComponent:
    """Build explicit token-history append and sequence-length update math."""
    graph, builder = _make_graph("token_state_update")
    op = builder.op
    current = builder.input("current", ir.DataType.INT64, ["batch", 1])
    update = builder.input("update", ir.DataType.INT64, ["batch"])
    next_state = op.Add(
        op.Mul(current, op.Constant(value_int=0)),
        op.Unsqueeze(update, op.Constant(value_ints=[-1])),
    )
    builder.add_output(next_state, "next")
    return _component(
        PolicyRole.STATE_UPDATE,
        graph,
        {
            "role": "state_update",
            "current": "current",
            "update": "update",
            "next": "next",
            "effect": "state",
        },
        "state",
    )
