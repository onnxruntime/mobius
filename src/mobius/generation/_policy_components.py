# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Small, model-agnostic ONNX graphs for generation policy and state math.

These components deliberately receive policy parameters as tensor inputs. They
contain no model-family dispatch and can therefore be invoked from a generic
workflow IR just like neural ONNX components.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import onnx_ir as ir
from onnxscript import GraphBuilder

from mobius._constants import OPSET_VERSION

_POLICY_CONTRACT_ID_METADATA = "mobius.generation.policy_contract_id"
_POLICY_CONTRACT_METADATA = "mobius.generation.policy_contract"
_POLICY_EFFECTS_METADATA = "mobius.generation.policy_effects"


@dataclass(frozen=True)
class PolicyComponent:
    """A versioned semantic contract and its executable ONNX model."""

    contract_id: str
    model: ir.Model
    contract: dict[str, object]
    effects: tuple[str, ...]

    def __post_init__(self) -> None:
        if "@" not in self.contract_id:
            raise ValueError("policy contract_id must include a version")
        self.model.graph.metadata_props[_POLICY_CONTRACT_ID_METADATA] = self.contract_id
        self.model.graph.metadata_props[_POLICY_CONTRACT_METADATA] = json.dumps(self.contract)
        self.model.graph.metadata_props[_POLICY_EFFECTS_METADATA] = json.dumps(self.effects)

    @classmethod
    def from_model(cls, model: ir.Model) -> PolicyComponent:
        """Restore a component from contract metadata embedded in its ONNX graph."""
        contract_id = model.graph.metadata_props.get(_POLICY_CONTRACT_ID_METADATA)
        if contract_id is None:
            raise ValueError("ONNX policy component is missing its versioned contract ID")
        contract = json.loads(model.graph.metadata_props[_POLICY_CONTRACT_METADATA])
        effects = tuple(json.loads(model.graph.metadata_props[_POLICY_EFFECTS_METADATA]))
        return cls(contract_id, model, contract, effects)


@dataclass(frozen=True)
class PolicyCapabilities:
    """Data-driven declaration of policy math required by a package."""

    sampler: str | None = None
    eos_termination: bool = False
    solver: str | None = None
    masked_update: bool = False
    speculative_acceptance: bool = False
    grammar_guidance: bool = False
    adaptive_k_max: int | None = None
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
    solvers = SOLVER_BUILDERS
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
    if capabilities.grammar_guidance:
        selected.append(("grammar_guidance", build_grammar_logits_processor()))
    if capabilities.adaptive_k_max is not None:
        selected.append(
            (
                "adaptive_k",
                build_adaptive_k_policy(max_k=capabilities.adaptive_k_max),
            )
        )
    if capabilities.token_state_update:
        selected.append(("token_state_update", build_token_state_update()))

    for name, component in selected:
        pkg.add_policy_component(name, component)
    return {name: f"policies/{name}.onnx" for name, _ in selected}


#: Element types the ONNX ``ConstantOfShape`` kernel can produce. The narrow
#: float types are absent, so a policy graph that must materialize an fp8 or
#: fp4 tensor has to fill a supported dtype and cast the result.
_CONSTANT_OF_SHAPE_DTYPES = frozenset(
    {
        ir.DataType.FLOAT,
        ir.DataType.FLOAT16,
        ir.DataType.BFLOAT16,
        ir.DataType.DOUBLE,
        ir.DataType.INT8,
        ir.DataType.INT16,
        ir.DataType.INT32,
        ir.DataType.INT64,
        ir.DataType.UINT8,
        ir.DataType.UINT16,
        ir.DataType.UINT32,
        ir.DataType.UINT64,
        ir.DataType.BOOL,
    }
)


def _component(
    contract_id: str,
    graph: ir.Graph,
    contract: dict[str, object],
    *_effects: str,
) -> PolicyComponent:
    # ONNX policy components are pure: RNG and state are explicit tensor data.
    model = ir.Model(graph, ir_version=11)
    model.producer_name = "mobius"
    return PolicyComponent(contract_id, model, contract, ())


def _make_graph(name: str) -> tuple[ir.Graph, GraphBuilder]:
    graph = ir.Graph(
        [],
        [],
        nodes=[],
        name=name,
        opset_imports={"": OPSET_VERSION},
    )
    return graph, GraphBuilder(graph)


def _set_public_shape(value: ir.Value, shape: list[str | int]) -> None:
    """Set ABI dimensions without GraphBuilder's graph-name qualification."""
    value.shape = ir.Shape(shape)


def build_greedy_sampler(
    *,
    effect: str = "sample",
    row_selective: bool = False,
) -> PolicyComponent:
    """Build row-selective greedy sampling over the final axis."""
    graph, builder = _make_graph("greedy_sampler")
    op = builder.op
    logits = builder.input(
        "logits",
        dtype=ir.DataType.FLOAT,
        shape=["batch", "vocabulary"],
    )
    sampled = op.ArgMax(logits, axis=-1, keepdims=0)
    if row_selective:
        active = builder.input("active", dtype=ir.DataType.BOOL, shape=["batch"])
        done = builder.input("done", dtype=ir.DataType.BOOL, shape=["batch"])
        enabled = op.And(active, op.Not(done))
        token_ids = op.Where(enabled, sampled, op.Constant(value_int=-1))
    else:
        token_ids = sampled
    builder.add_output(token_ids, "token")
    return _component(
        "onnx-genai.token-sampler@2" if row_selective else "onnx-genai.token-sampler@1",
        graph,
        {
            "role": "token_sampler",
            "mode": "greedy",
            **({"batching": "per_row", "inactive_rows": "preserve"} if row_selective else {}),
            "logits": "logits",
            **({"active": "active", "done": "done"} if row_selective else {}),
            "token": "token",
            "effect": effect,
        },
        effect,
    )


def build_last_token_logits(
    input_dtype: ir.DataType = ir.DataType.FLOAT,
) -> PolicyComponent:
    """Build ``[B,T,V] -> [B,V]`` selection and normalize logits to float32."""
    graph, builder = _make_graph("last_token_logits")
    logits = builder.input(
        "logits",
        dtype=input_dtype,
        shape=["batch", "sequence", "vocabulary"],
    )
    selected = builder.op.Gather(logits, builder.op.Constant(value_int=-1), axis=1)
    if input_dtype != ir.DataType.FLOAT:
        selected = builder.op.Cast(selected, to=ir.DataType.FLOAT)
    selected.shape = ir.Shape(["batch", "vocabulary"])
    builder.add_output(selected, "last_logits")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_boolean_not() -> PolicyComponent:
    """Build one synchronized ``continue = Not(Any(done))`` predicate."""
    graph, builder = _make_graph("boolean_not")
    done = builder.input("done", dtype=ir.DataType.BOOL, shape=["batch"])
    any_done = builder.op.ReduceMax(
        builder.op.Cast(done, to=ir.DataType.INT64),
        keepdims=1,
    )
    continued = builder.op.Equal(any_done, builder.op.Constant(value_int=0))
    continued.shape = ir.Shape([1])
    builder.add_output(continued, "continue")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_integer_minimum() -> PolicyComponent:
    """Compute the per-batch minimum of two integer lengths."""
    graph, builder = _make_graph("integer_minimum")
    left = builder.input("left", ir.DataType.INT64, ["batch"])
    right = builder.input("right", ir.DataType.INT64, ["batch"])
    minimum = builder.op.Min(left, right)
    minimum.shape = ir.Shape(["batch"])
    builder.add_output(minimum, "minimum")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_integer_add() -> PolicyComponent:
    """Add two per-row integer state values."""
    graph, builder = _make_graph("integer_add")
    left = builder.input("left", ir.DataType.INT64, ["batch"])
    right = builder.input("right", ir.DataType.INT64, ["batch"])
    total = builder.op.Add(left, right)
    total.shape = ir.Shape(["batch"])
    builder.add_output(total, "total")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_selective_integer_add() -> PolicyComponent:
    """Add per-row integer state only for active, unfinished rows."""
    graph, builder = _make_graph("selective_integer_add")
    op = builder.op
    left = builder.input("left", ir.DataType.INT64, ["batch"])
    right = builder.input("right", ir.DataType.INT64, ["batch"])
    active = builder.input("active", ir.DataType.BOOL, ["batch"])
    done = builder.input("done", ir.DataType.BOOL, ["batch"])
    enabled = op.And(active, op.Not(done))
    total = op.Where(enabled, op.Add(left, right), left)
    total.shape = ir.Shape(["batch"])
    builder.add_output(total, "total")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_integer_row_broadcast() -> PolicyComponent:
    """Broadcast one scalar loop control to the current dynamic batch."""
    graph, builder = _make_graph("integer_row_broadcast")
    op = builder.op
    value = builder.input("value", ir.DataType.INT64, [1])
    active = builder.input("active", ir.DataType.BOOL, ["batch"])
    rows = op.Shape(active)
    result = op.Expand(value, rows)
    result.shape = ir.Shape(["batch"])
    builder.add_output(result, "rows")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_termination_batch_initializer() -> PolicyComponent:
    """Normalize explicit ragged EOS sets and per-row generation limits."""
    graph, builder = _make_graph("termination_batch_initializer")
    op = builder.op
    eos_ids = builder.input("input_eos_ids", ir.DataType.INT64, ["batch", "num_eos"])
    eos_lengths = builder.input("input_eos_lengths", ir.DataType.INT64, ["batch"])
    max_iterations = builder.input("input_max_iterations", ir.DataType.INT64, ["batch"])
    fallback_max_iterations = builder.input("fallback_max_iterations", ir.DataType.INT64, [1])
    active = builder.input("active", ir.DataType.BOOL, ["batch"])
    batch_shape = op.Shape(active)
    row_max_iterations = op.Where(
        op.Greater(max_iterations, op.Constant(value_int=0)),
        max_iterations,
        op.Expand(fallback_max_iterations, batch_shape),
    )
    row_eos_ids = op.Identity(eos_ids)
    eos_count = op.Identity(eos_lengths)
    row_eos_ids.shape = ir.Shape(["batch", "num_eos"])
    eos_count.shape = ir.Shape(["batch"])
    row_max_iterations.shape = ir.Shape(["batch"])
    builder.add_output(row_eos_ids, "row_eos_ids")
    builder.add_output(eos_count, "eos_lengths")
    builder.add_output(row_max_iterations, "max_iterations")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_batch_minimum() -> PolicyComponent:
    """Synchronize a per-batch integer length to one conservative scalar."""
    graph, builder = _make_graph("batch_minimum")
    values = builder.input("values", ir.DataType.INT64, ["batch"])
    minimum = builder.op.ReduceMin(values, keepdims=1)
    minimum.shape = ir.Shape([1])
    builder.add_output(minimum, "minimum")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_proposal_metrics() -> PolicyComponent:
    """Derive evaluated width and budget fullness from a dense proposal."""
    graph, builder = _make_graph("proposal_metrics")
    op = builder.op
    tokens = builder.input("proposed_tokens", ir.DataType.INT64, ["batch", "proposal"])
    requested_k = builder.input("requested_k", ir.DataType.INT64, ["batch"])
    batch = op.Shape(tokens, start=0, end=1)
    length = op.Expand(op.Shape(tokens, start=1, end=2), batch)
    filled = op.Equal(length, requested_k)
    length.shape = ir.Shape(["batch"])
    filled.shape = ir.Shape(["batch"])
    builder.add_output(length, "evaluated")
    builder.add_output(filled, "filled_proposal_budget")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_sequence_length() -> PolicyComponent:
    """Expand a dense rank-2 token block's sequence width per batch."""
    graph, builder = _make_graph("sequence_length")
    op = builder.op
    tokens = builder.input("tokens", ir.DataType.INT64, ["batch", "sequence"])
    length = op.Expand(
        op.Shape(tokens, start=1, end=2),
        op.Shape(tokens, start=0, end=1),
    )
    length.shape = ir.Shape(["batch"])
    builder.add_output(length, "length")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_iteration_cast(dtype: ir.DataType) -> PolicyComponent:
    """Cast the generic int64 loop induction value for a model timestep port."""
    graph, builder = _make_graph("iteration_cast")
    iteration = builder.input("iteration", dtype=ir.DataType.INT64, shape=["batch"])
    timestep = builder.op.Cast(iteration, to=dtype)
    timestep.shape = ir.Shape(["batch"])
    builder.add_output(timestep, "timestep")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_schedule_constant(values: list[float]) -> PolicyComponent:
    """Materialize a producer-selected diffusion schedule inside ONNX."""
    if len(values) < 2:
        raise ValueError("a diffusion schedule requires at least two values")
    graph, builder = _make_graph("diffusion_schedule")
    schedule = builder.op.Constant(
        value=ir.tensor(values, dtype=ir.DataType.FLOAT),
    )
    schedule.shape = ir.Shape([len(values)])
    builder.add_output(schedule, "schedule")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_schedule_lookup(dtype: ir.DataType) -> PolicyComponent:
    """Gather the current schedule value and cast it for a model timestep port."""
    graph, builder = _make_graph("schedule_lookup")
    op = builder.op
    schedule = builder.input("schedule", ir.DataType.FLOAT, ["schedule_length"])
    step = builder.input("step", ir.DataType.INT64, ["batch"])
    timestep = op.Cast(op.Gather(schedule, step, axis=0), to=dtype)
    timestep.shape = ir.Shape(["batch"])
    builder.add_output(timestep, "timestep")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_tts_state_initializer(num_code_groups: int) -> PolicyComponent:
    """Create an empty codec history and zeroed current frame from prompt batch."""
    if num_code_groups < 1:
        raise ValueError("num_code_groups must be positive")
    graph, builder = _make_graph("tts_state_initializer")
    op = builder.op
    prompt = builder.input(
        "prompt_tokens", dtype=ir.DataType.INT64, shape=["batch", "sequence"]
    )
    batch = op.Shape(prompt, start=0, end=1)
    frame_shape = op.Concat(batch, op.Constant(value_ints=[num_code_groups]), axis=0)
    history_shape = op.Concat(batch, op.Constant(value_ints=[0, num_code_groups]), axis=0)
    frame = op.ConstantOfShape(frame_shape, value=ir.tensor([0], dtype=ir.DataType.INT64))
    token_slot = op.ConstantOfShape(
        op.Concat(batch, op.Constant(value_ints=[1]), axis=0),
        value=ir.tensor([0], dtype=ir.DataType.INT64),
    )
    history = op.ConstantOfShape(history_shape, value=ir.tensor([0], dtype=ir.DataType.INT64))
    frame.shape = ir.Shape(["batch", num_code_groups])
    token_slot.shape = ir.Shape(["batch", 1])
    history.shape = ir.Shape(["batch", 0, num_code_groups])
    builder.add_output(frame, "frame_codes")
    builder.add_output(token_slot, "token_slot")
    builder.add_output(history, "code_history")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_tts_decoder_state_initializer(
    decoder: ir.Model,
    *,
    graph_name: str,
    embedding_input: str,
    attention_mask_input: str,
    position_ids_input: str,
    cache_inputs: list[str],
) -> PolicyComponent:
    """Initialize masks, positions, and empty KV state from prefill embeddings."""
    graph, builder = _make_graph(graph_name)
    op = builder.op
    inputs = {value.name: value for value in decoder.graph.inputs}
    embedding = inputs[embedding_input]
    prompt = builder.input(
        "prefill_embeds",
        embedding.dtype,
        ["batch", "prefill_sequence", list(embedding.shape)[-1]],
    )
    batch_shape = op.Shape(prompt, start=0, end=1)
    sequence_shape = op.Shape(prompt, start=1, end=2)
    mask_shape = op.Concat(batch_shape, sequence_shape, axis=0)
    attention_value = inputs[attention_mask_input]
    attention = op.Cast(
        op.ConstantOfShape(mask_shape, value=ir.tensor([1])),
        to=attention_value.dtype,
    )
    attention.shape = attention_value.shape
    body_attention = op.Concat(
        attention,
        op.Cast(
            op.ConstantOfShape(
                op.Concat(batch_shape, op.Constant(value_ints=[1]), axis=0),
                value=ir.tensor([1]),
            ),
            to=attention_value.dtype,
        ),
        axis=1,
    )
    body_attention.shape = ir.Shape(["batch", "prefill_sequence + 1"])

    position_value = inputs[position_ids_input]
    position_rank = len(position_value.shape or [])
    position_range = op.Range(
        op.Constant(value_int=0),
        op.Squeeze(sequence_shape, [0]),
        op.Constant(value_int=1),
    )
    if position_rank == 2:
        position_shape = mask_shape
        positions = op.Expand(op.Unsqueeze(position_range, [0]), position_shape)
        body_position = op.Expand(
            op.Cast(sequence_shape, to=position_value.dtype),
            op.Concat(batch_shape, op.Constant(value_ints=[1]), axis=0),
        )
        body_position.shape = ir.Shape(["batch", 1])
    elif position_rank == 3:
        position_shape = op.Concat(
            op.Constant(value_ints=[3]), batch_shape, sequence_shape, axis=0
        )
        positions = op.Expand(op.Unsqueeze(position_range, [0, 1]), position_shape)
        body_position = op.Expand(
            op.Reshape(
                op.Cast(sequence_shape, to=position_value.dtype),
                op.Constant(value_ints=[1, 1, 1]),
            ),
            op.Concat(
                op.Constant(value_ints=[3]),
                batch_shape,
                op.Constant(value_ints=[1]),
                axis=0,
            ),
        )
        body_position.shape = ir.Shape([3, "batch", 1])
    else:
        raise ValueError("TTS decoder position_ids must be rank 2 or 3")
    positions = op.Cast(positions, to=position_value.dtype)
    positions.shape = position_value.shape

    builder.add_output(attention, attention_mask_input)
    builder.add_output(positions, position_ids_input)
    builder.add_output(body_attention, "body_attention_mask")
    builder.add_output(body_position, "body_position_ids")
    for name in cache_inputs:
        value = inputs[name]
        dimensions = list(value.shape or [])
        shape_parts = []
        for axis, dimension in enumerate(dimensions):
            text = str(getattr(dimension, "value", dimension))
            if axis == 0:
                shape_parts.append(batch_shape)
            elif "sequence" in text:
                shape_parts.append(op.Constant(value_ints=[0]))
            elif isinstance(dimension, int):
                shape_parts.append(op.Constant(value_ints=[dimension]))
            else:
                raise ValueError(f"cache input {name!r} has unsupported dimension {text!r}")
        empty = op.ConstantOfShape(
            op.Concat(*shape_parts, axis=0),
            value=ir.tensor(
                [0.0 if value.dtype.is_floating_point else 0],
                dtype=value.dtype,
            ),
        )
        empty.shape = value.shape
        builder.add_output(empty, name)
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_tts_decoder_step_update(
    *,
    graph_name: str,
    attention_dtype: ir.DataType,
    position_dtype: ir.DataType,
    position_rank: int,
) -> PolicyComponent:
    """Append one decoder mask slot and increment rank-2 or rank-3 positions."""
    graph, builder = _make_graph(graph_name)
    op = builder.op
    attention = builder.input("attention_mask", attention_dtype, ["batch", "context"])
    one_shape = op.Concat(
        op.Shape(attention, start=0, end=1),
        op.Constant(value_ints=[1]),
        axis=0,
    )
    one = op.CastLike(op.ConstantOfShape(one_shape, value=ir.tensor([1])), attention)
    next_attention = op.Concat(attention, one, axis=1)
    next_attention.shape = ir.Shape(["batch", "context + 1"])
    if position_rank == 2:
        position_shape: list[int | str] = ["batch", 1]
    elif position_rank == 3:
        position_shape = [3, "batch", 1]
    else:
        raise ValueError("TTS decoder position_ids must be rank 2 or 3")
    position = builder.input("position_ids", position_dtype, position_shape)
    next_position = op.Add(position, op.CastLike(op.Constant(value_int=1), position))
    next_position.shape = position.shape
    builder.add_output(next_attention, "next_attention_mask")
    builder.add_output(next_position, "next_position_ids")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_code_frame_update(
    num_code_groups: int, *, scalar_index: bool = False
) -> PolicyComponent:
    """Scatter one predicted code into the current codec frame."""
    graph, builder = _make_graph("code_frame_update")
    op = builder.op
    frame = builder.input(
        "frame_codes",
        dtype=ir.DataType.INT64,
        shape=["batch", num_code_groups],
    )
    token = builder.input("token", dtype=ir.DataType.INT64, shape=["batch"])
    index = builder.input(
        "index",
        dtype=ir.DataType.INT64,
        shape=[] if scalar_index else ["batch"],
    )
    if scalar_index:
        index = op.Expand(index, op.Shape(token))
    updated = op.ScatterElements(
        frame,
        op.Unsqueeze(index, op.Constant(value_ints=[-1])),
        op.Unsqueeze(token, op.Constant(value_ints=[-1])),
        axis=1,
    )
    updated.shape = frame.shape
    builder.add_output(updated, "next_frame")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_code_history_append(num_code_groups: int) -> PolicyComponent:
    """Append a completed codec frame to a growing frame history."""
    graph, builder = _make_graph("code_history_append")
    op = builder.op
    history = builder.input(
        "history",
        dtype=ir.DataType.INT64,
        shape=["batch", "frames", num_code_groups],
    )
    frame = builder.input("frame", dtype=ir.DataType.INT64, shape=["batch", num_code_groups])
    next_history = op.Concat(
        history,
        op.Unsqueeze(frame, op.Constant(value_ints=[1])),
        axis=1,
    )
    next_history.shape = ir.Shape(["batch", "frames + 1", num_code_groups])
    builder.add_output(next_history, "next_history")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_codec_layout_transpose(num_code_groups: int) -> PolicyComponent:
    """Convert frame-major ``[B,F,G]`` history to codec-major ``[B,G,F]``."""
    graph, builder = _make_graph("codec_layout_transpose")
    history = builder.input(
        "history",
        dtype=ir.DataType.INT64,
        shape=["batch", "frames", num_code_groups],
    )
    codes = builder.op.Transpose(history, perm=[0, 2, 1])
    codes.shape = ir.Shape(["batch", num_code_groups, "frames"])
    builder.add_output(codes, "codes")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_model_token_cast(dtype: ir.DataType) -> PolicyComponent:
    """Cast the canonical int64 token state to a decoder's integer dtype."""
    graph, builder = _make_graph("model_token_cast")
    token = builder.input("token", dtype=ir.DataType.INT64, shape=["batch", 1])
    model_token = builder.op.Cast(token, to=dtype)
    model_token.shape = ir.Shape(["batch", 1])
    builder.add_output(model_token, "model_token")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_empty_features(dtype: ir.DataType, feature_size: int) -> PolicyComponent:
    """Build the empty feature matrix used by multimodal text-only requests."""
    graph, builder = _make_graph("empty_features")
    features = builder.op.ConstantOfShape(
        builder.op.Constant(value_ints=[0, feature_size]),
        value=ir.tensor([0.0], dtype=dtype),
    )
    features.shape = ir.Shape([0, feature_size])
    builder.add_output(features, "features")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_empty_batched_features(
    dtype: ir.DataType,
    feature_size: int,
) -> PolicyComponent:
    """Build ``(1, 0, hidden)`` features for an optional single-image path."""
    graph, builder = _make_graph("empty_batched_features")
    features = builder.op.ConstantOfShape(
        builder.op.Constant(value_ints=[1, 0, feature_size]),
        value=ir.tensor([0.0], dtype=dtype),
    )
    features.shape = ir.Shape([1, 0, feature_size])
    builder.add_output(features, "features")
    return _component("mobius.policy.auxiliary@1", graph, {})


def rotary_axis_count(position_value: ir.Value) -> int | None:
    """Number of rotary axes a decoder's ``position_ids`` input carries.

    A plain decoder reads ``(batch, sequence)`` positions and gets ``None``. A
    decoder with multi-axis rotary embeddings reads
    ``(sections, batch, sequence)``, where the leading axis is a fixed count of
    rotary axes and must therefore be a static dimension: it is a property of
    the exported graph, not of the request.
    """
    shape = position_value.shape
    if shape is None or len(shape) != 3:
        return None
    leading = shape[0]
    sections = getattr(leading, "value", leading)
    if not isinstance(sections, int):
        raise TypeError(
            f"position input {position_value.name!r} declares a rank-3 shape "
            f"whose leading rotary-axis count {sections!r} is symbolic. The "
            "number of rotary axes is fixed by the exported graph, so it must "
            "be a static dimension."
        )
    return sections


def build_decoder_state_initializer(
    decoder: ir.Model,
    *,
    token_input: str | None,
    prompt_dtype: ir.DataType | None = None,
    attention_mask_input: str | None,
    position_ids_input: str | None,
    cache_inputs: list[str],
    fixed_capacity: bool = False,
    ragged: bool = False,
    write_indices_output: str | None = None,
) -> PolicyComponent:
    """Build prompt-derived decoder state, optionally with capture-stable storage.

    ``write_indices_output`` names the graph port of a static (indexed-scatter)
    KV cache. Prefill writes the whole prompt chunk starting at slot zero, so the
    initial destinations are zeros and the resulting logical length is the prompt
    length; both are emitted here so no consumer has to infer them.
    """
    if fixed_capacity and attention_mask_input is None:
        raise ValueError(
            "fixed-capacity decoder state requires an attention-mask input to carry "
            "each row's logical length"
        )
    graph, builder = _make_graph("decoder_state_initializer")
    op = builder.op
    decoder_inputs = {value.name: value for value in decoder.graph.inputs}
    if prompt_dtype is None:
        if token_input is None:
            raise ValueError("token_input or prompt_dtype is required")
        prompt_dtype = decoder_inputs[token_input].dtype
    prompt = builder.input(
        "prompt_tokens",
        dtype=prompt_dtype,
        shape=["batch", "prompt_sequence"],
    )
    prompt_shape = op.Shape(prompt)
    batch_shape = op.Shape(prompt, start=0, end=1)
    sequence_shape = op.Shape(prompt, start=1, end=2)
    sequence_length = op.Squeeze(sequence_shape, op.Constant(value_ints=[0]))
    if ragged:
        provided_prompt_lengths = builder.input(
            "prompt_lengths",
            dtype=ir.DataType.INT64,
            shape=["batch"],
        )
        full_prompt_lengths = op.Expand(
            op.Unsqueeze(sequence_length, [0]),
            batch_shape,
        )
        prompt_lengths = op.Where(
            op.Greater(provided_prompt_lengths, op.Constant(value_int=0)),
            provided_prompt_lengths,
            full_prompt_lengths,
        )
    else:
        prompt_lengths = op.Expand(op.Unsqueeze(sequence_length, [0]), batch_shape)
    capacity = None
    if fixed_capacity:
        max_iterations = builder.input(
            "max_iterations",
            dtype=ir.DataType.INT64,
            shape=[1],
        )
        capacity = op.Add(
            sequence_length,
            op.Squeeze(max_iterations, [0]),
        )
        attention_shape = op.Concat(
            batch_shape,
            op.Unsqueeze(capacity, op.Constant(value_ints=[0])),
            axis=0,
        )
        offsets = op.Range(
            op.Constant(value_int=0),
            capacity,
            op.Constant(value_int=1),
        )
        offsets = op.Expand(
            op.Unsqueeze(offsets, op.Constant(value_ints=[0])),
            attention_shape,
        )

    attention = None
    body_attention = None
    if attention_mask_input is not None:
        attention_value = decoder_inputs[attention_mask_input]
        if fixed_capacity:
            # Native ORT GenAI binds one persistent full-capacity mask for prefill
            # and decode, then enables the next logical slot before each decode.
            attention = op.Cast(
                op.Less(offsets, op.Unsqueeze(prompt_lengths, [-1])),
                to=attention_value.dtype,
            )
            attention.shape = ir.Shape(["batch", "capacity"])
            body_attention = op.Identity(attention)
            body_attention.shape = attention.shape
        else:
            offsets = op.Range(
                op.Constant(value_int=0),
                sequence_length,
                op.Constant(value_int=1),
            )
            offsets = op.Expand(op.Unsqueeze(offsets, [0]), prompt_shape)
            attention = op.Cast(
                op.Less(offsets, op.Unsqueeze(prompt_lengths, [-1])),
                to=attention_value.dtype,
            )
            attention.shape = attention_value.shape
            body_attention = op.Concat(
                attention,
                op.Cast(
                    op.ConstantOfShape(
                        op.Concat(batch_shape, op.Constant(value_ints=[1]), axis=0),
                        value=ir.tensor([1]),
                    ),
                    to=attention_value.dtype,
                ),
                axis=1,
            )
            body_attention.shape = ir.Shape(["batch", "prompt_sequence + 1"])
    token_slot = op.ConstantOfShape(
        op.Concat(batch_shape, op.Constant(value_ints=[1]), axis=0),
        value=ir.tensor([0], dtype=ir.DataType.INT64),
    )
    token_slot.shape = ir.Shape(["batch", 1])

    positions = None
    body_position = None
    if position_ids_input is not None:
        position_value = decoder_inputs[position_ids_input]
        sections = rotary_axis_count(position_value)
        positions = op.Range(
            op.Constant(value_int=0),
            op.Squeeze(sequence_shape, op.Constant(value_ints=[0])),
            op.Constant(value_int=1),
        )
        positions = op.Expand(
            op.Unsqueeze(positions, op.Constant(value_ints=[0])), prompt_shape
        )
        positions = op.Cast(positions, to=position_value.dtype)
        # (batch, prompt_sequence)
        positions.shape = ir.Shape(["batch", "prompt_sequence"])
        body_position = op.Unsqueeze(
            op.Cast(prompt_lengths, to=position_value.dtype),
            [-1],
        )
        # (batch, 1)
        body_position.shape = ir.Shape(["batch", 1])
        if sections is not None:
            # A decoder with multi-axis rotary positions reads
            # (sections, batch, sequence): one position row per rotary axis.
            # Every axis carries the same sequential position here, which is
            # what the axes agree on for a pure token stream. A component that
            # lays media out differently across axes states that layout itself;
            # this initializer never invents one.
            positions = op.Expand(
                op.Unsqueeze(positions, op.Constant(value_ints=[0])),
                op.Concat(op.Constant(value_ints=[sections]), prompt_shape, axis=0),
            )
            positions.shape = ir.Shape([sections, "batch", "prompt_sequence"])
            body_position = op.Expand(
                op.Unsqueeze(body_position, op.Constant(value_ints=[0])),
                op.Concat(
                    op.Constant(value_ints=[sections]),
                    batch_shape,
                    op.Constant(value_ints=[1]),
                    axis=0,
                ),
            )
            body_position.shape = ir.Shape([sections, "batch", 1])
    if attention_mask_input is not None:
        assert attention is not None and body_attention is not None
        builder.add_output(attention, attention_mask_input)
    if position_ids_input is not None:
        assert positions is not None and body_position is not None
        builder.add_output(positions, position_ids_input)
    if attention_mask_input is not None:
        assert body_attention is not None
        builder.add_output(body_attention, "body_attention_mask")
    if position_ids_input is not None:
        builder.add_output(body_position, "body_position_ids")
    builder.add_output(token_slot, "token_slot")
    if ragged:
        generated_lengths = op.ConstantOfShape(
            batch_shape,
            value=ir.tensor([0], dtype=ir.DataType.INT64),
        )
        generated_lengths.shape = ir.Shape(["batch"])
        builder.add_output(generated_lengths, "generated_lengths")
    if fixed_capacity or write_indices_output is not None:
        cache_lengths = op.Identity(prompt_lengths)
        cache_lengths.shape = ir.Shape(["batch"])
        builder.add_output(cache_lengths, "cache_lengths")
    if write_indices_output is not None:
        # Prefill scatters the whole prompt chunk from slot zero for every row;
        # the per-row cursor only diverges once decode advances rows separately.
        write_indices = op.ConstantOfShape(
            batch_shape,
            value=ir.tensor([0], dtype=ir.DataType.INT64),
        )
        write_indices.shape = ir.Shape(["batch"])
        builder.add_output(write_indices, write_indices_output)

    for name in cache_inputs:
        value = decoder_inputs[name]
        if value.shape is None:
            raise ValueError(f"cache input {name!r} must declare a shape")
        dimensions = list(value.shape)
        shape_parts = []
        for axis, dimension in enumerate(dimensions):
            dimension_text = str(getattr(dimension, "value", dimension))
            if axis == 0:
                shape_parts.append(batch_shape)
            elif "sequence" in dimension_text:
                if fixed_capacity:
                    assert capacity is not None
                    shape_parts.append(op.Unsqueeze(capacity, op.Constant(value_ints=[0])))
                else:
                    shape_parts.append(op.Constant(value_ints=[0]))
            elif isinstance(dimension, int):
                shape_parts.append(op.Constant(value_ints=[dimension]))
            else:
                raise ValueError(
                    f"cache input {name!r} has unsupported symbolic "
                    f"dimension {dimension_text!r}"
                )
        cache_shape = op.Concat(*shape_parts, axis=0)
        # ``ConstantOfShape`` kernels do not implement the narrow float types, so
        # an fp8 KV cache must be materialized in a supported dtype and cast.
        fill_dtype = (
            value.dtype if value.dtype in _CONSTANT_OF_SHAPE_DTYPES else ir.DataType.FLOAT
        )
        zero = 0.0 if fill_dtype.is_floating_point else 0
        empty = op.ConstantOfShape(
            cache_shape,
            value=ir.tensor([zero], dtype=fill_dtype),
        )
        if fill_dtype != value.dtype:
            empty = op.Cast(empty, to=value.dtype)
        empty.shape = (
            ir.Shape(
                [
                    "capacity" if "sequence" in str(getattr(d, "value", d)) else d
                    for d in dimensions
                ]
            )
            if fixed_capacity
            else value.shape
        )
        builder.add_output(empty, name)

    return _component("mobius.policy.auxiliary@1", graph, {})


def build_decoder_step_update(
    *,
    attention_dtype: ir.DataType | None,
    position_dtype: ir.DataType | None,
    fixed_capacity: bool = False,
    position_sections: int | None = None,
) -> PolicyComponent:
    """Build one-token attention-mask and position update."""
    if attention_dtype is None and position_dtype is None:
        raise ValueError("decoder step update requires an attention mask or position ids")
    if attention_dtype is None and fixed_capacity:
        raise ValueError("fixed-capacity decoder step update requires an attention mask")
    graph, builder = _make_graph("decoder_step_update")
    op = builder.op
    if attention_dtype is not None:
        attention = builder.input(
            "attention_mask",
            dtype=attention_dtype,
            shape=["batch", "context"],
        )
        if fixed_capacity:
            logical_length = builder.input(
                "logical_length",
                dtype=ir.DataType.INT64,
                shape=["batch"],
            )
            offsets = op.Range(
                op.Constant(value_int=0),
                op.Squeeze(op.Shape(attention, start=1, end=2), [0]),
                op.Constant(value_int=1),
            )
            slots = op.Equal(
                op.Unsqueeze(offsets, [0]),
                op.Unsqueeze(logical_length, [1]),
            )
            next_attention = op.Where(
                slots,
                op.CastLike(op.Constant(value_int=1), attention),
                attention,
            )
            next_attention.shape = ir.Shape(["batch", "context"])
        else:
            batch_shape = op.Shape(attention, start=0, end=1)
            one_shape = op.Concat(batch_shape, op.Constant(value_ints=[1]), axis=0)
            one = op.CastLike(op.ConstantOfShape(one_shape, value=ir.tensor([1])), attention)
            next_attention = op.Concat(attention, one, axis=1)
            next_attention.shape = ir.Shape(["batch", "context + 1"])
        builder.add_output(next_attention, "next_attention_mask")
    if position_dtype is not None:
        # A multi-axis rotary decoder carries one position row per rotary axis;
        # advancing one token advances every axis, so the update is the same
        # `+1` at either rank and only the declared shape differs.
        position_shape: list[int | str] = (
            ["batch", 1] if position_sections is None else [position_sections, "batch", 1]
        )
        position = builder.input(
            "position_ids",
            dtype=position_dtype,
            shape=position_shape,
        )
        next_position = op.Add(position, op.CastLike(op.Constant(value_int=1), position))
        next_position.shape = ir.Shape(position_shape)
        builder.add_output(next_position, "next_position_ids")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_grammar_logits_processor() -> PolicyComponent:
    """Apply a grammar adapter's mask and return a forced or sampled token."""
    graph, builder = _make_graph("grammar_guided_sampler")
    op = builder.op
    logits = builder.input("logits", ir.DataType.FLOAT, ["batch", "vocabulary"])
    logits_mask = builder.input("logits_mask", ir.DataType.BOOL, ["batch", "vocabulary"])
    forced_tokens = builder.input("forced_tokens", ir.DataType.INT64, ["batch", 1])
    forced_length = builder.input("forced_length", ir.DataType.INT64, ["batch"])
    blocked = op.CastLike(op.Constant(value_float=-3.4028235e38), logits)
    masked_logits = op.Where(logits_mask, logits, blocked)
    sampled = op.ArgMax(masked_logits, axis=-1, keepdims=1)
    token = op.Where(
        op.Unsqueeze(op.Greater(forced_length, op.Constant(value_int=0)), [-1]),
        forced_tokens,
        sampled,
    )
    token.shape = ir.Shape(["batch", 1])
    builder.add_output(token, "token")
    return _component("onnx-genai.grammar-guidance@1", graph, {})


def build_adaptive_k_policy(*, max_k: int = 16, min_k: int = 1) -> PolicyComponent:
    """Build the advisory adjacent-probe adaptive-K controller from ORT GenAI."""
    if not 1 <= min_k <= max_k:
        raise ValueError("adaptive K requires 1 <= min_k <= max_k")
    graph, builder = _make_graph("adaptive_k_policy")
    op = builder.op
    k_slots = max_k + 1
    estimate_slots = 4 * k_slots + 4

    current_k = builder.input("current_k", ir.DataType.INT64, ["batch"])
    accepted = builder.input("accepted", ir.DataType.INT64, ["batch"])
    evaluated = builder.input("evaluated", ir.DataType.INT64, ["batch"])
    committed_tokens = builder.input("committed_tokens", ir.DataType.INT64, ["batch"])
    filled_proposal_budget = builder.input(
        "filled_proposal_budget", ir.DataType.BOOL, ["batch"]
    )
    draft_ms = builder.input("draft_ms", ir.DataType.FLOAT, ["batch"])
    target_ms = builder.input("target_ms", ir.DataType.FLOAT, ["batch"])
    estimates = builder.input("estimates", ir.DataType.FLOAT, ["batch", estimate_slots])

    def section(start: int, end: int):
        return op.Slice(
            estimates,
            op.Constant(value_ints=[start]),
            op.Constant(value_ints=[end]),
            op.Constant(value_ints=[1]),
        )

    token_estimates = section(0, k_slots)
    millisecond_estimates = section(k_slots, 2 * k_slots)
    acceptance_estimates = section(2 * k_slots, 3 * k_slots)
    sample_counts = section(3 * k_slots, 4 * k_slots)
    controller = section(4 * k_slots, estimate_slots)
    probe_origin_k = op.Clip(
        op.Cast(op.Slice(controller, [0], [1], [1]), to=ir.DataType.INT64),
        op.Constant(value_int=0),
        op.Constant(value_int=max_k),
    )
    probe_observations = op.Cast(op.Slice(controller, [1], [2], [1]), to=ir.DataType.INT64)
    stable_observations = op.Cast(op.Slice(controller, [2], [3], [1]), to=ir.DataType.INT64)
    probe_cooldown = op.Cast(op.Slice(controller, [3], [4], [1]), to=ir.DataType.INT64)

    index = op.Unsqueeze(current_k, [-1])
    zero_i = op.Constant(value_int=0)
    one_i = op.Constant(value_int=1)
    two_i = op.Constant(value_int=2)
    total_ms = op.Add(draft_ms, target_ms)
    finite_time = op.Not(op.Or(op.IsNaN(total_ms), op.IsInf(total_ms)))
    valid = op.And(
        op.Greater(evaluated, zero_i),
        op.And(
            op.Greater(committed_tokens, zero_i),
            op.And(
                filled_proposal_budget,
                op.And(op.Greater(total_ms, op.Constant(value_float=0.0)), finite_time),
            ),
        ),
    )
    valid_col = op.Unsqueeze(valid, [-1])

    old_tokens = op.GatherElements(token_estimates, index, axis=1)
    old_ms = op.GatherElements(millisecond_estimates, index, axis=1)
    old_acceptance = op.GatherElements(acceptance_estimates, index, axis=1)
    old_samples = op.GatherElements(sample_counts, index, axis=1)
    sample_tokens = op.Unsqueeze(op.Cast(committed_tokens, to=ir.DataType.FLOAT), [-1])
    sample_acceptance = op.Unsqueeze(
        op.Div(
            op.Cast(accepted, to=ir.DataType.FLOAT),
            op.Cast(evaluated, to=ir.DataType.FLOAT),
        ),
        [-1],
    )
    sample_ms = op.Unsqueeze(total_ms, [-1])
    first_sample = op.Equal(old_samples, op.Constant(value_float=0.0))
    alpha = op.Constant(value_float=0.25)

    def ewma(old, sample):
        return op.Where(
            first_sample,
            sample,
            op.Add(old, op.Mul(alpha, op.Sub(sample, old))),
        )

    updated_tokens = op.Where(valid_col, ewma(old_tokens, sample_tokens), old_tokens)
    updated_ms = op.Where(valid_col, ewma(old_ms, sample_ms), old_ms)
    updated_acceptance = op.Where(
        valid_col, ewma(old_acceptance, sample_acceptance), old_acceptance
    )
    updated_samples = op.Where(
        valid_col,
        op.Add(old_samples, op.Constant(value_float=1.0)),
        old_samples,
    )
    next_tokens = op.ScatterElements(token_estimates, index, updated_tokens, axis=1)
    next_ms = op.ScatterElements(millisecond_estimates, index, updated_ms, axis=1)
    next_acceptance = op.ScatterElements(
        acceptance_estimates, index, updated_acceptance, axis=1
    )
    next_samples = op.ScatterElements(sample_counts, index, updated_samples, axis=1)

    current_throughput = op.Div(
        updated_tokens,
        op.Max(updated_ms, op.Constant(value_float=1e-12)),
    )
    origin_index = probe_origin_k
    origin_tokens = op.GatherElements(token_estimates, origin_index, axis=1)
    origin_ms = op.GatherElements(millisecond_estimates, origin_index, axis=1)
    origin_throughput = op.Div(
        origin_tokens,
        op.Max(origin_ms, op.Constant(value_float=1e-12)),
    )
    current_col = index
    in_probe = op.Greater(probe_origin_k, zero_i)
    next_probe_count = op.Add(probe_observations, one_i)
    probing_up = op.Greater(current_col, probe_origin_k)
    severe_regression = op.And(
        valid_col,
        op.And(
            in_probe,
            op.And(
                op.Greater(origin_throughput, op.Constant(value_float=0.0)),
                op.Less(
                    current_throughput,
                    op.Mul(origin_throughput, op.Constant(value_float=0.8)),
                ),
            ),
        ),
    )
    probe_ready = op.And(
        valid_col,
        op.And(
            in_probe,
            op.And(
                op.Not(severe_regression),
                op.GreaterOrEqual(next_probe_count, two_i),
            ),
        ),
    )
    throughput_safe = op.Or(
        op.LessOrEqual(origin_throughput, op.Constant(value_float=0.0)),
        op.GreaterOrEqual(
            current_throughput,
            op.Mul(origin_throughput, op.Constant(value_float=0.97)),
        ),
    )
    acceptance_safe = op.Or(
        op.Not(probing_up),
        op.GreaterOrEqual(updated_acceptance, op.Constant(value_float=0.75)),
    )
    keep_probe = op.And(probe_ready, op.And(throughput_safe, acceptance_safe))
    finish_probe = op.Or(severe_regression, probe_ready)
    probe_k = op.Where(
        finish_probe,
        op.Where(keep_probe, current_col, probe_origin_k),
        current_col,
    )
    probe_origin = op.Where(finish_probe, zero_i, probe_origin_k)
    probe_count = op.Where(
        finish_probe,
        zero_i,
        op.Where(valid_col, next_probe_count, probe_observations),
    )
    probe_stable = op.Where(finish_probe, zero_i, stable_observations)
    probe_next_cooldown = op.Where(
        finish_probe,
        op.Where(keep_probe, one_i, op.Constant(value_int=6)),
        probe_cooldown,
    )

    stable_count = op.Where(valid_col, op.Add(stable_observations, one_i), stable_observations)
    cooling = op.And(valid_col, op.Greater(probe_cooldown, zero_i))
    stable_cooldown = op.Where(cooling, op.Sub(probe_cooldown, one_i), probe_cooldown)
    can_probe = op.And(
        valid_col,
        op.And(
            op.Equal(probe_cooldown, zero_i),
            op.GreaterOrEqual(updated_samples, op.Constant(value_float=2.0)),
        ),
    )
    probe_down = op.And(
        can_probe,
        op.And(
            op.Less(updated_acceptance, op.Constant(value_float=0.5)),
            op.Greater(current_col, op.Constant(value_int=min_k)),
        ),
    )
    probe_up = op.And(
        can_probe,
        op.And(
            op.GreaterOrEqual(updated_acceptance, op.Constant(value_float=0.75)),
            op.And(
                op.Less(current_col, op.Constant(value_int=max_k)),
                op.GreaterOrEqual(stable_count, two_i),
            ),
        ),
    )
    start_probe = op.Or(probe_down, probe_up)
    candidate_k = op.Where(
        probe_down,
        op.Sub(current_col, one_i),
        op.Add(current_col, one_i),
    )
    stable_k = op.Where(start_probe, candidate_k, current_col)
    stable_origin = op.Where(start_probe, current_col, probe_origin_k)
    stable_count = op.Where(start_probe, zero_i, stable_count)

    computed_k = op.Squeeze(op.Where(in_probe, probe_k, stable_k), [-1])
    next_probe_origin = op.Where(in_probe, probe_origin, stable_origin)
    next_probe_observations = op.Where(
        in_probe,
        probe_count,
        op.Where(start_probe, zero_i, probe_observations),
    )
    next_stable_observations = op.Where(in_probe, probe_stable, stable_count)
    next_probe_cooldown = op.Where(in_probe, probe_next_cooldown, stable_cooldown)
    next_controller = op.Cast(
        op.Concat(
            next_probe_origin,
            next_probe_observations,
            next_stable_observations,
            next_probe_cooldown,
            axis=1,
        ),
        to=ir.DataType.FLOAT,
    )
    computed_estimates = op.Concat(
        next_tokens,
        next_ms,
        next_acceptance,
        next_samples,
        next_controller,
        axis=1,
    )
    next_k = op.Where(valid, computed_k, current_k)
    next_estimates = op.Where(valid_col, computed_estimates, estimates)
    next_k.shape = ir.Shape(["batch"])
    next_estimates.shape = ir.Shape(["batch", estimate_slots])
    builder.add_output(next_k, "next_k")
    builder.add_output(next_estimates, "next_estimates")
    return _component(
        "onnx-genai.adaptive-proposal-budget@1",
        graph,
        {
            "role": "adaptive_proposal_budget",
            "current_k": "current_k",
            "accepted": "accepted",
            "evaluated": "evaluated",
            "committed_tokens": "committed_tokens",
            "filled_proposal_budget": "filled_proposal_budget",
            "draft_ms": "draft_ms",
            "target_ms": "target_ms",
            "estimates": "estimates",
            "next_k": "next_k",
            "next_estimates": "next_estimates",
            "effect": "adaptive",
        },
        "adaptive",
    )


def build_seeded_categorical_sampler() -> PolicyComponent:
    """Build request-parameterized categorical sampling with explicit RNG state.

    Threefry is counter based: identical tensor inputs produce the same token.
    Temperature, top-k, top-p, and min-p remain request inputs;
    changing ordinary generation options never regenerates this artifact.
    """
    graph, builder = _make_graph("seeded_categorical_sampler")
    op = builder.op
    logits = builder.input("logits", ir.DataType.FLOAT, ["batch", "vocabulary"])
    temperature = builder.input("temperature", ir.DataType.FLOAT, ["batch"])
    top_k = builder.input("top_k", ir.DataType.INT64, ["batch"])
    top_p = builder.input("top_p", ir.DataType.FLOAT, ["batch"])
    min_p = builder.input("min_p", ir.DataType.FLOAT, ["batch"])
    seed = builder.input("seed", ir.DataType.INT64, ["batch"])
    counter = builder.input("counter", ir.DataType.INT64, ["batch"])
    active = builder.input("active", ir.DataType.BOOL, ["batch"])
    done = builder.input("done", ir.DataType.BOOL, ["batch"])
    enabled = op.And(active, op.Not(done))

    # Threefry2x64: a counter-based Random123 generator with no hidden state.
    # Unsigned arithmetic gives the specified modulo-2^64 round behavior.
    k0 = op.Cast(seed, to=ir.DataType.UINT64)
    k1 = op.Constant(value_int=0)
    k1 = op.Cast(k1, to=ir.DataType.UINT64)
    parity = op.Cast(op.Constant(value_int=0x1BD11BDAA9FC1A22), to=ir.DataType.UINT64)
    k2 = op.BitwiseXor(op.BitwiseXor(k0, k1), parity)
    keys = [k0, k1, k2]
    x0 = op.Add(op.Cast(counter, to=ir.DataType.UINT64), k0)
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

    blocked = op.CastLike(op.Constant(value_float=-3.4028235e38), logits)
    safe_temperature = op.Unsqueeze(
        op.Max(temperature, op.Constant(value_float=1e-6)),
        [-1],
    )
    scaled_logits = op.Div(logits, safe_temperature)
    safe_min_p = op.Unsqueeze(
        op.Clip(
            min_p,
            op.Constant(value_float=1e-20),
            op.Constant(value_float=1.0),
        ),
        [-1],
    )
    min_p_threshold = op.Add(
        op.ReduceMax(scaled_logits, axes=[-1], keepdims=1),
        op.Log(safe_min_p),
    )
    min_p_mask = op.Or(
        op.Unsqueeze(op.LessOrEqual(min_p, op.Constant(value_float=0.0)), [-1]),
        op.GreaterOrEqual(scaled_logits, min_p_threshold),
    )
    scaled_logits = op.Where(min_p_mask, scaled_logits, blocked)

    vocabulary = op.Shape(logits, start=1, end=2)
    requested_k = op.Where(
        op.Greater(top_k, op.Constant(value_int=0)),
        top_k,
        vocabulary,
    )
    effective_k = op.Min(op.Max(requested_k, op.Constant(value_int=1)), vocabulary)
    _, top_indices = op.TopK(
        scaled_logits,
        vocabulary,
        axis=-1,
        largest=1,
        sorted=1,
        _outputs=2,
    )
    ranks = op.Range(
        op.Constant(value_int=0),
        op.Squeeze(vocabulary, [0]),
        op.Constant(value_int=1),
    )
    ranks = op.Expand(op.Unsqueeze(ranks, [0]), op.Shape(logits))
    keep_top_k = op.Less(ranks, op.Unsqueeze(effective_k, [-1]))
    top_k_mask = op.Greater(
        op.ScatterElements(
            op.ConstantOfShape(
                op.Shape(logits),
                value=ir.tensor([0], dtype=ir.DataType.INT64),
            ),
            top_indices,
            op.Cast(keep_top_k, to=ir.DataType.INT64),
            axis=1,
        ),
        op.Constant(value_int=0),
    )
    top_k_logits = op.Where(top_k_mask, scaled_logits, blocked)
    probabilities = op.Softmax(top_k_logits, axis=-1)

    _, sorted_indices = op.TopK(
        probabilities,
        vocabulary,
        axis=-1,
        largest=1,
        sorted=1,
        _outputs=2,
    )
    sorted_probabilities = op.GatherElements(probabilities, sorted_indices, axis=1)
    cumulative_probabilities = op.CumSum(
        sorted_probabilities,
        op.Constant(value_int=1),
    )
    safe_top_p = op.Unsqueeze(
        op.Clip(
            top_p,
            op.Constant(value_float=1e-6),
            op.Constant(value_float=1.0),
        ),
        [-1],
    )
    keep_sorted = op.Less(
        op.Sub(cumulative_probabilities, sorted_probabilities),
        safe_top_p,
    )
    top_p_mask = op.Greater(
        op.ScatterElements(
            op.ConstantOfShape(
                op.Shape(logits),
                value=ir.tensor([0], dtype=ir.DataType.INT64),
            ),
            sorted_indices,
            op.Cast(keep_sorted, to=ir.DataType.INT64),
            axis=1,
        ),
        op.Constant(value_int=0),
    )
    probabilities = op.Where(
        top_p_mask,
        probabilities,
        op.CastLike(op.Constant(value_float=0.0), probabilities),
    )
    probabilities = op.Div(
        probabilities,
        op.Max(
            op.ReduceSum(probabilities, axes=[-1], keepdims=1),
            op.Constant(value_float=1e-20),
        ),
    )
    axis = op.Constant(value_int=-1)
    cumulative = op.CumSum(probabilities, axis)
    uniform = op.Unsqueeze(uniform, op.Constant(value_ints=[-1]))
    candidates = op.GreaterOrEqual(cumulative, uniform)
    token_ids = op.ArgMax(
        op.Cast(candidates, to=ir.DataType.INT64),
        axis=-1,
        keepdims=0,
    )
    token_ids = op.Where(enabled, token_ids, op.Constant(value_int=-1))
    next_counter = op.Where(
        enabled,
        op.Add(counter, op.Constant(value_int=1)),
        counter,
    )
    _set_public_shape(logits, ["batch", "vocabulary"])
    for value in (temperature, top_k, top_p, min_p, seed, counter, active, done):
        _set_public_shape(value, ["batch"])
    _set_public_shape(token_ids, ["batch"])
    _set_public_shape(next_counter, ["batch"])
    builder.add_output(token_ids, "token")
    builder.add_output(next_counter, "next_counter")
    return _component(
        "onnx-genai.token-sampler@2",
        graph,
        {
            "role": "token_sampler",
            "mode": "seeded_stochastic",
            "batching": "per_row",
            "inactive_rows": "preserve",
            "logits": "logits",
            "token": "token",
            "temperature": "temperature",
            "top_k": "top_k",
            "top_p": "top_p",
            "min_p": "min_p",
            "active": "active",
            "done": "done",
            "seed": "seed",
            "counter": "counter",
            "next_counter": "next_counter",
            "effect": "rng",
        },
        "rng",
    )


def build_eos_termination(*, row_selective: bool = False) -> PolicyComponent:
    """Build an EOS predicate for batched current tokens and an EOS-id set."""
    graph, builder = _make_graph("eos_termination")
    op = builder.op
    token_ids = builder.input("tokens", ir.DataType.INT64, ["batch"])
    eos_ids = builder.input(
        "eos_ids",
        ir.DataType.INT64,
        ["batch", "num_eos"] if row_selective else ["num_eos"],
    )
    eos_lengths = (
        builder.input("eos_lengths", ir.DataType.INT64, ["batch"]) if row_selective else None
    )
    iteration = builder.input(
        "iteration",
        ir.DataType.INT64,
        [1] if row_selective else ["batch"],
    )
    max_iterations = builder.input(
        "max_iterations",
        ir.DataType.INT64,
        ["batch"],
    )
    tokens = op.Unsqueeze(token_ids, op.Constant(value_ints=[-1]))
    eos = eos_ids if row_selective else op.Unsqueeze(eos_ids, [0])
    matches = op.Equal(tokens, eos)
    if row_selective:
        assert eos_lengths is not None
        eos_positions = op.Range(
            op.Constant(value_int=0),
            op.Squeeze(op.Shape(eos_ids, start=1, end=2), [0]),
            op.Constant(value_int=1),
        )
        valid_eos = op.Less(
            op.Unsqueeze(eos_positions, [0]),
            op.Unsqueeze(eos_lengths, [-1]),
        )
        matches = op.And(matches, valid_eos)
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
    if row_selective:
        active = builder.input("active", ir.DataType.BOOL, ["batch"])
        newly_done = op.And(active, op.Or(hit_eos, hit_limit))
        next_active = op.And(active, op.Not(newly_done))
        done = op.Not(next_active)
    else:
        done = op.Or(hit_eos, hit_limit)
        next_active = op.Not(done)
    continued = op.Greater(
        op.ReduceMax(op.Cast(next_active, to=ir.DataType.INT64), keepdims=1),
        op.Constant(value_int=0),
    )
    if row_selective:
        assert eos_lengths is not None
        _set_public_shape(token_ids, ["batch"])
        _set_public_shape(eos_ids, ["batch", "num_eos"])
        _set_public_shape(eos_lengths, ["batch"])
        _set_public_shape(iteration, [1])
        _set_public_shape(max_iterations, ["batch"])
        _set_public_shape(active, ["batch"])
        _set_public_shape(done, ["batch"])
        _set_public_shape(next_active, ["batch"])
    _set_public_shape(continued, [1])
    builder.add_output(done, "done")
    if row_selective:
        builder.add_output(next_active, "next_active")
    builder.add_output(continued, "continue")
    return _component(
        (
            "onnx-genai.termination-predicate@2"
            if row_selective
            else "onnx-genai.termination-predicate@1"
        ),
        graph,
        {
            "role": "termination_predicate",
            "tokens": "tokens",
            "eos_ids": "eos_ids",
            "iteration": "iteration",
            "max_iterations": "max_iterations",
            **(
                {
                    "eos_lengths": "eos_lengths",
                    "active": "active",
                    "batching": "per_row",
                    "inactive_rows": "preserve",
                }
                if row_selective
                else {}
            ),
            "done": "done",
            **({"next_active": "next_active"} if row_selective else {}),
            "continue": "continue",
            "effect": "termination",
        },
        "termination",
    )


def build_scalar_constant(value: float) -> PolicyComponent:
    """Materialize one producer-selected scalar as a rank-1 tensor."""
    graph, builder = _make_graph("scalar_constant")
    constant = builder.op.Constant(value=ir.tensor([value], dtype=ir.DataType.FLOAT))
    constant.shape = ir.Shape([1])
    builder.add_output(constant, "value")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_shape_constant(dims: list[int]) -> PolicyComponent:
    """Materialize a producer-selected integer shape vector."""
    graph, builder = _make_graph("shape_constant")
    constant = builder.op.Constant(
        value=ir.tensor([int(dim) for dim in dims], dtype=ir.DataType.INT64)
    )
    constant.shape = ir.Shape([len(dims)])
    builder.add_output(constant, "shape")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_tensor_scale(dtype: ir.DataType = ir.DataType.FLOAT) -> PolicyComponent:
    """Scale a rank-4 state tensor by a broadcast scalar factor."""
    graph, builder = _make_graph("tensor_scale")
    op = builder.op
    tensor = builder.input("tensor", dtype, ["batch", "channels", "height", "width"])
    scale = builder.input("scale", ir.DataType.FLOAT, [1])
    scaled = op.Mul(tensor, op.Cast(scale, to=dtype))
    scaled.shape = tensor.shape
    builder.add_output(scaled, "scaled")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_tensor_clamp(
    dtype: ir.DataType = ir.DataType.FLOAT,
    *,
    minimum: float,
    maximum: float,
) -> PolicyComponent:
    """Clamp a rank-4 tensor to an explicitly declared numeric range."""
    graph, builder = _make_graph("tensor_clamp")
    op = builder.op
    tensor = builder.input("tensor", dtype, ["batch", "channels", "height", "width"])
    clamped = op.Clip(
        tensor,
        op.Cast(op.Constant(value_float=minimum), to=dtype),
        op.Cast(op.Constant(value_float=maximum), to=dtype),
    )
    clamped.shape = tensor.shape
    builder.add_output(clamped, "clamped")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_zeros_like(dtype: ir.DataType = ir.DataType.FLOAT) -> PolicyComponent:
    """Produce a zero tensor shaped like its reference, for state initializers."""
    graph, builder = _make_graph("zeros_like")
    op = builder.op
    reference = builder.input("reference", dtype, ["batch", "channels", "height", "width"])
    zeros = op.Mul(reference, op.Cast(op.Constant(value_float=0.0), to=dtype))
    zeros.shape = reference.shape
    builder.add_output(zeros, "zeros")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_guidance_combine(dtype: ir.DataType = ir.DataType.FLOAT) -> PolicyComponent:
    """Combine two conditioned estimates by a per-row guidance scale.

    ``estimate = unconditional + scale * (conditional - unconditional)`` is the
    classifier-free-guidance extrapolation. The scale is an ordinary per-row
    tensor input, so a caller can vary it per request without a rebuild.
    """
    graph, builder = _make_graph("guidance_combine")
    op = builder.op
    unconditional = builder.input(
        "unconditional", dtype, ["batch", "channels", "height", "width"]
    )
    conditional = builder.input("conditional", dtype, ["batch", "channels", "height", "width"])
    scale = builder.input("scale", ir.DataType.FLOAT, ["batch"])
    # (batch,) -> (batch, 1, 1, 1) so the row scale broadcasts over the latent.
    factor = op.Unsqueeze(op.Cast(scale, to=dtype), op.Constant(value_ints=[1, 2, 3]))
    estimate = op.Add(unconditional, op.Mul(factor, op.Sub(conditional, unconditional)))
    estimate.shape = unconditional.shape
    builder.add_output(estimate, "estimate")
    return _component(
        "onnx-genai.guidance-combine@1",
        graph,
        {
            "role": "guidance_combine",
            "unconditional": "unconditional",
            "conditional": "conditional",
            "scale": "scale",
            "estimate": "estimate",
        },
    )


def build_tensor_cast(
    source_dtype: ir.DataType,
    target_dtype: ir.DataType,
    dims: Sequence[str] = ("batch", "heads", "sequence", "head_dim"),
) -> PolicyComponent:
    """Cast a typed tensor at an explicit component precision boundary."""
    graph, builder = _make_graph("tensor_cast")
    value = builder.input("value", source_dtype, list(dims))
    cast = builder.op.Cast(value, to=target_dtype)
    cast.type = ir.TensorType(target_dtype)
    _set_public_shape(cast, list(dims))
    builder.add_output(cast, "cast")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_x0_flow_velocity(
    dtype: ir.DataType = ir.DataType.FLOAT,
    *,
    t_eps: float = 0.02,
) -> PolicyComponent:
    """Convert a clean-sample prediction into flow velocity.

    Unified pixel-space generators commonly predict ``x0`` rather than
    velocity. Their flow-matching derivative is
    ``(x0 - sample) / max(1 - timestep, t_eps)``.
    """
    graph, builder = _make_graph("x0_flow_velocity")
    op = builder.op
    sample = builder.input("sample", dtype, ["batch", "channels", "height", "width"])
    x0 = builder.input("x0", dtype, ["batch", "channels", "height", "width"])
    timestep = builder.input("timestep", dtype, ["batch"])
    one = op.CastLike(op.Constant(value_float=1.0), timestep)
    epsilon = op.CastLike(op.Constant(value_float=float(t_eps)), timestep)
    denominator = op.Max(op.Sub(one, timestep), epsilon)
    denominator = op.Unsqueeze(denominator, op.Constant(value_ints=[1, 2, 3]))
    velocity = op.Div(op.Sub(x0, sample), denominator)
    velocity.type = ir.TensorType(dtype)
    _set_public_shape(velocity, ["batch", "channels", "height", "width"])
    builder.add_output(velocity, "velocity")
    return _component("onnx-genai.flow-velocity@1", graph, {})


def build_image_grid_positions(
    dtype: ir.DataType = ir.DataType.FLOAT,
    *,
    pixels_per_token: int,
) -> PolicyComponent:
    """Derive a three-axis image position grid from a pixel-space latent."""
    graph, builder = _make_graph("image_grid_positions")
    op = builder.op
    latent = builder.input("latent", dtype, ["batch", "channels", "height", "width"])
    prompt_tokens = builder.input(
        "prompt_tokens", ir.DataType.INT64, ["batch", "prompt_sequence"]
    )
    batch = op.Shape(latent, start=0, end=1)
    height = op.Div(
        op.Shape(latent, start=2, end=3),
        op.Constant(value_ints=[pixels_per_token]),
    )
    width = op.Div(
        op.Shape(latent, start=3, end=4),
        op.Constant(value_ints=[pixels_per_token]),
    )
    count = op.Mul(height, width)
    flat = op.Range(
        op.Constant(value_int=0),
        op.Squeeze(count, op.Constant(value_ints=[0])),
        op.Constant(value_int=1),
    )
    token_shape = op.Concat(batch, count, axis=0)
    height_positions = op.Expand(
        op.Unsqueeze(
            op.Div(flat, op.Squeeze(width, op.Constant(value_ints=[0]))),
            op.Constant(value_ints=[0]),
        ),
        token_shape,
    )
    width_positions = op.Expand(
        op.Unsqueeze(
            op.Mod(flat, op.Squeeze(width, op.Constant(value_ints=[0])), fmod=0),
            op.Constant(value_ints=[0]),
        ),
        token_shape,
    )
    prompt_length = op.Shape(prompt_tokens, start=1, end=2)
    temporal = op.Expand(prompt_length, token_shape)
    positions = op.Concat(
        op.Unsqueeze(temporal, op.Constant(value_ints=[0])),
        op.Unsqueeze(height_positions, op.Constant(value_ints=[0])),
        op.Unsqueeze(width_positions, op.Constant(value_ints=[0])),
        axis=0,
    )
    positions.shape = ir.Shape([3, "batch", "image_tokens"])
    token_grid = op.Concat(height, width, axis=0)
    token_grid.shape = ir.Shape([2])
    builder.add_output(positions, "position_ids")
    builder.add_output(token_grid, "token_grid")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_image_dimensions(
    dtype: ir.DataType = ir.DataType.FLOAT,
) -> PolicyComponent:
    """Read pixel height and width from a rank-4 image tensor."""
    graph, builder = _make_graph("image_dimensions")
    op = builder.op
    tensor = builder.input("tensor", dtype, ["batch", "channels", "height", "width"])
    height = op.Shape(tensor, start=2, end=3)
    width = op.Shape(tensor, start=3, end=4)
    height.shape = width.shape = ir.Shape([1])
    builder.add_output(height, "height")
    builder.add_output(width, "width")
    return _component("mobius.policy.auxiliary@1", graph, {})


_RNG_MODULUS = 2147483647
_RNG_MULTIPLIER = 48271
_RNG_STRIDE = 2654435761


def build_image_noise_geometry(
    *,
    channels: int = 3,
    token_stride: int = 32,
    base_image_sequence_length: int = 64,
    noise_scale: float = 1.0,
    maximum_noise_scale: float = 16.0,
) -> PolicyComponent:
    """Derive image-noise shape, token grid, and resolution-dependent scale."""
    graph, builder = _make_graph("image_noise_geometry")
    op = builder.op
    height = builder.input("height", ir.DataType.INT64, [1])
    width = builder.input("width", ir.DataType.INT64, [1])

    row_shape = op.Concat(
        op.Constant(value_ints=[channels]),
        height,
        width,
        axis=0,
    )
    row_shape.shape = ir.Shape([3])
    token_height = op.Div(height, op.Constant(value_int=token_stride))
    token_width = op.Div(width, op.Constant(value_int=token_stride))
    token_height.shape = token_width.shape = ir.Shape([1])

    image_sequence_length = op.Mul(token_height, token_width)
    relative_length = op.Div(
        op.Cast(image_sequence_length, to=ir.DataType.FLOAT),
        op.Constant(value_float=float(base_image_sequence_length)),
    )
    resolved_scale = op.Min(
        op.Mul(op.Sqrt(relative_length), op.Constant(value_float=noise_scale)),
        op.Constant(value_float=maximum_noise_scale),
    )
    resolved_scale.shape = ir.Shape([1])

    builder.add_output(row_shape, "row_shape")
    builder.add_output(token_height, "token_height")
    builder.add_output(token_width, "token_width")
    builder.add_output(resolved_scale, "noise_scale")
    return _component("mobius.policy.auxiliary@1", graph, {})


def _counter_uniform(op, key, counter):
    """Map a counter-based ``(key, counter)`` pair onto a uniform in ``(0, 1)``.

    The stream is a pure function of its inputs, so a row's noise depends only
    on its own seed and offset and never on batch position or iteration order.
    """
    modulus = op.Constant(value_int=_RNG_MODULUS)
    multiplier = op.Constant(value_int=_RNG_MULTIPLIER)
    state = op.Mod(
        op.Add(
            op.Add(key, op.Constant(value_int=1)),
            op.Mul(counter, op.Constant(value_int=_RNG_STRIDE)),
        ),
        modulus,
        fmod=0,
    )
    for _ in range(3):
        state = op.Mod(op.Mul(state, multiplier), modulus, fmod=0)
        # Integer division stands in for a right shift: BitShift is unsigned-only.
        state = op.BitwiseXor(state, op.Div(state, op.Constant(value_int=2048)))
        state = op.Mod(op.Mul(state, multiplier), modulus, fmod=0)
    # Shift off zero so the logarithm in the Box-Muller transform stays finite.
    return op.Div(
        op.Cast(op.Add(state, op.Constant(value_int=1)), to=ir.DataType.FLOAT),
        op.Constant(value_float=float(_RNG_MODULUS + 2)),
    )


def build_counter_rng_normal(dtype: ir.DataType = ir.DataType.FLOAT) -> PolicyComponent:
    """Draw standard-normal noise from explicit counter RNG state.

    Inputs are a per-row ``seed``, a per-row ``offset`` counter, and the target
    ``shape``. The component is pure: it consumes counter state and returns the
    advanced counter, so RNG progress is loop-carried workflow state rather than
    hidden session state inside an operator.
    """
    graph, builder = _make_graph("counter_rng_normal")
    op = builder.op
    seed = builder.input("seed", ir.DataType.INT64, ["batch"])
    offset = builder.input("offset", ir.DataType.INT64, ["batch"])
    row_shape = builder.input("row_shape", ir.DataType.INT64, ["row_rank"])

    # The batch extent comes from the per-row seed, so the draw is always
    # request-aligned no matter how many rows the runtime batched together.
    shape = op.Concat(op.Shape(seed), row_shape, axis=0)
    row_elements = op.ReduceProd(row_shape, keepdims=1)
    # (row_elements,) counter positions, shared by every row's private stream.
    positions = op.Range(
        op.Constant(value_int=0),
        op.Squeeze(row_elements, op.Constant(value_ints=[0])),
        op.Constant(value_int=1),
    )
    positions = op.Unsqueeze(positions, op.Constant(value_ints=[0]))
    base = op.Mul(op.Unsqueeze(offset, op.Constant(value_ints=[1])), row_elements)
    counter = op.Add(base, positions)
    key = op.Unsqueeze(seed, op.Constant(value_ints=[1]))

    # Box-Muller over two independent counter blocks keeps the draw stateless.
    uniform_radius = _counter_uniform(op, key, counter)
    uniform_angle = _counter_uniform(
        op, key, op.Add(counter, op.Mul(row_elements, op.Constant(value_int=2)))
    )
    radius = op.Sqrt(op.Mul(op.Constant(value_float=-2.0), op.Log(uniform_radius)))
    angle = op.Mul(op.Constant(value_float=6.283185307179586), uniform_angle)
    flat = op.Mul(radius, op.Cos(angle))
    noise = op.Cast(op.Reshape(flat, shape), to=dtype)
    noise.shape = ir.Shape(["batch", "channels", "height", "width"])
    next_offset = op.Add(offset, op.Constant(value_int=1))
    next_offset.shape = ir.Shape(["batch"])
    builder.add_output(noise, "noise")
    builder.add_output(next_offset, "next_offset")
    return _component(
        "onnx-genai.counter-rng@1",
        graph,
        {
            "role": "counter_rng",
            "seed": "seed",
            "offset": "offset",
            "row_shape": "row_shape",
            "noise": "noise",
            "next_offset": "next_offset",
        },
    )


_IMAGE_LATENT_DIMS: tuple[str, ...] = ("batch", "channels", "height", "width")


def build_euler_model_input(
    dtype: ir.DataType = ir.DataType.FLOAT,
    latent_dims: Sequence[str] = _IMAGE_LATENT_DIMS,
) -> PolicyComponent:
    """Scale a latent for the Euler denoiser input at the current sigma.

    ``latent_dims`` names the latent axes. The per-row sigma is broadcast over
    every axis after the batch, so a video latent that carries a temporal axis
    works without a separate component.
    """
    graph, builder = _make_graph("euler_model_input")
    op = builder.op
    sample = builder.input("sample", dtype, list(latent_dims))
    step = builder.input("step", ir.DataType.INT64, ["batch"])
    schedule = builder.input("schedule", ir.DataType.FLOAT, ["schedule_length"])
    sigma = op.Gather(schedule, step, axis=0)
    scale = op.Sqrt(op.Add(op.Mul(sigma, sigma), op.Constant(value_float=1.0)))
    scale = op.Cast(scale, to=dtype)
    scale = op.Unsqueeze(scale, op.Constant(value_ints=list(range(1, len(latent_dims)))))
    model_input = op.Div(sample, scale)
    model_input.shape = sample.shape
    builder.add_output(model_input, "model_input")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_euler_solver_step(
    dtype: ir.DataType = ir.DataType.FLOAT,
    latent_dims: Sequence[str] = _IMAGE_LATENT_DIMS,
) -> PolicyComponent:
    """Build the generic Euler update ``x_next = x + dx * (sigma_next-sigma)``.

    ``latent_dims`` names the latent axes so the same update serves image and
    video latents; the sigma delta broadcasts over every axis after the batch.
    """
    graph, builder = _make_graph("euler_solver_step")
    op = builder.op
    sample = builder.input("sample", dtype, list(latent_dims))
    derivative = builder.input("derivative", dtype, list(latent_dims))
    step = builder.input("step", ir.DataType.INT64, ["batch"])
    schedule = builder.input("schedule", ir.DataType.FLOAT, ["schedule_length"])
    final_index = op.Sub(op.Shape(schedule, start=0, end=1), op.Constant(value_ints=[1]))
    next_step = op.Min(op.Add(step, op.Constant(value_int=1)), final_index)
    sigma = op.Gather(schedule, step, axis=0)
    sigma_next = op.Gather(schedule, next_step, axis=0)
    delta = op.Sub(sigma_next, sigma)
    delta = op.Cast(delta, to=dtype)
    delta = op.Unsqueeze(delta, op.Constant(value_ints=list(range(1, len(latent_dims)))))
    next_sample = op.Add(sample, op.Mul(derivative, delta))
    builder.add_output(next_sample, "next_state")
    return _component(
        "onnx-genai.solver-step@1",
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


def build_multistep_solver_step(
    dtype: ir.DataType = ir.DataType.FLOAT,
) -> PolicyComponent:
    """Build a second-order multistep solver update with explicit history state.

    This is the DPM-Solver++(2M) midpoint update expressed entirely in ONNX. The
    solver's memory - the previous data-space estimate - is an ordinary tensor
    port, so the workflow carries it as declared state instead of the runtime
    holding hidden scheduler attributes.

    ``next = (sig_next/sig_now) * sample - alpha_next * (sig_next/sig_now_ratio - 1)
    * (D0 + 0.5 * D1)`` where ``D0`` is the current data estimate and ``D1`` the
    finite difference against the previous one. The first and last steps have no
    usable history, so ``D1`` is masked to zero there, which reduces the update to
    the first-order form exactly as a multistep scheme's warm-up and final step do.
    """
    graph, builder = _make_graph("multistep_solver_step")
    op = builder.op
    sample = builder.input("sample", dtype, ["batch", "channels", "height", "width"])
    estimate = builder.input("estimate", dtype, ["batch", "channels", "height", "width"])
    history = builder.input("history", dtype, ["batch", "channels", "height", "width"])
    step = builder.input("step", ir.DataType.INT64, ["batch"])
    schedule = builder.input("schedule", ir.DataType.FLOAT, ["schedule_length"])

    one = op.Constant(value_int=1)
    zero = op.Constant(value_int=0)
    final_index = op.Sub(op.Shape(schedule, start=0, end=1), op.Constant(value_ints=[1]))
    next_step = op.Min(op.Add(step, one), final_index)
    previous_step = op.Max(op.Sub(step, one), zero)
    sigma_now = op.Gather(schedule, step, axis=0)
    sigma_next = op.Gather(schedule, next_step, axis=0)
    sigma_previous = op.Gather(schedule, previous_step, axis=0)

    def alpha(sigma):
        return op.Div(
            op.Constant(value_float=1.0),
            op.Sqrt(op.Add(op.Mul(sigma, sigma), op.Constant(value_float=1.0))),
        )

    alpha_now = alpha(sigma_now)
    alpha_next = alpha(sigma_next)
    # The variance-preserving noise level, sigma_t in the DPM-Solver derivation.
    noise_now = op.Mul(sigma_now, alpha_now)
    noise_next = op.Mul(sigma_next, alpha_next)

    def row_scalar(value):
        return op.Unsqueeze(op.Cast(value, to=dtype), op.Constant(value_ints=[1, 2, 3]))

    # Data-space estimate x0 from the epsilon-space model output.
    data_estimate = op.Div(
        op.Sub(sample, op.Mul(row_scalar(noise_now), estimate)),
        row_scalar(alpha_now),
    )
    ratio = op.Div(sigma_next, sigma_now)
    sample_coefficient = row_scalar(op.Div(noise_next, noise_now))
    data_coefficient = row_scalar(
        op.Mul(alpha_next, op.Sub(ratio, op.Constant(value_float=1.0)))
    )

    # Half-log-SNR spacing ratio; lambda(sigma) = -log(sigma) for this schedule.
    interval = op.Sub(op.Log(sigma_now), op.Log(sigma_next))
    previous_interval = op.Sub(op.Log(sigma_previous), op.Log(sigma_now))
    difference = op.Mul(
        row_scalar(op.Div(interval, previous_interval)),
        op.Sub(data_estimate, history),
    )
    warm = op.Greater(step, zero)
    penultimate = op.Less(op.Add(step, one), final_index)
    usable = op.Unsqueeze(op.And(warm, penultimate), op.Constant(value_ints=[1, 2, 3]))
    difference = op.Where(usable, difference, op.Cast(op.Constant(value_float=0.0), to=dtype))

    next_state = op.Sub(
        op.Mul(sample_coefficient, sample),
        op.Mul(
            data_coefficient,
            op.Add(
                data_estimate,
                op.Mul(op.Cast(op.Constant(value_float=0.5), to=dtype), difference),
            ),
        ),
    )
    next_state.shape = sample.shape
    data_estimate.shape = sample.shape
    builder.add_output(next_state, "next_state")
    builder.add_output(data_estimate, "next_history")
    return _component(
        "onnx-genai.solver-step@1",
        graph,
        {
            "role": "solver_step",
            "state": "sample",
            "estimate": "estimate",
            "history": "history",
            "step": "step",
            "schedule": "schedule",
            "next_state": "next_state",
            "next_history": "next_history",
        },
    )


def build_flow_match_solver_step(
    dtype: ir.DataType = ir.DataType.FLOAT,
) -> PolicyComponent:
    """Build the Euler update for rank-3 (packed/patchified) latents.

    Identical arithmetic to :func:`build_euler_solver_step` — ``x_next = x + dx *
    (sigma_next - sigma)`` — but broadcasts the per-batch step size over a
    ``(batch, sequence, channels)`` latent instead of a rank-4 image latent.
    Flow-matching transformers (Qwen Image, Flux, SD3) carry latents in this
    packed layout.
    """
    graph, builder = _make_graph("flow_match_solver_step")
    op = builder.op
    sample = builder.input("sample", dtype, ["batch", "sequence", "channels"])
    derivative = builder.input("derivative", dtype, ["batch", "sequence", "channels"])
    step = builder.input("step", ir.DataType.INT64, ["batch"])
    schedule = builder.input("schedule", ir.DataType.FLOAT, ["schedule_length"])
    final_index = op.Sub(op.Shape(schedule, start=0, end=1), op.Constant(value_ints=[1]))
    next_step = op.Min(op.Add(step, op.Constant(value_int=1)), final_index)
    sigma = op.Gather(schedule, step, axis=0)
    sigma_next = op.Gather(schedule, next_step, axis=0)
    delta = op.Cast(op.Sub(sigma_next, sigma), to=dtype)
    delta = op.Unsqueeze(delta, op.Constant(value_ints=[1, 2]))
    next_sample = op.Add(sample, op.Mul(derivative, delta))
    builder.add_output(next_sample, "next_state")
    return _component(
        "onnx-genai.solver-step@1",
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


def build_ddim_solver_step(
    dtype: ir.DataType = ir.DataType.FLOAT,
    latent_dims: Sequence[str] = _IMAGE_LATENT_DIMS,
    *,
    clip_sample_range: float | None = None,
) -> PolicyComponent:
    """Build the deterministic DDIM update (``eta = 0``, epsilon prediction).

    ``schedule`` holds the cumulative alpha of every denoising step followed by
    the alpha of the final step's predecessor, so entry ``i + 1`` is the
    ``alpha_prev`` of step ``i``::

        pred_x0 = (x - sqrt(1 - a_t) * eps) / sqrt(a_t)
        x_prev  = sqrt(a_prev) * pred_x0 + sqrt(1 - a_prev) * eps

    ``clip_sample_range`` reproduces schedulers configured with
    ``clip_sample=True``, which clamp the predicted clean sample before the
    reverse step. ``latent_dims`` names the latent axes, so the same update
    serves image and video latents.
    """
    graph, builder = _make_graph("ddim_solver_step")
    op = builder.op
    sample = builder.input("sample", dtype, list(latent_dims))
    derivative = builder.input("derivative", dtype, list(latent_dims))
    step = builder.input("step", ir.DataType.INT64, ["batch"])
    schedule = builder.input("schedule", ir.DataType.FLOAT, ["schedule_length"])
    final_index = op.Sub(op.Shape(schedule, start=0, end=1), op.Constant(value_ints=[1]))
    next_step = op.Min(op.Add(step, op.Constant(value_int=1)), final_index)
    broadcast_axes = op.Constant(value_ints=list(range(1, len(latent_dims))))
    alpha = op.Unsqueeze(op.Cast(op.Gather(schedule, step, axis=0), to=dtype), broadcast_axes)
    alpha_prev = op.Unsqueeze(
        op.Cast(op.Gather(schedule, next_step, axis=0), to=dtype), broadcast_axes
    )
    one = op.CastLike(op.Constant(value_float=1.0), sample)
    # Recover the predicted clean latent, then re-noise it to alpha_prev.
    pred_original = op.Div(
        op.Sub(sample, op.Mul(op.Sqrt(op.Sub(one, alpha)), derivative)), op.Sqrt(alpha)
    )
    if clip_sample_range is not None:
        limit = op.CastLike(op.Constant(value_float=float(clip_sample_range)), sample)
        pred_original = op.Clip(pred_original, op.Neg(limit), limit)
    next_sample = op.Add(
        op.Mul(op.Sqrt(alpha_prev), pred_original),
        op.Mul(op.Sqrt(op.Sub(one, alpha_prev)), derivative),
    )
    _set_public_shape(next_sample, list(latent_dims))
    builder.add_output(next_sample, "next_state")
    return _component(
        "onnx-genai.solver-step@1",
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


def build_pack_latents_2x2(dtype: ir.DataType = ir.DataType.FLOAT) -> PolicyComponent:
    """Patchify a 3D VAE latent into the transformer's packed token layout.

    ``(B, C, T, H, W) -> (B, T*(H/2)*(W/2), C*4)`` by folding each 2x2 spatial
    patch into the channel axis, matching ``QwenImagePipeline._pack_latents``.
    Shapes are derived from the input at runtime so the component stays valid
    for any resolution.
    """
    graph, builder = _make_graph("pack_latents")
    op = builder.op
    latent = builder.input(
        "latent_sample", dtype, ["batch", "channels", "frames", "height", "width"]
    )
    two = op.Constant(value_ints=[2])
    batch = op.Shape(latent, start=0, end=1)
    channels = op.Shape(latent, start=1, end=2)
    frames = op.Shape(latent, start=2, end=3)
    height = op.Shape(latent, start=3, end=4)
    width = op.Shape(latent, start=4, end=5)
    half_h = op.Div(height, two)
    half_w = op.Div(width, two)
    # (B, C, T, H, W) -> (B, C, T, H/2, 2, W/2, 2)
    patched = op.Reshape(
        latent,
        op.Concat(batch, channels, frames, half_h, two, half_w, two, axis=0),
    )
    # -> (B, T, H/2, W/2, C, 2, 2) so each 2x2 patch is contiguous per channel
    patched = op.Transpose(patched, perm=[0, 2, 3, 5, 1, 4, 6])
    tokens = op.Mul(op.Mul(frames, half_h), half_w)
    packed = op.Reshape(
        patched,
        op.Concat(batch, tokens, op.Mul(channels, op.Constant(value_ints=[4])), axis=0),
    )
    _set_public_shape(packed, ["batch", "sequence", "packed_channels"])
    builder.add_output(packed, "packed_latent")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_unpack_latents_2x2(dtype: ir.DataType = ir.DataType.FLOAT) -> PolicyComponent:
    """Invert :func:`build_pack_latents_2x2` back to a 3D VAE latent.

    ``(B, S, C*4) -> (B, C, 1, H*2, W*2)`` given the packed latent grid
    ``height``/``width`` (in packed tokens). The single frame matches the image
    pipelines, whose VAE always encodes one temporal chunk.
    """
    graph, builder = _make_graph("unpack_latents")
    op = builder.op
    packed = builder.input("packed_latent", dtype, ["batch", "sequence", "packed_channels"])
    height = builder.input("height", ir.DataType.INT64, [1])
    width = builder.input("width", ir.DataType.INT64, [1])
    two = op.Constant(value_ints=[2])
    batch = op.Shape(packed, start=0, end=1)
    channels = op.Div(op.Shape(packed, start=2, end=3), op.Constant(value_ints=[4]))
    # (B, S, C*4) -> (B, H, W, C, 2, 2)
    grid = op.Reshape(packed, op.Concat(batch, height, width, channels, two, two, axis=0))
    # -> (B, C, H, 2, W, 2) -> (B, C, 1, H*2, W*2)
    grid = op.Transpose(grid, perm=[0, 3, 1, 4, 2, 5])
    latent = op.Reshape(
        grid,
        op.Concat(
            batch,
            channels,
            op.Constant(value_ints=[1]),
            op.Mul(height, two),
            op.Mul(width, two),
            axis=0,
        ),
    )
    _set_public_shape(latent, ["batch", "channels", "frames", "height", "width"])
    builder.add_output(latent, "latent_sample")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_sequence_concat(dtype: ir.DataType = ir.DataType.FLOAT) -> PolicyComponent:
    """Concatenate two token sequences along the sequence axis.

    Image-editing denoisers attend jointly over the generated tokens and the
    source-image tokens, so the model input is ``concat([target, source], 1)``
    and the estimate is sliced back to the target length inside the denoiser.
    """
    graph, builder = _make_graph("sequence_concat")
    op = builder.op
    target = builder.input("target", dtype, ["batch", "target_sequence", "channels"])
    source = builder.input("source", dtype, ["batch", "source_sequence", "channels"])
    joined = op.Concat(target, source, axis=1)
    _set_public_shape(joined, ["batch", "sequence", "channels"])
    builder.add_output(joined, "sequence")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_true_cfg(
    dtype: ir.DataType = ir.DataType.FLOAT,
    *,
    guidance_scale: float = 4.0,
) -> PolicyComponent:
    """Combine conditional/unconditional estimates with Qwen's true CFG.

    ``combined = uncond + scale * (cond - uncond)``, then rescaled to preserve
    the conditional estimate's per-token norm::

        noise_pred = combined * (||cond||_2 / ||combined||_2)

    The reduction runs in float32: squared activations overflow float16, and
    onnxruntime ships no bfloat16 ``ReduceL2`` kernel at all (a bfloat16 graph
    containing one cannot be assigned to any provider and fails to load).
    """
    graph, builder = _make_graph("true_cfg")
    op = builder.op
    cond = builder.input("conditional", dtype, ["batch", "sequence", "channels"])
    uncond = builder.input("unconditional", dtype, ["batch", "sequence", "channels"])
    cond_f32 = op.Cast(cond, to=ir.DataType.FLOAT)
    uncond_f32 = op.Cast(uncond, to=ir.DataType.FLOAT)
    scale = op.Constant(value_float=float(guidance_scale))
    combined = op.Add(uncond_f32, op.Mul(op.Sub(cond_f32, uncond_f32), scale))
    cond_norm = op.ReduceL2(cond_f32, [-1], keepdims=True)
    combined_norm = op.Max(op.ReduceL2(combined, [-1], keepdims=True), 1e-12)
    guided = op.Cast(op.Mul(combined, op.Div(cond_norm, combined_norm)), to=dtype)
    _set_public_shape(guided, ["batch", "sequence", "channels"])
    builder.add_output(guided, "estimate")
    return _component("mobius.policy.auxiliary@1", graph, {})


SOLVER_BUILDERS = {
    "euler": build_euler_solver_step,
    "multistep": build_multistep_solver_step,
}


def build_identity_model_input(
    dtype: ir.DataType = ir.DataType.FLOAT,
    latent_dims: Sequence[str] = _IMAGE_LATENT_DIMS,
) -> PolicyComponent:
    """Pass the latent to the denoiser unchanged.

    DDIM-style schedulers define ``scale_model_input`` as the identity. Keeping
    the node explicit means the workflow reads the same way for every solver
    and the input scaling stays a declared, swappable policy.
    """
    graph, builder = _make_graph("identity_model_input")
    op = builder.op
    sample = builder.input("sample", dtype, list(latent_dims))
    step = builder.input("step", ir.DataType.INT64, ["batch"])
    schedule = builder.input("schedule", ir.DataType.FLOAT, ["schedule_length"])
    model_input = op.Identity(sample)
    # The step and schedule are unused by this policy but stay on the signature
    # so a package can swap solvers without rewiring the workflow.
    _ = op.Gather(schedule, step, axis=0)
    _set_public_shape(model_input, list(latent_dims))
    builder.add_output(model_input, "model_input")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_masked_token_update() -> PolicyComponent:
    """Build replacement of masked positions with explicit RNG-counter threading."""
    graph, builder = _make_graph("masked_token_update")
    op = builder.op
    current = builder.input("current_tokens", ir.DataType.INT64, ["batch", "sequence"])
    proposed = builder.input("proposed_tokens", ir.DataType.INT64, ["batch", "sequence"])
    logits = builder.input(
        "logits",
        ir.DataType.FLOAT,
        ["batch", "sequence", "vocabulary"],
    )
    masked = builder.input("masked", ir.DataType.BOOL, ["batch", "sequence"])
    step = builder.input("step", ir.DataType.INT64, ["batch"])
    total_steps = builder.input("total_steps", ir.DataType.INT64, ["batch"])
    seed = builder.input("seed", ir.DataType.INT64, ["batch"])
    offset = builder.input("offset", ir.DataType.INT64, ["batch"])
    sequence_length = op.Shape(masked, start=1, end=2)
    positions = op.Range(
        op.Constant(value_int=0),
        op.Squeeze(sequence_length, op.Constant(value_ints=[0])),
        op.Constant(value_int=1),
    )
    positions = op.Unsqueeze(positions, op.Constant(value_ints=[0]))
    stream = op.Add(
        positions,
        op.Unsqueeze(
            op.Add(op.Add(seed, offset), op.Mul(step, op.Constant(value_int=17))),
            op.Constant(value_ints=[-1]),
        ),
    )
    tie_noise = op.Div(
        op.Cast(op.Mod(stream, op.Constant(value_int=997), fmod=0), to=ir.DataType.FLOAT),
        op.Constant(value_float=997_000_000.0),
    )
    probabilities = op.Softmax(logits, axis=-1)
    confidence = op.Squeeze(
        op.GatherElements(
            probabilities,
            op.Unsqueeze(proposed, op.Constant(value_ints=[-1])),
            axis=-1,
        ),
        op.Constant(value_ints=[-1]),
    )
    confidence = op.Add(confidence, tie_noise)
    negative = op.CastLike(op.Constant(value_float=-1.0), confidence)
    ranked_scores = op.Where(masked, confidence, negative)
    left = op.Unsqueeze(ranked_scores, op.Constant(value_ints=[-1]))
    right = op.Unsqueeze(ranked_scores, op.Constant(value_ints=[-2]))
    rank = op.ReduceSum(
        op.Cast(op.Greater(right, left), to=ir.DataType.INT64),
        axes=[-1],
        keepdims=0,
    )
    remaining_before = op.ReduceSum(
        op.Cast(masked, to=ir.DataType.INT64),
        axes=[-1],
        keepdims=0,
    )
    steps_left = op.Max(
        op.Constant(value_int=1),
        op.Sub(total_steps, step),
    )
    quota = op.Div(
        op.Add(op.Sub(remaining_before, op.Constant(value_int=1)), steps_left),
        steps_left,
    )
    committed = op.And(
        masked,
        op.Less(rank, op.Unsqueeze(quota, op.Constant(value_ints=[-1]))),
    )
    updated = op.Where(committed, proposed, current)
    updated.shape = ir.Shape(["batch", "sequence"])
    remaining = op.And(masked, op.Not(committed))
    remaining.shape = ir.Shape(["batch", "sequence"])
    remaining_count = op.ReduceSum(
        op.Cast(remaining, to=ir.DataType.INT64),
        axes=[-1],
        keepdims=0,
    )
    done = op.Equal(remaining_count, op.Constant(value_int=0))
    done.shape = ir.Shape(["batch"])
    continued = op.Equal(
        op.ReduceMax(op.Cast(done, to=ir.DataType.INT64), keepdims=1),
        op.Constant(value_int=0),
    )
    continued.shape = ir.Shape([1])
    next_offset = op.Add(
        op.Add(offset, op.Squeeze(sequence_length, op.Constant(value_ints=[0]))),
        op.Mul(seed, op.Constant(value_int=0)),
    )
    next_offset.shape = ir.Shape(["batch"])
    builder.add_output(updated, "next_state")
    builder.add_output(remaining, "next_mask")
    builder.add_output(next_offset, "next_offset")
    builder.add_output(done, "done")
    builder.add_output(continued, "continue")
    return _component(
        "onnx-genai.masked-update@1",
        graph,
        {
            "role": "masked_update",
            "state": "current_tokens",
            "proposal": "proposed_tokens",
            "mask": "masked",
            "step": "step",
            "next_state": "next_state",
            "next_mask": "next_mask",
            "continue": "continue",
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
    seed = builder.input("seed", ir.DataType.INT64, ["batch"])
    offset = builder.input("offset", ir.DataType.INT64, ["batch"])
    target_tokens = op.ArgMax(target_scores, axis=-1, keepdims=0)
    accepted = op.Equal(target_tokens, proposed_tokens)
    rejected = op.Cast(op.Not(accepted), to=ir.DataType.INT64)
    rejection_count = op.CumSum(rejected, op.Constant(value_int=-1))
    prefix = op.Cast(
        op.Equal(rejection_count, op.Constant(value_int=0)),
        to=ir.DataType.INT64,
    )
    verified_count = op.ReduceSum(prefix, axes=[-1], keepdims=0)
    first_rejection = op.And(
        op.Cast(rejected, to=ir.DataType.BOOL),
        op.Equal(rejection_count, op.Constant(value_int=1)),
    )
    zeros = op.ConstantOfShape(
        op.Shape(proposed_tokens),
        value=ir.tensor([0], dtype=ir.DataType.INT64),
    )
    # Publish the verified prefix plus the verifier's correction at the first
    # mismatch. Trailing slots remain zero and are bounded by accepted_len.
    accepted_tokens = op.Where(
        op.Cast(prefix, to=ir.DataType.BOOL),
        proposed_tokens,
        op.Where(first_rejection, target_tokens, zeros),
    )
    accepted_tokens.shape = ir.Shape(["batch", "draft_sequence"])
    draft_length = op.Shape(proposed_tokens, start=1, end=2)
    done = op.Equal(verified_count, draft_length)
    accepted_count = op.Min(
        op.Add(verified_count, op.Cast(op.Not(done), to=ir.DataType.INT64)),
        draft_length,
    )
    continued = op.Not(done)
    positions = op.Range(
        op.Constant(value_int=0),
        op.Squeeze(draft_length, op.Constant(value_ints=[0])),
        op.Constant(value_int=1),
    )
    valid = op.Less(
        op.Unsqueeze(positions, op.Constant(value_ints=[0])),
        op.Unsqueeze(accepted_count, op.Constant(value_ints=[-1])),
    )
    accepted_tokens = op.Where(valid, accepted_tokens, zeros)
    accepted_tokens.shape = ir.Shape(["batch", "draft_sequence"])
    next_offset = op.Add(
        offset,
        op.Squeeze(draft_length, op.Constant(value_ints=[0])),
    )
    next_offset = op.Add(next_offset, op.Mul(seed, op.Constant(value_int=0)))
    accepted_count.shape = ir.Shape(["batch"])
    done.shape = ir.Shape(["batch"])
    verified_count.shape = ir.Shape(["batch"])
    continued.shape = ir.Shape(["batch"])
    next_offset.shape = ir.Shape(["batch"])
    builder.add_output(accepted_tokens, "accepted_tokens")
    builder.add_output(accepted_count, "accepted_len")
    builder.add_output(done, "done")
    builder.add_output(next_offset, "next_offset")
    builder.add_output(verified_count, "rollback_len")
    builder.add_output(continued, "continue")
    return _component(
        "onnx-genai.speculative-verifier@1",
        graph,
        {
            "role": "speculative_verifier",
            "target_scores": "target_scores",
            "proposed_tokens": "proposed_tokens",
            "accepted_tokens": "accepted_tokens",
            "accepted_len": "accepted_len",
            "done": "done",
            "continue": "continue",
            "rng": {
                "seed": "seed",
                "offset": "offset",
                "next_offset": "next_offset",
            },
            "effect": "verify",
        },
        "verify",
    )


def build_speculative_state_rollback(
    dtype: ir.DataType,
    shape: list[int | str],
    *,
    sequence_axis: int,
    effect: str = "rollback",
) -> PolicyComponent:
    """Trim tentative recurrent state to ``past_length + accepted_length``."""
    if not 0 <= sequence_axis < len(shape):
        raise ValueError("sequence_axis must index the state shape")
    graph, builder = _make_graph("speculative_state_rollback")
    op = builder.op
    past = builder.input("past_state", dtype, shape)
    tentative_shape = list(shape)
    tentative_shape[sequence_axis] = "tentative_sequence"
    tentative = builder.input("tentative_state", dtype, tentative_shape)
    accepted_len = builder.input("accepted_len", ir.DataType.INT64, [1])
    past_len = op.Shape(past, start=sequence_axis, end=sequence_axis + 1)
    end = op.Add(past_len, accepted_len)
    corrected = op.Slice(
        tentative,
        op.Constant(value_ints=[0]),
        end,
        op.Constant(value_ints=[sequence_axis]),
        op.Constant(value_ints=[1]),
    )
    corrected_shape = list(shape)
    corrected_shape[sequence_axis] = "accepted_sequence"
    corrected.shape = ir.Shape(corrected_shape)
    builder.add_output(corrected, "corrected_state")
    return _component("mobius.policy.auxiliary@1", graph, {}, effect)


def build_effectful_identity(
    name: str,
    dtype: ir.DataType,
    shape: list[int | str],
    *,
    effect: str,
) -> PolicyComponent:
    """Publish a typed branch-local state value with a linear effect."""
    graph, builder = _make_graph(name)
    value = builder.input("value", dtype, shape)
    builder.add_output(builder.op.Identity(value), "next_value")
    return _component("mobius.policy.auxiliary@1", graph, {}, effect)


def build_token_block_identity() -> PolicyComponent:
    """Publish a branch-local speculative token block with a linear effect."""
    graph, builder = _make_graph("token_block_identity")
    tokens = builder.input("tokens", ir.DataType.INT64, ["batch", "draft_sequence"])
    builder.add_output(builder.op.Identity(tokens), "next_tokens")
    return _component("mobius.policy.auxiliary@1", graph, {}, "state")


def build_token_state_update(*, row_selective: bool = False) -> PolicyComponent:
    """Selectively update one-token state while preserving suppressed rows."""
    graph, builder = _make_graph("token_state_update")
    op = builder.op
    current = builder.input("current", ir.DataType.INT64, ["batch", 1])
    update = builder.input(
        "update",
        ir.DataType.INT64,
        ["batch", 1] if row_selective else ["batch"],
    )
    if row_selective:
        active = builder.input("active", ir.DataType.BOOL, ["batch"])
        done = builder.input("done", ir.DataType.BOOL, ["batch"])
        enabled = op.And(active, op.Not(done))
        next_state = op.Where(
            op.Unsqueeze(enabled, [-1]),
            update,
            current,
        )
    else:
        next_state = op.Unsqueeze(update, [-1])
    if row_selective:
        _set_public_shape(current, ["batch", 1])
        _set_public_shape(update, ["batch", 1])
        _set_public_shape(active, ["batch"])
        _set_public_shape(done, ["batch"])
    _set_public_shape(next_state, ["batch", 1])
    builder.add_output(next_state, "next")
    return _component(
        "onnx-genai.state-update@2" if row_selective else "onnx-genai.state-update@1",
        graph,
        {
            "role": "state_update",
            "current": "current",
            "update": "update",
            **({"batching": "per_row", "inactive_rows": "preserve"} if row_selective else {}),
            **({"active": "active", "done": "done"} if row_selective else {}),
            "next": "next",
            "effect": "state",
        },
        "state",
    )


def build_token_to_slot() -> PolicyComponent:
    """Convert a canonical sampled token vector to a one-token ID tensor."""
    graph, builder = _make_graph("token_to_slot")
    token = builder.input("token", ir.DataType.INT64, ["batch"])
    slot = builder.op.Unsqueeze(token, [-1])
    slot.shape = ir.Shape(["batch", 1])
    builder.add_output(slot, "slot")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_video_latent_initializer(
    dtype: ir.DataType,
    init_noise_sigma: float,
    history_dtype: ir.DataType = ir.DataType.INT64,
) -> PolicyComponent:
    """Scale request noise into the scheduler's starting video latent.

    The noise carries the temporal axis, so nothing here collapses a clip to a
    single frame: the component is rank-agnostic and only applies the
    scheduler's ``init_noise_sigma``.
    """
    graph, builder = _make_graph("video_latent_initializer")
    op = builder.op
    noise = builder.input("noise", dtype, ["batch", "frames", "channels", "height", "width"])
    latent = op.Mul(noise, op.CastLike(op.Constant(value_float=init_noise_sigma), noise))
    _set_public_shape(latent, ["batch", "frames", "channels", "height", "width"])
    builder.add_output(latent, "latent")
    # An empty scheduler history: the denoise loop appends one timestep per step.
    history = op.ConstantOfShape(
        op.Concat(op.Shape(noise, start=0, end=1), op.Constant(value_ints=[0]), axis=0),
        value=ir.tensor([0], dtype=history_dtype),
    )
    _set_public_shape(history, ["batch", 0])
    builder.add_output(history, "history")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_schedule_history_append(dtype: ir.DataType) -> PolicyComponent:
    """Append the current timestep to the scheduler's history.

    Multistep video schedulers consume the trajectory of previous timesteps, so
    the history is real state rather than telemetry.
    """
    graph, builder = _make_graph("schedule_history_append")
    op = builder.op
    history = builder.input("history", dtype, ["batch", "history"])
    timestep = builder.input("timestep", dtype, ["batch"])
    updated = op.Concat(history, op.Unsqueeze(timestep, op.Constant(value_ints=[1])), axis=1)
    _set_public_shape(updated, ["batch", "history"])
    builder.add_output(updated, "next")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_video_decode_chunk_count(latent_frame_axis: int = 2) -> PolicyComponent:
    """Number of causal decode chunks a latent clip is split into.

    Mirrors ``AutoencoderKLCogVideoX._decode``: ``max(latent_frames // 2, 1)``.
    """
    graph, builder = _make_graph("video_decode_chunk_count")
    op = builder.op
    latent = builder.input(
        "latent",
        ir.DataType.FLOAT,
        ["batch", "channels", "latent_frames", "height", "width"],
    )
    frames = op.Shape(latent, start=latent_frame_axis, end=latent_frame_axis + 1)
    count = op.Max(
        op.Div(frames, op.Constant(value_ints=[2])),
        op.Constant(value_ints=[1]),
    )
    _set_public_shape(count, [1])
    builder.add_output(count, "count")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_video_decode_chunk(latent_frame_axis: int = 2) -> PolicyComponent:
    """Slice the latent frames belonging to one causal decode chunk.

    Reproduces the reference chunk walk, where the odd frame left over by the
    two-frame stride is folded into the first chunk::

        remaining = latent_frames % 2
        start = 2 * step + (0 if step == 0 else remaining)
        end   = 2 * (step + 1) + remaining
    """
    graph, builder = _make_graph("video_decode_chunk")
    op = builder.op
    latent = builder.input(
        "latent",
        ir.DataType.FLOAT,
        ["batch", "channels", "latent_frames", "height", "width"],
    )
    step = builder.input("step", ir.DataType.INT64, ["batch"])

    two = op.Constant(value_ints=[2])
    zero = op.Constant(value_ints=[0])
    frames = op.Shape(latent, start=latent_frame_axis, end=latent_frame_axis + 1)
    remaining = op.Mod(frames, two)
    # The loop induction value is batch-broadcast; every row walks the same clip.
    index = op.Slice(step, zero, op.Constant(value_ints=[1]), zero)
    offset = op.Where(op.Equal(index, zero), zero, remaining)
    start = op.Add(op.Mul(index, two), offset)
    end = op.Add(op.Mul(op.Add(index, op.Constant(value_ints=[1])), two), remaining)
    chunk = op.Slice(latent, start, end, op.Constant(value_ints=[latent_frame_axis]))
    _set_public_shape(chunk, ["batch", "channels", "chunk_frames", "height", "width"])
    builder.add_output(chunk, "chunk")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_video_conv_cache_initializer(
    entries: list[tuple[str, int, int]],
    dtype: ir.DataType = ir.DataType.FLOAT,
) -> PolicyComponent:
    """Zero-length causal convolution caches sized from the latent grid.

    ``entries`` are ``(port, channels, spatial_scale)``. A zero-length temporal
    axis is the encoding of "no previous chunk", which makes the first decode
    chunk replicate its own first frame exactly as the reference does.
    """
    graph, builder = _make_graph("video_conv_cache_initializer")
    op = builder.op
    latent = builder.input(
        "latent",
        dtype,
        ["batch", "channels", "latent_frames", "height", "width"],
    )
    batch = op.Shape(latent, start=0, end=1)
    height = op.Shape(latent, start=3, end=4)
    width = op.Shape(latent, start=4, end=5)
    zero = ir.tensor([0.0], dtype=dtype)
    for port, channels, scale in entries:
        scaled_height = (
            height if scale == 1 else op.Mul(height, op.Constant(value_ints=[scale]))
        )
        scaled_width = width if scale == 1 else op.Mul(width, op.Constant(value_ints=[scale]))
        shape = op.Concat(
            batch,
            op.Constant(value_ints=[channels, 0]),
            scaled_height,
            scaled_width,
            axis=0,
        )
        cache = op.ConstantOfShape(shape, value=zero)
        _set_public_shape(
            cache,
            [
                "batch",
                channels,
                0,
                "height" if scale == 1 else f"{scale}*height",
                "width" if scale == 1 else f"{scale}*width",
            ],
        )
        builder.add_output(cache, port)
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_video_latent_permute(perm: list[int]) -> PolicyComponent:
    """Reorder a video latent between the denoiser and VAE layouts.

    CogVideoX denoises ``[batch, frames, channels, height, width]`` but decodes
    ``[batch, channels, frames, height, width]``; the transposition is part of
    the pipeline contract, not an implementation detail of either model.
    """
    graph, builder = _make_graph("video_latent_permute")
    op = builder.op
    source = builder.input(
        "latent", ir.DataType.FLOAT, ["batch", "frames", "channels", "height", "width"]
    )
    permuted = op.Transpose(source, perm=perm)
    _set_public_shape(permuted, ["batch", "channels", "frames", "height", "width"])
    builder.add_output(permuted, "permuted")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_video_latent_unscale(scaling_factor: float) -> PolicyComponent:
    """Undo the autoencoder's latent scaling before decoding."""
    graph, builder = _make_graph("video_latent_unscale")
    op = builder.op
    latent = builder.input(
        "latent", ir.DataType.FLOAT, ["batch", "channels", "frames", "height", "width"]
    )
    unscaled = op.Div(latent, op.CastLike(op.Constant(value_float=scaling_factor), latent))
    _set_public_shape(unscaled, ["batch", "channels", "frames", "height", "width"])
    builder.add_output(unscaled, "unscaled")
    return _component("mobius.policy.auxiliary@1", graph, {})


_INT64_MAX = 2**63 - 1


def build_scalar_integer_add() -> PolicyComponent:
    """Add two rank-0 integer control values.

    Loop induction values and substep indices are scalars, not per-row state, so
    they need a rank-0 adder rather than the ``[batch]`` :func:`build_integer_add`.
    """
    graph, builder = _make_graph("scalar_integer_add")
    left = builder.input("left", ir.DataType.INT64, [])
    right = builder.input("right", ir.DataType.INT64, [])
    total = builder.op.Add(left, right)
    total.shape = ir.Shape([])
    builder.add_output(total, "total")
    return _component("mobius.policy.auxiliary@1", graph, {})


def _duplex_positions(
    op: Any,
    offset: Any,
    delays: Any,
    cache_length: int,
    channels: int,
    batch_shape: Any,
) -> Any:
    """Build ``[batch, channels, 1]`` ring indices for ``(offset + delays) % CT``."""
    positions = op.Mod(op.Add(delays, offset), op.Constant(value_int=cache_length))
    positions = op.Reshape(positions, op.Constant(value_ints=[1, channels, 1]))
    target_shape = op.Concat(batch_shape, op.Constant(value_ints=[channels, 1]), axis=0)
    return op.Expand(positions, target_shape)


def build_duplex_frame_assemble(
    *, channels: int = 17, cache_length: int = 4
) -> PolicyComponent:
    """Write per-stream tokens into the delay ring cache and read one model frame.

    Full-duplex codec language models interleave several token streams (one text
    stream plus interleaved agent and user acoustic streams) that are each shifted
    by a small per-stream delay. The delay compensation is a ring buffer of
    ``cache_length`` frames indexed by ``(offset + delay) % cache_length``.

    ``stream_tokens`` carries every externally supplied token for this step; a
    negative entry means "this stream has nothing to contribute, the model must
    predict it". ``initial_tokens`` primes streams whose delay has not elapsed.
    """
    graph, builder = _make_graph("duplex_frame_assemble")
    op = builder.op
    cache = builder.input("token_cache", ir.DataType.INT64, ["batch", channels, cache_length])
    provided = builder.input(
        "token_provided", ir.DataType.BOOL, ["batch", channels, cache_length]
    )
    offset = builder.input("offset", ir.DataType.INT64, [])
    stream_tokens = builder.input("stream_tokens", ir.DataType.INT64, ["batch", channels])
    delays = builder.input("delays", ir.DataType.INT64, [channels])
    initial_tokens = builder.input("initial_tokens", ir.DataType.INT64, [channels])

    batch_shape = op.Shape(cache, start=0, end=1)

    # 1. scatter externally supplied tokens at their delayed ring slot.
    write_index = _duplex_positions(op, offset, delays, cache_length, channels, batch_shape)
    token_update = op.Unsqueeze(stream_tokens, [-1])
    has_token = op.GreaterOrEqual(token_update, op.Constant(value_int=0))
    cache = op.ScatterElements(
        cache,
        write_index,
        op.Where(has_token, token_update, op.GatherElements(cache, write_index, axis=2)),
        axis=2,
    )
    # ORT has no bool Where kernel; setting a flag is a saturating Or.
    provided = op.ScatterElements(
        provided,
        write_index,
        op.Or(op.GatherElements(provided, write_index, axis=2), has_token),
        axis=2,
    )

    # 2. prime streams whose delay has not yet elapsed (offset <= delay).
    ring = op.Mod(offset, op.Constant(value_int=cache_length))
    target_index = op.Expand(
        op.Reshape(ring, op.Constant(value_ints=[1, 1, 1])),
        op.Concat(batch_shape, op.Constant(value_ints=[channels, 1]), axis=0),
    )
    primed = op.Reshape(
        op.LessOrEqual(op.Expand(offset, op.Constant(value_ints=[channels])), delays),
        op.Constant(value_ints=[1, channels, 1]),
    )
    primed = op.Expand(primed, op.Shape(target_index))
    initial_update = op.Expand(
        op.Reshape(initial_tokens, op.Constant(value_ints=[1, channels, 1])),
        op.Shape(target_index),
    )
    cache = op.ScatterElements(
        cache,
        target_index,
        op.Where(primed, initial_update, op.GatherElements(cache, target_index, axis=2)),
        axis=2,
    )
    provided = op.ScatterElements(
        provided,
        target_index,
        op.Or(op.GatherElements(provided, target_index, axis=2), primed),
        axis=2,
    )

    # 3. read the model input frame (offset - 1) and the teacher-forcing target.
    input_index = op.Expand(
        op.Reshape(
            op.Mod(
                op.Add(
                    op.Sub(offset, op.Constant(value_int=1)),
                    op.Constant(value_int=cache_length),
                ),
                op.Constant(value_int=cache_length),
            ),
            op.Constant(value_ints=[1, 1, 1]),
        ),
        op.Shape(target_index),
    )
    input_frame = op.GatherElements(cache, input_index, axis=2)
    target = op.GatherElements(cache, target_index, axis=2)
    target_provided = op.GatherElements(provided, target_index, axis=2)

    cache.shape = ir.Shape(["batch", channels, cache_length])
    provided.shape = ir.Shape(["batch", channels, cache_length])
    for value in (input_frame, target, target_provided):
        value.shape = ir.Shape(["batch", channels, 1])
    builder.add_output(cache, "next_token_cache")
    builder.add_output(provided, "next_token_provided")
    builder.add_output(input_frame, "input_frame")
    builder.add_output(target, "target")
    builder.add_output(target_provided, "target_provided")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_duplex_frame_commit(
    *, channels: int = 17, cache_length: int = 4, max_delay: int = 1
) -> PolicyComponent:
    """Commit predicted tokens and read the delay-compensated output frame.

    Predictions only fill ring slots that were not externally supplied, so a
    teacher-forced stream keeps its supplied value. The emitted frame undoes the
    per-stream delay by reading each stream at ``offset - max_delay + delay``,
    which is only well defined once ``offset`` has passed ``max_delay``.
    """
    graph, builder = _make_graph("duplex_frame_commit")
    op = builder.op
    cache = builder.input("token_cache", ir.DataType.INT64, ["batch", channels, cache_length])
    provided = builder.input(
        "token_provided", ir.DataType.BOOL, ["batch", channels, cache_length]
    )
    offset = builder.input("offset", ir.DataType.INT64, [])
    frame = builder.input("frame", ir.DataType.INT64, ["batch", channels])
    delays = builder.input("delays", ir.DataType.INT64, [channels])

    batch_shape = op.Shape(cache, start=0, end=1)
    cell_shape = op.Concat(batch_shape, op.Constant(value_ints=[channels, 1]), axis=0)
    false_like = op.ConstantOfShape(cell_shape, value=ir.tensor([False]))

    def ring_index(value: Any) -> Any:
        return op.Expand(
            op.Reshape(
                op.Mod(
                    op.Add(value, op.Constant(value_int=cache_length)),
                    op.Constant(value_int=cache_length),
                ),
                op.Constant(value_ints=[1, 1, 1]),
            ),
            cell_shape,
        )

    # 1. retire the slot that was just consumed as model input.
    input_index = ring_index(op.Sub(offset, op.Constant(value_int=1)))
    provided = op.ScatterElements(provided, input_index, false_like, axis=2)

    # 2. fill only the slots the caller did not supply.
    target_index = ring_index(offset)
    target_provided = op.GatherElements(provided, target_index, axis=2)
    existing = op.GatherElements(cache, target_index, axis=2)
    cache = op.ScatterElements(
        cache,
        target_index,
        op.Where(target_provided, existing, op.Unsqueeze(frame, [-1])),
        axis=2,
    )

    # 3. undo the per-stream delay: stream k is read at offset - max_delay + delay[k].
    read_index = _duplex_positions(
        op,
        op.Sub(offset, op.Constant(value_int=max_delay)),
        delays,
        cache_length,
        channels,
        batch_shape,
    )
    out_frame = op.Squeeze(op.GatherElements(cache, read_index, axis=2), [-1])
    next_offset = op.Add(offset, op.Constant(value_int=1))
    emit = op.Greater(offset, op.Constant(value_int=max_delay))

    cache.shape = ir.Shape(["batch", channels, cache_length])
    provided.shape = ir.Shape(["batch", channels, cache_length])
    out_frame.shape = ir.Shape(["batch", channels])
    next_offset.shape = ir.Shape([])
    emit.shape = ir.Shape([])
    builder.add_output(cache, "next_token_cache")
    builder.add_output(provided, "next_token_provided")
    builder.add_output(out_frame, "out_frame")
    builder.add_output(next_offset, "next_offset")
    builder.add_output(emit, "emit")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_duplex_teacher_select(*, channels: int = 17) -> PolicyComponent:
    """Choose the supplied token over the sampled token for one stream index."""
    graph, builder = _make_graph("duplex_teacher_select")
    op = builder.op
    target = builder.input("target", ir.DataType.INT64, ["batch", channels, 1])
    target_provided = builder.input(
        "target_provided", ir.DataType.BOOL, ["batch", channels, 1]
    )
    sampled = builder.input("sampled", ir.DataType.INT64, ["batch"])
    index = builder.input("index", ir.DataType.INT64, [])
    stream = op.Reshape(index, op.Constant(value_ints=[1]))
    picked = op.Squeeze(op.Gather(target, stream, axis=1), [1, 2])
    flag = op.Squeeze(op.Gather(target_provided, stream, axis=1), [1, 2])
    token = op.Where(flag, picked, sampled)
    token.shape = ir.Shape(["batch"])
    builder.add_output(token, "token")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_duplex_stream_append(
    *, streams: int = 8, dtype: ir.DataType = ir.DataType.INT64
) -> PolicyComponent:
    """Append one frame to a growing ``[batch, streams, length]`` prefix."""
    graph, builder = _make_graph("duplex_stream_append")
    op = builder.op
    prefix = builder.input("prefix", dtype, ["batch", streams, "length"])
    frame = builder.input("frame", dtype, ["batch", streams])
    appended = op.Concat(prefix, op.Unsqueeze(frame, [-1]), axis=2)
    appended.shape = ir.Shape(["batch", streams, "length + 1"])
    builder.add_output(appended, "next_prefix")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_duplex_waveform_append(*, dtype: ir.DataType = ir.DataType.FLOAT) -> PolicyComponent:
    """Append a packed audio chunk to a growing ``[batch, 1, samples]`` prefix."""
    graph, builder = _make_graph("duplex_waveform_append")
    op = builder.op
    prefix = builder.input("prefix", dtype, ["batch", 1, "samples"])
    chunk = builder.input("chunk", dtype, ["batch", 1, "chunk"])
    appended = op.Concat(prefix, chunk, axis=2)
    appended.shape = ir.Shape(["batch", 1, "samples + chunk"])
    builder.add_output(appended, "next_prefix")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_duplex_stream_tail(
    *, streams: int = 8, dtype: ir.DataType = ir.DataType.INT64, rank: int = 3
) -> PolicyComponent:
    """Read the trailing ``count`` positions of a growing prefix.

    Stateless codec graphs are replayed over an accumulated prefix, so only the
    newest ``count`` positions belong to the current event.
    """
    graph, builder = _make_graph("duplex_stream_tail")
    op = builder.op
    shape = ["batch", streams, "length"] if rank == 3 else ["batch", "length"]
    prefix = builder.input("prefix", dtype, shape)
    count = builder.input("count", ir.DataType.INT64, [])
    axis = rank - 1
    length = op.Squeeze(op.Shape(prefix, start=axis, end=axis + 1), [0])
    start = op.Reshape(op.Sub(length, count), op.Constant(value_ints=[1]))
    tail = op.Slice(
        prefix,
        start,
        op.Constant(value_ints=[_INT64_MAX]),
        op.Constant(value_ints=[axis]),
    )
    tail.shape = ir.Shape([*shape[:-1], "count"])
    builder.add_output(tail, "tail")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_duplex_user_stream_merge(*, channels: int = 17, streams: int = 8) -> PolicyComponent:
    """Overlay freshly encoded user codes onto the supplied stream-token frame.

    The trailing ``streams`` channels of a full-duplex frame carry the incoming
    user audio, so the codec output always wins over the request-supplied frame.
    """
    graph, builder = _make_graph("duplex_user_stream_merge")
    op = builder.op
    frame = builder.input("frame_codes", ir.DataType.INT64, ["batch", channels])
    codes = builder.input("codes", ir.DataType.INT64, ["batch", streams, 1])
    head = op.Slice(
        frame,
        op.Constant(value_ints=[0]),
        op.Constant(value_ints=[channels - streams]),
        op.Constant(value_ints=[1]),
    )
    merged = op.Concat(head, op.Squeeze(codes, [-1]), axis=1)
    merged.shape = ir.Shape(["batch", channels])
    builder.add_output(merged, "stream_tokens")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_duplex_cell_to_frame(*, channels: int = 17) -> PolicyComponent:
    """Drop the single-position axis of one ring-buffer cell."""
    graph, builder = _make_graph("duplex_cell_to_frame")
    target = builder.input("target", ir.DataType.INT64, ["batch", channels, 1])
    frame = builder.op.Squeeze(target, [-1])
    frame.shape = ir.Shape(["batch", channels])
    builder.add_output(frame, "frame")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_duplex_agent_frame_select(
    *, channels: int = 17, streams: int = 8
) -> PolicyComponent:
    """Read the agent acoustic streams out of a delay-compensated frame."""
    graph, builder = _make_graph("duplex_agent_frame_select")
    op = builder.op
    frame = builder.input("frame", ir.DataType.INT64, ["batch", channels])
    codes = op.Slice(
        frame,
        op.Constant(value_ints=[1]),
        op.Constant(value_ints=[1 + streams]),
        op.Constant(value_ints=[1]),
    )
    codes.shape = ir.Shape(["batch", streams])
    builder.add_output(codes, "codes")
    return _component("mobius.policy.auxiliary@1", graph, {})
