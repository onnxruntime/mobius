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
from typing import Protocol

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


def build_greedy_sampler(*, effect: str = "sample") -> PolicyComponent:
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
        "onnx-genai.token-sampler@1",
        graph,
        {
            "role": "token_sampler",
            "mode": "greedy",
            "logits": "logits",
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


def build_decoder_state_initializer(
    decoder: ir.Model,
    *,
    token_input: str | None,
    prompt_dtype: ir.DataType | None = None,
    attention_mask_input: str,
    position_ids_input: str | None,
    cache_inputs: list[str],
    fixed_capacity: bool = False,
) -> PolicyComponent:
    """Build prompt-derived decoder state, optionally with capture-stable storage."""
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
    capacity = None
    if fixed_capacity:
        max_iterations = builder.input(
            "max_iterations",
            dtype=ir.DataType.INT64,
            shape=[1],
        )
        capacity = op.Add(
            sequence_length,
            op.Squeeze(max_iterations, op.Constant(value_ints=[0])),
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

    attention_value = decoder_inputs[attention_mask_input]
    if fixed_capacity:
        attention = op.Cast(op.Less(offsets, sequence_length), to=attention_value.dtype)
        attention.shape = ir.Shape(["batch", "capacity"])
        body_attention = op.Cast(
            op.Less(
                offsets,
                op.Add(sequence_length, op.Constant(value_int=1)),
            ),
            to=attention_value.dtype,
        )
        body_attention.shape = ir.Shape(["batch", "capacity"])
    else:
        attention = op.Cast(
            op.ConstantOfShape(prompt_shape, value=ir.tensor([1])),
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
        positions = op.Range(
            op.Constant(value_int=0),
            op.Squeeze(sequence_shape, op.Constant(value_ints=[0])),
            op.Constant(value_int=1),
        )
        positions = op.Expand(
            op.Unsqueeze(positions, op.Constant(value_ints=[0])), prompt_shape
        )
        positions = op.Cast(positions, to=position_value.dtype)
        positions.shape = position_value.shape
        body_position = op.Expand(
            op.Cast(sequence_shape, to=position_value.dtype),
            op.Concat(batch_shape, op.Constant(value_ints=[1]), axis=0),
        )
        body_position.shape = ir.Shape(["batch", 1])
    builder.add_output(attention, attention_mask_input)
    if position_ids_input is not None:
        assert positions is not None and body_position is not None
        builder.add_output(positions, position_ids_input)
    builder.add_output(body_attention, "body_attention_mask")
    if position_ids_input is not None:
        builder.add_output(body_position, "body_position_ids")
    builder.add_output(token_slot, "token_slot")
    cache_lengths = op.Expand(
        op.Unsqueeze(sequence_length, op.Constant(value_ints=[0])),
        batch_shape,
    )
    cache_lengths.shape = ir.Shape(["batch"])
    builder.add_output(cache_lengths, "cache_lengths")

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
        zero = 0.0 if value.dtype.is_floating_point else 0
        empty = op.ConstantOfShape(
            cache_shape,
            value=ir.tensor([zero], dtype=value.dtype),
        )
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
    attention_dtype: ir.DataType,
    position_dtype: ir.DataType | None,
    fixed_capacity: bool = False,
) -> PolicyComponent:
    """Build one-token attention-mask and position update."""
    graph, builder = _make_graph("decoder_step_update")
    op = builder.op
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
        position = builder.input(
            "position_ids",
            dtype=position_dtype,
            shape=["batch", 1],
        )
        next_position = op.Add(position, op.CastLike(op.Constant(value_int=1), position))
        next_position.shape = ir.Shape(["batch", 1])
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
    Temperature, top-k, top-p, and grammar constraints remain request inputs;
    changing ordinary generation options never regenerates this artifact.
    """
    graph, builder = _make_graph("seeded_categorical_sampler")
    op = builder.op
    logits = builder.input("logits", ir.DataType.FLOAT, ["batch", "vocabulary"])
    temperature = builder.input("temperature", ir.DataType.FLOAT, [1])
    top_k = builder.input("top_k", ir.DataType.INT64, [1])
    top_p = builder.input("top_p", ir.DataType.FLOAT, [1])
    min_p = builder.input("min_p", ir.DataType.FLOAT, [1])
    grammar_mask = builder.input("grammar_mask", ir.DataType.BOOL, ["batch", "vocabulary"])
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

    blocked = op.CastLike(op.Constant(value_float=-3.4028235e38), logits)
    constrained_logits = op.Where(grammar_mask, logits, blocked)
    safe_temperature = op.Max(temperature, op.Constant(value_float=1e-6))
    scaled_logits = op.Div(constrained_logits, safe_temperature)
    safe_min_p = op.Clip(
        min_p,
        op.Constant(value_float=1e-20),
        op.Constant(value_float=1.0),
    )
    min_p_threshold = op.Add(
        op.ReduceMax(scaled_logits, axes=[-1], keepdims=1),
        op.Log(safe_min_p),
    )
    min_p_mask = op.Or(
        op.LessOrEqual(min_p, op.Constant(value_float=0.0)),
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
        effective_k,
        axis=-1,
        largest=1,
        sorted=0,
        _outputs=2,
    )
    top_k_mask = op.Greater(
        op.ScatterElements(
            op.ConstantOfShape(
                op.Shape(logits),
                value=ir.tensor([0], dtype=ir.DataType.INT64),
            ),
            top_indices,
            op.ConstantOfShape(
                op.Shape(top_indices),
                value=ir.tensor([1], dtype=ir.DataType.INT64),
            ),
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
    safe_top_p = op.Clip(
        top_p,
        op.Constant(value_float=1e-6),
        op.Constant(value_float=1.0),
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
    has_allowed_token = op.ReduceMax(
        op.Cast(grammar_mask, to=ir.DataType.INT64),
        axes=[-1],
        keepdims=0,
    )
    token_ids = op.Where(
        op.Greater(has_allowed_token, op.Constant(value_int=0)),
        token_ids,
        op.Constant(value_int=-1),
    )
    next_offset = op.Add(offset, op.Constant(value_int=1))
    builder.add_output(token_ids, "token")
    builder.add_output(next_offset, "next_offset")
    return _component(
        "onnx-genai.token-sampler@1",
        graph,
        {
            "role": "token_sampler",
            "mode": "seeded_stochastic",
            "logits": "logits",
            "token": "token",
            "temperature": "temperature",
            "top_k": "top_k",
            "top_p": "top_p",
            "min_p": "min_p",
            "grammar_mask": "grammar_mask",
            "rng": {
                "rng_seed": "seed",
                "rng_offset": "offset",
                "rng_next_offset": "next_offset",
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
    continued = op.Equal(
        op.ReduceMax(op.Cast(done, to=ir.DataType.INT64), keepdims=1),
        op.Constant(value_int=0),
    )
    continued.shape = ir.Shape([1])
    builder.add_output(done, "done")
    builder.add_output(continued, "continue")
    return _component(
        "onnx-genai.termination-predicate@1",
        graph,
        {
            "role": "termination_predicate",
            "tokens": "token_ids",
            "eos_ids": "eos_ids",
            "iteration": "iteration",
            "max_iterations": "max_iterations",
            "done": "done",
            "continue": "continue",
            "effect": "termination",
        },
        "termination",
    )


def build_euler_model_input(dtype: ir.DataType = ir.DataType.FLOAT) -> PolicyComponent:
    """Scale a latent for the Euler denoiser input at the current sigma."""
    graph, builder = _make_graph("euler_model_input")
    op = builder.op
    sample = builder.input("sample", dtype, ["batch", "channels", "height", "width"])
    step = builder.input("step", ir.DataType.INT64, ["batch"])
    schedule = builder.input("schedule", ir.DataType.FLOAT, ["schedule_length"])
    sigma = op.Gather(schedule, step, axis=0)
    scale = op.Sqrt(op.Add(op.Mul(sigma, sigma), op.Constant(value_float=1.0)))
    scale = op.Cast(scale, to=dtype)
    scale = op.Unsqueeze(scale, op.Constant(value_ints=[1, 2, 3]))
    model_input = op.Div(sample, scale)
    model_input.shape = sample.shape
    builder.add_output(model_input, "model_input")
    return _component("mobius.policy.auxiliary@1", graph, {})


def build_euler_solver_step(
    dtype: ir.DataType = ir.DataType.FLOAT,
) -> PolicyComponent:
    """Build the generic Euler update ``x_next = x + dx * (sigma_next-sigma)``."""
    graph, builder = _make_graph("euler_solver_step")
    op = builder.op
    sample = builder.input(
        "sample",
        dtype,
        ["batch", "channels", "height", "width"],
    )
    derivative = builder.input(
        "derivative",
        dtype,
        ["batch", "channels", "height", "width"],
    )
    step = builder.input("step", ir.DataType.INT64, ["batch"])
    schedule = builder.input("schedule", ir.DataType.FLOAT, ["schedule_length"])
    final_index = op.Sub(op.Shape(schedule, start=0, end=1), op.Constant(value_ints=[1]))
    next_step = op.Min(op.Add(step, op.Constant(value_int=1)), final_index)
    sigma = op.Gather(schedule, step, axis=0)
    sigma_next = op.Gather(schedule, next_step, axis=0)
    delta = op.Sub(sigma_next, sigma)
    delta = op.Cast(delta, to=dtype)
    delta = op.Unsqueeze(delta, op.Constant(value_ints=[1, 2, 3]))
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
        "onnx-genai.state-update@1",
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


def build_token_to_slot() -> PolicyComponent:
    """Convert a canonical sampled token vector to a one-token ID tensor."""
    graph, builder = _make_graph("token_to_slot")
    token = builder.input("token", ir.DataType.INT64, ["batch"])
    slot = builder.op.Unsqueeze(token, [-1])
    slot.shape = ir.Shape(["batch", 1])
    builder.add_output(slot, "slot")
    return _component("mobius.policy.auxiliary@1", graph, {})
