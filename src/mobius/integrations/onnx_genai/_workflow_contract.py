# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Dependency-neutral primitives for published ONNX-GenAI workflow contracts."""

from __future__ import annotations

import dataclasses
from typing import Any

import onnx_ir as ir


@dataclasses.dataclass(frozen=True)
class _Port:
    """Structural description of one ONNX graph port."""

    value: Any
    name: str
    dtype: str
    rank: int | None
    dims: tuple[Any, ...]


_DTYPE_TAGS = {
    "FLOAT": "fp32",
    "FLOAT16": "fp16",
    "BFLOAT16": "bf16",
    "FLOAT8E4M3FN": "float8_e4m3fn",
    "FLOAT8E5M2": "float8_e5m2",
    "INT64": "int64",
    "INT32": "int32",
    "INT8": "int8",
    "UINT8": "uint8",
    "BOOL": "bool",
    "STRING": "string",
}


def _port(value: Any) -> _Port:
    shape = getattr(value, "shape", None)
    dims = tuple(shape) if shape is not None else ()
    dtype = getattr(getattr(value, "dtype", None), "name", "")
    return _Port(
        value=value,
        name=str(value.name),
        dtype=_DTYPE_TAGS.get(str(dtype).upper(), str(dtype).lower() or "fp32"),
        rank=len(dims) if shape is not None else None,
        dims=dims,
    )


def _shape_metadata(port: _Port) -> list[int | str]:
    """Return a YAML-safe graph shape without losing symbolic dimensions."""
    shape: list[int | str] = []
    for axis, dim in enumerate(port.dims):
        if isinstance(dim, int):
            shape.append(dim)
            continue
        value = getattr(dim, "value", None)
        # Metadata dimensions cannot be null. Preserve named graph dimensions;
        # give anonymous dynamic dimensions a stable, port-local name instead
        # of pretending they are static or serializing an invalid null.
        shape.append(str(value) if value is not None else f"{port.name}_dim_{axis}")
    return shape


#: Symbolic dimension Mobius emits for every request-aligned ONNX port.
REQUEST_AXIS_SYMBOL = "batch"
_BATCH_DIMENSION_NAMES = frozenset({"batch", "batch_size", "batch_dim", "b"})


def request_batch_layout(shape: list[Any] | None) -> dict[str, Any] | None:
    """Return the request-aligned batch layout implied by a port's shape."""
    axes = [
        axis
        for axis, dimension in enumerate(shape or [])
        if str(dimension) in _BATCH_DIMENSION_NAMES
    ]
    if len(axes) == 1:
        return {"kind": "request_aligned", "axis": axes[0]}
    return None


def declare_request_alignment(workflow: dict[str, Any]) -> None:
    """Stamp the request axis named by exactly one batch symbol.

    The runtime compacts finished rows out of a batch by applying one row
    permutation to every request-aligned tensor. A contract whose batch axis
    does not say so is unpermutable, so state,
    component ports, and outputs would silently drift apart after the first
    eviction. Deriving the declaration from the admitted graph's own batch
    symbol keeps alignment a property of the model interface rather than an
    annotation every workflow builder has to remember.
    """

    def stamp(contract: Any) -> None:
        if not isinstance(contract, dict) or "batch_layout" in contract:
            return
        shape = contract.get("shape") or []
        if layout := request_batch_layout(shape):
            contract["batch_layout"] = layout

    for section in ("inputs", "outputs", "state"):
        for declaration in (workflow.get(section) or {}).values():
            if isinstance(declaration, dict):
                stamp(declaration.get("contract"))
    for component in (workflow.get("components") or {}).values():
        ports = component.get("ports", {}) if isinstance(component, dict) else {}
        for side in ("inputs", "outputs"):
            for contract in (ports.get(side) or {}).values():
                stamp(contract)
    # A cell backed by a state-service group is stored by the runtime, not by
    # the workflow: the group owns the buffer and the eviction policy, so the
    # cell also needs an explicit boundary at which the runtime may free it.
    for declaration in (workflow.get("state") or {}).values():
        if isinstance(declaration, dict) and declaration.get("service_group"):
            declaration.setdefault("management", "runtime")
            declaration.setdefault("release_boundary", declaration.get("scope", "invocation"))


def published_value_references(workflow: dict[str, Any]) -> set[str]:
    """Every value name the published program reads, on any path.

    Reachability here is deliberately path-insensitive: it answers "does this
    workflow ever look at this value", which is what an admission decision
    needs. Whether a particular request reaches the branch that reads it is a
    runtime fact, and a package that guessed at it would be describing one
    caller rather than its own contract.
    """
    references: set[str] = set()

    def note(value: Any) -> None:
        if isinstance(value, str):
            references.add(value)
        elif isinstance(value, dict):
            for item in value.values():
                note(item)
        elif isinstance(value, list):
            for item in value:
                note(item)

    def visit(step: Any) -> None:
        if isinstance(step, list):
            for item in step:
                visit(item)
            return
        if not isinstance(step, dict):
            return
        kind = step.get("kind")
        if kind == "invoke":
            note(step.get("inputs"))
        elif kind == "emit":
            note(step.get("value"))
            note(step.get("valid_length"))
            note(step.get("when"))
        elif kind == "branch":
            note(step.get("predicate"))
            note(step.get("outputs"))
            for case in (step.get("cases") or {}).values():
                visit(case)
            visit(step.get("default"))
        elif kind == "loop":
            note(step.get("continue_when"))
            note(step.get("max_iterations"))
            for carry in step.get("carried") or []:
                note(carry.get("next"))
                note(carry.get("initial"))
        for key in ("steps", "setup", "nodes"):
            visit(step.get(key))

    visit(workflow.get("steps") or [])
    for declaration in (workflow.get("state") or {}).values():
        if isinstance(declaration, dict):
            note(declaration.get("initializer"))
            # A bounded or growing cell reads its own extent every step, so the
            # bound and the step size are read positions like any other. The
            # sibling ``kind``/``axis`` keys describe the recurrence rather than
            # naming values, so they are not references.
            recurrence = declaration.get("recurrence") or {}
            note(recurrence.get("max"))
            note(recurrence.get("increment"))
    # The serving block names the workflow values a runtime reads to drive
    # batching -- the active/done row masks and the accepted length.
    serving = workflow.get("serving") or {}
    for key, value in serving.items():
        if key != "state_service":
            note(value)
    # ``state_service`` is mostly port and cell names rather than values, with
    # one exception: a group's fixed update extent is itself a workflow value.
    for group in ((serving.get("state_service") or {}).get("groups") or {}).values():
        if isinstance(group, dict):
            note((group.get("update") or {}).get("capacity"))
    return references


def declare_input_admission(workflow: dict[str, Any]) -> None:
    """Publish every package input's admission requirement instead of implying it.

    ``required`` is what a runtime admits a request against: an input it holds
    required and the caller did not attach is a rejected request, on every path,
    before a single component runs. A consumer cannot see what an *absent*
    ``required`` key was meant to say, so it has to choose a default, and the
    choice it makes is the opposite of what omission means to a producer -- a
    value the workflow computes for itself, defaults for itself, or explicitly
    branches on the absence of, silently becomes a mandatory caller attachment.

    So the flag is derived from the published program rather than left to a
    reader, and stamped on every declaration:

    * a declaration the package can satisfy on its own -- it carries a
      ``default``, which is also the only thing a package-owned ``literal``
      source is ever bound from -- is not something a caller can be required to
      send;
    * a declaration whose absence the program *observes*, through a
      ``present_as`` symbol the steps actually branch on, is one the workflow
      has been written to run without;
    * anything else is genuinely externally required, and says so out loud.

    A builder that declares both an escape and ``required: True`` has written
    two contradictory contracts, and there is no reading of the package that
    satisfies both, so this fails closed rather than picking one. The mirror
    case fails closed for the same reason: an input marked optional that the
    package has no way to proceed without is not optional, it is a request that
    is admitted and then fails part-way through on an unbound value, which is
    strictly worse than the rejection it replaced.
    """
    inputs = workflow.get("inputs") or {}
    if not inputs:
        return
    references = published_value_references(workflow)
    for name, declaration in inputs.items():
        if not isinstance(declaration, dict):
            continue
        escapes = []
        literal = (declaration.get("source") or {}).get("kind") == "literal"
        if "default" in declaration:
            escapes.append("a package-supplied default" if literal else "a default")
        elif literal:
            # A literal source is resolved from the declaration's own default
            # and from nothing else, so one without a default names a value the
            # package neither holds nor can ask a caller for.
            raise ValueError(
                f"workflow input {name!r} is sourced from a package literal but "
                "carries no default, so nothing ever binds it"
            )
        present_as = declaration.get("present_as")
        if present_as is not None:
            if present_as not in references:
                raise ValueError(
                    f"workflow input {name!r} declares the presence symbol "
                    f"{present_as!r} that no step reads, so the workflow never "
                    "handles the input being absent"
                )
            escapes.append(f"the presence gate {present_as!r}")
        if not escapes:
            if declaration.get("required", True) is False:
                raise ValueError(
                    f"workflow input {name!r} is declared optional but the workflow "
                    "carries no default and no presence gate for it, so a request "
                    "that omits it has no defined behaviour"
                )
            declaration["required"] = True
            continue
        if declaration.get("required", False):
            raise ValueError(
                f"workflow input {name!r} is declared required but the workflow "
                f"already proceeds without it through {', '.join(escapes)}"
            )
        declaration["required"] = False


def add_policy_components_to_workflow(
    metadata: dict[str, Any],
    pkg: Any,
) -> dict[str, Any]:
    """Reference attached ONNX policy artifacts from an existing workflow.

    This helper intentionally does not synthesize a workflow or guess bindings.
    It only adds schema-defined component declarations when a producer has
    already emitted the exact workflow contract.
    """
    policy_components = getattr(pkg, "policy_components", {})
    if not policy_components:
        return metadata
    workflow = metadata.get("pipeline", {}).get("workflow")
    if not isinstance(workflow, dict):
        return metadata
    components = workflow.setdefault("components", {})

    def semantic_contract(component: Any) -> dict[str, Any]:
        contract = component.contract
        contract_name, version = component.contract_id.rsplit("@", 1)
        bindings = {
            key: value
            for key, value in contract.items()
            if key
            not in {
                "role",
                "mode",
                "effect",
                "rng",
                "state_class",
                "batching",
                "inactive_rows",
            }
            and isinstance(value, str)
        }
        rng = contract.get("rng")
        if isinstance(rng, dict):
            bindings.update(
                {key: value for key, value in rng.items() if isinstance(value, str)}
            )
        declaration: dict[str, Any] = {
            "id": contract_name,
            "version": version,
            "bindings": bindings,
        }
        parameters = {
            key: contract[key]
            for key in ("mode", "batching", "inactive_rows")
            if key in contract
        }
        if parameters:
            declaration["parameters"] = parameters
        return declaration

    def tensor_contract(value: Any) -> dict[str, Any]:
        port = _port(value)
        dtype = {
            "fp32": "float32",
            "fp16": "float16",
            "bf16": "bfloat16",
        }.get(port.dtype, port.dtype)
        shape = _shape_metadata(port)
        contract: dict[str, Any] = {
            "dtype": dtype,
            "rank": port.rank,
            "shape": shape,
        }
        layout = request_batch_layout(shape)
        if layout is not None:
            contract["batch_layout"] = layout
        return contract

    for name, component in policy_components.items():
        # A policy graph is synthesized by this producer to realize the
        # workflow's own control flow, so its port contracts are not a
        # transcription of an external interface: they are the type annotations
        # of the workflow's dataflow. A workflow value acquires its dtype, rank
        # and request axis from the port that produces it, and the validator
        # reads metadata without the artifacts, so a policy output that states
        # no contract leaves every value derived from it untyped.
        declaration = {
            "implementation": {
                "kind": "onnx",
                "artifact": f"policies/{name}.onnx",
            },
            "ports": {
                "inputs": {
                    value.name: tensor_contract(value)
                    for value in component.model.graph.inputs
                },
                "outputs": {
                    value.name: tensor_contract(value)
                    for value in component.model.graph.outputs
                },
            },
        }
        if component.contract:
            declaration["contract"] = semantic_contract(component)
            if component.contract.get("role") == "token_sampler":
                declaration["application_overridable"] = True
        components[name] = declaration
    declare_request_alignment(workflow)
    declare_input_admission(workflow)
    return metadata


def _contract(value: ir.Value) -> dict[str, Any]:
    port = _port(value)
    dtype = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}.get(
        port.dtype, port.dtype
    )
    shape = _shape_metadata(port)
    contract: dict[str, Any] = {
        "dtype": dtype,
        "rank": port.rank,
        "shape": shape,
    }
    layout = request_batch_layout(shape)
    if layout is not None:
        contract["batch_layout"] = layout
    return contract


def _request_aligned(contract: dict[str, Any], axis: int = 0) -> dict[str, Any]:
    """Mark a contract as carrying exactly one entry per in-flight request.

    This is a structural batching fact, not a row identity: it tells the runtime
    which axis to permute when it compacts the batch, while scheduler slots and
    sequence handles stay runtime-private.
    """
    return {**contract, "batch_layout": {"kind": "request_aligned", "axis": axis}}


# Translation between the port vocabulary this producer *mints* when it builds
# a graph and the runtime's architecture-neutral role vocabulary. Both sides are
# fixed vocabularies and Mobius owns one of them: the task builders in
# ``mobius.tasks`` choose these exact names, so reading them back here is a
# lookup, not an inference about a graph of unknown provenance. A port outside
# this vocabulary carries no role, because a workflow that guesses is worse than
# one that stays silent.
_PORT_ROLES: dict[str, str] = {
    "input_ids": "token_ids",
    "inputs_embeds": "inputs_embeds",
    "attention_mask": "attention_mask",
    "position_ids": "position_ids",
    "logits": "logits",
    "last_hidden_state": "hidden_states",
    "encoder_hidden_states": "encoder_hidden_states",
    "audio_features": "audio_features",
}


def _component(
    model: ir.Model,
    artifact: str,
    *,
    effects: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Declare one ONNX-backed workflow component: its artifact and port roles.

    A component declares only what its artifact cannot say about itself. The
    ``.onnx`` file is shipped inside the package and is authoritative for which
    ports exist and what dtype, rank and shape each one has, so transcribing
    that into YAML would create a second copy of a fact the package already
    carries — one that can drift from the graph and that nothing cross-checks
    at rest. The runtime resolves ports against the live session instead, which
    catches a name the graph does not expose rather than agreeing with a stale
    echo of it.

    What no graph carries is what a port *means*. ``input_ids`` and
    ``position_ids`` are both rank-2 ``int64``; nothing in the file says which
    one is the autoregressive sequence. An invocation binds an SSA value to a
    port, which records which value arrives but not whether it is tokens, a mask
    or logits — and that second fact is what a runtime needs before it can
    specialize a decode step. So ``roles`` is the whole declaration here.

    Only ports in this producer's own vocabulary get a role, and state ports
    never need one: the group that carries them already names its pairs, which
    is also where the fixed-capacity scatter ABI is stated.

    ``batch_capacity`` is intentionally absent. A request-aligned or dynamic
    batch axis is structural shape information, not proof that co-batching
    preserves each request's result. Builders may add that semantic permission
    only after the complete grouped contract has been authored and validated.
    """
    del effects
    named = [str(value.name) for value in (*model.graph.inputs, *model.graph.outputs)]
    roles = {name: _PORT_ROLES[name] for name in named if name in _PORT_ROLES}
    declaration: dict[str, Any] = {"implementation": {"kind": "onnx", "artifact": artifact}}
    if roles:
        declaration["ports"] = {"roles": roles}
    return declaration


def _effect(consumes: str, produces: str) -> dict[str, str]:
    return {"consumes": consumes, "produces": produces}


def _publish_workflow_v1(workflow: dict[str, Any]) -> dict[str, Any]:
    """Publish structured steps and logical carries without compiler bookkeeping."""
    graph = workflow.pop("graph")
    workflow.pop("initial_effects", None)
    for declaration in workflow.get("inputs", {}).values():
        source = declaration.get("source")
        if isinstance(source, dict) and source.get("kind") == "request":
            source.pop("field", None)

    # Every workflow value whose leading dimension is the batch symbol holds one
    # entry per in-flight request, so declare that structurally instead of leaving
    # a runtime to infer it. Graph-derived contracts already carry the layout;
    # this covers the hand-written declarations the runtime compares them against
    # when it validates a carry, a binding, or an emit.
    def _declare_row_alignment(contract: Any) -> Any:
        if (
            isinstance(contract, dict)
            and "batch_layout" not in contract
            and request_batch_layout(contract.get("shape")) is not None
        ):
            return _request_aligned(contract)
        return contract

    for section in ("inputs", "outputs", "state"):
        for declaration in workflow.get(section, {}).values():
            declaration["contract"] = _declare_row_alignment(declaration.get("contract"))
    for component in workflow.get("components", {}).values():
        for side in ("inputs", "outputs"):
            ports = component.get("ports", {}).get(side)
            if not ports:
                continue
            for port, contract in ports.items():
                ports[port] = _declare_row_alignment(contract)
    substitutions: dict[str, str] = {}
    loop_index = 0
    cell_aliases = {
        cell: f"{cell}_state" if cell in workflow.get("outputs", {}) else cell
        for cell in workflow.get("state", {})
    }
    if any(cell != alias for cell, alias in cell_aliases.items()):
        workflow["state"] = {
            cell_aliases[cell]: declaration for cell, declaration in workflow["state"].items()
        }

    def collect_carried(node: dict[str, Any]) -> None:
        if node["kind"] == "loop":
            for carry in node.get("carried", []):
                alias = cell_aliases.get(carry["cell"], carry["cell"])
                substitutions[carry["body_input"]] = alias
                substitutions[carry["next"]] = alias
            collect_carried(node["setup"])
            collect_carried(node["body"])
        elif node["kind"] == "sequence":
            for child in node["nodes"]:
                collect_carried(child)
        elif node["kind"] == "branch":
            for case in node["cases"].values():
                collect_carried(case)
            if "default" in node:
                collect_carried(node["default"])

    def rewrite(value: Any) -> Any:
        if isinstance(value, str):
            return substitutions.get(value, value)
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        return value

    def convert(node: dict[str, Any]) -> dict[str, Any]:
        nonlocal loop_index
        kind = node["kind"]
        if kind == "sequence":
            return {
                "kind": "sequence",
                "steps": [convert(child) for child in node["nodes"]],
            }
        if kind == "invoke":
            return {
                "kind": "invoke",
                "component": node["component"],
                "inputs": rewrite(node.get("inputs", {})),
                "outputs": rewrite(node.get("outputs", {})),
            }
        if kind == "emit":
            result = {
                "kind": "emit",
                "value": rewrite(node["value"]),
                "output": node["output"],
                "mode": node["mode"],
            }
            if "axis" in node:
                result["axis"] = node["axis"]
            if "valid_length" in node:
                result["valid_length"] = rewrite(node["valid_length"])
            if "when" in node:
                result["when"] = rewrite(node["when"])
            return result
        if kind == "branch":
            result = {
                "kind": "branch",
                "predicate": rewrite(node["predicate"]),
                "cases": {name: convert(case) for name, case in node["cases"].items()},
                "outputs": rewrite(node.get("outputs", {})),
            }
            if "default" in node:
                result["default"] = convert(node["default"])
            return result
        if kind == "loop":
            current_loop = loop_index
            loop_index += 1
            setup = node["setup"]
            body = node["body"]
            setup_steps = (
                [convert(child) for child in setup["nodes"]]
                if setup["kind"] == "sequence"
                else [convert(setup)]
            )
            body_steps = (
                [convert(child) for child in body["nodes"]]
                if body["kind"] == "sequence"
                else [convert(body)]
            )
            carried = []
            for carry in node.get("carried", []):
                cell = cell_aliases.get(carry["cell"], carry["cell"])
                published_carry = {
                    "cell": cell,
                    "next": rewrite(carry["body_output"]),
                }
                initial = rewrite(carry["current"])
                if workflow["state"][cell]["initializer"] != initial:
                    published_carry["initial"] = initial
                carried.append(published_carry)
            active_cell = node.get("active_cell")
            if active_cell is None:
                active_cell = f"loop_{current_loop}_active"
                active_initializer = f"package.{active_cell}"
                workflow["inputs"][active_initializer] = {
                    "contract": {"dtype": "bool", "rank": 1, "shape": [1]},
                    "role": {"kind": "opaque"},
                    "source": {"kind": "literal"},
                    "required": False,
                    "default": True,
                }
                workflow["state"][active_cell] = {
                    "contract": {"dtype": "bool", "rank": 1, "shape": [1]},
                    "scope": "invocation",
                    "initializer": active_initializer,
                    "recurrence": {"kind": "invariant"},
                }
                carried.append(
                    {
                        "cell": active_cell,
                        "next": rewrite(node["condition"]),
                    }
                )
            result = {
                "kind": "loop",
                "setup": setup_steps,
                "steps": body_steps,
                "continue_when": active_cell,
                "max_iterations": rewrite(node["max_iterations"]),
                "carried": carried,
            }
            if "termination" in node:
                result["termination"] = node["termination"]
            if "iteration" in node:
                result["iteration"] = node["iteration"]
            return result
        raise ValueError(f"unsupported workflow node kind {kind!r}")

    collect_carried(graph)
    published = convert(graph)
    workflow["steps"] = published["steps"] if published["kind"] == "sequence" else [published]
    declare_request_alignment(workflow)
    declare_input_admission(workflow)
    return workflow


def _invoke(
    component: str,
    inputs: dict[str, str],
    outputs: dict[str, str],
    _effects: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "kind": "invoke",
        "component": component,
        "inputs": inputs,
        "outputs": outputs,
    }


def _model_cache_pairs(model: ir.Model) -> list[tuple[ir.Value, ir.Value]]:
    outputs = {value.name: value for value in model.graph.outputs}
    pairs = []
    for past in model.graph.inputs:
        present = next(
            (
                outputs.get(name)
                for name in _cache_output_candidates(past.name or "")
                if name in outputs
            ),
            None,
        )
        if present is not None:
            pairs.append((past, present))
    return pairs


def _cache_output_candidates(past_name: str) -> tuple[str, ...]:
    """Names an exporter may give the output that continues a cache input.

    An appending cache renames ``past`` to ``present``; a static, indexed cache
    keeps the buffer's name and prefixes the written result instead, because the
    output is the same buffer rather than a longer one.
    """
    return (
        past_name.replace("past_key_values", "present"),
        past_name.replace("past.", "present."),
        past_name.replace("past_", "present_"),
        f"updated_{past_name}",
    )
