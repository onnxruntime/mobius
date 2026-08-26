# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""One serialized contract shape, for every package this producer can emit.

A package describes itself in exactly one place: ``pipeline.workflow``. That is
true of a three-graph vision-language package, and it is equally true of a bare
single-file decoder — the single-file case is a one-component workflow, not a
different kind of document with its own keys. A runtime is free to *lower* that
one component onto an optimized single-graph path; what it may not do is read a
second, independently writable statement of the same facts, because nothing
would force the two to agree and a reader of one never learns that the other
said something else.

These tests exist because that property is only worth anything if it cannot
quietly lapse. It would be easy for a feature added to one export shape — a
fixed-capacity cache, an FP8 buffer, a heterogeneous decoder — to grow its own
top-level block "just for this case", and easy for that to go unnoticed while
every feature-specific test kept passing. So the assertions here are
deliberately shape-agnostic: they are asked of dynamic, static-cache, FP8,
heterogeneous and composite packages through the same code path, and of every
checked-in fixture package, and none of them mentions a feature by name.
"""

from __future__ import annotations

import copy
import glob
import json
import os
import re
from typing import Any

import jsonschema
import onnx_ir as ir
import pytest
import yaml

from mobius import registry
from mobius._configs import ArchitectureConfig
from mobius._optimizations import optimize_model
from mobius.integrations.onnx_genai.inference_metadata import (
    published_value_references,
)
from mobius.integrations.onnx_genai.workflow_metadata import (
    build_decoder_workflow_metadata,
    build_vlm_workflow_metadata,
)
from mobius.tasks import CausalLMTask

CAPACITY = 64

# Enough layers that a cell label's lexicographic order stops agreeing with its
# numeric one: with ten or more cells, ``cache_10`` sorts between ``cache_1``
# and ``cache_2``. Below that threshold every ordering rule looks correct.
DEEP_LAYERS = 12

FIXTURE_ROOT = os.path.join(os.path.dirname(__file__), "fixtures", "onnx_genai_workflows")
SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "src",
    "mobius",
    "integrations",
    "onnx_genai",
    "_schema",
    "inference_metadata.schema.json",
)
with open(SCHEMA_PATH, encoding="utf-8") as _schema_handle:
    ONNX_GENAI_SCHEMA = json.load(_schema_handle)


def _text_config(**overrides: Any) -> ArchitectureConfig:
    params: dict[str, Any] = {
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 256,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "max_position_embeddings": 512,
    }
    params.update(overrides)
    return ArchitectureConfig(**params)


def _dynamic_decoder() -> tuple[Any, dict[str, Any]]:
    """One ONNX file, an appending cache: the simplest package there is."""
    config = _text_config()
    pkg = CausalLMTask().build(registry.get("qwen2")(config), config)
    return pkg, build_decoder_workflow_metadata(pkg, config)


def _static_cache_decoder() -> tuple[Any, dict[str, Any]]:
    """One ONNX file whose cache is scattered into fixed-capacity buffers."""
    config = _text_config()
    task = CausalLMTask(static_cache=True, max_seq_len=CAPACITY)
    pkg = task.build(registry.get("qwen2")(config), config)
    return pkg, build_decoder_workflow_metadata(pkg, config)


def _fp8_decoder() -> tuple[Any, dict[str, Any]]:
    """One ONNX file whose cache buffers are FP8 rather than the compute dtype."""
    config = _text_config()
    pkg = CausalLMTask().build(registry.get("qwen2")(config), config)
    optimize_model(
        pkg["model"],
        ep="cuda",
        dtype=ir.DataType.FLOAT16,
        model_role="decoder",
        fp8_kv_cache=True,
    )
    return pkg, build_decoder_workflow_metadata(pkg, config)


def _heterogeneous_decoder() -> tuple[Any, dict[str, Any]]:
    """One ONNX file that publishes two state groups instead of one.

    A hybrid decoder interleaves sliding and full attention, so its cells land
    in different groups with different eviction rules. The contract shape must
    not depend on how many groups a package happens to need.
    """
    config = _text_config(
        sliding_window=8, layer_types=["sliding_attention", "full_attention"]
    )
    pkg = CausalLMTask().build(registry.get("qwen2")(config), config)
    return pkg, build_decoder_workflow_metadata(pkg, config)


def _composite_vision_language() -> tuple[Any, dict[str, Any]]:
    """Three ONNX files driven by one workflow.

    This is the same package the runtime conformance suite executes, built by
    the same helper, so the composite arm of these tests and the composite arm
    of the executed fixtures cannot describe different things.
    """
    from generate_onnx_genai_validation_packages import _executable_vlm_package

    pkg = _executable_vlm_package()
    return pkg, build_vlm_workflow_metadata(pkg, pkg.config)


_PACKAGES = {
    "dynamic": _dynamic_decoder,
    "static_cache": _static_cache_decoder,
    "fp8": _fp8_decoder,
    "heterogeneous": _heterogeneous_decoder,
    "composite": _composite_vision_language,
}


@pytest.fixture(scope="module")
def built() -> dict[str, tuple[Any, dict[str, Any]]]:
    return {name: build() for name, build in _PACKAGES.items()}


@pytest.fixture(params=sorted(_PACKAGES), scope="module")
def package(request, built):
    return built[request.param]


def _onnx_components(workflow: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: component
        for name, component in workflow["components"].items()
        if component.get("implementation", {}).get("kind") == "onnx"
    }


def _graph_ports(model: ir.Model) -> tuple[set[str], set[str]]:
    """The ports an artifact actually exposes, which is the only authority."""
    return (
        {str(value.name) for value in model.graph.inputs},
        {str(value.name) for value in model.graph.outputs},
    )


def _artifacts(pkg: Any, workflow: dict[str, Any]) -> dict[str, ir.Model]:
    """Every ONNX component of *workflow* paired with the graph it references.

    A component names an artifact, and the artifact is what a runtime binds
    against. Resolving a declaration through this map is what makes these
    assertions checks of the package rather than checks of the metadata's
    internal consistency with itself.
    """
    policies = {
        name: component.model
        for name, component in getattr(pkg, "policy_components", {}).items()
    }
    graphs = {**dict(pkg.items()), **policies}
    return {name: graphs[name] for name in _onnx_components(workflow) if name in graphs}


def _walk_steps(steps: list[dict[str, Any]]):
    """Every step of a workflow, including the ones nested in loops and branches."""
    for step in steps:
        yield step
        for key in ("setup", "steps", "nodes"):
            if isinstance(step.get(key), list):
                yield from _walk_steps(step[key])
        for case in (step.get("cases") or {}).values():
            yield from _walk_steps([case])
        if isinstance(step.get("default"), dict):
            yield from _walk_steps([step["default"]])


def _groups(workflow: dict[str, Any]) -> dict[str, Any]:
    return (workflow.get("serving") or {}).get("state_service", {}).get("groups", {}) or {}


def _scatter_bound_components(workflow: dict[str, Any]) -> set[str]:
    """Components a fixed-capacity group hands its write cursor and valid length to.

    These two ports are consumed by exactly one thing: the driver that writes
    into a preallocated cache at an index. That driver binds its ports from the
    resolved decode ABI, so naming a component here is a claim that the
    component is resolvable as a decoder.
    """
    bound: set[str] = set()
    for group in _groups(workflow).values():
        update = group.get("update") or {}
        if update.get("kind") != "indexed_scatter":
            continue
        bound |= set(update.get("write_indices_ports") or {})
        bound |= set(update.get("kv_length_ports") or {})
    return bound


def _declares_a_sequence_role(component: dict[str, Any]) -> bool:
    roles = ((component.get("ports") or {}).get("roles")) or {}
    return bool({"token_ids", "inputs_embeds"} & set(roles.values()))


class TestOneSerializedContract:
    """Facts that hold for every package shape, asserted through one code path."""

    def test_the_workflow_is_where_a_package_describes_itself(self, package):
        _, metadata = package
        assert "workflow" in metadata["pipeline"]

    def test_no_producer_emits_retired_batching_declarations(self, package):
        _, metadata = package
        serialized = yaml.safe_dump(metadata)
        assert "batch_invariance:" not in serialized
        assert "continuous_batching:" not in serialized

    def test_no_package_states_its_graph_abi_a_second_time(self, package):
        """The point of one representation is that there is no other one.

        ``model`` may still carry package-wide geometry, but the moment it
        carries an ``io`` block the package has two writable answers to "what
        does the decode step look like", and a reader of either never learns the
        other exists.
        """
        _, metadata = package
        assert "io" not in (metadata.get("model") or {})

    def test_an_exported_graph_is_not_transcribed_into_the_workflow(self, package):
        """A model component declares its roles and nothing the artifact says.

        The ``.onnx`` file ships inside the package and is authoritative for
        which ports exist and what each one's dtype, rank and shape is. Copying
        that into YAML would create a second statement of the same fact with
        nothing to keep the two in agreement, which is the failure this whole
        module exists to prevent — the copy just happens to sit one level down
        from ``model.io`` rather than beside it.
        """
        pkg, metadata = package
        components = _onnx_components(metadata["pipeline"]["workflow"])
        exported = [name for name in components if name in pkg]
        assert exported, "a package with no exported graph describes nothing"
        for name in exported:
            ports = components[name].get("ports", {})
            assert not ports.get("inputs")
            assert not ports.get("outputs")

    def test_a_synthesized_policy_graph_still_types_the_dataflow(self, package):
        """The producer's own control graphs are the workflow's type annotations.

        A policy graph is not an external interface this producer describes; it
        is a graph this producer emits to realize the workflow's control flow,
        and the value an invocation produces takes its dtype, rank and request
        axis from the port that produced it. A validator reads metadata without
        the artifacts, so dropping these would leave every derived value
        untyped — the contract here is load-bearing, not a transcription.
        """
        pkg, metadata = package
        workflow = metadata["pipeline"]["workflow"]
        components = _onnx_components(workflow)
        policies = [
            components[name]
            for name in getattr(pkg, "policy_components", {})
            if name in components
        ]
        for component in policies:
            ports = component["ports"]
            assert ports["inputs"] or ports["outputs"]
            for contract in (*ports["inputs"].values(), *ports["outputs"].values()):
                assert contract["rank"] == len(contract["shape"])

    def test_every_declared_role_names_a_port_the_graph_exposes(self, package):
        """A role is only meaningful if it resolves in the artifact.

        This is what replaces the transcription: rather than restating the
        graph and checking the restatement against itself, the role is checked
        against the graph it claims to describe. A role naming a port the file
        does not expose binds nothing, and is caught here exactly as the
        runtime would catch it against a live session.
        """
        pkg, metadata = package
        workflow = metadata["pipeline"]["workflow"]
        components = _onnx_components(workflow)
        roled = 0
        for name, model in _artifacts(pkg, workflow).items():
            inputs, outputs = _graph_ports(model)
            for port in (components[name].get("ports") or {}).get("roles", {}):
                assert port in inputs or port in outputs
                roled += 1
        assert roled, "no component says what it does with any of its ports"

    def test_every_invocation_binds_a_port_of_the_artifact(self, package):
        """A binding to a port the graph lacks names nothing a runtime can feed."""
        pkg, metadata = package
        workflow = metadata["pipeline"]["workflow"]
        graphs = _artifacts(pkg, workflow)
        for step in _walk_steps(workflow["steps"]):
            if step.get("kind") != "invoke" or step["component"] not in graphs:
                continue
            inputs, outputs = _graph_ports(graphs[step["component"]])
            assert set(step.get("inputs", {})) <= inputs
            assert set(step.get("outputs", {})) <= outputs

    def test_every_state_pair_names_ports_of_the_artifact(self, package):
        """State is carried through ports, so both halves have to exist."""
        pkg, metadata = package
        workflow = metadata["pipeline"]["workflow"]
        graphs = _artifacts(pkg, workflow)
        for group in _groups(workflow).values():
            for component, aliases in (group.get("ports") or {}).items():
                if component not in graphs:
                    continue
                inputs, outputs = _graph_ports(graphs[component])
                for alias in aliases.values():
                    assert alias["input"] in inputs
                    assert alias["output"] in outputs

    def test_split_cache_halves_and_layers_are_stated_not_positional(self, package):
        """A layer's key and value buffers are indistinguishable once listed.

        They are the same dtype and the same shape, and a cell's label sorts
        lexicographically (``cache_10`` before ``cache_2``), so a consumer that
        paired them positionally would transpose two layers' caches without
        anything failing. When a pair carries a half, it says so, and it says
        which layer it belongs to.
        """
        _, metadata = package
        for group in _groups(metadata["pipeline"]["workflow"]).values():
            for aliases in (group.get("ports") or {}).values():
                halves = [alias.get("role") for alias in aliases.values()]
                if not any(halves):
                    continue
                assert all(half in {"key", "value", "combined"} for half in halves)
                assert all("layer" in alias for alias in aliases.values())
                keys = [alias for alias in aliases.values() if alias["role"] == "key"]
                values = [alias for alias in aliases.values() if alias["role"] == "value"]
                assert len(keys) == len(values)
                assert {alias["layer"] for alias in keys} == {
                    alias["layer"] for alias in values
                }

    def test_control_ports_of_a_fixed_capacity_cache_exist_in_the_artifact(self, package):
        """A scatter's two control vectors are rank-1 integers, so they must be named.

        Nothing distinguishes the write cursor from the valid length by shape.
        A package that scatters therefore names both against a component whose
        graph exposes both; a package that appends declares no scatter at all,
        and this assertion is vacuous for it — which is the point, since the
        same test runs over every shape.
        """
        pkg, metadata = package
        workflow = metadata["pipeline"]["workflow"]
        graphs = _artifacts(pkg, workflow)
        for group in _groups(workflow).values():
            update = group.get("update") or {}
            if update.get("kind") != "indexed_scatter":
                continue
            bound = set(group["ports"])
            assert set(update["write_indices_ports"]) == bound
            assert set(update["kv_length_ports"]) == bound
            for component in bound:
                inputs, _ = _graph_ports(graphs[component])
                assert update["write_indices_ports"][component] in inputs
                assert update["kv_length_ports"][component] in inputs

    def test_the_decode_step_is_recoverable_from_the_workflow_alone(self, package):
        """Reconstruct the decode ABI the way a runtime lowering would.

        This is the assertion that makes removing the second copy safe: if the
        sequence input, the logits output and the per-layer cache pairs can all
        be read off the workflow, then a separate block stating them again was
        never carrying information — only risk. Roles are the only thing read
        here, and every name they yield is checked against the graph, so the
        reconstruction never falls back to recognizing a spelling.
        """
        pkg, metadata = package
        workflow = metadata["pipeline"]["workflow"]
        graphs = _artifacts(pkg, workflow)
        decoders = []
        for name, component in _onnx_components(workflow).items():
            roles = (component.get("ports") or {}).get("roles", {})
            consumes = {"token_ids", "inputs_embeds"} & set(roles.values())
            owns_state = any(
                name in (group.get("ports") or {}) for group in _groups(workflow).values()
            )
            if consumes and ("logits" in roles.values() or owns_state):
                decoders.append((name, roles))
        assert decoders, "no component declares what it does with the sequence"
        for name, roles in decoders:
            inputs, outputs = _graph_ports(graphs[name])
            sequence = [
                port
                for port, role in roles.items()
                if role in {"token_ids", "inputs_embeds"} and port in inputs
            ]
            assert len(sequence) == 1
            assert all(port in outputs for port, role in roles.items() if role == "logits")
            pairs = [
                alias
                for group in _groups(workflow).values()
                for alias in (group.get("ports") or {}).get(name, {}).values()
            ]
            assert all(
                alias["input"] in inputs and alias["output"] in outputs for alias in pairs
            )


def _cell_order(alias: dict[str, Any], label: str) -> tuple[Any, ...]:
    """Canonical order of a state pair: its layer, then its half."""
    return (alias.get("layer", 0), alias.get("role", ""), label)


def _resolve_decode_abi(workflow: dict[str, Any], component: str) -> dict[str, Any]:
    """Bind the decode ABI the way a consumer with no port vocabulary must.

    Nothing here looks at how a port is spelled. The sequence and logits ports
    come from declared roles, and every cache port comes from a state-service
    alias, ordered by the layer and half the alias states. A consumer that took
    any other route would be recognizing names — which is the failure mode this
    resolution exists to make impossible.
    """
    declaration = workflow["components"][component]
    roles = (declaration.get("ports") or {}).get("roles", {})
    caches: list[tuple[str, str]] = []
    write_indices: str | None = None
    kv_length: str | None = None
    for group in _groups(workflow).values():
        aliases = (group.get("ports") or {}).get(component) or {}
        for label in sorted(aliases, key=lambda label: _cell_order(aliases[label], label)):
            caches.append((aliases[label]["input"], aliases[label]["output"]))
        update = group.get("update") or {}
        if update.get("kind") == "indexed_scatter":
            write_indices = update["write_indices_ports"][component]
            kv_length = update["kv_length_ports"][component]
    return {
        "token_ids": [port for port, role in roles.items() if role == "token_ids"],
        "inputs_embeds": [port for port, role in roles.items() if role == "inputs_embeds"],
        "logits": [port for port, role in roles.items() if role == "logits"],
        "cache_inputs": [pair[0] for pair in caches],
        "cache_outputs": [pair[1] for pair in caches],
        "write_indices": write_indices,
        "kv_sequence_length": kv_length,
    }


def _rename_component_ports(
    workflow: dict[str, Any], component: str, renames: dict[str, str]
) -> None:
    """Rewrite every place the workflow names a port of *component*.

    The four places are the whole surface: the role table, the invocations that
    bind the ports, the state pairs that carry buffers through them, and the
    scatter's two control ports.
    """
    declaration = workflow["components"][component]
    ports = declaration.get("ports") or {}
    if "roles" in ports:
        ports["roles"] = {renames.get(k, k): v for k, v in ports["roles"].items()}
    for side in ("inputs", "outputs"):
        if ports.get(side):
            ports[side] = {renames.get(k, k): v for k, v in ports[side].items()}
    for step in _walk_steps(workflow["steps"]):
        if step.get("kind") != "invoke" or step.get("component") != component:
            continue
        for side in ("inputs", "outputs"):
            if step.get(side):
                step[side] = {renames.get(k, k): v for k, v in step[side].items()}
    for group in _groups(workflow).values():
        for alias in ((group.get("ports") or {}).get(component) or {}).values():
            alias["input"] = renames.get(alias["input"], alias["input"])
            alias["output"] = renames.get(alias["output"], alias["output"])
        update = group.get("update") or {}
        for key in ("write_indices_ports", "kv_length_ports"):
            bindings = update.get(key) or {}
            if component in bindings:
                bindings[component] = renames.get(bindings[component], bindings[component])


class TestRolesAloneCarryTheDecodeAbi:
    """Omitting the port contracts must not push a consumer back to guessing.

    A component that ships an artifact declares no port contracts, so the only
    thing left saying what a port *means* is its role. The risk that creates is
    specific: a consumer that cannot find a role does not fail loudly, it falls
    back to matching the spelling ``input_ids`` — and then a package that
    spells it differently binds the wrong tensor with the right shape.

    These tests hold the producer to the side of that contract it owns. Every
    export shape must declare the roles a consumer needs, and the whole decode
    ABI must survive renaming every port to an opaque label: if any part of the
    binding still resolved after that, it was resolving by name.
    """

    @staticmethod
    def _roled(workflow: dict[str, Any]) -> list[str]:
        """Every component that says what any of its ports means."""
        return [
            name
            for name, component in _onnx_components(workflow).items()
            if ((component.get("ports") or {}).get("roles") or {})
        ]

    @staticmethod
    def _decoders(workflow: dict[str, Any]) -> list[str]:
        """Components that drive a decode step.

        A decoder consumes the sequence and either produces logits or owns
        cache state. An embedding component consumes tokens too, but it is not
        the step a runtime specializes.
        """
        decoders = []
        for name, component in _onnx_components(workflow).items():
            roles = set(((component.get("ports") or {}).get("roles") or {}).values())
            owns_state = any(
                name in (group.get("ports") or {}) for group in _groups(workflow).values()
            )
            if {"token_ids", "inputs_embeds"} & roles and ("logits" in roles or owns_state):
                decoders.append(name)
        return decoders

    def test_a_shipped_artifact_declares_roles_and_no_contracts(self, package):
        """The omission and the role are one decision, not two.

        Dropping the contracts is only safe because the role is there; a
        component with neither would be a graph a consumer can only guess at.
        """
        pkg, metadata = package
        workflow = metadata["pipeline"]["workflow"]
        for name in _artifacts(pkg, workflow):
            if name not in pkg:
                continue
            ports = workflow["components"][name].get("ports") or {}
            assert not ports.get("inputs") and not ports.get("outputs")
        assert self._decoders(workflow), "no exported graph says what consumes the sequence"

    def test_the_sequence_port_is_declared_not_spelled(self, package):
        """``input_ids`` and ``position_ids`` are both rank-2 int64.

        Nothing in a graph distinguishes them, so the producer states which one
        is the autoregressive sequence rather than leaving a runtime to infer
        it from a name it has no right to assume. A decode step consumes
        exactly one sequence — tokens or embeddings, never both.
        """
        _, metadata = package
        workflow = metadata["pipeline"]["workflow"]
        for name in self._decoders(workflow):
            abi = _resolve_decode_abi(workflow, name)
            assert len(abi["token_ids"]) + len(abi["inputs_embeds"]) == 1

    def test_the_whole_abi_survives_renaming_every_port(self, package):
        """Rename the ports to opaque labels; the ABI must resolve identically.

        This is the assertion with teeth. The renamed package contains no port
        called ``input_ids``, ``logits`` or ``key_cache.0``, so a resolution
        that still returns the right ports cannot have been reading names. The
        answers are compared as positions, because the names deliberately no
        longer match.
        """
        pkg, metadata = package
        workflow = copy.deepcopy(metadata)["pipeline"]["workflow"]
        roled = self._roled(workflow)
        assert roled
        original = {name: _resolve_decode_abi(workflow, name) for name in roled}
        renames = {}
        for index, name in enumerate(sorted(original)):
            inputs, outputs = _graph_ports(_artifacts(pkg, workflow)[name])
            mapping = {
                port: f"c{index}.p{position}"
                for position, port in enumerate(sorted(inputs | outputs))
            }
            renames[name] = mapping
            _rename_component_ports(workflow, name, mapping)
        recognizable = {"input_ids", "inputs_embeds", "logits"}
        for name, before in original.items():
            after = _resolve_decode_abi(workflow, name)
            mapping = renames[name]
            for key, value in before.items():
                expected = (
                    [mapping[port] for port in value]
                    if isinstance(value, list)
                    else (mapping[value] if value is not None else None)
                )
                assert after[key] == expected
            # Nothing recognizable is left to have matched on.
            resolved = (*after["token_ids"], *after["inputs_embeds"], *after["logits"])
            assert not recognizable & set(resolved)

    def test_deleting_the_roles_is_what_breaks_it(self, package):
        """The mutation that proves the role is doing the work.

        If the sequence port were still recoverable with the role table gone,
        then something else — a position, a shape, a spelling — was carrying
        the fact, and these tests would be asserting nothing.
        """
        _, metadata = package
        workflow = copy.deepcopy(metadata)["pipeline"]["workflow"]
        roled = self._roled(workflow)
        assert roled
        for name in roled:
            workflow["components"][name]["ports"].pop("roles")
        for name in roled:
            abi = _resolve_decode_abi(workflow, name)
            assert abi["token_ids"] == []
            assert abi["inputs_embeds"] == []
            assert abi["logits"] == []

    def test_state_pairs_still_bind_without_a_role_table(self, package):
        """The cache half of the ABI is the state service's, not the role table's.

        Deleting the roles must not disturb it: a runtime that resolved caches
        through the roles would break on any component whose cache ports carry
        no role, and the two halves are deliberately independent.
        """
        _, metadata = package
        workflow = copy.deepcopy(metadata)["pipeline"]["workflow"]
        roled = self._roled(workflow)
        before = {name: _resolve_decode_abi(workflow, name) for name in roled}
        for name in roled:
            workflow["components"][name]["ports"].pop("roles")
        for name in roled:
            after = _resolve_decode_abi(workflow, name)
            for key in (
                "cache_inputs",
                "cache_outputs",
                "write_indices",
                "kv_sequence_length",
            ):
                assert after[key] == before[name][key]

    def test_a_scatter_bound_component_is_always_resolvable_as_a_decoder(self, package):
        """Naming a component in an ``indexed_scatter`` group obliges it to say so.

        The write cursor and valid length exist for one consumer: the driver
        that writes into a preallocated buffer at an index. That driver binds
        its ports from the resolved decode ABI, and every field of that ABI is
        found by role — so a component handed those two control ports while
        declaring no sequence role cannot be resolved as a decoder at all.

        Nothing rejects that combination for us. Identifying the decoder
        requires a sequence role, so a component that omits one is invisible to
        the very check that would have caught it, and the package validates,
        loads, and quietly falls back to inferring ports from shapes. The
        producer is the only place the contradiction is visible, which is why
        it is asserted here.
        """
        _, metadata = package
        workflow = metadata["pipeline"]["workflow"]
        for name in _scatter_bound_components(workflow):
            component = workflow["components"][name]
            assert _declares_a_sequence_role(component), (
                f"{name} is handed a fixed-capacity write cursor but declares no "
                "sequence role, so no decode ABI can be resolved for it"
            )
            assert _resolve_decode_abi(workflow, name)["write_indices"]


def _deep_dynamic() -> dict[str, Any]:
    """An appending cache with enough layers that labels sort out of order."""
    config = _text_config(num_hidden_layers=DEEP_LAYERS)
    pkg = CausalLMTask().build(registry.get("qwen2")(config), config)
    return build_decoder_workflow_metadata(pkg, config)


def _deep_static_cache() -> dict[str, Any]:
    """The same depth, scattered into fixed-capacity buffers."""
    config = _text_config(num_hidden_layers=DEEP_LAYERS)
    task = CausalLMTask(static_cache=True, max_seq_len=CAPACITY)
    pkg = task.build(registry.get("qwen2")(config), config)
    return build_decoder_workflow_metadata(pkg, config)


def _deep_heterogeneous() -> dict[str, Any]:
    """A deep hybrid whose cache-owning layers are a non-contiguous subset.

    Alternating the layer types makes the full-attention group own layers
    1, 3, 5, ... only, so a cell's position in its group is never its layer.
    """
    config = _text_config(
        num_hidden_layers=DEEP_LAYERS,
        sliding_window=8,
        layer_types=["sliding_attention", "full_attention"] * (DEEP_LAYERS // 2),
    )
    pkg = CausalLMTask().build(registry.get("qwen2")(config), config)
    return build_decoder_workflow_metadata(pkg, config)


_DEEP_PACKAGES = {
    "dynamic": _deep_dynamic,
    "static_cache": _deep_static_cache,
    "heterogeneous": _deep_heterogeneous,
}


@pytest.fixture(scope="module")
def deep_built() -> dict[str, dict[str, Any]]:
    return {name: build() for name, build in _DEEP_PACKAGES.items()}


@pytest.fixture(params=sorted(_DEEP_PACKAGES), scope="module")
def deep_package(request, deep_built):
    return deep_built[request.param]


def _roled_aliases(metadata: dict[str, Any]) -> list[dict[str, dict[str, Any]]]:
    """Every per-component alias map that carries key/value halves."""
    maps = []
    for group in _groups(metadata["pipeline"]["workflow"]).values():
        for aliases in (group.get("ports") or {}).values():
            if any(alias.get("role") for alias in aliases.values()):
                maps.append(aliases)
    return maps


class TestDeepDecodersDeclareLayersRatherThanPositions:
    """The layer annotation only earns its place above nine layers.

    Every other package in this file has two layers, and with two layers a
    cell's label sorts the same way whichever rule is used: lexicographic,
    numeric, or insertion order all agree. So do the alternatives a producer
    could accidentally implement — ``layer`` taken from the enumeration index
    rather than parsed from the port name is indistinguishable from the
    correct value until some layer's cells outnumber a single digit.

    Real decoders have twenty to eighty layers, so that region is the normal
    case in production and the unreachable case in this suite. These tests put
    a package there. They are written to fail loudly if the annotation ever
    degrades into a restatement of position, because the failure it prevents
    is silent: two transposed caches have identical shapes and identical
    dtypes, and the only symptom is that generated text is subtly wrong.
    """

    def test_labels_really_do_sort_out_of_order_at_this_depth(self, deep_package):
        # Guards the premise of every other test in this class: if the labels
        # happened to sort numerically, the rest would prove nothing.
        for aliases in _roled_aliases(deep_package):
            by_label = [aliases[cell]["layer"] for cell in sorted(aliases)]
            assert by_label != sorted(by_label), (
                "labels sort into layer order, so this package cannot "
                "distinguish a declared layer from a positional one"
            )

    def test_the_declared_layer_is_the_one_the_port_name_states(self, deep_package):
        """The annotation restates the exporter's own port name, not an index."""
        for aliases in _roled_aliases(deep_package):
            for alias in aliases.values():
                stated = re.search(
                    r"\.(\d+)\.(?:key|value)$|(?:key|value)_cache\.(\d+)$", alias["input"]
                )
                assert stated is not None, alias["input"]
                assert alias["layer"] == int(stated.group(1) or stated.group(2))

    def test_ordering_by_the_declared_layer_recovers_the_buffer_lists(self, deep_package):
        """Sorting by (layer, half) is what a consumer does; it must be numeric.

        This mirrors how the runtime collects a group's ports, so the assertion
        fails here rather than as transposed caches at inference time.
        """
        for aliases in _roled_aliases(deep_package):
            ordered = sorted(
                aliases.values(), key=lambda alias: (alias["layer"], alias["role"])
            )
            layers = [alias["layer"] for alias in ordered]
            assert layers == sorted(layers)
            keys = [alias["input"] for alias in ordered if alias["role"] == "key"]
            values = [alias["input"] for alias in ordered if alias["role"] == "value"]
            assert len(keys) == len(values)
            # Each layer contributes exactly one key and one value, in step.
            assert [alias["layer"] for alias in ordered if alias["role"] == "key"] == [
                alias["layer"] for alias in ordered if alias["role"] == "value"
            ]

    def test_a_cells_position_in_its_group_is_not_its_layer(self, deep_built):
        """Each group of a hybrid owns an alternating half of the layers.

        This is the case that separates a declared layer from a positional one
        even for a producer that sorts numerically: the sliding group owns
        layers 0, 2, 4, ... and the full-attention group owns 1, 3, 5, ..., so
        a cell's position within its own group is never its layer.
        """
        aliases_by_group = _roled_aliases(deep_built["heterogeneous"])
        assert len(aliases_by_group) == 2, "expected one group per attention type"
        owned = [
            frozenset(alias["layer"] for alias in aliases.values() if alias["role"] == "key")
            for aliases in aliases_by_group
        ]
        evens = frozenset(index for index in range(DEEP_LAYERS) if index % 2 == 0)
        odds = frozenset(index for index in range(DEEP_LAYERS) if index % 2 == 1)
        assert set(owned) == {evens, odds}
        # Neither group's layers are its own positions, which is the property a
        # positional annotation would have satisfied by construction.
        for layers in owned:
            assert layers != frozenset(range(len(layers)))


def _fixture_packages() -> list[str]:
    return sorted(
        os.path.dirname(path)
        for path in glob.glob(os.path.join(FIXTURE_ROOT, "*", "inference_metadata.yaml"))
    )


_MULTI_REQUEST_COMPONENTS = {
    "adapter": {"overlay"},
    "decoder": {
        "model",
        "token_sampler",
        "termination",
        "token_state_update",
        "last_token_logits",
        "decoder_state_initializer",
        "decoder_step_update",
        "cache_length_update",
        "termination_batch_initializer",
        "token_to_slot",
        "generated_length_update",
    },
    "diffusion": {
        "text_encoder",
        "denoiser",
        "vae_decoder",
        "image_output_clamp",
        "solver_step",
        "continue_predicate",
        "model_input_scale",
        "diffusion_schedule",
        "diffusion_timesteps",
        "schedule_lookup",
        "tensor_scale",
        "initial_state_scale",
    },
    "diffusion_guided": {
        "text_encoder",
        "denoiser",
        "vae_decoder",
        "image_output_clamp",
        "solver_step",
        "continue_predicate",
        "diffusion_schedule",
        "diffusion_timesteps",
        "schedule_lookup",
        "tensor_scale",
        "decoder_input_scale",
        "history_initializer",
        "guidance_combine",
        "latent_row_shape",
        "latent_noise",
    },
    "static_cache": {
        "model",
        "token_sampler",
        "termination",
        "token_state_update",
        "last_token_logits",
        "decoder_state_initializer",
        "decoder_step_update",
        "cache_length_update",
        "termination_batch_initializer",
        "token_to_slot",
        "generated_length_update",
    },
    "tts": {
        "talker",
        "code_predictor",
        "embedding",
        "talker_step_embedder",
        "talker_prefill_embedder",
        "code_predictor_prefill",
        "code_predictor_step_embedder",
        "code_predictor_indices",
        "talker_text_step",
        "codec",
        "last_token_logits",
        "setup_talker_sampler",
        "setup_predictor_sampler",
        "talker_sampler",
        "predictor_prefill_sampler",
        "predictor_body_sampler",
        "continue_predicate",
        "tts_state_initializer",
        "token_to_slot",
        "code_frame_update",
        "code_history_append",
        "cache_length_update",
        "talker_state_initializer",
        "predictor_state_initializer",
        "talker_step_update",
        "predictor_step_update",
        "codec_layout",
    },
    "video": {
        "transformer",
        "vae_decoder",
        "model_input",
        "solver_step",
        "continue_predicate",
        "video_latent_init",
        "schedule_history_append",
        "video_latent_permute",
        "video_latent_unscale",
        "video_decode_chunks",
        "video_decode_chunk",
        "video_conv_cache_init",
        "diffusion_schedule",
        "diffusion_timesteps",
        "schedule_lookup",
    },
}


@pytest.mark.parametrize(
    "directory", _fixture_packages(), ids=lambda path: os.path.basename(path)
)
class TestCheckedInPackagesShareTheShape:
    """The same contract, asserted against every package checked into the tree.

    The built packages above cover the decoder shapes this producer can
    construct in a unit test. The fixtures cover the ones it cannot — audio
    codecs, diffusion, video, speculative decoding, text-to-speech — and they
    are the exact bytes the runtime conformance suite executes, so a shape that
    drifts here drifts in something already proven to run.
    """

    @staticmethod
    def _metadata(directory: str) -> dict[str, Any]:
        with open(
            os.path.join(directory, "inference_metadata.yaml"), encoding="utf-8"
        ) as handle:
            return yaml.safe_load(handle)

    def test_no_fixture_contains_retired_batching_declarations(self, directory):
        serialized = yaml.safe_dump(self._metadata(directory))
        assert "batch_invariance:" not in serialized
        assert "continuous_batching:" not in serialized

    def test_fixture_validates_against_current_onnx_genai_schema(self, directory):
        jsonschema.validate(self._metadata(directory), ONNX_GENAI_SCHEMA)

    def test_fixture_round_trips_through_yaml_and_schema(self, directory):
        metadata = self._metadata(directory)
        round_tripped = yaml.safe_load(yaml.safe_dump(metadata, sort_keys=False))
        assert round_tripped == metadata
        jsonschema.validate(round_tripped, ONNX_GENAI_SCHEMA)

    def test_multi_request_components_declare_capacity(self, directory):
        package = os.path.basename(directory)
        expected = _MULTI_REQUEST_COMPONENTS.get(package)
        if expected is None:
            return
        components = self._metadata(directory)["pipeline"]["workflow"]["components"]
        assert {
            name for name, component in components.items() if "batch_capacity" in component
        } == expected

    def test_unproven_encoder_capacity_remains_absent(self, directory):
        if os.path.basename(directory) not in {
            "esm2_protein_embeddings",
            "protbert_protein_embeddings",
        }:
            return
        encoder = self._metadata(directory)["pipeline"]["workflow"]["components"]["encoder"]
        assert "batch_capacity" not in encoder

    def test_hierarchical_audio_preserves_its_internal_row_expansion(self, directory):
        if os.path.basename(directory) != "hierarchical_audio":
            return
        components = self._metadata(directory)["pipeline"]["workflow"]["components"]
        for name in ("global_initializer", "global_step_update"):
            contracts = components[name]["ports"]
            for contract in (*contracts["inputs"].values(), *contracts["outputs"].values()):
                shape = contract.get("shape") or []
                if shape and shape[0] == "batch":
                    assert contract["batch_layout"] == {
                        "kind": "request_expanded",
                        "axis": 0,
                        "factor": 2,
                    }

    def test_fixture_matches_regenerated_metadata(
        self, directory, materialized_workflow_packages
    ):
        relative = os.path.relpath(directory, FIXTURE_ROOT)
        regenerated = os.path.join(
            materialized_workflow_packages, relative, "inference_metadata.yaml"
        )
        with open(regenerated, encoding="utf-8") as handle:
            assert self._metadata(directory) == yaml.safe_load(handle)

    @staticmethod
    def _artifact_root(directory: str, materialized: str) -> str:
        """Where this package's graphs live once they have been generated."""
        return os.path.join(materialized, os.path.basename(directory))

    @classmethod
    def _ports(
        cls, directory: str, workflow: dict[str, Any], materialized: str
    ) -> dict[str, tuple[set, set]]:
        """Read the ports each graph really exposes.

        The metadata under review is the committed one; the graph it describes
        is generated, because it is a deterministic function of the producer
        and committing it would store bytes no reviewer reads. Resolving the
        committed declaration against the generated graph is what makes this an
        assertion about the artifact rather than about a copy of itself.
        """
        root = cls._artifact_root(directory, materialized)
        ports: dict[str, tuple[set, set]] = {}
        for name, component in _onnx_components(workflow).items():
            artifact = os.path.join(root, component["implementation"]["artifact"])
            assert os.path.exists(artifact), (
                f"{os.path.basename(directory)} declares component {name!r} backed by "
                f"{component['implementation']['artifact']!r}, but the generator emitted "
                f"no such file. Two of the assertions below skip components they cannot "
                f"resolve, so a missing graph would quietly turn them into no-ops."
            )
            ports[name] = _graph_ports(ir.load(artifact))
        return ports

    def test_describes_itself_only_through_the_workflow(self, directory):
        """One serialized representation of the executable ABI, not two.

        The runtime resolves the workflow first and never consults a second
        declaration when one is present, so a surviving ``model.io`` or
        ``pipeline.models`` would not be a redundant copy that merely risks
        drifting: it would be inert, and nothing would say so. ONNX GenAI
        rejects a package that carries both rather than silently picking one.
        """
        metadata = self._metadata(directory)
        assert "workflow" in metadata["pipeline"]
        assert "model" not in metadata or "io" not in metadata["model"], (
            "model.io declares the same executable ABI as the workflow; the "
            "workflow is canonical, so this one would be discarded at load"
        )
        assert "models" not in metadata["pipeline"], (
            "pipeline.models is the legacy composite ABI, superseded by "
            "pipeline.workflow's components and invoke bindings"
        )

    def test_no_component_transcribes_the_ports_of_its_own_artifact(
        self, directory, materialized_workflow_packages
    ):
        """Whatever a component declares must not be a restatement of its graph.

        A policy graph declares contracts because they type the workflow's
        dataflow, and a workflow value has no other source for its dtype and
        request axis. What no component may do is declare a port the artifact
        does not have, or declare only some of them: either turns the
        declaration into a partial second truth that drifts silently.
        """
        workflow = self._metadata(directory)["pipeline"]["workflow"]
        graphs = self._ports(directory, workflow, materialized_workflow_packages)
        assert graphs, "a fixture that ships no artifact proves nothing"
        for name, component in _onnx_components(workflow).items():
            ports = component.get("ports") or {}
            if not (ports.get("inputs") or ports.get("outputs")):
                continue
            inputs, outputs = graphs[name]
            assert set(ports.get("inputs", {})) == inputs
            assert set(ports.get("outputs", {})) == outputs

    def test_every_binding_and_state_pair_resolves_in_the_artifact(
        self, directory, materialized_workflow_packages
    ):
        workflow = self._metadata(directory)["pipeline"]["workflow"]
        graphs = self._ports(directory, workflow, materialized_workflow_packages)
        for step in _walk_steps(workflow["steps"]):
            if step.get("kind") != "invoke" or step["component"] not in graphs:
                continue
            inputs, outputs = graphs[step["component"]]
            assert set(step.get("inputs", {})) <= inputs
            assert set(step.get("outputs", {})) <= outputs
        for group in _groups(workflow).values():
            for component, aliases in (group.get("ports") or {}).items():
                if component not in graphs:
                    continue
                inputs, outputs = graphs[component]
                for alias in aliases.values():
                    assert alias["input"] in inputs
                    assert alias["output"] in outputs

    def test_every_declared_role_resolves_in_the_artifact(
        self, directory, materialized_workflow_packages
    ):
        workflow = self._metadata(directory)["pipeline"]["workflow"]
        graphs = self._ports(directory, workflow, materialized_workflow_packages)
        for name, component in _onnx_components(workflow).items():
            if name not in graphs:
                continue
            inputs, outputs = graphs[name]
            for port in (component.get("ports") or {}).get("roles", {}):
                assert port in inputs or port in outputs

    def test_a_scatter_bound_component_is_always_resolvable_as_a_decoder(self, directory):
        """The shipped bytes carry the same obligation the built packages do.

        A fixture is edited and regenerated by hand far more often than the
        producer is changed, so this is the copy of the rule that catches a
        package whose sequence role was dropped on the way to disk.
        """
        workflow = self._metadata(directory)["pipeline"]["workflow"]
        for name in _scatter_bound_components(workflow):
            assert _declares_a_sequence_role(workflow["components"][name]), (
                f"{name} is handed a fixed-capacity write cursor but declares no "
                "sequence role, so no decode ABI can be resolved for it"
            )

    def test_no_input_leaves_its_admission_requirement_to_the_reader(self, directory):
        """Admission is a fact of the package, not a default the consumer picks.

        A runtime rejects a request that omits an input the package holds
        required, before any component runs. A declaration with no ``required``
        key does not say it is optional -- it says nothing, and the schema that
        reads it fills in ``true``, which is the opposite of what omission means
        to a producer whose workflow can supply the value itself.
        """
        workflow = self._metadata(directory)["pipeline"]["workflow"]
        for name, declaration in (workflow.get("inputs") or {}).items():
            assert "required" in declaration, (
                f"{name} does not publish whether a caller must attach it, so "
                "every reader has to guess, and the guess rejects requests this "
                "package can serve"
            )

    def test_a_required_input_is_one_the_package_cannot_supply_itself(self, directory):
        """No caller is obliged to attach a value the workflow already has.

        Three declarations mean "this runs without the caller": a ``default``, a
        package-owned ``literal`` source, and a ``present_as`` symbol the steps
        branch on. Any of them alongside ``required: true`` is two contracts
        that cannot both hold, and the runtime honours the one that turns an
        optional branch input into a universal obligation.
        """
        workflow = self._metadata(directory)["pipeline"]["workflow"]
        references = published_value_references(workflow)
        for name, declaration in (workflow.get("inputs") or {}).items():
            present_as = declaration.get("present_as")
            if present_as is not None:
                assert present_as in references, (
                    f"{name} advertises the presence symbol {present_as!r} that no "
                    "step reads, so the workflow never handles it being absent"
                )
            if (declaration.get("source") or {}).get("kind") == "literal":
                assert "default" in declaration, (
                    f"{name} is sourced from a package literal, which is bound from "
                    "its own default and nothing else, but carries no default"
                )
            if not declaration.get("required"):
                assert "default" in declaration or present_as is not None, (
                    f"{name} is published as optional but the package carries no "
                    "default and no presence gate for it, so omitting it has no "
                    "defined behaviour"
                )
                continue
            assert "default" not in declaration
            assert (declaration.get("source") or {}).get("kind") != "literal"
            assert present_as is None
