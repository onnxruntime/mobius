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

import glob
import os
import re
from typing import Any

import onnx_ir as ir
import pytest
import yaml

from mobius import registry
from mobius._configs import ArchitectureConfig
from mobius._optimizations import optimize_model
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


class TestOneSerializedContract:
    """Facts that hold for every package shape, asserted through one code path."""

    def test_the_workflow_is_where_a_package_describes_itself(self, package):
        _, metadata = package
        assert "workflow" in metadata["pipeline"]

    def test_no_package_states_its_graph_abi_a_second_time(self, package):
        """The point of one representation is that there is no other one.

        ``model`` may still carry package-wide geometry, but the moment it
        carries an ``io`` block the package has two writable answers to "what
        does the decode step look like", and a reader of either never learns the
        other exists.
        """
        _, metadata = package
        assert "io" not in (metadata.get("model") or {})

    def test_every_graph_declares_exactly_the_ports_it_has(self, package):
        """Declared ports are the graph's ports — not a subset, not a superset.

        A subset lets a runtime silently fall back to opening the artifact, and
        a superset is a promise the graph does not keep. Either way the
        declaration stops being usable as the single source of truth.
        """
        pkg, metadata = package
        components = _onnx_components(metadata["pipeline"]["workflow"])
        graphs = {name: pkg[name] for name in components if name in pkg}
        assert graphs, "a package with no ONNX component describes nothing"
        for name, model in graphs.items():
            ports = components[name]["ports"]
            assert set(ports["inputs"]) == {str(v.name) for v in model.graph.inputs}
            assert set(ports["outputs"]) == {str(v.name) for v in model.graph.outputs}

    def test_declared_contracts_match_the_graph_dtype_and_rank(self, package):
        """A contract that disagrees with its graph would mis-size every buffer."""
        pkg, metadata = package
        dtypes = {
            ir.DataType.FLOAT: "float32",
            ir.DataType.FLOAT16: "float16",
            ir.DataType.BFLOAT16: "bfloat16",
            ir.DataType.INT64: "int64",
            ir.DataType.INT32: "int32",
            ir.DataType.BOOL: "bool",
            ir.DataType.FLOAT8E4M3FN: "float8_e4m3fn",
        }
        components = _onnx_components(metadata["pipeline"]["workflow"])
        for name in (name for name in components if name in pkg):
            model = pkg[name]
            ports = components[name]["ports"]
            declared = {**ports["inputs"], **ports["outputs"]}
            for value in (*model.graph.inputs, *model.graph.outputs):
                contract = declared[str(value.name)]
                assert contract["rank"] == len(value.shape)
                if value.dtype in dtypes:
                    assert contract["dtype"] == dtypes[value.dtype]

    def test_every_invocation_binds_a_declared_port(self, package):
        """A binding to an undeclared port names nothing a consumer can resolve."""
        _, metadata = package
        workflow = metadata["pipeline"]["workflow"]
        components = workflow["components"]
        for step in _walk_steps(workflow["steps"]):
            if step.get("kind") != "invoke":
                continue
            ports = components[step["component"]]["ports"]
            assert set(step.get("inputs", {})) <= set(ports["inputs"])
            assert set(step.get("outputs", {})) <= set(ports["outputs"])

    def test_every_state_pair_names_declared_ports(self, package):
        """State is carried through ports, so both halves have to be declared."""
        _, metadata = package
        workflow = metadata["pipeline"]["workflow"]
        components = workflow["components"]
        for group in _groups(workflow).values():
            for component, aliases in (group.get("ports") or {}).items():
                ports = components[component]["ports"]
                for alias in aliases.values():
                    assert alias["input"] in ports["inputs"]
                    assert alias["output"] in ports["outputs"]

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

    def test_control_ports_of_a_fixed_capacity_cache_are_declared_ports(self, package):
        """A scatter's two control vectors are rank-1 integers, so they must be named.

        Nothing distinguishes the write cursor from the valid length by shape.
        A package that scatters therefore names both against a component that
        declares both; a package that appends declares no scatter at all, and
        this assertion is vacuous for it — which is the point, since the same
        test runs over every shape.
        """
        _, metadata = package
        workflow = metadata["pipeline"]["workflow"]
        for group in _groups(workflow).values():
            update = group.get("update") or {}
            if update.get("kind") != "indexed_scatter":
                continue
            bound = set(group["ports"])
            assert set(update["write_indices_ports"]) == bound
            assert set(update["kv_length_ports"]) == bound
            for component in bound:
                declared = workflow["components"][component]["ports"]["inputs"]
                assert update["write_indices_ports"][component] in declared
                assert update["kv_length_ports"][component] in declared

    def test_the_decode_step_is_recoverable_from_the_workflow_alone(self, package):
        """Reconstruct the decode ABI the way a runtime lowering would.

        This is the assertion that makes removing the second copy safe: if the
        sequence input, the logits output and the per-layer cache pairs can all
        be read off the workflow, then a separate block stating them again was
        never carrying information — only risk.
        """
        _, metadata = package
        workflow = metadata["pipeline"]["workflow"]
        decoders = []
        for name, component in _onnx_components(workflow).items():
            roles = component["ports"].get("roles", {})
            consumes = {"token_ids", "inputs_embeds"} & set(roles.values())
            owns_state = any(
                name in (group.get("ports") or {}) for group in _groups(workflow).values()
            )
            if consumes and ("logits" in roles.values() or owns_state):
                decoders.append((name, component, roles))
        assert decoders, "no component declares what it does with the sequence"
        for name, component, roles in decoders:
            inputs = component["ports"]["inputs"]
            outputs = component["ports"]["outputs"]
            sequence = [
                port
                for port, role in roles.items()
                if role in {"token_ids", "inputs_embeds"} and port in inputs
            ]
            assert len(sequence) == 1
            assert [port for port, role in roles.items() if role == "logits"] == [
                port for port in outputs if roles.get(port) == "logits"
            ]
            pairs = [
                alias
                for group in _groups(workflow).values()
                for alias in (group.get("ports") or {}).get(name, {}).values()
            ]
            assert all(
                alias["input"] in inputs and alias["output"] in outputs for alias in pairs
            )


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

    def test_describes_itself_only_through_the_workflow(self, directory):
        metadata = self._metadata(directory)
        assert "workflow" in metadata["pipeline"]
        assert "io" not in (metadata.get("model") or {})

    def test_every_onnx_component_declares_its_ports(self, directory):
        workflow = self._metadata(directory)["pipeline"]["workflow"]
        for component in _onnx_components(workflow).values():
            ports = component["ports"]
            assert ports["inputs"] or ports["outputs"]
            for contract in (*ports["inputs"].values(), *ports["outputs"].values()):
                assert contract["rank"] == len(contract["shape"])

    def test_every_binding_and_state_pair_resolves(self, directory):
        workflow = self._metadata(directory)["pipeline"]["workflow"]
        components = workflow["components"]
        for step in _walk_steps(workflow["steps"]):
            if step.get("kind") != "invoke":
                continue
            ports = components[step["component"]]["ports"]
            assert set(step.get("inputs", {})) <= set(ports["inputs"])
            assert set(step.get("outputs", {})) <= set(ports["outputs"])
        for group in _groups(workflow).values():
            for component, aliases in (group.get("ports") or {}).items():
                ports = components[component]["ports"]
                for alias in aliases.values():
                    assert alias["input"] in ports["inputs"]
                    assert alias["output"] in ports["outputs"]
