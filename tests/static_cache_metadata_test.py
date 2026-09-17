# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Workflow metadata for static (fixed-capacity, indexed) KV caches.

A static-cache export does not append to a growing tensor: it scatters each
step's keys and values into a preallocated buffer at a per-row cursor. The
published metadata therefore has to describe three things a dynamic cache
never needs — where the write lands (``write_indices``), how much of the
buffer is valid afterwards (``nonpad_kv_seqlen``), and how large the buffer is
(``package.cache_capacity``) — and it has to say that the cache tensors are
loop *invariant* rather than growing.

These tests pin that contract against real exported packages, not synthetic
graphs, so a change to the exporter's port names or scatter axis fails here
rather than at runtime.

Opt in to real TensorRT execution with ``MOBIUS_TEST_TENSORRT=1`` and set
``TENSORRT_ROOT`` to a TensorRT 11.3+ SDK containing ``bin/trtexec``. The
``tensorrt_static_cache_runtime`` test requires CUDA, TensorRT Python bindings,
cuda-python and Transformers. It builds a tiny seeded model without downloads;
once enabled, missing prerequisites or engine failures are test failures.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import onnx_ir as ir
import pytest
import yaml
from onnxscript import GraphBuilder

from mobius import registry
from mobius._build_context import build_context
from mobius._configs import ArchitectureConfig
from mobius._constants import (
    OPSET_VERSION,
    STATIC_CACHE_KV_SEQUENCE_LENGTH,
    STATIC_CACHE_SEQUENCE_AXIS,
    STATIC_CACHE_WRITE_INDICES,
)
from mobius._execution_providers import get_ep
from mobius._flags import override_flags
from mobius._testing.ort_inference import OnnxModelSession
from mobius.components import create_static_cache_attention_bias
from mobius.components._attention import StaticCacheState, _apply_attention
from mobius.integrations.onnx_genai.workflow_metadata import (
    build_decoder_workflow_metadata,
    write_decoder_workflow_metadata,
)
from mobius.tasks import CausalLMTask

CAPACITY = 128


def _text_config(**overrides) -> ArchitectureConfig:
    params = {
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


def _static_package(*, ep_name="default", **overrides):
    config = _text_config(**overrides)
    with build_context(get_ep(ep_name), dtype=config.dtype):
        module = registry.get("qwen2")(config)
        task = CausalLMTask(static_cache=True, max_seq_len=CAPACITY)
        return task.build(module, config), config


@pytest.mark.parametrize(
    "ep_name,axis,layout", [("default", 1, "bsh"), ("tensorrt", 2, "bnsh")]
)
def test_static_cache_metadata_layout(ep_name, axis, layout, tmp_path):
    package, config = _static_package(ep_name=ep_name)
    metadata = build_decoder_workflow_metadata(package, config)
    _, group = _scatter_group(metadata)
    assert group["sequence_axis"] == axis
    assert group["layout"] == layout
    assert _static_cache_abi(metadata)["capacity"] == CAPACITY
    path = write_decoder_workflow_metadata(package, str(tmp_path), config)
    saved = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    _, saved_group = _scatter_group(saved)
    assert saved_group["sequence_axis"] == axis
    assert saved_group["layout"] == layout
    abi = _static_cache_abi(saved)
    assert abi == _static_cache_abi(metadata)
    ports = _graph_ports(package)
    for input_name, output_name in zip(abi["cache_inputs"], abi["cache_outputs"], strict=True):
        assert ports[input_name].shape[axis] == CAPACITY
        assert ports[output_name].shape == ports[input_name].shape


@pytest.mark.parametrize("ep_name", ["default", "tensorrt"])
@pytest.mark.parametrize(
    "dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16]
)
@pytest.mark.parametrize("static_cache", [False, True])
def test_attention_cache_mask_ep_contract(ep_name, dtype, static_cache):
    config = _text_config(dtype=dtype)
    with build_context(get_ep(ep_name), dtype=dtype), override_flags(static_cache_bias=False):
        module = registry.get("qwen2")(config)
        package = CausalLMTask(static_cache=static_cache, max_seq_len=CAPACITY).build(
            module, config
        )
    graph = package["model"].graph
    attentions = [node for node in graph if node.op_type == "Attention"]
    assert len(attentions) == 2
    explicit_bias = static_cache and ep_name == "tensorrt"
    for node in attentions:
        if static_cache:
            assert (node.inputs[3] is not None) == explicit_bias
        assert node.attributes["is_causal"].as_int() == (0 if explicit_bias else 1)
        native_nonpad = node.inputs[6] if len(node.inputs) > 6 else None
        assert (native_nonpad is not None) == (static_cache and not explicit_bias)
        if explicit_bias:
            assert node.inputs[3].dtype == dtype
            assert len(node.outputs) == 1
            assert "q_num_heads" not in node.attributes
            assert "kv_num_heads" not in node.attributes
        if static_cache:
            assert node.inputs[4] is None and node.inputs[5] is None
            cache_input = next(value for value in graph.inputs if value.name == "key_cache.0")
            assert len(cache_input.shape) == (4 if ep_name == "tensorrt" else 3)
        else:
            assert node.inputs[4] is not None and node.inputs[5] is not None
            assert len(node.outputs) == 3
    if explicit_bias:
        assert attentions[0].inputs[3] is attentions[1].inputs[3]
        graph_inputs = {value.name: value for value in graph.inputs}
        assert graph_inputs["nonpad_kv_seqlen"].uses()
        assert graph_inputs["write_indices"].uses()


@pytest.mark.skipif(
    os.environ.get("MOBIUS_TEST_TENSORRT") != "1",
    reason="Set MOBIUS_TEST_TENSORRT=1 to build and execute a TensorRT engine",
)
def test_tensorrt_static_cache_runtime(tmp_path, monkeypatch):
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from tensorrt_debug.runtime import TensorRTRunner

    assert torch.cuda.is_available(), "The enabled TensorRT runtime test requires CUDA"
    sdk = Path(os.environ["TENSORRT_ROOT"])
    trtexec = sdk / "bin" / ("trtexec.exe" if os.name == "nt" else "trtexec")
    assert trtexec.is_file(), f"TensorRT builder not found: {trtexec}"
    monkeypatch.setenv(
        "PATH", os.pathsep.join([str(sdk / "bin"), str(sdk / "lib"), os.environ["PATH"]])
    )
    hf_config = Qwen2Config(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=256,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    hf_config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        reference = Qwen2ForCausalLM(hf_config).float().eval()
    config = ArchitectureConfig.from_transformers(hf_config)
    config.dtype = ir.DataType.FLOAT
    capacity = 16
    with build_context(get_ep("tensorrt"), dtype=config.dtype):
        module = registry.get("qwen2")(config)
        package = CausalLMTask(static_cache=True, max_seq_len=capacity).build(module, config)
    package.apply_weights(module.preprocess_weights(reference.state_dict()))
    model = package["model"]
    model_path = tmp_path / "model.onnx"
    engine_path = tmp_path / "model.engine"
    ir.save(model, model_path, external_data="model.onnx.data")
    command = [
        str(trtexec),
        f"--onnx={model_path}",
        f"--saveEngine={engine_path}",
        "--skipInference",
        "--noTF32",
        "--decomposableAttentions=*",
    ]
    for option, length in (("minShapes", 1), ("optShapes", 4), ("maxShapes", 8)):
        shapes = []
        for value in model.graph.inputs:
            shape = [
                dimension if isinstance(dimension, int) else 1 for dimension in value.shape
            ]
            if value.name in ("input_ids", "position_ids"):
                shape = [1, length]
            shapes.append(f"{value.name}:{'x'.join(map(str, shape))}")
        command.append(f"--{option}={','.join(shapes)}")
    build_log = tmp_path / "build.log"
    with build_log.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, timeout=300, check=False
        )
    assert result.returncode == 0, build_log.read_text(encoding="utf-8")[-12000:]

    history = []
    with TensorRTRunner(engine_path, sdk) as runner, torch.inference_mode():
        assert runner.capacity == capacity
        for tokens in ([5, 17, 23, 42], [71], [19, 11]):
            position = len(history)
            history.extend(tokens)
            length = len(history)
            feeds = {
                "input_ids": np.asarray([tokens], dtype=np.int64),
                "position_ids": np.arange(position, length, dtype=np.int64)[None, :],
                "write_indices": np.asarray([position], dtype=np.int64),
                "nonpad_kv_seqlen": np.asarray([length], dtype=np.int64),
            }
            runner.prepare_inputs(feeds)
            before = {name: runner.read_tensor(name) for name in runner.caches}
            for name in runner.caches:
                assert runner.context.get_tensor_address(
                    name
                ) == runner.context.get_tensor_address("updated_" + name)
            runner.execute()
            actual_logits = runner.read_tensor("logits")
            expected = reference(torch.tensor([history]), use_cache=True)
            np.testing.assert_allclose(
                actual_logits, expected.logits[:, position:].numpy(), atol=1e-3, rtol=1e-3
            )
            for layer in range(config.num_hidden_layers):
                cache_layer = expected.past_key_values.layers[layer]
                key, value = cache_layer.keys, cache_layer.values
                for role, target in (("key", key), ("value", value)):
                    name = f"{role}_cache.{layer}"
                    actual = runner.read_tensor(name)
                    assert actual.shape == (
                        1,
                        config.num_key_value_heads,
                        capacity,
                        config.head_dim,
                    )
                    np.testing.assert_allclose(
                        actual[:, :, :length], target.numpy(), atol=1e-3, rtol=1e-3
                    )
                    np.testing.assert_array_equal(
                        actual[:, :, :position], before[name][:, :, :position]
                    )
                    np.testing.assert_array_equal(
                        actual[:, :, length:], before[name][:, :, length:]
                    )
        cached_logits = actual_logits.copy()
        runner.reset_caches()
        runner.prepare_inputs(
            {
                "input_ids": np.asarray([history], dtype=np.int64),
                "position_ids": np.arange(len(history), dtype=np.int64)[None, :],
                "write_indices": np.asarray([0], dtype=np.int64),
                "nonpad_kv_seqlen": np.asarray([len(history)], dtype=np.int64),
            }
        )
        runner.execute()
        np.testing.assert_allclose(
            cached_logits,
            runner.read_tensor("logits")[:, -len(tokens) :],
            atol=1e-3,
            rtol=1e-3,
        )


@pytest.mark.parametrize("write,length,nonpad", [(0, 4, 4), (4, 1, 5), (4, 3, 7), (4, 3, 6)])
def test_tensorrt_static_bias_slot_geometry(write, length, nonpad):
    def input_value(name):
        return ir.Value(name=name, shape=ir.Shape([1]), type=ir.TensorType(ir.DataType.INT64))

    cursor = input_value("write_indices")
    valid_length = input_value("nonpad_kv_seqlen")
    graph = ir.Graph(
        inputs=[cursor, valid_length],
        outputs=[],
        nodes=[],
        opset_imports={"": OPSET_VERSION},
        name="static_bias_geometry",
    )
    op = GraphBuilder(graph).op
    with build_context(get_ep("tensorrt")):
        bias = create_static_cache_attention_bias(
            op,
            write_indices=cursor,
            seq_len=op.Constant(value_ints=[length]),
            nonpad_kv_seqlen=valid_length,
            max_seq_len=8,
        )
    bias.name = "bias"
    graph.outputs.append(bias)
    model = ir.Model(graph, ir_version=10)
    actual = OnnxModelSession(model).run(
        {
            "write_indices": np.asarray([write], dtype=np.int64),
            "nonpad_kv_seqlen": np.asarray([nonpad], dtype=np.int64),
        }
    )["bias"]
    key_slots = np.arange(8)[None, :]
    query_slots = write + np.arange(length)[:, None]
    allowed = (key_slots <= query_slots) & (key_slots < nonpad)
    expected = np.where(allowed, 0, np.finfo(np.float32).min).astype(np.float32)[None, None]
    np.testing.assert_array_equal(actual, expected)


def test_tensorrt_static_attention_rejects_missing_bias():
    def value(name, shape):
        return ir.Value(
            name=name, shape=ir.Shape(shape), type=ir.TensorType(ir.DataType.FLOAT)
        )

    query = value("query", [1, 1, 64])
    key = value("key", [1, 1, 32])
    cache = value("cache", [1, 2, 8, 16])
    index = ir.Value(name="index", shape=ir.Shape([1]), type=ir.TensorType(ir.DataType.INT64))
    graph = ir.Graph(
        inputs=[query, key, cache, index],
        outputs=[],
        nodes=[],
        opset_imports={"": OPSET_VERSION},
    )
    with (
        build_context(get_ep("tensorrt")),
        pytest.raises(ValueError, match="explicit static-cache"),
    ):
        _apply_attention(
            GraphBuilder(graph).op,
            query,
            key,
            key,
            None,
            None,
            None,
            num_attention_heads=4,
            num_key_value_heads=2,
            scale=0.25,
            static_cache=StaticCacheState(cache, cache, index, index),
        )


def _cache_cells(workflow) -> list[str]:
    """Names of the loop cells the state service publishes as cache buffers."""
    group = next(iter(workflow["serving"]["state_service"]["groups"].values()))
    return list(group["ports"]["model"])


def _model_invoke(steps) -> dict[str, str]:
    """Input bindings of the neural component's invoke step in *steps*."""
    return next(step for step in steps if step.get("component") == "model")["inputs"]


def _scatter_group(metadata) -> tuple[str, dict]:
    """The state-service group whose buffers are written by an indexed scatter."""
    groups = metadata["pipeline"]["workflow"]["serving"]["state_service"]["groups"]
    return next(
        (name, group)
        for name, group in groups.items()
        if group.get("update", {}).get("kind") == "indexed_scatter"
    )


def _static_cache_abi(metadata) -> dict:
    """Recover the whole scatter ABI from the workflow and nothing else.

    This is deliberately written the way a runtime lowers a one-component
    workflow onto a direct decode path. If it can reconstruct every port a
    driver needs, then the workflow is a complete description and republishing
    the same facts under a second top-level key would only create two truths
    that can disagree.

    Note what it does not read: the component declares no port contracts, so
    every name here comes from the scatter discipline and the state pairs. The
    ports themselves are checked against the graph, which is the only thing
    entitled to say what a port's dtype and shape are.
    """
    workflow = metadata["pipeline"]["workflow"]
    _, group = _scatter_group(metadata)
    update = group["update"]
    component = next(iter(update["write_indices_ports"]))
    aliases = group["ports"][component]
    buffers = [
        aliases[cell] for cell in sorted(aliases, key=lambda cell: int(cell.rsplit("_")[-1]))
    ]
    return {
        "component": component,
        "write_indices_input": update["write_indices_ports"][component],
        "kv_sequence_length_input": update["kv_length_ports"][component],
        "capacity": workflow["inputs"][update["capacity"]]["default"],
        "cache_inputs": [alias["input"] for alias in buffers],
        "cache_outputs": [alias["output"] for alias in buffers],
    }


def _graph_ports(pkg) -> dict[str, Any]:
    """Every port of the exported decoder, keyed by name.

    The artifact is authoritative for dtype, rank and shape, so assertions
    about those resolve here rather than against a transcription in the
    metadata — a transcription could agree with itself while disagreeing with
    the graph a runtime actually binds.
    """
    model = pkg["model"]
    return {str(value.name): value for value in (*model.graph.inputs, *model.graph.outputs)}


@pytest.fixture(scope="module")
def static_built():
    pkg, config = _static_package()
    return pkg, build_decoder_workflow_metadata(pkg, config)


@pytest.fixture(scope="module")
def static_workflow(static_built):
    return static_built[1]


@pytest.fixture(scope="module")
def static_ports(static_built):
    return _graph_ports(static_built[0])


@pytest.fixture(scope="module")
def mixed():
    """Gemma 4 text with `--features static-cache`: one static + one dynamic geometry."""
    from gemma4_prefill_prefix_test import _make_config

    from mobius.tasks._gemma4 import Gemma4TextCausalLMTask

    config = _make_config()
    # Widen the global head so a collapsed cache group would be observably wrong.
    config.global_head_dim = 32
    module = registry.get("gemma4_text")(config)
    pkg = Gemma4TextCausalLMTask(static_cache=True, max_seq_len=CAPACITY).build(module, config)
    return build_decoder_workflow_metadata(pkg, config)


class TestStaticCacheAbiLivesOnlyInTheWorkflow:
    """The workflow is the package's single description of its scatter ABI."""

    def test_no_second_top_level_port_declaration(self, static_workflow):
        # `model` carries package-wide geometry and capabilities, never a copy
        # of the port ABI: two declarations of one fact can drift apart, and
        # nothing in the format says which one wins.
        assert "io" not in static_workflow.get("model", {})

    def test_control_ports_are_declared_by_the_scatter_discipline(self, static_workflow):
        abi = _static_cache_abi(static_workflow)
        assert abi["write_indices_input"] == STATIC_CACHE_WRITE_INDICES
        assert abi["kv_sequence_length_input"] == STATIC_CACHE_KV_SEQUENCE_LENGTH

    def test_control_ports_are_real_ports_of_the_component(
        self, static_workflow, static_ports
    ):
        # Naming a port the component does not expose would bind nothing, and
        # the graph is the only thing that can settle whether it does.
        abi = _static_cache_abi(static_workflow)
        for role in ("write_indices_input", "kv_sequence_length_input"):
            value = static_ports[abi[role]]
            # Both are per-row integer vectors, which is exactly why they are
            # declared rather than recognized by shape.
            assert value.dtype == ir.DataType.INT64
            assert len(value.shape) == 1

    def test_every_buffer_pair_is_declared(self, static_workflow, static_ports):
        abi = _static_cache_abi(static_workflow)
        assert abi["cache_inputs"] == [
            "key_cache.0",
            "value_cache.0",
            "key_cache.1",
            "value_cache.1",
        ]
        assert abi["cache_outputs"] == [f"updated_{name}" for name in abi["cache_inputs"]]
        for name in (*abi["cache_inputs"], *abi["cache_outputs"]):
            assert static_ports[name].shape[STATIC_CACHE_SEQUENCE_AXIS] == CAPACITY

    def test_capacity_is_recoverable(self, static_workflow):
        assert _static_cache_abi(static_workflow)["capacity"] == CAPACITY


class TestStaticCacheWorkflow:
    """The loop body has to carry the buffers and drive the write cursor."""

    def test_capacity_is_a_declared_workflow_input(self, static_workflow):
        workflow = static_workflow["pipeline"]["workflow"]
        capacity = workflow["inputs"]["package.cache_capacity"]
        assert capacity["source"] == {"kind": "literal"}
        assert capacity["default"] == CAPACITY
        assert capacity["required"] is False
        assert capacity["contract"]["dtype"] == "int64"
        assert capacity["contract"]["rank"] == 1

    def test_cache_cells_are_invariant_not_growing(self, static_workflow):
        workflow = static_workflow["pipeline"]["workflow"]
        for cell_name in _cache_cells(workflow):
            cell = workflow["state"][cell_name]
            assert cell["recurrence"] == {"kind": "invariant"}
            # The buffer keeps its full capacity every step.
            assert cell["contract"]["shape"][STATIC_CACHE_SEQUENCE_AXIS] == CAPACITY

    def test_write_cursor_and_valid_length_are_bound_each_phase(self, static_workflow):
        loop = static_workflow["pipeline"]["workflow"]["steps"][0]

        setup_inputs = _model_invoke(loop["setup"])
        # Prefill starts every row at slot 0 and ends with the prompt length.
        assert setup_inputs[STATIC_CACHE_WRITE_INDICES] == "initializer.write_indices"
        assert setup_inputs[STATIC_CACHE_KV_SEQUENCE_LENGTH] == "initializer.cache_lengths"

        body_inputs = _model_invoke(loop["steps"])
        # Decode writes at the length carried in from the previous step and
        # reports the length that step produced.
        assert body_inputs[STATIC_CACHE_WRITE_INDICES] == "cache_lengths"
        assert body_inputs[STATIC_CACHE_KV_SEQUENCE_LENGTH] == "cache_lengths.next"

    def test_buffers_are_carried_by_the_loop_not_regrown(self, static_workflow):
        loop = static_workflow["pipeline"]["workflow"]["steps"][0]
        carried = {entry["cell"]: entry["next"] for entry in loop["carried"]}
        for cell in _cache_cells(static_workflow["pipeline"]["workflow"]):
            assert carried[cell].startswith("decoder.body.updated_")

    def test_state_service_publishes_an_indexed_scatter_discipline(self, static_workflow):
        groups = static_workflow["pipeline"]["workflow"]["serving"]["state_service"]["groups"]
        assert len(groups) == 1
        group = next(iter(groups.values()))
        assert group["update"] == {
            "kind": "indexed_scatter",
            "write_indices": "cache_lengths",
            "capacity": "package.cache_capacity",
            "write_indices_ports": {"model": STATIC_CACHE_WRITE_INDICES},
            "kv_length_ports": {"model": STATIC_CACHE_KV_SEQUENCE_LENGTH},
        }
        assert group["logical_lengths"] == "cache_lengths"
        assert group["sequence_axis"] == STATIC_CACHE_SEQUENCE_AXIS
        assert group["layout"] == "bsh"
        # Scattering writes in place, so the runtime may alias the buffers.
        assert group["aliasing"] == "permitted"
        # Every buffer port pair is published so a runtime can bind them.
        ports = group["ports"]["model"]
        assert len(ports) == 4
        assert ports["cache_0"] == {
            "input": "key_cache.0",
            "output": "updated_key_cache.0",
            "role": "key",
            "layer": 0,
        }

    def test_control_ports_are_not_advertised_as_request_inputs(self, static_workflow):
        # They are derived from loop state, so a caller must not be asked
        # to supply them.
        inputs = static_workflow["pipeline"]["workflow"]["inputs"]
        assert STATIC_CACHE_WRITE_INDICES not in inputs
        assert STATIC_CACHE_KV_SEQUENCE_LENGTH not in inputs


class TestStaticCachePortDerivation:
    """The ABI is read from the graph, never assumed."""

    def test_capacity_follows_the_requested_max_sequence_length(self):
        config = _text_config()
        module = registry.get("qwen2")(config)
        pkg = CausalLMTask(static_cache=True, max_seq_len=64).build(module, config)
        metadata = build_decoder_workflow_metadata(pkg, config)
        capacity = metadata["pipeline"]["workflow"]["inputs"]["package.cache_capacity"]
        assert capacity["default"] == 64

    def test_grouped_query_layouts_are_declared_flat(self, static_workflow):
        # The exporter stores the static cache as (batch, capacity, kv_hidden);
        # publishing a 4-D BNSH shape would misdescribe the buffer a runtime
        # has to allocate.
        workflow = static_workflow["pipeline"]["workflow"]
        contract = workflow["state"][_cache_cells(workflow)[0]]["contract"]
        assert contract["rank"] == 3
        # 2 kv heads x 16 head_dim
        assert contract["shape"] == ["batch", CAPACITY, 32]

    def test_layer_count_follows_the_config(self):
        pkg, config = _static_package(num_hidden_layers=3)
        abi = _static_cache_abi(build_decoder_workflow_metadata(pkg, config))
        assert abi["cache_inputs"] == [
            "key_cache.0",
            "value_cache.0",
            "key_cache.1",
            "value_cache.1",
            "key_cache.2",
            "value_cache.2",
        ]


class TestHeterogeneousStaticCache:
    """Gemma 4 mixes a static full-attention cache with a dynamic sliding one.

    Its two cache geometries have different ranks, layouts, sequence axes and
    head dimensions, and its KV-shared suffix owns no cache at all. Publishing
    one undifferentiated group — or listing the borrowing layers as if they
    owned buffers — would have a runtime allocate caches that do not exist and
    scatter into a sliding cache that is appended to.
    """

    def test_only_cache_owning_layers_are_declared(self, mixed):
        # layer_types = [sliding, full, sliding, full] with the last two layers
        # sharing KV: exactly one layer owns a static buffer.
        abi = _static_cache_abi(mixed)
        assert abi["cache_inputs"] == ["key_cache.1", "value_cache.1"]
        assert abi["cache_outputs"] == ["updated_key_cache.1", "updated_value_cache.1"]

    def test_each_geometry_gets_its_own_update_discipline(self, mixed):
        groups = mixed["pipeline"]["workflow"]["serving"]["state_service"]["groups"]
        assert set(groups) == {
            "decoder_cache_full_attention",
            "decoder_cache_sliding_attention",
        }
        full = groups["decoder_cache_full_attention"]
        sliding = groups["decoder_cache_sliding_attention"]

        assert full["update"]["kind"] == "indexed_scatter"
        assert full["sequence_axis"] == STATIC_CACHE_SEQUENCE_AXIS
        assert full["layout"] == "bsh"
        assert full["aliasing"] == "permitted"

        # The sliding layers still append into a growing BNSH tensor.
        assert "update" not in sliding or sliding["update"]["kind"] == "append"
        assert sliding["sequence_axis"] == 2
        assert sliding["layout"] == "bnsh"
        assert sliding["aliasing"] == "forbidden"
        assert sliding["reuse"]["evictable_prefix"] is True

    def test_dual_head_dims_survive_the_split(self, mixed):
        workflow = mixed["pipeline"]["workflow"]
        groups = workflow["serving"]["state_service"]["groups"]
        state = workflow["state"]

        full_cell = next(iter(groups["decoder_cache_full_attention"]["ports"]["model"]))
        sliding_cell = next(iter(groups["decoder_cache_sliding_attention"]["ports"]["model"]))
        # 1 kv head x global_head_dim 32, flattened into a fixed buffer.
        assert state[full_cell]["contract"]["shape"] == ["batch", CAPACITY, 32]
        # 1 kv head x head_dim 16, still growing.
        assert state[sliding_cell]["contract"]["shape"] == [
            "batch",
            1,
            "past_sequence_len",
            16,
        ]

    def test_a_hybrid_decoder_keeps_its_padding_mask(self, mixed):
        # The dynamic sliding layers still build their bias from attention_mask;
        # dropping it because *some* layers are static loses padding.
        loop = mixed["pipeline"]["workflow"]["steps"][0]
        assert "attention_mask" in _model_invoke(loop["setup"])


class TestFp8KvCacheMetadata:
    """FP8 storage is a graph-visible fact, so the metadata must repeat it.

    A runtime allocates the KV buffers from these contracts. Publishing
    ``float16`` for a cache the graph declares as ``float8_e4m3fn`` would size
    every buffer at twice the bytes the model reads, so the declared dtype has
    to be whatever the graph actually says — never the model's compute dtype.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def fp8_workflow():
        import onnx_ir as ir

        from mobius._optimizations import optimize_model

        config = _text_config()
        module = registry.get("qwen2")(config)
        pkg = CausalLMTask().build(module, config)
        optimize_model(
            pkg["model"],
            ep="cuda",
            dtype=ir.DataType.FLOAT16,
            model_role="decoder",
            fp8_kv_cache=True,
        )
        return pkg, build_decoder_workflow_metadata(pkg, config)

    def test_graph_ports_are_fp8(self, fp8_workflow):
        pkg, _ = fp8_workflow
        import onnx_ir as ir

        caches = [
            value
            for value in [*pkg["model"].graph.inputs, *pkg["model"].graph.outputs]
            if value.name.startswith(("past_key_values.", "present."))
        ]
        assert caches
        assert {value.dtype for value in caches} == {ir.DataType.FLOAT8E4M3FN}

    def test_state_contracts_declare_the_graph_dtype(self, fp8_workflow):
        _, metadata = fp8_workflow
        workflow = metadata["pipeline"]["workflow"]
        cells = _cache_cells(workflow)
        assert cells
        for cell in cells:
            assert workflow["state"][cell]["contract"]["dtype"] == "float8_e4m3fn"

    def test_carried_cache_is_still_an_appending_cache(self, fp8_workflow):
        # Quantizing the cells changes their dtype, not their update discipline.
        _, metadata = fp8_workflow
        group = next(
            iter(
                metadata["pipeline"]["workflow"]["serving"]["state_service"]["groups"].values()
            )
        )
        assert group["sequence_axis"] == 2
        assert group["layout"] == "bnsh"
        assert group.get("update", {}).get("kind") != "indexed_scatter"


class TestFeatureCombinations:
    """Which feature pairs are representable, and which are refused and why."""

    def test_fp8_requires_an_operator_that_can_dequantize_the_cache(self):
        # A static-cache graph scatters into buffers read by ai.onnx Attention,
        # which has no k_scale/v_scale inputs. Retyping those buffers would
        # declare FP8 over bytes that are read as float16, so the build must
        # refuse rather than emit either a wrong graph or a silently fp16 one.
        import onnx_ir as ir

        from mobius._optimizations import optimize_model

        config = _text_config()
        module = registry.get("qwen2")(config)
        pkg = CausalLMTask(static_cache=True, max_seq_len=CAPACITY).build(module, config)
        with pytest.raises(ValueError, match="no GroupQueryAttention KV cache"):
            optimize_model(
                pkg["model"],
                ep="cuda",
                dtype=ir.DataType.FLOAT16,
                model_role="decoder",
                fp8_kv_cache=True,
            )

    def test_static_cache_survives_cuda_optimization(self):
        # Optimizing must not rewrite the scatter into an appending cache.
        import onnx_ir as ir

        from mobius._optimizations import optimize_model

        config = _text_config()
        module = registry.get("qwen2")(config)
        pkg = CausalLMTask(static_cache=True, max_seq_len=CAPACITY).build(module, config)
        optimize_model(
            pkg["model"], ep="cuda", dtype=ir.DataType.FLOAT16, model_role="decoder"
        )
        metadata = build_decoder_workflow_metadata(pkg, config)
        _, group = _scatter_group(metadata)
        assert group["update"]["kind"] == "indexed_scatter"
        assert _static_cache_abi(metadata)["cache_inputs"] == [
            "key_cache.0",
            "value_cache.0",
            "key_cache.1",
            "value_cache.1",
        ]
