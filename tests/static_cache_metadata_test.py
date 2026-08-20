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
"""

from __future__ import annotations

import pytest

from mobius import registry
from mobius._configs import ArchitectureConfig
from mobius._constants import (
    STATIC_CACHE_KV_SEQUENCE_LENGTH,
    STATIC_CACHE_SEQUENCE_AXIS,
    STATIC_CACHE_WRITE_INDICES,
)
from mobius.integrations.onnx_genai.workflow_metadata import (
    build_decoder_workflow_metadata,
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


def _static_package(**overrides):
    config = _text_config(**overrides)
    module = registry.get("qwen2")(config)
    task = CausalLMTask(static_cache=True, max_seq_len=CAPACITY)
    return task.build(module, config), config


def _cache_cells(workflow) -> list[str]:
    """Names of the loop cells the state service publishes as cache buffers."""
    group = next(iter(workflow["serving"]["state_service"]["groups"].values()))
    return list(group["ports"]["model"])


def _model_invoke(steps) -> dict[str, str]:
    """Input bindings of the neural component's invoke step in *steps*."""
    return next(step for step in steps if step.get("component") == "model")["inputs"]


@pytest.fixture(scope="module")
def static_workflow():
    pkg, config = _static_package()
    return build_decoder_workflow_metadata(pkg, config)


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


class TestStaticCacheModelIo:
    """``model.io.static_cache`` is the authoritative port ABI."""

    def test_declares_control_and_buffer_ports(self, static_workflow):
        static_cache = static_workflow["model"]["io"]["static_cache"]
        assert static_cache["write_indices_input"] == STATIC_CACHE_WRITE_INDICES
        assert static_cache["kv_sequence_length_input"] == STATIC_CACHE_KV_SEQUENCE_LENGTH
        assert static_cache["key_cache_inputs"] == ["key_cache.0", "key_cache.1"]
        assert static_cache["value_cache_inputs"] == ["value_cache.0", "value_cache.1"]
        assert static_cache["key_cache_outputs"] == [
            "updated_key_cache.0",
            "updated_key_cache.1",
        ]
        assert static_cache["value_cache_outputs"] == [
            "updated_value_cache.0",
            "updated_value_cache.1",
        ]

    def test_per_layer_lists_are_paired(self, static_workflow):
        # A runtime binds these four lists positionally; unequal lengths would
        # silently pair layer i's key buffer with layer j's output.
        static_cache = static_workflow["model"]["io"]["static_cache"]
        lengths = {
            len(static_cache[key])
            for key in (
                "key_cache_inputs",
                "value_cache_inputs",
                "key_cache_outputs",
                "value_cache_outputs",
            )
        }
        assert lengths == {2}

    def test_owns_its_cache_and_declares_no_appending_ports(self, static_workflow):
        io = static_workflow["model"]["io"]
        assert io["kv_ownership"] == "owned"
        # A static cache has no past/present pair to advertise; declaring one
        # would invite a runtime to concatenate into a fixed buffer.
        assert "kv_inputs" not in io
        assert "kv_outputs" not in io


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
        io = build_decoder_workflow_metadata(pkg, config)["model"]["io"]["static_cache"]
        assert io["key_cache_inputs"] == ["key_cache.0", "key_cache.1", "key_cache.2"]


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
        static_cache = mixed["model"]["io"]["static_cache"]
        assert static_cache["key_cache_inputs"] == ["key_cache.1"]
        assert static_cache["value_cache_inputs"] == ["value_cache.1"]
        assert static_cache["key_cache_outputs"] == ["updated_key_cache.1"]
        assert static_cache["value_cache_outputs"] == ["updated_value_cache.1"]

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
        group = next(
            iter(
                metadata["pipeline"]["workflow"]["serving"]["state_service"]["groups"].values()
            )
        )
        assert group["update"]["kind"] == "indexed_scatter"
        assert metadata["model"]["io"]["static_cache"]["key_cache_inputs"] == [
            "key_cache.0",
            "key_cache.1",
        ]
