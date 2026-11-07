# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Packed storage, fail-closed rewrites and unchanged public decoder contracts."""

from __future__ import annotations

import dataclasses

import numpy as np
import onnx
import onnx_ir as ir
import pytest
from onnx_ir.passes import common as common_passes

from mobius import build_from_module
from mobius._configs import NemotronHOmniConfig
from mobius._configs.per_model._nemotron_h_omni_test import _omni_config
from mobius.models import NemotronHOmniModel
from mobius.models._nemotron_h_omni_packed import (
    _ggml_parts,
    _native_nvfp4_initializer,
    _pack_projection,
    optimize_nemotron_h_omni_cuda,
)


def _fixture(*, loaded=True, topk=1, experts=4):
    source = _omni_config()
    source["llm_config"]["hidden_size"] = 256
    source["llm_config"]["moe_latent_size"] = 1024
    source["llm_config"]["block_configs"][1]["num_experts_per_tok"] = topk
    source["llm_config"]["block_configs"][1]["n_routed_experts"] = experts
    plan = source["quantization_config"]["quantized_layers"]
    for layer in (1, 3):
        block = source["llm_config"]["block_configs"][layer]
        block["moe_intermediate_size"] = 256
        for expert in range(block["n_routed_experts"]):
            for projection in ("up_proj", "down_proj"):
                plan[
                    f"language_model.backbone.layers.{layer}.mixer.experts.{expert}.{projection}"
                ] = {
                    "quant_algo": "IQ2_XXS",
                    "group_size": 256,
                    "block_payload_bytes": 66,
                    "packing": "ggml",
                }
    config = dataclasses.replace(
        NemotronHOmniConfig.from_transformers(source), dtype=ir.DataType.BFLOAT16
    )
    package = build_from_module(
        NemotronHOmniModel(config),
        config,
        task="nemotron-h-omni",
        execution_provider="cuda",
    )
    for value in package["decoder"].graph.outputs:
        if value.shape is None:
            value.shape = ir.Shape(
                [None, None, 32] if value.name == "logits" else [None, 1, None, 8]
            )
    if loaded:
        for model in package.values():
            for value in model.graph.initializers.values():
                if value.const_value is not None:
                    continue
                data = np.zeros(tuple(value.shape), dtype=value.dtype.numpy())
                if value.dtype == ir.DataType.UINT8 and value.shape[-1] in (34, 66):
                    data[..., :2] = np.array([1.0009765625], np.float16).view(np.uint8)
                value.const_value = ir.tensor(data)
    return package


@pytest.mark.parametrize("iq2", [False, True])
def test_split_ggml_storage_preserves_fp16_scale_bits_and_signed_codes(iq2):
    block_size = 66 if iq2 else 34
    raw = np.arange(2 * 4 * block_size, dtype=np.uint8).reshape(2, 4, block_size)
    scale_bits = np.array([0x3C01, 0x8000, 0x0001, 0x7BFF], np.uint16)
    raw[..., :2] = scale_bits.view(np.uint8).reshape(4, 2)
    payload, scales = _ggml_parts(ir.tensor(raw), iq2=iq2)
    np.testing.assert_array_equal(scales.view(np.uint16), np.broadcast_to(scale_bits, (2, 4)))
    expected = raw[..., 2:] if iq2 else raw[..., 2:] ^ np.uint8(128)
    np.testing.assert_array_equal(payload.reshape(expected.shape), expected)


@pytest.mark.parametrize("native_blocks", [False, True])
def test_compact_decoder_preserves_interfaces_media_and_source_graph(tmp_path, native_blocks):
    original = _fixture()
    for name, value in original["decoder"].graph.initializers.items():
        if ".experts." in name:
            expert = int(name.split(".experts.")[1].split(".")[0])
            data = value.const_value.numpy().copy()
            data[..., 2:] = expert + 1
            data[..., :2] = np.array([0.001003 * (expert + 1)], np.float16).view(np.uint8)
            value.const_value = ir.tensor(data)
    source_nodes = list(original["decoder"].graph)
    source_initializers = dict(original["decoder"].graph.initializers)
    compact = optimize_nemotron_h_omni_cuda(original, native_blocks=native_blocks)
    before, after = original["decoder"], compact["decoder"]
    assert compact["vision_encoder"] is original["vision_encoder"]
    assert compact["embedding"] is original["embedding"]
    for old, new in zip(
        [*before.graph.inputs, *before.graph.outputs],
        [*after.graph.inputs, *after.graph.outputs],
    ):
        assert (old.name, old.type, old.shape) == (new.name, new.type, new.shape)
    assert list(before.graph) == source_nodes
    assert dict(before.graph.initializers) == source_initializers
    assert len(after.graph) < len(before.graph)
    moe_op = "MoEPacked" if native_blocks else "MoEPackedIQ2XXS"
    moe_nodes = [n for n in after.graph.all_nodes() if n.op_type == moe_op]
    assert len(moe_nodes) == 2
    if native_blocks:
        assert all(
            n.attributes["activation_type"].as_string() == "relu_squared"
            and n.attributes["gate_type"].as_int() == 16
            and n.attributes["down_type"].as_int() == 16
            and n.attributes["normalize_routing_weights"].as_int() == 0
            for n in moe_nodes
        )
        assert all(
            len(n.inputs) == 7
            and n.inputs[1] is None
            and n.inputs[3] is None
            and n.inputs[5].dtype == ir.DataType.INT32
            and n.inputs[6].dtype == ir.DataType.FLOAT
            and len(n.outputs) == 1
            for n in moe_nodes
        )
        assert not any(n.op_type.startswith("MatMulPackedQ") for n in after.graph.all_nodes())
        assert compact.weight_loading_report["packed_cuda"]["storage"] == "native_blocks"
    assert not any(".experts." in name for name in after.graph.initializers)
    for name in after.graph.initializers:
        if name in before.graph.initializers:
            assert (
                after.graph.initializers[name].const_value
                is before.graph.initializers[name].const_value
            )
    assert optimize_nemotron_h_omni_cuda(compact, native_blocks=native_blocks) is compact
    with pytest.raises(ValueError, match="original source package"):
        optimize_nemotron_h_omni_cuda(compact, native_blocks=not native_blocks)
    path = tmp_path / "compact.onnx"
    ir.save(after, path, external_data="compact.data", size_threshold_bytes=1024)
    onnx.checker.check_model(str(path))
    reloaded = ir.load(path)
    assert [v.name for v in reloaded.graph.outputs] == [v.name for v in before.graph.outputs]
    for layer, experts in ((1, 4), (3, 3)):
        root = f"decoder.model.layers.{layer}.moe"
        for projection in ("up_proj", "down_proj"):
            if native_blocks:
                native = reloaded.graph.initializers[
                    root + f".native_{projection}_weight"
                ].const_value.numpy()
                for expert in range(experts):
                    source = before.graph.initializers[
                        root + f".experts.{expert}.{projection}.weight"
                    ].const_value.numpy()
                    np.testing.assert_array_equal(
                        native[expert], source.reshape(source.shape[0], -1)
                    )
                continue
            payload = reloaded.graph.initializers[
                root + f".packed_{projection}_weight"
            ].const_value.numpy()
            scales = reloaded.graph.initializers[
                root + f".packed_{projection}_scales"
            ].const_value.numpy()
            for expert in range(experts):
                source = before.graph.initializers[
                    root + f".experts.{expert}.{projection}.weight"
                ].const_value
                expected_payload, expected_scales = _ggml_parts(source, iq2=True)
                np.testing.assert_array_equal(payload[expert], expected_payload)
                np.testing.assert_array_equal(
                    scales[expert].view(np.uint16), expected_scales.view(np.uint16)
                )


@pytest.mark.parametrize("native_blocks", [False, True])
def test_compact_decoder_reuses_shared_projection_storage(native_blocks):
    package = _fixture()
    graph = package["decoder"].graph
    head = next(n for n in graph if n.op_type == "MatMulGGMLQ8_0")
    extra = ir.Value(
        name="extra_logits", type=head.outputs[0].type, shape=head.outputs[0].shape
    )
    graph.append(ir.Node(head.domain, head.op_type, head.inputs, outputs=[extra]))
    graph.outputs.append(extra)
    graph = optimize_nemotron_h_omni_cuda(package, native_blocks=native_blocks)[
        "decoder"
    ].graph
    suffix = ".native_weight" if native_blocks else ".packed_weight"
    assert sum(name.endswith(suffix) for name in graph.initializers) == 1
    assert [v.name for v in graph.outputs][-1] == "extra_logits"
    if native_blocks:
        native = next(v for v in graph.initializers.values() if v.name.endswith(suffix))
        np.testing.assert_array_equal(
            native.const_value.numpy(), head.inputs[1].const_value.numpy().reshape(32, -1)
        )
        assert len(native.uses()) >= 2


@pytest.mark.parametrize("native_blocks", [False, True])
def test_compact_decoder_rejects_unloaded_or_observed_experts_without_mutation(native_blocks):
    package = _fixture(loaded=False)
    with pytest.raises(ValueError, match="loaded weights"):
        optimize_nemotron_h_omni_cuda(package, native_blocks=native_blocks)
    package = _fixture()
    node = next(n for n in package["decoder"].graph if n.op_type == "MatMulGGMLIQ2XXS")
    package["decoder"].graph.outputs.append(node.outputs[0])
    before = list(package["decoder"].graph)
    with pytest.raises(ValueError, match="observed expert"):
        optimize_nemotron_h_omni_cuda(package, native_blocks=native_blocks)
    assert list(package["decoder"].graph) == before


def test_native_nvfp4_preserves_all_nibbles_and_scale_bytes():
    n, k = 3, 4096
    codes = np.arange(n * k // 2, dtype=np.uint8).reshape(n, k // 2)
    scales = np.arange(n * k // 16, dtype=np.uint8).reshape(n, k // 16) % 127
    graph = ir.Graph([], [], nodes=[])
    values = [
        ir.Value(
            name=name,
            type=ir.TensorType(ir.DataType.UINT8),
            shape=ir.Shape(array.shape),
            const_value=ir.tensor(array),
        )
        for name, array in (("codes", codes), ("scales", scales))
    ]
    native = _native_nvfp4_initializer(graph, *values, n=n, k=k).const_value.numpy()
    blocks = native.reshape(n, k // 64, 36)
    np.testing.assert_array_equal(blocks[..., :4].reshape(scales.shape), scales)
    gguf_codes = blocks[..., 4:].reshape(n, k // 16, 8)
    elements = np.concatenate((gguf_codes & 15, gguf_codes >> 4), axis=-1)
    adjacent = elements[..., 0::2] | (elements[..., 1::2] << 4)
    np.testing.assert_array_equal(adjacent.reshape(codes.shape), codes)
    values[1].dtype = ir.DataType.FLOAT16
    with pytest.raises(ValueError, match="byte scales"):
        _native_nvfp4_initializer(ir.Graph([], [], nodes=[]), *values, n=n, k=k)


@pytest.mark.parametrize("invalid", [127, 128, 255])
def test_native_nvfp4_rejects_signed_or_nonfinite_source_scales(invalid):
    n, k = 1, 4096
    values = [
        ir.Value(
            name=name,
            type=ir.TensorType(ir.DataType.UINT8),
            shape=ir.Shape(array.shape),
            const_value=ir.tensor(array),
        )
        for name, array in (
            ("codes", np.zeros((n, k // 2), np.uint8)),
            ("scales", np.full((n, k // 16), invalid, np.uint8)),
        )
    ]
    native = _native_nvfp4_initializer(ir.Graph([], [], nodes=[]), *values, n=n, k=k)
    with pytest.raises(ValueError, match="nonnegative finite"):
        native.const_value.numpy()


def test_native_nvfp4_graph_preserves_global_scale_and_reuses_block_storage():
    n, k = 3, 4096
    inputs = [
        ir.Value(
            name=name,
            type=ir.TensorType(ir.DataType.from_numpy(array.dtype)),
            shape=ir.Shape(array.shape),
            const_value=ir.tensor(array),
        )
        for name, array in (
            ("codes", np.zeros((n, k // 2), np.uint8)),
            ("scales", np.full((n, k // 16), 56, np.uint8)),
            ("global", np.array([0.12345679], np.float32)),
        )
    ]
    x = ir.Value(name="x", type=ir.TensorType(ir.DataType.BFLOAT16), shape=ir.Shape([None, k]))
    outputs = [
        ir.Value(
            name=name, type=ir.TensorType(ir.DataType.BFLOAT16), shape=ir.Shape([None, n])
        )
        for name in ("y", "shared_y")
    ]
    nodes = [
        ir.Node("com.microsoft", "MatMulNVFP4A16", [x, *inputs], outputs=[output])
        for output in outputs
    ]
    graph = ir.Graph(
        [x],
        outputs,
        nodes=nodes,
        initializers=inputs,
        opset_imports={"": 24, "com.microsoft": 1},
    )
    cache = {}
    for node in nodes:
        assert _pack_projection(graph, node, cache, native_blocks=True)
    common_passes.RemoveUnusedNodesPass()(ir.Model(graph, ir_version=11))
    projections = [node for node in graph.all_nodes() if node.op_type == "MatMulPacked"]
    assert len(projections) == 4
    assert len({node.inputs[1] for node in projections}) == 1
    assert all(node.inputs[2] is inputs[2] for node in projections)
    assert all(
        node.attributes["packed_type"].as_int() == 40
        and node.attributes["dequantize_to_activation_dtype"].as_int() == 1
        and node.attributes["K"].as_int() == k
        and node.attributes["N"].as_int() == n
        for node in projections
    )
    assert (
        inputs[0].name not in graph.initializers and inputs[1].name not in graph.initializers
    )
    assert graph.initializers["global"].const_value is inputs[2].const_value
    assert [value.name for value in graph.outputs] == ["y", "shared_y"]


def test_native_packed_legacy_artifacts_require_original_source():
    compact = optimize_nemotron_h_omni_cuda(_fixture(), native_blocks=True)
    del compact["decoder"].metadata_props["mobius.bolt.native_packed_cuda_abi"]
    with pytest.raises(ValueError, match="original source package"):
        optimize_nemotron_h_omni_cuda(compact, native_blocks=True)
