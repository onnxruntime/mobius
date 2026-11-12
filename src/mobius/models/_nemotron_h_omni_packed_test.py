# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Packed storage, fail-closed rewrites and unchanged public decoder contracts."""

from __future__ import annotations

import dataclasses

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest

from mobius import build_from_module
from mobius._configs import NemotronHOmniConfig
from mobius._configs.per_model._nemotron_h_omni_test import _omni_config
from mobius.models import NemotronHOmniModel
from mobius.models._nemotron_h_omni_packed import (
    _packed_initializer,
    optimize_nemotron_h_omni_cuda,
)


def _fixture(*, loaded=True, topk=1, experts=4, normalize=True, scale=1.0):
    source = _omni_config()
    source["llm_config"]["hidden_size"] = 256
    source["llm_config"]["moe_latent_size"] = 1024
    source["llm_config"]["block_configs"][1]["num_experts_per_tok"] = topk
    source["llm_config"]["block_configs"][1]["n_routed_experts"] = experts
    source["llm_config"]["norm_topk_prob"] = normalize
    source["llm_config"]["routed_scaling_factor"] = scale
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
def test_packed_ggml_storage_preserves_complete_source_blocks(iq2):
    block_size = 66 if iq2 else 34
    raw = np.arange(2 * 4 * block_size, dtype=np.uint8).reshape(2, 4, block_size)
    scale_bits = np.array([0x3C01, 0x8000, 0x0001, 0x7BFF], np.uint16)
    raw[..., :2] = scale_bits.view(np.uint8).reshape(4, 2)
    graph = ir.Graph([], [], nodes=[], opset_imports={"": 24})
    source = ir.Value(
        name="source",
        type=ir.TensorType(ir.DataType.UINT8),
        shape=ir.Shape(raw.shape),
        const_value=ir.tensor(raw),
    )
    packed = _packed_initializer(graph, "packed", [source], iq2=iq2, bank=False)
    np.testing.assert_array_equal(packed.const_value.numpy(), raw.reshape(2, -1))


def test_compact_decoder_preserves_interfaces_media_and_source_graph(tmp_path):
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
    compact = optimize_nemotron_h_omni_cuda(original)
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
    assert sum(n.op_type == "MoEPacked" for n in after.graph.all_nodes()) == 2
    assert after.metadata_props["mobius.bolt.packed_cuda"] == "3"
    assert not any(".experts." in name for name in after.graph.initializers)
    for name in after.graph.initializers:
        if name in before.graph.initializers:
            assert (
                after.graph.initializers[name].const_value
                is before.graph.initializers[name].const_value
            )
    assert optimize_nemotron_h_omni_cuda(compact) is compact
    path = tmp_path / "compact.onnx"
    ir.save(after, path, external_data="compact.data", size_threshold_bytes=1024)
    onnx.checker.check_model(str(path))
    reloaded = ir.load(path)
    assert [v.name for v in reloaded.graph.outputs] == [v.name for v in before.graph.outputs]
    for layer, experts in ((1, 4), (3, 3)):
        root = f"decoder.model.layers.{layer}.moe"
        for projection in ("up_proj", "down_proj"):
            packed = reloaded.graph.initializers[
                root + f".packed_{projection}_weight"
            ].const_value.numpy()
            for expert in range(experts):
                source = before.graph.initializers[
                    root + f".experts.{expert}.{projection}.weight"
                ].const_value
                np.testing.assert_array_equal(
                    packed[expert],
                    source.numpy().reshape(256, -1)
                    if projection == "up_proj"
                    else source.numpy().reshape(1024, -1),
                )
    for router in (n for n in reloaded.graph if n.op_type == "MoEGroupRouter"):
        assert len(router.outputs) == 3
        assert router.outputs[0].name == ""


def test_compact_decoder_reuses_shared_projection_storage():
    package = _fixture()
    graph = package["decoder"].graph
    head = next(n for n in graph if n.op_type == "MatMulGGMLQ8_0")
    extra = ir.Value(
        name="extra_logits", type=head.outputs[0].type, shape=head.outputs[0].shape
    )
    graph.append(ir.Node(head.domain, head.op_type, head.inputs, outputs=[extra]))
    graph.outputs.append(extra)
    graph = optimize_nemotron_h_omni_cuda(package)["decoder"].graph
    assert sum(name.endswith(".packed_weight") for name in graph.initializers) == 1
    assert [v.name for v in graph.outputs][-1] == "extra_logits"


def test_compact_decoder_rejects_unloaded_or_observed_experts_without_mutation():
    package = _fixture(loaded=False)
    with pytest.raises(ValueError, match="loaded weights"):
        optimize_nemotron_h_omni_cuda(package)
    package = _fixture()
    node = next(n for n in package["decoder"].graph if n.op_type == "MatMulGGMLIQ2XXS")
    package["decoder"].graph.outputs.append(node.outputs[0])
    before = list(package["decoder"].graph)
    with pytest.raises(ValueError, match="observed expert"):
        optimize_nemotron_h_omni_cuda(package)
    assert list(package["decoder"].graph) == before


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("scale", [1.0, 2.5])
def test_compact_decoder_wires_full_fp32_routing_without_compact_ids(normalize, scale):
    original = _fixture(topk=2, normalize=normalize, scale=scale)
    graph = optimize_nemotron_h_omni_cuda(original)["decoder"].graph
    routers = [node for node in graph if node.op_type == "MoEGroupRouter"]
    consumers = [node for node in graph if node.op_type == "MoEPacked"]
    assert len(routers) == len(consumers) == 2
    for router, consumer in zip(routers, consumers):
        assert len(router.outputs) == 3
        assert router.outputs[0].name == ""
        assert router.inputs[0].dtype == ir.DataType.FLOAT
        assert router.inputs[1].dtype == ir.DataType.FLOAT
        assert consumer.inputs[1] is router.outputs[1]
        if scale == 1.0:  # noqa: RUF069
            assert consumer.inputs[5] is router.outputs[2]
        assert consumer.inputs[1].dtype == consumer.inputs[5].dtype == ir.DataType.FLOAT
        assert consumer.inputs[0].dtype == consumer.outputs[0].dtype == ir.DataType.BFLOAT16
        assert consumer.inputs[3] is None
        assert consumer.attributes.get_int("k") == router.attributes.get_int("top_k")
        assert consumer.attributes.get_int("normalize_routing_weights") == int(
            normalize and scale == 1.0  # noqa: RUF069
        )
        assert (
            consumer.attributes.get_int("gate_type")
            == consumer.attributes.get_int("down_type")
            == 16
        )
        if scale != 1.0:  # noqa: RUF069
            scaled = consumer.inputs[5].producer()
            assert scaled.op_type == "Mul"
            assert scaled.inputs[1].const_value.numpy().item() == np.float32(scale)
            assert next(iter(consumer.outputs[0].uses())).node.op_type != "Cast"
    assert not any(node.op_type == "TopK" for node in graph)
    assert not any(
        node.op_type in {"MoEPackedIQ2XXS", "MatMulPackedIQ2XXS", "MatMulPackedQ8"}
        for node in graph.all_nodes()
    )
    head = next(node for node in graph.all_nodes() if node.op_type == "MatMulPacked")
    assert head.attributes.get_int("packed_type") == 8
    source = next(
        node for node in original["decoder"].graph if node.op_type == "MatMulGGMLQ8_0"
    )
    np.testing.assert_array_equal(
        head.inputs[1].const_value.numpy(),
        source.inputs[1].const_value.numpy().reshape(32, -1),
    )


@pytest.mark.parametrize("provider", ["CPUExecutionProvider", "CUDAExecutionProvider"])
@pytest.mark.parametrize("topk", [1, 2, 8])
@pytest.mark.parametrize("normalize", [False, True])
def test_compact_decoder_prescales_selected_mixing_in_fp32(
    tmp_path, topk, normalize, provider
):
    if provider not in ort.get_available_providers():
        pytest.skip(f"{provider} is unavailable")
    graph = optimize_nemotron_h_omni_cuda(
        _fixture(topk=topk, experts=8, normalize=normalize, scale=5.0)
    )["decoder"].graph
    router = next(node for node in graph if node.op_type == "MoEGroupRouter")
    consumer = next(node for node in graph if node.op_type == "MoEPacked")
    preparation = ir.convenience.extract(
        graph, inputs=list(router.outputs[1:]), outputs=[consumer.inputs[5]]
    )
    path = tmp_path / "mixing.onnx"
    ir.save(ir.Model(preparation, ir_version=10), path)
    session = ort.InferenceSession(str(path), providers=[provider])
    assert session.get_providers()[0] == provider
    scores = np.array(
        [
            [30, 29.577760696411133, 29.577760696411133, 4, 3, 2, 1, 0],
            [0, 1, 2, 3, 4, 5, 6, 7],
            [7, 6, 5, 4, 3, 2, 1, 0],
        ],
        np.float32,
    )
    mixing = np.array(
        [
            [0.8, 0.2, 0.6, 0.9, 0.3, 0.4, 0.5, 0.1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 2**-24, 2**-24, 2**-24, 2**-24, 2**-24, 2**-24, 2**-24],
        ],
        np.float32,
    )
    actual = session.run(
        None, dict(zip([value.name for value in preparation.inputs], [scores, mixing]))
    )[0]
    chosen = np.argsort(-scores, axis=-1, kind="stable")[:, :topk]
    expected = mixing
    if normalize:
        selected = np.take_along_axis(mixing, chosen, axis=-1)
        total = selected[:, :1].copy()
        for rank in range(1, topk):
            total = total + selected[:, rank : rank + 1]
        expected = expected / (total + np.float32(1e-20))
    expected = expected * np.float32(5.0)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()


@pytest.mark.parametrize("old_api", ["1", "2"])
def test_compact_decoder_rejects_observed_routes_and_old_packed_abi(old_api):
    package = _fixture()
    graph = package["decoder"].graph
    topk = next(node for node in graph if node.op_type == "TopK")
    graph.outputs.append(topk.outputs[1])
    with pytest.raises(ValueError, match="observed routing"):
        optimize_nemotron_h_omni_cuda(package)
    graph.outputs.pop()
    package["decoder"].metadata_props["mobius.bolt.packed_cuda"] = old_api
    with pytest.raises(ValueError, match="Older packed CUDA API"):
        optimize_nemotron_h_omni_cuda(package)


@pytest.mark.parametrize("changed", ["biased_mixing", "ascending_topk"])
def test_compact_decoder_rejects_changed_routing_without_mutating_source(changed):
    package = _fixture()
    graph = package["decoder"].graph
    topk = next(node for node in graph if node.op_type == "TopK")
    if changed == "ascending_topk":
        topk.attributes["largest"] = ir.AttrInt64("largest", 0)
    else:
        gather = next(
            use.node for use in topk.outputs[1].uses() if use.node.op_type == "GatherElements"
        )
        gather.replace_input_with(0, topk.inputs[0])
    before = list(graph)
    with pytest.raises(ValueError, match=r"configured top-k|unbiased sigmoid"):
        optimize_nemotron_h_omni_cuda(package)
    assert list(graph) == before
