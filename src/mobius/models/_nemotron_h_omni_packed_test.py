# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Packed storage, fail-closed rewrites and unchanged public decoder contracts."""

from __future__ import annotations

import dataclasses

import numpy as np
import onnx
import onnx_ir as ir
import pytest

from mobius import build_from_module
from mobius._configs import NemotronHOmniConfig
from mobius._configs.per_model._nemotron_h_omni_test import _omni_config
from mobius.models import NemotronHOmniModel
from mobius.models._nemotron_h_omni_packed import (
    _ggml_parts,
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
    assert sum(n.op_type == "MoEPackedIQ2XXS" for n in after.graph.all_nodes()) == 2
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
