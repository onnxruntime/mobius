# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Packed hybrid metadata must describe pages and fixed state independently."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import jsonschema
import numpy as np
import onnx_ir as ir
import onnx_ir.passes.common as common_passes
import onnxruntime as ort
import pytest

from mobius import build_from_module
from mobius._testing import make_config
from mobius.integrations._paged_hybrid import inspect_paged_hybrid
from mobius.integrations.onnx_genai.workflow_metadata import build_decoder_workflow_metadata
from mobius.integrations.ort_genai.auto_export import _inspect_decoder_abi, _write_genai_config
from mobius.models.qwen35 import Qwen35CausalLMModel
from mobius.tasks import HybridCausalLMTask, PagedHybridCausalLMTask


def tiny_config(dtype=ir.DataType.FLOAT16):
    return make_config(
        model_type="qwen3_5_text",
        dtype=dtype,
        hidden_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        intermediate_size=192,
        vocab_size=128,
        max_position_embeddings=2048,
        num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        partial_rotary_factor=0.5,
        mrope_section=[8, 4, 4],
        mrope_interleaved=True,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=64,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
    )


@pytest.fixture
def package():
    config = tiny_config()
    return build_from_module(
        Qwen35CausalLMModel(config),
        config,
        task=PagedHybridCausalLMTask(),
        execution_provider="cuda",
    )


def test_exact_state_disciplines_and_application_scheduling(package):
    abi = inspect_paged_hybrid(package["model"])
    assert abi is not None
    assert abi.full_layers == (3,)
    assert abi.linear_layers == (0, 1, 2)

    metadata = build_decoder_workflow_metadata(package, package.config)
    schema = json.loads(
        (Path(__file__).with_name("_schema") / "inference_metadata.schema.json").read_text()
    )
    jsonschema.validate(metadata, schema)
    workflow = metadata["pipeline"]["workflow"]
    assert workflow["steps"][0]["kind"] == "invoke"
    assert all(step["kind"] == "emit" for step in workflow["steps"][1:])
    assert all(
        value["source"]["kind"] == "application" for value in workflow["inputs"].values()
    )
    groups = workflow["serving"]["state_service"]["groups"]
    assert set(groups) == {"paged_kv", "fixed_conv", "fixed_recurrent"}
    assert groups["paged_kv"]["update"]["kind"] == "paged_scatter"
    assert groups["paged_kv"]["aliasing"] == "required"
    for name in ("fixed_conv", "fixed_recurrent"):
        assert groups[name]["update"] == {"kind": "replace"}
        assert "sequence_axis" not in groups[name]
    aliases = groups["paged_kv"]["ports"]["decoder"]
    assert {alias["layer"] for alias in aliases.values()} == {3}
    assert {alias["role"] for alias in aliases.values()} == {"key", "value"}
    for cell in workflow["state"].values():
        if cell["service_group"] == "paged_kv":
            assert "batch_layout" not in cell["contract"]
        else:
            assert cell["contract"]["batch_layout"]["kind"] == "request_aligned"


def test_genai_export_uses_verified_groups_and_no_capture(package, tmp_path):
    _write_genai_config(
        package.config,
        str(tmp_path),
        pkg=package,
        ort_model_type="decoder",
        context_length=2048,
        ep="cuda",
        bos_token_id=1,
        eos_token_id=127,
        pad_token_id=0,
        is_vlm=False,
        has_speech=False,
    )
    config = json.loads((tmp_path / "genai_config.json").read_text())
    decoder = config["model"]["decoder"]
    assert decoder["state_groups"] == [
        {"kind": "paged_kv", "layer_ids": [3]},
        {"kind": "fixed_conv", "layer_ids": [0, 1, 2]},
        {"kind": "fixed_recurrent", "layer_ids": [0, 1, 2]},
    ]
    assert decoder["inputs"]["past_conv_names"] == "past_key_values.%d.conv_state"
    assert decoder["inputs"]["past_recurrent_names"] == "past_key_values.%d.recurrent_state"
    assert config["engine"]["dynamic_batching"]["block_size"] == 256
    assert config["search"]["past_present_share_buffer"] is True
    assert (
        decoder["session_options"]["provider_options"][0]["cuda"].get("enable_cuda_graph", "0")
        != "1"
    )


def test_packed_export_precision_and_rotary_shapes(package):
    graph = package["model"].graph
    assert graph.outputs[0].dtype == ir.DataType.FLOAT
    for layer in range(3):
        for suffix in ("A_log", "dt_bias"):
            assert (
                graph.initializers[f"model.layers.{layer}.linear_attn.{suffix}"].dtype
                == ir.DataType.FLOAT
            )
        assert f"model.layers.{layer}.linear_attn.conv1d.weight" in graph.initializers
    assert not any(
        key[1]
        in {
            "GatedDeltaNet",
            "LinearAttention",
            "VarlenCausalConvWithState",
            "CausalConvWithState",
        }
        for key in package["model"].functions
    )
    assert not any(
        node.op_type in {"Attention", "LinearAttention", "CausalConvWithState"}
        for node in graph
    )


@pytest.mark.parametrize("mutation", ["missing_state", "bad_dtype", "bad_pages", "bad_node"])
def test_incomplete_hybrid_cannot_claim_engine_compatibility(package, mutation):
    model = package["model"]
    if mutation == "missing_state":
        model.graph.inputs.remove(
            next(v for v in model.graph.inputs if v.name.endswith("0.conv_state"))
        )
    elif mutation == "bad_dtype":
        model.graph.outputs[0].type = ir.TensorType(ir.DataType.FLOAT16)
    elif mutation == "bad_pages":
        model.metadata_props["mobius.paged_block_size"] = "512"
    else:
        node = next(node for node in model.graph if node.op_type == "PagedAttention")
        model.graph.remove(node, safe=False)
    with pytest.raises(ValueError):
        _inspect_decoder_abi(model, model_type="decoder")


def test_unmarked_block_table_is_not_hybrid_compatibility(package):
    del package["model"].metadata_props["mobius.paged_hybrid"]
    assert inspect_paged_hybrid(package["model"]) is None


def test_verifier_accepts_normal_saved_export(package, tmp_path):
    package.save(str(tmp_path), check_weights=False, progress_bar=False)

    assert inspect_paged_hybrid(ir.load(tmp_path / "model.onnx")) is not None


def test_verifier_accepts_unrelated_symbolic_request_dimensions(package):
    graph = package["model"].graph
    request_inputs = {
        value.name: value
        for value in graph.inputs
        if value.name
        in {"block_table", "past_sequence_lengths", "cumulative_sequence_lengths"}
        or value.name.endswith((".conv_state", ".recurrent_state"))
    }
    request_inputs["block_table"].shape = ir.Shape(["blocks_batch", "max_blocks"])
    request_inputs["past_sequence_lengths"].shape = ir.Shape(["lengths_batch"])
    request_inputs["cumulative_sequence_lengths"].shape = ir.Shape(["boundaries"])
    for name, state in request_inputs.items():
        if not name.endswith((".conv_state", ".recurrent_state")):
            continue
        state.shape = ir.Shape([f"{name}.batch", *state.shape[1:]])
        layer, kind = name.split(".")[1:3]
        present = next(
            value for value in graph.outputs if value.name == f"present.{layer}.{kind}"
        )
        present.shape = ir.Shape([f"present.{name}.batch", *state.shape[1:]])

    key = next(value for value in graph.inputs if value.name == "past_key_values.3.key")
    value = next(value for value in graph.inputs if value.name == "past_key_values.3.value")
    key.shape = ir.Shape(["key_pages", 256, "key_heads", "key_dim"])
    value.shape = ir.Shape(["value_pages", 256, "value_heads", "value_dim"])
    next(value for value in graph.outputs if value.name == "present.3.key").shape = ir.Shape(
        ["present_key_pages", 256, "present_key_heads", "present_key_dim"]
    )
    next(value for value in graph.outputs if value.name == "present.3.value").shape = ir.Shape(
        ["present_value_pages", 256, "present_value_heads", "present_value_dim"]
    )

    paged = next(node for node in graph if node.op_type == "PagedAttention")
    for index, name in enumerate(("query", "key", "value")):
        operand = paged.inputs[index]
        operand.shape = ir.Shape([f"{name}_tokens", operand.shape[1]])
    paged.outputs[0].shape = ir.Shape(["output_tokens", paged.outputs[0].shape[1]])

    assert inspect_paged_hybrid(package["model"]) is not None


@pytest.mark.parametrize(
    ("operand_index", "mutation"),
    [
        (0, "dtype"),
        (1, "dtype"),
        (2, "dtype"),
        (0, "rank"),
        (1, "rank"),
        (2, "rank"),
        (0, "width"),
        (1, "width"),
        (2, "width"),
    ],
)
def test_verifier_rejects_invalid_paged_attention_qkv(package, operand_index, mutation):
    node = next(node for node in package["model"].graph if node.op_type == "PagedAttention")
    operand = node.inputs[operand_index]
    assert operand is not None and operand.shape is not None
    if mutation == "dtype":
        operand.type = ir.TensorType(ir.DataType.FLOAT)
    elif mutation == "rank":
        operand.shape = ir.Shape([1, *operand.shape])
    else:
        operand.shape = ir.Shape([operand.shape[0], 7])

    with pytest.raises(ValueError, match="PagedAttention"):
        inspect_paged_hybrid(package["model"])


@pytest.mark.parametrize("mutation", ["dtype", "rank", "width", "disconnected"])
def test_verifier_rejects_invalid_paged_attention_data_output(package, mutation):
    node = next(node for node in package["model"].graph if node.op_type == "PagedAttention")
    output = node.outputs[0]
    assert output.shape is not None
    if mutation == "dtype":
        output.type = ir.TensorType(ir.DataType.FLOAT)
    elif mutation == "rank":
        output.shape = ir.Shape([1, *output.shape])
    elif mutation == "width":
        output.shape = ir.Shape([output.shape[0], 7])
    else:
        replacement = ir.Value(type=output.type, shape=output.shape)
        output.replace_all_uses_with(replacement)

    with pytest.raises(ValueError, match="PagedAttention data output"):
        inspect_paged_hybrid(package["model"])


@pytest.mark.parametrize("mutation", ["disconnected_key", "swapped_key_value"])
def test_verifier_rejects_paged_attention_output_misbinding(package, mutation):
    graph = package["model"].graph
    key = next(value for value in graph.outputs if value.name == "present.3.key")
    value = next(value for value in graph.outputs if value.name == "present.3.value")
    if mutation == "swapped_key_value":
        key.name, value.name = value.name, key.name
    else:
        key.name = "orphaned.present.3.key"
        graph.outputs.remove(key)
        graph.outputs.append(
            ir.Value(
                name="present.3.key",
                type=key.type,
                shape=key.shape,
            )
        )

    with pytest.raises(ValueError, match="PagedAttention"):
        inspect_paged_hybrid(package["model"])


@pytest.mark.parametrize(
    ("state_name", "native_op"),
    [
        ("present.0.conv_state", "VarlenCausalConvWithState"),
        ("present.0.recurrent_state", "GatedDeltaNet"),
    ],
)
def test_verifier_rejects_disconnected_linear_state_output(package, state_name, native_op):
    graph = package["model"].graph
    state = next(value for value in graph.outputs if value.name == state_name)
    state.name = f"orphaned.{state_name}"
    graph.outputs.remove(state)
    graph.outputs.append(ir.Value(name=state_name, type=state.type, shape=state.shape))

    with pytest.raises(ValueError, match=native_op):
        inspect_paged_hybrid(package["model"])


@pytest.mark.parametrize(
    ("op_type", "attribute", "replacement"),
    [
        (
            "VarlenCausalConvWithState",
            "activation",
            ir.AttrString("activation", "relu"),
        ),
        ("GatedDeltaNet", "gate_activation", ir.AttrString("gate_activation", "silu")),
        ("GatedDeltaNet", "scale", ir.AttrFloat32("scale", 1.0)),
    ],
)
def test_verifier_rejects_malformed_native_attributes(
    package, op_type, attribute, replacement
):
    node = next(node for node in package["model"].graph if node.op_type == op_type)
    node.attributes[attribute] = replacement

    with pytest.raises(ValueError, match=op_type):
        inspect_paged_hybrid(package["model"])


@pytest.mark.parametrize(
    ("op_type", "attribute", "replacement"),
    [
        ("PagedAttention", "scale", ir.AttrFloat32("scale", 0.5)),
        ("PagedAttention", "is_causal", ir.AttrInt64("is_causal", 0)),
        (
            "PagedAttention",
            "local_window_size",
            ir.AttrInt64("local_window_size", 128),
        ),
        ("PagedAttention", "softcap", ir.AttrFloat32("softcap", 30.0)),
        ("VarlenCausalConvWithState", "dilation", ir.AttrInt64("dilation", 2)),
        (
            "VarlenCausalConvWithState",
            "state_update_capacity",
            ir.AttrInt64("state_update_capacity", 1),
        ),
        ("GatedDeltaNet", "state_update_capacity", ir.AttrInt64("state_update_capacity", 1)),
    ],
)
def test_verifier_rejects_non_phase1_native_options(package, op_type, attribute, replacement):
    node = next(node for node in package["model"].graph if node.op_type == op_type)
    node.attributes[attribute] = replacement

    with pytest.raises(ValueError, match=op_type):
        inspect_paged_hybrid(package["model"])


@pytest.mark.parametrize(("mutation", "size"), [("input", 10), ("output", 3)])
def test_verifier_rejects_extended_gated_delta_contract(package, mutation, size):
    node = next(node for node in package["model"].graph if node.op_type == "GatedDeltaNet")
    if mutation == "input":
        node.resize_inputs(size)
    else:
        node.resize_outputs(size)

    with pytest.raises(ValueError, match="GatedDeltaNet"):
        inspect_paged_hybrid(package["model"])


@pytest.mark.parametrize("operand_index", [4, 5, 7, 8])
def test_verifier_rejects_non_float_gated_delta_operands(package, operand_index):
    node = next(node for node in package["model"].graph if node.op_type == "GatedDeltaNet")
    operand = node.inputs[operand_index]
    assert operand is not None
    operand.type = ir.TensorType(ir.DataType.FLOAT16)

    with pytest.raises(ValueError, match="GatedDeltaNet"):
        inspect_paged_hybrid(package["model"])


def test_verifier_requires_matching_gated_delta_qkv_dtypes(package):
    node = next(node for node in package["model"].graph if node.op_type == "GatedDeltaNet")
    key = node.inputs[1]
    assert key is not None
    key.type = ir.TensorType(ir.DataType.BFLOAT16)

    with pytest.raises(ValueError, match="GatedDeltaNet QKV layout"):
        inspect_paged_hybrid(package["model"])


@pytest.mark.parametrize("operand_index", [2, 4, 5])
def test_verifier_requires_matching_gated_delta_token_extents(package, operand_index):
    node = next(node for node in package["model"].graph if node.op_type == "GatedDeltaNet")
    for index in (0, 1, 2, 4, 5):
        value = node.inputs[index]
        assert value is not None and value.shape is not None
        value.shape = ir.Shape([8, *value.shape[1:]])
    node.outputs[0].shape = ir.Shape([8, *node.outputs[0].shape[1:]])
    operand = node.inputs[operand_index]
    assert operand is not None and operand.shape is not None
    operand.shape = ir.Shape([7, *operand.shape[1:]])

    with pytest.raises(ValueError, match="GatedDeltaNet"):
        inspect_paged_hybrid(package["model"])


def test_verifier_requires_depthwise_varlen_conv_weight(package):
    node = next(
        node for node in package["model"].graph if node.op_type == "VarlenCausalConvWithState"
    )
    weight = node.inputs[1]
    assert weight is not None and weight.shape is not None
    weight.shape = ir.Shape([weight.shape[0], 2, weight.shape[2]])

    with pytest.raises(ValueError, match="VarlenCausalConvWithState"):
        inspect_paged_hybrid(package["model"])


@pytest.mark.parametrize(
    "target",
    [
        "past_sequence_lengths",
        "conv_state",
        "recurrent_state",
        "cumulative_sequence_lengths",
    ],
)
def test_verifier_rejects_concrete_request_dimension_mismatch(package, target):
    graph = package["model"].graph
    inputs = {value.name: value for value in graph.inputs}
    inputs["block_table"].shape = ir.Shape([2, inputs["block_table"].shape[1]])
    inputs["past_sequence_lengths"].shape = ir.Shape([2])
    inputs["cumulative_sequence_lengths"].shape = ir.Shape([3])
    for layer in range(3):
        for kind in ("conv_state", "recurrent_state"):
            state = inputs[f"past_key_values.{layer}.{kind}"]
            state.shape = ir.Shape([2, *state.shape[1:]])
            present = next(
                value for value in graph.outputs if value.name == f"present.{layer}.{kind}"
            )
            present.shape = state.shape

    if target in {"past_sequence_lengths", "cumulative_sequence_lengths"}:
        inputs[target].shape = ir.Shape([4])
    else:
        state = inputs[f"past_key_values.0.{target}"]
        state.shape = ir.Shape([4, *state.shape[1:]])
        present = next(value for value in graph.outputs if value.name == f"present.0.{target}")
        present.shape = state.shape

    with pytest.raises(ValueError, match=r"request-aligned|batch \+ 1"):
        inspect_paged_hybrid(package["model"])


def test_verifier_rejects_cross_layer_recurrent_state_binding(package):
    nodes = [node for node in package["model"].graph if node.op_type == "GatedDeltaNet"]
    assert len(nodes) >= 2
    first_state, second_state = nodes[0].inputs[6], nodes[1].inputs[6]
    nodes[0].replace_input_with(6, second_state)
    nodes[1].replace_input_with(6, first_state)

    with pytest.raises(ValueError, match=r"state output|cross-bound"):
        inspect_paged_hybrid(package["model"])


@pytest.mark.parametrize("ep", ["default", "cpu", "dml"])
def test_direct_task_rejects_non_cuda(ep):
    config = tiny_config()
    with pytest.raises(ValueError, match="requires execution_provider"):
        build_from_module(
            Qwen35CausalLMModel(config),
            config,
            PagedHybridCausalLMTask(),
            execution_provider=ep,
        )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"dtype": ir.DataType.FLOAT}, "float16/bfloat16"),
        ({"model_type": "llama"}, "text-only"),
        ({"mrope_section": None}, "interleaved MRoPE"),
        ({"mrope_interleaved": False}, "interleaved MRoPE"),
        ({"layer_types": ["full_attention"] * 4}, "layer_types"),
    ],
)
def test_direct_task_rejects_incompatible_contracts(overrides, match):
    config = dataclasses.replace(tiny_config(), **overrides)
    with pytest.raises(ValueError, match=match):
        build_from_module(
            Qwen35CausalLMModel(config),
            config,
            PagedHybridCausalLMTask(),
            execution_provider="cuda",
        )


def test_paged_flag_cannot_be_combined_with_a_dense_task():
    config = dataclasses.replace(tiny_config(), export_paged_attention=True)
    with pytest.raises(ValueError, match="requires PagedHybridCausalLMTask"):
        build_from_module(
            Qwen35CausalLMModel(config),
            config,
            HybridCausalLMTask(),
            execution_provider="cuda",
        )


def test_64_layer_manifest_matches_native_operator_topology():
    config = dataclasses.replace(
        tiny_config(),
        num_hidden_layers=64,
        layer_types=tiny_config().layer_types * 16,
    )
    package = build_from_module(
        Qwen35CausalLMModel(config),
        config,
        PagedHybridCausalLMTask(),
        execution_provider="cuda",
    )
    abi = inspect_paged_hybrid(package["model"])
    assert abi.full_layers == tuple(range(3, 64, 4))
    assert len(abi.linear_layers) == 48
    nodes = list(package["model"].graph)
    assert sum(node.op_type == "PagedAttention" for node in nodes) == 16
    assert sum(node.op_type == "VarlenCausalConvWithState" for node in nodes) == 48
    assert sum(node.op_type == "GatedDeltaNet" for node in nodes) == 48


def test_dense_gate_parameters_still_follow_compute_dtype():
    from mobius._builder import _cast_module_dtype

    config = dataclasses.replace(tiny_config(), export_paged_attention=False)
    module = Qwen35CausalLMModel(config)
    _cast_module_dtype(module, config.dtype)
    parameters = dict(module.named_parameters())
    assert parameters["model.layers.0.linear_attn.A_log"].dtype == config.dtype
    assert parameters["model.layers.0.linear_attn.dt_bias"].dtype == config.dtype


def test_packed_qk_norm_and_interleaved_mrope_match_dense_on_cpu(tmp_path):
    """Execute the real projection/RoPE subgraphs without requiring paged kernels."""
    config = dataclasses.replace(
        tiny_config(),
        num_hidden_layers=2,
        layer_types=["full_attention", "linear_attention"],
    )
    sessions = []
    for packed in (False, True):
        module = Qwen35CausalLMModel(config)
        rng = np.random.default_rng(51)
        for _, parameter in module.named_parameters():
            if parameter.const_value is None:
                parameter.const_value = ir.tensor(
                    rng.normal(0, 0.1, tuple(parameter.shape)).astype(np.float32)
                )
        package = build_from_module(
            module,
            config,
            PagedHybridCausalLMTask() if packed else HybridCausalLMTask(),
            execution_provider="cuda" if packed else "cpu",
        )
        model = package["model"]
        attention = next(
            node
            for node in model.graph
            if node.op_type == ("PagedAttention" if packed else "Attention")
        )
        model.graph.outputs.clear()
        for index, value in enumerate(attention.inputs[:3]):
            value.name = f"projected_{index}"
            model.graph.outputs.append(value)
        common_passes.RemoveUnusedNodesPass()(model)
        common_passes.RemoveUnusedFunctionsPass()(model)
        # Production cleanup retains cache ports as an ABI promise. This
        # projection-only test has deliberately removed that ABI.
        for value in list(model.graph.inputs):
            if not value.uses():
                model.graph.inputs.remove(value)
        path = tmp_path / f"projections_{packed}.onnx"
        ir.save(model, path)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sessions.append(
            ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        )
    tokens = np.asarray([[4, 7, 9], [12, 5, 3]], np.int64)
    positions = np.asarray([[17, 18, 19], [41, 42, 43]], np.int64)
    dense = sessions[0].run(None, {"input_ids": tokens, "position_ids": positions})
    packed = sessions[1].run(
        None, {"input_ids": tokens.ravel(), "position_ids": np.tile(positions.ravel(), (3, 1))}
    )
    for reference, actual in zip(dense, packed):
        # Attention accepts both (B,H,S,D) and (B,S,H*D).
        if reference.ndim == 4:
            reference = reference.transpose(0, 2, 1, 3)
        reference = reference.reshape(actual.shape)
        np.testing.assert_allclose(actual, reference, atol=2e-3, rtol=2e-3)
