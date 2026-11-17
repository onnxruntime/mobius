# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Normal-export compact functions preserve graph semantics and parameter bindings."""

from __future__ import annotations

import hashlib

import numpy as np
import onnx
import onnx_ir as ir
import pytest
from onnx_ir.passes import common

from mobius._builder import build_from_module
from mobius._flags import override_flags
from mobius._model_package import ModelPackage
from mobius._optimizations import optimize_model, validate_standard_onnx
from mobius._testing.ort_inference import OnnxModelSession
from mobius.functions._lowering_test_utils import expand_grouped_matmul
from mobius.functions.linear_attention import linear_attention
from mobius.models.qwen4_exp import Qwen4ExpCausalLMModel, Qwen4ExpForConditionalGeneration
from mobius.models.qwen4_exp_test import _config, _initial_states, _vl_config

pytestmark = pytest.mark.opset29


@pytest.fixture(autouse=True)
def _require_flash_next_schema():
    if "v_num_heads" not in onnx.defs.get_schema("LinearAttention", 29).attributes:
        pytest.skip("Requires opset-29 LinearAttention schemas with v_num_heads support")


def _fingerprint(model: ir.Model) -> tuple:
    """Hash the full dataflow, ignoring generated names and constant storage layout."""
    cache: dict[ir.Value, str] = {}

    def tensor_key(tensor):
        return (
            str(tensor.dtype),
            tuple(tensor.shape),
            hashlib.sha256(tensor.tobytes()).hexdigest(),
        )

    def graph_key(graph):
        for index, value in enumerate(graph.inputs):
            cache[value] = repr(("input", index))
        for name, value in graph.initializers.items():
            cache[value] = repr(
                ("parameter", name, str(value.dtype), str(value.shape))
                if value.const_value is None
                else ("constant", tensor_key(value.const_value))
            )
        return tuple(value_key(value) for value in graph.outputs)

    def value_key(value):
        if value is None:
            return "absent"
        if value in cache:
            return cache[value]
        node = value.producer()
        assert node is not None, value
        if node.op_type == "Constant":
            result = repr(("constant", tensor_key(value.const_value)))
        else:
            attrs = []
            for name, attr in sorted(node.attributes.items()):
                if attr.type == ir.AttributeType.GRAPH:
                    data = graph_key(attr.as_graph())
                elif attr.type == ir.AttributeType.GRAPHS:
                    data = tuple(graph_key(graph) for graph in attr.as_graphs())
                elif attr.type == ir.AttributeType.TENSOR:
                    data = tensor_key(attr.as_tensor())
                elif attr.type == ir.AttributeType.FLOAT:
                    data = np.float32(attr.as_float()).tobytes()
                elif attr.type == ir.AttributeType.FLOATS:
                    data = np.asarray(attr.as_floats(), dtype=np.float32).tobytes()
                else:
                    data = str(attr)
                attrs.append((name, data))
            result = hashlib.sha256(
                repr(
                    (
                        node.op_identifier(),
                        tuple(value_key(input_) for input_ in node.inputs),
                        attrs,
                    )
                ).encode()
            ).hexdigest()
            result += f":{list(node.outputs).index(value)}"
        cache[value] = result
        return result

    return graph_key(model.graph)


@pytest.mark.parametrize(
    "multimodal,dtype",
    [
        (False, ir.DataType.FLOAT),
        (False, ir.DataType.FLOAT16),
        (False, ir.DataType.BFLOAT16),
        (True, ir.DataType.FLOAT),
        (True, ir.DataType.FLOAT16),
    ],
)
def test_compact_normal_export_roundtrip_and_inline_topology(tmp_path, multimodal, dtype):
    packages = []
    for compact in (False, True):
        config = (_vl_config if multimodal else _config)(
            dtype=dtype, num_experts_per_tok=2, output_gate_type="sigmoid"
        )
        module = (Qwen4ExpForConditionalGeneration if multimodal else Qwen4ExpCausalLMModel)(
            config
        )
        with override_flags(experimental_opset29=True, compact_standard_functions=compact):
            packages.append(
                build_from_module(
                    module,
                    config,
                    task="qwen4-exp-vision-language"
                    if multimodal
                    else "qwen4-exp-text-generation",
                    execution_provider="onnx-standard",
                )
            )
    plain, compact = packages
    for name in plain:
        before, after = plain[name], compact[name]
        assert not before.functions
        if name != "embedding":
            assert after.functions
            assert len(after.graph) < len(before.graph)

        def parameters(model):
            return {
                name: (value.dtype, tuple(value.shape))
                for name, value in model.graph.initializers.items()
                if value.const_value is None
            }

        assert parameters(before) == parameters(after)
        assert [(v.name, v.dtype, str(v.shape)) for v in before.graph.inputs] == [
            (v.name, v.dtype, str(v.shape)) for v in after.graph.inputs
        ]
        assert [(v.name, v.dtype, str(v.shape)) for v in before.graph.outputs] == [
            (v.name, v.dtype, str(v.shape)) for v in after.graph.outputs
        ]
        for model in (before, after):
            for value in list(model.graph.initializers.values()):
                if value.const_value is None:
                    model.graph.inputs.append(value)
                    del model.graph.initializers[value.name]
        validate_standard_onnx(after)
    compact.save(str(tmp_path), progress_bar=False)
    restored = ModelPackage.load(str(tmp_path))
    for name, model in restored.items():
        assert set(model.functions) == set(compact[name].functions)
        validate_standard_onnx(model)
        path = tmp_path / "model.onnx" if name == "model" else tmp_path / name / "model.onnx"
        onnx.checker.check_model(str(path), full_check=True)
        onnx.shape_inference.infer_shapes(
            ir.serde.serialize_model(model), strict_mode=True, data_prop=True
        )
        common.InlinePass()(model)
        assert not model.functions
        # Normalize constant storage/CSE on both sides after public inlining.
        with override_flags(experimental_opset29=True, compact_standard_functions=False):
            for candidate in (plain[name], model):
                optimize_model(candidate, ep="onnx-standard", dtype=dtype)
        assert _fingerprint(model) == _fingerprint(plain[name])


def test_compact_grouped_decoder_prefill_and_reused_states():
    config = _config(num_experts_per_tok=2, output_gate_type="sigmoid")
    models = []
    for compact in (False, True):
        with override_flags(experimental_opset29=True, compact_standard_functions=compact):
            model = build_from_module(
                Qwen4ExpCausalLMModel(config),
                config,
                task="qwen4-exp-text-generation",
                execution_provider="onnx-standard",
            )["model"]
        common.InlinePass()(model)
        # Test-only expansion: released ORT does not implement the new standard
        # recurrence yet. Use Mobius's reference body before the existing
        # GroupedMatMul test-copy lowering; production retains both standard ops.
        recurrence = linear_attention(
            q_num_heads=config.linear_num_key_heads,
            kv_num_heads=config.linear_num_key_heads,
            v_num_heads=config.linear_num_value_heads,
            scale=config.linear_key_head_dim**-0.5,
            stash_type=ir.DataType.FLOAT,
            opset_version=29,
            domain="test.compact",
        )
        for node in model.graph:
            if node.op_type == "LinearAttention":
                node.domain = recurrence.domain
        model.graph.opset_imports[recurrence.domain] = 1
        model.functions[recurrence.identifier()] = recurrence
        common.InlinePass()(model)
        rng = np.random.default_rng(42)
        for name in sorted(model.graph.initializers):
            value = model.graph.initializers[name]
            if value.const_value is None:
                value.const_value = ir.tensor(
                    rng.normal(0, 0.02, tuple(value.shape)).astype(np.float32)
                )
        models.append(expand_grouped_matmul(model))
    sessions = [OnnxModelSession(model) for model in models]
    states = [_initial_states(), _initial_states()]
    try:
        for start, tokens in [(0, [2, 3, 4]), (3, [5, 6]), (5, [7])]:
            outputs = [
                session.run(
                    state
                    | {
                        "input_ids": np.asarray([tokens], dtype=np.int64),
                        "position_ids": np.arange(start, start + len(tokens))[None, :],
                        "attention_mask": np.ones((1, start + len(tokens)), dtype=np.int64),
                    }
                )
                for session, state in zip(sessions, states, strict=True)
            ]
            for name in outputs[0]:
                np.testing.assert_array_equal(outputs[0][name], outputs[1][name])
            states = [
                {
                    input_name: output[
                        "present_position_ids"
                        if input_name == "past_position_ids"
                        else input_name.replace("past_key_values.", "present.")
                    ]
                    for input_name in state
                }
                for output, state in zip(outputs, states, strict=True)
            ]
    finally:
        for session in sessions:
            session.close()
