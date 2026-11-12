# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Build, registration, ownership, and inlining contracts independent of runtime support."""

from __future__ import annotations

import dataclasses
import hashlib

import onnx
import onnx_ir as ir
import pytest
from onnx_ir.passes import PreconditionError
from onnx_ir.passes.common import InlinePass
from onnxscript._internal.builder import build_function

from mobius._build_context import build_context, get_build_opset_version
from mobius._execution_providers import EpCapabilities, ep_registry
from mobius._flags import override_flags
from mobius._optimizations import _inline_function_bodies, optimize_model
from mobius._testing import create_test_input, make_config
from mobius.components import (
    MoELayer,
    SigmoidTopKGate,
    SoftmaxTopKGate,
    SparseMixerGate,
    TopKGate,
    create_body_graph,
)
from mobius.functions import (
    _config_identity,
    causal_conv1d_with_state,
    causal_conv_nd_with_state,
    get_function,
    linear_attention,
    matmul_block_quantized_fp4_weight,
    matmul_nbits,
    moe,
    packed_multi_head_attention,
    qmoe,
    register_function_bodies,
    skip_layer_normalization,
    skip_simplified_layer_normalization,
)
from mobius.functions.matmul_nbits import (
    FUNCTION_BODY_METADATA_KEY,
    FUNCTION_BODY_METADATA_VALUE,
)
from mobius.functions.moe import validate_call as validate_moe
from mobius.functions.moe_test import _model as float_model
from mobius.functions.qmoe import validate_call as validate_qmoe
from mobius.functions.qmoe_test import _model as quantized_model
from mobius.tasks._base import _make_graph

requires_schema = pytest.mark.skipif(
    not onnx.defs.has("GroupedMatMul", 29), reason="GroupedMatMul-29 schemas unavailable"
)


@pytest.fixture(autouse=True)
def experimental():
    with override_flags(experimental_opset29=True):
        yield


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("native", [False, True])
def test_build_opset_selected_before_builder_and_snapshotted(enabled, native):
    caps = EpCapabilities(name="test", supports_fused_moe=native)
    expected = 29 if enabled and not native else 24
    assert get_build_opset_version() == 24
    with override_flags(experimental_opset29=enabled), build_context(caps):
        graph, builder = _make_graph()
        assert graph.opset_imports[""] == expected
        assert (
            builder.op.Identity(builder.op.Constant(value_int=1)).producer().version
            == expected
        )
        body, _ = create_body_graph([], [], "positional_name")
        assert body.opset_imports[""] == expected
        explicit, _ = create_body_graph([], [], opset_version=24)
        assert explicit.opset_imports[""] == 24
        with override_flags(experimental_opset29=not enabled):
            assert get_build_opset_version() == expected

        def fail_in_nested_build():
            with build_context(ep_registry.require("cpu")):
                assert get_build_opset_version() == 24
                raise RuntimeError("restore")

        with pytest.raises(RuntimeError, match="restore"):
            fail_in_nested_build()
        assert get_build_opset_version() == expected
    assert get_build_opset_version() == 24


@pytest.mark.parametrize("version", [24, 29])
@pytest.mark.parametrize(
    "factory",
    [
        matmul_nbits,
        packed_multi_head_attention,
        skip_layer_normalization,
        skip_simplified_layer_normalization,
        matmul_block_quantized_fp4_weight,
    ],
)
def test_ordinary_factories_create_at_requested_opset(factory, version):
    function = factory(opset_version=version)
    assert function.opset_imports[""] == version
    assert all(node.version == version for node in function.graph.all_nodes())
    assert factory().opset_imports[""] == 24
    assert function is not factory(opset_version=version)


@pytest.mark.parametrize("version", [24, 29])
def test_parametric_factories_and_recurrence_are_created_at_requested_opset(version):
    for factory in (causal_conv_nd_with_state, causal_conv1d_with_state):
        fn = factory(kernel_size=3, channels=4, opset_version=version)
        assert fn.opset_imports[""] == version
        assert all(node.version == version for node in fn)
    fn = linear_attention(q_num_heads=2, kv_num_heads=2, opset_version=version)
    assert fn.opset_imports[""] == version
    assert all(node.version == version for node in fn.graph.all_nodes())
    assert all(graph.opset_imports[""] == version for graph in fn.graph.subgraphs())


@pytest.mark.parametrize("version", [24, 29])
def test_registry_constructs_fresh_bodies_at_root_version(version):
    model = ir.Model(ir.Graph([], [], nodes=[], opset_imports={"": version}), ir_version=10)
    register_function_bodies(model)
    assert all(fn.opset_imports[""] == version for fn in model.functions.values())
    fn = get_function(("com.microsoft", "MatMulNBits", ""), opset_version=version)
    assert fn is not model.functions[fn.identifier()]
    assert fn.metadata_props[FUNCTION_BODY_METADATA_KEY] == FUNCTION_BODY_METADATA_VALUE
    assert fn.opset_imports[""] == version


@pytest.mark.parametrize("model_version,function_version", [(24, 29), (29, 24), (29, 29)])
@pytest.mark.parametrize("in_function", [False, True])
def test_public_inliner_rejects_both_import_mismatch_directions(
    model_version, function_version, in_function
):
    fn = build_function(
        lambda op, x: op.Identity(x),
        [ir.Value(name="x")],
        domain="test",
        name="Identity",
        opset_imports={"": function_version},
    )
    x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2]))
    call = ir.Node(
        "test", "Identity", [x], outputs=[ir.Value(name="y", type=x.type, shape=x.shape)]
    )
    graph = ir.Graph(
        [x], call.outputs, nodes=[call], opset_imports={"": model_version, "test": 1}
    )
    functions = [fn]
    root = graph
    if in_function:
        functions.append(ir.Function("test", "Outer", graph=graph, attributes=[]))
        root = ir.Graph([], [], nodes=[], opset_imports={"": model_version, "test": 1})
    model = ir.Model(root, functions=functions, ir_version=10)
    inline = InlinePass(criteria=lambda function: function.name == "Identity")
    if model_version != function_version:
        with pytest.raises(ValueError, match="Opset mismatch"):
            inline(model)
    else:
        inline(model)
        assert next(iter(graph)).domain == ""


@pytest.mark.parametrize("make_model", [float_model, quantized_model])
def test_24_target_rejection_is_atomic_without_schema_requirement(make_model):
    model = make_model(opset=24)
    node = next(iter(model.graph))
    inputs, attributes = tuple(node.inputs), dict(node.attributes)
    with pytest.raises(ValueError, match="rebuild under experimental opset29"):
        register_function_bodies(model, moe_fallback=True)
    assert tuple(node.inputs) == inputs and dict(node.attributes) == attributes
    assert node.overload == "" and not model.functions
    assert model.graph.opset_imports[""] == 24


@pytest.mark.parametrize("unused_leaf", [False, True])
@pytest.mark.parametrize("selected", [False, True])
def test_inline_preserves_public_cycle_precondition_with_unused_leaf(unused_leaf, selected):
    formal = ir.Value(name="z")
    recursive = ir.Node("test", "Loop", [formal], outputs=[ir.Value(name="out")])
    function = ir.Function(
        "test",
        "Loop",
        graph=ir.Graph(
            [formal],
            recursive.outputs,
            nodes=[recursive],
            opset_imports={"": 24, "test": 1},
        ),
        attributes=[],
    )
    x = ir.Value(name="x")
    call = ir.Node("test", "Loop", [x], outputs=[ir.Value(name="y")])
    model = ir.Model(
        ir.Graph([x], call.outputs, nodes=[call], opset_imports={"": 24, "test": 1}),
        functions=[function],
        ir_version=10,
    )
    if unused_leaf:
        a = ir.Value(name="a")
        unused = ir.Function(
            "test",
            "Unused",
            graph=ir.Graph([a], [a], nodes=[], opset_imports={"": 24}),
            attributes=[],
        )
        model.functions[unused.identifier()] = unused
    with pytest.raises(PreconditionError, match="Cyclic dependency detected"):
        InlinePass(criteria=lambda function: selected)(model)
    with pytest.raises(PreconditionError, match="Cyclic dependency detected"):
        _inline_function_bodies(model, lambda function: selected)


@pytest.mark.parametrize("make_model", [float_model, quantized_model])
@pytest.mark.parametrize("enabled", [False, True])
def test_native_capable_optimizer_does_not_validate_restricted_fallback_abi(
    make_model, enabled
):
    model = make_model(opset=24, dtype=ir.DataType.BFLOAT16)
    node = next(iter(model.graph))
    node.attributes["activation_type"] = ir.AttrString("activation_type", "geglu")
    if node.op_type == "QMoE":
        node.attributes["quant_type"] = ir.AttrString("quant_type", "fp4")
    inputs, attributes = tuple(node.inputs), dict(node.attributes)
    with override_flags(experimental_opset29=enabled):
        optimize_model(model, ep="cpu")
    assert node.graph is model.graph and node.overload == ""
    assert tuple(node.inputs) == inputs and dict(node.attributes) == attributes
    assert model.graph.opset_imports[""] == 24
    assert not any(fn.name in {"MoE", "QMoE"} for fn in model.functions.values())


@pytest.mark.parametrize("make_model", [float_model, quantized_model])
def test_flag_off_fallback_ep_keeps_native_calls(make_model):
    model = make_model(opset=24)
    node = next(iter(model.graph))
    with override_flags(experimental_opset29=False):
        optimize_model(model, ep="default")
    assert node.graph is model.graph and node.overload == ""
    assert model.graph.opset_imports[""] == 24


@pytest.mark.parametrize("make_model", [float_model, quantized_model])
@pytest.mark.parametrize(
    "failure,message",
    [
        ("abi", "k must"),
        ("output", "output metadata"),
        ("scope", "containing function must use opset 29"),
        ("source", "caller-owned source"),
        ("destination", "conflicting destination"),
    ],
)
def test_all_targets_validated_before_schema_or_body_construction(
    make_model, failure, message, monkeypatch, tmp_path
):
    model = make_model()
    first = next(iter(model.graph))
    first.outputs[0].type = first.outputs[0].shape = None
    if failure == "scope":
        inner = make_model(opset=24).graph
        inner.inputs.extend(inner.initializers.values())
        inner.initializers.clear()
        function = ir.Function("test", "Experts", graph=inner, attributes=[])
        model.functions[function.identifier()] = function
    else:
        second = ir.Node(
            first.domain,
            first.op_type,
            first.inputs,
            outputs=[ir.Value(name="second")],
            attributes=first.attributes,
        )
        second.attributes["k"] = ir.AttrInt64("k", 1)
        model.graph.append(second)
        if failure == "abi":
            second.attributes["k"] = ir.AttrInt64("k", 0)
        elif failure == "output":
            second.outputs[0].dtype = ir.DataType.BFLOAT16
        else:
            if failure == "source":
                overload = second.overload = "caller"
            else:
                validate = validate_moe if second.op_type == "MoE" else validate_qmoe
                identity = _config_identity(validate(second))
                overload = "mobius_" + hashlib.sha256(identity.encode()).hexdigest()
            function = ir.Function(
                first.domain,
                first.op_type,
                overload=overload,
                graph=ir.Graph([], [], nodes=[], opset_imports={"": 29}),
                attributes=[],
            )
            model.functions[function.identifier()] = function

    def unexpected_schema_or_build(*args, **kwargs):
        pytest.fail("All targets must be validated before schema checks or body construction")

    monkeypatch.setattr(onnx.defs, "has", unexpected_schema_or_build)
    monkeypatch.setattr("mobius.functions.moe", unexpected_schema_or_build)
    monkeypatch.setattr("mobius.functions.qmoe", unexpected_schema_or_build)
    before, after = tmp_path / "before.onnx", tmp_path / "after.onnx"
    ir.save(model, before)
    with pytest.raises(ValueError, match=message):
        register_function_bodies(model, moe_fallback=True)
    ir.save(model, after)
    assert before.read_bytes() == after.read_bytes()
    assert first.overload == ""
    assert first.outputs[0].type is None and first.outputs[0].shape is None


@requires_schema
@pytest.mark.parametrize(
    "make_model,validate,factory,arity",
    [
        (float_model, validate_moe, moe, 8),
        (quantized_model, validate_qmoe, qmoe, 15),
    ],
)
def test_factory_formals_and_complete_config_identity(make_model, validate, factory, arity):
    model = make_model()
    node = next(iter(model.graph))
    config = validate(node)
    fn = factory(config)
    assert len(fn.inputs) == arity and fn.opset_imports[""] == 29
    assert all(n.version == 29 for n in fn)
    assert len(next(n for n in fn if n.op_type == "TopK").outputs) == 2
    assert dataclasses.is_dataclass(config)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.k = 1
    register_function_bodies(model, moe_fallback=True)
    first = node.overload
    assert first.startswith("mobius_")
    register_function_bodies(model, moe_fallback=True)
    assert node.overload == first
    other = make_model(rank=3)
    register_function_bodies(other, moe_fallback=True)
    other_node = next(iter(other.graph))
    assert other_node.overload == first  # Includes the stable +infinity default.
    assert (
        model.functions[node.op_identifier()]
        is not other.functions[other_node.op_identifier()]
    )


@requires_schema
@pytest.mark.parametrize("make_model", [float_model, quantized_model])
def test_mixed_configs_masks_and_ownership_collisions_are_atomic(make_model):
    model = make_model()
    first = next(iter(model.graph))
    second_inputs = list(first.inputs)
    optional = 3 if first.op_type == "MoE" else 11
    second_inputs[optional] = None
    second = ir.Node(
        first.domain,
        first.op_type,
        second_inputs,
        outputs=[ir.Value(name="second")],
        attributes=first.attributes,
    )
    model.graph.append(second)
    original_inputs = [tuple(node.inputs) for node in (first, second)]
    register_function_bodies(model, moe_fallback=True)
    assert first.overload != second.overload
    assert [tuple(node.inputs) for node in (first, second)] == original_inputs
    owned = model.functions[second.op_identifier()]
    owned.metadata_props.pop(FUNCTION_BODY_METADATA_KEY)
    first.overload = second.overload = ""
    before_functions = dict(model.functions)
    with pytest.raises(ValueError, match="destination"):
        register_function_bodies(model, moe_fallback=True)
    assert first.overload == second.overload == ""
    assert dict(model.functions) == before_functions
    # Caller-owned source definitions are also never specialized or replaced.
    owned.overload = ""
    model.functions[(owned.domain, owned.name, "")] = owned
    with pytest.raises(ValueError, match="source"):
        register_function_bodies(model, moe_fallback=True)
    assert first.overload == second.overload == ""


@pytest.mark.parametrize("make_model", [float_model, quantized_model])
def test_containing_function24_rejected_even_with_child29_hint(make_model):
    inner = make_model().graph
    inner.inputs.extend(inner.initializers.values())
    inner.initializers.clear()
    condition = ir.Value(
        name="condition", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([])
    )
    # Loop/If graph hints are not serialized imports. A function's root owns them.
    branch_inputs = list(inner.inputs)
    inner.inputs.clear()
    otherwise_output = ir.Value(
        name="otherwise", type=branch_inputs[0].type, shape=branch_inputs[0].shape
    )
    otherwise = ir.Graph(
        [],
        [otherwise_output],
        nodes=[ir.Node("", "Identity", [branch_inputs[0]], outputs=[otherwise_output])],
        opset_imports={"": 29},
    )
    branch = ir.Node(
        "",
        "If",
        [condition],
        attributes=[
            ir.AttrGraph("then_branch", inner),
            ir.AttrGraph("else_branch", otherwise),
        ],
        outputs=[ir.Value(name="out")],
    )
    root = ir.Graph(
        [condition, *branch_inputs],
        branch.outputs,
        nodes=[branch],
        opset_imports={"": 24, "com.microsoft": 1},
    )
    fn = ir.Function("test", "Experts", graph=root, attributes=[])
    model = ir.Model(
        ir.Graph([], [], nodes=[], opset_imports={"": 29}), functions=[fn], ir_version=10
    )
    target = next(iter(inner))
    with pytest.raises(ValueError, match="containing function must use opset 29"):
        register_function_bodies(model, moe_fallback=True)
    assert target.overload == "" and tuple(model.functions) == (fn.identifier(),)


@requires_schema
def test_nested_function_expansion_preserves_moe_output_annotations():
    inner = float_model().graph
    inner.inputs.extend(inner.initializers.values())
    inner.initializers.clear()
    original = inner.outputs[0]
    original.doc_string = "nested result"
    original.metadata_props["annotation"] = "value"
    marker = object()
    original.meta["marker"] = marker
    fn = ir.Function("test", "Experts", graph=inner, attributes=[])
    inputs = [
        ir.Value(name=f"input_{i}", shape=v.shape, type=v.type)
        for i, v in enumerate(inner.inputs)
    ]
    output = ir.Value(
        name="result",
        shape=original.shape,
        type=original.type,
    )
    call_marker = object()
    output.meta["call_marker"] = call_marker
    call = ir.Node(fn.domain, fn.name, inputs, outputs=[output])
    model = ir.Model(
        ir.Graph(
            inputs,
            [output],
            nodes=[call],
            opset_imports={"": 29, "com.microsoft": 1, "test": 1},
        ),
        functions=[fn],
        ir_version=10,
    )
    register_function_bodies(model, moe_fallback=True)
    _inline_function_bodies(model, lambda function: function.domain not in {"", "ai.onnx"})
    assert not any(node.domain for node in model.graph.all_nodes())
    result = model.graph.outputs[0]
    assert result.name == "result" and result.doc_string == "nested result"
    assert result.metadata_props["annotation"] == "value"
    assert result.meta["marker"] is marker
    assert result.meta["call_marker"] is call_marker


@pytest.mark.parametrize("version", [24, 29])
def test_passthrough_function_preserves_call_output_annotations(version):
    formal = ir.Value(
        name="formal", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2])
    )
    function = ir.Function(
        "test",
        "Passthrough",
        graph=ir.Graph([formal], [formal], nodes=[], opset_imports={"": version}),
        attributes=[],
    )
    x = ir.Value(
        name="x",
        type=formal.type,
        shape=formal.shape,
        doc_string="input docs",
        metadata_props={"label": "input"},
    )
    input_marker = x.meta["marker"] = object()
    result = ir.Value(
        name="y",
        type=x.type,
        shape=x.shape,
        doc_string="call output",
        metadata_props={"label": "preserved"},
    )
    marker = result.meta["marker"] = object()
    call = ir.Node("test", "Passthrough", [x], outputs=[result])
    model = ir.Model(
        ir.Graph([x], [result], nodes=[call], opset_imports={"": version, "test": 1}),
        functions=[function],
        ir_version=10,
    )
    with override_flags(experimental_opset29=version == 29):
        _inline_function_bodies(model, lambda function: True)
    output = model.graph.outputs[0]
    assert output is not x
    assert x.name == "x" and x.doc_string == "input docs"
    assert x.metadata_props == {"label": "input"}
    assert x.meta["marker"] is input_marker
    assert [node.op_type for node in model.graph] == ["Identity"]
    assert output.name == "y" and output.doc_string == "call output"
    assert output.metadata_props["label"] == "preserved"
    assert output.meta["marker"] is marker
    assert output.type == x.type and output.shape == x.shape
    assert function.outputs[0] is formal and not list(function.graph)


def test_passthrough_shared_upstream_keeps_separate_output_boundaries():
    formal = ir.Value(name="formal")
    function = ir.Function(
        "test",
        "Passthrough",
        graph=ir.Graph([formal], [formal], nodes=[], opset_imports={"": 24}),
        attributes=[],
    )
    x = ir.Value(name="x")
    upstream = ir.Node(
        "", "Identity", [x], outputs=[ir.Value(name="upstream", doc_string="upstream docs")]
    )
    source = upstream.outputs[0]
    source.metadata_props["label"] = "source"
    source.meta["marker"] = object()
    calls = [
        ir.Node(
            "test",
            "Passthrough",
            [source],
            outputs=[
                ir.Value(name=f"y{i}", doc_string=f"docs{i}", metadata_props={"label": str(i)})
            ],
        )
        for i in range(2)
    ]
    model = ir.Model(
        ir.Graph(
            [x],
            [source, *(call.outputs[0] for call in calls)],
            nodes=[upstream, *calls],
            opset_imports={"": 24, "test": 1},
        ),
        functions=[function],
        ir_version=10,
    )
    _inline_function_bodies(model, lambda function: True)
    assert model.graph.outputs[0] is source
    assert source.name == "upstream" and source.doc_string == "upstream docs"
    assert source.metadata_props == {"label": "source"}
    for i, output in enumerate(model.graph.outputs[1:]):
        assert output is not source and output.name == f"y{i}"
        assert output.doc_string == f"docs{i}" and output.metadata_props["label"] == str(i)


@requires_schema
@pytest.mark.parametrize("make_model", [float_model, quantized_model])
def test_registered_not_inlined_save_load_reregister_is_stable(make_model, tmp_path):
    model = make_model(clipped=True)
    register_function_bodies(model, moe_fallback=True)
    original = next(iter(model.graph))
    overload, identifiers = original.overload, set(model.functions)
    path = tmp_path / "registered.onnx"
    ir.save(model, path)
    loaded = ir.load(path)
    call = next(iter(loaded.graph))
    assert call.domain == "com.microsoft" and call.overload == overload
    assert call.attributes.get_float("activation_alpha").hex() == (1.7000000476837158).hex()
    register_function_bodies(loaded, moe_fallback=True)
    assert call.overload == overload and set(loaded.functions) == identifiers
    # Compare executable copies, leaving both saved/registered graphs uninlined.
    import numpy as np

    from mobius._testing.ort_inference import OnnxModelSession
    from mobius.functions._lowering_test_utils import expand_grouped_matmul
    from mobius.functions._moe_test_utils import inline_native_calls
    from mobius.functions.moe_test import _feeds as float_feeds
    from mobius.functions.qmoe_test import _feeds as quantized_feeds

    feeds = float_feeds() if call.op_type == "MoE" else quantized_feeds()
    results = []
    for registered in (model, loaded):
        executable = registered.clone()
        inline_native_calls(executable)
        expand_grouped_matmul(executable)
        session = OnnxModelSession(executable, device="cpu")
        try:
            results.append(session.run(feeds)["y"])
        finally:
            session.close()
    assert np.isfinite(results[0]).all()
    np.testing.assert_array_equal(results[0], results[1])


@requires_schema
def test_unnamed_call_output_documentation_survives_inline_boundary():
    model = float_model()
    model.graph.inputs[0].name = "mobius_inline_output_0"
    original = model.graph.outputs[0]
    original.name = None
    original.doc_string = "unnamed output"
    original.metadata_props["annotation"] = "preserved"
    marker = object()
    original.meta["marker"] = marker
    register_function_bodies(model, moe_fallback=True)
    _inline_function_bodies(model, lambda function: function.name == "MoE")
    output = model.graph.outputs[0]
    assert output.name is not None
    assert output.name != model.graph.inputs[0].name
    assert output.doc_string == "unnamed output"
    assert output.metadata_props["annotation"] == "preserved"
    assert output.meta["marker"] is marker


@pytest.mark.parametrize(
    "gate_type", [TopKGate, SoftmaxTopKGate, SigmoidTopKGate, SparseMixerGate]
)
def test_generic_dense_gate_dispatch_is_unchanged_under_flag(gate_type):
    signatures = []
    for enabled in (False, True):
        config = make_config(
            hidden_size=16, intermediate_size=32, num_local_experts=3, num_experts_per_tok=2
        )
        gate = gate_type(hidden_size=16, num_experts=3, top_k=2)
        with (
            override_flags(experimental_opset29=enabled),
            build_context(ep_registry.require("onnx-standard")),
        ):
            graph, builder = _make_graph()
            hidden = create_test_input(builder, "hidden", [2, 3, 16])
            output = MoELayer(config, gate=gate)(builder.op, hidden)
            graph.outputs.append(output)
        signatures.append([node.op_identifier() for node in graph.all_nodes()])
        assert not any(node.op_type in {"MoE", "QMoE", "GroupedMatMul"} for node in graph)
    assert signatures[0] == signatures[1]


def test_qwen4_probability_dispatch_is_not_replaced_or_softmaxed_again():
    from mobius.models.qwen4_exp import Qwen4ExpExperts
    from mobius.models.qwen4_exp_test import _config

    signatures = []
    for enabled in (False, True):
        config = _config()
        with (
            override_flags(experimental_opset29=enabled),
            build_context(ep_registry.require("onnx-standard")),
        ):
            graph, builder = _make_graph()
            hidden = create_test_input(builder, "hidden", [2, 3, 16])
            probabilities = create_test_input(builder, "probabilities", [2, 3, 2])
            output = Qwen4ExpExperts(config)(builder.op, hidden, probabilities)
            graph.outputs.append(output)
        signatures.append([node.op_identifier() for node in graph.all_nodes()])
        assert any(node.op_type == "Scan" for node in graph)
        assert not any(
            node.op_type in {"MoE", "QMoE", "GroupedMatMul", "Softmax"}
            for node in graph.all_nodes()
        )
    assert signatures[0] == signatures[1]


@pytest.mark.parametrize("model_type", ["plamo2", "kimi_linear", "kimi_k3"])
def test_direct_task_registrations_construct29_bodies(model_type):
    if model_type == "plamo2":
        from mobius.models.plamo2 import Plamo2ForCausalLM as Module
        from mobius.models.plamo2_test import _config
        from mobius.tasks import Plamo2CausalLMTask as Task
    elif model_type == "kimi_linear":
        from mobius.models.kimi_linear import KimiLinearCausalLMModel as Module
        from mobius.models.kimi_linear_test import _config
        from mobius.tasks import KimiLinearCausalLMTask as Task
    else:
        from mobius.models.kimi_k3 import KimiK3CausalLMModel as Module
        from mobius.models.kimi_k3_test import _config
        from mobius.tasks import KimiK3CausalLMTask as Task

    config = _config()
    with build_context(ep_registry.require("onnx-standard")):
        model = Task().build(Module(config), config)["model"]
    assert model.graph.opset_imports[""] == 29
    functions = [fn for fn in model.functions.values() if fn.domain == "com.microsoft"]
    assert {fn.name for fn in functions} == {"CausalConvWithState", "LinearAttention"}
    assert all(fn.opset_imports[""] == 29 for fn in functions)
    assert all(node.version == 29 for fn in functions for node in fn.graph.all_nodes())


@pytest.mark.parametrize("version", [24, 29])
@pytest.mark.parametrize("layer_type", ["linear_attention", "lightning_attention", "conv"])
def test_cache_function_registration_uses_model_root_not_ambient_context(version, layer_type):
    from mobius.tasks._cache_utils import _register_linear_attention_functions

    config = make_config(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=[layer_type],
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
    )
    model = ir.Model(ir.Graph([], [], nodes=[], opset_imports={"": version}), ir_version=10)
    # Outside a context the build opset is 24, including when model root is 29.
    assert get_build_opset_version() == 24
    _register_linear_attention_functions(model, config)
    assert model.functions
    assert all(fn.opset_imports[""] == version for fn in model.functions.values())
    assert all(
        node.version == version
        for fn in model.functions.values()
        for node in fn.graph.all_nodes()
    )


def test_extended_qmoe_slots_are_rejected_even_when_empty():
    model = quantized_model()
    original = next(iter(model.graph))
    extended = ir.Node(
        original.domain,
        original.op_type,
        [*original.inputs, None],
        outputs=[ir.Value(name="extended")],
        attributes=original.attributes,
    )
    model.graph.insert_before(original, extended)
    with pytest.raises(ValueError, match="extended ABI"):
        register_function_bodies(model, moe_fallback=True)
    assert not model.functions
    assert original.overload == extended.overload == ""
