# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for QuantizedLinear component."""

from __future__ import annotations

import math

import onnx_ir as ir
import pytest

from mobius._build_context import build_context
from mobius._execution_providers import ep_registry
from mobius._testing import (
    count_op_type,
    create_test_builder,
    create_test_input,
)
from mobius.components._quantized_linear import (
    BlockQuantizedLinear,
    ClippableQuantizedLinear,
    NVFP4QuantizedLinear,
    QuantizedEmbedding,
    QuantizedLinear,
    TiedQuantizedLMHead,
)

# Test dimensions
IN_FEATURES = 64
OUT_FEATURES = 32
BITS_4 = 4
BITS_8 = 8
BLOCK_SIZE = 32


@pytest.mark.parametrize("ep", ["default", "cpu", "qnn"])
@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("level", [None, 0, 1, 2, 3, 4])
@pytest.mark.parametrize("kind", ["linear", "clippable", "tied"])
def test_matmul_nbits_accuracy_override(ep, bits, level, kind):
    embedding = QuantizedEmbedding(OUT_FEATURES, IN_FEATURES, bits=bits, block_size=32)
    if kind == "tied":
        linear = TiedQuantizedLMHead(embedding, IN_FEATURES, OUT_FEATURES)
    else:
        cls = QuantizedLinear if kind == "linear" else ClippableQuantizedLinear
        linear = cls(IN_FEATURES, OUT_FEATURES, bits=bits, block_size=32)
    with build_context(ep_registry.require(ep), matmul_nbits_accuracy_level=level):
        builder, op, graph = create_test_builder()
        x = create_test_input(builder, "x", [1, 4, IN_FEATURES])
        builder.add_output(linear(op, x), "y")
        ids = create_test_input(builder, "ids", [1, 4], dtype=ir.DataType.INT64)
        builder.add_output(embedding(op, ids), "embeddings")

    node = next(node for node in graph if node.op_type == "MatMulNBits")
    assert node.attributes["bits"].value == bits
    assert node.attributes["block_size"].value == 32
    expected = level
    if expected is None and bits == 4:
        expected = ep_registry.require(ep).default_int4_accuracy_level or None
    if expected is None:
        assert "accuracy_level" not in node.attributes
    else:
        assert node.attributes["accuracy_level"].value == expected
    gather = next(node for node in graph if node.op_type == "GatherBlockQuantized")
    assert "accuracy_level" not in gather.attributes
    assert gather.attributes["bits"].value == bits


class TestQuantizedLinearInit:
    def test_creates_packed_weight(self):
        ql = QuantizedLinear(IN_FEATURES, OUT_FEATURES, bits=4, block_size=32)
        n_blocks = math.ceil(IN_FEATURES / 32)
        blob_size = 32 * 4 // 8  # = 16
        assert ql.weight.shape == [OUT_FEATURES, n_blocks, blob_size]

    def test_creates_scales(self):
        ql = QuantizedLinear(IN_FEATURES, OUT_FEATURES, bits=4, block_size=32)
        n_blocks = math.ceil(IN_FEATURES / 32)
        assert ql.scales.shape == [OUT_FEATURES, n_blocks]

    def test_no_zero_points_by_default(self):
        ql = QuantizedLinear(IN_FEATURES, OUT_FEATURES)
        assert ql.zero_points is None

    def test_creates_zero_points_when_requested(self):
        ql = QuantizedLinear(IN_FEATURES, OUT_FEATURES, has_zero_point=True)
        n_blocks = math.ceil(IN_FEATURES / BLOCK_SIZE)
        # 4-bit: two zero-point values packed per byte
        zp_dim = math.ceil(n_blocks / 2)
        assert ql.zero_points is not None
        assert ql.zero_points.shape == [OUT_FEATURES, zp_dim]

    def test_no_bias_by_default(self):
        ql = QuantizedLinear(IN_FEATURES, OUT_FEATURES)
        assert ql.bias is None

    def test_creates_bias_when_requested(self):
        ql = QuantizedLinear(IN_FEATURES, OUT_FEATURES, bias=True)
        assert ql.bias is not None
        assert ql.bias.shape == [OUT_FEATURES]

    def test_8bit_packed_shape(self):
        ql = QuantizedLinear(IN_FEATURES, OUT_FEATURES, bits=8, block_size=32)
        n_blocks = math.ceil(IN_FEATURES / 32)
        blob_size = 32 * 8 // 8  # = 32
        assert ql.weight.shape == [OUT_FEATURES, n_blocks, blob_size]

    def test_rejects_invalid_bits(self):
        with pytest.raises(ValueError, match="bits must be 2, 4, or 8"):
            QuantizedLinear(IN_FEATURES, OUT_FEATURES, bits=3)

    def test_2bit_packed_shape(self):
        ql = QuantizedLinear(IN_FEATURES, OUT_FEATURES, bits=2, block_size=32)
        n_blocks = math.ceil(IN_FEATURES / 32)
        blob_size = 32 * 2 // 8  # = 8
        assert ql.weight.shape == [OUT_FEATURES, n_blocks, blob_size]

    def test_2bit_zero_points_packed_4_per_byte(self):
        ql = QuantizedLinear(
            IN_FEATURES, OUT_FEATURES, bits=2, block_size=32, has_zero_point=True
        )
        n_blocks = math.ceil(IN_FEATURES / 32)
        zp_dim = math.ceil(n_blocks * 2 / 8)  # 4 ZPs per byte
        assert ql.zero_points.shape == [OUT_FEATURES, zp_dim]

    def test_rejects_non_power_of_2_block_size(self):
        with pytest.raises(ValueError, match="block_size must be a power of 2 >= 16"):
            QuantizedLinear(IN_FEATURES, OUT_FEATURES, block_size=48)

    def test_rejects_small_block_size(self):
        with pytest.raises(ValueError, match="block_size must be a power of 2 >= 16"):
            QuantizedLinear(IN_FEATURES, OUT_FEATURES, block_size=8)

    @pytest.mark.parametrize("bits", [2, 4, 8])
    @pytest.mark.parametrize("k", [1, 7, 9, 16, 33, 64, 96, 1536, 6144])
    @pytest.mark.parametrize("explicit_block_size", [False, True])
    def test_per_channel_uses_exact_input_width(self, bits, k, explicit_block_size):
        linear = QuantizedLinear(
            k, OUT_FEATURES, bits=bits, block_size=k if explicit_block_size else -1
        )
        assert linear._k == k
        assert linear._block_size == k
        assert linear.weight.shape == [OUT_FEATURES, 1, math.ceil(k * bits / 8)]
        assert linear.scales.shape == [OUT_FEATURES, 1]

    @pytest.mark.parametrize("k", [0, -1])
    def test_per_channel_rejects_nonpositive_width(self, k):
        with pytest.raises(ValueError, match="requires K > 0"):
            QuantizedLinear(k, OUT_FEATURES, block_size=-1)

    @pytest.mark.parametrize("block_size", [0, -2, -16])
    def test_other_nonpositive_block_sizes_remain_invalid(self, block_size):
        with pytest.raises(ValueError, match="block_size must be a power of 2 >= 16"):
            QuantizedLinear(IN_FEATURES, OUT_FEATURES, block_size=block_size)


@pytest.mark.parametrize(
    ("bits", "inline", "k"),
    [
        *((bits, False, k) for bits in (4, 8) for k in (16, 32, 64, 128, 256)),
        *((4, True, k) for k in (1, 7, 9, 33, 96, 1536, 6144)),
    ],
)
@pytest.mark.parametrize("asymmetric", [False, True])
def test_per_channel_olive_export_matches_dequantized_weights(
    tmp_path, k, bits, inline, asymmetric
):
    import numpy as np
    import onnxruntime as ort
    import torch
    from onnx_ir.passes.common import InlinePass

    from mobius._weight_utils import preprocess_olive_weights
    from mobius.functions import register_function_bodies

    n = 8
    rng = np.random.default_rng(7)
    values_per_byte = 8 // bits
    packed_k = math.ceil(k / values_per_byte)
    codes = rng.integers(0, 1 << bits, (n, packed_k * values_per_byte), dtype=np.uint8)
    packed = np.bitwise_or.reduce(
        codes.reshape(n, packed_k, values_per_byte)
        << (np.arange(values_per_byte, dtype=np.uint8) * bits),
        axis=-1,
    )
    scales = rng.uniform(0.01, 0.05, (n, 1)).astype(np.float32)
    zeros = (
        rng.integers(0, 1 << bits, (n, 1), dtype=np.uint8)
        if asymmetric
        else np.full((n, 1), 1 << (bits - 1), dtype=np.uint8)
    )
    raw = {
        "proj.weight_qweight": torch.from_numpy(packed),
        "proj.weight_scales": torch.from_numpy(scales),
    }
    if asymmetric:
        raw["proj.weight_qzeros"] = torch.from_numpy(zeros)
    weights = preprocess_olive_weights(raw, bits=bits, group_size=-1, in_features=k)
    linear = QuantizedLinear(k, n, bits=bits, block_size=-1, has_zero_point=asymmetric)
    builder, op, graph = create_test_builder()
    x = create_test_input(builder, "x", [2, 3, k])
    builder.add_output(linear(op, x), "y")
    graph.opset_imports["com.microsoft"] = 1
    for name, weight in weights.items():
        value = graph.initializers[name.removeprefix("proj.")]
        assert list(value.shape) == list(weight.shape)
        value.const_value = ir.tensor(weight.numpy())
    node = next(node for node in graph if node.op_type == "MatMulNBits")
    assert node.attributes["K"].value == k
    assert node.attributes["block_size"].value == k
    model = ir.Model(graph, ir_version=10)
    if inline:
        register_function_bodies(model)
        InlinePass()(model)
        assert not any(node.op_type == "MatMulNBits" for node in graph)
    path = tmp_path / "per-channel.onnx"
    ir.save(model, path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    inputs = rng.standard_normal((2, 3, k)).astype(np.float32)
    actual = session.run(None, {"x": inputs})[0]
    dequantized = (codes[:, :k].astype(np.float32) - zeros.astype(np.float32)) * scales
    np.testing.assert_allclose(actual, inputs @ dequantized.T, rtol=1e-5, atol=1e-4)


class TestQuantizedLinearForward:
    def test_graph_has_matmulnbits_node(self):
        ql = QuantizedLinear(IN_FEATURES, OUT_FEATURES, bits=4, block_size=32)
        b, op, graph = create_test_builder()
        x = create_test_input(b, "x", [1, 4, IN_FEATURES])
        result = ql(op, x)
        b._adapt_outputs([result], "")
        assert count_op_type(graph, "MatMulNBits") == 1

    def test_matmulnbits_domain_is_microsoft(self):
        ql = QuantizedLinear(IN_FEATURES, OUT_FEATURES, bits=4, block_size=32)
        b, op, graph = create_test_builder()
        x = create_test_input(b, "x", [1, 4, IN_FEATURES])
        result = ql(op, x)
        b._adapt_outputs([result], "")
        for node in graph:
            if node.op_type == "MatMulNBits":
                assert node.domain == "com.microsoft"
                break
        else:
            pytest.fail("MatMulNBits node not found")

    def test_matmulnbits_attributes(self):
        ql = QuantizedLinear(IN_FEATURES, OUT_FEATURES, bits=4, block_size=32)
        b, op, graph = create_test_builder()
        x = create_test_input(b, "x", [1, 4, IN_FEATURES])
        result = ql(op, x)
        b._adapt_outputs([result], "")
        for node in graph:
            if node.op_type == "MatMulNBits":
                attrs = {a.name: a.value for a in node.attributes.values()}
                assert attrs["K"] == IN_FEATURES
                assert attrs["N"] == OUT_FEATURES
                assert attrs["bits"] == 4
                assert attrs["block_size"] == 32
                # No active build context -> default EP (accuracy_level 0),
                # so the attribute is omitted and ORT keeps its default path.
                assert "accuracy_level" not in attrs
                break
        else:
            pytest.fail("MatMulNBits node not found")

    def test_no_accuracy_level_without_context(self):
        """Default EP omits accuracy_level (ORT default / highest precision)."""
        ql = QuantizedLinear(IN_FEATURES, OUT_FEATURES, bits=4, block_size=32)
        b, op, graph = create_test_builder()
        x = create_test_input(b, "x", [1, 4, IN_FEATURES])
        result = ql(op, x)
        b._adapt_outputs([result], "")
        for node in graph:
            if node.op_type == "MatMulNBits":
                attrs = {a.name: a.value for a in node.attributes.values()}
                assert "accuracy_level" not in attrs
                break
        else:
            pytest.fail("MatMulNBits node not found")

    def test_cpu_ep_emits_accuracy_level_4(self):
        """CPU EP context stamps accuracy_level=4 (int8 MLAS path)."""
        with build_context(ep_registry.require("cpu")):
            ql = QuantizedLinear(IN_FEATURES, OUT_FEATURES, bits=4, block_size=32)
            b, op, graph = create_test_builder()
            x = create_test_input(b, "x", [1, 4, IN_FEATURES])
            result = ql(op, x)
            b._adapt_outputs([result], "")
        for node in graph:
            if node.op_type == "MatMulNBits":
                attrs = {a.name: a.value for a in node.attributes.values()}
                assert attrs["accuracy_level"] == 4
                break
        else:
            pytest.fail("MatMulNBits node not found")

    def test_cpu_ep_omits_accuracy_level_for_non_int4(self):
        """Without an override, 8-bit weights preserve the existing ORT default."""
        with build_context(ep_registry.require("cpu")):
            ql = QuantizedLinear(IN_FEATURES, OUT_FEATURES, bits=8, block_size=32)
            b, op, graph = create_test_builder()
            x = create_test_input(b, "x", [1, 4, IN_FEATURES])
            result = ql(op, x)
            b._adapt_outputs([result], "")
        for node in graph:
            if node.op_type == "MatMulNBits":
                attrs = {a.name: a.value for a in node.attributes.values()}
                assert attrs["bits"] == 8
                assert "accuracy_level" not in attrs
                break
        else:
            pytest.fail("MatMulNBits node not found")

    def test_3_inputs_without_zero_points(self):
        ql = QuantizedLinear(IN_FEATURES, OUT_FEATURES)
        b, op, graph = create_test_builder()
        x = create_test_input(b, "x", [1, 4, IN_FEATURES])
        result = ql(op, x)
        b._adapt_outputs([result], "")
        for node in graph:
            if node.op_type == "MatMulNBits":
                assert len(node.inputs) == 3
                break

    def test_4_inputs_with_zero_points(self):
        ql = QuantizedLinear(IN_FEATURES, OUT_FEATURES, has_zero_point=True)
        b, op, graph = create_test_builder()
        x = create_test_input(b, "x", [1, 4, IN_FEATURES])
        result = ql(op, x)
        b._adapt_outputs([result], "")
        for node in graph:
            if node.op_type == "MatMulNBits":
                assert len(node.inputs) == 4
                break

    def test_bias_adds_add_node(self):
        ql = QuantizedLinear(IN_FEATURES, OUT_FEATURES, bias=True)
        b, op, graph = create_test_builder()
        x = create_test_input(b, "x", [1, 4, IN_FEATURES])
        result = ql(op, x)
        b._adapt_outputs([result], "")
        assert count_op_type(graph, "MatMulNBits") == 1
        assert count_op_type(graph, "Add") == 1

    def test_no_add_without_bias(self):
        ql = QuantizedLinear(IN_FEATURES, OUT_FEATURES)
        b, op, graph = create_test_builder()
        x = create_test_input(b, "x", [1, 4, IN_FEATURES])
        result = ql(op, x)
        b._adapt_outputs([result], "")
        assert count_op_type(graph, "Add") == 0

    def test_8bit_forward(self):
        ql = QuantizedLinear(IN_FEATURES, OUT_FEATURES, bits=8, block_size=32)
        b, op, graph = create_test_builder()
        x = create_test_input(b, "x", [1, 4, IN_FEATURES])
        result = ql(op, x)
        b._adapt_outputs([result], "")
        for node in graph:
            if node.op_type == "MatMulNBits":
                attrs = {a.name: a.value for a in node.attributes.values()}
                assert attrs["bits"] == 8
                break

    def test_2bit_forward_attributes(self):
        ql = QuantizedLinear(
            IN_FEATURES, OUT_FEATURES, bits=2, block_size=32, has_zero_point=True
        )
        b, op, graph = create_test_builder()
        x = create_test_input(b, "x", [1, 4, IN_FEATURES])
        result = ql(op, x)
        b._adapt_outputs([result], "")
        for node in graph:
            if node.op_type == "MatMulNBits":
                attrs = {a.name: a.value for a in node.attributes.values()}
                assert attrs["bits"] == 2
                assert attrs["block_size"] == 32
                assert len(node.inputs) == 4  # x, w, scales, zero_points
                break
        else:
            pytest.fail("MatMulNBits node not found")

    def test_parameter_names(self):
        ql = QuantizedLinear(IN_FEATURES, OUT_FEATURES, has_zero_point=True, bias=True)
        names = [n for n, _ in ql.named_parameters()]
        assert "weight" in names
        assert "scales" in names
        assert "zero_points" in names
        assert "bias" in names


class TestNVFP4QuantizedLinear:
    def test_exact_parameter_abi(self):
        linear = NVFP4QuantizedLinear(IN_FEATURES, OUT_FEATURES)

        assert linear.weight.shape == [OUT_FEATURES, IN_FEATURES // 2]
        assert linear.weight.dtype.name == "UINT8"
        assert linear.weight_scale.shape == [OUT_FEATURES, IN_FEATURES // 16]
        assert linear.weight_scale.dtype.name == "UINT8"
        assert linear.weight_scale_2.shape == [1]
        assert linear.weight_scale_2.dtype.name == "FLOAT"

    @pytest.mark.parametrize(("in_features", "block_size"), [(63, 16), (64, 32)])
    def test_rejects_unsupported_layout(self, in_features, block_size):
        with pytest.raises(ValueError, match="requires"):
            NVFP4QuantizedLinear(
                in_features,
                OUT_FEATURES,
                block_size=block_size,
            )

    def test_emits_exact_native_node_and_separate_bias(self):
        linear = NVFP4QuantizedLinear(
            IN_FEATURES,
            OUT_FEATURES,
            bias=True,
        )
        builder, op, graph = create_test_builder()
        x = create_test_input(
            builder,
            "x",
            [2, 3, IN_FEATURES],
            dtype=ir.DataType.FLOAT16,
        )

        result = linear(op, x)
        builder._adapt_outputs([result], "")

        native = next(
            node for node in graph if node.op_type == "MatMulBlockQuantizedFp4Weight"
        )
        assert native.domain == "com.microsoft"
        assert len(native.inputs) == 4
        assert native.attributes["block_size"].value == 16
        assert native.outputs[0].shape == [2, 3, OUT_FEATURES]
        assert count_op_type(graph, "Add") == 1
        assert graph.opset_imports["com.microsoft"] == 1

    def test_rejects_non_fp16_activation_abi(self):
        linear = NVFP4QuantizedLinear(IN_FEATURES, OUT_FEATURES)
        builder, op, _graph = create_test_builder()
        x = create_test_input(builder, "x", [1, IN_FEATURES])

        with pytest.raises(ValueError, match="requires FLOAT16 activations"):
            linear(op, x)


class TestBlockQuantizedLinear:
    @pytest.mark.parametrize(
        ("format_name", "block_elements", "block_bytes"),
        [
            ("mxfp4", 32, 17),
            ("iq4_nl", 32, 18),
            ("iq4_xs", 256, 136),
            ("iq3_s", 256, 110),
            ("iq3_xxs", 256, 98),
            ("iq2_xxs", 256, 66),
            ("iq2_xs", 256, 74),
            ("iq2_s", 256, 82),
            ("iq1_s", 256, 50),
            ("iq1_m", 256, 56),
        ],
    )
    def test_emits_native_block_contract(
        self,
        format_name: str,
        block_elements: int,
        block_bytes: int,
    ):
        linear = BlockQuantizedLinear(
            IN_FEATURES,
            OUT_FEATURES,
            format=format_name,
            bias=True,
        )
        expected_blocks = (IN_FEATURES + block_elements - 1) // block_elements
        assert linear.weight.shape == [OUT_FEATURES, expected_blocks, block_bytes]

        b, op, graph = create_test_builder()
        x = create_test_input(b, "x", [1, 4, IN_FEATURES])
        result = linear(op, x)
        b._adapt_outputs([result], "")

        node = next(node for node in graph if node.op_type == "BlockQuantizedMatMul")
        assert node.domain == "pkg.nxrt"
        assert graph.opset_imports["pkg.nxrt"] == 1
        assert len(node.inputs) == 3
        attrs = {attribute.name: attribute.value for attribute in node.attributes.values()}
        assert attrs == {
            "K": IN_FEATURES,
            "N": OUT_FEATURES,
            "format": format_name,
            "block_layout_version": 1,
        }

    def test_rejects_runtime_unsupported_iq_format(self):
        with pytest.raises(ValueError, match="format must be one of"):
            BlockQuantizedLinear(IN_FEATURES, OUT_FEATURES, format="q4_k")


class TestClippableQuantizedLinear:
    def test_keeps_clipping_parameters(self):
        linear = ClippableQuantizedLinear(IN_FEATURES, OUT_FEATURES)
        names = {name for name, _ in linear.named_parameters()}
        assert {
            "weight",
            "scales",
            "input_min",
            "input_max",
            "output_min",
            "output_max",
        } <= names

    def test_graph_clips_around_matmulnbits(self):
        linear = ClippableQuantizedLinear(
            IN_FEATURES,
            OUT_FEATURES,
            bits=8,
            block_size=32,
        )
        b, op, graph = create_test_builder()
        x = create_test_input(b, "x", [1, 4, IN_FEATURES])
        result = linear(op, x)
        b._adapt_outputs([result], "")

        assert count_op_type(graph, "MatMulNBits") == 1
        assert count_op_type(graph, "Clip") == 2


class TestMakeQuantizedLinearFactory:
    """Tests for the make_quantized_linear_factory closure."""

    def test_factory_returns_class(self):
        from mobius.components._quantized_linear import (
            make_quantized_linear_factory,
        )

        factory = make_quantized_linear_factory(bits=4, block_size=32)
        assert isinstance(factory, type)

    def test_factory_creates_quantized_linear(self):
        from mobius.components._quantized_linear import (
            make_quantized_linear_factory,
        )

        factory = make_quantized_linear_factory(bits=4, block_size=32, has_zero_point=True)
        instance = factory(64, 128)
        assert isinstance(instance, QuantizedLinear)
        assert instance._bits == 4
        assert instance._block_size == 32
        assert instance.zero_points is not None

    def test_factory_matches_linear_signature(self):
        """Factory class must accept (in_features, out_features, bias=True)."""
        from mobius.components._quantized_linear import (
            make_quantized_linear_factory,
        )

        factory = make_quantized_linear_factory(bits=4, block_size=128)
        instance = factory(32, 64, bias=False)
        assert instance._k == 32
        assert instance._n == 64
        assert instance.bias is None

    def test_clippable_factory_uses_requested_layout(self):
        from mobius.components._quantized_linear import (
            make_clippable_quantized_linear_factory,
        )

        factory = make_clippable_quantized_linear_factory(
            bits=8,
            block_size=32,
            has_zero_point=True,
        )
        linear = factory(IN_FEATURES, OUT_FEATURES, bias=False)

        assert isinstance(linear, ClippableQuantizedLinear)
        assert linear.weight.shape == [OUT_FEATURES, 2, 32]
        assert linear.zero_points is not None


class TestQuantizedEmbeddingInit:
    VOCAB = 64
    DIM = 64

    def test_creates_packed_qweight_2d(self):
        from mobius.components._quantized_linear import QuantizedEmbedding

        qe = QuantizedEmbedding(self.VOCAB, self.DIM, bits=4, block_size=32)
        packed = self.DIM * 4 // 8  # 32
        assert qe.qweight.shape == [self.VOCAB, packed]

    def test_creates_scales(self):
        from mobius.components._quantized_linear import QuantizedEmbedding

        qe = QuantizedEmbedding(self.VOCAB, self.DIM, bits=4, block_size=32)
        n_blocks = self.DIM // 32
        assert qe.scales.shape == [self.VOCAB, n_blocks]

    def test_zero_points_by_default(self):
        from mobius.components._quantized_linear import QuantizedEmbedding

        qe = QuantizedEmbedding(self.VOCAB, self.DIM)
        assert qe.zero_points is not None

    def test_no_zero_points_when_symmetric(self):
        from mobius.components._quantized_linear import QuantizedEmbedding

        qe = QuantizedEmbedding(self.VOCAB, self.DIM, has_zero_point=False)
        assert qe.zero_points is None

    def test_rejects_invalid_bits(self):
        from mobius.components._quantized_linear import QuantizedEmbedding

        with pytest.raises(ValueError, match="bits must be 2, 4, or 8"):
            QuantizedEmbedding(self.VOCAB, self.DIM, bits=3)

    def test_rejects_indivisible_dim(self):
        from mobius.components._quantized_linear import QuantizedEmbedding

        with pytest.raises(ValueError, match="must be divisible by"):
            QuantizedEmbedding(self.VOCAB, 48, block_size=32)


class TestQuantizedEmbeddingForward:
    VOCAB = 64
    DIM = 64

    def test_graph_has_gather_block_quantized_node(self):
        import onnx_ir as ir

        from mobius.components._quantized_linear import QuantizedEmbedding

        qe = QuantizedEmbedding(self.VOCAB, self.DIM, bits=4, block_size=32)
        b, op, graph = create_test_builder()
        ids = create_test_input(b, "input_ids", [1, 4], dtype=ir.DataType.INT64)
        result = qe(op, ids)
        b._adapt_outputs([result], "")
        assert count_op_type(graph, "GatherBlockQuantized") == 1
        assert result.dtype == ir.DataType.FLOAT
        assert result.shape == ir.Shape([1, 4, self.DIM])

    def test_node_domain_and_attributes(self):
        import onnx_ir as ir

        from mobius.components._quantized_linear import QuantizedEmbedding

        qe = QuantizedEmbedding(self.VOCAB, self.DIM, bits=4, block_size=32)
        b, op, graph = create_test_builder()
        ids = create_test_input(b, "input_ids", [1, 4], dtype=ir.DataType.INT64)
        result = qe(op, ids)
        b._adapt_outputs([result], "")
        for node in graph:
            if node.op_type == "GatherBlockQuantized":
                assert node.domain == "com.microsoft"
                attrs = {a.name: a.value for a in node.attributes.values()}
                assert attrs["bits"] == 4
                assert attrs["block_size"] == 32
                assert attrs["gather_axis"] == 0
                assert attrs["quantize_axis"] == 1
                assert len(node.inputs) == 4  # qweight, ids, scales, zero_points
                break
        else:
            pytest.fail("GatherBlockQuantized node not found")

    def test_3_inputs_without_zero_points(self):
        import onnx_ir as ir

        from mobius.components._quantized_linear import QuantizedEmbedding

        qe = QuantizedEmbedding(self.VOCAB, self.DIM, has_zero_point=False)
        b, op, graph = create_test_builder()
        ids = create_test_input(b, "input_ids", [1, 4], dtype=ir.DataType.INT64)
        result = qe(op, ids)
        b._adapt_outputs([result], "")
        for node in graph:
            if node.op_type == "GatherBlockQuantized":
                assert len(node.inputs) == 3
                break
        else:
            pytest.fail("GatherBlockQuantized node not found")


class TestTiedQuantizedLMHead:
    VOCAB = 64
    DIM = 64

    def _make(self, **kw):
        from mobius.components._quantized_linear import (
            QuantizedEmbedding,
            TiedQuantizedLMHead,
        )

        emb = QuantizedEmbedding(self.VOCAB, self.DIM, bits=4, block_size=32, **kw)
        return emb, TiedQuantizedLMHead(emb, self.DIM, self.VOCAB)

    def test_shares_embedding_parameters(self):
        emb, head = self._make()
        # Same Parameter objects -> a single ONNX initializer each.
        assert head.qweight is emb.qweight
        assert head.scales is emb.scales
        assert head.zero_points is emb.zero_points

    def test_rejects_indivisible_hidden(self):
        from mobius.components._quantized_linear import (
            QuantizedEmbedding,
            TiedQuantizedLMHead,
        )

        emb = QuantizedEmbedding(self.VOCAB, self.DIM, bits=4, block_size=32)
        with pytest.raises(ValueError, match="must be divisible by"):
            TiedQuantizedLMHead(emb, 48, self.VOCAB)

    def test_graph_has_reshape_and_matmulnbits(self):
        _, head = self._make()
        b, op, graph = create_test_builder()
        x = create_test_input(b, "x", [1, 4, self.DIM])
        result = head(op, x)
        b._adapt_outputs([result], "")
        assert count_op_type(graph, "MatMulNBits") == 1
        # Reshape rewrites the shared 2-D table to the 3-D MatMulNBits layout.
        assert count_op_type(graph, "Reshape") == 1

    def test_matmulnbits_attributes(self):
        _, head = self._make()
        b, op, graph = create_test_builder()
        x = create_test_input(b, "x", [1, 4, self.DIM])
        result = head(op, x)
        b._adapt_outputs([result], "")
        for node in graph:
            if node.op_type == "MatMulNBits":
                attrs = {a.name: a.value for a in node.attributes.values()}
                assert attrs["K"] == self.DIM
                assert attrs["N"] == self.VOCAB
                assert attrs["bits"] == 4
                assert attrs["block_size"] == 32
                assert node.domain == "com.microsoft"
                break
        else:
            pytest.fail("MatMulNBits node not found")

    def test_no_zero_points_when_symmetric(self):
        _, head = self._make(has_zero_point=False)
        assert head.zero_points is None
        b, op, graph = create_test_builder()
        x = create_test_input(b, "x", [1, 4, self.DIM])
        result = head(op, x)
        b._adapt_outputs([result], "")
        for node in graph:
            if node.op_type == "MatMulNBits":
                assert len(node.inputs) == 3  # x, weight, scales
                break
        else:
            pytest.fail("MatMulNBits node not found")
