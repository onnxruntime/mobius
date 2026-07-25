# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the quantized GGUF → ONNX build pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _write_quantized_gguf(
    path: Path,
    *,
    hidden_size: int = 64,
    num_layers: int = 1,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    intermediate_size: int = 128,
    vocab_size: int = 256,
    quantize_embedding: bool = False,
    embedding_quantization: str | None = None,
    projection_quantization: str = "q4_0",
    value_projection_quantization: str | None = None,
    output_quantization: str | None = None,
    tie_embeddings: bool = False,
) -> None:
    """Write a GGUF file with quantized projection weights.

    Norms are float32; all linear-layer weights in decoder blocks are
    encoded with *projection_quantization*. The embedding can optionally
    be Q4_0 and tied to the LM head.
    """
    from gguf import GGMLQuantizationType, GGUFWriter

    writer = GGUFWriter(str(path), "llama")
    writer.add_context_length(512)
    writer.add_embedding_length(hidden_size)
    writer.add_feed_forward_length(intermediate_size)
    writer.add_block_count(num_layers)
    writer.add_head_count(num_heads)
    writer.add_head_count_kv(num_kv_heads)
    writer.add_rope_freq_base(10000.0)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_vocab_size(vocab_size)

    head_dim = hidden_size // num_heads

    def _add_f32(name: str, shape: tuple[int, ...]) -> None:
        writer.add_tensor(name, np.random.randn(*shape).astype(np.float32))

    def _add_q4_0(name: str, n_out: int, k_in: int) -> None:
        """Write a Q4_0-quantized weight tensor."""
        block_size = 32
        block_bytes = 18  # 2B scale + 16B quants
        n_blocks = k_in // block_size
        bytes_per_row = n_blocks * block_bytes
        raw = np.zeros((n_out, bytes_per_row), dtype=np.uint8)
        for row in range(n_out):
            for b in range(n_blocks):
                off = b * block_bytes
                # Random fp16 scale
                scale = np.random.uniform(0.01, 1.0)
                raw[row, off : off + 2] = np.array([scale], dtype=np.float16).view(np.uint8)
                # Random packed nibbles
                raw[row, off + 2 : off + 18] = np.random.randint(
                    0, 256, size=16, dtype=np.uint8
                )
        writer.add_tensor(name, raw, raw_dtype=GGMLQuantizationType.Q4_0)

    def _add_q8_0(name: str, n_out: int, k_in: int) -> None:
        """Write a Q8_0-quantized weight tensor."""
        block_size = 32
        block_bytes = 34  # 2B scale + 32B quants
        n_blocks = k_in // block_size
        bytes_per_row = n_blocks * block_bytes
        raw = np.zeros((n_out, bytes_per_row), dtype=np.uint8)
        for row in range(n_out):
            for b in range(n_blocks):
                off = b * block_bytes
                scale = np.random.uniform(0.01, 1.0)
                raw[row, off : off + 2] = np.array([scale], dtype=np.float16).view(np.uint8)
                raw[row, off + 2 : off + 34] = (
                    np.random.randint(-127, 128, size=32, dtype=np.int16)
                    .astype(np.int8, copy=False)
                    .view(np.uint8)
                )
        writer.add_tensor(name, raw, raw_dtype=GGMLQuantizationType.Q8_0)

    def _add_q5_1(name: str, n_out: int, k_in: int) -> None:
        """Write a Q5_1-quantized weight tensor."""
        block_size = 32
        block_bytes = 24  # 2B scale + 2B min + 4B high bits + 16B low nibbles
        n_blocks = k_in // block_size
        bytes_per_row = n_blocks * block_bytes
        raw = np.zeros((n_out, bytes_per_row), dtype=np.uint8)
        for row in range(n_out):
            for b in range(n_blocks):
                off = b * block_bytes
                scale = np.random.uniform(0.01, 1.0)
                minimum = np.random.uniform(-0.5, 0.5)
                raw[row, off : off + 2] = np.array([scale], dtype=np.float16).view(np.uint8)
                raw[row, off + 2 : off + 4] = np.array([minimum], dtype=np.float16).view(
                    np.uint8
                )
                raw[row, off + 4 : off + 8] = np.random.randint(0, 256, size=4, dtype=np.uint8)
                raw[row, off + 8 : off + 24] = np.random.randint(
                    0, 256, size=16, dtype=np.uint8
                )
        writer.add_tensor(name, raw, raw_dtype=GGMLQuantizationType.Q5_1)

    def _add_native(name: str, n_out: int, k_in: int, format_name: str) -> None:
        """Write deterministic native blocks with embedded scales."""
        qtype, block_elements, block_bytes = {
            "mxfp4": (GGMLQuantizationType.MXFP4, 32, 17),
            "iq4_nl": (GGMLQuantizationType.IQ4_NL, 32, 18),
            "iq4_xs": (GGMLQuantizationType.IQ4_XS, 256, 136),
            "iq3_s": (GGMLQuantizationType.IQ3_S, 256, 110),
            "iq3_xxs": (GGMLQuantizationType.IQ3_XXS, 256, 98),
            "iq2_xxs": (GGMLQuantizationType.IQ2_XXS, 256, 66),
            "iq2_xs": (GGMLQuantizationType.IQ2_XS, 256, 74),
            "iq2_s": (GGMLQuantizationType.IQ2_S, 256, 82),
            "iq1_s": (GGMLQuantizationType.IQ1_S, 256, 50),
            "iq1_m": (GGMLQuantizationType.IQ1_M, 256, 56),
        }[format_name]
        n_blocks = (k_in + block_elements - 1) // block_elements
        raw = np.arange(n_out * n_blocks * block_bytes, dtype=np.uint8).reshape(
            n_out, n_blocks * block_bytes
        )
        writer.add_tensor(name, raw, raw_dtype=qtype)

    if quantize_embedding:
        _add_q4_0("token_embd.weight", vocab_size, hidden_size)
    elif embedding_quantization == "q8_0":
        _add_q8_0("token_embd.weight", vocab_size, hidden_size)
    else:
        _add_f32("token_embd.weight", (vocab_size, hidden_size))

    if projection_quantization in {"q4_0", "q8_0"}:
        add_projection = {
            "q4_0": _add_q4_0,
            "q8_0": _add_q8_0,
        }[projection_quantization]
    else:

        def add_projection(name: str, n_out: int, k_in: int) -> None:
            _add_native(name, n_out, k_in, projection_quantization)

    add_value_projection = (
        _add_q5_1 if value_projection_quantization == "q5_1" else add_projection
    )

    for i in range(num_layers):
        add_projection(
            f"blk.{i}.attn_q.weight",
            num_heads * head_dim,
            hidden_size,
        )
        add_projection(
            f"blk.{i}.attn_k.weight",
            num_kv_heads * head_dim,
            hidden_size,
        )
        add_value_projection(
            f"blk.{i}.attn_v.weight",
            num_kv_heads * head_dim,
            hidden_size,
        )
        add_projection(
            f"blk.{i}.attn_output.weight",
            hidden_size,
            num_heads * head_dim,
        )
        add_projection(f"blk.{i}.ffn_gate.weight", intermediate_size, hidden_size)
        add_projection(f"blk.{i}.ffn_up.weight", intermediate_size, hidden_size)
        add_projection(f"blk.{i}.ffn_down.weight", hidden_size, intermediate_size)
        # Norms (float32)
        _add_f32(f"blk.{i}.attn_norm.weight", (hidden_size,))
        _add_f32(f"blk.{i}.ffn_norm.weight", (hidden_size,))

    # Output norm + optional untied lm_head
    _add_f32("output_norm.weight", (hidden_size,))
    if not tie_embeddings:
        if output_quantization == "q4_0":
            _add_q4_0("output.weight", vocab_size, hidden_size)
        elif output_quantization == "q8_0":
            _add_q8_0("output.weight", vocab_size, hidden_size)
        else:
            _add_f32("output.weight", (vocab_size, hidden_size))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.fixture
def q4_0_gguf(tmp_path: Path) -> Path:
    """Create a Q4_0 quantized GGUF test file."""
    path = tmp_path / "test_q4_0.gguf"
    _write_quantized_gguf(path)
    return path


@pytest.fixture
def q4_0_embedding_gguf(tmp_path: Path) -> Path:
    """Create a GGUF with a Q4_0 token embedding."""
    path = tmp_path / "test_q4_0_embedding.gguf"
    _write_quantized_gguf(path, quantize_embedding=True)
    return path


@pytest.fixture
def q4_0_tied_embedding_gguf(tmp_path: Path) -> Path:
    """Create a GGUF with one Q4_0 table shared by embedding and LM head."""
    path = tmp_path / "test_q4_0_tied_embedding.gguf"
    _write_quantized_gguf(path, quantize_embedding=True, tie_embeddings=True)
    return path


@pytest.fixture
def q4_0_embedding_q8_head_gguf(tmp_path: Path) -> Path:
    """Create a GGUF with a Q4 embedding and an untied Q8 output head."""
    path = tmp_path / "test_q4_0_embedding_q8_head.gguf"
    _write_quantized_gguf(
        path,
        quantize_embedding=True,
        output_quantization="q8_0",
    )
    return path


@pytest.fixture
def q8_0_projection_q4_head_gguf(tmp_path: Path) -> Path:
    """Create a GGUF whose Q4 output head would require Q8 requantization."""
    path = tmp_path / "test_q8_0_projection_q4_head.gguf"
    _write_quantized_gguf(
        path,
        projection_quantization="q8_0",
        output_quantization="q4_0",
    )
    return path


@pytest.fixture(
    params=[
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
    ]
)
def native_block_gguf(tmp_path: Path, request) -> tuple[Path, str, int, int]:
    """Create a GGUF whose projection weights use a runtime-native block format."""
    format_name, block_elements, block_bytes = request.param
    path = tmp_path / f"test_{format_name}.gguf"
    _write_quantized_gguf(
        path,
        hidden_size=256,
        num_kv_heads=4,
        intermediate_size=256,
        projection_quantization=format_name,
    )
    return path, format_name, block_elements, block_bytes


@pytest.fixture
def mixed_native_q5_q8_gguf(tmp_path: Path) -> Path:
    """Create native IQ projections with a Q5_1 value projection and Q8 embedding."""
    path = tmp_path / "test_mixed_native_q5_q8.gguf"
    _write_quantized_gguf(
        path,
        hidden_size=256,
        num_kv_heads=4,
        intermediate_size=256,
        embedding_quantization="q8_0",
        projection_quantization="iq4_xs",
        value_projection_quantization="q5_1",
    )
    return path


class TestBuildQuantizedGguf:
    """Tests for build_from_gguf(keep_quantized=True)."""

    def test_produces_model_package(self, q4_0_gguf: Path):
        """Quantized build returns a valid ModelPackage."""
        from mobius.integrations.gguf import build_from_gguf

        pkg = build_from_gguf(q4_0_gguf, keep_quantized=True)
        assert "model" in pkg
        assert pkg["model"].graph is not None

    def test_model_has_matmulnbits_ops(self, q4_0_gguf: Path):
        """Quantized model uses MatMulNBits instead of MatMul."""
        from mobius.integrations.gguf import build_from_gguf

        pkg = build_from_gguf(q4_0_gguf, keep_quantized=True)
        model = pkg["model"]

        op_types = {node.op_type for node in model.graph if node.op_type}
        assert "MatMulNBits" in op_types, (
            f"Expected MatMulNBits in ops, got: {sorted(op_types)}"
        )

    def test_native_blocks_emit_block_quantized_matmul_and_preserve_bytes(
        self,
        native_block_gguf: tuple[Path, str, int, int],
    ):
        """Runtime-native IQ/MXFP4 projections retain their exact GGUF bytes."""
        import onnx_ir as ir

        from mobius.integrations.gguf import build_from_gguf

        path, format_name, block_elements, block_bytes = native_block_gguf
        model = build_from_gguf(path, keep_quantized=True)["model"]
        nodes = [node for node in model.graph if node.op_type == "BlockQuantizedMatMul"]
        assert len(nodes) == 7
        assert all(node.domain == "com.github.onnxruntime.genai" for node in nodes)
        for node in nodes:
            attrs = {attribute.name: attribute.value for attribute in node.attributes.values()}
            assert attrs["format"] == format_name
            assert attrs["block_layout_version"] == 1
            assert attrs["K"] == 256
            assert attrs["N"] == 256

        weight = model.graph.initializers["model.layers.0.self_attn.o_proj.weight"]
        assert weight.dtype == ir.DataType.UINT8
        n_blocks = (256 + block_elements - 1) // block_elements
        assert list(weight.shape) == [256, n_blocks, block_bytes]
        expected = np.arange(256 * n_blocks * block_bytes, dtype=np.uint8).reshape(
            256, n_blocks, block_bytes
        )
        np.testing.assert_array_equal(weight.const_value.numpy(), expected)
        assert "model.layers.0.self_attn.o_proj.scales" not in model.graph.initializers

        assert model.graph.opset_imports["com.github.onnxruntime.genai"] == 1
        proto = ir.serde.serialize_model(model)
        imports = {opset.domain: opset.version for opset in proto.opset_import}
        assert imports["com.github.onnxruntime.genai"] == 1

    def test_mixed_native_quantization_uses_q4_scaffold(self, mixed_native_q5_q8_gguf: Path):
        """Native IQ tensors force Q5_1 fallback weights onto a Q4 scaffold."""
        from mobius.integrations.gguf import build_from_gguf

        model = build_from_gguf(mixed_native_q5_q8_gguf, keep_quantized=True)["model"]
        native_nodes = [node for node in model.graph if node.op_type == "BlockQuantizedMatMul"]
        assert len(native_nodes) == 6

        value_nodes = [
            node
            for node in model.graph
            if node.op_type == "MatMulNBits"
            and node.inputs[1].name == "model.layers.0.self_attn.v_proj.weight"
        ]
        assert len(value_nodes) == 1
        assert value_nodes[0].attributes["bits"].value == 4
        assert value_nodes[0].attributes["block_size"].value == 32
        assert "model.layers.0.self_attn.v_proj.zero_points" in model.graph.initializers

        native_weight = model.graph.initializers["model.layers.0.self_attn.o_proj.weight"]
        expected = np.arange(256 * 136, dtype=np.uint8).reshape(256, 1, 136)
        np.testing.assert_array_equal(native_weight.const_value.numpy(), expected)

    def test_quantized_embedding_uses_gatherblockquantized(self, q4_0_embedding_gguf: Path):
        """A quantized GGUF embedding remains packed in the ONNX graph."""
        import onnx_ir as ir

        from mobius.integrations.gguf import build_from_gguf

        model = build_from_gguf(q4_0_embedding_gguf, keep_quantized=True)["model"]
        gather_nodes = [node for node in model.graph if node.op_type == "GatherBlockQuantized"]
        assert len(gather_nodes) == 1
        assert gather_nodes[0].domain == "com.microsoft"

        qweight = model.graph.initializers["model.embed_tokens.qweight"]
        assert qweight.dtype == ir.DataType.UINT8
        assert list(qweight.shape) == [256, 32]
        assert list(model.graph.initializers["model.embed_tokens.scales"].shape) == [
            256,
            2,
        ]
        assert "model.embed_tokens.weight" not in model.graph.initializers

    def test_tied_quantized_embedding_drives_matmulnbits_head(
        self, q4_0_tied_embedding_gguf: Path
    ):
        """Tied embedding/head share one packed table across both contrib ops."""
        from mobius.integrations.gguf import build_from_gguf

        model = build_from_gguf(q4_0_tied_embedding_gguf, keep_quantized=True)["model"]
        op_types = [node.op_type for node in model.graph]
        assert op_types.count("GatherBlockQuantized") == 1
        assert "MatMulNBits" in op_types
        assert "model.embed_tokens.qweight" in model.graph.initializers
        assert not any(name.startswith("lm_head.") for name in model.graph.initializers)

    def test_untied_quantized_head_uses_q4_matmulnbits(
        self, q4_0_embedding_q8_head_gguf: Path
    ):
        """An untied quantized output is requantized to the graph's Q4 layout."""
        import onnx_ir as ir

        from mobius.integrations.gguf import build_from_gguf

        model = build_from_gguf(q4_0_embedding_q8_head_gguf, keep_quantized=True)["model"]
        head_nodes = [
            node
            for node in model.graph
            if node.op_type == "MatMulNBits" and node.outputs[0].name == "logits"
        ]
        assert len(head_nodes) == 1
        assert head_nodes[0].attributes["bits"].value == 4
        assert head_nodes[0].attributes["block_size"].value == 32

        qweight = model.graph.initializers["lm_head.weight"]
        assert qweight.dtype == ir.DataType.UINT8
        assert list(qweight.shape) == [256, 2, 16]
        assert list(model.graph.initializers["lm_head.scales"].shape) == [256, 2]
        assert "lm_head.weight_t" not in model.graph.initializers

    def test_unsupported_requantization_target_has_clear_error(
        self, q8_0_projection_q4_head_gguf: Path
    ):
        """Mixed targets outside 4-bit/block-32 fail before the Q4 repacker."""
        from mobius.integrations.gguf import build_from_gguf

        with pytest.raises(
            ValueError,
            match=(
                "keep_quantized MatMulNBits requantization currently supports only "
                r"4-bit/block-32 targets; got bits=8 block=32 for tensor lm_head\.weight"
            ),
        ):
            build_from_gguf(q8_0_projection_q4_head_gguf, keep_quantized=True)

    def test_norms_are_float(self, q4_0_gguf: Path):
        """Norm weights remain float, not quantized."""
        import onnx_ir as ir

        from mobius.integrations.gguf import build_from_gguf

        pkg = build_from_gguf(q4_0_gguf, keep_quantized=True)
        model = pkg["model"]

        for init in model.graph.initializers.values():
            name = init.name or ""
            if "norm" in name and "weight" in name:
                assert init.dtype != ir.DataType.UINT8, (
                    f"Norm {name} should be float, not uint8"
                )

    def test_dequantized_path_no_matmulnbits(self, q4_0_gguf: Path):
        """Without keep_quantized, no MatMulNBits ops."""
        from mobius.integrations.gguf import build_from_gguf

        pkg = build_from_gguf(q4_0_gguf, keep_quantized=False)
        model = pkg["model"]

        op_types = {node.op_type for node in model.graph if node.op_type}
        assert "MatMulNBits" not in op_types

    def test_detect_quant_params(self, q4_0_gguf: Path):
        """_detect_quant_params finds Q4_0 as dominant type."""
        from mobius.integrations.gguf._builder import (
            _detect_quant_params,
        )
        from mobius.integrations.gguf._reader import GGUFModel

        gguf_model = GGUFModel(q4_0_gguf)
        bits, block_size, is_sym = _detect_quant_params(gguf_model, gguf_model.architecture)
        assert bits == 4
        assert block_size == 32
        assert is_sym is True

    def test_embedding_quantization_check_is_metadata_only(self, monkeypatch):
        """Embedding compatibility does not read or repack tensor data."""
        from types import SimpleNamespace

        from gguf import GGMLQuantizationType

        from mobius.integrations.gguf import _tencent_q1_0
        from mobius.integrations.gguf._builder import _can_quantize_embedding

        monkeypatch.setattr(_tencent_q1_0, "is_tencent_q1_0_layout", lambda _model: False)
        tensor = SimpleNamespace(
            name="token_embd.weight",
            tensor_type=GGMLQuantizationType.Q4_0,
            shape=(64, 256),
        )
        model = SimpleNamespace(_reader=SimpleNamespace(tensors=[tensor]))

        assert _can_quantize_embedding(model, "llama", bits=4, block_size=32)

    def test_tencent_q1_0_embedding_is_not_quantized(self, monkeypatch):
        """Tencent Q1_0 detection short-circuits before inspecting tensors."""
        from types import SimpleNamespace

        from mobius.integrations.gguf import _tencent_q1_0
        from mobius.integrations.gguf._builder import _can_quantize_embedding

        monkeypatch.setattr(_tencent_q1_0, "is_tencent_q1_0_layout", lambda _model: True)
        model = SimpleNamespace()

        assert not _can_quantize_embedding(model, "llama", bits=4, block_size=128)

    def test_detect_q4_k_m_mixed_profile(self):
        """Q4_K presence selects a 4-bit target despite more Q5_0 tensors."""
        from gguf import GGMLQuantizationType

        from mobius.integrations.gguf._builder import _detect_quant_params

        class _MixedModel:
            def tensor_items_raw(self):
                for i in range(5):
                    yield (
                        f"blk.{i}.attn_q.weight",
                        np.empty(0, dtype=np.uint8),
                        GGMLQuantizationType.Q5_0,
                        (64, 64),
                    )
                yield (
                    "blk.0.ffn_down.weight",
                    np.empty(0, dtype=np.uint8),
                    GGMLQuantizationType.Q4_K,
                    (64, 128),
                )

        bits, block_size, is_sym = _detect_quant_params(_MixedModel(), "llama")
        assert (bits, block_size, is_sym) == (4, 32, False)

    def test_runtime_unsupported_format_does_not_select_native_op(self):
        """A GGUF type outside the runtime contract remains on the fallback."""
        from gguf import GGMLQuantizationType

        from mobius.integrations.gguf._builder import _native_block_format

        assert _native_block_format(GGMLQuantizationType.Q5_0) is None

    @pytest.mark.parametrize("moe_container", ["experts", "moe.experts"])
    def test_native_moe_tensor_maps_to_each_expert(self, moe_container: str):
        """Stacked GGUF MoE blocks route to standard and DeepSeek expert paths."""
        from mobius.integrations.gguf._builder import _native_block_target_stems

        available = {f"model.layers.0.mlp.{moe_container}.{i}.gate_proj" for i in range(3)}
        assert _native_block_target_stems(
            "model.layers.0.mlp.experts.gate_proj.weight",
            (3, 64, 64),
            available,
        ) == sorted(available, key=lambda name: int(name.split(".")[-2]))


class TestBuildGgufStaticCache:
    """Tests for build_from_gguf(static_cache=True).

    Static cache mode replaces the dynamic concat-grow KV cache with
    pre-allocated fixed-width buffers (written in place via TensorScatter),
    producing a fully static-shaped graph required by fixed-shape runtimes
    such as the QNN HTP backend. Llama uses the base ``DecoderLayer`` which
    supports the StaticCacheState dispatch.
    """

    def test_static_cache_emits_fixed_width_cache_io(self, q4_0_gguf: Path):
        """Static cache build exposes fixed-width key_cache/value_cache I/O."""
        from mobius.integrations.gguf import build_from_gguf

        max_seq_len = 128
        model = build_from_gguf(
            q4_0_gguf, keep_quantized=True, static_cache=True, max_seq_len=max_seq_len
        )["model"]

        input_names = {i.name for i in model.graph.inputs}
        # Static cache uses key_cache.N / value_cache.N inputs, not the
        # dynamic past_key_values.N.key / .value pair.
        assert any(n and n.startswith("key_cache.") for n in input_names), (
            f"Expected key_cache.* inputs, got {sorted(input_names)}"
        )
        assert not any(n and n.startswith("past_key_values.") for n in input_names), (
            f"Static cache must not emit past_key_values.* inputs, got {sorted(input_names)}"
        )

        # The KV axis of every cache buffer must be a concrete int == max_seq_len,
        # i.e. fully static (no symbolic dims).
        for inp in model.graph.inputs:
            name = inp.name or ""
            if name.startswith(("key_cache.", "value_cache.")):
                assert inp.shape is not None
                assert inp.shape[1] == max_seq_len, (
                    f"{name} KV axis {inp.shape[1]!r} != max_seq_len {max_seq_len}"
                )

    def test_static_cache_rejects_explicit_task(self, q4_0_gguf: Path):
        """static_cache=True with an explicit task override is a ValueError."""
        from mobius.integrations.gguf import build_from_gguf

        with pytest.raises(ValueError, match="static_cache"):
            build_from_gguf(
                q4_0_gguf,
                keep_quantized=True,
                static_cache=True,
                task="text-generation",
            )


class TestRawTensorIterator:
    """Tests for GGUFModel.tensor_items_raw()."""

    def test_yields_raw_data(self, q4_0_gguf: Path):
        from gguf import GGMLQuantizationType

        from mobius.integrations.gguf._reader import GGUFModel

        model = GGUFModel(q4_0_gguf)
        items = list(model.tensor_items_raw())

        # Should have tensors
        assert len(items) > 0

        # Check a quantized tensor
        q_items = [(n, d, qt, s) for n, d, qt, s in items if qt == GGMLQuantizationType.Q4_0]
        assert len(q_items) > 0
        _name, raw, _qtype, shape = q_items[0]
        assert raw.dtype == np.uint8
        assert len(shape) == 2

    def test_float_tensors_have_correct_type(self, q4_0_gguf: Path):
        from gguf import GGMLQuantizationType

        from mobius.integrations.gguf._reader import GGUFModel

        model = GGUFModel(q4_0_gguf)

        f32_items = [
            (n, d, qt, s)
            for n, d, qt, s in model.tensor_items_raw()
            if qt == GGMLQuantizationType.F32
        ]
        assert len(f32_items) > 0
        for _name, _raw, qtype, _shape in f32_items:
            assert qtype == GGMLQuantizationType.F32

    def test_dequantize_raw_tensor_matches_get_tensor(self, q4_0_gguf: Path):
        from mobius.integrations.gguf._reader import GGUFModel

        model = GGUFModel(q4_0_gguf)
        name, raw, qtype, shape = next(
            item for item in model.tensor_items_raw() if len(item[3]) == 2
        )
        expected = model.get_tensor(name)
        actual = model.dequantize_raw_tensor(raw, qtype, shape)
        np.testing.assert_array_equal(actual, expected)
