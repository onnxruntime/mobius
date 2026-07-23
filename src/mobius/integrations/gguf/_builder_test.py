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
    projection_quantization: str = "q4_0",
    output_quantization: str | None = None,
    tie_embeddings: bool = False,
) -> None:
    """Write a GGUF file with Q4_0 quantized projection weights.

    Norms are float32; all linear-layer weights in decoder blocks are
    Q4_0 (4-bit symmetric, block_size=32). The embedding can optionally
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

    if quantize_embedding:
        _add_q4_0("token_embd.weight", vocab_size, hidden_size)
    else:
        _add_f32("token_embd.weight", (vocab_size, hidden_size))

    add_projection = {
        "q4_0": _add_q4_0,
        "q8_0": _add_q8_0,
    }[projection_quantization]

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
        add_projection(
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

    def test_q8_target_requantizes_mixed_q4_head(self, q8_0_projection_q4_head_gguf: Path):
        """A mixed Q8 target requantizes its Q4 output head to block-32 Q8."""
        from mobius.integrations.gguf import build_from_gguf

        model = build_from_gguf(q8_0_projection_q4_head_gguf, keep_quantized=True)["model"]
        head = next(
            node
            for node in model.graph
            if node.op_type == "MatMulNBits" and node.outputs[0].name == "logits"
        )
        assert head.attributes["bits"].value == 8
        assert head.attributes["block_size"].value == 32

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


def test_normalize_glm_dsa_split_kv_b_and_router_bias():
    import torch

    from mobius.integrations.gguf._builder import _normalize_gguf_weights

    k_proj = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    v_proj = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    result = _normalize_gguf_weights(
        {
            "model.layers.0.self_attn.kv_b_proj.k_proj.weight": k_proj,
            "model.layers.0.self_attn.kv_b_proj.v_proj.weight": v_proj,
            "model.layers.0.mlp.gate.e_score_correction_bias.bias": torch.zeros(2),
        }
    )

    expected = torch.cat((k_proj.transpose(1, 2), v_proj), dim=1).reshape(18, 3)
    assert torch.equal(result["model.layers.0.self_attn.kv_b_proj.weight"], expected)
    assert "model.layers.0.mlp.gate.e_score_correction_bias" in result


def test_load_quantized_glm_experts_repacked_individually(monkeypatch):
    from types import SimpleNamespace

    from gguf import GGMLQuantizationType

    from mobius.integrations.gguf import _repacker, _tencent_q1_0, _tensor_mapping
    from mobius.integrations.gguf._builder import _load_quantized_state_dict

    repacked_inputs = []

    def _repack(raw, qtype, shape):
        repacked_inputs.append((raw.copy(), qtype, shape))
        return SimpleNamespace(
            weight=np.full((2, 1, 16), len(repacked_inputs), dtype=np.uint8),
            scales=np.ones((2, 1), dtype=np.float32),
            zero_points=None,
        )

    monkeypatch.setattr(_tencent_q1_0, "is_tencent_q1_0_layout", lambda _model: False)
    monkeypatch.setattr(
        _tensor_mapping,
        "map_gguf_to_hf_names",
        lambda _name, _arch: "model.layers.0.mlp.experts.gate_proj.weight",
    )
    monkeypatch.setattr(_repacker, "repack_quant_params", lambda _qtype: (4, 32))
    monkeypatch.setattr(_repacker, "repack_gguf_tensor", _repack)

    raw = np.arange(8, dtype=np.uint8)
    model = SimpleNamespace(
        _tensor_index={"experts": object()},
        tensor_items_raw=lambda: [("experts", raw, GGMLQuantizationType.Q4_0, (2, 2, 2))],
    )
    module = SimpleNamespace(named_modules=list)
    config = SimpleNamespace(
        quantization=SimpleNamespace(bits=4, group_size=32, sym=True),
        num_attention_heads=1,
        num_key_value_heads=1,
        model_type="glm_moe_dsa",
    )

    result = _load_quantized_state_dict(model, "glm-dsa", module, config)

    assert [call[2] for call in repacked_inputs] == [(2, 2), (2, 2)]
    assert np.array_equal(repacked_inputs[0][0], raw[:4])
    assert np.array_equal(repacked_inputs[1][0], raw[4:])
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" in result
    assert "model.layers.0.mlp.experts.1.gate_proj.weight" in result


def test_load_quantized_experts_requantizes_when_zero_point_presence_differs(
    monkeypatch,
):
    from types import SimpleNamespace

    from gguf import GGMLQuantizationType

    from mobius.integrations.gguf import _repacker, _tencent_q1_0, _tensor_mapping
    from mobius.integrations.gguf._builder import _load_quantized_state_dict

    requantized = []

    def _repack(*_args, **_kwargs):
        return SimpleNamespace(
            weight=np.zeros((2, 1, 16), dtype=np.uint8),
            scales=np.ones((2, 1), dtype=np.float32),
            zero_points=None,
        )

    def _requantize(values, **_kwargs):
        requantized.append(values)
        return SimpleNamespace(
            weight=np.zeros((2, 1, 16), dtype=np.uint8),
            scales=np.ones((2, 1), dtype=np.float32),
            zero_points=np.zeros((2, 1), dtype=np.uint8),
        )

    monkeypatch.setattr(_tencent_q1_0, "is_tencent_q1_0_layout", lambda _model: False)
    monkeypatch.setattr(
        _tensor_mapping,
        "map_gguf_to_hf_names",
        lambda _name, _arch: "model.layers.0.mlp.experts.gate_proj.weight",
    )
    monkeypatch.setattr(_repacker, "repack_quant_params", lambda _qtype: (4, 32))
    monkeypatch.setattr(_repacker, "repack_gguf_tensor", _repack)
    monkeypatch.setattr(_repacker, "repack_dequantized_tensor", _requantize)

    raw = np.arange(8, dtype=np.uint8)
    model = SimpleNamespace(
        _tensor_index={"experts": object()},
        tensor_items_raw=lambda: [
            ("experts", raw, GGMLQuantizationType.Q4_0, (2, 2, 2))
        ],
        dequantize_raw_tensor=lambda *_args: np.ones((2, 2), dtype=np.float32),
    )
    config = SimpleNamespace(
        quantization=SimpleNamespace(bits=4, group_size=32, sym=False),
        num_attention_heads=1,
        num_key_value_heads=1,
        model_type="glm_moe_dsa",
    )

    result = _load_quantized_state_dict(model, "glm-dsa", SimpleNamespace(named_modules=list), config)

    assert len(requantized) == 2
    assert "model.layers.0.mlp.experts.0.gate_proj.zero_points" in result
    assert "model.layers.0.mlp.experts.1.gate_proj.zero_points" in result


def test_load_quantized_glm_fuses_split_kv_b(monkeypatch):
    from types import SimpleNamespace

    from gguf import GGMLQuantizationType

    from mobius.integrations.gguf import _repacker, _tencent_q1_0, _tensor_mapping
    from mobius.integrations.gguf._builder import _load_quantized_state_dict

    fused_values = []

    def _repack(values, **_kwargs):
        fused_values.append(values.copy())
        return SimpleNamespace(
            weight=np.zeros((18, 1, 16), dtype=np.uint8),
            scales=np.ones((18, 1), dtype=np.float32),
            zero_points=np.zeros((18, 1), dtype=np.uint8),
        )

    names = {
        "k": "model.layers.0.self_attn.kv_b_proj.k_proj.weight",
        "v": "model.layers.0.self_attn.kv_b_proj.v_proj.weight",
    }
    monkeypatch.setattr(_tencent_q1_0, "is_tencent_q1_0_layout", lambda _model: False)
    monkeypatch.setattr(
        _tensor_mapping,
        "map_gguf_to_hf_names",
        lambda name, _arch: names[name],
    )
    monkeypatch.setattr(_repacker, "repack_dequantized_tensor", _repack)

    k_proj = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    v_proj = np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3)
    tensors = [
        ("k", k_proj, GGMLQuantizationType.F32, k_proj.shape),
        ("v", v_proj, GGMLQuantizationType.F32, v_proj.shape),
    ]
    model = SimpleNamespace(
        _tensor_index={"k": object(), "v": object()},
        tensor_items_raw=lambda: tensors,
        dequantize_raw_tensor=lambda raw, _qtype, _shape: raw,
    )
    module = SimpleNamespace(named_modules=list)
    config = SimpleNamespace(
        quantization=SimpleNamespace(bits=4, group_size=32, sym=False),
        num_attention_heads=1,
        num_key_value_heads=1,
        model_type="glm_moe_dsa",
    )

    result = _load_quantized_state_dict(model, "glm-dsa", module, config)

    expected = np.concatenate((k_proj.transpose(0, 2, 1), v_proj), axis=1).reshape(18, 3)
    np.testing.assert_array_equal(fused_values[0], expected)
    assert "model.layers.0.self_attn.kv_b_proj.weight" in result
    assert "model.layers.0.self_attn.kv_b_proj.scales" in result
    assert "model.layers.0.self_attn.kv_b_proj.zero_points" in result
