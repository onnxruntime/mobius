# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the GGUF reader and config mapping modules.

These tests use a synthetic GGUF file created via ``gguf.GGUFWriter``
to avoid requiring model downloads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mobius.integrations.gguf._config_mapping import (
    GGUF_ARCH_TO_MODEL_TYPE,
    _extract_config_fields,
    _infer_attn_o_bias,
    _infer_attn_qkv_bias,
    _infer_mlp_bias,
    _infer_tie_embeddings,
    gguf_to_config,
)
from mobius.integrations.gguf._reader import GGUFModel


def _write_test_gguf(
    path: Path,
    architecture: str = "llama",
    hidden_size: int = 64,
    num_layers: int = 2,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    intermediate_size: int = 128,
    vocab_size: int = 256,
    context_length: int = 512,
    rope_theta: float = 10000.0,
    rms_norm_eps: float = 1e-5,
    include_output_weight: bool = True,
) -> None:
    """Write a minimal GGUF file for testing.

    Creates a syntactically valid GGUF file with metadata and tiny
    float32 tensors.
    """
    from gguf import GGUFWriter

    writer = GGUFWriter(str(path), architecture)

    # Core metadata
    writer.add_context_length(context_length)
    writer.add_embedding_length(hidden_size)
    writer.add_feed_forward_length(intermediate_size)
    writer.add_block_count(num_layers)
    writer.add_head_count(num_heads)
    writer.add_head_count_kv(num_kv_heads)
    writer.add_rope_freq_base(rope_theta)
    writer.add_layer_norm_rms_eps(rms_norm_eps)
    writer.add_vocab_size(vocab_size)

    # Tiny tensors: token embedding + one layer + output
    head_dim = hidden_size // num_heads

    # token_embd.weight: (vocab_size, hidden_size)
    writer.add_tensor(
        "token_embd.weight",
        np.random.randn(vocab_size, hidden_size).astype(np.float32),
    )

    for i in range(num_layers):
        # Attention weights
        writer.add_tensor(
            f"blk.{i}.attn_q.weight",
            np.random.randn(num_heads * head_dim, hidden_size).astype(np.float32),
        )
        writer.add_tensor(
            f"blk.{i}.attn_k.weight",
            np.random.randn(num_kv_heads * head_dim, hidden_size).astype(np.float32),
        )
        writer.add_tensor(
            f"blk.{i}.attn_v.weight",
            np.random.randn(num_kv_heads * head_dim, hidden_size).astype(np.float32),
        )
        writer.add_tensor(
            f"blk.{i}.attn_output.weight",
            np.random.randn(hidden_size, num_heads * head_dim).astype(np.float32),
        )
        # MLP weights
        writer.add_tensor(
            f"blk.{i}.ffn_gate.weight",
            np.random.randn(intermediate_size, hidden_size).astype(np.float32),
        )
        writer.add_tensor(
            f"blk.{i}.ffn_up.weight",
            np.random.randn(intermediate_size, hidden_size).astype(np.float32),
        )
        writer.add_tensor(
            f"blk.{i}.ffn_down.weight",
            np.random.randn(hidden_size, intermediate_size).astype(np.float32),
        )
        # Norm weights
        writer.add_tensor(
            f"blk.{i}.attn_norm.weight",
            np.ones(hidden_size, dtype=np.float32),
        )
        writer.add_tensor(
            f"blk.{i}.ffn_norm.weight",
            np.ones(hidden_size, dtype=np.float32),
        )

    # Output norm
    writer.add_tensor(
        "output_norm.weight",
        np.ones(hidden_size, dtype=np.float32),
    )

    # Output (lm_head) — only if not tied
    if include_output_weight:
        writer.add_tensor(
            "output.weight",
            np.random.randn(vocab_size, hidden_size).astype(np.float32),
        )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.fixture
def llama_gguf(tmp_path: Path) -> Path:
    """Create a minimal Llama GGUF file for testing."""
    path = tmp_path / "test_llama.gguf"
    _write_test_gguf(path)
    return path


@pytest.fixture
def tied_gguf(tmp_path: Path) -> Path:
    """Create a GGUF file with tied embeddings (no output.weight)."""
    path = tmp_path / "test_tied.gguf"
    _write_test_gguf(path, include_output_weight=False)
    return path


def _write_gemma4_test_gguf(
    path: Path,
    hidden_size: int = 64,
    num_layers: int = 6,
    num_heads: int = 4,
    num_kv_heads_sliding: int = 2,
    num_kv_heads_full: int = 2,
    intermediate_size: int = 128,
    vocab_size: int = 256,
    context_length: int = 512,
    sliding_window: int = 128,
    global_head_dim: int = 32,
    swa_head_dim: int = 16,
    global_rope_theta: float = 1_000_000.0,
    swa_rope_theta: float = 10_000.0,
    final_logit_softcapping: float = 30.0,
    num_kv_shared_layers: int = 0,
    hidden_size_per_layer_input: int = 0,
) -> None:
    """Write a minimal Gemma4 GGUF file for testing.

    Creates a syntactically valid GGUF with Gemma4-specific metadata:
    dual head_dim, dual RoPE theta, sliding_window_pattern, softcapping,
    and per-layer KV head counts.
    """
    from gguf import GGUFWriter

    writer = GGUFWriter(str(path), "gemma4")

    # Core metadata
    writer.add_context_length(context_length)
    writer.add_embedding_length(hidden_size)
    writer.add_feed_forward_length(intermediate_size)
    writer.add_block_count(num_layers)
    writer.add_head_count(num_heads)
    writer.add_vocab_size(vocab_size)

    # Per-layer KV head counts: sliding layers get more, full layers get fewer.
    # Pattern: every 3rd layer (0-indexed: 2, 5, ...) is full_attention.
    sliding_pattern = [(i + 1) % 3 != 0 for i in range(num_layers)]
    kv_heads_per_layer = [
        num_kv_heads_sliding if is_sliding else num_kv_heads_full
        for is_sliding in sliding_pattern
    ]
    writer.add_head_count_kv(kv_heads_per_layer)
    writer.add_sliding_window_pattern(sliding_pattern)

    # Dual head_dim
    writer.add_key_length(global_head_dim)
    # key_length_swa is a non-standard extension — write it as a raw key
    writer.add_uint32("gemma4.attention.key_length_swa", swa_head_dim)

    # Dual RoPE theta
    writer.add_rope_freq_base(global_rope_theta)
    writer.add_rope_freq_base_swa(swa_rope_theta)
    writer.add_rope_dimension_count(global_head_dim)

    # Gemma4-specific metadata
    writer.add_sliding_window(sliding_window)
    writer.add_final_logit_softcapping(final_logit_softcapping)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_shared_kv_layers(num_kv_shared_layers)
    writer.add_embedding_length_per_layer_input(hidden_size_per_layer_input)

    # Tensors
    # token_embd.weight: (vocab_size, hidden_size)
    writer.add_tensor(
        "token_embd.weight",
        np.random.randn(vocab_size, hidden_size).astype(np.float32),
    )

    for i in range(num_layers):
        is_sliding = sliding_pattern[i]
        head_dim = swa_head_dim if is_sliding else global_head_dim
        kv_heads = num_kv_heads_sliding if is_sliding else num_kv_heads_full

        writer.add_tensor(
            f"blk.{i}.attn_q.weight",
            np.random.randn(num_heads * head_dim, hidden_size).astype(np.float32),
        )
        writer.add_tensor(
            f"blk.{i}.attn_k.weight",
            np.random.randn(kv_heads * head_dim, hidden_size).astype(np.float32),
        )
        writer.add_tensor(
            f"blk.{i}.attn_v.weight",
            np.random.randn(kv_heads * head_dim, hidden_size).astype(np.float32),
        )
        writer.add_tensor(
            f"blk.{i}.attn_output.weight",
            np.random.randn(hidden_size, num_heads * head_dim).astype(np.float32),
        )
        # Q/K norms (per-head)
        writer.add_tensor(
            f"blk.{i}.attn_q_norm.weight",
            np.ones(head_dim, dtype=np.float32),
        )
        writer.add_tensor(
            f"blk.{i}.attn_k_norm.weight",
            np.ones(head_dim, dtype=np.float32),
        )
        # MLP weights
        writer.add_tensor(
            f"blk.{i}.ffn_gate.weight",
            np.random.randn(intermediate_size, hidden_size).astype(np.float32),
        )
        writer.add_tensor(
            f"blk.{i}.ffn_up.weight",
            np.random.randn(intermediate_size, hidden_size).astype(np.float32),
        )
        writer.add_tensor(
            f"blk.{i}.ffn_down.weight",
            np.random.randn(hidden_size, intermediate_size).astype(np.float32),
        )
        # Norm weights (Gemma4 has 4 norms per layer)
        writer.add_tensor(
            f"blk.{i}.attn_norm.weight",
            np.ones(hidden_size, dtype=np.float32),
        )
        writer.add_tensor(
            f"blk.{i}.post_attention_norm.weight",
            np.ones(hidden_size, dtype=np.float32),
        )
        writer.add_tensor(
            f"blk.{i}.ffn_norm.weight",
            np.ones(hidden_size, dtype=np.float32),
        )
        writer.add_tensor(
            f"blk.{i}.post_ffw_norm.weight",
            np.ones(hidden_size, dtype=np.float32),
        )
        # Layer scalar
        writer.add_tensor(
            f"blk.{i}.layer_output_scale.weight",
            np.ones(1, dtype=np.float32),
        )

    # Output norm
    writer.add_tensor(
        "output_norm.weight",
        np.ones(hidden_size, dtype=np.float32),
    )
    # Output (lm_head)
    writer.add_tensor(
        "output.weight",
        np.random.randn(vocab_size, hidden_size).astype(np.float32),
    )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.fixture
def gemma4_gguf(tmp_path: Path) -> Path:
    """Create a minimal Gemma4 GGUF file for testing."""
    path = tmp_path / "test_gemma4.gguf"
    _write_gemma4_test_gguf(path)
    return path


class TestGGUFModelReader:
    """Tests for GGUFModel reading and metadata extraction."""

    def test_read_architecture(self, llama_gguf: Path):
        model = GGUFModel(llama_gguf)
        assert model.architecture == "llama"

    def test_read_metadata(self, llama_gguf: Path):
        model = GGUFModel(llama_gguf)
        meta = model.metadata
        assert meta["llama.embedding_length"] == 64
        assert meta["llama.block_count"] == 2
        assert meta["llama.attention.head_count"] == 4
        assert meta["llama.attention.head_count_kv"] == 2
        assert meta["llama.feed_forward_length"] == 128
        assert meta["llama.context_length"] == 512

    def test_get_metadata_with_default(self, llama_gguf: Path):
        model = GGUFModel(llama_gguf)
        assert model.get_metadata("nonexistent.key", 42) == 42
        assert model.get_metadata("llama.embedding_length") == 64

    def test_tensor_names(self, llama_gguf: Path):
        model = GGUFModel(llama_gguf)
        names = model.tensor_names
        assert "token_embd.weight" in names
        assert "blk.0.attn_q.weight" in names
        assert "output.weight" in names
        # 1 embed + 2 layers x 9 tensors + 1 norm + 1 output = 21
        assert len(names) == 21

    def test_get_tensor(self, llama_gguf: Path):
        model = GGUFModel(llama_gguf)
        tensor = model.get_tensor("token_embd.weight")
        assert tensor.shape == (256, 64)
        assert tensor.dtype == np.float32

    def test_get_tensor_not_found(self, llama_gguf: Path):
        model = GGUFModel(llama_gguf)
        with pytest.raises(KeyError, match="not_a_tensor"):
            model.get_tensor("not_a_tensor")

    def test_tensor_items_iterates_all(self, llama_gguf: Path):
        model = GGUFModel(llama_gguf)
        items = list(model.tensor_items())
        assert len(items) == 21
        names = [name for name, _ in items]
        assert "token_embd.weight" in names

    def test_tensor_shapes(self, llama_gguf: Path):
        """Verify key tensor shapes match the tiny config."""
        model = GGUFModel(llama_gguf)
        embed = model.get_tensor("token_embd.weight")
        assert embed.shape == (256, 64)

        q = model.get_tensor("blk.0.attn_q.weight")
        assert q.shape == (64, 64)  # num_heads * head_dim = 4*16 = 64

        k = model.get_tensor("blk.0.attn_k.weight")
        assert k.shape == (32, 64)  # num_kv_heads * head_dim = 2*16 = 32

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="not found"):
            GGUFModel(tmp_path / "nonexistent.gguf")

    def test_repr(self, llama_gguf: Path):
        model = GGUFModel(llama_gguf)
        r = repr(model)
        assert "llama" in r
        assert "21" in r  # tensor count

    def test_tied_embeddings_no_output(self, tied_gguf: Path):
        model = GGUFModel(tied_gguf)
        assert "output.weight" not in model.tensor_names


class TestConfigMapping:
    """Tests for GGUF metadata → ArchitectureConfig mapping."""

    def test_llama_config_extraction(self, llama_gguf: Path):
        model = GGUFModel(llama_gguf)
        config = gguf_to_config(model)

        assert config.hidden_size == 64
        assert config.intermediate_size == 128
        assert config.num_hidden_layers == 2
        assert config.num_attention_heads == 4
        assert config.num_key_value_heads == 2
        assert config.max_position_embeddings == 512
        assert config.head_dim == 16  # 64 // 4
        assert config.hidden_act == "silu"  # default for llama

    def test_model_type_is_set(self, llama_gguf: Path):
        model = GGUFModel(llama_gguf)
        config = gguf_to_config(model)
        assert config._gguf_model_type == "llama"

    def test_arch_to_model_type_mapping(self):
        assert GGUF_ARCH_TO_MODEL_TYPE["llama"] == "llama"
        assert GGUF_ARCH_TO_MODEL_TYPE["mistral"] == "llama"
        assert GGUF_ARCH_TO_MODEL_TYPE["qwen2"] == "qwen2"
        assert GGUF_ARCH_TO_MODEL_TYPE["qwen35"] == "qwen3_5_text"
        assert GGUF_ARCH_TO_MODEL_TYPE["phi3"] == "phi3"

    def test_tie_embeddings_detected(self, tied_gguf: Path):
        model = GGUFModel(tied_gguf)
        config = gguf_to_config(model)
        assert config.tie_word_embeddings is True

    def test_no_tie_embeddings_with_output(self, llama_gguf: Path):
        model = GGUFModel(llama_gguf)
        config = gguf_to_config(model)
        assert config.tie_word_embeddings is False

    def test_infer_projection_biases_from_tensor_names(self):
        """Q/K/V, output, and MLP biases are inferred from tensor names.

        Regression for the invalid Q4 bug: Qwen2 carries Q/K/V biases in
        GGUF (``blk.N.attn_{q,k,v}.bias``). If ``attn_qkv_bias`` is not
        inferred, the graph builder omits the bias Add and the model emits
        garbage. Attention-output and MLP biases are absent for Qwen2.
        """

        class _FakeModel:
            def __init__(self, names):
                self.tensor_names = names

        qwen_like = _FakeModel(
            [
                "blk.0.attn_q.weight",
                "blk.0.attn_q.bias",
                "blk.0.attn_k.bias",
                "blk.0.attn_v.bias",
                "blk.0.ffn_down.weight",
            ]
        )
        assert _infer_attn_qkv_bias(qwen_like) is True
        assert _infer_attn_o_bias(qwen_like) is False
        assert _infer_mlp_bias(qwen_like) is False

        no_bias = _FakeModel(["blk.0.attn_q.weight", "blk.0.ffn_down.weight"])
        assert _infer_attn_qkv_bias(no_bias) is False
        assert _infer_attn_o_bias(no_bias) is False
        assert _infer_mlp_bias(no_bias) is False

        full_bias = _FakeModel(
            [
                "blk.0.attn_qkv.bias",
                "blk.0.attn_output.bias",
                "blk.0.ffn_up.bias",
            ]
        )
        assert _infer_attn_qkv_bias(full_bias) is True
        assert _infer_attn_o_bias(full_bias) is True
        assert _infer_mlp_bias(full_bias) is True

    def test_custom_rope_theta(self, tmp_path: Path):
        path = tmp_path / "rope.gguf"
        _write_test_gguf(path, rope_theta=500000.0)
        model = GGUFModel(path)
        config = gguf_to_config(model)
        assert config.rope_theta == pytest.approx(500000.0)

    def test_custom_rms_norm_eps(self, tmp_path: Path):
        path = tmp_path / "eps.gguf"
        _write_test_gguf(path, rms_norm_eps=1e-6)
        model = GGUFModel(path)
        config = gguf_to_config(model)
        assert config.rms_norm_eps == pytest.approx(1e-6)

    def test_qwen2_architecture(self, tmp_path: Path):
        path = tmp_path / "qwen2.gguf"
        _write_test_gguf(path, architecture="qwen2")
        model = GGUFModel(path)
        config = gguf_to_config(model)
        assert config._gguf_model_type == "qwen2"

    def test_extract_config_fields_with_prefix(self):
        """Test field extraction with architecture-prefixed keys."""
        metadata = {
            "llama.embedding_length": 4096,
            "llama.block_count": 32,
            "llama.attention.head_count": 32,
            "llama.feed_forward_length": 11008,
        }
        fields = _extract_config_fields("llama", metadata)
        assert fields["hidden_size"] == 4096
        assert fields["num_hidden_layers"] == 32

    def test_infer_tie_embeddings_true(self, tied_gguf: Path):
        model = GGUFModel(tied_gguf)
        assert _infer_tie_embeddings(model) is True

    def test_infer_tie_embeddings_false(self, llama_gguf: Path):
        model = GGUFModel(llama_gguf)
        assert _infer_tie_embeddings(model) is False

    def test_missing_required_field_raises(self, tmp_path: Path):
        """GGUF files missing critical metadata raise ValueError."""
        from gguf import GGUFWriter

        path = tmp_path / "incomplete.gguf"
        writer = GGUFWriter(str(path), "llama")
        # Only set embedding_length — omit block_count and head_count
        writer.add_embedding_length(64)
        writer.add_tensor(
            "token_embd.weight",
            np.ones((8, 64), dtype=np.float32),
        )
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        model = GGUFModel(path)
        with pytest.raises(ValueError, match="missing required metadata"):
            gguf_to_config(model)


class TestGemma4ConfigMapping:
    """Tests for Gemma4-specific GGUF → Gemma4Config extraction."""

    def test_gemma4_maps_to_text_model_type(self):
        assert GGUF_ARCH_TO_MODEL_TYPE["gemma4"] == "gemma4_text"

    def test_gemma4_returns_gemma4_config(self, gemma4_gguf: Path):
        from mobius._configs import Gemma4Config

        model = GGUFModel(gemma4_gguf)
        config = gguf_to_config(model)
        assert isinstance(config, Gemma4Config)
        assert config.model_type == "gemma4_text"

    def test_gemma4_dual_head_dim(self, gemma4_gguf: Path):
        """head_dim should be the SWA value; global_head_dim the full value."""
        model = GGUFModel(gemma4_gguf)
        config = gguf_to_config(model)
        assert config.head_dim == 16  # SWA head_dim
        assert config.global_head_dim == 32  # global head_dim

    def test_gemma4_dual_rope_theta(self, gemma4_gguf: Path):
        """rope_theta should be the SWA value; global_rope_theta the full value."""
        model = GGUFModel(gemma4_gguf)
        config = gguf_to_config(model)
        assert config.rope_theta == pytest.approx(10_000.0)  # SWA
        assert config.global_rope_theta == pytest.approx(1_000_000.0)  # global

    def test_gemma4_layer_types(self, gemma4_gguf: Path):
        """sliding_window_pattern bool[] → layer_types str[]."""
        model = GGUFModel(gemma4_gguf)
        config = gguf_to_config(model)
        # 6 layers, every 3rd (idx 2, 5) is full_attention
        assert config.layer_types == [
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ]

    def test_gemma4_sliding_window(self, gemma4_gguf: Path):
        model = GGUFModel(gemma4_gguf)
        config = gguf_to_config(model)
        assert config.sliding_window == 128

    def test_gemma4_softcapping(self, gemma4_gguf: Path):
        model = GGUFModel(gemma4_gguf)
        config = gguf_to_config(model)
        assert config.final_logit_softcapping == pytest.approx(30.0)
        # attn_logit_softcapping not set in test fixture → defaults to 0.0
        assert config.attn_logit_softcapping == pytest.approx(0.0)

    def test_gemma4_kv_shared_layers(self, gemma4_gguf: Path):
        model = GGUFModel(gemma4_gguf)
        config = gguf_to_config(model)
        assert config.num_kv_shared_layers == 0

    def test_gemma4_per_layer_input(self, gemma4_gguf: Path):
        model = GGUFModel(gemma4_gguf)
        config = gguf_to_config(model)
        # hidden_size_per_layer_input=0 → vocab_size_per_layer_input=0
        assert config.hidden_size_per_layer_input == 0
        assert config.vocab_size_per_layer_input == 0

    def test_gemma4_per_layer_kv_heads_uses_majority(self, tmp_path: Path):
        """Per-layer KV head array should collapse to the majority value."""
        path = tmp_path / "gemma4_mixed_kv.gguf"
        _write_gemma4_test_gguf(path, num_kv_heads_full=1)
        model = GGUFModel(path)
        config = gguf_to_config(model)
        # 4 sliding layers with 2 kv_heads, 2 full layers with 1 → majority is 2
        assert config.num_key_value_heads == 2

    def test_gemma4_partial_rotary_factor(self, gemma4_gguf: Path):
        """Base partial_rotary_factor should be 1.0 (full rotation for SWA)."""
        model = GGUFModel(gemma4_gguf)
        config = gguf_to_config(model)
        assert config.partial_rotary_factor == pytest.approx(1.0)
        assert config.global_partial_rotary_factor == pytest.approx(0.25)

    def test_gemma4_basic_fields(self, gemma4_gguf: Path):
        """Verify basic fields are correctly extracted."""
        model = GGUFModel(gemma4_gguf)
        config = gguf_to_config(model)
        assert config.hidden_size == 64
        assert config.intermediate_size == 128
        assert config.num_hidden_layers == 6
        assert config.num_attention_heads == 4
        assert config.vocab_size == 256
        assert config.rms_norm_eps == pytest.approx(1e-6)

    def test_gemma4_per_layer_input_enables_vocab(self, tmp_path: Path):
        """When hidden_size_per_layer_input > 0, vocab_size_per_layer_input = vocab_size."""
        path = tmp_path / "gemma4_pli.gguf"
        _write_gemma4_test_gguf(path, hidden_size_per_layer_input=16)
        model = GGUFModel(path)
        config = gguf_to_config(model)
        assert config.hidden_size_per_layer_input == 16
        assert config.vocab_size_per_layer_input == 256  # = vocab_size

    def test_gemma4_global_kv_heads_from_mixed_array(self, tmp_path: Path):
        """When GGUF has mixed per-layer KV heads, extract num_global_key_value_heads."""
        path = tmp_path / "gemma4_mixed_kv.gguf"
        _write_gemma4_test_gguf(path, num_kv_heads_full=1)
        model = GGUFModel(path)
        config = gguf_to_config(model)
        # Sliding layers have 2 KV heads, full layers have 1
        assert config.num_key_value_heads == 2  # majority (sliding)
        assert config.num_global_key_value_heads == 1  # minority (full)

    def test_gemma4_uniform_kv_heads_no_global(self, gemma4_gguf: Path):
        """When all layers have the same KV heads, num_global_key_value_heads is None."""
        model = GGUFModel(gemma4_gguf)
        config = gguf_to_config(model)
        # Default fixture has uniform KV heads (2 for all layers)
        assert config.num_key_value_heads == 2
        assert config.num_global_key_value_heads is None


class TestBuildFromGguf:
    """Tests for the build_from_gguf pipeline (integration)."""

    def test_import_build_from_gguf(self):
        """Verify the public API is importable."""
        from mobius.integrations.gguf import build_from_gguf

        assert callable(build_from_gguf)

    def test_build_from_gguf_file_not_found(self, tmp_path: Path):
        from mobius.integrations.gguf import build_from_gguf

        with pytest.raises(FileNotFoundError):
            build_from_gguf(tmp_path / "nonexistent.gguf")

    def test_build_from_gguf_produces_model_package(self, llama_gguf: Path):
        """Produce a ModelPackage from a valid GGUF file.

        Verifies that build_from_gguf returns a package with a
        ``'model'`` component containing an ONNX graph.
        """
        from mobius.integrations.gguf import build_from_gguf

        pkg = build_from_gguf(llama_gguf)
        assert "model" in pkg
        # Check the model has a graph
        model = pkg["model"]
        assert model.graph is not None

    def test_build_gemma4_from_gguf(self, gemma4_gguf: Path):
        """Build a Gemma4 text model from GGUF and verify the graph."""
        from mobius.integrations.gguf import build_from_gguf

        pkg = build_from_gguf(gemma4_gguf)
        assert "model" in pkg
        model = pkg["model"]
        assert model.graph is not None
