# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mobius import build_from_gguf
from mobius._testing import make_config
from mobius.integrations.gguf._arch_registry import Support, get_arch_spec
from mobius.integrations.gguf._config_mapping import _plm_postprocess
from mobius.integrations.gguf._repacker import repack_gguf_tensor
from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names

_HIDDEN = 32
_INTERMEDIATE = 64
_HEADS = 2
_KEY = 12
_ROPE = 4
_NOPE = 8
_VALUE = 6
_RANK = 8
_VOCAB = 24


def _metadata() -> dict[str, object]:
    return {
        "plm.attention.key_length": _KEY,
        "plm.attention.value_length": _VALUE,
        "plm.attention.kv_lora_rank": _RANK,
        "plm.rope.dimension_count": _ROPE,
    }


def _shapes() -> dict[str, tuple[int, ...]]:
    result = {
        "token_embd.weight": (_VOCAB, _HIDDEN),
        "output_norm.weight": (_HIDDEN,),
    }
    for layer in range(2):
        prefix = f"blk.{layer}."
        result.update(
            {
                prefix + "attn_norm.weight": (_HIDDEN,),
                prefix + "attn_q.weight": (_HEADS * _KEY, _HIDDEN),
                prefix + "attn_kv_a_mqa.weight": (_RANK + _ROPE, _HIDDEN),
                prefix + "attn_kv_a_norm.weight": (_RANK,),
                prefix + "attn_kv_b.weight": (
                    _HEADS * (_NOPE + _VALUE),
                    _RANK,
                ),
                prefix + "attn_output.weight": (_HIDDEN, _HEADS * _VALUE),
                prefix + "ffn_norm.weight": (_HIDDEN,),
                prefix + "ffn_up.weight": (_INTERMEDIATE, _HIDDEN),
                prefix + "ffn_down.weight": (_HIDDEN, _INTERMEDIATE),
            }
        )
    return result


class _FakePLM:
    architecture = "plm"

    def __init__(self, shapes=None):
        self.shapes = dict(_shapes() if shapes is None else shapes)
        self.tensor_names = list(self.shapes)

    def tensor_items_raw(self):
        return [(name, memoryview(b""), 0, shape) for name, shape in self.shapes.items()]


def _base_config():
    return make_config(
        model_type="plm",
        hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
        num_hidden_layers=2,
        num_attention_heads=_HEADS,
        num_key_value_heads=1,
        vocab_size=_VOCAB,
    )


def test_plm_config_extracts_exact_geometry_and_policy() -> None:
    config = _plm_postprocess(_base_config(), _metadata(), _FakePLM())

    assert config.head_dim == _KEY
    assert config.qk_nope_head_dim == _NOPE
    assert config.qk_rope_head_dim == _ROPE
    assert config.v_head_dim == _VALUE
    assert config.kv_lora_rank == _RANK
    assert config.num_key_value_heads == _HEADS
    assert config.q_lora_rank is None
    assert config.hidden_act == "relu2"
    assert config.tie_word_embeddings
    assert config.rope_interleave


def test_plm_tensor_mapping_keeps_fused_kv_b_and_has_no_gate() -> None:
    assert map_gguf_to_hf_names("blk.3.attn_kv_b.weight", "plm") == (
        "model.layers.3.self_attn.kv_b_proj.weight"
    )
    assert map_gguf_to_hf_names("blk.3.ffn_up.weight", "plm") == (
        "model.layers.3.mlp.up_proj.weight"
    )
    assert map_gguf_to_hf_names("blk.3.ffn_gate.weight", "plm") is None

    spec = get_arch_spec("plm")
    assert spec.quantized_import is Support.SUPPORTED
    assert spec.tensor_processor is None
    assert not spec.llama_qk_permute
    assert not spec.v_head_reorder


def test_plm_rejects_standalone_output_owner() -> None:
    shapes = _shapes()
    shapes["output.weight"] = (_VOCAB, _HIDDEN)
    with pytest.raises(ValueError, match="sole tied output owner"):
        _plm_postprocess(_base_config(), _metadata(), _FakePLM(shapes))


def test_plm_rejects_missing_or_wrong_tensor_shape() -> None:
    shapes = _shapes()
    del shapes["blk.1.attn_kv_a_norm.weight"]
    shapes["blk.0.attn_kv_b.weight"] = (_HEADS * (_NOPE + _VALUE) + 1, _RANK)

    with pytest.raises(
        ValueError,
        match=r"missing=.*attn_kv_a_norm.*shape_mismatches=.*attn_kv_b",
    ):
        _plm_postprocess(_base_config(), _metadata(), _FakePLM(shapes))


def test_plm_rejects_invalid_rope_geometry() -> None:
    metadata = _metadata()
    metadata["plm.rope.dimension_count"] = 5
    with pytest.raises(ValueError, match="must be even"):
        _plm_postprocess(_base_config(), metadata, _FakePLM())


def _write_quantized_plm(path: Path) -> dict[str, np.ndarray]:
    from gguf import GGMLQuantizationType, GGUFWriter

    hidden, intermediate, heads = 32, 64, 4
    key_dim, rope_dim, value_dim, rank = 12, 4, 8, 32
    nope_dim, vocab = key_dim - rope_dim, 32
    writer = GGUFWriter(str(path), "plm")
    writer.add_context_length(64)
    writer.add_embedding_length(hidden)
    writer.add_feed_forward_length(intermediate)
    writer.add_block_count(1)
    writer.add_head_count(heads)
    writer.add_head_count_kv(heads)
    writer.add_rope_freq_base(100_000.0)
    writer.add_rope_dimension_count(rope_dim)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_vocab_size(vocab)
    writer.add_uint32("plm.attention.key_length", key_dim)
    writer.add_uint32("plm.attention.value_length", value_dim)
    writer.add_uint32("plm.attention.kv_lora_rank", rank)

    rng = np.random.default_rng(0)
    packed: dict[str, np.ndarray] = {}

    def add_q4(name: str, shape: tuple[int, int]) -> None:
        rows, columns = shape
        assert columns % 32 == 0
        raw = np.empty((rows, columns // 32 * 18), dtype=np.uint8)
        for row in range(rows):
            for block in range(columns // 32):
                offset = block * 18
                raw[row, offset : offset + 2] = np.asarray(
                    [rng.uniform(0.01, 1.0)], dtype=np.float16
                ).view(np.uint8)
                raw[row, offset + 2 : offset + 18] = rng.integers(
                    0, 256, size=16, dtype=np.uint8
                )
        packed[name] = raw.copy()
        writer.add_tensor(name, raw, raw_dtype=GGMLQuantizationType.Q4_0)

    writer.add_tensor("output_norm.weight", np.ones(hidden, dtype=np.float32))
    writer.add_tensor("blk.0.attn_norm.weight", np.ones(hidden, dtype=np.float32))
    writer.add_tensor("blk.0.attn_kv_a_norm.weight", np.ones(rank, dtype=np.float32))
    writer.add_tensor("blk.0.ffn_norm.weight", np.ones(hidden, dtype=np.float32))
    add_q4("token_embd.weight", (vocab, hidden))
    add_q4("blk.0.attn_q.weight", (heads * key_dim, hidden))
    add_q4("blk.0.attn_kv_a_mqa.weight", (rank + rope_dim, hidden))
    add_q4(
        "blk.0.attn_kv_b.weight",
        (heads * (nope_dim + value_dim), rank),
    )
    add_q4("blk.0.attn_output.weight", (hidden, heads * value_dim))
    add_q4("blk.0.ffn_up.weight", (intermediate, hidden))
    add_q4("blk.0.ffn_down.weight", (hidden, intermediate))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return packed


def test_quantized_plm_preserves_fused_kv_b_blocks(tmp_path: Path) -> None:
    from gguf import GGMLQuantizationType

    path = tmp_path / "plm-q4.gguf"
    packed = _write_quantized_plm(path)
    package = build_from_gguf(path, keep_quantized=True)
    graph = package["model"].graph

    stem = "model.layers.0.self_attn.kv_b_proj"
    expected = repack_gguf_tensor(
        packed["blk.0.attn_kv_b.weight"],
        GGMLQuantizationType.Q4_0,
        shape=(64, 32),
    )
    np.testing.assert_array_equal(
        graph.initializers[stem + ".weight"].const_value.numpy(), expected.weight
    )
    np.testing.assert_array_equal(
        graph.initializers[stem + ".scales"].const_value.numpy(), expected.scales
    )
    np.testing.assert_array_equal(
        graph.initializers[stem + ".zero_points"].const_value.numpy(),
        expected.zero_points,
    )
    assert "lm_head.weight" not in graph.initializers
