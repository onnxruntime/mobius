# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx_ir as ir
import pytest

from mobius._registry import registry
from mobius._testing.ort_inference import OnnxModelSession
from mobius.integrations.gguf._arch_registry import get_arch_spec
from mobius.integrations.gguf._builder import (
    _raise_for_invalid_maincoder_tensor_contract,
)
from mobius.integrations.gguf._config_mapping import gguf_to_config
from mobius.integrations.gguf._spec import Support
from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names
from mobius.models.maincoder import MaincoderCausalLMModel
from mobius.tasks import CausalLMTask

_LLAMA_CPP_COMMIT = "cb300598d5f90189cb69d2702f4930aaf99d32a2"
_SOURCE_MODEL = "Maincode/Maincoder-1B"
_SOURCE_REVISION = "088ec98640bdeb105f46a9ef6a1370ed5d0d2ea5"
_GGUF_REPOSITORY = "mradermacher/Maincoder-1B-GGUF"
_GGUF_REVISION = "1c963c98dfb478ea3b4719299bd85fbb5cf30899"
_GGUF_FILENAME = "Maincoder-1B.Q2_K.gguf"
_GGUF_SHA256 = "48c22fed9eb682c01580a1472330528bfa02c25c41d430409dd3947f0ec8c931"
_GGUF_SIZE = 490_467_072
_DOWNLOADED_HEADER_BYTES = 16_777_216
_GGUF_QTYPE_COUNTS = {"F32": 129, "Q2_K": 128, "Q3_K": 64, "Q4_K": 32, "Q6_K": 1}


class _FakeGGUF:
    def __init__(
        self,
        metadata: dict,
        tensors: dict[str, tuple[int, ...]],
        *,
        qtypes: dict[str, int] | None = None,
    ):
        self.architecture = "maincoder"
        self.metadata = metadata
        self._tensors = tensors
        self._qtypes = qtypes or {}
        self.tensor_names = list(tensors)

    def get_metadata(self, key, default=None):
        return self.metadata.get(key, default)

    def tensor_items_raw(self):
        for name, shape in self._tensors.items():
            yield name, None, SimpleNamespace(value=self._qtypes.get(name, 0)), shape


def _fixture(
    *,
    hidden: int = 8,
    layers: int = 1,
    heads: int = 2,
    kv_heads: int = 1,
    head_dim: int = 4,
    intermediate: int = 12,
    vocab: int = 16,
    context: int = 16,
) -> _FakeGGUF:
    metadata = {
        "general.architecture": "maincoder",
        "tokenizer.ggml.tokens": [""] * vocab,
        "maincoder.context_length": context,
        "maincoder.embedding_length": hidden,
        "maincoder.feed_forward_length": intermediate,
        "maincoder.block_count": layers,
        "maincoder.attention.head_count": heads,
        "maincoder.attention.head_count_kv": kv_heads,
        "maincoder.attention.key_length": head_dim,
        "maincoder.attention.value_length": head_dim,
        "maincoder.attention.layer_norm_rms_epsilon": 1e-5,
        "maincoder.rope.freq_base": 10_000.0,
        "maincoder.rope.dimension_count": head_dim,
    }
    tensors: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (vocab, hidden),
        "output_norm.weight": (hidden,),
    }
    kv_hidden = kv_heads * head_dim
    for layer in range(layers):
        prefix = f"blk.{layer}."
        tensors.update(
            {
                prefix + "attn_norm.weight": (hidden,),
                prefix + "attn_q.weight": (hidden, hidden),
                prefix + "attn_k.weight": (kv_hidden, hidden),
                prefix + "attn_v.weight": (kv_hidden, hidden),
                prefix + "attn_output.weight": (hidden, hidden),
                prefix + "attn_q_norm.weight": (head_dim,),
                prefix + "attn_k_norm.weight": (head_dim,),
                prefix + "ffn_norm.weight": (hidden,),
                prefix + "ffn_gate.weight": (intermediate, hidden),
                prefix + "ffn_up.weight": (intermediate, hidden),
                prefix + "ffn_down.weight": (hidden, intermediate),
            }
        )
    return _FakeGGUF(metadata, tensors)


def _write_tiny_maincoder(path: Path) -> None:
    from gguf import GGUFWriter

    model = _fixture()
    writer = GGUFWriter(str(path), "maincoder")
    writer.add_context_length(16)
    writer.add_embedding_length(8)
    writer.add_feed_forward_length(12)
    writer.add_block_count(1)
    writer.add_head_count(2)
    writer.add_head_count_kv(1)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_vocab_size(16)
    writer.add_rope_dimension_count(4)
    writer.add_rope_freq_base(10_000.0)
    writer.add_uint32("maincoder.attention.key_length", 4)
    writer.add_uint32("maincoder.attention.value_length", 4)
    rng = np.random.default_rng(11)
    for name, shape in model._tensors.items():
        values = (rng.standard_normal(shape) * 0.05).astype(np.float32)
        if name.endswith("norm.weight"):
            values.fill(1.0)
        writer.add_tensor(name, values)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_pinned_maincoder_config_and_real_tensor_census() -> None:
    """Validate the immutable Maincoder-1B profile without downloading tensor payloads."""
    assert len(_LLAMA_CPP_COMMIT) == len(_SOURCE_REVISION) == len(_GGUF_REVISION) == 40
    assert _SOURCE_MODEL == "Maincode/Maincoder-1B"
    assert _GGUF_REPOSITORY == "mradermacher/Maincoder-1B-GGUF"
    assert _GGUF_FILENAME == "Maincoder-1B.Q2_K.gguf"
    assert len(_GGUF_SHA256) == 64
    assert _GGUF_SIZE == 490_467_072
    assert _DOWNLOADED_HEADER_BYTES == 16_777_216
    assert sum(_GGUF_QTYPE_COUNTS.values()) == 354

    model = _fixture(
        hidden=1536,
        layers=32,
        heads=16,
        kv_heads=4,
        head_dim=96,
        intermediate=4096,
        vocab=151936,
        context=2048,
    )
    model.metadata["maincoder.rope.freq_base"] = 1_000_000.0
    _raise_for_invalid_maincoder_tensor_contract(model)
    assert len(model.tensor_names) == 354

    config = gguf_to_config(model)
    assert config.model_type == "maincoder"
    assert config.hidden_size == 1536
    assert config.num_hidden_layers == 32
    assert config.num_attention_heads == 16
    assert config.num_key_value_heads == 4
    assert config.head_dim == 96
    assert config.intermediate_size == 4096
    assert config.vocab_size == 151936
    assert config.max_position_embeddings == 2048
    assert config.rms_norm_eps == pytest.approx(1e-5)
    assert config.rope_theta == pytest.approx(1_000_000.0)
    assert config.hidden_act == "silu"
    assert config.attn_qk_norm is True
    assert config.attn_qk_norm_full is False
    assert config.rope_interleave is True
    assert config.tie_word_embeddings is True


def test_maincoder_route_mapping_and_graph_order_are_exact() -> None:
    model = _fixture()
    config = gguf_to_config(model)
    spec = get_arch_spec("maincoder")
    assert spec is not None
    assert spec.quantized_import is Support.REJECTED
    assert spec.runtime is Support.DEFERRED
    assert spec.llama_qk_permute is False
    assert registry.get("maincoder") is MaincoderCausalLMModel
    assert (
        map_gguf_to_hf_names("blk.7.attn_q_norm.weight", "maincoder")
        == "model.layers.7.self_attn.q_norm.weight"
    )
    assert (
        map_gguf_to_hf_names("blk.7.attn_k_norm.weight", "maincoder")
        == "model.layers.7.self_attn.k_norm.weight"
    )

    graph = CausalLMTask().build(MaincoderCausalLMModel(config), config)["model"]
    initializers = graph.graph.initializers
    assert "model.embed_tokens.weight" in initializers
    assert "lm_head.weight" not in initializers
    assert tuple(graph.graph.inputs[3].shape) == ("batch", 1, "past_sequence_len", 4)
    assert tuple(graph.graph.outputs[1].shape) == (
        "batch",
        1,
        "past_sequence_len + sequence_len",
        4,
    )

    q_norm = next(
        node
        for node in graph.graph
        if node.op_type == "RMSNormalization"
        and node.inputs[1].name == "model.layers.0.self_attn.q_norm.weight"
    )
    reshape = q_norm.inputs[0].producer()
    assert reshape is not None and reshape.op_type == "Reshape"
    rope = reshape.inputs[0].producer()
    assert rope is not None and rope.op_type == "RotaryEmbedding"
    assert int(rope.attributes["interleaved"].as_int()) == 1


def test_tiny_maincoder_gguf_closes_weight_loading_and_tied_head(tmp_path: Path) -> None:
    from mobius.integrations.gguf import build_from_gguf

    path = tmp_path / "maincoder.gguf"
    _write_tiny_maincoder(path)
    package = build_from_gguf(path)
    assert package.config.model_type == "maincoder"
    assert all(
        initializer.const_value is not None
        for initializer in package["model"].graph.initializers.values()
    )
    assert "model.embed_tokens.weight" in package["model"].graph.initializers
    assert "lm_head.weight" not in package["model"].graph.initializers


def test_maincoder_static_cache_uses_dedicated_decoder_dispatch() -> None:
    config = gguf_to_config(_fixture())
    graph = CausalLMTask(static_cache=True, max_seq_len=8).build(
        MaincoderCausalLMModel(config), config
    )["model"]
    inputs = {value.name: tuple(value.shape) for value in graph.graph.inputs}
    assert inputs["key_cache.0"] == ("batch", 8, 4)
    assert inputs["value_cache.0"] == ("batch", 8, 4)
    assert "attention_mask" not in inputs
    assert any(node.op_type == "TensorScatter" for node in graph.graph)
    assert any(node.op_type == "Attention" for node in graph.graph)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing_q_norm", "missing"),
        ("output_weight", "unexpected"),
        ("q_bias", "unexpected"),
        ("partial_rope", "equal full-head"),
        ("scaled_rope", "scaled/sectioned"),
        ("bad_k_shape", "malformed"),
        ("quantized_norm", "must use float"),
    ],
)
def test_maincoder_rejects_unsupported_layouts(mutation: str, match: str) -> None:
    model = _fixture()
    if mutation == "missing_q_norm":
        del model._tensors["blk.0.attn_q_norm.weight"]
    elif mutation == "output_weight":
        model._tensors["output.weight"] = (16, 8)
    elif mutation == "q_bias":
        model._tensors["blk.0.attn_q.bias"] = (8,)
    elif mutation == "partial_rope":
        model.metadata["maincoder.rope.dimension_count"] = 2
    elif mutation == "scaled_rope":
        model.metadata["maincoder.rope.scaling.type"] = "yarn"
    elif mutation == "bad_k_shape":
        model._tensors["blk.0.attn_k.weight"] = (8, 8)
    elif mutation == "quantized_norm":
        model._qtypes["blk.0.attn_q_norm.weight"] = 10

    model.tensor_names = list(model._tensors)
    with pytest.raises(ValueError, match=match):
        _raise_for_invalid_maincoder_tensor_contract(model)


def _interleaved_rope(x: np.ndarray, positions: np.ndarray, theta: float) -> np.ndarray:
    head_dim = x.shape[-1]
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    angles = positions[..., None, None] * inv_freq
    cos = np.cos(angles)
    sin = np.sin(angles)
    even = x[..., 0::2]
    odd = x[..., 1::2]
    result = np.empty_like(x)
    result[..., 0::2] = even * cos - odd * sin
    result[..., 1::2] = even * sin + odd * cos
    return result


def test_maincoder_synthetic_execution_cache_and_post_rope_k_norm_values() -> None:
    config = gguf_to_config(_fixture())
    graph = CausalLMTask().build(MaincoderCausalLMModel(config), config)["model"]
    rng = np.random.default_rng(29)
    for initializer in graph.graph.initializers.values():
        if initializer.const_value is not None:
            continue
        values = (rng.standard_normal(initializer.shape) * 0.08).astype(np.float32)
        if initializer.name.endswith("norm.weight"):
            values = rng.uniform(0.7, 1.3, initializer.shape).astype(np.float32)
        initializer.const_value = ir.tensor(values, name=initializer.name)

    session = OnnxModelSession(graph)
    tokens = np.array([[2, 5, 7]], dtype=np.int64)
    empty = np.empty((1, 1, 0, 4), dtype=np.float32)

    def run(input_ids: np.ndarray, past_key: np.ndarray, past_value: np.ndarray):
        total = past_key.shape[2] + input_ids.shape[1]
        return session.run(
            {
                "input_ids": input_ids,
                "attention_mask": np.ones((1, total), dtype=np.int64),
                "position_ids": np.arange(past_key.shape[2], total, dtype=np.int64).reshape(
                    1, -1
                ),
                "past_key_values.0.key": past_key,
                "past_key_values.0.value": past_value,
            }
        )

    full = run(tokens, empty, empty)
    prefix = run(tokens[:, :2], empty, empty)
    decoded = run(
        tokens[:, 2:],
        prefix["present.0.key"],
        prefix["present.0.value"],
    )
    np.testing.assert_allclose(decoded["logits"], full["logits"][:, -1:], rtol=1e-4, atol=1e-5)

    values = {
        name: value.const_value.numpy()
        for name, value in graph.graph.initializers.items()
        if value.const_value is not None
    }
    hidden = values["model.embed_tokens.weight"][tokens]
    input_norm = values["model.layers.0.input_layernorm.weight"]
    hidden = hidden / np.sqrt(np.mean(hidden * hidden, axis=-1, keepdims=True) + 1e-5)
    hidden = hidden * input_norm
    key = hidden @ values["model.layers.0.self_attn.k_proj.weight"].T
    key = key.reshape(1, 3, 1, 4)
    key = _interleaved_rope(
        key,
        np.arange(3, dtype=np.float32).reshape(1, 3),
        config.rope_theta,
    )
    key_norm = values["model.layers.0.self_attn.k_norm.weight"]
    key = key / np.sqrt(np.mean(key * key, axis=-1, keepdims=True) + 1e-5)
    key = (key * key_norm).transpose(0, 2, 1, 3)
    np.testing.assert_allclose(full["present.0.key"], key, rtol=2e-5, atol=2e-6)
