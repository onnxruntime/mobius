# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mobius._registry import registry
from mobius._testing.ort_inference import OnnxModelSession
from mobius.integrations.gguf._arch_registry import get_arch_spec
from mobius.integrations.gguf._builder import (
    _raise_for_invalid_embedding_tensor_contract,
    build_from_gguf,
)
from mobius.integrations.gguf._config_mapping import gguf_to_config
from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names
from mobius.models.gguf_embeddings import GemmaEmbeddingGGUFModel, LlamaEmbedGGUFModel
from mobius.tasks import GGUFEmbeddingFeatureExtractionTask


class _FakeEmbeddingGGUF:
    def __init__(
        self,
        architecture: str,
        *,
        pooling_type: int = 0,
        dense: bool = False,
    ):
        self.architecture = architecture
        self.metadata = {
            f"{architecture}.context_length": 32,
            f"{architecture}.embedding_length": 4,
            f"{architecture}.feed_forward_length": 8,
            f"{architecture}.block_count": 1,
            f"{architecture}.attention.head_count": 1,
            f"{architecture}.attention.head_count_kv": 1,
            f"{architecture}.attention.layer_norm_rms_epsilon": 1e-5,
            f"{architecture}.attention.causal": False,
            f"{architecture}.rope.freq_base": 10_000.0,
            f"{architecture}.rope.dimension_count": 4,
            f"{architecture}.vocab_size": 8,
            f"{architecture}.pooling_type": pooling_type,
        }
        self.tensors = _embedding_tensors(architecture, dense=dense)
        self.tensor_names = list(self.tensors)
        if architecture == "gemma-embedding":
            self.metadata.update(
                {
                    f"{architecture}.attention.sliding_window": 2,
                    f"{architecture}.attention.sliding_window_pattern": 2,
                    f"{architecture}.rope.freq_base_swa": 10_000.0,
                }
            )
            if dense:
                self.metadata.update(
                    {
                        f"{architecture}.dense_2_feat_in": 4,
                        f"{architecture}.dense_2_feat_out": 4,
                    }
                )

    def get_metadata(self, key, default=None):
        return self.metadata.get(key, default)

    def get_tensor(self, name):
        return self.tensors[name]

    def tensor_items_raw(self):
        for name, value in self.tensors.items():
            yield name, None, SimpleNamespace(value=0, name="F32"), value.shape

    def reader_tensors(self):
        for name, value in self.tensors.items():
            yield SimpleNamespace(
                name=name,
                tensor_type=SimpleNamespace(value=0, name="F32"),
                shape=tuple(reversed(value.shape)),
                n_bytes=value.nbytes,
            )

    def tensor_items(self):
        return self.tensors.items()

    @property
    def num_tensors(self):
        return len(self.tensors)


def _embedding_tensors(architecture: str, *, dense: bool) -> dict[str, np.ndarray]:
    hidden = 4
    intermediate = 8
    tensors = {
        "token_embd.weight": np.arange(32, dtype=np.float32).reshape(8, hidden) / 16,
        "output_norm.weight": np.ones(hidden, dtype=np.float32),
        "blk.0.attn_norm.weight": np.ones(hidden, dtype=np.float32),
        "blk.0.attn_q.weight": np.zeros((hidden, hidden), dtype=np.float32),
        "blk.0.attn_k.weight": np.zeros((hidden, hidden), dtype=np.float32),
        "blk.0.attn_v.weight": np.eye(hidden, dtype=np.float32),
        "blk.0.attn_output.weight": np.eye(hidden, dtype=np.float32),
        "blk.0.ffn_norm.weight": np.ones(hidden, dtype=np.float32),
        "blk.0.ffn_gate.weight": np.zeros((intermediate, hidden), dtype=np.float32),
        "blk.0.ffn_up.weight": np.zeros((intermediate, hidden), dtype=np.float32),
        "blk.0.ffn_down.weight": np.zeros((hidden, intermediate), dtype=np.float32),
    }
    if architecture == "gemma-embedding":
        tensors.update(
            {
                "blk.0.attn_q_norm.weight": np.ones(hidden, dtype=np.float32),
                "blk.0.attn_k_norm.weight": np.ones(hidden, dtype=np.float32),
                "blk.0.post_attention_norm.weight": np.ones(hidden, dtype=np.float32),
                "blk.0.post_ffw_norm.weight": np.ones(hidden, dtype=np.float32),
            }
        )
        if dense:
            tensors["dense_2.weight"] = np.eye(hidden, dtype=np.float32)
    return tensors


def _rms_norm(value: np.ndarray, eps: float) -> np.ndarray:
    return value / np.sqrt(np.mean(value * value, axis=-1, keepdims=True) + eps)


def _reference_output(
    source: _FakeEmbeddingGGUF,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
) -> np.ndarray:
    eps = float(source.metadata[f"{source.architecture}.attention.layer_norm_rms_epsilon"])
    hidden = source.tensors["token_embd.weight"][input_ids]
    if source.architecture == "gemma-embedding":
        hidden = hidden * np.sqrt(hidden.shape[-1])
    normalized = _rms_norm(hidden, eps)
    output = np.empty_like(hidden)
    for row in range(hidden.shape[0]):
        for query in range(hidden.shape[1]):
            valid = attention_mask[row].astype(bool)
            if source.architecture == "gemma-embedding":
                positions = np.arange(hidden.shape[1])
                valid &= np.abs(positions - query) <= 1
            attended = normalized[row, valid].mean(axis=0)
            if source.architecture == "gemma-embedding":
                attended = _rms_norm(attended, eps)
            output[row, query] = hidden[row, query] + attended
    output = _rms_norm(output, eps)
    pooling = int(source.metadata[f"{source.architecture}.pooling_type"])
    if pooling == 0:
        return output
    if pooling == 1:
        mask = attention_mask[..., None]
        return (output * mask).sum(axis=1) / mask.sum(axis=1)
    index = np.argmax(attention_mask, axis=1)
    if pooling == 3:
        index = attention_mask.shape[1] - 1 - np.argmax(attention_mask[:, ::-1], axis=1)
    return output[np.arange(output.shape[0]), index]


@pytest.mark.parametrize(
    ("architecture", "module_type", "module_class"),
    [
        ("gemma-embedding", "gemma_embedding_gguf", GemmaEmbeddingGGUFModel),
        ("llama-embed", "llama_embed_gguf", LlamaEmbedGGUFModel),
    ],
)
def test_embedding_architectures_use_dedicated_stateless_tasks(
    architecture: str,
    module_type: str,
    module_class: type,
) -> None:
    spec = get_arch_spec(architecture)
    assert spec.module_type == module_type
    assert registry.get(module_type) is module_class
    source = _FakeEmbeddingGGUF(architecture)
    config = gguf_to_config(source)
    package = GGUFEmbeddingFeatureExtractionTask().build(module_class(config), config)
    graph = package["model"].graph
    assert [value.name for value in graph.inputs] == ["input_ids", "attention_mask"]
    assert [value.name for value in graph.outputs] == ["last_hidden_state"]
    assert not any("past_key_values" in value.name for value in graph.inputs)


@pytest.mark.parametrize("architecture", ["gemma-embedding", "llama-embed"])
def test_embedding_build_from_gguf_returns_stateless_package(architecture: str) -> None:
    package = build_from_gguf(
        f"{architecture}.gguf",
        keep_quantized=False,
        _gguf_model=_FakeEmbeddingGGUF(architecture),
    )
    graph = package["model"].graph
    assert [value.name for value in graph.inputs] == ["input_ids", "attention_mask"]
    assert [value.name for value in graph.outputs] == ["last_hidden_state"]


@pytest.mark.parametrize("architecture", ["gemma-embedding", "llama-embed"])
def test_embedding_tensor_mapping_exactly_owns_graph_weights(architecture: str) -> None:
    source = _FakeEmbeddingGGUF(architecture)
    _raise_for_invalid_embedding_tensor_contract(source)
    config = gguf_to_config(source)
    module = registry.get(get_arch_spec(architecture).module_type)(config)
    graph = GGUFEmbeddingFeatureExtractionTask().build(module, config)["model"].graph
    mapped = {
        mapped
        for name in source.tensor_names
        if (mapped := map_gguf_to_hf_names(name, architecture)) is not None
    }
    graph_weights = {
        name
        for name in graph.initializers
        if not name.startswith("const_") and "rotary_emb" not in name
    }
    assert graph_weights == mapped


@pytest.mark.parametrize(
    ("architecture", "mutation", "message"),
    [
        ("gemma-embedding", "missing", "missing"),
        ("gemma-embedding", "unexpected", "unexpected"),
        ("gemma-embedding", "malformed", "malformed"),
        ("llama-embed", "bias", "unsupported"),
        ("llama-embed", "moe", "MoE"),
        ("llama-embed", "rope", "rope scaling"),
        ("gemma-embedding", "dense_without_pooling", "pooled output"),
    ],
)
def test_embedding_tensor_contract_fails_closed(
    architecture: str,
    mutation: str,
    message: str,
) -> None:
    source = _FakeEmbeddingGGUF(architecture)
    if mutation == "missing":
        source.tensors.pop("blk.0.ffn_down.weight")
    elif mutation == "unexpected":
        source.tensors["blk.0.unknown.weight"] = np.ones((4, 4), dtype=np.float32)
    elif mutation == "malformed":
        source.tensors["blk.0.attn_q.weight"] = np.ones((3, 4), dtype=np.float32)
    elif mutation == "bias":
        source.tensors["blk.0.attn_output.bias"] = np.ones(4, dtype=np.float32)
    elif mutation == "moe":
        source.metadata[f"{architecture}.expert_count"] = 2
    elif mutation == "dense_without_pooling":
        source = _FakeEmbeddingGGUF(architecture, dense=True, pooling_type=0)
    else:
        source.metadata[f"{architecture}.rope.scaling.type"] = "linear"
    source.tensor_names = list(source.tensors)
    with pytest.raises(ValueError, match=message):
        _raise_for_invalid_embedding_tensor_contract(source)


@pytest.mark.parametrize(
    ("architecture", "pooling_type", "dense"),
    [
        ("llama-embed", 0, False),
        ("llama-embed", 1, False),
        ("gemma-embedding", 0, False),
        ("gemma-embedding", 3, True),
    ],
)
def test_embedding_nonzero_bidirectional_padding_pooling_and_dense_parity(
    architecture: str,
    pooling_type: int,
    dense: bool,
) -> None:
    source = _FakeEmbeddingGGUF(
        architecture,
        pooling_type=pooling_type,
        dense=dense,
    )
    config = gguf_to_config(source)
    module = registry.get(get_arch_spec(architecture).module_type)(config)
    package = GGUFEmbeddingFeatureExtractionTask().build(module, config)
    package.apply_weights(
        {
            mapped: torch.from_numpy(value.copy())
            for name, value in source.tensors.items()
            if (mapped := map_gguf_to_hf_names(name, architecture)) is not None
        }
    )
    input_ids = np.array([[1, 2, 3, 0]], dtype=np.int64)
    attention_mask = np.array([[1, 1, 1, 0]], dtype=np.int64)
    expected = _reference_output(source, input_ids, attention_mask)
    session = OnnxModelSession(package["model"])
    try:
        output_name = "last_hidden_state" if pooling_type == 0 else "sentence_embedding"
        actual = session.run({"input_ids": input_ids, "attention_mask": attention_mask})[
            output_name
        ]
    finally:
        session.close()
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
