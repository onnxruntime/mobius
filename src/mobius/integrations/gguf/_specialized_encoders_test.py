# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mobius._registry import registry
from mobius.integrations.gguf._arch_registry import get_arch_spec
from mobius.integrations.gguf._builder import (
    _raise_for_invalid_specialized_encoder_tensor_contract,
)
from mobius.integrations.gguf._config_mapping import gguf_to_config
from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names
from mobius.tasks import FeatureExtractionTask


class _FakeGGUF:
    def __init__(self, architecture: str, metadata: dict, tensors: dict[str, tuple[int, ...]]):
        self.architecture = architecture
        self.metadata = metadata
        self.tensor_names = list(tensors)
        self._tensors = tensors

    def get_metadata(self, key, default=None):
        return self.metadata.get(key, default)

    def tensor_items_raw(self):
        for name, shape in self._tensors.items():
            yield name, None, SimpleNamespace(value=0, name="F32"), shape


def _metadata(arch: str, *, intermediate: int = 16) -> dict:
    epsilon = (
        "attention.layer_norm_rms_epsilon"
        if arch in {"eurobert", "neo-bert"}
        else "attention.layer_norm_epsilon"
    )
    metadata = {
        f"{arch}.context_length": 128,
        f"{arch}.embedding_length": 8,
        f"{arch}.feed_forward_length": intermediate,
        f"{arch}.block_count": 2,
        f"{arch}.attention.head_count": 2,
        f"{arch}.attention.head_count_kv": 2,
        f"{arch}.attention.causal": False,
        f"{arch}.{epsilon}": 1e-5,
        f"{arch}.vocab_size": 32,
        f"{arch}.pooling_type": 0,
        "tokenizer.ggml.token_type_count": 2,
    }
    if arch != "jina-bert-v2":
        metadata.update(
            {
                f"{arch}.rope.freq_base": 1000.0,
                f"{arch}.rope.dimension_count": 4,
            }
        )
    return metadata


def _tensors(arch: str, *, jina_fused: bool = True) -> dict[str, tuple[int, ...]]:
    hidden = 8
    intermediate = 16
    tensors: dict[str, tuple[int, ...]] = {"token_embd.weight": (32, hidden)}
    if arch == "eurobert":
        tensors["output_norm.weight"] = (hidden,)
    elif arch == "neo-bert":
        tensors["enc.output_norm.weight"] = (hidden,)
    else:
        tensors.update(
            {
                "token_embd_norm.weight": (hidden,),
                "token_embd_norm.bias": (hidden,),
            }
        )
        if arch == "jina-bert-v2":
            tensors["token_types.weight"] = (2, hidden)

    for layer in range(2):
        prefix = f"blk.{layer}."
        if arch in {"eurobert", "neo-bert"}:
            tensors.update(
                {
                    prefix + "attn_norm.weight": (hidden,),
                    prefix + "attn_output.weight": (hidden, hidden),
                    prefix + "ffn_norm.weight": (hidden,),
                    prefix + "ffn_down.weight": (hidden, intermediate),
                }
            )
            if arch == "eurobert":
                tensors.update(
                    {
                        prefix + "attn_q.weight": (hidden, hidden),
                        prefix + "attn_k.weight": (hidden, hidden),
                        prefix + "attn_v.weight": (hidden, hidden),
                        prefix + "ffn_gate.weight": (intermediate, hidden),
                        prefix + "ffn_up.weight": (intermediate, hidden),
                    }
                )
            else:
                tensors[prefix + "attn_qkv.weight"] = (3 * hidden, hidden)
                tensors[prefix + "ffn_up.weight"] = (2 * intermediate, hidden)
            continue

        tensors.update(
            {
                prefix + "attn_q.weight": (hidden, hidden),
                prefix + "attn_k.weight": (hidden, hidden),
                prefix + "attn_v.weight": (hidden, hidden),
                prefix + "attn_output.weight": (hidden, hidden),
                prefix + "attn_output_norm.weight": (hidden,),
                prefix + "attn_output_norm.bias": (hidden,),
                prefix + "ffn_down.weight": (hidden, intermediate),
                prefix + "layer_output_norm.weight": (hidden,),
                prefix + "layer_output_norm.bias": (hidden,),
            }
        )
        if arch == "nomic-bert":
            tensors[prefix + "ffn_gate.weight"] = (intermediate, hidden)
            tensors[prefix + "ffn_up.weight"] = (intermediate, hidden)
        else:
            tensors.update(
                {
                    prefix + "attn_q.bias": (hidden,),
                    prefix + "attn_k.bias": (hidden,),
                    prefix + "attn_v.bias": (hidden,),
                    prefix + "attn_output.bias": (hidden,),
                    prefix + "ffn_up.weight": (
                        2 * intermediate if jina_fused else intermediate,
                        hidden,
                    ),
                    prefix + "ffn_up.bias": (
                        2 * intermediate if jina_fused else intermediate,
                    ),
                    prefix + "ffn_down.bias": (hidden,),
                }
            )
            if not jina_fused:
                tensors[prefix + "ffn_gate.weight"] = (intermediate, hidden)
    return tensors


@pytest.mark.parametrize(
    ("arch", "model_type", "module_type"),
    [
        ("eurobert", "eurobert", "eurobert_gguf"),
        ("neo-bert", "neobert", "neo_bert_gguf"),
        ("nomic-bert", "nomic_bert", "nomic_bert_gguf"),
        ("jina-bert-v2", "bert", "jina_bert_v2_gguf"),
    ],
)
def test_promoted_capabilities_are_float_only(arch, model_type, module_type) -> None:
    spec = get_arch_spec(arch)
    assert spec.is_importable
    assert spec.model_type == model_type
    assert spec.module_type == module_type
    assert spec.runtime.value == "deferred"
    assert spec.quantized_import.value == "rejected"


@pytest.mark.parametrize("arch", ["eurobert", "neo-bert", "nomic-bert", "jina-bert-v2"])
def test_config_and_tiny_graph_follow_encoder_abi(arch: str) -> None:
    tensors = _tensors(arch)
    config = gguf_to_config(_FakeGGUF(arch, _metadata(arch), tensors))
    assert config.num_key_value_heads == config.num_attention_heads == 2
    assert config.max_position_embeddings == 128
    assert config.pooling_type == 0
    assert (config.rope_type is None) == (arch == "jina-bert-v2")
    if arch == "neo-bert":
        # The converter has already applied floor(2 * HF intermediate_size / 3).
        assert config.intermediate_size == 16
    if arch == "jina-bert-v2":
        assert config.encoder_fused_geglu
        assert config.attn_o_bias

    module = registry.get(get_arch_spec(arch).module_type)(config)
    package = FeatureExtractionTask().build(module, config)
    graph = package["model"].graph
    assert [value.name for value in graph.outputs] == ["last_hidden_state"]
    assert not any("past_key_values" in value.name for value in graph.inputs)
    mapped = {map_gguf_to_hf_names(name, arch) for name in tensors}
    owned_initializers = {
        name
        for name in graph.initializers
        if name.startswith(("token_", "layers.", "output_norm."))
    }
    assert mapped == owned_initializers


@pytest.mark.parametrize("arch", ["eurobert", "neo-bert", "nomic-bert", "jina-bert-v2"])
def test_exact_closure_accepts_and_mutations_fail(arch: str) -> None:
    tensors = _tensors(arch)
    model = _FakeGGUF(arch, _metadata(arch), tensors)
    _raise_for_invalid_specialized_encoder_tensor_contract(model)

    missing = dict(tensors)
    missing.pop(next(name for name in missing if name.startswith("blk.0.")))
    with pytest.raises(ValueError, match="tensor closure"):
        _raise_for_invalid_specialized_encoder_tensor_contract(
            _FakeGGUF(arch, model.metadata, missing)
        )

    extra = dict(tensors)
    extra["blk.0.unowned.weight"] = (8, 8)
    with pytest.raises(ValueError, match="unexpected"):
        _raise_for_invalid_specialized_encoder_tensor_contract(
            _FakeGGUF(arch, model.metadata, extra)
        )


@pytest.mark.parametrize(
    ("arch", "gguf_name", "target"),
    [
        ("eurobert", "blk.1.ffn_gate.weight", "layers.1.mlp.gate.weight"),
        ("neo-bert", "blk.0.attn_qkv.weight", "layers.0.attention.qkv.weight"),
        (
            "nomic-bert",
            "blk.1.attn_output_norm.bias",
            "layers.1.attention_output_norm.bias",
        ),
        (
            "jina-bert-v2",
            "blk.0.attn_q_norm.weight",
            "layers.0.attention.q_norm.weight",
        ),
    ],
)
def test_exact_tensor_mapping(arch: str, gguf_name: str, target: str) -> None:
    assert map_gguf_to_hf_names(gguf_name, arch) == target


def test_jina_separate_and_fused_geglu_are_discriminated_by_shape() -> None:
    for fused in (False, True):
        tensors = _tensors("jina-bert-v2", jina_fused=fused)
        model = _FakeGGUF("jina-bert-v2", _metadata("jina-bert-v2"), tensors)
        _raise_for_invalid_specialized_encoder_tensor_contract(model)
        config = gguf_to_config(model)
        assert config.encoder_fused_geglu is fused


def test_synthetic_fused_split_values_preserve_gate_up_order() -> None:
    values = np.arange(64, dtype=np.float32).reshape(1, 32, 2)
    gate, up = np.split(values, 2, axis=1)
    expected = (gate / (1.0 + np.exp(-gate))) * up
    actual = (values[:, :16] / (1.0 + np.exp(-values[:, :16]))) * values[:, 16:]
    np.testing.assert_allclose(actual, expected)


def test_neobert_graph_preserves_per_head_qkv_and_interleaved_rope() -> None:
    arch = "neo-bert"
    config = gguf_to_config(_FakeGGUF(arch, _metadata(arch), _tensors(arch)))
    module = registry.get(get_arch_spec(arch).module_type)(config)
    graph = FeatureExtractionTask().build(module, config)["model"].graph

    attention_splits = [
        node
        for node in graph
        if node.op_type == "Split" and "/attention/" in (node.name or "")
    ]
    assert len(attention_splits) == config.num_hidden_layers
    assert all(node.inputs[0].producer().op_type == "Reshape" for node in attention_splits)
    rotary_nodes = [node for node in graph if node.op_type == "RotaryEmbedding"]
    assert len(rotary_nodes) == 2 * config.num_hidden_layers
    assert all(node.attributes["interleaved"].value == 1 for node in rotary_nodes)

    # A head-interleaved projection must split each head's 3*head_dim chunk.
    projected = np.arange(24, dtype=np.float32).reshape(1, 1, 2, 12)
    query, key, value = np.split(projected, 3, axis=-1)
    np.testing.assert_array_equal(query.reshape(1, 1, 8), [[[*range(4), *range(12, 16)]]])
    np.testing.assert_array_equal(key.reshape(1, 1, 8), [[[*range(4, 8), *range(16, 20)]]])
    np.testing.assert_array_equal(value.reshape(1, 1, 8), [[[*range(8, 12), *range(20, 24)]]])


def test_jina_uses_tanh_approximate_gelu() -> None:
    arch = "jina-bert-v2"
    config = gguf_to_config(_FakeGGUF(arch, _metadata(arch), _tensors(arch)))
    module = registry.get(get_arch_spec(arch).module_type)(config)
    graph = FeatureExtractionTask().build(module, config)["model"].graph
    gelu_nodes = [node for node in graph if node.op_type == "Gelu"]
    assert len(gelu_nodes) == config.num_hidden_layers
    assert all(node.attributes["approximate"].value == "tanh" for node in gelu_nodes)


@pytest.mark.parametrize("arch", ["eurobert", "neo-bert", "nomic-bert", "jina-bert-v2"])
def test_pooled_specialized_encoder_files_fail_closed(arch: str) -> None:
    metadata = _metadata(arch)
    metadata[f"{arch}.pooling_type"] = 1
    with pytest.raises(ValueError, match="token-level last_hidden_state only"):
        gguf_to_config(_FakeGGUF(arch, metadata, _tensors(arch)))


@pytest.mark.parametrize(
    ("arch", "revision", "metadata_updates", "expected"),
    [
        (
            "eurobert",
            "EuroBERT/EuroBERT-210m@39b51e15dd1f1a06f58b5cbf6a8a188cec60bd0e",
            {
                "context_length": 8192,
                "embedding_length": 768,
                "feed_forward_length": 3072,
                "block_count": 12,
                "attention.head_count": 12,
                "attention.head_count_kv": 12,
                "rope.dimension_count": 64,
                "rope.freq_base": 250000.0,
            },
            (768, 3072, 12, 8192, 250000.0),
        ),
        (
            "neo-bert",
            "chandar-lab/NeoBERT@5424c8efeea6491b151d62dee55a752165407430",
            {
                "context_length": 4096,
                "embedding_length": 768,
                # conversion/bert.py stores floor(2 * HF intermediate_size / 3).
                "feed_forward_length": 2048,
                "block_count": 28,
                "attention.head_count": 12,
                "attention.head_count_kv": 12,
                "rope.dimension_count": 64,
                "rope.freq_base": 10000.0,
            },
            (768, 2048, 28, 4096, 10000.0),
        ),
        (
            "nomic-bert",
            "nomic-ai/nomic-embed-text-v1.5@e9b6763023c676ca8431644204f50c2b100d9aab",
            {
                # conversion/bert.py normalizes n_positions=8192/max_trained=2048.
                "context_length": 2048,
                "embedding_length": 768,
                "feed_forward_length": 3072,
                "block_count": 12,
                "attention.head_count": 12,
                "attention.head_count_kv": 12,
                "rope.dimension_count": 64,
                "rope.freq_base": 1000.0,
            },
            (768, 3072, 12, 2048, 1000.0),
        ),
        (
            "jina-bert-v2",
            "jinaai/jina-embeddings-v2-small-en@44e7d1d6caec8c883c2d4b207588504d519788d0",
            {
                "context_length": 8192,
                "embedding_length": 512,
                "feed_forward_length": 2048,
                "block_count": 4,
                "attention.head_count": 8,
                "attention.head_count_kv": 8,
            },
            (512, 2048, 4, 8192, None),
        ),
    ],
)
def test_pinned_hf_config_semantics_survive_gguf_conversion(
    arch: str, revision: str, metadata_updates: dict, expected: tuple
) -> None:
    assert "@" in revision
    metadata = _metadata(arch)
    metadata.update({f"{arch}.{key}": value for key, value in metadata_updates.items()})
    names = {"token_embd.weight": (1, 1)}
    if arch in {"nomic-bert", "jina-bert-v2"}:
        names["token_types.weight"] = (2, metadata_updates["embedding_length"])
    config = gguf_to_config(_FakeGGUF(arch, metadata, names))
    assert (
        config.hidden_size,
        config.intermediate_size,
        config.num_hidden_layers,
        config.max_position_embeddings,
        config.rope_theta,
    ) == expected


@pytest.mark.parametrize("arch", ["eurobert", "neo-bert", "nomic-bert", "jina-bert-v2"])
def test_synthetic_zero_weight_ort_parity(arch: str) -> None:
    from mobius._testing.ort_inference import OnnxModelSession

    tensors = _tensors(arch)
    config = gguf_to_config(_FakeGGUF(arch, _metadata(arch), tensors))
    module = registry.get(get_arch_spec(arch).module_type)(config)
    package = FeatureExtractionTask().build(module, config)
    package.apply_weights(
        {
            map_gguf_to_hf_names(name, arch): torch.zeros(shape)
            for name, shape in tensors.items()
        }
    )
    session = OnnxModelSession(package["model"])
    try:
        output = session.run(
            {
                "input_ids": np.array([[1, 2, 0]], dtype=np.int64),
                "attention_mask": np.array([[1, 1, 0]], dtype=np.int64),
                "token_type_ids": np.array([[0, 1, 0]], dtype=np.int64),
            }
        )["last_hidden_state"]
    finally:
        session.close()
    np.testing.assert_array_equal(output, np.zeros((1, 3, 8), dtype=np.float32))
