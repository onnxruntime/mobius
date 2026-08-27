# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import onnx_ir as ir
import pytest
import torch

from mobius._registry import registry
from mobius.integrations.gguf._arch_registry import get_arch_spec
from mobius.integrations.gguf._builder import (
    _raise_for_invalid_specialized_encoder_tensor_contract,
)
from mobius.integrations.gguf._config_mapping import gguf_to_config
from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names
from mobius.tasks import FeatureExtractionTask, GGUFEncoderFeatureExtractionTask


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


def _jina_v3_tensors(*, moe: bool = False, fused_qkv: bool = True):
    hidden = 8
    intermediate = 16
    experts = 3
    tensors: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (32, hidden),
        "token_types.weight": (2, hidden),
        "token_embd_norm.weight": (hidden,),
        "token_embd_norm.bias": (hidden,),
    }
    for layer in range(2):
        prefix = f"blk.{layer}."
        if fused_qkv:
            tensors[prefix + "attn_qkv.weight"] = (3 * hidden, hidden)
            tensors[prefix + "attn_qkv.bias"] = (3 * hidden,)
        else:
            for projection in ("q", "k", "v"):
                tensors[prefix + f"attn_{projection}.weight"] = (hidden, hidden)
                tensors[prefix + f"attn_{projection}.bias"] = (hidden,)
        tensors.update(
            {
                prefix + "attn_output.weight": (hidden, hidden),
                prefix + "attn_output.bias": (hidden,),
                prefix + "attn_output_norm.weight": (hidden,),
                prefix + "attn_output_norm.bias": (hidden,),
                prefix + "layer_output_norm.weight": (hidden,),
                prefix + "layer_output_norm.bias": (hidden,),
            }
        )
        if moe and layer == 1:
            tensors.update(
                {
                    prefix + "ffn_gate_inp.weight": (experts, hidden),
                    prefix + "ffn_up_exps.weight": (experts, intermediate, hidden),
                    prefix + "ffn_down_exps.weight": (
                        experts,
                        hidden,
                        intermediate,
                    ),
                }
            )
        else:
            tensors.update(
                {
                    prefix + "ffn_up.weight": (intermediate, hidden),
                    prefix + "ffn_up.bias": (intermediate,),
                    prefix + "ffn_down.weight": (hidden, intermediate),
                    prefix + "ffn_down.bias": (hidden,),
                }
            )
    return tensors


def _jina_v3_metadata(*, moe: bool = False):
    metadata = _metadata("jina-bert-v3")
    metadata.pop("jina-bert-v3.rope.dimension_count")
    if moe:
        metadata.update(
            {
                "jina-bert-v3.moe_every_n_layers": 2,
                "jina-bert-v3.expert_count": 3,
                "jina-bert-v3.expert_used_count": 2,
                "jina-bert-v3.expert_weights_scale": 0.5,
                "jina-bert-v3.expert_weights_norm": False,
            }
        )
    return metadata


@pytest.mark.parametrize(
    ("arch", "model_type", "module_type"),
    [
        ("eurobert", "eurobert", "eurobert_gguf"),
        ("neo-bert", "neobert", "neo_bert_gguf"),
        ("nomic-bert", "nomic_bert", "nomic_bert_gguf"),
        ("jina-bert-v2", "bert", "jina_bert_v2_gguf"),
        ("jina-bert-v3", "jina-bert-v3", "jina_bert_v3_gguf"),
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
    package = GGUFEncoderFeatureExtractionTask().build(module, config)
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


@pytest.mark.parametrize("fused_qkv", [False, True])
def test_jina_v3_config_graph_and_tensor_closure(fused_qkv: bool) -> None:
    arch = "jina-bert-v3"
    tensors = _jina_v3_tensors(fused_qkv=fused_qkv)
    model = _FakeGGUF(arch, _jina_v3_metadata(), tensors)
    _raise_for_invalid_specialized_encoder_tensor_contract(model)
    config = gguf_to_config(model)
    assert config.encoder_fused_qkv is fused_qkv

    module = registry.get(get_arch_spec(arch).module_type)(config)
    graph = GGUFEncoderFeatureExtractionTask().build(module, config)["model"].graph
    mapped = {map_gguf_to_hf_names(name, arch) for name in tensors}
    owned_initializers = {
        name for name in graph.initializers if name.startswith(("token_", "layers."))
    }
    assert mapped == owned_initializers
    rotary = [node for node in graph if node.op_type == "RotaryEmbedding"]
    assert len(rotary) == 2 * config.num_hidden_layers
    assert all(node.attributes["interleaved"].value == 0 for node in rotary)
    gelu = [node for node in graph if node.op_type == "Gelu"]
    assert all(node.attributes["approximate"].value == "tanh" for node in gelu)
    assert not any(node.op_type in {"TopK", "Softmax"} for node in graph)


def test_jina_v3_rejects_malformed_schedule_and_tensor_mix() -> None:
    arch = "jina-bert-v3"
    tensors = _jina_v3_tensors(moe=True)
    metadata = _jina_v3_metadata(moe=True)
    with pytest.raises(ValueError, match="pinned loader"):
        _raise_for_invalid_specialized_encoder_tensor_contract(
            _FakeGGUF(arch, metadata, tensors)
        )

    with pytest.raises(ValueError, match="pinned loader"):
        gguf_to_config(_FakeGGUF(arch, metadata, tensors))


def test_jina_v3_exact_tensor_mapping_and_singleton_token_type_transform() -> None:
    from mobius.integrations.gguf._builder import _normalize_gguf_weights

    assert (
        map_gguf_to_hf_names("blk.1.attn_qkv.weight", "jina-bert-v3")
        == "layers.1.attention.qkv.weight"
    )
    normalized = _normalize_gguf_weights(
        {"token_type_embeddings.weight": torch.arange(8.0)},
        "jina-bert-v3",
        SimpleNamespace(num_local_experts=None),
    )
    assert normalized["token_type_embeddings.weight"].shape == (1, 8)


@pytest.mark.parametrize("fused_qkv", [False, True])
def test_jina_v3_synthetic_execution_matches_post_norm_oracle(
    fused_qkv: bool,
) -> None:
    from mobius._testing.ort_inference import OnnxModelSession

    arch = "jina-bert-v3"
    tensors = _jina_v3_tensors(fused_qkv=fused_qkv)
    metadata = _jina_v3_metadata()
    config = gguf_to_config(_FakeGGUF(arch, metadata, tensors))
    module = registry.get(get_arch_spec(arch).module_type)(config)
    package = FeatureExtractionTask().build(module, config)

    weights = {
        map_gguf_to_hf_names(name, arch): torch.zeros(shape) for name, shape in tensors.items()
    }
    token = torch.arange(1, 9, dtype=torch.float32)
    weights["token_embeddings.weight"][1] = token
    for name in tuple(weights):
        if name.endswith(
            (
                "token_embeddings_norm.weight",
                "attention_output_norm.weight",
                "layer_output_norm.weight",
            )
        ):
            weights[name] = torch.ones_like(weights[name])
    package.apply_weights(weights)

    session = OnnxModelSession(package["model"])
    try:
        output = session.run(
            {
                "input_ids": np.array([[1, 0]], dtype=np.int64),
                "attention_mask": np.array([[1, 0]], dtype=np.int64),
                "token_type_ids": np.array([[1, 1]], dtype=np.int64),
            }
        )["last_hidden_state"]
    finally:
        session.close()

    expected = token.numpy()
    for _ in range(1 + 2 * config.num_hidden_layers):
        expected = (expected - expected.mean()) / np.sqrt(expected.var() + config.rms_norm_eps)
    np.testing.assert_allclose(output[0, 0], expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(output[0, 1], np.zeros(8), rtol=0, atol=0)


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


def test_neobert_graph_preserves_contiguous_qkv_and_interleaved_rope() -> None:
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
    assert all(node.inputs[0].producer().op_type == "MatMul" for node in attention_splits)
    rotary_nodes = [node for node in graph if node.op_type == "RotaryEmbedding"]
    assert len(rotary_nodes) == 2 * config.num_hidden_layers
    assert all(node.attributes["interleaved"].value == 1 for node in rotary_nodes)

    # The pinned fused build_qkv takes three contiguous model-width views.
    projected = np.arange(24, dtype=np.float32).reshape(1, 1, 24)
    query, key, value = np.split(projected, 3, axis=-1)
    np.testing.assert_array_equal(query, [[[*range(8)]]])
    np.testing.assert_array_equal(key, [[[*range(8, 16)]]])
    np.testing.assert_array_equal(value, [[[*range(16, 24)]]])


def test_jina_uses_tanh_approximate_gelu() -> None:
    arch = "jina-bert-v2"
    config = gguf_to_config(_FakeGGUF(arch, _metadata(arch), _tensors(arch)))
    module = registry.get(get_arch_spec(arch).module_type)(config)
    graph = FeatureExtractionTask().build(module, config)["model"].graph
    gelu_nodes = [node for node in graph if node.op_type == "Gelu"]
    assert len(gelu_nodes) == config.num_hidden_layers
    assert all(node.attributes["approximate"].value == "tanh" for node in gelu_nodes)


@pytest.mark.parametrize(
    ("arch", "pooling_type"),
    [("eurobert", 1), ("neo-bert", 2), ("nomic-bert", 1), ("jina-bert-v2", 1)],
)
def test_specialized_encoder_pooling_uses_sentence_embedding_abi(
    arch: str, pooling_type: int
) -> None:
    metadata = _metadata(arch)
    metadata[f"{arch}.pooling_type"] = pooling_type
    tensors = _tensors(arch)
    config = gguf_to_config(_FakeGGUF(arch, metadata, tensors))
    module = registry.get(get_arch_spec(arch).module_type)(config)
    package = GGUFEncoderFeatureExtractionTask().build(module, config)
    assert [value.name for value in package["model"].graph.outputs] == ["sentence_embedding"]
    package.apply_weights(
        {
            map_gguf_to_hf_names(name, arch): torch.zeros(shape)
            for name, shape in tensors.items()
        }
    )
    from mobius._testing.ort_inference import OnnxModelSession

    session = OnnxModelSession(package["model"])
    try:
        output = session.run(
            {
                "input_ids": np.array([[1, 2, 0]], dtype=np.int64),
                "attention_mask": np.array([[1, 1, 0]], dtype=np.int64),
                "token_type_ids": np.array([[0, 1, 0]], dtype=np.int64),
            }
        )["sentence_embedding"]
    finally:
        session.close()
    assert output.shape == (1, 8)


def test_unknown_specialized_encoder_pooling_fails_closed() -> None:
    arch = "neo-bert"
    metadata = _metadata(arch)
    metadata[f"{arch}.pooling_type"] = 3
    with pytest.raises(ValueError, match="not a known encoder pooling"):
        gguf_to_config(_FakeGGUF(arch, metadata, _tensors(arch)))


def test_cls_pooling_selects_first_valid_left_padded_token() -> None:
    from mobius._testing.ort_inference import OnnxModelSession

    arch = "neo-bert"
    metadata = _metadata(arch)
    metadata[f"{arch}.pooling_type"] = 2
    tensors = _tensors(arch)
    config = gguf_to_config(_FakeGGUF(arch, metadata, tensors))
    module = registry.get(get_arch_spec(arch).module_type)(config)
    package = GGUFEncoderFeatureExtractionTask().build(module, config)

    weights = {
        map_gguf_to_hf_names(name, arch): torch.zeros(shape) for name, shape in tensors.items()
    }
    embeddings = torch.zeros((32, 8))
    embeddings[0] = 100.0  # Left-padding row must never be selected.
    embeddings[1] = torch.arange(1, 9, dtype=torch.float32)
    weights["token_embeddings.weight"] = embeddings
    for name in tuple(weights):
        if name.endswith("_norm.weight") or name == "output_norm.weight":
            weights[name] = torch.ones_like(weights[name])
    package.apply_weights(weights)

    session = OnnxModelSession(package["model"])
    try:
        output = session.run(
            {
                "input_ids": np.array([[0, 1, 2]], dtype=np.int64),
                "attention_mask": np.array([[0, 1, 1]], dtype=np.int64),
                "token_type_ids": np.zeros((1, 3), dtype=np.int64),
            }
        )["sentence_embedding"]
    finally:
        session.close()
    token = np.arange(1, 9, dtype=np.float32)
    expected = token / np.sqrt(np.mean(token * token) + config.rms_norm_eps)
    np.testing.assert_allclose(output, expected[None, :], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
def test_jina_alibi_graph_loads_in_reduced_precision(dtype: ir.DataType) -> None:
    from mobius._testing.ort_inference import OnnxModelSession

    arch = "jina-bert-v2"
    tensors = _tensors(arch)
    config = dataclasses.replace(
        gguf_to_config(_FakeGGUF(arch, _metadata(arch), tensors)),
        dtype=dtype,
    )
    module = registry.get(get_arch_spec(arch).module_type)(config)
    package = FeatureExtractionTask().build(module, config)
    torch_dtype = torch.float16 if dtype is ir.DataType.FLOAT16 else torch.bfloat16
    package.apply_weights(
        {
            map_gguf_to_hf_names(name, arch): torch.zeros(shape, dtype=torch_dtype)
            for name, shape in tensors.items()
        }
    )
    session = OnnxModelSession(package["model"])
    session.close()


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
