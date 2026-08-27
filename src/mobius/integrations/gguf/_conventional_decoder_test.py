# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mobius._registry import registry
from mobius.integrations.gguf._arch_registry import get_arch_spec, try_get_arch_spec
from mobius.integrations.gguf._builder import (
    _raise_for_invalid_conventional_decoder_tensor_contract,
)
from mobius.integrations.gguf._config_mapping import gguf_to_config
from mobius.integrations.gguf._spec import Support
from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names
from mobius.integrations.gguf._tensor_processors import process_tensors
from mobius.integrations.ort_genai.genai_config import GenaiConfigGenerator
from mobius.tasks import CausalLMTask

_ARCHITECTURES = (
    "bloom",
    "codeshell",
    "command-r",
    "jais2",
    "orion",
    "pangu-embedded",
    "qwen",
    "starcoder",
    "xverse",
)


class _FakeGGUF:
    def __init__(self, architecture: str, metadata: dict, tensors: dict[str, tuple[int, ...]]):
        self.architecture = architecture
        self.metadata = metadata
        self._tensors = tensors
        self.tensor_names = list(tensors)

    def get_metadata(self, key, default=None):
        return self.metadata.get(key, default)

    def tensor_items_raw(self):
        for name, shape in self._tensors.items():
            yield name, None, SimpleNamespace(value=0), shape


def _fixture(architecture: str) -> _FakeGGUF:
    hidden, intermediate, heads = 8, 16, 2
    kv_heads = 1 if architecture in {"codeshell", "command-r", "starcoder"} else heads
    serialized_ffn = 2 * intermediate if architecture == "qwen" else intermediate
    epsilon = (
        "attention.layer_norm_rms_epsilon"
        if architecture in {"pangu-embedded", "qwen", "xverse"}
        else "attention.layer_norm_epsilon"
    )
    metadata = {
        f"{architecture}.context_length": 32,
        f"{architecture}.embedding_length": hidden,
        f"{architecture}.feed_forward_length": serialized_ffn,
        f"{architecture}.block_count": 1,
        f"{architecture}.attention.head_count": heads,
        f"{architecture}.attention.head_count_kv": kv_heads,
        f"{architecture}.vocab_size": 24,
        f"{architecture}.{epsilon}": 1e-5,
    }
    if architecture not in {"bloom", "starcoder"}:
        metadata[f"{architecture}.rope.freq_base"] = 10_000.0
        metadata[f"{architecture}.rope.dimension_count"] = hidden // heads
    if architecture == "command-r":
        metadata["command-r.logit_scale"] = 0.0625

    tensors: dict[str, tuple[int, ...]] = {"token_embd.weight": (24, hidden)}
    if architecture == "bloom":
        tensors.update(
            {
                "token_embd_norm.weight": (hidden,),
                "token_embd_norm.bias": (hidden,),
            }
        )
    if architecture == "starcoder":
        tensors["position_embd.weight"] = (32, hidden)
    tensors["output_norm.weight"] = (hidden,)
    if architecture in {"bloom", "codeshell", "jais2", "orion", "starcoder"}:
        tensors["output_norm.bias"] = (hidden,)
    if architecture in {"codeshell", "orion", "pangu-embedded", "qwen", "xverse"}:
        tensors["output.weight"] = (24, hidden)

    prefix = "blk.0."
    tensors[prefix + "attn_norm.weight"] = (hidden,)
    if architecture in {"bloom", "codeshell", "jais2", "orion", "starcoder"}:
        tensors[prefix + "attn_norm.bias"] = (hidden,)
    tensors[prefix + "attn_output.weight"] = (hidden, hidden)
    if architecture in {"bloom", "codeshell", "jais2", "pangu-embedded", "starcoder"}:
        tensors[prefix + "attn_output.bias"] = (hidden,)

    kv_dim = kv_heads * (hidden // heads)
    if architecture in {"bloom", "codeshell", "qwen", "starcoder"}:
        tensors[prefix + "attn_qkv.weight"] = (hidden + 2 * kv_dim, hidden)
        tensors[prefix + "attn_qkv.bias"] = (hidden + 2 * kv_dim,)
    else:
        tensors.update(
            {
                prefix + "attn_q.weight": (hidden, hidden),
                prefix + "attn_k.weight": (kv_dim, hidden),
                prefix + "attn_v.weight": (kv_dim, hidden),
            }
        )
        if architecture == "jais2":
            tensors.update(
                {
                    prefix + "attn_q.bias": (hidden,),
                    prefix + "attn_k.bias": (kv_dim,),
                    prefix + "attn_v.bias": (kv_dim,),
                }
            )
        if architecture == "pangu-embedded":
            tensors.update(
                {
                    prefix + "attn_q.bias": (hidden,),
                    prefix + "attn_k.bias": (kv_dim,),
                    prefix + "attn_v.bias": (kv_dim,),
                }
            )
    if architecture != "command-r":
        tensors[prefix + "ffn_norm.weight"] = (hidden,)
    if architecture in {"bloom", "codeshell", "jais2", "orion", "starcoder"}:
        tensors[prefix + "ffn_norm.bias"] = (hidden,)
    if architecture in {"command-r", "orion", "pangu-embedded", "qwen", "xverse"}:
        tensors[prefix + "ffn_gate.weight"] = (intermediate, hidden)
    tensors[prefix + "ffn_up.weight"] = (intermediate, hidden)
    tensors[prefix + "ffn_down.weight"] = (hidden, intermediate)
    if architecture in {"bloom", "codeshell", "jais2", "starcoder"}:
        tensors[prefix + "ffn_up.bias"] = (intermediate,)
        tensors[prefix + "ffn_down.bias"] = (hidden,)
    return _FakeGGUF(architecture, metadata, tensors)


@pytest.mark.parametrize("architecture", _ARCHITECTURES)
def test_conventional_decoder_config_tensor_and_graph_closure(architecture: str) -> None:
    model = _fixture(architecture)
    _raise_for_invalid_conventional_decoder_tensor_contract(model)
    config = gguf_to_config(model)
    module = registry.get(config.model_type)(config)
    graph = CausalLMTask().build(module, config)["model"]

    state = {
        mapped: torch.from_numpy(np.zeros(shape, dtype=np.float32))
        for name, shape in model._tensors.items()
        if (mapped := map_gguf_to_hf_names(name, architecture)) is not None
    }
    state = module.preprocess_weights(process_tensors(state, config))
    graph_weights = {
        name
        for name in graph.graph.initializers
        if not name.startswith("const_") and ".rotary_emb." not in name
    }
    # Tied heads have one graph owner, while preprocessors may retain the alias.
    assert graph_weights - {"model.embed_tokens.weight", "transformer.wte.weight"} <= set(
        state
    )


@pytest.mark.parametrize(
    ("architecture", "model_type", "activation", "intermediate"),
    [
        ("bloom", "bloom", "gelu", 16),
        ("codeshell", "kclgpt", "gelu_pytorch_tanh", 16),
        ("command-r", "command_r", "silu", 16),
        ("jais2", "jais2", "relu2", 16),
        ("orion", "orion", "silu", 16),
        ("pangu-embedded", "pangu_embedded", "silu", 16),
        ("qwen", "qwen", "silu", 16),
        ("starcoder", "gpt_bigcode", "gelu_pytorch_tanh", 16),
        ("xverse", "xverse", "silu", 16),
    ],
)
def test_conventional_decoder_config_defaults_match_source_configs(
    architecture: str, model_type: str, activation: str, intermediate: int
) -> None:
    config = gguf_to_config(_fixture(architecture))
    assert config.model_type == model_type
    assert config.hidden_act == activation
    assert config.intermediate_size == intermediate
    if architecture == "command-r":
        assert config.logit_scale == pytest.approx(0.0625)
        module = registry.get(config.model_type)(config)
        assert module.logit_scale == pytest.approx(0.0625)
        graph = CausalLMTask().build(module, config)["model"]
        assert any(node.op_type == "Mul" for node in graph.graph)


def test_conventional_decoder_support_and_runtime_verdicts_are_explicit() -> None:
    supported_quantized = {"command-r", "jais2", "pangu-embedded"}
    for architecture in _ARCHITECTURES:
        spec = get_arch_spec(architecture)
        assert spec.is_importable
        assert spec.runtime is Support.DEFERRED
        expected = (
            Support.SUPPORTED if architecture in supported_quantized else Support.REJECTED
        )
        assert spec.quantized_import is expected


@pytest.mark.parametrize(
    ("present", "tied"),
    [
        ({"token_embd.weight"}, True),
        ({"output.weight"}, True),
        ({"token_embd.weight", "output.weight"}, False),
    ],
)
def test_codeshell_embedding_output_alternatives_are_truthful(
    present: set[str], tied: bool
) -> None:
    model = _fixture("codeshell")
    for name in {"token_embd.weight", "output.weight"} - present:
        model._tensors.pop(name)
    model.tensor_names = list(model._tensors)
    _raise_for_invalid_conventional_decoder_tensor_contract(model)
    config = gguf_to_config(model)
    assert config.tie_word_embeddings is tied

    module = registry.get(config.model_type)(config)
    state = {
        mapped: torch.zeros(shape)
        for name, shape in model._tensors.items()
        if (mapped := map_gguf_to_hf_names(name, "codeshell")) is not None
    }
    processed = module.preprocess_weights(state)
    assert "model.embed_tokens.weight" in processed
    assert ("lm_head.weight" in processed) is not tied


def test_codeshell_rejects_missing_embedding_and_output() -> None:
    model = _fixture("codeshell")
    for name in ("token_embd.weight", "output.weight"):
        model._tensors.pop(name)
    model.tensor_names = list(model._tensors)

    with pytest.raises(ValueError, match=r"token_embd\.weight or output\.weight"):
        _raise_for_invalid_conventional_decoder_tensor_contract(model)


@pytest.mark.parametrize("name", ["token_embd.weight", "output.weight"])
def test_codeshell_validates_each_embedding_output_shape(name: str) -> None:
    model = _fixture("codeshell")
    model._tensors[name] = (23, 8)

    with pytest.raises(ValueError, match=name):
        _raise_for_invalid_conventional_decoder_tensor_contract(model)


def test_starcoder_tied_head_has_single_token_embedding_graph_owner() -> None:
    config = gguf_to_config(_fixture("starcoder"))
    module = registry.get(config.model_type)(config)
    assert config.tie_word_embeddings
    assert module.lm_head.weight is module.transformer.wte.weight

    graph = CausalLMTask().build(module, config)["model"]
    assert "transformer.wte.weight" in graph.graph.initializers
    assert "lm_head.weight" not in graph.graph.initializers


def test_jais2_tied_head_has_single_embedding_graph_owner() -> None:
    model = _fixture("jais2")
    config = gguf_to_config(model)
    assert config.tie_word_embeddings
    module = registry.get(config.model_type)(config)
    assert module.lm_head.weight is module.model.embed_tokens.weight

    graph = CausalLMTask().build(module, config)["model"]
    assert "model.embed_tokens.weight" in graph.graph.initializers
    assert "lm_head.weight" not in graph.graph.initializers


def test_command_r_requires_canonical_logit_scale_metadata() -> None:
    model = _fixture("command-r")
    model.metadata.pop("command-r.logit_scale")
    with pytest.raises(ValueError, match=r"command-r\.logit_scale"):
        _raise_for_invalid_conventional_decoder_tensor_contract(model)


@pytest.mark.parametrize("logit_scale", [0.0, -0.125, float("nan"), float("inf")])
def test_command_r_rejects_nonpositive_or_nonfinite_logit_scale(
    logit_scale: float,
) -> None:
    model = _fixture("command-r")
    model.metadata["command-r.logit_scale"] = logit_scale
    with pytest.raises(ValueError, match="finite positive"):
        _raise_for_invalid_conventional_decoder_tensor_contract(model)


@pytest.mark.parametrize("architecture", ["command-r", "orion", "pangu-embedded", "xverse"])
def test_unimplemented_fused_qkv_forms_are_rejected_early(architecture: str) -> None:
    model = _fixture(architecture)
    hidden = model.metadata[f"{architecture}.embedding_length"]
    heads = model.metadata[f"{architecture}.attention.head_count"]
    kv_heads = model.metadata[f"{architecture}.attention.head_count_kv"]
    kv_dim = kv_heads * (hidden // heads)
    for suffix in ("weight",):
        for projection in ("q", "k", "v"):
            name = f"blk.0.attn_{projection}.{suffix}"
            model._tensors.pop(name)
            model.tensor_names.remove(name)
    fused = "blk.0.attn_qkv.weight"
    model._tensors[fused] = (hidden + 2 * kv_dim, hidden)
    model.tensor_names.append(fused)

    with pytest.raises(ValueError, match="fused QKV tensors are not supported"):
        _raise_for_invalid_conventional_decoder_tensor_contract(model)


def test_pangu_embedded_requires_attention_output_bias() -> None:
    model = _fixture("pangu-embedded")
    model._tensors.pop("blk.0.attn_output.bias")
    model.tensor_names = list(model._tensors)

    with pytest.raises(ValueError, match=r"attn_output\.bias"):
        _raise_for_invalid_conventional_decoder_tensor_contract(model)


def test_pangu_embedded_rejects_partial_qkv_bias() -> None:
    model = _fixture("pangu-embedded")
    model._tensors.pop("blk.0.attn_k.bias")
    model.tensor_names = list(model._tensors)

    with pytest.raises(ValueError, match=r"attn_k\.bias"):
        _raise_for_invalid_conventional_decoder_tensor_contract(model)


@pytest.mark.parametrize(
    "factor",
    ["rope_freqs.weight", "rope_factors_long.weight", "rope_factors_short.weight"],
)
def test_pangu_embedded_tensor_rope_factors_fail_closed(factor: str) -> None:
    model = _fixture("pangu-embedded")
    model._tensors[factor] = (4,)
    model.tensor_names = list(model._tensors)
    model.metadata["pangu-embedded.rope.scaling.type"] = "longrope"

    with pytest.raises(ValueError, match=r"rope_(freqs|factors)"):
        _raise_for_invalid_conventional_decoder_tensor_contract(model)


def test_pangu_embedded_scaled_rope_metadata_fails_closed() -> None:
    model = _fixture("pangu-embedded")
    model.metadata["pangu-embedded.rope.scaling.type"] = "longrope"
    _raise_for_invalid_conventional_decoder_tensor_contract(model)

    with pytest.raises(ValueError, match="ordinary full-head RoPE"):
        gguf_to_config(model)


@pytest.mark.parametrize(
    ("suffix", "value"),
    [
        ("attention.key_length", 3),
        ("attention.value_length", 3),
        ("rope.dimension_count", 3),
    ],
)
def test_pangu_embedded_requires_full_head_rope_geometry(suffix: str, value: int) -> None:
    model = _fixture("pangu-embedded")
    model.metadata[f"pangu-embedded.{suffix}"] = value
    _raise_for_invalid_conventional_decoder_tensor_contract(model)

    with pytest.raises(ValueError, match="must equal head_dim"):
        gguf_to_config(model)


def test_pangu_embedded_tied_output_has_single_embedding_owner() -> None:
    model = _fixture("pangu-embedded")
    model._tensors.pop("output.weight")
    model.tensor_names = list(model._tensors)
    config = gguf_to_config(model)
    module = registry.get(config.model_type)(config)

    assert config.tie_word_embeddings
    assert module.lm_head.weight is module.model.embed_tokens.weight
    graph = CausalLMTask().build(module, config)["model"]
    assert "model.embed_tokens.weight" in graph.graph.initializers
    assert "lm_head.weight" not in graph.graph.initializers


@pytest.mark.parametrize(
    ("architecture", "reason_fragments"),
    [
        ("maincoder", ("after RoPE", "ordering")),
        ("mistral4", ("conditional dense/MoE", "overrides graph construction")),
        ("plamo3", ("fused QKV", "periodic full/sliding", "iSWA cache ABI")),
    ],
)
def test_remaining_dense_architecture_blockers_are_precise(
    architecture: str, reason_fragments: tuple[str, ...]
) -> None:
    spec = try_get_arch_spec(architecture)
    assert spec is not None
    assert not spec.is_importable
    assert spec.reason is not None
    for fragment in reason_fragments:
        assert fragment in spec.reason


@pytest.mark.parametrize("fused", [False, True])
def test_codeshell_split_and_fused_qkv_forms_match_import_contract(fused: bool) -> None:
    model = _fixture("codeshell")
    hidden = model.metadata["codeshell.embedding_length"]
    heads = model.metadata["codeshell.attention.head_count"]
    kv_heads = model.metadata["codeshell.attention.head_count_kv"]
    kv_dim = kv_heads * (hidden // heads)
    if not fused:
        for suffix in ("weight", "bias"):
            fused_name = f"blk.0.attn_qkv.{suffix}"
            model._tensors.pop(fused_name)
            model.tensor_names.remove(fused_name)
        model._tensors.update(
            {
                "blk.0.attn_q.weight": (hidden, hidden),
                "blk.0.attn_k.weight": (kv_dim, hidden),
                "blk.0.attn_v.weight": (kv_dim, hidden),
                "blk.0.attn_q.bias": (hidden,),
                "blk.0.attn_k.bias": (kv_dim,),
                "blk.0.attn_v.bias": (kv_dim,),
            }
        )
        model.tensor_names = list(model._tensors)

    _raise_for_invalid_conventional_decoder_tensor_contract(model)
    config = gguf_to_config(model)
    module = registry.get(config.model_type)(config)
    state = {
        mapped: torch.zeros(shape)
        for name, shape in model._tensors.items()
        if (mapped := map_gguf_to_hf_names(name, "codeshell")) is not None
    }
    processed = module.preprocess_weights(process_tensors(state, config))
    for projection in ("q", "k", "v"):
        assert f"model.layers.0.self_attn.{projection}_proj.weight" in processed


def test_large_command_r_per_head_qk_norm_is_rejected_before_graph() -> None:
    model = _fixture("command-r")
    model.metadata["command-r.block_count"] = 64
    with pytest.raises(ValueError, match="distinct per-head Q/K LayerNorm"):
        _raise_for_invalid_conventional_decoder_tensor_contract(model)


@pytest.mark.parametrize("architecture", _ARCHITECTURES)
def test_conventional_decoders_emit_generic_runtime_metadata(architecture: str) -> None:
    config = gguf_to_config(_fixture(architecture))
    metadata = GenaiConfigGenerator(
        config.model_type,
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
    ).generate()
    assert metadata["model"]["type"] == "decoder"


def test_xverse_processor_exactly_inverts_pinned_q_and_gqa_k_permutations() -> None:
    config = SimpleNamespace(
        _gguf_arch="xverse",
        model_type="xverse",
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    hidden_size = 32  # head_dim=8, so the inverse is not accidentally self-inverse.
    q = torch.arange(hidden_size * 5, dtype=torch.float32).reshape(hidden_size, 5)
    k = torch.arange(16 * 5, dtype=torch.float32).reshape(16, 5) + 1_000

    def pinned_permute(tensor: torch.Tensor, heads: int) -> torch.Tensor:
        dim = tensor.shape[0] // heads // 2
        return (
            tensor.reshape(heads, 2, dim, *tensor.shape[1:]).swapaxes(1, 2).reshape_as(tensor)
        )

    # Xverse's pinned converter uses all attention heads for Q, but the
    # query-group factor (q_heads / kv_heads) for GQA K.
    converted = {
        "model.layers.0.self_attn.q_proj.weight": pinned_permute(q, 4),
        "model.layers.0.self_attn.k_proj.weight": pinned_permute(k, 2),
    }
    result = process_tensors(converted, config)
    torch.testing.assert_close(result["model.layers.0.self_attn.q_proj.weight"], q)
    torch.testing.assert_close(result["model.layers.0.self_attn.k_proj.weight"], k)


def test_bloom_processor_restores_head_interleaved_qkv() -> None:
    config = SimpleNamespace(
        _gguf_arch="bloom",
        model_type="bloom",
        hidden_size=8,
        num_attention_heads=2,
    )
    canonical = torch.arange(24, dtype=torch.float32)
    q, k, v = canonical.reshape(3, 8)
    expected = torch.stack((q.reshape(2, 4), k.reshape(2, 4), v.reshape(2, 4)), dim=1).reshape(
        24
    )
    result = process_tensors(
        {"transformer.h.0.self_attention.query_key_value.bias": canonical}, config
    )
    torch.testing.assert_close(
        result["transformer.h.0.self_attention.query_key_value.bias"], expected
    )
