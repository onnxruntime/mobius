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
    _raise_for_invalid_conventional_decoder_tensor_contract,
)
from mobius.integrations.gguf._config_mapping import gguf_to_config
from mobius.integrations.gguf._spec import Support
from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names
from mobius.tasks import CausalLMTask

_ARCHITECTURES = ("gptneox", "jais", "mpt", "refact", "ernie4_5", "openelm")


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
    hidden, layers, head_dim, vocab = 16, 2, 4, 32
    if architecture == "openelm":
        heads = [2, 4]
        kv_heads = [1, 2]
        intermediate = [24, 32]
    else:
        heads = 4
        kv_heads = 1 if architecture == "refact" else 4
        intermediate = 32
    metadata = {
        f"{architecture}.context_length": 32,
        f"{architecture}.embedding_length": hidden,
        f"{architecture}.feed_forward_length": intermediate,
        f"{architecture}.block_count": layers,
        f"{architecture}.attention.head_count": heads,
        f"{architecture}.attention.head_count_kv": kv_heads,
        f"{architecture}.attention.key_length": head_dim,
        f"{architecture}.vocab_size": vocab,
    }
    if architecture in {"gptneox", "jais", "mpt"}:
        metadata[f"{architecture}.attention.layer_norm_epsilon"] = 1e-5
    else:
        metadata[f"{architecture}.attention.layer_norm_rms_epsilon"] = 1e-5
    if architecture == "gptneox":
        metadata["gptneox.use_parallel_residual"] = True
        metadata["gptneox.rope.dimension_count"] = 2
        metadata["gptneox.rope.freq_base"] = 10_000.0
    elif architecture in {"jais", "mpt"}:
        metadata[f"{architecture}.attention.max_alibi_bias"] = 8.0
    elif architecture in {"ernie4_5", "openelm"}:
        metadata[f"{architecture}.rope.dimension_count"] = head_dim
        metadata[f"{architecture}.rope.freq_base"] = 10_000.0

    tensors: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (vocab, hidden),
        "output_norm.weight": (hidden,),
    }
    if architecture in {"gptneox", "jais"}:
        tensors["output_norm.bias"] = (hidden,)
        tensors["output.weight"] = (vocab, hidden)
    elif architecture not in {"openelm"}:
        tensors["output.weight"] = (vocab, hidden)

    for layer in range(layers):
        layer_heads = heads[layer] if isinstance(heads, list) else heads
        layer_kv_heads = kv_heads[layer] if isinstance(kv_heads, list) else kv_heads
        layer_intermediate = (
            intermediate[layer] if isinstance(intermediate, list) else intermediate
        )
        q_dim = layer_heads * head_dim
        kv_dim = layer_kv_heads * head_dim
        prefix = f"blk.{layer}."
        tensors[prefix + "attn_norm.weight"] = (hidden,)
        tensors[prefix + "ffn_norm.weight"] = (hidden,)
        tensors[prefix + "attn_output.weight"] = (hidden, q_dim)
        if architecture in {"gptneox", "jais"}:
            tensors[prefix + "attn_norm.bias"] = (hidden,)
            tensors[prefix + "ffn_norm.bias"] = (hidden,)
            tensors[prefix + "attn_output.bias"] = (hidden,)
        if architecture in {"gptneox", "jais", "mpt", "openelm"}:
            tensors[prefix + "attn_qkv.weight"] = (q_dim + 2 * kv_dim, hidden)
            if architecture in {"gptneox", "jais"}:
                tensors[prefix + "attn_qkv.bias"] = (q_dim + 2 * kv_dim,)
        else:
            tensors[prefix + "attn_q.weight"] = (q_dim, hidden)
            tensors[prefix + "attn_k.weight"] = (kv_dim, hidden)
            tensors[prefix + "attn_v.weight"] = (kv_dim, hidden)
        if architecture == "openelm":
            tensors[prefix + "attn_q_norm.weight"] = (head_dim,)
            tensors[prefix + "attn_k_norm.weight"] = (head_dim,)
        if architecture in {"jais", "refact", "ernie4_5", "openelm"}:
            tensors[prefix + "ffn_gate.weight"] = (layer_intermediate, hidden)
        tensors[prefix + "ffn_up.weight"] = (layer_intermediate, hidden)
        tensors[prefix + "ffn_down.weight"] = (hidden, layer_intermediate)
        if architecture == "gptneox":
            tensors[prefix + "ffn_up.bias"] = (layer_intermediate,)
            tensors[prefix + "ffn_down.bias"] = (hidden,)
        elif architecture == "jais":
            tensors[prefix + "ffn_gate.bias"] = (layer_intermediate,)
            tensors[prefix + "ffn_up.bias"] = (layer_intermediate,)
            tensors[prefix + "ffn_down.bias"] = (hidden,)
    return _FakeGGUF(architecture, metadata, tensors)


@pytest.mark.parametrize("architecture", _ARCHITECTURES)
def test_exact_legacy_config_tensor_and_graph_closure(architecture: str) -> None:
    model = _fixture(architecture)
    _raise_for_invalid_conventional_decoder_tensor_contract(model)
    config = gguf_to_config(model)
    spec = get_arch_spec(architecture)
    module = registry.get(spec.module_type)(config)
    graph = CausalLMTask().build(module, config)["model"]

    state = {
        mapped: torch.from_numpy(np.arange(np.prod(shape), dtype=np.float32).reshape(shape))
        for name, shape in model._tensors.items()
        if (mapped := map_gguf_to_hf_names(name, architecture)) is not None
    }
    processed = module.preprocess_weights(state)
    graph_weights = {
        name
        for name in graph.graph.initializers
        if not name.startswith("const_") and ".rotary_emb." not in name
    }
    assert graph_weights - {"model.embed_tokens.weight"} <= set(processed)
    assert (
        config.model_type
        == {
            "gptneox": "gpt_neox",
            "jais": "jais",
            "mpt": "mpt",
            "refact": "refact",
            "ernie4_5": "ernie4_5",
            "openelm": "openelm",
        }[architecture]
    )
    assert spec.runtime is Support.DEFERRED
    if architecture == "ernie4_5":
        assert config.rope_interleave


def test_openelm_preserves_per_layer_geometry_and_fused_row_order() -> None:
    model = _fixture("openelm")
    config = gguf_to_config(model)
    assert config.layer_attention_head_counts == (2, 4)
    assert config.layer_attention_kv_head_counts == (1, 2)
    assert config.layer_intermediate_sizes == (24, 32)
    module = registry.get("gguf_legacy")(config)
    fused = torch.arange((8 + 2 * 4) * 16, dtype=torch.float32).reshape(16, 16)
    state = module.preprocess_weights({"model.layers.0.self_attn.qkv_proj.weight": fused})
    torch.testing.assert_close(state["model.layers.0.self_attn.q_proj.weight"], fused[:8])
    torch.testing.assert_close(state["model.layers.0.self_attn.k_proj.weight"], fused[8:12])
    torch.testing.assert_close(state["model.layers.0.self_attn.v_proj.weight"], fused[12:])


def test_gptneox_uses_default_rope_when_frequency_base_is_omitted() -> None:
    model = _fixture("gptneox")
    del model.metadata["gptneox.rope.freq_base"]

    config = gguf_to_config(model)
    assert config.rope_type == "default"
    assert config.rope_theta == pytest.approx(10_000.0)
    assert config.partial_rotary_factor == pytest.approx(0.5)
    assert config.head_dim == 4

    module = registry.get("gguf_legacy")(config)
    CausalLMTask().build(module, config)


def test_gptneox_splits_converter_reformatted_qkv_rows() -> None:
    model = _fixture("gptneox")
    config = gguf_to_config(model)
    module = registry.get("gguf_legacy")(config)
    heads, head_dim, hidden = 4, 4, 16

    # The pinned converter changes HF's per-head [Q, K, V] rows into
    # llama.cpp's contiguous [all Q, all K, all V] fused tensor.
    hf_fused = torch.arange(3 * hidden * hidden, dtype=torch.float32).reshape(
        heads, 3, head_dim, hidden
    )
    converted = torch.cat(
        tuple(hf_fused[:, index].reshape(hidden, hidden) for index in range(3))
    )
    state = module.preprocess_weights({"model.layers.0.self_attn.qkv_proj.weight": converted})

    for index, projection in enumerate(("q", "k", "v")):
        torch.testing.assert_close(
            state[f"model.layers.0.self_attn.{projection}_proj.weight"],
            hf_fused[:, index].reshape(hidden, hidden),
        )


def test_mpt_accepts_complete_bias_family_and_tied_output() -> None:
    model = _fixture("mpt")
    del model._tensors["output.weight"]
    model._tensors["output_norm.bias"] = (16,)
    for layer in range(2):
        prefix = f"blk.{layer}."
        model._tensors.update(
            {
                prefix + "attn_norm.bias": (16,),
                prefix + "ffn_norm.bias": (16,),
                prefix + "attn_qkv.bias": (48,),
                prefix + "attn_output.bias": (16,),
                prefix + "ffn_up.bias": (32,),
                prefix + "ffn_down.bias": (16,),
            }
        )
    model.tensor_names = list(model._tensors)

    _raise_for_invalid_conventional_decoder_tensor_contract(model)
    config = gguf_to_config(model)
    assert config.tie_word_embeddings
    assert config.attn_qkv_bias
    assert config.attn_o_bias
    assert config.mlp_bias
    CausalLMTask().build(registry.get("gguf_legacy")(config), config)


@pytest.mark.parametrize("architecture", ["refact", "ernie4_5"])
def test_split_projection_architectures_accept_tied_output(architecture: str) -> None:
    model = _fixture(architecture)
    del model._tensors["output.weight"]
    model.tensor_names = list(model._tensors)

    _raise_for_invalid_conventional_decoder_tensor_contract(model)
    config = gguf_to_config(model)
    assert config.tie_word_embeddings
    CausalLMTask().build(registry.get("gguf_legacy")(config), config)


def test_exact_legacy_quantized_import_boundary_matches_required_transforms() -> None:
    for architecture in _ARCHITECTURES:
        spec = get_arch_spec(architecture)
        assert spec.quantized_import is Support.REJECTED
        reason = (spec.reason or "").lower()
        assert "quant" in reason
        assert "split" in reason or "projection targets" in reason


def test_exact_legacy_static_cache_fails_closed() -> None:
    model = _fixture("openelm")
    config = gguf_to_config(model)
    module = registry.get("gguf_legacy")(config)
    with pytest.raises(TypeError, match="Static cache mode"):
        CausalLMTask(static_cache=True).build(module, config)


@pytest.mark.parametrize(
    ("architecture", "mutation", "message"),
    [
        (
            "gptneox",
            lambda model: model.metadata.__setitem__("gptneox.use_parallel_residual", False),
            "parallel",
        ),
        (
            "mpt",
            lambda model: model.metadata.__setitem__("mpt.attention.clamp_kqv", 1.0),
            "clamp_kqv",
        ),
        (
            "refact",
            lambda model: model._tensors.__setitem__("blk.0.attn_output.bias", (16,)),
            "unexpected",
        ),
        (
            "ernie4_5",
            lambda model: model.metadata.__setitem__(
                "ernie4_5.rope.dimension_sections", [1, 1, 2]
            ),
            "dimension_sections",
        ),
        (
            "openelm",
            lambda model: model.metadata.__setitem__("openelm.attention.head_count_kv", [1]),
            "block_count",
        ),
    ],
)
def test_exact_legacy_loader_alternatives_fail_closed(
    architecture: str, mutation, message: str
) -> None:
    model = _fixture(architecture)
    mutation(model)
    model.tensor_names = list(model._tensors)
    with pytest.raises(ValueError, match=message):
        _raise_for_invalid_conventional_decoder_tensor_contract(model)
