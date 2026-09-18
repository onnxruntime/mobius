# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mobius.integrations.gguf._builder import (
    _raise_for_invalid_conventional_moe_tensor_contract,
)


class _FakeGGUF:
    def __init__(
        self,
        architecture: str,
        metadata: dict[str, object],
        tensors: dict[str, tuple[int, ...]],
    ):
        self.architecture = architecture
        self.metadata = metadata
        self._tensors = tensors
        self.tensor_names = list(tensors)

    def tensor_items_raw(self):
        for name, shape in self._tensors.items():
            yield name, None, SimpleNamespace(value=0), shape


def _fixture(
    architecture: str,
    *,
    dense_prefix: int = 0,
    fused_qkv: bool = False,
) -> _FakeGGUF:
    hidden, intermediate = 8, 16
    heads, head_dim = 2, 4
    kv_heads = heads if architecture == "dots1" else 1
    experts, expert_intermediate, shared_experts = 4, 6, 1
    layers, vocab = 2, 24
    metadata: dict[str, object] = {
        f"{architecture}.context_length": 32,
        f"{architecture}.embedding_length": hidden,
        f"{architecture}.feed_forward_length": intermediate,
        f"{architecture}.block_count": layers,
        f"{architecture}.attention.head_count": heads,
        f"{architecture}.attention.head_count_kv": kv_heads,
        f"{architecture}.attention.layer_norm_rms_epsilon": 1e-5,
        f"{architecture}.rope.dimension_count": head_dim,
        f"{architecture}.vocab_size": vocab,
        f"{architecture}.expert_count": experts,
        f"{architecture}.expert_used_count": 2,
        f"{architecture}.expert_feed_forward_length": expert_intermediate,
        f"{architecture}.expert_shared_count": shared_experts,
        f"{architecture}.leading_dense_block_count": dense_prefix,
    }
    tensors: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (vocab, hidden),
        "output_norm.weight": (hidden,),
        "output.weight": (vocab, hidden),
    }
    q_width, kv_width = heads * head_dim, kv_heads * head_dim
    for layer in range(layers):
        prefix = f"blk.{layer}."
        tensors.update(
            {
                prefix + "attn_norm.weight": (hidden,),
                prefix + "attn_output.weight": (hidden, q_width),
                prefix + "ffn_norm.weight": (hidden,),
            }
        )
        if fused_qkv:
            tensors[prefix + "attn_qkv.weight"] = (q_width + 2 * kv_width, hidden)
            tensors[prefix + "attn_qkv.bias"] = (q_width + 2 * kv_width,)
        else:
            tensors.update(
                {
                    prefix + "attn_q.weight": (q_width, hidden),
                    prefix + "attn_k.weight": (kv_width, hidden),
                    prefix + "attn_v.weight": (kv_width, hidden),
                }
            )
        if architecture == "dots1":
            tensors[prefix + "attn_q_norm.weight"] = (head_dim,)
            tensors[prefix + "attn_k_norm.weight"] = (head_dim,)
        if layer < dense_prefix:
            tensors.update(
                {
                    prefix + "ffn_gate.weight": (intermediate, hidden),
                    prefix + "ffn_up.weight": (intermediate, hidden),
                    prefix + "ffn_down.weight": (hidden, intermediate),
                }
            )
        else:
            tensors.update(
                {
                    prefix + "ffn_gate_inp.weight": (experts, hidden),
                    prefix + "ffn_gate_exps.weight": (
                        experts,
                        expert_intermediate,
                        hidden,
                    ),
                    prefix + "ffn_up_exps.weight": (
                        experts,
                        expert_intermediate,
                        hidden,
                    ),
                    prefix + "ffn_down_exps.weight": (
                        experts,
                        hidden,
                        expert_intermediate,
                    ),
                    prefix + "ffn_gate_shexp.weight": (expert_intermediate, hidden),
                    prefix + "ffn_up_shexp.weight": (expert_intermediate, hidden),
                    prefix + "ffn_down_shexp.weight": (hidden, expert_intermediate),
                }
            )
            if architecture == "dots1":
                tensors[prefix + "exp_probs_b.bias"] = (experts,)
    return _FakeGGUF(architecture, metadata, tensors)


def test_dots1_preserves_qwen2_qk_layout_while_llama_derived_promotions_permute() -> None:
    from mobius.integrations.gguf._arch_registry import get_arch_spec

    assert get_arch_spec("dots1").llama_qk_permute is False
    assert get_arch_spec("bailingmoe").llama_qk_permute is True
    assert get_arch_spec("deepseek").llama_qk_permute is True


@pytest.mark.parametrize("architecture", ["bailingmoe", "deepseek", "dots1"])
@pytest.mark.parametrize("fused_qkv", [False, True])
def test_conventional_moe_exact_tensor_closure(architecture: str, fused_qkv: bool) -> None:
    dense_prefix = 0 if architecture == "bailingmoe" else 1
    model = _fixture(architecture, dense_prefix=dense_prefix, fused_qkv=fused_qkv)

    _raise_for_invalid_conventional_moe_tensor_contract(model)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing_expert", "missing="),
        ("mixed_schedule", "unexpected="),
        ("partial_qkv", "complete QKV layout"),
        ("partial_bias", "partial attention Q/K/V"),
        ("malformed_shape", "malformed="),
        ("out_of_range", "out_of_range="),
        ("unexpected", "unexpected="),
    ],
)
def test_conventional_moe_tensor_contract_rejects_malformed_sources(
    mutation: str, match: str
) -> None:
    model = _fixture("dots1", dense_prefix=1)
    if mutation == "missing_expert":
        del model._tensors["blk.1.ffn_up_exps.weight"]
    elif mutation == "mixed_schedule":
        model._tensors["blk.0.ffn_gate_inp.weight"] = (4, 8)
    elif mutation == "partial_qkv":
        del model._tensors["blk.0.attn_v.weight"]
    elif mutation == "partial_bias":
        model._tensors["blk.0.attn_q.bias"] = (8,)
    elif mutation == "malformed_shape":
        model._tensors["blk.1.ffn_down_exps.weight"] = (4, 7, 6)
    elif mutation == "out_of_range":
        model._tensors["blk.2.ffn_gate_inp.weight"] = (4, 8)
    elif mutation == "unexpected":
        model._tensors["blk.1.ffn_gate_up_exps.weight"] = (4, 12, 8)
    model.tensor_names = list(model._tensors)

    with pytest.raises(ValueError, match=match):
        _raise_for_invalid_conventional_moe_tensor_contract(model)


def test_deepseek_allows_tied_output_but_other_promotions_require_head() -> None:
    deepseek = _fixture("deepseek")
    del deepseek._tensors["output.weight"]
    deepseek.tensor_names = list(deepseek._tensors)
    _raise_for_invalid_conventional_moe_tensor_contract(deepseek)

    dots1 = _fixture("dots1")
    del dots1._tensors["output.weight"]
    dots1.tensor_names = list(dots1._tensors)
    with pytest.raises(ValueError, match="missing="):
        _raise_for_invalid_conventional_moe_tensor_contract(dots1)


@pytest.mark.parametrize("architecture", ["deepseek", "dots1"])
@pytest.mark.parametrize("expert_metadata", ["absent", "zero"])
def test_conventional_moe_all_dense_schedule_is_valid_without_active_experts(
    architecture: str, expert_metadata: str
) -> None:
    model = _fixture(architecture, dense_prefix=2)
    suffixes = (
        "expert_count",
        "expert_used_count",
        "expert_feed_forward_length",
        "expert_shared_count",
    )
    for suffix in suffixes:
        key = f"{architecture}.{suffix}"
        if expert_metadata == "absent":
            model.metadata.pop(key)
        else:
            model.metadata[key] = 0

    _raise_for_invalid_conventional_moe_tensor_contract(model)


@pytest.mark.parametrize("dense_prefix", [-1, 3])
def test_conventional_moe_rejects_dense_prefix_outside_layer_range(
    dense_prefix: int,
) -> None:
    model = _fixture("deepseek")
    model.metadata["deepseek.leading_dense_block_count"] = dense_prefix

    with pytest.raises(ValueError, match="invalid conventional MoE geometry"):
        _raise_for_invalid_conventional_moe_tensor_contract(model)


@pytest.mark.parametrize("metadata_key", ["dots1.attention.head_count_kv"])
def test_dots1_rejects_non_authoritative_attention_geometry(
    metadata_key: str,
) -> None:
    model = _fixture("dots1")
    model.metadata[metadata_key] = 1

    with pytest.raises(ValueError, match="invalid attention geometry"):
        _raise_for_invalid_conventional_moe_tensor_contract(model)


@pytest.mark.parametrize("architecture", ["bailingmoe", "deepseek", "dots1"])
def test_conventional_moe_rejects_partial_rope(architecture: str) -> None:
    model = _fixture(architecture)
    model.metadata[f"{architecture}.rope.dimension_count"] = 2

    with pytest.raises(ValueError, match="invalid attention geometry"):
        _raise_for_invalid_conventional_moe_tensor_contract(model)


@pytest.mark.parametrize("architecture", ["bailingmoe", "deepseek", "dots1"])
def test_conventional_moe_tensor_contract_rejects_unsupported_rope_scaling(
    architecture: str,
) -> None:
    model = _fixture(architecture)
    model.metadata[f"{architecture}.rope.scaling.type"] = "linear"

    with pytest.raises(ValueError, match="only unscaled and YaRN RoPE are exact"):
        _raise_for_invalid_conventional_moe_tensor_contract(model)


@pytest.mark.parametrize("dense_prefix", [0, 2])
def test_deepseek_accepts_single_expert_routed_and_all_dense(
    dense_prefix: int,
) -> None:
    model = _fixture("deepseek", dense_prefix=dense_prefix)
    model.metadata["deepseek.expert_count"] = 1
    model.metadata["deepseek.expert_used_count"] = 1
    for name, shape in list(model._tensors.items()):
        if name.endswith("ffn_gate_inp.weight"):
            model._tensors[name] = (1, shape[1])
        elif "_exps." in name:
            model._tensors[name] = (1, *shape[1:])

    _raise_for_invalid_conventional_moe_tensor_contract(model)


@pytest.mark.parametrize(
    ("metadata_key", "value"),
    [
        ("attention.layer_norm_rms_epsilon", 0.0),
        ("attention.layer_norm_rms_epsilon", float("nan")),
        ("expert_weights_scale", -1.0),
        ("expert_weights_scale", float("inf")),
    ],
)
def test_conventional_moe_rejects_invalid_scalar_metadata(
    metadata_key: str, value: float
) -> None:
    model = _fixture("deepseek")
    model.metadata[f"deepseek.{metadata_key}"] = value

    with pytest.raises(ValueError, match="invalid normalization or routing scale"):
        _raise_for_invalid_conventional_moe_tensor_contract(model)
