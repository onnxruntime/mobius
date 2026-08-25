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


@pytest.mark.parametrize("architecture", ["bailingmoe", "deepseek", "dots1"])
@pytest.mark.parametrize("fused_qkv", [False, True])
def test_conventional_moe_exact_tensor_closure(
    architecture: str, fused_qkv: bool
) -> None:
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
def test_conventional_moe_all_dense_schedule_is_valid(architecture: str) -> None:
    model = _fixture(architecture, dense_prefix=2)

    _raise_for_invalid_conventional_moe_tensor_contract(model)


@pytest.mark.parametrize("dense_prefix", [-1, 3])
def test_conventional_moe_rejects_dense_prefix_outside_layer_range(
    dense_prefix: int,
) -> None:
    model = _fixture("deepseek")
    model.metadata["deepseek.leading_dense_block_count"] = dense_prefix

    with pytest.raises(ValueError, match="invalid conventional MoE geometry"):
        _raise_for_invalid_conventional_moe_tensor_contract(model)


@pytest.mark.parametrize(
    ("metadata_key", "value"),
    [
        ("dots1.attention.head_count_kv", 1),
        ("dots1.rope.dimension_count", 2),
    ],
)
def test_dots1_rejects_non_authoritative_attention_geometry(
    metadata_key: str, value: int
) -> None:
    model = _fixture("dots1")
    model.metadata[metadata_key] = value

    with pytest.raises(ValueError, match="invalid attention geometry"):
        _raise_for_invalid_conventional_moe_tensor_contract(model)
