# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Fail-closed metadata, tensor-closure, shape, and storage checks for HYV3 GGUF."""

from __future__ import annotations

import math
import re
from typing import Any

from mobius.integrations.gguf._quant_registry import get_quant_spec
from mobius.integrations.gguf._spec import StorageRole
from mobius.integrations.gguf._tensor_mapping import is_known_skip

_BLOCK_RE = re.compile(r"^blk\.(\d+)\.")


def _integer(metadata: dict[str, Any], key: str) -> int:
    value = metadata[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer, got {value!r}")
    return value


def _add_attention(
    required: dict[str, tuple[int, ...]],
    optional: dict[str, tuple[int, ...]],
    names: set[str],
    *,
    layer: int,
    hidden: int,
    q_width: int,
    kv_width: int,
    head_dim: int,
) -> None:
    prefix = f"blk.{layer}."
    required[prefix + "attn_norm.weight"] = (hidden,)
    required[prefix + "attn_q_norm.weight"] = (head_dim,)
    required[prefix + "attn_k_norm.weight"] = (head_dim,)
    required[prefix + "ffn_norm.weight"] = (hidden,)
    required[prefix + "attn_output.weight"] = (hidden, q_width)

    fused = prefix + "attn_qkv.weight"
    split = tuple(prefix + f"attn_{name}.weight" for name in ("q", "k", "v"))
    if fused in names and any(name in names for name in split):
        raise ValueError(f"hy_v3 layer {layer} mixes fused and split Q/K/V weights")
    if fused in names:
        required[fused] = (q_width + 2 * kv_width, hidden)
        optional[prefix + "attn_qkv.bias"] = (q_width + 2 * kv_width,)
    else:
        required.update(
            {
                split[0]: (q_width, hidden),
                split[1]: (kv_width, hidden),
                split[2]: (kv_width, hidden),
            }
        )
        optional.update(
            {
                prefix + "attn_q.bias": (q_width,),
                prefix + "attn_k.bias": (kv_width,),
                prefix + "attn_v.bias": (kv_width,),
            }
        )


def _add_feed_forward(
    required: dict[str, tuple[int, ...]],
    optional: dict[str, tuple[int, ...]],
    names: set[str],
    *,
    layer: int,
    hidden: int,
    dense_width: int,
    expert_width: int,
    shared_width: int,
    experts: int,
) -> bool:
    prefix = f"blk.{layer}."
    router = prefix + "ffn_gate_inp.weight"
    dense_names = {
        prefix + "ffn_gate.weight",
        prefix + "ffn_up.weight",
        prefix + "ffn_down.weight",
    }
    routed_names = {
        router,
        prefix + "ffn_down_exps.weight",
        prefix + "ffn_gate_exps.weight",
        prefix + "ffn_up_exps.weight",
        prefix + "ffn_gate_up_exps.weight",
        prefix + "ffn_gate_shexp.weight",
        prefix + "ffn_up_shexp.weight",
        prefix + "ffn_down_shexp.weight",
    }
    if router not in names:
        if names & routed_names:
            raise ValueError(f"hy_v3 layer {layer} has routed tensors without a router")
        required.update(
            {
                prefix + "ffn_gate.weight": (dense_width, hidden),
                prefix + "ffn_up.weight": (dense_width, hidden),
                prefix + "ffn_down.weight": (hidden, dense_width),
            }
        )
        return False
    if names & dense_names:
        raise ValueError(f"hy_v3 layer {layer} mixes dense and routed FFN tensors")

    required[router] = (experts, hidden)
    required[prefix + "ffn_down_exps.weight"] = (
        experts,
        hidden,
        expert_width,
    )
    fused = prefix + "ffn_gate_up_exps.weight"
    split = (
        prefix + "ffn_gate_exps.weight",
        prefix + "ffn_up_exps.weight",
    )
    if fused in names and any(name in names for name in split):
        raise ValueError(f"hy_v3 layer {layer} mixes fused and split expert gate/up tensors")
    if fused in names:
        required[fused] = (experts, 2 * expert_width, hidden)
    else:
        required[split[0]] = (experts, expert_width, hidden)
        required[split[1]] = (experts, expert_width, hidden)
    required.update(
        {
            prefix + "ffn_gate_shexp.weight": (shared_width, hidden),
            prefix + "ffn_up_shexp.weight": (shared_width, hidden),
            prefix + "ffn_down_shexp.weight": (hidden, shared_width),
        }
    )
    optional[prefix + "exp_probs_b"] = (experts,)
    return True


def validate_hy_v3_tensor_contract(gguf_model) -> None:
    """Validate the exact HYV3 trunk and optional combined NextN tensor closure."""
    if gguf_model.architecture != "hy_v3":
        return
    metadata = gguf_model.metadata
    prefix = "hy_v3."
    hidden = _integer(metadata, prefix + "embedding_length")
    total_layers = _integer(metadata, prefix + "block_count")
    heads = _integer(metadata, prefix + "attention.head_count")
    kv_heads = _integer(metadata, prefix + "attention.head_count_kv")
    dense_width = _integer(metadata, prefix + "feed_forward_length")
    experts = _integer(metadata, prefix + "expert_count")
    top_k = _integer(metadata, prefix + "expert_used_count")
    expert_width = _integer(metadata, prefix + "expert_feed_forward_length")
    nextn = int(metadata.get(prefix + "nextn_predict_layers", 0))
    head_dim = int(
        metadata.get(
            prefix + "attention.key_length", metadata[prefix + "rope.dimension_count"]
        )
    )
    value_dim = int(metadata.get(prefix + "attention.value_length", head_dim))
    rope_dim = _integer(metadata, prefix + "rope.dimension_count")
    shared_width = int(
        metadata.get(prefix + "expert_shared_feed_forward_length", expert_width)
    )
    vocab = int(metadata.get(prefix + "vocab_size", 0))
    if not vocab:
        vocab = len(metadata.get("tokenizer.ggml.tokens", ()))
    epsilon = float(metadata[prefix + "attention.layer_norm_rms_epsilon"])
    route_scale = float(metadata.get(prefix + "expert_weights_scale", 1.0))
    gating = int(metadata.get(prefix + "expert_gating_func", 2))

    if (
        min(
            hidden,
            total_layers,
            heads,
            kv_heads,
            dense_width,
            experts,
            top_k,
            expert_width,
            shared_width,
            vocab,
        )
        <= 0
        or hidden % heads
        or heads % kv_heads
        or head_dim <= 0
        or value_dim != head_dim
        or rope_dim != head_dim
        or top_k > experts
        or nextn not in (0, 1)
        or nextn >= total_layers
        or not math.isfinite(epsilon)
        or epsilon <= 0
        or not math.isfinite(route_scale)
        or route_scale < 0
        or gating != 2
    ):
        raise ValueError("hy_v3 GGUF has invalid attention, MoE, or NextN geometry")

    names = set(gguf_model.tensor_names)
    has_nextn_tensors = nextn == 1 and f"blk.{total_layers - 1}.nextn.eh_proj.weight" in names
    if not nextn and any(".nextn." in name for name in names):
        raise ValueError("hy_v3 GGUF contains NextN tensors without nextn_predict_layers=1")

    trunk_layers = total_layers - nextn
    q_width = heads * head_dim
    kv_width = kv_heads * head_dim
    required: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (vocab, hidden),
        "output_norm.weight": (hidden,),
    }
    optional: dict[str, tuple[int, ...]] = {"output.weight": (vocab, hidden)}
    routed_schedule: list[bool] = []
    serialized_layers = total_layers if has_nextn_tensors else trunk_layers
    for layer in range(serialized_layers):
        _add_attention(
            required,
            optional,
            names,
            layer=layer,
            hidden=hidden,
            q_width=q_width,
            kv_width=kv_width,
            head_dim=head_dim,
        )
        routed_schedule.append(
            _add_feed_forward(
                required,
                optional,
                names,
                layer=layer,
                hidden=hidden,
                dense_width=dense_width,
                expert_width=expert_width,
                shared_width=shared_width,
                experts=experts,
            )
        )

    dense_prefix = 0
    while dense_prefix < trunk_layers and not routed_schedule[dense_prefix]:
        dense_prefix += 1
    if dense_prefix == trunk_layers or not all(routed_schedule[dense_prefix:trunk_layers]):
        raise ValueError(
            "hy_v3 trunk must have a contiguous leading dense prefix followed by routed layers"
        )

    if has_nextn_tensors:
        mtp = f"blk.{trunk_layers}.nextn."
        required.update(
            {
                mtp + "eh_proj.weight": (hidden, 2 * hidden),
                mtp + "enorm.weight": (hidden,),
                mtp + "hnorm.weight": (hidden,),
            }
        )
        optional.update(
            {
                mtp + "embed_tokens.weight": (vocab, hidden),
                mtp + "shared_head_norm.weight": (hidden,),
                mtp + "shared_head_head.weight": (vocab, hidden),
            }
        )

    actual: dict[str, tuple[int, ...]] = {}
    quant_specs = {}
    for name, _raw, qtype, shape in gguf_model.tensor_items_raw():
        if is_known_skip(name):
            continue
        actual[name] = tuple(int(dimension) for dimension in shape)
        spec = get_quant_spec(qtype)
        if spec is None:
            raise ValueError(f"hy_v3 tensor {name} has an unclassified GGUF storage type")
        quant_specs[name] = spec

    invalid_weight_storage = sorted(
        name
        for name, spec in quant_specs.items()
        if spec.role not in {StorageRole.FLOAT, StorageRole.QUANTIZED} and spec.name != "F64"
    )
    if invalid_weight_storage:
        raise ValueError(
            "hy_v3 model tensors must use float, F64, or quantized weight storage, "
            f"got non-weight storage for {invalid_weight_storage}"
        )

    allowed = set(required) | set(optional)
    missing = sorted(set(required) - set(actual))
    unexpected = sorted(set(actual) - allowed)
    malformed = {
        name: (required.get(name, optional.get(name)), actual[name])
        for name in allowed & set(actual)
        if actual[name] != required.get(name, optional.get(name))
    }
    out_of_range = sorted(
        name
        for name in actual
        if (match := _BLOCK_RE.match(name)) and int(match.group(1)) >= total_layers
    )
    if missing or unexpected or malformed or out_of_range:
        raise ValueError(
            "Invalid hy_v3 GGUF tensor closure: "
            f"missing={missing}, unexpected={unexpected}, malformed={malformed}, "
            f"out_of_range={out_of_range}"
        )

    explicit_float = {
        name
        for name in actual
        if name.endswith(
            (
                "_norm.weight",
                ".ffn_gate_inp.weight",
                ".enorm.weight",
                ".hnorm.weight",
                ".exp_probs_b",
            )
        )
    }
    quantized_auxiliary = sorted(
        name
        for name in explicit_float
        if quant_specs[name].role is not StorageRole.FLOAT and quant_specs[name].name != "F64"
    )
    if quantized_auxiliary:
        raise ValueError(
            "hy_v3 normalization and expert-selection tensors must use explicit float "
            f"storage, got quantized tensors {quantized_auxiliary}"
        )

    trunk_biases = [
        f"blk.{layer}.exp_probs_b" in actual for layer in range(dense_prefix, trunk_layers)
    ]
    if any(trunk_biases) and not all(trunk_biases):
        raise ValueError(
            "hy_v3 expert selection bias must be present in every routed trunk layer or absent"
        )
