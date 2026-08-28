# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Strict tensor closure for MiniMax-M2, Mistral4, and GLM-DSA GGUF graphs."""

from __future__ import annotations

import re

from mobius._configs import ArchitectureConfig
from mobius.integrations.gguf._arch_registry import try_get_arch_spec
from mobius.integrations.gguf._tensor_mapping import is_known_skip


def _tensor_shapes(model) -> dict[str, tuple[int, ...]]:
    return {
        name: tuple(int(dimension) for dimension in shape)
        for name, _raw, _qtype, shape in model.tensor_items_raw()
    }


def _validate_exact_closure(
    architecture: str,
    shapes: dict[str, tuple[int, ...]],
    required: dict[str, tuple[int, ...]],
    optional: dict[str, tuple[int, ...]],
    *,
    layers: int,
) -> None:
    actual = set(shapes)
    allowed = set(required) | set(optional)
    unexpected = sorted(name for name in actual - allowed if not is_known_skip(name))
    out_of_range = sorted(
        name
        for name in actual
        if (match := re.match(r"^blk\.(\d+)\.", name)) and int(match.group(1)) >= layers
    )
    missing = sorted(set(required) - actual)
    malformed = {
        name: (expected, shapes.get(name))
        for name, expected in {**required, **optional}.items()
        if name in shapes and shapes[name] != expected
    }
    if missing or unexpected or malformed or out_of_range:
        raise ValueError(
            f"Invalid {architecture} GGUF tensor closure: missing={missing}, "
            f"unexpected={unexpected}, malformed={malformed}, "
            f"out_of_range={out_of_range}"
        )


def _require_float_auxiliaries(model, names: set[str], *, architecture: str) -> None:
    non_float = sorted(
        name
        for name, _raw, qtype, _shape in model.tensor_items_raw()
        if name in names and getattr(qtype, "name", "") not in {"F32", "F16", "BF16"}
    )
    if non_float:
        raise ValueError(
            f"{architecture} normalization/router sidecars must use float storage: {non_float}"
        )


def _validate_minimax_m2(model) -> None:
    metadata = model.metadata
    arch = "minimax-m2"
    layers = int(metadata[f"{arch}.block_count"])
    hidden = int(metadata[f"{arch}.embedding_length"])
    heads = int(metadata[f"{arch}.attention.head_count"])
    kv_heads = int(metadata[f"{arch}.attention.head_count_kv"])
    head_dim = int(metadata[f"{arch}.attention.key_length"])
    value_dim = int(metadata[f"{arch}.attention.value_length"])
    intermediate = int(metadata[f"{arch}.feed_forward_length"])
    expert_width = int(metadata[f"{arch}.expert_feed_forward_length"])
    experts = int(metadata[f"{arch}.expert_count"])
    top_k = int(metadata[f"{arch}.expert_used_count"])
    vocab = int(metadata.get(f"{arch}.vocab_size", 0)) or len(
        metadata.get("tokenizer.ggml.tokens", ())
    )
    if (
        min(
            layers,
            hidden,
            heads,
            kv_heads,
            head_dim,
            intermediate,
            experts,
            top_k,
            vocab,
        )
        <= 0
        or heads % kv_heads
        or value_dim != head_dim
        or expert_width != intermediate
        or top_k > experts
    ):
        raise ValueError("MiniMax-M2 GGUF has inconsistent architecture geometry")

    actual = set(model.tensor_names)
    fused = sorted(name for name in actual if ".attn_qkv." in name)
    ignored_biases = sorted(
        name for name in actual if name.endswith(("attn_q.bias", "attn_k.bias", "attn_v.bias"))
    )
    if fused:
        raise ValueError(
            "MiniMax-M2 fused QKV is accepted by the pinned loader but its graph "
            f"cannot execute it: {fused}"
        )
    if ignored_biases:
        raise ValueError(
            "MiniMax-M2 Q/K/V biases are accepted by the pinned loader but ignored "
            f"by its graph: {ignored_biases}"
        )

    q_width = heads * head_dim
    kv_width = kv_heads * head_dim
    required: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (vocab, hidden),
        "output_norm.weight": (hidden,),
        "output.weight": (vocab, hidden),
    }
    float_auxiliaries = {"output_norm.weight"}
    for layer in range(layers):
        prefix = f"blk.{layer}."
        required.update(
            {
                prefix + "attn_norm.weight": (hidden,),
                prefix + "attn_q.weight": (q_width, hidden),
                prefix + "attn_k.weight": (kv_width, hidden),
                prefix + "attn_v.weight": (kv_width, hidden),
                prefix + "attn_q_norm.weight": (q_width,),
                prefix + "attn_k_norm.weight": (kv_width,),
                prefix + "attn_output.weight": (hidden, q_width),
                prefix + "ffn_norm.weight": (hidden,),
                prefix + "ffn_gate_inp.weight": (experts, hidden),
                prefix + "ffn_gate_exps.weight": (
                    experts,
                    intermediate,
                    hidden,
                ),
                prefix + "ffn_up_exps.weight": (
                    experts,
                    intermediate,
                    hidden,
                ),
                prefix + "ffn_down_exps.weight": (
                    experts,
                    hidden,
                    intermediate,
                ),
                prefix + "exp_probs_b.bias": (experts,),
            }
        )
        float_auxiliaries.update(
            {
                prefix + "attn_norm.weight",
                prefix + "attn_q_norm.weight",
                prefix + "attn_k_norm.weight",
                prefix + "ffn_norm.weight",
                prefix + "ffn_gate_inp.weight",
                prefix + "exp_probs_b.bias",
            }
        )
    shapes = _tensor_shapes(model)
    _validate_exact_closure(arch, shapes, required, {}, layers=layers)
    _require_float_auxiliaries(model, float_auxiliaries, architecture=arch)


def _validate_mistral4(model) -> None:
    metadata = model.metadata
    arch = "mistral4"
    nextn = int(metadata.get(f"{arch}.nextn_predict_layers", 0))
    if nextn:
        raise ValueError("Mistral4's pinned graph does not execute NextN tensors")
    layers = int(metadata[f"{arch}.block_count"])
    hidden = int(metadata[f"{arch}.embedding_length"])
    dense_width = int(metadata[f"{arch}.feed_forward_length"])
    heads = int(metadata[f"{arch}.attention.head_count"])
    kv_heads = int(metadata[f"{arch}.attention.head_count_kv"])
    q_lora = int(metadata[f"{arch}.attention.q_lora_rank"])
    kv_lora = int(metadata[f"{arch}.attention.kv_lora_rank"])
    qk_dim = int(metadata[f"{arch}.attention.key_length_mla"])
    rope_dim = int(metadata[f"{arch}.rope.dimension_count"])
    nope_dim = qk_dim - rope_dim
    value_dim = int(metadata[f"{arch}.attention.value_length_mla"])
    dense_prefix = int(metadata.get(f"{arch}.leading_dense_block_count", 0))
    experts = int(metadata[f"{arch}.expert_count"])
    top_k = int(metadata[f"{arch}.expert_used_count"])
    expert_width = int(metadata[f"{arch}.expert_feed_forward_length"])
    shared = int(metadata[f"{arch}.expert_shared_count"])
    vocab = int(metadata.get(f"{arch}.vocab_size", 0)) or len(
        metadata.get("tokenizer.ggml.tokens", ())
    )
    if (
        min(
            layers,
            hidden,
            dense_width,
            heads,
            q_lora,
            kv_lora,
            nope_dim,
            rope_dim,
            value_dim,
            experts,
            top_k,
            expert_width,
            shared,
            vocab,
        )
        <= 0
        or kv_heads != 1
        or int(metadata[f"{arch}.attention.key_length"]) != kv_lora + rope_dim
        or int(metadata[f"{arch}.attention.value_length"]) != kv_lora
        or top_k > experts
        or not 0 <= dense_prefix < layers
    ):
        raise ValueError("Mistral4 GGUF has inconsistent MLA/MoE geometry")

    required: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (vocab, hidden),
        "output_norm.weight": (hidden,),
    }
    optional: dict[str, tuple[int, ...]] = {
        "output.weight": (vocab, hidden),
    }
    float_auxiliaries = {"output_norm.weight"}
    actual = set(model.tensor_names)
    for layer in range(layers):
        prefix = f"blk.{layer}."
        required.update(
            {
                prefix + "attn_norm.weight": (hidden,),
                prefix + "attn_q_a_norm.weight": (q_lora,),
                prefix + "attn_kv_a_norm.weight": (kv_lora,),
                prefix + "attn_q_a.weight": (q_lora, hidden),
                prefix + "attn_q_b.weight": (heads * qk_dim, q_lora),
                prefix + "attn_kv_a_mqa.weight": (
                    kv_lora + rope_dim,
                    hidden,
                ),
                prefix + "attn_k_b.weight": (heads, kv_lora, nope_dim),
                prefix + "attn_v_b.weight": (heads, value_dim, kv_lora),
                prefix + "attn_output.weight": (hidden, heads * value_dim),
                prefix + "ffn_norm.weight": (hidden,),
            }
        )
        float_auxiliaries.update(
            {
                prefix + "attn_norm.weight",
                prefix + "attn_q_a_norm.weight",
                prefix + "attn_kv_a_norm.weight",
                prefix + "ffn_norm.weight",
            }
        )
        if layer < dense_prefix:
            required.update(
                {
                    prefix + "ffn_gate.weight": (dense_width, hidden),
                    prefix + "ffn_up.weight": (dense_width, hidden),
                    prefix + "ffn_down.weight": (hidden, dense_width),
                }
            )
            continue

        required.update(
            {
                prefix + "ffn_gate_inp.weight": (experts, hidden),
                prefix + "ffn_down_exps.weight": (
                    experts,
                    hidden,
                    expert_width,
                ),
                prefix + "ffn_gate_shexp.weight": (
                    expert_width * shared,
                    hidden,
                ),
                prefix + "ffn_up_shexp.weight": (
                    expert_width * shared,
                    hidden,
                ),
                prefix + "ffn_down_shexp.weight": (
                    hidden,
                    expert_width * shared,
                ),
            }
        )
        fused = prefix + "ffn_gate_up_exps.weight"
        gate = prefix + "ffn_gate_exps.weight"
        up = prefix + "ffn_up_exps.weight"
        has_fused = fused in actual
        has_separate = gate in actual or up in actual
        if has_fused == has_separate or (has_separate and not {gate, up} <= actual):
            raise ValueError(
                f"Mistral4 layer {layer} must contain exactly one fused or split "
                "expert gate/up representation"
            )
        if has_fused:
            required[fused] = (experts, 2 * expert_width, hidden)
        else:
            required[gate] = (experts, expert_width, hidden)
            required[up] = (experts, expert_width, hidden)
        optional[prefix + "exp_probs_b.bias"] = (experts,)
        float_auxiliaries.add(prefix + "ffn_gate_inp.weight")
        if prefix + "exp_probs_b.bias" in actual:
            float_auxiliaries.add(prefix + "exp_probs_b.bias")

    shapes = _tensor_shapes(model)
    _validate_exact_closure(arch, shapes, required, optional, layers=layers)
    _require_float_auxiliaries(model, float_auxiliaries, architecture=arch)


def _validate_glm_dsa(model) -> None:
    metadata = model.metadata
    arch = model.architecture
    nextn = int(metadata.get(f"{arch}.nextn_predict_layers", 0))
    total_layers = int(metadata[f"{arch}.block_count"])
    layers = total_layers - nextn
    if layers <= 0:
        raise ValueError("GLM-DSA block_count must exceed nextn_predict_layers")
    hidden = int(metadata[f"{arch}.embedding_length"])
    dense_width = int(metadata[f"{arch}.feed_forward_length"])
    heads = int(metadata[f"{arch}.attention.head_count"])
    q_lora = int(metadata[f"{arch}.attention.q_lora_rank"])
    kv_lora = int(metadata[f"{arch}.attention.kv_lora_rank"])
    qk_dim = int(metadata[f"{arch}.attention.key_length_mla"])
    rope_dim = int(metadata[f"{arch}.rope.dimension_count"])
    nope_dim = qk_dim - rope_dim
    value_dim = int(metadata[f"{arch}.attention.value_length_mla"])
    dense_prefix = int(metadata.get(f"{arch}.leading_dense_block_count", 0))
    experts = int(metadata[f"{arch}.expert_count"])
    top_k = int(metadata[f"{arch}.expert_used_count"])
    expert_width = int(metadata[f"{arch}.expert_feed_forward_length"])
    shared = int(metadata[f"{arch}.expert_shared_count"])
    index_heads = int(metadata[f"{arch}.attention.indexer.head_count"])
    index_dim = int(metadata[f"{arch}.attention.indexer.key_length"])
    vocab = int(metadata.get(f"{arch}.vocab_size", 0)) or len(
        metadata.get("tokenizer.ggml.tokens", ())
    )
    if (
        min(
            hidden,
            dense_width,
            heads,
            q_lora,
            kv_lora,
            nope_dim,
            rope_dim,
            value_dim,
            experts,
            top_k,
            expert_width,
            shared,
            index_heads,
            index_dim,
            vocab,
        )
        <= 0
    ):
        raise ValueError("GLM-DSA GGUF has inconsistent MLA/DSA/MoE geometry")
    if (
        int(metadata[f"{arch}.attention.head_count_kv"]) != 1
        or int(metadata[f"{arch}.attention.key_length"]) != kv_lora + rope_dim
        or int(metadata[f"{arch}.attention.value_length"]) != kv_lora
        or top_k > experts
        or not 0 <= dense_prefix < layers
    ):
        raise ValueError("GLM-DSA GGUF has invalid expert or dense-prefix geometry")

    from mobius.integrations.gguf._config_mapping import _glm_dsa_indexer_types

    indexer_types = _glm_dsa_indexer_types(
        ArchitectureConfig(
            num_hidden_layers=layers,
            max_position_embeddings=int(metadata[f"{arch}.context_length"]),
        ),
        metadata,
        arch=arch,
    )
    required: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (vocab, hidden),
        "output_norm.weight": (hidden,),
    }
    optional: dict[str, tuple[int, ...]] = {
        "output.weight": (vocab, hidden),
    }
    actual = set(model.tensor_names)
    float_auxiliaries = {"output_norm.weight"}
    for layer in range(layers):
        prefix = f"blk.{layer}."
        required.update(
            {
                prefix + "attn_norm.weight": (hidden,),
                prefix + "attn_q_a_norm.weight": (q_lora,),
                prefix + "attn_kv_a_norm.weight": (kv_lora,),
                prefix + "attn_q_a.weight": (q_lora, hidden),
                prefix + "attn_q_b.weight": (heads * qk_dim, q_lora),
                prefix + "attn_kv_a_mqa.weight": (
                    kv_lora + rope_dim,
                    hidden,
                ),
                prefix + "attn_k_b.weight": (heads, kv_lora, nope_dim),
                prefix + "attn_v_b.weight": (heads, value_dim, kv_lora),
                prefix + "attn_output.weight": (hidden, heads * value_dim),
                prefix + "ffn_norm.weight": (hidden,),
            }
        )
        float_auxiliaries.update(
            {
                prefix + "attn_norm.weight",
                prefix + "attn_q_a_norm.weight",
                prefix + "attn_kv_a_norm.weight",
                prefix + "ffn_norm.weight",
            }
        )
        if indexer_types[layer] == "full":
            required.update(
                {
                    prefix + "indexer.k_norm.weight": (index_dim,),
                    prefix + "indexer.k_norm.bias": (index_dim,),
                    prefix + "indexer.proj.weight": (index_heads, hidden),
                    prefix + "indexer.attn_k.weight": (index_dim, hidden),
                    prefix + "indexer.attn_q_b.weight": (
                        index_heads * index_dim,
                        q_lora,
                    ),
                }
            )
            float_auxiliaries.update(
                {
                    prefix + "indexer.k_norm.weight",
                    prefix + "indexer.k_norm.bias",
                }
            )
        if layer < dense_prefix:
            required.update(
                {
                    prefix + "ffn_gate.weight": (dense_width, hidden),
                    prefix + "ffn_up.weight": (dense_width, hidden),
                    prefix + "ffn_down.weight": (hidden, dense_width),
                }
            )
            continue
        required.update(
            {
                prefix + "ffn_gate_inp.weight": (experts, hidden),
                prefix + "ffn_gate_exps.weight": (
                    experts,
                    expert_width,
                    hidden,
                ),
                prefix + "ffn_up_exps.weight": (
                    experts,
                    expert_width,
                    hidden,
                ),
                prefix + "ffn_down_exps.weight": (
                    experts,
                    hidden,
                    expert_width,
                ),
                prefix + "ffn_gate_shexp.weight": (
                    expert_width * shared,
                    hidden,
                ),
                prefix + "ffn_up_shexp.weight": (
                    expert_width * shared,
                    hidden,
                ),
                prefix + "ffn_down_shexp.weight": (
                    hidden,
                    expert_width * shared,
                ),
            }
        )
        optional[prefix + "exp_probs_b.bias"] = (experts,)
        float_auxiliaries.add(prefix + "ffn_gate_inp.weight")
        if prefix + "exp_probs_b.bias" in actual:
            float_auxiliaries.add(prefix + "exp_probs_b.bias")

    shapes = _tensor_shapes(model)
    _validate_exact_closure(arch, shapes, required, optional, layers=layers)
    _require_float_auxiliaries(model, float_auxiliaries, architecture=arch)


def validate_remaining_dense_tensor_contract(model) -> None:
    """Validate the exact executable tensor subset for the three promoted routes."""
    spec = try_get_arch_spec(model.architecture)
    architecture = spec.gguf_arch if spec is not None else model.architecture
    if architecture == "minimax-m2":
        _validate_minimax_m2(model)
    elif architecture == "mistral4":
        _validate_mistral4(model)
    elif architecture == "glm-dsa":
        _validate_glm_dsa(model)
