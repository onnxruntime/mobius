# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Exact Qwen3.8 Flash-Next GGUF identity, header, and resource-policy gates."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

__all__ = [
    "QWEN4EXP_GGUF_REPO",
    "QWEN4EXP_GGUF_REVISION",
    "QWEN4EXP_GGUF_SHARDS",
    "Qwen4ExpGGUFImportError",
    "validate_qwen4exp_hub_artifact",
    "validate_qwen4exp_hub_source",
    "validate_qwen4exp_tensor_contract",
]

QWEN4EXP_GGUF_REPO = "unsloth/Qwen3.8-Flash-Next-GGUF"
QWEN4EXP_GGUF_REVISION = "d3bc75ee6ccef3efc1e228ec00a6cc2cdb1e2249"


@dataclass(frozen=True, slots=True)
class _PinnedShard:
    filename: str
    size: int
    lfs_sha256: str
    tensor_count: int


QWEN4EXP_GGUF_SHARDS = (
    _PinnedShard(
        "UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf",
        10_946_624,
        "88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd",
        0,
    ),
    _PinnedShard(
        "UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00002-of-00003.gguf",
        49_990_818_368,
        "3a62e35bbf9add4733bd1438ebd3a67649d5edd6cb0e72bb78e33c913992b2b6",
        595,
    ),
    _PinnedShard(
        "UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00003-of-00003.gguf",
        22_544_696_352,
        "0e25ceaeb89b8a80aa973c6c0c7448943682f7408c2855b2ebd016b7643a861a",
        629,
    ),
)
_EXPECTED_TENSOR_COUNT = 1224
_PINNED_TENSOR_MANIFEST_SHA256 = (
    "25a1e6a2073caf19d3a3835dd23702a19fa09cc651506e11a13de7b48076359d"
)
_MAX_DEQUANTIZED_SINGLE_TENSOR_BYTES = 8 << 30
_IQ2_EXPERT_LAYERS = frozenset({1, 2, 4, 14, 16, 25, 30, 32, 37, 39, 42, 45, 46, 47})


class Qwen4ExpGGUFImportError(NotImplementedError):
    """The pinned Qwen4Exp payload has no truthful executable import route."""


def _lfs_sha256(info: Any) -> str | None:
    lfs = getattr(info, "lfs", None)
    if isinstance(lfs, dict):
        value = lfs.get("sha256") or lfs.get("oid")
    else:
        value = getattr(lfs, "sha256", None) or getattr(lfs, "oid", None)
    if isinstance(value, str) and value.startswith("sha256:"):
        value = value.removeprefix("sha256:")
    return value if isinstance(value, str) else None


def _payload_blocker(*, keep_quantized: bool) -> Qwen4ExpGGUFImportError:
    if keep_quantized:
        detail = (
            "per_layer_token_embd.weight is an IQ4_NL embedding with no matching "
            "GatherBlockQuantized ABI, and the rank-3 routed experts mix IQ1_S "
            "gate/up banks with an IQ4_NL down bank. MatMulNBits is an affine "
            "rank-2 projection ABI, while the released BlockQuantizedMoE path has "
            "no mixed-format expert-bank ABI or real-weight runtime evidence."
        )
    else:
        detail = (
            "per_layer_token_embd.weight expands to roughly 191 GiB as float32 "
            "(about 95 GiB at its source BF16 logical width) by itself, "
            f"exceeding the {_MAX_DEQUANTIZED_SINGLE_TENSOR_BYTES >> 30} GiB "
            "single-tensor materialization limit before the expert banks are "
            "included. The current importer materializes a complete torch tensor, "
            "so dequantization is not a bounded-memory route."
        )
    return Qwen4ExpGGUFImportError(
        "Qwen3.8 Flash-Next GGUF payload import is intentionally fail-closed. "
        f"{detail} No GGUF tensor payload was downloaded or materialized. The exact header, "
        "configuration, shard closure, and tensor-name mapping remain supported "
        "for preflight and future ABI work."
    )


def validate_qwen4exp_hub_artifact(
    api: Any,
    *,
    repo_id: str,
    revision: str,
    shard_filenames: list[str],
    keep_quantized: bool,
) -> None:
    """Verify the only published artifact identity, then reject before download."""
    validate_qwen4exp_hub_source(repo_id=repo_id, revision=revision)
    expected_names = [shard.filename for shard in QWEN4EXP_GGUF_SHARDS]
    if shard_filenames != expected_names:
        raise Qwen4ExpGGUFImportError(
            "Qwen4Exp GGUF shard set does not match the pinned three-file route: "
            f"expected {expected_names}, got {shard_filenames}. No payload was downloaded."
        )

    infos = api.get_paths_info(
        repo_id,
        shard_filenames,
        revision=revision,
        expand=True,
    )
    by_path = {getattr(info, "path", None): info for info in infos}
    for shard in QWEN4EXP_GGUF_SHARDS:
        info = by_path.get(shard.filename)
        actual_size = int(getattr(info, "size", 0) or 0)
        actual_sha256 = _lfs_sha256(info)
        if actual_size != shard.size or actual_sha256 != shard.lfs_sha256:
            raise Qwen4ExpGGUFImportError(
                f"Pinned Qwen4Exp shard identity mismatch for {shard.filename!r}: "
                f"expected size/SHA-256 {shard.size}/{shard.lfs_sha256}, got "
                f"{actual_size}/{actual_sha256}. No payload was downloaded."
            )
    raise _payload_blocker(keep_quantized=keep_quantized)


def validate_qwen4exp_hub_source(*, repo_id: str, revision: str) -> None:
    """Reject unpinned Qwen4Exp repositories and revisions before payload access."""
    if repo_id != QWEN4EXP_GGUF_REPO:
        raise Qwen4ExpGGUFImportError(
            f"Qwen4Exp Hub GGUF source {repo_id!r} is not the pinned repository "
            f"{QWEN4EXP_GGUF_REPO!r}; refusing an unverified payload before download."
        )
    if revision != QWEN4EXP_GGUF_REVISION:
        raise Qwen4ExpGGUFImportError(
            f"Qwen4Exp Hub GGUF revision {revision!r} is not the pinned revision "
            f"{QWEN4EXP_GGUF_REVISION}; refusing mutable or unverified payloads."
        )


def _qtype_name(qtype: Any) -> str:
    return str(getattr(qtype, "name", qtype)).upper()


def _tensor_manifest_sha256(
    entries: list[tuple[str, tuple[int, ...], str]],
) -> str:
    """Hash all header-owned names, logical shapes, and GGML qtypes."""
    lines = [
        f"{name}|{','.join(str(dim) for dim in shape)}|{qtype}"
        for name, shape, qtype in sorted(entries)
    ]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _expected_shapes(metadata: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    prefix = "qwen4exp."
    hidden = int(metadata[f"{prefix}embedding_length"])
    vocab = len(metadata.get("tokenizer.ggml.tokens", ())) or int(
        metadata.get(f"{prefix}vocab_size", 0)
    )
    layers = int(metadata[f"{prefix}block_count"])
    hc_count = int(metadata[f"{prefix}hyper_connection.count"])
    hc_lowrank = int(metadata[f"{prefix}hyper_connection.low_rank"])
    experts = int(metadata[f"{prefix}expert_count"])
    expert_width = int(metadata[f"{prefix}expert_feed_forward_length"])
    shared_width = int(metadata[f"{prefix}expert_shared_feed_forward_length"])
    q_heads = int(metadata[f"{prefix}attention.head_count"])
    kv_heads = int(metadata[f"{prefix}attention.head_count_kv"])
    head_dim = int(metadata[f"{prefix}attention.key_length"])
    key_heads = int(metadata[f"{prefix}ssm.group_count"])
    value_heads = int(metadata[f"{prefix}ssm.time_step_rank"])
    state_size = int(metadata[f"{prefix}ssm.state_size"])
    inner_size = int(metadata[f"{prefix}ssm.inner_size"])
    conv_kernel = int(metadata[f"{prefix}ssm.conv_kernel"])
    index_heads = int(metadata[f"{prefix}attention.indexer.head_count"])
    index_dim = int(metadata[f"{prefix}attention.indexer.key_length"])
    index_kv_heads = 1
    hc_hidden = hc_count * hidden
    linear_key = key_heads * state_size
    linear_value = inner_size
    ratios = [int(value) for value in metadata[f"{prefix}attention.compress_ratios"]]
    ple_layers = {int(value) for value in metadata[f"{prefix}ple.layers"]}
    ngram_heads = (int(metadata[f"{prefix}ple.ngram_size"]) - 1) * int(
        metadata[f"{prefix}ple.heads_per_ngram"]
    )
    ple_width = int(metadata[f"{prefix}embedding_length_per_layer_input"])
    head_vocab_sizes = [int(value) for value in metadata[f"{prefix}ple.head_vocab_sizes"]]
    ple_rows = math.ceil(sum(head_vocab_sizes) / 128) * 128

    shapes: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (vocab, hidden),
        "output.weight": (vocab, hidden),
        "per_layer_token_embd.weight": (ple_rows, ple_width),
        "output_hc_down.weight": (hc_lowrank, hc_hidden),
        "output_hc_up.weight": (hc_hidden, hc_lowrank),
        "output_hc_norm.weight": (hc_hidden,),
    }
    for layer in range(layers):
        root = f"blk.{layer}"
        shapes.update(
            {
                f"{root}.ffn_gate_exps.weight": (experts, expert_width, hidden),
                f"{root}.ffn_up_exps.weight": (experts, expert_width, hidden),
                f"{root}.ffn_down_exps.weight": (experts, hidden, expert_width),
                f"{root}.ffn_gate_inp.weight": (experts, hidden),
                f"{root}.ffn_gate_shexp.weight": (shared_width, hidden),
                f"{root}.ffn_up_shexp.weight": (shared_width, hidden),
                f"{root}.ffn_down_shexp.weight": (hidden, shared_width),
                f"{root}.ffn_gate_inp_shexp.weight": (hidden,),
            }
        )
        for block in ("attn", "ffn"):
            shapes.update(
                {
                    f"{root}.hc_{block}_down.weight": (hc_lowrank, hc_hidden),
                    f"{root}.hc_{block}_up.weight": (hc_hidden, hc_lowrank),
                    f"{root}.hc_{block}_norm.weight": (hc_hidden,),
                    f"{root}.hc_{block}_inject.weight": (hc_count, hc_hidden),
                }
            )
        if ratios[layer] == 0:
            shapes.update(
                {
                    f"{root}.attn_qkv.weight": (
                        2 * linear_key + linear_value,
                        hidden,
                    ),
                    f"{root}.attn_gate.weight": (linear_value, hidden),
                    f"{root}.ssm_alpha.weight": (value_heads, hidden),
                    f"{root}.ssm_beta.weight": (value_heads, hidden),
                    f"{root}.ssm_a": (value_heads,),
                    f"{root}.ssm_dt.bias": (value_heads,),
                    f"{root}.ssm_conv1d.weight": (
                        2 * linear_key + linear_value,
                        conv_kernel,
                    ),
                    f"{root}.ssm_norm.weight": (state_size,),
                    f"{root}.ssm_out.weight": (hidden, inner_size),
                }
            )
        else:
            q_width = q_heads * head_dim
            shapes.update(
                {
                    f"{root}.attn_q.weight": (2 * q_width, hidden),
                    f"{root}.attn_k.weight": (kv_heads * head_dim, hidden),
                    f"{root}.attn_v.weight": (kv_heads * head_dim, hidden),
                    f"{root}.attn_output.weight": (hidden, q_width),
                    f"{root}.attn_q_norm.weight": (head_dim,),
                    f"{root}.attn_k_norm.weight": (head_dim,),
                    f"{root}.indexer.q_proj.weight": (index_heads * index_dim, hidden),
                    f"{root}.indexer.k_proj.weight": (
                        index_kv_heads * index_dim,
                        hidden,
                    ),
                    f"{root}.indexer.q_norm.weight": (index_dim,),
                    f"{root}.indexer.k_norm.weight": (index_dim,),
                }
            )
        if layer in ple_layers:
            ple_dim = ple_width * ngram_heads
            shapes.update(
                {
                    f"{root}.ple_conv1d.weight": (hc_hidden, conv_kernel),
                    f"{root}.ple_key.weight": (hc_hidden, ple_dim),
                    f"{root}.ple_value.weight": (hidden, ple_dim),
                    f"{root}.ple_norm_key.weight": (hc_hidden,),
                    f"{root}.ple_norm_query.weight": (hc_hidden,),
                    f"{root}.ple_norm_conv.weight": (hc_hidden,),
                }
            )
    return shapes


def _expected_qtypes(metadata: dict[str, Any]) -> dict[str, str]:
    """Return the complete dynamic-quantization assignment from pinned headers."""
    layers = int(metadata["qwen4exp.block_count"])
    ratios = [int(value) for value in metadata["qwen4exp.attention.compress_ratios"]]
    ple_layers = {int(value) for value in metadata["qwen4exp.ple.layers"]}
    qtypes = {
        "output.weight": "Q4_K",
        "token_embd.weight": "Q4_K",
        "per_layer_token_embd.weight": "IQ4_NL",
        "output_hc_down.weight": "Q8_0",
        "output_hc_up.weight": "Q8_0",
        "output_hc_norm.weight": "F32",
    }
    for layer in range(layers):
        root = f"blk.{layer}"
        expert_qtype = "IQ2_XXS" if layer in _IQ2_EXPERT_LAYERS else "IQ1_S"
        shared_input_qtype = "Q6_K" if layer == 2 else "Q5_K"
        qtypes.update(
            {
                f"{root}.ffn_gate_exps.weight": expert_qtype,
                f"{root}.ffn_up_exps.weight": expert_qtype,
                f"{root}.ffn_down_exps.weight": "IQ4_NL",
                f"{root}.ffn_gate_inp.weight": "F32",
                f"{root}.ffn_gate_shexp.weight": shared_input_qtype,
                f"{root}.ffn_up_shexp.weight": shared_input_qtype,
                f"{root}.ffn_down_shexp.weight": "Q8_0",
                f"{root}.ffn_gate_inp_shexp.weight": "F32",
            }
        )
        for block in ("attn", "ffn"):
            qtypes.update(
                {
                    f"{root}.hc_{block}_down.weight": "Q8_0",
                    f"{root}.hc_{block}_up.weight": "Q8_0",
                    f"{root}.hc_{block}_norm.weight": "F32",
                    f"{root}.hc_{block}_inject.weight": "F32",
                }
            )
        if ratios[layer] == 0:
            projection_qtype = "Q6_K" if layer == 2 else "Q5_K"
            qtypes.update(
                {
                    f"{root}.attn_qkv.weight": projection_qtype,
                    f"{root}.attn_gate.weight": projection_qtype,
                    f"{root}.ssm_a": "F32",
                    f"{root}.ssm_alpha.weight": "F32",
                    f"{root}.ssm_beta.weight": "F32",
                    f"{root}.ssm_conv1d.weight": "F32",
                    f"{root}.ssm_dt.bias": "F32",
                    f"{root}.ssm_norm.weight": "F32",
                    f"{root}.ssm_out.weight": "Q6_K",
                }
            )
        else:
            qtypes.update(
                {
                    f"{root}.attn_q.weight": "Q5_K",
                    f"{root}.attn_k.weight": "Q5_K",
                    f"{root}.attn_v.weight": "Q5_K",
                    f"{root}.attn_output.weight": "Q5_K",
                    f"{root}.attn_q_norm.weight": "F32",
                    f"{root}.attn_k_norm.weight": "F32",
                    f"{root}.indexer.q_proj.weight": "BF16",
                    f"{root}.indexer.k_proj.weight": "BF16",
                    f"{root}.indexer.q_norm.weight": "F32",
                    f"{root}.indexer.k_norm.weight": "F32",
                }
            )
        if layer in ple_layers:
            qtypes.update(
                {
                    f"{root}.ple_conv1d.weight": "F32",
                    f"{root}.ple_key.weight": "Q8_0",
                    f"{root}.ple_value.weight": "Q8_0",
                    f"{root}.ple_norm_key.weight": "F32",
                    f"{root}.ple_norm_query.weight": "F32",
                    f"{root}.ple_norm_conv.weight": "F32",
                }
            )
    return qtypes


def validate_qwen4exp_tensor_contract(
    model: Any,
    *,
    source: str,
    keep_quantized: bool | None = None,
) -> None:
    """Validate the exact 1,224-tensor pinned header without touching payloads."""
    if model.architecture != "qwen4exp":
        return
    metadata = model.metadata
    expected_scalars = {
        "general.architecture": "qwen4exp",
        "general.type": "model",
        "general.name": "Qwen3.8 Flash Next",
        "general.description": "A Preview of the Qwen4 Architecture",
        "general.size_label": "512x56B",
        "qwen4exp.block_count": 48,
        "qwen4exp.context_length": 262144,
        "qwen4exp.embedding_length": 2560,
        "qwen4exp.attention.head_count": 24,
        "qwen4exp.attention.head_count_kv": 2,
        "qwen4exp.attention.key_length": 256,
        "qwen4exp.attention.value_length": 256,
        "qwen4exp.attention.layer_norm_rms_epsilon": 1e-6,
        "qwen4exp.rope.dimension_count": 64,
        "qwen4exp.rope.freq_base": 10_000_000.0,
        "qwen4exp.full_attention_interval": 4,
        "qwen4exp.expert_count": 512,
        "qwen4exp.expert_used_count": 10,
        "qwen4exp.expert_feed_forward_length": 640,
        "qwen4exp.expert_shared_feed_forward_length": 640,
        "qwen4exp.hyper_connection.count": 4,
        "qwen4exp.hyper_connection.low_rank": 320,
        "qwen4exp.attention.indexer.head_count": 4,
        "qwen4exp.attention.indexer.key_length": 128,
        "qwen4exp.attention.indexer.top_k": 2048,
        "qwen4exp.ssm.conv_kernel": 4,
        "qwen4exp.ssm.group_count": 16,
        "qwen4exp.ssm.inner_size": 6144,
        "qwen4exp.ssm.state_size": 128,
        "qwen4exp.ssm.time_step_rank": 48,
        "qwen4exp.ple.ngram_size": 3,
        "qwen4exp.ple.heads_per_ngram": 8,
        "qwen4exp.ple.conv_kernel": 4,
        "qwen4exp.ple.eos_token_id": 248044,
        "qwen4exp.embedding_length_per_layer_input": 160,
        "tokenizer.ggml.eos_token_id": 248044,
    }
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in expected_scalars.items()
        if metadata.get(key) != expected
    }
    expected_arrays = {
        "qwen4exp.attention.compress_ratios": [0, 0, 0, 4] * 12,
        "qwen4exp.ple.layers": [1],
        "qwen4exp.rope.dimension_sections": [11, 11, 10],
    }
    for key, expected in expected_arrays.items():
        if list(metadata.get(key, ())) != expected:
            mismatches[key] = (metadata.get(key), expected)
    if metadata.get("tokenizer.ggml.pre") != "qwen35":
        mismatches["tokenizer.ggml.pre"] = (metadata.get("tokenizer.ggml.pre"), "qwen35")
    from mobius.models.qwen4_exp import (
        _build_layer_multipliers,
        _find_nth_prime_after,
    )

    ngram_size = int(metadata["qwen4exp.ple.ngram_size"])
    heads_per_ngram = int(metadata["qwen4exp.ple.heads_per_ngram"])
    ngram_heads = (ngram_size - 1) * heads_per_ngram
    expected_vocab_sizes = [
        _find_nth_prime_after(20_000_000 - 1, index + 1) for index in range(ngram_heads)
    ]
    expected_offsets: list[int] = []
    offset = 0
    for size in expected_vocab_sizes:
        expected_offsets.append(offset)
        offset += size
    vocab_size = len(metadata.get("tokenizer.ggml.tokens", ())) or int(
        metadata.get("qwen4exp.vocab_size", 0)
    )
    expected_multipliers = _build_layer_multipliers(
        vocab_size,
        ngram_size,
        0,
        1234,
    ).tolist()
    concrete_hash_arrays = {
        "qwen4exp.ple.head_vocab_sizes": expected_vocab_sizes,
        "qwen4exp.ple.head_offsets": expected_offsets,
        "qwen4exp.ple.layer_multipliers": expected_multipliers,
    }
    for key, expected in concrete_hash_arrays.items():
        if list(metadata.get(key, ())) != expected:
            mismatches[key] = (metadata.get(key), expected)
    if "qwen4exp.feed_forward_length" in metadata:
        mismatches["qwen4exp.feed_forward_length"] = (
            metadata["qwen4exp.feed_forward_length"],
            "absent in the pinned MoE-only header",
        )
    if mismatches:
        raise ValueError(
            f"Qwen4Exp GGUF metadata does not match the pinned source contract: {mismatches}"
        )

    expected_shapes = _expected_shapes(metadata)
    actual_names = set(model.tensor_names)
    expected_names = set(expected_shapes)
    if actual_names != expected_names:
        raise ValueError(
            f"Qwen4Exp GGUF tensor closure mismatch for {source!r}: missing "
            f"{sorted(expected_names - actual_names)}, unexpected "
            f"{sorted(actual_names - expected_names)}"
        )
    if len(actual_names) != _EXPECTED_TENSOR_COUNT:
        raise ValueError(
            f"Qwen4Exp GGUF must contain exactly {_EXPECTED_TENSOR_COUNT} tensors, "
            f"got {len(actual_names)}"
        )

    qtypes: dict[str, str] = {}
    manifest_entries: list[tuple[str, tuple[int, ...], str]] = []
    shape_errors: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for name, _raw, qtype, shape in model.tensor_items_raw():
        actual_shape = tuple(int(dim) for dim in shape)
        expected_shape = expected_shapes[name]
        if actual_shape != expected_shape:
            shape_errors[name] = (actual_shape, expected_shape)
        qtypes[name] = _qtype_name(qtype)
        manifest_entries.append((name, actual_shape, qtypes[name]))
    if shape_errors:
        raise ValueError(f"Qwen4Exp GGUF tensor shape mismatch: {shape_errors}")
    manifest_sha256 = _tensor_manifest_sha256(manifest_entries)
    if manifest_sha256 != _PINNED_TENSOR_MANIFEST_SHA256:
        raise ValueError(
            "Qwen4Exp GGUF complete tensor manifest mismatch: expected SHA-256 "
            f"{_PINNED_TENSOR_MANIFEST_SHA256}, got {manifest_sha256}. The digest "
            "covers all 1,224 names, logical shapes, and GGML qtypes."
        )

    expected_qtypes = _expected_qtypes(metadata)
    qtype_errors = {
        name: (qtypes.get(name), expected)
        for name, expected in expected_qtypes.items()
        if qtypes.get(name) != expected
    }
    if qtype_errors:
        raise ValueError(f"Qwen4Exp GGUF pinned qtype mismatch: {qtype_errors}")

    manifest = getattr(model, "manifest", None)
    if manifest is None:
        raise ValueError("Qwen4Exp GGUF requires the pinned complete three-shard set")
    shard_counts = [int(shard.tensor_count) for shard in manifest.shards]
    if manifest.split_count != 3 or shard_counts != [0, 595, 629]:
        raise ValueError(
            "Qwen4Exp GGUF shard closure must be metadata-only shard 0 plus "
            f"0+595+629=1224 tensors, got split_count={manifest.split_count}, "
            f"counts={shard_counts}"
        )
    if keep_quantized is not None:
        raise _payload_blocker(keep_quantized=keep_quantized)
