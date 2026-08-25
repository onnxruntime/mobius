# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""GGUF → ONNX build pipeline.

Converts ``.gguf`` model files to ONNX using the standard build
pipeline. Quantized preservation is the default: affine linear-layer weights
are repacked into MatMulNBits format and compatible token embeddings into
GatherBlockQuantized format. For text-only builds, operator-native
IQ/MXFP4 projection blocks are preserved for BlockQuantizedMatMul. Multimodal
text backbones and mixed presets such as Q4_K_M are normalized to one affine
layout, so not every source tensor is byte-preserved. Other tensors are
dequantized. Set ``keep_quantized=False`` to dequantize all weights to float
explicitly.
"""

from __future__ import annotations

__all__ = ["build_from_gguf"]

import logging
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import tqdm
from huggingface_hub import HfApi, hf_hub_download

from mobius._model_package import ModelPackage
from mobius.integrations.gguf._arch_registry import (
    MMPROJ_ARCHITECTURE,
    arch_names_with,
    try_get_arch_spec,
)
from mobius.integrations.gguf._errors import (
    DisabledGGUFArchitectureError,
    ShardedGGUFNotSupportedError,
)
from mobius.integrations.gguf._spec import Support

_HUB_PREFLIGHT_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (OSError,)
try:
    from httpx import TransportError as _HttpxTransportError
except ImportError:
    pass
else:
    _HUB_PREFLIGHT_TRANSPORT_ERRORS += (_HttpxTransportError,)

if TYPE_CHECKING:
    from mobius.tasks import ModelTask

logger = logging.getLogger(__name__)

_GGUF_SHARD_FILENAME_RE = re.compile(
    r"-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$",
    re.IGNORECASE,
)
_NEMOTRON_H_MOE_ARCHITECTURE = "nemotron_h_moe"


def _summarize_nemotron_h_moe_layout(
    tensor_names: Iterable[str],
) -> tuple[Counter[str], tuple[int, ...], dict[int, frozenset[str]]]:
    """Summarize base-layer and MTP mixer types from Nemotron-H GGUF names."""
    layer_kinds: dict[int, set[str]] = {}
    mtp_blocks: set[int] = set()
    for name in tensor_names:
        match = re.match(r"^blk\.(\d+)\.(.+)$", name)
        if match is None:
            continue
        block_index = int(match.group(1))
        suffix = match.group(2)
        kinds = layer_kinds.setdefault(block_index, set())
        if suffix.startswith("nextn."):
            mtp_blocks.add(block_index)
        elif suffix.startswith("ssm_"):
            kinds.add("mamba")
        elif suffix.startswith(("ffn_", "exp_probs_")):
            kinds.add("moe")
        elif suffix.startswith(("attn_q.", "attn_k.", "attn_v.", "attn_output.")):
            kinds.add("attention")

    base_counts: Counter[str] = Counter()
    for block_index, kinds in layer_kinds.items():
        if block_index not in mtp_blocks:
            base_counts.update(kinds)
    mtp_kinds = {
        block_index: frozenset(layer_kinds.get(block_index, set()))
        for block_index in sorted(mtp_blocks)
    }
    return base_counts, tuple(sorted(mtp_blocks)), mtp_kinds


def _raise_for_unsupported_gguf_architecture(
    architecture: str,
    *,
    source: str,
    tensor_names: Iterable[str] | None = None,
) -> None:
    """Reject GGUF architectures that do not have semantic conversion evidence.

    Only architectures the registry marks ``REJECTED`` are refused here. An
    architecture that is merely unregistered still reaches the tensor-mapping
    gate, which produces the actionable "not imported yet" message; failing
    early would change which error a caller sees.
    """
    spec = try_get_arch_spec(architecture)
    if spec is None:
        return
    if spec.gguf_arch == MMPROJ_ARCHITECTURE:
        # mmproj sidecars are opened deliberately by the multimodal path, which
        # pairs them with a text backbone. They are rejected only when someone
        # passes one as the model itself, and the tensor-mapping gate already
        # produces that message.
        return
    rejected = [name for name, verdict in spec.verdicts.items() if verdict is Support.REJECTED]
    if not rejected:
        return

    layout = ""
    if architecture == _NEMOTRON_H_MOE_ARCHITECTURE and tensor_names is not None:
        counts, mtp_blocks, mtp_kinds = _summarize_nemotron_h_moe_layout(tensor_names)
        mtp_kind_names = {index: sorted(kinds) for index, kinds in mtp_kinds.items()}
        layout = (
            " Detected base schedule: "
            f"{counts['mamba']} Mamba + {counts['moe']} MoE + "
            f"{counts['attention']} attention layers; auxiliary MTP blocks: "
            f"{list(mtp_blocks)} with mixer types {mtp_kind_names}."
        )

    raise DisabledGGUFArchitectureError(
        f"Direct GGUF conversion for architecture {spec.gguf_arch!r} is intentionally "
        f"disabled for {source!r}.{layout} {spec.reason} No ONNX artifacts were emitted."
    )


def _raise_for_sharded_gguf(
    *,
    source: str,
    filename: str | None = None,
    split_count: int | None = None,
) -> None:
    """Reject one-shard inputs before they can produce an incomplete model."""
    shard_index: int | None = None
    if filename is not None:
        match = _GGUF_SHARD_FILENAME_RE.search(filename)
        if match is not None:
            shard_index = int(match.group("index"))
            split_count = int(match.group("count"))
    if split_count is None or split_count <= 1:
        return

    shard_detail = f" shard {shard_index} of {split_count}" if shard_index else ""
    raise ShardedGGUFNotSupportedError(
        f"Sharded GGUF input is not supported: {source!r} is{shard_detail}. "
        "The GGUF builder reads one file and cannot assemble split tensor tables; "
        "continuing would emit an incomplete ONNX model. Select a single-file GGUF "
        "variant, join the shards with a GGUF-aware tool, or build from the original "
        "Hugging Face checkpoint."
    )


class SparseMoEExportError(NotImplementedError):
    """Routed MoE experts have no sparse fusion for their quantization.

    Raised before export when a checkpoint's routed experts lower to per-expert
    native ``pkg.nxrt::BlockQuantizedMatMul`` nodes (e.g. GLM-5.2 UD-IQ1_{S,M})
    for which no sparse top-k ``BlockQuantizedMoE`` fusion exists — exporting
    anyway would recompute every expert for every token (dense-all-expert
    compute) with no performance guarantee.
    """


def _routed_dense_block_expert_paths(module) -> list[str]:
    """Return module paths of routed experts lowered as native block linears.

    A routed expert is a :class:`BlockQuantizedLinear` whose qualified name
    contains a ``.moe.experts.`` (or bare ``.experts.``) segment but is *not* a
    ``shared_expert``. These are the modules that, absent a sparse top-k MoE
    fusion, force a dense loop over all experts.
    """
    from mobius.components import BlockQuantizedLinear

    paths: list[str] = []
    for name, mod in module.named_modules():
        if not isinstance(mod, BlockQuantizedLinear):
            continue
        if "shared_expert" in name:
            continue
        if ".experts." not in name:
            continue
        paths.append(name)
    return paths


def _assert_sparse_moe_capability(module, config, *, source: str, allow_dense: bool) -> None:
    """Fail closed when routed IQ-block experts have no sparse MoE fusion.

    The int4 ``MatMulNBits`` dense-fallback → ``com.microsoft::QMoE`` rewrite is
    the only sparse MoE path today; native IQ/MXFP4 block experts stay as
    per-expert ``BlockQuantizedMatMul`` nodes. Exporting those yields a graph
    that evaluates every expert for every token. Refuse by default; only proceed
    (with a loud warning) when the caller explicitly opts in.
    """
    num_experts = int(getattr(config, "num_local_experts", 0) or 0)
    if num_experts <= 0:
        return
    dense_paths = _routed_dense_block_expert_paths(module)
    if not dense_paths:
        return

    example = ", ".join(sorted(dense_paths)[:3])
    detail = (
        f"{len(dense_paths)} routed expert projections (e.g. {example}) in "
        f"{source!r} lower to per-expert native BlockQuantizedMatMul nodes."
    )
    if allow_dense:
        logger.warning(
            "allow_dense_moe: exporting %d routed MoE expert(s) as independent "
            "BlockQuantizedMatMul nodes for %s. This recomputes EVERY expert for "
            "every token (dense-all-expert compute); it is NOT a performance path "
            "and makes no throughput claim. %s",
            len(dense_paths),
            source,
            detail,
        )
        return

    raise SparseMoEExportError(
        "Sparse-MoE export blocker: "
        f"{detail} No sparse top-k BlockQuantizedMoE fusion exists for these "
        "block formats — only int4 MatMulNBits experts are fused into "
        "com.microsoft::QMoE. Exporting anyway would build a dense-all-expert "
        "graph (every expert evaluated for every token) with no performance "
        "guarantee, so the build fails closed. To study the dense fallback, "
        "re-run with allow_dense_moe=True (or set "
        "MOBIUS_ALLOW_DENSE_MOE_EXPERTS=1); this makes no throughput claim. "
        "The supported fix is a sparse IQ-block BlockQuantizedMoE fusion "
        "(top-k gather over native-block expert weights)."
    )


def _fuse_native_block_moe(pkg, *, allow_dense: bool) -> int:
    """Collapse routed native-block expert storms into sparse ``BlockQuantizedMoE``.

    Runs on the final, fully-weighted graph so the fusion can stack each layer's
    per-expert native blocks byte-for-byte into one expert-major bank. Every
    candidate layer is validated before any node is emitted, so an unfusable
    layer (per-expert bias, ragged/incomplete group, ...) raises
    :class:`SparseMoEExportError` atomically with the graph untouched, unless
    ``allow_dense`` downgrades it to a warning + dense keep.

    A layer that mixes native formats across its fc1/fc2/fc3 banks (GLM-5.2
    UD-IQ1) can only be expressed with the ``block_layout_version=2``
    per-projection ABI, which no shipped onnx-genai runtime executes yet. The
    production builder therefore never enables v2: such layers always
    typed-reject here rather than emit an unrunnable node. There is no
    environment or CLI opt-in -- v2 stays a schema-construction test path until a
    real typed runtime-capability handshake exists.
    """
    # Imported lazily: the generic rewrite lives in the rewrite_rules package and
    # must not be pulled into the GGUF import graph at module load time.
    from mobius.rewrite_rules import fuse_block_quantized_moe

    fused = 0
    for model in pkg.values():
        # No ``_allow_perproj_v2_schema`` argument: the production authority path
        # always fails closed for mixed-format v2 (fail-safe default).
        fused += fuse_block_quantized_moe(model, allow_dense_moe=allow_dense)
    return fused


def _routed_dense_block_matmul_nodes(model) -> list:
    """Routed per-expert ``BlockQuantizedMatMul`` nodes surviving in a graph.

    A routed expert projection is a ``pkg.nxrt::BlockQuantizedMatMul`` whose
    packed-weight initializer path carries an ``.experts.`` segment and is not a
    ``shared_expert``. After :func:`_fuse_native_block_moe` runs, any that remain
    are an un-collapsed dense-all-expert storm.
    """
    hits = []
    for node in model.graph:
        if node.op_type != "BlockQuantizedMatMul":
            continue
        weight = node.inputs[1] if len(node.inputs) > 1 else None
        name = getattr(weight, "name", None) or ""
        if ".experts." in name and "shared_expert" not in name:
            hits.append(node)
    return hits


def _assert_sparse_moe_graph(pkg, *, source: str, allow_dense: bool) -> None:
    """Sparse-MoE honesty gate over the final (post-fusion) graph state.

    :func:`_fuse_native_block_moe` already fails closed with a precise reason for
    every routed native-block storm it recognises. This backstop catches any
    routed per-expert ``BlockQuantizedMatMul`` storm that survived fusion (e.g. a
    dispatch shape the rewrite did not recognise): shipping it silently would be
    a dense-all-expert graph with no throughput guarantee. Opting into the dense
    fallback (``allow_dense``) is already warned about by the fusion, so this
    gate only enforces the fail-closed default.
    """
    if allow_dense:
        return
    storm = [n for model in pkg.values() for n in _routed_dense_block_matmul_nodes(model)]
    if not storm:
        return
    raise SparseMoEExportError(
        "Sparse-MoE export blocker: "
        f"{len(storm)} routed expert projection(s) in {source!r} remain as "
        "per-expert pkg.nxrt::BlockQuantizedMatMul nodes after native-block MoE "
        "fusion (no sparse top-k BlockQuantizedMoE was applied). Exporting anyway "
        "would build a dense-all-expert graph (every expert evaluated for every "
        "token) with no performance guarantee, so the build fails closed. To "
        "study the dense fallback, re-run with allow_dense_moe=True (or set "
        "MOBIUS_ALLOW_DENSE_MOE_EXPERTS=1); this makes no throughput claim."
    )


def _preflight_hf_gguf(api: HfApi, repo_id: str, filename: str) -> None:
    """Use Hub metadata to reject known-bad inputs before a multi-GB download."""
    source = f"{repo_id}:{filename}"
    _raise_for_sharded_gguf(source=source, filename=filename)
    try:
        info = api.model_info(repo_id, expand=["gguf"])
    except TypeError as error:
        if "expand" not in str(error):
            raise
        logger.info(
            "Skipping Hub GGUF architecture preflight for %s because this "
            "huggingface_hub version has no model_info(expand=...) support; "
            "the downloaded or cached local header will still be validated.",
            source,
        )
        return
    except _HUB_PREFLIGHT_TRANSPORT_ERRORS as error:
        logger.warning(
            "Hub GGUF architecture preflight failed for %s (%s); continuing to "
            "hf_hub_download so an authenticated or cached file can still be used. "
            "The local header will be validated before graph construction.",
            source,
            error,
        )
        return
    gguf_metadata = getattr(info, "gguf", None)
    if isinstance(gguf_metadata, Mapping):
        architecture = gguf_metadata.get("architecture")
    else:
        architecture = getattr(gguf_metadata, "architecture", None)
    if isinstance(architecture, str):
        _raise_for_unsupported_gguf_architecture(
            architecture,
            source=source,
        )


def _validate_gguf_model(gguf_model, *, source: str) -> None:
    """Validate a parsed GGUF before config extraction or graph construction."""
    from mobius.integrations.gguf._shard_set import GgufShardSet

    # A GgufShardSet has already assembled and structurally validated the whole
    # split set, so the single-file "sharded input is unsupported" guard must
    # not fire for it. Plain single-file GGUFModels still reject a lone shard.
    if not isinstance(gguf_model, GgufShardSet):
        split_count = int(gguf_model.get_metadata("split.count", 1))
        _raise_for_sharded_gguf(source=source, split_count=split_count)
    _raise_for_unsupported_gguf_architecture(
        gguf_model.architecture,
        source=source,
        tensor_names=gguf_model.tensor_names,
    )
    _raise_for_unsupported_auxiliary_quantization(gguf_model)
    _raise_for_invalid_t5_tensor_contract(gguf_model)
    _raise_for_malformed_recurrent_tensors(gguf_model)
    _raise_for_unsupported_encoder_heads(gguf_model)
    _raise_for_invalid_encoder_tensor_contract(gguf_model)
    from mobius.integrations.gguf._draft import validate_draft_tensor_contract

    validate_draft_tensor_contract(gguf_model)


def _raise_for_unsupported_encoder_heads(gguf_model) -> None:
    """Reject optional llama.cpp encoder heads that the token-output graph omits."""
    if gguf_model.architecture not in {"bert", "modern-bert"}:
        return
    head_tensors = [
        name
        for name in gguf_model.tensor_names
        if name.startswith(("cls.", "cls_out.", "cls_norm."))
    ]
    if head_tensors:
        raise ValueError(
            f"{gguf_model.architecture} GGUF contains unsupported pooler/classifier "
            f"tensor(s): {head_tensors}. Mobius will not silently discard encoder heads."
        )


def _raise_for_invalid_encoder_tensor_contract(gguf_model) -> None:
    """Validate required encoder tensors and their exact logical ranks/shapes."""
    from mobius.integrations.gguf._quant_registry import float_storage_type_ids

    architecture = gguf_model.architecture
    if architecture not in {"bert", "modern-bert"}:
        return

    metadata = gguf_model.metadata
    hidden = int(metadata[f"{architecture}.embedding_length"])
    intermediate = int(metadata[f"{architecture}.feed_forward_length"])
    layers = int(metadata[f"{architecture}.block_count"])
    num_heads = int(metadata[f"{architecture}.attention.head_count"])
    num_kv_heads = int(metadata.get(f"{architecture}.attention.head_count_kv", num_heads))
    if num_kv_heads != num_heads:
        raise ValueError(
            f"{architecture} GGUF grouped-query attention is not supported: "
            f"attention.head_count={num_heads}, attention.head_count_kv={num_kv_heads}"
        )
    vocab = int(metadata.get(f"{architecture}.vocab_size", 0))
    if not vocab:
        vocab = len(metadata.get("tokenizer.ggml.tokens", ()))
    context = int(metadata[f"{architecture}.context_length"])

    required: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (vocab, hidden),
    }
    if architecture == "bert":
        token_types = int(metadata.get("tokenizer.ggml.token_type_count", 0))
        required.update(
            {
                "token_types.weight": (token_types, hidden),
                "position_embd.weight": (context, hidden),
                "token_embd_norm.weight": (hidden,),
                "token_embd_norm.bias": (hidden,),
            }
        )
        layer_shapes = {
            "attn_output.weight": (hidden, hidden),
            "attn_output.bias": (hidden,),
            "attn_output_norm.weight": (hidden,),
            "attn_output_norm.bias": (hidden,),
            "ffn_up.weight": (intermediate, hidden),
            "ffn_up.bias": (intermediate,),
            "ffn_down.weight": (hidden, intermediate),
            "ffn_down.bias": (hidden,),
            "layer_output_norm.weight": (hidden,),
            "layer_output_norm.bias": (hidden,),
        }
    else:
        required.update(
            {
                "token_embd_norm.weight": (hidden,),
                "output_norm.weight": (hidden,),
            }
        )
        layer_shapes = {
            "attn_qkv.weight": (3 * hidden, hidden),
            "attn_output.weight": (hidden, hidden),
            "ffn_up.weight": (2 * intermediate, hidden),
            "ffn_down.weight": (hidden, intermediate),
            "ffn_norm.weight": (hidden,),
        }

    actual_items = list(gguf_model.tensor_items_raw())
    actual = {
        name: tuple(int(dim) for dim in shape) for name, _raw, _qtype, shape in actual_items
    }
    qtypes = {name: qtype for name, _raw, qtype, _shape in actual_items}

    for layer in range(layers):
        for suffix, shape in layer_shapes.items():
            required[f"blk.{layer}.{suffix}"] = shape
        if architecture == "bert":
            fused_weight = f"blk.{layer}.attn_qkv.weight"
            fused_bias = f"blk.{layer}.attn_qkv.bias"
            split_names = {
                f"blk.{layer}.attn_{projection}.{suffix}"
                for projection in ("q", "k", "v")
                for suffix in ("weight", "bias")
            }
            has_fused = fused_weight in actual or fused_bias in actual
            has_split = bool(split_names & set(actual))
            if has_fused and has_split:
                raise ValueError(
                    f"bert GGUF layer {layer} contains both fused and split Q/K/V tensors; "
                    "the import contract requires one unambiguous representation"
                )
            if has_fused:
                required[fused_weight] = (3 * hidden, hidden)
                required[fused_bias] = (3 * hidden,)
                qtype = qtypes.get(fused_weight)
                qtype_id = getattr(qtype, "value", qtype)
                if qtype is not None and qtype_id not in float_storage_type_ids():
                    raise ValueError(
                        "Quantized fused BERT attn_qkv.weight cannot be split losslessly; "
                        "use a GGUF with split Q/K/V tensors or float fused QKV"
                    )
            else:
                for projection in ("q", "k", "v"):
                    required[f"blk.{layer}.attn_{projection}.weight"] = (hidden, hidden)
                    required[f"blk.{layer}.attn_{projection}.bias"] = (hidden,)
        if architecture == "modern-bert" and layer > 0:
            required[f"blk.{layer}.attn_norm.weight"] = (hidden,)

    missing = sorted(set(required) - set(actual))
    if missing:
        raise ValueError(
            f"{architecture} GGUF is missing required encoder tensor(s): {missing}"
        )
    malformed = {
        name: (required[name], actual[name])
        for name in required
        if actual[name] != required[name]
    }
    if malformed:
        raise ValueError(
            f"{architecture} GGUF has invalid encoder tensor shape(s): {malformed}"
        )
    if architecture == "modern-bert" and "blk.0.attn_norm.weight" in actual:
        raise ValueError(
            "modern-bert blk.0.attn_norm.weight is present, but Mobius models layer 0 "
            "as the pinned identity-norm variant and will not ignore the tensor"
        )


def _raise_for_invalid_t5_tensor_contract(gguf_model) -> None:
    """Validate the pinned T5/T5-encoder tensor closure and logical shapes."""
    from mobius.integrations.gguf._quant_registry import float_storage_type_ids

    architecture = gguf_model.architecture
    if architecture not in {"t5", "t5encoder"}:
        return

    metadata = gguf_model.metadata
    hidden = int(metadata[f"{architecture}.embedding_length"])
    intermediate = int(metadata[f"{architecture}.feed_forward_length"])
    encoder_layers = int(metadata[f"{architecture}.block_count"])
    decoder_layers = (
        int(metadata.get("t5.decoder_block_count", encoder_layers))
        if architecture == "t5"
        else 0
    )
    heads = int(metadata[f"{architecture}.attention.head_count"])
    key_length = int(metadata.get(f"{architecture}.attention.key_length", hidden // heads))
    value_length = int(metadata.get(f"{architecture}.attention.value_length", hidden // heads))
    buckets = int(metadata[f"{architecture}.attention.relative_buckets_count"])
    vocab = int(metadata.get(f"{architecture}.vocab_size", 0))
    if not vocab:
        vocab = len(metadata.get("tokenizer.ggml.tokens", ()))

    actual_items = list(gguf_model.tensor_items_raw())
    actual = {
        name: tuple(int(dim) for dim in shape) for name, _raw, _qtype, shape in actual_items
    }
    qtypes = {name: qtype for name, _raw, qtype, _shape in actual_items}
    layer_limits = {"enc": encoder_layers, "dec": decoder_layers}
    invalid_layer_indices = []
    for name in actual:
        match = re.match(r"^(enc|dec)\.blk\.([^.]+)\.", name)
        if match is None:
            continue
        stack, raw_index = match.groups()
        if not re.fullmatch(r"0|[1-9][0-9]*", raw_index):
            invalid_layer_indices.append(
                f"{name} (layer index {raw_index!r} is not canonical)"
            )
            continue
        layer = int(raw_index)
        if layer >= layer_limits[stack]:
            invalid_layer_indices.append(
                f"{name} (layer index {layer} is outside declared {stack} layer "
                f"count {layer_limits[stack]})"
            )
    if invalid_layer_indices:
        raise ValueError(
            f"{architecture} GGUF has invalid T5 layer tensor index(es): "
            f"{sorted(invalid_layer_indices)}"
        )

    required: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (vocab, hidden),
        "enc.output_norm.weight": (hidden,),
        "enc.blk.0.attn_rel_b.weight": (buckets, heads),
    }
    optional: dict[str, tuple[int, ...]] = {
        "output.weight": (vocab, hidden),
    }
    if architecture == "t5":
        required.update(
            {
                "dec.output_norm.weight": (hidden,),
                "dec.blk.0.attn_rel_b.weight": (buckets, heads),
            }
        )

    def add_stack(prefix: str, layers: int, *, decoder: bool) -> None:
        for layer in range(layers):
            base = f"{prefix}.blk.{layer}"
            required.update(
                {
                    f"{base}.attn_norm.weight": (hidden,),
                    f"{base}.attn_q.weight": (heads * key_length, hidden),
                    f"{base}.attn_k.weight": (heads * key_length, hidden),
                    f"{base}.attn_v.weight": (heads * value_length, hidden),
                    f"{base}.attn_o.weight": (hidden, heads * value_length),
                    f"{base}.ffn_norm.weight": (hidden,),
                    f"{base}.ffn_up.weight": (intermediate, hidden),
                    f"{base}.ffn_down.weight": (hidden, intermediate),
                }
            )
            optional[f"{base}.attn_rel_b.weight"] = (buckets, heads)
            optional[f"{base}.ffn_gate.weight"] = (intermediate, hidden)
            if decoder:
                required.update(
                    {
                        f"{base}.cross_attn_norm.weight": (hidden,),
                        f"{base}.cross_attn_q.weight": (heads * key_length, hidden),
                        f"{base}.cross_attn_k.weight": (heads * key_length, hidden),
                        f"{base}.cross_attn_v.weight": (heads * value_length, hidden),
                        f"{base}.cross_attn_o.weight": (hidden, heads * value_length),
                    }
                )
                # llama.cpp loads this tensor but deliberately does not consume
                # it in the cross-attention graph.
                optional[f"{base}.cross_attn_rel_b.weight"] = (buckets, heads)

    add_stack("enc", encoder_layers, decoder=False)
    add_stack("dec", decoder_layers, decoder=True)

    missing = sorted(set(required) - set(actual))
    if missing:
        raise ValueError(f"{architecture} GGUF is missing required T5 tensor(s): {missing}")
    expected = {**optional, **required}
    malformed = {
        name: (expected[name], actual[name])
        for name in actual
        if name in expected and actual[name] != expected[name]
    }
    if malformed:
        raise ValueError(f"{architecture} GGUF has invalid T5 tensor shape(s): {malformed}")

    float_type_ids = float_storage_type_ids()
    small_non_float = []
    for name, qtype in qtypes.items():
        if name.endswith(("_norm.weight", "attn_rel_b.weight")):
            qtype_id = getattr(qtype, "value", qtype)
            if qtype_id not in float_type_ids:
                small_non_float.append(name)
    if small_non_float:
        raise ValueError(
            f"{architecture} GGUF relative-bias and norm tensors must remain float: "
            f"{sorted(small_non_float)}"
        )
    ignored_names = (
        {"output.weight"}
        if architecture == "t5encoder"
        else {f"dec.blk.{layer}.cross_attn_rel_b.weight" for layer in range(decoder_layers)}
    )
    ignored = sorted(set(actual) & ignored_names)
    if ignored:
        logger.warning(
            "Ignoring pinned llama.cpp T5 tensor(s) that do not participate in "
            "the architecture's output graph: %s",
            ignored,
        )


def _raise_for_malformed_recurrent_tensors(gguf_model) -> None:
    """Reject suffixes not created by the pinned C++ tensor loaders."""
    from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names
    from mobius.integrations.gguf._upstream import upstream_architecture

    upstream = upstream_architecture(gguf_model.architecture)
    if upstream is None or not upstream.tensor_names:
        return

    expected = set(upstream.tensor_names)
    malformed = []
    for name in gguf_model.tensor_names:
        template = re.sub(
            r"^((?:enc\.|dec\.)?blk)\.\d+\.",
            r"\1.{bid}.",
            name,
        )
        mapped_sidecar = name.endswith((".scale", ".input_scale")) and (
            map_gguf_to_hf_names(name, gguf_model.architecture) is not None
        )
        if template not in expected and not mapped_sidecar:
            malformed.append(name)
    if malformed:
        raise ValueError(
            f"Malformed {gguf_model.architecture} GGUF tensor name(s): {malformed}. "
            "The suffixes do not match the pinned llama.cpp tensor creation sites."
        )


def _raise_for_unsupported_auxiliary_quantization(gguf_model) -> None:
    """Reject scale sidecars whose semantics the target quantization ABI cannot express."""
    from mobius.integrations.gguf._mtp import map_gguf_mtp_to_hf_names
    from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names

    architecture = gguf_model.architecture
    expert_count = gguf_model.get_metadata(f"{architecture}.expert_count")
    block_count = int(gguf_model.get_metadata(f"{architecture}.block_count", 0))
    mtp_count = int(gguf_model.get_metadata(f"{architecture}.nextn_predict_layers", 0))
    mtp_block_indices = range(max(0, block_count - mtp_count), block_count)
    for gguf_name, _raw, _qtype, shape in gguf_model.tensor_items_raw():
        suffix = next(
            (
                candidate
                for candidate in (".input_scale", ".scale")
                if gguf_name.endswith(candidate)
            ),
            None,
        )
        if suffix is None:
            continue
        hf_name = map_gguf_to_hf_names(gguf_name, architecture)
        if hf_name is None:
            hf_name = next(
                (
                    mapped
                    for block_index in mtp_block_indices
                    if (mapped := map_gguf_mtp_to_hf_names(gguf_name, block_index)) is not None
                ),
                None,
            )
        if hf_name is None:
            continue

        is_expert_scale = ".mlp.experts." in hf_name
        expected = (int(expert_count),) if is_expert_scale and expert_count else (1,)
        actual = tuple(int(dim) for dim in shape)
        if actual != expected:
            raise ValueError(
                f"Invalid GGUF auxiliary quantization tensor {gguf_name!r}: "
                f"expected shape {expected}, got {actual}"
            )
        raise ValueError(
            f"GGUF auxiliary quantization tensor {gguf_name!r} maps to {hf_name!r}, "
            "but Mobius cannot represent GGUF scale/input_scale sidecars "
            "(including NVFP4 scale2) in its expert/projection quantization ABI. "
            "This file is rejected before graph construction to avoid silently "
            "dropping required quantization data."
        )


def _looks_like_hf_repo_id(value: str) -> bool:
    """Heuristic: ``value`` matches ``owner/repo`` (no path separators, no .gguf suffix)."""
    if value.startswith((".", "/", "~")):
        return False
    parts = value.split("/")
    return len(parts) == 2 and all(p and not p.endswith(".gguf") for p in parts)


def _resolve_gguf_path(gguf_path: str | Path) -> str:
    """Resolve a GGUF reference to a local file path.

    Accepts:
    - An existing local filesystem path (returned unchanged).
    - A HuggingFace Hub reference ``"owner/repo"`` — the repo must contain
      exactly one ``*.gguf`` file, which is downloaded.
    - A HuggingFace Hub reference ``"owner/repo:filename.gguf"`` to pick a
      specific file from a multi-file repo.
    """
    raw = str(gguf_path)
    if Path(raw).exists():
        return raw

    # Split the optional ":filename" suffix before classifying so HF refs like
    # "owner/repo:weights.gguf" are not mistaken for a local path ending in .gguf.
    repo_id, _, filename = raw.partition(":")
    if not _looks_like_hf_repo_id(repo_id):
        # Looks like a local path that doesn't exist; let GGUFModel raise
        # FileNotFoundError with the original path.
        return raw

    api = HfApi()
    if not filename:
        files = [f for f in api.list_repo_files(repo_id) if f.endswith(".gguf")]
        if not files:
            raise FileNotFoundError(f"No *.gguf files found in HF repo {repo_id!r}")
        if len(files) > 1:
            raise ValueError(
                f"HF repo {repo_id!r} contains multiple .gguf files: {files}. "
                f"Specify one via '{repo_id}:<filename.gguf>'."
            )
        filename = files[0]

    _preflight_hf_gguf(api, repo_id, filename)
    logger.info("Downloading %s from %s", filename, repo_id)
    return hf_hub_download(repo_id=repo_id, filename=filename)


def build_from_gguf(
    gguf_path: str | Path,
    *,
    task: str | None = None,
    dtype: str | None = None,
    keep_quantized: bool = True,
    execution_provider: str = "default",
    mmproj: str | Path | None = None,
    static_cache: bool = False,
    max_seq_len: int | None = None,
    allow_dense_moe: bool | None = None,
    reuse_gguf_weights: bool = False,
    target_config: str | Path | Mapping[str, object] | None = None,
) -> ModelPackage:
    """Build an ONNX :class:`ModelPackage` from a GGUF file.

    1. Parse GGUF metadata → :class:`ArchitectureConfig`
    2. Look up the model class and task from the registry
    3. Build the ONNX graph (standard ``build_from_module`` pipeline)
    4. Map GGUF tensor names → HuggingFace names
    5. Replace native-block projection modules when present
    6. Apply architecture-specific tensor processors
    7. Run ``preprocess_weights()`` (HF → ONNX name mapping)
    8. Apply weights to the ONNX model

    By default, supported affine tensors are repacked into MatMulNBits format.
    For text-only builds, operator-native IQ/MXFP4 projection blocks
    are retained byte-for-byte for BlockQuantizedMatMul. Multimodal text
    backbones normalize quantized projections to a common affine layout. GGUFs
    containing only F32, F16, or BF16 weights use the float path because there
    is no quantization to preserve.
    Quantized files with no supported preservation target raise an actionable
    error instead of silently falling back to a float model.

    Args:
        gguf_path: Path to the ``.gguf`` file, *or* a HuggingFace Hub
            reference of the form ``"owner/repo"`` (the repo must
            contain exactly one ``*.gguf`` file) or
            ``"owner/repo:filename.gguf"`` to pick a specific file. HF
            references are downloaded via ``huggingface_hub`` into the
            standard local cache.
        task: Override the model task (e.g. ``"text-generation"``).
            When ``None``, the task is auto-detected from the
            model type.
        dtype: Override model dtype (e.g. ``"f16"``). When ``None``,
            defaults to float32.
        keep_quantized: Preserve quantization when quantized tensors are
            present. This is the default. Supported affine blocks are repacked,
            text-only operator-native IQ/MXFP4 projection blocks
            retain their bytes, and mixed or multimodal source types can be
            normalized to a common affine layout. Set to ``False`` to
            dequantize all weights.
        execution_provider: Target execution provider for EP-aware
            optimisations (e.g. ``"cpu"`` to apply the
            GroupQueryAttention rewrite). Defaults to ``"default"``
            (portable, no vendor fusions).
        mmproj: Optional path (or HF ref) to a companion ``clip``
            multimodal-projector GGUF. When set, this becomes the single
            entry point for a multimodal build: the text GGUF and the
            mmproj vision/audio encoder are fused into one multimodal
            :class:`ModelPackage` (delegates to
            :func:`build_gemma4_vlm_from_gguf`).
        static_cache: When ``True``, build with a pre-allocated static KV
            cache (fixed-width buffers written in place via ``TensorScatter``)
            instead of the default dynamic concat-grow cache. Produces a
            fully static-shaped graph, which is required by fixed-shape
            runtimes such as the QNN HTP backend. Cannot be combined with an
            explicit *task* override.
        max_seq_len: Maximum sequence length for the static cache buffers.
            Only used when ``static_cache=True``. Defaults to the model's
            ``max_position_embeddings``.
        allow_dense_moe: Opt in to exporting routed MoE experts that have no
            sparse top-k fusion (they lower to per-expert native
            ``BlockQuantizedMatMul`` nodes, i.e. dense-all-expert compute with
            no performance guarantee). When ``None`` (default), the value of
            the ``allow_dense_moe_experts`` flag is used, which defaults to
            ``False`` — the build fails closed with a typed capability error
            rather than silently shipping a dense graph. This is a research /
            correctness knob and makes no throughput claim.
        reuse_gguf_weights: Reuse compatible tensor payloads directly from the
            original GGUF via ONNX external-data ranges. The GGUF must be a real
            file in the final flat output directory. Converted tensors are
            written once to ``model.onnx.data``.
        target_config: Exact target model directory, config path, or explicit
            config mapping for a ``dflash``/``eagle3`` speculative draft. A
            mapping must include the complete ``tokenizer_json`` object. Required for draft GGUFs
            and rejected for standalone architectures.

    Returns:
        A :class:`ModelPackage` containing the built model(s).

    Raises:
        ImportError: If the ``gguf`` package is not installed.
        FileNotFoundError: If the GGUF file does not exist.
        KeyError: If the GGUF architecture is not in the registry.
        ValueError: If *static_cache* is combined with an explicit *task*, or
            if a quantized input has no supported preservation target.
    """
    import dataclasses

    # A companion mmproj GGUF turns this into a multimodal build: the text +
    # vision/audio encoders are assembled by the dedicated VLM builder. Keep
    # build_from_gguf as the single public entry point (text-only or multimodal).
    if mmproj is not None:
        if reuse_gguf_weights:
            raise ValueError(
                "reuse_gguf_weights=True does not yet support multimodal/mmproj packages."
            )
        if static_cache:
            raise ValueError("static_cache=True is not supported with a companion mmproj.")
        from mobius.integrations.gguf._mmproj import build_vlm_from_gguf

        return build_vlm_from_gguf(
            gguf_path,
            mmproj,
            dtype=dtype,
            execution_provider=execution_provider,
            keep_quantized=keep_quantized,
        )

    from mobius._builder import (
        build_from_module,
        resolve_dtype,
    )
    from mobius._registry import registry
    from mobius.integrations.gguf._arch_registry import get_arch_spec
    from mobius.integrations.gguf._config_mapping import (
        GGUF_ARCH_TO_MODEL_TYPE,
        gguf_to_config,
    )
    from mobius.integrations.gguf._shard_set import GgufShardSet, open_gguf_model
    from mobius.integrations.gguf._tensor_processors import (
        process_tensors,
    )
    from mobius.integrations.transformers import (
        _default_task_for_model,
    )

    if static_cache and task is not None:
        raise ValueError(
            "static_cache=True cannot be combined with an explicit task "
            "override; the static cache is wired through CausalLMTask."
        )

    from mobius._flags import flags as _mobius_flags

    if allow_dense_moe is None:
        allow_dense_moe = _mobius_flags.allow_dense_moe_experts

    # 1. Parse GGUF file (auto-download from HF Hub when given "owner/repo[:filename]").
    #    A ``-000i-of-000N.gguf`` split set is assembled directly from its shards
    #    (never merged into a second on-disk GGUF); a plain file opens as before.
    gguf_path = _resolve_gguf_path(gguf_path)
    gguf_model = open_gguf_model(gguf_path)
    _validate_gguf_model(gguf_model, source=str(gguf_path))
    if reuse_gguf_weights and isinstance(gguf_model, GgufShardSet):
        raise ValueError(
            "reuse_gguf_weights=True does not yet support multi-shard GGUF sets. "
            "Build without reuse to preserve current multi-shard import behavior."
        )
    if reuse_gguf_weights and not gguf_model.is_little_endian:
        raise ValueError(
            "reuse_gguf_weights=True requires a little-endian GGUF because ONNX "
            "external tensors interpret the referenced bytes as little-endian. "
            "Build without reuse to convert this file."
        )
    gguf_arch = gguf_model.architecture
    logger.info("Loaded GGUF model: %s (arch=%s)", gguf_path, gguf_arch)
    preserve_quantization = keep_quantized and _has_quantized_weights(gguf_model, gguf_arch)
    arch_spec = try_get_arch_spec(gguf_arch)
    if (
        preserve_quantization
        and arch_spec is not None
        and arch_spec.quantized_import is not Support.SUPPORTED
    ):
        raise ValueError(
            f"GGUF architecture {gguf_arch!r} does not support keep_quantized=True: "
            f"{arch_spec.reason}"
        )
    if keep_quantized and not preserve_quantization:
        logger.info("GGUF contains no mapped quantized weights; using the float import path")
    _reject_quantized_diffusion_fused_qkv(
        gguf_model,
        gguf_arch,
        preserve_quantization=preserve_quantization,
    )

    # 2. Extract config from GGUF metadata
    config = gguf_to_config(gguf_model)
    spec = get_arch_spec(gguf_arch)
    from mobius.integrations.gguf._draft import (
        is_draft_architecture,
        validate_draft_pairing,
    )

    if target_config is not None and not is_draft_architecture(gguf_arch):
        raise ValueError(
            f"target_config is only valid for dflash/eagle3 draft GGUFs, got {gguf_arch!r}"
        )
    draft_manifest = (
        validate_draft_pairing(gguf_model, config, target_config)
        if is_draft_architecture(gguf_arch)
        else None
    )
    model_type = getattr(config, "_gguf_model_type", None)
    if model_type is None:
        model_type = GGUF_ARCH_TO_MODEL_TYPE.get(gguf_arch, gguf_arch)
    if gguf_arch in {"dream", "llada", "llada-moe", "rnd1"}:
        from mobius.tasks import MaskedDiffusionTask

        if static_cache:
            raise ValueError(
                f"static_cache=True is not valid for masked-diffusion {gguf_arch} GGUF; "
                "the model is bidirectional and has no KV cache"
            )
        if (
            task is not None
            and task != "masked-diffusion"
            and not isinstance(task, MaskedDiffusionTask)
        ):
            raise ValueError(
                f"{gguf_arch} GGUF only supports task='masked-diffusion', got {task!r}"
            )
    if static_cache and model_type in {"mamba", "mamba2"}:
        raise ValueError(
            f"static_cache=True is not supported for recurrent {model_type} GGUF models; "
            "they carry per-layer conv_state and ssm_state rather than a KV cache."
        )
    if model_type in {"bert", "modernbert", "t5encoder"}:
        if static_cache:
            raise ValueError("static_cache is not valid for encoder-only GGUF architectures")
        expected_task = (
            "t5-text-encoding" if model_type == "t5encoder" else "feature-extraction"
        )
        if task is not None and task != expected_task:
            raise ValueError(
                f"{gguf_arch} GGUF only supports task={expected_task!r}, got {task!r}"
            )
    if model_type == "t5":
        if static_cache:
            raise ValueError(
                "static_cache=True is not valid for T5 seq2seq GGUF; use the "
                "encoder/decoder cache contract"
            )
        if task is not None and task != "seq2seq":
            raise ValueError(f"t5 GGUF only supports task='seq2seq', got {task!r}")
    if is_draft_architecture(gguf_arch):
        if static_cache:
            raise ValueError(f"static_cache=True is not supported for {gguf_arch} drafts")
        expected_task = f"{gguf_arch}-draft"
        if task is not None and task != expected_task:
            raise ValueError(
                f"{gguf_arch} GGUF only supports task={expected_task!r}, got {task!r}"
            )

    # 2b. Architecture-resolution safety rail. When the GGUF architecture string
    # bridges to a specialised registry key, verify the metadata-derived config
    # actually describes that architecture before selecting its model class, so
    # a mislabelled or incomplete GGUF fails closed with precise reasons instead
    # of silently building the wrong graph.
    if model_type == "glm_moe_dsa":
        from mobius.integrations.gguf._config_mapping import assert_glm_moe_dsa_resolvable

        assert_glm_moe_dsa_resolvable(config, gguf_arch, source=str(gguf_path))

    # ``dataclasses.replace`` below (dtype / quantization) returns a fresh
    # instance that does NOT carry the private ``_gguf_*`` attributes set by
    # gguf_to_config (they are plain attributes, not dataclass fields). Capture
    # the MTP head metadata here — like ``model_type`` above — so it can be
    # re-attached onto the final config for auto-detection (step 4b).
    mtp_predict_layers = getattr(config, "_gguf_nextn_predict_layers", 0)
    mtp_block_indices = list(getattr(config, "_gguf_mtp_block_indices", []) or [])

    if dtype is not None:
        resolved = resolve_dtype(dtype)
        if resolved is not None:
            config = dataclasses.replace(config, dtype=resolved)

    # 3. Quantized path: detect dominant type and set config
    if preserve_quantization:
        from mobius._configs import QuantizationConfig
        from mobius._flags import flags
        from mobius.integrations.gguf._tencent_q1_0 import is_tencent_q1_0_layout

        bits, block_size, is_sym = _detect_quant_params(gguf_model, gguf_arch)
        # Float zero-point only when actually using Tencent's native 2-bit form.
        float_zp = is_tencent_q1_0_layout(gguf_model) and flags.tencent_q1_0_use_native_2bit
        quantize_embeddings = _can_quantize_embedding(
            gguf_model,
            gguf_arch,
            bits=bits,
            block_size=block_size,
        )
        quantize_lm_head = (
            quantize_embeddings
            if config.tie_word_embeddings
            else _can_quantize_lm_head(gguf_model, gguf_arch)
        )
        config = dataclasses.replace(
            config,
            quantization=QuantizationConfig(
                bits=bits,
                group_size=block_size,
                quant_method="gguf",
                sym=is_sym,
                float_zero_point=float_zp,
                quantize_embeddings=quantize_embeddings,
                quantize_lm_head=quantize_lm_head,
                tie_word_embeddings=quantize_lm_head and config.tie_word_embeddings,
            ),
        )
        logger.info(
            "Quantized mode: bits=%d, block_size=%d, symmetric=%s, "
            "float_zp=%s, embedding=%s, lm_head=%s",
            bits,
            block_size,
            is_sym,
            float_zp,
            quantize_embeddings,
            quantize_lm_head,
        )

    # 4. Look up module class and resolve task
    module_type = spec.module_type or model_type
    module_class = registry.get(module_type)
    resolved_task: str | ModelTask
    if model_type == "t5":
        from mobius.tasks import Seq2SeqTask

        resolved_task = Seq2SeqTask(
            use_cross_attention_cache=True,
            use_attention_masks=True,
        )
    elif static_cache:
        from mobius.tasks import CausalLMTask

        resolved_task = CausalLMTask(static_cache=True, max_seq_len=max_seq_len)
    elif task is None:
        resolved_task = _default_task_for_model(model_type)
    else:
        resolved_task = task

    # 4b. Auto-detect the Qwen3.5/3.8 MTP / "nextn" self-speculative head: if
    # the source GGUF ships the trailing nextn head block (surfaced by
    # ``has_mtp_head`` from ``<arch>.nextn_predict_layers`` > 0 + the
    # ``blk.<N>.nextn.*`` tensors), always emit the MTP sidecar — it is a purely
    # additive artifact that text-only consumers ignore. No opt-in flag: the
    # decision is driven entirely by presence in the source. When present, expose
    # the backbone's final-layer hidden state as a graph output so the
    # orchestrator can seed the head with it (must be set before the graph is
    # built). Direct field assignment (not dataclasses.replace) preserves the
    # ``_gguf_*`` metadata attributes on the config. Skipped under static_cache
    # (the head needs the dynamic concat-grow cache), leaving those exports
    # byte-identical to today.
    from mobius.integrations.gguf._mtp import build_mtp_head_from_gguf, has_mtp_head

    # Re-attach the GGUF metadata dropped by the dtype/quantization
    # ``dataclasses.replace`` calls above so auto-detection sees it on the final
    # config instance (and ``derive_mtp_config`` can read model_type/quant/dtype).
    # ``_gguf_arch`` matters most: it is the key ``process_tensors`` dispatches
    # on, so losing it here would silently demote every non-float32 and every
    # quantized import to the ``model_type`` fallback.
    config._gguf_arch = gguf_arch
    config._gguf_model_type = model_type
    config._gguf_nextn_predict_layers = mtp_predict_layers
    config._gguf_mtp_block_indices = mtp_block_indices
    emit_mtp_head = has_mtp_head(config) and not static_cache
    if has_mtp_head(config) and static_cache:
        logger.info(
            "GGUF ships an MTP/nextn head but static_cache=True is incompatible "
            "with the head's dynamic cache; skipping the self-speculative sidecar."
        )

    if emit_mtp_head:
        seed_index = int(config.num_hidden_layers) - 1
        existing = list(config.output_layer_indices or [])
        if seed_index not in existing:
            existing.append(seed_index)
        config.output_layer_indices = existing
        logger.info(
            "MTP head detected in source: exposing backbone hidden-state seed "
            "output hidden_states.%d",
            seed_index,
        )

    # 5. Build ONNX graph
    module = module_class(config)
    if preserve_quantization:
        _replace_native_block_linears(module, gguf_model, gguf_arch)
        # The sparse-MoE honesty gate runs post-export on the final graph state
        # (see step 9b): routed native-block experts are first collapsed into a
        # sparse top-k pkg.nxrt::BlockQuantizedMoE by fuse_block_quantized_moe,
        # then the gate fails closed if any per-expert dense storm survives.
        # Enforcing here (pre-export, module level) would reject the very layers
        # the fusion can now collapse, so the authority moved to the graph.
    pkg = build_from_module(
        module, config, resolved_task, execution_provider=execution_provider
    )
    logger.info(
        "Built ONNX graph for %s (%d components)",
        model_type,
        len(pkg),
    )

    # 6. Load tensors from GGUF → state_dict
    reuse_candidates_by_id = {} if reuse_gguf_weights else None
    if preserve_quantization:
        state_dict = _load_quantized_state_dict(
            gguf_model,
            gguf_arch,
            module,
            config,
            reuse_candidates=reuse_candidates_by_id,
        )
    else:
        state_dict = _load_dequantized_state_dict(
            gguf_model,
            gguf_arch,
            reuse_candidates=reuse_candidates_by_id,
        )

    logger.info(
        "Mapped %d state_dict entries from GGUF tensors",
        len(state_dict),
    )

    # 7. Apply architecture-specific tensor processors.
    # For the quantized path, only float tensors go through
    # process_tensors; quantized Q/K tensors were permuted in
    # _load_quantized_state_dict already.
    if preserve_quantization:
        float_keys = {
            k
            for k in state_dict
            if not (
                k.endswith((".scales", ".zero_points"))
                or _is_quantized_weight(k, state_dict)
                or _is_native_block_weight(k, state_dict)
            )
        }
        float_dict = {k: state_dict[k] for k in float_keys}
        quant_dict = {k: state_dict[k] for k in state_dict if k not in float_keys}
        before_processing = dict(float_dict)
        float_dict = process_tensors(float_dict, config)
        if reuse_candidates_by_id is not None:
            _record_reuse_process_transforms(
                before_processing,
                float_dict,
                reuse_candidates_by_id,
                config,
            )
        state_dict = {**float_dict, **quant_dict}
    else:
        before_processing = dict(state_dict)
        state_dict = process_tensors(state_dict, config)
        if reuse_candidates_by_id is not None:
            _record_reuse_process_transforms(
                before_processing,
                state_dict,
                reuse_candidates_by_id,
                config,
            )

    # 7b. Normalize GGUF-specific weight shapes to match HF conventions.
    # This converts GGUF tensor quirks (stacked experts, 1D gates, 2D
    # conv weights, suffix artifacts) into the shapes that HF models
    # produce, so preprocess_weights only needs to handle HF→ONNX.
    state_dict = _normalize_gguf_weights(state_dict, gguf_arch, config)

    # 8. Run model-specific preprocess_weights (HF → ONNX names)
    if hasattr(module, "preprocess_weights"):
        state_dict = module.preprocess_weights(state_dict)
    if is_draft_architecture(gguf_arch):
        state_dict.pop("d2t", None)

    # 9. Apply weights to ONNX model
    prefix_map = getattr(module, "weight_prefix_map", None)
    pkg.apply_weights(
        state_dict,
        prefix_map=prefix_map,
        fold_constants=not reuse_gguf_weights,
    )
    if reuse_gguf_weights:
        from mobius.integrations.gguf._reuse import attach_reused_initializers

        if emit_mtp_head:
            raise ValueError(
                "reuse_gguf_weights=True does not yet support packages with an MTP sidecar."
            )
        final_candidates = {
            name: reuse_candidates_by_id[id(tensor)]
            for name, tensor in state_dict.items()
            if id(tensor) in reuse_candidates_by_id
        }
        attach_reused_initializers(pkg, gguf_path, final_candidates)

    # 9b. Sparse-MoE fusion + honesty gate (final graph state).
    # Now that every native block carries its real packed bytes, collapse the
    # routed per-expert BlockQuantizedMatMul storm into one sparse top-k
    # pkg.nxrt::BlockQuantizedMoE per layer (byte-for-byte, no requantization),
    # then fail closed if any dense-all-expert storm still survives.
    if preserve_quantization:
        _fuse_native_block_moe(pkg, allow_dense=allow_dense_moe)
        _assert_sparse_moe_graph(pkg, source=str(gguf_path), allow_dense=allow_dense_moe)

    # 10. Build the trailing MTP / "nextn" self-speculative head sidecar from
    # the GGUF's ``blk.<nextn>.*`` tensors (dropped by the backbone build) and
    # attach it to the package so the CLI can save it into a ``mtp/`` subdir.
    # Auto-detected from source-tensor presence (see step 4b).
    if emit_mtp_head:
        mtp_pkg = build_mtp_head_from_gguf(
            gguf_model,
            config,
            preserve_quantization=preserve_quantization,
            execution_provider=execution_provider,
        )
        if mtp_pkg is not None:
            pkg.mtp_head = mtp_pkg

    if draft_manifest is not None:
        pkg.draft_manifest = draft_manifest

    return pkg


def _record_reuse_process_transforms(
    before: dict,
    after: dict,
    candidates: dict,
    config,
) -> None:
    """Carry exact-byte candidates through known graph-expressible transforms."""
    import dataclasses

    from mobius.integrations.gguf._tensor_processors import needs_llama_qk_permute

    for name, transformed in after.items():
        original = before.get(name)
        if original is None or id(original) == id(transformed):
            continue
        candidate = candidates.get(id(original))
        if candidate is None:
            continue

        transform: str | None = None
        parameter: int | None = None
        if needs_llama_qk_permute(getattr(config, "model_type", None)) and (
            ".q_proj." in name or ".k_proj." in name
        ):
            transform = "llama_qk_permute"
            parameter = (
                config.num_attention_heads
                if ".q_proj." in name
                else config.num_key_value_heads
            )
        elif "A_log" in name:
            transform = "log_neg"
        elif "norm" in name and original.shape == transformed.shape:
            transform = "subtract_one"
        elif tuple(reversed(original.shape)) == tuple(transformed.shape):
            transform = "transpose"
        elif original.numel() == transformed.numel():
            transform = "reshape"

        if transform is not None:
            candidates[id(transformed)] = dataclasses.replace(
                candidate,
                transform=transform,
                transform_parameter=parameter,
            )


def _is_quantized_weight(key: str, state_dict: dict) -> bool:
    """Check if a .weight key has a matching .scales companion."""
    if not key.endswith(".weight"):
        return False
    stem = key[: -len(".weight")]
    return f"{stem}.scales" in state_dict


def _is_native_block_weight(key: str, state_dict: dict) -> bool:
    """Check for a packed runtime-native GGUF weight."""
    from mobius.integrations.gguf._repacker import NATIVE_BLOCK_BYTE_SIZES

    if not key.endswith(".weight"):
        return False
    value = state_dict[key]
    return (
        value.dtype.is_floating_point is False
        and value.dim() == 3
        and value.shape[-1] in NATIVE_BLOCK_BYTE_SIZES
    )


def _native_block_spec(qtype):
    """Return the runtime-native layout for a GGUF quantization enum."""
    from mobius.integrations.gguf._repacker import native_block_spec

    qtype_val = qtype.value if hasattr(qtype, "value") else qtype
    return native_block_spec(qtype_val)


def _native_block_format(qtype) -> str | None:
    """Return the runtime format string for supported native GGUF blocks."""
    spec = _native_block_spec(qtype)
    return spec.format if spec is not None else None


def _native_block_target_stems(
    hf_name: str,
    np_shape: tuple[int, ...],
    available_stems: set[str],
) -> list[str]:
    """Map a GGUF weight to one or more native-block module stems."""
    if not hf_name.endswith(".weight"):
        return []
    stem = hf_name[: -len(".weight")]
    if stem in available_stems:
        return [stem]

    if len(np_shape) == 3 and ".experts." in stem:
        prefix, projection = stem.rsplit(".experts.", 1)
        for container in (f"{prefix}.experts", f"{prefix}.moe.experts"):
            candidates = [f"{container}.{i}.{projection}" for i in range(np_shape[0])]
            if all(candidate in available_stems for candidate in candidates):
                return candidates
    return []


def _fused_projection_target_stems(hf_name: str, available_stems: set[str]) -> list[str]:
    """Return separate quantized targets for a fused GGUF projection."""
    if hf_name.endswith(".qkv_proj.weight"):
        stem = hf_name[: -len(".qkv_proj.weight")]
        candidates = [f"{stem}.{projection}_proj" for projection in ("q", "k", "v")]
    elif hf_name.endswith(".attn.Wqkv.weight"):
        stem = hf_name[: -len(".Wqkv.weight")]
        candidates = [f"{stem}.{projection}_proj" for projection in ("q", "k", "v")]
    elif hf_name.endswith(".attention.self.qkv.weight"):
        stem = hf_name[: -len(".qkv.weight")]
        candidates = [f"{stem}.{projection}" for projection in ("query", "key", "value")]
    elif hf_name.endswith(".mlp.Wi.weight"):
        stem = hf_name[: -len(".Wi.weight")]
        candidates = [f"{stem}.{projection}_proj" for projection in ("gate", "up")]
    else:
        return []
    return candidates if all(candidate in available_stems for candidate in candidates) else []


def _replace_child_module(root, path: str, replacement) -> None:
    """Replace a named ONNXScript child module while retaining its graph name."""
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        try:
            parent = getattr(parent, part)
        except AttributeError as error:
            raise AttributeError(f"Module path {path!r} has no child {part!r}") from error
    child_name = parts[-1]
    try:
        old = getattr(parent, child_name)
    except AttributeError as error:
        raise AttributeError(f"Module path {path!r} has no child {child_name!r}") from error
    if hasattr(replacement, "_set_name") and hasattr(old, "name"):
        replacement._set_name(old.name)
    setattr(parent, child_name, replacement)


def _replace_native_block_linears(
    module,
    gguf_model,
    gguf_arch: str,
    *,
    name_mapper: Callable[[str, str], str | None] | None = None,
) -> None:
    """Swap MatMulNBits scaffolding for runtime-supported native linears."""
    from mobius.components import BlockQuantizedLinear, QuantizedLinear
    from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names

    if name_mapper is None:
        name_mapper = map_gguf_to_hf_names

    module_map = dict(module.named_modules())
    quantized_stems = {
        name for name, child in module_map.items() if isinstance(child, QuantizedLinear)
    }
    replacements: dict[str, str] = {}
    for gguf_name, _raw, qtype, np_shape in gguf_model.tensor_items_raw():
        format_name = _native_block_format(qtype)
        if format_name is None:
            continue
        hf_name = name_mapper(gguf_name, gguf_arch)
        if hf_name is None:
            continue
        for stem in _native_block_target_stems(hf_name, np_shape, quantized_stems):
            replacements[stem] = format_name

    for stem, format_name in replacements.items():
        old = module_map[stem]
        replacement = BlockQuantizedLinear(
            old._k,
            old._n,
            format=format_name,
            bias=old.bias is not None,
        )
        _replace_child_module(module, stem, replacement)

    if replacements:
        logger.info(
            "Preserving %d GGUF projection weights as runtime-native IQ/MXFP4 blocks",
            len(replacements),
        )


#: GGUF architectures whose transformer RMSNorms are zero-centered
#: (``output = norm(x) * (1 + weight)``, mobius :class:`OffsetRMSNorm`). Their
#: llama.cpp converter bakes the ``+1`` into every ``*norm.weight`` *except* the
#: Gated-DeltaNet internal ``linear_attn.norm`` (a plain gated RMSNorm), so the
#: GGUF path must undo it — see :func:`_normalize_gguf_weights`.
_OFFSET_NORM_GGUF_ARCHS: frozenset[str] = arch_names_with(lambda spec: spec.offset_norm)

#: GGUF architectures whose llama.cpp converter reorders Gated-DeltaNet V-heads
#: from HuggingFace *grouped* order (``head = group * v_per_k + j``) into ggml
#: *tiled* order (``head = j * num_k_heads + group``) whenever the linear layer
#: is grouped (``num_value_heads != num_key_heads``). mobius's ``GatedDeltaNet``
#: forward consumes the HF grouped order, so the GGUF path must undo the tiling —
#: see :func:`_reorder_deltanet_v_heads`.
_V_HEAD_REORDER_GGUF_ARCHS: frozenset[str] = arch_names_with(lambda spec: spec.v_head_reorder)


def _normalize_gguf_weights(
    state_dict: dict,
    gguf_arch: str | None = None,
    config=None,
) -> dict:
    """Normalize GGUF-specific weight shapes to match HF conventions.

    GGUF tensor mapping + dequantization produces weights that differ
    from HuggingFace in several ways. This function converts them so
    that ``preprocess_weights`` only needs to handle HF→ONNX mapping.

    Transforms applied:

    - **Stacked expert weights**: GGUF provides separate 3D tensors
      ``experts.{gate,up,down}_proj.weight`` with shape
      ``[num_experts, out, in]``. These are unpacked into per-expert
      ``experts.{i}.{proj}.weight`` tensors, matching the HF
      ``experts.down_proj`` format that ``preprocess_weights`` expects.
    - **1D shared_expert_gate**: GGUF stores as ``[hidden]``; HF/ONNX
      ``Linear(hidden, 1)`` expects ``[1, hidden]``.
    - **2D conv1d**: GGUF stores as ``[channels, kernel]``; depthwise
      ``Conv1d`` expects ``[channels, 1, kernel]``.
    - **dt_bias suffix**: GGUF ``ssm_dt.bias`` maps to
      ``dt_bias.bias`` after suffix splitting, but the model parameter
      is just ``dt_bias`` (an ``nn.Parameter``, not a module bias).
    - **DeltaNet A_log**: GGUF stores the SSM decay pre-transformed as
      ``ssm_a = -exp(A_log)``; mobius's ``GatedDeltaNet`` re-derives
      ``-exp(A_log)`` at runtime, so the raw log is recovered via
      ``A_log = log(-ssm_a)`` (scoped to ``linear_attn.A_log``).
    - **Zero-centered RMSNorm** (``gguf_arch`` in
      :data:`_OFFSET_NORM_GGUF_ARCHS`): the converter bakes ``+1`` into every
      ``*norm.weight`` except ``linear_attn.norm.weight``; mobius applies the
      ``1 +`` at runtime via :class:`OffsetRMSNorm`, so subtract ``1`` back out
      to avoid double-counting.
    - **Gated-DeltaNet V-head tiling** (``gguf_arch`` in
      :data:`_V_HEAD_REORDER_GGUF_ARCHS`, grouped linear attention): the
      converter reorders every V-indexed ``linear_attn`` tensor from HF grouped
      order into ggml tiled order; mobius consumes grouped order, so the tiling
      is undone via :func:`_reorder_deltanet_v_heads`.

    Args:
        state_dict: Dequantized GGUF weights keyed by HF tensor names.
        gguf_arch: The source GGUF architecture string (e.g. ``"qwen35"``),
            used to gate architecture-specific value transforms such as the
            zero-centered RMSNorm offset.
        config: The resolved :class:`ArchitectureConfig`; supplies the
            Gated-DeltaNet head counts / dims used to undo the V-head tiling.
    """
    import torch

    offset_norms = gguf_arch in _OFFSET_NORM_GGUF_ARCHS

    result: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if config is not None:
            _validate_moe_weight_shape(key, tuple(value.shape), config)
        if gguf_arch in {"dream", "llada-moe", "rnd1"} and ".self_attn.qkv_proj." in key:
            suffix = key.rsplit(".", 1)[-1]
            q_width = int(config.num_attention_heads) * int(config.head_dim)
            kv_width = int(config.num_key_value_heads) * int(config.head_dim)
            expected_width = q_width + 2 * kv_width
            if value.shape[0] != expected_width:
                raise ValueError(
                    f"Invalid fused {gguf_arch} QKV {suffix} width: expected "
                    f"{expected_width}, got {value.shape[0]}"
                )
            query, key_projection, value_projection = value.split(
                [q_width, kv_width, kv_width], dim=0
            )
            stem = key.rsplit(".qkv_proj.", 1)[0]
            result[f"{stem}.q_proj.{suffix}"] = query
            result[f"{stem}.k_proj.{suffix}"] = key_projection
            result[f"{stem}.v_proj.{suffix}"] = value_projection
            continue
        if gguf_arch == "bert" and ".attention.self.qkv." in key:
            suffix = key.rsplit(".", 1)[-1]
            expected_width = 3 * int(config.hidden_size)
            if value.shape[0] != expected_width:
                raise ValueError(
                    f"Invalid fused BERT QKV {suffix} width: expected {expected_width}, "
                    f"got {value.shape[0]}"
                )
            query, key_value, value_projection = value.chunk(3, dim=0)
            stem = key.rsplit(".qkv.", 1)[0]
            result[f"{stem}.query.{suffix}"] = query
            result[f"{stem}.key.{suffix}"] = key_value
            result[f"{stem}.value.{suffix}"] = value_projection
            continue
        # Stacked expert weights [num_experts, out, in] → per-expert
        unpacked = False
        for proj in ("gate_proj", "up_proj", "down_proj"):
            suffix = f".mlp.experts.{proj}.weight"
            if key.endswith(suffix) and value.dim() == 3:
                prefix = key[: -len(suffix)]
                for i in range(value.shape[0]):
                    result[f"{prefix}.mlp.experts.{i}.{proj}.weight"] = value[i]
                unpacked = True
                break
        if unpacked:
            continue

        # 1D shared_expert_gate → [1, hidden]
        if key.endswith(".mlp.shared_expert_gate.weight") and value.dim() == 1:
            result[key] = value.unsqueeze(0)
            continue

        # 2D conv1d → [channels, 1, kernel]
        if key.endswith(".conv1d.weight") and value.dim() == 2:
            result[key] = value.unsqueeze(1)
            continue

        # dt_bias.bias → dt_bias (nn.Parameter, not module bias)
        if key.endswith(".dt_bias.bias"):
            result[key[: -len(".bias")]] = value
            continue

        # DeltaNet A_log: undo the converter's pre-transform. GGUF's converter
        # stores the SSM decay already transformed as ``ssm_a = -exp(A_log)``
        # (llama.cpp applies ``-torch.exp`` to every ``.A_log`` tensor and the
        # reference then uses it *directly* as the decay coefficient ``a`` in
        # ``a * softplus(dt)``). mobius's ``GatedDeltaNet`` parameter is the raw
        # ``A_log`` and recomputes ``g = -exp(A_log) * softplus(...)`` at
        # runtime, so feeding it the already-negated-exp value squashes every
        # head's decay to ``-exp(-exp(A_log)) ≈ -1`` and the linear-attention
        # recurrence emits garbage. Invert to recover the raw log parameter,
        # ``A_log = log(-ssm_a)``, so mobius's ``-exp(A_log)`` reproduces the
        # original ``ssm_a`` exactly. Scoped to the GatedDeltaNet ``linear_attn``
        # projection so Mamba/PLaMo SSM modules (which consume ``A = -exp(A_log)``
        # directly) are left untouched.
        if key.endswith(".linear_attn.A_log"):
            result[key] = torch.log(-value)
            continue

        # Zero-centered RMSNorm: undo the converter's baked-in ``+1`` so
        # mobius's OffsetRMSNorm (which adds it back at runtime) does not
        # double-count. The DeltaNet internal ``linear_attn.norm`` is a plain
        # gated RMSNorm (no offset) and is excluded — mirroring exactly which
        # tensors the llama.cpp converter transforms.
        if (
            offset_norms
            and key.endswith("norm.weight")
            and not key.endswith(".linear_attn.norm.weight")
        ):
            result[key] = value - 1.0
            continue

        # layer_scalar.weight → layer_scalar (Gemma4 per-layer output scale is an
        # nn.Parameter, not a module weight). GGUF stores it as
        # blk.{i}.layer_output_scale.weight, which the tensor mapping renames to
        # model.layers.{i}.layer_scalar.weight; strip the artefact .weight suffix.
        if key.endswith(".layer_scalar.weight"):
            result[key[: -len(".weight")]] = value
            continue

        result[key] = value

    # DeltaNet V-head tiling: undo the converter's grouped→tiled permutation of
    # every V-indexed linear_attn tensor so mobius's GatedDeltaNet (which expects
    # HF grouped order) reads consistent heads. Runs last so it operates on the
    # already-normalized keys/shapes (renamed dt_bias, unsqueezed conv1d, ...).
    if gguf_arch in _V_HEAD_REORDER_GGUF_ARCHS:
        result = _reorder_deltanet_v_heads(result, config)

    return result


def _reorder_deltanet_v_heads(state_dict: dict, config) -> dict:
    """Undo the GGUF converter's grouped→tiled V-head permutation.

    llama.cpp's ``_LinearAttentionVReorderBase`` reorders every V-indexed
    Gated-DeltaNet tensor from HuggingFace *grouped* order (V-head
    ``s = group * v_per_k + j``) into ggml *tiled* order
    (``t = j * num_k_heads + group``) whenever the linear layer is grouped
    (``num_value_heads != num_key_heads``). mobius's ``GatedDeltaNet`` forward
    reshapes the value stream as ``num_value_heads`` contiguous ``head_v_dim``
    blocks in the original grouped order, so the GGUF tensors must be permuted
    back: grouped position ``s`` is fetched from tiled slot
    ``perm[s] = (s % v_per_k) * num_key_heads + (s // v_per_k)``.

    The permutation is applied (all derived from ``config`` — no hardcoded head
    counts) to:

    - ``in_proj_qkv`` output rows — V rows only (after ``2 * key_dim``);
    - ``in_proj_z`` output rows — all rows;
    - ``in_proj_a`` / ``in_proj_b`` output rows — one row per V-head;
    - ``A_log`` / ``dt_bias`` — one element per V-head;
    - ``conv1d`` channels — V channels only (after ``2 * key_dim``);
    - ``out_proj`` input columns — all columns (a block-granular permutation of
      the quantized ``K`` axis).

    Quantized projections are stored as MatMulNBits triplets
    (``weight`` ``[N, K/block, block/2]``, ``scales`` ``[N, K/block]``,
    ``zero_points`` ``[N, K/block/2]``). Output-row permutations reindex axis 0
    of all three; the ``out_proj`` input permutation reindexes the block axis
    (axis 1), valid because ``head_v_dim`` is a whole number of quant blocks.
    """
    import torch

    num_k_heads = getattr(config, "linear_num_key_heads", None)
    num_v_heads = getattr(config, "linear_num_value_heads", None)
    head_k_dim = getattr(config, "linear_key_head_dim", None)
    head_v_dim = getattr(config, "linear_value_head_dim", None)
    # Nothing to do unless this is a grouped linear-attention model.
    if not (num_k_heads and num_v_heads and head_k_dim and head_v_dim):
        return state_dict
    if num_v_heads == num_k_heads or num_v_heads % num_k_heads != 0:
        return state_dict

    v_per_k = num_v_heads // num_k_heads
    key_dim = head_k_dim * num_k_heads
    v_offset = 2 * key_dim  # in_proj_qkv / conv1d layout is [Q | K | V]

    # perm[s] = tiled slot holding grouped V-head s.
    head_perm = torch.tensor(
        [(s % v_per_k) * num_k_heads + (s // v_per_k) for s in range(num_v_heads)],
        dtype=torch.long,
    )

    def _expand(perm: torch.Tensor, stride: int) -> torch.Tensor:
        # Expand a per-head permutation into a per-row/-channel index.
        base = (perm * stride).unsqueeze(1) + torch.arange(stride)
        return base.reshape(-1)

    v_rows = _expand(head_perm, head_v_dim)  # length value_dim

    def _index_dim0(t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return t.index_select(0, idx)

    def _apply_rows(stem: str, idx: torch.Tensor) -> None:
        # Permute axis 0 of a float weight or a quantized triplet in place.
        for suffix in (".weight", ".scales", ".zero_points"):
            key = stem + suffix
            if key in state_dict:
                state_dict[key] = _index_dim0(state_dict[key], idx)

    def _apply_bare(key: str, idx: torch.Tensor) -> None:
        if key in state_dict:
            state_dict[key] = _index_dim0(state_dict[key], idx)

    layer_stems = {k.rsplit(".", 1)[0] for k in state_dict if ".linear_attn." in k}
    for stem in layer_stems:
        name = stem.rsplit(".", 1)[-1]
        if name == "in_proj_z":
            _apply_rows(stem, v_rows)
        elif name in ("in_proj_a", "in_proj_b"):
            _apply_rows(stem, head_perm)
        elif name == "in_proj_qkv":
            n_rows = state_dict[stem + ".weight"].shape[0]
            full = torch.cat([torch.arange(v_offset), v_offset + v_rows])
            assert full.numel() == n_rows, (n_rows, full.numel())
            _apply_rows(stem, full)
        elif name == "out_proj":
            _reorder_out_proj_cols(state_dict, stem, head_perm, head_v_dim)

    # Bare (non-".weight") linear_attn parameters.
    for k in list(state_dict):
        if k.endswith((".linear_attn.A_log", ".linear_attn.dt_bias")):
            _apply_bare(k, head_perm)
        elif k.endswith(".linear_attn.conv1d.weight"):
            conv = state_dict[k]
            n_ch = conv.shape[0]
            full = torch.cat([torch.arange(v_offset), v_offset + v_rows])
            assert full.numel() == n_ch, (n_ch, full.numel())
            state_dict[k] = _index_dim0(conv, full)

    return state_dict


def _reorder_out_proj_cols(state_dict: dict, stem: str, head_perm, head_v_dim: int) -> None:
    """Permute the quantized ``out_proj`` input (K) axis by V-head.

    ``out_proj`` maps ``value_dim -> hidden``; its input columns are the V
    stream, so they carry the same head tiling. In MatMulNBits form the K axis is
    the block axis (axis 1) of ``weight``/``scales`` and the packed block axis of
    ``zero_points`` (two 4-bit blocks per byte). ``head_v_dim`` spans a whole
    number of blocks, so the permutation is block-granular and lossless.
    """
    import torch

    weight = state_dict.get(stem + ".weight")
    if weight is None or weight.dim() < 2:
        return
    n_blocks = weight.shape[1]
    if n_blocks % head_perm.numel() != 0:
        raise ValueError(
            f"{stem}: cannot map {n_blocks} quant blocks onto "
            f"{head_perm.numel()} V-heads for column reorder"
        )
    blocks_per_head = n_blocks // head_perm.numel()

    def _expand(perm, stride):
        base = (perm * stride).unsqueeze(1) + torch.arange(stride)
        return base.reshape(-1)

    blk_idx = _expand(head_perm, blocks_per_head)
    state_dict[stem + ".weight"] = weight.index_select(1, blk_idx)

    scales = state_dict.get(stem + ".scales")
    if scales is not None:
        state_dict[stem + ".scales"] = scales.index_select(1, blk_idx)

    zp = state_dict.get(stem + ".zero_points")
    if zp is not None and zp.dim() >= 2:
        # zero_points pack two 4-bit blocks per byte along the block axis.
        if blocks_per_head % 2 != 0:
            raise ValueError(
                f"{stem}: {blocks_per_head} blocks/head is not byte-aligned for "
                "packed zero_points reorder"
            )
        zp_bytes_per_head = blocks_per_head // 2
        zp_idx = _expand(head_perm, zp_bytes_per_head)
        state_dict[stem + ".zero_points"] = zp.index_select(1, zp_idx)


def _has_quantized_weights(gguf_model, gguf_arch: str) -> bool:
    """Return whether a GGUF has mapped weights with a quantized tensor type."""
    from mobius.integrations.gguf._quant_registry import float_storage_type_ids
    from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names

    float_type_ids = float_storage_type_ids()

    for name, _raw, qtype, _shape in gguf_model.tensor_items_raw():
        hf_name = map_gguf_to_hf_names(name, gguf_arch)
        type_id = getattr(qtype, "value", qtype)
        if (
            hf_name is not None
            and hf_name.endswith(".weight")
            and type_id not in float_type_ids
        ):
            return True
    return False


def _reject_quantized_diffusion_fused_qkv(
    gguf_model,
    gguf_arch: str,
    *,
    preserve_quantization: bool,
) -> None:
    """Reject fused diffusion QKV before a quantized graph can be constructed.

    The diffusion graph owns separate QuantizedLinear Q/K/V modules, while the
    fused GGUF family maps to a synthetic ``qkv_proj`` stem that is not a graph
    target. The quantized loader therefore cannot attach packed blocks to it.
    Dequantizing the fused tensor and splitting it later is also invalid because
    the graph still expects packed parameters. This applies even when the fused
    tensor itself is float: any other quantized mapped tensor selects the packed
    graph, so the split Q/K/V targets remain quantized.
    """
    if not preserve_quantization or gguf_arch not in {"dream", "llada-moe", "rnd1"}:
        return

    fused = sorted(
        name
        for name in gguf_model.tensor_names
        if re.fullmatch(r"blk\.\d+\.attn_qkv\.weight", name)
    )
    if fused:
        raise ValueError(
            f"Quantization-preserving import of fused QKV is not supported for "
            f"{gguf_arch} GGUF ({fused[0]}). The graph has separate packed Q/K/V "
            "targets, so splitting after weight loading would leave them uninitialized. "
            "Use keep_quantized=False (or --dequantize) for a float import."
        )


def _detect_quant_params(gguf_model, gguf_arch: str) -> tuple[int, int, bool]:
    """Detect the common MatMulNBits target for GGUF projection weights.

    Q4_K-containing mixed presets target 4-bit, block-32 asymmetric
    MatMulNBits. Other files use the most common directly repackable type.

    Returns:
        ``(bits, block_size, is_symmetric)`` tuple.

    Raises:
        ValueError: If no mapped or repackable weight tensors are found.
    """
    from gguf import GGMLQuantizationType

    from mobius.integrations.gguf._quant_registry import (
        explicit_zero_point_type_names,
        float_storage_type_ids,
        get_quant_spec,
    )
    from mobius.integrations.gguf._repacker import can_repack, repack_quant_params
    from mobius.integrations.gguf._spec import QuantImportRoute
    from mobius.integrations.gguf._tencent_q1_0 import is_tencent_q1_0_layout
    from mobius.integrations.gguf._tensor_mapping import (
        map_gguf_to_hf_names,
    )

    counts: Counter = Counter()
    float_type_ids = float_storage_type_ids()
    for name, _raw, qtype, _shape in gguf_model.tensor_items_raw():
        hf_name = map_gguf_to_hf_names(name, gguf_arch)
        if hf_name is None or not hf_name.endswith(".weight"):
            continue
        type_id = getattr(qtype, "value", qtype)
        if type_id not in float_type_ids:
            counts[qtype] += 1

    if not counts:
        raise ValueError(
            "No mapped weight tensors found in GGUF file. "
            "Use keep_quantized=False for dequantized import."
        )

    quant_specs = {qtype: get_quant_spec(qtype) for qtype in counts}
    unknown = [qtype for qtype, spec in quant_specs.items() if spec is None]
    if unknown:
        names = ", ".join(sorted(getattr(qtype, "name", str(qtype)) for qtype in unknown))
        raise ValueError(
            f"GGUF contains qtypes outside the pinned llama.cpp census: {names}. "
            "Update the importer policy before loading this file."
        )
    rejected = [
        spec
        for spec in quant_specs.values()
        if spec is not None and spec.import_route is QuantImportRoute.REJECTED
    ]
    if rejected:
        details = "; ".join(
            f"{spec.name}: {spec.reason or 'no supported importer route'}" for spec in rejected
        )
        raise ValueError(
            "GGUF contains stored qtypes that cannot be imported safely: "
            f"{details} Re-quantize those tensors to a supported qtype."
        )

    native_counts = Counter(
        {qtype: count for qtype, count in counts.items() if _native_block_format(qtype)}
    )
    if native_counts:
        explicit_zero_point_types = explicit_zero_point_type_names()
        can_omit_zero_points = not any(
            getattr(qtype, "name", None) in explicit_zero_point_types
            for qtype in counts
            if qtype not in native_counts
        )
        logger.info(
            "Native GGUF quant types present; using 4-bit/block-32 module "
            "scaffolding for non-native quantized tensors",
        )
        return 4, 32, can_omit_zero_points

    if any(
        spec is not None and spec.import_route is QuantImportRoute.DEQUANTIZE_REQUANTIZE
        for spec in quant_specs.values()
    ):
        logger.info(
            "GGUF contains qtypes requiring dequantize/requantize; using the "
            "supported 4-bit/block-32 affine target"
        )
        return 4, 32, False

    # Q4_K_M is deliberately a mixed preset. Depending on tensor dimensions
    # and importance it may contain mostly Q5_0 plus Q4_K, Q6_K, and Q8_0.
    # The presence of Q4_K identifies the desired 4-bit MatMulNBits target;
    # choosing only among already-repackable types would incorrectly select
    # Q8_0 for the official Qwen2.5-0.5B Q4_K_M file.
    if GGMLQuantizationType.Q4_K in counts:
        dominant = GGMLQuantizationType.Q4_K
    else:
        repackable_counts = Counter(
            {
                qtype: count
                for qtype, count in counts.items()
                if can_repack(qtype.value if hasattr(qtype, "value") else qtype)
            }
        )
        if not repackable_counts:
            source_types = ", ".join(
                sorted({getattr(qtype, "name", str(qtype)) for qtype in counts})
            )
            raise ValueError(
                "No supported quantized preservation target for GGUF weight "
                f"types: {source_types}. Use keep_quantized=False (API) or "
                "--dequantize (CLI) for explicit float import."
            )
        dominant = repackable_counts.most_common(1)[0][0]
    dominant_value = dominant.value if hasattr(dominant, "value") else dominant
    params = repack_quant_params(dominant_value)
    assert params is not None
    bits, block_size = params
    # Whether the repacked form may drop zero points is a property of the repack
    # *target*, not of the source format: Q4_K and Q6_K both requantize through
    # the asymmetric affine path, so both need zero points even though Q6_K's
    # source form is symmetric around 32. Q4_0/Q8_0 look symmetric too, but
    # their GGUF dequantization is still ``(q - 8) * scale`` / ``(q - 128) *
    # scale``, and GatherBlockQuantized has diverging CPU/CUDA defaults when the
    # input is omitted, which corrupts embeddings before the first decoder layer.
    dominant_spec = get_quant_spec(dominant)
    assert dominant_spec is not None and dominant_spec.affine_repack is not None
    is_sym = dominant_spec.affine_repack.omit_zero_points

    # Tencent Q1_0 files reuse the Q1_0 type id but ship a different
    # on-disk layout (2-bit SEQ, 512-element blocks, fp16 scale per block).
    # Override the mainline defaults so the resulting QuantizedLinear
    # matches what parse_tencent_q1_0_tensor produces (4-bit packed).
    if dominant == GGMLQuantizationType.Q1_0 and is_tencent_q1_0_layout(gguf_model):
        # See _tencent_q1_0.py — the bits/zp flavour depends on a flag:
        #   default (fast):  bits=4 packed-uint8 zp=3 (inflated codebook)
        #   opt-in (small):  bits=2 float zp=1.5 (native SEQ layout)
        from mobius._flags import flags

        if flags.tencent_q1_0_use_native_2bit:
            bits, block_size, is_sym = 2, 128, False
        else:
            bits, block_size, is_sym = 4, 128, False

    logger.info(
        "Dominant GGUF quant type: %s (%d tensors, bits=%d, block_size=%d)",
        dominant,
        counts[dominant],
        bits,
        block_size,
    )
    return bits, block_size, is_sym


def _can_quantize_embedding(
    gguf_model,
    gguf_arch: str,
    *,
    bits: int,
    block_size: int,
) -> bool:
    """Return whether the token embedding can use GatherBlockQuantized.

    The graph has one quantization configuration shared by its quantized
    modules. Preserve the GGUF embedding only when its repacked representation
    uses the same bit width and block size as the projection weights.
    """
    from mobius.integrations.gguf._quant_registry import quant_import_decision
    from mobius.integrations.gguf._repacker import repack_quant_params
    from mobius.integrations.gguf._spec import QuantImportRoute, TensorRole
    from mobius.integrations.gguf._tencent_q1_0 import is_tencent_q1_0_layout
    from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names

    # Encoder modules currently expose plain Embedding parameters. Keep their
    # token tables explicitly dequantized until they have a QuantizedEmbedding
    # factory and a validated GatherBlockQuantized ABI.
    if gguf_arch in {"bert", "modern-bert", "t5", "t5encoder"}:
        return False

    # Tencent files reuse the Q1_0 type id for a custom layout that gguf-py
    # cannot size correctly. Embeddings from those files must stay dequantized.
    if is_tencent_q1_0_layout(gguf_model):
        return False

    for tensor in gguf_model.reader_tensors():
        mapped_name = map_gguf_to_hf_names(tensor.name, gguf_arch)
        if mapped_name is None or not mapped_name.endswith(
            ("model.embed_tokens.weight", "shared.weight")
        ):
            continue
        shape = tuple(reversed(tensor.shape))
        if len(shape) != 2:
            return False
        qtype = tensor.tensor_type
        qtype_val = qtype.value if hasattr(qtype, "value") else qtype
        if repack_quant_params(qtype_val) == (bits, block_size):
            return True
        route, _, _ = quant_import_decision(
            qtype,
            TensorRole.EMBEDDING,
            target_bits=bits,
            target_block_size=block_size,
        )
        return route is QuantImportRoute.DEQUANTIZE_REQUANTIZE
    return False


def _can_quantize_lm_head(gguf_model, gguf_arch: str) -> bool:
    """Return whether an untied GGUF output head can be kept quantized."""
    from mobius.integrations.gguf._quant_registry import lm_head_preserve_type_names
    from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names

    supported_types = lm_head_preserve_type_names()
    for name, _raw, qtype, shape in gguf_model.tensor_items_raw():
        mapped = map_gguf_to_hf_names(name, gguf_arch)
        if mapped is None or not mapped.endswith("lm_head.weight"):
            continue
        return len(shape) == 2 and getattr(qtype, "name", None) in supported_types
    return False


def _require_supported_requantization(
    *,
    bits: int,
    block_size: int,
    tensor_name: str,
) -> None:
    if bits != 4 or block_size != 32:
        raise ValueError(
            "keep_quantized MatMulNBits requantization currently supports only "
            f"4-bit/block-32 targets; got bits={bits} block={block_size} "
            f"for tensor {tensor_name}. Use keep_quantized=False or a "
            "4-bit/block-32 target."
        )


def repack_gguf_weight_to_target(
    gguf_model,
    raw,
    qtype,
    np_shape,
    *,
    target_bits: int,
    target_block_size: int,
    target_symmetric: bool,
    tensor_name: str,
    tensor_role=None,
):
    """Repack one 2-D GGUF weight to the graph's common MatMulNBits target.

    Reuses the shared repacker machinery: a tensor whose native repacked layout
    already matches ``(target_bits, target_block_size)`` is repacked directly;
    otherwise it is dequantized and requantized to the target layout. This is
    the single-tensor building block reused by both the text-only
    (:func:`_load_quantized_state_dict`) and the multimodal quantized loaders.

    Args:
        gguf_model: The source :class:`GGUFModel`.
        raw: Raw tensor bytes (as returned by ``tensor_items_raw``).
        qtype: The GGUF quantization type of the tensor.
        np_shape: The tensor's logical ``(N, K)`` shape.
        target_bits: Target MatMulNBits bit width.
        target_block_size: Target MatMulNBits block size.
        target_symmetric: Whether the requantization path should omit
            zero-points (symmetric). Only used when requantizing.
        tensor_name: Name used for error messages.

    Returns:
        A ``RepackedTensor`` with the target ``(bits, block_size)`` layout.
    """
    import numpy as np

    from mobius.integrations.gguf._quant_registry import quant_import_decision
    from mobius.integrations.gguf._repacker import (
        can_repack,
        repack_dequantized_tensor,
        repack_gguf_tensor,
    )
    from mobius.integrations.gguf._spec import QuantImportRoute, TensorRole

    if tensor_role is None:
        tensor_role = TensorRole.PROJECTION
    route, _exactness, reason = quant_import_decision(
        qtype,
        tensor_role,
        target_bits=target_bits,
        target_block_size=target_block_size,
    )
    if route is QuantImportRoute.REJECTED:
        qtype_name = getattr(qtype, "name", str(qtype))
        raise ValueError(
            f"Cannot import GGUF tensor {tensor_name} ({qtype_name}, "
            f"role={tensor_role.value}): {reason}"
        )

    qtype_val = qtype.value if hasattr(qtype, "value") else qtype
    if route is QuantImportRoute.AFFINE_REPACK and can_repack(qtype_val):
        shape_2d = (int(np_shape[0]), int(np_shape[1]))
        repacked = repack_gguf_tensor(raw.ravel().view(np.uint8), qtype_val, shape_2d)
        if repacked.bits == target_bits and repacked.block_size == target_block_size:
            return repacked

    _require_supported_requantization(
        bits=target_bits,
        block_size=target_block_size,
        tensor_name=tensor_name,
    )
    values = gguf_model.dequantize_raw_tensor(raw, qtype, np_shape)
    return repack_dequantized_tensor(
        values,
        bits=target_bits,
        block_size=target_block_size,
        symmetric=target_symmetric,
    )


def _load_dequantized_state_dict(
    gguf_model,
    gguf_arch: str,
    name_mapper: Callable[[str, str], str | None] | None = None,
    warn_unmapped: bool = True,
    reuse_candidates: dict | None = None,
) -> dict:
    """Load all tensors dequantized to float (Phase 1 path).

    ``name_mapper`` maps a GGUF tensor name to its target state-dict key (or
    ``None`` to skip). Defaults to the main-model mapping; the MTP head builder
    injects a head-scoped mapper so the trailing ``blk.<mtp>.nextn.*`` /
    attention block is routed to the head module instead of being dropped.
    """
    import numpy as np
    import torch

    from mobius.integrations.gguf._tensor_mapping import (
        map_gguf_to_hf_names,
    )

    if name_mapper is None:
        name_mapper = map_gguf_to_hf_names

    if gguf_arch in {"dflash", "eagle3"}:
        # d2t is an integer orchestration table, not an ONNX initializer. The
        # gguf package intentionally has no I64 "dequantizer", so exclude it
        # before asking the reader to materialize neural tensors.
        tensors = (
            (name, gguf_model.dequantize_raw_tensor(raw, qtype, shape))
            for name, raw, qtype, shape in gguf_model.tensor_items_raw()
            if name != "d2t"
        )
    else:
        tensors = gguf_model.tensor_items()

    state_dict = {}
    for gguf_name, np_array in tqdm.tqdm(
        tensors,
        desc="Dequantizing tensors",
        total=gguf_model.num_tensors,
    ):
        hf_name = name_mapper(gguf_name, gguf_arch)
        if hf_name is not None:
            # F32/F16 tensors are mmap'd read-only views; make
            # writable so PyTorch can mutate if needed.
            if not np_array.flags.writeable:
                np_array = np.array(np_array)
            tensor = torch.from_numpy(np_array)
            state_dict[hf_name] = tensor
            if reuse_candidates is not None:
                qtype = gguf_model.get_tensor_type(gguf_name)
                if getattr(qtype, "name", None) in {"F32", "F16"}:
                    from mobius.integrations.gguf._reuse import GGUFReuseCandidate

                    offset, length, qtype_name = gguf_model.tensor_storage_range(gguf_name)
                    reuse_candidates[id(tensor)] = GGUFReuseCandidate(
                        gguf_name,
                        offset,
                        length,
                        qtype_name,
                        tuple(int(dim) for dim in np_array.shape),
                    )
        else:
            if warn_unmapped:
                logger.warning("Unmapped GGUF tensor: %s (skipped)", gguf_name)
    return state_dict


def _load_quantized_state_dict(
    gguf_model,
    gguf_arch: str,
    module,
    config,
    name_mapper: Callable[[str, str], str | None] | None = None,
    warn_unmapped: bool = True,
    reuse_candidates: dict | None = None,
) -> dict:
    """Load tensors, preserving native blocks or normalizing to MatMulNBits.

    Projection weights (Q/K/V/O, MLP, and a quantized output head) are
    converted to the graph's common MatMulNBits format, and compatible token
    embeddings to GatherBlockQuantized format. Mixed or unsupported source
    types are dequantized and requantized when they do not match that target.
    Incompatible embeddings, norms, and other non-linear tensors remain
    dequantized.

    For llama-family models, quantized Q/K weights receive the
    row-level reverse-permutation that ``process_tensors`` would
    normally apply.
    """
    import numpy as np
    import torch
    from gguf import GGMLQuantizationType, dequantize

    from mobius.components import (
        BlockQuantizedLinear,
        Embedding,
        Linear,
        QuantizedEmbedding,
        QuantizedLinear,
    )
    from mobius.integrations.gguf._quant_registry import (
        get_quant_spec,
        quant_import_decision,
    )
    from mobius.integrations.gguf._repacker import (
        can_repack,
        preserve_native_blocks,
        repack_dequantized_tensor,
        repack_gguf_tensor,
    )
    from mobius.integrations.gguf._spec import QuantImportRoute, TensorRole
    from mobius.integrations.gguf._tencent_q1_0 import (
        is_tencent_q1_0_layout,
        parse_tencent_q1_0_tensor,
    )
    from mobius.integrations.gguf._tensor_mapping import (
        map_gguf_to_hf_names,
    )
    from mobius.integrations.gguf._tensor_processors import (
        _reverse_permute,
    )

    if name_mapper is None:
        name_mapper = map_gguf_to_hf_names

    # Collect module paths that use QuantizedLinear so we know
    # which .weight parameters should receive repacked data.
    quantized_stems = set()
    quantized_output_sizes: dict[str, int] = {}
    native_block_stems: dict[str, str] = {}
    quantized_embedding_stems = set()
    float_linear_stems = set()
    embedding_stems = set()
    for mod_name, mod in module.named_modules():
        if isinstance(mod, QuantizedLinear) or getattr(mod, "_gguf_quantized_linear", False):
            quantized_stems.add(mod_name)
            quantized_output_sizes[mod_name] = mod._n
        elif isinstance(mod, BlockQuantizedLinear):
            native_block_stems[mod_name] = mod._format
        elif isinstance(mod, QuantizedEmbedding):
            quantized_embedding_stems.add(mod_name)
            embedding_stems.add(mod_name)
        elif isinstance(mod, Linear):
            float_linear_stems.add(mod_name)
        elif isinstance(mod, Embedding):
            embedding_stems.add(mod_name)

    num_heads = getattr(config, "num_attention_heads", None)
    num_kv_heads = getattr(config, "num_key_value_heads", None)
    model_type = getattr(config, "model_type", None)

    # Detect Tencent's non-mainline Q1_0 layout once per file. Reading
    # such tensors requires a custom parser keyed on the explicit
    # per-tensor file offset (mainline byte sizes are wrong).
    tencent_q1_0 = is_tencent_q1_0_layout(gguf_model)
    if tencent_q1_0:
        gguf_path = str(gguf_model._path)
        data_section_offset = gguf_model._reader.data_offset
        tensors_by_name = {t.name: t for t in gguf_model._reader.tensors}
        logger.info(
            "Detected Tencent Q1_0 layout (block_size=512, 2-bit SEQ); "
            "using custom per-tensor parser"
        )

    state_dict: dict[str, torch.Tensor] = {}
    n_repacked = 0
    n_requantized = 0
    target_bits = config.quantization.bits
    target_block_size = config.quantization.group_size
    target_symmetric = config.quantization.sym

    for gguf_name, raw, qtype, np_shape in tqdm.tqdm(
        gguf_model.tensor_items_raw(),
        desc="Repacking tensors",
        total=gguf_model.num_tensors,
    ):
        if gguf_arch in {"dflash", "eagle3"} and gguf_name == "d2t":
            continue
        hf_name = name_mapper(gguf_name, gguf_arch)
        if hf_name is None:
            if warn_unmapped:
                logger.warning("Unmapped GGUF tensor: %s (skipped)", gguf_name)
            continue
        _validate_moe_weight_shape(hf_name, tuple(int(dim) for dim in np_shape), config)
        module_hf_name = hf_name
        if gguf_arch == "bert" and module_hf_name.startswith("bert."):
            module_hf_name = module_hf_name[len("bert.") :]
        elif gguf_arch == "modern-bert" and module_hf_name.startswith("model."):
            module_hf_name = module_hf_name[len("model.") :]
        elif gguf_arch in {"t5", "t5encoder"}:
            from mobius.models.t5 import _rename_t5_weight

            renamed = _rename_t5_weight(
                module_hf_name,
                is_gated_act=bool(getattr(config, "is_gated_act", False)),
            )
            if renamed is not None:
                module_hf_name = renamed

        # Determine the int value of the quant type for can_repack
        qtype_val = qtype.value if hasattr(qtype, "value") else qtype

        # Repack every target QuantizedLinear weight. Mixed GGUF presets
        # otherwise leave unsupported source types as full float matrices,
        # which cannot fit the graph's packed MatMulNBits initializer shape.
        stem = hf_name[: -len(".weight")] if hf_name.endswith(".weight") else None
        module_stem = (
            module_hf_name[: -len(".weight")] if module_hf_name.endswith(".weight") else None
        )
        is_tencent_q1_0_tensor = tencent_q1_0 and qtype == GGMLQuantizationType.Q1_0
        is_quantized_embedding = (
            module_stem is not None and module_stem in quantized_embedding_stems
        )
        should_repack = module_stem is not None and (
            module_stem in quantized_stems or is_quantized_embedding
        )

        native_targets = _native_block_target_stems(
            hf_name,
            np_shape,
            set(native_block_stems),
        )
        affine_targets = _native_block_target_stems(
            module_hf_name,
            np_shape,
            quantized_stems,
        )
        fused_projection_targets = _fused_projection_target_stems(
            module_hf_name, quantized_stems
        )
        native_spec = _native_block_spec(qtype)
        quant_spec = get_quant_spec(qtype)
        is_embedding_tensor = module_stem is not None and module_stem in embedding_stems
        is_encoder_embedding = is_embedding_tensor and gguf_arch in {
            "bert",
            "modern-bert",
        }
        tensor_role = (
            TensorRole.EMBEDDING
            if is_embedding_tensor
            else TensorRole.EXPERT
            if len(np_shape) == 3 and ".experts." in hf_name
            else TensorRole.OUTPUT
            if hf_name == "lm_head.weight"
            else TensorRole.PROJECTION
            if module_stem is not None
            and (
                module_stem in quantized_stems
                or module_stem in native_block_stems
                or module_stem in float_linear_stems
            )
            else TensorRole.NON_MATMUL
        )
        route = QuantImportRoute.DEQUANTIZE_REQUANTIZE
        if quant_spec is not None and quant_spec.is_quantized_storage:
            route, _exactness, reason = quant_import_decision(
                qtype,
                tensor_role,
                target_bits=target_bits,
                target_block_size=target_block_size,
            )
            if route is QuantImportRoute.REJECTED:
                raise ValueError(
                    f"Cannot import GGUF tensor {gguf_name} mapped to {hf_name} "
                    f"({quant_spec.name}, role={tensor_role.value}): {reason}"
                )
            if route is QuantImportRoute.NATIVE_BYTES and not native_targets:
                raise ValueError(
                    f"Cannot preserve native {quant_spec.name} bytes for {hf_name}: "
                    "the mapped module does not expose a compatible "
                    "BlockQuantizedMatMul target."
                )
            if (
                tensor_role is TensorRole.EMBEDDING
                and route is not QuantImportRoute.DEQUANTIZE_FLOAT
                and not is_quantized_embedding
                and not is_encoder_embedding
            ):
                raise ValueError(
                    f"Cannot keep {quant_spec.name} embedding {hf_name} quantized: "
                    "the model graph does not expose GatherBlockQuantized. Use "
                    "keep_quantized=False for explicit float import."
                )
            if (
                tensor_role in {TensorRole.PROJECTION, TensorRole.OUTPUT}
                and route
                in {
                    QuantImportRoute.AFFINE_REPACK,
                    QuantImportRoute.DEQUANTIZE_REQUANTIZE,
                }
                and not should_repack
                and not fused_projection_targets
            ):
                raise ValueError(
                    f"Cannot keep {quant_spec.name} {tensor_role.value} {hf_name} "
                    "quantized: the model graph does not expose MatMulNBits. Use "
                    "keep_quantized=False for explicit float import."
                )
        if fused_projection_targets:
            values = gguf_model.dequantize_raw_tensor(raw, qtype, np_shape)
            offset = 0
            for target_stem in fused_projection_targets:
                n_out = quantized_output_sizes[target_stem]
                target_values = values[offset : offset + n_out]
                offset += n_out
                repacked = repack_dequantized_tensor(
                    target_values,
                    bits=target_bits,
                    block_size=target_block_size,
                    symmetric=target_symmetric,
                )
                state_dict[f"{target_stem}.weight"] = torch.from_numpy(repacked.weight)
                state_dict[f"{target_stem}.scales"] = torch.from_numpy(repacked.scales)
                if repacked.zero_points is not None:
                    state_dict[f"{target_stem}.zero_points"] = torch.from_numpy(
                        repacked.zero_points
                    )
            if offset != int(np_shape[0]):
                raise ValueError(
                    f"Fused QKV tensor {hf_name!r} has {np_shape[0]} rows, "
                    f"but Q/K/V targets require {offset}"
                )
            n_repacked += len(fused_projection_targets)
            n_requantized += 1
        elif native_targets and native_spec is not None:
            n_out = int(np_shape[-2])
            k_in = int(np_shape[-1])
            packed = preserve_native_blocks(
                raw,
                qtype_val,
                (len(native_targets) * n_out, k_in),
            )
            packed = packed.reshape(
                len(native_targets),
                n_out,
                packed.shape[-2],
                native_spec.bytes,
            )
            for index, native_stem in enumerate(native_targets):
                w = torch.from_numpy(np.array(packed[index], copy=True))
                target_name = f"{native_stem}.weight"
                needs_permute = _needs_qk_permute(
                    target_name,
                    num_heads,
                    num_kv_heads,
                    model_type,
                    gguf_arch,
                )
                if needs_permute:
                    n_head = (
                        num_heads
                        if ".q_proj." in target_name or ".qkv_proj." in target_name
                        else num_kv_heads
                    )
                    w = _reverse_permute(w, n_head)
                state_dict[target_name] = w
                if (
                    reuse_candidates is not None
                    and len(native_targets) == 1
                    and not needs_permute
                ):
                    from mobius.integrations.gguf._reuse import GGUFReuseCandidate

                    offset, length, qtype_name = gguf_model.tensor_storage_range(gguf_name)
                    reuse_candidates[id(w)] = GGUFReuseCandidate(
                        gguf_name,
                        offset,
                        length,
                        qtype_name,
                        tuple(int(dim) for dim in w.shape),
                    )
            n_repacked += len(native_targets)
        elif affine_targets:
            # GGUF stores routed experts as one expert-major 3-D tensor
            # [E, N, K]. MatMulNBits parameters remain per expert, so repack
            # the contiguous [E*N, K] matrix once and split only after packing.
            num_experts = len(affine_targets)
            n_out = int(np_shape[-2])
            k_in = int(np_shape[-1])
            aggregate_shape = (num_experts * n_out, k_in)
            if can_repack(qtype_val):
                repacked = repack_gguf_tensor(
                    raw.ravel().view(np.uint8),
                    qtype_val,
                    aggregate_shape,
                )
                if repacked.bits != target_bits or repacked.block_size != target_block_size:
                    _require_supported_requantization(
                        bits=target_bits,
                        block_size=target_block_size,
                        tensor_name=hf_name,
                    )
                    values = gguf_model.dequantize_raw_tensor(raw, qtype, np_shape)
                    repacked = repack_dequantized_tensor(
                        values.reshape(aggregate_shape),
                        bits=target_bits,
                        block_size=target_block_size,
                        symmetric=target_symmetric,
                    )
                    n_requantized += 1
            else:
                _require_supported_requantization(
                    bits=target_bits,
                    block_size=target_block_size,
                    tensor_name=hf_name,
                )
                values = gguf_model.dequantize_raw_tensor(raw, qtype, np_shape)
                repacked = repack_dequantized_tensor(
                    values.reshape(aggregate_shape),
                    bits=target_bits,
                    block_size=target_block_size,
                    symmetric=target_symmetric,
                )
                n_requantized += 1

            packed = repacked.weight.reshape(num_experts, n_out, *repacked.weight.shape[1:])
            scales = repacked.scales.reshape(num_experts, n_out, *repacked.scales.shape[1:])
            zero_points = (
                repacked.zero_points.reshape(
                    num_experts,
                    n_out,
                    *repacked.zero_points.shape[1:],
                )
                if repacked.zero_points is not None
                else None
            )
            for index, target_stem in enumerate(affine_targets):
                target_name = f"{target_stem}.weight"
                weight = torch.from_numpy(np.array(packed[index], copy=True))
                scale = torch.from_numpy(np.array(scales[index], copy=True))
                zero_point = (
                    torch.from_numpy(np.array(zero_points[index], copy=True))
                    if zero_points is not None
                    else None
                )
                if _needs_qk_permute(
                    target_name,
                    num_heads,
                    num_kv_heads,
                    model_type,
                    gguf_arch,
                ):
                    n_head = num_heads if ".q_proj." in target_name else num_kv_heads
                    weight = _reverse_permute(weight, n_head)
                    scale = _reverse_permute(scale, n_head)
                    if zero_point is not None:
                        zero_point = _reverse_permute(zero_point, n_head)
                state_dict[target_name] = weight
                state_dict[f"{target_stem}.scales"] = scale
                if zero_points is not None:
                    assert zero_point is not None
                    state_dict[f"{target_stem}.zero_points"] = zero_point
            n_repacked += num_experts
        elif should_repack:
            if is_tencent_q1_0_tensor:
                repacked = parse_tencent_q1_0_tensor(
                    gguf_path,
                    data_section_offset,
                    tensors_by_name[gguf_name],
                )
            elif route is QuantImportRoute.AFFINE_REPACK and can_repack(qtype_val):
                shape_2d = (int(np_shape[0]), int(np_shape[1]))
                repacked = repack_gguf_tensor(
                    raw.ravel().view(np.uint8),
                    qtype_val,
                    shape_2d,
                )
                if repacked.bits != target_bits or repacked.block_size != target_block_size:
                    _require_supported_requantization(
                        bits=target_bits,
                        block_size=target_block_size,
                        tensor_name=hf_name,
                    )
                    values = gguf_model.dequantize_raw_tensor(raw, qtype, np_shape)
                    repacked = repack_dequantized_tensor(
                        values,
                        bits=target_bits,
                        block_size=target_block_size,
                        symmetric=target_symmetric,
                    )
                    n_requantized += 1
            else:
                _require_supported_requantization(
                    bits=target_bits,
                    block_size=target_block_size,
                    tensor_name=hf_name,
                )
                values = gguf_model.dequantize_raw_tensor(raw, qtype, np_shape)
                repacked = repack_dequantized_tensor(
                    values,
                    bits=target_bits,
                    block_size=target_block_size,
                    symmetric=target_symmetric,
                )
                n_requantized += 1
            w = torch.from_numpy(repacked.weight)
            s = torch.from_numpy(repacked.scales)

            # Apply Q/K row permutation to quantized tensors
            # (same transform as _process_llama, on all arrays). Only
            # llama-family archs use the interleaved-rope permute; Qwen
            # and others must NOT be permuted.
            if _needs_qk_permute(hf_name, num_heads, num_kv_heads, model_type, gguf_arch):
                n_head = (
                    num_heads
                    if ".q_proj." in hf_name or ".qkv_proj." in hf_name
                    else num_kv_heads
                )
                w = _reverse_permute(w, n_head)
                s = _reverse_permute(s, n_head)

            if is_quantized_embedding:
                state_dict[f"{stem}.qweight"] = w.reshape(w.shape[0], -1)
            else:
                state_dict[hf_name] = w
            state_dict[f"{stem}.scales"] = s
            if repacked.zero_points is not None:
                zp = torch.from_numpy(repacked.zero_points)
                if _needs_qk_permute(hf_name, num_heads, num_kv_heads, model_type, gguf_arch):
                    zp = _reverse_permute(zp, n_head)
                state_dict[f"{stem}.zero_points"] = zp
            n_repacked += 1
        else:
            # Dequantize to float
            if qtype in (
                GGMLQuantizationType.F32,
                GGMLQuantizationType.F16,
            ):
                arr = gguf_model.get_tensor(gguf_name)
                # F32/F16 tensors are mmap'd read-only views
                if not arr.flags.writeable:
                    arr = np.array(arr)
            else:
                arr = dequantize(raw, qtype).reshape(np_shape)
            tensor = torch.from_numpy(arr)
            state_dict[hf_name] = tensor
            if reuse_candidates is not None and qtype in (
                GGMLQuantizationType.F32,
                GGMLQuantizationType.F16,
            ):
                from mobius.integrations.gguf._reuse import GGUFReuseCandidate

                offset, length, qtype_name = gguf_model.tensor_storage_range(gguf_name)
                reuse_candidates[id(tensor)] = GGUFReuseCandidate(
                    gguf_name,
                    offset,
                    length,
                    qtype_name,
                    tuple(int(dim) for dim in arr.shape),
                )

    logger.info(
        "Loaded %d state_dict entries (%d GGUF tensors repacked for quantized ops, "
        "%d requantized from mixed source types)",
        len(state_dict),
        n_repacked,
        n_requantized,
    )
    return state_dict


def _validate_moe_weight_shape(
    name: str,
    shape: tuple[int, ...],
    config,
) -> None:
    """Reject router/expert tensors that could otherwise be partially routed."""
    num_experts = getattr(config, "num_local_experts", None)
    if num_experts is None:
        return
    expert_size = getattr(config, "moe_intermediate_size", None) or config.intermediate_size
    if ".mlp.experts." in name:
        projection = name.rsplit(".mlp.experts.", 1)[1].split(".", 1)[0]
        if projection not in {"gate_proj", "up_proj", "down_proj"}:
            return
        expected = (
            (num_experts, config.hidden_size, expert_size)
            if projection == "down_proj"
            else (num_experts, expert_size, config.hidden_size)
        )
        if shape != expected:
            raise ValueError(
                f"Invalid stacked expert shape for {name}: expected {expected}, got {shape}"
            )
    elif name.endswith(".mlp.gate.weight"):
        expected = (num_experts, config.hidden_size)
        if shape != expected:
            raise ValueError(
                f"Invalid router shape for {name}: expected {expected}, got {shape}"
            )


def _needs_qk_permute(
    hf_name: str,
    num_heads: int | None,
    num_kv_heads: int | None,
    model_type: str | None = None,
    gguf_arch: str | None = None,
) -> bool:
    """Check if this tensor needs Q/K reverse-permutation.

    Two conditions must hold: the tensor must be a Q/K projection weight,
    AND the model architecture must actually use llama.cpp's
    interleaved-rope permute. Name-based gating alone is insufficient —
    Qwen2/Qwen3 use ``.q_proj.``/``.k_proj.`` names too but store Q/K in
    plain HF order (NEOX rope) and must NOT be permuted, or their
    attention heads get scrambled and the model emits garbage.
    """
    from mobius.integrations.gguf._arch_registry import try_get_arch_spec
    from mobius.integrations.gguf._tensor_processors import needs_llama_qk_permute

    if num_heads is None or num_kv_heads is None:
        return False
    spec = try_get_arch_spec(gguf_arch) if gguf_arch is not None else None
    permute = spec.llama_qk_permute if spec is not None else needs_llama_qk_permute(model_type)
    if not permute:
        return False
    return (
        ".q_proj." in hf_name or ".k_proj." in hf_name or ".qkv_proj." in hf_name
    ) and hf_name.endswith(".weight")
