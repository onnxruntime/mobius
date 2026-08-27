# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Validate and emit GGUF multi-token-prediction (MTP) auxiliary heads.

The dense ``unsloth/Qwen3.8-27B`` (and ``Qwen/Qwen3.6-27B``) GGUF ships a
trained MTP head as the trailing ``blk.<N>`` block, tagged by the
``<arch>.nextn_predict_layers`` metadata key.  The base decoder build drops
that block (see :func:`gguf_to_config`), so the exported ONNX has no
self-speculative drafter.  This module rebuilds the head as a standalone
:class:`~mobius.models.qwen35_mtp.Qwen35MtpModel` sidecar and loads the
``blk.<N>.nextn.*`` / attention / FFN tensors into it.

The head graph mirrors llama.cpp's NEXTN tensors and vLLM's
``Qwen3_5MultiTokenPredictor``::

    blk.<N>.nextn.enorm            -> pre_fc_norm_embedding   (OffsetRMSNorm)
    blk.<N>.nextn.hnorm            -> pre_fc_norm_hidden       (OffsetRMSNorm)
    blk.<N>.nextn.eh_proj          -> fc                       (2H -> H GEMM)
    blk.<N>.nextn.embed_tokens     -> embed_tokens             (optional)
    blk.<N>.nextn.shared_head_norm -> norm                     (optional)
    blk.<N>.nextn.shared_head_head -> lm_head                  (optional)
    blk.<N>.attn_norm              -> layers.0.input_layernorm
    blk.<N>.attn_q/k/v/output      -> layers.0.self_attn.{q,k,v,o}_proj
    blk.<N>.attn_q_norm/k_norm     -> layers.0.self_attn.{q,k}_norm
    blk.<N>.post_attention_norm    -> layers.0.post_attention_layernorm
    blk.<N>.ffn_gate/up/down       -> layers.0.mlp.{gate,up,down}_proj

Absent optional tables fall back exactly to the target embedding, final norm,
and output head. Dedicated embedding/head tables become sidecar modules;
fallback embedding/head ownership remains explicit in runtime metadata.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from collections.abc import Callable
from types import MappingProxyType
from typing import TYPE_CHECKING

from mobius.integrations.gguf._spec import Support

if TYPE_CHECKING:
    from mobius._model_package import ModelPackage
    from mobius.integrations.gguf._quantization_report import GGUFQuantizationReport

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, slots=True)
class MtpArchitectureCapability:
    """Pinned llama.cpp MTP capability for one ``general.architecture`` value."""

    support: Support
    loader_behavior: str
    block_kind: str
    reason: str


_DENSE_QWEN35_MTP = MtpArchitectureCapability(
    support=Support.SUPPORTED,
    loader_behavior="executed-sidecar",
    block_kind="dense-full-attention",
    reason=(
        "The appended block is exactly one dense Qwen3.5 full-attention decoder layer "
        "with the standard per-block nextn conditioning tensors."
    ),
)

_PINNED_MTP_CAPABILITIES = MappingProxyType(
    {
        # Exact dense sidecar supported by this module.
        "qwen35": _DENSE_QWEN35_MTP,
        # Pinned loaders execute these heads, but their routed/stateful decoder
        # blocks are not represented by Qwen35MtpModel.
        "bailingmoe3": MtpArchitectureCapability(
            Support.REJECTED,
            "executed-sidecar",
            "routed-kda-mla",
            "the MTP block carries routed experts and KDA/MLA state semantics",
        ),
        "cohere2moe": MtpArchitectureCapability(
            Support.REJECTED,
            "executed-sidecar",
            "routed-full-attention",
            "the MTP block carries Cohere2 routed and shared experts",
        ),
        "deepseek2": MtpArchitectureCapability(
            Support.REJECTED,
            "executed-sidecar",
            "routed-mla",
            "the MTP block carries DeepSeek MLA and routed-expert semantics",
        ),
        "deepseek32": MtpArchitectureCapability(
            Support.REJECTED,
            "executed-sidecar",
            "routed-dsa-mla",
            "the MTP block carries DSA/MLA attention and routed experts",
        ),
        "deepseek4": MtpArchitectureCapability(
            Support.REJECTED,
            "executed-sidecar",
            "routed-hyperconnection",
            "the MTP block carries hyper-connections, compressed state, and routed experts",
        ),
        "glm-dsa": MtpArchitectureCapability(
            Support.REJECTED,
            "executed-sidecar",
            "routed-dsa-mla",
            "the MTP block carries GLM DSA/MLA and routed-expert semantics",
        ),
        "hy_v3": MtpArchitectureCapability(
            Support.REJECTED,
            "executed-sidecar",
            "routed-hyperconnection",
            "the MTP block carries HY-V3 hyper-connections and routed experts",
        ),
        "mimo2": MtpArchitectureCapability(
            Support.REJECTED,
            "executed-sidecar",
            "routed-full-attention",
            "the MTP block carries MiMo2 routed experts and layer-output norm fallback",
        ),
        "nemotron_h_moe": MtpArchitectureCapability(
            Support.REJECTED,
            "executed-sidecar",
            "routed-hybrid",
            "the MTP block carries Nemotron-H routed experts and hybrid-state semantics",
        ),
        "qwen35moe": MtpArchitectureCapability(
            Support.REJECTED,
            "executed-sidecar",
            "routed-full-attention",
            "MTP blocks use routed experts and a shared expert that are not represented",
        ),
        "qwen3next": MtpArchitectureCapability(
            Support.REJECTED,
            "executed-sidecar",
            "routed-hybrid",
            "the MTP block carries Qwen3-Next routed experts and hybrid-state semantics",
        ),
        "step35": MtpArchitectureCapability(
            Support.REJECTED,
            "executed-sidecar",
            "routed-full-attention",
            "the MTP block carries Step3.5 routed and shared experts",
        ),
        # These pinned loaders consume/preserve NextN metadata or tensors but do
        # not expose an executable decoder-MTP graph for the serialized head.
        "bailingmoe2": MtpArchitectureCapability(
            Support.REJECTED,
            "preserved-unused",
            "routed-full-attention",
            "the pinned loader skips the appended NextN blocks during execution",
        ),
        "dots3note": MtpArchitectureCapability(
            Support.REJECTED,
            "preserved-unused",
            "routed-dsa",
            "the pinned loader marks MTP tensors skipped and has no MTP graph",
        ),
        "exaone-moe": MtpArchitectureCapability(
            Support.REJECTED,
            "preserved-unused",
            "routed-full-attention",
            "the pinned loader preserves NextN tensors without an MTP graph",
        ),
        "exaone4": MtpArchitectureCapability(
            Support.REJECTED,
            "preserved-unused",
            "dense-full-attention",
            "the pinned loader preserves NextN tensors without an MTP graph",
        ),
        "glm4": MtpArchitectureCapability(
            Support.REJECTED,
            "preserved-unused",
            "dense-full-attention",
            "the pinned loader preserves NextN tensors without an MTP graph",
        ),
        "glm4moe": MtpArchitectureCapability(
            Support.REJECTED,
            "preserved-unused",
            "routed-full-attention",
            "the pinned loader preserves NextN tensors without an MTP graph",
        ),
        "nemotron_h": MtpArchitectureCapability(
            Support.REJECTED,
            "converter-conditional",
            "hybrid",
            "only the distinct nemotron_h_moe loader owns the routed MTP graph",
        ),
        # Gemma4 Assistant is a standalone drafter with global projection
        # tensors, not an appended target-owned sidecar.
        "gemma4-assistant": MtpArchitectureCapability(
            Support.REJECTED,
            "standalone-drafter",
            "legacy-global-projections",
            "nextn.pre_projection/post_projection describe a standalone assistant model",
        ),
        # Granite Switch repurposes n_layer_nextn for its router-cache layer.
        # llama.cpp's generic saver can therefore re-emit this metadata, but it
        # never denotes an MTP head for this architecture.
        "graniteswitch": MtpArchitectureCapability(
            Support.REJECTED,
            "metadata-reemitted-non-mtp",
            "router-cache-sentinel",
            "nextn_predict_layers is a round-trip artifact for the extra router layer",
        ),
    }
)

_MTP_METADATA_SUFFIXES = (".nextn_predict_layers", ".nextn.predict_layers")
_LEGACY_NEXTN_TENSORS = frozenset(
    {
        "nextn.pre_projection.weight",
        "nextn.post_projection.weight",
    }
)

# GGUF ``blk.<N>.<stem>`` -> Qwen35MtpModel-internal parameter stem. Keys are
# the GGUF stems (without ``.weight``); values are the head module's own
# submodule paths (no ``mtp.`` prefix, so no ``preprocess_weights`` rename is
# needed). The single MTP block always maps onto the head's ``layers.0``.
_MTP_STEM_MAP: dict[str, str] = {
    # Cross-conditioning input projection.
    "nextn.enorm": "pre_fc_norm_embedding",
    "nextn.hnorm": "pre_fc_norm_hidden",
    "nextn.eh_proj": "fc",
    "nextn.embed_tokens": "embed_tokens",
    "nextn.shared_head_norm": "norm",
    "nextn.shared_head_head": "lm_head",
    # The single full-attention decoder layer.
    "attn_norm": "layers.0.input_layernorm",
    "attn_q": "layers.0.self_attn.q_proj",
    "attn_k": "layers.0.self_attn.k_proj",
    "attn_v": "layers.0.self_attn.v_proj",
    "attn_output": "layers.0.self_attn.o_proj",
    "attn_q_norm": "layers.0.self_attn.q_norm",
    "attn_k_norm": "layers.0.self_attn.k_norm",
    "post_attention_norm": "layers.0.post_attention_layernorm",
    "ffn_gate": "layers.0.mlp.gate_proj",
    "ffn_up": "layers.0.mlp.up_proj",
    "ffn_down": "layers.0.mlp.down_proj",
}
_SUPPORTED_MTP_BLOCK_TENSORS: frozenset[str] = frozenset(
    f"{stem}.weight" for stem in _MTP_STEM_MAP
)

# Head norms that are ``OffsetRMSNorm`` (``1 + weight``) but whose target name
# does not end in ``norm.weight`` — the generic offset-strip in
# ``_normalize_gguf_weights`` would miss them, so they are handled explicitly.
_MTP_EXTRA_OFFSET_NORMS: frozenset[str] = frozenset(
    {"pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight"}
)

_BLK_PATTERN = re.compile(r"^blk\.(\d+)\.(.+)$")


def mtp_architecture_capabilities() -> MappingProxyType[str, MtpArchitectureCapability]:
    """Return the immutable pinned architecture-by-MTP capability policy."""
    return _PINNED_MTP_CAPABILITIES


def validate_mtp_tensor_contract(gguf_model) -> None:
    """Reject unsupported or incomplete MTP metadata/tensors before graph construction."""
    architecture = gguf_model.architecture
    metadata = gguf_model.metadata
    tensor_names = set(gguf_model.tensor_names)
    exact_key = f"{architecture}.nextn_predict_layers"

    mtp_metadata_keys = sorted(
        key
        for key in metadata
        if isinstance(key, str)
        and (key.endswith(_MTP_METADATA_SUFFIXES) or ".mtp" in key.lower())
    )
    unexpected_metadata = [key for key in mtp_metadata_keys if key != exact_key]
    if unexpected_metadata:
        raise ValueError(
            f"{architecture} GGUF contains unsupported MTP metadata key(s): "
            f"{unexpected_metadata}; pinned llama.cpp uses only {exact_key!r}"
        )

    modern_tensors: dict[int, set[str]] = {}
    unknown_mtp_tensors: list[str] = []
    auxiliary_tensors: list[str] = []
    for name in tensor_names:
        if name in _LEGACY_NEXTN_TENSORS:
            continue
        match = _BLK_PATTERN.match(name)
        if match is not None and match.group(2).startswith("nextn."):
            block_index = int(match.group(1))
            suffix = match.group(2)
            if suffix.endswith((".scale", ".input_scale")):
                auxiliary_tensors.append(name)
                continue
            if not suffix.endswith(".weight"):
                unknown_mtp_tensors.append(name)
                continue
            stem = suffix[: -len(".weight")]
            if stem not in _MTP_STEM_MAP:
                unknown_mtp_tensors.append(name)
                continue
            modern_tensors.setdefault(block_index, set()).add(stem)
            continue
        if name.startswith(("nextn.", "mtp.")) or ".nextn." in name or ".mtp." in name:
            unknown_mtp_tensors.append(name)

    if auxiliary_tensors:
        raise ValueError(
            f"{sorted(auxiliary_tensors)}: Mobius cannot represent GGUF scale/input_scale "
            "sidecars in the MTP quantization ABI"
        )
    if unknown_mtp_tensors:
        raise ValueError(
            f"{architecture} GGUF contains unsupported nextn tensor(s) or mtp tensor(s): "
            f"{sorted(unknown_mtp_tensors)}"
        )

    legacy_tensors = sorted(tensor_names & _LEGACY_NEXTN_TENSORS)
    raw_count = metadata.get(exact_key, 0)
    if isinstance(raw_count, bool) or not isinstance(raw_count, int):
        raise TypeError(f"{exact_key} must be an integer, got {raw_count!r}")
    count = raw_count
    if count < 0:
        raise ValueError(f"{exact_key} must be non-negative, got {count}")
    if count == 0 and (modern_tensors or legacy_tensors):
        raise ValueError(
            f"{architecture} GGUF contains MTP tensors but does not declare a positive "
            f"{exact_key}"
        )
    if count == 0:
        return

    capability = _PINNED_MTP_CAPABILITIES.get(architecture)
    if capability is None:
        raise ValueError(
            f"{architecture} GGUF declares MTP, but that architecture is absent from the "
            "pinned llama.cpp MTP loader/converter census"
        )
    if capability.support is not Support.SUPPORTED:
        raise NotImplementedError(
            f"{architecture} GGUF MTP is rejected: {capability.reason}; "
            f"pinned loader behavior={capability.loader_behavior}, "
            f"block kind={capability.block_kind}"
        )
    if not (modern_tensors or legacy_tensors):
        raise ValueError(
            f"{architecture} GGUF declares {exact_key}={count} but contains no MTP tensors"
        )
    if count != 1:
        raise ValueError(
            f"{architecture} GGUF declares {exact_key}={count}, but only exactly one "
            "appended MTP block can be represented; refusing to silently truncate "
            "the remaining heads"
        )
    if legacy_tensors:
        raise ValueError(
            f"{architecture} GGUF mixes unsupported legacy global NextN tensors "
            f"{legacy_tensors} with the modern per-block namespace"
        )

    block_count_key = f"{architecture}.block_count"
    raw_block_count = metadata.get(block_count_key, 0)
    if isinstance(raw_block_count, bool) or not isinstance(raw_block_count, int):
        raise TypeError(f"{block_count_key} must be an integer, got {raw_block_count!r}")
    block_count = raw_block_count
    expected_block = block_count - 1
    if block_count <= count:
        raise ValueError(
            f"{architecture} GGUF block_count={block_count} must exceed MTP head count {count}"
        )
    if set(modern_tensors) != {expected_block}:
        raise ValueError(
            f"{architecture} GGUF MTP tensors must all belong to trailing block "
            f"{expected_block}, got blocks {sorted(modern_tensors)}"
        )
    block_prefix = f"blk.{expected_block}."
    unexpected_block_tensors = sorted(
        name
        for name in tensor_names
        if name.startswith(block_prefix)
        and name.removeprefix(block_prefix) not in _SUPPORTED_MTP_BLOCK_TENSORS
    )
    if unexpected_block_tensors:
        raise ValueError(
            f"{architecture} GGUF MTP block contains unsupported tensor(s): "
            f"{unexpected_block_tensors}"
        )

    required = {"nextn.eh_proj", "nextn.enorm", "nextn.hnorm"}
    missing = sorted(required - modern_tensors[expected_block])
    if missing:
        raise ValueError(
            f"{architecture} GGUF MTP block {expected_block} is missing required tensor "
            f"stem(s): {missing}"
        )

    hidden = int(metadata[f"{architecture}.embedding_length"])
    vocab = int(metadata.get(f"{architecture}.vocab_size", 0))
    if not vocab:
        vocab = len(metadata.get("tokenizer.ggml.tokens", ()))
    num_heads = int(metadata[f"{architecture}.attention.head_count"])
    num_kv_heads = int(metadata.get(f"{architecture}.attention.head_count_kv", num_heads))
    head_dim = int(metadata.get(f"{architecture}.attention.key_length", hidden // num_heads))
    raw_ffn = metadata[f"{architecture}.feed_forward_length"]
    if isinstance(raw_ffn, (list, tuple)):
        if len(raw_ffn) != block_count:
            raise ValueError(
                f"{architecture}.feed_forward_length must contain {block_count} entries, "
                f"got {len(raw_ffn)}"
            )
        ffn = int(raw_ffn[expected_block])
    else:
        ffn = int(raw_ffn)

    prefix = f"blk.{expected_block}."
    expected_shapes: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (vocab, hidden),
        "output_norm.weight": (hidden,),
        f"{prefix}nextn.eh_proj.weight": (hidden, 2 * hidden),
        f"{prefix}nextn.enorm.weight": (hidden,),
        f"{prefix}nextn.hnorm.weight": (hidden,),
        f"{prefix}attn_norm.weight": (hidden,),
        f"{prefix}attn_q.weight": (2 * num_heads * head_dim, hidden),
        f"{prefix}attn_k.weight": (num_kv_heads * head_dim, hidden),
        f"{prefix}attn_v.weight": (num_kv_heads * head_dim, hidden),
        f"{prefix}attn_output.weight": (hidden, num_heads * head_dim),
        f"{prefix}attn_q_norm.weight": (head_dim,),
        f"{prefix}attn_k_norm.weight": (head_dim,),
        f"{prefix}post_attention_norm.weight": (hidden,),
        f"{prefix}ffn_gate.weight": (ffn, hidden),
        f"{prefix}ffn_up.weight": (ffn, hidden),
        f"{prefix}ffn_down.weight": (hidden, ffn),
    }
    optional_shapes = {
        "output.weight": (vocab, hidden),
        f"{prefix}nextn.embed_tokens.weight": (vocab, hidden),
        f"{prefix}nextn.shared_head_norm.weight": (hidden,),
        f"{prefix}nextn.shared_head_head.weight": (vocab, hidden),
    }
    actual_shapes = {
        name: tuple(int(dimension) for dimension in shape)
        for name, _raw, _qtype, shape in gguf_model.tensor_items_raw()
        if name in expected_shapes or name in optional_shapes
    }
    missing_shapes = sorted(set(expected_shapes) - set(actual_shapes))
    malformed_shapes = {
        name: (expected_shapes.get(name, optional_shapes.get(name)), actual)
        for name, actual in actual_shapes.items()
        if actual != expected_shapes.get(name, optional_shapes.get(name))
    }
    if missing_shapes or malformed_shapes:
        raise ValueError(
            f"Invalid {architecture} GGUF MTP tensor shapes: missing={missing_shapes}, "
            f"malformed={malformed_shapes}"
        )


def map_gguf_mtp_to_hf_names(gguf_name: str, mtp_block_index: int) -> str | None:
    """Map a GGUF MTP-block tensor name to its head-module parameter name.

    Returns ``None`` for every tensor that is not part of the MTP block at
    ``mtp_block_index`` (backbone layers, embeddings, tokenizer, ...), so the
    shared GGUF state-dict loaders route only the head weights here.
    """
    match = _BLK_PATTERN.match(gguf_name)
    if match is None:
        return None
    if int(match.group(1)) != mtp_block_index:
        return None
    stem_with_suffix = match.group(2)
    # The pinned generic llama.cpp loader also creates optional quantization
    # sidecars for reachable MTP projections. Keep them visible to the common
    # pre-build validator instead of silently treating them as unknown tensors.
    for suffix in (".input_scale", ".weight", ".scale", ".bias"):
        if stem_with_suffix.endswith(suffix):
            stem, tail = stem_with_suffix[: -len(suffix)], suffix
            break
    else:
        stem, tail = stem_with_suffix, ""
    target = _MTP_STEM_MAP.get(stem)
    if target is None:
        return None
    return target + tail


def has_mtp_head(config) -> bool:
    """Return ``True`` when *config* carries a trailing GGUF MTP block."""
    return bool(getattr(config, "_gguf_mtp_block_indices", None))


def derive_mtp_config(
    config,
    *,
    use_dedicated_embeddings: bool = False,
    use_dedicated_lm_head: bool = False,
):
    """Derive a :class:`Qwen35MtpConfig` from the resolved backbone *config*.

    All dimensions (hidden size, head dim, KV heads, rope, quantization,
    dtype, ...) are inherited from the backbone so the reused
    :class:`Qwen35DecoderLayer` stays bit-compatible with the target's
    full-attention layers.  Only the head-specific overrides — a single
    ``full_attention`` layer — are forced.
    """
    from mobius._configs import Qwen35MtpConfig

    accepted = {f.name for f in dataclasses.fields(Qwen35MtpConfig)}
    fields = {
        f.name: getattr(config, f.name)
        for f in dataclasses.fields(config)
        if f.name in accepted
    }
    fields["num_hidden_layers"] = 1
    fields["layer_types"] = ["full_attention"]
    fields["output_layer_indices"] = None
    fields["output_final_hidden_state"] = False
    fields["use_dedicated_embeddings"] = use_dedicated_embeddings
    fields["use_dedicated_lm_head"] = use_dedicated_lm_head
    mtp_config = Qwen35MtpConfig(**fields)
    # Preserve the model_type so tensor processors / quantization dispatch the
    # same way as the backbone.
    mtp_config.model_type = getattr(config, "model_type", None)
    mtp_config._gguf_model_type = getattr(config, "_gguf_model_type", None)
    mtp_config._gguf_arch = getattr(config, "_gguf_arch", None)
    return mtp_config


def _strip_mtp_offset_norms(state_dict: dict, gguf_arch: str) -> dict:
    """Subtract the baked-in ``+1`` from head norms missed by the generic pass.

    ``_normalize_gguf_weights`` already strips the offset from every tensor
    whose name ends in ``norm.weight`` (``norm``, ``input_layernorm``,
    ``q_norm``, ...).  The two ``pre_fc_norm_*`` tensors end in
    ``embedding.weight`` / ``hidden.weight`` and are handled here so all three
    :class:`OffsetRMSNorm` inputs are treated consistently.
    """
    from mobius.integrations.gguf._builder import _OFFSET_NORM_GGUF_ARCHS

    if gguf_arch not in _OFFSET_NORM_GGUF_ARCHS:
        return state_dict
    for key in _MTP_EXTRA_OFFSET_NORMS:
        if key in state_dict:
            state_dict[key] = state_dict[key] - 1.0
    return state_dict


def build_mtp_head_from_gguf(
    gguf_model,
    config,
    *,
    preserve_quantization: bool,
    execution_provider: str = "default",
    on_preflight: Callable[[GGUFQuantizationReport], None] | None = None,
) -> ModelPackage | None:
    """Build the Qwen3.5/3.8 MTP head sidecar :class:`ModelPackage` from GGUF.

    Returns ``None`` when the GGUF has no MTP block.  The returned package has
    a single ``"model"`` component and is intended to be saved into a ``mtp/``
    subdirectory alongside the backbone.
    """
    from mobius._builder import build_from_module
    from mobius.integrations.gguf._builder import (
        _is_native_block_weight,
        _is_quantized_weight,
        _load_dequantized_state_dict,
        _load_quantized_state_dict,
        _normalize_gguf_weights,
        _preflight_quantization_report,
        _replace_native_block_linears,
    )
    from mobius.integrations.gguf._tensor_processors import process_tensors
    from mobius.models.qwen35_mtp import Qwen35MtpModel
    from mobius.tasks import Qwen35MtpTask

    mtp_blocks = getattr(config, "_gguf_mtp_block_indices", None)
    if not mtp_blocks:
        return None
    validate_mtp_tensor_contract(gguf_model)
    if len(mtp_blocks) != 1:
        raise ValueError(
            f"GGUF declares {len(mtp_blocks)} MTP blocks, but ModelPackage can "
            "represent exactly one MTP sidecar head. Multi-head MTP export is not "
            f"supported; got blocks {list(mtp_blocks)} and produced no partial sidecar."
        )
    mtp_block_index = int(mtp_blocks[0])
    gguf_arch = gguf_model.architecture
    tensor_names = set(gguf_model.tensor_names)
    mtp_prefix = f"blk.{mtp_block_index}.nextn."
    use_dedicated_embeddings = f"{mtp_prefix}embed_tokens.weight" in tensor_names
    use_dedicated_norm = f"{mtp_prefix}shared_head_norm.weight" in tensor_names
    use_dedicated_lm_head = f"{mtp_prefix}shared_head_head.weight" in tensor_names

    mtp_config = derive_mtp_config(
        config,
        use_dedicated_embeddings=use_dedicated_embeddings,
        use_dedicated_lm_head=use_dedicated_lm_head,
    )
    module = Qwen35MtpModel(mtp_config)

    def _mapper(gguf_name: str, _arch: str) -> str | None:
        if not use_dedicated_norm and gguf_name == "output_norm.weight":
            return "norm.weight"
        return map_gguf_mtp_to_hf_names(gguf_name, mtp_block_index)

    if preserve_quantization:
        _replace_native_block_linears(
            module,
            gguf_model,
            gguf_arch,
            name_mapper=_mapper,
        )
    quantization_report = _preflight_quantization_report(
        gguf_model,
        gguf_arch,
        module,
        mtp_config,
        preserve_quantization=preserve_quantization,
        target_bits=(mtp_config.quantization.bits if preserve_quantization else None),
        target_block_size=(
            mtp_config.quantization.group_size if preserve_quantization else None
        ),
        execution_provider=execution_provider,
        name_mapper=_mapper,
        dequantize_float_linear_types={"lm_head": {"Q4_1"}},
        emit_warning=False,
    )
    if on_preflight is not None:
        on_preflight(quantization_report)
    pkg = build_from_module(
        module, mtp_config, Qwen35MtpTask(), execution_provider=execution_provider
    )
    pkg.gguf_quantization_report = quantization_report

    if preserve_quantization:
        state_dict = _load_quantized_state_dict(
            gguf_model,
            gguf_arch,
            module,
            mtp_config,
            name_mapper=_mapper,
            warn_unmapped=False,
            dequantize_float_linear_types={"lm_head": {"Q4_1"}},
        )
        # Mirror the backbone (build_from_gguf steps 7/7b): run process_tensors
        # over only the float tensors (native quant blocks were already
        # permuted during load), then normalize the full merged dict.
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
        float_dict = process_tensors(float_dict, mtp_config)
        state_dict = {**float_dict, **quant_dict}
    else:
        state_dict = _load_dequantized_state_dict(
            gguf_model,
            gguf_arch,
            name_mapper=_mapper,
            warn_unmapped=False,
        )
        state_dict = process_tensors(state_dict, mtp_config)

    state_dict = _normalize_gguf_weights(state_dict, gguf_arch, mtp_config)
    state_dict = _strip_mtp_offset_norms(state_dict, gguf_arch)

    if hasattr(module, "preprocess_weights"):
        # NOTE: intentionally NOT calling ``module.preprocess_weights`` here.
        # ``Qwen35MtpModel.preprocess_weights`` strips a ``mtp.`` prefix and
        # drops everything else; our mapper already emits the final
        # head-internal names (``fc.weight``, ``layers.0.*``), so running it
        # would drop the whole state dict.
        pass

    prefix_map = getattr(module, "weight_prefix_map", None)
    pkg.apply_weights(state_dict, prefix_map=prefix_map)
    logger.info(
        "Built Qwen3.5/3.8 MTP head sidecar from GGUF block %d (%d tensors).",
        mtp_block_index,
        len(state_dict),
    )
    return pkg
