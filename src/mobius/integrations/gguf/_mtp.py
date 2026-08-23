# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Emit the Qwen3.5/3.8 multi-token-prediction (MTP) self-speculative head from GGUF.

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
    blk.<N>.nextn.shared_head_norm -> norm                     (OffsetRMSNorm)
    blk.<N>.attn_norm              -> layers.0.input_layernorm
    blk.<N>.attn_q/k/v/output      -> layers.0.self_attn.{q,k,v,o}_proj
    blk.<N>.attn_q_norm/k_norm     -> layers.0.self_attn.{q,k}_norm
    blk.<N>.post_attention_norm    -> layers.0.post_attention_layernorm
    blk.<N>.ffn_gate/up/down       -> layers.0.mlp.{gate,up,down}_proj

The head borrows the target's shared ``embed_tokens`` / ``lm_head`` (it has
none of its own), so the orchestrator embeds the drafted token, runs the
head, and decodes ``mtp_hidden`` through the target LM head.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mobius._model_package import ModelPackage

logger = logging.getLogger(__name__)

# GGUF ``blk.<N>.<stem>`` -> Qwen35MtpModel-internal parameter stem. Keys are
# the GGUF stems (without ``.weight``); values are the head module's own
# submodule paths (no ``mtp.`` prefix, so no ``preprocess_weights`` rename is
# needed). The single MTP block always maps onto the head's ``layers.0``.
_MTP_STEM_MAP: dict[str, str] = {
    # Cross-conditioning input projection.
    "nextn.enorm": "pre_fc_norm_embedding",
    "nextn.hnorm": "pre_fc_norm_hidden",
    "nextn.eh_proj": "fc",
    "nextn.shared_head_norm": "norm",
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

# Head norms that are ``OffsetRMSNorm`` (``1 + weight``) but whose target name
# does not end in ``norm.weight`` — the generic offset-strip in
# ``_normalize_gguf_weights`` would miss them, so they are handled explicitly.
_MTP_EXTRA_OFFSET_NORMS: frozenset[str] = frozenset(
    {"pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight"}
)

_BLK_PATTERN = re.compile(r"^blk\.(\d+)\.(.+)$")


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
    for suffix in (".weight", ".bias"):
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


def derive_mtp_config(config):
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
    )
    from mobius.integrations.gguf._tensor_processors import process_tensors
    from mobius.models.qwen35_mtp import Qwen35MtpModel
    from mobius.tasks import Qwen35MtpTask

    mtp_blocks = getattr(config, "_gguf_mtp_block_indices", None)
    if not mtp_blocks:
        return None
    if len(mtp_blocks) > 1:
        logger.warning(
            "GGUF declares %d MTP blocks; only the first (block %d) is exported "
            "as the self-speculative head.",
            len(mtp_blocks),
            mtp_blocks[0],
        )
    mtp_block_index = int(mtp_blocks[0])
    gguf_arch = gguf_model.architecture

    mtp_config = derive_mtp_config(config)
    module = Qwen35MtpModel(mtp_config)
    pkg = build_from_module(
        module, mtp_config, Qwen35MtpTask(), execution_provider=execution_provider
    )

    def _mapper(gguf_name: str, _arch: str) -> str | None:
        return map_gguf_mtp_to_hf_names(gguf_name, mtp_block_index)

    if preserve_quantization:
        state_dict = _load_quantized_state_dict(
            gguf_model,
            gguf_arch,
            module,
            mtp_config,
            name_mapper=_mapper,
            warn_unmapped=False,
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
