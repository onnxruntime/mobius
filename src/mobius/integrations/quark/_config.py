# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Parsing for AMD Quark ``quantization_config`` blocks.

Quark's schema differs from the GPTQ/AWQ one in both nesting and naming: the
weight layout lives under ``global_quant_config.weight`` and the non-quantized
modules under ``exclude``. Letting a Quark config fall through to the generic
affine reader would silently substitute defaults (``group_size=128`` instead of
32, ``sym=True`` instead of ``False``, no exclusions) and mis-dequantize every
weight without raising, so this module parses it explicitly and rejects any
layout it cannot honour.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mobius._configs._quantization import QuantizationConfig

# Quark spells its weight width as a dtype string rather than a bit count.
# Only ``uint4`` is mapped: it is the one layout whose packing this path has
# been verified against (8 codes per int32, AWQ-GEMM interleaved nibble order).
# ``int4`` stores sign-extended codes and ``int8``/``uint8`` are not int32-packed
# at all, so each would need its own unpack and zero-point convention.
_QUARK_DTYPE_BITS: dict[str, int] = {
    "uint4": 4,
}


def is_quark_quant_config(qc: Mapping) -> bool:
    """Whether a ``quantization_config`` mapping was produced by AMD Quark."""
    return isinstance(qc, Mapping) and qc.get("quant_method") == "quark"


def parse_quark_config(qc: Mapping) -> QuantizationConfig:
    """Parse an AMD Quark ``quantization_config`` into mobius's flat config.

    Anything this affine INT4 path cannot honour is rejected rather than
    approximated, because a wrong ``group_size`` or ``sym`` produces
    plausible-looking but numerically wrong weights instead of an error.
    """
    # Imported here rather than at module scope: ``_quantization`` dispatches
    # into this module from ``QuantizationConfig.from_value``, so a top-level
    # import would close the cycle.
    from mobius._configs._quantization import QuantizationConfig, _compile_pattern

    weight = qc.get("global_quant_config", {}).get("weight")
    if weight is None:
        raise ValueError(
            "Quark quantization_config is missing global_quant_config.weight; "
            "cannot determine the weight layout."
        )
    if not isinstance(weight, Mapping):
        raise TypeError(
            "Quark global_quant_config.weight must be a mapping, got "
            f"{type(weight).__name__}."
        )

    dtype = weight.get("dtype")
    bits = _QUARK_DTYPE_BITS.get(dtype)
    if bits is None:
        raise NotImplementedError(
            f"Unsupported Quark weight dtype {dtype!r}. This path handles "
            f"integer affine weights only: {sorted(_QUARK_DTYPE_BITS)}."
        )

    qscheme = weight.get("qscheme")
    if qscheme != "per_group":
        raise NotImplementedError(
            f"Unsupported Quark qscheme {qscheme!r}; only 'per_group' is "
            "implemented. Per-tensor/per-channel Quark weights would need a "
            "different scale/zero-point broadcast."
        )

    group_size = weight.get("group_size")
    if not isinstance(group_size, int) or group_size <= 0:
        raise ValueError(
            f"Quark global_quant_config.weight.group_size must be a positive "
            f"int for per_group quantization, got {group_size!r}."
        )

    if weight.get("is_dynamic"):
        raise NotImplementedError(
            "Dynamic Quark weight quantization is not supported; weights must "
            "carry static scales and zero points to become ONNX initializers."
        )

    # Per-layer overrides would need to map Quark module paths onto mobius's
    # component/override model. Reject rather than silently applying the
    # global layout to layers the quantizer treated differently.
    for field in ("layer_quant_config", "layer_type_quant_config"):
        if qc.get(field):
            raise NotImplementedError(
                f"Quark {field} is not supported; this path applies one global "
                "weight layout to every quantized projection."
            )

    raw_exclusions = qc.get("exclude") or ()
    if not isinstance(raw_exclusions, (list, tuple)):
        raise TypeError("Quark quantization_config.exclude must be a list or tuple")
    exclusions = tuple(str(pattern) for pattern in raw_exclusions)
    for pattern in exclusions:
        if pattern.startswith("re:"):
            _compile_pattern(pattern[3:])

    # Quark signals float modules via ``exclude``; a module absent from
    # that list is quantized. The AWQ/GPTQ parser reads explicit
    # ``lm_head``/``embeds`` keys, which Quark never writes -- without this
    # the flag silently defaults False and the graph builds a dense float
    # head while the checkpoint supplies packed MatMulNBits weights.
    # NOTE: quantize_embeddings is deliberately NOT derived the same way --
    # Quark ships embed_tokens as plain BF16 with no packed sidecars.
    lm_head_excluded = any(
        QuantizationConfig._matches_module(pattern, "lm_head") for pattern in exclusions
    )
    return QuantizationConfig(
        bits=bits,
        group_size=group_size,
        quant_method="quark",
        sym=bool(weight.get("symmetric", False)),
        modules_to_not_convert=exclusions,
        quantize_lm_head=not lm_head_excluded,
    )
