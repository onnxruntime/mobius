# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Pass that converts a ``GroupQueryAttention`` KV cache to FP8 (E4M3).

``com.microsoft::GroupQueryAttention`` can store its past/present key-value
cache as ``FLOAT8E4M3FN`` instead of the model dtype (fp16/bf16), halving the
KV-cache memory footprint at long context. The kernel keeps the ``query`` /
``key`` / ``value`` node inputs at the model dtype and quantizes the new K/V to
FP8 internally on write, dequantizing on read, using a per-tensor scale.

This pass runs **after** the GQA fusion rules (see
:func:`~mobius._optimizations.optimize_model`). For every decoder
``GroupQueryAttention`` node whose ``past_key`` / ``past_value`` inputs are
graph inputs it:

1. Retypes the ``past_key`` / ``past_value`` inputs and the
   ``present_key`` / ``present_value`` outputs (all graph I/O) to
   ``FLOAT8E4M3FN`` while preserving their shapes.
2. Adds ``k_scale`` / ``v_scale`` scalar FLOAT initializers (input slots 12/13).
   Scales default to ``1.0`` (the "legacy" export shape) or are taken from a
   caller-provided per-layer calibration map so out-of-range K/V do not
   saturate the ~±448 FP8-E4M3 range.
3. Sets the quantization attributes ``k_quant_type`` / ``v_quant_type`` to
   ``"PER_TENSOR"`` and ``kv_cache_bit_width`` to ``8``.

The GQA op signature this targets (ORT >= 1.28) is::

    query, key, value, past_key, past_value, seqlens_k, total_sequence_length,
    cos_cache, sin_cache, position_ids, attention_bias, head_sink,
    k_scale, v_scale, q_norm_weight, k_norm_weight

so the ``k_scale`` / ``v_scale`` inputs live at positions 12 and 13. mobius
emits 9 inputs (through ``sin_cache``); this pass grows the input list to 14,
leaving the optional ``position_ids`` / ``attention_bias`` / ``head_sink``
slots (9/10/11) empty.

Requires an ORT build with the FP8 KV-cache GQA kernel (SM89+ CUDA, e.g. Ada /
Hopper / Blackwell). The FP8 KV cache is IO-bound at runtime as device
``FLOAT8E4M3FN`` OrtValues (as onnxruntime-genai does), not fed as numpy.
"""

from __future__ import annotations

import json
import logging
import math
import re
import warnings

import numpy as np
import onnx_ir as ir

logger = logging.getLogger(__name__)

_FP8 = ir.DataType.FLOAT8E4M3FN

# GQA (ORT >= 1.28) optional KV-scale input slots.
_K_SCALE_INDEX = 12
_V_SCALE_INDEX = 13
_MIN_INPUTS = _V_SCALE_INDEX + 1  # 14: grow to expose both scale slots.

# Parses the layer index ``i`` from a KV-cache input named
# ``past_key_values.{i}.key`` (the name minted by ``_make_kv_cache_inputs``).
_LAYER_ID_RE = re.compile(r"\.(\d+)\.")


def _layer_id_from_name(name: str | None) -> int | None:
    """Return the layer index encoded in a ``past_key_values.{i}.key`` name."""
    if not name:
        return None
    match = _LAYER_ID_RE.search(name)
    return int(match.group(1)) if match else None


def _retype_fp8(value: ir.Value | None) -> None:
    """Retype *value* to ``FLOAT8E4M3FN`` in place, preserving its shape."""
    if value is None:
        return
    value.type = ir.TensorType(_FP8)


def _is_retypable_cache(value: ir.Value) -> bool:
    """Whether *value* may be safely retyped to FP8.

    A KV cache is retypable when it is a graph input (no ``const_value``) or an
    empty placeholder initializer. A non-empty FLOAT/FLOAT16 initializer must
    NOT be retyped: declaring FP8 over FLOAT bytes would corrupt the graph.
    """
    const = value.const_value
    if const is None:
        return True
    return const.size == 0


class Fp8KvCachePass(ir.passes.InPlacePass):
    """Convert decoder ``GroupQueryAttention`` KV caches to FP8-E4M3.

    Args:
        scales: Optional mapping ``layer_id -> (k_scale, v_scale)`` of
            per-tensor FP8 scales (typically produced by an offline
            calibration). Layers absent from the map — and every layer when
            *scales* is ``None`` — use a unit scale of ``1.0``.
    """

    def __init__(self, scales: dict[int, tuple[float, float]] | None = None) -> None:
        super().__init__()
        self._scales = scales or {}

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        graph = model.graph
        modified = False
        converted = 0

        for node in graph:
            if node.domain != "com.microsoft" or node.op_type != "GroupQueryAttention":
                continue

            inputs = node.inputs
            past_key = inputs[3] if len(inputs) > 4 else None
            past_value = inputs[4] if len(inputs) > 4 else None
            if past_key is None or past_value is None:
                # Prompt-processing GQA without a KV cache — nothing to convert.
                continue

            # Contract: only KV caches that are graph inputs (or empty
            # placeholders) may be retyped. Retyping a non-empty FLOAT/FLOAT16
            # initializer would declare FP8 over FLOAT bytes and corrupt the
            # graph, so skip such nodes loudly rather than emit an invalid model.
            if not _is_retypable_cache(past_key) or not _is_retypable_cache(past_value):
                warnings.warn(
                    f"Fp8KvCachePass: skipping {node.name!r} — its KV cache is a "
                    f"non-empty initializer, not a graph input. Only graph-input "
                    f"KV caches can be converted to FP8.",
                    stacklevel=2,
                )
                continue

            # No early skip when past_key is already FP8: every step below is
            # idempotent (retype is a no-op, scale initializers dedup by name,
            # attributes overwrite, inputs only grow when < 14), so a re-run
            # cannot leave a node in a half-converted state.

            layer_id = _layer_id_from_name(past_key.name)
            k_val, v_val = self._scales.get(
                layer_id if layer_id is not None else -1, (1.0, 1.0)
            )

            # (1) Retype the cache graph I/O: past inputs + present outputs.
            # Guard each present output on its own index so a hypothetical
            # 2-output GQA variant cannot leave present_key (index 1) as a
            # different dtype from past_key.
            _retype_fp8(past_key)
            _retype_fp8(past_value)
            present_key = node.outputs[1] if len(node.outputs) > 1 else None
            present_value = node.outputs[2] if len(node.outputs) > 2 else None
            _retype_fp8(present_key)
            _retype_fp8(present_value)

            # (2) Scale initializers (unit 1.0 unless calibrated). Names are
            # derived from the cache tensor names — not just the numeric layer
            # id — so distinct caches at the same layer index (e.g. seq2seq
            # self- vs cross-attention) never share a scale initializer.
            k_scale = self._scale_initializer(graph, f"{past_key.name}.fp8_scale", k_val)
            v_scale = self._scale_initializer(graph, f"{past_value.name}.fp8_scale", v_val)

            if len(node.inputs) < _MIN_INPUTS:
                node.resize_inputs(_MIN_INPUTS)
            node.replace_input_with(_K_SCALE_INDEX, k_scale)
            node.replace_input_with(_V_SCALE_INDEX, v_scale)

            # (3) Quantization attributes.
            node.attributes.add(ir.AttrString("k_quant_type", "PER_TENSOR"))
            node.attributes.add(ir.AttrString("v_quant_type", "PER_TENSOR"))
            node.attributes.add(ir.AttrInt64("kv_cache_bit_width", 8))

            modified = True
            converted += 1

        if converted:
            logger.info(
                "Fp8KvCachePass: converted %d GroupQueryAttention KV cache(s) to FP8",
                converted,
            )
        return ir.passes.PassResult(model, modified=modified)

    @staticmethod
    def _scale_initializer(graph: ir.Graph, name: str, value: float) -> ir.Value:
        """Return (creating if needed) a scalar FLOAT scale initializer."""
        existing = graph.initializers.get(name)
        if existing is not None:
            return existing
        tensor = ir.tensor(np.array([value], dtype=np.float32), name=name)
        scale = ir.Value(name=name, type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1]))
        scale.const_value = tensor
        graph.initializers[name] = scale
        return scale


def load_kv_cache_scale_file(path: str) -> dict[int, tuple[float, float]]:
    """Load per-layer FP8 KV-cache scales from a calibration JSON file.

    Accepts the onnxruntime-genai calibration format::

        {"scales": {"k_scales": [s0, s1, ...], "v_scales": [s0, s1, ...]}}

    The lists are indexed positionally: entry ``i`` maps to layer id ``i``
    (matching the ``past_key_values.{i}`` naming). ``k_scales`` and
    ``v_scales`` must have equal length.

    Args:
        path: Path to the calibration JSON file.

    Returns:
        A ``layer_id -> (k_scale, v_scale)`` mapping.

    Raises:
        ValueError: If the file lacks ``scales.k_scales`` / ``scales.v_scales``
            or the two lists differ in length.
    """
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    try:
        k_scales = data["scales"]["k_scales"]
        v_scales = data["scales"]["v_scales"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"{path!r} must contain scales.k_scales and scales.v_scales."
        ) from error
    if not isinstance(k_scales, list) or not isinstance(v_scales, list):
        raise ValueError(  # noqa: TRY004 — malformed calibration file is a value error, not a type error
            f"{path!r}: scales.k_scales and scales.v_scales must be JSON arrays."
        )
    if len(k_scales) != len(v_scales):
        raise ValueError(
            f"{path!r}: k_scales and v_scales must have equal length "
            f"(got k={len(k_scales)}, v={len(v_scales)})."
        )
    scales: dict[int, tuple[float, float]] = {}
    for i, (k, v) in enumerate(zip(k_scales, v_scales)):
        k_f, v_f = float(k), float(v)
        if not (math.isfinite(k_f) and k_f > 0.0 and math.isfinite(v_f) and v_f > 0.0):
            raise ValueError(
                f"{path!r}: layer {i} has a non-positive or non-finite scale "
                f"(k={k}, v={v}); FP8 KV-cache scales must be finite and > 0."
            )
        scales[i] = (k_f, v_f)
    return scales
