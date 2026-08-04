# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""NVFP4 / FP8 weight reconstruction for NVIDIA ModelOpt checkpoints.

NVIDIA TensorRT Model Optimizer (ModelOpt) exports mixed-precision checkpoints
where different modules use different numeric formats. For Qwen3.6 the layout is:

- **Routed MoE experts, shared expert, lm_head**: ``W4A16_NVFP4`` — block-16
  E2M1 (fp4) 4-bit weights, FP8-E4M3 per-block scales, and a per-tensor FP32
  global scale stored as ``weight_scale_2``.
- **Attention / linear-attention projections**: ``FP8`` (E4M3) — float8 weights
  with a per-tensor ``weight_scale``.

ONNX Runtime has no FP8 attention GEMM, so — following ORT GenAI's ModelOpt
loader — the dense FP8 projections and the NVFP4 shared-expert / lm_head are
**dequantized back to BF16**, which reconstructs them exactly and lets the
standard (BF16) build path consume the checkpoint. Only the *routed* MoE experts
are consumed natively by the CUDA QMoE ``quant_type="nvfp4"`` op; that native
emission is Blackwell/``onnxruntime_USE_FP4_QMOE=ON``-only and is intentionally
NOT implemented here (it cannot be built or verified without that runtime).

The functions below are the numeric core of the loader and are format-faithful
to ModelOpt's on-disk encoding (verified in ``_dequant_test.py``).

E2M1 (fp4) magnitude table, indexed by the low 3 bits of each 4-bit code
(bit 3 is the sign): ``{0, 0.5, 1, 1.5, 2, 3, 4, 6}``.
"""

from __future__ import annotations

import ml_dtypes
import numpy as np

# E2M1 (fp4) decoded magnitudes, indexed by ``code & 0x7`` (bit 3 is the sign).
FP4_E2M1_LUT = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)

# NVFP4 pins the weight block size to 16 K-elements per FP8-E4M3 block scale.
NVFP4_BLOCK_SIZE = 16


def unpack_nvfp4_codes(packed_nk2: np.ndarray) -> np.ndarray:
    """Unpack a ModelOpt NVFP4 weight tensor to per-element E2M1 codes.

    ``packed_nk2`` is uint8 ``[N, K/2]`` where each byte holds two adjacent
    K-axis E2M1 codes for the same output row ``N`` (low nibble = even ``K``,
    high nibble = odd ``K``) — the layout ModelOpt writes. Returns uint8 codes
    ``[N, K]`` in ``0..15``.

    Raises:
        ValueError: if ``packed_nk2`` is not a 2D uint8 ``[N, K/2]`` array.
            Loader code should surface an upstream shape/dtype mistake rather than
            silently reinterpreting it.
    """
    packed = np.asarray(packed_nk2)
    if packed.ndim != 2:
        raise ValueError(
            f"NVFP4 packed codes must be 2D [N, K/2], got shape {packed.shape}."
        )
    if packed.dtype != np.uint8:
        raise ValueError(f"NVFP4 packed codes must be uint8, got dtype {packed.dtype}.")
    packed = np.ascontiguousarray(packed)
    low = packed & 0x0F
    high = packed >> 4
    n, k2 = packed.shape
    # Interleave low/high nibbles (even-K / odd-K) to produce [N, K].
    codes = np.empty((n, k2 * 2), dtype=np.uint8)
    codes[:, 0::2] = low
    codes[:, 1::2] = high
    return np.ascontiguousarray(codes)


def dequantize_nvfp4(
    weight_u8: np.ndarray,
    block_scale_e4m3: np.ndarray,
    global_scale: float | np.floating,
) -> np.ndarray:
    """Reconstruct a BF16 weight from ModelOpt NVFP4 tensors.

    Args:
        weight_u8: uint8 ``[N, K/2]`` packed E2M1 codes (low nibble = even
            ``K``, high nibble = odd ``K``).
        block_scale_e4m3: FP8-E4M3 ``[N, K/16]`` per-block scales
            (``ml_dtypes.float8_e4m3fn`` or a uint8 raw-code view).
        global_scale: per-tensor FP32 scalar.

    Returns:
        ``ml_dtypes.bfloat16`` array ``[N, K]`` where
        ``w = e2m1(code) * e4m3(block_scale[n, k // 16]) * global_scale``.
    """
    codes = unpack_nvfp4_codes(weight_u8).astype(np.int64)  # [N, K]
    mag = FP4_E2M1_LUT[codes & 0x7]  # [N, K]
    val = np.where((codes & 0x8) > 0, -mag, mag).astype(np.float32)  # [N, K]

    block_scale = _to_float32(block_scale_e4m3)  # [N, K/16]
    k = codes.shape[1]
    n_blocks = block_scale.shape[1]
    if n_blocks == 0 or k % n_blocks != 0:
        raise ValueError(f"NVFP4 K={k} is not divisible by the block count {n_blocks}.")
    block_size = k // n_blocks
    if block_size != NVFP4_BLOCK_SIZE:
        # NVFP4 pins the block size to 16; a different derived size means the
        # weight/scale shapes are mismatched (silently-wrong reconstruction).
        raise ValueError(
            f"NVFP4 block size must be {NVFP4_BLOCK_SIZE}, got {block_size} "
            f"(K={k}, block scales={n_blocks})."
        )
    block_scale = np.repeat(block_scale, block_size, axis=1)  # [N, K]

    dequant = val * block_scale * np.float32(global_scale)
    return dequant.astype(ml_dtypes.bfloat16)


def dequantize_fp8(
    weight_f8: np.ndarray,
    weight_scale: float | np.floating,
) -> np.ndarray:
    """Reconstruct a BF16 weight from an FP8 (E4M3) weight + per-tensor scale.

    ``w = e4m3(weight) * weight_scale``.
    """
    weight = _to_float32(weight_f8)
    return (weight * np.float32(weight_scale)).astype(ml_dtypes.bfloat16)


def is_modelopt_quant_config(quantization_config: dict | None) -> bool:
    """Return ``True`` if a HF ``quantization_config`` is a ModelOpt export.

    ModelOpt writes ``quant_method="modelopt"`` (newer HF serialisations) or a
    ``quant_algo`` / ``quant_cfg`` naming an ``NVFP4`` / ``FP8`` scheme. This
    keeps detection tolerant of both spellings.
    """
    if not isinstance(quantization_config, dict):
        return False
    method = str(quantization_config.get("quant_method", "")).lower()
    if method == "modelopt":
        return True
    algo = str(
        quantization_config.get("quant_algo") or quantization_config.get("quant_cfg") or ""
    ).upper()
    return "NVFP4" in algo or "FP8" in algo


def _to_float32(arr: np.ndarray) -> np.ndarray:
    """View/convert an FP8-E4M3 (or raw uint8 code) tensor as float32.

    Raw ``uint8`` inputs are reinterpreted as ``float8_e4m3fn`` byte codes
    before widening, matching how ModelOpt stores block scales.
    """
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        arr = arr.view(ml_dtypes.float8_e4m3fn)
    return arr.astype(np.float32)
