# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for ModelOpt NVFP4 / FP8 weight reconstruction."""

from __future__ import annotations

import ml_dtypes
import numpy as np

from mobius.integrations.modelopt import (
    FP4_E2M1_LUT,
    dequantize_fp8,
    dequantize_nvfp4,
    is_modelopt_quant_config,
    unpack_nvfp4_codes,
)


def _e2m1_code(sign: int, mag_index: int) -> int:
    """Build a 4-bit E2M1 code from a sign bit and a magnitude index (0..7)."""
    return (sign << 3) | (mag_index & 0x7)


def test_fp4_lut_values():
    assert FP4_E2M1_LUT.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def test_unpack_nvfp4_codes_splits_nibbles():
    # byte = low | (high << 4); low nibble is the even-K code.
    packed = np.array([[0x21, 0xC6]], dtype=np.uint8)  # (1, 0x21=33, 0xC6=198)
    codes = unpack_nvfp4_codes(packed)
    # 0x21 -> low=1, high=2 ; 0xC6 -> low=6, high=12
    assert codes.tolist() == [[1, 2, 6, 12]]
    assert codes.dtype == np.uint8


def test_dequantize_nvfp4_known_values():
    # K=4 codes: k0=+1.0, k1=-0.5, k2=+4.0, k3=-2.0
    k0 = _e2m1_code(0, 2)  # mag 1.0
    k1 = _e2m1_code(1, 1)  # -0.5
    k2 = _e2m1_code(0, 6)  # 4.0
    k3 = _e2m1_code(1, 4)  # -2.0
    byte0 = k0 | (k1 << 4)
    byte1 = k2 | (k3 << 4)
    weight_u8 = np.array([[byte0, byte1]], dtype=np.uint8)

    block_scale = np.array([[2.0]], dtype=ml_dtypes.float8_e4m3fn)  # one block
    global_scale = 0.5

    out = dequantize_nvfp4(weight_u8, block_scale, global_scale)
    assert out.dtype == ml_dtypes.bfloat16
    # val * 2.0 (block) * 0.5 (global) == val
    expected = np.array([[1.0, -0.5, 4.0, -2.0]], dtype=np.float32)
    np.testing.assert_array_equal(out.astype(np.float32), expected)


def test_dequantize_nvfp4_block_scale_repeat():
    # Two 16-element blocks with distinct block scales exercise repeat_interleave.
    n, k = 1, 32
    # All codes = +1.0 (index 2) -> byte 0x22 packs two such codes.
    weight_u8 = np.full((n, k // 2), 0x22, dtype=np.uint8)
    block_scale = np.array([[1.0, 4.0]], dtype=ml_dtypes.float8_e4m3fn)
    out = dequantize_nvfp4(weight_u8, block_scale, 1.0).astype(np.float32)
    assert out.shape == (1, 32)
    # First 16 elements scaled by 1.0, next 16 by 4.0.
    np.testing.assert_array_equal(out[0, :16], np.ones(16, dtype=np.float32))
    np.testing.assert_array_equal(out[0, 16:], np.full(16, 4.0, dtype=np.float32))


def test_dequantize_nvfp4_raw_uint8_block_scale():
    # Block scales may arrive as raw uint8 e4m3 code bytes; result must match a
    # typed float8 view.
    k0 = _e2m1_code(0, 2)
    weight_u8 = np.array([[k0 | (k0 << 4)]], dtype=np.uint8)  # K=2, both +1.0
    typed = np.array([[3.0]], dtype=ml_dtypes.float8_e4m3fn)
    raw = typed.view(np.uint8)
    a = dequantize_nvfp4(weight_u8, typed, 1.0).astype(np.float32)
    b = dequantize_nvfp4(weight_u8, raw, 1.0).astype(np.float32)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, np.array([[3.0, 3.0]], dtype=np.float32))


def test_dequantize_fp8_per_tensor_scale():
    weight = np.array([[1.0, -2.0, 0.5, 3.0]], dtype=ml_dtypes.float8_e4m3fn)
    out = dequantize_fp8(weight, 0.25)
    assert out.dtype == ml_dtypes.bfloat16
    expected = np.array([[0.25, -0.5, 0.125, 0.75]], dtype=np.float32)
    np.testing.assert_array_equal(out.astype(np.float32), expected)


def test_is_modelopt_quant_config():
    assert is_modelopt_quant_config({"quant_method": "modelopt"})
    assert is_modelopt_quant_config({"quant_algo": "NVFP4"})
    assert is_modelopt_quant_config({"quant_cfg": "W4A16_NVFP4"})
    assert is_modelopt_quant_config({"quant_algo": "FP8"})
    # Non-ModelOpt schemes and empty configs are rejected.
    assert not is_modelopt_quant_config({"quant_method": "gptq"})
    assert not is_modelopt_quant_config({"quant_method": "awq"})
    assert not is_modelopt_quant_config({})
    assert not is_modelopt_quant_config(None)
