# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for AMD Quark weight unpacking."""

from __future__ import annotations

import pytest
import torch

from mobius._configs import QuantizationConfig
from mobius._weight_utils import preprocess_quantized_weights
from mobius.integrations.quark import preprocess_quark_weights

def _quark_pack(codes: torch.Tensor) -> torch.Tensor:
    """Pack ``[R, C]`` uint4 codes the way AMD Quark does.

    Mirrors ``quark.torch.utils.pack.Pack_4_bits.pack(reorder=True)``: 8 codes
    per int32, slot ``i`` holding logical index ``[0, 2, 4, 6, 1, 3, 5, 7][i]``.
    """
    order = [0, 2, 4, 6, 1, 3, 5, 7]
    rows, cols = codes.shape
    packed = torch.zeros(rows, cols // 8, dtype=torch.int32)
    for block in range(cols // 8):
        for slot, logical in enumerate(order):
            packed[:, block] |= (codes[:, block * 8 + logical].to(torch.int32) & 0xF) << (
                slot * 4
            )
    return packed


class TestPreprocessQuarkWeights:
    """Tests for AMD Quark weight preprocessing.

    Quark packs along the *output* axis, so a projection is stored as
    ``[K, N // 8]`` — the transpose of the GPTQ/AWQ convention.
    """

    K = 256
    N = 128
    BITS = 4
    GROUP_SIZE = 32
    N_GROUPS = K // GROUP_SIZE  # 8
    BLOB_SIZE = GROUP_SIZE * BITS // 8  # 16

    def _state_dict(self, codes, zeros, scales=None):
        sd = {
            "q_proj.weight": _quark_pack(codes),
            "q_proj.weight_scale": (
                scales if scales is not None else torch.randn(self.N_GROUPS, self.N)
            ),
        }
        if zeros is not None:
            sd["q_proj.weight_zero_point"] = _quark_pack(zeros)
        return sd

    @staticmethod
    def _unpack_nibbles(packed: torch.Tensor, count: int) -> torch.Tensor:
        out = torch.zeros(packed.shape[0], count, dtype=torch.int64)
        for byte in range(packed.shape[1]):
            value = packed[:, byte].to(torch.int64)
            out[:, 2 * byte] = value & 0xF
            if 2 * byte + 1 < count:
                out[:, 2 * byte + 1] = (value >> 4) & 0xF
        return out

    def test_quark_keys_replaced_by_matmulnbits_names(self):
        codes = torch.randint(0, 16, (self.K, self.N))
        zeros = torch.randint(0, 16, (self.N_GROUPS, self.N))
        result = preprocess_quark_weights(
            self._state_dict(codes, zeros), bits=self.BITS, group_size=self.GROUP_SIZE
        )
        assert set(result) == {"q_proj.weight", "q_proj.scales", "q_proj.zero_points"}

    def test_weight_shape_and_dtype(self):
        """Weight must be [N, n_blocks, blob_size] uint8 for MatMulNBits."""
        codes = torch.randint(0, 16, (self.K, self.N))
        zeros = torch.randint(0, 16, (self.N_GROUPS, self.N))
        result = preprocess_quark_weights(
            self._state_dict(codes, zeros), bits=self.BITS, group_size=self.GROUP_SIZE
        )
        weight = result["q_proj.weight"]
        assert weight.shape == (self.N, self.N_GROUPS, self.BLOB_SIZE)
        assert weight.dtype == torch.uint8

    def test_codes_round_trip_exactly(self):
        """Every uint4 code survives the unpack/transpose/repack unchanged."""
        codes = torch.randint(0, 16, (self.K, self.N))
        result = preprocess_quark_weights(
            self._state_dict(codes, None), bits=self.BITS, group_size=self.GROUP_SIZE
        )
        recovered = self._unpack_nibbles(result["q_proj.weight"].reshape(self.N, -1), self.K)
        # Quark packs along the output axis, so the result is the transpose.
        assert torch.equal(recovered, codes.t().to(torch.int64))

    def test_scales_transposed(self):
        scales = torch.randn(self.N_GROUPS, self.N)
        result = preprocess_quark_weights(
            self._state_dict(torch.randint(0, 16, (self.K, self.N)), None, scales),
            bits=self.BITS,
            group_size=self.GROUP_SIZE,
        )
        assert torch.equal(result["q_proj.scales"], scales.t())

    def test_zero_point_of_15_survives(self):
        """Quark zero points are unbiased, so 15 must round-trip.

        The AWQ path cannot represent this: its implicit ``+1`` offset wraps
        ``15`` to ``0``. Guards the one numerical difference between the two.
        """
        zeros = torch.full((self.N_GROUPS, self.N), 15)
        result = preprocess_quark_weights(
            self._state_dict(torch.randint(0, 16, (self.K, self.N)), zeros),
            bits=self.BITS,
            group_size=self.GROUP_SIZE,
        )
        recovered = self._unpack_nibbles(result["q_proj.zero_points"], self.N_GROUPS)
        assert (recovered == 15).all()

    def test_no_zero_point_offset_applied(self):
        """Zero points pass through verbatim — no GPTQ/AWQ style correction."""
        zeros = torch.randint(0, 16, (self.N_GROUPS, self.N))
        result = preprocess_quark_weights(
            self._state_dict(torch.randint(0, 16, (self.K, self.N)), zeros),
            bits=self.BITS,
            group_size=self.GROUP_SIZE,
        )
        recovered = self._unpack_nibbles(result["q_proj.zero_points"], self.N_GROUPS)
        assert torch.equal(recovered, zeros.t().to(torch.int64))

    def test_float_weights_untouched(self):
        """A ``.weight`` without a ``.weight_scale`` sibling is not quantized."""
        norm = torch.randn(self.N)
        result = preprocess_quark_weights(
            {"input_layernorm.weight": norm}, bits=self.BITS, group_size=self.GROUP_SIZE
        )
        assert torch.equal(result["input_layernorm.weight"], norm)

    def test_missing_payload_raises(self):
        sd = {"q_proj.weight_scale": torch.randn(self.N_GROUPS, self.N)}
        with pytest.raises(ValueError, match=r"Missing q_proj\.weight"):
            preprocess_quark_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)

    def test_group_size_mismatch_raises(self):
        codes = torch.randint(0, 16, (self.K, self.N))
        sd = self._state_dict(codes, None, torch.randn(self.N_GROUPS, self.N))
        with pytest.raises(ValueError, match="implies"):
            preprocess_quark_weights(sd, bits=self.BITS, group_size=64)

    def test_dispatches_from_quantized_weights(self):
        """``preprocess_quantized_weights`` routes quant_method='quark' here."""
        codes = torch.randint(0, 16, (self.K, self.N))
        zeros = torch.randint(0, 16, (self.N_GROUPS, self.N))
        result = preprocess_quantized_weights(
            self._state_dict(codes, zeros),
            QuantizationConfig(
                bits=self.BITS,
                group_size=self.GROUP_SIZE,
                quant_method="quark",
                sym=False,
            ),
        )
        assert result["q_proj.weight"].shape == (self.N, self.N_GROUPS, self.BLOB_SIZE)
