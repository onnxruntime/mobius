"""Unit tests for BitNet ternary weight unpacking."""

from __future__ import annotations

import torch

from mobius.models.bitnet import _unpack_ternary_weights


class TestUnpackTernaryWeights:
    """Tests for the _unpack_ternary_weights helper."""

    def test_output_shape(self):
        """Packed [out//4, in] unpacks to [out, in]."""
        packed = torch.zeros(8, 16, dtype=torch.uint8)
        result = _unpack_ternary_weights(packed)
        assert result.shape == (32, 16)

    def test_values_in_ternary_set(self):
        """All output values must be in {-1, 0, +1} for valid packed data."""
        # Build valid packed bytes: only 2-bit values 0, 1, 2 (not 3)
        rng = torch.Generator().manual_seed(42)
        values = torch.randint(0, 3, (4, 8, 4), generator=rng, dtype=torch.uint8)
        packed = (
            values[..., 0]
            | (values[..., 1] << 2)
            | (values[..., 2] << 4)
            | (values[..., 3] << 6)
        )
        result = _unpack_ternary_weights(packed)
        unique = set(result.unique().tolist())
        assert unique.issubset({-1.0, 0.0, 1.0})

    def test_known_byte_round_trip(self):
        """Verify bit extraction for a known byte.

        Packing: ternary {-1, 0, +1} → unsigned {0, 1, 2}
        Byte = v0 | (v1 << 2) | (v2 << 4) | (v3 << 6)

        Example: values [-1, 0, +1, -1] → unsigned [0, 1, 2, 0]
        Byte = 0b_00_10_01_00 = 0x24 = 36
        """
        packed = torch.tensor([[36]], dtype=torch.uint8)  # [1, 1]
        result = _unpack_ternary_weights(packed)
        # Unpacks to [4, 1]: four values from the single byte
        assert result.shape == (4, 1)
        expected = torch.tensor([[-1.0], [0.0], [1.0], [-1.0]])
        assert torch.equal(result, expected)

    def test_all_zeros_byte(self):
        """Byte 0x00: all four packed values are 0 → all -1."""
        packed = torch.tensor([[0]], dtype=torch.uint8)
        result = _unpack_ternary_weights(packed)
        assert torch.all(result == -1.0)

    def test_all_ones_byte(self):
        """Byte 0x55 = 0b_01_01_01_01: all four packed values are 1 → all 0."""
        packed = torch.tensor([[0x55]], dtype=torch.uint8)
        result = _unpack_ternary_weights(packed)
        assert torch.all(result == 0.0)

    def test_all_twos_byte(self):
        """Byte 0xAA = 0b_10_10_10_10: all four packed values are 2 → all +1."""
        packed = torch.tensor([[0xAA]], dtype=torch.uint8)
        result = _unpack_ternary_weights(packed)
        assert torch.all(result == 1.0)

    def test_output_dtype_is_float(self):
        """Result should be float32."""
        packed = torch.zeros(2, 4, dtype=torch.uint8)
        result = _unpack_ternary_weights(packed)
        assert result.dtype == torch.float32

    def test_invalid_ndim_raises(self):
        """1-D input should raise ValueError."""
        packed = torch.zeros(4, dtype=torch.uint8)
        try:
            _unpack_ternary_weights(packed)
            assert False, "Expected ValueError"
        except ValueError:
            pass

    def test_multi_column(self):
        """Verify correct unpacking across multiple columns.

        Two bytes per row, 2 rows → packed [2, 2] → unpacked [8, 2].
        """
        # Column 0: all -1 (byte 0x00), Column 1: all +1 (byte 0xAA)
        packed = torch.tensor([[0x00, 0xAA], [0x00, 0xAA]], dtype=torch.uint8)
        result = _unpack_ternary_weights(packed)
        assert result.shape == (8, 2)
        assert torch.all(result[:, 0] == -1.0)
        assert torch.all(result[:, 1] == 1.0)
