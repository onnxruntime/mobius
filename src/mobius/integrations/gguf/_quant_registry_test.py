# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Invariants for the GGUF quantization registry.

Two jobs:

1. **Behavior preservation.** The tables in ``_repacker`` and ``_builder`` used
   to be literals maintained by hand. They are now derived. Every literal is
   pinned here verbatim, so the refactor is provably a no-op and a future edit
   to the registry that changes a repack target has to say so out loud.
2. **Internal consistency.** A quantization type cannot be readable and removed,
   preserved natively and repacked, or given a zero-point rule with no repack
   target. Those combinations used to be expressible across five tables.
"""

from __future__ import annotations

import pytest

from mobius.integrations.gguf import _repacker
from mobius.integrations.gguf._quant_registry import (
    explicit_zero_point_type_names,
    float_storage_type_ids,
    get_quant_spec,
    iter_quant_specs,
    lm_head_preserve_type_names,
    quant_spec_by_name,
)
from mobius.integrations.gguf._spec import StorageRole, Support
from mobius.integrations.gguf._upstream import upstream_quant_types

# --------------------------------------------------------------------------
# Pre-refactor literals, copied verbatim from the tables this registry replaced.
# --------------------------------------------------------------------------

#: ``_repacker._REPACK_PARAMS``
_LEGACY_REPACK_PARAMS = {
    2: (4, 32),
    3: (4, 32),
    8: (8, 32),
    12: (4, 32),
    14: (4, 32),
    41: (2, 128),
}

#: ``_repacker._BLOCK_BYTES``
_LEGACY_BLOCK_BYTES = {2: 18, 3: 20, 8: 34, 12: 144, 14: 210, 41: 18}

#: ``_repacker._GGUF_BLOCK_ELEMENTS``
_LEGACY_BLOCK_ELEMENTS = {2: 32, 3: 32, 8: 32, 12: 256, 14: 256, 41: 128}

#: ``_repacker._NATIVE_BLOCK_SPECS`` as ``id -> (format, elements, bytes)``
_LEGACY_NATIVE_BLOCKS = {
    39: ("mxfp4", 32, 17),
    20: ("iq4_nl", 32, 18),
    23: ("iq4_xs", 256, 136),
    21: ("iq3_s", 256, 110),
    18: ("iq3_xxs", 256, 98),
    16: ("iq2_xxs", 256, 66),
    17: ("iq2_xs", 256, 74),
    22: ("iq2_s", 256, 82),
    19: ("iq1_s", 256, 50),
    29: ("iq1_m", 256, 56),
}

#: ``_builder._detect_quant_params.type_can_omit_zero_points`` — every repack
#: target emitted zero points explicitly.
_LEGACY_OMIT_ZERO_POINTS = {2: False, 3: False, 12: False, 14: False, 8: False, 41: False}

#: ``_builder._detect_quant_params.explicit_zero_point_types``
_LEGACY_EXPLICIT_ZERO_POINT_TYPES = frozenset(
    {"Q1_0", "Q2_K", "Q4_0", "Q4_1", "Q4_K", "Q5_1", "Q5_K", "Q8_0"}
)

#: ``_builder._can_quantize_lm_head.supported_types``
_LEGACY_LM_HEAD_PRESERVE = frozenset(
    {
        "Q1_0",
        "Q2_K",
        "Q3_K",
        "Q4_0",
        "Q4_1",
        "Q4_K",
        "Q5_0",
        "Q5_1",
        "Q5_K",
        "Q6_K",
        "Q8_0",
        "MXFP4",
        "IQ4_NL",
        "IQ4_XS",
        "IQ3_S",
        "IQ3_XXS",
        "IQ2_XXS",
        "IQ2_XS",
        "IQ2_S",
        "IQ1_S",
        "IQ1_M",
    }
)

#: ``_builder._has_quantized_weights.float_types`` — F32, F16, BF16, plus F64
#: when the installed ``gguf`` exposes it.
_LEGACY_FLOAT_TYPE_IDS = frozenset({0, 1, 30, 28})


class TestBehaviorPreservation:
    """Derived tables must equal the literals they replaced."""

    def test_repack_params(self) -> None:
        assert dict(_repacker._REPACK_PARAMS) == _LEGACY_REPACK_PARAMS

    def test_block_bytes(self) -> None:
        assert dict(_repacker._BLOCK_BYTES) == _LEGACY_BLOCK_BYTES

    def test_block_elements(self) -> None:
        assert dict(_repacker._GGUF_BLOCK_ELEMENTS) == _LEGACY_BLOCK_ELEMENTS

    def test_supported_types(self) -> None:
        assert frozenset(_LEGACY_REPACK_PARAMS) == _repacker._SUPPORTED_TYPES

    def test_native_block_specs(self) -> None:
        derived = {
            type_id: (spec.format, spec.elements, spec.bytes)
            for type_id, spec in _repacker._NATIVE_BLOCK_SPECS.items()
        }
        assert derived == _LEGACY_NATIVE_BLOCKS

    def test_native_block_byte_sizes(self) -> None:
        expected = frozenset(size for _, _, size in _LEGACY_NATIVE_BLOCKS.values())
        assert expected == _repacker.NATIVE_BLOCK_BYTE_SIZES

    def test_omit_zero_points(self) -> None:
        derived = {
            spec.ggml_type_id: spec.affine_repack.omit_zero_points
            for spec in iter_quant_specs()
            if spec.affine_repack is not None
        }
        assert derived == _LEGACY_OMIT_ZERO_POINTS

    def test_explicit_zero_point_types(self) -> None:
        assert explicit_zero_point_type_names() == _LEGACY_EXPLICIT_ZERO_POINT_TYPES

    def test_lm_head_preserve_types(self) -> None:
        assert lm_head_preserve_type_names() == _LEGACY_LM_HEAD_PRESERVE

    def test_float_storage_types(self) -> None:
        assert float_storage_type_ids() == _LEGACY_FLOAT_TYPE_IDS

    def test_type_ids_match_the_gguf_package(self) -> None:
        """The ids resolved from the census must match the installed enum."""
        from gguf import GGMLQuantizationType

        for member in GGMLQuantizationType:
            spec = get_quant_spec(member)
            assert spec is not None, f"{member.name} has no registry entry"
            assert spec.name == member.name


class TestCensusCoverage:
    """Every pinned ggml slot is accounted for, with upstream geometry."""

    def test_all_slots_are_registered(self) -> None:
        assert len(iter_quant_specs()) == 43
        assert len(upstream_quant_types()) == 43

    @pytest.mark.parametrize("spec", iter_quant_specs(), ids=lambda s: s.name)
    def test_geometry_matches_upstream(self, spec) -> None:
        upstream = upstream_quant_types()[spec.ggml_type_id]
        assert spec.block_elements == upstream.block_elements
        assert spec.block_bytes == upstream.block_bytes
        assert spec.readable == upstream.readable


class TestStorageInvariants:
    """Combinations that used to be expressible across five tables are illegal."""

    @pytest.mark.parametrize("spec", iter_quant_specs(), ids=lambda s: s.name)
    def test_readability_follows_block_size(self, spec) -> None:
        """Upstream rejects a tensor whose type has ``blck_size == 0`` at parse time."""
        assert spec.readable == (spec.block_elements > 0)

    @pytest.mark.parametrize("spec", iter_quant_specs(), ids=lambda s: s.name)
    def test_a_type_is_never_both_preserved_and_repacked(self, spec) -> None:
        assert not (spec.native_preserve is not None and spec.affine_repack is not None)

    @pytest.mark.parametrize("spec", iter_quant_specs(), ids=lambda s: s.name)
    def test_unsupported_dequantization_carries_a_reason(self, spec) -> None:
        if spec.dequantize is not Support.SUPPORTED:
            assert spec.reason

    @pytest.mark.parametrize("spec", iter_quant_specs(), ids=lambda s: s.name)
    def test_lm_head_preservation_has_a_path(self, spec) -> None:
        """An lm_head can only stay quantized if some path can consume it."""
        if not spec.lm_head_preserve:
            return
        assert (
            spec.native_preserve is not None
            or spec.affine_repack is not None
            or spec.dequantize is Support.SUPPORTED
        ), f"{spec.name}: lm_head preservation claimed with no way to read the blocks"

    def test_removed_types_are_rejected(self) -> None:
        """The 8 retired slots are unreadable, so nothing downstream can help."""
        removed = [s for s in iter_quant_specs() if s.role is StorageRole.REMOVED]
        assert len(removed) == 8
        for spec in removed:
            assert not spec.readable
            assert spec.dequantize is Support.REJECTED
            assert spec.native_preserve is None
            assert spec.affine_repack is None

    def test_compute_only_types_are_rejected(self) -> None:
        """``q8_1``/``q8_K`` have no ``to_float``; stored as weights they are malformed."""
        compute_only = {
            s.name for s in iter_quant_specs() if s.role is StorageRole.COMPUTE_ONLY
        }
        assert compute_only == {"Q8_1", "Q8_K"}
        for name in compute_only:
            spec = quant_spec_by_name(name)
            assert spec is not None
            assert spec.dequantize is Support.REJECTED

    @pytest.mark.parametrize("name", ["Q1_0", "Q2_0"])
    def test_types_without_a_python_dequantizer_are_deferred(self, name: str) -> None:
        """The float path calls ``gguf.dequantize``; these have no implementation."""
        spec = quant_spec_by_name(name)
        assert spec is not None
        assert spec.dequantize is Support.DEFERRED
        assert "dequantizer" in spec.reason

    def test_q1_0_is_still_importable_through_the_repack_path(self) -> None:
        """Deferred *dequantization* must not be read as "unsupported type"."""
        spec = quant_spec_by_name("Q1_0")
        assert spec is not None
        assert spec.affine_repack is not None
        assert spec.repack_params == (2, 128)
        assert spec.lm_head_preserve
