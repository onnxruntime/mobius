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

import pathlib

import pytest

from mobius.integrations.gguf import _repacker
from mobius.integrations.gguf._quant_registry import (
    explicit_zero_point_type_names,
    float_storage_type_ids,
    get_quant_spec,
    iter_quant_specs,
    lm_head_preserve_type_names,
    lossless_preservation_type_names,
    quant_import_decision,
    quant_spec_by_name,
    render_quant_support_matrix,
)
from mobius.integrations.gguf._spec import (
    QuantImportRoute,
    RepackExactness,
    StorageRole,
    Support,
    TensorRole,
)
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

#: Every non-rejected projection/output route may stay quantized.
_EXPECTED_LM_HEAD_PRESERVE = frozenset(
    {
        "Q1_0",
        "Q4_0",
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

_PINNED_STORED_ROUTES = {
    "Q4_0": ("affine repack", "exact"),
    "Q4_1": ("affine repack", "lossy"),
    "Q5_0": ("dequantize/requantize", None),
    "Q5_1": ("dequantize/requantize", None),
    "Q8_0": ("affine repack", "exact"),
    "Q2_K": ("dequantize/requantize", None),
    "Q3_K": ("dequantize/requantize", None),
    "Q4_K": ("affine repack", "lossy"),
    "Q5_K": ("dequantize/requantize", None),
    "Q6_K": ("affine repack", "lossy"),
    "TQ1_0": ("dequantize/requantize", None),
    "TQ2_0": ("dequantize/requantize", None),
    "IQ4_NL": ("native byte-preserved", None),
    "IQ4_XS": ("native byte-preserved", None),
    "IQ3_S": ("native byte-preserved", None),
    "IQ3_XXS": ("native byte-preserved", None),
    "IQ2_XXS": ("native byte-preserved", None),
    "IQ2_XS": ("native byte-preserved", None),
    "IQ2_S": ("native byte-preserved", None),
    "IQ1_S": ("native byte-preserved", None),
    "IQ1_M": ("native byte-preserved", None),
    "MXFP4": ("native byte-preserved", None),
    "NVFP4": ("dequantize/requantize", None),
    "Q1_0": ("affine repack", "exact"),
    "Q2_0": ("rejected", None),
}


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
        assert lm_head_preserve_type_names() == _EXPECTED_LM_HEAD_PRESERVE

    def test_only_value_preserving_affine_types_are_advertised(self) -> None:
        preserved = lossless_preservation_type_names()
        assert {"Q1_0", "Q4_0", "Q8_0"} <= preserved
        assert {"Q4_1", "Q4_K", "Q6_K"}.isdisjoint(preserved)

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

    def test_every_active_stored_qtype_has_one_pinned_route(self) -> None:
        actual = {
            spec.name: (
                spec.import_route.value,
                None if spec.repack_exactness is None else spec.repack_exactness.value,
            )
            for spec in iter_quant_specs()
            if spec.is_quantized_storage
        }
        assert actual == _PINNED_STORED_ROUTES
        assert len(actual) == 25

    @pytest.mark.parametrize(
        "spec",
        [
            spec
            for spec in iter_quant_specs()
            if spec.import_route is QuantImportRoute.DEQUANTIZE_REQUANTIZE
        ],
        ids=lambda spec: spec.name,
    )
    def test_declared_float_decoders_execute_one_pinned_block(self, spec) -> None:
        import numpy as np
        from gguf import GGMLQuantizationType, quants

        qtype = GGMLQuantizationType(spec.ggml_type_id)
        values = quants.dequantize(np.zeros(spec.block_bytes, dtype=np.uint8), qtype)
        assert values.size == spec.block_elements
        assert np.isfinite(values).all()

    def test_all_pinned_stored_quantized_types_are_registered(self) -> None:
        stored_quantized = [
            spec
            for spec in iter_quant_specs()
            if spec.readable and spec.role is StorageRole.QUANTIZED
        ]
        assert len(stored_quantized) == 25

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

    @pytest.mark.parametrize("spec", iter_quant_specs(), ids=lambda s: s.name)
    def test_runtime_stays_deferred_without_real_execution_evidence(self, spec) -> None:
        if spec.is_quantized_storage:
            assert spec.runtime is Support.DEFERRED
            assert "runtime" in spec.runtime_reason.lower()

    def test_native_projection_does_not_imply_native_embedding(self) -> None:
        projection = quant_import_decision(20, TensorRole.PROJECTION)
        embedding = quant_import_decision(20, TensorRole.EMBEDDING)
        assert projection[0] is QuantImportRoute.NATIVE_BYTES
        assert embedding[0] is QuantImportRoute.DEQUANTIZE_REQUANTIZE
        assert "GatherBlockQuantized" in embedding[2]

    def test_native_experts_require_contiguous_expert_major_layout(self) -> None:
        route, exactness, reason = quant_import_decision(39, TensorRole.EXPERT)
        assert route is QuantImportRoute.NATIVE_BYTES
        assert exactness is None
        assert "contiguous" in reason

    def test_non_matmul_q1_0_is_rejected_without_a_decoder(self) -> None:
        route, _, reason = quant_import_decision(41, TensorRole.NON_MATMUL)
        assert route is QuantImportRoute.REJECTED
        assert "decoder" in reason

    def test_affine_expert_major_route_uses_complete_per_expert_targets(self) -> None:
        route, exactness, _ = quant_import_decision(2, TensorRole.EXPERT)
        assert route is QuantImportRoute.AFFINE_REPACK
        assert exactness is RepackExactness.EXACT

    def test_expert_major_route_without_a_decoder_is_rejected(self) -> None:
        route, _, reason = quant_import_decision(
            41,
            TensorRole.EXPERT,
            target_bits=4,
            target_block_size=32,
        )
        assert route is QuantImportRoute.REJECTED
        assert "no trusted decoder" in reason

    def test_exact_q8_route_becomes_lossy_for_four_bit_target(self) -> None:
        route, exactness, reason = quant_import_decision(
            8,
            TensorRole.PROJECTION,
            target_bits=4,
            target_block_size=32,
        )
        assert route is QuantImportRoute.DEQUANTIZE_REQUANTIZE
        assert exactness is RepackExactness.LOSSY
        assert "lossy" in reason

    def test_q1_route_rejects_an_incompatible_target_without_decoder(self) -> None:
        route, _, reason = quant_import_decision(
            41,
            TensorRole.PROJECTION,
            target_bits=4,
            target_block_size=32,
        )
        assert route is QuantImportRoute.REJECTED
        assert "no trusted decoder" in reason

    @pytest.mark.parametrize("name", ["Q4_0", "Q8_0", "Q1_0"])
    def test_exact_affine_routes_are_declared(self, name: str) -> None:
        spec = quant_spec_by_name(name)
        assert spec is not None
        assert spec.import_route is QuantImportRoute.AFFINE_REPACK
        assert spec.repack_exactness is RepackExactness.EXACT


class TestDocumentedQuantizationMatrix:
    _DOC = pathlib.Path(__file__).resolve().parents[4] / "docs" / "api" / "build_from_gguf.md"
    _BEGIN = "<!-- BEGIN GGUF QUANTIZATION MATRIX (generated; see _quant_registry.py) -->"
    _END = "<!-- END GGUF QUANTIZATION MATRIX -->"

    def test_generated_matrix_is_current(self) -> None:
        text = self._DOC.read_text(encoding="utf-8")
        documented = text.split(self._BEGIN, 1)[1].split(self._END, 1)[0].strip()
        assert documented == render_quant_support_matrix()
