# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Validation rules for the GGUF capability specs.

The specs are the vocabulary the registries are written in, so their guards are
what make an incoherent registration impossible to write in the first place.
"""

from __future__ import annotations

import pytest

from mobius.integrations.gguf._spec import (
    AffineRepackSpec,
    GGUFArchitectureSpec,
    GGUFQuantSpec,
    NativeBlockSpec,
    StorageRole,
    Support,
)


class TestArchitectureSpecValidation:
    def test_a_minimal_supported_spec_is_accepted(self) -> None:
        spec = GGUFArchitectureSpec(
            gguf_arch="llama", model_type="llama", tensor_map_recipe=("llama",)
        )
        assert spec.is_importable
        assert spec.names == frozenset({"llama"})

    def test_an_unsupported_capability_needs_a_reason(self) -> None:
        with pytest.raises(ValueError, match="must say why"):
            GGUFArchitectureSpec(
                gguf_arch="bloom", model_type="bloom", tensor_map=Support.DEFERRED
            )

    def test_a_buildable_spec_needs_a_model_type(self) -> None:
        with pytest.raises(ValueError, match="requires a model_type"):
            GGUFArchitectureSpec(gguf_arch="llama", tensor_map_recipe=("llama",))

    def test_a_mappable_spec_needs_a_recipe(self) -> None:
        with pytest.raises(ValueError, match="non-empty tensor_map_recipe"):
            GGUFArchitectureSpec(gguf_arch="llama", model_type="llama")

    def test_a_recipe_may_not_contradict_an_unsupported_verdict(self) -> None:
        """A recipe on a rejected mapping would imply the mapping works."""
        with pytest.raises(ValueError, match="would imply it works"):
            GGUFArchitectureSpec(
                gguf_arch="bloom",
                model_type="bloom",
                tensor_map=Support.DEFERRED,
                tensor_map_recipe=("llama",),
                reason="no mapping",
            )

    def test_an_alias_may_not_repeat_the_canonical_name(self) -> None:
        with pytest.raises(ValueError, match="must not repeat in aliases"):
            GGUFArchitectureSpec(
                gguf_arch="llama",
                model_type="llama",
                aliases=frozenset({"llama"}),
                tensor_map_recipe=("llama",),
            )

    def test_verdicts_are_reported_separately(self) -> None:
        spec = GGUFArchitectureSpec(
            gguf_arch="t5",
            model_type="t5",
            tensor_map=Support.DEFERRED,
            reason="encoder-decoder",
        )
        assert spec.verdicts["config"] is Support.SUPPORTED
        assert spec.verdicts["tensor_map"] is Support.DEFERRED
        assert not spec.is_importable


class TestQuantSpecValidation:
    def _quantized(self, **overrides):
        fields = dict(
            ggml_type_id=12,
            name="Q4_K",
            role=StorageRole.QUANTIZED,
            block_elements=256,
            block_bytes=144,
        )
        fields.update(overrides)
        return GGUFQuantSpec(**fields)

    def test_a_minimal_quantized_spec_is_accepted(self) -> None:
        spec = self._quantized(affine_repack=AffineRepackSpec(4, 32))
        assert spec.readable
        assert spec.repack_params == (4, 32)

    def test_removed_slots_cannot_be_dequantized(self) -> None:
        with pytest.raises(ValueError, match="unreadable slots cannot be dequantized"):
            GGUFQuantSpec(
                ggml_type_id=4,
                name="Q4_2",
                role=StorageRole.REMOVED,
                block_elements=0,
                block_bytes=0,
                dequantize=Support.SUPPORTED,
            )

    def test_readability_must_agree_with_block_size(self) -> None:
        with pytest.raises(ValueError, match="contradicts"):
            GGUFQuantSpec(
                ggml_type_id=4,
                name="Q4_2",
                role=StorageRole.REMOVED,
                block_elements=256,
                block_bytes=144,
                dequantize=Support.REJECTED,
                reason="removed",
            )

    def test_native_and_affine_paths_are_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="never both"):
            self._quantized(
                native_preserve=NativeBlockSpec("q4_k", 256, 144),
                affine_repack=AffineRepackSpec(4, 32),
            )

    def test_native_geometry_must_match_upstream(self) -> None:
        with pytest.raises(ValueError, match="must match the upstream block size"):
            self._quantized(native_preserve=NativeBlockSpec("q4_k", 32, 144))
        with pytest.raises(ValueError, match="must match the upstream type size"):
            self._quantized(native_preserve=NativeBlockSpec("q4_k", 256, 17))

    def test_preservation_flags_require_quantized_storage(self) -> None:
        with pytest.raises(ValueError, match="lm_head_preserve only applies"):
            GGUFQuantSpec(
                ggml_type_id=0,
                name="F32",
                role=StorageRole.FLOAT,
                block_elements=1,
                block_bytes=4,
                lm_head_preserve=True,
            )

    def test_a_deferred_dequantization_needs_a_reason(self) -> None:
        with pytest.raises(ValueError, match="must say why"):
            self._quantized(dequantize=Support.DEFERRED)
