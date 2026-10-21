# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import numpy as np
import pytest

from mobius.integrations.mattergen._contract import (
    HOST_OWNED_STEPS,
    MATTERGEN_HUB_REVISION,
    MATTERGEN_SOURCE_COMMIT,
    OFFICIAL_CHECKPOINT_CONDITIONS,
    chemical_system_multihot,
    validate_final_crystal,
)


class TestMatterGenHostContract:
    def test_pins_the_hub_and_source_implementation(self) -> None:
        assert MATTERGEN_HUB_REVISION == "5244495dd9a979ff71abc7548a0b14b9deb0069a"
        assert MATTERGEN_SOURCE_COMMIT == "842ffe735f7d06cec89d56aa23d9f001e1124b30"

    def test_declares_all_official_checkpoint_condition_contracts(self) -> None:
        assert OFFICIAL_CHECKPOINT_CONDITIONS == {
            "mattergen_base": (),
            "mp_20_base": (),
            "chemical_system": ("chemical_system",),
            "chemical_system_energy_above_hull": ("chemical_system", "energy_above_hull"),
            "space_group": ("space_group",),
            "dft_band_gap": ("dft_band_gap",),
            "dft_mag_density": ("dft_mag_density",),
            "dft_mag_density_hhi_score": ("dft_mag_density", "hhi_score"),
            "ml_bulk_modulus": ("ml_bulk_modulus",),
        }

    def test_declares_the_complete_host_orchestration_boundary(self) -> None:
        assert HOST_OWNED_STEPS == (
            "periodic_radius_graph",
            "symmetric_edge_reordering",
            "sparse_triplet_construction",
            "d3pm_species_sampling",
            "wrapped_ve_coordinate_update",
            "vp_lattice_update",
            "classifier_free_guidance",
            "fractional_coordinate_wrapping",
            "lattice_projection",
            "crystal_validation",
        )

    def test_chemical_system_uses_one_based_atomic_number_slots(self) -> None:
        multihot = chemical_system_multihot("Li-O")

        assert multihot.dtype == np.float32
        assert multihot.shape == (101,)
        np.testing.assert_array_equal(multihot[[0, 3, 8]], [0.0, 1.0, 1.0])
        assert np.count_nonzero(multihot) == 2

    @pytest.mark.parametrize("chemical_system", ["He", "Li-Li", "Xx", ""])
    def test_chemical_system_rejects_unsampleable_or_invalid_elements(
        self, chemical_system: str
    ) -> None:
        with pytest.raises(ValueError):
            chemical_system_multihot(chemical_system)

    def test_accepts_wrapped_bounded_crystal(self) -> None:
        validate_final_crystal(
            np.array([3, 8], dtype=np.int64),
            np.array([[0.0, 0.5, 0.9], [0.25, 0.75, 0.125]], dtype=np.float32),
            np.diag(np.array([3.0, 3.0, 3.0], dtype=np.float32)),
        )

    @pytest.mark.parametrize(
        ("numbers", "fractional", "cell", "message"),
        [
            (
                np.array([3], dtype=np.int64),
                np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
                np.eye(3, dtype=np.float32),
                "wrapped",
            ),
            (
                np.array([2], dtype=np.int64),
                np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                np.eye(3, dtype=np.float32),
                "allowlist",
            ),
            (
                np.array([3], dtype=np.int64),
                np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                np.diag(np.array([-1.0, 1.0, 1.0], dtype=np.float32)),
                "positive",
            ),
        ],
    )
    def test_rejects_invalid_final_crystal(
        self, numbers: np.ndarray, fractional: np.ndarray, cell: np.ndarray, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            validate_final_crystal(numbers, fractional, cell)
