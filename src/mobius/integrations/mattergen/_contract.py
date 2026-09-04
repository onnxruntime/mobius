# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Host-owned contract for the pinned Microsoft MatterGen score model.

The ONNX component produced by this integration is deliberately only the
deterministic GemNet-T score core.  MatterGen rebuilds a ragged periodic
neighbor graph for each diffusion evaluation; its D3PM/SDE sampling loop,
classifier-free guidance, coordinate wrapping, and final crystal validation
therefore remain a source-compatible host responsibility.
"""

from __future__ import annotations

__all__ = [
    "MATTERGEN_HUB_ID",
    "MATTERGEN_HUB_REVISION",
    "MATTERGEN_SOURCE_COMMIT",
    "MATTERGEN_SOURCE_REPOSITORY",
    "MAX_ATOMS",
    "MAX_ATOMIC_NUMBER",
    "HOST_OWNED_STEPS",
    "OFFICIAL_CHECKPOINT_CONDITIONS",
    "SELECTED_ATOMIC_NUMBERS",
    "chemical_system_multihot",
    "validate_final_crystal",
]

from collections.abc import Sequence

import numpy as np

# The Hub commit pins configs and Lightning checkpoints together.  It is not
# the MatterGen source commit; 842ffe is the v1.0.3 implementation reference.
MATTERGEN_HUB_ID = "microsoft/mattergen"
MATTERGEN_HUB_REVISION = "5244495dd9a979ff71abc7548a0b14b9deb0069a"
MATTERGEN_SOURCE_REPOSITORY = "https://github.com/microsoft/mattergen"
MATTERGEN_SOURCE_COMMIT = "842ffe735f7d06cec89d56aa23d9f001e1124b30"

MAX_ATOMS = 20
MAX_ATOMIC_NUMBER = 100

# These stages must be performed by a caller around every ONNX score-core
# invocation.  The order preserves MatterGen's source sampling semantics.
HOST_OWNED_STEPS = (
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

# ``mattergen.common.utils.globals.SELECTED_ATOMIC_NUMBERS`` at the pinned
# source commit.  It is the sampling allowlist, not the D3PM vocabulary.
SELECTED_ATOMIC_NUMBERS = (
    1,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    37,
    38,
    39,
    40,
    41,
    42,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    55,
    56,
    57,
    58,
    59,
    60,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
)

# These names are a checkpoint routing contract, derived from each pinned
# Hydra config's ``condition_on_adapt`` field.  The score graph accepts raw
# values plus an explicit per-condition unconditional selector for every item.
OFFICIAL_CHECKPOINT_CONDITIONS: dict[str, tuple[str, ...]] = {
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

_SYMBOLS = (
    "",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
)
_ATOMIC_NUMBER_BY_SYMBOL = {symbol.casefold(): index for index, symbol in enumerate(_SYMBOLS)}
_SELECTED_ATOMIC_NUMBER_SET = frozenset(SELECTED_ATOMIC_NUMBERS)


def chemical_system_multihot(chemical_system: str | Sequence[str]) -> np.ndarray:
    """Convert MatterGen chemical-system input to its ``[101]`` float vector.

    A string uses the upstream hyphen-separated convention (for example,
    ``"Li-O"``).  Atomic-number zero is intentionally unused.  Inputs are
    constrained to the upstream generation allowlist because requesting a
    disallowed element would make the host's mandatory sampling logit mask
    unsatisfiable.
    """
    symbols = chemical_system.split("-") if isinstance(chemical_system, str) else chemical_system
    if not symbols:
        raise ValueError("chemical_system must contain at least one element.")

    multihot = np.zeros(MAX_ATOMIC_NUMBER + 1, dtype=np.float32)
    seen: set[int] = set()
    for raw_symbol in symbols:
        if not isinstance(raw_symbol, str) or not raw_symbol:
            raise ValueError("chemical_system elements must be non-empty symbols.")
        atomic_number = _ATOMIC_NUMBER_BY_SYMBOL.get(raw_symbol.casefold())
        if atomic_number is None or atomic_number == 0:
            raise ValueError(f"Unknown chemical-system element: {raw_symbol!r}.")
        if atomic_number not in _SELECTED_ATOMIC_NUMBER_SET:
            raise ValueError(
                f"Element {raw_symbol!r} (Z={atomic_number}) is outside MatterGen's "
                "pinned sampling allowlist."
            )
        if atomic_number in seen:
            raise ValueError(f"chemical_system contains duplicate element {raw_symbol!r}.")
        seen.add(atomic_number)
        multihot[atomic_number] = 1.0
    return multihot


def validate_final_crystal(
    atomic_numbers: np.ndarray,
    fractional_coordinates: np.ndarray,
    cell: np.ndarray,
) -> None:
    """Validate the bounded crystal artifact emitted by a MatterGen host loop.

    This is deliberately structural validation only.  It does not replace
    Pymatgen's site-distance checks or create a crystal object, both of which
    remain host implementation choices outside the pure score graph.
    """
    numbers = np.asarray(atomic_numbers)
    fractional = np.asarray(fractional_coordinates)
    lattice = np.asarray(cell)
    if numbers.ndim != 1 or not 1 <= len(numbers) <= MAX_ATOMS:
        raise ValueError(f"atomic_numbers must have a length in [1, {MAX_ATOMS}].")
    if fractional.shape != (len(numbers), 3):
        raise ValueError("fractional_coordinates must have shape [N, 3].")
    if lattice.shape != (3, 3):
        raise ValueError("cell must have shape [3, 3].")
    if not np.isfinite(fractional).all() or not np.isfinite(lattice).all():
        raise ValueError("fractional_coordinates and cell must be finite.")
    if not np.issubdtype(numbers.dtype, np.integer):
        raise TypeError("atomic_numbers must use an integer dtype.")
    if any(int(number) not in _SELECTED_ATOMIC_NUMBER_SET for number in numbers):
        raise ValueError("atomic_numbers contains an element outside MatterGen's allowlist.")
    if np.any(fractional < 0.0) or np.any(fractional >= 1.0):
        raise ValueError("fractional_coordinates must be wrapped to [0, 1).")
    if not np.isfinite(np.linalg.det(lattice)) or np.linalg.det(lattice) <= 0.0:
        raise ValueError("cell must have positive finite volume.")
