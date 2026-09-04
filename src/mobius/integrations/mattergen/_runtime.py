# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Source-faithful host runtime for the MatterGen v1.0.3 score-core ABI.

This module ports the host-only portions of the pinned MatterGen implementation
without importing MatterGen, PyTorch Geometric, ``torch_scatter``, or
``torch_sparse``.  In particular, :func:`build_periodic_graph` reproduces the
periodic-image enumeration and GemNet-T edge/triplet ordering used by
``mattergen.common.utils.ocp_graph_utils.radius_graph_pbc`` and
``mattergen.common.gemnet.gemnet.GemNetT.generate_interaction_graph``.

The score callback is deliberately explicit.  It can call ONNX Runtime (using
:func:`create_onnxruntime_score_callback`) or another execution host, while
this module owns the scientifically significant graph and sampler semantics.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from mobius.integrations.mattergen._configs import MATTERGEN_CONDITION_FAMILY
from mobius.integrations.mattergen._contract import (
    MAX_ATOMS,
    SELECTED_ATOMIC_NUMBERS,
    validate_final_crystal,
)

__all__ = [
    "MATTERGEN_LATTICE_LIMIT_DENSITY",
    "MATTERGEN_SAMPLING_STEPS",
    "MatterGenCrystal",
    "MatterGenGraph",
    "MatterGenHostSampler",
    "MatterGenSampleBatch",
    "MatterGenScoreCallback",
    "MatterGenScoreInputs",
    "MatterGenScoreOutputs",
    "build_periodic_graph",
    "create_onnxruntime_score_callback",
]

# ``mattergen/conf/data_module/{mp_20,alex_mp_20}.yaml``.  Both published
# generation datasets resolve the corruption interpolation to this same value.
MATTERGEN_LATTICE_LIMIT_DENSITY = 0.05771451654022283
MATTERGEN_SAMPLING_STEPS = 1000
_EPS_T = 1.0 / MATTERGEN_SAMPLING_STEPS
_D3PM_CLASSES = 101
_D3PM_MASK_CLASS = _D3PM_CLASSES - 1
_D3PM_EPSILON = 1e-20
# ``sampling_conf/default.yaml`` configures both released Langevin correctors
# with this cap, including their defined all-zero-score branch.
_LANGEVIN_MAX_STEP_SIZE = 1e6
# Materialized exactly as ``MaskDiffusion._create_state`` does for
# ``create_discrete_diffusion_schedule(kind="standard", num_steps=1000)``.
_D3PM_BETAS = torch.cat(
    [
        torch.tensor([0.0], device="cpu"),
        1.0
        / (MATTERGEN_SAMPLING_STEPS - torch.arange(MATTERGEN_SAMPLING_STEPS, device="cpu")),
    ]
).to(torch.float64)
_D3PM_STATE = torch.cumprod(1.0 - _D3PM_BETAS, dim=0).to(torch.float32)
_D3PM_STATE[-1] = 0.0


@dataclass(frozen=True)
class MatterGenGraph:
    """Ragged source-ordered GemNet-T geometry for one batched score call.

    ``edge_index`` has source/target rows.  ``edge_direction`` is the source
    ``V_st`` convention, i.e. the negative normalized periodic distance vector.
    """

    edge_index: torch.Tensor
    edge_distance: torch.Tensor
    edge_direction: torch.Tensor
    edge_lattice_cosines: torch.Tensor
    id_swap: torch.Tensor
    id3_ba: torch.Tensor
    id3_ca: torch.Tensor
    id3_ragged_idx: torch.Tensor


@dataclass(frozen=True)
class MatterGenScoreInputs:
    """All inputs required by one pure-ONNX MatterGen score-core invocation."""

    atomic_numbers: torch.Tensor
    batch: torch.Tensor
    timestep: torch.Tensor
    graph: MatterGenGraph
    condition_values: Mapping[str, torch.Tensor]
    use_unconditional: Mapping[str, torch.Tensor]

    def as_onnx_inputs(self) -> dict[str, np.ndarray]:
        """Return the exact named ABI expected by ``MatterGenScoreTask``.

        ONNX Runtime consumes CPU NumPy arrays.  Keeping this conversion at the
        callback boundary prevents scheduler code from depending on an ORT API.
        """
        graph = self.graph
        values = {
            "atomic_numbers": _as_numpy(self.atomic_numbers),
            "batch": _as_numpy(self.batch),
            "timestep": _as_numpy(self.timestep),
            "edge_index": _as_numpy(graph.edge_index),
            "edge_distance": _as_numpy(graph.edge_distance),
            "edge_direction": _as_numpy(graph.edge_direction),
            "edge_lattice_cosines": _as_numpy(graph.edge_lattice_cosines),
            "id_swap": _as_numpy(graph.id_swap),
            "id3_ba": _as_numpy(graph.id3_ba),
            "id3_ca": _as_numpy(graph.id3_ca),
            "id3_ragged_idx": _as_numpy(graph.id3_ragged_idx),
        }
        values.update(
            {
                f"condition.{name}": _as_numpy(value)
                for name, value in self.condition_values.items()
            }
        )
        values.update(
            {
                f"condition.{name}.use_unconditional": _as_numpy(value)
                for name, value in self.use_unconditional.items()
            }
        )
        return values


@dataclass(frozen=True)
class MatterGenScoreOutputs:
    """Raw score-core outputs before MatterGen host postprocessing."""

    atom_logits: torch.Tensor
    coordinate_score: torch.Tensor
    lattice_score: torch.Tensor
    energy: torch.Tensor | None = None


MatterGenScoreCallback = Callable[[MatterGenScoreInputs], MatterGenScoreOutputs]


class _OnnxRuntimeSession(Protocol):
    """Minimal structural type accepted from an ONNX Runtime inference session."""

    def run(
        self, output_names: Sequence[str] | None, input_feed: Mapping[str, np.ndarray]
    ) -> Sequence[np.ndarray]: ...


def create_onnxruntime_score_callback(session: _OnnxRuntimeSession) -> MatterGenScoreCallback:
    """Adapt an ``onnxruntime.InferenceSession`` without importing onnxruntime.

    The returned callback preserves the model's raw Cartesian coordinate score;
    :class:`MatterGenHostSampler` performs the source ``cell^{-T}``
    conversion before applying the wrapped VE scheduler.
    """

    def score(inputs: MatterGenScoreInputs) -> MatterGenScoreOutputs:
        atom_logits, coordinate_score, lattice_score, energy = session.run(
            ["atom_logits", "coordinate_score", "lattice_score", "energy"],
            inputs.as_onnx_inputs(),
        )
        return MatterGenScoreOutputs(
            atom_logits=torch.from_numpy(atom_logits),
            coordinate_score=torch.from_numpy(coordinate_score),
            lattice_score=torch.from_numpy(lattice_score),
            energy=torch.from_numpy(energy),
        )

    return score


@dataclass(frozen=True)
class MatterGenCrystal:
    """A validated crystal artifact represented in MatterGen row-vector convention."""

    atomic_numbers: torch.Tensor
    fractional_coordinates: torch.Tensor
    cell: torch.Tensor

    def validate(self) -> None:
        """Run dependency-free structural checks before exposing this artifact."""
        validate_final_crystal(
            self.atomic_numbers.detach().cpu().numpy(),
            self.fractional_coordinates.detach().cpu().numpy(),
            self.cell.detach().cpu().numpy(),
        )


@dataclass(frozen=True)
class MatterGenSampleBatch:
    """Final mean sample returned by MatterGen's predictor-corrector sampler."""

    atomic_numbers: torch.Tensor
    fractional_coordinates: torch.Tensor
    cell: torch.Tensor
    num_atoms: torch.Tensor

    @property
    def batch(self) -> torch.Tensor:
        """Source-compatible crystal index for each atom."""
        return torch.repeat_interleave(
            torch.arange(len(self.num_atoms), dtype=torch.long, device=self.num_atoms.device),
            self.num_atoms,
        )

    def crystals(self) -> tuple[MatterGenCrystal, ...]:
        """Split the packed batch and validate every final crystal."""
        crystals: list[MatterGenCrystal] = []
        start = 0
        for count, lattice in zip(self.num_atoms.tolist(), self.cell, strict=True):
            stop = start + count
            crystal = MatterGenCrystal(
                atomic_numbers=self.atomic_numbers[start:stop],
                fractional_coordinates=self.fractional_coordinates[start:stop],
                cell=lattice,
            )
            crystal.validate()
            crystals.append(crystal)
            start = stop
        return tuple(crystals)


@dataclass(frozen=True)
class _State:
    """Packed fields corrupted jointly by the source multi-corruption sampler."""

    atomic_numbers: torch.Tensor
    fractional_coordinates: torch.Tensor
    cell: torch.Tensor
    num_atoms: torch.Tensor

    @property
    def batch(self) -> torch.Tensor:
        return torch.repeat_interleave(
            torch.arange(len(self.num_atoms), dtype=torch.long, device=self.num_atoms.device),
            self.num_atoms,
        )


def build_periodic_graph(
    fractional_coordinates: torch.Tensor,
    cell: torch.Tensor,
    num_atoms: torch.Tensor,
    *,
    cutoff: float = 7.0,
    max_neighbors: int = 50,
    max_cell_images_per_dim: int = 5,
) -> MatterGenGraph:
    """Construct the pinned-source periodic GemNet-T graph without PyG.

    This follows MatterGen's batched OCP graph construction exactly: the
    maximum periodic-image extent is shared across the batch, candidates are
    ordered by target atom / source atom / image, then directed candidates are
    symmetrized image-by-image before triplets are constructed.
    """
    _validate_geometry_inputs(fractional_coordinates, cell, num_atoms)
    if cutoff <= 0.0:
        raise ValueError("cutoff must be positive.")
    if max_neighbors <= 0 or max_cell_images_per_dim <= 0:
        raise ValueError("max_neighbors and max_cell_images_per_dim must be positive.")

    batch = torch.repeat_interleave(
        torch.arange(len(num_atoms), dtype=torch.long, device=num_atoms.device), num_atoms
    )
    # Source ``frac_to_cart_coords_with_lattice`` uses row vectors:
    # (N, 3) @ (N, 3, 3) -> (N, 3) Cartesian positions.
    cartesian = torch.einsum("ni,nij->nj", fractional_coordinates, cell[batch])
    cell_offsets = _periodic_image_offsets(cell, cutoff, max_cell_images_per_dim)
    edge_index, to_jimages, neighbors = _radius_graph_pbc(
        cartesian,
        cell,
        num_atoms,
        cell_offsets,
        cutoff,
        max_neighbors,
    )
    if edge_index.shape[1] == 0:
        raise ValueError(
            "MatterGen source ordering cannot construct GemNet-T triplets for an empty "
            "periodic radius graph. Increase the cutoff or use a physically valid cell."
        )

    # ``get_pbc_distances`` uses j->i edge order and a row-vector image shift.
    lattice_edges = torch.repeat_interleave(cell, neighbors, dim=0)
    distance_vectors = (
        cartesian[edge_index[0]]
        - cartesian[edge_index[1]]
        + torch.einsum("ei,eij->ej", to_jimages, lattice_edges)
    )
    distances = torch.linalg.vector_norm(distance_vectors, dim=-1)
    edge_direction = -distance_vectors / distances[:, None]

    edge_index, to_jimages, neighbors, distances, edge_direction = _reorder_symmetric_edges(
        edge_index, to_jimages, neighbors, distances, edge_direction
    )
    if edge_index.shape[1] == 0:
        raise ValueError(
            "MatterGen source symmetric edge reordering removed every edge; "
            "the current graph cannot be scored faithfully."
        )
    id_swap = _symmetric_edge_swaps(neighbors)
    id3_ba, id3_ca, id3_ragged_idx = _triplets(edge_index)

    # GemNet appends cosine alignment to each edge embedding.  ``batch`` is
    # indexed by the source node exactly as in ``GemNetT.forward``.
    edge_lattice_cosines = torch.cosine_similarity(
        edge_direction[:, None, :], cell[batch[edge_index[0]]], dim=-1
    )
    return MatterGenGraph(
        edge_index=edge_index,
        edge_distance=distances,
        edge_direction=edge_direction,
        edge_lattice_cosines=edge_lattice_cosines,
        id_swap=id_swap,
        id3_ba=id3_ba,
        id3_ca=id3_ca,
        id3_ragged_idx=id3_ragged_idx,
    )


class MatterGenHostSampler:
    """Run the official v1.0.3 MatterGen PC loop around an explicit score callback.

    It implements the released 1,000-step absorbing-mask D3PM, the
    number-of-atoms-adjusted wrapped VE coordinate process, the lattice VP
    process, source predictor/corrector ordering, classifier-free guidance,
    source logit masking, and final structural validation.  No shortened or
    rescheduled path is accepted because it would not be a source-compatible
    MatterGen sampler.
    """

    def __init__(
        self,
        score_callback: MatterGenScoreCallback,
        *,
        condition_names: Sequence[str] = (),
        cutoff: float = 7.0,
        max_neighbors: int = 50,
        max_cell_images_per_dim: int = 5,
        lattice_limit_density: float = MATTERGEN_LATTICE_LIMIT_DENSITY,
    ) -> None:
        if not callable(score_callback):
            raise TypeError("score_callback must be callable.")
        if len(condition_names) != len(set(condition_names)):
            raise ValueError("condition_names must be unique.")
        if unsupported := set(condition_names).difference(MATTERGEN_CONDITION_FAMILY):
            raise ValueError(
                f"Unsupported MatterGen condition names: {sorted(unsupported)!r}."
            )
        if cutoff <= 0.0 or max_neighbors <= 0 or max_cell_images_per_dim <= 0:
            raise ValueError("graph construction limits must be positive.")
        if lattice_limit_density <= 0.0:
            raise ValueError("lattice_limit_density must be positive.")
        self._score_callback = score_callback
        self._condition_names = tuple(condition_names)
        self._cutoff = cutoff
        self._max_neighbors = max_neighbors
        self._max_cell_images_per_dim = max_cell_images_per_dim
        self._lattice_limit_density = lattice_limit_density

    def sample(
        self,
        num_atoms: torch.Tensor,
        *,
        condition_values: Mapping[str, torch.Tensor] | None = None,
        guidance_scale: float = 0.0,
        seed: int | None = None,
    ) -> MatterGenSampleBatch:
        """Draw source-scheduled crystal samples for requested atom counts.

        ``condition_values`` contains exactly the raw exported condition inputs.
        A non-``None`` ``seed`` supplies a private CPU Torch generator; omitting
        it deliberately uses the source-compatible global Torch RNG behavior.
        """
        _validate_num_atoms(num_atoms)
        supplied_conditions = dict(condition_values or {})
        _validate_conditions(supplied_conditions, self._condition_names, len(num_atoms))
        condition_values = _complete_condition_values(
            supplied_conditions,
            self._condition_names,
            len(num_atoms),
        )
        if not math.isfinite(guidance_scale):
            raise ValueError("guidance_scale must be finite.")

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)

        with torch.no_grad():
            state = self._sample_prior(num_atoms, generator)
            dt = -torch.tensor(  # Source uses CPU float32 for a CPU host state.
                (1.0 - _EPS_T) / (MATTERGEN_SAMPLING_STEPS - 1),
                dtype=torch.float32,
                device="cpu",
            )
            # Source ``torch.linspace(T, eps_t, N)`` is float32 and includes
            # both ends.  Its D3PM conversion therefore remains coupled to N=1000.
            timesteps = torch.linspace(1.0, _EPS_T, MATTERGEN_SAMPLING_STEPS, device="cpu")
            final_mean = state
            for timestep in timesteps:
                t = torch.full((len(num_atoms),), timestep, dtype=torch.float32, device="cpu")

                # Correctors update positions and cells from the same score
                # evaluation, first positions then lattice, as ``apply`` does.
                score = self._guided_score(
                    state,
                    t,
                    condition_values,
                    guidance_scale,
                    supplied_condition_names=frozenset(supplied_conditions),
                )
                corrected_pos, _ = self._wrapped_langevin(
                    state.fractional_coordinates,
                    score.coordinate_score,
                    t,
                    dt,
                    state.batch,
                    snr=0.4,
                    generator=generator,
                )
                corrected_cell, _ = self._lattice_langevin(
                    state.cell, score.lattice_score, t, dt, generator=generator
                )
                state = _State(
                    atomic_numbers=state.atomic_numbers,
                    fractional_coordinates=corrected_pos,
                    cell=corrected_cell,
                    num_atoms=state.num_atoms,
                )

                # Predictors recompute the score after both corrector updates.
                score = self._guided_score(
                    state,
                    t,
                    condition_values,
                    guidance_scale,
                    supplied_condition_names=frozenset(supplied_conditions),
                )
                predicted_pos, mean_pos = self._wrapped_ancestral(
                    state.fractional_coordinates,
                    score.coordinate_score,
                    t,
                    dt,
                    state.num_atoms,
                    state.batch,
                    generator,
                )
                predicted_cell, mean_cell = self._lattice_ancestral(
                    state.cell, score.lattice_score, t, dt, state.num_atoms, generator
                )
                predicted_atoms, mean_atoms = self._d3pm_ancestral(
                    state.atomic_numbers, score.atom_logits, t, state.batch, generator
                )
                state = _State(
                    atomic_numbers=predicted_atoms,
                    fractional_coordinates=predicted_pos,
                    cell=predicted_cell,
                    num_atoms=state.num_atoms,
                )
                final_mean = _State(
                    atomic_numbers=mean_atoms,
                    fractional_coordinates=mean_pos,
                    cell=mean_cell,
                    num_atoms=state.num_atoms,
                )

        result = MatterGenSampleBatch(
            atomic_numbers=final_mean.atomic_numbers,
            fractional_coordinates=final_mean.fractional_coordinates,
            cell=final_mean.cell,
            num_atoms=final_mean.num_atoms,
        )
        # Match the source's final Structure creation with a dependency-free,
        # fail-closed structural gate.  A caller may then serialize ``crystals``.
        result.crystals()
        return result

    def _sample_prior(
        self, num_atoms: torch.Tensor, generator: torch.Generator | None
    ) -> _State:
        batch = torch.repeat_interleave(
            torch.arange(len(num_atoms), device=num_atoms.device), num_atoms
        )
        atom_count_scale = num_atoms.to(torch.float32).pow(-1.0 / 3.0)[batch, None]
        # LatticeVPSDE.prior_sampling: symmetric IID noise around the diagonal
        # density-derived limit mean, with n^(2/3) * 0.25 elementwise variance.
        # MultiCorruption sorts fields, so the source consumes cell noise before
        # the wrapped VE position prior (atomic_numbers has no random draw).
        cell_noise = _symmetric_noise(_randn((len(num_atoms), 3, 3), generator))
        limit_mean = _lattice_limit_mean(num_atoms, self._lattice_limit_density)
        limit_var = _lattice_limit_var(num_atoms)
        cell = cell_noise * limit_var.sqrt() + limit_mean
        # NumAtomsVarianceAdjustedWrappedVESDE.prior_sampling: wrapped N(0,
        # (5 / n^(1/3))^2) fractional coordinates.
        fractional = torch.remainder(
            _randn((int(num_atoms.sum()), 3), generator) * 5.0 * atom_count_scale, 1.0
        )
        return _State(
            atomic_numbers=torch.full(
                (int(num_atoms.sum()),),
                _D3PM_MASK_CLASS + 1,
                dtype=torch.long,
                device="cpu",
            ),
            fractional_coordinates=fractional,
            cell=cell,
            num_atoms=num_atoms.clone(),
        )

    def _guided_score(
        self,
        state: _State,
        timestep: torch.Tensor,
        condition_values: Mapping[str, torch.Tensor],
        guidance_scale: float,
        *,
        supplied_condition_names: frozenset[str] | None = None,
    ) -> MatterGenScoreOutputs:
        if supplied_condition_names is None:
            supplied_condition_names = frozenset(condition_values)
        conditional = {
            name: torch.full(
                (len(state.num_atoms),),
                name not in supplied_condition_names,
                dtype=torch.bool,
                device="cpu",
            )
            for name in self._condition_names
        }
        unconditional = {
            name: torch.ones(len(state.num_atoms), dtype=torch.bool, device="cpu")
            for name in self._condition_names
        }

        def score(use_unconditional: Mapping[str, torch.Tensor]) -> MatterGenScoreOutputs:
            graph = build_periodic_graph(
                state.fractional_coordinates,
                state.cell,
                state.num_atoms,
                cutoff=self._cutoff,
                max_neighbors=self._max_neighbors,
                max_cell_images_per_dim=self._max_cell_images_per_dim,
            )
            outputs = self._score_callback(
                MatterGenScoreInputs(
                    atomic_numbers=state.atomic_numbers,
                    batch=state.batch,
                    timestep=timestep,
                    graph=graph,
                    condition_values=condition_values,
                    use_unconditional=use_unconditional,
                )
            )
            _validate_score_outputs(outputs, state)
            # MatterGen's denoiser converts raw Cartesian GemNet forces to
            # fractional scores before the wrapped coordinate process.
            coordinate_score = torch.bmm(
                torch.linalg.inv(state.cell).transpose(1, 2)[state.batch],
                outputs.coordinate_score.unsqueeze(-1),
            ).squeeze(-1)
            atom_logits = _mask_atom_logits(
                outputs.atom_logits,
                condition_values.get("chemical_system"),
                use_unconditional.get("chemical_system"),
                state.batch,
            )
            return MatterGenScoreOutputs(
                atom_logits=atom_logits,
                coordinate_score=coordinate_score,
                lattice_score=outputs.lattice_score,
                energy=outputs.energy,
            )

        # ``GuidedPredictorCorrector`` avoids unnecessary model calls at 0 and
        # 1, otherwise computing unconditional + gamma*(conditional-unconditional).
        if abs(guidance_scale - 1.0) < 1e-15:
            return score(conditional)
        if abs(guidance_scale) < 1e-15:
            return score(unconditional)
        conditional_score = score(conditional)
        unconditional_score = score(unconditional)
        return MatterGenScoreOutputs(
            atom_logits=torch.lerp(
                unconditional_score.atom_logits, conditional_score.atom_logits, guidance_scale
            ),
            coordinate_score=torch.lerp(
                unconditional_score.coordinate_score,
                conditional_score.coordinate_score,
                guidance_scale,
            ),
            lattice_score=torch.lerp(
                unconditional_score.lattice_score,
                conditional_score.lattice_score,
                guidance_scale,
            ),
            energy=None,
        )

    def _wrapped_langevin(
        self,
        value: torch.Tensor,
        score: torch.Tensor,
        timestep: torch.Tensor,
        dt: torch.Tensor,
        batch: torch.Tensor,
        *,
        snr: float,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del dt  # VE's source Langevin alpha is identically one.
        noise = _randn_like(score, generator)
        grad_norm = _per_crystal_norm(score, batch, len(timestep)).mean()
        noise_norm = _per_crystal_norm(noise, batch, len(timestep)).mean()
        step_size = _langevin_step_size(snr, noise_norm, grad_norm, len(timestep))
        expanded_step = step_size[batch, None]
        mean = value + expanded_step * score
        sample = mean + torch.sqrt(expanded_step * 2.0) * noise
        return torch.remainder(sample, 1.0), torch.remainder(mean, 1.0)

    def _lattice_langevin(
        self,
        value: torch.Tensor,
        score: torch.Tensor,
        timestep: torch.Tensor,
        dt: torch.Tensor,
        *,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = _vp_alpha(timestep) ** 2 / _vp_alpha(timestep + dt) ** 2
        noise = _symmetric_noise(_randn_like(score, generator))
        grad_norm = torch.square(score).sum(dim=(1, 2)).sqrt().mean()
        noise_norm = torch.square(noise).sum(dim=(1, 2)).sqrt().mean()
        step_size = _langevin_step_size(
            0.2,
            noise_norm,
            grad_norm,
            len(timestep),
            alpha=alpha,
        )
        expanded_step = step_size[:, None, None]
        mean = value + expanded_step * score
        sample = mean + torch.sqrt(expanded_step * 2.0) * noise
        return _polar_lattice(sample), _polar_lattice(mean)

    def _wrapped_ancestral(
        self,
        value: torch.Tensor,
        score: torch.Tensor,
        timestep: torch.Tensor,
        dt: torch.Tensor,
        num_atoms: torch.Tensor,
        batch: torch.Tensor,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sigma_t = _position_sigma(timestep, batch, num_atoms)
        sigma_s = _position_sigma(timestep + dt, batch, num_atoms)
        is_time_zero = (timestep + dt)[batch] <= 0
        sigma_s[is_time_zero] = 0.0
        score_coeff = sigma_t.square() - sigma_s.square()
        std = torch.sqrt(score_coeff) * sigma_s / sigma_t
        mean = value + score_coeff * score
        sample = mean + std * _randn_like(value, generator)
        return torch.remainder(sample, 1.0), torch.remainder(mean, 1.0)

    def _lattice_ancestral(
        self,
        value: torch.Tensor,
        score: torch.Tensor,
        timestep: torch.Tensor,
        dt: torch.Tensor,
        num_atoms: torch.Tensor,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha_t = _vp_alpha(timestep)[:, None, None]
        alpha_s = _vp_alpha(timestep + dt)[:, None, None]
        limit_var = _lattice_limit_var(num_atoms)
        sigma_t = torch.sqrt((1.0 - alpha_t.square()) * limit_var)
        sigma_s = torch.sqrt((1.0 - alpha_s.square()) * limit_var)
        is_time_zero = (timestep + dt) <= 0
        sigma_s[is_time_zero] = 0.0
        alpha_t_given_s = torch.clamp(alpha_t / alpha_s, min=0.001, max=1.0)
        sigma2_t_given_s = (
            sigma_t.square() - sigma_s.square() * alpha_t.square() / alpha_s.square()
        )
        std = torch.sqrt(sigma2_t_given_s) * sigma_s / sigma_t
        std[is_time_zero] = 0.0
        x_coeff = 1.0 / alpha_t_given_s
        score_coeff = sigma2_t_given_s / alpha_t_given_s
        limit_mean = _lattice_limit_mean(num_atoms, self._lattice_limit_density)
        mean = x_coeff * value + score_coeff * score + (1.0 - x_coeff) * limit_mean
        # Source samples ``randn_like(x_coeff)`` where x_coeff is [B, 1, 1],
        # then broadcasts it through the 3x3 symmetric-noise transform.
        sample = mean + std * _symmetric_noise(_randn_like(x_coeff, generator))
        return sample, mean

    def _d3pm_ancestral(
        self,
        atomic_numbers: torch.Tensor,
        logits: torch.Tensor,
        timestep: torch.Tensor,
        batch: torch.Tensor,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        discrete_time = (timestep * (MATTERGEN_SAMPLING_STEPS - 1)).long()[batch]
        # The source computes this preliminary sample before the predict-x0
        # posterior.  Retaining it preserves the source RNG consumption order.
        _categorical(logits, generator)
        class_probs = torch.softmax(logits, dim=-1)
        state = _mask_diffusion_state(discrete_time)
        q_t = torch.empty_like(class_probs)
        q_t[:, :-1] = state[:, None] * class_probs[:, :-1]
        q_t[:, -1] = 1.0 - q_t[:, :-1].sum(dim=-1)

        beta = 1.0 / (MATTERGEN_SAMPLING_STEPS - discrete_time).to(torch.float32)
        current = atomic_numbers - 1
        transition = torch.zeros_like(class_probs)
        is_mask = current == _D3PM_MASK_CLASS
        transition[is_mask, :-1] = beta[is_mask, None]
        transition[is_mask, -1] = 1.0
        non_mask_rows = torch.nonzero(~is_mask, as_tuple=False).flatten()
        transition[non_mask_rows, current[non_mask_rows]] = 1.0 - beta[non_mask_rows]
        posterior_logits = torch.log(q_t + _D3PM_EPSILON) + torch.log(
            transition + _D3PM_EPSILON
        )
        sample = _categorical(posterior_logits, generator) + 1
        mean = torch.argmax(torch.softmax(posterior_logits, dim=-1), dim=-1) + 1
        return sample, mean


def _as_numpy(value: torch.Tensor) -> np.ndarray:
    if value.device.type != "cpu":
        raise ValueError("MatterGen ONNX Runtime callback inputs must be CPU tensors.")
    return value.detach().contiguous().numpy()


def _validate_num_atoms(num_atoms: torch.Tensor) -> None:
    if not isinstance(num_atoms, torch.Tensor) or num_atoms.dtype != torch.long:
        raise TypeError("num_atoms must be a CPU torch.int64 tensor.")
    if num_atoms.ndim != 1 or len(num_atoms) == 0 or num_atoms.device.type != "cpu":
        raise ValueError("num_atoms must be a non-empty rank-1 CPU tensor.")
    if torch.any(num_atoms < 1) or torch.any(num_atoms > MAX_ATOMS):
        raise ValueError(f"Each MatterGen crystal must contain 1 through {MAX_ATOMS} atoms.")


def _validate_geometry_inputs(
    fractional_coordinates: torch.Tensor, cell: torch.Tensor, num_atoms: torch.Tensor
) -> None:
    _validate_num_atoms(num_atoms)
    if (
        not isinstance(fractional_coordinates, torch.Tensor)
        or fractional_coordinates.dtype != torch.float32
        or fractional_coordinates.device.type != "cpu"
        or fractional_coordinates.shape != (int(num_atoms.sum()), 3)
    ):
        raise ValueError(
            "fractional_coordinates must be a CPU float32 tensor with shape [N, 3]."
        )
    if (
        not isinstance(cell, torch.Tensor)
        or cell.dtype != torch.float32
        or cell.device.type != "cpu"
        or cell.shape != (len(num_atoms), 3, 3)
    ):
        raise ValueError("cell must be a CPU float32 tensor with shape [B, 3, 3].")
    if not torch.isfinite(fractional_coordinates).all() or not torch.isfinite(cell).all():
        raise ValueError("fractional_coordinates and cell must be finite.")
    if torch.any(fractional_coordinates < 0.0) or torch.any(fractional_coordinates >= 1.0):
        raise ValueError("fractional_coordinates must be wrapped to [0, 1).")
    if torch.any(torch.linalg.det(cell) == 0):
        raise ValueError("cell must have nonzero volume while constructing a periodic graph.")


def _validate_conditions(
    values: Mapping[str, torch.Tensor], names: Sequence[str], batch_size: int
) -> None:
    unexpected = set(values).difference(names)
    if unexpected:
        raise ValueError(
            f"condition_values contains unsupported names {sorted(unexpected)!r}; "
            f"expected a subset of {tuple(names)!r}."
        )
    for name, value in values.items():
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
            raise TypeError(f"condition {name!r} must be a CPU torch tensor.")
        if value.ndim == 0 or value.shape[0] != batch_size:
            raise ValueError(f"condition {name!r} must have batch dimension {batch_size}.")
        if name == "chemical_system":
            if value.dtype != torch.float32 or value.shape != (batch_size, _D3PM_CLASSES):
                raise ValueError(
                    "chemical_system must have shape [B, 101] and dtype torch.float32."
                )
            if (
                not torch.all(value.eq(0.0) | value.eq(1.0))
                or torch.any(value[:, 0].ne(0.0))
                or torch.any(value.sum(dim=1).eq(0.0))
            ):
                raise ValueError(
                    "chemical_system must be a non-empty one-based binary multihot."
                )
            allowed = torch.zeros(_D3PM_CLASSES, dtype=torch.bool, device="cpu")
            allowed[torch.tensor(SELECTED_ATOMIC_NUMBERS, dtype=torch.long, device="cpu")] = (
                True
            )
            if torch.any(value[:, ~allowed].ne(0.0)):
                raise ValueError(
                    "chemical_system contains an element outside MatterGen's allowlist."
                )
        elif name == "space_group":
            if value.dtype != torch.long or value.shape != (batch_size,):
                raise ValueError("space_group must have shape [B] and dtype torch.int64.")
            if torch.any(value < 1) or torch.any(value > 230):
                raise ValueError("space_group must be in the inclusive range [1, 230].")
        elif (
            value.dtype != torch.float32
            or value.shape != (batch_size,)
            or not torch.isfinite(value).all()
        ):
            raise ValueError(
                f"Scalar condition {name!r} must be a finite float32 tensor with shape [B]."
            )
        elif name in {"dft_bulk_modulus", "ml_bulk_modulus"} and torch.any(value <= 0.0):
            raise ValueError(
                f"Scalar condition {name!r} must be strictly positive for log10 scaling."
            )


def _complete_condition_values(
    values: Mapping[str, torch.Tensor], names: Sequence[str], batch_size: int
) -> dict[str, torch.Tensor]:
    """Fill absent source conditions with shape-valid values for unconditional ONNX paths."""
    completed = dict(values)
    for name in names:
        if name in completed:
            continue
        if name == "chemical_system":
            # The selector keeps this placeholder out of both property and
            # species-mask semantics; hydrogen simply makes it source-shaped.
            placeholder = torch.zeros((batch_size, _D3PM_CLASSES), dtype=torch.float32)
            placeholder[:, 1] = 1.0
        elif name == "space_group":
            placeholder = torch.ones(batch_size, dtype=torch.long)
        else:
            # Positive placeholders also avoid evaluating log10(0) in the
            # graph's unused conditional branch for bulk-modulus adapters.
            placeholder = torch.ones(batch_size, dtype=torch.float32)
        completed[name] = placeholder
    return completed


def _validate_score_outputs(outputs: MatterGenScoreOutputs, state: _State) -> None:
    if not isinstance(outputs, MatterGenScoreOutputs):
        raise TypeError("score_callback must return MatterGenScoreOutputs.")
    expected = {
        "atom_logits": (int(state.num_atoms.sum()), _D3PM_CLASSES),
        "coordinate_score": (int(state.num_atoms.sum()), 3),
        "lattice_score": (len(state.num_atoms), 3, 3),
    }
    for name, shape in expected.items():
        value = getattr(outputs, name)
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            raise TypeError(f"score_callback {name} must be a torch.float32 tensor.")
        if value.device.type != "cpu" or tuple(value.shape) != shape:
            raise ValueError(f"score_callback {name} must have CPU shape {shape}.")
        if not torch.isfinite(value).all():
            raise ValueError(f"score_callback {name} must be finite.")


def _periodic_image_offsets(
    cell: torch.Tensor, cutoff: float, max_cell_images_per_dim: int
) -> torch.Tensor:
    cross_a2a3 = torch.cross(cell[:, 1], cell[:, 2], dim=-1)
    volume = torch.sum(cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)
    reciprocal_norms = (
        torch.linalg.vector_norm(cross_a2a3 / volume, dim=-1),
        torch.linalg.vector_norm(torch.cross(cell[:, 2], cell[:, 0], dim=-1) / volume, dim=-1),
        torch.linalg.vector_norm(torch.cross(cell[:, 0], cell[:, 1], dim=-1) / volume, dim=-1),
    )
    repetitions = [
        min(int(torch.ceil(cutoff * reciprocal_norm).max()), max_cell_images_per_dim)
        for reciprocal_norm in reciprocal_norms
    ]
    return torch.cartesian_prod(
        *[
            torch.arange(-repetition, repetition + 1, dtype=torch.float32, device="cpu")
            for repetition in repetitions
        ]
    )


def _radius_graph_pbc(
    cartesian: torch.Tensor,
    cell: torch.Tensor,
    num_atoms: torch.Tensor,
    image_offsets: torch.Tensor,
    cutoff: float,
    max_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Port ``ocp_graph_utils.radius_graph_pbc`` through neighbor truncation."""
    index1_parts: list[torch.Tensor] = []
    index2_parts: list[torch.Tensor] = []
    for offset, count in zip(
        torch.cat([num_atoms.new_zeros(1), num_atoms.cumsum(0)[:-1]]), num_atoms
    ):
        local = torch.arange(int(count), dtype=torch.long, device="cpu") + offset
        # Source creates pairs with index1 as target and index2 as source.
        index1_parts.append(local.repeat_interleave(int(count)))
        index2_parts.append(local.repeat(int(count)))
    index1 = torch.cat(index1_parts).repeat_interleave(len(image_offsets))
    index2 = torch.cat(index2_parts).repeat_interleave(len(image_offsets))
    offsets = image_offsets.repeat(int(num_atoms.square().sum()), 1)
    pair_cells = torch.repeat_interleave(
        torch.repeat_interleave(cell, num_atoms.square(), dim=0),
        len(image_offsets),
        dim=0,
    )
    # The OCP implementation applies the image displacement to index2 before
    # measuring index1 - index2; the later GemNet V_st convention negates it.
    shifted_source = cartesian[index2] + torch.einsum("ei,eij->ej", offsets, pair_cells)
    distance_squared = torch.square(cartesian[index1] - shifted_source).sum(dim=-1)
    mask = (distance_squared <= cutoff * cutoff) & (distance_squared > 0.0001)
    index1 = index1[mask]
    index2 = index2[mask]
    offsets = offsets[mask]
    distance_squared = distance_squared[mask]

    neighbor_mask, neighbors = _max_neighbors_mask(
        num_atoms, index1, distance_squared, max_neighbors
    )
    index1 = index1[neighbor_mask]
    index2 = index2[neighbor_mask]
    offsets = offsets[neighbor_mask]
    return torch.stack((index2, index1)), offsets, neighbors


def _max_neighbors_mask(
    num_atoms: torch.Tensor,
    target: torch.Tensor,
    distance_squared: torch.Tensor,
    max_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_total_atoms = int(num_atoms.sum())
    counts = torch.bincount(target, minlength=num_total_atoms)
    thresholded = counts.clamp(max=max_neighbors)
    atom_batch = torch.repeat_interleave(torch.arange(len(num_atoms), device="cpu"), num_atoms)
    neighbors = torch.zeros(len(num_atoms), dtype=torch.long, device="cpu")
    neighbors.scatter_add_(0, atom_batch, thresholded)
    if target.numel() == 0 or int(counts.max()) <= max_neighbors:
        return torch.ones_like(target, dtype=torch.bool), neighbors

    # ``get_max_neighbors_mask`` writes the target-sorted candidate distances
    # into a dense matrix, sorts each target row, then retains its original
    # candidate order through a Boolean mask.
    max_count = int(counts.max())
    starts = torch.cumsum(counts, dim=0) - counts
    dense = torch.full(
        (num_total_atoms, max_count), float("inf"), dtype=torch.float32, device="cpu"
    )
    columns = torch.arange(len(target), device="cpu") - torch.repeat_interleave(starts, counts)
    dense[target, columns] = distance_squared
    _, sorted_columns = torch.sort(dense, dim=1)
    retained = sorted_columns[:, :max_neighbors] + starts[:, None]
    selected = retained[
        torch.isfinite(torch.gather(dense, 1, sorted_columns[:, :max_neighbors]))
    ]
    mask = torch.zeros(len(target), dtype=torch.bool, device="cpu")
    mask[selected] = True
    return mask, neighbors


def _reorder_symmetric_edges(
    edge_index: torch.Tensor,
    cell_offsets: torch.Tensor,
    neighbors: torch.Tensor,
    distances: torch.Tensor,
    edge_direction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    source, target = edge_index
    earlier_cell = (
        (cell_offsets[:, 0] < 0)
        | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] < 0))
        | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] == 0) & (cell_offsets[:, 2] < 0))
    )
    mask = (source < target) | ((source == target) & earlier_cell)
    directed_edge_index = edge_index[:, mask]
    directed_offsets = cell_offsets[mask]
    directed_distances = distances[mask]
    directed_directions = edge_direction[mask]

    edge_batch = torch.repeat_interleave(
        torch.arange(len(neighbors), device="cpu"), neighbors
    )[mask]
    symmetric_neighbors = 2 * torch.bincount(edge_batch, minlength=len(neighbors))
    count_per_image = symmetric_neighbors // 2
    directed_total = len(directed_offsets)
    directed_starts = torch.cumsum(count_per_image, dim=0) - count_per_image
    reorder_parts = [
        torch.cat(
            [
                torch.arange(start, start + count, device="cpu"),
                torch.arange(
                    directed_total + start, directed_total + start + count, device="cpu"
                ),
            ]
        )
        for start, count in zip(
            directed_starts.tolist(), count_per_image.tolist(), strict=True
        )
        if count > 0
    ]
    reorder = (
        torch.cat(reorder_parts)
        if reorder_parts
        else torch.empty(0, dtype=torch.long, device="cpu")
    )
    edge_index_cat = torch.cat(
        [directed_edge_index, torch.stack([directed_edge_index[1], directed_edge_index[0]])],
        dim=1,
    )
    return (
        edge_index_cat[:, reorder],
        torch.cat([directed_offsets, -directed_offsets], dim=0)[reorder],
        symmetric_neighbors,
        torch.cat([directed_distances, directed_distances], dim=0)[reorder],
        torch.cat([directed_directions, -directed_directions], dim=0)[reorder],
    )


def _symmetric_edge_swaps(neighbors: torch.Tensor) -> torch.Tensor:
    swaps: list[torch.Tensor] = []
    start = 0
    for count in neighbors.tolist():
        half = count // 2
        if half:
            swaps.append(
                torch.cat(
                    [
                        torch.arange(start + half, start + count, device="cpu"),
                        torch.arange(start, start + half, device="cpu"),
                    ]
                )
            )
        start += count
    return torch.cat(swaps) if swaps else torch.empty(0, dtype=torch.long, device="cpu")


def _triplets(edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Port ``SparseTensor(row=target, col=source)[target]`` deterministically."""
    source, target = edge_index
    triplet_ba: list[torch.Tensor] = []
    triplet_ca: list[torch.Tensor] = []
    ragged: list[torch.Tensor] = []
    for ca in range(edge_index.shape[1]):
        candidates = torch.nonzero(target == target[ca], as_tuple=False).flatten()
        # torch_sparse stores CSR rows by source column; retain edge-id order
        # for periodic duplicate columns, which is the source insertion order.
        candidate_order = torch.argsort(source[candidates], stable=True)
        candidates = candidates[candidate_order]
        candidates = candidates[candidates != ca]
        triplet_ba.append(candidates)
        triplet_ca.append(torch.full((len(candidates),), ca, dtype=torch.long, device="cpu"))
        ragged.append(torch.arange(len(candidates), dtype=torch.long, device="cpu"))
    if not triplet_ba:
        empty = torch.empty(0, dtype=torch.long, device="cpu")
        return empty, empty, empty
    return torch.cat(triplet_ba), torch.cat(triplet_ca), torch.cat(ragged)


def _mask_atom_logits(
    logits: torch.Tensor,
    chemical_system: torch.Tensor | None,
    use_unconditional: torch.Tensor | None,
    batch: torch.Tensor,
) -> torch.Tensor:
    # MatterGen's ``mask_disallowed_elements`` treats score logits as zero-based
    # atomic numbers and reserves the final 101st class for the absorbing mask.
    selected = torch.tensor(SELECTED_ATOMIC_NUMBERS, dtype=torch.long, device="cpu")
    keep = torch.zeros((1, _D3PM_CLASSES), dtype=torch.float32, device="cpu")
    keep[0, selected - 1] = 1.0
    masked = logits + (1.0 - keep) * -1e10
    if chemical_system is None:
        return masked
    if use_unconditional is None:
        raise ValueError("chemical_system requires its use_unconditional selector.")
    chemical_keep = torch.zeros(
        (len(chemical_system), _D3PM_CLASSES), dtype=torch.float32, device="cpu"
    )
    chemical_keep[:, :-1] = chemical_system[:, 1:]
    keep = torch.where(
        use_unconditional[:, None],
        torch.ones((len(chemical_system), 1), dtype=torch.float32, device="cpu"),
        chemical_keep,
    )
    return masked + (1.0 - keep[batch]) * -1e10


def _randn(shape: tuple[int, ...], generator: torch.Generator | None) -> torch.Tensor:
    return torch.randn(shape, dtype=torch.float32, device="cpu", generator=generator)


def _randn_like(value: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    return torch.randn(value.shape, dtype=value.dtype, device="cpu", generator=generator)


def _categorical(logits: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    if generator is None:
        return torch.distributions.Categorical(logits=logits).sample()
    return torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator).squeeze(-1)


def _mask_diffusion_state(timestep: torch.Tensor) -> torch.Tensor:
    """Index the source-materialized MaskDiffusion cumulative state."""
    return _D3PM_STATE[timestep]


def _position_sigma(
    timestep: torch.Tensor, batch: torch.Tensor, num_atoms: torch.Tensor
) -> torch.Tensor:
    sigma = 0.01 * (5.0 / 0.01) ** timestep
    return (sigma * num_atoms.to(torch.float32).pow(-1.0 / 3.0))[batch, None]


def _vp_alpha(timestep: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.25 * timestep.square() * (20.0 - 0.1) - 0.5 * timestep * 0.1)


def _lattice_limit_mean(num_atoms: torch.Tensor, density: float) -> torch.Tensor:
    return torch.pow(
        torch.eye(3, device="cpu").expand(len(num_atoms), 3, 3)
        * num_atoms.to(torch.float32)[:, None, None]
        / density,
        1.0 / 3.0,
    )


def _lattice_limit_var(num_atoms: torch.Tensor) -> torch.Tensor:
    return num_atoms.to(torch.float32)[:, None, None].expand(-1, 3, 3).pow(2.0 / 3.0) * 0.25


def _symmetric_noise(noise: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(3, device="cpu")[None]
    return (1.0 / math.sqrt(2.0)) * (1.0 - eye) * (noise + noise.transpose(1, 2)) + eye * noise


def _polar_lattice(lattice: torch.Tensor) -> torch.Tensor:
    # ``compute_lattice_polar_decomposition`` projects corrector updates to the
    # rotation-equivalent symmetric lattice representation used by MatterGen.
    w, singular_values, v_transpose = torch.linalg.svd(lattice)
    v = v_transpose.transpose(1, 2)
    orthogonal = w @ v_transpose
    positive = v @ torch.diag_embed(singular_values) @ v_transpose
    return orthogonal @ positive @ orthogonal.transpose(1, 2)


def _per_crystal_norm(
    score: torch.Tensor, batch: torch.Tensor, batch_size: int
) -> torch.Tensor:
    norms = torch.square(score).sum(dim=1)
    summed = torch.zeros(batch_size, dtype=score.dtype, device="cpu")
    summed.scatter_add_(0, batch, norms)
    return torch.sqrt(summed)


def _langevin_step_size(
    snr: float,
    noise_norm: torch.Tensor,
    grad_norm: torch.Tensor,
    batch_size: int,
    *,
    alpha: torch.Tensor | None = None,
) -> torch.Tensor:
    # Both released correctors explicitly take their configured cap when the
    # aggregate score is zero; this is not an error or a substitute schedule.
    if not bool(grad_norm):
        return torch.full(
            (batch_size,),
            _LANGEVIN_MAX_STEP_SIZE,
            dtype=torch.float32,
            device="cpu",
        )
    step_size = torch.full(
        (batch_size,),
        float((snr * noise_norm / grad_norm) ** 2 * 2.0),
        dtype=torch.float32,
        device="cpu",
    )
    if alpha is not None:
        step_size *= alpha
    return torch.clamp(step_size, max=_LANGEVIN_MAX_STEP_SIZE)
