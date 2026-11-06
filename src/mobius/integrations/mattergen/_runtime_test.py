# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import numpy as np
import pytest
import torch

from mobius.integrations.mattergen import (
    MatterGenHostSampler,
    MatterGenScoreInputs,
    MatterGenScoreOutputs,
    build_periodic_graph,
    create_onnxruntime_score_callback,
)
from mobius.integrations.mattergen._runtime import (
    _LANGEVIN_MAX_STEP_SIZE,
    MATTERGEN_SAMPLING_STEPS,
    _langevin_step_size,
    _State,
)


class TestMatterGenPeriodicGraph:
    def test_matches_source_self_image_edge_and_triplet_order(self) -> None:
        """Exercise the source's PBC/reorder/triplet sequence without PyG."""
        graph = build_periodic_graph(
            torch.zeros((1, 3), dtype=torch.float32),
            torch.eye(3, dtype=torch.float32).unsqueeze(0) * 4.0,
            torch.tensor([1], dtype=torch.long),
            cutoff=4.1,
        )

        torch.testing.assert_close(
            graph.edge_index,
            torch.zeros((2, 6), dtype=torch.long),
        )
        torch.testing.assert_close(graph.edge_distance, torch.full((6,), 4.0))
        torch.testing.assert_close(
            graph.edge_direction,
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0],
                ]
            ),
        )
        torch.testing.assert_close(graph.edge_lattice_cosines, graph.edge_direction)
        torch.testing.assert_close(
            graph.id_swap,
            torch.tensor([3, 4, 5, 0, 1, 2]),
        )
        torch.testing.assert_close(
            graph.id3_ba,
            torch.tensor(
                [
                    1,
                    2,
                    3,
                    4,
                    5,
                    0,
                    2,
                    3,
                    4,
                    5,
                    0,
                    1,
                    3,
                    4,
                    5,
                    0,
                    1,
                    2,
                    4,
                    5,
                    0,
                    1,
                    2,
                    3,
                    5,
                    0,
                    1,
                    2,
                    3,
                    4,
                ]
            ),
        )
        torch.testing.assert_close(
            graph.id3_ca,
            torch.arange(6, dtype=torch.long).repeat_interleave(5),
        )
        torch.testing.assert_close(
            graph.id3_ragged_idx,
            torch.arange(5, dtype=torch.long).repeat(6),
        )

    def test_rejects_an_empty_source_graph(self) -> None:
        with pytest.raises(ValueError, match="empty periodic radius graph"):
            build_periodic_graph(
                torch.zeros((1, 3), dtype=torch.float32),
                torch.eye(3, dtype=torch.float32).unsqueeze(0) * 20.0,
                torch.tensor([1], dtype=torch.long),
            )


class TestMatterGenScoreCallback:
    def test_onnxruntime_adapter_uses_the_exported_named_abi(self) -> None:
        graph = build_periodic_graph(
            torch.zeros((1, 3), dtype=torch.float32),
            torch.eye(3, dtype=torch.float32).unsqueeze(0) * 4.0,
            torch.tensor([1], dtype=torch.long),
            cutoff=4.1,
        )
        captured: dict[str, np.ndarray] = {}

        class Session:
            def run(self, output_names, input_feed):
                assert output_names == [
                    "atom_logits",
                    "coordinate_score",
                    "lattice_score",
                    "energy",
                ]
                captured.update(input_feed)
                return [
                    np.zeros((1, 101), dtype=np.float32),
                    np.zeros((1, 3), dtype=np.float32),
                    np.zeros((1, 3, 3), dtype=np.float32),
                    np.zeros((1, 1), dtype=np.float32),
                ]

        callback = create_onnxruntime_score_callback(Session())
        outputs = callback(
            MatterGenScoreInputs(
                atomic_numbers=torch.tensor([101], dtype=torch.long),
                batch=torch.tensor([0], dtype=torch.long),
                timestep=torch.tensor([1.0], dtype=torch.float32),
                graph=graph,
                condition_values={
                    "chemical_system": torch.zeros((1, 101), dtype=torch.float32),
                },
                use_unconditional={
                    "chemical_system": torch.tensor([True], dtype=torch.bool),
                },
            )
        )

        assert set(captured) == {
            "atomic_numbers",
            "batch",
            "timestep",
            "edge_index",
            "edge_distance",
            "edge_direction",
            "edge_lattice_cosines",
            "id_swap",
            "id3_ba",
            "id3_ca",
            "id3_ragged_idx",
            "condition.chemical_system",
            "condition.chemical_system.use_unconditional",
        }
        assert outputs.atom_logits.shape == (1, 101)

    def test_cfg_uses_source_conditional_then_unconditional_order(self) -> None:
        calls: list[MatterGenScoreInputs] = []

        def score(inputs: MatterGenScoreInputs) -> MatterGenScoreOutputs:
            calls.append(inputs)
            conditional = not bool(inputs.use_unconditional["chemical_system"][0])
            return MatterGenScoreOutputs(
                atom_logits=torch.full(
                    (1, 101), 10.0 if conditional else 2.0, dtype=torch.float32
                ),
                coordinate_score=torch.full(
                    (1, 3), 4.0 if conditional else 2.0, dtype=torch.float32
                ),
                lattice_score=torch.full(
                    (1, 3, 3), 4.0 if conditional else 2.0, dtype=torch.float32
                ),
            )

        sampler = MatterGenHostSampler(score, condition_names=("chemical_system",))
        chemical_system = torch.zeros((1, 101), dtype=torch.float32)
        chemical_system[0, 3] = 1.0
        state = _State(
            atomic_numbers=torch.tensor([101], dtype=torch.long),
            fractional_coordinates=torch.zeros((1, 3), dtype=torch.float32),
            cell=torch.eye(3, dtype=torch.float32).unsqueeze(0) * 4.0,
            num_atoms=torch.tensor([1], dtype=torch.long),
        )

        guided = sampler._guided_score(
            state,
            torch.tensor([1.0], dtype=torch.float32),
            {"chemical_system": chemical_system},
            guidance_scale=0.5,
        )

        assert [bool(call.use_unconditional["chemical_system"][0]) for call in calls] == [
            False,
            True,
        ]
        torch.testing.assert_close(guided.atom_logits[:, 2], torch.tensor([6.0]))
        torch.testing.assert_close(guided.coordinate_score, torch.full((1, 3), 0.75))
        torch.testing.assert_close(guided.lattice_score, torch.full((1, 3, 3), 3.0))

    def test_omitted_adapter_values_use_source_unconditional_semantics(self) -> None:
        """Conditioned checkpoints may sample unconditionally without a property value."""
        captured: list[MatterGenScoreInputs] = []

        def score(inputs: MatterGenScoreInputs) -> MatterGenScoreOutputs:
            captured.append(inputs)
            raise RuntimeError("stop after inspecting the first source score call")

        with pytest.raises(RuntimeError, match="stop after"):
            MatterGenHostSampler(score, condition_names=("ml_bulk_modulus",)).sample(
                torch.tensor([1], dtype=torch.long),
                seed=3,
            )

        assert len(captured) == 1
        assert bool(captured[0].use_unconditional["ml_bulk_modulus"][0])
        torch.testing.assert_close(
            captured[0].condition_values["ml_bulk_modulus"],
            torch.ones(1, dtype=torch.float32),
        )


class TestMatterGenHostSampler:
    def test_zero_langevin_score_uses_the_released_cap(self) -> None:
        """The source correctors define, rather than reject, a zero-score update."""
        step_size = _langevin_step_size(
            snr=0.4,
            noise_norm=torch.tensor(3.0),
            grad_norm=torch.tensor(0.0),
            batch_size=2,
        )

        torch.testing.assert_close(
            step_size,
            torch.full((2,), _LANGEVIN_MAX_STEP_SIZE, dtype=torch.float32),
        )

    def test_lattice_langevin_caps_after_vp_alpha_scaling(self) -> None:
        """Mirror LatticeLangevinDiffCorrector's cap ordering."""
        step_size = _langevin_step_size(
            snr=0.2,
            noise_norm=torch.tensor(100.0),
            grad_norm=torch.tensor(0.001),
            batch_size=2,
            alpha=torch.tensor([0.0001, 1.0], dtype=torch.float32),
        )

        torch.testing.assert_close(
            step_size,
            torch.tensor([80_000.0, _LANGEVIN_MAX_STEP_SIZE], dtype=torch.float32),
        )

    def test_rejects_invalid_source_condition_before_sampling(self) -> None:
        def score(_: MatterGenScoreInputs) -> MatterGenScoreOutputs:
            raise AssertionError("invalid conditioning must fail before a score invocation")

        with pytest.raises(ValueError, match="strictly positive"):
            MatterGenHostSampler(score, condition_names=("ml_bulk_modulus",)).sample(
                torch.tensor([1], dtype=torch.long),
                condition_values={
                    "ml_bulk_modulus": torch.tensor([0.0], dtype=torch.float32),
                },
            )

    def test_seeded_full_source_schedule_is_deterministic_and_validates_output(self) -> None:
        """Exercise all 1,000 host timesteps and the final structural gate."""
        calls = 0
        skew = torch.tensor(
            [[0.0, 100.0, 0.0], [-100.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=torch.float32,
        )

        def score(inputs: MatterGenScoreInputs) -> MatterGenScoreOutputs:
            nonlocal calls
            calls += 1
            atom_logits = torch.zeros((len(inputs.atomic_numbers), 101), dtype=torch.float32)
            atom_logits[:, 0] = 2.0
            # A high-norm anti-symmetric fixture keeps test-only Langevin
            # steps bounded without modeling a physical score field.
            lattice_score = skew.unsqueeze(0).repeat(len(inputs.timestep), 1, 1)
            return MatterGenScoreOutputs(
                atom_logits=atom_logits,
                coordinate_score=torch.full(
                    (len(inputs.atomic_numbers), 3), 5.0, dtype=torch.float32
                ),
                lattice_score=lattice_score,
            )

        sampler = MatterGenHostSampler(score, cutoff=100.0, max_neighbors=2)
        sample = sampler.sample(
            torch.tensor([1], dtype=torch.long),
            seed=19,
        )
        repeated = sampler.sample(
            torch.tensor([1], dtype=torch.long),
            seed=19,
        )

        assert calls == 4 * MATTERGEN_SAMPLING_STEPS
        torch.testing.assert_close(sample.atomic_numbers, repeated.atomic_numbers)
        torch.testing.assert_close(
            sample.fractional_coordinates, repeated.fractional_coordinates
        )
        torch.testing.assert_close(sample.cell, repeated.cell)
        assert sample.atomic_numbers.shape == (1,)
        assert sample.fractional_coordinates.shape == (1, 3)
        assert torch.all(
            (sample.fractional_coordinates >= 0.0) & (sample.fractional_coordinates < 1.0)
        )
        assert torch.linalg.det(sample.cell).item() > 0.0
        assert len(sample.crystals()) == 1
