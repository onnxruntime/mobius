# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Task wiring for the host-orchestrated MatterGen GemNet-T score core."""

from __future__ import annotations

import json
from typing import ClassVar

import onnx_ir as ir
from onnxscript import nn

from mobius._export_report import ComponentExportDisposition, ComponentExportReport
from mobius._model_package import ModelPackage
from mobius.integrations.mattergen._configs import MatterGenConfig
from mobius.integrations.mattergen._contract import HOST_OWNED_STEPS
from mobius.integrations.mattergen._contract import (
    MAX_ATOMS,
    OFFICIAL_CHECKPOINT_CONDITIONS,
    SELECTED_ATOMIC_NUMBERS,
)
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class MatterGenScoreTask(ModelTask):
    """Build the pure neural score stage of the MatterGen crystal generator.

    The host must rebuild MatterGen's source-ordered periodic graph before every
    invocation.  Inputs include its ragged graph and triplet tensors rather
    than fractional positions/cells, because data-dependent periodic image
    enumeration and neighbor sorting cannot be represented faithfully by a
    portable dynamic-shape ONNX graph.
    """

    model_roles: ClassVar[dict[str, str]] = {"model": "encoder"}

    def build(self, module: nn.Module, config: MatterGenConfig) -> ModelPackage:
        """Build an encoder-role ONNX score graph with config-specific condition ports."""
        if not isinstance(config, MatterGenConfig):
            raise TypeError("MatterGenScoreTask requires a MatterGenConfig.")
        graph, builder = _make_graph("mattergen_score")
        atoms = "atoms"
        crystals = "crystals"
        edges = "edges"
        triplets = "triplets"

        atomic_numbers = builder.input(
            "atomic_numbers", dtype=ir.DataType.INT64, shape=[atoms]
        )
        batch = builder.input("batch", dtype=ir.DataType.INT64, shape=[atoms])
        # The source basis and fitted scalar encoders are always float32, even
        # when a future export introduces a separately assessed compute dtype.
        timestep = builder.input("timestep", dtype=ir.DataType.FLOAT, shape=[crystals])
        edge_index = builder.input(
            "edge_index", dtype=ir.DataType.INT64, shape=[2, edges]
        )
        edge_distance = builder.input(
            "edge_distance", dtype=ir.DataType.FLOAT, shape=[edges]
        )
        edge_direction = builder.input(
            "edge_direction", dtype=ir.DataType.FLOAT, shape=[edges, 3]
        )
        edge_lattice_cosines = builder.input(
            "edge_lattice_cosines", dtype=ir.DataType.FLOAT, shape=[edges, 3]
        )
        id_swap = builder.input("id_swap", dtype=ir.DataType.INT64, shape=[edges])
        id3_ba = builder.input("id3_ba", dtype=ir.DataType.INT64, shape=[triplets])
        id3_ca = builder.input("id3_ca", dtype=ir.DataType.INT64, shape=[triplets])
        id3_ragged_idx = builder.input(
            "id3_ragged_idx", dtype=ir.DataType.INT64, shape=[triplets]
        )

        condition_values: dict[str, ir.Value] = {}
        condition_masks: dict[str, ir.Value] = {}
        for spec in config.condition_input_specs:
            condition_values[spec.name] = builder.input(
                f"condition.{spec.name}",
                dtype=(
                    ir.DataType.INT64
                    if spec.kind == "space_group_index"
                    else ir.DataType.FLOAT
                ),
                shape=[crystals, *spec.input_shape_suffix],
            )
            condition_masks[spec.name] = builder.input(
                f"condition.{spec.name}.use_unconditional",
                dtype=ir.DataType.BOOL,
                shape=[crystals],
            )

        atom_logits, coordinate_score, lattice_score, energy = module(
            builder.op,
            atomic_numbers,
            batch,
            timestep,
            edge_index,
            edge_distance,
            edge_direction,
            edge_lattice_cosines,
            id_swap,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
            condition_values,
            condition_masks,
        )
        atom_logits.shape = ir.Shape([atoms, 101])
        coordinate_score.shape = ir.Shape([atoms, 3])
        lattice_score.shape = ir.Shape([crystals, 3, 3])
        energy.shape = ir.Shape([crystals, 1])
        builder.add_output(atom_logits, "atom_logits")
        builder.add_output(coordinate_score, "coordinate_score")
        builder.add_output(lattice_score, "lattice_score")
        # Denoiser inference does not consume GemNet's energy result. Exposing
        # it as a diagnostic output keeps every trained OutputBlock parameter
        # reachable and makes strict checkpoint routing auditable.
        builder.add_output(energy, "energy")

        model = _make_model(graph)
        model.metadata_props.update(
            {
                "mobius.model_type": "mattergen",
                "mobius.source_model": config.model_id,
                "mobius.source_revision": config.revision,
                "mobius.source_commit": config.source_commit,
                "mobius.task": "mattergen-score",
                "mobius.checkpoint_family": config.variant,
                "mobius.max_atoms": str(MAX_ATOMS),
                "mobius.sampling_atomic_numbers": json.dumps(SELECTED_ATOMIC_NUMBERS),
                "mobius.official_checkpoint_conditions": json.dumps(
                    OFFICIAL_CHECKPOINT_CONDITIONS,
                    sort_keys=True,
                ),
                "mobius.coordinate_convention": (
                    "host uses row-vector cells: cartesian = fractional @ cell; "
                    "coordinate_score is Cartesian"
                ),
                "mobius.periodic_graph_abi": (
                    "host supplies source-ordered edge_index, edge_distance, "
                    "V_st edge_direction, edge_lattice_cosines, id_swap, and triplet ids"
                ),
                "mobius.host_orchestration": json.dumps(HOST_OWNED_STEPS),
                "mobius.runtime_support": (
                    "ONNX score core only; ONNX Runtime GenAI cannot execute MatterGen's "
                    "periodic graph construction, stochastic scheduler, or crystal validation."
                ),
            }
        )
        package = ModelPackage({"model": model}, config=config)
        package.export_report = ComponentExportReport.create(
            (
                ComponentExportDisposition(
                    name="crystal_validation",
                    route="MatterGen host postprocessing",
                    requested=True,
                    discovered=True,
                    support="deferred",
                    output="omitted",
                    blocker_category="host-scientific-runtime",
                    reason="Pymatgen Structure/CIF validation is not a neural ONNX operation.",
                    impact="The ONNX package cannot itself claim to generate a valid crystal artifact.",
                    remediation="Validate final wrapped fractional coordinates and cell in a host runtime.",
                ),
                ComponentExportDisposition(
                    name="periodic_graph",
                    route="MatterGen host preprocessing",
                    requested=True,
                    discovered=True,
                    support="deferred",
                    output="omitted",
                    blocker_category="dynamic-ragged-pbc",
                    reason=(
                        "Source-faithful periodic image enumeration, neighbor sorting, "
                        "symmetric reordering, and sparse triplets are data-dependent."
                    ),
                    impact="The score graph requires the documented host graph ABI per evaluation.",
                    remediation="Build graph tensors with the pinned MatterGen v1.0.3 semantics.",
                ),
                ComponentExportDisposition(
                    name="sampling_scheduler",
                    route="MatterGen host sampling",
                    requested=True,
                    discovered=True,
                    support="deferred",
                    output="omitted",
                    blocker_category="stochastic-host-loop",
                    reason=(
                        "D3PM, wrapped VE/VP updates, RNG, classifier-free guidance, "
                        "and lattice projection are source host-loop behavior."
                    ),
                    impact="This package is not an end-to-end crystal generator.",
                    remediation="Run the pinned-source scheduler around repeated score-core calls.",
                ),
                ComponentExportDisposition(
                    name="score_core",
                    route="standard ONNX GemNet-T encoder graph",
                    requested=True,
                    discovered=True,
                    support="supported",
                    output="exported",
                    runtime_validation_status="validated",
                    evidence_id="mattergen-score-core-ort",
                ),
            ),
            end_to_end_runnable=False,
        )
        return package
