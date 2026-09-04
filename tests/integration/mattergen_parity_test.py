# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""L3/L4 CPU parity for the pinned MatterGen GemNet-T score core.

These tests intentionally require local, immutable MatterGen artifacts rather
than downloading them.  Set ``MOBIUS_MATTERGEN_SOURCE_DIR`` to source commit
``842ffe735f7d06cec89d56aa23d9f001e1124b30`` and
``MOBIUS_MATTERGEN_MP20_CHECKPOINT`` to the official ``mp_20_base``
``last.ckpt``. Optionally set ``MOBIUS_MATTERGEN_DFT_BAND_GAP_CHECKPOINT`` to
the released ``dft_band_gap`` adapter checkpoint. The source repository's
optional PyG extensions do not publish Apple-Silicon wheels for the Torch
version used by this project. The narrowly scoped native-Torch compatibility
definitions below implement only the sum scatter, CSR segment sum, sparse-row
lookup, and Gaussian basis operations that the pinned source invokes on this
fixture. The network layers, periodic graph construction, triplet generation,
checkpoint loading, and score computation are all executed from the pinned
source tree.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import types
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from mobius import build_from_module
from mobius._testing.ort_inference import OnnxModelSession
from mobius.integrations.mattergen import (
    MatterGenConfig,
    MatterGenHostSampler,
    MatterGenModel,
    build_periodic_graph,
    create_onnxruntime_score_callback,
)
from mobius.integrations.mattergen._configs import MATTERGEN_SOURCE_COMMIT
from mobius.integrations.mattergen._weights import apply_mattergen_checkpoint
from mobius.tasks import MatterGenScoreTask

pytestmark = pytest.mark.integration

_SOURCE_DIR_ENV = "MOBIUS_MATTERGEN_SOURCE_DIR"
_MP20_CHECKPOINT_ENV = "MOBIUS_MATTERGEN_MP20_CHECKPOINT"
_DFT_BAND_GAP_CHECKPOINT_ENV = "MOBIUS_MATTERGEN_DFT_BAND_GAP_CHECKPOINT"
_GOLDEN_PATH = (
    Path(__file__).parents[2]
    / "testdata"
    / "golden"
    / "diffusion"
    / "mattergen-mp20-score.json"
)
_HOST_SAMPLE_GOLDEN_PATH = (
    Path(__file__).parents[2]
    / "testdata"
    / "golden"
    / "diffusion"
    / "mattergen-mp20-host-sample.json"
)
_RTOL = 1e-3
_ATOL = 1e-3
# The fine-tuned adapter checkpoint amplifies sub-ULP differences between
# PyTorch and ORT float32 kernels; this remains an output-level parity bound.
_ADAPTER_RTOL = 1e-2
_ADAPTER_ATOL = 5e-3


@dataclass(frozen=True)
class _SourceModules:
    """Pinned source modules used to evaluate the independent PyTorch reference."""

    gemnet: Any
    gemnet_ctrl: Any
    data_utils: Any
    atom_embedding: Any
    model_utils: Any
    property_embeddings: Any
    d3pm: Any
    d3pm_corruption: Any
    d3pm_predictors_correctors: Any


@dataclass(frozen=True)
class _Crystal:
    """A small periodic two-atom crystal in MatterGen's row-vector cell convention."""

    atomic_numbers: np.ndarray
    fractional_coordinates: np.ndarray
    cell: np.ndarray


@dataclass
class _ReferenceModel:
    """Executable pinned-source GemNet and its timestep encoder/classification head."""

    gemnet: Any
    noise_level_encoding: Any
    fc_atom: torch.nn.Linear
    adapter_property: Any | None = None
    adapter_name: str | None = None


@dataclass(frozen=True)
class _ScoreCase:
    """Host ABI feeds and the exact pinned-source outputs for one score evaluation."""

    feeds: dict[str, np.ndarray]
    outputs: dict[str, np.ndarray]


@dataclass(frozen=True)
class _SourceGraph:
    """Initial PBC graph for source evaluation plus the normalized ONNX host ABI."""

    initial_edges: torch.Tensor
    to_jimages: torch.Tensor
    num_bonds: torch.Tensor
    feeds: dict[str, np.ndarray]


@dataclass
class _Mp20Runtime:
    """Shared real-weight source references and a loaded ONNX Runtime score graph."""

    session: OnnxModelSession
    original: dict[float, _ScoreCase]
    translated: _ScoreCase
    permuted: _ScoreCase
    batched_graph: _SourceGraph
    checkpoint_sha256: str

    def close(self) -> None:
        self.session.close()


def _required_artifact(environment_variable: str) -> Path:
    """Resolve an explicit local integration artifact or skip without network access."""
    raw_path = os.environ.get(environment_variable)
    if raw_path is None:
        pytest.skip(f"set {environment_variable} to run MatterGen source parity")
    path = Path(raw_path)
    if not path.is_file() and environment_variable == _SOURCE_DIR_ENV:
        if not path.is_dir():
            pytest.skip(f"{environment_variable} is not a readable source directory: {path}")
    elif not path.is_file():
        pytest.skip(f"{environment_variable} is not a readable checkpoint: {path}")
    return path


def _source_revision(source_dir: Path) -> str:
    """Return the checked-out source revision, refusing an unpinned reference tree."""
    completed = subprocess.run(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _module_is_available(name: str) -> bool:
    """Handle persistent in-process compatibility modules without re-resolving specs."""
    return name in sys.modules or importlib.util.find_spec(name) is not None


def _install_source_dependency_compatibility() -> None:
    """Provide native-Torch equivalents only for unavailable source extension APIs.

    MatterGen's pinned source calls all of these helpers only with leading-axis
    sum reductions.  Keeping the compatibility surface this small avoids a
    second implementation of any GemNet layer in this test.
    """
    # MatterGen v1.0.3 declares NumPy <2.  NumPy 2 removed this historical
    # alias; restoring the identical stdlib module lets its source basis
    # generator run without changing any generated polynomial.
    if not hasattr(np, "math"):
        np.math = math  # type: ignore[attr-defined]

    if not _module_is_available("omegaconf"):
        omegaconf = types.ModuleType("omegaconf")

        class OmegaConf:
            """Minimal import-time resolver registration interface."""

            @staticmethod
            def register_new_resolver(*_args: object, **_kwargs: object) -> None:
                return None

        omegaconf.OmegaConf = OmegaConf
        sys.modules["omegaconf"] = omegaconf

    if not _module_is_available("torch_scatter"):
        torch_scatter = types.ModuleType("torch_scatter")

        def scatter(
            source: torch.Tensor,
            index: torch.Tensor,
            dim: int = 0,
            dim_size: int | torch.Tensor | None = None,
            reduce: str = "sum",
        ) -> torch.Tensor:
            if dim != 0 or reduce not in {"sum", "add"}:
                raise NotImplementedError(
                    "MatterGen fixture needs only leading-axis sum scatter"
                )
            if index.ndim != 1 or index.shape[0] != source.shape[0]:
                raise ValueError("source and index must share their leading dimension")
            rows = (
                int(index.max().item()) + 1
                if dim_size is None
                else int(torch.as_tensor(dim_size).item())
            )
            result = source.new_zeros((rows, *source.shape[1:]))
            return result.index_add_(0, index, source)

        def segment_coo(
            source: torch.Tensor,
            index: torch.Tensor,
            dim_size: int | torch.Tensor | None = None,
            reduce: str = "sum",
        ) -> torch.Tensor:
            return scatter(source, index, dim_size=dim_size, reduce=reduce)

        def segment_csr(
            source: torch.Tensor, indptr: torch.Tensor, reduce: str = "sum"
        ) -> torch.Tensor:
            if reduce not in {"sum", "add"}:
                raise NotImplementedError("MatterGen fixture needs only CSR segment sums")
            return torch.stack(
                [
                    source[int(start.item()) : int(end.item())].sum(dim=0)
                    for start, end in pairwise(indptr)
                ]
            )

        torch_scatter.scatter = scatter
        torch_scatter.scatter_add = scatter
        torch_scatter.segment_coo = segment_coo
        torch_scatter.segment_csr = segment_csr
        sys.modules["torch_scatter"] = torch_scatter

    if not _module_is_available("torch_sparse"):
        torch_sparse = types.ModuleType("torch_sparse")

        class _SparseStorage:
            """Storage view exposing the two accessors GemNetT.get_triplets uses."""

            def __init__(self, row: torch.Tensor, value: torch.Tensor):
                self._row = row
                self._value = value

            def row(self) -> torch.Tensor:
                return self._row

            def value(self) -> torch.Tensor:
                return self._value

        class SparseTensor:
            """Row-indexed multigraph lookup preserving MatterGen's edge ordering."""

            def __init__(
                self,
                *,
                row: torch.Tensor,
                col: torch.Tensor,
                value: torch.Tensor,
                sparse_sizes: tuple[torch.Tensor, torch.Tensor] | tuple[int, int],
            ):
                del sparse_sizes
                self._row = row
                self._col = col
                self._value = value
                self.storage = _SparseStorage(row.new_empty(0), value.new_empty(0))

            def __getitem__(self, queried_rows: torch.Tensor) -> SparseTensor:
                result_rows: list[torch.Tensor] = []
                result_values: list[torch.Tensor] = []
                for output_row, source_row in enumerate(queried_rows):
                    match = torch.nonzero(self._row == source_row, as_tuple=False).squeeze(1)
                    # torch_sparse stores COO entries in row/column order.
                    # MatterGen's id3_ba therefore follows the edge source ID
                    # within each target row, not the caller's incidental COO order.
                    match = match[torch.argsort(self._col[match], stable=True)]
                    result_rows.append(
                        torch.full(
                            (len(match),),
                            output_row,
                            dtype=queried_rows.dtype,
                            device=queried_rows.device,
                        )
                    )
                    result_values.append(self._value[match])
                result = object.__new__(SparseTensor)
                result._row = self._row
                result._col = self._col
                result._value = self._value
                result.storage = _SparseStorage(
                    torch.cat(result_rows),
                    torch.cat(result_values),
                )
                return result

            def to(self, _device: torch.device) -> SparseTensor:
                return self

        torch_sparse.SparseTensor = SparseTensor
        sys.modules["torch_sparse"] = torch_sparse

    if not _module_is_available("torch_geometric"):
        torch_geometric = types.ModuleType("torch_geometric")
        torch_geometric.__path__ = []
        pyg_data = types.ModuleType("torch_geometric.data")
        pyg_typing = types.ModuleType("torch_geometric.typing")
        pyg_utils = types.ModuleType("torch_geometric.utils")
        pyg_nn = types.ModuleType("torch_geometric.nn")
        pyg_nn.__path__ = []
        pyg_models = types.ModuleType("torch_geometric.nn.models")
        pyg_models.__path__ = []
        pyg_schnet = types.ModuleType("torch_geometric.nn.models.schnet")

        class Data:
            """Import-only Data base sufficient for source type declarations."""

            def __init__(self, **kwargs: object):
                self.__dict__.update(kwargs)

        class Batch:
            """Import-only Batch factory used while defining ChemGraphBatch."""

            def __new__(cls, _base_cls: type | None = None, **_kwargs: object):
                if _base_cls is None:
                    return super().__new__(cls)
                return type(f"{_base_cls.__name__}Batch", (_base_cls, cls), {})()

        class GaussianSmearing(torch.nn.Module):
            """Pinned PyG GaussianSmearing formula used by MatterGen's radial basis."""

            def __init__(
                self, start: float, stop: float, num_gaussians: int, **_kwargs: object
            ):
                super().__init__()
                offset = torch.linspace(start, stop, num_gaussians)
                self.register_buffer("offset", offset)
                self.coeff = -0.5 / float((offset[1] - offset[0]).square())

            def forward(self, distance: torch.Tensor) -> torch.Tensor:
                return torch.exp(
                    self.coeff * (distance.view(-1, 1) - self.offset.view(1, -1)).square()
                )

        pyg_data.Data = Data
        pyg_data.Batch = Batch
        pyg_typing.OptTensor = object
        pyg_schnet.GaussianSmearing = GaussianSmearing
        torch_geometric.data = pyg_data
        torch_geometric.typing = pyg_typing
        torch_geometric.utils = pyg_utils
        torch_geometric.nn = pyg_nn
        pyg_nn.models = pyg_models
        pyg_models.schnet = pyg_schnet
        sys.modules.update(
            {
                "torch_geometric": torch_geometric,
                "torch_geometric.data": pyg_data,
                "torch_geometric.typing": pyg_typing,
                "torch_geometric.utils": pyg_utils,
                "torch_geometric.nn": pyg_nn,
                "torch_geometric.nn.models": pyg_models,
                "torch_geometric.nn.models.schnet": pyg_schnet,
            }
        )

    if not _module_is_available("pymatgen"):
        pymatgen = types.ModuleType("pymatgen")
        pymatgen.__path__ = []
        pymatgen_core = types.ModuleType("pymatgen.core")

        class Element:
            """Import-only placeholder; this fixture supplies atomic numbers directly."""

            def __init__(self, *_args: object, **_kwargs: object):
                raise RuntimeError(
                    "MatterGen parity fixture does not construct pymatgen Elements"
                )

        pymatgen_core.Element = Element
        pymatgen.core = pymatgen_core
        sys.modules["pymatgen"] = pymatgen
        sys.modules["pymatgen.core"] = pymatgen_core

    if not _module_is_available("emmet"):
        emmet = types.ModuleType("emmet")
        emmet.__path__ = []
        emmet_core = types.ModuleType("emmet.core")
        emmet_core.__path__ = []
        emmet_material = types.ModuleType("emmet.core.material")

        class PropertyOrigin:
            """Import-only provenance type used only by source annotations."""

        emmet_material.PropertyOrigin = PropertyOrigin
        emmet.core = emmet_core
        emmet_core.material = emmet_material
        sys.modules.update(
            {
                "emmet": emmet,
                "emmet.core": emmet_core,
                "emmet.core.material": emmet_material,
            }
        )


@contextmanager
def _pinned_source_modules(source_dir: Path) -> Iterator[_SourceModules]:
    """Import all reference layers from the requested immutable source checkout."""
    if _source_revision(source_dir) != MATTERGEN_SOURCE_COMMIT:
        pytest.fail(
            f"MatterGen source must be {MATTERGEN_SOURCE_COMMIT}, got {_source_revision(source_dir)}"
        )
    _install_source_dependency_compatibility()
    stale = [
        name for name in sys.modules if name == "mattergen" or name.startswith("mattergen.")
    ]
    for name in stale:
        del sys.modules[name]
    sys.path.insert(0, str(source_dir))
    try:
        yield _SourceModules(
            gemnet=importlib.import_module("mattergen.common.gemnet.gemnet"),
            gemnet_ctrl=importlib.import_module("mattergen.common.gemnet.gemnet_ctrl"),
            data_utils=importlib.import_module("mattergen.common.utils.data_utils"),
            atom_embedding=importlib.import_module(
                "mattergen.common.gemnet.layers.embedding_block"
            ),
            model_utils=importlib.import_module("mattergen.diffusion.model_utils"),
            property_embeddings=importlib.import_module("mattergen.property_embeddings"),
            d3pm=importlib.import_module("mattergen.diffusion.d3pm.d3pm"),
            d3pm_corruption=importlib.import_module(
                "mattergen.diffusion.corruption.d3pm_corruption"
            ),
            d3pm_predictors_correctors=importlib.import_module(
                "mattergen.diffusion.d3pm.d3pm_predictors_correctors"
            ),
        )
    finally:
        sys.path.remove(str(source_dir))
        for name in [
            name
            for name in sys.modules
            if name == "mattergen" or name.startswith("mattergen.")
        ]:
            del sys.modules[name]


def _load_checkpoint(checkpoint: Path) -> tuple[dict[str, torch.Tensor], Mapping[str, object]]:
    """Safely read the official Lightning checkpoint and return its score-core state."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("MatterGen checkpoint must deserialize to a mapping")
    state_dict = payload.get("state_dict")
    config = payload.get("config")
    if not isinstance(state_dict, Mapping) or not isinstance(config, Mapping):
        raise TypeError(
            "MatterGen checkpoint must provide mapping state_dict and config values"
        )
    prefix = "diffusion_module.model."
    state = {
        name.removeprefix(prefix): value.detach().clone()
        for name, value in state_dict.items()
        if isinstance(name, str)
        and name.startswith(prefix)
        and isinstance(value, torch.Tensor)
    }
    if not state:
        raise ValueError("MatterGen checkpoint contains no score-core tensors")
    return state, config


def _gemnet_kwargs(config: MatterGenConfig, atom_embedding: Any) -> dict[str, object]:
    """Build the exact GemNet constructor argument set reflected in the Hydra config."""
    return {
        "num_targets": config.num_targets,
        "latent_dim": config.latent_dim,
        "atom_embedding": atom_embedding,
        "num_spherical": config.num_spherical,
        "num_radial": config.num_radial,
        "num_blocks": config.num_blocks,
        "emb_size_atom": config.emb_size_atom,
        "emb_size_edge": config.emb_size_edge,
        "emb_size_trip": config.emb_size_trip,
        "emb_size_rbf": config.emb_size_rbf,
        "emb_size_cbf": config.emb_size_cbf,
        "emb_size_bil_trip": config.emb_size_bil_trip,
        "num_before_skip": config.num_before_skip,
        "num_after_skip": config.num_after_skip,
        "num_concat": config.num_concat,
        "num_atom": config.num_atom,
        "regress_stress": config.regress_stress,
        "cutoff": config.cutoff,
        "max_neighbors": config.max_neighbors,
        "max_cell_images_per_dim": config.max_cell_images_per_dim,
        "otf_graph": False,
    }


def _make_reference_model(
    source: _SourceModules,
    config: MatterGenConfig,
    state: Mapping[str, torch.Tensor],
) -> _ReferenceModel:
    """Load official tensors into the independently imported source neural modules."""
    atom_embedding = source.atom_embedding.AtomEmbedding(
        config.hidden_size,
        with_mask_type=True,
    )
    gemnet = source.gemnet.GemNetT(**_gemnet_kwargs(config, atom_embedding))
    gemnet.load_state_dict(
        {
            name.removeprefix("gemnet."): value
            for name, value in state.items()
            if name.startswith("gemnet.")
        },
        strict=True,
    )
    noise_level_encoding = source.model_utils.NoiseLevelEncoding(config.hidden_size)
    noise_level_encoding.load_state_dict(
        {
            name.removeprefix("noise_level_encoding."): value
            for name, value in state.items()
            if name.startswith("noise_level_encoding.")
        },
        strict=True,
    )
    fc_atom = torch.nn.Linear(config.hidden_size, config.num_atom_types)
    fc_atom.load_state_dict(
        {
            name.removeprefix("fc_atom."): value
            for name, value in state.items()
            if name.startswith("fc_atom.")
        },
        strict=True,
    )
    gemnet.eval()
    noise_level_encoding.eval()
    fc_atom.eval()
    return _ReferenceModel(gemnet, noise_level_encoding, fc_atom)


class _SourcePropertyBatch(dict[str, object]):
    """Minimal mapping interface consumed by source PropertyEmbedding.forward."""

    def __init__(
        self,
        *,
        value: torch.Tensor,
        use_unconditional: torch.Tensor,
        position: torch.Tensor,
    ):
        super().__init__(
            dft_band_gap=value,
            num_atoms=torch.tensor([len(position)], dtype=torch.long),
            _USE_UNCONDITIONAL_EMBEDDING={"dft_band_gap": use_unconditional},
        )
        self.pos = position


def _make_dft_band_gap_reference_model(
    source: _SourceModules,
    config: MatterGenConfig,
    state: Mapping[str, torch.Tensor],
) -> _ReferenceModel:
    """Load the source GemNet-T control adapter and its property encoder."""
    assert config.condition_on_adapt == ("dft_band_gap",)
    atom_embedding = source.atom_embedding.AtomEmbedding(
        config.hidden_size,
        with_mask_type=True,
    )
    gemnet = source.gemnet_ctrl.GemNetTCtrl(
        list(config.condition_on_adapt),
        **_gemnet_kwargs(config, atom_embedding),
    )
    gemnet.load_state_dict(
        {
            name.removeprefix("gemnet."): value
            for name, value in state.items()
            if name.startswith("gemnet.")
        },
        strict=True,
    )
    noise_level_encoding = source.model_utils.NoiseLevelEncoding(config.hidden_size)
    noise_level_encoding.load_state_dict(
        {
            name.removeprefix("noise_level_encoding."): value
            for name, value in state.items()
            if name.startswith("noise_level_encoding.")
        },
        strict=True,
    )
    property_embedding = source.property_embeddings.PropertyEmbedding(
        name="dft_band_gap",
        conditional_embedding_module=source.model_utils.NoiseLevelEncoding(config.hidden_size),
        unconditional_embedding_module=source.property_embeddings.ZerosEmbedding(
            config.hidden_size
        ),
        scaler=source.data_utils.StandardScalerTorch(),
    )
    property_embedding.load_state_dict(
        {
            name.removeprefix("property_embeddings_adapt.dft_band_gap."): value
            for name, value in state.items()
            if name.startswith("property_embeddings_adapt.dft_band_gap.")
        },
        strict=True,
    )
    fc_atom = torch.nn.Linear(config.hidden_size, config.num_atom_types)
    fc_atom.load_state_dict(
        {
            name.removeprefix("fc_atom."): value
            for name, value in state.items()
            if name.startswith("fc_atom.")
        },
        strict=True,
    )
    gemnet.eval()
    noise_level_encoding.eval()
    property_embedding.eval()
    fc_atom.eval()
    return _ReferenceModel(
        gemnet,
        noise_level_encoding,
        fc_atom,
        adapter_property=property_embedding,
        adapter_name="dft_band_gap",
    )


def _crystal(
    *,
    translated: bool = False,
    permutation: np.ndarray | None = None,
) -> _Crystal:
    """Return a non-boundary periodic fixture with an optional rigid translation/permutation."""
    fractional_coordinates = np.array(
        [[0.10, 0.15, 0.20], [0.40, 0.45, 0.35]],
        dtype=np.float32,
    )
    if translated:
        # No coordinate wraps, so this is a true rigid translation in the source cell convention.
        fractional_coordinates = fractional_coordinates + np.array(
            [0.10, 0.20, 0.10], dtype=np.float32
        )
    atomic_numbers = np.array([3, 8], dtype=np.int64)
    if permutation is not None:
        atomic_numbers = atomic_numbers[permutation]
        fractional_coordinates = fractional_coordinates[permutation]
    return _Crystal(
        atomic_numbers=atomic_numbers,
        fractional_coordinates=fractional_coordinates,
        cell=np.diag(np.array([6.0, 6.0, 6.0], dtype=np.float32))[None, ...],
    )


def _source_host_feeds(
    reference: _ReferenceModel,
    source: _SourceModules,
    crystal: _Crystal,
    timestep: float,
) -> _SourceGraph:
    """Build host ABI tensors through source PBC graph, symmetrization, and triplet code."""
    atomic_numbers = torch.from_numpy(crystal.atomic_numbers)
    fractional_coordinates = torch.from_numpy(crystal.fractional_coordinates)
    lattice = torch.from_numpy(crystal.cell)
    num_atoms = torch.tensor([len(atomic_numbers)], dtype=torch.long)
    return _source_host_feeds_for_batch(
        reference,
        source,
        atomic_numbers=atomic_numbers,
        fractional_coordinates=fractional_coordinates,
        lattice=lattice,
        num_atoms=num_atoms,
        timestep=timestep,
    )


def _source_host_feeds_for_batch(
    reference: _ReferenceModel,
    source: _SourceModules,
    *,
    atomic_numbers: torch.Tensor,
    fractional_coordinates: torch.Tensor,
    lattice: torch.Tensor,
    num_atoms: torch.Tensor,
    timestep: float,
) -> _SourceGraph:
    """Build one exact source graph for arbitrary packed crystals."""
    batch = torch.repeat_interleave(torch.arange(len(num_atoms), dtype=torch.long), num_atoms)
    cartesian_coordinates = source.data_utils.frac_to_cart_coords_with_lattice(
        fractional_coordinates, num_atoms, lattice
    )
    initial_edges, to_jimages, num_bonds = source.data_utils.radius_graph_pbc(
        cart_coords=cartesian_coordinates,
        lattice=lattice,
        num_atoms=num_atoms,
        radius=7.0,
        max_num_neighbors_threshold=50,
        max_cell_images_per_dim=5,
    )
    (
        edge_index,
        _neighbors,
        edge_distance,
        edge_direction,
        id_swap,
        id3_ba,
        id3_ca,
        id3_ragged_idx,
        _cell_offsets,
    ) = reference.gemnet.generate_interaction_graph(
        cartesian_coordinates,
        lattice,
        num_atoms,
        initial_edges,
        to_jimages,
        num_bonds,
    )
    edge_batch = batch[edge_index[0]]
    edge_lattice_cosines = torch.cosine_similarity(
        edge_direction[:, None],
        lattice[edge_batch],
        dim=-1,
    )
    feeds = {
        "atomic_numbers": atomic_numbers.numpy(),
        "batch": batch.numpy(),
        "timestep": np.array([timestep], dtype=np.float32),
        "edge_index": edge_index.numpy(),
        "edge_distance": edge_distance.numpy(),
        "edge_direction": edge_direction.numpy(),
        "edge_lattice_cosines": edge_lattice_cosines.numpy(),
        "id_swap": id_swap.numpy(),
        "id3_ba": id3_ba.numpy(),
        "id3_ca": id3_ca.numpy(),
        "id3_ragged_idx": id3_ragged_idx.numpy(),
    }
    return _SourceGraph(initial_edges, to_jimages, num_bonds, feeds)


def _source_score(
    reference: _ReferenceModel,
    source: _SourceModules,
    crystal: _Crystal,
    timestep: float,
    *,
    adapter_value: float | None = None,
    use_unconditional: bool = False,
) -> _ScoreCase:
    """Evaluate all score heads in the pinned source with the same host graph passed to ONNX."""
    graph = _source_host_feeds(reference, source, crystal, timestep)
    adapter_inputs: dict[str, object] = {}
    if reference.adapter_property is not None:
        assert reference.adapter_name is not None
        if adapter_value is None:
            raise ValueError("adapter score requires a concrete source property value")
        adapter_mask = torch.tensor([[use_unconditional]], dtype=torch.bool)
        property_value = torch.tensor([adapter_value], dtype=torch.float32)
        adapter_embedding = reference.adapter_property(
            _SourcePropertyBatch(
                value=property_value,
                use_unconditional=adapter_mask,
                position=torch.from_numpy(crystal.fractional_coordinates),
            )
        )
        adapter_inputs = {
            "cond_adapt": {reference.adapter_name: adapter_embedding},
            "cond_adapt_mask": {reference.adapter_name: adapter_mask},
        }
        graph.feeds[f"condition.{reference.adapter_name}"] = property_value.numpy()
        graph.feeds[f"condition.{reference.adapter_name}.use_unconditional"] = (
            adapter_mask.squeeze(1).numpy()
        )
    with torch.inference_mode():
        output = reference.gemnet(
            z=reference.noise_level_encoding(torch.from_numpy(graph.feeds["timestep"])),
            frac_coords=torch.from_numpy(crystal.fractional_coordinates),
            atom_types=torch.from_numpy(graph.feeds["atomic_numbers"]),
            num_atoms=torch.tensor([len(crystal.atomic_numbers)], dtype=torch.long),
            batch=torch.from_numpy(graph.feeds["batch"]),
            edge_index=graph.initial_edges,
            to_jimages=graph.to_jimages,
            num_bonds=graph.num_bonds,
            lattice=torch.from_numpy(crystal.cell),
            **adapter_inputs,
        )
        outputs = {
            "atom_logits": reference.fc_atom(output.node_embeddings).numpy(),
            "coordinate_score": output.forces.numpy(),
            "lattice_score": output.stress.numpy(),
            "energy": output.energy.numpy(),
        }
    return _ScoreCase(graph.feeds, outputs)


def _sha256(path: Path) -> str:
    """Hash an immutable checkpoint for golden-fixture provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def mp20_runtime() -> Iterator[_Mp20Runtime]:
    """Run pinned-source references first, then load the same checkpoint into standard ONNX."""
    source_dir = _required_artifact(_SOURCE_DIR_ENV)
    checkpoint = _required_artifact(_MP20_CHECKPOINT_ENV)
    with _pinned_source_modules(source_dir) as source:
        state, hydra_config = _load_checkpoint(checkpoint)
        config = MatterGenConfig.from_hydra_config(hydra_config, variant="mp_20_base")
        reference = _make_reference_model(source, config, state)
        original = {
            timestep: _source_score(reference, source, _crystal(), timestep)
            for timestep in (0.25, 0.75)
        }
        translated = _source_score(reference, source, _crystal(translated=True), 0.25)
        permuted = _source_score(
            reference,
            source,
            _crystal(permutation=np.array([1, 0], dtype=np.int64)),
            0.25,
        )
        batched_graph = _source_host_feeds_for_batch(
            reference,
            source,
            atomic_numbers=torch.tensor([3, 8, 14], dtype=torch.long),
            fractional_coordinates=torch.tensor(
                [[0.10, 0.15, 0.20], [0.40, 0.45, 0.35], [0.25, 0.75, 0.50]],
                dtype=torch.float32,
            ),
            lattice=torch.stack(
                [
                    torch.diag(torch.tensor([6.0, 5.5, 6.5], dtype=torch.float32)),
                    torch.tensor(
                        [[4.5, 0.0, 0.0], [0.4, 5.0, 0.0], [0.2, 0.3, 5.5]],
                        dtype=torch.float32,
                    ),
                ]
            ),
            num_atoms=torch.tensor([2, 1], dtype=torch.long),
            timestep=0.25,
        )
        del reference
        del state

    module = MatterGenModel(config)
    package = build_from_module(module, config, task=MatterGenScoreTask())
    assert package.export_report is not None
    score_core_report = package.export_report.component("score_core")
    assert score_core_report.runtime_validation_status == "validated"
    assert score_core_report.evidence_id == "mattergen-score-core-ort"
    apply_mattergen_checkpoint(package, module, checkpoint)
    # This materializes the reported standard-ONNX score core in CPU ORT; the
    # tests below execute all exported score heads against pinned-source values.
    session = OnnxModelSession(package["model"], device="cpu")
    del module
    del package
    runtime = _Mp20Runtime(
        session=session,
        original=original,
        translated=translated,
        permuted=permuted,
        batched_graph=batched_graph,
        checkpoint_sha256=_sha256(checkpoint),
    )
    try:
        yield runtime
    finally:
        runtime.close()


def _assert_outputs_close(
    actual: Mapping[str, np.ndarray],
    expected: Mapping[str, np.ndarray],
    *,
    rtol: float = _RTOL,
    atol: float = _ATOL,
) -> None:
    """Compare every source score-core output, retaining array-specific failure context."""
    assert set(actual) == {"atom_logits", "coordinate_score", "lattice_score", "energy"}
    for name in actual:
        np.testing.assert_allclose(
            actual[name],
            expected[name],
            rtol=rtol,
            atol=atol,
            err_msg=f"{name} differs between ONNX Runtime and pinned MatterGen source",
        )


def _unpermute_atom_outputs(
    outputs: Mapping[str, np.ndarray], permutation: np.ndarray
) -> dict[str, np.ndarray]:
    """Restore atom-indexed outputs after evaluating a source-compatible input atom permutation."""
    inverse = np.argsort(permutation)
    return {
        name: value[inverse] if name in {"atom_logits", "coordinate_score"} else value
        for name, value in outputs.items()
    }


def test_mp20_source_score_core_matches_onnx_at_two_timesteps(
    mp20_runtime: _Mp20Runtime,
) -> None:
    """L3: exact-source synthetic periodic graph parity for all requested score heads."""
    for expected in mp20_runtime.original.values():
        _assert_outputs_close(mp20_runtime.session.run(expected.feeds), expected.outputs)


def test_mp20_periodic_translation_and_permutation_invariance(
    mp20_runtime: _Mp20Runtime,
) -> None:
    """L3: establish PBC translation/permutation invariance in source before asserting it in ONNX."""
    baseline = mp20_runtime.original[0.25].outputs
    translation = mp20_runtime.translated
    permutation = np.array([1, 0], dtype=np.int64)
    permuted = _unpermute_atom_outputs(mp20_runtime.permuted.outputs, permutation)

    _assert_outputs_close(translation.outputs, baseline)
    _assert_outputs_close(permuted, baseline)
    _assert_outputs_close(mp20_runtime.session.run(translation.feeds), translation.outputs)
    _assert_outputs_close(
        _unpermute_atom_outputs(
            mp20_runtime.session.run(mp20_runtime.permuted.feeds), permutation
        ),
        baseline,
    )


def test_host_periodic_graph_matches_source_for_batched_neighbor_truncation(
    mp20_runtime: _Mp20Runtime,
) -> None:
    """Host PBC edges, symmetric pairs, and triplets match source for packed crystals."""
    expected = mp20_runtime.batched_graph.feeds
    graph = build_periodic_graph(
        torch.from_numpy(
            np.array(
                [[0.10, 0.15, 0.20], [0.40, 0.45, 0.35], [0.25, 0.75, 0.50]],
                dtype=np.float32,
            )
        ),
        torch.from_numpy(
            np.array(
                [
                    [[6.0, 0.0, 0.0], [0.0, 5.5, 0.0], [0.0, 0.0, 6.5]],
                    [[4.5, 0.0, 0.0], [0.4, 5.0, 0.0], [0.2, 0.3, 5.5]],
                ],
                dtype=np.float32,
            )
        ),
        torch.tensor([2, 1], dtype=torch.long),
        cutoff=7.0,
        max_neighbors=50,
        max_cell_images_per_dim=5,
    )
    actual = {
        "edge_index": graph.edge_index.numpy(),
        "edge_distance": graph.edge_distance.numpy(),
        "edge_direction": graph.edge_direction.numpy(),
        "edge_lattice_cosines": graph.edge_lattice_cosines.numpy(),
        "id_swap": graph.id_swap.numpy(),
        "id3_ba": graph.id3_ba.numpy(),
        "id3_ca": graph.id3_ca.numpy(),
        "id3_ragged_idx": graph.id3_ragged_idx.numpy(),
    }
    for name, value in actual.items():
        np.testing.assert_array_equal(
            value,
            expected[name],
            err_msg=f"{name} source graph mismatch",
        )


def test_host_d3pm_predictor_matches_source_schedule_and_rng() -> None:
    """Host absorbing-mask posterior and both categorical draws match the source."""
    source_dir = _required_artifact(_SOURCE_DIR_ENV)
    with _pinned_source_modules(source_dir) as source:
        schedule = source.d3pm.create_discrete_diffusion_schedule(
            kind="standard",
            num_steps=1000,
        )
        corruption = source.d3pm_corruption.D3PMCorruption(
            source.d3pm.MaskDiffusion(dim=101, schedule=schedule),
            offset=1,
        )
        predictor = source.d3pm_predictors_correctors.D3PMAncestralSamplingPredictor(
            corruption=corruption,
            score_fn=None,
            predict_x0=True,
        )
        atomic_numbers = torch.tensor([101, 3, 101], dtype=torch.long)
        logits = torch.linspace(-2.0, 2.0, 303, dtype=torch.float32).reshape(3, 101)
        timestep = torch.tensor([0.75], dtype=torch.float32)
        batch = torch.zeros(3, dtype=torch.long)
        rng_state = torch.random.get_rng_state()
        try:
            torch.manual_seed(814)
            expected_sample, expected_mean = predictor.update_given_score(
                x=atomic_numbers,
                t=timestep,
                dt=torch.tensor(-0.001, dtype=torch.float32),
                batch_idx=batch,
                score=logits,
                batch=None,
            )
        finally:
            torch.random.set_rng_state(rng_state)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(814)
    actual_sample, actual_mean = MatterGenHostSampler(
        lambda _inputs: (_ for _ in ()).throw(
            AssertionError("score callback must not be used")
        )
    )._d3pm_ancestral(atomic_numbers, logits, timestep, batch, generator)

    torch.testing.assert_close(actual_sample, expected_sample)
    torch.testing.assert_close(actual_mean, expected_mean)


@pytest.mark.golden
@pytest.mark.generation
def test_mp20_real_onnx_host_sampling_golden(mp20_runtime: _Mp20Runtime) -> None:
    """L5: the full released scheduler yields a deterministic valid crystal artifact."""

    class _NamedSession:
        """Bridge test inference wrapper to the public ONNX Runtime callback ABI."""

        def run(
            self,
            output_names: list[str] | None,
            input_feed: Mapping[str, np.ndarray],
        ) -> list[np.ndarray]:
            if output_names is None:
                raise AssertionError("MatterGen callback requests explicit output names")
            outputs = mp20_runtime.session.run(dict(input_feed))
            return [outputs[name] for name in output_names]

    golden = json.loads(_HOST_SAMPLE_GOLDEN_PATH.read_text(encoding="utf-8"))
    assert golden["source_commit"] == MATTERGEN_SOURCE_COMMIT
    assert golden["checkpoint_sha256"] == mp20_runtime.checkpoint_sha256
    assert golden["timesteps"] == 1000

    sample = MatterGenHostSampler(create_onnxruntime_score_callback(_NamedSession())).sample(
        torch.tensor(golden["num_atoms"], dtype=torch.long),
        seed=golden["seed"],
    )
    crystal = sample.crystals()[0]
    np.testing.assert_array_equal(
        crystal.atomic_numbers.numpy(),
        np.asarray(golden["sample"]["atomic_numbers"], dtype=np.int64),
    )
    np.testing.assert_allclose(
        crystal.fractional_coordinates.numpy(),
        np.asarray(golden["sample"]["fractional_coordinates"], dtype=np.float32),
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        crystal.cell.numpy(),
        np.asarray(golden["sample"]["cell"], dtype=np.float32),
        rtol=1e-4,
        atol=1e-4,
    )
    assert np.isclose(
        np.linalg.det(crystal.cell.numpy()),
        golden["sample"]["volume"],
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.golden
def test_mp20_one_step_source_golden(mp20_runtime: _Mp20Runtime) -> None:
    """L4: compare a one-step real ``mp_20_base`` source output with committed provenance."""
    golden = json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))
    assert golden["source_commit"] == MATTERGEN_SOURCE_COMMIT
    assert golden["checkpoint_sha256"] == mp20_runtime.checkpoint_sha256
    assert np.isclose(golden["timestep"], 0.25)
    expected = {
        name: np.asarray(value, dtype=np.float32) for name, value in golden["outputs"].items()
    }
    source_case = mp20_runtime.original[0.25]
    _assert_outputs_close(source_case.outputs, expected)
    _assert_outputs_close(mp20_runtime.session.run(source_case.feeds), expected)


def test_dft_band_gap_adapter_matches_source_for_conditional_and_unconditional_scores() -> (
    None
):
    """L3: exercise the released scalar adapter through both source embedding modes."""
    source_dir = _required_artifact(_SOURCE_DIR_ENV)
    checkpoint = _required_artifact(_DFT_BAND_GAP_CHECKPOINT_ENV)
    with _pinned_source_modules(source_dir) as source:
        state, hydra_config = _load_checkpoint(checkpoint)
        config = MatterGenConfig.from_hydra_config(hydra_config, variant="dft_band_gap")
        reference = _make_dft_band_gap_reference_model(source, config, state)
        conditional = _source_score(
            reference,
            source,
            _crystal(),
            0.0,
            adapter_value=1.5,
        )
        unconditional = _source_score(
            reference,
            source,
            _crystal(),
            0.0,
            adapter_value=1.5,
            use_unconditional=True,
        )

    # The property value must reach GemNet-T control blocks; otherwise the two
    # source evaluations would be indistinguishable despite different adapter masks.
    assert not np.allclose(
        conditional.outputs["coordinate_score"],
        unconditional.outputs["coordinate_score"],
        rtol=_RTOL,
        atol=_ATOL,
    )

    module = MatterGenModel(config)
    package = build_from_module(module, config, task=MatterGenScoreTask())
    apply_mattergen_checkpoint(package, module, checkpoint)
    session = OnnxModelSession(package["model"], device="cpu")
    try:
        _assert_outputs_close(
            session.run(conditional.feeds),
            conditional.outputs,
            rtol=_ADAPTER_RTOL,
            atol=_ADAPTER_ATOL,
        )
        _assert_outputs_close(
            session.run(unconditional.feeds),
            unconditional.outputs,
            rtol=_ADAPTER_RTOL,
            atol=_ADAPTER_ATOL,
        )
    finally:
        session.close()
