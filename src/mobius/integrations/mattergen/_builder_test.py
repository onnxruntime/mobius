# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from unittest import mock

import pytest
import yaml

from mobius import build_from_module
from mobius.integrations.mattergen import MatterGenConfig, MatterGenModel
from mobius.integrations.mattergen import _builder
from mobius.integrations.mattergen._contract import (
    MATTERGEN_HUB_ID,
    MATTERGEN_HUB_REVISION,
    OFFICIAL_CHECKPOINT_CONDITIONS,
)


def _tiny_hydra_config(*, adapter: bool = False) -> dict[str, object]:
    """Return a source-shaped configuration small enough for graph unit tests."""
    model: dict[str, object] = {
        "hidden_dim": 8,
        "denoise_atom_types": True,
        "atom_type_diffusion": "mask",
        "gemnet": {
            "num_targets": 1,
            "num_spherical": 2,
            "num_radial": 4,
            "num_blocks": 1,
            "emb_size_atom": 8,
            "emb_size_edge": 8,
            "emb_size_trip": 4,
            "emb_size_rbf": 4,
            "emb_size_cbf": 4,
            "emb_size_bil_trip": 4,
            "num_before_skip": 1,
            "num_after_skip": 1,
            "num_concat": 1,
            "num_atom": 1,
            "max_neighbors": 4,
            "max_cell_images_per_dim": 1,
            "regress_stress": True,
            "atom_embedding": {"with_mask_type": True},
        },
    }
    if adapter:
        model["property_embeddings_adapt"] = {
            "dft_band_gap": {
                "conditional_embedding_module": {
                    "_target_": "mattergen.property_embeddings.NoiseLevelEncoding"
                },
                "scaler": {"_target_": "mattergen.common.utils.data_utils.StandardScalerTorch"},
            }
        }
        gemnet = model["gemnet"]
        assert isinstance(gemnet, dict)
        gemnet["condition_on_adapt"] = ["dft_band_gap"]
    return {"lightning_module": {"diffusion_module": {"model": model}}}


class TestMatterGenGraphTask:
    @pytest.mark.parametrize("adapter", [False, True])
    def test_builds_tiny_score_graph_with_exact_condition_ports(self, adapter: bool) -> None:
        config = MatterGenConfig.from_hydra_config(_tiny_hydra_config(adapter=adapter))
        package = build_from_module(MatterGenModel(config), config, task="mattergen-score")
        model = package["model"]

        expected_inputs = {
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
        }
        if adapter:
            expected_inputs |= {
                "condition.dft_band_gap",
                "condition.dft_band_gap.use_unconditional",
            }
        assert {value.name for value in model.graph.inputs} == expected_inputs
        assert [value.name for value in model.graph.outputs] == [
            "atom_logits",
            "coordinate_score",
            "lattice_score",
            "energy",
        ]
        assert model.metadata_props["mobius.source_revision"] == MATTERGEN_HUB_REVISION
        assert model.metadata_props["mobius.max_atoms"] == "20"
        assert model.metadata_props["mobius.checkpoint_family"] == "mattergen_base"
        assert package.export_report is not None
        assert package.export_report.status == "partial"
        assert package.export_report.component("score_core").runtime_validation_status == "validated"


class TestMatterGenBuilder:
    def test_builds_no_weights_from_a_safe_local_checkpoint_root(self, tmp_path) -> None:
        root = tmp_path / "mattergen"
        config_path = root / "checkpoints" / "mp_20_base" / "config.yaml"
        config_path.parent.mkdir(parents=True)
        config_path.write_text(yaml.safe_dump(_tiny_hydra_config()), encoding="utf-8")

        package = _builder.build_mattergen(root, load_weights=False)

        assert package.config.variant == "mp_20_base"
        assert package["model"].graph.name == f"{root}/mp_20_base/model"
        assert _builder.is_mattergen_checkpoint(root)

    def test_no_weights_package_saves_an_atomic_partial_export_report(self, tmp_path) -> None:
        root = tmp_path / "mattergen"
        config_path = root / "checkpoints" / "mp_20_base" / "config.yaml"
        config_path.parent.mkdir(parents=True)
        config_path.write_text(yaml.safe_dump(_tiny_hydra_config()), encoding="utf-8")
        package = _builder.build_mattergen(root, load_weights=False)
        output = tmp_path / "score-core"

        package.save(output, check_weights=False, progress_bar=False)

        assert (output / "model.onnx").is_file()
        assert (output / "export_report.json").is_file()

    def test_resolves_the_immutable_hub_revision_for_config_download(self, tmp_path, monkeypatch) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(_tiny_hydra_config()), encoding="utf-8")
        download = mock.Mock(return_value=str(config_path))
        monkeypatch.setattr(_builder, "hf_hub_download", download)

        config, checkpoint, revision = _builder._load_config(
            MATTERGEN_HUB_ID,
            "mp_20_base",
            None,
            load_weights=False,
        )

        assert config == _tiny_hydra_config()
        assert checkpoint is None
        assert revision == MATTERGEN_HUB_REVISION
        assert download.call_args.kwargs == {
            "repo_id": MATTERGEN_HUB_ID,
            "filename": "checkpoints/mp_20_base/config.yaml",
            "revision": MATTERGEN_HUB_REVISION,
        }

    def test_rejects_mutable_or_incompatible_build_options(self) -> None:
        with pytest.raises(ValueError, match="pinned Hub revision"):
            _builder.build_mattergen(revision="main", load_weights=False)
        with pytest.raises(ValueError, match="float32"):
            _builder.build_mattergen(dtype="f16", load_weights=False)
        with pytest.raises(ValueError, match="default/CPU"):
            _builder.build_mattergen(execution_provider="cuda", load_weights=False)

    def test_rejects_a_local_config_with_the_wrong_declared_family_conditions(self, tmp_path) -> None:
        root = tmp_path / "mattergen"
        config_path = root / "checkpoints" / "mp_20_base" / "config.yaml"
        config_path.parent.mkdir(parents=True)
        config_path.write_text(yaml.safe_dump(_tiny_hydra_config(adapter=True)), encoding="utf-8")

        with pytest.raises(ValueError, match="expected the pinned contract"):
            _builder.build_mattergen(root, load_weights=False)

    @pytest.mark.arch_validation
    @pytest.mark.parametrize(
        ("family", "conditions"),
        sorted(OFFICIAL_CHECKPOINT_CONDITIONS.items()),
    )
    def test_all_official_pinned_hydra_configs_build_score_graph(
        self,
        family: str,
        conditions: tuple[str, ...],
    ) -> None:
        package = _builder.build_mattergen(
            MATTERGEN_HUB_ID,
            checkpoint=family,
            revision=MATTERGEN_HUB_REVISION,
            load_weights=False,
        )

        assert package.config.variant == family
        assert tuple(spec.name for spec in package.config.condition_input_specs) == conditions
        assert package["model"].graph.outputs[0].name == "atom_logits"
