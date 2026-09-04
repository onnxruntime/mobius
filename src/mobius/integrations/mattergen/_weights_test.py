# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import pytest
import torch

from mobius.integrations.mattergen._weights import (
    MATTERGEN_MODEL_STATE_PREFIX,
    _assert_exact_tensor_routing,
    apply_mattergen_checkpoint,
    load_mattergen_state_dict,
)


class _Initializer:
    def __init__(self, shape: tuple[int, ...], *, is_literal: bool) -> None:
        self.shape = shape
        self.const_value = object() if is_literal else None


class _Graph:
    def __init__(self) -> None:
        self.initializers = {
            "trained.weight": _Initializer((2, 3), is_literal=False),
            "const_1.0_f32": _Initializer((), is_literal=True),
        }


class _ScoreModel:
    def __init__(self) -> None:
        self.graph = _Graph()
        self.metadata_props: dict[str, str] = {}


class _Package(dict[str, _ScoreModel]):
    def __init__(self) -> None:
        super().__init__({"model": _ScoreModel()})
        self.applied: dict[str, torch.Tensor] | None = None
        self.weight_loading_report: dict[str, object] | None = None

    def apply_weights(self, state_dict, **_kwargs) -> None:
        self.applied = state_dict


class TestMatterGenWeightLoading:
    def test_strips_only_the_inference_model_prefix(self, tmp_path) -> None:
        checkpoint = tmp_path / "mattergen.ckpt"
        weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        torch.save(
            {"state_dict": {f"{MATTERGEN_MODEL_STATE_PREFIX}gemnet.weight": weight}},
            checkpoint,
        )

        state_dict = load_mattergen_state_dict(checkpoint)

        assert state_dict.keys() == {"gemnet.weight"}
        assert torch.equal(state_dict["gemnet.weight"], weight)

    def test_rejects_training_or_unknown_state_tensors(self, tmp_path) -> None:
        checkpoint = tmp_path / "mattergen.ckpt"
        torch.save(
            {
                "state_dict": {
                    f"{MATTERGEN_MODEL_STATE_PREFIX}gemnet.weight": torch.ones(1),
                    "optimizer.step": torch.ones(1),
                }
            },
            checkpoint,
        )

        with pytest.raises(ValueError, match="outside"):
            load_mattergen_state_dict(checkpoint)

    @pytest.mark.parametrize("payload", [{}, {"state_dict": []}, []])
    def test_rejects_non_mapping_checkpoint_schema(self, tmp_path, payload) -> None:
        checkpoint = tmp_path / "mattergen.ckpt"
        torch.save(payload, checkpoint)

        with pytest.raises(TypeError, match="mapping"):
            load_mattergen_state_dict(checkpoint)

    def test_exact_routing_ignores_graph_literal_initializers(self) -> None:
        package = _Package()

        _assert_exact_tensor_routing(
            package,
            {"trained.weight": torch.ones((2, 3), dtype=torch.float32)},
        )

    def test_report_accounts_for_validated_checkpoint_aliases(self, monkeypatch, tmp_path) -> None:
        package = _Package()
        source_tensors = {
            "trained.weight": torch.ones((2, 3), dtype=torch.float32),
            "duplicate.alias": torch.ones(1, dtype=torch.float32),
        }
        monkeypatch.setattr(
            "mobius.integrations.mattergen._weights.load_mattergen_state_dict",
            lambda _path: source_tensors,
        )

        class Module:
            @staticmethod
            def preprocess_weights(_state_dict):
                return {"trained.weight": source_tensors["trained.weight"]}

        apply_mattergen_checkpoint(package, Module(), tmp_path / "mattergen.ckpt")

        assert package.applied is not None
        assert torch.equal(package.applied["trained.weight"], source_tensors["trained.weight"])
        assert package.weight_loading_report["source_tensors"] == 2
        assert package.weight_loading_report["assigned_tensors"] == 1
        assert package.weight_loading_report["canonicalized_alias_tensors"] == 1
        assert package.weight_loading_report["ignored_tensors"] == 0
