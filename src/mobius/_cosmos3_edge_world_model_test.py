# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import mock

from mobius._cosmos3_edge_world_model import (
    _edge_text_model_type,
    build_cosmos3_edge_world_model,
)
from mobius._cosmos3_world_model import build_cosmos3_world_model
from mobius._world_model_builder import world_model_registry
from mobius.models.cosmos import Cosmos3EdgeVLModel


def test_edge_model_type_is_registered() -> None:
    assert "cosmos3_edge" in world_model_registry.model_types()


def test_edge_text_model_type_uses_nested_config() -> None:
    assert (
        _edge_text_model_type(
            {"model_type": "cosmos3_omni", "text_config": {"model_type": "cosmos3_edge_text"}}
        )
        == "cosmos3_edge_text"
    )
    assert _edge_text_model_type({"model_type": "cosmos3_edge"}) is None


def test_omni_dispatch_delegates_mislabeled_edge_checkpoint(tmp_path) -> None:
    (tmp_path / "model_index.json").write_text("{}", encoding="utf-8")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "cosmos3_omni",
                "text_config": {"model_type": "cosmos3_edge_text"},
            }
        ),
        encoding="utf-8",
    )
    package = object()

    with mock.patch(
        "mobius._cosmos3_edge_world_model.build_cosmos3_edge_world_model",
        return_value=package,
    ) as edge_builder:
        result = build_cosmos3_world_model(
            str(tmp_path),
            dtype="bf16",
            load_weights=False,
            execution_provider="cuda",
            trace_optimization=True,
            custom=True,
        )

    assert result is package
    edge_builder.assert_called_once_with(
        str(tmp_path),
        dtype="bf16",
        load_weights=False,
        execution_provider="cuda",
        trace_optimization=True,
        custom=True,
    )


def test_edge_builder_uses_edge_reasoner_and_shared_generator_pipeline() -> None:
    loaded = {
        "config.json": (
            {
                "model_type": "cosmos3_edge",
                "text_config": {"model_type": "cosmos3_edge_text"},
            },
            "config.json",
        ),
        "model_index.json": (
            {
                "transformer": ["diffusers", "Cosmos3OmniTransformer"],
                "vae": ["diffusers", "AutoencoderKLWan"],
                "sound_tokenizer": [None, None],
            },
            "model_index.json",
        ),
        "transformer/config.json": ({"hidden_size": 8}, "transformer/config.json"),
        "vae/config.json": ({"z_dim": 2}, "vae/config.json"),
        "scheduler/scheduler_config.json": (
            {"prediction_type": "flow_prediction"},
            "scheduler/scheduler_config.json",
        ),
    }
    reasoner_package = mock.sentinel.reasoner_package
    reasoner_module = mock.sentinel.reasoner_module
    generator_package = mock.sentinel.generator_package
    generator_module = SimpleNamespace(config=mock.sentinel.generator_config)
    vae_package = mock.sentinel.vae_package
    vae_module = SimpleNamespace(config=mock.sentinel.vae_config)
    package = mock.sentinel.pipeline_package

    with (
        mock.patch(
            "mobius._cosmos3_edge_world_model._load_json",
            side_effect=lambda _model_id, filename: loaded[filename],
        ),
        mock.patch(
            "mobius._cosmos3_edge_world_model._build_components",
            return_value=(
                reasoner_package,
                reasoner_module,
                generator_package,
                generator_module,
                vae_package,
                vae_module,
                None,
                None,
            ),
        ) as build_components,
        mock.patch(
            "mobius._cosmos3_edge_world_model._collect_assets",
            return_value={},
        ),
        mock.patch(
            "mobius._cosmos3_edge_world_model._load_optional_json",
            return_value={},
        ),
        mock.patch(
            "mobius._cosmos3_edge_world_model._resolve_file",
            return_value=None,
        ),
        mock.patch(
            "mobius._cosmos3_edge_world_model._compose_pipeline",
            return_value=package,
        ) as compose,
    ):
        result = build_cosmos3_edge_world_model("nvidia/Cosmos3-Edge", load_weights=False)

    assert result is package
    assert build_components.call_args.kwargs["reasoner_module_class"] is Cosmos3EdgeVLModel
    assert build_components.call_args.kwargs["reasoner_task"] == "cosmos3-edge-vl"
    assert compose.call_args.kwargs["pipeline_model_type"] == "cosmos3_edge"
    assert compose.call_args.kwargs["reasoner_architecture"] == "cosmos3_edge"
    assert compose.call_args.kwargs["extra_metadata"]["edge"]["checkpoint_model_type"] == (
        "cosmos3_edge"
    )
