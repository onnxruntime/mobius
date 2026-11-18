# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import json
from unittest import mock

import pytest

from mobius._world_model_builder import (
    WorldModelBuilderRegistry,
    build_world_model,
    world_model_registry,
)


def test_registry_rejects_conflicting_builder() -> None:
    registry = WorldModelBuilderRegistry()

    def first(*_args, **_kwargs):
        return object()

    def second(*_args, **_kwargs):
        return object()

    registry.register("example", first)
    registry.register("example", first)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("example", second)


def test_registry_reports_supported_model_types() -> None:
    registry = WorldModelBuilderRegistry()
    registry.register("zeta", mock.Mock())
    registry.register("alpha", mock.Mock())

    assert registry.model_types() == ("alpha", "zeta")
    with pytest.raises(ValueError, match=r"alpha, zeta"):
        registry.get("missing")


def test_build_world_model_dispatches_local_checkpoint(tmp_path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "test_world"}))
    package = object()
    builder = mock.Mock(return_value=package)

    with mock.patch.dict(world_model_registry._builders, {"test_world": builder}, clear=True):
        result = build_world_model(
            str(tmp_path),
            dtype="bf16",
            load_weights=False,
            execution_provider="cuda",
            trace_optimization=True,
            custom_option=7,
        )

    assert result is package
    builder.assert_called_once_with(
        str(tmp_path),
        dtype="bf16",
        load_weights=False,
        execution_provider="cuda",
        trace_optimization=True,
        custom_option=7,
    )


def test_build_world_model_rejects_missing_model_type(tmp_path) -> None:
    (tmp_path / "config.json").write_text("{}")

    with pytest.raises(ValueError, match="model_type"):
        build_world_model(str(tmp_path), load_weights=False)


def test_build_world_model_dispatches_pure_diffusers_pipeline(tmp_path) -> None:
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "ExampleWorldPipeline"})
    )
    package = object()
    builder = mock.Mock(return_value=package)

    with mock.patch.dict(
        world_model_registry._builders,
        {"ExampleWorldPipeline": builder},
        clear=True,
    ):
        result = build_world_model(str(tmp_path), load_weights=False)

    assert result is package
