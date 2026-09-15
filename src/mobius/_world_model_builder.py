# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Registry and public build entry point for compositional world models."""

from __future__ import annotations

__all__ = [
    "WorldModelBuilderRegistry",
    "build_world_model",
    "world_model_registry",
]

import json
import pathlib
from collections.abc import Callable
from typing import Any

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from mobius._pipeline import PipelinePackage

WorldModelBuilder = Callable[..., PipelinePackage]


class WorldModelBuilderRegistry:
    """Map checkpoint ``model_type`` values to compositional exporters."""

    def __init__(self) -> None:
        self._builders: dict[str, WorldModelBuilder] = {}

    def register(self, model_type: str, builder: WorldModelBuilder) -> None:
        """Register one world-model pipeline builder.

        Re-registering the same callable is idempotent. Replacing an existing
        builder is rejected so import order cannot silently change export
        semantics.
        """
        if not model_type:
            raise ValueError("World-model model_type must be non-empty")
        existing = self._builders.get(model_type)
        if existing is builder:
            return
        if existing is not None:
            raise ValueError(f"World-model builder for {model_type!r} is already registered")
        self._builders[model_type] = builder

    def get(self, model_type: str) -> WorldModelBuilder:
        """Return the registered builder for *model_type*."""
        try:
            return self._builders[model_type]
        except KeyError as error:
            supported = ", ".join(sorted(self._builders)) or "<none>"
            raise ValueError(
                f"No complete world-model pipeline is registered for model_type "
                f"{model_type!r}. Registered types: {supported}."
            ) from error

    def model_types(self) -> tuple[str, ...]:
        """Return registered model types in deterministic order."""
        return tuple(sorted(self._builders))


def _load_model_type(model_id: str) -> str:
    root = pathlib.Path(model_id)
    candidates = ("config.json", "model_index.json")
    for filename in candidates:
        if root.is_dir():
            path = root / filename
            if not path.is_file():
                continue
        else:
            try:
                path = pathlib.Path(hf_hub_download(repo_id=model_id, filename=filename))
            except EntryNotFoundError:
                continue
        with path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        field = "model_type" if filename == "config.json" else "_class_name"
        model_type = config.get(field)
        if isinstance(model_type, str) and model_type:
            return model_type
    raise ValueError(
        f"Checkpoint {model_id!r} has neither a non-empty config.json model_type "
        "nor a model_index.json _class_name."
    )


world_model_registry = WorldModelBuilderRegistry()


def build_world_model(
    model_id: str,
    *,
    dtype: Any | None = None,
    load_weights: bool = True,
    execution_provider: str = "default",
    trace_optimization: bool = False,
    **options: Any,
) -> PipelinePackage:
    """Build every neural component of a registered world-model checkpoint.

    Unlike :func:`mobius.build`, which exports one architecture task, this
    entry point dispatches to a model-specific pipeline builder that may
    combine reasoners, dynamics or diffusion generators, tokenizers/codecs,
    observation decoders, reward models, and action policies.
    """
    model_type = _load_model_type(model_id)
    builder = world_model_registry.get(model_type)
    return builder(
        model_id,
        dtype=dtype,
        load_weights=load_weights,
        execution_provider=execution_provider,
        trace_optimization=trace_optimization,
        **options,
    )


def _register_builtin_world_models() -> None:
    from mobius._cosmos3_edge_world_model import build_cosmos3_edge_world_model
    from mobius._cosmos3_world_model import build_cosmos3_world_model

    world_model_registry.register("cosmos3_edge", build_cosmos3_edge_world_model)
    world_model_registry.register("cosmos3_omni", build_cosmos3_world_model)


_register_builtin_world_models()
