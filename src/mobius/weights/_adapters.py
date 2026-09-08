# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Narrow model-specific boundary in the checkpoint loading pipeline."""

from __future__ import annotations

__all__ = [
    "ModelWeightAdapter",
    "WeightAdapterContext",
    "adapt_model_weights",
]

import dataclasses
from collections.abc import Mapping
from typing import Any, Protocol

import torch
from onnxscript import nn

from mobius._component_manifest import ComponentManifest
from mobius._configs import BaseModelConfig


@dataclasses.dataclass(frozen=True)
class WeightAdapterContext:
    """Generic metadata available to a model-specific semantic adapter."""

    config: BaseModelConfig
    manifest: ComponentManifest


class ModelWeightAdapter(Protocol):
    """Architecture-specific rename/split/fuse operations only."""

    def adapt(
        self,
        module: nn.Module,
        state_dict: Mapping[str, torch.Tensor],
        context: WeightAdapterContext,
    ) -> dict[str, torch.Tensor]:
        """Return semantically aligned weights without format normalization."""
        ...


def adapt_model_weights(
    module: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    config: BaseModelConfig,
    manifest: ComponentManifest,
) -> dict[str, torch.Tensor]:
    """Run an explicit adapter or the legacy ``preprocess_weights`` hook."""
    context = WeightAdapterContext(config=config, manifest=manifest)
    adapter: ModelWeightAdapter | None = getattr(module, "weight_adapter", None)
    if adapter is not None:
        return adapter.adapt(module, state_dict, context)

    preprocess = getattr(module, "preprocess_weights", None)
    if preprocess is None:
        return dict(state_dict)
    result: Any = preprocess(dict(state_dict))
    if not isinstance(result, dict):
        raise TypeError(
            f"{type(module).__name__}.preprocess_weights must return a dict, "
            f"got {type(result).__name__}"
        )
    return result
