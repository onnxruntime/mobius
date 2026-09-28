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
from mobius._weight_utils import materialize_split_tied_olive_lm_head


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


def _materialize_shared_quantized_weights(
    state_dict: dict[str, torch.Tensor],
    context: WeightAdapterContext,
) -> None:
    """Materialize non-canonical packed aliases before model-specific routing."""
    if context.config.component_quantization is None:
        return
    for shared_weight in context.manifest.shared_weights:
        if shared_weight.kind != "tied_word_embeddings":
            continue
        canonical = shared_weight.canonical
        canonical_module = canonical.parameter.removesuffix(".weight")
        embedding_quantization = context.config.quantization_for_source_paths(
            canonical.component,
            (canonical_module,),
            ignored_source_names=(),
        )
        for alias in shared_weight.aliases:
            alias_module = alias.parameter.removesuffix(".weight")
            materialize_split_tied_olive_lm_head(
                state_dict,
                embed_key=canonical.parameter,
                head_key=alias.parameter,
                embedding_quantization=embedding_quantization,
                head_quantization=context.config.quantization_for_source_paths(
                    alias.component,
                    (alias_module,),
                    ignored_source_names=(),
                ),
            )


def adapt_model_weights(
    module: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    config: BaseModelConfig,
    manifest: ComponentManifest,
) -> dict[str, torch.Tensor]:
    """Run an explicit adapter or the legacy ``preprocess_weights`` hook."""
    context = WeightAdapterContext(config=config, manifest=manifest)
    state_dict = dict(state_dict)
    _materialize_shared_quantized_weights(state_dict, context)
    adapter: ModelWeightAdapter | None = getattr(module, "weight_adapter", None)
    if adapter is not None:
        return adapter.adapt(module, state_dict, context)

    preprocess = getattr(module, "preprocess_weights", None)
    if preprocess is None:
        return state_dict
    result: Any = preprocess(state_dict)
    if not isinstance(result, dict):
        raise TypeError(
            f"{type(module).__name__}.preprocess_weights must return a dict, "
            f"got {type(result).__name__}"
        )
    return result
