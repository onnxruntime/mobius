# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Pinned MatterGen checkpoint configuration integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "MATTERGEN_CONDITION_FAMILY",
    "MATTERGEN_CONDITION_SPECS",
    "MATTERGEN_HUB_REVISION",
    "MATTERGEN_MODEL_ID",
    "MATTERGEN_SOURCE_COMMIT",
    "MatterGenConditionSpec",
    "MatterGenConfig",
    "MatterGenGemNetTModel",
    "MatterGenGraph",
    "MatterGenHostSampler",
    "MatterGenSampleBatch",
    "MatterGenScoreCallback",
    "MatterGenScoreInputs",
    "MatterGenScoreOutputs",
    "MatterGenModel",
    "MatterGenCrystal",
    "build_mattergen",
    "build_periodic_graph",
    "create_onnxruntime_score_callback",
    "is_mattergen_checkpoint",
]

if TYPE_CHECKING:
    from mobius.integrations.mattergen._builder import (
        build_mattergen,
        is_mattergen_checkpoint,
    )
    from mobius.integrations.mattergen._configs import (
        MATTERGEN_CONDITION_FAMILY,
        MATTERGEN_CONDITION_SPECS,
        MATTERGEN_HUB_REVISION,
        MATTERGEN_MODEL_ID,
        MATTERGEN_SOURCE_COMMIT,
        MatterGenConditionSpec,
        MatterGenConfig,
    )
    from mobius.integrations.mattergen._runtime import (
        MatterGenCrystal,
        MatterGenGraph,
        MatterGenHostSampler,
        MatterGenSampleBatch,
        MatterGenScoreCallback,
        MatterGenScoreInputs,
        MatterGenScoreOutputs,
        build_periodic_graph,
        create_onnxruntime_score_callback,
    )
    from mobius.models.mattergen import MatterGenGemNetTModel, MatterGenModel


def __getattr__(name: str):
    """Lazily expose configuration without creating model import cycles."""
    if name in {
        "MATTERGEN_CONDITION_FAMILY",
        "MATTERGEN_CONDITION_SPECS",
        "MATTERGEN_HUB_REVISION",
        "MATTERGEN_MODEL_ID",
        "MATTERGEN_SOURCE_COMMIT",
        "MatterGenConditionSpec",
        "MatterGenConfig",
    }:
        from mobius.integrations.mattergen import _configs

        return getattr(_configs, name)
    if name in {"MatterGenGemNetTModel", "MatterGenModel"}:
        from mobius.models.mattergen import MatterGenGemNetTModel, MatterGenModel

        return {
            "MatterGenGemNetTModel": MatterGenGemNetTModel,
            "MatterGenModel": MatterGenModel,
        }[name]
    if name in {"build_mattergen", "is_mattergen_checkpoint"}:
        from mobius.integrations.mattergen import _builder

        return getattr(_builder, name)
    if name in {
        "MatterGenCrystal",
        "MatterGenGraph",
        "MatterGenHostSampler",
        "MatterGenSampleBatch",
        "MatterGenScoreCallback",
        "MatterGenScoreInputs",
        "MatterGenScoreOutputs",
        "build_periodic_graph",
        "create_onnxruntime_score_callback",
    }:
        from mobius.integrations.mattergen import _runtime

        return getattr(_runtime, name)
    raise AttributeError(name)
