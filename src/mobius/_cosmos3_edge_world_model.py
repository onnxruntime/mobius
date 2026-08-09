# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Complete world-model exporter for NVIDIA Cosmos3-Edge checkpoints."""

from __future__ import annotations

__all__ = ["build_cosmos3_edge_world_model"]

import json
from collections.abc import Mapping
from typing import Any

import onnx_ir as ir

from mobius._configs.per_model import _cosmos3_edge_vision  # noqa: F401
from mobius._cosmos3_world_model import (
    _apply_checkpoint_weights,
    _build_components,
    _collect_assets,
    _compose_pipeline,
)
from mobius._diffusers_checkpoint import (
    component_class,
    load_checkpoint_json,
    load_optional_checkpoint_json,
    resolve_checkpoint_file,
)
from mobius._pipeline import PipelinePackage
from mobius._world_model_config import (
    WorldModelBuildConfig,
    WorldModelGenerationConfig,
    WorldModelPipelineConfig,
)
from mobius.models.cosmos import Cosmos3EdgeVLModel

# Cosmos3-Edge tunes the rectified-flow scheduler per generation mode. These
# values belong to the Edge checkpoint contract, not to a generic default.
_SCHEDULER_MODE_OVERRIDES: dict[str, Any] = {
    "image_to_video": {
        "flow_shift": 3.0,
        "use_karras_sigmas": False,
    },
    "action": {
        "flow_shift": 10.0,
        "use_karras_sigmas": False,
    },
}
_DEFAULT_INFERENCE_STEPS = 50


def _edge_text_model_type(root_config: Mapping[str, Any]) -> str | None:
    text_config = root_config.get("text_config")
    if isinstance(text_config, Mapping):
        model_type = text_config.get("model_type")
        return model_type if isinstance(model_type, str) else None
    return None


def build_cosmos3_edge_world_model(
    model_id: str,
    *,
    dtype: str | ir.DataType | None = None,
    load_weights: bool = True,
    execution_provider: str = "default",
    trace_optimization: bool = False,
    **_options: Any,
) -> PipelinePackage:
    """Build the complete Cosmos3-Edge Reasoner/Generator/VAE/Action package.

    Both ``nvidia/Cosmos3-Edge`` and the historically mislabeled
    ``nvidia/Cosmos3-Edge-Policy-DROID`` use the Edge text/vision architecture.
    The latter advertises top-level ``model_type="cosmos3_omni"``; dispatch is
    therefore based on ``text_config.model_type="cosmos3_edge_text"``.
    """
    build_config = WorldModelBuildConfig(
        dtype=dtype,
        load_weights=load_weights,
        execution_provider=execution_provider,
        trace_optimization=trace_optimization,
    )
    root_config, _ = load_checkpoint_json(model_id, "config.json")
    if _edge_text_model_type(root_config) != "cosmos3_edge_text":
        raise ValueError(
            f"{model_id!r} is not a Cosmos3-Edge checkpoint: expected "
            "text_config.model_type='cosmos3_edge_text'."
        )

    pipeline_index, _ = load_checkpoint_json(model_id, "model_index.json")
    transformer_class = component_class(pipeline_index, "transformer")
    vae_class = component_class(pipeline_index, "vae")
    sound_class = component_class(pipeline_index, "sound_tokenizer")
    if transformer_class != "Cosmos3OmniTransformer":
        raise ValueError(f"Unsupported Cosmos3-Edge transformer class {transformer_class!r}")
    if vae_class != "AutoencoderKLWan":
        raise ValueError(f"Unsupported Cosmos3-Edge VAE class {vae_class!r}")
    if sound_class is not None:
        raise ValueError(
            "Cosmos3-Edge Sound generation is not supported by the public architecture; "
            f"unexpected sound tokenizer {sound_class!r}."
        )

    transformer_config, _ = load_checkpoint_json(model_id, "transformer/config.json")
    vae_config, _ = load_checkpoint_json(model_id, "vae/config.json")
    scheduler_config, _ = load_checkpoint_json(model_id, "scheduler/scheduler_config.json")
    generation_config = WorldModelGenerationConfig.from_generation_config(
        load_optional_checkpoint_json(model_id, "generation_config.json"),
        default_inference_steps=_DEFAULT_INFERENCE_STEPS,
        scheduler_mode_overrides=_SCHEDULER_MODE_OVERRIDES,
    )

    (
        reasoner_package,
        reasoner_module,
        generator_package,
        generator_module,
        vae_package,
        vae_module,
        audio_package,
        audio_module,
    ) = _build_components(
        model_id,
        build_config=build_config,
        pipeline_index=pipeline_index,
        transformer_config_dict=transformer_config,
        vae_config_dict=vae_config,
        audio_config_dict=None,
        audio_weight_names=None,
        has_reasoner_vision=True,
        reasoner_module_class=Cosmos3EdgeVLModel,
        reasoner_task="cosmos3-edge-vl",
    )
    assert audio_package is None and audio_module is None

    if build_config.load_weights:
        _apply_checkpoint_weights(
            model_id,
            reasoner_package=reasoner_package,
            reasoner_module=reasoner_module,
            generator_package=generator_package,
            generator_module=generator_module,
            vae_package=vae_package,
            vae_module=vae_module,
            audio_package=None,
            audio_module=None,
        )

    assets = _collect_assets(model_id, has_sound_tokenizer=False)
    policy: dict[str, Any] | None = None
    checkpoint_path = resolve_checkpoint_file(model_id, "checkpoint.json", required=False)
    if checkpoint_path is not None:
        with open(checkpoint_path, encoding="utf-8") as handle:
            checkpoint = json.load(handle)
        if isinstance(checkpoint, Mapping) and isinstance(checkpoint.get("policy"), dict):
            policy = dict(checkpoint["policy"])
    return _compose_pipeline(
        pipeline_config=WorldModelPipelineConfig(
            model_id=model_id,
            model_type="cosmos3_edge",
            build=build_config,
            generation=generation_config,
            extra_metadata={
                "edge": {
                    "checkpoint_model_type": root_config.get("model_type"),
                    "policy": policy,
                }
            },
        ),
        reasoner_package=reasoner_package,
        generator_package=generator_package,
        vae_package=vae_package,
        audio_package=None,
        generator_config=generator_module.config,
        vae_config=vae_module.config,
        scheduler_config=scheduler_config,
        assets=assets,
        reasoner_architecture="cosmos3_edge",
        default_action_domain=(
            policy.get("domain_name", "no_action") if policy is not None else "no_action"
        ),
    )
