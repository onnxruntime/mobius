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
    _REASONER_NAMES,
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

# Official Edge generation recipes use 50 steps for I2V and a mode-specific
# flow schedule. Action recipes use a separate flow schedule.
_SCHEDULER_MODE_OVERRIDES: dict[str, Any] = {
    "image_to_video": {
        "flow_shift": 3.0,
        "use_karras_sigmas": False,
        "num_inference_steps": 50,
        "guidance_scale": 5.0,
    },
    "action": {
        "flow_shift": 10.0,
        "use_karras_sigmas": False,
        "num_inference_steps": 30,
        "guidance_scale": 1.0,
    },
}
_DEFAULT_INFERENCE_STEPS = 50

# Cosmos3-Edge Reasoner vision/video-understanding contract. The exported
# ``reasoner_vision_encoder`` graph consumes the *pre-patchified* tensor that
# ``Cosmos3EdgeImageProcessor`` / ``Cosmos3EdgeVideoProcessor`` produce, so the
# manifest has to spell out that preprocessing for a runtime to reproduce it.
_IMAGE_PROCESSOR_ASSET = "preprocessor_config.json"
_VIDEO_PROCESSOR_ASSET = "video_preprocessor_config.json"


def _vision_understanding_metadata(
    root_config: Mapping[str, Any],
    reasoner_config: Any,
) -> dict[str, Any]:
    """Describe the Reasoner's packed image/video input contract."""
    vision = getattr(reasoner_config, "vision", None)
    patch_size = getattr(vision, "patch_size", None) or 16
    merge_size = getattr(vision, "spatial_merge_size", None) or 2
    channels = getattr(vision, "in_channels", None) or 3
    temporal_patch_size = getattr(vision, "temporal_patch_size", None) or 1
    return {
        "encoder": _REASONER_NAMES["vision_encoder"],
        "invocation": "once_per_visual_item",
        "tokens": {
            "image": root_config.get("image_token_id", 19),
            "video": root_config.get("video_token_id", 18),
            "vision_start": root_config.get("vision_start_token_id", 20),
            "vision_end": root_config.get("vision_end_token_id", 21),
        },
        "token_expansion": {
            "image": (
                "<|vision_start|> + grid_h*grid_w/merge_size**2 image tokens + <|vision_end|>"
            ),
            "video": (
                "one '<T.T seconds>' timestamp followed by "
                "<|vision_start|> + grid_h*grid_w/merge_size**2 video tokens + "
                "<|vision_end|>, repeated per sampled frame"
            ),
        },
        "routing": {
            "image": f"{_REASONER_NAMES['embedding']}.image_features",
            "video": f"{_REASONER_NAMES['embedding']}.video_features",
        },
        "presence": {"video": "video_understanding"},
        "preprocessing": {
            "image_processor_asset": _IMAGE_PROCESSOR_ASSET,
            "video_processor_asset": _VIDEO_PROCESSOR_ASSET,
            # ``size`` in the shipped processor assets holds pixel *areas*
            # (shortest_edge/longest_edge), not edge lengths.
            "resize": "smart_resize_area_bounded_multiple_of_patch_times_merge",
            "alignment": patch_size * merge_size,
            # Class defaults of Cosmos3EdgeImageProcessor /
            # Cosmos3EdgeVideoProcessor; they are NOT serialised into the
            # shipped *_preprocessor_config.json assets.
            "resample": "bicubic",
            "convert_rgb": True,
            "rescale_factor": 1 / 255,
            "normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
            "video_frame_sampling": {"fps": 2, "min_frames": 4, "max_frames": 768},
            "patchify": {
                "layout": "time_major_block_major",
                "patch_value_order": "patch_height_patch_width_channel",
                "patch_size": patch_size,
                "merge_size": merge_size,
                "temporal_patch_size": temporal_patch_size,
                "patch_dim": patch_size * patch_size * channels * temporal_patch_size,
            },
        },
        "grid_thw": {
            "layout": "t_h_w",
            "units": "patches",
            "source": "image_grid_thw[i] / video_grid_thw[i]",
            "note": "grid_h and grid_w must be multiples of merge_size",
        },
        "position_ids": {
            "mrope": "interleaved",
            "mrope_section": list(getattr(reasoner_config, "mrope_section", None) or []),
            "axis_assignment": (
                "channel i uses height when i%3==1 and i<3*mrope_section[1], "
                "width when i%3==2 and i<3*mrope_section[2], temporal otherwise"
            ),
            "video_index_rule": (
                "expand video_grid_thw to one row per frame and set grid_t=1, so "
                "every frame is an independent visual span for position indexing"
            ),
        },
    }


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
    negative_prompt_asset = "assets/negative_prompt.json"
    i2v_prompt: dict[str, Any] = {
        "positive": "json_or_text",
        "negative_default": (
            "asset" if negative_prompt_asset in assets else "empty"
        ),
        "add_resolution_template": False,
        "add_duration_template": False,
        "use_system_prompt": False,
    }
    if negative_prompt_asset in assets:
        i2v_prompt["negative_asset"] = negative_prompt_asset
    text_config = root_config.get("text_config")
    eos_token_id = (
        text_config.get("eos_token_id")
        if isinstance(text_config, Mapping)
        else root_config.get("eos_token_id")
    )
    vision_start_token_id = root_config.get("vision_start_token_id")
    if not isinstance(eos_token_id, int) or not isinstance(vision_start_token_id, int):
        raise TypeError(
            "Cosmos3-Edge requires integer text_config.eos_token_id and "
            "vision_start_token_id for generator prompt packing."
        )
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
                },
                "generation_recipes": {
                    "image_to_video": {
                        "conditioning": {
                            "modality": "image",
                            "encoder_stage": "encode_video",
                            "conditioned_latent_frames": [0],
                        },
                        "prompt": i2v_prompt,
                        "height": 480,
                        "width": 832,
                        "frames": 121,
                        "fps": 24.0,
                    },
                },
                "generator_prompt": {
                    "chat": {
                        "add_generation_prompt": True,
                        "add_vision_id": False,
                        "enable_thinking": True,
                    },
                    "suffix_token_ids": [
                        eos_token_id,
                        vision_start_token_id,
                    ],
                },
                "vision_understanding": _vision_understanding_metadata(
                    root_config, reasoner_package.config
                ),
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
