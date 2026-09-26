# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import mock

import onnx_ir as ir
import pytest

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
                "text_config": {
                    "model_type": "cosmos3_edge_text",
                    "eos_token_id": 11,
                },
                "vision_start_token_id": 20,
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
    reasoner_package = SimpleNamespace(
        config=SimpleNamespace(
            vision=SimpleNamespace(
                patch_size=16,
                spatial_merge_size=2,
                in_channels=3,
                temporal_patch_size=1,
            ),
            mrope_section=[24, 20, 20],
        )
    )
    reasoner_module = mock.sentinel.reasoner_module
    generator_package = mock.sentinel.generator_package
    generator_module = SimpleNamespace(config=mock.sentinel.generator_config)
    vae_package = mock.sentinel.vae_package
    vae_module = SimpleNamespace(config=mock.sentinel.vae_config)
    package = mock.sentinel.pipeline_package

    with (
        mock.patch(
            "mobius._cosmos3_edge_world_model.load_checkpoint_json",
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
            return_value={
                "assets/negative_prompt.json": (
                    "cached/assets/negative_prompt.json",
                    False,
                )
            },
        ),
        mock.patch(
            "mobius._cosmos3_edge_world_model.load_optional_checkpoint_json",
            return_value={},
        ),
        mock.patch(
            "mobius._cosmos3_edge_world_model.resolve_checkpoint_file",
            return_value=None,
        ),
        mock.patch(
            "mobius._cosmos3_edge_world_model._compose_pipeline",
            return_value=package,
        ) as compose,
    ):
        result = build_cosmos3_edge_world_model(
            "nvidia/Cosmos3-Edge",
            dtype="bf16",
            load_weights=False,
        )

    assert result is package
    assert build_components.call_args.kwargs["reasoner_module_class"] is Cosmos3EdgeVLModel
    assert build_components.call_args.kwargs["reasoner_task"] == "cosmos3-edge-vl"
    build_config = build_components.call_args.kwargs["build_config"]
    assert build_config.resolved_dtype() is ir.DataType.BFLOAT16
    assert build_config.load_weights is False
    pipeline_config = compose.call_args.kwargs["pipeline_config"]
    assert pipeline_config.model_type == "cosmos3_edge"
    assert pipeline_config.model_id == "nvidia/Cosmos3-Edge"
    assert pipeline_config.build is build_config
    assert compose.call_args.kwargs["reasoner_architecture"] == "cosmos3_edge"
    assert pipeline_config.extra_metadata["edge"]["checkpoint_model_type"] == "cosmos3_edge"
    assert pipeline_config.generation.default_inference_steps == 50
    assert pipeline_config.generation.scheduler_mode_overrides_manifest() == {
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
    i2v = pipeline_config.extra_metadata["generation_recipes"]["image_to_video"]
    assert i2v["conditioning"]["conditioned_latent_frames"] == [0]
    assert i2v["prompt"]["negative_asset"] == "assets/negative_prompt.json"
    assert i2v["prompt"]["negative_default"] == "asset"
    assert (i2v["width"], i2v["height"], i2v["frames"]) == (832, 480, 121)
    assert pipeline_config.extra_metadata["generator_prompt"] == {
        "chat": {
            "add_generation_prompt": True,
            "add_vision_id": False,
            "enable_thinking": True,
        },
        "suffix_token_ids": [11, 20],
    }
    vision = pipeline_config.extra_metadata["vision_understanding"]
    assert vision["encoder"] == "reasoner_vision_encoder"
    assert vision["tokens"] == {
        "image": 19,
        "video": 18,
        "vision_start": 20,
        "vision_end": 21,
    }
    assert vision["routing"]["video"] == "reasoner_embedding.video_features"
    preprocessing = vision["preprocessing"]
    assert preprocessing["patchify"] == {
        "layout": "time_major_block_major",
        "patch_value_order": "patch_height_patch_width_channel",
        "patch_size": 16,
        "merge_size": 2,
        "temporal_patch_size": 1,
        "patch_dim": 16 * 16 * 3,
    }
    assert preprocessing["alignment"] == 32
    # Class defaults that the shipped preprocessor_config.json assets omit.
    assert preprocessing["resample"] == "bicubic"
    assert preprocessing["rescale_factor"] == pytest.approx(1 / 255)
    assert preprocessing["convert_rgb"] is True
    assert vision["position_ids"]["mrope"] == "interleaved"
    assert vision["position_ids"]["mrope_section"] == [24, 20, 20]
    assert "grid_t=1" in vision["position_ids"]["video_index_rule"]


def test_edge_builder_rejects_non_edge_checkpoint(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "cosmos3_omni"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not a Cosmos3-Edge checkpoint"):
        build_cosmos3_edge_world_model(str(tmp_path), load_weights=False)
