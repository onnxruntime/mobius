# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import dataclasses

import onnx_ir as ir
import pytest

from mobius import (
    WorldModelBuildConfig as PublicWorldModelBuildConfig,
)
from mobius import (
    WorldModelGenerationConfig as PublicWorldModelGenerationConfig,
)
from mobius import (
    WorldModelPipelineConfig as PublicWorldModelPipelineConfig,
)
from mobius._world_model_config import (
    WorldModelBuildConfig,
    WorldModelGenerationConfig,
    WorldModelPipelineConfig,
)


def test_world_model_configs_are_public() -> None:
    assert PublicWorldModelBuildConfig is WorldModelBuildConfig
    assert PublicWorldModelGenerationConfig is WorldModelGenerationConfig
    assert PublicWorldModelPipelineConfig is WorldModelPipelineConfig


def test_build_config_resolves_dtype_aliases() -> None:
    assert WorldModelBuildConfig().resolved_dtype() is None
    assert WorldModelBuildConfig(dtype="f16").resolved_dtype() is ir.DataType.FLOAT16
    assert (
        WorldModelBuildConfig(dtype=ir.DataType.BFLOAT16).resolved_dtype()
        is ir.DataType.BFLOAT16
    )


def test_build_config_rejects_unknown_settings() -> None:
    with pytest.raises(ValueError, match="Unknown dtype"):
        WorldModelBuildConfig(dtype="float128")
    with pytest.raises(ValueError, match="execution_provider"):
        WorldModelBuildConfig(execution_provider="")


def test_build_config_is_immutable() -> None:
    config = WorldModelBuildConfig(dtype="f32")

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.dtype = "f16"  # type: ignore[misc]


def test_preferred_execution_providers_follow_dtype_support() -> None:
    default = WorldModelBuildConfig()

    assert default.preferred_execution_providers(ir.DataType.FLOAT) == ("cuda", "dml", "cpu")
    # DirectML has no bfloat16 support, so it is omitted for bfloat16 graphs.
    assert default.preferred_execution_providers(ir.DataType.BFLOAT16) == ("cuda", "cpu")
    assert WorldModelBuildConfig(execution_provider="cpu").preferred_execution_providers(
        ir.DataType.FLOAT16
    ) == ("cpu",)


def test_generation_config_parses_huggingface_mapping() -> None:
    config = WorldModelGenerationConfig.from_generation_config(
        {
            "do_sample": True,
            "temperature": 0.6,
            "top_k": 20,
            "top_p": 0.95,
            "repetition_penalty": 1.05,
            "max_new_tokens": 1024,
            "eos_token_id": [151645, 151643],
        },
        default_inference_steps=35,
    )

    assert config.do_sample is True
    assert config.temperature == pytest.approx(0.6)
    assert config.top_k == 20
    assert config.top_p == pytest.approx(0.95)
    assert config.repetition_penalty == pytest.approx(1.05)
    assert config.max_new_tokens == 1024
    assert config.eos_token_ids == (151645, 151643)
    assert config.default_inference_steps == 35


def test_generation_config_normalizes_scalar_and_absent_eos() -> None:
    scalar = WorldModelGenerationConfig.from_generation_config({"eos_token_id": 7})
    explicit_null = WorldModelGenerationConfig.from_generation_config({"eos_token_id": None})

    assert scalar.eos_token_ids == (7,)
    assert explicit_null.eos_token_ids == ()
    assert WorldModelGenerationConfig.from_generation_config({}).eos_token_ids == ()
    assert WorldModelGenerationConfig.from_generation_config(None).eos_token_ids == ()


def test_generation_config_defaults_match_huggingface() -> None:
    config = WorldModelGenerationConfig.from_generation_config({})

    assert config.sampling_manifest() == {
        "do_sample": False,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
    }


def test_generation_config_treats_explicit_nulls_as_unset() -> None:
    config = WorldModelGenerationConfig.from_generation_config(
        {
            "do_sample": None,
            "temperature": None,
            "top_k": None,
            "top_p": None,
            "repetition_penalty": None,
            "max_new_tokens": None,
        }
    )

    assert config.sampling_manifest() == {
        "do_sample": False,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
    }
    assert config.max_new_tokens is None


def test_generation_manifest_blocks_are_deterministic() -> None:
    config = WorldModelGenerationConfig(
        do_sample=True,
        temperature=0.6,
        top_k=20,
        top_p=0.95,
        repetition_penalty=1.05,
        max_new_tokens=64,
        eos_token_ids=(2,),
        default_inference_steps=4,
        scheduler_mode_overrides={"action": {"flow_shift": 10.0}},
    )

    assert list(config.sampling_manifest()) == [
        "do_sample",
        "temperature",
        "top_k",
        "top_p",
        "repetition_penalty",
    ]
    assert config.stop_manifest(max_sequence_length=128) == {
        "kind": "token_ids",
        "eos_token_ids": [2],
        "max_sequence_length": 128,
    }
    assert config.max_tokens_manifest(limit=128) == {
        "default": 64,
        "required_override": False,
        "limit": 128,
    }
    assert config.scheduler_mode_overrides_manifest() == {"action": {"flow_shift": 10.0}}


def test_generation_config_requires_runtime_budget_without_max_new_tokens() -> None:
    config = WorldModelGenerationConfig()

    assert config.max_tokens_manifest(limit=None) == {
        "default": None,
        "required_override": True,
        "limit": None,
    }


def test_generation_config_copies_and_freezes_overrides() -> None:
    overrides = {"action": {"flow_shift": 10.0}}
    config = WorldModelGenerationConfig(scheduler_mode_overrides=overrides)
    overrides["action"] = {"flow_shift": 1.0}

    assert config.scheduler_mode_overrides["action"] == {"flow_shift": 10.0}
    with pytest.raises(TypeError):
        config.scheduler_mode_overrides["extra"] = {}  # type: ignore[index]
    # The manifest mapping is a mutable copy, so callers cannot corrupt the config.
    manifest = config.scheduler_mode_overrides_manifest()
    manifest["extra"] = {}
    assert "extra" not in config.scheduler_mode_overrides


def test_generation_config_rejects_unusable_values() -> None:
    with pytest.raises(ValueError, match="temperature"):
        WorldModelGenerationConfig(temperature=-1.0)
    with pytest.raises(ValueError, match="top_k"):
        WorldModelGenerationConfig(top_k=-1)
    with pytest.raises(ValueError, match="max_new_tokens"):
        WorldModelGenerationConfig(max_new_tokens=0)
    with pytest.raises(ValueError, match="default_inference_steps"):
        WorldModelGenerationConfig(default_inference_steps=0)
    with pytest.raises(TypeError, match="eos_token_ids"):
        WorldModelGenerationConfig(eos_token_ids=("2",))  # type: ignore[arg-type]


def test_pipeline_config_derives_profile_and_metadata() -> None:
    config = WorldModelPipelineConfig(
        model_id="nvidia/Cosmos3-Omni",
        model_type="cosmos3_omni",
    )

    assert config.profile_name == "cosmos3-omni"
    assert config.profile_version == "1.0"
    assert config.manifest_metadata() == {
        "profile": "world-model",
        "model_type": "cosmos3_omni",
        "source": "nvidia/Cosmos3-Omni",
    }
    assert list(config.manifest_metadata()) == ["profile", "model_type", "source"]


def test_pipeline_config_holds_build_and_generation_settings() -> None:
    build = WorldModelBuildConfig(dtype="bf16", execution_provider="cuda")
    generation = WorldModelGenerationConfig(default_inference_steps=50)
    config = WorldModelPipelineConfig(
        model_id="example/world",
        model_type="example_world",
        build=build,
        generation=generation,
        extra_metadata={"edge": {"policy": None}},
    )

    assert config.build is build
    assert config.generation is generation
    assert config.extra_metadata["edge"] == {"policy": None}
    with pytest.raises(TypeError):
        config.extra_metadata["edge"] = {}  # type: ignore[index]


def test_pipeline_config_requires_identity() -> None:
    with pytest.raises(ValueError, match="model_id"):
        WorldModelPipelineConfig(model_id="", model_type="example")
    with pytest.raises(ValueError, match="model_type"):
        WorldModelPipelineConfig(model_id="example/world", model_type="")
    with pytest.raises(ValueError, match="profile_version"):
        WorldModelPipelineConfig(
            model_id="example/world",
            model_type="example",
            profile_version="",
        )


def test_pipeline_config_defaults_are_independent() -> None:
    first = WorldModelPipelineConfig(model_id="a/b", model_type="t")
    second = WorldModelPipelineConfig(model_id="a/b", model_type="t")

    assert first.build == second.build
    assert first.generation == second.generation
    assert first.build is not second.build
