# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import onnx_ir as ir
import pytest
import safetensors.torch
import torch

from mobius._configs import Cosmos3AudioConfig, Cosmos3OmniGeneratorConfig, WanVAEConfig
from mobius._cosmos3_world_model import (
    _component_weight_names,
    _compose_pipeline,
)
from mobius._model_package import ModelPackage


def _value(
    name: str,
    dtype: ir.DataType,
    shape: list[int | str],
) -> ir.Value:
    return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))


def _model(
    inputs: dict[str, tuple[ir.DataType, list[int | str]]],
    outputs: dict[str, tuple[ir.DataType, list[int | str]]],
) -> ir.Model:
    input_values = [_value(name, dtype, shape) for name, (dtype, shape) in inputs.items()]
    nodes: list[ir.Node] = []
    output_values: list[ir.Value] = []
    for name, (dtype, shape) in outputs.items():
        node = ir.Node("", "Identity", inputs=[input_values[0]], num_outputs=1)
        output = node.outputs[0]
        output.name = name
        output.type = ir.TensorType(dtype)
        output.shape = ir.Shape(shape)
        nodes.append(node)
        output_values.append(output)
    graph = ir.Graph(
        input_values,
        output_values,
        nodes=nodes,
        name="component",
        opset_imports={"": 24},
    )
    return ir.Model(graph, ir_version=10)


def _generator_config(*, sound: bool = True, action: bool = True):
    return Cosmos3OmniGeneratorConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        hidden_act="silu",
        rope_axes_dim=(2, 1, 1),
        latent_channel=16,
        latent_patch_size=2,
        patch_latent_dim=64,
        sound_gen=sound,
        sound_dim=4 if sound else None,
        action_gen=action,
        action_dim=64 if action else None,
        max_action_dim=64 if action else None,
        num_embodiment_domains=32,
        dtype=ir.DataType.FLOAT,
    )


def _packages(*, sound: bool = True, action: bool = True, audio_encoder: bool = True):
    f32 = ir.DataType.FLOAT
    i64 = ir.DataType.INT64
    reasoner_config = SimpleNamespace(dtype=f32)
    reasoner = ModelPackage(
        {
            "vision_encoder": _model(
                {"pixel_values": (f32, ["b", 3, 8, 8])},
                {"image_features": (f32, ["b", "n", 16])},
            ),
            "embedding": _model(
                {
                    "input_ids": (i64, ["b", "s"]),
                    "image_features": (f32, ["b", "n", 16]),
                },
                {"inputs_embeds": (f32, ["b", "s", 16])},
            ),
            "decoder": _model(
                {
                    "inputs_embeds": (f32, ["b", "s", 16]),
                    "attention_mask": (i64, ["b", "s"]),
                    "position_ids": (i64, [3, "b", "s"]),
                    "past_key_values.0.key": (f32, ["b", 1, "p", 8]),
                    "past_key_values.0.value": (f32, ["b", 1, "p", 8]),
                },
                {
                    "logits": (f32, ["b", "s", 32]),
                    "present.0.key": (f32, ["b", 1, "total", 8]),
                    "present.0.value": (f32, ["b", 1, "total", 8]),
                },
            ),
        },
        config=reasoner_config,
    )

    generator_inputs = {
        "input_ids": (i64, ["text"]),
        "position_ids": (i64, [3, "sequence"]),
        "vision_tokens": (f32, ["vision", 64]),
        "vision_timesteps": (f32, ["vision"]),
    }
    generator_outputs = {"vision_pred": (f32, ["vision", 64])}
    if sound:
        generator_inputs["sound_tokens"] = (f32, ["sound", 4])
        generator_inputs["sound_timesteps"] = (f32, ["sound"])
        generator_outputs["sound_pred"] = (f32, ["sound", 4])
    if action:
        generator_inputs["action_tokens"] = (f32, ["action", 64])
        generator_inputs["action_timesteps"] = (f32, ["action"])
        generator_outputs["action_pred"] = (f32, ["action", 64])
    generator = ModelPackage(
        {"model": _model(generator_inputs, generator_outputs)},
        config=_generator_config(sound=sound, action=action),
    )

    vae_config = WanVAEConfig()
    vae = ModelPackage(
        {
            "encoder": _model(
                {"sample": (f32, ["b", 3, "t", "h", "w"])},
                {"latent": (f32, ["b", 16, "lt", "lh", "lw"])},
            ),
            "decoder": _model(
                {"latent": (f32, ["b", 16, "lt", "lh", "lw"])},
                {"sample": (f32, ["b", 3, "t", "h", "w"])},
            ),
        },
        config=vae_config,
    )

    audio = None
    if sound:
        audio_config = Cosmos3AudioConfig(
            dec_dim=4,
            dec_c_mults=(1,),
            dec_strides=(2,),
            enc_dim=4,
            enc_intermediate_dim=16,
            enc_num_layers=1,
            enc_num_blocks=1,
            enc_n_fft=4,
            enc_hop_length=2,
            enc_c_mults=(1,),
            enc_strides=(1,),
            enc_latent_dim=8,
            vocoder_input_dim=4,
            hop_size=2,
            encoder_enabled=audio_encoder,
        )
        audio_models = {
            "decoder": _model(
                {"latents": (f32, ["b", 4, "t"])},
                {"waveform": (f32, ["b", 2, "samples"])},
            )
        }
        if audio_encoder:
            audio_models["encoder"] = _model(
                {"audio": (f32, ["b", 2, "samples"])},
                {"latent_mean": (f32, ["b", 4, "t"])},
            )
        audio = ModelPackage(audio_models, config=audio_config)
    return reasoner, generator, vae, audio


def _runtime_assets(tmp_path):
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    scheduler_dir = tmp_path / "scheduler"
    scheduler_dir.mkdir(exist_ok=True)
    scheduler = scheduler_dir / "scheduler_config.json"
    scheduler.write_text("{}", encoding="utf-8")
    return {
        "tokenizer.json": (str(tokenizer), True),
        "scheduler/scheduler_config.json": (str(scheduler), True),
    }


def test_compose_complete_cosmos3_pipeline(tmp_path) -> None:
    reasoner, generator, vae, audio = _packages()
    config = generator.config
    package = _compose_pipeline(
        model_id="nvidia/Cosmos3-Nano",
        reasoner_package=reasoner,
        generator_package=generator,
        vae_package=vae,
        audio_package=audio,
        generator_config=config,
        vae_config=vae.config,
        scheduler_config={"prediction_type": "flow_prediction"},
        assets=_runtime_assets(tmp_path),
    )

    assert set(package) == {
        "reasoner_decoder",
        "reasoner_vision_encoder",
        "reasoner_embedding",
        "generator",
        "video_encoder",
        "video_decoder",
        "audio_encoder",
        "audio_decoder",
    }
    recurrent = [edge for edge in package.manifest.connections if edge.recurrent]
    assert {edge.target.qualified for edge in recurrent} >= {
        "generator.vision_tokens",
        "generator.sound_tokens",
        "generator.action_tokens",
        "reasoner_decoder.past_key_values.0.key",
        "reasoner_decoder.past_key_values.0.value",
    }
    assert "iterative_scheduler" in package.manifest.required_capabilities
    assert {output.name for output in package.manifest.outputs} >= {
        "logits",
        "video",
        "sound",
        "action",
    }
    assert package.manifest.metadata["profile"] == "world-model"
    assert package.manifest.metadata["action"]["domain_ids"]["droid_lerobot"] == 8
    assert package.manifest.metadata["action"]["raw_dimensions"]["hand_pose"] == 57
    assert package.manifest.profile is not None
    assert package.manifest.profile.name == "cosmos3-omni"
    assert {state.kind for state in package.manifest.states} >= {
        "kv_cache",
        "diffusion_latent",
        "action_state",
    }
    position_input = next(
        value
        for value in package.manifest.inputs
        if value.port.qualified == "generator.position_ids"
    )
    assert position_input.generator is not None
    assert position_input.generator.kind == "multimodal_position_ids"
    cache_input = next(
        value
        for value in package.manifest.inputs
        if value.port.qualified == "reasoner_decoder.past_key_values.0.key"
    )
    assert cache_input.generator is not None
    assert cache_input.generator.parameters["dynamic_axes"] == {"p": 0}
    assert package.manifest.component("generator").parameter_dtype == "FLOAT"
    assert package.manifest.component("generator").preferred_execution_providers[0] == ("cuda")


def test_compose_without_optional_sound_or_action(tmp_path) -> None:
    reasoner, generator, vae, audio = _packages(sound=False, action=False)
    package = _compose_pipeline(
        model_id="example/world",
        reasoner_package=reasoner,
        generator_package=generator,
        vae_package=vae,
        audio_package=audio,
        generator_config=generator.config,
        vae_config=vae.config,
        scheduler_config={},
        assets=_runtime_assets(tmp_path),
    )

    assert "audio_decoder" not in package
    assert {output.name for output in package.manifest.outputs}.isdisjoint({"sound", "action"})


def test_sound_head_requires_sound_tokenizer() -> None:
    reasoner, generator, vae, _audio = _packages(sound=True)

    with pytest.raises(ValueError, match="sound_gen=True"):
        _compose_pipeline(
            model_id="example/world",
            reasoner_package=reasoner,
            generator_package=generator,
            vae_package=vae,
            audio_package=None,
            generator_config=generator.config,
            vae_config=vae.config,
            scheduler_config={},
            assets={},
        )


def test_local_audio_weight_metadata_detects_encoder(tmp_path) -> None:
    component = tmp_path / "sound_tokenizer"
    component.mkdir()
    safetensors.torch.save_file(
        {
            "encoder.layers.0.weight": torch.zeros(1),
            "decoder.conv1.weight": torch.zeros(1),
        },
        str(component / "diffusion_pytorch_model.safetensors"),
    )

    names = _component_weight_names(str(tmp_path), "sound_tokenizer")

    assert names == {"encoder.layers.0.weight", "decoder.conv1.weight"}


def test_decoder_only_audio_package_is_supported(tmp_path) -> None:
    reasoner, generator, vae, audio = _packages(audio_encoder=False)
    package = _compose_pipeline(
        model_id="example/world",
        reasoner_package=reasoner,
        generator_package=generator,
        vae_package=vae,
        audio_package=audio,
        generator_config=generator.config,
        vae_config=vae.config,
        scheduler_config={},
        assets=_runtime_assets(tmp_path),
    )

    assert "audio_encoder" not in package
    assert "audio_decoder" in package


def test_reasoner_without_standalone_vision_tower_is_supported(tmp_path) -> None:
    reasoner, generator, vae, audio = _packages(sound=False, action=False)
    del reasoner["vision_encoder"]
    package = _compose_pipeline(
        model_id="example/distilled-world",
        reasoner_package=reasoner,
        generator_package=generator,
        vae_package=vae,
        audio_package=audio,
        generator_config=generator.config,
        vae_config=vae.config,
        scheduler_config={},
        assets=_runtime_assets(tmp_path),
    )

    assert "reasoner_vision_encoder" not in package
    prompt_stage = next(
        stage for stage in package.manifest.stages if stage.name == "reasoner_prompt"
    )
    assert prompt_stage.components == ("reasoner_embedding",)


def test_generator_and_video_vae_latent_width_must_match() -> None:
    reasoner, generator, vae, audio = _packages(sound=False, action=False)
    incompatible = dataclasses.replace(
        generator.config,
        latent_channel=8,
        patch_latent_dim=32,
    )

    with pytest.raises(ValueError, match="latent width"):
        _compose_pipeline(
            model_id="example/world",
            reasoner_package=reasoner,
            generator_package=generator,
            vae_package=vae,
            audio_package=audio,
            generator_config=incompatible,
            vae_config=vae.config,
            scheduler_config={},
            assets={},
        )
