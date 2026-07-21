# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for onnx-genai diffusion inference_metadata generation."""

from __future__ import annotations

import os

import pytest
import yaml

from mobius.integrations.onnx_genai.inference_metadata import (
    SchedulerConfig,
    build_diffusion_pipeline_metadata,
    build_language_diffusion_pipeline_metadata,
    build_multimodal_pipeline_metadata,
    build_tts_pipeline_metadata,
    load_diffusers_scheduler_config,
    write_diffusion_pipeline_metadata,
    write_tts_pipeline_metadata,
)


def _onnx_genai_schema_path() -> str | None:
    """Locate onnx-genai's committed pipeline JSON schema, if available."""
    candidates = [
        os.environ.get("ONNX_GENAI_SCHEMA"),
        os.path.join(
            os.path.dirname(__file__),
            "../../../../../onnx-genai/schema/inference_metadata.schema.json",
        ),
        os.path.expanduser(
            "~/Documents/GitHub/onnx-genai/schema/inference_metadata.schema.json"
        ),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


class TestBuildDiffusionPipelineMetadata:
    def test_denoiser_only_minimal(self):
        meta = build_diffusion_pipeline_metadata(num_inference_steps=20)
        pipe = meta["pipeline"]
        assert set(pipe["models"]) == {"denoiser"}
        assert pipe["strategy"]["kind"] == "iterative"
        assert pipe["strategy"]["denoiser"] == "denoiser"
        assert pipe["strategy"]["num_steps"] == 20
        assert pipe["strategy"]["timestep_input"] == "timestep"
        # Loop-carried self-edge is present.
        assert {"from": "denoiser.noise_pred", "to": "denoiser.sample"} in pipe["dataflow"]
        # Default DDIM scheduler config.
        sched = pipe["strategy"]["scheduler_config"]
        assert sched["kind"] == "ddim"
        assert sched["num_train_timesteps"] == 1000

    def test_full_pipeline_with_vae_and_text_encoder(self):
        meta = build_diffusion_pipeline_metadata(
            num_inference_steps=4,
            vae_filename="vae.onnx",
            text_encoder_filename="text_encoder.onnx",
            guidance_scale=7.5,
        )
        pipe = meta["pipeline"]
        assert set(pipe["models"]) == {"denoiser", "vae", "text_encoder"}
        # Text encoder feeds conditioning (prompt-phase); VAE decodes final latent.
        assert {
            "from": "text_encoder.last_hidden_state",
            "to": "denoiser.encoder_hidden_states",
        } in pipe["dataflow"]
        assert {"from": "denoiser.sample", "to": "vae.latent"} in pipe["dataflow"]
        assert pipe["phases"]["text_encoder"] == {"run_on": "prompt_only"}
        assert pipe["phases"]["vae"] == {"run_on": "final_only"}
        # CFG enabled -> conditioning input declared for the unconditional pass.
        assert pipe["strategy"]["guidance_scale"] == pytest.approx(7.5)
        assert pipe["strategy"]["cfg_conditioning_input"] == "encoder_hidden_states"

    def test_sdxl_dual_conditioning_edges(self):
        meta = build_diffusion_pipeline_metadata(
            num_inference_steps=4,
            vae_filename="vae.onnx",
            text_encoder_filename="text_encoder.onnx",
            guidance_scale=7.5,
            text_encoder_edges=[
                ("encoder_hidden_states", "encoder_hidden_states"),
                ("text_embeds", "text_embeds"),
            ],
        )
        flow = meta["pipeline"]["dataflow"]
        assert {
            "from": "text_encoder.encoder_hidden_states",
            "to": "denoiser.encoder_hidden_states",
        } in flow
        assert {"from": "text_encoder.text_embeds", "to": "denoiser.text_embeds"} in flow

    def test_guidance_scale_one_does_not_enable_cfg(self):
        meta = build_diffusion_pipeline_metadata(num_inference_steps=2, guidance_scale=1.0)
        assert "cfg_conditioning_input" not in meta["pipeline"]["strategy"]

    def test_scheduler_from_diffusers_config(self):
        sched = SchedulerConfig.from_diffusers(
            {
                "_class_name": "DDIMScheduler",
                "num_train_timesteps": 1000,
                "beta_start": 0.0001,
                "beta_end": 0.02,
                "prediction_type": "epsilon",
            }
        )
        assert sched.kind == "ddim"
        assert sched.beta_end == pytest.approx(0.02)

    def test_scheduler_maps_euler_class(self):
        sched = SchedulerConfig.from_diffusers(
            {"_class_name": "EulerDiscreteScheduler", "beta_schedule": "scaled_linear"}
        )
        assert sched.kind == "euler"
        assert sched.beta_schedule == "scaled_linear"

    def test_scheduler_maps_dpmsolver_class(self):
        sched = SchedulerConfig.from_diffusers({"_class_name": "DPMSolverMultistepScheduler"})
        assert sched.kind == "dpmpp_2m"

    def test_scheduler_defaults_to_ddim_when_class_absent(self):
        assert SchedulerConfig.from_diffusers({}).kind == "ddim"

    def test_scheduler_maps_euler_ancestral(self):
        sched = SchedulerConfig.from_diffusers(
            {"_class_name": "EulerAncestralDiscreteScheduler"}
        )
        assert sched.kind == "euler_ancestral"

    def test_scheduler_rejects_ancestral(self):
        # Other ancestral / SDE samplers have no onnx-genai equivalent yet.
        with pytest.raises(ValueError, match="stochastic"):
            SchedulerConfig.from_diffusers({"_class_name": "DPMSolverSDEScheduler"})

    def test_scheduler_rejects_unsupported_class(self):
        with pytest.raises(ValueError, match="unsupported"):
            SchedulerConfig.from_diffusers({"_class_name": "LMSDiscreteScheduler"})

    def test_load_scheduler_from_local_checkpoint(self, tmp_path):
        import json

        sd = tmp_path / "scheduler"
        sd.mkdir()
        (sd / "scheduler_config.json").write_text(
            json.dumps({"_class_name": "EulerDiscreteScheduler", "beta_end": 0.015})
        )
        sched = load_diffusers_scheduler_config(str(tmp_path))
        assert sched is not None
        assert sched.kind == "euler"
        assert sched.beta_end == pytest.approx(0.015)

    def test_load_scheduler_none_when_absent(self, tmp_path):
        assert load_diffusers_scheduler_config(str(tmp_path)) is None
        assert load_diffusers_scheduler_config(None) is None

    def test_load_scheduler_falls_back_on_unsupported(self, tmp_path):
        import json

        sd = tmp_path / "scheduler"
        sd.mkdir()
        (sd / "scheduler_config.json").write_text(
            json.dumps({"_class_name": "LMSDiscreteScheduler"})
        )
        # Unsupported scheduler must not raise from the loader; returns None so
        # the caller falls back to the DDIM default.
        assert load_diffusers_scheduler_config(str(tmp_path)) is None

    def test_rejects_zero_steps(self):
        with pytest.raises(ValueError):
            build_diffusion_pipeline_metadata(num_inference_steps=0)

    def test_write_roundtrips_yaml(self, tmp_path):
        path = write_diffusion_pipeline_metadata(
            str(tmp_path), num_inference_steps=3, vae_filename="vae.onnx"
        )
        with open(path) as handle:
            loaded = yaml.safe_load(handle)
        assert loaded["pipeline"]["strategy"]["num_steps"] == 3
        assert "vae" in loaded["pipeline"]["models"]

    def test_matches_onnx_genai_json_schema(self):
        """The emitted metadata validates against onnx-genai's published schema."""
        schema_path = _onnx_genai_schema_path()
        if schema_path is None:
            pytest.skip("onnx-genai schema not found (set ONNX_GENAI_SCHEMA)")
        import json

        import jsonschema

        with open(schema_path) as handle:
            schema = json.load(handle)
        meta = build_diffusion_pipeline_metadata(
            num_inference_steps=4,
            vae_filename="vae.onnx",
            text_encoder_filename="text_encoder.onnx",
            guidance_scale=7.5,
        )
        # Validate the whole InferenceMetadata document.
        jsonschema.validate(instance=meta, schema=schema)


class TestLanguageDiffusionMetadata:
    def test_minimal_masked_diffusion_pipeline(self):
        meta = build_language_diffusion_pipeline_metadata(
            mask_token_id=126336, num_inference_steps=128
        )
        pipeline = meta["pipeline"]
        assert pipeline["models"]["denoiser"] == {
            "filename": "model.onnx",
            "type": "denoiser",
        }
        # Loop-carried self-edge: logits refine the token sequence.
        assert pipeline["dataflow"] == [
            {"from": "denoiser.logits", "to": "denoiser.input_ids"}
        ]
        strategy = pipeline["strategy"]
        assert strategy["kind"] == "iterative"
        assert strategy["num_steps"] == 128
        assert strategy["scheduler_config"] == {
            "kind": "masked_diffusion",
            "mask_token_id": 126336,
        }
        assert "guidance_scale" not in strategy

    def test_semi_autoregressive_with_temperature_and_cfg(self):
        meta = build_language_diffusion_pipeline_metadata(
            mask_token_id=5,
            num_inference_steps=64,
            block_length=32,
            temperature=0.2,
            guidance_scale=2.5,  # LLaDA cfg_scale=1.5 => cfg_scale + 1
        )
        strategy = meta["pipeline"]["strategy"]
        assert strategy["guidance_scale"] == pytest.approx(2.5)
        assert strategy["scheduler_config"]["block_length"] == 32
        assert strategy["scheduler_config"]["temperature"] == pytest.approx(0.2)

    def test_custom_ports(self):
        meta = build_language_diffusion_pipeline_metadata(
            mask_token_id=1,
            num_inference_steps=8,
            model_filename="llada.onnx",
            input_ids_port="tokens",
            logits_port="scores",
        )
        pipeline = meta["pipeline"]
        assert pipeline["models"]["denoiser"]["filename"] == "llada.onnx"
        assert pipeline["dataflow"] == [{"from": "denoiser.scores", "to": "denoiser.tokens"}]

    def test_rejects_zero_steps(self):
        with pytest.raises(ValueError):
            build_language_diffusion_pipeline_metadata(mask_token_id=1, num_inference_steps=0)

    def test_matches_onnx_genai_json_schema(self):
        schema_path = _onnx_genai_schema_path()
        if schema_path is None:
            pytest.skip("onnx-genai schema not found (set ONNX_GENAI_SCHEMA)")
        import json

        import jsonschema

        with open(schema_path) as handle:
            schema = json.load(handle)
        meta = build_language_diffusion_pipeline_metadata(
            mask_token_id=126336,
            num_inference_steps=64,
            block_length=32,
            temperature=0.0,
            guidance_scale=2.5,
        )
        jsonschema.validate(instance=meta, schema=schema)


class TestBuildMultimodalPipelineMetadata:
    def test_vision_only_pipeline(self):
        metadata = build_multimodal_pipeline_metadata(
            vision_encoder_filename="vision_encoder.onnx"
        )

        assert metadata == {
            "pipeline": {
                "models": {
                    "vision_encoder": {
                        "filename": "vision_encoder.onnx",
                        "type": "vision_encoder",
                    },
                    "embedding": {"filename": "embedding.onnx", "type": "encoder"},
                    "decoder": {
                        "filename": "decoder.onnx",
                        "type": "decoder",
                        "tokenizer": "tokenizer.json",
                    },
                },
                "dataflow": [
                    {
                        "from": "vision_encoder.image_features",
                        "to": "embedding.image_features",
                        "dtype": "fp32",
                        "device_transfer": False,
                    },
                    {
                        "from": "embedding.inputs_embeds",
                        "to": "decoder.inputs_embeds",
                        "dtype": "fp32",
                        "device_transfer": False,
                    },
                ],
                "strategy": {
                    "kind": "composite",
                    "stages": [
                        {
                            "name": "encode_vision",
                            "strategy": {
                                "kind": "single_pass",
                                "model": "vision_encoder",
                            },
                            "run_on": "prompt_only",
                        },
                        {
                            "name": "fuse_embeddings",
                            "strategy": {
                                "kind": "single_pass",
                                "model": "embedding",
                            },
                            "run_on": "prompt_only",
                        },
                        {
                            "name": "decode",
                            "strategy": {
                                "kind": "autoregressive",
                                "decoder": "decoder",
                            },
                            "run_on": "every_step",
                        },
                    ],
                },
                "phases": {
                    "vision_encoder": {"run_on": "prompt_only"},
                    "embedding": {"run_on": "prompt_only"},
                    "decoder": {"run_on": "every_step"},
                },
            }
        }

    def test_vision_and_audio_pipeline(self):
        metadata = build_multimodal_pipeline_metadata(
            vision_encoder_filename="vision_encoder.onnx",
            audio_encoder_filename="audio_encoder.onnx",
        )
        pipeline = metadata["pipeline"]

        assert pipeline["models"] == {
            "vision_encoder": {
                "filename": "vision_encoder.onnx",
                "type": "vision_encoder",
            },
            "audio_encoder": {
                "filename": "audio_encoder.onnx",
                "type": "audio_encoder",
            },
            "embedding": {"filename": "embedding.onnx", "type": "encoder"},
            "decoder": {
                "filename": "decoder.onnx",
                "type": "decoder",
                "tokenizer": "tokenizer.json",
            },
        }
        assert pipeline["dataflow"] == [
            {
                "from": "vision_encoder.image_features",
                "to": "embedding.image_features",
                "dtype": "fp32",
                "device_transfer": False,
            },
            {
                "from": "audio_encoder.audio_features",
                "to": "embedding.audio_features",
                "dtype": "fp32",
                "device_transfer": False,
            },
            {
                "from": "embedding.inputs_embeds",
                "to": "decoder.inputs_embeds",
                "dtype": "fp32",
                "device_transfer": False,
            },
        ]
        assert pipeline["strategy"]["stages"] == [
            {
                "name": "encode_vision",
                "strategy": {"kind": "single_pass", "model": "vision_encoder"},
                "run_on": "prompt_only",
            },
            {
                "name": "encode_audio",
                "strategy": {"kind": "single_pass", "model": "audio_encoder"},
                "run_on": "prompt_only",
            },
            {
                "name": "fuse_embeddings",
                "strategy": {"kind": "single_pass", "model": "embedding"},
                "run_on": "prompt_only",
            },
            {
                "name": "decode",
                "strategy": {"kind": "autoregressive", "decoder": "decoder"},
                "run_on": "every_step",
            },
        ]


class TestBuildTTSPipelineMetadata:
    """Pre-embedder-driven multi-decoder TTS (Qwen3-TTS) metadata."""

    def test_minimal_nested_autoregressive_with_pre_embedder(self):
        meta = build_tts_pipeline_metadata(num_code_groups=16, max_frames=1000)
        pipe = meta["pipeline"]
        assert set(pipe["models"]) == {"talker", "talker_step_embedder", "code_predictor"}
        assert pipe["models"]["talker"]["type"] == "decoder"
        assert pipe["models"]["talker"]["tokenizer"] == "tokenizer.json"
        assert pipe["models"]["talker_step_embedder"]["type"] == "embedding"

        stage = pipe["strategy"]["stages"][0]["strategy"]
        assert stage["kind"] == "nested_autoregressive"
        assert stage["outer"] == "talker"
        assert stage["inner"] == "code_predictor"
        assert stage["pre_embedder"] == "talker_step_embedder"
        assert stage["num_code_groups"] == 16
        assert stage["max_tokens"] == 1000

        # Required pre-embedder feed edge + inner seed edge.
        assert {
            "from": "talker_step_embedder.inputs_embeds",
            "to": "talker.inputs_embeds",
            "dtype": "fp32",
            "device_transfer": False,
        } in pipe["dataflow"]
        assert {
            "from": "talker.last_hidden_state",
            "to": "code_predictor.inputs_embeds",
            "dtype": "fp32",
            "device_transfer": False,
        } in pipe["dataflow"]
        # No in-package vocoder.
        assert "vocoder" not in pipe["models"]
        # Pre-embedder is a loop-internal on_demand component.
        assert pipe["phases"]["talker_step_embedder"]["run_on"] == "on_demand"

    def test_rejects_invalid_code_groups(self):
        with pytest.raises(ValueError, match="num_code_groups"):
            build_tts_pipeline_metadata(num_code_groups=0)

    def test_write_roundtrip(self, tmp_path):
        path = write_tts_pipeline_metadata(str(tmp_path), num_code_groups=8)
        with open(path) as handle:
            loaded = yaml.safe_load(handle)
        stage = loaded["pipeline"]["strategy"]["stages"][0]["strategy"]
        assert stage["pre_embedder"] == "talker_step_embedder"
        assert stage["num_code_groups"] == 8

    def test_matches_onnx_genai_json_schema(self):
        """Emitted TTS metadata validates against onnx-genai's published schema."""
        schema_path = _onnx_genai_schema_path()
        if schema_path is None:
            pytest.skip("onnx-genai schema not found (set ONNX_GENAI_SCHEMA)")
        import json

        import jsonschema

        with open(schema_path) as handle:
            schema = json.load(handle)
        meta = build_tts_pipeline_metadata(num_code_groups=16, max_frames=2000)
        jsonschema.validate(instance=meta, schema=schema)
