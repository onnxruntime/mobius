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
    load_diffusers_scheduler_config,
    write_diffusion_pipeline_metadata,
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
        assert pipe["strategy"]["guidance_scale"] == 7.5
        assert pipe["strategy"]["cfg_conditioning_input"] == "encoder_hidden_states"

    def test_guidance_scale_one_does_not_enable_cfg(self):
        meta = build_diffusion_pipeline_metadata(
            num_inference_steps=2, guidance_scale=1.0
        )
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
        assert sched.beta_end == 0.02

    def test_scheduler_maps_euler_class(self):
        sched = SchedulerConfig.from_diffusers(
            {"_class_name": "EulerDiscreteScheduler", "beta_schedule": "scaled_linear"}
        )
        assert sched.kind == "euler"
        assert sched.beta_schedule == "scaled_linear"

    def test_scheduler_defaults_to_ddim_when_class_absent(self):
        assert SchedulerConfig.from_diffusers({}).kind == "ddim"

    def test_scheduler_rejects_ancestral(self):
        with pytest.raises(ValueError, match="stochastic"):
            SchedulerConfig.from_diffusers({"_class_name": "EulerAncestralDiscreteScheduler"})

    def test_scheduler_rejects_unsupported_class(self):
        with pytest.raises(ValueError, match="unsupported"):
            SchedulerConfig.from_diffusers({"_class_name": "DPMSolverMultistepScheduler"})

    def test_load_scheduler_from_local_checkpoint(self, tmp_path):
        import json

        sd = tmp_path / "scheduler"
        sd.mkdir()
        (sd / "scheduler_config.json").write_text(
            json.dumps(
                {"_class_name": "EulerDiscreteScheduler", "beta_end": 0.015}
            )
        )
        sched = load_diffusers_scheduler_config(str(tmp_path))
        assert sched is not None
        assert sched.kind == "euler"
        assert sched.beta_end == 0.015

    def test_load_scheduler_none_when_absent(self, tmp_path):
        assert load_diffusers_scheduler_config(str(tmp_path)) is None
        assert load_diffusers_scheduler_config(None) is None

    def test_load_scheduler_falls_back_on_unsupported(self, tmp_path):
        import json

        sd = tmp_path / "scheduler"
        sd.mkdir()
        (sd / "scheduler_config.json").write_text(
            json.dumps({"_class_name": "DPMSolverMultistepScheduler"})
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
        loaded = yaml.safe_load(open(path))
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
