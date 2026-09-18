# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the ComfyUI -> onnx-genai conversion reconciliation (no torch)."""

from __future__ import annotations

import json

import pytest

from mobius.integrations.onnx_genai.comfyui import parse_comfyui_workflow
from mobius.integrations.onnx_genai.convert import build_pipeline_metadata_for_workflow
from mobius.integrations.onnx_genai.inference_metadata import SchedulerConfig

_WF = {
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 7,
            "steps": 25,
            "cfg": 6.5,
            "sampler_name": "ddim",
            "scheduler": "normal",
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0],
        },
    },
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd15.safetensors"}},
    "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 640, "height": 512}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a dog"}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry"}},
    "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
}

_SCHEDULER = SchedulerConfig(
    kind="ddim",
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
)


def test_reconciles_workflow_sampler_with_checkpoint_schedule():
    wf = parse_comfyui_workflow(_WF)
    fake_ts = [float(x) for x in range(25)]
    meta = build_pipeline_metadata_for_workflow(wf, _SCHEDULER, timesteps=fake_ts)
    workflow = meta["pipeline"]["workflow"]
    # sampler steps/cfg from the ComfyUI graph ...
    assert workflow["inputs"]["request.max_iterations"]["default"] == 25
    assert workflow["inputs"]["request.guidance_scale"]["default"] == pytest.approx(6.5)
    assert {"denoiser", "vae", "text_encoder"} <= set(workflow["components"])
    # ... and the checkpoint-derived timestep table (never present in the
    # ComfyUI JSON) is materialized as a constant component rather than a
    # scheduler_config block the runtime would have to interpret.
    assert workflow["components"]["diffusion_timesteps"]["implementation"]["kind"] == "onnx"
    # DDIM consumes the latent unscaled; only Euler pre-scales it.
    assert "model_input_scale" not in workflow["components"]


def test_run_params_capture_prompt_and_dims():
    wf = parse_comfyui_workflow(_WF)
    assert wf.prompt == "a dog"
    assert wf.negative_prompt == "blurry"
    assert (wf.width, wf.height) == (640, 512)
    assert wf.checkpoint == "sd15.safetensors"


def test_cfg_one_reconciles_without_guidance():
    wf_json = json.loads(json.dumps(_WF))
    wf_json["3"]["inputs"]["cfg"] = 1.0
    wf = parse_comfyui_workflow(wf_json)
    meta = build_pipeline_metadata_for_workflow(wf, _SCHEDULER)
    workflow = meta["pipeline"]["workflow"]
    assert "guidance_combine" not in workflow["components"]
    assert "request.guidance_scale" not in workflow["inputs"]


def test_sdxl_exported_routes_dual_conditioning():
    wf = parse_comfyui_workflow(_WF)
    meta = build_pipeline_metadata_for_workflow(wf, _SCHEDULER, sdxl=True)
    loop = next(
        step for step in meta["pipeline"]["workflow"]["steps"] if step["kind"] == "loop"
    )
    encoder = next(step for step in loop["setup"] if step.get("component") == "text_encoder")
    assert encoder["outputs"] == {
        "encoder_hidden_states": "conditioning.encoder_hidden_states",
        "text_embeds": "conditioning.text_embeds",
    }
    # CFG runs the denoiser twice; the second call carries the positive prompt.
    denoiser = [step for step in loop["steps"] if step.get("component") == "denoiser"][-1]
    assert denoiser["inputs"]["encoder_hidden_states"] == (
        "conditioning.encoder_hidden_states"
    )
    assert denoiser["inputs"]["text_embeds"] == "conditioning.text_embeds"
