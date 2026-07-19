# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the ComfyUI -> onnx-genai conversion reconciliation (no torch)."""

from __future__ import annotations

import dataclasses
import json

from mobius.integrations.onnx_genai.checkpoint_export import ExportedCheckpoint
from mobius.integrations.onnx_genai.comfyui import parse_comfyui_workflow
from mobius.integrations.onnx_genai.convert import build_pipeline_metadata_for_workflow

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

_EXPORTED = ExportedCheckpoint(
    denoiser_filename="denoiser.onnx",
    vae_filename="vae.onnx",
    text_encoder_filename="text_encoder.onnx",
    in_channels=4,
    cross_attention_dim=768,
    sample_size=64,
    model_max_length=77,
    scaling_factor=0.18215,
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
)


def test_reconciles_workflow_sampler_with_checkpoint_schedule():
    wf = parse_comfyui_workflow(_WF)
    fake_ts = [float(x) for x in range(25)]
    meta = build_pipeline_metadata_for_workflow(
        wf, _EXPORTED, timesteps=fake_ts
    )
    strat = meta["pipeline"]["strategy"]
    # sampler kind/steps/cfg from the ComfyUI graph ...
    assert strat["num_steps"] == 25
    assert strat["scheduler_config"]["kind"] == "ddim"
    assert strat["guidance_scale"] == 6.5
    assert strat["timesteps"] == fake_ts
    # ... betas from the checkpoint (never present in the ComfyUI JSON).
    assert strat["scheduler_config"]["beta_start"] == 0.00085
    assert strat["scheduler_config"]["beta_schedule"] == "scaled_linear"
    models = meta["pipeline"]["models"]
    assert {"denoiser", "vae", "text_encoder"} <= set(models)


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
    meta = build_pipeline_metadata_for_workflow(wf, _EXPORTED)
    assert "guidance_scale" not in meta["pipeline"]["strategy"]


def test_sdxl_exported_routes_dual_conditioning():
    wf = parse_comfyui_workflow(_WF)
    sdxl_exported = dataclasses.replace(_EXPORTED, sdxl=True, pooled_dim=1280)
    meta = build_pipeline_metadata_for_workflow(wf, sdxl_exported)
    flow = meta["pipeline"]["dataflow"]
    assert {"from": "text_encoder.encoder_hidden_states",
            "to": "denoiser.encoder_hidden_states"} in flow
    assert {"from": "text_encoder.text_embeds", "to": "denoiser.text_embeds"} in flow
