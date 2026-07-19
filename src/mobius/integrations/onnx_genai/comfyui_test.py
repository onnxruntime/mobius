# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the ComfyUI workflow -> onnx-genai pipeline translator."""

from __future__ import annotations

import json

import pytest

from mobius.integrations.onnx_genai.comfyui import (
    ComfyUIWorkflow,
    parse_comfyui_workflow,
    translate_comfyui_workflow,
    translate_comfyui_workflow_file,
)

# ComfyUI's canonical "default" text-to-image API-format workflow (trimmed).
_DEFAULT_TXT2IMG = {
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 42,
            "steps": 20,
            "cfg": 8.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0],
        },
    },
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "v1-5.safetensors"}},
    "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat", "clip": ["4", 1]}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["4", 1]}},
    "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
    "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "ComfyUI"}},
}


def test_translate_default_txt2img():
    meta = translate_comfyui_workflow(_DEFAULT_TXT2IMG)
    strat = meta["pipeline"]["strategy"]
    assert strat["kind"] == "iterative"
    assert strat["num_steps"] == 20
    assert strat["scheduler_config"]["kind"] == "euler"
    assert strat["guidance_scale"] == 8.0
    # CFG conditioning + text encoder + VAE all present.
    assert strat["cfg_conditioning_input"] == "encoder_hidden_states"
    models = meta["pipeline"]["models"]
    assert "denoiser" in models and "vae" in models and "text_encoder" in models


def test_parse_recovers_full_run_params():
    wf = parse_comfyui_workflow(_DEFAULT_TXT2IMG)
    assert isinstance(wf, ComfyUIWorkflow)
    assert wf.prompt == "a cat"
    assert wf.negative_prompt == ""
    assert (wf.width, wf.height) == (512, 512)
    assert wf.seed == 42
    assert wf.steps == 20
    assert wf.cfg == 8.0
    assert wf.sampler_name == "euler"
    assert wf.scheduler_kind == "euler"
    assert wf.checkpoint == "v1-5.safetensors"


def test_parse_traces_checkpoint_through_lora():
    wf = json.loads(json.dumps(_DEFAULT_TXT2IMG))
    # Insert a LoraLoader between the checkpoint and the sampler's model input.
    wf["11"] = {
        "class_type": "LoraLoader",
        "inputs": {"lora_name": "x.safetensors", "strength_model": 0.8, "model": ["4", 0], "clip": ["4", 1]},
    }
    wf["3"]["inputs"]["model"] = ["11", 0]
    parsed = parse_comfyui_workflow(wf)
    assert parsed.checkpoint == "v1-5.safetensors"


def test_parse_collects_stacked_loras_in_order():
    wf = json.loads(json.dumps(_DEFAULT_TXT2IMG))
    # Two stacked LoRAs: checkpoint -> loraA -> loraB -> sampler.
    wf["11"] = {
        "class_type": "LoraLoader",
        "inputs": {"lora_name": "a.safetensors", "strength_model": 0.5, "model": ["4", 0], "clip": ["4", 1]},
    }
    wf["12"] = {
        "class_type": "LoraLoader",
        "inputs": {"lora_name": "b.safetensors", "strength_model": 1.0, "model": ["11", 0], "clip": ["11", 1]},
    }
    wf["3"]["inputs"]["model"] = ["12", 0]
    parsed = parse_comfyui_workflow(wf)
    # Applied base-first: a then b.
    assert parsed.loras == (("a.safetensors", 0.5), ("b.safetensors", 1.0))


def test_parse_no_loras_by_default():
    assert parse_comfyui_workflow(_DEFAULT_TXT2IMG).loras == ()


def test_parse_collects_controlnet():
    wf = json.loads(json.dumps(_DEFAULT_TXT2IMG))
    wf["10"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": "canny.safetensors"}}
    wf["11"] = {"class_type": "ControlNetApply", "inputs": {
        "conditioning": ["6", 0], "control_net": ["10", 0], "image": ["99", 0], "strength": 0.7}}
    wf["3"]["inputs"]["positive"] = ["11", 0]
    parsed = parse_comfyui_workflow(wf)
    assert parsed.controlnet == ("canny.safetensors", 0.7)


def test_parse_no_controlnet_by_default():
    assert parse_comfyui_workflow(_DEFAULT_TXT2IMG).controlnet is None


def test_parse_non_square_dims():
    wf = json.loads(json.dumps(_DEFAULT_TXT2IMG))
    wf["5"]["inputs"].update({"width": 768, "height": 512})
    parsed = parse_comfyui_workflow(wf)
    assert (parsed.width, parsed.height) == (768, 512)


def test_parse_batch_size():
    assert parse_comfyui_workflow(_DEFAULT_TXT2IMG).batch_size == 1
    wf = json.loads(json.dumps(_DEFAULT_TXT2IMG))
    wf["5"]["inputs"]["batch_size"] = 4
    assert parse_comfyui_workflow(wf).batch_size == 4



def test_ddim_sampler_maps_to_ddim():
    wf = json.loads(json.dumps(_DEFAULT_TXT2IMG))
    wf["3"]["inputs"]["sampler_name"] = "ddim"
    meta = translate_comfyui_workflow(wf)
    assert meta["pipeline"]["strategy"]["scheduler_config"]["kind"] == "ddim"


def test_dpmpp_2m_sampler_maps_to_dpmpp_2m():
    wf = json.loads(json.dumps(_DEFAULT_TXT2IMG))
    wf["3"]["inputs"]["sampler_name"] = "dpmpp_2m"
    meta = translate_comfyui_workflow(wf)
    assert meta["pipeline"]["strategy"]["scheduler_config"]["kind"] == "dpmpp_2m"


def test_euler_ancestral_sampler_maps():
    wf = json.loads(json.dumps(_DEFAULT_TXT2IMG))
    wf["3"]["inputs"]["sampler_name"] = "euler_ancestral"
    meta = translate_comfyui_workflow(wf)
    assert meta["pipeline"]["strategy"]["scheduler_config"]["kind"] == "euler_ancestral"


def test_karras_scheduler_enables_karras_sigmas():
    wf = json.loads(json.dumps(_DEFAULT_TXT2IMG))
    wf["3"]["inputs"]["sampler_name"] = "dpmpp_2m"
    wf["3"]["inputs"]["scheduler"] = "karras"
    parsed = parse_comfyui_workflow(wf)
    assert parsed.scheduler_spacing == "karras"
    assert parsed.metadata["pipeline"]["strategy"]["scheduler_config"]["use_karras_sigmas"] is True


def test_normal_scheduler_omits_karras_sigmas():
    meta = translate_comfyui_workflow(_DEFAULT_TXT2IMG)
    assert "use_karras_sigmas" not in meta["pipeline"]["strategy"]["scheduler_config"]


def test_exponential_scheduler_enables_exponential_sigmas():
    wf = json.loads(json.dumps(_DEFAULT_TXT2IMG))
    wf["3"]["inputs"]["scheduler"] = "exponential"
    parsed = parse_comfyui_workflow(wf)
    assert parsed.scheduler_spacing == "exponential"
    assert parsed.metadata["pipeline"]["strategy"]["scheduler_config"]["use_exponential_sigmas"] is True


def test_denoise_less_than_one_sets_start_step():
    wf = json.loads(json.dumps(_DEFAULT_TXT2IMG))
    wf["3"]["inputs"]["denoise"] = 0.5  # steps=20 -> start_step = 20 - round(10) = 10
    parsed = parse_comfyui_workflow(wf)
    assert parsed.denoise == 0.5
    assert parsed.start_step == 10
    assert parsed.metadata["pipeline"]["strategy"]["start_step"] == 10


def test_denoise_one_omits_start_step():
    meta = translate_comfyui_workflow(_DEFAULT_TXT2IMG)
    assert "start_step" not in meta["pipeline"]["strategy"]


def test_cfg_one_disables_guidance():
    wf = json.loads(json.dumps(_DEFAULT_TXT2IMG))
    wf["3"]["inputs"]["cfg"] = 1.0
    meta = translate_comfyui_workflow(wf)
    assert "guidance_scale" not in meta["pipeline"]["strategy"]


def test_prompt_wrapper_is_accepted():
    meta = translate_comfyui_workflow({"prompt": _DEFAULT_TXT2IMG})
    assert meta["pipeline"]["strategy"]["num_steps"] == 20


def test_unsupported_sampler_rejected():
    wf = json.loads(json.dumps(_DEFAULT_TXT2IMG))
    wf["3"]["inputs"]["sampler_name"] = "dpmpp_2m_sde"
    with pytest.raises(ValueError, match="no onnx-genai equivalent"):
        translate_comfyui_workflow(wf)


def test_missing_sampler_rejected():
    with pytest.raises(ValueError, match="no sampler"):
        translate_comfyui_workflow({"5": _DEFAULT_TXT2IMG["5"]})


def test_multiple_samplers_rejected():
    wf = json.loads(json.dumps(_DEFAULT_TXT2IMG))
    wf["10"] = json.loads(json.dumps(_DEFAULT_TXT2IMG["3"]))
    with pytest.raises(ValueError, match="multiple"):
        translate_comfyui_workflow(wf)


def test_no_vae_no_text_encoder_denoiser_only():
    # A latent-only graph (no VAEDecode / no CLIPTextEncode) yields a denoiser-only
    # pipeline.
    wf = {
        "3": {
            "class_type": "KSampler",
            "inputs": {"steps": 5, "cfg": 1.0, "sampler_name": "euler", "scheduler": "normal"},
        },
    }
    meta = translate_comfyui_workflow(wf)
    models = meta["pipeline"]["models"]
    assert "denoiser" in models and "vae" not in models and "text_encoder" not in models


def test_translate_from_file(tmp_path):
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps(_DEFAULT_TXT2IMG))
    meta = translate_comfyui_workflow_file(str(path))
    assert meta["pipeline"]["strategy"]["num_steps"] == 20


def _onnx_genai_schema_path() -> str | None:
    import os

    candidates = [
        os.environ.get("ONNX_GENAI_SCHEMA"),
        os.path.join(
            os.path.dirname(__file__),
            "../../../../../onnx-genai/schema/inference_metadata.schema.json",
        ),
        os.path.expanduser("~/Documents/GitHub/onnx-genai/schema/inference_metadata.schema.json"),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def test_translated_metadata_matches_onnx_genai_schema():
    """A ComfyUI-translated pipeline validates against onnx-genai's real schema."""
    schema_path = _onnx_genai_schema_path()
    if schema_path is None:
        pytest.skip("onnx-genai schema not found (set ONNX_GENAI_SCHEMA)")
    import jsonschema

    with open(schema_path) as handle:
        schema = json.load(handle)
    meta = translate_comfyui_workflow(_DEFAULT_TXT2IMG)
    jsonschema.validate(instance=meta, schema=schema)

