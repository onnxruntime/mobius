# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the ComfyUI workflow -> onnx-genai pipeline translator."""

from __future__ import annotations

import json

import pytest
import yaml

from mobius.integrations.onnx_genai.comfyui import (
    ComfyUIWorkflow,
    parse_comfyui_workflow,
    translate_comfyui_workflow,
    translate_comfyui_workflow_file,
)
from mobius.integrations.onnx_genai.convert import convert_comfyui_workflow
from mobius.integrations.onnx_genai.inference_metadata import SchedulerConfig
from mobius.integrations.onnx_genai.inference_metadata_test import (
    _onnx_genai_schema_path,
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
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 512, "height": 512, "batch_size": 1},
    },
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat", "clip": ["4", 1]}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["4", 1]}},
    "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
    "9": {
        "class_type": "SaveImage",
        "inputs": {"images": ["8", 0], "filename_prefix": "ComfyUI"},
    },
}


def test_translate_default_txt2img():
    meta = translate_comfyui_workflow(_DEFAULT_TXT2IMG)
    # onnx-genai's PipelineSpec carries a typed SSA workflow and nothing else;
    # the denoise loop is executable steps, not an "iterative" strategy block.
    assert set(meta["pipeline"]) == {"workflow"}
    workflow = meta["pipeline"]["workflow"]
    assert {"denoiser", "vae", "text_encoder", "solver_step"} <= set(workflow["components"])
    assert workflow["inputs"]["request.max_iterations"]["default"] == 20
    # CFG is two denoiser invocations plus a combine component.
    assert "guidance_combine" in workflow["components"]
    assert workflow["inputs"]["request.guidance_scale"]["default"] == pytest.approx(8.0)


def test_parse_recovers_full_run_params():
    wf = parse_comfyui_workflow(_DEFAULT_TXT2IMG)
    assert isinstance(wf, ComfyUIWorkflow)
    assert wf.prompt == "a cat"
    assert wf.negative_prompt == ""
    assert (wf.width, wf.height) == (512, 512)
    assert wf.seed == 42
    assert wf.steps == 20
    assert wf.cfg == pytest.approx(8.0)
    assert wf.sampler_name == "euler"
    assert wf.scheduler_kind == "euler"
    assert wf.checkpoint == "v1-5.safetensors"


def test_conversion_forwards_revision_to_scheduler_loader(tmp_path, monkeypatch):
    calls = []

    def fake_scheduler(source, *, revision=None):
        calls.append((source, revision))
        return SchedulerConfig(kind="euler")

    monkeypatch.setattr(
        "mobius.integrations.onnx_genai.convert.load_diffusers_scheduler_config",
        fake_scheduler,
    )

    convert_comfyui_workflow(
        _DEFAULT_TXT2IMG,
        "nota-ai/bk-sdm-small",
        str(tmp_path),
        revision="pinned-revision",
        compute_timesteps=False,
    )

    assert calls == [("nota-ai/bk-sdm-small", "pinned-revision")]


def test_parse_traces_checkpoint_through_lora():
    wf = json.loads(json.dumps(_DEFAULT_TXT2IMG))
    # Insert a LoraLoader between the checkpoint and the sampler's model input.
    wf["11"] = {
        "class_type": "LoraLoader",
        "inputs": {
            "lora_name": "x.safetensors",
            "strength_model": 0.8,
            "model": ["4", 0],
            "clip": ["4", 1],
        },
    }
    wf["3"]["inputs"]["model"] = ["11", 0]
    parsed = parse_comfyui_workflow(wf)
    assert parsed.checkpoint == "v1-5.safetensors"


def test_parse_collects_stacked_loras_in_order():
    wf = json.loads(json.dumps(_DEFAULT_TXT2IMG))
    # Two stacked LoRAs: checkpoint -> loraA -> loraB -> sampler.
    wf["11"] = {
        "class_type": "LoraLoader",
        "inputs": {
            "lora_name": "a.safetensors",
            "strength_model": 0.5,
            "model": ["4", 0],
            "clip": ["4", 1],
        },
    }
    wf["12"] = {
        "class_type": "LoraLoader",
        "inputs": {
            "lora_name": "b.safetensors",
            "strength_model": 1.0,
            "model": ["11", 0],
            "clip": ["11", 1],
        },
    }
    wf["3"]["inputs"]["model"] = ["12", 0]
    parsed = parse_comfyui_workflow(wf)
    # Applied base-first: a then b.
    assert parsed.loras == (("a.safetensors", 0.5), ("b.safetensors", 1.0))


def test_parse_no_loras_by_default():
    assert parse_comfyui_workflow(_DEFAULT_TXT2IMG).loras == ()


def test_parse_collects_controlnet():
    wf = json.loads(json.dumps(_DEFAULT_TXT2IMG))
    wf["10"] = {
        "class_type": "ControlNetLoader",
        "inputs": {"control_net_name": "canny.safetensors"},
    }
    wf["11"] = {
        "class_type": "ControlNetApply",
        "inputs": {
            "conditioning": ["6", 0],
            "control_net": ["10", 0],
            "image": ["99", 0],
            "strength": 0.7,
        },
    }
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


def test_ddim_sampler_selects_an_unscaled_solver():
    wf = json.loads(json.dumps(_DEFAULT_TXT2IMG))
    wf["3"]["inputs"]["sampler_name"] = "ddim"
    parsed = parse_comfyui_workflow(wf)
    assert parsed.scheduler_kind == "ddim"
    # DDIM consumes the latent directly; only Euler pre-scales it by sigma.
    components = parsed.metadata["pipeline"]["workflow"]["components"]
    assert "solver_step" in components
    assert "model_input_scale" not in components


def test_euler_sampler_scales_the_model_input():
    components = translate_comfyui_workflow(_DEFAULT_TXT2IMG)["pipeline"]["workflow"][
        "components"
    ]
    assert "model_input_scale" in components


def test_dpmpp_2m_sampler_carries_solver_history():
    wf = json.loads(json.dumps(_DEFAULT_TXT2IMG))
    wf["3"]["inputs"]["sampler_name"] = "dpmpp_2m"
    parsed = parse_comfyui_workflow(wf)
    assert parsed.scheduler_kind == "dpmpp_2m"
    # A multistep solver's previous data estimate is a declared state cell, not
    # a hidden scheduler attribute.
    assert "history" in parsed.metadata["pipeline"]["workflow"]["state"]


def test_ancestral_sampler_is_rejected():
    wf = json.loads(json.dumps(_DEFAULT_TXT2IMG))
    wf["3"]["inputs"]["sampler_name"] = "euler_ancestral"
    # The runtime executes a declared solver component; mobius ships no
    # stochastic solver, so lowering it to deterministic Euler would silently
    # run the wrong dynamics.
    with pytest.raises(ValueError, match="euler_ancestral"):
        translate_comfyui_workflow(wf)


@pytest.mark.parametrize("spacing", ["karras", "exponential"])
def test_unmaterializable_sigma_spacing_is_rejected(spacing):
    wf = json.loads(json.dumps(_DEFAULT_TXT2IMG))
    wf["3"]["inputs"]["scheduler"] = spacing
    # The workflow ships the sigma table as a constant component, so a spacing
    # mobius cannot materialize has to fail rather than be silently dropped.
    with pytest.raises(ValueError, match="Karras or exponential sigmas"):
        parse_comfyui_workflow(wf)


def test_denoise_less_than_one_shortens_the_schedule():
    wf = json.loads(json.dumps(_DEFAULT_TXT2IMG))
    wf["3"]["inputs"]["denoise"] = 0.5  # steps=20 -> start_step = 20 - round(10) = 10
    parsed = parse_comfyui_workflow(wf)
    assert parsed.denoise == pytest.approx(0.5)
    assert parsed.start_step == 10
    # img2img skips the noisiest steps, which lowers to a sliced schedule.
    inputs = parsed.metadata["pipeline"]["workflow"]["inputs"]
    assert inputs["request.max_iterations"]["default"] == 10


def test_denoise_one_runs_every_step():
    workflow = translate_comfyui_workflow(_DEFAULT_TXT2IMG)["pipeline"]["workflow"]
    assert workflow["inputs"]["request.max_iterations"]["default"] == 20


def test_cfg_one_disables_guidance():
    wf = json.loads(json.dumps(_DEFAULT_TXT2IMG))
    wf["3"]["inputs"]["cfg"] = 1.0
    workflow = translate_comfyui_workflow(wf)["pipeline"]["workflow"]
    assert "guidance_combine" not in workflow["components"]
    assert "request.guidance_scale" not in workflow["inputs"]


def test_prompt_wrapper_is_accepted():
    meta = translate_comfyui_workflow({"prompt": _DEFAULT_TXT2IMG})
    workflow = meta["pipeline"]["workflow"]
    assert workflow["inputs"]["request.max_iterations"]["default"] == 20


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


def test_latent_only_graph_is_rejected():
    # A graph with no VAEDecode produces latents, and the published workflow
    # terminates in a decoded image output, so there is nothing to declare.
    wf = {
        "3": {
            "class_type": "KSampler",
            "inputs": {"steps": 5, "cfg": 1.0, "sampler_name": "euler", "scheduler": "normal"},
        },
    }
    with pytest.raises(ValueError, match="no VAE decode node"):
        translate_comfyui_workflow(wf)


def test_translate_from_file(tmp_path):
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps(_DEFAULT_TXT2IMG))
    meta = translate_comfyui_workflow_file(str(path))
    workflow = meta["pipeline"]["workflow"]
    assert workflow["inputs"]["request.max_iterations"]["default"] == 20


def _referenced_artifacts(metadata: dict) -> set[str]:
    """Every ONNX artifact the published workflow declares."""
    return {
        declaration["implementation"]["artifact"]
        for declaration in metadata["pipeline"]["workflow"]["components"].values()
        if declaration["implementation"]["kind"] == "onnx"
    }


def test_parsed_workflow_can_materialize_the_policies_it_references(tmp_path):
    """The generated sampler graphs travel with the parse result.

    The document declares them as ``policies/*.onnx`` artifacts, so a caller
    that only had the metadata dict could not produce a loadable package.
    """
    parsed = parse_comfyui_workflow(_DEFAULT_TXT2IMG)
    written = parsed.save_policy_components(str(tmp_path))
    assert written, "a diffusion workflow always generates sampler components"
    for artifact in _referenced_artifacts(parsed.metadata):
        if artifact.startswith("policies/"):
            assert (tmp_path / artifact).is_file(), artifact


def test_conversion_writes_every_referenced_policy_artifact(tmp_path):
    convert_comfyui_workflow(_DEFAULT_TXT2IMG, None, str(tmp_path), compute_timesteps=False)
    with open(tmp_path / "inference_metadata.yaml", encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle)
    policies = {
        artifact
        for artifact in _referenced_artifacts(metadata)
        if artifact.startswith("policies/")
    }
    assert policies
    for artifact in policies:
        assert (tmp_path / artifact).is_file(), artifact
    # And nothing stale: the reconciled solver's components are the only ones
    # written, so a package never ships a graph its document does not declare.
    written = {f"policies/{path.name}" for path in (tmp_path / "policies").iterdir()}
    assert written == policies


def test_translated_metadata_matches_onnx_genai_schema():
    """A ComfyUI-translated pipeline validates against onnx-genai's real schema."""
    import jsonschema

    with open(_onnx_genai_schema_path()) as handle:
        schema = json.load(handle)
    meta = translate_comfyui_workflow(_DEFAULT_TXT2IMG)
    jsonschema.validate(instance=meta, schema=schema)
