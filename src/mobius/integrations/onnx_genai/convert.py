# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""One-shot ComfyUI workflow -> runnable onnx-genai pipeline directory.

Ties the three pieces together:

1. :func:`parse_comfyui_workflow` — recover run params + topology from the JSON.
2. :func:`export_checkpoint` — export the referenced ``.safetensors`` to the ONNX
   components the pipeline runs.
3. Reconcile the ComfyUI sampler (kind / steps / cfg) with the checkpoint's own
   noise schedule (betas / num_train_timesteps), emit ``inference_metadata.yaml``,
   and write a ``run.json`` capturing the prompt / seed / resolution.

The result is a directory onnx-genai can load and run.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from typing import Any

import yaml

from mobius.integrations.onnx_genai.checkpoint_export import (
    ExportedCheckpoint,
    export_checkpoint,
)
from mobius.integrations.onnx_genai.comfyui import (
    ComfyUIWorkflow,
    parse_comfyui_workflow,
)
from mobius.integrations.onnx_genai.inference_metadata import (
    SchedulerConfig,
    build_diffusion_pipeline_metadata,
)

_LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ConversionResult:
    output_dir: str
    metadata_path: str
    run_params_path: str
    workflow: ComfyUIWorkflow


def build_pipeline_metadata_for_workflow(
    wf: ComfyUIWorkflow,
    exported: ExportedCheckpoint,
    *,
    timesteps: list[float] | None = None,
) -> dict[str, Any]:
    """Reconcile a parsed workflow with an exported checkpoint into pipeline metadata.

    The sampler *kind* / *steps* / *cfg* come from the ComfyUI graph; the noise
    *schedule* (betas, ``num_train_timesteps``) come from the checkpoint — the
    ComfyUI JSON never carries betas, so reading them from the model is the only
    correct source.
    """
    has_vae = "vae" in wf.metadata["pipeline"]["models"]
    has_text = "text_encoder" in wf.metadata["pipeline"]["models"]
    scheduler = SchedulerConfig(
        kind=wf.scheduler_kind,
        num_train_timesteps=exported.num_train_timesteps,
        beta_start=exported.beta_start,
        beta_end=exported.beta_end,
        beta_schedule=exported.beta_schedule,
        prediction_type="epsilon",
        use_karras_sigmas=(wf.scheduler_spacing == "karras"),
    )
    guidance = wf.cfg if wf.cfg != 1.0 else None
    # SDXL routes two conditioning edges (concatenated hidden states + pooled
    # text_embeds); its time_ids is an external denoiser input the driver supplies.
    text_encoder_edges = None
    if exported.sdxl:
        text_encoder_edges = [
            ("encoder_hidden_states", "encoder_hidden_states"),
            ("text_embeds", "text_embeds"),
        ]
    return build_diffusion_pipeline_metadata(
        num_inference_steps=wf.steps,
        scheduler=scheduler,
        guidance_scale=guidance,
        start_step=wf.start_step or None,
        timesteps=timesteps,
        denoiser_filename=exported.denoiser_filename,
        vae_filename=exported.vae_filename if has_vae else None,
        vae_latent_input="latent",
        text_encoder_filename=exported.text_encoder_filename if has_text else None,
        text_encoder_edges=text_encoder_edges,
    )


def _diffusers_timesteps(
    kind: str, exported: ExportedCheckpoint, steps: int, use_karras: bool = False
) -> list[float] | None:
    """Compute the exact inference timesteps diffusers would use, so the denoiser
    is fed the right timestep values. Best-effort; ``None`` on any failure."""
    try:
        if kind == "euler":
            from diffusers import EulerDiscreteScheduler as _Sched

            extra = {
                "timestep_spacing": "linspace",
                "interpolation_type": "linear",
                "use_karras_sigmas": use_karras,
            }
        elif kind == "euler_ancestral":
            from diffusers import EulerAncestralDiscreteScheduler as _Sched

            extra = {"timestep_spacing": "linspace"}
        elif kind == "dpmpp_2m":
            from diffusers import DPMSolverMultistepScheduler as _Sched

            extra = {
                "algorithm_type": "dpmsolver++",
                "solver_order": 2,
                "solver_type": "midpoint",
                "use_karras_sigmas": use_karras,
                "timestep_spacing": "linspace",
                "final_sigmas_type": "zero",
            }
        else:
            from diffusers import DDIMScheduler as _Sched

            extra = {"set_alpha_to_one": True, "steps_offset": 0, "clip_sample": False}
        sched = _Sched(
            num_train_timesteps=exported.num_train_timesteps,
            beta_start=exported.beta_start,
            beta_end=exported.beta_end,
            beta_schedule=exported.beta_schedule,
            prediction_type="epsilon",
            **extra,
        )
        sched.set_timesteps(steps)
        return [float(t) for t in sched.timesteps]
    except Exception as err:  # noqa: BLE001 - timesteps are an optimization, not required
        _LOGGER.warning("could not compute diffusers timesteps (%s); omitting", err)
        return None


def convert_comfyui_workflow(
    workflow: dict[str, Any],
    checkpoint_source: str,
    output_dir: str,
    *,
    opset: int = 17,
) -> ConversionResult:
    """Convert a ComfyUI workflow + checkpoint into a runnable onnx-genai pipeline dir.

    Args:
        workflow: Parsed ComfyUI API-format JSON.
        checkpoint_source: The ``.safetensors``/``.ckpt`` file, diffusers dir, or HF
            id to export (ComfyUI references checkpoints by name; the caller resolves
            that name to a real source).
        output_dir: Destination directory for the ONNX components + metadata.

    Returns:
        A :class:`ConversionResult` with the written paths and parsed workflow.
    """
    wf = parse_comfyui_workflow(workflow)
    os.makedirs(output_dir, exist_ok=True)
    use_karras = wf.scheduler_spacing == "karras"
    # Fractional inference timesteps (Euler/Euler-ancestral always; any Karras
    # schedule) need a float32 denoiser timestep to avoid truncation before the
    # time embedding; DPM++/DDIM linspace timesteps are integer and fine as int64.
    fractional = wf.scheduler_kind in ("euler", "euler_ancestral") or use_karras
    timestep_dtype = "float32" if fractional else "int64"
    exported = export_checkpoint(
        checkpoint_source,
        output_dir,
        height=wf.height,
        width=wf.width,
        opset=opset,
        timestep_dtype=timestep_dtype,
    )
    timesteps = _diffusers_timesteps(wf.scheduler_kind, exported, wf.steps, use_karras)
    metadata = build_pipeline_metadata_for_workflow(wf, exported, timesteps=timesteps)

    metadata_path = os.path.join(output_dir, "inference_metadata.yaml")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)

    run_params = {
        "prompt": wf.prompt,
        "negative_prompt": wf.negative_prompt,
        "seed": wf.seed,
        "width": wf.width,
        "height": wf.height,
        "steps": wf.steps,
        "cfg": wf.cfg,
        "sampler_name": wf.sampler_name,
        "scheduler_kind": wf.scheduler_kind,
        "checkpoint": wf.checkpoint,
        "latent_channels": exported.in_channels,
        "cross_attention_dim": exported.cross_attention_dim,
        "model_max_length": exported.model_max_length,
    }
    run_params_path = os.path.join(output_dir, "run.json")
    with open(run_params_path, "w", encoding="utf-8") as handle:
        json.dump(run_params, handle, indent=2)

    _LOGGER.info("wrote runnable onnx-genai pipeline to %s", output_dir)
    return ConversionResult(
        output_dir=output_dir,
        metadata_path=metadata_path,
        run_params_path=run_params_path,
        workflow=wf,
    )
