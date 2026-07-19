# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Translate a ComfyUI workflow into an onnx-genai pipeline metadata directory.

This ties the *translation* pieces together (no model export — Mobius builds the
neural ONNX components from scratch via :func:`build_diffusers_pipeline`):

1. :func:`parse_comfyui_workflow` — recover run params + topology from the JSON.
2. Reconcile the ComfyUI sampler (kind / steps / cfg / spacing) with the
   checkpoint's own noise schedule (betas / ``num_train_timesteps``), read from
   the diffusers ``scheduler/scheduler_config.json`` (the ComfyUI JSON never
   carries betas), and emit ``inference_metadata.yaml`` + a ``run.json``.

The ONNX component graphs (denoiser / VAE / text encoder) are produced
separately by Mobius's from-scratch builder — see
:func:`mobius.build_diffusers_pipeline` and
:func:`mobius.integrations.onnx_genai.write_onnx_genai_config`. This module does
**not** export or fuse any weights.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from typing import Any

import yaml

from mobius.integrations.onnx_genai.comfyui import (
    ComfyUIWorkflow,
    parse_comfyui_workflow,
)
from mobius.integrations.onnx_genai.inference_metadata import (
    SchedulerConfig,
    build_diffusion_pipeline_metadata,
    load_diffusers_scheduler_config,
)

_LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ConversionResult:
    output_dir: str
    metadata_path: str
    run_params_path: str
    workflow: ComfyUIWorkflow


def _scheduler_for_workflow(
    workflow: ComfyUIWorkflow, checkpoint_source: str | None
) -> SchedulerConfig:
    """Build the scheduler config: sampler *kind*/*spacing* from the ComfyUI graph,
    noise *schedule* (betas) from the diffusers checkpoint config (SD defaults if
    unavailable)."""
    base = load_diffusers_scheduler_config(checkpoint_source) if checkpoint_source else None
    return SchedulerConfig(
        kind=workflow.scheduler_kind,
        num_train_timesteps=base.num_train_timesteps if base else 1000,
        beta_start=base.beta_start if base else 0.00085,
        beta_end=base.beta_end if base else 0.012,
        beta_schedule=base.beta_schedule if base else "scaled_linear",
        prediction_type="epsilon",
        use_karras_sigmas=(workflow.scheduler_spacing == "karras"),
        use_exponential_sigmas=(workflow.scheduler_spacing == "exponential"),
    )


def build_pipeline_metadata_for_workflow(
    workflow: ComfyUIWorkflow,
    scheduler: SchedulerConfig,
    *,
    sdxl: bool = False,
    timesteps: list[float] | None = None,
) -> dict[str, Any]:
    """Reconcile a parsed workflow with a scheduler config into pipeline metadata.

    The sampler *kind* / *steps* / *cfg* come from the ComfyUI graph; the noise
    *schedule* comes from ``scheduler`` (read from the checkpoint's diffusers
    config — the ComfyUI JSON never carries betas).
    """
    has_vae = "vae" in workflow.metadata["pipeline"]["models"]
    has_text = "text_encoder" in workflow.metadata["pipeline"]["models"]
    guidance = workflow.cfg if workflow.cfg != 1.0 else None
    # SDXL routes two conditioning edges (concatenated hidden states + pooled
    # text_embeds); its time_ids is an external denoiser input the driver supplies.
    text_encoder_edges = None
    if sdxl:
        text_encoder_edges = [
            ("encoder_hidden_states", "encoder_hidden_states"),
            ("text_embeds", "text_embeds"),
        ]
    return build_diffusion_pipeline_metadata(
        num_inference_steps=workflow.steps,
        scheduler=scheduler,
        guidance_scale=guidance,
        start_step=workflow.start_step or None,
        timesteps=timesteps,
        denoiser_filename="denoiser.onnx",
        vae_filename="vae.onnx" if has_vae else None,
        vae_latent_input="latent",
        text_encoder_filename="text_encoder.onnx" if has_text else None,
        text_encoder_edges=text_encoder_edges,
    )


def _diffusers_timesteps(
    kind: str,
    scheduler: SchedulerConfig,
    steps: int,
    use_karras: bool = False,
    use_exponential: bool = False,
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
                "use_exponential_sigmas": use_exponential,
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
                "use_exponential_sigmas": use_exponential,
                "timestep_spacing": "linspace",
                "final_sigmas_type": "zero",
            }
        else:
            from diffusers import DDIMScheduler as _Sched

            extra = {"set_alpha_to_one": True, "steps_offset": 0, "clip_sample": False}
        sched = _Sched(
            num_train_timesteps=scheduler.num_train_timesteps,
            beta_start=scheduler.beta_start,
            beta_end=scheduler.beta_end,
            beta_schedule=scheduler.beta_schedule,
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
    checkpoint_source: str | None,
    output_dir: str,
    *,
    sdxl: bool = False,
    compute_timesteps: bool = True,
) -> ConversionResult:
    """Translate a ComfyUI workflow into an onnx-genai pipeline metadata directory.

    Writes ``inference_metadata.yaml`` (topology + reconciled scheduler) and
    ``run.json`` (prompt / seed / resolution). It does **not** build or export the
    ONNX component graphs — Mobius builds those from scratch; see
    :func:`mobius.build_diffusers_pipeline` +
    :func:`mobius.integrations.onnx_genai.write_onnx_genai_config`.

    Args:
        workflow: Parsed ComfyUI API-format JSON.
        checkpoint_source: The diffusers directory or HF id whose
            ``scheduler/scheduler_config.json`` supplies the noise-schedule betas.
            May be ``None`` (Stable Diffusion beta defaults are used).
        output_dir: Destination directory for the metadata files.
        sdxl: Whether the target is an SDXL pipeline (routes the dual-encoder
            conditioning edges).
        compute_timesteps: Whether to precompute the diffusers inference timesteps
            (requires ``diffusers``); when False they are omitted.

    Returns:
        A :class:`ConversionResult` with the written paths and parsed workflow.
    """
    wf = parse_comfyui_workflow(workflow)
    os.makedirs(output_dir, exist_ok=True)
    use_karras = wf.scheduler_spacing == "karras"
    use_exponential = wf.scheduler_spacing == "exponential"
    scheduler = _scheduler_for_workflow(wf, checkpoint_source)

    timesteps = None
    if compute_timesteps:
        timesteps = _diffusers_timesteps(
            wf.scheduler_kind, scheduler, wf.steps, use_karras, use_exponential
        )
    metadata = build_pipeline_metadata_for_workflow(
        wf, scheduler, sdxl=sdxl, timesteps=timesteps
    )

    metadata_path = os.path.join(output_dir, "inference_metadata.yaml")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)

    run_params = {
        "prompt": wf.prompt,
        "negative_prompt": wf.negative_prompt,
        "seed": wf.seed,
        "width": wf.width,
        "height": wf.height,
        "batch_size": wf.batch_size,
        "steps": wf.steps,
        "cfg": wf.cfg,
        "sampler_name": wf.sampler_name,
        "scheduler_kind": wf.scheduler_kind,
        "checkpoint": wf.checkpoint,
        "sdxl": sdxl,
    }
    run_params_path = os.path.join(output_dir, "run.json")
    with open(run_params_path, "w", encoding="utf-8") as handle:
        json.dump(run_params, handle, indent=2)

    _LOGGER.info("wrote onnx-genai pipeline metadata to %s", output_dir)
    return ConversionResult(
        output_dir=output_dir,
        metadata_path=metadata_path,
        run_params_path=run_params_path,
        workflow=wf,
    )
