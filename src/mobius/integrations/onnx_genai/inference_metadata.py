# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Emit onnx-genai ``inference_metadata`` for diffusion pipelines.

Mobius builds the neural components of a diffusion model (denoiser transformer,
VAE, and — externally — a text encoder) as separate ONNX graphs, but does not
itself carry a scheduler loop. onnx-genai's *iterative* pipeline supplies that
loop declaratively: given an ``inference_metadata`` document describing the
components, the loop-carried dataflow, a timestep input, a scheduler, and
(optionally) classifier-free guidance, it drives the denoise loop and returns
the decoded output.

This module produces that document from the component filenames + a scheduler
config. It reads no torch/diffusers state — only plain values — so it is cheap
to unit-test and safe to call anywhere.

The emitted contract matches onnx-genai's pipeline schema:
``schema/inference_metadata.schema.json`` (kind ``iterative`` with
``denoiser`` / ``num_steps`` / ``timestep_input`` / ``scheduler_config`` /
``cfg_conditioning_input`` and denoiser self-edge loop-carried dataflow).
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

import yaml


@dataclasses.dataclass(frozen=True)
class SchedulerConfig:
    """Diffusion noise-schedule parameters for onnx-genai's DDIM scheduler."""

    kind: str = "ddim"
    num_train_timesteps: int = 1000
    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_schedule: str = "scaled_linear"
    prediction_type: str = "epsilon"

    def to_metadata(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "num_train_timesteps": self.num_train_timesteps,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "beta_schedule": self.beta_schedule,
            "prediction_type": self.prediction_type,
        }

    @classmethod
    def from_diffusers(cls, config: dict[str, Any]) -> "SchedulerConfig":
        """Build from a diffusers ``scheduler/scheduler_config.json`` dict.

        Unknown/absent fields fall back to the (Stable Diffusion) defaults. Only
        the schedule parameters onnx-genai consumes are read; the diffusers
        scheduler class name is mapped to ``kind`` where recognized (DDIM).
        """
        name = str(config.get("_class_name", "")).lower()
        kind = "ddim" if "ddim" in name or not name else "ddim"
        return cls(
            kind=kind,
            num_train_timesteps=int(config.get("num_train_timesteps", 1000)),
            beta_start=float(config.get("beta_start", 0.00085)),
            beta_end=float(config.get("beta_end", 0.012)),
            beta_schedule=str(config.get("beta_schedule", "scaled_linear")),
            prediction_type=str(config.get("prediction_type", "epsilon")),
        )


def build_diffusion_pipeline_metadata(
    *,
    num_inference_steps: int,
    denoiser_filename: str = "denoiser.onnx",
    denoiser_sample_input: str = "sample",
    denoiser_timestep_input: str = "timestep",
    denoiser_conditioning_input: str = "encoder_hidden_states",
    denoiser_output: str = "noise_pred",
    scheduler: SchedulerConfig | None = None,
    timesteps: list[float] | None = None,
    guidance_scale: float | None = None,
    vae_filename: str | None = None,
    vae_latent_input: str = "latent",
    text_encoder_filename: str | None = None,
    text_encoder_output: str = "last_hidden_state",
) -> dict[str, Any]:
    """Build the onnx-genai ``inference_metadata`` dict for a diffusion pipeline.

    The denoiser runs an iterative loop: its ``denoiser_output`` (a noise
    prediction) is fed back to ``denoiser_sample_input`` each step (a
    loop-carried self-edge), the scheduler combines it with the current latent,
    the per-step timestep is injected into ``denoiser_timestep_input``, and the
    conditioning is supplied on ``denoiser_conditioning_input``.

    Args:
        num_inference_steps: Number of denoise steps (``strategy.num_steps``).
        denoiser_*: Denoiser component filename and I/O port names.
        scheduler: Noise-schedule config (defaults to DDIM defaults).
        guidance_scale: When set and != 1.0, enables classifier-free guidance
            (the conditioning input is zeroed on the unconditional pass).
        vae_filename: Optional VAE decoder; runs ``final_only`` on the final
            latent (``denoiser_sample_input``).
        vae_latent_input: VAE latent input port name.
        text_encoder_filename: Optional text encoder; runs ``prompt_only`` and
            feeds ``denoiser_conditioning_input``.
        text_encoder_output: Text encoder output port name.

    Returns:
        A dict with a top-level ``pipeline`` key, ready to serialize to
        ``inference_metadata.yaml``.
    """
    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be >= 1")
    scheduler = scheduler or SchedulerConfig()

    models: dict[str, Any] = {
        "denoiser": {"filename": denoiser_filename, "type": "denoiser"},
    }
    dataflow: list[dict[str, Any]] = [
        # Loop-carried self-edge: previous step's prediction seeds the next.
        {
            "from": f"denoiser.{denoiser_output}",
            "to": f"denoiser.{denoiser_sample_input}",
        },
    ]
    phases: dict[str, Any] = {}

    if text_encoder_filename is not None:
        models["text_encoder"] = {
            "filename": text_encoder_filename,
            "type": "encoder",
        }
        dataflow.append(
            {
                "from": f"text_encoder.{text_encoder_output}",
                "to": f"denoiser.{denoiser_conditioning_input}",
            }
        )
        phases["text_encoder"] = {"run_on": "prompt_only"}

    if vae_filename is not None:
        models["vae"] = {"filename": vae_filename, "type": "vae"}
        # The VAE decodes the final post-scheduler latent (the sample port).
        dataflow.append(
            {
                "from": f"denoiser.{denoiser_sample_input}",
                "to": f"vae.{vae_latent_input}",
            }
        )
        phases["vae"] = {"run_on": "final_only"}

    strategy: dict[str, Any] = {
        "kind": "iterative",
        "denoiser": "denoiser",
        "num_steps": num_inference_steps,
        "timestep_input": denoiser_timestep_input,
        "scheduler_config": scheduler.to_metadata(),
    }
    if timesteps is not None:
        if len(timesteps) != num_inference_steps:
            raise ValueError(
                f"timesteps has {len(timesteps)} entries but num_inference_steps is "
                f"{num_inference_steps}"
            )
        strategy["timesteps"] = [float(t) for t in timesteps]
    if guidance_scale is not None:
        strategy["guidance_scale"] = guidance_scale
        if guidance_scale != 1.0:
            strategy["cfg_conditioning_input"] = denoiser_conditioning_input

    pipeline: dict[str, Any] = {
        "models": models,
        "dataflow": dataflow,
        "strategy": strategy,
    }
    if phases:
        pipeline["phases"] = phases
    return {"pipeline": pipeline}


def write_diffusion_pipeline_metadata(
    directory: str,
    *,
    filename: str = "inference_metadata.yaml",
    **kwargs: Any,
) -> str:
    """Build and write ``inference_metadata.yaml`` into ``directory``.

    Extra keyword arguments are forwarded to
    :func:`build_diffusion_pipeline_metadata`. Returns the written path.
    """
    metadata = build_diffusion_pipeline_metadata(**kwargs)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)
    return path
