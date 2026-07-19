# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Export a diffusion checkpoint (``.safetensors``) to the ONNX components
onnx-genai's iterative pipeline runs.

ComfyUI references a checkpoint by filename (e.g. ``v1-5-pruned.safetensors``).
onnx-genai runs ONNX graphs, so the checkpoint must first be exported to three
components with the exact ports the pipeline metadata declares (matching the
validated ``diffusion_e2e`` contract):

    text_encoder.onnx : input_ids            -> last_hidden_state
    denoiser.onnx     : sample, timestep,      -> noise_pred
                        encoder_hidden_states
    vae.onnx          : latent               -> image   (1/scaling_factor baked in)

Loading strategy (the "safetensors -> ONNX" bridge):

* A single ``.safetensors`` / ``.ckpt`` file (ComfyUI's original-SD layout) is
  loaded with diffusers ``StableDiffusionPipeline.from_single_file`` — diffusers
  converts the original SD state dict into its UNet / VAE / CLIP modules.
* A diffusers-layout directory or a Hugging Face repo id is loaded with
  ``from_pretrained``.

Either way we then ``torch.onnx.export`` the three modules. This module imports
torch/diffusers lazily so it is cheap to import in environments without them.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any

_LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ExportedCheckpoint:
    """Result of :func:`export_checkpoint`: component filenames + schedule params."""

    denoiser_filename: str
    vae_filename: str
    text_encoder_filename: str
    in_channels: int
    cross_attention_dim: int
    sample_size: int
    model_max_length: int
    scaling_factor: float
    num_train_timesteps: int
    beta_start: float
    beta_end: float
    beta_schedule: str


def _looks_like_single_file(source: str) -> bool:
    return os.path.isfile(source) and source.endswith((".safetensors", ".ckpt"))


def _load_pipeline(source: str) -> Any:
    """Load a diffusers ``StableDiffusionPipeline`` from a checkpoint file, a
    diffusers directory, or a Hugging Face repo id."""
    from diffusers import StableDiffusionPipeline

    if _looks_like_single_file(source):
        _LOGGER.info("loading single-file checkpoint %s", source)
        return StableDiffusionPipeline.from_single_file(source, safety_checker=None)
    _LOGGER.info("loading diffusers checkpoint %s", source)
    return StableDiffusionPipeline.from_pretrained(source, safety_checker=None)


def export_checkpoint(
    source: str,
    output_dir: str,
    *,
    height: int | None = None,
    width: int | None = None,
    opset: int = 17,
    components: tuple[str, ...] = ("text_encoder", "denoiser", "vae"),
) -> ExportedCheckpoint:
    """Export a checkpoint's components to ONNX in ``output_dir``.

    Args:
        source: A ``.safetensors``/``.ckpt`` file, a diffusers directory, or a
            Hugging Face model id.
        output_dir: Destination directory for the ONNX files.
        height/width: Sample resolution used to build export dummy inputs. The
            exported graphs keep dynamic spatial axes; these only size the trace.
        opset: ONNX opset version.
        components: Which components to export (subset of the three).

    Returns:
        An :class:`ExportedCheckpoint` describing the emitted files and the
        checkpoint's noise-schedule parameters (for the scheduler metadata).
    """
    import torch

    os.makedirs(output_dir, exist_ok=True)
    pipe = _load_pipeline(source)
    unet = pipe.unet.eval()
    vae = pipe.vae.eval()
    text_encoder = pipe.text_encoder.eval()
    tokenizer = pipe.tokenizer
    sched = pipe.scheduler

    scaling_factor = float(getattr(vae.config, "scaling_factor", 0.18215))
    in_channels = int(unet.config.in_channels)
    cross_attention_dim = int(unet.config.cross_attention_dim)
    unet_sample = int(getattr(unet.config, "sample_size", 64))
    latent_h = (height // 8) if height else unet_sample
    latent_w = (width // 8) if width else unet_sample
    max_len = int(tokenizer.model_max_length)

    ids = torch.ones(1, max_len, dtype=torch.long)
    with torch.no_grad():
        emb = text_encoder(ids)[0]
    latent0 = torch.zeros(1, in_channels, latent_h, latent_w)

    denoiser_file = "denoiser.onnx"
    vae_file = "vae.onnx"
    text_file = "text_encoder.onnx"

    class _UNetWrap(torch.nn.Module):
        def __init__(self, u):
            super().__init__()
            self.u = u

        def forward(self, sample, timestep, encoder_hidden_states):
            return self.u(sample, timestep, encoder_hidden_states=encoder_hidden_states).sample

    class _TextWrap(torch.nn.Module):
        def __init__(self, t):
            super().__init__()
            self.t = t

        def forward(self, input_ids):
            return self.t(input_ids)[0]

    class _VaeWrap(torch.nn.Module):
        def __init__(self, v, scale):
            super().__init__()
            self.v = v
            self.scale = scale

        def forward(self, latent):
            return self.v.decode(latent / self.scale).sample

    if "text_encoder" in components:
        _LOGGER.info("exporting text_encoder -> %s", text_file)
        torch.onnx.export(
            _TextWrap(text_encoder), (ids,), os.path.join(output_dir, text_file),
            input_names=["input_ids"], output_names=["last_hidden_state"],
            dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}},
            opset_version=opset, dynamo=False,
        )
    if "denoiser" in components:
        _LOGGER.info("exporting denoiser (unet) -> %s", denoiser_file)
        torch.onnx.export(
            _UNetWrap(unet),
            (latent0, torch.tensor([1], dtype=torch.long), emb),
            os.path.join(output_dir, denoiser_file),
            input_names=["sample", "timestep", "encoder_hidden_states"],
            output_names=["noise_pred"],
            dynamic_axes={
                "sample": {0: "batch", 2: "height", 3: "width"},
                "encoder_hidden_states": {0: "batch", 1: "sequence"},
            },
            opset_version=opset, dynamo=False,
        )
    if "vae" in components:
        _LOGGER.info("exporting vae -> %s", vae_file)
        torch.onnx.export(
            _VaeWrap(vae, scaling_factor), (latent0,), os.path.join(output_dir, vae_file),
            input_names=["latent"], output_names=["image"],
            dynamic_axes={"latent": {0: "batch", 2: "height", 3: "width"}},
            opset_version=opset, dynamo=False,
        )

    return ExportedCheckpoint(
        denoiser_filename=denoiser_file,
        vae_filename=vae_file,
        text_encoder_filename=text_file,
        in_channels=in_channels,
        cross_attention_dim=cross_attention_dim,
        sample_size=unet_sample,
        model_max_length=max_len,
        scaling_factor=scaling_factor,
        num_train_timesteps=int(getattr(sched.config, "num_train_timesteps", 1000)),
        beta_start=float(getattr(sched.config, "beta_start", 0.00085)),
        beta_end=float(getattr(sched.config, "beta_end", 0.012)),
        beta_schedule=str(getattr(sched.config, "beta_schedule", "scaled_linear")),
    )
