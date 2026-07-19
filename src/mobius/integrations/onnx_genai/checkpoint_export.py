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
    # SDXL: two text encoders (concatenated) + a pooled text_embeds + time_ids.
    sdxl: bool = False
    pooled_dim: int = 0
    # ControlNet: the denoiser is a fused ControlNet+UNet taking an extra constant
    # `controlnet_cond` image input; `conditioning_channels` is its channel count.
    controlnet: bool = False
    conditioning_channels: int = 3
    # Inpainting: a 9-channel UNet; the denoiser takes extra constant `mask` (1ch)
    # and `masked_latent` (4ch) inputs concatenated to the latent.
    inpaint: bool = False


def _looks_like_single_file(source: str) -> bool:
    return os.path.isfile(source) and source.endswith((".safetensors", ".ckpt"))


def _load_pipeline(source: str) -> Any:
    """Load a diffusers pipeline (auto-selecting SD vs SDXL) from a checkpoint
    file, a diffusers directory, or a Hugging Face repo id."""
    from diffusers import DiffusionPipeline

    load = (
        DiffusionPipeline.from_single_file
        if _looks_like_single_file(source)
        else DiffusionPipeline.from_pretrained
    )
    _LOGGER.info("loading checkpoint %s", source)
    try:
        return load(source, safety_checker=None)
    except TypeError:
        # SDXL pipelines don't take a `safety_checker` argument.
        return load(source)


def export_checkpoint(
    source: str,
    output_dir: str,
    *,
    height: int | None = None,
    width: int | None = None,
    opset: int = 17,
    components: tuple[str, ...] = ("text_encoder", "denoiser", "vae"),
    timestep_dtype: str = "int64",
    loras: list[tuple[str, float]] | None = None,
    controlnet: str | None = None,
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
        timestep_dtype: ``"int64"`` (default, integer timesteps as DDIM uses) or
            ``"float32"``. Schedulers with *fractional* inference timesteps (Euler
            linspace) need ``"float32"`` so the value is not truncated before the
            UNet's continuous time embedding.
        loras: Optional ``[(path, strength), ...]`` LoRA weights to **fuse** into
            the base model before export (so the exported ONNX already carries the
            LoRA deltas — no runtime LoRA support needed). Applied in order.

    Returns:
        An :class:`ExportedCheckpoint` describing the emitted files and the
        checkpoint's noise-schedule parameters (for the scheduler metadata).
    """
    import torch

    if timestep_dtype not in ("int64", "float32"):
        raise ValueError(f"timestep_dtype must be 'int64' or 'float32', got {timestep_dtype!r}")
    os.makedirs(output_dir, exist_ok=True)
    pipe = _load_pipeline(source)
    for path, strength in loras or []:
        _LOGGER.info("fusing LoRA %s (strength %s)", path, strength)
        pipe.load_lora_weights(path)
        pipe.fuse_lora(lora_scale=float(strength))
        pipe.unload_lora_weights()
    unet = pipe.unet.eval()
    vae = pipe.vae.eval()
    text_encoder = pipe.text_encoder.eval()
    tokenizer = pipe.tokenizer
    sched = pipe.scheduler
    sdxl = getattr(pipe, "text_encoder_2", None) is not None

    scaling_factor = float(getattr(vae.config, "scaling_factor", 0.18215))
    # An inpainting UNet takes 9 input channels (latent[4] + mask[1] + masked_latent[4])
    # but still predicts a 4-channel latent. The loop-carried latent is `out_channels`;
    # mask + masked_latent are extra constant conditioning.
    unet_in = int(unet.config.in_channels)
    latent_channels = int(getattr(unet.config, "out_channels", unet_in))
    inpaint = unet_in == latent_channels + 5
    in_channels = latent_channels
    cross_attention_dim = int(unet.config.cross_attention_dim)
    unet_sample = int(getattr(unet.config, "sample_size", 64))
    latent_h = (height // 8) if height else unet_sample
    latent_w = (width // 8) if width else unet_sample
    max_len = int(tokenizer.model_max_length)

    latent0 = torch.zeros(1, latent_channels, latent_h, latent_w)
    if timestep_dtype == "float32":
        timestep_arg = torch.tensor([1.0], dtype=torch.float32)
    else:
        timestep_arg = torch.tensor([1], dtype=torch.long)

    denoiser_file = "denoiser.onnx"
    vae_file = "vae.onnx"
    text_file = "text_encoder.onnx"

    class _VaeWrap(torch.nn.Module):
        def __init__(self, v, scale):
            super().__init__()
            self.v = v
            self.scale = scale

        def forward(self, latent):
            return self.v.decode(latent / self.scale).sample

    pooled_dim = 0
    cond_channels = 0
    if controlnet and sdxl:
        pooled_dim, cond_channels = _export_sdxl_controlnet(
            pipe, controlnet, output_dir, latent0, timestep_arg, latent_h, latent_w, max_len,
            opset, components, text_file, denoiser_file,
        )
    elif controlnet:
        cond_channels = _export_sd_controlnet(
            text_encoder, unet, controlnet, output_dir, latent0, timestep_arg, latent_h, latent_w,
            max_len, opset, components, text_file, denoiser_file,
        )
    elif inpaint:
        _export_sd_inpaint(
            text_encoder, unet, output_dir, latent0, timestep_arg, max_len, opset, components,
            text_file, denoiser_file,
        )
    elif sdxl:
        pooled_dim = _export_sdxl(
            pipe, output_dir, latent0, timestep_arg, max_len, opset, components, text_file,
            denoiser_file,
        )
    else:
        _export_sd(
            text_encoder, unet, output_dir, latent0, timestep_arg, max_len, opset, components,
            text_file, denoiser_file,
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
        sdxl=sdxl,
        pooled_dim=pooled_dim,
        controlnet=bool(controlnet),
        conditioning_channels=cond_channels,
        inpaint=inpaint,
    )


def _export_sd(
    text_encoder, unet, output_dir, latent0, timestep_arg, max_len, opset, components,
    text_file, denoiser_file,
) -> None:
    """Export a single-text-encoder SD 1.x pipeline's text encoder + UNet."""
    import torch

    ids = torch.ones(1, max_len, dtype=torch.long)
    with torch.no_grad():
        emb = text_encoder(ids)[0]

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
            _UNetWrap(unet), (latent0, timestep_arg, emb),
            os.path.join(output_dir, denoiser_file),
            input_names=["sample", "timestep", "encoder_hidden_states"],
            output_names=["noise_pred"],
            dynamic_axes={
                "sample": {0: "batch", 2: "height", 3: "width"},
                "encoder_hidden_states": {0: "batch", 1: "sequence"},
            },
            opset_version=opset, dynamo=False,
        )


def _export_sd_inpaint(
    text_encoder, unet, output_dir, latent0, timestep_arg, max_len, opset, components,
    text_file, denoiser_file,
) -> None:
    """Export an SD text encoder + a 9-channel inpainting UNet denoiser.

    The denoiser takes the 4-channel loop-carried `sample` plus two extra constant
    inputs — `mask` (1ch) and `masked_latent` (4ch) — which it concatenates to form
    the UNet's 9-channel input and predicts the 4-channel noise.
    """
    import torch

    ids = torch.ones(1, max_len, dtype=torch.long)
    with torch.no_grad():
        emb = text_encoder(ids)[0]
    b, _, h, w = latent0.shape
    mask0 = torch.zeros(b, 1, h, w)
    masked0 = torch.zeros_like(latent0)

    class _TextWrap(torch.nn.Module):
        def __init__(self, t):
            super().__init__()
            self.t = t

        def forward(self, input_ids):
            return self.t(input_ids)[0]

    class _InpaintUNetWrap(torch.nn.Module):
        def __init__(self, u):
            super().__init__()
            self.u = u

        def forward(self, sample, timestep, encoder_hidden_states, mask, masked_latent):
            latent_in = torch.cat([sample, mask, masked_latent], dim=1)
            return self.u(latent_in, timestep, encoder_hidden_states=encoder_hidden_states).sample

    if "text_encoder" in components:
        _LOGGER.info("exporting text_encoder -> %s", text_file)
        torch.onnx.export(
            _TextWrap(text_encoder), (ids,), os.path.join(output_dir, text_file),
            input_names=["input_ids"], output_names=["last_hidden_state"],
            dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}},
            opset_version=opset, dynamo=False,
        )
    if "denoiser" in components:
        _LOGGER.info("exporting inpainting (9-channel) denoiser -> %s", denoiser_file)
        torch.onnx.export(
            _InpaintUNetWrap(unet), (latent0, timestep_arg, emb, mask0, masked0),
            os.path.join(output_dir, denoiser_file),
            input_names=["sample", "timestep", "encoder_hidden_states", "mask", "masked_latent"],
            output_names=["noise_pred"],
            dynamic_axes={"sample": {0: "batch", 2: "height", 3: "width"},
                          "encoder_hidden_states": {0: "batch", 1: "sequence"},
                          "mask": {0: "batch", 2: "height", 3: "width"},
                          "masked_latent": {0: "batch", 2: "height", 3: "width"}},
            opset_version=opset, dynamo=False,
        )


def _export_sd_controlnet(
    text_encoder, unet, controlnet_source, output_dir, latent0, timestep_arg, latent_h, latent_w,
    max_len, opset, components, text_file, denoiser_file,
) -> int:
    """Export an SD text encoder + a **fused ControlNet+UNet** denoiser.

    The denoiser takes an extra constant ``controlnet_cond`` image input: the
    ControlNet runs each step producing down/mid residuals that are injected into
    the UNet. Returns the ControlNet conditioning channel count.
    """
    import torch
    from diffusers import ControlNetModel, UNet2DConditionModel

    if controlnet_source == "__from_unet__":
        controlnet = ControlNetModel.from_unet(unet).eval()
    elif os.path.isfile(controlnet_source):
        controlnet = ControlNetModel.from_single_file(controlnet_source).eval()
    else:
        controlnet = ControlNetModel.from_pretrained(controlnet_source).eval()
    cond_channels = int(getattr(controlnet.config, "conditioning_channels", 3))

    ids = torch.ones(1, max_len, dtype=torch.long)
    with torch.no_grad():
        emb = text_encoder(ids)[0]
    cond_img = torch.zeros(1, cond_channels, latent_h * 8, latent_w * 8)

    class _TextWrap(torch.nn.Module):
        def __init__(self, t):
            super().__init__()
            self.t = t

        def forward(self, input_ids):
            return self.t(input_ids)[0]

    class _CtrlUNetWrap(torch.nn.Module):
        def __init__(self, u, c):
            super().__init__()
            self.u, self.c = u, c

        def forward(self, sample, timestep, encoder_hidden_states, controlnet_cond):
            down, mid = self.c(sample, timestep, encoder_hidden_states=encoder_hidden_states,
                               controlnet_cond=controlnet_cond, return_dict=False)
            return self.u(sample, timestep, encoder_hidden_states=encoder_hidden_states,
                          down_block_additional_residuals=down,
                          mid_block_additional_residual=mid).sample

    if "text_encoder" in components:
        _LOGGER.info("exporting text_encoder -> %s", text_file)
        torch.onnx.export(
            _TextWrap(text_encoder), (ids,), os.path.join(output_dir, text_file),
            input_names=["input_ids"], output_names=["last_hidden_state"],
            dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}},
            opset_version=opset, dynamo=False,
        )
    if "denoiser" in components:
        _LOGGER.info("exporting fused ControlNet+UNet denoiser -> %s", denoiser_file)
        torch.onnx.export(
            _CtrlUNetWrap(unet, controlnet),
            (latent0, timestep_arg, emb, cond_img),
            os.path.join(output_dir, denoiser_file),
            input_names=["sample", "timestep", "encoder_hidden_states", "controlnet_cond"],
            output_names=["noise_pred"],
            dynamic_axes={"sample": {0: "batch", 2: "height", 3: "width"},
                          "encoder_hidden_states": {0: "batch", 1: "sequence"},
                          "controlnet_cond": {0: "batch", 2: "height", 3: "width"}},
            opset_version=opset, dynamo=False,
        )
    return cond_channels


def _export_sdxl_controlnet(
    pipe, controlnet_source, output_dir, latent0, timestep_arg, latent_h, latent_w, max_len,
    opset, components, text_file, denoiser_file,
) -> tuple[int, int]:
    """Export SDXL + ControlNet: the SDXL dual text encoder and a fused
    ControlNet+SDXL-UNet denoiser (sample/timestep/encoder_hidden_states/text_embeds/
    time_ids/controlnet_cond). Returns (pooled_dim, conditioning_channels)."""
    import torch
    from diffusers import ControlNetModel

    unet = pipe.unet.eval()
    # Reuse the SDXL dual-encoder export (also gives pooled_dim) and the encoder
    # dummy tensors for the denoiser trace.
    pooled_dim = _export_sdxl(
        pipe, output_dir, latent0, timestep_arg, max_len, opset,
        tuple(c for c in components if c == "text_encoder"), text_file, "__unused__.onnx",
    )
    te1, te2, tok2 = pipe.text_encoder.eval(), pipe.text_encoder_2.eval(), pipe.tokenizer_2
    ids1 = torch.ones(1, max_len, dtype=torch.long)
    ids2 = torch.ones(1, int(tok2.model_max_length), dtype=torch.long)
    with torch.no_grad():
        o1 = te1(ids1, output_hidden_states=True)
        o2 = te2(ids2, output_hidden_states=True)
        enc = torch.cat([o1.hidden_states[-2], o2.hidden_states[-2]], dim=-1)
        pooled = o2[0]
    time_ids = torch.zeros(1, 6, dtype=torch.float32)

    if controlnet_source == "__from_unet__":
        controlnet = ControlNetModel.from_unet(unet).eval()
    elif os.path.isfile(controlnet_source):
        controlnet = ControlNetModel.from_single_file(controlnet_source).eval()
    else:
        controlnet = ControlNetModel.from_pretrained(controlnet_source).eval()
    cond_channels = int(getattr(controlnet.config, "conditioning_channels", 3))
    cond_img = torch.zeros(1, cond_channels, latent_h * 8, latent_w * 8)

    class _SdxlCtrlUNetWrap(torch.nn.Module):
        def __init__(self, u, c):
            super().__init__()
            self.u, self.c = u, c

        def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids, controlnet_cond):
            added = {"text_embeds": text_embeds, "time_ids": time_ids}
            down, mid = self.c(sample, timestep, encoder_hidden_states=encoder_hidden_states,
                               added_cond_kwargs=added, controlnet_cond=controlnet_cond,
                               return_dict=False)
            return self.u(sample, timestep, encoder_hidden_states=encoder_hidden_states,
                          added_cond_kwargs=added, down_block_additional_residuals=down,
                          mid_block_additional_residual=mid).sample

    if "denoiser" in components:
        _LOGGER.info("exporting fused SDXL ControlNet+UNet denoiser -> %s", denoiser_file)
        torch.onnx.export(
            _SdxlCtrlUNetWrap(unet, controlnet),
            (latent0, timestep_arg, enc, pooled, time_ids, cond_img),
            os.path.join(output_dir, denoiser_file),
            input_names=["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids",
                         "controlnet_cond"],
            output_names=["noise_pred"],
            dynamic_axes={"sample": {0: "batch", 2: "height", 3: "width"},
                          "encoder_hidden_states": {0: "batch", 1: "sequence"},
                          "controlnet_cond": {0: "batch", 2: "height", 3: "width"}},
            opset_version=opset, dynamo=False,
        )
    return pooled_dim, cond_channels


def _export_sdxl(
    pipe, output_dir, latent0, timestep_arg, max_len, opset, components, text_file, denoiser_file,
) -> int:
    """Export an SDXL pipeline: a combined dual text encoder (concatenated
    penultimate hidden states + pooled ``text_embeds``) and the SDXL UNet
    (``sample``/``timestep``/``encoder_hidden_states``/``text_embeds``/``time_ids``).
    Returns the pooled ``text_embeds`` dim."""
    import torch

    te1, te2 = pipe.text_encoder.eval(), pipe.text_encoder_2.eval()
    tok2 = pipe.tokenizer_2
    unet = pipe.unet.eval()
    max_len_2 = int(tok2.model_max_length)

    ids1 = torch.ones(1, max_len, dtype=torch.long)
    ids2 = torch.ones(1, max_len_2, dtype=torch.long)

    class _SdxlTextWrap(torch.nn.Module):
        def __init__(self, a, b):
            super().__init__()
            self.a, self.b = a, b

        def forward(self, input_ids, input_ids_2):
            o1 = self.a(input_ids, output_hidden_states=True)
            o2 = self.b(input_ids_2, output_hidden_states=True)
            # SDXL uses the penultimate hidden layer of both encoders, concatenated,
            # plus the pooled projection from the second (bigG) encoder.
            enc = torch.cat([o1.hidden_states[-2], o2.hidden_states[-2]], dim=-1)
            return enc, o2[0]

    with torch.no_grad():
        enc, pooled = _SdxlTextWrap(te1, te2)(ids1, ids2)
    time_ids = torch.zeros(1, 6, dtype=torch.float32)

    class _SdxlUNetWrap(torch.nn.Module):
        def __init__(self, u):
            super().__init__()
            self.u = u

        def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
            added = {"text_embeds": text_embeds, "time_ids": time_ids}
            return self.u(sample, timestep, encoder_hidden_states=encoder_hidden_states,
                          added_cond_kwargs=added).sample

    if "text_encoder" in components:
        _LOGGER.info("exporting SDXL dual text_encoder -> %s", text_file)
        torch.onnx.export(
            _SdxlTextWrap(te1, te2), (ids1, ids2), os.path.join(output_dir, text_file),
            input_names=["input_ids", "input_ids_2"],
            output_names=["encoder_hidden_states", "text_embeds"],
            dynamic_axes={"input_ids": {0: "batch", 1: "sequence"},
                          "input_ids_2": {0: "batch", 1: "sequence"}},
            opset_version=opset, dynamo=False,
        )
    if "denoiser" in components:
        _LOGGER.info("exporting SDXL denoiser (unet) -> %s", denoiser_file)
        torch.onnx.export(
            _SdxlUNetWrap(unet), (latent0, timestep_arg, enc, pooled, time_ids),
            os.path.join(output_dir, denoiser_file),
            input_names=["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"],
            output_names=["noise_pred"],
            dynamic_axes={"sample": {0: "batch", 2: "height", 3: "width"},
                          "encoder_hidden_states": {0: "batch", 1: "sequence"}},
            opset_version=opset, dynamo=False,
        )
    return int(pooled.shape[-1])
