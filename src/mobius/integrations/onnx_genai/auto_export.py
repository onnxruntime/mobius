# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Write onnx-genai ``inference_metadata.yaml`` for a built Mobius package.

Dispatches on the package/config: a decoder-only LLM emits the
``model.attention`` + ``kv_cache`` document; a diffusion package (denoiser +
optional VAE / text encoder) emits the iterative ``pipeline`` document. This is
the onnx-genai analogue of :func:`mobius.integrations.ort_genai.write_ort_genai_config`.
"""

from __future__ import annotations

import os
from typing import Any

from mobius.integrations.onnx_genai.decoder_metadata import write_decoder_metadata
from mobius.integrations.onnx_genai.inference_metadata import (
    SchedulerConfig,
    load_diffusers_scheduler_config,
    write_diffusion_pipeline_metadata,
)

_DENOISER_KEYS = ("denoiser", "transformer", "unet")


def _looks_like_diffusion(pkg: Any) -> bool:
    try:
        names = set(pkg.keys())
    except AttributeError:
        return False
    return any(k in names for k in _DENOISER_KEYS) or any(
        k in names for k in ("vae", "vae_decoder", "vae_encoder")
    )


def _diffusion_component_kwargs(pkg: Any) -> dict[str, Any]:
    """Derive diffusion-pipeline metadata filenames from a package's component keys.

    Mobius saves a multi-component package into ``<component>/model.onnx``
    subfolders. This inspects the built package's keys and returns the
    ``denoiser_filename`` / ``text_encoder_filename`` / ``vae_filename`` (and the
    VAE latent port) that :func:`build_diffusion_pipeline_metadata` expects, so a
    classic Stable Diffusion package (``text_encoder`` + ``unet`` +
    ``vae_decoder``) is described in full instead of only its denoiser. Values
    already supplied by the caller take precedence and are never overridden.
    """
    try:
        names = set(pkg.keys())
    except AttributeError:
        return {}

    derived: dict[str, Any] = {}
    for key in _DENOISER_KEYS:
        if key in names:
            derived["denoiser_filename"] = f"{key}/model.onnx"
            break
    if "text_encoder" in names:
        derived["text_encoder_filename"] = "text_encoder/model.onnx"
    if "vae_decoder" in names:
        derived["vae_filename"] = "vae_decoder/model.onnx"
        derived["vae_latent_input"] = "latent_sample"
    elif "vae" in names:
        derived["vae_filename"] = "vae/model.onnx"
    return derived


def write_onnx_genai_config(
    pkg: Any,
    output_dir: str,
    *,
    config: Any | None = None,
    kv_native_dtype: str | None = None,
    num_inference_steps: int = 30,
    scheduler: SchedulerConfig | None = None,
    guidance_scale: float | None = None,
    source: str | None = None,
    **kwargs: Any,
) -> dict[str, str]:
    """Write ``inference_metadata.yaml`` into ``output_dir`` and return its path.

    For a decoder LLM, ``config`` (or ``pkg.config``) supplies the attention
    dimensions. For a diffusion package, the denoiser/VAE/text-encoder filenames
    are taken from ``kwargs`` (see :func:`build_diffusion_pipeline_metadata`) or
    defaulted; ``num_inference_steps`` / ``scheduler`` / ``guidance_scale`` set
    the loop. When ``scheduler`` is not given and ``source`` (the diffusers
    checkpoint dir or HF id) is provided, the scheduler is auto-read from the
    checkpoint's ``scheduler/scheduler_config.json`` (falling back to DDIM).
    """
    os.makedirs(output_dir, exist_ok=True)
    if _looks_like_diffusion(pkg):
        if scheduler is None:
            scheduler = load_diffusers_scheduler_config(source)
        # Fill in component filenames from the package layout, letting any
        # caller-supplied values win.
        derived = _diffusion_component_kwargs(pkg)
        for name, value in derived.items():
            kwargs.setdefault(name, value)
        # Classic text-conditioned diffusion (a text encoder is present) uses
        # classifier-free guidance by default; SD's canonical scale is 7.5.
        if guidance_scale is None and "text_encoder_filename" in kwargs:
            guidance_scale = 7.5
        path = write_diffusion_pipeline_metadata(
            output_dir,
            num_inference_steps=num_inference_steps,
            scheduler=scheduler,
            guidance_scale=guidance_scale,
            **kwargs,
        )
        return {"inference_metadata": path}

    cfg = config if config is not None else getattr(pkg, "config", None)
    if cfg is None:
        raise ValueError(
            "onnx-genai decoder metadata requires a model config (pass config=... "
            "or a package carrying `.config`)"
        )
    path = write_decoder_metadata(output_dir, config=cfg, kv_native_dtype=kv_native_dtype)
    return {"inference_metadata": path}
