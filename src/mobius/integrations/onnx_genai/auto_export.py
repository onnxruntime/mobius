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
    return any(k in names for k in _DENOISER_KEYS) or "vae" in names


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
