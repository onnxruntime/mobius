# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Translate a ComfyUI *API-format* workflow JSON into onnx-genai pipeline metadata.

ComfyUI is a node-graph UI for diffusion. Its "Save (API Format)" export is a flat
dict ``{node_id: {"class_type": str, "inputs": {port: value | [src_id, slot]}}}``.
The canonical text-to-image graph is *KSampler-centric* and maps directly onto
onnx-genai's composite iterative pipeline:

    EmptyLatentImage ─► KSampler ─► VAEDecode ─► SaveImage
    CLIPTextEncode(+/-) ─┘  (positive / negative → CFG cond / uncond)
    CheckpointLoaderSimple ─► model / clip / vae

Mapping to :func:`build_diffusion_pipeline_metadata`:

    KSampler.steps            -> num_inference_steps
    KSampler.cfg              -> guidance_scale (CFG; 1.0 disables)
    KSampler.sampler_name     -> scheduler kind (euler, ddim)
    CLIPTextEncode (present)  -> text_encoder component (prompt phase)
    VAEDecode (present)       -> vae component (final phase)

This is a topology/params translator only — it does not carry weights. The actual
ONNX component graphs come from a Mobius build of the same checkpoint; the ComfyUI
JSON supplies *how to run them* (loop shape, scheduler, guidance).

Only the core txt2img subset is supported today; unsupported samplers or nodes
raise a clear ``ValueError`` rather than silently producing wrong dynamics.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mobius.integrations.onnx_genai.inference_metadata import (
    SchedulerConfig,
    build_diffusion_pipeline_metadata,
)

_LOGGER = logging.getLogger(__name__)

_SAMPLER_NODES = ("KSampler", "KSamplerAdvanced")
_VAE_DECODE_NODES = ("VAEDecode", "VAEDecodeTiled")

# ComfyUI sampler_name -> onnx-genai scheduler kind. Only deterministic samplers
# with an onnx-genai implementation are mapped; ancestral / multistep solvers are
# rejected until onnx-genai grows an equivalent scheduler.
_SAMPLER_KIND = {
    "euler": "euler",
    "ddim": "ddim",
}

# Sigma spacings onnx-genai's Euler currently reproduces (linspace). Others
# (karras/exponential) change the schedule and are warned about.
_SUPPORTED_SPACINGS = {"normal", "simple", "ddim_uniform"}


def _nodes(workflow: dict[str, Any]) -> dict[str, Any]:
    """Return the flat node map, tolerating a ``{"prompt": {...}}`` wrapper."""
    if "prompt" in workflow and isinstance(workflow["prompt"], dict):
        return workflow["prompt"]
    return workflow


def _find_single(nodes: dict[str, Any], class_types: tuple[str, ...], what: str) -> tuple[str, dict]:
    hits = [
        (nid, node)
        for nid, node in nodes.items()
        if isinstance(node, dict) and node.get("class_type") in class_types
    ]
    if not hits:
        raise ValueError(
            f"ComfyUI workflow has no {what} node ({' / '.join(class_types)}); "
            "this translator supports the core text-to-image (KSampler) graph"
        )
    if len(hits) > 1:
        raise ValueError(f"ComfyUI workflow has multiple {what} nodes; only one is supported")
    return hits[0]


def _sampler_kind(sampler_name: str) -> str:
    try:
        return _SAMPLER_KIND[sampler_name]
    except KeyError:
        raise ValueError(
            f"ComfyUI sampler {sampler_name!r} has no onnx-genai equivalent yet; "
            f"supported: {', '.join(sorted(_SAMPLER_KIND))}"
        ) from None


def translate_comfyui_workflow(
    workflow: dict[str, Any],
    *,
    denoiser_filename: str = "denoiser.onnx",
    vae_filename: str = "vae.onnx",
    text_encoder_filename: str = "text_encoder.onnx",
    scheduler: SchedulerConfig | None = None,
) -> dict[str, Any]:
    """Translate a ComfyUI API-format workflow into onnx-genai pipeline metadata.

    Args:
        workflow: Parsed ComfyUI API-format JSON (or a ``{"prompt": {...}}`` wrap).
        denoiser_filename / vae_filename / text_encoder_filename: ONNX component
            filenames the emitted metadata should reference (from a Mobius build).
        scheduler: Override the scheduler noise-schedule params (betas, etc.). When
            omitted, the sampler_name sets the ``kind`` and SD defaults fill the rest.

    Returns:
        The onnx-genai ``inference_metadata`` dict (top-level ``pipeline`` key).

    Raises:
        ValueError: No/duplicate sampler, or an unsupported sampler.
    """
    nodes = _nodes(workflow)
    _, sampler = _find_single(nodes, _SAMPLER_NODES, "sampler")
    inputs = sampler.get("inputs", {})

    if "steps" not in inputs:
        raise ValueError("ComfyUI sampler node is missing 'steps'")
    steps = int(inputs["steps"])
    cfg = float(inputs.get("cfg", 1.0))
    sampler_name = str(inputs.get("sampler_name", "euler"))
    spacing = str(inputs.get("scheduler", "normal"))

    kind = _sampler_kind(sampler_name)
    if spacing not in _SUPPORTED_SPACINGS:
        _LOGGER.warning(
            "ComfyUI scheduler spacing %r is not reproduced exactly (onnx-genai uses "
            "linspace); results may differ slightly",
            spacing,
        )

    # A negative CLIPTextEncode + cfg != 1.0 means classifier-free guidance.
    text_nodes = [n for n in nodes.values() if isinstance(n, dict) and n.get("class_type") == "CLIPTextEncode"]
    has_text_encoder = bool(text_nodes)
    has_vae = any(
        isinstance(n, dict) and n.get("class_type") in _VAE_DECODE_NODES for n in nodes.values()
    )
    guidance = cfg if cfg != 1.0 else None

    sched = scheduler or SchedulerConfig(kind=kind)
    if sched.kind != kind:
        _LOGGER.warning(
            "overriding sampler-derived scheduler kind %r with %r from the supplied config",
            kind,
            sched.kind,
        )

    return build_diffusion_pipeline_metadata(
        num_inference_steps=steps,
        scheduler=sched,
        guidance_scale=guidance,
        denoiser_filename=denoiser_filename,
        vae_filename=vae_filename if has_vae else None,
        text_encoder_filename=text_encoder_filename if has_text_encoder else None,
    )


def translate_comfyui_workflow_file(path: str, **kwargs: Any) -> dict[str, Any]:
    """Load a ComfyUI API-format JSON file and translate it (see
    :func:`translate_comfyui_workflow`)."""
    with open(path, encoding="utf-8") as handle:
        workflow = json.load(handle)
    return translate_comfyui_workflow(workflow, **kwargs)
