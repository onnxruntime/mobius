# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Translate a ComfyUI *API-format* workflow JSON into onnx-genai pipeline metadata.

ComfyUI is a node-graph UI for diffusion. Its "Save (API Format)" export is a flat
dict ``{node_id: {"class_type": str, "inputs": {port: value | [src_id, slot]}}}``
where a value of the form ``[src_id, slot]`` is a *link* to another node's output.

The canonical text-to-image graph is *KSampler-centric* and maps directly onto
onnx-genai's composite iterative pipeline:

    EmptyLatentImage ─► KSampler ─► VAEDecode ─► SaveImage
    CLIPTextEncode(+/-) ─┘  (positive / negative → CFG cond / uncond)
    CheckpointLoaderSimple ─► model / clip / vae

This module walks that graph to recover everything needed to *run* the pipeline:

    KSampler.steps            -> num_inference_steps
    KSampler.cfg              -> guidance_scale (CFG; 1.0 disables)
    KSampler.sampler_name     -> scheduler kind (euler, ddim)
    KSampler.seed             -> seed
    KSampler.positive/negative-> prompt / negative_prompt (followed to CLIPTextEncode)
    KSampler.latent_image     -> width / height (followed to EmptyLatentImage)
    KSampler.model            -> checkpoint name (traced to CheckpointLoaderSimple)
    CLIPTextEncode (present)  -> text_encoder component (prompt phase)
    VAEDecode (present)       -> vae component (final phase)

The translator carries topology + run parameters only; it does NOT carry weights.
The actual ONNX component graphs are built from scratch by Mobius's diffusers
builder (see :func:`mobius.build_diffusers_pipeline`).

Only the core txt2img subset is supported today; unsupported samplers or missing
sampler nodes raise a clear ``ValueError`` rather than silently producing wrong
dynamics.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
from typing import Any

from mobius.integrations.onnx_genai.inference_metadata import (
    SchedulerConfig,
    build_diffusion_pipeline_metadata,
)

_LOGGER = logging.getLogger(__name__)

_SAMPLER_NODES = ("KSampler", "KSamplerAdvanced")
_VAE_DECODE_NODES = ("VAEDecode", "VAEDecodeTiled")
_TEXT_ENCODE_NODES = ("CLIPTextEncode",)
_LATENT_NODES = ("EmptyLatentImage", "EmptySD3LatentImage")

# ComfyUI sampler_name -> onnx-genai scheduler kind. Only deterministic samplers
# with an onnx-genai implementation are mapped; ancestral / multistep solvers are
# rejected until onnx-genai grows an equivalent scheduler.
_SAMPLER_KIND = {
    "euler": "euler",
    "euler_ancestral": "euler_ancestral",
    "ddim": "ddim",
    "dpmpp_2m": "dpmpp_2m",
    "dpm_2m": "dpmpp_2m",
}

# ComfyUI sigma spacings onnx-genai reproduces. "normal"/"simple"/"ddim_uniform"
# map to linspace; "karras" enables the Karras schedule; others are warned about.
_SUPPORTED_SPACINGS = {"normal", "simple", "ddim_uniform", "karras", "exponential"}

_MAX_TRACE_DEPTH = 16


@dataclasses.dataclass(frozen=True)
class ComfyUIWorkflow:
    """Everything needed to run a translated ComfyUI txt2img workflow.

    ``metadata`` is the onnx-genai ``inference_metadata`` document (topology +
    scheduler + guidance). The remaining fields are the per-run inputs recovered
    from the graph so a caller can actually drive the pipeline.

    The document declares the sampler as executable components under
    ``policies/*.onnx``. Those graphs are generated during translation rather
    than read off disk, so they are carried here in ``policy_components``;
    :meth:`save_policy_components` writes them next to the metadata. A package
    whose metadata references them without shipping them is not loadable.
    """

    metadata: dict[str, Any]
    prompt: str | None
    negative_prompt: str | None
    width: int
    height: int
    batch_size: int
    seed: int
    steps: int
    cfg: float
    sampler_name: str
    scheduler_kind: str
    scheduler_spacing: str
    checkpoint: str | None
    denoise: float = 1.0
    start_step: int = 0
    loras: tuple[tuple[str, float], ...] = ()
    controlnet: tuple[str, float] | None = None
    policy_components: Any | None = None

    def save_policy_components(self, directory: str) -> dict[str, str]:
        """Write the generated sampler graphs ``metadata`` references.

        Returns the package-relative artifact paths, or an empty mapping when
        the workflow carries no generated components.
        """
        if self.policy_components is None:
            return {}
        os.makedirs(directory, exist_ok=True)
        return self.policy_components.save_policy_components(directory)


def _nodes(workflow: dict[str, Any]) -> dict[str, Any]:
    """Return the flat node map, tolerating a ``{"prompt": {...}}`` wrapper."""
    if "prompt" in workflow and isinstance(workflow["prompt"], dict):
        return workflow["prompt"]
    return workflow


def _is_link(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
    )


def _resolve(nodes: dict[str, Any], link: Any) -> dict[str, Any] | None:
    """Follow a ``[src_id, slot]`` link to the referenced node dict."""
    if not _is_link(link):
        return None
    node = nodes.get(link[0])
    return node if isinstance(node, dict) else None


def _find_single(
    nodes: dict[str, Any], class_types: tuple[str, ...], what: str
) -> tuple[str, dict]:
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


def _follow_prompt_text(nodes: dict[str, Any], link: Any) -> str | None:
    """Resolve a KSampler conditioning link to its CLIPTextEncode prompt text."""
    node = _resolve(nodes, link)
    if node is None:
        return None
    if node.get("class_type") in _TEXT_ENCODE_NODES:
        text = node.get("inputs", {}).get("text")
        return text if isinstance(text, str) else None
    # Some graphs wrap conditioning (e.g. ConditioningCombine/SetArea); follow a
    # single "conditioning" link one hop as a best effort.
    inner = node.get("inputs", {}).get("conditioning")
    if _is_link(inner):
        return _follow_prompt_text(nodes, inner)
    return None


def _follow_dims(nodes: dict[str, Any], link: Any) -> tuple[int, int, int]:
    """Resolve a KSampler latent link to (width, height, batch_size); defaults 512x512x1."""
    node = _resolve(nodes, link)
    if node is not None and node.get("class_type") in _LATENT_NODES:
        inputs = node.get("inputs", {})
        try:
            return (
                int(inputs.get("width", 512)),
                int(inputs.get("height", 512)),
                max(1, int(inputs.get("batch_size", 1))),
            )
        except (TypeError, ValueError):
            pass
    return 512, 512, 1


def _trace_checkpoint(nodes: dict[str, Any], link: Any) -> str | None:
    """Trace a KSampler.model link back to a checkpoint filename.

    Follows intermediate model-transforming nodes (LoraLoader, ModelSamplingDiscrete,
    ...) by their ``model`` input up to a bounded depth.
    """
    for _ in range(_MAX_TRACE_DEPTH):
        node = _resolve(nodes, link)
        if node is None:
            return None
        inputs = node.get("inputs", {})
        for key in ("ckpt_name", "unet_name", "model_name"):
            name = inputs.get(key)
            if isinstance(name, str):
                return name
        link = inputs.get("model")
        if not _is_link(link):
            return None
    return None


def _trace_loras(nodes: dict[str, Any], link: Any) -> list[tuple[str, float]]:
    """Collect LoraLoader nodes along a KSampler.model chain, in application order.

    ComfyUI stacks LoraLoaders (each `model` input chains to the previous); returns
    ``[(lora_name, strength_model), ...]`` from the base checkpoint outward.
    """
    loras: list[tuple[str, float]] = []
    for _ in range(_MAX_TRACE_DEPTH):
        node = _resolve(nodes, link)
        if node is None:
            break
        inputs = node.get("inputs", {})
        if node.get("class_type") in ("LoraLoader", "LoraLoaderModelOnly"):
            name = inputs.get("lora_name")
            if isinstance(name, str):
                strength = float(inputs.get("strength_model", inputs.get("strength", 1.0)))
                loras.append((name, strength))
        link = inputs.get("model")
        if not _is_link(link):
            break
    loras.reverse()  # base checkpoint applies first
    return loras


def _find_controlnet(nodes: dict[str, Any]) -> tuple[str, float] | None:
    """Find a ControlNetApply node's (control_net_name, strength), if any."""
    for node in nodes.values():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") in ("ControlNetApply", "ControlNetApplyAdvanced"):
            inputs = node.get("inputs", {})
            strength = float(inputs.get("strength", 1.0))
            loader = _resolve(nodes, inputs.get("control_net"))
            if loader is not None and loader.get("class_type") in (
                "ControlNetLoader",
                "DiffControlNetLoader",
            ):
                name = loader.get("inputs", {}).get("control_net_name")
                if isinstance(name, str):
                    return (name, strength)
    return None


def parse_comfyui_workflow(
    workflow: dict[str, Any],
    *,
    denoiser_filename: str = "denoiser.onnx",
    vae_filename: str = "vae.onnx",
    text_encoder_filename: str = "text_encoder.onnx",
    scheduler: SchedulerConfig | None = None,
) -> ComfyUIWorkflow:
    """Parse a ComfyUI API-format workflow into a structured :class:`ComfyUIWorkflow`.

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
    seed = int(inputs.get("seed", inputs.get("noise_seed", 0)))
    denoise = float(inputs.get("denoise", 1.0))

    kind = _sampler_kind(sampler_name)
    if spacing not in _SUPPORTED_SPACINGS:
        _LOGGER.warning(
            "ComfyUI scheduler spacing %r is not reproduced exactly (onnx-genai uses "
            "linspace); results may differ slightly",
            spacing,
        )

    # img2img: a KSampler `denoise` < 1.0 skips the earliest (noisiest) steps.
    # Matches diffusers get_timesteps: start_step = num_steps - round(num_steps*denoise).
    start_step = 0
    if 0.0 < denoise < 1.0:
        start_step = max(0, min(steps - 1, steps - round(steps * denoise)))

    prompt = _follow_prompt_text(nodes, inputs.get("positive"))
    negative_prompt = _follow_prompt_text(nodes, inputs.get("negative"))
    width, height, batch_size = _follow_dims(nodes, inputs.get("latent_image"))
    checkpoint = _trace_checkpoint(nodes, inputs.get("model"))
    loras = tuple(_trace_loras(nodes, inputs.get("model")))
    controlnet = _find_controlnet(nodes)

    has_text_encoder = any(
        isinstance(n, dict) and n.get("class_type") in _TEXT_ENCODE_NODES
        for n in nodes.values()
    )
    has_vae = any(
        isinstance(n, dict) and n.get("class_type") in _VAE_DECODE_NODES
        for n in nodes.values()
    )
    guidance = cfg if not math.isclose(cfg, 1.0) else None

    sched = scheduler or SchedulerConfig(
        kind=kind,
        use_karras_sigmas=(spacing == "karras"),
        use_exponential_sigmas=(spacing == "exponential"),
    )
    if sched.kind != kind:
        _LOGGER.warning(
            "overriding sampler-derived scheduler kind %r with %r from the supplied config",
            kind,
            sched.kind,
        )

    if not has_vae:
        raise ValueError(
            "ComfyUI workflow has no VAE decode node, so it produces latents rather "
            "than an image. Why: onnx-genai's pipeline workflow terminates in a decoded "
            f"image output, and there is nothing to decode with; supported decode nodes: "
            f"{', '.join(sorted(_VAE_DECODE_NODES))}. How to fix: add a VAEDecode node to "
            "the workflow before converting it."
        )

    from mobius._model_package import ModelPackage

    # The sampler components the document declares are generated here, so they
    # travel with the parsed workflow instead of being discarded.
    policy_components = ModelPackage({})
    metadata = build_diffusion_pipeline_metadata(
        num_inference_steps=steps,
        scheduler=sched,
        guidance_scale=guidance,
        start_step=start_step or None,
        denoiser_filename=denoiser_filename,
        vae_filename=vae_filename,
        text_encoder_filename=text_encoder_filename if has_text_encoder else None,
        package=policy_components,
    )
    return ComfyUIWorkflow(
        metadata=metadata,
        policy_components=policy_components,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        batch_size=batch_size,
        seed=seed,
        steps=steps,
        cfg=cfg,
        sampler_name=sampler_name,
        scheduler_kind=sched.kind,
        scheduler_spacing=spacing,
        checkpoint=checkpoint,
        denoise=denoise,
        start_step=start_step,
        loras=loras,
        controlnet=controlnet,
    )


def translate_comfyui_workflow(workflow: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Translate a ComfyUI workflow to just the onnx-genai ``inference_metadata`` dict.

    Convenience wrapper over :func:`parse_comfyui_workflow` for callers that only
    need the pipeline document (see that function for the full run parameters).

    The document references generated sampler graphs under ``policies/*.onnx``,
    which this wrapper drops. To write a loadable package use
    :func:`convert_comfyui_workflow`, or keep the
    :class:`ComfyUIWorkflow` and call
    :meth:`~ComfyUIWorkflow.save_policy_components`.
    """
    return parse_comfyui_workflow(workflow, **kwargs).metadata


def parse_comfyui_workflow_file(path: str, **kwargs: Any) -> ComfyUIWorkflow:
    """Load and parse a ComfyUI API-format JSON file (see :func:`parse_comfyui_workflow`)."""
    with open(path, encoding="utf-8") as handle:
        workflow = json.load(handle)
    return parse_comfyui_workflow(workflow, **kwargs)


def translate_comfyui_workflow_file(path: str, **kwargs: Any) -> dict[str, Any]:
    """Load a ComfyUI API-format JSON file and translate it to metadata.

    Returns the document only; see :func:`translate_comfyui_workflow` for how to
    materialize the sampler graphs it references.
    """
    return parse_comfyui_workflow_file(path, **kwargs).metadata
