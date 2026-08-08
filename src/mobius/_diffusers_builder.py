# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Diffusers pipeline building support.

This module handles building ONNX models from HuggingFace diffusers
pipelines (Flux, Stable Diffusion 3, VAEs, etc.).
"""

from __future__ import annotations

__all__ = [
    "build_diffusers_pipeline",
]

import json
import logging

import onnx_ir as ir
import torch
import tqdm

from mobius._builder import build_from_module, resolve_dtype
from mobius._model_package import ModelPackage
from mobius._weight_loading import _parallel_download, apply_weights

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diffusers pipeline support
# ---------------------------------------------------------------------------

#: Mapping of diffusers ``_class_name`` to (module_class, config_class, task_name).
#: Each entry maps a diffusers component class to the mobius
#: module class, config parser, and task used to build the ONNX graph.
_DIFFUSERS_CLASS_MAP: dict[str, tuple[type, type, str]] = {}


def _init_diffusers_class_map() -> None:
    """Lazily populate the diffusers class map on first use."""
    if _DIFFUSERS_CLASS_MAP:
        return

    from mobius._configs import Cosmos3OmniGeneratorConfig, WanVAEConfig
    from mobius._diffusers_configs import (
        CLIPTextConfig,
        CogVideoXConfig,
        QwenImageConfig,
        QwenImageVAEConfig,
        UNet2DConfig,
        VAEConfig,
    )
    from mobius.models.clip import CLIPTextModel
    from mobius.models.cogvideox import (
        CogVideoXTransformer3DModel,
    )
    from mobius.models.cosmos3_omni_generator import Cosmos3OmniGeneratorModel
    from mobius.models.dit import DiTConfig, DiTTransformer2DModel
    from mobius.models.flux_sd3 import (
        FluxConfig,
        FluxTransformer2DModel,
        SD3Config,
        SD3Transformer2DModel,
    )
    from mobius.models.hunyuan_dit import HunyuanDiT2DModel, HunyuanDiTConfig
    from mobius.models.qwen_image import QwenImageTransformer2DModel
    from mobius.models.qwen_image_vae import AutoencoderKLQwenImageModel
    from mobius.models.unet import UNet2DConditionModel
    from mobius.models.vae import AutoencoderKLModel
    from mobius.models.video_vae import VideoAutoencoderModel, VideoVAEConfig
    from mobius.models.wan_vae import AutoencoderKLWanModel

    _DIFFUSERS_CLASS_MAP.update(
        {
            "DiTTransformer2DModel": (DiTTransformer2DModel, DiTConfig, "denoising"),
            "HunyuanDiT2DModel": (HunyuanDiT2DModel, HunyuanDiTConfig, "denoising"),
            "PixArtTransformer2DModel": (DiTTransformer2DModel, DiTConfig, "denoising"),
            "FluxTransformer2DModel": (FluxTransformer2DModel, FluxConfig, "denoising"),
            "SD3Transformer2DModel": (SD3Transformer2DModel, SD3Config, "denoising"),
            # Classic Stable Diffusion 1.x/2.x: cross-attention UNet denoiser plus
            # a CLIP text prompt encoder, both built from scratch by Mobius.
            "UNet2DConditionModel": (
                UNet2DConditionModel,
                UNet2DConfig,
                "denoising",
            ),
            "CLIPTextModel": (CLIPTextModel, CLIPTextConfig, "feature-extraction"),
            "QwenImageTransformer2DModel": (
                QwenImageTransformer2DModel,
                QwenImageConfig,
                "denoising",
            ),
            "AutoencoderKL": (AutoencoderKLModel, VAEConfig, "vae"),
            "AutoencoderKLQwenImage": (
                AutoencoderKLQwenImageModel,
                QwenImageVAEConfig,
                "qwen-image-vae",
            ),
            "AutoencoderKLCogVideoX": (
                VideoAutoencoderModel,
                VideoVAEConfig,
                "vae",
            ),
            "CogVideoXTransformer3DModel": (
                CogVideoXTransformer3DModel,
                CogVideoXConfig,
                "video-denoising",
            ),
            "Cosmos3OmniTransformer": (
                Cosmos3OmniGeneratorModel,
                Cosmos3OmniGeneratorConfig,
                "cosmos3-omni-generator",
            ),
            "AutoencoderKLWan": (
                AutoencoderKLWanModel,
                WanVAEConfig,
                "wan-vae",
            ),
        }
    )


def _load_diffusers_pipeline_index(model_id: str) -> dict | None:
    """Try to load a diffusers ``model_index.json`` from HuggingFace.

    Returns the parsed JSON dict, or ``None`` if not found.
    """
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(repo_id=model_id, filename="model_index.json")
    except (OSError, ValueError) as e:
        logger.debug("Failed to download model_index.json for %s: %s", model_id, e)
        return None

    with open(path) as f:
        return json.load(f)


def _download_diffusers_component_weights(
    model_id: str, component_name: str
) -> dict[str, torch.Tensor]:
    """Download weights for a specific component of a diffusers pipeline.

    Diffusers pipelines store weights in subdirectories using either
    ``diffusion_pytorch_model.safetensors`` (standard) or ``model.safetensors``
    as the weight filename. Sharded weights use a corresponding
    ``.index.json`` file.
    """
    import safetensors.torch
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    prefix = f"{component_name}/"
    # Diffusers uses two naming conventions for the weight basename, and either
    # safetensors (preferred) or PyTorch .bin serialization. Some real repos
    # (e.g. OFA-Sys/small-stable-diffusion-v0) ship only .bin.
    weight_basenames = ["diffusion_pytorch_model", "pytorch_model", "model"]

    all_files = None
    for ext in ("safetensors", "bin"):
        # Sharded weights: <basename>.<ext>.index.json maps params -> shard files.
        for basename in weight_basenames:
            try:
                index_path = hf_hub_download(
                    repo_id=model_id,
                    filename=f"{prefix}{basename}.{ext}.index.json",
                )
                with open(index_path) as f:
                    index = json.load(f)
                all_files = sorted(set(index["weight_map"].values()))
                break
            except EntryNotFoundError:
                continue
        if all_files is not None:
            break
        # Single-file weights.
        for basename in weight_basenames:
            try:
                hf_hub_download(repo_id=model_id, filename=f"{prefix}{basename}.{ext}")
                all_files = [f"{basename}.{ext}"]
                break
            except EntryNotFoundError:
                continue
        if all_files is not None:
            break

    if all_files is None:
        raise FileNotFoundError(
            f"Could not find weight files for component '{component_name}' "
            f"in '{model_id}'. Tried {weight_basenames} with .safetensors and .bin."
        )

    paths = _parallel_download(
        model_id,
        [f"{prefix}{f}" for f in all_files],
        desc=f"{component_name} weights",
    )

    state_dict: dict[str, torch.Tensor] = {}
    for path in tqdm.tqdm(paths, desc=f"Loading {component_name} weights"):
        if path.endswith(".safetensors"):
            state_dict.update(safetensors.torch.load_file(path))
        else:
            state_dict.update(torch.load(path, map_location="cpu", weights_only=True))
    return state_dict


def _load_diffusers_component_config(model_id: str, component_name: str) -> dict:
    """Load the config.json for a specific diffusers pipeline component."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=model_id, filename=f"{component_name}/config.json")
    with open(path) as f:
        return json.load(f)


def _prepare_unet_loras(unet_loras: dict) -> tuple[tuple, dict]:
    """Load each UNet LoRA ``.safetensors``; return baked-adapter specs + merged weights.

    ``unet_loras`` maps ``adapter_name -> safetensors path``. The rank is inferred
    from each adapter's ``lora_A`` factor (``[rank, in]``); the baked scale is
    ``1.0`` because the runtime ``lora_gate.{name}`` input supplies the effective
    strength (0 = off, 1 = on, or a blend). Returns
    ``(((name, rank, 1.0), ...), merged_state_dict)``.
    """
    from mobius.models.unet import load_unet_lora_safetensors

    adapters = []
    merged: dict = {}
    for name, path in unet_loras.items():
        remapped = load_unet_lora_safetensors(path, name)
        rank = None
        for key, value in remapped.items():
            if f".lora_A.{name}.weight" in key:
                rank = int(value.shape[0])
                break
        if rank is None:
            raise ValueError(f"no lora_A weights found for adapter {name!r} in {path}")
        adapters.append((name, rank, 1.0))
        merged.update(remapped)
    return tuple(adapters), merged


def build_diffusers_pipeline(
    model_id: str,
    *,
    dtype: str | ir.DataType | None = None,
    load_weights: bool = True,
    unet_loras: dict | None = None,
    execution_provider: str = "default",
    trace_optimization: bool = False,
) -> ModelPackage:
    """Build ONNX models for all supported components in a diffusers pipeline.

    Parses the pipeline's ``model_index.json`` and builds each neural network
    component (transformer, VAE, etc.) as a separate ONNX model in the
    returned :class:`ModelPackage`.

    Components that are not neural networks (schedulers, tokenizers) or that
    don't have a registered ONNX model class are skipped.

    Args:
        model_id: HuggingFace model repository ID for a diffusers pipeline.
        dtype: Override the model dtype.
        load_weights: Whether to download and apply weights.
        unet_loras: Optional ``{adapter_name: lora.safetensors}`` map. Each LoRA
            is baked into the UNet denoiser as a runtime-gated adapter (rank
            inferred from the file); at inference a ``lora_gate.{name}`` scalar
            input switches/blends it. Requires ``load_weights=True`` to apply the
            adapter weights.
        execution_provider: Target execution provider for component-specific
            optimization and lowering.
        trace_optimization: Whether to log each component optimization stage.

    Returns:
        A :class:`ModelPackage` containing the built component model(s).

    Raises:
        ValueError: If the model does not have a ``model_index.json``.
    """
    _init_diffusers_class_map()

    pipeline_index = _load_diffusers_pipeline_index(model_id)
    if pipeline_index is None:
        raise ValueError(
            f"'{model_id}' does not appear to be a diffusers pipeline "
            f"(no model_index.json found)."
        )

    if dtype is not None and isinstance(dtype, str):
        dtype = resolve_dtype(dtype)

    package = ModelPackage({})

    for component_name, component_info in pipeline_index.items():
        if component_name.startswith("_"):
            continue
        if not isinstance(component_info, list) or len(component_info) != 2:
            continue

        library, class_name = component_info
        if class_name not in _DIFFUSERS_CLASS_MAP:
            logger.info(
                "Skipping diffusers component '%s' (class '%s' from '%s' is not registered).",
                component_name,
                class_name,
                library,
            )
            continue

        module_class, config_class, task_name = _DIFFUSERS_CLASS_MAP[class_name]
        logger.info(
            "Building diffusers component '%s' (%s)...",
            component_name,
            class_name,
        )

        component_config_dict = _load_diffusers_component_config(model_id, component_name)
        config = config_class.from_diffusers(component_config_dict)

        if dtype is not None and hasattr(config, "dtype"):
            import dataclasses

            config = dataclasses.replace(config, dtype=dtype)

        # Runtime LoRA: bake the requested adapters into the UNet denoiser and
        # merge their (remapped) weights alongside the base weights.
        lora_weights: dict = {}
        if unet_loras and task_name == "denoising" and hasattr(config, "lora_adapters"):
            import dataclasses

            adapters, lora_weights = _prepare_unet_loras(unet_loras)
            config = dataclasses.replace(config, lora_adapters=adapters)

        model_module = module_class(config)

        sub_pkg = build_from_module(
            model_module,
            config,
            task_name,
            execution_provider=execution_provider,
            trace_optimization=trace_optimization,
        )

        # Flatten sub-package into the top-level package
        if len(sub_pkg) == 1 and "model" in sub_pkg:
            sub_pkg["model"].graph.name = f"{model_id}/{component_name}"
            package[component_name] = sub_pkg["model"]
        else:
            for sub_name, sub_model in sub_pkg.items():
                sub_model.graph.name = f"{model_id}/{component_name}_{sub_name}"
                package[f"{component_name}_{sub_name}"] = sub_model

        if load_weights:
            state_dict = _download_diffusers_component_weights(model_id, component_name)
            if hasattr(model_module, "preprocess_weights"):
                state_dict = model_module.preprocess_weights(state_dict)
            if lora_weights:
                state_dict = {**state_dict, **lora_weights}
            for model in sub_pkg.values():
                apply_weights(model, state_dict)

    if not package:
        raise ValueError(
            f"No supported neural network components found in '{model_id}'. "
            f"Supported diffusers classes: {sorted(_DIFFUSERS_CLASS_MAP)}."
        )

    return package
