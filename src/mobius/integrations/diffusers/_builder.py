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
import os
import re
from importlib import resources

import onnx_ir as ir
import torch
import tqdm

from mobius._builder import build_from_module, resolve_dtype
from mobius._model_package import ModelPackage
from mobius._optimizations import fold_initializers_after_weights
from mobius.integrations._weight_loading import _parallel_download, apply_weights

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diffusers pipeline support
# ---------------------------------------------------------------------------

#: Mapping of diffusers ``_class_name`` to (module_class, config_class, task_name).
#: Each entry maps a diffusers component class to the mobius
#: module class, config parser, and task used to build the ONNX graph.
_DIFFUSERS_CLASS_MAP: dict[str, tuple[type, type, str]] = {}

# A component class can require a different graph contract in a specific
# pipeline while retaining the same neural-network implementation.
_PIPELINE_COMPONENT_TASK_OVERRIDES: dict[str, dict[str, str]] = {
    "QwenImageEditPlusPipeline": {
        "AutoencoderKLQwenImage": "qwen-image-edit-vae",
    },
}

_PIPELINE_MODEL_TYPES: dict[str, str] = {
    "MiniMaxMusic3ModularPipeline": "minimax_music3",
    "QwenImageEditPlusPipeline": "qwen_image_edit",
}


def _tokenizer_id_ceiling(tokenizer_data: dict) -> int | None:
    """Return the largest validated ID represented by a tokenizer.json."""
    ids: list[int] = []
    model = tokenizer_data.get("model")
    if isinstance(model, dict):
        vocabulary = model.get("vocab")
        if isinstance(vocabulary, dict):
            ids.extend(vocabulary.values())
        elif isinstance(vocabulary, list):
            for entry in vocabulary:
                if isinstance(entry, list) and len(entry) >= 2:
                    ids.append(entry[1])
    for token in tokenizer_data.get("added_tokens", []):
        if isinstance(token, dict) and "id" in token:
            ids.append(token["id"])
    if not ids or any(
        not isinstance(token_id, int) or isinstance(token_id, bool) for token_id in ids
    ):
        return None
    if min(ids) < 0:
        return None
    return max(ids)


def _resolve_hierarchical_workflow_config(
    *,
    roles: dict[str, str],
    component_configs: dict[str, dict],
    tokenizer_data: dict,
    contract: dict,
) -> dict | None:
    """Resolve tokenizer-dependent workflow facts or fail closed."""
    required_roles = {
        "global_decoder",
        "global_embedding",
        "semantic_embedding",
        "local_decoder",
        "local_projection",
        "local_embedding",
        "local_feedback_embedding",
        "local_heads",
        "condition_encoder",
        "flow_transformer",
        "vocoder",
    }
    if not required_roles <= roles.keys():
        return None
    global_component = roles["global_decoder"]
    global_config = component_configs.get(global_component)
    if not isinstance(global_config, dict):
        return None
    global_vocabulary_size = global_config.get("vocab_size")
    global_context = global_config.get("max_position_embeddings")
    if not isinstance(global_vocabulary_size, int) or not isinstance(global_context, int):
        return None

    added_tokens = {
        token.get("content"): token.get("id")
        for token in tokenizer_data.get("added_tokens", [])
        if isinstance(token, dict)
    }
    required_tokens = set(
        re.findall(
            r"<\|[^|]+\|>",
            "".join(segment.get("literal", "") for segment in contract["prompt_segments"]),
        )
    )
    required_tokens.update(contract["tokens"].values())
    if not required_tokens <= added_tokens.keys():
        return None
    tokenizer_ceiling = _tokenizer_id_ceiling(tokenizer_data)
    if tokenizer_ceiling is None:
        return None
    semantic_start = tokenizer_ceiling + 1
    stop_token_id = added_tokens[contract["tokens"]["stop"]]
    unconditional_token_id = added_tokens[contract["tokens"]["unconditional"]]
    semantic_size = contract["semantic_vocabulary_size"]
    if (
        not isinstance(stop_token_id, int)
        or not isinstance(unconditional_token_id, int)
        or stop_token_id >= semantic_start
        or unconditional_token_id >= semantic_start
        or not isinstance(semantic_size, int)
        or semantic_size < 1
        or semantic_start + semantic_size > global_vocabulary_size
    ):
        return None
    return {
        **contract,
        "components": roles,
        "semantic_vocabulary_start": semantic_start,
        "stop_token_id": stop_token_id,
        "unconditional_token_id": unconditional_token_id,
        "global_context": global_context,
    }


def _init_diffusers_class_map() -> None:
    """Lazily populate the diffusers class map on first use."""
    if _DIFFUSERS_CLASS_MAP:
        return

    from mobius.integrations.diffusers._configs import (
        CLIPTextConfig,
        CogVideoXConfig,
        MiniMaxMusic3ConditionConfig,
        MiniMaxMusic3LanguageConfig,
        MiniMaxMusic3RVQConfig,
        MiniMaxMusic3TransformerConfig,
        MiniMaxMusic3VocoderConfig,
        QwenImageConfig,
        QwenImageTextEncoderConfig,
        QwenImageVAEConfig,
        T5TextEncoderConfig,
        UNet2DConfig,
        VAEConfig,
    )
    from mobius.models.clip import CLIPTextModel
    from mobius.models.cogvideox import (
        CogVideoXTransformer3DModel,
    )
    from mobius.models.cogvideox_vae import (
        AutoencoderKLCogVideoXModel,
        CogVideoXVAEConfig,
    )
    from mobius.models.dit import DiTConfig, DiTTransformer2DModel
    from mobius.models.flux_sd3 import (
        FluxConfig,
        FluxTransformer2DModel,
        SD3Config,
        SD3Transformer2DModel,
    )
    from mobius.models.hunyuan_dit import HunyuanDiT2DModel, HunyuanDiTConfig
    from mobius.models.minimax_music3 import (
        MiniMaxMusic3ConditionEncoder,
        MiniMaxMusic3LanguageModel,
        MiniMaxMusic3RVQDepthDecoder,
        MiniMaxMusic3Transformer1DModel,
        MiniMaxMusic3Vocoder,
    )
    from mobius.models.qwen_image import QwenImageTransformer2DModel
    from mobius.models.qwen_image_vae import AutoencoderKLQwenImageModel
    from mobius.models.qwen_vl import Qwen25VLCausalLMModel
    from mobius.models.t5 import T5EncoderModel
    from mobius.models.unet import UNet2DConditionModel
    from mobius.models.vae import AutoencoderKLModel

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
            "T5EncoderModel": (
                T5EncoderModel,
                T5TextEncoderConfig,
                "t5-text-encoding",
            ),
            "QwenImageTransformer2DModel": (
                QwenImageTransformer2DModel,
                QwenImageConfig,
                "qwen-image-denoising",
            ),
            "Qwen2_5_VLForConditionalGeneration": (
                Qwen25VLCausalLMModel,
                QwenImageTextEncoderConfig,
                "qwen-image-text-encoding",
            ),
            "AutoencoderKL": (AutoencoderKLModel, VAEConfig, "vae"),
            "AutoencoderKLQwenImage": (
                AutoencoderKLQwenImageModel,
                QwenImageVAEConfig,
                "qwen-image-vae",
            ),
            "AutoencoderKLCogVideoX": (
                AutoencoderKLCogVideoXModel,
                CogVideoXVAEConfig,
                "video-vae",
            ),
            "CogVideoXTransformer3DModel": (
                CogVideoXTransformer3DModel,
                CogVideoXConfig,
                "video-denoising",
            ),
            "Qwen3ForCausalLM": (
                MiniMaxMusic3LanguageModel,
                MiniMaxMusic3LanguageConfig,
                "minimax-music3-language",
            ),
            "MiniMaxMusic3RVQDepthDecoder": (
                MiniMaxMusic3RVQDepthDecoder,
                MiniMaxMusic3RVQConfig,
                "minimax-music3-rvq",
            ),
            "MiniMaxMusic3ConditionEncoder": (
                MiniMaxMusic3ConditionEncoder,
                MiniMaxMusic3ConditionConfig,
                "minimax-music3-condition",
            ),
            "MiniMaxMusic3Transformer1DModel": (
                MiniMaxMusic3Transformer1DModel,
                MiniMaxMusic3TransformerConfig,
                "minimax-music3-denoising",
            ),
            "MiniMaxMusic3Vocoder": (
                MiniMaxMusic3Vocoder,
                MiniMaxMusic3VocoderConfig,
                "minimax-music3-vocoder",
            ),
        }
    )


def _load_diffusers_pipeline_index(
    model_id: str, *, revision: str | None = None
) -> dict | None:
    """Try to load a diffusers ``model_index.json`` from HuggingFace.

    Returns the parsed JSON dict, or ``None`` if not found.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    if os.path.isdir(model_id):
        for filename in ("model_index.json", "modular_model_index.json"):
            path = os.path.join(model_id, filename)
            if os.path.isfile(path):
                with open(path) as f:
                    return json.load(f)
        return None

    path = None
    for filename in ("model_index.json", "modular_model_index.json"):
        try:
            path = hf_hub_download(
                repo_id=model_id,
                filename=filename,
                revision=revision,
            )
            break
        except (EntryNotFoundError, OSError, ValueError) as e:
            logger.debug("Failed to download %s for %s: %s", filename, model_id, e)
    if path is None:
        return None

    with open(path) as f:
        return json.load(f)


def _download_diffusers_component_weights(
    model_id: str,
    component_name: str,
    *,
    revision: str | None = None,
    subfolder: str | None = None,
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

    resolved_subfolder = component_name if subfolder is None else subfolder
    prefix = f"{resolved_subfolder}/" if resolved_subfolder else ""
    # Diffusers uses two naming conventions for the weight basename, and either
    # safetensors (preferred) or PyTorch .bin serialization. Some real repos
    # (e.g. OFA-Sys/small-stable-diffusion-v0) ship only .bin.
    weight_basenames = ["diffusion_pytorch_model", "pytorch_model", "model"]

    all_files = None
    local_root = model_id if os.path.isdir(model_id) else None
    for ext in ("safetensors", "bin"):
        # Sharded weights: <basename>.<ext>.index.json maps params -> shard files.
        for basename in weight_basenames:
            local_index = (
                os.path.join(local_root, prefix, f"{basename}.{ext}.index.json")
                if local_root
                else None
            )
            try:
                if local_index is not None:
                    if not os.path.isfile(local_index):
                        continue
                    index_path = local_index
                else:
                    index_path = hf_hub_download(
                        repo_id=model_id,
                        filename=f"{prefix}{basename}.{ext}.index.json",
                        revision=revision,
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
            local_weight = (
                os.path.join(local_root, prefix, f"{basename}.{ext}") if local_root else None
            )
            try:
                if local_weight is not None:
                    if not os.path.isfile(local_weight):
                        continue
                else:
                    hf_hub_download(
                        repo_id=model_id,
                        filename=f"{prefix}{basename}.{ext}",
                        revision=revision,
                    )
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

    paths = (
        [os.path.join(model_id, prefix, filename) for filename in all_files]
        if local_root
        else _parallel_download(
            model_id,
            [f"{prefix}{f}" for f in all_files],
            revision=revision,
            desc=f"{component_name} weights",
        )
    )

    state_dict: dict[str, torch.Tensor] = {}
    for path in tqdm.tqdm(paths, desc=f"Loading {component_name} weights"):
        if path.endswith(".safetensors"):
            state_dict.update(safetensors.torch.load_file(path))
        else:
            state_dict.update(torch.load(path, map_location="cpu", weights_only=True))
    return state_dict


def _load_diffusers_component_config(
    model_id: str,
    component_name: str,
    *,
    revision: str | None = None,
    subfolder: str | None = None,
) -> dict:
    """Load the config.json for a specific diffusers pipeline component."""
    from huggingface_hub import hf_hub_download

    resolved_subfolder = component_name if subfolder is None else subfolder
    filename = f"{resolved_subfolder}/config.json" if resolved_subfolder else "config.json"
    local_path = os.path.join(model_id, filename)
    path = (
        local_path
        if os.path.isdir(model_id) and os.path.isfile(local_path)
        else hf_hub_download(
            repo_id=model_id,
            filename=filename,
            revision=revision,
        )
    )
    with open(path) as f:
        return json.load(f)


def _resolve_diffusers_component_source(
    root_model_id: str,
    root_revision: str | None,
    component_name: str,
    component_info: list,
) -> tuple[str, str | None, str | None]:
    """Resolve a modular component's repository, revision, and subfolder.

    Legacy two-item entries always use ``root_model_id/component_name`` at the
    caller's revision. For modular three-item entries, an external repository
    uses its metadata revision independently of the root pin. A component that
    still references the root repository uses the explicit caller revision when
    supplied, otherwise its metadata revision.
    """
    if len(component_info) == 2 or not isinstance(component_info[2], dict):
        return root_model_id, root_revision, component_name

    metadata = component_info[2]
    component_model_id = metadata.get("pretrained_model_name_or_path") or root_model_id
    metadata_revision = metadata.get("revision")
    if component_model_id == root_model_id:
        component_revision = root_revision if root_revision is not None else metadata_revision
    else:
        component_revision = metadata_revision
    subfolder = metadata.get("subfolder") if "subfolder" in metadata else component_name
    if subfolder is None:
        subfolder = ""
    if os.path.isdir(root_model_id) and os.path.isfile(
        os.path.join(root_model_id, subfolder, "config.json")
    ):
        return root_model_id, None, subfolder
    return component_model_id, component_revision, subfolder


def _load_optional_diffusers_json(
    model_id: str, filename: str, *, revision: str | None = None
) -> dict:
    """Load optional non-neural pipeline metadata without failing the build."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    try:
        local_path = os.path.join(model_id, filename)
        if os.path.isdir(model_id):
            if not os.path.isfile(local_path):
                return {}
            path = local_path
        else:
            path = hf_hub_download(repo_id=model_id, filename=filename, revision=revision)
    except EntryNotFoundError:
        return {}
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
    revision: str | None = None,
    dtype: str | ir.DataType | None = None,
    load_weights: bool = True,
    unet_loras: dict | None = None,
    components: set[str] | None = None,
    execution_provider: str = "default",
) -> ModelPackage:
    """Build ONNX models for all supported components in a diffusers pipeline.

    Parses the pipeline's ``model_index.json`` and builds each neural network
    component (transformer, VAE, etc.) as a separate ONNX model in the
    returned :class:`ModelPackage`.

    Components that are not neural networks (schedulers, tokenizers) or that
    don't have a registered ONNX model class are skipped.

    Args:
        model_id: HuggingFace model repository ID for a diffusers pipeline.
        revision: Optional Hugging Face revision used for all pipeline artifacts.
        dtype: Override the model dtype.
        load_weights: Whether to download and apply weights.
        unet_loras: Optional ``{adapter_name: lora.safetensors}`` map. Each LoRA
            is baked into the UNet denoiser as a runtime-gated adapter (rank
            inferred from the file); at inference a ``lora_gate.{name}`` scalar
            input switches/blends it. Requires ``load_weights=True`` to apply the
            adapter weights.
        components: Optional component-name allowlist. Non-neural pipeline metadata
            is still retained so a single-component export preserves its contract.
        execution_provider: Target execution provider for EP-aware graph optimization.

    Returns:
        A :class:`ModelPackage` containing the built component model(s).

    Raises:
        ValueError: If the model does not have a ``model_index.json``.
    """
    _init_diffusers_class_map()

    pipeline_index = _load_diffusers_pipeline_index(model_id, revision=revision)
    if pipeline_index is None:
        raise ValueError(
            f"'{model_id}' does not appear to be a diffusers pipeline "
            f"(no model_index.json found)."
        )

    if dtype is not None and isinstance(dtype, str):
        dtype = resolve_dtype(dtype)

    package = ModelPackage({})
    component_configs: dict[str, dict] = {}
    pipeline_class = str(pipeline_index.get("_class_name", "DiffusionPipeline"))

    for component_name, component_info in pipeline_index.items():
        if component_name.startswith("_"):
            continue
        if components is not None and component_name not in components:
            continue
        if not isinstance(component_info, list) or len(component_info) not in (2, 3):
            continue

        library, class_name = component_info[:2]
        component_model_id, component_revision, component_subfolder = (
            _resolve_diffusers_component_source(
                model_id,
                revision,
                component_name,
                component_info,
            )
        )
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

        component_source_kwargs = {"revision": component_revision}
        if len(component_info) == 3 and isinstance(component_info[2], dict):
            component_source_kwargs["subfolder"] = component_subfolder
        component_config_dict = _load_diffusers_component_config(
            component_model_id,
            component_name,
            **component_source_kwargs,
        )
        component_configs[component_name] = component_config_dict
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

        task_name = _PIPELINE_COMPONENT_TASK_OVERRIDES.get(pipeline_class, {}).get(
            class_name, task_name
        )
        sub_pkg = build_from_module(
            model_module,
            config,
            task_name,
            execution_provider=execution_provider,
        )

        # Flatten sub-package into the top-level package
        if len(sub_pkg) == 1 and "model" in sub_pkg:
            sub_pkg["model"].graph.name = f"{model_id}/{component_name}"
            package[component_name] = sub_pkg["model"]
        else:
            for sub_name, sub_model in sub_pkg.items():
                package_name = (
                    component_name if sub_name == "model" else f"{component_name}_{sub_name}"
                )
                sub_model.graph.name = f"{model_id}/{package_name}"
                package[package_name] = sub_model

        if load_weights:
            state_dict = _download_diffusers_component_weights(
                component_model_id,
                component_name,
                **component_source_kwargs,
            )
            if hasattr(model_module, "preprocess_weights"):
                state_dict = model_module.preprocess_weights(state_dict)
            if lora_weights:
                state_dict = {**state_dict, **lora_weights}
            for model in sub_pkg.values():
                apply_weights(model, state_dict)
                fold_initializers_after_weights(model)

    if not package:
        raise ValueError(
            f"No supported neural network components found in '{model_id}'. "
            f"Supported diffusers classes: {sorted(_DIFFUSERS_CLASS_MAP)}."
        )

    from mobius.integrations.diffusers._configs import DiffusersPipelineConfig

    def load_optional_component_json(component_name: str, filename: str) -> dict:
        component_info = pipeline_index.get(component_name)
        if not isinstance(component_info, list) or len(component_info) not in (2, 3):
            return {}
        source_model_id, source_revision, source_subfolder = (
            _resolve_diffusers_component_source(
                model_id,
                revision,
                component_name,
                component_info,
            )
        )
        component_filename = f"{source_subfolder}/{filename}" if source_subfolder else filename
        return _load_optional_diffusers_json(
            source_model_id,
            component_filename,
            revision=source_revision,
        )

    role_specs = {
        "Qwen3ForCausalLM": {
            "global_decoder": "",
            "global_embedding": "_embedding",
            "semantic_embedding": "_semantic_embedding",
        },
        "MiniMaxMusic3RVQDepthDecoder": {
            "local_decoder": "",
            "local_projection": "_projection",
            "local_embedding": "_embedding",
            "local_feedback_embedding": "_feedback_embedding",
            "local_heads": "_heads",
        },
        "MiniMaxMusic3ConditionEncoder": {"condition_encoder": ""},
        "MiniMaxMusic3Transformer1DModel": {"flow_transformer": ""},
        "MiniMaxMusic3Vocoder": {"vocoder": ""},
    }
    workflow_roles: dict[str, str] = {}
    for component_name, component_info in pipeline_index.items():
        if not isinstance(component_info, list) or len(component_info) not in (2, 3):
            continue
        for role, suffix in role_specs.get(component_info[1], {}).items():
            workflow_roles[role] = f"{component_name}{suffix}"
    contract_path = resources.files("mobius.integrations.diffusers").joinpath(
        "workflow_contracts/hierarchical_audio_flow_v1.json"
    )
    with contract_path.open(encoding="utf-8") as handle:
        hierarchical_contract = json.load(handle)
    tokenizer_data = (
        load_optional_component_json("tokenizer", "tokenizer.json")
        if "tokenizer" in pipeline_index
        else {}
    )
    package.config = DiffusersPipelineConfig(
        source_model_id=model_id,
        pipeline_class=pipeline_class,
        component_configs=component_configs,
        scheduler_config=(
            load_optional_component_json("scheduler", "scheduler_config.json")
            if "scheduler" in pipeline_index
            else {}
        ),
        processor_config=(
            load_optional_component_json("processor", "preprocessor_config.json")
            if "processor" in pipeline_index
            else {}
        ),
        workflow_config=_resolve_hierarchical_workflow_config(
            roles=workflow_roles,
            component_configs=component_configs,
            tokenizer_data=tokenizer_data,
            contract=hierarchical_contract,
        ),
        model_type=_PIPELINE_MODEL_TYPES.get(pipeline_class, "diffusers"),
    )
    return package
