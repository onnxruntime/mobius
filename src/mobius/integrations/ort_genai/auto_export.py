# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Auto-export pipeline for onnxruntime-genai.

Three entry points, in order of increasing convenience:

- :func:`write_ort_genai_config` — config-only API.  Generates the ORT-GenAI
  config artifacts (``genai_config.json``, tokenizer files,
  ``processor_config.json`` / ``image_processor.json``) for an already-built
  :class:`~mobius._model_package.ModelPackage`.  Does **not** write any ONNX
  files — call :meth:`ModelPackage.save` separately if the ONNX models are
  not already on disk.  The package only needs ``pkg.config`` and the model
  graph metadata; weights need not have been written yet.

- :func:`export_package` — save+config API. Takes an already-built
  ``ModelPackage`` and writes both the ONNX models AND the ORT-GenAI config
  artifacts in one call.  Use this when you built the package manually
  (e.g. with custom dtype / quantization).

- :func:`auto_export` — end-to-end API. Builds the model from a HuggingFace
  ID and calls :func:`export_package`. Use this for the common
  HF-model-id → ORT-GenAI-directory case.

All three produce a directory that ``onnxruntime-genai`` can load directly.

Example::

    # Config-only — assumes you've already saved the ONNX files yourself
    from mobius import build
    from mobius.integrations.ort_genai import write_ort_genai_config

    pkg = build("Qwen/Qwen3-0.6B", load_weights=True)
    pkg.save("/output/qwen3")  # write ONNX + weights first
    write_ort_genai_config(pkg, "/output/qwen3", hf_model_id="Qwen/Qwen3-0.6B")

    # Save + config — single call, when you have a built package in memory
    from mobius.integrations.ort_genai import export_package

    pkg = build("Qwen/Qwen3-0.6B", load_weights=True)
    export_package(pkg, "/output/qwen3", hf_model_id="Qwen/Qwen3-0.6B", ep="cuda")

    # End-to-end — when you only have an HF model id
    from mobius.integrations.ort_genai.auto_export import auto_export

    auto_export("Qwen/Qwen3-0.6B", "/output/qwen3", ep="cuda")
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import onnx_ir as ir

    from mobius._model_package import ModelPackage

logger = logging.getLogger(__name__)

# ORT-GenAI model type overrides for model types whose ORT-GenAI name
# differs from the HuggingFace model_type.
_ORT_GENAI_MODEL_TYPE: dict[str, str] = {
    "llama": "llama",
    "qwen2": "qwen2",
    "qwen3": "qwen2",
    "phi3": "phi3",
    "phi": "phi",
    "phi4mm": "phi4mm",
    "phi4_multimodal": "phi4mm",
    "gemma": "gemma",
    "gemma2": "gemma",
    "gemma4": "gemma4",
    "gemma4_text": "gemma4_text",
    # gemma-4-12B "unified" (encoder-free) variant reuses the gemma4 ORT GenAI
    # pipelines: the multimodal package (decoder taking inputs_embeds + vision
    # embedder + embedding fusion) maps to "gemma4"; the standalone text
    # backbone maps to "gemma4_text".
    "gemma4_unified": "gemma4",
    "gemma4_unified_text": "gemma4_text",
    "mistral": "mistral",
    "mistral3": "mistral3",
    # HunYuan-V1 dense / Hy-MT1.5 — generic decoder LLM type accepted by
    # ORT GenAI (see onnxruntime-genai/src/models/model_type.h LLM list).
    "hunyuan_v1_dense": "decoder",
    # Qwen VL models all use the same GenAI pipeline as qwen2_5_vl
    "qwen2_vl": "qwen2_5_vl",
    "qwen3_vl": "qwen2_5_vl",
    "qwen3_vl_text": "qwen2_5_vl",
    "qwen3_5": "qwen2_5_vl",
    "qwen3_5_vl": "qwen2_5_vl",
}

_GEMMA4_MODEL_TYPES = frozenset(
    {"gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text"}
)
# Encoder-free gemma-4-12B "unified" variants. Their image/audio inputs are raw
# merged pixel patches (48px, 6912-dim) / raw waveform frames (640-dim), NOT the
# SigLIP 16px / 128-dim log-mel contract that the ort-extensions
# ``Gemma4ImageTransform`` / ``Gemma4LogMel`` ops implement. There is no
# genai-native transform for the unified contract, so we deliberately do NOT
# emit image_processor.json / audio_processor.json for these models — callers
# must preprocess with the HuggingFace processor and feed tensors via
# ``Generator.set_inputs`` (see examples/gemma4_unified_ort_genai.py).
_GEMMA4_UNIFIED_MODEL_TYPES = frozenset({"gemma4_unified", "gemma4_unified_text"})
_PIXTRAL_MODEL_TYPES = frozenset({"mistral3"})
_QWEN_VL_MODEL_TYPES = frozenset(
    {
        "qwen2_vl",
        "qwen2_5_vl",
        "qwen3_vl",
        "qwen3_vl_text",
        "qwen3_5",
        "qwen3_5_vl",
        "qwen3_5_moe",
        "videochat_flash_qwen",
    }
)

_TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",  # SentencePiece
    "added_tokens.json",
    "merges.txt",  # BPE
    "vocab.json",  # BPE
    "chat_template.jinja",  # Chat template for ORT GenAI
]


def _resolve_ort_genai_model_type(model_type: str) -> str:
    """Map HuggingFace model_type to ORT-GenAI model type string."""
    return _ORT_GENAI_MODEL_TYPE.get(model_type, model_type)


def _select_ort_model_type(
    config_model_type: str | None,
    hf_model_type: str | None,
    *,
    is_decoder_only: bool,
) -> str:
    """Choose the ORT-GenAI model type for an exported package.

    Decoder-only packages prefer the built package's ``config.model_type`` so
    text-only / overridden builds (e.g. ``gemma4_unified -> gemma4_unified_text``)
    resolve to the decoder-only ORT type. Multimodal packages keep the HF
    parent ``model_type``: ``build()`` unwraps composite configs to their text
    sub-config, so ``config.model_type`` would otherwise be the text type even
    for a full multimodal export.
    """
    if is_decoder_only and config_model_type:
        return _resolve_ort_genai_model_type(config_model_type)
    return _resolve_ort_genai_model_type(hf_model_type or "unknown")


def _graph_input_names(model: ir.Model) -> list[str]:
    """Return non-KV-cache input names from an ONNX model graph.

    Filters out KV cache inputs (``past_key_values.*`` and ``past_*``)
    since those are represented as template patterns in genai_config.json,
    not as literal graph input names.
    """
    return [
        inp.name
        for inp in model.graph.inputs
        if inp.name is not None
        and not inp.name.startswith("past_key_values.")
        and not inp.name.startswith("past_")
    ]


def _introspect_inputs(pkg: ModelPackage, key: str) -> dict[str, str] | None:
    """Return ``{name: name}`` identity mapping for a sub-model's inputs.

    Returns ``None`` when *key* is absent from *pkg*, letting callers
    fall back to hard-coded defaults.
    """
    model = pkg.get(key)
    if model is None:
        return None
    return {n: n for n in _graph_input_names(model)}


def _introspect_outputs(pkg: ModelPackage, key: str) -> dict[str, str] | None:
    """Return ``{name: name}`` identity mapping for a sub-model's outputs.

    Returns ``None`` when *key* is absent from *pkg*.
    """
    model = pkg.get(key)
    if model is None:
        return None
    return {out.name: out.name for out in model.graph.outputs if out.name is not None}


def _copy_tokenizer_files(
    model_id: str,
    output_dir: str,
) -> list[str]:
    """Download and copy tokenizer files from HuggingFace Hub.

    Returns list of copied filenames.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    copied: list[str] = []
    for filename in _TOKENIZER_FILES:
        try:
            src = hf_hub_download(model_id, filename)
            dst = os.path.join(output_dir, filename)
            shutil.copy2(src, dst)
            copied.append(filename)
        except (EntryNotFoundError, OSError):
            continue
    return copied


def _copy_tokenizer_files_from_local(
    source_dir: str,
    output_dir: str,
) -> list[str]:
    """Copy tokenizer files from a local model directory.

    Silently skips files that are absent (not all tokenizer variants have
    all files — e.g. SentencePiece models have ``tokenizer.model`` but not
    ``merges.txt``).

    Returns list of copied filenames.
    """
    if not os.path.isdir(source_dir):
        logger.warning(
            "Local tokenizer source directory does not exist: %s — no tokenizer files copied.",
            source_dir,
        )
        return []
    copied: list[str] = []
    for filename in _TOKENIZER_FILES:
        src = os.path.join(source_dir, filename)
        if os.path.isfile(src):
            dst = os.path.join(output_dir, filename)
            shutil.copy2(src, dst)
            copied.append(filename)
    return copied


# Tokenizer class remapping: HF tokenizer classes that ORT GenAI
# (ort-extensions) does not support, mapped to compatible alternatives.
_TOKENIZER_CLASS_REMAP: dict[str, str] = {
    "TokenizersBackend": "LlamaTokenizer",
}


def _fix_tokenizer_config(output_dir: str) -> bool:
    """Remap unsupported tokenizer classes for ORT GenAI compatibility.

    Some HuggingFace models use tokenizer classes (e.g.
    ``TokenizersBackend``) that ORT GenAI's ort-extensions
    tokenizer doesn't support. This fixes tokenizer_config.json
    to use a compatible class.

    Returns True if a fix was applied, False otherwise.
    """
    tc_path = os.path.join(output_dir, "tokenizer_config.json")
    if not os.path.exists(tc_path):
        return False

    with open(tc_path, encoding="utf-8") as f:
        tc = json.load(f)

    original_class = tc.get("tokenizer_class", "")
    replacement = _TOKENIZER_CLASS_REMAP.get(original_class)
    if replacement is None:
        return False

    tc["tokenizer_class"] = replacement
    with open(tc_path, "w", encoding="utf-8") as f:
        json.dump(tc, f, indent=2, ensure_ascii=False)
    logger.info(
        "Fixed tokenizer_class: %s -> %s",
        original_class,
        replacement,
    )
    return True


def _fix_chat_template(output_dir: str, hf_model_id: str | None) -> bool:
    """Ensure chat_template is present in tokenizer_config.json.

    Some HuggingFace models don't store ``chat_template`` in the
    raw ``tokenizer_config.json`` file — transformers injects it
    dynamically from the model class at runtime. ORT GenAI reads
    the file directly and needs it to be present.

    This function loads the tokenizer via transformers (which
    applies the dynamic template) and writes it back.

    Returns True if the template was added, False otherwise.
    """
    tc_path = os.path.join(output_dir, "tokenizer_config.json")
    if not os.path.exists(tc_path):
        return False

    with open(tc_path, encoding="utf-8") as f:
        tc = json.load(f)

    if tc.get("chat_template"):
        return False

    if hf_model_id is None:
        return False

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        template = getattr(tokenizer, "chat_template", None)
        if template:
            tc["chat_template"] = template
            with open(tc_path, "w", encoding="utf-8") as f:
                json.dump(tc, f, indent=2, ensure_ascii=False)
            logger.info(
                "Added chat_template to tokenizer_config.json from %s",
                hf_model_id,
            )
            return True
    except Exception:
        logger.warning(
            "Could not load tokenizer for %s to extract chat_template",
            hf_model_id,
            exc_info=True,
        )
    return False


def _build_vision_transform_pipeline(
    *,
    image_size: int,
    patch_size: int,
    merge_size: int,
    rescale_factor: float,
    image_mean: list[float],
    image_std: list[float],
    min_pixels: int = 784,
    max_pixels: int = 2371600,
) -> list[dict[str, Any]]:
    """Build the common 5-step vision transform pipeline.

    Returns the base transforms: DecodeImage → ConvertRGB → Resize →
    Rescale → Normalize.  Callers may append model-specific steps
    (e.g. Permute3D, PixtralImageSizes) after this.
    """
    return [
        {
            "operation": {
                "name": "decode_image",
                "type": "DecodeImage",
                "attrs": {"color_space": "RGB"},
            }
        },
        {
            "operation": {
                "name": "convert_to_rgb",
                "type": "ConvertRGB",
            }
        },
        {
            "operation": {
                "name": "resize",
                "type": "Resize",
                "attrs": {
                    "height": image_size,
                    "width": image_size,
                    "smart_resize": 1,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    "patch_size": patch_size,
                    "merge_size": merge_size,
                },
            }
        },
        {
            "operation": {
                "name": "rescale",
                "type": "Rescale",
                "attrs": {
                    "rescale_factor": rescale_factor,
                },
            }
        },
        {
            "operation": {
                "name": "normalize",
                "type": "Normalize",
                "attrs": {
                    "mean": image_mean,
                    "std": image_std,
                },
            }
        },
    ]


def _write_vision_processor_config(
    config: Any,
    output_dir: str,
    *,
    hf_model_id: str | None = None,
) -> str | None:
    """Write the vision processor config file for VLM models.

    Generates the ORT-extensions image transform pipeline derived from the
    HuggingFace image processor config. When ``hf_model_id`` is provided,
    loads the HF processor to extract normalization values and resize
    parameters. Otherwise falls back to CLIP-standard defaults.

    The output format depends on the model type:

    - **Gemma4** (``gemma4``, ``gemma4_text``): Writes ``image_processor.json``
      with a ``DecodeImage → Gemma4ImageTransform`` pipeline.
    - **Gemma4 unified** (``gemma4_unified*``): Returns ``None`` — the
      encoder-free model has no matching ort-extensions transform; callers feed
      HF-preprocessed pixel_values via ``Generator.set_inputs``.
    - **Pixtral / Mistral3**: Writes ``processor_config.json`` with a 7-step
      pipeline (DecodeImage → ConvertRGB → Resize → Rescale → Normalize →
      Permute3D → PixtralImageSizes).
    - **Other VLMs**: Writes ``processor_config.json`` with a 5-step pipeline
      (DecodeImage → ConvertRGB → Resize → Rescale → Normalize).

    Returns the written file path, or None if the config has no vision section.
    """
    vision = getattr(config, "vision", None)
    if vision is None:
        return None

    model_type = getattr(config, "model_type", "")
    if model_type in _GEMMA4_UNIFIED_MODEL_TYPES:
        # Encoder-free unified model: no ort-extensions transform matches its
        # raw merged-patch contract. Emit no image_processor.json; callers feed
        # HF-preprocessed pixel_values via Generator.set_inputs.
        logger.info(
            "Skipping image_processor.json for encoder-free %s "
            "(no native ort-extensions transform; use HF processor + set_inputs)",
            model_type,
        )
        return None

    vision_model_type = getattr(vision, "model_type", None)
    is_pixtral = vision_model_type == "pixtral" or model_type in _PIXTRAL_MODEL_TYPES

    if model_type in _GEMMA4_MODEL_TYPES:
        # Gemma4 needs an onnxruntime-extensions format processor config
        # with a transforms pipeline (DecodeImage -> Gemma4ImageTransform).
        max_soft_tokens = (
            getattr(vision, "mm_tokens_per_image", None)
            or getattr(config, "mm_tokens_per_image", None)
            or 280
        )
        patch_size = getattr(vision, "patch_size", None) or 16
        pooling_kernel_size = getattr(vision, "pooling_kernel_size", None) or 3
        processor_config: dict[str, Any] = {
            "processor": {
                "name": "gemma_4_image_processing",
                "transforms": [
                    {
                        "operation": {
                            "name": "decode_image",
                            "type": "DecodeImage",
                            "attrs": {"color_space": "RGB"},
                        }
                    },
                    {
                        "operation": {
                            "name": "gemma4_image_transform",
                            "type": "Gemma4ImageTransform",
                            "attrs": {
                                "patch_size": patch_size,
                                "max_soft_tokens": max_soft_tokens,
                                "pooling_kernel_size": pooling_kernel_size,
                            },
                        }
                    },
                ],
            }
        }
        path = os.path.join(output_dir, "image_processor.json")
    else:
        # Pixtral and generic VLMs share the same base pipeline;
        # Pixtral adds Permute3D + PixtralImageSizes at the end.
        patch_size = getattr(vision, "patch_size", 14) or 14
        merge_size = (
            getattr(vision, "spatial_merge_size", None)
            or getattr(config, "spatial_merge_size", 2)
            or 2
        )

        # CLIP-standard normalization defaults
        image_mean = [0.48145466, 0.4578275, 0.40821073]
        image_std = [0.26862954, 0.26130258, 0.27577711]
        rescale_factor = 1.0 / 255.0
        min_pixels = 784
        max_pixels = 2371600
        image_size = getattr(vision, "image_size", None)

        if hf_model_id is not None:
            try:
                from transformers import AutoProcessor

                hf_proc = AutoProcessor.from_pretrained(hf_model_id)
                ip = getattr(hf_proc, "image_processor", None)
                if ip is not None:
                    image_mean = list(getattr(ip, "image_mean", image_mean))
                    image_std = list(getattr(ip, "image_std", image_std))
                    rescale_factor = getattr(ip, "rescale_factor", rescale_factor)
                    if hasattr(ip, "size"):
                        size = ip.size
                        if isinstance(size, dict):
                            if "longest_edge" in size:
                                image_size = size["longest_edge"]
                            min_pixels = size.get("shortest_edge", min_pixels)
                            max_pixels = size.get("longest_edge", max_pixels)
                        elif isinstance(size, int):
                            image_size = size
            except Exception:
                logger.warning(
                    "Could not load HF processor for %s; "
                    "using CLIP-standard normalization defaults",
                    hf_model_id,
                    exc_info=True,
                )

        if image_size is None:
            image_size = 1540 if is_pixtral else 448

        transforms = _build_vision_transform_pipeline(
            image_size=image_size,
            patch_size=patch_size,
            merge_size=merge_size,
            rescale_factor=rescale_factor,
            image_mean=image_mean,
            image_std=image_std,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        if is_pixtral:
            # Pixtral requires Permute3D (HWC→CHW) and PixtralImageSizes
            # for the per-image slicing loop in PixtralVisionState.
            transforms.append(
                {
                    "operation": {
                        "name": "permute",
                        "type": "Permute3D",
                        "attrs": {"dims": [2, 0, 1]},
                    }
                }
            )
            transforms.append(
                {
                    "operation": {
                        "name": "pixtral_image_sizes",
                        "type": "PixtralImageSizes",
                    }
                }
            )
        elif model_type in _QWEN_VL_MODEL_TYPES:
            # Qwen-VL models need the PatchImage transform to extract
            # temporal+spatial patches, and qwen2_5_vl/qwen3_vl flag
            # on Normalize for correct interleaving.
            temporal_patch_size = config.temporal_patch_size
            # Add qwen3_vl flag to the Normalize step
            for t in transforms:
                op = t.get("operation", {})
                if op.get("type") == "Normalize":
                    op.setdefault("attrs", {})["qwen2_5_vl"] = 1
            transforms.append(
                {
                    "operation": {
                        "name": "patch_image",
                        "type": "PatchImage",
                        "attrs": {
                            "patch_size": patch_size,
                            "temporal_patch_size": temporal_patch_size,
                            "merge_size": merge_size,
                        },
                    }
                }
            )

        processor_name = (
            "pixtral_image_processor"
            if is_pixtral
            else "qwen2_5_image_processor"
            if model_type in _QWEN_VL_MODEL_TYPES
            else "image_processor"
        )
        processor_config = {
            "processor": {
                "name": processor_name,
                "transforms": transforms,
            }
        }
        path = os.path.join(output_dir, "processor_config.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(processor_config, f, indent=4)
    return path


def _write_audio_processor_config(
    config: Any,
    output_dir: str,
) -> str | None:
    """Write audio_processor.json for models with audio encoders.

    Returns the path if written, None otherwise.
    """
    audio = getattr(config, "audio", None)
    if audio is None:
        return None

    model_type = getattr(config, "model_type", "")

    if model_type in _GEMMA4_UNIFIED_MODEL_TYPES:
        # Encoder-free unified model: raw 640-dim waveform frames, not the
        # 128-dim log-mel Gemma4LogMel contract. Emit no audio_processor.json;
        # callers feed HF-preprocessed input_features via Generator.set_inputs.
        logger.info(
            "Skipping audio_processor.json for encoder-free %s "
            "(no native ort-extensions transform; use HF processor + set_inputs)",
            model_type,
        )
        return None

    if model_type in _GEMMA4_MODEL_TYPES:
        # Gemma4 USM-style 128-dim log-mel spectrogram.
        # OrtxCreateSpeechFeatureExtractor requires the feature_extraction.sequence format.
        processor = {
            "feature_extraction": {
                "sequence": [
                    {
                        "operation": {
                            "name": "audio_decoder",
                            "type": "AudioDecoder",
                        }
                    },
                    {
                        "operation": {
                            "name": "gemma4_log_mel",
                            "type": "Gemma4LogMel",
                            "attrs": {
                                "feature_size": 128,
                                "sampling_rate": 16000,
                                "frame_length_ms": 20.0,
                                "hop_length_ms": 10.0,
                                "min_frequency": 0.0,
                                "max_frequency": 8000.0,
                                "preemphasis": 0.0,
                                "preemphasis_htk_flavor": 1,
                                "fft_overdrive": 0,
                                "mel_floor": 0.001,
                            },
                        }
                    },
                ]
            }
        }
        proc_filename = "audio_feature_extraction.json"
    else:
        # Generic audio processor — add model-specific branches as needed.
        return None

    path = os.path.join(output_dir, proc_filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(processor, f, indent=4)
    return path


def _write_genai_config(
    config: Any,
    output_dir: str,
    *,
    pkg: ModelPackage,
    ort_model_type: str,
    ep: str,
    context_length: int,
    bos_token_id: int | None,
    eos_token_id: int | list[int] | None,
    pad_token_id: int | None,
    is_vlm: bool,
    has_speech: bool,
) -> str:
    """Generate and write genai_config.json.

    Input names for each sub-model (decoder, vision, embedding) are
    introspected from the ONNX graphs in *pkg* rather than hard-coded
    per model type.

    Returns the path to the written file.
    """
    from mobius.integrations.ort_genai.genai_config import GenaiConfigGenerator

    # --- Discover decoder inputs from the ONNX graph ---
    decoder_key = "decoder" if "decoder" in pkg else "model"
    decoder_inputs = _introspect_inputs(pkg, decoder_key)
    if decoder_inputs is not None:
        # KV cache entries are template-based, not per-input
        decoder_inputs["past_key_names"] = "past_key_values.%d.key"
        decoder_inputs["past_value_names"] = "past_key_values.%d.value"

    # Derive decoder filename from the actual package key
    decoder_filename = f"{decoder_key}/model.onnx" if decoder_key != "model" else "model.onnx"

    # ORT GenAI's ``past_present_share_buffer`` mode requires the decoder
    # graph to write the KV cache in place. Only ``com.microsoft.
    # GroupQueryAttention`` does that; the standard ONNX ``Attention`` op
    # concatenates ``past_key`` with the new ``K`` and returns a dynamic-
    # shape ``present_key``, which is incompatible with the pre-allocated
    # shared buffer. Introspect the graph: if there is at least one GQA
    # node, the model supports shared-buffer mode; otherwise force it off
    # regardless of the EP capability flag.
    decoder_model = pkg.get(decoder_key)
    supports_in_place_kv_cache: bool | None = None
    if decoder_model is not None:
        supports_in_place_kv_cache = any(
            node.op_type == "GroupQueryAttention" and node.domain == "com.microsoft"
            for node in decoder_model.graph
        )

    generator = GenaiConfigGenerator.from_config(
        config,
        ort_model_type,
        context_length=context_length,
        ep=ep,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        decoder_inputs=decoder_inputs,
        decoder_filename=decoder_filename,
        supports_in_place_kv_cache=supports_in_place_kv_cache,
    )

    if is_vlm:
        image_token_id = getattr(config, "image_token_id", None)
        if image_token_id is not None:
            vision_input_mapping = _introspect_inputs(pkg, "vision_encoder")
            embedding_input_mapping = _introspect_inputs(pkg, "embedding")

            # spatial_merge_size and config_filename are config-level
            # properties that cannot be inferred from the graph.
            vision_kwargs: dict[str, Any] = {}
            model_type = getattr(config, "model_type", "")
            if model_type in _GEMMA4_MODEL_TYPES:
                vision_cfg = getattr(config, "vision", None)
                vision_kwargs["spatial_merge_size"] = getattr(
                    vision_cfg, "spatial_merge_size", 2
                )
            elif has_speech:
                vision_kwargs["spatial_merge_size"] = None
            elif (
                model_type in _PIXTRAL_MODEL_TYPES
                or getattr(getattr(config, "vision", None), "model_type", None) == "pixtral"
            ):
                vision_cfg = getattr(config, "vision", None)
                sms = getattr(vision_cfg, "spatial_merge_size", None) or getattr(
                    config, "spatial_merge_size", 2
                )
                vision_kwargs["spatial_merge_size"] = sms
                vision_kwargs["config_filename"] = "processor_config.json"
            else:
                # All other VLMs (Qwen-VL, LLaVA, InternVL, etc.) use
                # processor_config.json written by _write_vision_processor_config.
                vision_cfg = getattr(config, "vision", None)
                sms = getattr(vision_cfg, "spatial_merge_size", None) or getattr(
                    config, "spatial_merge_size", None
                )
                if sms is not None:
                    vision_kwargs["spatial_merge_size"] = sms
                vision_kwargs["config_filename"] = "processor_config.json"

            if vision_input_mapping is not None:
                vision_kwargs["input_names"] = vision_input_mapping
            if embedding_input_mapping is not None:
                vision_kwargs["embedding_input_names"] = embedding_input_mapping

            embedding_output_mapping = _introspect_outputs(pkg, "embedding")
            if embedding_output_mapping is not None:
                vision_kwargs["embedding_output_names"] = embedding_output_mapping

            generator.with_vision(image_token_id=image_token_id, **vision_kwargs)

    if has_speech:
        audio_input_mapping = _introspect_inputs(pkg, "audio_encoder")

        audio_config = getattr(config, "audio", None)
        audio_token_id = (
            getattr(audio_config, "token_id", None)
            or getattr(audio_config, "audio_token_id", None)
            or getattr(config, "audio_token_id", None)
        )
        boa_token_id = getattr(config, "boa_token_id", None)

        audio_kwargs: dict[str, Any] = {}
        model_type = getattr(config, "model_type", "")
        if model_type in _GEMMA4_MODEL_TYPES:
            # Gemma4 audio encoder uses different filename and config
            audio_kwargs["filename"] = "audio_encoder/model.onnx"
            audio_kwargs["config_filename"] = "audio_feature_extraction.json"
            # Gemma4 speech model input is 'input_features' + 'input_features_mask'
            audio_kwargs["input_names"] = {
                "audio_embeds": "input_features",
                "attention_mask": "input_features_mask",
            }
        elif audio_input_mapping is not None:
            audio_kwargs["input_names"] = audio_input_mapping
        generator.with_audio(
            audio_token_id=audio_token_id,
            boa_token_id=boa_token_id,
            **audio_kwargs,
        )

    return generator.write(output_dir)


def write_ort_genai_config(
    pkg: ModelPackage,
    directory: str,
    *,
    hf_model_id: str | None = None,
    ep: str = "cpu",
    context_length: int = 4096,
    local_config_dir: str | None = None,
) -> dict[str, str]:
    """Generate ORT-GenAI config artifacts for an already-built ModelPackage.

    Writes ``genai_config.json``, optionally copies tokenizer files from
    HuggingFace Hub or a local directory, and writes ``image_processor.json``
    for VLM models.  Does NOT build or save ONNX models — call
    :meth:`~mobius._model_package.ModelPackage.save` separately before or
    after this function.

    Args:
        pkg: Already-built :class:`~mobius._model_package.ModelPackage` with
            weights applied and ``config`` set.
        directory: Output directory (created if needed).
        hf_model_id: HuggingFace model ID. When provided, used to fetch token
            IDs (``bos``/``eos``/``pad``) and download tokenizer files.
            When ``None``, token IDs are read from ``pkg.config`` fields
            (``bos_token_id``, ``eos_token_id``, ``pad_token_id``) populated
            by :meth:`~mobius._configs.ArchitectureConfig.from_transformers`,
            and tokenizer files are not copied unless ``local_config_dir`` is set.
        ep: Execution provider for ``session_options`` in
            ``genai_config.json`` (e.g. ``"cpu"``, ``"cuda"``, ``"dml"``,
            ``"trt-rtx"``). Defaults to ``"cpu"``.
        context_length: Minimum context length written to
            ``genai_config.json``. Overridden upward by
            ``max_position_embeddings`` from ``pkg.config``.
        local_config_dir: Path to a local model directory. When provided
            and ``hf_model_id`` is ``None``, tokenizer files are copied from
            this directory instead of downloaded from HuggingFace Hub.
            Typically set when the CLI ``--config`` flag points to a local
            directory rather than a HuggingFace model ID.

    Returns:
        Dict mapping artifact name to file path, e.g.::

            {
                "genai_config": "/output/genai_config.json",
                "tokenizer.json": "/output/tokenizer.json",
                "image_processor": "/output/image_processor.json",
            }

    Raises:
        ValueError: If ``pkg.config`` is ``None`` (required for config
            generation).
    """
    config = getattr(pkg, "config", None)
    if config is None:
        raise ValueError(
            "write_ort_genai_config requires ModelPackage.config to be set. "
            "This is set automatically when building with mobius.build(). "
            "Diffusion models (which have no config) are not supported."
        )

    os.makedirs(directory, exist_ok=True)

    # Normalize EP: 'default' and 'onnx-standard' are portable-ONNX modes
    # that carry no EP-specific session options → treat as CPU.
    if ep in ("default", "onnx-standard"):
        ep = "cpu"

    # Resolve token IDs and ORT model type from HF config (if provided)
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    pad_token_id: int | None = None
    ort_model_type: str

    # Detect multimodal capabilities from the package keys. Needed before
    # resolving the ORT model type so decoder-only (text-only) packages can
    # prefer their own config.model_type (see below).
    is_vlm = "vision_encoder" in pkg and "embedding" in pkg
    has_speech = "audio_encoder" in pkg
    is_decoder_only = not is_vlm and not has_speech

    if hf_model_id is not None:
        import transformers

        hf_config = transformers.AutoConfig.from_pretrained(hf_model_id)
        model_type = hf_config.model_type
        # For decoder-only packages, prefer the built package's model_type so
        # text-only / overridden builds (e.g. gemma4_unified -> gemma4_unified_text)
        # resolve to the correct decoder-only ORT-GenAI type rather than the
        # multimodal one reported by the HuggingFace config. For multimodal
        # packages keep the HF parent model_type: build() unwraps composite
        # configs to their text sub-config, so pkg.config.model_type would
        # otherwise be the text type even for a full multimodal export.
        cfg_model_type = getattr(config, "model_type", None)
        ort_model_type = _select_ort_model_type(
            cfg_model_type, model_type, is_decoder_only=is_decoder_only
        )
        # Token IDs may live on the parent config or the text sub-config
        # (e.g. Gemma4Config has text_config with bos_token_id=2).
        _tok_cfg = getattr(hf_config, "text_config", hf_config)
        bos_token_id = getattr(
            hf_config,
            "bos_token_id",
            getattr(_tok_cfg, "bos_token_id", None),
        )
        eos_token_id = getattr(
            hf_config,
            "eos_token_id",
            getattr(_tok_cfg, "eos_token_id", None),
        )
        pad_token_id = getattr(
            hf_config,
            "pad_token_id",
            getattr(_tok_cfg, "pad_token_id", None),
        )
    else:
        # Fall back to fields stored in ArchitectureConfig (set by from_transformers()).
        # This path is taken when hf_model_id is not provided (e.g. --config mode).
        raw_type = getattr(config, "model_type", None) or "unknown"
        ort_model_type = _resolve_ort_genai_model_type(raw_type)
        if ort_model_type == "unknown":
            logger.warning(
                "Could not determine ORT-GenAI model type: pkg.config.model_type "
                "is missing, None, or not mapped to an ORT-GenAI type (got %r). "
                "Pass hf_model_id to resolve it from the HuggingFace config, or "
                "the generated genai_config.json may not load correctly.",
                raw_type,
            )
        # Read token IDs from ArchitectureConfig (populated by from_transformers()
        # when --config is used with a local directory).
        bos_token_id = getattr(config, "bos_token_id", None)
        eos_token_id = getattr(config, "eos_token_id", None)
        # pad_token_id lives in BaseModelConfig with DEFAULT_INT (-42) as the
        # "not set" sentinel (negative IDs are never valid token positions).
        _pad = getattr(config, "pad_token_id", None)
        pad_token_id = None if (_pad is None or _pad < 0) else _pad

    # Phi4MM quirk: HF reports model_type='phi' but the model package
    # includes an 'audio_encoder' component that distinguishes it from plain Phi.
    # Override to 'phi4mm' so ORT-GenAI loads the correct pipeline.
    if ort_model_type == "phi" and has_speech:
        ort_model_type = "phi4mm"

    logger.info("Generating genai_config.json for %s (ep=%s)", ort_model_type, ep)
    genai_path = _write_genai_config(
        config,
        directory,
        pkg=pkg,
        ort_model_type=ort_model_type,
        ep=ep,
        context_length=context_length,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        is_vlm=is_vlm,
        has_speech=has_speech,
    )

    result: dict[str, str] = {"genai_config": genai_path}

    # Copy tokenizer files — HF Hub takes precedence; local dir is the fallback
    # for --config mode where no HF model ID is available.
    if hf_model_id is not None:
        logger.info("Copying tokenizer files from %s", hf_model_id)
        tokenizer_files = _copy_tokenizer_files(hf_model_id, directory)
        for tf in tokenizer_files:
            result[tf] = os.path.join(directory, tf)
    elif local_config_dir is not None:
        logger.info("Copying tokenizer files from local directory %s", local_config_dir)
        tokenizer_files = _copy_tokenizer_files_from_local(local_config_dir, directory)
        if not tokenizer_files:
            logger.warning(
                "No tokenizer files were copied from local directory %s. "
                "The export may be missing tokenizer artifacts required by ORT-GenAI.",
                local_config_dir,
            )

        for tf in tokenizer_files:
            result[tf] = os.path.join(directory, tf)

    # Write processor config for VLMs
    processor_path = _write_vision_processor_config(config, directory, hf_model_id=hf_model_id)
    if processor_path:
        result["processor_config"] = processor_path

    # Write audio_processor.json for models with audio encoders
    audio_proc_path = _write_audio_processor_config(config, directory)
    if audio_proc_path:
        result["audio_processor"] = audio_proc_path

    # Fix unsupported tokenizer classes
    _fix_tokenizer_config(directory)

    # Ensure chat_template is in tokenizer_config.json
    _fix_chat_template(directory, hf_model_id)

    logger.info("ORT-GenAI artifacts written: %d files", len(result))
    return result


def export_package(
    pkg: ModelPackage,
    output_dir: str,
    *,
    hf_model_id: str | None = None,
    ep: str = "cpu",
    context_length: int = 4096,
    local_config_dir: str | None = None,
    external_data: str = "onnx",
    progress_bar: bool = True,
) -> dict[str, str]:
    """Save an already-built ModelPackage as a complete ORT-GenAI directory.

    This is the convenience function for users who built a ``ModelPackage``
    themselves (e.g. with custom dtype / quantization / weight overrides) and
    want a single call that produces an ``onnxruntime-genai``-loadable
    directory.  It calls :meth:`ModelPackage.save` followed by
    :func:`write_ort_genai_config`.

    For the end-to-end case where you start from a HuggingFace model id, use
    :func:`auto_export` instead — it builds the package for you.

    Args:
        pkg: Already-built :class:`~mobius._model_package.ModelPackage` with
            weights applied and ``config`` set.  All components in *pkg* are
            saved; selective export via :meth:`ModelPackage.save`'s
            ``components`` filter is intentionally not exposed here, because
            the ORT-GenAI runtime expects the on-disk file layout to match
            what ``model.type`` in :file:`genai_config.json` declares (e.g. a
            multimodal ``model.type`` implies a vision encoder file is
            present).  If you need a partial export, build a separate
            filtered ``ModelPackage`` first.
        output_dir: Output directory (created if needed).
        hf_model_id: HuggingFace model ID for tokenizer download / token-id
            resolution.  When ``None``, token IDs are read from ``pkg.config``
            and tokenizer files are not copied (unless ``local_config_dir``
            is provided).
        ep: Execution provider written to ``session_options`` in
            ``genai_config.json`` (e.g. ``"cpu"``, ``"cuda"``, ``"dml"``,
            ``"webgpu"``, ``"trt-rtx"``).
        context_length: Minimum context length written to ``genai_config.json``.
            Overridden upward by ``pkg.config.max_position_embeddings`` when
            larger.
        local_config_dir: Local model directory to copy tokenizer files from
            when ``hf_model_id`` is ``None``.
        external_data: External-data format passed to :meth:`ModelPackage.save`
            (``"onnx"`` or ``"safetensors"``).
        progress_bar: Whether to show the save progress bar.

    Returns:
        Manifest dict mapping artifact names to paths::

            {
                "model": "/output/model.onnx",          # or per-component paths
                "genai_config": "/output/genai_config.json",
                "tokenizer.json": "/output/tokenizer.json",
                ...
            }

    Raises:
        ValueError: If ``pkg.config`` is ``None`` (required for genai_config
            generation; e.g. diffusion models have no config and are not
            supported).

    Example::

        from mobius import build
        from mobius.integrations.ort_genai import export_package

        pkg = build("Qwen/Qwen3-0.6B", load_weights=True)
        export_package(pkg, "/output/qwen3", hf_model_id="Qwen/Qwen3-0.6B", ep="cuda")
    """
    # Preflight: fail fast before writing ONNX so the user doesn't end up
    # with a half-exported directory containing only the model file.
    if getattr(pkg, "config", None) is None:
        raise ValueError(
            "export_package requires ModelPackage.config to be set. "
            "This is set automatically when building with mobius.build(). "
            "Diffusion models (which have no config) are not supported — "
            "use ModelPackage.save() directly for those."
        )

    os.makedirs(output_dir, exist_ok=True)

    # 1. Save ONNX models + weights
    logger.info("Saving ONNX models to %s", output_dir)
    pkg.save(
        output_dir,
        external_data=external_data,
        progress_bar=progress_bar,
    )

    # 2. Write ORT-GenAI config artifacts
    result = write_ort_genai_config(
        pkg,
        output_dir,
        hf_model_id=hf_model_id,
        ep=ep,
        context_length=context_length,
        local_config_dir=local_config_dir,
    )

    # 3. Add ONNX paths to the manifest
    if len(pkg) == 1:
        result["model"] = os.path.join(output_dir, "model.onnx")
    else:
        for name in pkg:
            result[name] = os.path.join(output_dir, name, "model.onnx")

    logger.info("Export complete: %d artifacts", len(result))
    return result


def auto_export(
    model_id: str,
    output_dir: str,
    *,
    dtype: str | None = None,
    task: str | None = None,
    external_data: str = "onnx",
    trust_remote_code: bool = False,
    context_length: int = 4096,
    ep: str = "cpu",
    progress_bar: bool = True,
    text_only: bool = False,
) -> dict[str, str]:
    """Build and export a model for onnxruntime-genai.

    This is the end-to-end convenience function for producing ORT-GenAI-ready
    model directories. It:

    1. Builds the ONNX graph(s) via :func:`~mobius._builder.build`
    2. Downloads and applies HuggingFace weights
    3. Saves ONNX model(s) with external data
    4. Calls :func:`write_ort_genai_config` to write ``genai_config.json``,
       tokenizer files, and ``image_processor.json``

    Args:
        model_id: HuggingFace model repository ID.
        output_dir: Directory to write all output files.
        dtype: Override model dtype (``"f32"``, ``"f16"``, ``"bf16"``).
        task: Override model task (auto-detected if ``None``).
        external_data: External data format (``"onnx"`` or
            ``"safetensors"``).
        trust_remote_code: Trust remote code for HuggingFace config.
        context_length: Minimum context length for genai_config.json.
        ep: Execution provider for ``session_options`` in
            ``genai_config.json``. Defaults to ``"cpu"``. For non-CPU providers
            this value also drives build-time ``execution_provider`` so the
            exported ONNX graph is fused for the same provider the runtime will
            use (e.g. ``"cuda"`` enables ``GroupQueryAttention`` fusion).
            ``"cpu"`` builds the portable ``"default"`` graph (unchanged
            behavior).
        progress_bar: Show progress bar during save.
        text_only: When ``True``, export the text backbone of a multimodal
            checkpoint as a standalone decoder-only LLM (see
            :func:`~mobius._builder.build`). Produces a single ``model.onnx``
            with a decoder-only ``genai_config.json`` (no vision/audio
            sections). Currently supported for ``gemma4_unified``
            (``google/gemma-4-12B``).

    Returns:
        Dict mapping output artifact names to file paths, e.g.::

            {
                "genai_config": "/output/genai_config.json",
                "model": "/output/model.onnx",
                "tokenizer.json": "/output/tokenizer.json",
            }
    """
    from mobius._builder import build

    os.makedirs(output_dir, exist_ok=True)

    # Build ONNX graph(s) with weights. The runtime EP (``ep``) also drives
    # EP-aware graph construction so fused ops (e.g. GroupQueryAttention on
    # CUDA) match the provider declared in genai_config.json. ``cpu`` maps to
    # the portable ``default`` build to preserve historical CPU/f32 output
    # (the CPU EP would otherwise fuse f32 GroupQueryAttention).
    build_ep = "default" if ep == "cpu" else ep
    logger.info("Building ONNX model for %s (build ep=%s)", model_id, build_ep)
    pkg = build(
        model_id,
        task=task,
        dtype=dtype,
        load_weights=True,
        trust_remote_code=trust_remote_code,
        execution_provider=build_ep,
        text_only=text_only,
    )

    if getattr(pkg, "config", None) is None:
        raise ValueError(
            f"Model package for '{model_id}' has no config attribute. "
            "auto_export requires a config to generate genai_config.json. "
            "Diffusion models are not yet supported."
        )

    # Delegate save + config generation to the integration helper.
    # `export_package` already logs "Export complete" so we don't repeat it here.
    result = export_package(
        pkg,
        output_dir,
        hf_model_id=model_id,
        ep=ep,
        context_length=context_length,
        external_data=external_data,
        progress_bar=progress_bar,
    )

    return result
