# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Auto-export pipeline for onnxruntime-genai.

Two entry points:

- :func:`write_ort_genai_config` — programmatic API. Takes an already-built
  :class:`~mobius._model_package.ModelPackage` (with weights) and writes the
  ORT-GenAI config artifacts (``genai_config.json``, tokenizer files,
  ``processor_config.json``) alongside the ONNX models.

- :func:`auto_export` — end-to-end convenience function. Builds the model
  from a HuggingFace ID, saves the ONNX files, then calls
  :func:`write_ort_genai_config` to write the config artifacts.

Both functions produce a directory that ``onnxruntime-genai`` can load
directly.

Example::

    # Programmatic API — build first, then export configs
    from mobius import build
    from mobius.integrations.ort_genai import write_ort_genai_config

    pkg = build("Qwen/Qwen3-0.6B", load_weights=True)
    pkg.save("/output/qwen3")
    write_ort_genai_config(pkg, "/output/qwen3", hf_model_id="Qwen/Qwen3-0.6B")

    # End-to-end convenience
    from mobius.integrations.ort_genai.auto_export import auto_export

    auto_export("Qwen/Qwen3-0.6B", "/output/qwen3")
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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
    "mistral": "mistral",
}


def _resolve_ort_genai_model_type(model_type: str) -> str:
    """Map HuggingFace model_type to ORT-GenAI model type string."""
    return _ORT_GENAI_MODEL_TYPE.get(model_type, model_type)


def _copy_tokenizer_files(
    model_id: str,
    output_dir: str,
) -> list[str]:
    """Download and copy tokenizer files from HuggingFace Hub.

    Returns list of copied filenames.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",  # SentencePiece
        "added_tokens.json",
        "merges.txt",  # BPE
        "vocab.json",  # BPE
    ]
    copied: list[str] = []
    for filename in tokenizer_files:
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
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",  # SentencePiece
        "added_tokens.json",
        "merges.txt",  # BPE
        "vocab.json",  # BPE
    ]
    copied: list[str] = []
    for filename in tokenizer_files:
        src = os.path.join(source_dir, filename)
        if os.path.isfile(src):
            dst = os.path.join(output_dir, filename)
            shutil.copy2(src, dst)
            copied.append(filename)
    return copied


def _write_processor_config(
    config: Any,
    output_dir: str,
) -> str | None:
    """Write a minimal processor_config.json for VLM models.

    Returns the path if written, None otherwise.
    """
    vision = getattr(config, "vision", None)
    if vision is None:
        return None

    processor: dict[str, Any] = {
        "image_size": getattr(vision, "image_size", 448),
        "patch_size": getattr(vision, "patch_size", 14),
    }
    path = os.path.join(output_dir, "processor_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(processor, f, indent=4)
    return path


def _write_genai_config(
    config: Any,
    output_dir: str,
    *,
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

    Returns the path to the written file.
    """
    from mobius.integrations.ort_genai.genai_config import GenaiConfigGenerator

    generator = GenaiConfigGenerator.from_config(
        config,
        ort_model_type,
        context_length=context_length,
        ep=ep,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )

    if is_vlm:
        image_token_id = getattr(config, "image_token_id", None)
        if image_token_id is not None:
            vision_kwargs: dict[str, Any] = {}
            if has_speech:
                # Phi4MM uses different vision inputs than Qwen2.5-VL
                vision_kwargs["spatial_merge_size"] = None
                vision_kwargs["config_filename"] = "vision_processor.json"
                vision_kwargs["input_names"] = {
                    "pixel_values": "pixel_values",
                    "image_sizes": "image_sizes",
                }
            generator.with_vision(image_token_id=image_token_id, **vision_kwargs)

    if has_speech:
        audio_config = getattr(config, "audio", None)
        audio_token_id = (
            getattr(audio_config, "token_id", None) if audio_config is not None else None
        )
        generator.with_speech(audio_token_id=audio_token_id)

    return generator.write(output_dir)


def write_ort_genai_config(
    pkg: ModelPackage,
    directory: str,
    *,
    hf_model_id: str | None = None,
    ep: str = "cpu",
    context_length: int = 4096,
    local_config_path: str | None = None,
) -> dict[str, str]:
    """Generate ORT-GenAI config artifacts for an already-built ModelPackage.

    Writes ``genai_config.json``, optionally copies tokenizer files from
    HuggingFace Hub or a local directory, and writes ``processor_config.json``
    for VLM models.  Does NOT build or save ONNX models — call
    :meth:`~mobius._model_package.ModelPackage.save` separately before or
    after this function.

    Args:
        pkg: Already-built :class:`~mobius._model_package.ModelPackage` with
            weights applied and ``config`` set.
        directory: Output directory (created if needed).
        hf_model_id: HuggingFace model ID. When provided, used to fetch token
            IDs (``bos``/``eos``/``pad``) and download tokenizer files.
            When ``None``, token IDs default to ``None`` and tokenizer files
            are not copied unless ``local_config_path`` is set.
        ep: Execution provider for ``session_options`` in
            ``genai_config.json`` (e.g. ``"cpu"``, ``"cuda"``, ``"dml"``,
            ``"trt-rtx"``). Defaults to ``"cpu"``.
        context_length: Minimum context length written to
            ``genai_config.json``. Overridden upward by
            ``max_position_embeddings`` from ``pkg.config``.
        local_config_path: Path to a local model directory. When provided
            and ``hf_model_id`` is ``None``, tokenizer files are copied from
            this directory instead of downloaded from HuggingFace Hub.
            Typically set when the CLI ``--config`` flag points to a local
            directory rather than a HuggingFace model ID.

    Returns:
        Dict mapping artifact name to file path, e.g.::

            {
                "genai_config": "/output/genai_config.json",
                "tokenizer.json": "/output/tokenizer.json",
                "processor_config": "/output/processor_config.json",
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

    if hf_model_id is not None:
        import transformers

        hf_config = transformers.AutoConfig.from_pretrained(hf_model_id)
        model_type = hf_config.model_type
        ort_model_type = _resolve_ort_genai_model_type(model_type)
        bos_token_id = getattr(hf_config, "bos_token_id", None)
        eos_token_id = getattr(hf_config, "eos_token_id", None)
        pad_token_id = getattr(hf_config, "pad_token_id", None)
    else:
        # Fall back to model_type from the mobius ArchitectureConfig.
        # This path is taken when hf_model_id is not provided, so HF config
        # is unavailable. The ArchitectureConfig.model_type may be absent on
        # older configs, producing 'unknown' — ORT-GenAI may reject it.
        raw_type = getattr(config, "model_type", "unknown")
        ort_model_type = _resolve_ort_genai_model_type(raw_type)
        if ort_model_type == "unknown":
            logger.warning(
                "Could not determine ORT-GenAI model type: pkg.config has no "
                "'model_type' attribute. Pass hf_model_id to resolve it from "
                "the HuggingFace config, or the generated genai_config.json "
                "may not load correctly."
            )

    # Detect multimodal capabilities from the package keys
    is_vlm = "vision" in pkg and "embedding" in pkg
    has_speech = "speech" in pkg

    # Phi4MM quirk: HF reports model_type='phi' but the model package
    # includes a 'speech' component that distinguishes it from plain Phi.
    # Override to 'phi4mm' so ORT-GenAI loads the correct pipeline.
    if ort_model_type == "phi" and has_speech:
        ort_model_type = "phi4mm"

    logger.info("Generating genai_config.json for %s (ep=%s)", ort_model_type, ep)
    genai_path = _write_genai_config(
        config,
        directory,
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

    # Copy tokenizer files — from HuggingFace Hub or local directory
    if hf_model_id is not None:
        logger.info("Copying tokenizer files from %s", hf_model_id)
        tokenizer_files = _copy_tokenizer_files(hf_model_id, directory)
        for tf in tokenizer_files:
            result[tf] = os.path.join(directory, tf)
    elif local_config_path is not None:
        if not os.path.isdir(local_config_path):
            raise ValueError(
                f"local_config_path must be an existing directory: {local_config_path}"
            )
        logger.info("Copying tokenizer files from local path %s", local_config_path)
        tokenizer_files = _copy_tokenizer_files_from_local(local_config_path, directory)
        if not tokenizer_files:
            logger.warning(
                "No tokenizer files were copied from local path %s. "
                "The export may be missing tokenizer artifacts required by ORT-GenAI.",
                local_config_path,
            )
        for tf in tokenizer_files:
            result[tf] = os.path.join(directory, tf)

    # Write processor_config.json for VLMs
    processor_path = _write_processor_config(config, directory)
    if processor_path:
        result["processor_config"] = processor_path

    logger.info("ORT-GenAI artifacts written: %d files", len(result))
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
) -> dict[str, str]:
    """Build and export a model for onnxruntime-genai.

    This is the end-to-end convenience function for producing ORT-GenAI-ready
    model directories. It:

    1. Builds the ONNX graph(s) via :func:`~mobius._builder.build`
    2. Downloads and applies HuggingFace weights
    3. Saves ONNX model(s) with external data
    4. Calls :func:`write_ort_genai_config` to write ``genai_config.json``,
       tokenizer files, and ``processor_config.json``

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
            ``genai_config.json``. Defaults to ``"cpu"``.
        progress_bar: Show progress bar during save.

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

    # Build ONNX graph(s) with weights
    logger.info("Building ONNX model for %s", model_id)
    pkg = build(
        model_id,
        task=task,
        dtype=dtype,
        load_weights=True,
        trust_remote_code=trust_remote_code,
    )

    if getattr(pkg, "config", None) is None:
        raise ValueError(
            f"Model package for '{model_id}' has no config attribute. "
            "auto_export requires a config to generate genai_config.json. "
            "Diffusion models are not yet supported."
        )

    # Save ONNX models
    logger.info("Saving ONNX models to %s", output_dir)
    pkg.save(
        output_dir,
        external_data=external_data,
        progress_bar=progress_bar,
    )

    # Write ORT-GenAI config artifacts (genai_config.json, tokenizer, processor)
    result = write_ort_genai_config(
        pkg,
        output_dir,
        hf_model_id=model_id,
        ep=ep,
        context_length=context_length,
    )

    # Add ONNX model paths to manifest
    if len(pkg) == 1:
        result["model"] = os.path.join(output_dir, "model.onnx")
    else:
        for name in pkg:
            result[name] = os.path.join(output_dir, name, "model.onnx")

    logger.info("Export complete: %d artifacts", len(result))
