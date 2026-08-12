# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Write onnx-genai ``inference_metadata.yaml`` for a built Mobius package.

Dispatches on the package/config: a decoder-only LLM emits the
``model.attention`` + ``kv_cache`` document; a multimodal package emits a
composite encoder/fusion/decoder pipeline; and a diffusion package emits an
iterative pipeline. This is the onnx-genai analogue of
:func:`mobius.integrations.ort_genai.write_ort_genai_config`.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import yaml

from mobius.integrations.onnx_genai.decoder_metadata import (
    decoder_metadata_from_config,
    write_decoder_metadata,
)
from mobius.integrations.onnx_genai.inference_metadata import (
    SchedulerConfig,
    add_explicit_package_io,
    add_policy_components_to_workflow,
    load_diffusers_scheduler_config,
    write_audio_codec_pipeline_metadata,
    write_diffusion_pipeline_metadata,
    write_multimodal_pipeline_metadata,
    write_speech_to_text_pipeline_metadata,
    write_tts_pipeline_metadata,
)

_LOGGER = logging.getLogger(__name__)

_DENOISER_KEYS = ("denoiser", "transformer", "unet")


def _add_explicit_io_to_file(path: str, pkg: Any, config: Any) -> None:
    """Augment an emitted sidecar with roles derived from the actual ONNX ports."""
    try:
        models = list(pkg.values())
    except AttributeError:
        return
    if not models or any(not hasattr(model, "graph") for model in models):
        return
    with open(path, encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle)
    add_explicit_package_io(metadata, pkg, config)
    add_policy_components_to_workflow(metadata, pkg)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)


def _write_clip_tokenizer(output_dir: str, source: str | None) -> str | None:
    """Emit ``tokenizer.json`` for a text-conditioned diffusion package.

    Classic Stable Diffusion conditions on a CLIP text encoder, and the
    onnx-genai runners (e.g. ``render_sd``) load ``<package>/tokenizer.json`` —
    the ``tokenizers``-library fast-tokenizer serialization. This builds that
    file from the source pipeline's ``tokenizer/`` subfolder (via a fast
    ``CLIPTokenizerFast``, which is constructed from ``vocab.json`` + ``merges.txt``
    even when the repo ships no ``tokenizer.json``), so the package is
    self-contained. Best-effort: returns ``None`` (with a warning) if transformers
    is unavailable or the source has no CLIP tokenizer, without failing the build.

    Args:
        output_dir: Package directory to write ``tokenizer.json`` into.
        source: The diffusers checkpoint directory or Hugging Face id the
            components were built from.

    Returns:
        The written ``tokenizer.json`` path, or ``None`` if it could not be emitted.
    """
    if not source:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError:
        _LOGGER.warning(
            "transformers is not available; skipping tokenizer.json emission. "
            "The onnx-genai runners will need a tokenizer.json supplied separately."
        )
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(source, subfolder="tokenizer", use_fast=True)
    except Exception as error:  # best-effort; never block the build
        _LOGGER.warning(
            "Could not load a CLIP tokenizer from %r (subfolder 'tokenizer'): %s; "
            "skipping tokenizer.json emission.",
            source,
            error,
        )
        return None
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        _LOGGER.warning(
            "Loaded a slow tokenizer for %r with no fast backend; skipping "
            "tokenizer.json emission.",
            source,
        )
        return None
    path = os.path.join(output_dir, "tokenizer.json")
    backend.save(path)
    return path


def _write_hf_tokenizer(output_dir: str, source: str | None) -> str | None:
    """Emit ``tokenizer.json`` for a text-producing package from its HF source.

    Decoder-LM, multimodal (VLM / speech-language ASR), and Whisper-style ASR
    packages all reference ``<package>/tokenizer.json`` in their emitted metadata
    so the onnx-genai runtime can tokenize prompts from the package alone. This
    reconstructs that file from the source model's fast tokenizer, mirroring the
    diffusion CLIP tokenizer helper. Best-effort: it logs a warning and returns
    ``None`` (never raising) when ``transformers`` is unavailable, no ``source``
    is known, or the source has no fast tokenizer, so the build is not blocked.

    Args:
        output_dir: Package directory to write ``tokenizer.json`` into.
        source: The Hugging Face model id or local directory carrying the
            tokenizer.

    Returns:
        The written ``tokenizer.json`` path, or ``None`` if it could not be
        emitted.
    """
    if not source:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError:
        _LOGGER.warning(
            "transformers is not available; skipping tokenizer.json emission. "
            "The onnx-genai runners will need a tokenizer.json supplied separately."
        )
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True)
    except Exception as error:  # best-effort; never block the build
        _LOGGER.warning(
            "Could not load a tokenizer from %r: %s; skipping tokenizer.json emission.",
            source,
            error,
        )
        return None
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        _LOGGER.warning(
            "Loaded a slow tokenizer for %r with no fast backend; skipping "
            "tokenizer.json emission.",
            source,
        )
        return None
    path = os.path.join(output_dir, "tokenizer.json")
    backend.save(path)
    return path


def _looks_like_diffusion(pkg: Any) -> bool:
    try:
        names = set(pkg.keys())
    except AttributeError:
        return False
    return any(k in names for k in _DENOISER_KEYS) or any(
        k in names for k in ("vae", "vae_decoder", "vae_encoder")
    )


def _looks_like_multimodal(pkg: Any) -> bool:
    try:
        names = set(pkg.keys())
    except AttributeError:
        return False
    # Require the `embedding` fusion component: the multimodal metadata wires
    # encoder(s) -> embedding -> decoder, so without it we would emit metadata
    # referencing a non-existent embedding model.
    return (
        "decoder" in names
        and "embedding" in names
        and bool(names & {"vision_encoder", "audio_encoder"})
    )


def _looks_like_speech_to_text(pkg: Any) -> bool:
    """Detect a cross-attention encoder-decoder ASR package (e.g. Whisper).

    The signal is structural rather than name-based: an ``encoder`` and a
    ``decoder`` component where the decoder consumes ``encoder_hidden_states``
    (cross-attention). This separates Whisper-style ASR from a codec, whose
    ``encoder``/``decoder`` are pure single-pass and share no such input.
    """
    try:
        names = set(pkg.keys())
    except AttributeError:
        return False
    if not {"encoder", "decoder"} <= names:
        return False
    decoder = pkg["decoder"]
    try:
        decoder_inputs = {value.name for value in decoder.graph.inputs}
    except AttributeError:
        return False
    return "encoder_hidden_states" in decoder_inputs


def _looks_like_audio_codec(pkg: Any) -> bool:
    """Detect an audio-to-audio neural codec package.

    The signal is structural: an ``encoder`` that outputs ``codes`` feeding a
    ``decoder`` that consumes ``codes`` via a pure single-pass path (no
    ``encoder_hidden_states`` cross-attention, no autoregressive text decode).
    This separates a codec from Whisper-style ASR and from an image VAE (which
    exchanges ``latent``/``sample`` rather than ``codes``).
    """
    try:
        names = set(pkg.keys())
    except AttributeError:
        return False
    if not {"encoder", "decoder"} <= names:
        return False
    try:
        encoder_outputs = {value.name for value in pkg["encoder"].graph.outputs}
        decoder_inputs = {value.name for value in pkg["decoder"].graph.inputs}
    except AttributeError:
        return False
    return (
        "codes" in encoder_outputs
        and "codes" in decoder_inputs
        and "encoder_hidden_states" not in decoder_inputs
    )


def _audio_codec_codes_dtype(pkg: Any) -> str:
    """Return the metadata dtype of the codec ``codes`` tensor (default int64)."""
    # ONNX elem-type names -> onnx-genai metadata dtype tags. Float codes keep
    # their precision (fp16/bf16/fp32) so the runtime binds the right buffer type.
    float_dtypes = {"FLOAT": "fp32", "FLOAT16": "fp16", "BFLOAT16": "bf16"}
    try:
        for value in pkg["decoder"].graph.inputs:
            if value.name == "codes" and value.dtype is not None:
                return float_dtypes.get(value.dtype.name, "int64")
    except (AttributeError, KeyError):
        # Missing/partial codec structure: fall back to the documented default.
        return "int64"
    return "int64"


def _looks_like_multi_decoder_tts(pkg: Any) -> bool:
    """Detect a nested multi-decoder TTS package (e.g. Qwen3-TTS).

    The defining signal is a ``talker`` plus a ``code_predictor`` decoder — a
    dual, nested autoregressive shape (the code_predictor expands each talker
    frame's residual codebooks). When the package also carries the
    ``talker_step_embedder`` pre-embedder (see :func:`_has_tts_pre_embedder`),
    the dispatcher emits a runnable ``pre_embedder``-driven
    ``nested_autoregressive`` contract; without it the component graph is not yet
    mappable, so detection triggers a precise, actionable error rather than
    mis-emitting (see DESIGN.md §20.3).
    """
    try:
        names = set(pkg.keys())
    except AttributeError:
        return False
    return {"talker", "code_predictor"} <= names


def _has_tts_pre_embedder(pkg: Any) -> bool:
    """True when a multi-decoder TTS package carries the pre-embedder component.

    The ``talker_step_embedder`` materializes the talker's per-step
    ``inputs_embeds`` (``frame_codes [+ text_embed] -> inputs_embeds``); its
    presence is what makes the package emittable to the ``pre_embedder``-driven
    ``nested_autoregressive`` contract.
    """
    try:
        names = set(pkg.keys())
    except AttributeError:
        return False
    return "talker_step_embedder" in names


def _tts_component_kwargs(pkg: Any, config: Any) -> dict[str, Any]:
    """Derive pre-embedder-driven TTS metadata kwargs from a package + config.

    Mobius saves each component into ``<component>/model.onnx``. ``num_code_groups``
    comes from the TTS config (the RVQ residual count per frame).
    """
    tts = getattr(config, "tts", None)
    num_code_groups = getattr(tts, "num_code_groups", None) if tts is not None else None
    if not num_code_groups:
        raise ValueError(
            "TTS metadata requires config.tts.num_code_groups (RVQ codes per frame)"
        )
    kwargs: dict[str, Any] = {
        "num_code_groups": num_code_groups,
        "talker_filename": "talker/model.onnx",
        "code_predictor_filename": "code_predictor/model.onnx",
        "pre_embedder_filename": "talker_step_embedder/model.onnx",
    }
    # Emit the prefill/trailing-text component only when the package carries it;
    # otherwise the prefill-less shape (talker frame 0 + zero text_embed) is used.
    try:
        names = set(pkg.keys())
    except (AttributeError, TypeError):
        names = set()
    kwargs["prefill_embedder_filename"] = (
        "talker_prefill_embedder/model.onnx" if "talker_prefill_embedder" in names else None
    )
    kwargs["activation_dtype"] = _activation_dtype_tag(config)
    return kwargs


def _activation_dtype_tag(config: Any) -> str:
    """Map a model config's activation dtype to the metadata dtype tag.

    The composite dataflow edges (inputs_embeds, encoder_hidden_states, …) carry
    the model's activation dtype, so metadata must reflect it (fp16/bf16 builds
    would otherwise be mislabeled fp32).
    """
    dtype = getattr(config, "dtype", None)
    name = getattr(dtype, "name", "") or ""
    return {"FLOAT16": "fp16", "BFLOAT16": "bf16"}.get(name.upper(), "fp32")


def _multimodal_component_kwargs(pkg: Any) -> dict[str, str]:
    """Derive multimodal component filenames from a package's component keys."""
    try:
        names = set(pkg.keys())
    except AttributeError:
        return {}

    derived = {
        "decoder_filename": "decoder/model.onnx",
        "embedding_filename": "embedding/model.onnx",
    }
    if "vision_encoder" in names:
        derived["vision_encoder_filename"] = "vision_encoder/model.onnx"
    if "audio_encoder" in names:
        derived["audio_encoder_filename"] = "audio_encoder/model.onnx"
    return derived


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

    This is the single entry point that inspects a built ``ModelPackage`` and
    emits the onnx-genai runtime contract for it. Package shapes are matched in
    order of decreasing specificity; the first match wins:

    ===================== ============================================ =================================
    Pipeline shape        Structural signal (detector)                 Emitted ``strategy``
    ===================== ============================================ =================================
    Diffusion             denoiser / VAE present                       ``iterative``
    Audio codec           encoder→``codes``→decoder, no cross-attn     ``composite`` (two single_pass)
    Multimodal VLM        decoder + vision/audio encoder + fusion      ``composite`` (encoders→fuse→AR)
    Speech-to-text (ASR)  decoder consumes ``encoder_hidden_states``   ``composite`` (encode→AR)
    Decoder LM            fallback (a config is required)              bare decoder (``kv_cache`` + attn)
    ===================== ============================================ =================================

    Detection is **structural** (graph input/output names + component roles),
    not name-based, so a codec is never mistaken for ASR and vice versa. To add a
    modality, add a ``_looks_like_X`` structural detector plus a
    ``build_X_pipeline_metadata`` builder and insert a dispatch branch ordered by
    specificity. Tensor-only pipelines (diffusion, codec) are matched before the
    ``config`` requirement because they carry no autoregressive decoder.

    For decoder and multimodal packages, ``config`` (or ``pkg.config``) supplies
    the decoder attention dimensions. Diffusion component filenames are taken
    from ``kwargs`` or discovered from the package; ``num_inference_steps`` /
    ``scheduler`` / ``guidance_scale`` set the loop.
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
        artifacts = {"inference_metadata": path}
        # Emit the CLIP tokenizer.json for text-conditioned pipelines so the
        # onnx-genai runners can tokenize prompts from the package alone.
        if "text_encoder_filename" in kwargs:
            tokenizer_path = _write_clip_tokenizer(output_dir, source)
            if tokenizer_path is not None:
                artifacts["tokenizer"] = tokenizer_path
        return artifacts

    if _looks_like_audio_codec(pkg):
        # A neural codec produces tensors (waveform), not tokens, so it needs no
        # decoder config — emit before the config requirement below.
        path = write_audio_codec_pipeline_metadata(
            output_dir, codes_dtype=_audio_codec_codes_dtype(pkg)
        )
        return {"inference_metadata": path}

    resolved_config = config if config is not None else getattr(pkg, "config", None)
    if resolved_config is None:
        raise ValueError(
            "onnx-genai decoder metadata requires a model config (pass config=... "
            "or a package carrying `.config`)"
        )
    if _looks_like_multimodal(pkg):
        derived = _multimodal_component_kwargs(pkg)
        for name, value in derived.items():
            kwargs.setdefault(name, value)
        decoder_metadata = decoder_metadata_from_config(
            resolved_config, kv_native_dtype=kv_native_dtype
        )
        path = write_multimodal_pipeline_metadata(
            output_dir,
            decoder_metadata=decoder_metadata,
            activation_dtype=_activation_dtype_tag(resolved_config),
            **kwargs,
        )
        _add_explicit_io_to_file(path, pkg, resolved_config)
        artifacts = {"inference_metadata": path}
        tokenizer_path = _write_hf_tokenizer(output_dir, source)
        if tokenizer_path is not None:
            artifacts["tokenizer"] = tokenizer_path
        return artifacts

    if _looks_like_speech_to_text(pkg):
        decoder_metadata = decoder_metadata_from_config(
            resolved_config, kv_native_dtype=kv_native_dtype
        )
        path = write_speech_to_text_pipeline_metadata(
            output_dir,
            decoder_metadata=decoder_metadata,
            activation_dtype=_activation_dtype_tag(resolved_config),
            **kwargs,
        )
        _add_explicit_io_to_file(path, pkg, resolved_config)
        artifacts = {"inference_metadata": path}
        tokenizer_path = _write_hf_tokenizer(output_dir, source)
        if tokenizer_path is not None:
            artifacts["tokenizer"] = tokenizer_path
        return artifacts

    # A nested multi-decoder TTS stack (talker + code_predictor) uses the
    # nested_autoregressive strategy. When the package also carries the
    # `talker_step_embedder` pre-embedder (the real Qwen3-TTS shape), emit the
    # pre-embedder-driven contract the onnx-genai runtime executes; otherwise the
    # component graph is not yet mappable, so fail with a precise, actionable error.
    if _looks_like_multi_decoder_tts(pkg):
        if not _has_tts_pre_embedder(pkg):
            raise NotImplementedError(
                "Multi-decoder TTS packages (talker + code_predictor, e.g. Qwen3-TTS) "
                "use the nested_autoregressive strategy. This package lacks the "
                "`talker_step_embedder` pre-embedder that materializes the talker "
                "inputs_embeds, so it cannot yet be mapped to the runtime contract — "
                "see onnx-genai docs/DESIGN.md §20.3 'Multi-decoder TTS'."
            )
        decoder_metadata = decoder_metadata_from_config(
            resolved_config, kv_native_dtype=kv_native_dtype
        )
        path = write_tts_pipeline_metadata(
            output_dir,
            decoder_metadata=decoder_metadata,
            **_tts_component_kwargs(pkg, resolved_config),
        )
        _add_explicit_io_to_file(path, pkg, resolved_config)
        artifacts = {"inference_metadata": path}
        tokenizer_path = _write_hf_tokenizer(output_dir, source)
        if tokenizer_path is not None:
            artifacts["tokenizer"] = tokenizer_path
        return artifacts

    # Fallback: a single-component decoder language model. A multi-component
    # package that matched none of the composite shapes above would be silently
    # mis-emitted as a bare decoder — fail loudly instead so an unsupported shape
    # is obvious rather than producing wrong metadata.
    try:
        component_names = sorted(pkg.keys())
    except (AttributeError, TypeError):
        component_names = []
    if len(component_names) > 1:
        raise ValueError(
            "onnx-genai config emission does not recognize this multi-component "
            f"package shape (components: {component_names}). Supported composite "
            "shapes: diffusion, audio codec, multimodal VLM, speech-to-text. "
            "Multi-decoder pipelines such as TTS require a dedicated emitter."
        )

    path = write_decoder_metadata(
        output_dir, config=resolved_config, kv_native_dtype=kv_native_dtype
    )
    _add_explicit_io_to_file(path, pkg, resolved_config)
    artifacts = {"inference_metadata": path}
    tokenizer_path = _write_hf_tokenizer(output_dir, source)
    if tokenizer_path is not None:
        artifacts["tokenizer"] = tokenizer_path
    return artifacts
