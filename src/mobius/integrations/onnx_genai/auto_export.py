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

import numpy as np
import onnx_ir as ir
import yaml

from mobius.integrations.onnx_genai.inference_metadata import (
    _TEXT_RUNTIME_ASSET_NAMES,
    SchedulerConfig,
    _copy_runtime_assets,
    add_adapter_service_to_metadata,
    add_explicit_package_io,
    add_policy_components_to_workflow,
    load_diffusers_scheduler_config,
)
from mobius.integrations.onnx_genai.workflow_metadata import (
    write_audio_codec_workflow_metadata,
    write_decoder_workflow_metadata,
    write_diffusion_workflow_metadata,
    write_language_diffusion_workflow_metadata,
    write_speculative_workflow_metadata,
    write_speech_to_text_workflow_metadata,
    write_tts_workflow_metadata,
    write_vlm_workflow_metadata,
)

_LOGGER = logging.getLogger(__name__)


def _euler_schedule(
    scheduler: SchedulerConfig, num_inference_steps: int
) -> tuple[list[float], list[float]]:
    """Materialize diffusers-compatible Euler timesteps and sigma values."""
    if scheduler.kind != "euler" or scheduler.prediction_type != "epsilon":
        raise ValueError(
            "workflow diffusion currently supports deterministic Euler epsilon "
            f"schedulers, got kind={scheduler.kind!r}, "
            f"prediction_type={scheduler.prediction_type!r}"
        )
    if scheduler.use_karras_sigmas or scheduler.use_exponential_sigmas:
        raise ValueError(
            "workflow diffusion does not yet materialize Karras or exponential sigmas"
        )
    if scheduler.beta_schedule == "scaled_linear":
        betas = (
            np.linspace(
                np.sqrt(scheduler.beta_start),
                np.sqrt(scheduler.beta_end),
                scheduler.num_train_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif scheduler.beta_schedule == "linear":
        betas = np.linspace(
            scheduler.beta_start,
            scheduler.beta_end,
            scheduler.num_train_timesteps,
            dtype=np.float64,
        )
    else:
        raise ValueError(
            f"workflow diffusion does not support beta schedule {scheduler.beta_schedule!r}"
        )
    training_sigmas = np.sqrt((1.0 - np.cumprod(1.0 - betas)) / np.cumprod(1.0 - betas))
    timesteps = np.linspace(
        scheduler.num_train_timesteps - 1,
        0,
        num_inference_steps,
        dtype=np.float64,
    )
    sigmas = np.interp(
        timesteps,
        np.arange(scheduler.num_train_timesteps, dtype=np.float64),
        training_sigmas,
    )
    return timesteps.tolist(), [*sigmas.tolist(), 0.0]


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
    add_adapter_service_to_metadata(metadata, pkg, os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)


def _write_clip_tokenizer(
    output_dir: str,
    source: str | None,
    *,
    revision: str | None = None,
) -> str | None:
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
        tokenizer = AutoTokenizer.from_pretrained(
            source,
            subfolder="tokenizer",
            use_fast=True,
            revision=revision,
        )
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


def _write_text_runtime_assets(output_dir: str, source: str | None) -> dict[str, str]:
    """Emit the tokenizer *and* chat-template assets a text package needs.

    ``tokenizer.json`` alone is not enough for an instruction-tuned decoder: the
    runtime applies the package's chat template to build the prompt, and without
    it the raw user text (no leading BOS, no turn markers) reaches the model.
    Gemma 4 answers such a prompt with unbounded repetition, so shipping the
    template is a correctness requirement rather than a convenience.

    Args:
        output_dir: Package directory to write the assets into.
        source: Hugging Face model id or local directory holding them.

    Returns:
        A mapping of asset stem to written path for every asset materialized.
    """
    artifacts = _copy_runtime_assets(output_dir, source, _TEXT_RUNTIME_ASSET_NAMES)
    if "tokenizer" not in artifacts:
        fallback = _write_hf_tokenizer(output_dir, source)
        if fallback is not None:
            artifacts["tokenizer"] = fallback
    return artifacts


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
        tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True, revision=revision)
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


def _write_hf_audio_processor(
    output_dir: str,
    source: str | None,
    *,
    revision: str | None = None,
) -> str | None:
    """Emit the Hugging Face audio feature-extractor contract for ASR packages."""
    if not source:
        return None
    try:
        from transformers import AutoFeatureExtractor
    except ImportError:
        _LOGGER.warning(
            "transformers is not available; skipping audio_processor.json emission."
        )
        return None
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(source, revision=revision)
    except Exception as error:
        _LOGGER.warning(
            "Could not load an audio processor from %r: %s; "
            "skipping audio_processor.json emission.",
            source,
            error,
        )
        return None
    path = os.path.join(output_dir, "audio_processor.json")
    feature_extractor.to_json_file(path)
    return path


def _audio_preprocessing_program(
    processor_path: str | None, encoder: Any
) -> dict[str, Any] | None:
    """Derive a declarative log-mel program from a HF feature-extractor config.

    The program is the executable contract the runtime audio adapter follows:
    decode -> resample -> pad/trim to the fixed window -> log-mel -> normalize.
    Its single output binds to the encoder's rank-3 feature input.
    """
    if processor_path is None:
        return None
    import json

    with open(processor_path, encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("feature_extractor_type") != "WhisperFeatureExtractor":
        _LOGGER.warning(
            "Audio feature extractor %r is not a log-mel window extractor; "
            "skipping declarative audio preprocessing.",
            config.get("feature_extractor_type"),
        )
        return None
    feature_inputs = [
        value
        for value in encoder.graph.inputs
        if value.shape is not None and len(value.shape) == 3
    ]
    if len(feature_inputs) != 1:
        raise ValueError(
            "audio preprocessing requires exactly one rank-3 encoder feature input, "
            f"got {[value.name for value in feature_inputs]}"
        )
    sampling_rate = int(config["sampling_rate"])
    num_mel_bins = int(config["feature_size"])
    n_fft = int(config["n_fft"])
    hop_length = int(config["hop_length"])
    n_samples = int(config.get("n_samples", config["chunk_length"] * sampling_rate))
    return {
        "transforms": [
            {"op": "decode", "outputs": ["samples"]},
            {
                "op": "resample",
                "inputs": ["samples"],
                "outputs": ["resampled"],
                "sampling_rate": sampling_rate,
            },
            {
                "op": "pad",
                "inputs": ["resampled"],
                "outputs": ["windowed"],
                "mode": "fixed_window",
                "target_samples": n_samples,
                "pad_value": float(config.get("padding_value", 0.0)),
            },
            {
                "op": "log_mel",
                "inputs": ["windowed"],
                "outputs": ["mel"],
                "num_mel_bins": num_mel_bins,
                "n_fft": n_fft,
                "hop_length": hop_length,
                "window": "hann",
                "mel_scale": "slaney",
                "sampling_rate": sampling_rate,
            },
            {
                "op": "normalize",
                "inputs": ["mel"],
                "outputs": ["features"],
                "mode": "whisper_log_mel",
            },
        ],
        "outputs": [
            {
                "source": "features",
                "name": feature_inputs[0].name,
                "content": "audio_features",
            }
        ],
    }


def _looks_like_diffusion(pkg: Any) -> bool:
    try:
        names = set(pkg.keys())
    except AttributeError:
        return False
    return any(k in names for k in _DENOISER_KEYS) or any(
        k in names for k in ("vae", "vae_decoder", "vae_encoder")
    )


def _looks_like_language_diffusion(pkg: Any) -> bool:
    """Detect a full-sequence token denoiser with an executable proposal output."""
    try:
        if len(pkg) != 1:
            return False
        model = next(iter(pkg.values()))
        inputs = list(model.graph.inputs)
        outputs = list(model.graph.outputs)
    except (AttributeError, TypeError):
        return False
    return (
        len(inputs) == 1
        and inputs[0].dtype in {ir.DataType.INT32, ir.DataType.INT64}
        and inputs[0].shape is not None
        and len(inputs[0].shape) == 2
        and {"logits", "proposed_tokens"} <= {value.name for value in outputs}
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


def _looks_like_multi_decoder_tts(pkg: Any) -> bool:
    """Detect a nested multi-decoder TTS package (e.g. Qwen3-TTS)."""
    try:
        names = set(pkg.keys())
    except AttributeError:
        return False
    return {"talker", "code_predictor"} <= names


def _looks_like_speculative(pkg: Any) -> bool:
    try:
        return {"proposer", "verifier"} <= set(pkg.keys())
    except AttributeError:
        return False


def _has_tts_pre_embedder(pkg: Any) -> bool:
    """True when a multi-decoder TTS package carries the pre-embedder component.

    The ``talker_step_embedder`` materializes the talker's per-step
    ``inputs_embeds`` (``frame_codes [+ text_embed] -> inputs_embeds``). It is
    necessary, but not sufficient until generic loops expose induction SSA.
    """
    try:
        names = set(pkg.keys())
    except AttributeError:
        return False
    return "talker_step_embedder" in names


def _activation_dtype_tag(config: Any) -> str:
    """Map a model config's activation dtype to the metadata dtype tag."""
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
    single_component = len(names) == 1

    def filename(key: str) -> str:
        return "model.onnx" if single_component else f"{key}/model.onnx"

    for key in _DENOISER_KEYS:
        if key in names:
            derived["denoiser_filename"] = filename(key)
            break
    if "text_encoder" in names:
        derived["text_encoder_filename"] = filename("text_encoder")
    if "vae_decoder" in names:
        derived["vae_filename"] = filename("vae_decoder")
        derived["vae_latent_input"] = "latent_sample"
    elif "vae" in names:
        derived["vae_filename"] = filename("vae")
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
    grammar_guidance: bool = False,
    adaptive_k_max: int | None = None,
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
    Audio codec           encoder→``codes``→decoder, no cross-attn     typed SSA workflow
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
    if _looks_like_language_diffusion(pkg):
        path = write_language_diffusion_workflow_metadata(
            pkg,
            output_dir,
            num_inference_steps=num_inference_steps,
        )
        artifacts = {"inference_metadata": path}
        artifacts.update(_write_text_runtime_assets(output_dir, source))
        return artifacts

    if _looks_like_diffusion(pkg):
        is_qwen_image_edit = getattr(getattr(pkg, "config", None), "model_type", None) == (
            "qwen_image_edit"
        )
        if is_qwen_image_edit:
            raise ValueError(
                "onnx-genai cannot execute Qwen Image Edit packages: the runtime "
                "does not support source-latent packing, target/source token "
                "concatenation, target-only denoiser outputs, or the required "
                "Qwen true-CFG path. Export the ONNX components without "
                "--runtime onnx-genai and orchestrate the pipeline directly."
            )
        if scheduler is None:
            scheduler = load_diffusers_scheduler_config(source, revision=revision)
        # Fill in component filenames from the package layout, letting any
        # caller-supplied values win.
        derived = _diffusion_component_kwargs(pkg)
        for name, value in derived.items():
            kwargs.setdefault(name, value)
        if "text_encoder_filename" in kwargs and guidance_scale is None:
            raise ValueError(
                "text-conditioned workflow diffusion does not implement "
                "classifier-free guidance; pass guidance_scale=1.0 explicitly "
                "to request unguided generation"
            )
        if guidance_scale is not None and not np.isclose(guidance_scale, 1.0):
            raise ValueError(
                "workflow diffusion requires an explicit classifier-free guidance "
                "component before guidance_scale can differ from 1.0"
            )
        resolved_scheduler = scheduler or SchedulerConfig(kind="euler")
        timesteps, sigma_schedule = _euler_schedule(resolved_scheduler, num_inference_steps)
        path = write_diffusion_workflow_metadata(
            pkg,
            output_dir,
            num_inference_steps=num_inference_steps,
            schedule=sigma_schedule,
            timesteps=timesteps,
        )
        artifacts = {"inference_metadata": path}
        # Emit the CLIP tokenizer.json for text-conditioned pipelines so the
        # onnx-genai runners can tokenize prompts from the package alone.
        if "text_encoder_filename" in kwargs:
            tokenizer_path = _write_clip_tokenizer(output_dir, source, revision=revision)
            if tokenizer_path is not None:
                artifacts["tokenizer"] = tokenizer_path
        return artifacts

    if _looks_like_audio_codec(pkg):
        # A neural codec produces tensors (waveform), not tokens, so it needs no
        # decoder config — emit before the config requirement below.
        path = write_audio_codec_workflow_metadata(pkg, output_dir)
        return {"inference_metadata": path}

    if _looks_like_speculative(pkg):
        if kv_native_dtype is not None:
            raise ValueError(
                "workflow speculative export derives KV state dtype from ONNX ports; "
                "kv_native_dtype overrides are unsupported"
            )
        path = write_speculative_workflow_metadata(
            pkg,
            output_dir,
            grammar_guidance=grammar_guidance,
            adaptive_k_max=adaptive_k_max,
        )
        return {"inference_metadata": path}

    resolved_config = config if config is not None else getattr(pkg, "config", None)
    if resolved_config is None:
        raise ValueError(
            "onnx-genai decoder metadata requires a model config (pass config=... "
            "or a package carrying `.config`)"
        )
    if _looks_like_multimodal(pkg):
        if kv_native_dtype is not None:
            raise ValueError(
                "workflow VLM export derives KV state dtype from ONNX ports; "
                "kv_native_dtype overrides are unsupported"
            )
        path = write_vlm_workflow_metadata(
            pkg,
            output_dir,
            resolved_config,
            source=source,
        )
        artifacts = {"inference_metadata": path}
        # A multimodal package needs the processor assets as well as the
        # tokenizer, because the runtime resolves image/audio preprocessing
        # parameters from them.
        artifacts.update(_copy_runtime_assets(output_dir, source))
        if "tokenizer" not in artifacts:
            tokenizer_path = _write_hf_tokenizer(output_dir, source)
            if tokenizer_path is not None:
                artifacts["tokenizer"] = tokenizer_path
        return artifacts

    if _looks_like_speech_to_text(pkg):
        if kv_native_dtype is not None:
            raise ValueError(
                "workflow speech-to-text export derives KV state dtype from ONNX ports; "
                "kv_native_dtype overrides are unsupported"
            )
        audio_processor_path = _write_hf_audio_processor(output_dir, source)
        path = write_speech_to_text_workflow_metadata(
            pkg,
            output_dir,
            resolved_config,
            audio_preprocessing=_audio_preprocessing_program(
                audio_processor_path, pkg["encoder"]
            ),
        )
        artifacts = {"inference_metadata": path}
        # An ASR decoder is still a text producer: ship its tokenizer and chat
        # template alongside the audio processor.
        artifacts.update(_write_text_runtime_assets(output_dir, source))
        if audio_processor_path is not None:
            artifacts["audio_processor"] = audio_processor_path
        return artifacts

    # A nested multi-decoder TTS stack requires the generic workflow loop to expose
    # its induction value. The current producer contract cannot wire step_index or
    # per-group embedding selection without host preprocessing, so the workflow
    # writer reports that exact contract defect.
    if _looks_like_multi_decoder_tts(pkg):
        if kv_native_dtype is not None:
            raise ValueError(
                "workflow TTS export derives KV state dtype from ONNX ports; "
                "kv_native_dtype overrides are unsupported"
            )
        if not _has_tts_pre_embedder(pkg):
            raise NotImplementedError(
                "Multi-decoder TTS packages (talker + code_predictor, e.g. Qwen3-TTS) "
                "require nested generic workflow loops. This package lacks the "
                "`talker_step_embedder` pre-embedder that materializes the talker "
                "inputs_embeds, so it cannot be mapped to the workflow contract."
            )
        path = write_tts_workflow_metadata(pkg, output_dir, resolved_config)
        return {"inference_metadata": path}

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

    if kv_native_dtype is not None:
        raise ValueError(
            "workflow decoder export derives KV state dtype from ONNX ports; "
            "kv_native_dtype overrides are unsupported"
        )
    path = write_decoder_workflow_metadata(
        pkg,
        output_dir,
        resolved_config,
        sampler=str(getattr(resolved_config, "workflow_sampler", "greedy")),
    )
    artifacts = {"inference_metadata": path}
    artifacts.update(_write_text_runtime_assets(output_dir, source))
    return artifacts
