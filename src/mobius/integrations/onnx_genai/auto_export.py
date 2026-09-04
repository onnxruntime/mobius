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

import json
import logging
import os
from typing import Any

import numpy as np
import onnx_ir as ir

from mobius.integrations.onnx_genai.inference_metadata import (
    _TEXT_RUNTIME_ASSET_NAMES,
    SchedulerConfig,
    _copy_runtime_assets,
    load_diffusers_scheduler_config,
    load_diffusers_vae_scaling_factor,
)
from mobius.integrations.onnx_genai.shared_state_flow_metadata import (
    is_shared_state_pixel_flow_package,
    write_shared_state_pixel_flow_workflow_metadata,
)
from mobius.integrations.onnx_genai.workflow_metadata import (
    HierarchicalAudioWorkflowConfig,
    _validate_reuse_rate_selection,
    write_audio_codec_workflow_metadata,
    write_ctc_asr_workflow_metadata,
    write_decoder_workflow_metadata,
    write_diffusion_workflow_metadata,
    write_encoder_embedding_workflow_metadata,
    write_hierarchical_audio_workflow_metadata,
    write_image_edit_workflow_metadata,
    write_language_diffusion_workflow_metadata,
    write_speculative_workflow_metadata,
    write_speech_enhancement_workflow_metadata,
    write_speech_to_text_workflow_metadata,
    write_tts_workflow_metadata,
    write_video_diffusion_workflow_metadata,
    write_vlm_workflow_metadata,
)
from mobius.models.reuse import ReUseConfig

_LOGGER = logging.getLogger(__name__)


#: Scheduler kind -> (workflow solver component, whether the sampler rescales
#: the denoiser's input). A sampler that keeps its state variance-preserving
#: feeds the raw state to the denoiser and starts from unit-variance noise; a
#: sampler that carries state in sigma space divides by ``sqrt(sigma**2 + 1)``
#: and starts from ``sigma_max`` scaled noise.
_DIFFUSION_SOLVERS: dict[str, tuple[str, bool]] = {
    "euler": ("euler", True),
    "dpmpp_2m": ("multistep", False),
}


def _diffusion_schedule(
    scheduler: SchedulerConfig, num_inference_steps: int
) -> tuple[list[float], list[float]]:
    """Materialize diffusers-compatible timesteps and sigma values."""
    if scheduler.kind not in _DIFFUSION_SOLVERS or scheduler.prediction_type != "epsilon":
        raise ValueError(
            "workflow diffusion currently supports deterministic epsilon schedulers "
            f"{sorted(_DIFFUSION_SOLVERS)}, got kind={scheduler.kind!r}, "
            f"prediction_type={scheduler.prediction_type!r}"
        )
    if scheduler.kind == "dpmpp_2m" and (
        scheduler.algorithm_type != "dpmsolver++"
        or scheduler.solver_order != 2
        or scheduler.solver_type != "midpoint"
        or not scheduler.lower_order_final
        or scheduler.final_sigmas_type != "zero"
    ):
        raise ValueError(
            "the workflow multistep solver implements second-order dpmsolver++ with "
            "midpoint updates, a lower-order final step, and a zero terminal sigma; "
            f"got algorithm_type={scheduler.algorithm_type!r}, "
            f"solver_order={scheduler.solver_order}, "
            f"solver_type={scheduler.solver_type!r}, "
            f"lower_order_final={scheduler.lower_order_final}, "
            f"final_sigmas_type={scheduler.final_sigmas_type!r}"
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
    if scheduler.kind == "dpmpp_2m":
        # Multistep solvers place the boundary at the terminal sigma, so the
        # linspace spans one extra point and drops the trailing zero timestep.
        timesteps = (
            np.linspace(0, scheduler.num_train_timesteps - 1, num_inference_steps + 1)
            .round()[::-1][:-1]
            .astype(np.float64)
        )
    else:
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


def _ddim_alpha_schedule(
    scheduler: SchedulerConfig, num_inference_steps: int
) -> tuple[list[float], list[float]]:
    """Materialize DDIM timesteps and cumulative alphas from diffusers config."""
    if scheduler.kind != "ddim":
        raise ValueError(f"video workflow requires a DDIM scheduler, got {scheduler.kind!r}")
    if num_inference_steps > scheduler.num_train_timesteps:
        raise ValueError("num_inference_steps exceeds the DDIM training schedule")
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
        raise ValueError(f"unsupported DDIM beta schedule {scheduler.beta_schedule!r}")
    alphas_cumprod = np.cumprod(1.0 - betas)
    alphas_cumprod = alphas_cumprod / (
        scheduler.snr_shift_scale + (1.0 - scheduler.snr_shift_scale) * alphas_cumprod
    )
    if scheduler.rescale_betas_zero_snr:
        alpha_sqrt = np.sqrt(alphas_cumprod)
        first, last = alpha_sqrt[0], alpha_sqrt[-1]
        alpha_sqrt = (alpha_sqrt - last) * first / (first - last)
        alphas_cumprod = alpha_sqrt**2
    if scheduler.timestep_spacing == "linspace":
        timesteps = np.linspace(
            0, scheduler.num_train_timesteps - 1, num_inference_steps
        ).round()[::-1]
    elif scheduler.timestep_spacing == "leading":
        step_ratio = scheduler.num_train_timesteps // num_inference_steps
        timesteps = (np.arange(num_inference_steps) * step_ratio).round()[::-1]
        timesteps += scheduler.steps_offset
    elif scheduler.timestep_spacing == "trailing":
        step_ratio = scheduler.num_train_timesteps / num_inference_steps
        timesteps = np.round(np.arange(scheduler.num_train_timesteps, 0, -step_ratio)) - 1
    else:
        raise ValueError(f"unsupported DDIM timestep spacing {scheduler.timestep_spacing!r}")
    timesteps = timesteps.astype(np.int64)
    final_alpha = 1.0 if scheduler.set_alpha_to_one else float(alphas_cumprod[0])
    schedule = [*(float(alphas_cumprod[index]) for index in timesteps), final_alpha]
    return timesteps.astype(np.float64).tolist(), schedule


_DENOISER_KEYS = ("denoiser", "transformer", "unet")


def _flow_match_euler_schedule(
    scheduler: SchedulerConfig, num_inference_steps: int, image_seq_len: int
) -> tuple[list[float], list[float]]:
    """Materialize diffusers ``FlowMatchEulerDiscreteScheduler`` timesteps/sigmas.

    Reproduces ``set_timesteps(sigmas=linspace(1, 1/n, n), mu=calculate_shift(...))``
    including resolution-dependent dynamic shifting and terminal stretching, so
    the baked schedule matches the pipeline that produced the reference image.
    Timesteps are emitted as sigmas (``t / num_train_timesteps``) because the
    Qwen Image denoiser consumes the normalized timestep directly.
    """
    if scheduler.kind != "flow_match_euler":
        raise ValueError(
            f"image-edit workflow requires a flow-match Euler scheduler, got {scheduler.kind!r}"
        )
    sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps, dtype=np.float64)
    if scheduler.use_dynamic_shifting:
        base_seq_len = scheduler.base_image_seq_len or 256
        max_seq_len = scheduler.max_image_seq_len or 4096
        base_shift = scheduler.base_shift if scheduler.base_shift is not None else 0.5
        max_shift = scheduler.max_shift if scheduler.max_shift is not None else 1.15
        slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        mu = image_seq_len * slope + (base_shift - slope * base_seq_len)
        if (scheduler.time_shift_type or "exponential") != "exponential":
            raise ValueError(
                f"unsupported flow-match time shift {scheduler.time_shift_type!r}"
            )
        sigmas = np.exp(mu) / (np.exp(mu) + (1.0 / sigmas - 1.0))
    elif scheduler.shift is not None:
        sigmas = scheduler.shift * sigmas / (1.0 + (scheduler.shift - 1.0) * sigmas)
    if scheduler.shift_terminal is not None:
        # stretch_shift_to_terminal: map the last sigma onto shift_terminal.
        one_minus = 1.0 - sigmas
        sigmas = 1.0 - one_minus / (one_minus[-1] / (1.0 - scheduler.shift_terminal))
    return sigmas.tolist(), [*sigmas.tolist(), 0.0]


def _looks_like_image_edit(pkg: Any) -> bool:
    """Detect a source-image-conditioned flow-matching editing pipeline.

    Structural signals: a VAE encoder and decoder pair, plus a denoiser that
    takes rank-3 packed latents and exposes a ``target_sequence_length`` port —
    i.e. it consumes concatenated target+source tokens and slices its estimate
    back to the target block.
    """
    try:
        names = set(pkg.keys())
    except AttributeError:
        return False
    if not {"vae_encoder", "vae_decoder"} <= names:
        return False
    denoiser_name = next((key for key in _DENOISER_KEYS if key in names), None)
    if denoiser_name is None:
        return False
    inputs = {value.name: value for value in pkg[denoiser_name].graph.inputs}
    sample = inputs.get("sample")
    return (
        "target_sequence_length" in inputs
        and sample is not None
        and sample.shape is not None
        and len(sample.shape) == 3
    )


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


def _write_text_runtime_assets(
    output_dir: str,
    source: str | None,
    *,
    revision: str | None = None,
) -> dict[str, str]:
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
    artifacts = _copy_runtime_assets(
        output_dir, source, _TEXT_RUNTIME_ASSET_NAMES, revision=revision
    )
    if "tokenizer" not in artifacts:
        fallback = _write_hf_tokenizer(output_dir, source, revision=revision)
        if fallback is not None:
            artifacts["tokenizer"] = fallback
    return artifacts


def _write_hf_tokenizer(
    output_dir: str,
    source: str | None,
    *,
    revision: str | None = None,
) -> str | None:
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


def _looks_like_hierarchical_audio_generation(pkg: Any) -> bool:
    """Detect the global-frame/local-codebook/flow/vocoder package topology.

    This check intentionally precedes ordinary diffusion detection. A package
    with a flow transformer is not an executable diffusion pipeline when its
    conditioning must first be generated by nested autoregressive loops.

    A package is treated as hierarchical audio either because it carries a typed
    :class:`HierarchicalAudioWorkflowConfig`, or because the builder recognized
    the topology structurally (``workflow_kind == "hierarchical_audio"``) but
    could not resolve a workflow config. The latter is routed here so metadata
    emission fails closed with a targeted instruction rather than being
    misclassified as diffusion.
    """
    package_config = getattr(pkg, "config", None)
    if isinstance(
        getattr(package_config, "workflow_config", None), HierarchicalAudioWorkflowConfig
    ):
        return True
    return getattr(package_config, "workflow_kind", None) == "hierarchical_audio"


def _looks_like_video_diffusion(pkg: Any) -> bool:
    try:
        denoiser = next(pkg[name] for name in _DENOISER_KEYS if name in pkg)
        sample = next(
            value
            for value in denoiser.graph.inputs
            if value.name in {"sample", "latent", "hidden_states"}
        )
    except (AttributeError, KeyError, StopIteration, TypeError):
        return False
    return sample.shape is not None and len(sample.shape) == 5


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


def _has_audio_encoder(pkg: Any) -> bool:
    """Whether a package fuses audio, and so needs a feature extractor."""
    try:
        return "audio_encoder" in set(pkg.keys())
    except AttributeError:
        return False


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
    if names != {"encoder", "decoder"}:
        return False
    decoder = pkg["decoder"]
    try:
        decoder_inputs = {value.name for value in decoder.graph.inputs}
    except AttributeError:
        return False
    return "encoder_hidden_states" in decoder_inputs


def _looks_like_ctc_asr(pkg: Any) -> bool:
    """Detect a non-generative CTC ASR package.

    The signal is structural: a single ``model`` component that consumes a raw
    waveform plus a sample-level mask and emits per-frame ``logits`` with no KV
    cache.  The absent cache is what separates CTC from a Whisper-style
    autoregressive decoder that also emits ``logits``.
    """
    try:
        names = set(pkg.keys())
    except AttributeError:
        return False
    if names != {"model"}:
        return False
    try:
        model = pkg["model"]
        inputs = {value.name for value in model.graph.inputs}
        outputs = {value.name for value in model.graph.outputs}
    except (AttributeError, KeyError):
        return False
    if not {"input_values", "attention_mask"} <= inputs:
        return False
    if "logits" not in outputs:
        return False
    return not any(name.startswith("past_key_values") for name in inputs)


def _looks_like_encoder_embedding(pkg: Any) -> bool:
    """Detect a bidirectional encoder that returns embeddings, not tokens.

    The signal is structural: a single ``model`` component that consumes
    ``input_ids`` and emits ``last_hidden_state`` with no ``logits`` port and
    no KV cache.  The absent ``logits`` is what separates an embedding encoder
    from every generative package -- there is nothing to sample, so there is no
    decode step to describe.
    """
    try:
        names = set(pkg.keys())
    except AttributeError:
        return False
    if names != {"model"}:
        return False
    try:
        model = pkg["model"]
        inputs = {value.name for value in model.graph.inputs}
        outputs = {value.name for value in model.graph.outputs}
    except (AttributeError, KeyError):
        return False
    if "input_ids" not in inputs or "last_hidden_state" not in outputs:
        return False
    if any(name == "logits" or str(name).startswith("present") for name in outputs):
        return False
    return not any(str(name).startswith("past_key_values") for name in inputs)


def _looks_like_speech_enhancement(pkg: Any) -> bool:
    """Detect a spectral speech-enhancement package.

    The signal is structural: a single ``model`` component that consumes a
    noisy magnitude and phase spectrogram and emits the enhanced pair. There
    is no ``logits`` port and no KV cache, so nothing about it is generative
    -- it is a single pure spectrum-to-spectrum call.
    """
    try:
        names = set(pkg.keys())
    except AttributeError:
        return False
    if names != {"model"}:
        return False
    try:
        model = pkg["model"]
        inputs = {str(value.name) for value in model.graph.inputs}
        outputs = {str(value.name) for value in model.graph.outputs}
    except (AttributeError, KeyError):
        return False
    if not {"noisy_mag", "noisy_pha"} <= inputs:
        return False
    return {"denoised_mag", "denoised_pha"} <= outputs


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
    if names != {"encoder", "decoder"}:
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


def _looks_like_vibevoice_tts(pkg: Any) -> bool:
    """Detect the continuous-token VibeVoice component topology."""
    try:
        names = set(pkg.keys())
    except AttributeError:
        return False
    return {
        "audio_encoder",
        "audio_projection",
        "embedding",
        "decoder",
        "diffusion_head",
        "audio_decoder",
        "semantic_encoder",
        "semantic_projection",
    } <= names


def _looks_like_vibevoice_asr(pkg: Any) -> bool:
    """Detect the offline VibeVoice-ASR dual-encoder component topology."""
    try:
        names = set(pkg.keys())
    except AttributeError:
        return False
    return {
        "acoustic_encoder",
        "semantic_encoder",
        "connectors",
        "embedding",
        "decoder",
    } <= names


def _write_vibevoice_asr_processor_contract(output_dir: str, config: Any) -> str:
    """Materialize the source processor protocol absent from the ASR checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "preprocessor_config.json")
    payload = {
        "processor_class": "VibeVoiceASRProcessor",
        "speech_tok_compress_ratio": int(
            getattr(config.acoustic_tokenizer, "hop_length", 3_200)
        ),
        "target_sample_rate": int(getattr(config, "sampling_rate", 24_000)),
        "normalize_audio": True,
        "target_dB_FS": -25.0,
        "eps": 1e-6,
        "acoustic_tokenizer_chunk_size": int(
            getattr(config, "acoustic_tokenizer_chunk_size", 1_440_000)
        ),
        "acoustic_sampling": {
            "noise_scale_input": "acoustic_noise_scale",
            "latent_noise_input": "acoustic_latent_noise",
            "formula": "latents + vae_std * noise_scale[:, None, None] * latent_noise",
        },
        "host_orchestration": [
            (
                "Split each 24 kHz waveform into chunk_samples windows and carry every "
                "past_conv.* output into the matching input of both audio encoders; "
                "zero-pad the terminal window to speech_tok_compress_ratio."
            ),
            (
                "Concatenate encoder chunks per utterance; call connectors once with the "
                "full sample-level padding mask and explicit reproducible noise draws."
            ),
            (
                "Create one audio placeholder token for each emitted audio_features row, "
                "then run embedding and the left-padded cached decoder autoregressively."
            ),
            "Parse generated JSON records into start_time, end_time, speaker_id, and text.",
        ],
        "prompt_protocol": {
            "system": (
                "You are a helpful assistant that transcribes audio input into text output "
                "in JSON format."
            ),
            "speech_tokens": ["<|speech_start|>", "<|speech_pad|>", "<|speech_end|>"],
            "context_info": "Optional background information or hotwords inserted in the user prompt.",
            "fields": ["Start time", "End time", "Speaker ID", "Content"],
        },
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return path


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


def _write_advisory_component_contract(
    pkg: Any,
    output_dir: str,
    *,
    warning: str,
) -> dict[str, str]:
    """Write exact component metadata for a package unsupported by the tested runtime."""
    try:
        package_items = list(pkg.items())
    except (AttributeError, TypeError) as error:
        raise ValueError("Package must expose named model components.") from error
    if not package_items:
        raise ValueError("Package must contain at least one model component.")

    components: dict[str, dict[str, Any]] = {}
    for name, model in package_items:
        if not isinstance(name, str) or not name:
            raise ValueError("Package component names must be non-empty strings.")
        graph = getattr(model, "graph", None)
        if graph is None:
            raise ValueError(f"Package component {name!r} has no graph contract.")
        inputs = [value.name for value in graph.inputs]
        outputs = [value.name for value in graph.outputs]
        if any(not isinstance(value, str) or not value for value in (*inputs, *outputs)):
            raise ValueError(f"Package component {name!r} has unnamed graph ports.")
        if not outputs:
            raise ValueError(f"Package component {name!r} has no graph outputs.")
        components[name] = {
            "filename": ("model.onnx" if len(package_items) == 1 else f"{name}/model.onnx"),
            "inputs": inputs,
            "outputs": outputs,
            "metadata": dict(getattr(model, "metadata_props", {})),
        }

    os.makedirs(output_dir, exist_ok=True)
    metadata = {
        "runtime_validation_status": "unsupported-by-tested-runtime",
        "warnings": [warning],
        "components": components,
    }
    inference_path = os.path.join(output_dir, "inference_metadata.yaml")
    compatibility_path = os.path.join(output_dir, "runtime_compatibility.json")
    for path in (inference_path, compatibility_path):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")
    _LOGGER.warning("%s", warning)
    return {
        "inference_metadata": inference_path,
        "runtime_compatibility": compatibility_path,
    }


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
    revision: str | None = None,
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
    Image edit            VAE pair + denoiser w/ target_sequence_len   typed SSA workflow (edit loop)
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
    package_config = getattr(pkg, "config", None)
    resolved_config = config if config is not None else package_config
    if isinstance(resolved_config, ReUseConfig):
        _validate_reuse_rate_selection(resolved_config)
    config_types = {
        getattr(candidate, "model_type", None)
        for candidate in (package_config, config)
        if candidate is not None
    }
    decoder = pkg.get("decoder") or pkg.get("model")
    decoder_inputs = (
        {value.name for value in decoder.graph.inputs} if decoder is not None else set()
    )
    qwen4_signature = {"ple_input_ids", "past_position_ids"} <= decoder_inputs
    if config_types & {"qwen4_exp", "qwen4_exp_text"} or qwen4_signature:
        warning = (
            "The tested onnx-genai runtime cannot orchestrate Qwen4-Exp's ple_input_ids "
            "and four-axis position state; component graphs and their exact contracts "
            "are exported without claiming runtime validation."
        )
        return _write_advisory_component_contract(pkg, output_dir, warning=warning)
    component_names = set(pkg)
    if component_names in ({"audio_encoder"}, {"speaker_encoder"}) and getattr(
        pkg, "gguf_projector_type", None
    ):
        warning = (
            "The tested onnx-genai runtime has no standalone GGUF audio/speaker "
            "sidecar orchestrator; exact component and processor contracts are advisory."
        )
        return _write_advisory_component_contract(pkg, output_dir, warning=warning)
    os.makedirs(output_dir, exist_ok=True)
    if is_shared_state_pixel_flow_package(pkg):
        if resolved_config is None:
            raise ValueError("shared-state pixel-flow metadata requires a model config")
        path = write_shared_state_pixel_flow_workflow_metadata(
            pkg,
            resolved_config,
            output_dir,
            num_inference_steps=num_inference_steps,
            guidance_scale=4.0 if guidance_scale is None else guidance_scale,
        )
        artifacts = {"inference_metadata": path}
        artifacts.update(_copy_runtime_assets(output_dir, source, revision=revision))
        processor_path = os.path.join(output_dir, "preprocessor_config.json")
        if not os.path.isfile(processor_path):
            with open(processor_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "do_convert_rgb": True,
                        "do_resize": True,
                        "min_pixels": 512 * 512,
                        "max_pixels": 2048 * 2048,
                        "size_multiple": 32,
                        "resample": "lanczos3",
                        "do_rescale": True,
                        "rescale_factor": 1.0 / 255.0,
                        "do_normalize": True,
                        "image_mean": [0.485, 0.456, 0.406],
                        "image_std": [0.229, 0.224, 0.225],
                    },
                    handle,
                    indent=2,
                )
                handle.write("\n")
            artifacts["preprocessor_config"] = processor_path
        if "tokenizer" not in artifacts:
            tokenizer_path = _write_hf_tokenizer(output_dir, source, revision=revision)
            if tokenizer_path is not None:
                artifacts["tokenizer"] = tokenizer_path
        return artifacts

    if _looks_like_language_diffusion(pkg):
        path = write_language_diffusion_workflow_metadata(
            pkg,
            output_dir,
            num_inference_steps=num_inference_steps,
        )
        artifacts = {"inference_metadata": path}
        artifacts.update(_write_text_runtime_assets(output_dir, source, revision=revision))
        return artifacts

    if _looks_like_hierarchical_audio_generation(pkg):
        path = write_hierarchical_audio_workflow_metadata(pkg, output_dir)
        artifacts = {"inference_metadata": path}
        artifacts.update(_write_text_runtime_assets(output_dir, source, revision=revision))
        return artifacts

    if _looks_like_diffusion(pkg):
        is_image_edit = _looks_like_image_edit(pkg)
        if is_image_edit:
            if guidance_scale is None:
                raise ValueError(
                    "image-edit workflow must declare its true-CFG guidance: "
                    "pass guidance_scale explicitly using the source pipeline's default"
                )
            if scheduler is None:
                scheduler = load_diffusers_scheduler_config(source, revision=revision)
            if scheduler is None:
                raise ValueError(
                    "image-edit workflow requires the diffusers scheduler config; "
                    "pass scheduler=SchedulerConfig(...) or a resolvable source"
                )
            image_seq_len = kwargs.pop("image_seq_len", None)
            if image_seq_len is None:
                raise ValueError(
                    "image-edit workflow requires image_seq_len (the packed target "
                    "token count) to materialize the resolution-dependent schedule"
                )
            timesteps, sigma_schedule = _flow_match_euler_schedule(
                scheduler, num_inference_steps, int(image_seq_len)
            )
            path = write_image_edit_workflow_metadata(
                pkg,
                output_dir,
                num_inference_steps=num_inference_steps,
                schedule=sigma_schedule,
                timesteps=timesteps,
                guidance_scale=guidance_scale,
                artifact_paths=kwargs.pop("artifact_paths", None),
            )
            artifacts = {"inference_metadata": path}
            tokenizer_path = _write_hf_tokenizer(output_dir, source, revision=revision)
            if tokenizer_path is not None:
                artifacts["tokenizer"] = tokenizer_path
            return artifacts
        if _looks_like_video_diffusion(pkg):
            if scheduler is None:
                scheduler = load_diffusers_scheduler_config(source, revision=revision)
            resolved_scheduler = scheduler or SchedulerConfig(kind="ddim")
            timesteps, alpha_schedule = _ddim_alpha_schedule(
                resolved_scheduler, num_inference_steps
            )
            if guidance_scale is None:
                raise ValueError(
                    "video diffusion workflow must declare its guidance: "
                    "pass guidance_scale=1.0 for unguided generation, or the source "
                    "pipeline's classifier-free guidance default"
                )
            scaling_factor = (
                load_diffusers_vae_scaling_factor(source, revision=revision) or 1.0
            )
            path = write_video_diffusion_workflow_metadata(
                pkg,
                output_dir,
                num_inference_steps=num_inference_steps,
                schedule=alpha_schedule,
                timesteps=timesteps,
                solver="ddim",
                prediction_type=resolved_scheduler.prediction_type,
                clip_sample_range=(
                    resolved_scheduler.clip_sample_range
                    if resolved_scheduler.clip_sample
                    else None
                ),
                scaling_factor=scaling_factor,
                guidance_scale=(
                    None
                    if guidance_scale is None or np.isclose(guidance_scale, 1.0)
                    else float(guidance_scale)
                ),
            )
            artifacts = {"inference_metadata": path}
            tokenizer_path = _write_hf_tokenizer(output_dir, source, revision=revision)
            if tokenizer_path is not None:
                artifacts["tokenizer"] = tokenizer_path
            return artifacts
        if scheduler is None:
            scheduler = load_diffusers_scheduler_config(source, revision=revision)
        # Fill in component filenames from the package layout, letting any
        # caller-supplied values win.
        derived = _diffusion_component_kwargs(pkg)
        for name, value in derived.items():
            kwargs.setdefault(name, value)
        resolved_scheduler = scheduler or SchedulerConfig(kind="euler")
        timesteps, sigma_schedule = _diffusion_schedule(
            resolved_scheduler, num_inference_steps
        )
        solver, scale_model_input = _DIFFUSION_SOLVERS[resolved_scheduler.kind]
        # A sigma-space sampler starts from noise scaled by the largest sigma; a
        # variance-preserving one starts from the unit-variance draw itself.
        initial_state_scale = sigma_schedule[0] if scale_model_input else 1.0
        conditioned = "text_encoder_filename" in kwargs
        if conditioned and guidance_scale is None:
            raise ValueError(
                "a text-conditioned diffusion package must declare its guidance: "
                "pass guidance_scale=1.0 for unguided generation, or the pipeline's "
                "classifier-free guidance scale to run the guided denoiser path"
            )
        if guidance_scale is not None and not conditioned:
            raise ValueError(
                "classifier-free guidance requires a text-conditioned diffusion package"
            )
        guidance = (
            None
            if guidance_scale is None or np.isclose(guidance_scale, 1.0)
            else float(guidance_scale)
        )
        decoder_input_scale = 1.0
        scaling_factor = load_diffusers_vae_scaling_factor(source, revision=revision)
        if scaling_factor:
            decoder_input_scale = 1.0 / scaling_factor
        path = write_diffusion_workflow_metadata(
            pkg,
            output_dir,
            num_inference_steps=num_inference_steps,
            schedule=sigma_schedule,
            timesteps=timesteps,
            solver=solver,
            scale_model_input=scale_model_input,
            initial_state_scale=initial_state_scale,
            decoder_input_scale=decoder_input_scale,
            guidance_scale=guidance,
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

    if _looks_like_ctc_asr(pkg):
        # CTC ASR is frame-synchronous: the encoder runs once and the transcript
        # comes from the profile's decoding contract, so no decoder/KV metadata
        # is produced.
        ctc_config = config if config is not None else getattr(pkg, "config", None)
        if ctc_config is None:
            raise ValueError(
                "CTC ASR metadata requires a model config (pass config=... or a "
                "package carrying `.config`)"
            )
        # The package's own tokenizer assets are materialized first: the
        # metadata declares their package-relative locations, so they have to
        # exist before the document that names them is written.
        artifacts: dict[str, str] = {}
        tokenizer_path = _write_hf_tokenizer(output_dir, source, revision=revision)
        if tokenizer_path is not None:
            artifacts["tokenizer"] = tokenizer_path
        artifacts["inference_metadata"] = write_ctc_asr_workflow_metadata(
            pkg, output_dir, ctc_config, source=source
        )
        return artifacts

    if _looks_like_speech_enhancement(pkg):
        # A spectral enhancement model has no logits and no cache: one pure
        # spectrum-to-spectrum call.  Emit before the config requirement below
        # because nothing here needs a decoder config.
        path = write_speech_enhancement_workflow_metadata(
            pkg, output_dir, config if config is not None else getattr(pkg, "config", None)
        )
        return {"inference_metadata": path}

    if _looks_like_encoder_embedding(pkg):
        # A bidirectional encoder has no logits and no cache: it runs once and
        # returns one hidden vector per position.  Emit before the config
        # requirement below because nothing here needs a decoder config.
        path = write_encoder_embedding_workflow_metadata(pkg, output_dir, config)
        artifacts = {"inference_metadata": path}
        tokenizer_path = _write_hf_tokenizer(output_dir, source, revision=revision)
        if tokenizer_path is not None:
            artifacts["tokenizer"] = tokenizer_path
        return artifacts

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
            source=source,
        )
        return {"inference_metadata": path}

    try:
        component_names = sorted(pkg.keys())
    except (AttributeError, TypeError):
        component_names = []
    resolved_config = config if config is not None else getattr(pkg, "config", None)
    if resolved_config is None:
        known_config_topology = (
            _looks_like_multimodal(pkg)
            or _looks_like_speech_to_text(pkg)
            or _looks_like_multi_decoder_tts(pkg)
        )
        if len(component_names) > 1 and not known_config_topology:
            return _write_advisory_component_contract(
                pkg,
                output_dir,
                warning=(
                    "The tested onnx-genai runtime does not recognize this multi-component "
                    f"package topology (components: {component_names}); exact component "
                    "contracts are exported without runtime orchestration."
                ),
            )
        raise ValueError(
            "onnx-genai decoder metadata requires a model config (pass config=... "
            "or a package carrying `.config`)"
        )
    if _looks_like_vibevoice_tts(pkg):
        if kv_native_dtype is not None:
            raise ValueError(
                "workflow VibeVoice export derives KV and convolution state dtypes "
                "from ONNX ports; kv_native_dtype overrides are unsupported"
            )
        artifacts = _write_text_runtime_assets(output_dir, source, revision=revision)
        artifacts.update(
            _copy_runtime_assets(
                output_dir,
                source,
                ("processor_config.json", "generation_config.json"),
                revision=revision,
            )
        )
        audio_processor_path = _write_hf_audio_processor(
            output_dir,
            source,
            revision=revision,
        )
        if audio_processor_path is not None:
            artifacts["audio_processor"] = audio_processor_path
        artifacts.update(
            _write_advisory_component_contract(
                pkg,
                output_dir,
                warning=(
                    "The tested onnx-genai runtime does not implement VibeVoice's "
                    "positive/negative Qwen2 caches, DPM-Solver diffusion loop, and "
                    "streaming convolution state. Exact graph and processor contracts "
                    "are exported without claiming downstream orchestration."
                ),
            )
        )
        return artifacts

    if _looks_like_vibevoice_asr(pkg):
        if kv_native_dtype is not None:
            raise ValueError(
                "VibeVoice-ASR derives KV and convolution state dtypes from ONNX ports; "
                "kv_native_dtype overrides are unsupported"
            )
        artifacts = _write_text_runtime_assets(output_dir, source, revision=revision)
        artifacts["processor_contract"] = _write_vibevoice_asr_processor_contract(
            output_dir, resolved_config
        )
        artifacts.update(
            _write_advisory_component_contract(
                pkg,
                output_dir,
                warning=(
                    "The tested onnx-genai runtime does not orchestrate VibeVoice-ASR's "
                    "dual cached audio encoders, source-defined latent sampling, or "
                    "diarization JSON protocol. Exact graph and processor contracts are "
                    "exported without claiming downstream orchestration."
                ),
            )
        )
        return artifacts

    if _looks_like_multimodal(pkg):
        if kv_native_dtype is not None:
            raise ValueError(
                "workflow VLM export derives KV state dtype from ONNX ports; "
                "kv_native_dtype overrides are unsupported"
            )
        # The package's own tokenizer and processor assets are materialized
        # first: the metadata declares their package-relative locations, so they
        # have to exist before the document that names them is written.
        #
        # A multimodal package needs the processor assets as well as the
        # tokenizer, because the runtime resolves image/audio preprocessing
        # parameters from them.
        artifacts = _copy_runtime_assets(output_dir, source, revision=revision)
        if "tokenizer" not in artifacts:
            tokenizer_path = _write_hf_tokenizer(output_dir, source, revision=revision)
            if tokenizer_path is not None:
                artifacts["tokenizer"] = tokenizer_path
        # A speech-language package fuses audio embeddings, so it needs the
        # feature extractor too: the runtime cannot turn a waveform into the
        # encoder's input without it, and no other asset carries those
        # parameters.
        if _has_audio_encoder(pkg):
            audio_processor_path = _write_hf_audio_processor(
                output_dir, source, revision=revision
            )
            if audio_processor_path is not None:
                artifacts["audio_processor"] = audio_processor_path
        artifacts["inference_metadata"] = write_vlm_workflow_metadata(
            pkg,
            output_dir,
            resolved_config,
            source=source,
        )
        return artifacts

    if _looks_like_speech_to_text(pkg):
        if kv_native_dtype is not None:
            raise ValueError(
                "workflow speech-to-text export derives KV state dtype from ONNX ports; "
                "kv_native_dtype overrides are unsupported"
            )
        audio_processor_path = _write_hf_audio_processor(output_dir, source, revision=revision)
        # An ASR decoder is still a text producer: ship its tokenizer and chat
        # template alongside the audio processor, before the metadata names them.
        artifacts = _write_text_runtime_assets(output_dir, source, revision=revision)
        if audio_processor_path is not None:
            artifacts["audio_processor"] = audio_processor_path
        artifacts["inference_metadata"] = write_speech_to_text_workflow_metadata(
            pkg,
            output_dir,
            resolved_config,
            audio_preprocessing=_audio_preprocessing_program(
                audio_processor_path, pkg["encoder"]
            ),
            source=source,
        )
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

    # Fallback: a single-component decoder language model. Preserve unknown
    # multi-component packages as exact component contracts without claiming that
    # the tested runtime can orchestrate them.
    if len(component_names) > 1:
        return _write_advisory_component_contract(
            pkg,
            output_dir,
            warning=(
                "The tested onnx-genai runtime does not recognize this multi-component "
                f"package topology (components: {component_names}); exact component contracts "
                "are exported without runtime orchestration."
            ),
        )

    if kv_native_dtype is not None:
        raise ValueError(
            "workflow decoder export derives KV state dtype from ONNX ports; "
            "kv_native_dtype overrides are unsupported"
        )
    # The package's own tokenizer and chat-template assets are materialized
    # first: the metadata declares their package-relative locations, so they
    # have to exist before the document that names them is written.
    artifacts = _write_text_runtime_assets(output_dir, source, revision=revision)
    artifacts["inference_metadata"] = write_decoder_workflow_metadata(
        pkg,
        output_dir,
        resolved_config,
        sampler=str(getattr(resolved_config, "workflow_sampler", "greedy")),
        source=source,
    )
    return artifacts
