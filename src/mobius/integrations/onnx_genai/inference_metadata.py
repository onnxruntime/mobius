# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Emit onnx-genai ``inference_metadata`` for multi-model pipelines.

Mobius builds the neural components of a diffusion model (denoiser transformer,
VAE, and — externally — a text encoder) as separate ONNX graphs, but does not
itself carry a scheduler loop. onnx-genai's *iterative* pipeline supplies that
loop declaratively: given an ``inference_metadata`` document describing the
components, the loop-carried dataflow, a timestep input, a scheduler, and
(optionally) classifier-free guidance, it drives the denoise loop and returns
the decoded output.

This module produces that document from the component filenames + a scheduler
config. It reads no torch/diffusers state — only plain values — so it is cheap
to unit-test and safe to call anywhere.

The emitted contract matches onnx-genai's pipeline schema:
``schema/inference_metadata.schema.json`` (kind ``iterative`` with
``denoiser`` / ``num_steps`` / ``timestep_input`` / ``scheduler_config`` /
``cfg_conditioning_input`` and denoiser self-edge loop-carried dataflow).

Autoregressive decoder-only LLM metadata (``model.attention`` + ``kv_cache``)
lives in the sibling :mod:`mobius.integrations.onnx_genai.decoder_metadata`
module. Composite multimodal pipelines retain those decoder properties while
declaring their encoder, fusion, and decoder execution stages here.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
from typing import Any

import yaml

_LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class SchedulerConfig:
    """Diffusion noise-schedule parameters for an onnx-genai scheduler."""

    kind: str = "ddim"
    num_train_timesteps: int = 1000
    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_schedule: str = "scaled_linear"
    prediction_type: str = "epsilon"
    use_karras_sigmas: bool = False
    use_exponential_sigmas: bool = False

    def to_metadata(self) -> dict[str, Any]:
        meta = {
            "kind": self.kind,
            "num_train_timesteps": self.num_train_timesteps,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "beta_schedule": self.beta_schedule,
            "prediction_type": self.prediction_type,
        }
        if self.use_karras_sigmas:
            meta["use_karras_sigmas"] = True
        if self.use_exponential_sigmas:
            meta["use_exponential_sigmas"] = True
        return meta

    @classmethod
    def from_diffusers(cls, config: dict[str, Any]) -> SchedulerConfig:
        """Build from a diffusers ``scheduler/scheduler_config.json`` dict.

        Unknown/absent schedule parameters fall back to the (Stable Diffusion)
        defaults. The diffusers scheduler class name (``_class_name``) is mapped
        to an onnx-genai scheduler ``kind``:

        * ``DDIMScheduler``  -> ``ddim``
        * ``EulerDiscreteScheduler`` (non-ancestral) -> ``euler``

        Ancestral samplers (which inject fresh noise every step) have no
        deterministic onnx-genai equivalent and are rejected, as are scheduler
        classes onnx-genai does not implement, so a Mobius-built package never
        silently runs the wrong denoise dynamics.
        """
        raw_name = str(config.get("_class_name", ""))
        name = raw_name.lower()
        if "eulerancestral" in name:
            kind = "euler_ancestral"
        elif "ancestral" in name or "sde" in name:
            raise ValueError(
                f"onnx-genai has no equivalent for the stochastic diffusers scheduler "
                f"{raw_name!r}; supported: DDIMScheduler, EulerDiscreteScheduler, "
                f"EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler"
            )
        elif not name or "ddim" in name:
            kind = "ddim"
        elif "dpmsolvermultistep" in name or "dpm++" in name or "dpmpp" in name:
            kind = "dpmpp_2m"
        elif "euler" in name:
            kind = "euler"
        else:
            raise ValueError(
                f"unsupported diffusers scheduler {raw_name!r} for onnx-genai; "
                f"supported kinds: ddim (DDIMScheduler), euler (EulerDiscreteScheduler)"
            )
        return cls(
            kind=kind,
            num_train_timesteps=int(config.get("num_train_timesteps", 1000)),
            beta_start=float(config.get("beta_start", 0.00085)),
            beta_end=float(config.get("beta_end", 0.012)),
            beta_schedule=str(config.get("beta_schedule", "scaled_linear")),
            prediction_type=str(config.get("prediction_type", "epsilon")),
            use_karras_sigmas=bool(config.get("use_karras_sigmas")),
            use_exponential_sigmas=bool(config.get("use_exponential_sigmas")),
        )


def load_diffusers_scheduler_config(source: str | None) -> SchedulerConfig | None:
    """Best-effort load of a diffusers ``scheduler/scheduler_config.json``.

    ``source`` may be a local diffusers checkpoint directory or a Hugging Face
    model id. Returns a :class:`SchedulerConfig` on success, or ``None`` when the
    config cannot be found or names a scheduler onnx-genai does not implement
    (in which case a warning is logged and the caller should fall back to the
    DDIM default). This never raises for a missing/unsupported scheduler so a
    model build is not blocked by scheduler-metadata resolution.
    """
    if not source:
        return None
    raw: dict[str, Any] | None = None
    local = os.path.join(source, "scheduler", "scheduler_config.json")
    if os.path.isfile(local):
        try:
            with open(local, encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, ValueError) as err:
            _LOGGER.warning("could not read %s: %s", local, err)
            return None
    else:
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(source, "scheduler/scheduler_config.json")
            with open(path, encoding="utf-8") as handle:
                raw = json.load(handle)
        except Exception as err:
            _LOGGER.info("no diffusers scheduler config for %r (%s)", source, err)
            return None
    try:
        return SchedulerConfig.from_diffusers(raw)
    except ValueError as err:
        _LOGGER.warning(
            "%s; falling back to onnx-genai's default DDIM scheduler metadata", err
        )
        return None


def build_language_diffusion_pipeline_metadata(
    *,
    mask_token_id: int,
    num_inference_steps: int,
    model_filename: str = "model.onnx",
    input_ids_port: str = "input_ids",
    logits_port: str = "logits",
    block_length: int | None = None,
    temperature: float | None = None,
    guidance_scale: float | None = None,
) -> dict[str, Any]:
    """Build the onnx-genai ``inference_metadata`` for a masked language-diffusion model.

    For a masked (discrete) language-diffusion model (e.g. LLaDA / Dream).

    The model is a mask predictor: it takes an int64 token sequence on
    ``input_ids_port`` (prompt tokens plus a masked generation region) and emits
    ``[B, S, V]`` logits on ``logits_port``. onnx-genai's ``masked_diffusion``
    scheduler drives the reverse process — each step commits the highest-confidence
    still-masked positions (LLaDA low-confidence remasking) via a loop-carried
    ``logits -> input_ids`` self-edge, unmasking progressively.

    Args:
        mask_token_id: The ``[MASK]`` token id (e.g. 126336 for LLaDA-8B).
        num_inference_steps: Total reverse-process steps (``strategy.num_steps``).
        model_filename: The mask-predictor ONNX filename.
        input_ids_port / logits_port: Model I/O port names.
        block_length: Semi-autoregressive block length in tokens. When set, the
            generation region is decoded in contiguous left-to-right blocks and
            ``num_inference_steps`` must be divisible by the block count.
        temperature: Gumbel-max sampling temperature (default 0 = argmax).
        guidance_scale: Unsupervised classifier-free guidance multiplier. LLaDA's
            effective multiplier is ``cfg_scale + 1``, so pass ``cfg_scale + 1``.

    Returns:
        A dict with a top-level ``pipeline`` key, ready to serialize to
        ``inference_metadata.yaml``.
    """
    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be >= 1")
    if block_length is not None and block_length < 1:
        raise ValueError("block_length must be >= 1")

    scheduler_config: dict[str, Any] = {
        "kind": "masked_diffusion",
        "mask_token_id": int(mask_token_id),
    }
    if temperature is not None:
        scheduler_config["temperature"] = float(temperature)
    if block_length is not None:
        scheduler_config["block_length"] = int(block_length)

    strategy: dict[str, Any] = {
        "kind": "iterative",
        "denoiser": "denoiser",
        "num_steps": num_inference_steps,
        "scheduler_config": scheduler_config,
    }
    if guidance_scale is not None and not math.isclose(guidance_scale, 1.0):
        strategy["guidance_scale"] = guidance_scale

    pipeline: dict[str, Any] = {
        "models": {"denoiser": {"filename": model_filename, "type": "denoiser"}},
        # Loop-carried self-edge: the emitted logits refine the token sequence.
        "dataflow": [{"from": f"denoiser.{logits_port}", "to": f"denoiser.{input_ids_port}"}],
        "strategy": strategy,
    }
    return {"pipeline": pipeline}


def build_diffusion_pipeline_metadata(
    *,
    num_inference_steps: int,
    denoiser_filename: str = "denoiser.onnx",
    denoiser_sample_input: str = "sample",
    denoiser_timestep_input: str = "timestep",
    denoiser_conditioning_input: str = "encoder_hidden_states",
    denoiser_output: str = "noise_pred",
    scheduler: SchedulerConfig | None = None,
    timesteps: list[float] | None = None,
    guidance_scale: float | None = None,
    start_step: int | None = None,
    vae_filename: str | None = None,
    vae_latent_input: str = "latent",
    text_encoder_filename: str | None = None,
    text_encoder_output: str = "last_hidden_state",
    text_encoder_edges: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Build the onnx-genai ``inference_metadata`` dict for a diffusion pipeline.

    The denoiser runs an iterative loop: its ``denoiser_output`` (a noise
    prediction) is fed back to ``denoiser_sample_input`` each step (a
    loop-carried self-edge), the scheduler combines it with the current latent,
    the per-step timestep is injected into ``denoiser_timestep_input``, and the
    conditioning is supplied on ``denoiser_conditioning_input``.

    Args:
        num_inference_steps: Number of denoise steps (``strategy.num_steps``).
        denoiser_*: Denoiser component filename and I/O port names.
        scheduler: Noise-schedule config (defaults to DDIM defaults).
        guidance_scale: When set and != 1.0, enables classifier-free guidance
            (the conditioning input is zeroed on the unconditional pass).
        vae_filename: Optional VAE decoder; runs ``final_only`` on the final
            latent (``denoiser_sample_input``).
        vae_latent_input: VAE latent input port name.
        text_encoder_filename: Optional text encoder; runs ``prompt_only`` and
            feeds ``denoiser_conditioning_input``.
        text_encoder_output: Text encoder output port name.

    Returns:
        A dict with a top-level ``pipeline`` key, ready to serialize to
        ``inference_metadata.yaml``.
    """
    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be >= 1")
    scheduler = scheduler or SchedulerConfig()

    models: dict[str, Any] = {
        "denoiser": {"filename": denoiser_filename, "type": "denoiser"},
    }
    dataflow: list[dict[str, Any]] = [
        # Loop-carried self-edge: previous step's prediction seeds the next.
        {
            "from": f"denoiser.{denoiser_output}",
            "to": f"denoiser.{denoiser_sample_input}",
        },
    ]
    phases: dict[str, Any] = {}

    if text_encoder_filename is not None:
        models["text_encoder"] = {
            "filename": text_encoder_filename,
            "type": "encoder",
        }
        # Route each text-encoder output to its denoiser conditioning input. SD
        # has one edge (hidden states -> encoder_hidden_states); SDXL has two
        # (concatenated hidden states + pooled text_embeds). `time_ids` is not
        # routed here — it is an external denoiser input the caller supplies.
        edges = text_encoder_edges or [(text_encoder_output, denoiser_conditioning_input)]
        for enc_out, denoiser_in in edges:
            dataflow.append(
                {"from": f"text_encoder.{enc_out}", "to": f"denoiser.{denoiser_in}"}
            )
        phases["text_encoder"] = {"run_on": "prompt_only"}

    if vae_filename is not None:
        models["vae"] = {"filename": vae_filename, "type": "vae"}
        # The VAE decodes the final post-scheduler latent (the sample port).
        dataflow.append(
            {
                "from": f"denoiser.{denoiser_sample_input}",
                "to": f"vae.{vae_latent_input}",
            }
        )
        phases["vae"] = {"run_on": "final_only"}

    strategy: dict[str, Any] = {
        "kind": "iterative",
        "denoiser": "denoiser",
        "num_steps": num_inference_steps,
        "timestep_input": denoiser_timestep_input,
        "scheduler_config": scheduler.to_metadata(),
    }
    if timesteps is not None:
        if len(timesteps) != num_inference_steps:
            raise ValueError(
                f"timesteps has {len(timesteps)} entries but num_inference_steps is "
                f"{num_inference_steps}"
            )
        strategy["timesteps"] = [float(t) for t in timesteps]
    if guidance_scale is not None:
        strategy["guidance_scale"] = guidance_scale
        if not math.isclose(guidance_scale, 1.0):
            strategy["cfg_conditioning_input"] = denoiser_conditioning_input
    if start_step:
        if not 0 < start_step < num_inference_steps:
            raise ValueError(
                f"start_step ({start_step}) must be in 1..{num_inference_steps - 1}"
            )
        strategy["start_step"] = start_step

    pipeline: dict[str, Any] = {
        "models": models,
        "dataflow": dataflow,
        "strategy": strategy,
    }
    if phases:
        pipeline["phases"] = phases
    return {"pipeline": pipeline}


def build_multimodal_pipeline_metadata(
    *,
    decoder_filename: str = "decoder.onnx",
    embedding_filename: str = "embedding.onnx",
    vision_encoder_filename: str | None = None,
    audio_encoder_filename: str | None = None,
    tokenizer_filename: str = "tokenizer.json",
    activation_dtype: str = "fp32",
    decoder_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build metadata for an encoder-to-fusion-to-decoder multimodal pipeline.

    At least one modality encoder is required. Each encoder and the embedding
    fusion model runs once for the prompt; the decoder then runs
    autoregressively for every generation step.

    Args:
        decoder_filename: Decoder ONNX filename relative to the package root.
        embedding_filename: Embedding fusion ONNX filename.
        vision_encoder_filename: Optional vision encoder ONNX filename.
        audio_encoder_filename: Optional audio encoder ONNX filename.
        tokenizer_filename: Tokenizer filename used by the decoder.
        decoder_metadata: Optional output from
            :func:`decoder_metadata_from_config`. Its decoder capabilities are
            retained at the document top level.

    Returns:
        A dict with a top-level ``pipeline`` key and any decoder capabilities.
    """
    if vision_encoder_filename is None and audio_encoder_filename is None:
        raise ValueError("a multimodal pipeline requires a vision or audio encoder")

    models: dict[str, Any] = {}
    dataflow: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    phases: dict[str, Any] = {}

    def add_encoder(
        name: str,
        filename: str,
        model_type: str,
        output_name: str,
        stage_name: str,
    ) -> None:
        models[name] = {"filename": filename, "type": model_type}
        dataflow.append(
            {
                "from": f"{name}.{output_name}",
                "to": f"embedding.{output_name}",
                "dtype": activation_dtype,
                "device_transfer": False,
            }
        )
        stages.append(
            {
                "name": stage_name,
                "strategy": {"kind": "single_pass", "model": name},
                "run_on": "prompt_only",
            }
        )
        phases[name] = {"run_on": "prompt_only"}

    if vision_encoder_filename is not None:
        add_encoder(
            "vision_encoder",
            vision_encoder_filename,
            "vision_encoder",
            "image_features",
            "encode_vision",
        )
    if audio_encoder_filename is not None:
        add_encoder(
            "audio_encoder",
            audio_encoder_filename,
            "audio_encoder",
            "audio_features",
            "encode_audio",
        )

    models["embedding"] = {"filename": embedding_filename, "type": "encoder"}
    models["decoder"] = {
        "filename": decoder_filename,
        "type": "decoder",
        "tokenizer": tokenizer_filename,
    }
    dataflow.append(
        {
            "from": "embedding.inputs_embeds",
            "to": "decoder.inputs_embeds",
            "dtype": activation_dtype,
            "device_transfer": False,
        }
    )
    stages.extend(
        [
            {
                "name": "fuse_embeddings",
                "strategy": {"kind": "single_pass", "model": "embedding"},
                "run_on": "prompt_only",
            },
            {
                "name": "decode",
                "strategy": {"kind": "autoregressive", "decoder": "decoder"},
                "run_on": "every_step",
            },
        ]
    )
    phases["embedding"] = {"run_on": "prompt_only"}
    phases["decoder"] = {"run_on": "every_step"}

    metadata = dict(decoder_metadata or {})
    metadata["pipeline"] = {
        "models": models,
        "dataflow": dataflow,
        "strategy": {"kind": "composite", "stages": stages},
        "phases": phases,
    }
    return metadata


def write_multimodal_pipeline_metadata(
    directory: str,
    *,
    filename: str = "inference_metadata.yaml",
    **kwargs: Any,
) -> str:
    """Build and write composite multimodal metadata into ``directory``."""
    metadata = build_multimodal_pipeline_metadata(**kwargs)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)
    return path


def build_speech_to_text_pipeline_metadata(
    *,
    encoder_filename: str = "encoder/model.onnx",
    decoder_filename: str = "decoder/model.onnx",
    tokenizer_filename: str = "tokenizer.json",
    activation_dtype: str = "fp32",
    decoder_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build metadata for a cross-attention encoder-decoder ASR pipeline.

    This is the Whisper-style speech-to-text shape (DESIGN.md §20): the audio
    encoder runs once for the prompt and produces ``encoder_hidden_states``,
    which the autoregressive decoder consumes via cross-attention (distinct from
    the multimodal ``inputs_embeds`` fusion shape). The decoder then runs for
    every generation step.

    Args:
        encoder_filename: Audio encoder ONNX filename relative to the package
            root.
        decoder_filename: Decoder ONNX filename relative to the package root.
        tokenizer_filename: Tokenizer filename used by the decoder.
        decoder_metadata: Optional output from
            :func:`decoder_metadata_from_config`; its decoder capabilities are
            retained at the document top level.

    Returns:
        A dict with a top-level ``pipeline`` key and any decoder capabilities.
    """
    metadata = dict(decoder_metadata or {})
    metadata["pipeline"] = {
        "models": {
            "encoder": {"filename": encoder_filename, "type": "encoder"},
            "decoder": {
                "filename": decoder_filename,
                "type": "decoder",
                "tokenizer": tokenizer_filename,
            },
        },
        "dataflow": [
            {
                "from": "encoder.encoder_hidden_states",
                "to": "decoder.encoder_hidden_states",
                "dtype": activation_dtype,
                "device_transfer": False,
            }
        ],
        "strategy": {
            "kind": "composite",
            "stages": [
                {
                    "name": "encode_audio",
                    "strategy": {"kind": "single_pass", "model": "encoder"},
                    "run_on": "prompt_only",
                },
                {
                    "name": "decode_transcript",
                    "strategy": {"kind": "autoregressive", "decoder": "decoder"},
                    "run_on": "every_step",
                },
            ],
        },
        "phases": {
            "encoder": {"run_on": "prompt_only"},
            "decoder": {"run_on": "every_step"},
        },
    }
    return metadata


def write_speech_to_text_pipeline_metadata(
    directory: str,
    *,
    filename: str = "inference_metadata.yaml",
    **kwargs: Any,
) -> str:
    """Build and write composite speech-to-text metadata into ``directory``."""
    metadata = build_speech_to_text_pipeline_metadata(**kwargs)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)
    return path


def build_audio_codec_pipeline_metadata(
    *,
    encoder_filename: str = "encoder/model.onnx",
    decoder_filename: str = "decoder/model.onnx",
    codes_dtype: str = "int64",
) -> dict[str, Any]:
    """Build metadata for an audio-to-audio neural codec pipeline.

    This is the pure single-pass composite shape (DESIGN.md §20): an audio
    encoder maps a waveform to ``codes``, and a decoder reconstructs a waveform
    from those codes. Both stages run once over the shared tensor pool (there is
    no autoregressive decode and no tokenizer), wired ``encoder.codes ->
    decoder.codes``.

    Args:
        encoder_filename: Waveform-to-codes encoder ONNX filename.
        decoder_filename: Codes-to-waveform decoder ONNX filename.
        codes_dtype: Metadata dtype of the ``codes`` tensor exchanged between the
            two stages (neural codecs typically emit ``int64`` code indices).

    Returns:
        A dict with a top-level ``pipeline`` key. No decoder capabilities are
        emitted because the pipeline produces tensors, not tokens.
    """
    return {
        "pipeline": {
            "models": {
                "encoder": {"filename": encoder_filename, "type": "audio_encoder"},
                "decoder": {"filename": decoder_filename, "type": "vocoder"},
            },
            "dataflow": [
                {
                    "from": "encoder.codes",
                    "to": "decoder.codes",
                    "dtype": codes_dtype,
                    "device_transfer": False,
                }
            ],
            "strategy": {
                "kind": "composite",
                "stages": [
                    {
                        "name": "encode_waveform",
                        "strategy": {"kind": "single_pass", "model": "encoder"},
                        "run_on": "prompt_only",
                    },
                    {
                        "name": "decode_waveform",
                        "strategy": {"kind": "single_pass", "model": "decoder"},
                        "run_on": "prompt_only",
                    },
                ],
            },
            "phases": {
                "encoder": {"run_on": "prompt_only"},
                "decoder": {"run_on": "prompt_only"},
            },
        }
    }


def write_audio_codec_pipeline_metadata(
    directory: str,
    *,
    filename: str = "inference_metadata.yaml",
    **kwargs: Any,
) -> str:
    """Build and write composite audio-codec metadata into ``directory``."""
    metadata = build_audio_codec_pipeline_metadata(**kwargs)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)
    return path


def build_tts_pipeline_metadata(
    *,
    num_code_groups: int,
    max_frames: int = 2000,
    talker_filename: str = "talker/model.onnx",
    code_predictor_filename: str = "code_predictor/model.onnx",
    pre_embedder_filename: str = "talker_step_embedder/model.onnx",
    prefill_embedder_filename: str | None = "talker_prefill_embedder/model.onnx",
    tokenizer_filename: str = "tokenizer.json",
    activation_dtype: str = "fp32",
    decoder_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build metadata for a pre-embedder-driven multi-decoder TTS pipeline.

    This is the real Qwen3-TTS shape (DESIGN.md §20.3, ``nested_autoregressive``
    with the optional ``pre_embedder`` extension): an OUTER ``talker`` AR loop
    where each frame drives an INNER ``code_predictor`` AR loop of
    ``num_code_groups`` steps (seeded by the talker's ``last_hidden_state``).
    Unlike the plain nested shape, the talker is **not** driven by ``input_ids``:
    each frame its ``inputs_embeds`` is materialized from the previous frame's
    codes by the ``talker_step_embedder`` pre-embedder (``frame_codes
    [+ text_embed] -> inputs_embeds``), keeping the engine generic.

    When ``prefill_embedder_filename`` is set (the default), a
    ``talker_prefill_embedder`` prompt-phase component is also emitted. It maps
    the tokenized prompt ``text_ids -> prefill_embeds + trailing_text_embeds``:
    the runtime feeds ``prefill_embeds`` to the talker on frame 0 and threads
    ``trailing_text_embeds[:, k-1, :]`` as the pre-embedder's ``text_embed`` on
    frames k>=1 (see the ``prefill_embedder`` field). Pass ``None`` to emit the
    prefill-less shape (talker frame 0 + ``text_embed`` fed zeros).

    The engine-driven components are emitted (``talker``, ``code_predictor``,
    ``talker_step_embedder``, and ``talker_prefill_embedder`` when present). The
    package's ``embedding`` and optional ``speaker_encoder`` models are internal
    weight sources already folded into the pre-/prefill-embedders, so they are
    not declared as pipeline models. There is **no in-package vocoder** — the
    assembled ``talker.output_codes`` are decoded by a separate codec model.

    Args:
        num_code_groups: Codes collected per outer frame (RVQ residual count).
        max_frames: Maximum number of outer talker frames to generate.
        talker_filename: Outer decoder (talker) ONNX filename.
        code_predictor_filename: Inner decoder ONNX filename.
        pre_embedder_filename: ``talker_step_embedder`` ONNX filename.
        prefill_embedder_filename: ``talker_prefill_embedder`` ONNX filename, or
            ``None`` to omit the prefill/trailing-text path.
        tokenizer_filename: Tokenizer filename used by the talker.
        decoder_metadata: Optional output from
            :func:`decoder_metadata_from_config`; its decoder capabilities are
            retained at the document top level.

    Returns:
        A dict with a top-level ``pipeline`` key and any decoder capabilities.
    """
    if num_code_groups < 1:
        raise ValueError("num_code_groups must be at least 1")
    if max_frames < 1:
        raise ValueError("max_frames must be at least 1")

    models: dict[str, Any] = {
        "talker": {
            "filename": talker_filename,
            "type": "decoder",
            "tokenizer": tokenizer_filename,
        },
        "talker_step_embedder": {
            "filename": pre_embedder_filename,
            "type": "embedding",
        },
        "code_predictor": {
            "filename": code_predictor_filename,
            "type": "decoder",
        },
    }
    dataflow: list[dict[str, Any]] = [
        {
            "from": "talker_step_embedder.inputs_embeds",
            "to": "talker.inputs_embeds",
            "dtype": activation_dtype,
            "device_transfer": False,
        },
        {
            "from": "talker.last_hidden_state",
            "to": "code_predictor.inputs_embeds",
            "dtype": activation_dtype,
            "device_transfer": False,
        },
    ]
    stage_strategy: dict[str, Any] = {
        "kind": "nested_autoregressive",
        "outer": "talker",
        "inner": "code_predictor",
        "pre_embedder": {
            "component": "talker_step_embedder",
            "frame_codes_input": "frame_codes",
            "text_embed_input": "text_embed",
        },
        "num_code_groups": num_code_groups,
        "max_tokens": max_frames,
    }
    phases: dict[str, Any] = {
        "talker": {"run_on": "every_step"},
        "talker_step_embedder": {"run_on": "on_demand"},
        "code_predictor": {"run_on": "every_step"},
    }

    if prefill_embedder_filename is not None:
        models["talker_prefill_embedder"] = {
            "filename": prefill_embedder_filename,
            "type": "embedding",
        }
        # Runs once in the prompt phase; the runtime seeds the declared
        # `prompt_input` with the tokenized prompt and reads the two named
        # outputs from the pool. Every port is declared explicitly (the engine
        # never guesses tensor names).
        stage_strategy["prefill_embedder"] = {
            "component": "talker_prefill_embedder",
            "prompt_input": "text_ids",
            "prefill_output": "prefill_embeds",
            "trailing_output": "trailing_text_embeds",
        }
        phases["talker_prefill_embedder"] = {"run_on": "prompt_only"}

    metadata = dict(decoder_metadata or {})
    metadata["pipeline"] = {
        "models": models,
        "dataflow": dataflow,
        "strategy": {
            "kind": "composite",
            "stages": [
                {
                    "name": "generate_codes",
                    "strategy": stage_strategy,
                    "run_on": "every_step",
                },
            ],
        },
        "phases": phases,
    }
    return metadata


def write_tts_pipeline_metadata(
    directory: str,
    *,
    filename: str = "inference_metadata.yaml",
    **kwargs: Any,
) -> str:
    """Build and write pre-embedder-driven TTS metadata into ``directory``."""
    metadata = build_tts_pipeline_metadata(**kwargs)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)
    return path


def write_diffusion_pipeline_metadata(
    directory: str,
    *,
    filename: str = "inference_metadata.yaml",
    **kwargs: Any,
) -> str:
    """Build and write ``inference_metadata.yaml`` into ``directory``.

    Extra keyword arguments are forwarded to
    :func:`build_diffusion_pipeline_metadata`. Returns the written path.
    """
    metadata = build_diffusion_pipeline_metadata(**kwargs)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)
    return path
