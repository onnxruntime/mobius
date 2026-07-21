# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""onnx-genai integration: inference_metadata.yaml generation.

onnx-genai (https://github.com/justinchuby/onnx-genai) is a standard-driven
runtime whose behavior is declared by an ``inference_metadata`` document rather
than hardcoded per-model dispatch. This package emits that document for both
model families Mobius builds:

* **Decoder-only language models** — :func:`write_decoder_metadata` /
  :func:`decoder_metadata_from_config` (in ``decoder_metadata``) emit the
  ``model.attention`` + ``kv_cache`` sections for an autoregressive LLM.
* **Multimodal pipelines** — :func:`write_multimodal_pipeline_metadata` emits a
  composite encoder/fusion/autoregressive-decoder pipeline.
* **Speech-to-text (ASR)** — :func:`write_speech_to_text_pipeline_metadata` emits
  a Whisper-style cross-attention encode→decode pipeline.
* **Audio codec / multi-decoder TTS** — :func:`write_audio_codec_pipeline_metadata`
  and :func:`write_tts_pipeline_metadata` emit audio-to-audio and
  ``pre_embedder``-driven ``nested_autoregressive`` (Qwen3-TTS) pipelines.
* **Diffusion pipelines** — :func:`write_diffusion_pipeline_metadata` emits an
  iterative pipeline for a denoiser plus optional VAE / text encoder.

:func:`write_onnx_genai_config` is the unified entry point: it inspects the
built package and dispatches to the matching writer, so
``mobius build --runtime onnx-genai`` works across every model family above.

The core model/task/component layers remain runtime-agnostic; all onnx-genai
specific code lives here.
"""

from mobius.integrations.onnx_genai.auto_export import write_onnx_genai_config
from mobius.integrations.onnx_genai.comfyui import (
    ComfyUIWorkflow,
    parse_comfyui_workflow,
    parse_comfyui_workflow_file,
    translate_comfyui_workflow,
    translate_comfyui_workflow_file,
)
from mobius.integrations.onnx_genai.convert import (
    ConversionResult,
    build_pipeline_metadata_for_workflow,
    convert_comfyui_workflow,
)
from mobius.integrations.onnx_genai.decoder_metadata import (
    build_decoder_metadata,
    decoder_metadata_from_config,
    write_decoder_metadata,
)
from mobius.integrations.onnx_genai.inference_metadata import (
    SchedulerConfig,
    build_audio_codec_pipeline_metadata,
    build_diffusion_pipeline_metadata,
    build_language_diffusion_pipeline_metadata,
    build_multimodal_pipeline_metadata,
    build_speech_to_text_pipeline_metadata,
    build_tts_pipeline_metadata,
    load_diffusers_scheduler_config,
    write_audio_codec_pipeline_metadata,
    write_diffusion_pipeline_metadata,
    write_multimodal_pipeline_metadata,
    write_speech_to_text_pipeline_metadata,
    write_tts_pipeline_metadata,
)

__all__ = [
    "ComfyUIWorkflow",
    "ConversionResult",
    "SchedulerConfig",
    "build_decoder_metadata",
    "build_diffusion_pipeline_metadata",
    "build_language_diffusion_pipeline_metadata",
    "build_audio_codec_pipeline_metadata",
    "build_multimodal_pipeline_metadata",
    "build_pipeline_metadata_for_workflow",
    "build_speech_to_text_pipeline_metadata",
    "build_tts_pipeline_metadata",
    "convert_comfyui_workflow",
    "decoder_metadata_from_config",
    "load_diffusers_scheduler_config",
    "parse_comfyui_workflow",
    "parse_comfyui_workflow_file",
    "translate_comfyui_workflow",
    "translate_comfyui_workflow_file",
    "write_decoder_metadata",
    "write_diffusion_pipeline_metadata",
    "write_audio_codec_pipeline_metadata",
    "write_multimodal_pipeline_metadata",
    "write_speech_to_text_pipeline_metadata",
    "write_tts_pipeline_metadata",
    "write_onnx_genai_config",
]
