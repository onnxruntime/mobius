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
* **Full-duplex speech-to-speech** — :func:`write_full_duplex_workflow_metadata`
  emits one-event-per-invocation SSA with session-scoped conversational state
  (Moshi / PersonaPlex).
* **Audio codec / multi-decoder TTS** —
  :func:`write_audio_codec_workflow_metadata` emits typed codec SSA, while
  :func:`write_tts_workflow_metadata` reports the current nested-loop induction
  contract blocker precisely.
* **Diffusion pipelines** — :func:`write_diffusion_pipeline_metadata` emits a
  typed SSA denoise loop for a denoiser plus VAE and optional text encoder,
  shipping the sampler's solver/schedule components alongside it.

:func:`write_onnx_genai_config` is the unified entry point: it inspects the
built package and dispatches to the matching writer, so
``mobius build --runtime onnx-genai`` works across every model family above.

The core model/task/component layers remain runtime-agnostic; all onnx-genai
specific code lives here.
"""

from mobius.integrations.onnx_genai._workflow_contract import (
    add_policy_components_to_workflow,
)
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
    moe_metadata_from_config,
    write_decoder_metadata,
)
from mobius.integrations.onnx_genai.inference_metadata import (
    SchedulerConfig,
    build_diffusion_pipeline_metadata,
    build_multimodal_pipeline_metadata,
    build_speech_to_text_pipeline_metadata,
    load_diffusers_scheduler_config,
    write_diffusion_pipeline_metadata,
    write_multimodal_pipeline_metadata,
    write_speech_to_text_pipeline_metadata,
)
from mobius.integrations.onnx_genai.package_facts import (
    IMAGE_PLACEHOLDER_ROLE,
    MEDIA_TOKEN_ROLES,
    TEXT_TOKEN_ROLES,
    PackageFacts,
    SpecialTokenFact,
    SpecialTokenRole,
    TokenFacts,
    TokenizerArtifact,
    TokenizerFacts,
    build_tokenizer_facts,
)
from mobius.integrations.onnx_genai.shared_state_flow_metadata import (
    build_shared_state_pixel_flow_workflow_metadata,
    is_shared_state_pixel_flow_package,
    write_shared_state_pixel_flow_workflow_metadata,
)
from mobius.integrations.onnx_genai.workflow_metadata import (
    HIERARCHICAL_AUDIO_ROLES,
    HierarchicalAudioWorkflowConfig,
    build_audio_codec_workflow_metadata,
    build_decoder_workflow_metadata,
    build_diffusion_workflow_metadata,
    build_full_duplex_workflow_metadata,
    build_hierarchical_audio_workflow_metadata,
    build_image_edit_workflow_metadata,
    build_language_diffusion_pipeline_metadata,
    build_speculative_workflow_metadata,
    build_speech_enhancement_workflow_metadata,
    build_tts_workflow_metadata,
    build_video_diffusion_workflow_metadata,
    build_vlm_workflow_metadata,
    write_audio_codec_workflow_metadata,
    write_decoder_workflow_metadata,
    write_diffusion_workflow_metadata,
    write_full_duplex_workflow_metadata,
    write_hierarchical_audio_workflow_metadata,
    write_image_edit_workflow_metadata,
    write_language_diffusion_workflow_metadata,
    write_speculative_workflow_metadata,
    write_speech_enhancement_workflow_metadata,
    write_tts_workflow_metadata,
    write_video_diffusion_workflow_metadata,
    write_vlm_workflow_metadata,
)

__all__ = [
    "ComfyUIWorkflow",
    "ConversionResult",
    "HIERARCHICAL_AUDIO_ROLES",
    "IMAGE_PLACEHOLDER_ROLE",
    "MEDIA_TOKEN_ROLES",
    "TEXT_TOKEN_ROLES",
    "HierarchicalAudioWorkflowConfig",
    "PackageFacts",
    "SchedulerConfig",
    "SpecialTokenFact",
    "SpecialTokenRole",
    "TokenFacts",
    "TokenizerArtifact",
    "TokenizerFacts",
    "add_policy_components_to_workflow",
    "build_audio_codec_workflow_metadata",
    "build_full_duplex_workflow_metadata",
    "build_decoder_metadata",
    "build_decoder_workflow_metadata",
    "build_diffusion_pipeline_metadata",
    "build_diffusion_workflow_metadata",
    "build_hierarchical_audio_workflow_metadata",
    "build_image_edit_workflow_metadata",
    "build_language_diffusion_pipeline_metadata",
    "build_multimodal_pipeline_metadata",
    "build_pipeline_metadata_for_workflow",
    "build_speculative_workflow_metadata",
    "build_speech_to_text_pipeline_metadata",
    "build_shared_state_pixel_flow_workflow_metadata",
    "build_speech_enhancement_workflow_metadata",
    "build_tokenizer_facts",
    "build_tts_workflow_metadata",
    "build_video_diffusion_workflow_metadata",
    "build_vlm_workflow_metadata",
    "convert_comfyui_workflow",
    "decoder_metadata_from_config",
    "load_diffusers_scheduler_config",
    "is_shared_state_pixel_flow_package",
    "moe_metadata_from_config",
    "parse_comfyui_workflow",
    "parse_comfyui_workflow_file",
    "translate_comfyui_workflow",
    "translate_comfyui_workflow_file",
    "write_audio_codec_workflow_metadata",
    "write_full_duplex_workflow_metadata",
    "write_decoder_metadata",
    "write_decoder_workflow_metadata",
    "write_diffusion_pipeline_metadata",
    "write_diffusion_workflow_metadata",
    "write_hierarchical_audio_workflow_metadata",
    "write_image_edit_workflow_metadata",
    "write_language_diffusion_workflow_metadata",
    "write_multimodal_pipeline_metadata",
    "write_onnx_genai_config",
    "write_speculative_workflow_metadata",
    "write_speech_to_text_pipeline_metadata",
    "write_shared_state_pixel_flow_workflow_metadata",
    "write_speech_enhancement_workflow_metadata",
    "write_tts_workflow_metadata",
    "write_video_diffusion_workflow_metadata",
    "write_vlm_workflow_metadata",
]
