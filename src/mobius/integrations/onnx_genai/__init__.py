# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""onnx-genai integration: inference_metadata.yaml generation.

onnx-genai (https://github.com/justinchuby/onnx-genai) is a standard-driven
runtime whose behavior is declared by an ``inference_metadata`` document rather
than hardcoded per-model dispatch. This module emits the pipeline section for
multi-model diffusion packages (denoiser + optional VAE / text encoder) so a
Mobius-built diffusion package is directly runnable by onnx-genai's iterative
pipeline (`PipelineEngine::run_pipeline`).

The core model/task/component layers remain runtime-agnostic; all onnx-genai
specific code lives here.
"""

from mobius.integrations.onnx_genai.auto_export import write_onnx_genai_config
from mobius.integrations.onnx_genai.decoder_metadata import (
    build_decoder_metadata,
    decoder_metadata_from_config,
    write_decoder_metadata,
)
from mobius.integrations.onnx_genai.inference_metadata import (
    SchedulerConfig,
    build_diffusion_pipeline_metadata,
    write_diffusion_pipeline_metadata,
)

__all__ = [
    "SchedulerConfig",
    "build_decoder_metadata",
    "build_diffusion_pipeline_metadata",
    "decoder_metadata_from_config",
    "write_decoder_metadata",
    "write_diffusion_pipeline_metadata",
    "write_onnx_genai_config",
]
