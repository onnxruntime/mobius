# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

__all__ = [
    "ArchitectureConfig",
    "AudioConfig",
    "BaseModelConfig",
    "CausalLMConfig",
    "CausalLMTask",
    "DepthAnythingConfig",
    "EncoderConfig",
    "EpCapabilities",
    "Gemma2Config",
    "Gemma3nConfig",
    "Gemma4AudioConfig",
    "Gemma4Config",
    "MambaConfig",
    "MllamaConfig",
    "ModelPackage",
    "ModelRegistration",
    "ModelRegistry",
    "ModelTask",
    "MMSConfig",
    "OPSET_VERSION",
    "Sam2Config",
    "SegformerConfig",
    "VisionConfig",
    "VisionLanguageConfig",
    "WhisperConfig",
    "YolosConfig",
    "apply_weights",
    "build",
    "build_context",
    "build_diffusers_pipeline",
    "build_from_module",
    "components",
    "ep_capabilities",
    "ep_registry",
    "get_build_dtype",
    "get_ep",
    "models",
    "optimize_model",
    "register_ep",
    "registry",
    "tasks",
]

__version__ = "0.1.0"

from mobius import components, models, tasks
from mobius._build_context import build_context, ep_capabilities, get_build_dtype
from mobius._builder import (
    build,
    build_from_module,
)
from mobius._configs import (
    ArchitectureConfig,
    AudioConfig,
    BaseModelConfig,
    CausalLMConfig,
    DepthAnythingConfig,
    EncoderConfig,
    Gemma2Config,
    Gemma3nConfig,
    Gemma4AudioConfig,
    Gemma4Config,
    MambaConfig,
    MllamaConfig,
    MMSConfig,
    Sam2Config,
    SegformerConfig,
    VisionConfig,
    VisionLanguageConfig,
    WhisperConfig,
    YolosConfig,
)
from mobius._constants import OPSET_VERSION
from mobius._diffusers_builder import build_diffusers_pipeline
from mobius._execution_providers import EpCapabilities, ep_registry, get_ep, register_ep
from mobius._model_package import ModelPackage
from mobius._optimizations import optimize_model
from mobius._registry import (
    ModelRegistration,
    ModelRegistry,
    registry,
)
from mobius._weight_loading import apply_weights
from mobius.tasks import CausalLMTask, ModelTask
