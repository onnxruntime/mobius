# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

__all__ = [
    "ArchitectureConfig",
    "AdapterApplication",
    "AdapterArtifact",
    "AdapterBatchSelection",
    "AdapterSlotSelection",
    "AdapterSelectionTensors",
    "AdapterServiceOptions",
    "AdapterSource",
    "AdapterTarget",
    "AdapterTargetDescriptor",
    "AdapterTargetManifest",
    "AdapterTargetSlice",
    "AdapterWeights",
    "AudioConfig",
    "BaseModelConfig",
    "CausalLMConfig",
    "CausalLMTask",
    "ComponentInfo",
    "ComponentExportDisposition",
    "ComponentExportReport",
    "DepthAnythingConfig",
    "EncoderConfig",
    "EpCapabilities",
    "Gemma2Config",
    "Gemma3nConfig",
    "Gemma3nMultiModalConfig",
    "Gemma4AudioConfig",
    "Gemma4Config",
    "MambaConfig",
    "MllamaConfig",
    "MoonshineConfig",
    "MoonshineStreamingConfig",
    "ModelPackage",
    "ModelRegistration",
    "ModelRegistry",
    "ModelTask",
    "MLPWorldModel",
    "MMSConfig",
    "OPSET_VERSION",
    "Sam2Config",
    "SegformerConfig",
    "SpeechToTextConfig",
    "VisionConfig",
    "VisionLanguageConfig",
    "WhisperConfig",
    "WorldModelConfig",
    "WorldModelTask",
    "YolosConfig",
    "apply_weights",
    "adapter_source_from_onnx_adapter",
    "attach_peft_adapter",
    "build",
    "build_context",
    "build_diffusers_pipeline",
    "build_from_gguf",
    "build_mattergen",
    "build_from_module",
    "build_from_nemo",
    "compose_adapter_deltas",
    "components",
    "ep_capabilities",
    "ep_registry",
    "fingerprint_model_weights",
    "load_peft_adapter",
    "generation",
    "get_build_dtype",
    "get_ep",
    "inspect_components",
    "models",
    "optimize_model",
    "register_ep",
    "registry",
    "stream_safetensors_to_model",
    "tasks",
]

__version__ = "0.1.0"

from mobius import components, generation, models, tasks
from mobius._build_context import build_context, ep_capabilities, get_build_dtype
from mobius._builder import build_from_module
from mobius._configs import (
    ArchitectureConfig,
    AudioConfig,
    BaseModelConfig,
    CausalLMConfig,
    DepthAnythingConfig,
    EncoderConfig,
    Gemma2Config,
    Gemma3nConfig,
    Gemma3nMultiModalConfig,
    Gemma4AudioConfig,
    Gemma4Config,
    MambaConfig,
    MllamaConfig,
    MMSConfig,
    MoonshineConfig,
    MoonshineStreamingConfig,
    Sam2Config,
    SegformerConfig,
    SpeechToTextConfig,
    VisionConfig,
    VisionLanguageConfig,
    WhisperConfig,
    WorldModelConfig,
    YolosConfig,
)
from mobius._constants import OPSET_VERSION
from mobius._execution_providers import EpCapabilities, ep_registry, get_ep, register_ep
from mobius._export_report import ComponentExportDisposition, ComponentExportReport
from mobius._inspect import ComponentInfo, inspect_components
from mobius._model_package import ModelPackage
from mobius._optimizations import optimize_model
from mobius._registry import (
    ModelRegistration,
    ModelRegistry,
    registry,
)
from mobius.adapter_io import (
    adapter_source_from_onnx_adapter,
    attach_peft_adapter,
    load_peft_adapter,
)
from mobius.adapters import (
    AdapterApplication,
    AdapterArtifact,
    AdapterBatchSelection,
    AdapterSelectionTensors,
    AdapterServiceOptions,
    AdapterSlotSelection,
    AdapterSource,
    AdapterTarget,
    AdapterTargetDescriptor,
    AdapterTargetManifest,
    AdapterTargetSlice,
    AdapterWeights,
    compose_adapter_deltas,
    fingerprint_model_weights,
)
from mobius.integrations._weight_loading import apply_weights, stream_safetensors_to_model
from mobius.integrations.diffusers import build_diffusers_pipeline
from mobius.integrations.gguf import build_from_gguf
from mobius.integrations.mattergen import build_mattergen
from mobius.integrations.nemo import build_from_nemo
from mobius.integrations.transformers import build
from mobius.models import MLPWorldModel
from mobius.tasks import CausalLMTask, ModelTask, WorldModelTask
