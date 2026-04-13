# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""ORT-GenAI integration: genai_config.json generation and runtime helpers.

All onnxruntime-genai specific code lives here. The core model/task/component
layers remain runtime-agnostic.
"""

from mobius.integrations.ort_genai.auto_export import write_ort_genai_config
from mobius.integrations.ort_genai.ep_config import (
    make_genai_decoder_config,
    make_kv_cache_dim_name,
    make_provider_options,
    make_sliding_window_config,
)
from mobius.integrations.ort_genai.genai_config import (
    GenaiConfigGenerator,
)

__all__ = [
    "GenaiConfigGenerator",
    "write_ort_genai_config",
    "make_genai_decoder_config",
    "make_kv_cache_dim_name",
    "make_provider_options",
    "make_sliding_window_config",
]
