# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""GGUF model import support for mobius.

This package provides tools to load GGUF model files and convert them
to ONNX models using the existing graph construction pipeline.

Usage::

    from mobius.integrations.gguf import build_from_gguf

    # Text-only model
    pkg = build_from_gguf("path/to/model.gguf")
    # Supported quantization is preserved by default; pass
    # keep_quantized=False for a fully float model.

    # Multimodal (text + companion mmproj vision/audio encoder)
    pkg = build_from_gguf("path/to/model.gguf", mmproj="path/to/mmproj.gguf")

:func:`build_from_gguf` is the single entry point; passing ``mmproj`` delegates
to :func:`build_gemma4_vlm_from_gguf` for the multimodal assembly.
"""

from __future__ import annotations

from mobius.integrations.gguf._builder import build_from_gguf
from mobius.integrations.gguf._mmproj import build_gemma4_vlm_from_gguf
from mobius.integrations.gguf._tokenizer import write_gguf_tokenizer_json

__all__ = [
    "build_from_gguf",
    "build_gemma4_vlm_from_gguf",
    "write_gguf_tokenizer_json",
]
