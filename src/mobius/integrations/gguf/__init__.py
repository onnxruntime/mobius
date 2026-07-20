# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""GGUF model import support for mobius.

This package provides tools to load GGUF model files and convert them
to ONNX models using the existing graph construction pipeline.

Usage::

    from mobius.integrations.gguf import build_from_gguf

    pkg = build_from_gguf("path/to/model.gguf")
"""

from __future__ import annotations

from mobius.integrations.gguf._builder import build_from_gguf
from mobius.integrations.gguf._tokenizer import write_gguf_tokenizer_json

__all__ = ["build_from_gguf", "write_gguf_tokenizer_json"]
