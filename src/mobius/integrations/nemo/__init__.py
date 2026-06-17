# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""NeMo ``.nemo`` model import support for mobius.

This package loads NVIDIA NeMo ``.nemo`` archives and converts them to ONNX
models using the standard graph construction pipeline.

Usage::

    from mobius.integrations.nemo import build_from_nemo

    pkg = build_from_nemo("nvidia/nemotron-speech-streaming-en-0.6b")
"""

from __future__ import annotations

from mobius.integrations.nemo._builder import build_from_nemo

__all__ = ["build_from_nemo"]
