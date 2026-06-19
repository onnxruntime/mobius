# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Kyutai Moshi / Mimi native checkpoint import support for mobius.

The Moshi family (incl. ``nvidia/personaplex-7b-v1``) ships native Kyutai
``safetensors`` checkpoints rather than HuggingFace ``config.json`` bundles.
This package loads those checkpoints and builds the corresponding ONNX
:class:`~mobius._model_package.ModelPackage` via the standard
``build_from_module`` pipeline.

Usage::

    from mobius.integrations.moshi import build_mimi

    pkg = build_mimi("nvidia/personaplex-7b-v1")
    pkg.save("mimi-onnx")
"""

from __future__ import annotations

from mobius.integrations.moshi._builder import build_mimi

__all__ = ["build_mimi"]
