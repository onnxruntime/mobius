# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Integrations with external model ecosystems and runtimes.

Ecosystem-specific model discovery and loading lives in subpackages such as
:mod:`mobius.integrations.transformers` and
:mod:`mobius.integrations.diffusers`. Core ONNX graph construction remains
available through :func:`build_from_module`.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "build_diffusers_pipeline",
    "build_from_module",
    "build_transformers_model",
]


def build_from_module(*args: Any, **kwargs: Any) -> Any:
    """Build directly from an ONNXScript module."""
    from mobius.integrations._builder import build_from_module as _build_from_module

    return _build_from_module(*args, **kwargs)


def build_transformers_model(*args: Any, **kwargs: Any) -> Any:
    """Build a model from a Transformers checkpoint."""
    from mobius.integrations.transformers import (
        build_transformers_model as _build_transformers_model,
    )

    return _build_transformers_model(*args, **kwargs)


def build_diffusers_pipeline(*args: Any, **kwargs: Any) -> Any:
    """Build neural components from a Diffusers pipeline."""
    from mobius.integrations.diffusers import (
        build_diffusers_pipeline as _build_diffusers_pipeline,
    )

    return _build_diffusers_pipeline(*args, **kwargs)
