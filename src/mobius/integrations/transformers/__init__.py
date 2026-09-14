# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Hugging Face Transformers integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "build",
    "build_transformers_model",
]

if TYPE_CHECKING:
    from mobius.integrations.transformers._builder import build_transformers_model

    build = build_transformers_model


def __getattr__(name: str):
    """Load public builder functions only after model registration completes."""
    if name in ("build", "build_transformers_model"):
        from mobius.integrations.transformers._builder import build_transformers_model

        return build_transformers_model
    raise AttributeError(name)
