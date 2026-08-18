# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Hugging Face Diffusers integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["build_diffusers_pipeline"]

if TYPE_CHECKING:
    from mobius.integrations.diffusers._builder import build_diffusers_pipeline


def __getattr__(name: str):
    """Load public builder functions only after model registration completes."""
    if name == "build_diffusers_pipeline":
        from mobius.integrations.diffusers._builder import build_diffusers_pipeline

        return build_diffusers_pipeline
    raise AttributeError(name)
