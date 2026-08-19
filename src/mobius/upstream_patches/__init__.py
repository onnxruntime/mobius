# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Corrections to runtime assets that ship broken from upstream model repos."""

from __future__ import annotations

__all__ = [
    "AssetPatch",
    "PatchError",
    "apply_asset_patches",
    "apply_unified_diff",
    "available_patches",
]

from mobius.upstream_patches._diff import PatchError, apply_unified_diff
from mobius.upstream_patches._patches import (
    AssetPatch,
    apply_asset_patches,
    available_patches,
)
