# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for ecosystem integration boundaries and the public façade."""

from __future__ import annotations

import inspect

import mobius
from mobius._builder import build_from_module as canonical_module_build
from mobius.integrations import diffusers, transformers
from mobius.integrations._weight_loading import apply_weights as canonical_apply_weights
from mobius.integrations.diffusers._builder import (
    build_diffusers_pipeline as canonical_diffusers_build,
)
from mobius.integrations.transformers._builder import (
    build_transformers_model as canonical_transformers_build,
)


def test_public_build_uses_transformers_integration() -> None:
    assert mobius.build is transformers.build


def test_public_diffusers_build_uses_diffusers_integration() -> None:
    assert mobius.build_diffusers_pipeline is diffusers.build_diffusers_pipeline


def test_public_core_functions_use_canonical_implementations() -> None:
    assert mobius.build_from_module is canonical_module_build
    assert mobius.apply_weights is canonical_apply_weights


def test_public_builder_signatures_are_preserved() -> None:
    assert inspect.signature(mobius.build) == inspect.signature(canonical_transformers_build)
    assert inspect.signature(mobius.build_from_module) == inspect.signature(
        canonical_module_build
    )
    assert inspect.signature(mobius.build_diffusers_pipeline) == inspect.signature(
        canonical_diffusers_build
    )
    assert inspect.signature(mobius.apply_weights) == inspect.signature(
        canonical_apply_weights
    )
