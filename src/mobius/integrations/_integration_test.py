# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for ecosystem integration boundaries and compatibility aliases."""

from __future__ import annotations

import importlib
import inspect

import mobius
from mobius.integrations import diffusers, transformers
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


def test_public_builder_signatures_are_preserved() -> None:
    assert inspect.signature(mobius.build) == inspect.signature(canonical_transformers_build)
    assert inspect.signature(mobius.build_diffusers_pipeline) == inspect.signature(
        canonical_diffusers_build
    )


def test_legacy_private_modules_alias_integration_implementations() -> None:
    aliases = {
        "mobius._builder": "mobius.integrations._builder",
        "mobius._config_resolver": "mobius.integrations.transformers._config_resolver",
        "mobius._diffusers_builder": "mobius.integrations.diffusers._builder",
        "mobius._diffusers_configs": "mobius.integrations.diffusers._configs",
        "mobius._weight_loading": "mobius.integrations._weight_loading",
    }
    for legacy_name, canonical_name in aliases.items():
        legacy = importlib.import_module(legacy_name)
        canonical = importlib.import_module(canonical_name)
        assert legacy is canonical
