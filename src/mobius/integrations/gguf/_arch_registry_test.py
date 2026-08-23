# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Invariants for the GGUF architecture registry.

These tests exist to fail. Each one encodes a way the pre-registry code drifted
apart, so that the same divergence cannot be reintroduced silently:

* capability sets that disagreed (``bloom``/``t5`` configurable but unmappable;
  ``gemma``/``internlm2`` mappable but unconfigured),
* behavior tables reachable under one key and registered under another
  (the Gemma 3 weight processor),
* mobius ``model_type`` strings masquerading as GGUF architectures
  (``qwen2_moe``, ``qwen3_moe``, ``hunyuan_v1_dense``),
* support implied by listing rather than by evidence.
"""

from __future__ import annotations

import re
from typing import ClassVar

import pytest

from mobius._registry import _REGISTRATIONS
from mobius.integrations.gguf._arch_registry import (
    MMPROJ_ARCHITECTURE,
    get_arch_spec,
    iter_arch_specs,
    supported_architectures,
    try_get_arch_spec,
)
from mobius.integrations.gguf._config_mapping import (
    _CONFIG_POSTPROCESSORS,
    _KEY_MAP_TABLES,
    GGUF_ARCH_TO_MODEL_TYPE,
)
from mobius.integrations.gguf._errors import (
    DisabledGGUFArchitectureError,
    UnsupportedGGUFArchitectureError,
)
from mobius.integrations.gguf._mmproj import _VLM_BUILDERS
from mobius.integrations.gguf._spec import Support
from mobius.integrations.gguf._tensor_mapping import _MAPPING_TABLES
from mobius.integrations.gguf._tensor_processors import _PROCESSOR_IMPLS
from mobius.integrations.gguf._upstream import upstream_architectures

#: Number of importable architectures. Pinned so that adding support is a
#: deliberate act that also updates the documented support matrix, and so that
#: accidentally losing an architecture is a failure rather than a silence.
_EXPECTED_SUPPORTED_COUNT = 23


class TestCapabilityClosure:
    """The four capability verdicts must not contradict each other."""

    @pytest.mark.parametrize("spec", iter_arch_specs(), ids=lambda s: s.gguf_arch)
    def test_a_mappable_architecture_is_also_configurable_and_buildable(self, spec) -> None:
        """Tensor mapping is useless without a config and a graph to feed.

        ``bloom`` and ``t5`` used to fail this from the other direction: they
        were configurable but unmappable, so config extraction succeeded and the
        build then died with a message contradicting the config map.
        """
        if spec.tensor_map is not Support.SUPPORTED:
            return
        assert spec.config is Support.SUPPORTED, (
            f"{spec.gguf_arch}: tensor mapping is supported but config extraction "
            f"is {spec.config.value}"
        )
        assert spec.graph is Support.SUPPORTED, (
            f"{spec.gguf_arch}: tensor mapping is supported but graph construction "
            f"is {spec.graph.value}"
        )

    @pytest.mark.parametrize("spec", iter_arch_specs(), ids=lambda s: s.gguf_arch)
    def test_every_unsupported_capability_carries_a_reason(self, spec) -> None:
        """Support must never be denied silently."""
        unsupported = [
            name for name, verdict in spec.verdicts.items() if verdict is not Support.SUPPORTED
        ]
        if unsupported:
            assert spec.reason, f"{spec.gguf_arch}: {unsupported} lack a reason"

    @pytest.mark.parametrize("spec", iter_arch_specs(), ids=lambda s: s.gguf_arch)
    def test_a_buildable_architecture_resolves_in_the_mobius_registry(self, spec) -> None:
        """A ``graph=SUPPORTED`` claim has to be backed by a real model class."""
        if spec.graph is not Support.SUPPORTED:
            return
        assert spec.model_type in _REGISTRATIONS, (
            f"{spec.gguf_arch}: model_type {spec.model_type!r} is not registered in "
            "mobius._registry, so the graph cannot actually be built"
        )

    def test_the_supported_set_is_pinned(self) -> None:
        """Gaining or losing support is a deliberate, reviewable change."""
        assert len(supported_architectures()) == _EXPECTED_SUPPORTED_COUNT


class TestNameResolutionClosure:
    """Every named behavior resolves, and every implementation is reachable.

    This replaces an import edge with a test, which is what lets
    ``_arch_registry`` stay a leaf. It is also what catches a dead
    implementation: ``_PROCESSORS`` previously held ``gemma``, ``gemma3`` and
    ``mistral`` entries keyed by ``model_type``, of which ``gemma3`` was the one
    that mattered and never ran.
    """

    _TABLES: ClassVar[dict[str, tuple[object, bool]]] = {
        "tensor_map_recipe": (_MAPPING_TABLES, True),
        "config_key_map": (_KEY_MAP_TABLES, False),
        "config_postprocessor": (_CONFIG_POSTPROCESSORS, False),
        "tensor_processor": (_PROCESSOR_IMPLS, False),
        "vlm_builder": (_VLM_BUILDERS, False),
    }

    @staticmethod
    def _referenced(field: str, is_sequence: bool) -> set[str]:
        names: set[str] = set()
        for spec in iter_arch_specs():
            value = getattr(spec, field)
            if value is None:
                continue
            names.update(value if is_sequence else {value})
        return names

    @pytest.mark.parametrize("field", sorted(_TABLES))
    def test_every_referenced_name_resolves(self, field: str) -> None:
        table, is_sequence = self._TABLES[field]
        missing = self._referenced(field, is_sequence) - set(table)
        assert not missing, (
            f"specs reference {field} names {sorted(missing)} that no module provides"
        )

    @pytest.mark.parametrize("field", sorted(_TABLES))
    def test_every_implementation_is_referenced(self, field: str) -> None:
        table, is_sequence = self._TABLES[field]
        orphans = set(table) - self._referenced(field, is_sequence)
        assert not orphans, (
            f"{field} implementations {sorted(orphans)} are registered but no "
            "architecture uses them, so they are dead code"
        )


class TestNamespaceHygiene:
    """GGUF architecture strings and mobius model types are different namespaces."""

    @pytest.mark.parametrize("spec", iter_arch_specs(), ids=lambda s: s.gguf_arch)
    def test_canonical_names_are_real_upstream_architectures(self, spec) -> None:
        """A canonical name llama.cpp never writes can never match a real file.

        ``qwen2_moe``/``qwen3_moe``/``hunyuan_v1_dense`` were mobius model types
        sitting in the architecture namespace; the architectures llama.cpp
        actually emits (``qwen2moe``, ``qwen3moe``, ``hunyuan-dense``) either
        went unmapped or were unreachable.
        """
        assert spec.gguf_arch in upstream_architectures(), (
            f"{spec.gguf_arch!r} is not one of the 147 architectures llama.cpp "
            "defines at the pinned commit. If it is a defensive spelling, declare "
            "it in `aliases` instead of as the canonical name."
        )

    def test_aliases_do_not_collide(self) -> None:
        seen: dict[str, str] = {}
        for spec in iter_arch_specs():
            for name in spec.names:
                assert name not in seen or seen[name] == spec.gguf_arch, (
                    f"{name!r} is claimed by both {seen[name]!r} and {spec.gguf_arch!r}"
                )
                seen[name] = spec.gguf_arch

    @pytest.mark.parametrize("spec", iter_arch_specs(), ids=lambda s: s.gguf_arch)
    def test_alias_resolution_is_total(self, spec) -> None:
        for name in spec.names:
            assert try_get_arch_spec(name) is spec
            assert try_get_arch_spec(name.upper()) is spec


class TestDerivedViewsAgree:
    """Views built from the registry must stay consistent with it."""

    def test_model_type_map_covers_every_name_with_a_model_type(self) -> None:
        expected = {
            name: spec.model_type
            for spec in iter_arch_specs()
            if spec.model_type is not None
            for name in spec.names
        }
        assert dict(GGUF_ARCH_TO_MODEL_TYPE) == expected

    def test_model_type_map_is_read_only(self) -> None:
        with pytest.raises(TypeError):
            GGUF_ARCH_TO_MODEL_TYPE["nope"] = "nope"  # type: ignore[index]


class TestRejectionsAreActionable:
    """An unsupported input must say what it is and what to do instead."""

    @pytest.mark.parametrize("architecture", ["bloom", "t5"])
    def test_configurable_but_unmappable_architectures_are_refused(
        self, architecture: str
    ) -> None:
        with pytest.raises(UnsupportedGGUFArchitectureError, match="Unsupported GGUF"):
            get_arch_spec(architecture)

    @pytest.mark.parametrize("architecture", ["nemotron_h_moe", MMPROJ_ARCHITECTURE])
    def test_deliberately_disabled_architectures_raise_the_disabled_error(
        self, architecture: str
    ) -> None:
        with pytest.raises(DisabledGGUFArchitectureError):
            get_arch_spec(architecture)

    def test_a_dead_upstream_architecture_says_so(self) -> None:
        """``gptj`` is registered upstream but has no loader, so nothing can read it."""
        with pytest.raises(
            UnsupportedGGUFArchitectureError, match=re.escape("no llama.cpp model loader")
        ):
            get_arch_spec("gptj")

    def test_an_unimported_upstream_architecture_names_its_cohort(self) -> None:
        with pytest.raises(UnsupportedGGUFArchitectureError, match="C05-pure-recurrent"):
            get_arch_spec("mamba2")

    def test_an_unknown_architecture_is_distinguished_from_an_upstream_one(self) -> None:
        with pytest.raises(UnsupportedGGUFArchitectureError, match="not among the 147"):
            get_arch_spec("definitely-not-real")

    def test_legacy_exception_types_still_catch_everything(self) -> None:
        """Callers written against the pre-registry errors keep working."""
        with pytest.raises(ValueError):
            get_arch_spec("definitely-not-real")
        with pytest.raises(NotImplementedError):
            get_arch_spec("nemotron_h_moe")
