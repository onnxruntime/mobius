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

import inspect
import pathlib
import re
from types import SimpleNamespace
from typing import ClassVar

import pytest

from mobius._registry import _REGISTRATIONS
from mobius.integrations.gguf._arch_registry import (
    MMPROJ_ARCHITECTURE,
    _validate_census_closure,
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
from mobius.integrations.gguf._tensor_mapping import (
    _MAPPING_TABLES,
    _build_mapping,
    is_known_skip,
    map_gguf_to_hf_names,
)
from mobius.integrations.gguf._tensor_processors import (
    _PROCESSOR_IMPLS,
    PACKED_SAFE_PROCESSORS,
)
from mobius.integrations.gguf._upstream import upstream_architectures

#: Number of importable architectures. Pinned so that adding support is a
#: deliberate act that also updates the documented support matrix, and so that
#: accidentally losing an architecture is a failure rather than a silence.
_EXPECTED_SUPPORTED_COUNT = 94
_PROMOTED_CONVENTIONAL_DECODERS = frozenset(
    {
        "bitnet",
        "codeshell",
        "command-r",
        "ernie4_5",
        "gptneox",
        "granite",
        "jais",
        "jais2",
        "minicpm",
        "mpt",
        "openelm",
        "orion",
        "pangu-embedded",
        "plm",
        "plamo",
        "qwen",
        "refact",
        "gemma-embedding",
        "llama-embed",
        "maincoder",
        "starcoder",
        "xverse",
    }
)
_FINAL_CENSUS_CLOSURE = frozenset(
    {
        "afmoe",
        "bailingmoe2",
        "bitnet",
        "codeshell",
        "cohere2moe",
        "command-r",
        "deepseek2",
        "deepseek32",
        "dots3note",
        "ernie4_5",
        "exaone-moe",
        "exaone4",
        "gemma-embedding",
        "gemma4-assistant",
        "glm4",
        "glm4moe",
        "gptj",
        "gptneox",
        "granite",
        "granite_swa",
        "graniteswitch",
        "hunyuan-moe",
        "hy_v3",
        "jais",
        "jais2",
        "laguna",
        "llama-embed",
        "maincoder",
        "mellum",
        "mimo2",
        "minicpm",
        "minimax-m2",
        "minimax-m3",
        "mistral4",
        "nanbeige",
        "orion",
        "pangu-embedded",
        "plamo",
        "plamo3",
        "plm",
        "qwen",
        "refact",
        "starcoder",
        "step35",
        "xverse",
    }
)
_QKV_HELPER_TENSOR_NAMES = frozenset(
    {
        "blk.{bid}.attn_qkv.weight",
        "blk.{bid}.attn_qkv.bias",
        "blk.{bid}.attn_q.weight",
        "blk.{bid}.attn_q.bias",
        "blk.{bid}.attn_k.weight",
        "blk.{bid}.attn_k.bias",
        "blk.{bid}.attn_v.weight",
        "blk.{bid}.attn_v.bias",
    }
)
_GATE_UP_EXPS_HELPER_TENSOR_NAMES = frozenset(
    {
        "blk.{bid}.ffn_gate_up_exps.weight",
        "blk.{bid}.ffn_gate_exps.weight",
        "blk.{bid}.ffn_up_exps.weight",
    }
)
_FINAL_CENSUS_SUFFIXLESS_TENSORS = frozenset(
    {
        ("gemma4-assistant", "masked_embd_ordering"),
        ("hy_v3", "blk.{bid}.exp_probs_b"),
        ("plamo3", "blk.{bid}.post_attention_norm"),
        ("plamo3", "blk.{bid}.post_ffw_norm"),
    }
)

# Quantized reachability is separately pinned from float importability. A new
# architecture must explicitly prove that its graph exposes packed projection
# modules before joining this set.
_EXPECTED_QUANTIZED_IMPORT_ARCHITECTURES = frozenset(
    {
        "apertus",
        "arcee",
        "baichuan",
        "bailingmoe",
        "bert",
        "cohere2",
        "deci",
        "deepseek",
        "dots1",
        "dflash",
        "dream",
        "eagle3",
        "exaone",
        "falcon",
        "falcon-h1",
        "gemma",
        "gemma2",
        "gemma3",
        "gemma4",
        "granite",
        "granitemoe",
        "hunyuan-dense",
        "jamba",
        "jais2",
        "kimi-k3",
        "kimi-linear",
        "lfm2",
        "llada",
        "llada-moe",
        "llama",
        "minicpm",
        "minicpm3",
        "minimax-01",
        "modern-bert",
        "muse-glimmer",
        "nemotron",
        "olmo",
        "olmo2",
        "olmoe",
        "pangu-embedded",
        "phi3",
        "phimoe",
        "plamo2",
        "plm",
        "qwen2",
        "qwen2vl",
        "qwen2moe",
        "qwen3",
        "qwen35",
        "qwen35moe",
        "qwen3moe",
        "qwen3next",
        "rnd1",
        "seed_oss",
        "command-r",
        "smollm3",
        "stablelm",
        "starcoder2",
        "t5",
        "t5encoder",
    }
)


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
        if spec.preflight_only:
            assert spec.graph is not Support.SUPPORTED
            return
        assert spec.graph is Support.SUPPORTED, (
            f"{spec.gguf_arch}: tensor mapping is supported but graph construction "
            f"is {spec.graph.value}"
        )

    @pytest.mark.parametrize("spec", iter_arch_specs(), ids=lambda s: s.gguf_arch)
    def test_every_unsupported_capability_carries_a_reason(self, spec) -> None:
        """Support must never be denied silently."""
        unsupported = [
            name
            for name, verdict in spec.capabilities.items()
            if verdict is not Support.SUPPORTED
        ]
        if unsupported:
            assert spec.reason, f"{spec.gguf_arch}: {unsupported} lack a reason"

    @pytest.mark.parametrize("spec", iter_arch_specs(), ids=lambda s: s.gguf_arch)
    def test_a_buildable_architecture_resolves_in_the_mobius_registry(self, spec) -> None:
        """A ``graph=SUPPORTED`` claim has to be backed by a real model class."""
        if spec.graph is not Support.SUPPORTED:
            return
        module_type = spec.module_type or spec.model_type
        assert module_type in _REGISTRATIONS, (
            f"{spec.gguf_arch}: module type {module_type!r} is not registered in "
            "mobius._registry, so the graph cannot actually be built"
        )

    def test_the_supported_set_is_pinned(self) -> None:
        """Gaining or losing support is a deliberate, reviewable change."""
        assert len(supported_architectures()) == _EXPECTED_SUPPORTED_COUNT

    def test_falcon_h1_is_not_a_generic_falcon_alias(self) -> None:
        """Falcon-H1 resolves only to its dedicated parallel hybrid graph."""
        spec = try_get_arch_spec("falcon-h1")
        assert spec is not None
        assert spec.model_type == "falcon_h1"
        assert spec.graph is Support.SUPPORTED
        assert spec.runtime is Support.DEFERRED
        assert _REGISTRATIONS["falcon_h1"].module_class.__name__ == "FalconH1ForCausalLM"

    def test_quantized_import_set_is_pinned(self) -> None:
        """Builder acceptance and rejection must come from an explicit policy set."""
        actual = frozenset(
            spec.gguf_arch
            for spec in iter_arch_specs()
            if spec.is_importable and spec.quantized_import is Support.SUPPORTED
        )
        assert actual == _EXPECTED_QUANTIZED_IMPORT_ARCHITECTURES

    def test_every_float_importable_architecture_has_a_quantized_verdict(self) -> None:
        """Float graph support must not be mistaken for quantized graph reachability."""
        actual = {
            spec.gguf_arch: spec.quantized_import
            for spec in iter_arch_specs()
            if spec.is_importable
        }
        assert set(actual) == set(supported_architectures())
        rejected = {
            "bitnet",
            "chatglm",
            "ernie4_5",
            "ernie4_5-moe",
            "eurobert",
            "gemma-embedding",
            "granitehybrid",
            "internlm2",
            "jina-bert-v2",
            "jina-bert-v3",
            "llama-embed",
            "lfm2moe",
            "mamba",
            "mamba2",
            "maincoder",
            "nemotron_h",
            "nemotron_h_moe",
            "neo-bert",
            "nomic-bert",
            "nomic-bert-moe",
            "arctic",
            "dbrx",
            "phi2",
            "bloom",
            "gpt2",
            "codeshell",
            "gptneox",
            "jais",
            "mpt",
            "openelm",
            "orion",
            "plamo",
            "qwen",
            "refact",
            "starcoder",
            "smallthinker",
            "talkie",
            "xverse",
        }
        assert all(actual[arch] is Support.REJECTED for arch in rejected)
        assert all(
            verdict is Support.SUPPORTED
            for arch, verdict in actual.items()
            if arch not in rejected
        )


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

    def test_quantized_import_never_skips_required_tensor_processor(self) -> None:
        unsafe = {
            spec.gguf_arch: spec.tensor_processor
            for spec in iter_arch_specs()
            if spec.quantized_import is Support.SUPPORTED
            and spec.tensor_processor is not None
            and spec.tensor_processor not in PACKED_SAFE_PROCESSORS
        }
        assert unsafe == {}, (
            "keep_quantized=True bypasses float tensor processors. Mark these "
            "architectures quantized_import=REJECTED or implement an exact packed-domain "
            f"transform for weight/scales/zero-points: {unsafe}"
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
            f"{spec.gguf_arch!r} is not one of the 148 pinned GGUF formats. "
            "If it is a defensive spelling, declare it in `aliases` instead of "
            "as the canonical name."
        )

    def test_aliases_do_not_collide(self) -> None:
        seen: dict[str, str] = {}
        for spec in iter_arch_specs():
            for name in spec.names:
                assert name not in seen or seen[name] == spec.gguf_arch, (
                    f"{name!r} is claimed by both {seen[name]!r} and {spec.gguf_arch!r}"
                )
                seen[name] = spec.gguf_arch

    def test_registry_exactly_closes_the_pinned_census(self) -> None:
        specs = iter_arch_specs()
        _validate_census_closure(specs, frozenset(upstream_architectures()))
        assert len(specs) == 148
        assert {spec.gguf_arch for spec in specs} == set(upstream_architectures())

    @pytest.mark.parametrize(
        ("mutation", "match"),
        [
            ("missing", "missing="),
            ("extra", "extra="),
            ("duplicate", "duplicates="),
            ("alias-drift", "aliases_that_became_canonical="),
        ],
    )
    def test_census_mutation_is_detected(self, mutation: str, match: str) -> None:
        specs = iter_arch_specs()
        upstream = set(upstream_architectures())
        if mutation == "missing":
            specs = specs[:-1]
        elif mutation == "extra":
            upstream.remove(specs[0].gguf_arch)
        elif mutation == "duplicate":
            specs = (*specs, specs[0])
        else:
            upstream.add("mistral")
        with pytest.raises(ValueError, match=match):
            _validate_census_closure(tuple(specs), frozenset(upstream))

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


class TestPinnedTensorClosure:
    """Every pinned source tensor must map or be an intentional computed skip."""

    _NEW_ARCHITECTURES = (
        "olmo",
        "olmo2",
        "cohere2",
        "arcee",
        "smollm3",
        "exaone",
        "olmoe",
        "phimoe",
        "qwen2moe",
        "qwen3moe",
        "granitemoe",
        "mamba",
        "mamba2",
        "bert",
        "modern-bert",
        "t5",
        "t5encoder",
        "dream",
        "llada",
        "llada-moe",
        "rnd1",
        "baichuan",
        "chatglm",
        "phi2",
        "seed_oss",
    )

    @staticmethod
    def _unmapped(architecture: str) -> list[str]:
        upstream = upstream_architectures()[architecture]
        unmapped = []
        names = (
            tuple((name, name) for name in upstream.tensor_names)
            if upstream.tensor_names
            else tuple((family + ".weight", family) for family in upstream.tensor_families)
        )
        for pinned_name, label in names:
            name = pinned_name.replace("{bid}", "0")
            if map_gguf_to_hf_names(name, architecture) is None and not is_known_skip(name):
                unmapped.append(label)
        return unmapped

    @pytest.mark.parametrize("architecture", _NEW_ARCHITECTURES)
    def test_every_pinned_tensor_family_closes(self, architecture: str) -> None:
        assert not self._unmapped(architecture)

    @pytest.mark.parametrize(
        "architecture",
        [
            "olmoe",
            "phimoe",
            "qwen2moe",
            "qwen3moe",
            "granitemoe",
            "llada-moe",
            "rnd1",
        ],
    )
    def test_pinned_expert_suffixes_close_without_drops(self, architecture: str) -> None:
        upstream = upstream_architectures()[architecture]
        assert upstream.expert_tensor_suffixes == ("weight", "scale", "input_scale")
        expert_families = [
            family for family in upstream.tensor_families if family.endswith("_exps")
        ]
        assert expert_families
        for family in expert_families:
            for suffix in upstream.expert_tensor_suffixes:
                name = family.replace("{bid}", "0") + f".{suffix}"
                assert map_gguf_to_hf_names(name, architecture) is not None, name

    @pytest.mark.parametrize("architecture", ["dream", "llada", "llada-moe", "rnd1"])
    @pytest.mark.parametrize(
        "malformed",
        [
            "token_embd.scale",
            "blk.0.attn_q.weight.extra",
            "blk.0.diffusion_timestep.weight",
            "blk.0.noise_schedule.weight",
        ],
    )
    def test_diffusion_tensor_closure_rejects_unpinned_suffixes(
        self, architecture: str, malformed: str
    ) -> None:
        assert map_gguf_to_hf_names(malformed, architecture) is None

    @pytest.mark.parametrize("architecture", ["llada-moe", "rnd1"])
    @pytest.mark.parametrize(
        "malformed",
        [
            "blk.0.ffn_gate_inp.scale",
            "blk.0.ffn_gate_exps.scales",
            "blk.0.ffn_down_exps.zero_point",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
        ],
    )
    def test_diffusion_moe_rejects_invalid_expert_sidecars(
        self, architecture: str, malformed: str
    ) -> None:
        assert map_gguf_to_hf_names(malformed, architecture) is None

    def test_deleting_one_mapping_breaks_closure(self) -> None:
        """Falsify the support claim rather than only testing the happy path."""
        olmo_mapping = _MAPPING_TABLES["olmo"]
        removed = olmo_mapping.pop("blk.{bid}.attn_q")
        _build_mapping.cache_clear()
        try:
            assert self._unmapped("olmo") == ["blk.{bid}.attn_q"]
        finally:
            olmo_mapping["blk.{bid}.attn_q"] = removed
            _build_mapping.cache_clear()

    @pytest.mark.parametrize(
        ("architecture", "mapping_key"),
        [
            ("bert", "blk.{bid}.attn_q"),
            ("bert", "blk.{bid}.attn_qkv"),
            ("modern-bert", "blk.{bid}.attn_qkv"),
        ],
    )
    def test_deleting_encoder_mapping_breaks_closure(
        self, architecture: str, mapping_key: str
    ) -> None:
        table_name = "bert" if architecture == "bert" else "modern_bert"
        mapping = _MAPPING_TABLES[table_name]
        removed = mapping.pop(mapping_key)
        _build_mapping.cache_clear()
        try:
            assert any(name.startswith(mapping_key) for name in self._unmapped(architecture))
        finally:
            mapping[mapping_key] = removed
            _build_mapping.cache_clear()

    @pytest.mark.parametrize(
        ("architecture", "mapping_key"),
        [
            ("t5", "dec.blk.{bid}.cross_attn_q"),
            ("t5encoder", "enc.blk.{bid}.attn_rel_b"),
        ],
    )
    def test_deleting_t5_mapping_breaks_closure(
        self, architecture: str, mapping_key: str
    ) -> None:
        mapping = _MAPPING_TABLES["t5"]
        removed = mapping.pop(mapping_key)
        _build_mapping.cache_clear()
        try:
            assert any(name.startswith(mapping_key) for name in self._unmapped(architecture))
        finally:
            mapping[mapping_key] = removed
            _build_mapping.cache_clear()

    def test_deleting_expert_mapping_breaks_moe_closure(self) -> None:
        moe_mapping = _MAPPING_TABLES["moe_extras"]
        removed = moe_mapping.pop("blk.{bid}.ffn_gate_exps")
        _build_mapping.cache_clear()
        try:
            assert "blk.{bid}.ffn_gate_exps" in self._unmapped("qwen3moe")
        finally:
            moe_mapping["blk.{bid}.ffn_gate_exps"] = removed
            _build_mapping.cache_clear()

    _RECURRENT_TENSOR_NAMES: ClassVar[dict[str, set[str]]] = {
        "mamba": {
            "token_embd.weight",
            "output_norm.weight",
            "output.weight",
            "blk.{bid}.attn_norm.weight",
            "blk.{bid}.ssm_in.weight",
            "blk.{bid}.ssm_conv1d.weight",
            "blk.{bid}.ssm_conv1d.bias",
            "blk.{bid}.ssm_x.weight",
            "blk.{bid}.ssm_dt.weight",
            "blk.{bid}.ssm_dt.bias",
            "blk.{bid}.ssm_a",
            "blk.{bid}.ssm_d",
            "blk.{bid}.ssm_out.weight",
        },
        "mamba2": {
            "token_embd.weight",
            "output_norm.weight",
            "output.weight",
            "blk.{bid}.attn_norm.weight",
            "blk.{bid}.ssm_in.weight",
            "blk.{bid}.ssm_conv1d.weight",
            "blk.{bid}.ssm_conv1d.bias",
            "blk.{bid}.ssm_dt.bias",
            "blk.{bid}.ssm_a",
            "blk.{bid}.ssm_d",
            "blk.{bid}.ssm_norm.weight",
            "blk.{bid}.ssm_out.weight",
        },
    }

    @pytest.mark.parametrize("architecture", ["mamba", "mamba2"])
    def test_pure_recurrent_pinned_tensor_names_are_suffix_exact(
        self, architecture: str
    ) -> None:
        upstream = upstream_architectures()[architecture]
        assert set(upstream.tensor_names) == self._RECURRENT_TENSOR_NAMES[architecture]
        for pinned_name in upstream.tensor_names:
            name = pinned_name.replace("{bid}", "7")
            assert map_gguf_to_hf_names(name, architecture) is not None, name

    @pytest.mark.parametrize(
        ("architecture", "malformed"),
        [
            ("mamba", "blk.0.ssm_a.weight"),
            ("mamba", "blk.0.ssm_d.bias"),
            ("mamba", "blk.0.ssm_in.bias"),
            ("mamba2", "blk.0.ssm_dt.weight"),
            ("mamba2", "blk.0.ssm_a.bias"),
            ("mamba2", "blk.0.ssm_norm.bias"),
        ],
    )
    def test_pure_recurrent_malformed_suffixes_do_not_map(
        self, architecture: str, malformed: str
    ) -> None:
        assert map_gguf_to_hf_names(malformed, architecture) is None

    @pytest.mark.parametrize(
        ("architecture", "mapping_key"),
        [
            ("mamba", "blk.{bid}.ssm_x"),
            ("mamba2", "blk.{bid}.ssm_norm"),
        ],
    )
    def test_deleting_recurrent_mapping_breaks_closure(
        self, architecture: str, mapping_key: str
    ) -> None:
        mapping = _MAPPING_TABLES[architecture]
        removed = mapping.pop(mapping_key)
        _build_mapping.cache_clear()
        try:
            assert any(name.startswith(mapping_key) for name in self._unmapped(architecture))
        finally:
            mapping[mapping_key] = removed
            _build_mapping.cache_clear()


class TestFinalCensusClosure:
    @pytest.mark.parametrize(
        "architecture", sorted(_FINAL_CENSUS_CLOSURE - _PROMOTED_CONVENTIONAL_DECODERS)
    )
    def test_every_newly_closed_id_has_one_nonimportable_spec(self, architecture: str) -> None:
        spec = try_get_arch_spec(architecture)
        assert spec is not None
        assert spec.gguf_arch == architecture
        assert spec.model_type is None
        assert spec.aliases == frozenset()
        assert all(verdict is not Support.SUPPORTED for verdict in spec.verdicts.values())
        assert spec.reason

    @pytest.mark.parametrize(
        "architecture",
        sorted(_FINAL_CENSUS_CLOSURE - {"gptj"}),
    )
    def test_loader_and_converter_audits_are_pinned(self, architecture: str) -> None:
        upstream = upstream_architectures()[architecture]
        assert upstream.cpp_loader
        assert upstream.loader_source.startswith("src/models/")
        assert upstream.tensor_names
        assert upstream.tensor_families
        assert upstream.required_metadata
        assert upstream.tensor_closure_status == "audited-direct-loader-conditional-union"
        assert (
            upstream.converter_inventory_status
            == "exact-pinned-MODEL_TENSORS-family-inventory"
        )
        assert all(
            name.endswith((".weight", ".bias", ".scale", ".input_scale", ".lora_a", ".lora_b"))
            or (architecture, name) in _FINAL_CENSUS_SUFFIXLESS_TENSORS
            for name in upstream.tensor_names
        )

    def test_shared_loader_helpers_are_fully_expanded(self) -> None:
        expected = {
            "create_tensor_qkv": _QKV_HELPER_TENSOR_NAMES,
            "create_tensor_gate_up_exps": _GATE_UP_EXPS_HELPER_TENSOR_NAMES,
        }
        for architecture in sorted(_FINAL_CENSUS_CLOSURE):
            upstream = upstream_architectures()[architecture]
            for helper in upstream.loader_helpers:
                assert helper in expected
                assert expected[helper] <= set(upstream.tensor_names), (
                    f"{architecture} does not contain the full {helper} conditional union"
                )

    def test_suffixless_loader_tensors_are_exact(self) -> None:
        suffixless = {
            (architecture, name)
            for architecture in _FINAL_CENSUS_CLOSURE
            for name in upstream_architectures()[architecture].tensor_names
            if not name.endswith(
                (".weight", ".bias", ".scale", ".input_scale", ".lora_a", ".lora_b")
            )
        }
        assert suffixless == _FINAL_CENSUS_SUFFIXLESS_TENSORS

    def test_step35_metadata_reads_are_exact(self) -> None:
        upstream = upstream_architectures()["step35"]
        assert set(upstream.required_metadata) == {
            "attention.layer_norm_rms_epsilon",
            "attention.sliding_window",
            "attention.sliding_window_pattern",
            "expert_feed_forward_length",
        }
        assert set(upstream.optional_metadata) == {
            "expert_gating_func",
            "expert_shared_feed_forward_length",
            "expert_weights_norm",
            "expert_weights_scale",
            "nextn_predict_layers",
            "rope.freq_base_swa",
            "swiglu_clamp_exp",
            "swiglu_clamp_shexp",
        }

    @pytest.mark.parametrize(
        ("architecture", "optional_keys"),
        [
            ("granite", {"expert_shared_feed_forward_length"}),
            ("granite_swa", {"expert_shared_feed_forward_length"}),
            (
                "graniteswitch",
                {"adapters.router_gain", "expert_shared_feed_forward_length"},
            ),
            ("minicpm", {"embedding_scale", "logit_scale", "residual_scale"}),
        ],
    )
    def test_defaulted_metadata_is_not_classified_as_required(
        self, architecture: str, optional_keys: set[str]
    ) -> None:
        upstream = upstream_architectures()[architecture]
        assert optional_keys <= set(upstream.optional_metadata)
        assert optional_keys.isdisjoint(upstream.required_metadata)

    def test_array_metadata_reads_are_complete_and_classified(self) -> None:
        expected = {
            "dots3note": (
                {"attention.indexer.types", "attention.sliding_window_pattern"},
                set(),
            ),
            "granite": (set(), {"deepstack_mapping"}),
            "granite_swa": (
                {"attention.sliding_window_pattern"},
                {"attention.rope_pattern", "deepstack_mapping"},
            ),
            "graniteswitch": (
                {"adapters.token_ids_activate", "adapters.token_ids_substitute"},
                set(),
            ),
        }
        actual = {
            architecture: set(upstream_architectures()[architecture].array_metadata)
            for architecture in expected
        }
        assert actual == {
            architecture: required | optional
            for architecture, (required, optional) in expected.items()
        }
        for architecture, (required, optional) in expected.items():
            upstream = upstream_architectures()[architecture]
            assert required <= set(upstream.required_metadata)
            assert required.isdisjoint(upstream.optional_metadata)
            assert optional <= set(upstream.optional_metadata)
            assert optional.isdisjoint(upstream.required_metadata)

    def test_dead_gptj_id_is_explicitly_rejected(self) -> None:
        upstream = upstream_architectures()["gptj"]
        assert not upstream.cpp_loader
        assert upstream.tensor_closure_status == "no-loader"
        assert upstream.converter_inventory_status == "no-converter"
        with pytest.raises(DisabledGGUFArchitectureError, match="no model loader"):
            get_arch_spec("gptj")

    @pytest.mark.parametrize("architecture", ["arwkv7", "rwkv6", "rwkv6qwen2", "rwkv7"])
    def test_rwkv_variants_have_distinct_state_abi_reasons(self, architecture: str) -> None:
        spec = try_get_arch_spec(architecture)
        assert spec is not None and spec.reason is not None
        assert "state" in spec.reason.lower()
        assert "Mamba" in spec.reason
        upstream = upstream_architectures()[architecture]
        assert upstream.tensor_names
        assert upstream.tensor_families
        assert upstream.required_metadata


class TestPinnedAudioCohort:
    """Pin the audited C09 inventories without implying graph support."""

    _LAYER_COUNTS: ClassVar[dict[str, int]] = {
        "pockettts": 24,
        "qwen3tts": 28,
        "talkie": 40,
        # wavtokenizer's only {bid} family is the ConvNeXt stack. PosNet's
        # heterogeneous six-layer schedule is enumerated literally in the pin.
        "wavtokenizer-dec": 12,
    }
    _EXPECTED_COUNTS: ClassVar[dict[str, int]] = {
        "pockettts": 3 + 10 * 24,
        "qwen3tts": 3 + 11 * 28,
        "talkie": 2 + 9 * 40,
        "wavtokenizer-dec": 161,
    }

    @classmethod
    def _expanded_patterns(cls, architecture: str, field: str) -> set[str]:
        names: set[str] = set()
        for pattern in getattr(upstream_architectures()[architecture], field):
            if "{bid}" in pattern:
                names.update(
                    pattern.replace("{bid}", str(index))
                    for index in range(cls._LAYER_COUNTS[architecture])
                )
            else:
                names.add(pattern)
        return names

    @classmethod
    def _expanded_names(cls, architecture: str) -> set[str]:
        return cls._expanded_patterns(architecture, "tensor_names")

    @pytest.mark.parametrize("architecture", sorted(_LAYER_COUNTS))
    def test_loader_inventory_is_suffix_exact_and_complete(self, architecture: str) -> None:
        names = self._expanded_names(architecture)
        assert len(names) == self._EXPECTED_COUNTS[architecture]
        assert all(name.endswith((".weight", ".bias")) for name in names)

    def test_pockettts_inventory_has_no_semantic_output_head(self) -> None:
        names = self._expanded_names("pockettts")
        assert "token_embd.weight" in names
        assert "output.weight" not in names
        assert "blk.0.attn_q.bias" not in names

    def test_qwen3tts_inventory_is_the_transformed_talker_only(self) -> None:
        names = self._expanded_names("qwen3tts")
        assert "output.weight" in names
        assert "blk.0.attn_q_norm.weight" in names
        assert not any(name.startswith("a.gen.") for name in names)

    def test_talkie_loader_and_converter_closures_are_distinct(self) -> None:
        loader_names = self._expanded_names("talkie")
        converter_extras = self._expanded_patterns("talkie", "converter_extra_tensor_names")

        assert len(loader_names) == 362
        assert len(converter_extras) == 80
        assert len(loader_names | converter_extras) == 442
        assert not loader_names & converter_extras
        assert "blk.0.attn_output.scale" in converter_extras
        assert "blk.0.ffn_down.scale" in converter_extras
        assert "blk.0.layer_output_scale.weight" in loader_names
        assert "blk.0.attn_output.input_scale" not in converter_extras

    def test_wavtokenizer_heterogeneous_stacks_are_literal(self) -> None:
        names = self._expanded_names("wavtokenizer-dec")
        assert "posnet.2.attn_q.weight" in names
        assert "posnet.5.attn_norm.weight" in names
        assert "posnet.5.norm1.weight" not in names
        assert "convnext.11.gamma.weight" in names
        assert "convnext.12.gamma.weight" not in names

    @pytest.mark.parametrize(
        "architecture",
        [
            "pockettts",
            "qwen3tts",
            "wavtokenizer-dec",
        ],
    )
    def test_task_misdispatch_is_refused(self, architecture: str) -> None:
        spec = try_get_arch_spec(architecture)
        assert spec is not None
        assert spec.model_type is None
        assert spec.graph is not Support.SUPPORTED
        assert all(verdict is Support.DEFERRED for verdict in spec.verdicts.values())
        with pytest.raises(UnsupportedGGUFArchitectureError):
            get_arch_spec(architecture)


class TestPinnedRemainingHybridCohort:
    """Pin C06 loader closure while refusing incompatible state/task ABIs."""

    _ARCHITECTURES = (
        "bailingmoe3",
        "deepseek4",
    )
    _EXPECTED_TENSOR_COUNTS: ClassVar[dict[str, int]] = {
        "bailingmoe3": 41,
        "deepseek4": 44,
    }

    @pytest.mark.parametrize("architecture", _ARCHITECTURES)
    def test_loader_inventory_is_suffix_exact(self, architecture: str) -> None:
        upstream = upstream_architectures()[architecture]
        names = set(upstream.tensor_names)
        assert len(names) == self._EXPECTED_TENSOR_COUNTS[architecture]
        assert upstream.expert_tensor_suffixes == ("weight",)
        assert all(name.endswith((".weight", ".bias", ".ssm_a")) for name in names)
        assert not any(name.endswith((".scale", ".input_scale")) for name in names)

    def test_conditional_tensor_representations_are_not_conflated(self) -> None:
        bailing = set(upstream_architectures()["bailingmoe3"].tensor_names)
        deepseek = set(upstream_architectures()["deepseek4"].tensor_names)
        kimi_k3 = set(upstream_architectures()["kimi-k3"].tensor_names)
        kimi_linear = set(upstream_architectures()["kimi-linear"].tensor_names)
        lfm2moe = set(upstream_architectures()["lfm2moe"].tensor_names)

        assert "blk.{bid}.nextn.eh_proj.weight" in bailing
        assert "blk.{bid}.ffn_gate_tid2eid.weight" in deepseek
        assert "blk.{bid}.indexer_compressor_kv.weight" in deepseek
        assert "blk.{bid}.ffn_routed_down.weight" in kimi_k3
        assert "blk.{bid}.attn_res_score.weight" in kimi_k3
        assert "blk.{bid}.ssm_g_b.weight" in kimi_linear
        assert "blk.{bid}.ssm_g_b.weight" not in kimi_k3
        assert "blk.{bid}.shortconv.conv.weight" in lfm2moe
        assert "blk.{bid}.ffn_gate_shexp.weight" not in lfm2moe

    @pytest.mark.parametrize("architecture", _ARCHITECTURES)
    def test_no_unpinned_alias_or_config_mutation_is_reachable(
        self, architecture: str
    ) -> None:
        spec = try_get_arch_spec(architecture)
        assert spec is not None
        assert not spec.aliases
        assert spec.model_type is None
        assert spec.config_key_map is None
        assert all(verdict is Support.DEFERRED for verdict in spec.verdicts.values())
        assert architecture not in GGUF_ARCH_TO_MODEL_TYPE

    @pytest.mark.parametrize(
        ("architecture", "state_terms"),
        [
            ("bailingmoe3", ("convolution histories", "matrix state", "NextN")),
            ("deepseek4", ("compressed-cache", "rollback", "ordinary KV")),
        ],
    )
    def test_state_and_schedule_mismatch_is_explicit(
        self, architecture: str, state_terms: tuple[str, ...]
    ) -> None:
        reason = try_get_arch_spec(architecture).reason
        assert reason is not None
        for term in state_terms:
            assert term in reason

    def test_lfm2moe_graph_and_import_advance_but_runtime_stays_deferred(self) -> None:
        spec = try_get_arch_spec("lfm2moe")
        assert spec is not None
        assert spec.model_type == "lfm2_moe"
        assert spec.config is Support.SUPPORTED
        assert spec.tensor_map is Support.SUPPORTED
        assert spec.graph is Support.SUPPORTED
        assert spec.runtime is Support.DEFERRED
        assert spec.quantized_import is Support.REJECTED
        assert spec.reason is not None
        assert "representative real-weight GGUF" in spec.reason
        assert "keep_quantized=False" in spec.reason

    def test_kimi_linear_graph_and_import_advance_but_runtime_stays_deferred(self) -> None:
        spec = try_get_arch_spec("kimi-linear")
        assert spec is not None
        assert spec.model_type == "kimi_linear"
        assert spec.config is Support.SUPPORTED
        assert spec.tensor_map is Support.SUPPORTED
        assert spec.graph is Support.SUPPORTED
        assert spec.runtime is Support.DEFERRED
        assert spec.config_key_map == "kimi_linear"
        assert spec.tensor_processor == "kimi_linear"

    def test_kimi_k3_graph_and_import_advance_but_runtime_stays_deferred(self) -> None:
        spec = try_get_arch_spec("kimi-k3")
        assert spec is not None
        assert spec.model_type == "kimi_k3"
        assert spec.config is Support.SUPPORTED
        assert spec.tensor_map is Support.SUPPORTED
        assert spec.graph is Support.SUPPORTED
        assert spec.runtime is Support.DEFERRED
        assert spec.config_key_map == "kimi_k3"
        assert spec.tensor_processor == "kimi_k3"
        assert spec.reason is not None
        assert "heterogeneous KV plus convolution/matrix state ABI" in spec.reason

    def test_hugging_face_deepseek_v4_registration_remains_valid(self) -> None:
        assert "deepseek_v4" in _REGISTRATIONS
        assert try_get_arch_spec("deepseek4").model_type is None


class TestPinnedRemainingConventionalMoECohort:
    """Pin the bounded C02 closure without aliasing incompatible MoE graphs."""

    _ARCHITECTURES = (
        "arctic",
        "dbrx",
        "gpt-oss",
        "grok",
        "grovemoe",
    )
    _DEFERRED_ARCHITECTURES = tuple(
        architecture
        for architecture in _ARCHITECTURES
        if architecture not in {"arctic", "dbrx"}
    )
    _EXPECTED_TENSOR_COUNTS: ClassVar[dict[str, int]] = {
        "arctic": 22,
        "dbrx": 11,
        "gpt-oss": 24,
        "grok": 24,
        "grovemoe": 23,
    }

    @pytest.mark.parametrize("architecture", _DEFERRED_ARCHITECTURES)
    def test_loader_inventory_and_expert_sidecars_are_suffix_exact(
        self, architecture: str
    ) -> None:
        upstream = upstream_architectures()[architecture]
        names = set(upstream.tensor_names)
        assert len(names) == self._EXPECTED_TENSOR_COUNTS[architecture]
        assert upstream.expert_tensor_suffixes == ()
        suffixes = dict(upstream.tensor_suffixes)
        expected_families = {
            "output",
            "blk.{bid}.attn_qkv",
            "blk.{bid}.attn_output",
            "blk.{bid}.ffn_gate_exps",
            "blk.{bid}.ffn_down_exps",
            "blk.{bid}.ffn_up_exps",
        }
        if architecture != "dbrx":
            expected_families.update(
                {
                    "blk.{bid}.attn_q",
                    "blk.{bid}.attn_k",
                    "blk.{bid}.attn_v",
                }
            )
        if architecture in {"arctic", "grok"}:
            expected_families.update(
                {
                    "blk.{bid}.ffn_gate",
                    "blk.{bid}.ffn_down",
                    "blk.{bid}.ffn_up",
                }
            )
        if architecture == "grovemoe":
            expected_families.update(
                {
                    "blk.{bid}.ffn_gate_chexps",
                    "blk.{bid}.ffn_down_chexps",
                    "blk.{bid}.ffn_up_chexps",
                }
            )
        assert set(suffixes) == expected_families
        projection_suffixes = ("weight", "scale", "input_scale")
        biased_projection_suffixes = ("weight", "bias", "scale", "input_scale")
        assert suffixes["blk.{bid}.attn_qkv"] == (
            projection_suffixes if architecture == "dbrx" else biased_projection_suffixes
        )
        for family in ("blk.{bid}.attn_q", "blk.{bid}.attn_k", "blk.{bid}.attn_v"):
            if architecture != "dbrx":
                assert suffixes[family] == biased_projection_suffixes
        assert suffixes["blk.{bid}.attn_output"] == (
            biased_projection_suffixes if architecture == "gpt-oss" else projection_suffixes
        )
        for family in (
            "blk.{bid}.ffn_gate_exps",
            "blk.{bid}.ffn_down_exps",
            "blk.{bid}.ffn_up_exps",
        ):
            expected = (
                biased_projection_suffixes
                if architecture == "gpt-oss"
                else projection_suffixes
            )
            assert suffixes[family] == expected
        assert suffixes["output"] == projection_suffixes
        assert all(name.endswith((".weight", ".bias")) for name in names)
        assert {
            "blk.{bid}.ffn_gate_exps.weight",
            "blk.{bid}.ffn_down_exps.weight",
            "blk.{bid}.ffn_up_exps.weight",
        } <= names

    def test_conditional_and_auxiliary_representations_are_not_conflated(self) -> None:
        arctic = set(upstream_architectures()["arctic"].tensor_names)
        dbrx = set(upstream_architectures()["dbrx"].tensor_names)
        gpt_oss = set(upstream_architectures()["gpt-oss"].tensor_names)
        grok = set(upstream_architectures()["grok"].tensor_names)
        grove = set(upstream_architectures()["grovemoe"].tensor_names)
        small = set(upstream_architectures()["smallthinker"].tensor_names)

        assert "blk.{bid}.ffn_norm_exps.weight" in arctic
        assert "blk.{bid}.attn_qkv.weight" in dbrx
        assert not {"blk.{bid}.attn_q.weight", "blk.{bid}.attn_k.weight"} & dbrx
        for names in (arctic, gpt_oss, grok, grove, small):
            assert {
                "blk.{bid}.attn_qkv.weight",
                "blk.{bid}.attn_qkv.bias",
                "blk.{bid}.attn_q.weight",
                "blk.{bid}.attn_q.bias",
                "blk.{bid}.attn_k.weight",
                "blk.{bid}.attn_k.bias",
                "blk.{bid}.attn_v.weight",
                "blk.{bid}.attn_v.bias",
            } <= names
        assert "blk.{bid}.attn_sinks.weight" in gpt_oss
        assert "blk.{bid}.ffn_gate_exps.bias" in gpt_oss
        assert {
            "blk.{bid}.layer_output_norm.weight",
            "blk.{bid}.post_ffw_norm.weight",
        } <= grok
        assert {
            "blk.{bid}.ffn_gate_chexps.weight",
            "blk.{bid}.ffn_down_chexps.weight",
            "blk.{bid}.ffn_up_chexps.weight",
        } <= grove
        grove_suffixes = dict(upstream_architectures()["grovemoe"].tensor_suffixes)
        assert grove_suffixes["blk.{bid}.ffn_gate_chexps"] == ("weight",)
        assert grove_suffixes["blk.{bid}.ffn_down_chexps"] == ("weight",)
        assert grove_suffixes["blk.{bid}.ffn_up_chexps"] == ("weight",)
        assert not any("_chexps." in name for name in small)

    @pytest.mark.parametrize("architecture", _DEFERRED_ARCHITECTURES)
    def test_no_false_alias_config_or_tensor_claim_is_reachable(
        self, architecture: str
    ) -> None:
        spec = try_get_arch_spec(architecture)
        assert spec is not None
        assert not spec.aliases
        assert spec.model_type is None
        assert spec.config_key_map is None
        assert spec.tensor_map_recipe == ()
        assert all(verdict is Support.DEFERRED for verdict in spec.verdicts.values())
        assert architecture not in GGUF_ARCH_TO_MODEL_TYPE

    @pytest.mark.parametrize(
        ("architecture", "reason_terms"),
        [
            ("gpt-oss", ("MXFP4", "expert biases", "attention sinks")),
            ("grok", ("softcaps", "sqrt(2)/2", "dense-plus-routed")),
            ("grovemoe", ("separate selections", "adjugate", "Q/K RMSNorm")),
        ],
    )
    def test_graph_and_routing_mismatch_is_explicit(
        self, architecture: str, reason_terms: tuple[str, ...]
    ) -> None:
        reason = try_get_arch_spec(architecture).reason
        assert reason is not None
        for term in reason_terms:
            assert term in reason

    @pytest.mark.parametrize("architecture", _DEFERRED_ARCHITECTURES)
    def test_every_architecture_fails_before_graph_construction(
        self, architecture: str
    ) -> None:
        with pytest.raises(UnsupportedGGUFArchitectureError, match=architecture):
            get_arch_spec(architecture)

    def test_valid_hugging_face_registrations_are_not_reused_as_gguf_aliases(self) -> None:
        assert {"arctic", "dbrx", "gpt_oss"} <= set(_REGISTRATIONS)
        assert try_get_arch_spec("arctic").module_type == "arctic_gguf"
        assert try_get_arch_spec("dbrx").module_type == "dbrx_gguf"
        assert try_get_arch_spec("gpt-oss").model_type is None


class TestPinnedRemainingVLMTextCohort:
    """Refuse incomplete text or paired packages for every remaining VLM identifier."""

    _ARCHITECTURES = (
        "chameleon",
        "cogvlm",
        "deepseek2-ocr",
        "gemma3n",
        "hunyuan_vl",
        "llama4",
        "mistral3",
        "paddleocr",
        "qwen3vl",
        "qwen3vlmoe",
    )
    _PAIRED_ARCHITECTURES = frozenset(_ARCHITECTURES) - {"chameleon"}
    _EXPECTED_LOADER_COUNTS: ClassVar[dict[str, int]] = {
        "chameleon": 21,
        "cogvlm": 16,
        "deepseek2-ocr": 21,
        "gemma3n": 37,
        "hunyuan_vl": 19,
        "llama4": 25,
        "mistral3": 31,
        "qwen2vl": 18,
        "qwen3vl": 20,
        "qwen3vlmoe": 20,
    }

    @pytest.mark.parametrize("architecture", sorted(_EXPECTED_LOADER_COUNTS))
    def test_direct_loader_inventory_and_projection_sidecars_are_suffix_exact(
        self, architecture: str
    ) -> None:
        upstream = upstream_architectures()[architecture]
        assert upstream.tensor_closure_status == "exact-direct-loader-conditional-union"
        assert len(upstream.tensor_names) == self._EXPECTED_LOADER_COUNTS[architecture]
        assert all(name.endswith((".weight", ".bias")) for name in upstream.tensor_names)
        suffixes = dict(upstream.tensor_suffixes)
        assert suffixes["output"] in {
            ("weight", "scale", "input_scale"),
            ("weight", "bias", "scale", "input_scale"),
        }
        assert all(
            "zero_point" not in suffix for values in suffixes.values() for suffix in values
        )
        for family, family_suffixes in suffixes.items():
            assert family.endswith(
                (
                    "output",
                    "attn_q",
                    "attn_k",
                    "attn_v",
                    "attn_qkv",
                    "attn_output",
                    "ffn_gate",
                    "ffn_down",
                    "ffn_up",
                    "ffn_gate_exps",
                    "ffn_down_exps",
                    "ffn_up_exps",
                    "ffn_gate_up_exps",
                    "ffn_gate_shexp",
                    "ffn_down_shexp",
                    "ffn_up_shexp",
                    "vis_attn_qkv",
                    "vis_attn_output",
                    "vis_gate",
                    "vis_down",
                    "vis_up",
                    "altup_proj",
                    "altup_unembd_proj",
                    "per_layer_model_proj",
                    "inp_gate",
                    ".proj",
                    "altup_router",
                    "laurel_l",
                    "laurel_r",
                    "cls.output",
                )
            ), family
            assert family_suffixes[0] == "weight"

    def test_paddleocr_records_only_the_mechanically_proven_converter_inventory(
        self,
    ) -> None:
        upstream = upstream_architectures()["paddleocr"]
        assert upstream.tensor_closure_status == (
            "strongest-converter-family-inventory-loader-inherited-from-ernie4_5-with-"
            "optional-attn-output-bias"
        )
        assert upstream.tensor_names == ()
        assert set(upstream.tensor_families) == {
            "token_embd",
            "output_norm",
            "output",
            "blk.{bid}.attn_norm",
            "blk.{bid}.attn_q",
            "blk.{bid}.attn_k",
            "blk.{bid}.attn_v",
            "blk.{bid}.attn_output",
            "blk.{bid}.ffn_norm",
            "blk.{bid}.ffn_gate",
            "blk.{bid}.ffn_down",
            "blk.{bid}.ffn_up",
        }

    def test_custom_loader_tensors_do_not_inherit_generic_quant_sidecars(self) -> None:
        expected_weight_only = {
            "cogvlm": {
                "blk.{bid}.vis_attn_qkv",
                "blk.{bid}.vis_attn_output",
                "blk.{bid}.vis_gate",
                "blk.{bid}.vis_down",
                "blk.{bid}.vis_up",
            },
            "deepseek2-ocr": {"blk.{bid}.ffn_gate_up_exps"},
            "gemma3n": {
                "altup_proj",
                "altup_unembd_proj",
                "per_layer_model_proj",
                "blk.{bid}.inp_gate",
                "blk.{bid}.proj",
                "blk.{bid}.altup_router",
                "blk.{bid}.laurel_l",
                "blk.{bid}.laurel_r",
            },
            "qwen3vl": {"cls.output"},
        }
        for architecture, families in expected_weight_only.items():
            suffixes = dict(upstream_architectures()[architecture].tensor_suffixes)
            assert all(suffixes[family] == ("weight",) for family in families)

    @pytest.mark.parametrize(
        "architecture",
        [
            "chameleon",
            "gemma3n",
            "hunyuan_vl",
            "llama4",
            "mistral3",
            "qwen2vl",
            "qwen3vl",
            "qwen3vlmoe",
        ],
    )
    def test_qkv_helper_loader_inventories_include_fused_and_split_branches(
        self, architecture: str
    ) -> None:
        upstream = upstream_architectures()[architecture]
        names = set(upstream.tensor_names)
        suffixes = dict(upstream.tensor_suffixes)
        for projection in ("qkv", "q", "k", "v"):
            family = f"blk.{{bid}}.attn_{projection}"
            assert {f"{family}.weight", f"{family}.bias"} <= names
            assert suffixes[family] == ("weight", "bias", "scale", "input_scale")

    def test_converter_inventory_status_does_not_turn_an_allowlist_into_a_claim(
        self,
    ) -> None:
        chameleon = upstream_architectures()["chameleon"]
        assert chameleon.converter_inventory_status == "exact-no-extra-tensors"
        assert chameleon.converter_extra_tensor_names == ()
        for architecture in self._PAIRED_ARCHITECTURES:
            upstream = upstream_architectures()[architecture]
            assert upstream.converter_inventory_status == (
                "unresolved-inherited-conditional-converter-hooks"
            )
            assert upstream.converter_extra_tensor_names == ()

    def test_conditional_tensor_representations_remain_distinct(self) -> None:
        chameleon = set(upstream_architectures()["chameleon"].tensor_names)
        cogvlm = set(upstream_architectures()["cogvlm"].tensor_names)
        deepseek = set(upstream_architectures()["deepseek2-ocr"].tensor_names)
        gemma3n = set(upstream_architectures()["gemma3n"].tensor_names)
        llama4 = set(upstream_architectures()["llama4"].tensor_names)
        mistral3 = set(upstream_architectures()["mistral3"].tensor_names)
        qwen2vl = set(upstream_architectures()["qwen2vl"].tensor_names)
        qwen3vl = set(upstream_architectures()["qwen3vl"].tensor_names)
        qwen3vlmoe = set(upstream_architectures()["qwen3vlmoe"].tensor_names)

        assert {
            "blk.{bid}.attn_q_norm.bias",
            "blk.{bid}.attn_k_norm.bias",
        } <= chameleon
        assert {
            "blk.{bid}.vis_attn_qkv.weight",
            "blk.{bid}.vis_gate.weight",
        } <= cogvlm
        assert {
            "blk.{bid}.ffn_gate_up_exps.weight",
            "blk.{bid}.exp_probs_b.bias",
        } <= deepseek
        assert {
            "per_layer_token_embd.weight",
            "blk.{bid}.altup_router.weight",
            "blk.{bid}.laurel_l.weight",
        } <= gemma3n
        assert {
            "blk.{bid}.ffn_gate.weight",
            "blk.{bid}.ffn_gate_exps.weight",
            "blk.{bid}.ffn_gate_shexp.weight",
        } <= llama4
        assert {
            "rope_factors_long.weight",
            "rope_factors_short.weight",
            "blk.{bid}.ffn_gate.bias",
            "blk.{bid}.ffn_gate_exps.weight",
        } <= mistral3
        assert "output.bias" in qwen2vl
        assert "cls.output.weight" in qwen3vl
        assert "cls.output.weight" not in qwen3vlmoe
        assert "blk.{bid}.ffn_gate_exps.weight" in qwen3vlmoe
        assert "blk.{bid}.ffn_gate.weight" not in qwen3vlmoe

    def test_loader_global_rope_tensors_are_not_layer_qualified(self) -> None:
        expected = {
            "cogvlm": {"rope_freqs.weight"},
            "llama4": {"rope_freqs.weight"},
            "mistral3": {
                "rope_freqs.weight",
                "rope_factors_long.weight",
                "rope_factors_short.weight",
            },
        }
        for architecture, global_names in expected.items():
            names = set(upstream_architectures()[architecture].tensor_names)
            assert global_names <= names
            assert not {f"blk.{{bid}}.{name}" for name in global_names} & names

    @pytest.mark.parametrize(
        "architecture", ["deepseek2-ocr", "llama4", "mistral3", "qwen3vlmoe"]
    )
    def test_generic_expert_sidecars_are_never_dropped(self, architecture: str) -> None:
        upstream = upstream_architectures()[architecture]
        assert upstream.expert_tensor_suffixes == ("weight", "scale", "input_scale")
        expert_families = [
            family
            for family in dict(upstream.tensor_suffixes)
            if family.endswith(("_exps", "_shexp")) and not family.endswith("ffn_gate_up_exps")
        ]
        assert expert_families
        for family in expert_families:
            assert dict(upstream.tensor_suffixes)[family] == (
                "weight",
                "scale",
                "input_scale",
            )

    @pytest.mark.parametrize("architecture", _ARCHITECTURES)
    def test_no_false_alias_config_tensor_or_runtime_claim_is_reachable(
        self, architecture: str
    ) -> None:
        spec = try_get_arch_spec(architecture)
        assert spec is not None
        assert not spec.aliases
        assert spec.model_type is None
        assert spec.config_key_map is None
        assert spec.tensor_map_recipe == ()
        assert spec.vlm_builder is None
        assert all(verdict is Support.DEFERRED for verdict in spec.verdicts.values())
        assert architecture not in GGUF_ARCH_TO_MODEL_TYPE

    @pytest.mark.parametrize(
        ("architecture", "reason_terms"),
        [
            ("chameleon", ("VQ image tokenizer", "swin_norm", "text-only")),
            ("cogvlm", ("visual-expert", "cogvlm clip sidecar", "wrong package")),
            ("deepseek2-ocr", ("text-plus-vision", "SAM/projector", "partial")),
            ("gemma3n", ("vision-and-audio", "per-layer embeddings", "package roles")),
            ("hunyuan_vl", ("M-RoPE", "Hunyuan-VL-MoT", "different")),
            ("llama4", ("routed experts", "llama4 clip", "text-backbone")),
            ("mistral3", ("dense or routed-expert", "temperature", "Pixtral")),
            ("paddleocr", ("optional bias", "image-token", "ordinary Qwen2")),
            ("qwen3vl", ("multimodal position IDs", "deep-stack", "text-only")),
            ("qwen3vlmoe", ("routed experts", "effective tied head", "cache ABI")),
        ],
    )
    def test_family_specific_blocker_is_explicit(
        self, architecture: str, reason_terms: tuple[str, ...]
    ) -> None:
        reason = try_get_arch_spec(architecture).reason
        assert reason is not None
        for term in reason_terms:
            assert term in reason

    @pytest.mark.parametrize("architecture", _ARCHITECTURES)
    def test_every_architecture_fails_before_config_or_graph_construction(
        self, architecture: str
    ) -> None:
        with pytest.raises(UnsupportedGGUFArchitectureError, match=architecture):
            get_arch_spec(architecture)

    def test_standalone_text_and_paired_package_verdicts_are_not_conflated(self) -> None:
        chameleon_reason = try_get_arch_spec("chameleon").reason
        assert chameleon_reason is not None
        assert "converter deliberately omits the VQ image tokenizer" in chameleon_reason
        for architecture in self._PAIRED_ARCHITECTURES:
            reason = try_get_arch_spec(architecture).reason
            assert reason is not None
            assert any(term in reason for term in ("clip", "sidecar", "vision-and-audio"))

    def test_valid_hugging_face_registrations_are_not_reused_as_gguf_aliases(self) -> None:
        assert {
            "chameleon",
            "gemma3n_text",
            "llama4_text",
            "mistral3",
            "qwen3_vl_text",
            "qwen3_vl_moe",
        } <= set(_REGISTRATIONS)
        for architecture in self._ARCHITECTURES:
            assert try_get_arch_spec(architecture).model_type is None

    def test_existing_exact_paired_builders_remain_the_only_supported_pairs(self) -> None:
        exact = {
            spec.gguf_arch: spec.vlm_builder
            for spec in iter_arch_specs()
            if spec.vlm_builder is not None
        }
        assert exact == {
            "chatglm": "generic_projector",
            "gemma3": "gemma3",
            "gemma4": "gemma4",
            "llama": "generic_projector",
            "minicpm": "generic_projector",
            "muse-glimmer": "muse_glimmer",
            "qwen2vl": "qwen_vl",
        }


class TestRejectionsAreActionable:
    """An unsupported input must say what it is and what to do instead."""

    @pytest.mark.parametrize("architecture", [])
    def test_configurable_but_unmappable_architectures_are_refused(
        self, architecture: str
    ) -> None:
        with pytest.raises(UnsupportedGGUFArchitectureError, match="Unsupported GGUF"):
            get_arch_spec(architecture)

    @pytest.mark.parametrize("architecture", [MMPROJ_ARCHITECTURE])
    def test_deliberately_disabled_architectures_raise_the_disabled_error(
        self, architecture: str
    ) -> None:
        with pytest.raises(DisabledGGUFArchitectureError):
            get_arch_spec(architecture)

    def test_a_dead_upstream_architecture_says_so(self) -> None:
        """``gptj`` is registered upstream but has no loader, so nothing can read it."""
        with pytest.raises(
            DisabledGGUFArchitectureError, match=re.escape("llama.cpp has no model loader")
        ):
            get_arch_spec("gptj")

    def test_an_unimported_upstream_architecture_names_its_cohort(self) -> None:
        with pytest.raises(UnsupportedGGUFArchitectureError, match="RWKV6 carries"):
            get_arch_spec("rwkv6")

    def test_an_unknown_architecture_is_distinguished_from_an_upstream_one(self) -> None:
        with pytest.raises(
            UnsupportedGGUFArchitectureError,
            match="not among the 148 pinned",
        ):
            get_arch_spec("definitely-not-real")

    def test_legacy_exception_types_still_catch_everything(self) -> None:
        """Callers written against the pre-registry errors keep working."""
        with pytest.raises(ValueError):
            get_arch_spec("definitely-not-real")
        with pytest.raises(NotImplementedError):
            get_arch_spec(MMPROJ_ARCHITECTURE)

    def test_nemotron_h_moe_graph_is_supported_but_runtime_is_deferred(self) -> None:
        spec = get_arch_spec("nemotron_h_moe")
        assert spec.model_type == "nemotron_h"
        assert spec.tensor_processor == "nemotron_h"
        assert spec.llama_qk_permute is True
        assert spec.config is Support.SUPPORTED
        assert spec.tensor_map is Support.SUPPORTED
        assert spec.graph is Support.SUPPORTED
        assert spec.runtime is Support.DEFERRED
        assert spec.quantized_import is Support.REJECTED
        assert spec.reason is not None
        assert "onnxruntime/mobius#605" in spec.reason

        dense_spec = get_arch_spec("nemotron_h")
        assert dense_spec.tensor_processor == "nemotron_h"
        assert dense_spec.llama_qk_permute is True

        granite_spec = get_arch_spec("granitehybrid")
        assert granite_spec.tensor_processor == "granitehybrid"
        assert granite_spec.llama_qk_permute is True


class TestDocumentedSupportMatrix:
    """The published matrix must be the registry, not a stale copy of it.

    ``docs/api/build_from_gguf.md`` previously claimed "Most decoder-only LLM
    architectures are supported", which is not a checkable statement. This test
    is what makes the replacement checkable.
    """

    _DOC = pathlib.Path(__file__).resolve().parents[4] / "docs" / "api" / "build_from_gguf.md"
    _BEGIN = "<!-- BEGIN GGUF SUPPORT MATRIX (generated; see _arch_registry.py) -->"
    _END = "<!-- END GGUF SUPPORT MATRIX -->"

    @staticmethod
    def _expected_rows() -> list[str]:
        rows = []
        for spec in sorted(iter_arch_specs(), key=lambda s: s.gguf_arch):
            aliases = ", ".join(f"`{a}`" for a in sorted(spec.aliases)) or "—"
            model_type = f"`{spec.model_type}`" if spec.model_type else "—"
            core_verdicts = spec.verdicts
            status = (
                "supported"
                if all(verdict is Support.SUPPORTED for verdict in spec.verdicts.values())
                else "; ".join(
                    f"{name} {verdict.value}"
                    for name, verdict in core_verdicts.items()
                    if verdict is not Support.SUPPORTED
                )
            )
            reason = spec.reason or "Validated graph and runtime contract."
            rows.append(
                f"| `{spec.gguf_arch}` | {aliases} | {model_type} | {status} | {reason} |"
            )
        return rows

    def test_the_doc_table_matches_the_registry(self) -> None:
        from mobius.integrations.gguf._docs import check_document

        assert check_document(), (
            "docs/api/build_from_gguf.md is out of date; run "
            "`python scripts/generate_gguf_support_docs.py`."
        )


class TestOffsetNormCompensation:
    """Offset-norm architectures must have their llama.cpp ``+1`` removed.

    llama.cpp bakes the ``1 +`` of a centered RMSNorm into the stored weight so
    its generic kernel can use the tensor directly. Any mobius model that
    normalizes with ``OffsetRMSNorm`` re-applies that ``1 +`` at runtime, so the
    import path has to subtract it back out — through the Gemma weight
    processor or through the ``offset_norm`` normalization hook.

    Gemma 3 failed exactly this: it used ``OffsetRMSNorm`` and had neither,
    because its processor was registered under ``model_type`` ``gemma3`` while
    GGUF ``gemma3`` resolves to ``gemma3_text``. Every norm was left doubled.

    Nemotron failed it in the other direction: it uses ``OffsetLayerNorm`` and
    had a processor, but that processor *added* one instead of subtracting it,
    so the effective scale came out ``w_hf + 3``. The check therefore matches
    both offset norm classes and both failure shapes.

    Gemma 4 deliberately passes with neither, because ``models/gemma4.py``
    normalizes with plain ``RMSNorm``.
    """

    _OFFSET_CALL = re.compile(
        r"(?<![\w.])Offset(?:RMSNorm|LayerNorm)\s*\(|(?:rms_)?norm_class=Offset\w+"
    )

    @classmethod
    def _model_uses_offset_norm(cls, model_type: str) -> bool:
        registration = _REGISTRATIONS[model_type]
        module_class = getattr(registration, "module_class", None) or getattr(
            registration, "model_class", None
        )
        if module_class is None:
            return False
        module = inspect.getmodule(module_class)
        if module is None:
            return False
        try:
            source = inspect.getsource(module)
        except OSError:
            return False
        return bool(cls._OFFSET_CALL.search(source))

    @pytest.mark.parametrize(
        "spec",
        [s for s in iter_arch_specs() if s.is_importable],
        ids=lambda s: s.gguf_arch,
    )
    def test_offset_norm_models_have_compensation(self, spec) -> None:
        assert spec.model_type is not None
        if not self._model_uses_offset_norm(spec.module_type or spec.model_type):
            return
        compensated = spec.tensor_processor == "unoffset_norm" or spec.offset_norm
        assert compensated, (
            f"{spec.gguf_arch}: {spec.model_type} normalizes with an offset norm, so "
            "llama.cpp's baked-in +1 must be removed on import via the "
            "'unoffset_norm' tensor processor or the offset_norm hook. Without it "
            "the offset lands twice and the model produces garbage."
        )

    def test_gemma4_is_deliberately_uncompensated(self) -> None:
        """Guard the inverse: applying the un-offset here would corrupt Gemma 4."""
        spec = get_arch_spec("gemma4")
        assert not self._model_uses_offset_norm("gemma4_text")
        assert spec.tensor_processor is None
        assert not spec.offset_norm

    @pytest.mark.parametrize("architecture", ["gemma", "gemma2", "gemma3", "nemotron"])
    def test_the_offset_is_removed_in_the_right_direction(self, architecture: str) -> None:
        """Pin the sign, not just which processor is declared.

        Declaring a processor is not enough — Nemotron declared one that added
        one instead of subtracting it. llama.cpp writes ``w_gguf = w_hf + 1``
        and the graph re-applies ``1 +``, so the initializer must come out at
        ``w_gguf - 1`` and the effective scale back at ``w_gguf``.
        """
        import torch

        from mobius.integrations.gguf._tensor_processors import process_tensors

        config = SimpleNamespace(
            _gguf_arch=architecture,
            model_type=get_arch_spec(architecture).model_type,
            num_attention_heads=4,
            num_key_value_heads=4,
        )
        w_gguf = 1.25
        processed = process_tensors(
            {"model.layers.0.input_layernorm.weight": torch.full((8,), w_gguf)}, config
        )
        initializer = float(processed["model.layers.0.input_layernorm.weight"][0])
        assert initializer == pytest.approx(w_gguf - 1.0), (
            f"{architecture}: expected the llama.cpp +1 to be subtracted"
        )
        assert 1.0 + initializer == pytest.approx(w_gguf), (
            f"{architecture}: effective scale must round-trip to the stored weight"
        )
