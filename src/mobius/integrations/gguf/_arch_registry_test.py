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
from mobius.integrations.gguf._tensor_processors import _PROCESSOR_IMPLS
from mobius.integrations.gguf._upstream import upstream_architectures

#: Number of importable architectures. Pinned so that adding support is a
#: deliberate act that also updates the documented support matrix, and so that
#: accidentally losing an architecture is a failure rather than a silence.
_EXPECTED_SUPPORTED_COUNT = 37

# Quantized reachability is separately pinned from float importability. A new
# architecture must explicitly prove that its graph exposes packed projection
# modules before joining this set.
_EXPECTED_QUANTIZED_IMPORT_ARCHITECTURES = frozenset(
    {
        "arcee",
        "bert",
        "cohere2",
        "deci",
        "deepseek4",
        "exaone",
        "falcon",
        "gemma",
        "gemma2",
        "gemma3",
        "gemma4",
        "gpt2",
        "granitemoe",
        "hunyuan-dense",
        "llama",
        "modern-bert",
        "muse-glimmer",
        "nemotron",
        "olmo",
        "olmo2",
        "olmoe",
        "phi3",
        "phimoe",
        "qwen2",
        "qwen2moe",
        "qwen3",
        "qwen35",
        "qwen35moe",
        "qwen3moe",
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
        assert spec.model_type in _REGISTRATIONS, (
            f"{spec.gguf_arch}: model_type {spec.model_type!r} is not registered in "
            "mobius._registry, so the graph cannot actually be built"
        )

    def test_the_supported_set_is_pinned(self) -> None:
        """Gaining or losing support is a deliberate, reviewable change."""
        assert len(supported_architectures()) == _EXPECTED_SUPPORTED_COUNT

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
        rejected = {"internlm2", "mamba", "mamba2"}
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
        "architecture", ["olmoe", "phimoe", "qwen2moe", "qwen3moe", "granitemoe"]
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


class TestRejectionsAreActionable:
    """An unsupported input must say what it is and what to do instead."""

    @pytest.mark.parametrize("architecture", ["bloom"])
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
        with pytest.raises(UnsupportedGGUFArchitectureError, match="distinct RWKV"):
            get_arch_spec("rwkv6")

    def test_an_unknown_architecture_is_distinguished_from_an_upstream_one(self) -> None:
        with pytest.raises(UnsupportedGGUFArchitectureError, match="not among the 147"):
            get_arch_spec("definitely-not-real")

    def test_legacy_exception_types_still_catch_everything(self) -> None:
        """Callers written against the pre-registry errors keep working."""
        with pytest.raises(ValueError):
            get_arch_spec("definitely-not-real")
        with pytest.raises(NotImplementedError):
            get_arch_spec("nemotron_h_moe")


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
            core_verdicts = {
                name: verdict
                for name, verdict in spec.verdicts.items()
                if name != "quantized_import"
            }
            status = (
                "supported"
                if all(verdict is Support.SUPPORTED for verdict in spec.verdicts.values())
                else "; ".join(
                    f"{name} {verdict.value}"
                    for name, verdict in core_verdicts.items()
                    if verdict is not Support.SUPPORTED
                )
            )
            quantized = spec.quantized_import.value if spec.is_importable else "unreachable"
            rows.append(
                f"| `{spec.gguf_arch}` | {aliases} | {model_type} | {status} | {quantized} |"
            )
        return rows

    def test_the_doc_table_matches_the_registry(self) -> None:
        text = self._DOC.read_text(encoding="utf-8")
        assert self._BEGIN in text and self._END in text, (
            f"{self._DOC} is missing the generated support-matrix markers"
        )
        block = text.split(self._BEGIN, 1)[1].split(self._END, 1)[0]
        documented = [line for line in block.splitlines() if line.startswith("| `")]
        assert documented == self._expected_rows(), (
            "docs/api/build_from_gguf.md is out of date with the architecture "
            "registry. Regenerate the support matrix between its markers."
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
        if not self._model_uses_offset_norm(spec.model_type):
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
