# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Coverage gap tests against the pinned llama.cpp census.

The census is vendored so coverage is *measurable*, not so it can be claimed.
These tests assert the separation directly: mobius covers a small, named subset
of the 147 upstream architectures, everything else resolves to an explicit
refusal with a reason, and no architecture becomes "supported" merely by
appearing in the data file.
"""

from __future__ import annotations

import pytest

from mobius.integrations.gguf._arch_registry import (
    get_arch_spec,
    iter_arch_specs,
    supported_architectures,
    try_get_arch_spec,
)
from mobius.integrations.gguf._errors import (
    DisabledGGUFArchitectureError,
    UnsupportedGGUFArchitectureError,
)
from mobius.integrations.gguf._upstream import (
    UPSTREAM_COMMIT,
    upstream_architectures,
    upstream_quant_types,
)

_PINNED_VLM_TEXT_ARCHITECTURES = frozenset(
    {
        "chameleon",
        "cogvlm",
        "deepseek2-ocr",
        "gemma3",
        "gemma3n",
        "gemma4",
        "hunyuan_vl",
        "llama4",
        "mistral3",
        "muse-glimmer",
        "paddleocr",
        "qwen2vl",
        "qwen3vl",
        "qwen3vlmoe",
    }
)


class TestPinIntegrity:
    """The vendored payload has to be the census it claims to be."""

    def test_commit_is_pinned(self) -> None:
        assert UPSTREAM_COMMIT == "8d9af256337d1a501250f9bbf4c0859a654bddd6"

    def test_counts_match_the_survey(self) -> None:
        assert len(upstream_architectures()) == 147
        assert len(upstream_quant_types()) == 43

    def test_known_upstream_facts_survive_the_trim(self) -> None:
        """Spot-check the facts the registry actually reasons about."""
        archs = upstream_architectures()
        # gptj is the one architecture with no C++ loader, so nothing can read it.
        assert not archs["gptj"].cpp_loader
        # clip is the mmproj sidecar cohort.
        assert archs["clip"].cohort == "C08-multimodal-projector"
        # 47 architectures switch tensor shape on expert_count rather than on name.
        dual = [a for a in archs.values() if a.dual_moe]
        assert len(dual) == 47

    def test_multimodal_text_census_is_explicit(self) -> None:
        assert not (_PINNED_VLM_TEXT_ARCHITECTURES - upstream_architectures().keys())
        assert {
            architecture
            for architecture in _PINNED_VLM_TEXT_ARCHITECTURES
            if try_get_arch_spec(architecture) is None
        } == set()


class TestCoverageIsHonest:
    """Being in the census must never imply being supported."""

    def test_every_upstream_architecture_has_one_explicit_verdict(self) -> None:
        assert len(iter_arch_specs()) == len(upstream_architectures()) == 147
        assert {spec.gguf_arch for spec in iter_arch_specs()} == set(upstream_architectures())
        assert len(supported_architectures()) < len(upstream_architectures())

    @pytest.mark.parametrize(
        "architecture",
        sorted(upstream_architectures()),
    )
    def test_every_pinned_architecture_resolves_to_a_spec(self, architecture: str) -> None:
        spec = try_get_arch_spec(architecture)
        assert spec is not None
        assert spec.gguf_arch == architecture

    @pytest.mark.parametrize("spec", iter_arch_specs(), ids=lambda s: s.gguf_arch)
    def test_every_registered_architecture_resolves_one_way_or_the_other(self, spec) -> None:
        """No spec may be inert: it either imports, or it refuses with a reason."""
        if spec.is_importable:
            assert get_arch_spec(spec.gguf_arch) is spec
            return
        with pytest.raises(
            (UnsupportedGGUFArchitectureError, DisabledGGUFArchitectureError)
        ) as e:
            get_arch_spec(spec.gguf_arch)
        assert spec.reason is not None
        assert spec.reason.split(".")[0] in str(e.value)

    @pytest.mark.parametrize(
        "architecture",
        ["bitnet", "deepseek2", "gemma4-assistant", "graniteswitch", "rwkv7"],
    )
    def test_deferred_verdict_precedes_mtp_and_qtype_policy(
        self, architecture: str, monkeypatch
    ) -> None:
        from mobius.integrations.gguf import _builder, _mtp

        class FakeGGUF:
            tensor_names = ()

            @staticmethod
            def get_metadata(_key: str, default):
                return default

        model = FakeGGUF()
        model.architecture = architecture
        monkeypatch.setattr(
            _mtp,
            "validate_mtp_tensor_contract",
            lambda _model: pytest.fail("MTP policy must not run for a deferred architecture"),
        )
        monkeypatch.setattr(
            _builder,
            "_raise_for_unsupported_auxiliary_quantization",
            lambda _model: pytest.fail(
                "qtype policy must not run for a deferred architecture"
            ),
        )
        with pytest.raises(
            (UnsupportedGGUFArchitectureError, DisabledGGUFArchitectureError),
            match="before config extraction",
        ):
            _builder._validate_gguf_model(model, source="synthetic.gguf")
