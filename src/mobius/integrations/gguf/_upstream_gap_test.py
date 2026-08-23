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

_UNCOVERED = sorted(set(upstream_architectures()) - {s.gguf_arch for s in iter_arch_specs()})


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
        dual = [a for a in archs.values() if a.moe_mode.startswith("dual")]
        assert len(dual) == 47


class TestCoverageIsHonest:
    """Being in the census must never imply being supported."""

    def test_most_upstream_architectures_are_not_covered(self) -> None:
        assert len(_UNCOVERED) == 147 - len({s.gguf_arch for s in iter_arch_specs()})
        assert len(supported_architectures()) < len(upstream_architectures())

    @pytest.mark.parametrize("architecture", _UNCOVERED)
    def test_an_uncovered_architecture_has_no_spec(self, architecture: str) -> None:
        assert try_get_arch_spec(architecture) is None

    @pytest.mark.parametrize("architecture", _UNCOVERED)
    def test_an_uncovered_architecture_is_refused_with_a_reason(
        self, architecture: str
    ) -> None:
        with pytest.raises(UnsupportedGGUFArchitectureError) as excinfo:
            get_arch_spec(architecture)
        message = str(excinfo.value)
        assert architecture in message
        # Either it names the upstream cohort and the alternative, or it says the
        # architecture cannot be loaded by anything.
        assert "mobius build" in message or "no llama.cpp model loader" in message

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
