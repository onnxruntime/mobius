# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from collections import Counter

from mobius.integrations.gguf._arch_registry import iter_arch_specs
from mobius.integrations.gguf._mmproj_registry import iter_projector_specs
from mobius.integrations.gguf._mtp import mtp_architecture_capabilities
from mobius.integrations.gguf._route_census import (
    RECENT_PR_DEPENDENCIES,
    iter_remaining_route_work,
    render_remaining_route_batches,
)
from mobius.integrations.gguf._spec import Support
from mobius.integrations.gguf._tokenizer_census import tokenizer_route_census


def test_remaining_route_census_closes_every_authoritative_registry() -> None:
    items = iter_remaining_route_work()
    actual = {item.route_id for item in items}
    expected = {
        *(
            f"architecture:{spec.gguf_arch}"
            for spec in iter_arch_specs()
            if spec.runtime is not Support.SUPPORTED
        ),
        *(
            f"projector:{spec.projector_type}"
            for spec in iter_projector_specs()
            if spec.runtime is not Support.SUPPORTED
        ),
        *(
            f"tokenizer:{record.identifier}"
            for record in tokenizer_route_census()
            if record.current_status != "validated-pinned-source"
        ),
        *(f"mtp:{architecture}" for architecture in mtp_architecture_capabilities()),
        "draft:dflash",
        "draft:eagle3",
    }
    assert actual == expected
    assert len(actual) == len(items)


def test_every_route_has_one_actionable_classification() -> None:
    allowed = {
        "immediately-implementable",
        "evidence-only",
        "dependency-or-runtime-abi-blocked",
        "artifact-unavailable",
        "intentionally-rejected",
    }
    items = iter_remaining_route_work()
    assert {item.category for item in items} == allowed
    assert all(item.batch and item.dependencies and item.reason.strip() for item in items)
    assert Counter(item.kind for item in items) == {
        "architecture": 144,
        "projector": 60,
        "tokenizer": 56,
        "mtp": 22,
        "draft": 2,
    }
    assert Counter(item.category for item in items) == {
        "dependency-or-runtime-abi-blocked": 102,
        "evidence-only": 105,
        "immediately-implementable": 53,
        "intentionally-rejected": 19,
        "artifact-unavailable": 5,
    }


def test_route_reasons_are_sourced_from_authoritative_records() -> None:
    by_id = {item.route_id: item for item in iter_remaining_route_work()}
    for spec in iter_arch_specs():
        item = by_id.get(f"architecture:{spec.gguf_arch}")
        if item is not None:
            assert item.reason == spec.reason
    for spec in iter_projector_specs():
        item = by_id.get(f"projector:{spec.projector_type}")
        if item is not None:
            assert item.reason == spec.reason
    for architecture, capability in mtp_architecture_capabilities().items():
        assert by_id[f"mtp:{architecture}"].reason == capability.reason
    for record in tokenizer_route_census():
        item = by_id.get(f"tokenizer:{record.identifier}")
        if item is not None:
            assert item.reason == (
                record.candidate_disposition or str(record.blocker_category)
            )


def test_known_route_boundaries_are_not_collapsed() -> None:
    by_id = {item.route_id: item for item in iter_remaining_route_work()}
    assert by_id["architecture:bitnet"].category == "evidence-only"
    assert by_id["architecture:deepseek4"].category == "dependency-or-runtime-abi-blocked"
    assert by_id["architecture:rwkv6"].category == "dependency-or-runtime-abi-blocked"
    assert by_id["architecture:bailingmoe2"].category == "intentionally-rejected"
    assert by_id["architecture:plamo3"].category == "dependency-or-runtime-abi-blocked"
    assert by_id["architecture:plamo2"].category == "dependency-or-runtime-abi-blocked"
    assert "issue #605" in by_id["architecture:plamo2"].dependencies[0]
    assert by_id["projector:qwen2vl_merger"].category == "evidence-only"
    assert by_id["projector:lfm2"].category == "immediately-implementable"
    assert by_id["projector:pixtral"].category == "immediately-implementable"
    assert by_id["projector:qwen3tts_gen"].category == "intentionally-rejected"
    assert by_id["tokenizer:llama4"].category == "artifact-unavailable"
    assert "PR #652" in by_id["tokenizer:llama4"].dependencies
    assert by_id["mtp:qwen35"].category == "evidence-only"
    assert by_id["mtp:deepseek2"].category == "dependency-or-runtime-abi-blocked"
    assert by_id["mtp:exaone4"].category == "intentionally-rejected"


def test_recent_pr_reconciliation_is_explicit() -> None:
    assert [(record.number, record.state_at_audit) for record in RECENT_PR_DEPENDENCIES] == [
        (645, "merged"),
        (651, "open"),
        (652, "open"),
        (656, "open"),
    ]


def test_batch_table_is_deterministic_and_complete() -> None:
    rendered = render_remaining_route_batches()
    assert rendered == render_remaining_route_batches()
    for item in iter_remaining_route_work():
        assert f"`{item.route_id}`" in rendered
