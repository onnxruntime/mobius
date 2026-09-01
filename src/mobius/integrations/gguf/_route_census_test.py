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
            and spec.gguf_arch not in {"dflash", "eagle3"}
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
    }
    assert actual == expected
    assert len(actual) == len(items)


def test_every_route_has_one_actionable_classification() -> None:
    allowed = {
        "evidence-only",
        "dependency-or-mobius-abi-blocked",
        "artifact-unavailable",
        "intentionally-rejected",
    }
    items = iter_remaining_route_work()
    assert {item.category for item in items} == allowed
    assert all(item.batch and item.dependencies and item.reason.strip() for item in items)
    assert Counter(item.kind for item in items) == {
        "architecture": 135,
        "projector": 60,
        "tokenizer": 56,
        "mtp": 22,
    }
    assert Counter(item.category for item in items) == {
        "dependency-or-mobius-abi-blocked": 99,
        "evidence-only": 150,
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
    assert by_id["architecture:glm-dsa"].category == "evidence-only"
    assert by_id["architecture:minimax-m2"].category == "evidence-only"
    assert by_id["architecture:mistral4"].category == "evidence-only"
    assert by_id["architecture:deepseek4"].category == "dependency-or-mobius-abi-blocked"
    assert by_id["architecture:rwkv6"].category == "dependency-or-mobius-abi-blocked"
    assert by_id["architecture:bailingmoe2"].category == "intentionally-rejected"
    assert by_id["architecture:plamo3"].category == "dependency-or-mobius-abi-blocked"
    assert by_id["architecture:plamo2"].category == "evidence-only"
    assert by_id["architecture:jamba"].category == "evidence-only"
    assert by_id["projector:qwen2vl_merger"].category == "evidence-only"
    for projector_type in (
        "glm4v",
        "glma",
        "qwen2.5o",
        "qwen2a",
        "qwen3a",
        "qwen3vl_merger",
        "qwen3tts_spkenc",
    ):
        assert by_id[f"projector:{projector_type}"].category == "evidence-only"
    assert by_id["projector:lfm2"].category == "evidence-only"
    assert by_id["projector:minimax_m3"].category == "evidence-only"
    assert by_id["projector:pixtral"].category == "evidence-only"
    for projector_type in (
        "gemma3na",
        "gemma3nv",
        "gemma4a",
        "gemma4ua",
        "gemma4uv",
        "idefics3",
        "internvl",
        "llama4",
    ):
        assert by_id[f"projector:{projector_type}"].category == "evidence-only"
    assert by_id["projector:qwen3tts_gen"].category == "intentionally-rejected"
    assert by_id["tokenizer:llama4"].category == "artifact-unavailable"
    assert "PR #652" not in by_id["tokenizer:llama4"].dependencies
    for identifier in (
        "bailingmoe",
        "bailingmoe2",
        "chatglm-bpe",
        "cohere2moe",
        "glm4",
        "llada-moe",
        "tiny_aya",
    ):
        item = by_id[f"tokenizer:{identifier}"]
        assert item.category == "dependency-or-mobius-abi-blocked"
        assert item.batch == "tokenizer-compiled-semantics"
    assert by_id["mtp:qwen35"].category == "evidence-only"
    assert by_id["mtp:deepseek2"].category == "dependency-or-mobius-abi-blocked"
    assert by_id["mtp:exaone4"].category == "intentionally-rejected"
    assert "draft:dflash" not in by_id
    assert "draft:eagle3" not in by_id


def test_recent_pr_reconciliation_is_explicit() -> None:
    assert [(record.number, record.state_at_audit) for record in RECENT_PR_DEPENDENCIES] == [
        (645, "merged"),
        (651, "merged"),
        (652, "closed"),
        (656, "merged"),
        (675, "merged"),
        (672, "merged"),
        (674, "merged"),
        (677, "merged"),
        (678, "merged"),
        (679, "merged"),
        (680, "merged"),
    ]


def test_batch_table_is_deterministic_and_complete() -> None:
    rendered = render_remaining_route_batches()
    assert rendered == render_remaining_route_batches()
    for item in iter_remaining_route_work():
        assert f"`{item.route_id}`" in rendered
