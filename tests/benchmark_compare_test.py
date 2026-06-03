# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the deterministic-metric regression comparator.

Focuses on the ``EXPECTED_CHANGES`` allowlist used to accept intended
structural node-count changes (e.g. the PR #328 static-cache phase-split)
without masking accidental regressions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import tests.benchmark_compare as bc

# A blue square marks an accepted intended structural change.
_ACCEPTED = "\U0001f7e6"
# A red circle marks a blocking regression.
_BLOCKER = "\U0001f534"


def _write(path: Path, models: dict[str, dict[str, int]]) -> str:
    path.write_text(json.dumps({"_metadata": {"commit": "deadbeef"}, "models": models}))
    return str(path)


def _run(tmp_path: Path, base: dict, curr: dict) -> tuple[str, bool]:
    baseline = _write(tmp_path / "baseline.json", base)
    current = _write(tmp_path / "current.json", curr)
    return bc.compare(current, baseline)


def test_exact_expected_static_cache_delta_is_waived(tmp_path: Path):
    """The exact intended +10 static-cache node delta must not block."""
    md, has_blocker = _run(
        tmp_path,
        base={"llama (static-cache)": {"num_nodes": 58, "model_size_bytes": 1000}},
        curr={"llama (static-cache)": {"num_nodes": 68, "model_size_bytes": 1000}},
    )
    assert has_blocker is False
    assert _ACCEPTED in md
    assert _BLOCKER not in md


@pytest.mark.parametrize(
    "model",
    ["llama (static-cache)", "qwen2 (static-cache)", "phi3 (static-cache)"],
)
def test_all_allowlisted_models_waive_their_change(tmp_path: Path, model: str):
    base_nodes, curr_nodes = bc.EXPECTED_CHANGES[model]["num_nodes"]
    md, has_blocker = _run(
        tmp_path,
        base={model: {"num_nodes": base_nodes}},
        curr={model: {"num_nodes": curr_nodes}},
    )
    assert has_blocker is False
    assert _ACCEPTED in md


def test_delta_larger_than_expected_still_blocks(tmp_path: Path):
    """An extra unexpected node beyond the intended delta must still block."""
    md, has_blocker = _run(
        tmp_path,
        base={"llama (static-cache)": {"num_nodes": 58}},
        curr={"llama (static-cache)": {"num_nodes": 69}},  # +11, not the intended +10
    )
    assert has_blocker is True
    assert _BLOCKER in md


def test_non_allowlisted_model_still_blocks(tmp_path: Path):
    """A model outside the allowlist gets no waiver."""
    md, has_blocker = _run(
        tmp_path,
        base={"llama": {"num_nodes": 58}},
        curr={"llama": {"num_nodes": 68}},  # +17%, over the 10% block threshold
    )
    assert has_blocker is True
    assert _BLOCKER in md


def test_smaller_than_expected_delta_is_not_waived(tmp_path: Path):
    """Only the exact delta is waived; a smaller change uses normal thresholds."""
    # +5 on 58 = +8.6%, between warn (5%) and block (10%): a warning, not waived.
    md, has_blocker = _run(
        tmp_path,
        base={"llama (static-cache)": {"num_nodes": 58}},
        curr={"llama (static-cache)": {"num_nodes": 63}},
    )
    assert has_blocker is False
    assert _ACCEPTED not in md
    assert "\u26a0\ufe0f" in md  # warning


def test_post_merge_zero_delta_is_inert(tmp_path: Path):
    """Once merged, base==head -> 0 delta: no blocker, no accepted-marker."""
    md, has_blocker = _run(
        tmp_path,
        base={"llama (static-cache)": {"num_nodes": 68}},
        curr={"llama (static-cache)": {"num_nodes": 68}},
    )
    assert has_blocker is False
    assert _ACCEPTED not in md
    assert _BLOCKER not in md


def test_post_merge_repeat_delta_still_blocks(tmp_path: Path):
    """Absolute pinning closes the stale-waiver hole.

    After the +10 phase-split merges (baseline becomes 68), a *future* +10
    regression (68 -> 78) must NOT match the (58, 68) waiver and must still block.
    """
    md, has_blocker = _run(
        tmp_path,
        base={"llama (static-cache)": {"num_nodes": 68}},
        curr={"llama (static-cache)": {"num_nodes": 78}},
    )
    assert has_blocker is True
    assert _BLOCKER in md
    assert _ACCEPTED not in md


def test_non_waived_metric_on_allowlisted_model_still_blocks(tmp_path: Path):
    """A non-waived metric on an allowlisted model still blocks.

    The waiver is scoped per-metric: an allowlisted model whose num_nodes
    matches the pinned (58, 68) must still block on a DIFFERENT metric
    (model_size_bytes +30%), guarding against over-broadening the waiver.
    """
    md, has_blocker = _run(
        tmp_path,
        base={"llama (static-cache)": {"num_nodes": 58, "model_size_bytes": 1000}},
        curr={"llama (static-cache)": {"num_nodes": 68, "model_size_bytes": 1300}},
    )
    assert has_blocker is True
    assert _BLOCKER in md
    assert _ACCEPTED in md  # the num_nodes row is still waived


def test_expected_changes_keys_are_real_model_display_keys():
    """Guard against a silent waiver no-op if model keys drift.

    A model_type/task rename would move the comparator's allowlist keys away
    from the benchmarked display keys, silently re-REDing the intended +10.
    """
    from tests.benchmark_build import BENCHMARK_MODELS, _display_key

    valid_keys = {_display_key(e.model_type, e.task_name) for e in BENCHMARK_MODELS}
    for model_key in bc.EXPECTED_CHANGES:
        assert model_key in valid_keys, (
            f"EXPECTED_CHANGES key {model_key!r} is not a benchmarked model "
            f"display key; the waiver would silently never fire."
        )
