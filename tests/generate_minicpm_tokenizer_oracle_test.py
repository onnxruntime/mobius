# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Network-free tests for the MiniCPM tokenizer oracle refresh workflow."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from generate_minicpm_tokenizer_oracle import (  # noqa: E402
    FIXED_INPUTS,
    MODES,
    RANDOM_ALPHABET,
    RANDOM_COUNT,
    RANDOM_LENGTH_STOP,
    SEED,
    _generator_sha256,
    _json_bytes,
    _render_fixture,
    _write_or_check_fixture,
    build_corpus,
    summarize_results,
)

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = _ROOT / "tests/data/gguf_minicpm_tokenizer_oracle.json"
_GENERATOR = _ROOT / "scripts/generate_minicpm_tokenizer_oracle.py"


def test_committed_fixture_binds_exact_generator_and_corpus() -> None:
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    corpus = build_corpus()

    assert fixture["generator"] == "scripts/generate_minicpm_tokenizer_oracle.py"
    assert fixture["generator_sha256"] == _generator_sha256(_GENERATOR)
    assert fixture["serialization"] == (
        "UTF-8 compact JSON arrays (ensure_ascii=false; separators=(',',':'))"
    )
    assert fixture["seed"] == SEED
    assert fixture["random_count"] == RANDOM_COUNT
    assert fixture["random_length_stop_exclusive"] == RANDOM_LENGTH_STOP
    assert fixture["fixed_count"] == len(FIXED_INPUTS)
    assert fixture["fixed_inputs"] == list(FIXED_INPUTS)
    assert fixture["random_alphabet"] == "".join(RANDOM_ALPHABET)
    assert (
        fixture["random_alphabet_sha256"]
        == hashlib.sha256("".join(RANDOM_ALPHABET).encode()).hexdigest()
    )
    assert fixture["corpus_sha256"] == hashlib.sha256(_json_bytes(corpus)).hexdigest()
    assert fixture["modes"] == [list(mode) for mode in MODES]
    assert fixture["case_count_per_route"] == len(MODES) * len(corpus)


def test_generator_and_fixture_bytes_are_deterministic_across_line_endings(
    tmp_path: Path,
) -> None:
    lf_source = tmp_path / "generator-lf.py"
    crlf_source = tmp_path / "generator-crlf.py"
    source = bytes("# UTF-8: café\nprint('MiniCPM')\n", encoding="utf-8")
    lf_source.write_bytes(source)
    crlf_source.write_bytes(source.replace(b"\n", b"\r\n"))

    lf_rendered = _render_fixture({"generator_sha256": _generator_sha256(lf_source)})
    crlf_rendered = _render_fixture({"generator_sha256": _generator_sha256(crlf_source)})
    lf_output = tmp_path / "oracle-lf.json"
    crlf_output = tmp_path / "oracle-crlf.json"
    _write_or_check_fixture(lf_output, lf_rendered, check=False)
    _write_or_check_fixture(crlf_output, crlf_rendered, check=False)

    assert _generator_sha256(lf_source) == hashlib.sha256(source).hexdigest()
    assert lf_output.read_bytes() == crlf_output.read_bytes() == lf_rendered
    assert b"\r\n" not in lf_rendered


def test_fixture_check_requires_exact_utf8_lf_bytes(tmp_path: Path) -> None:
    output = tmp_path / "oracle.json"
    rendered = _render_fixture({"text": "café", "lines": ["first", "second"]})
    output.write_bytes(rendered)

    _write_or_check_fixture(output, rendered, check=True)

    output.write_bytes(rendered.replace(b"\n", b"\r\n"))
    with pytest.raises(SystemExit, match="is stale; regenerate without --check"):
        _write_or_check_fixture(output, rendered, check=True)


def test_result_summary_recomputes_hashes_counts_and_first_witness() -> None:
    corpus = ("same", "different")
    source_mode = [[1], [2]]
    llama_mode = [[1], [3]]
    source = source_mode * len(MODES)
    llama = llama_mode * len(MODES)

    summary = summarize_results(llama, source, corpus=corpus)

    assert summary == {
        "ordered_results_sha256": hashlib.sha256(_json_bytes(llama)).hexdigest(),
        "source_results_sha256": hashlib.sha256(_json_bytes(source)).hexdigest(),
        "mismatch_count": 3,
        "mismatch_count_by_mode": [1, 1, 1],
        "mismatch_input_indices": [1],
        "first_mismatch": {
            "mode": list(MODES[0]),
            "text": "different",
            "llamacpp_ids": [3],
            "source_ids": [2],
        },
    }


def test_result_summary_rejects_incomplete_or_mode_inconsistent_results() -> None:
    with pytest.raises(ValueError, match="ordered results"):
        summarize_results([], [], corpus=("text",))

    source = [[1], [2], [1], [2], [1], [2]]
    llama = [[1], [3], [1], [2], [1], [3]]
    with pytest.raises(ValueError, match="differ across tokenization modes"):
        summarize_results(llama, source, corpus=("same", "different"))
