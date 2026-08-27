# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Network-free tests for the Gemma4 tokenizer oracle refresh workflow."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from generate_gemma4_tokenizer_oracle import (  # noqa: E402
    FIXED_INPUT_PREFIX,
    FIXED_INPUT_SUFFIX,
    MODES,
    RANDOM_ALPHABET,
    RANDOM_COUNT,
    RANDOM_LENGTH_STOP,
    ROUTE_FIXED_INPUTS,
    SEED,
    _generator_sha256,
    _write_or_check_fixture,
    build_corpus,
    ordered_results_sha256,
    render_fixture,
)

_FIXTURE = _ROOT / "tests/data/gguf_gemma4_tokenizer_oracle.json"


def _fixture() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def test_committed_fixture_binds_exact_generators_corpora_and_outputs() -> None:
    fixture = _fixture()
    sources = {item["path"]: item["sha256"] for item in fixture["generator_sources"]}
    assert sources == {
        path: _generator_sha256(_ROOT / path)
        for path in (
            "scripts/generate_gemma4_tokenizer_oracle.py",
            "scripts/generate_minicpm_tokenizer_oracle.py",
        )
    }
    assert fixture["seed"] == SEED
    assert fixture["random_count"] == RANDOM_COUNT
    assert fixture["random_length_stop_exclusive"] == RANDOM_LENGTH_STOP
    assert fixture["fixed_input_prefix"] == list(FIXED_INPUT_PREFIX)
    assert fixture["fixed_input_suffix"] == list(FIXED_INPUT_SUFFIX)
    assert fixture["route_fixed_inputs"] == {
        name: list(values) for name, values in ROUTE_FIXED_INPUTS.items()
    }
    assert fixture["random_alphabet"] == "".join(RANDOM_ALPHABET)
    assert fixture["modes"] == [list(mode) for mode in MODES]

    for route in fixture["routes"]:
        corpus = build_corpus(route["name"])
        outputs = route["expected_outputs"]
        assert route["case_count"] == len(outputs) == len(MODES) * len(corpus)
        assert route["ordered_results_sha256"] == ordered_results_sha256(corpus, outputs)
        assert route["native_tokenize_mismatch_count"] == 0
        assert route["native_detokenize_mismatch_count"] == 0
        assert route["official_copy_tokenize_mismatch_count"] > 0
        for token_ids, decoded_hex in outputs:
            assert all(type(token_id) is int for token_id in token_ids)
            bytes.fromhex(decoded_hex)


def test_fixture_is_canonical_utf8_lf_generator_output() -> None:
    rendered = render_fixture(_fixture())
    assert _FIXTURE.read_bytes() == rendered
    assert b"\r\n" not in rendered


def test_generator_hash_and_fixture_bytes_ignore_checkout_line_endings(
    tmp_path: Path,
) -> None:
    source = bytes("# UTF-8: café\nprint('Gemma4')\n", encoding="utf-8")
    lf_source = tmp_path / "generator-lf.py"
    crlf_source = tmp_path / "generator-crlf.py"
    lf_source.write_bytes(source)
    crlf_source.write_bytes(source.replace(b"\n", b"\r\n"))

    assert _generator_sha256(lf_source) == _generator_sha256(crlf_source)
    rendered = render_fixture(
        {
            "generator_sha256": _generator_sha256(lf_source),
            "routes": [{"expected_outputs": []}],
        }
    )
    output = tmp_path / "oracle.json"
    _write_or_check_fixture(output, rendered, check=False)
    _write_or_check_fixture(output, rendered, check=True)
    output.write_bytes(rendered.replace(b"\n", b"\r\n"))
    with pytest.raises(SystemExit, match="is stale; regenerate without --check"):
        _write_or_check_fixture(output, rendered, check=True)
