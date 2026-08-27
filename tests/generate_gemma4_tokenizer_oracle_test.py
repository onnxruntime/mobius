# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Network-free tests for the Gemma4 tokenizer oracle refresh workflow."""

from __future__ import annotations

import hashlib
import io
import json
import lzma
import sys
import tarfile
from pathlib import Path

import pytest
from tokenizers import Tokenizer

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

import generate_gemma4_tokenizer_oracle as oracle_generator  # noqa: E402
from generate_gemma4_tokenizer_oracle import (  # noqa: E402
    FIXED_INPUT_PREFIX,
    FIXED_INPUT_SUFFIX,
    MODES,
    RANDOM_ALPHABET,
    RANDOM_COUNT,
    RANDOM_LENGTH_STOP,
    ROUTE_FIXED_INPUTS,
    ROUTES,
    SEED,
    _generator_sha256,
    _read_header,
    _write_or_check_fixture,
    build_corpus,
    load_qualification_inputs,
    ordered_results_sha256,
    render_fixture,
)

from mobius.integrations.gguf import _tokenizer  # noqa: E402
from mobius.integrations.gguf._tokenizer_evidence import tokenizer_evidence  # noqa: E402

_FIXTURE = _ROOT / "tests/data/gguf_gemma4_tokenizer_oracle.json"
_QUALIFICATION_INPUTS = _ROOT / "tests/data/gguf_gemma4_qualification_inputs.tar.xz"


def _fixture() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _write_test_archive(
    tmp_path: Path,
    entries: list[tuple[tarfile.TarInfo, bytes | None]],
) -> Path:
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for info, payload in entries:
            tar.addfile(info, None if payload is None else io.BytesIO(payload))
    path = tmp_path / "qualification.tar.xz"
    path.write_bytes(lzma.compress(archive.getvalue(), format=lzma.FORMAT_XZ))
    return path


def _regular_member(name: str, payload: bytes) -> tuple[tarfile.TarInfo, bytes]:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    return info, payload


def _trust_test_archive(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        oracle_generator,
        "_QUALIFICATION_INPUTS_SIZE",
        path.stat().st_size,
    )
    monkeypatch.setattr(
        oracle_generator,
        "_QUALIFICATION_INPUTS_SHA256",
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )


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
    assert fixture["qualification_inputs"] == {
        "path": "tests/data/gguf_gemma4_qualification_inputs.tar.xz",
        "sha256": hashlib.sha256(_QUALIFICATION_INPUTS.read_bytes()).hexdigest(),
    }

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


def test_loader_rejects_compressed_identity_before_decompression(tmp_path: Path) -> None:
    tampered = bytearray(_QUALIFICATION_INPUTS.read_bytes())
    tampered[len(tampered) // 2] ^= 1
    same_size = tmp_path / "tampered.tar.xz"
    same_size.write_bytes(tampered)
    with pytest.raises(ValueError, match="compressed identity differs"):
        load_qualification_inputs(same_size)

    truncated = tmp_path / "truncated.tar.xz"
    truncated.write_bytes(_QUALIFICATION_INPUTS.read_bytes()[:-1])
    with pytest.raises(ValueError, match="compressed identity differs"):
        load_qualification_inputs(truncated)


def test_loader_rejects_unapproved_and_duplicate_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_qualification_inputs(_QUALIFICATION_INPUTS)["source/config.json"]
    extra = _write_test_archive(
        tmp_path,
        [_regular_member("unexpected", b"")],
    )
    _trust_test_archive(extra, monkeypatch)
    with pytest.raises(ValueError, match="contains unexpected"):
        load_qualification_inputs(extra)

    duplicate = _write_test_archive(
        tmp_path,
        [
            _regular_member("source/config.json", config),
            _regular_member("source/config.json", config),
        ],
    )
    _trust_test_archive(duplicate, monkeypatch)
    with pytest.raises(ValueError, match="duplicates"):
        load_qualification_inputs(duplicate)


def test_loader_rejects_non_regular_and_incomplete_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_qualification_inputs(_QUALIFICATION_INPUTS)["source/config.json"]
    link = tarfile.TarInfo("source/config.json")
    link.type = tarfile.SYMTYPE
    link.linkname = "gemma4.header"
    linked = _write_test_archive(tmp_path, [(link, None)])
    _trust_test_archive(linked, monkeypatch)
    with pytest.raises(ValueError, match="must be regular files"):
        load_qualification_inputs(linked)

    incomplete = _write_test_archive(
        tmp_path,
        [_regular_member("source/config.json", config)],
    )
    _trust_test_archive(incomplete, monkeypatch)
    with pytest.raises(ValueError, match="member inventory differs"):
        load_qualification_inputs(incomplete)


def test_loader_rejects_oversize_bad_hash_and_decompression_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversize_info = tarfile.TarInfo("source/config.json")
    oversize_info.size = 1_000_000_000
    raw_oversize = oversize_info.tobuf(format=tarfile.USTAR_FORMAT) + b"\0" * 1024
    oversize = tmp_path / "oversize.tar.xz"
    oversize.write_bytes(lzma.compress(raw_oversize, format=lzma.FORMAT_XZ))
    _trust_test_archive(oversize, monkeypatch)
    with pytest.raises(ValueError, match=r"member 'source/config\.json' size differs"):
        load_qualification_inputs(oversize)

    evidence = tokenizer_evidence("gemma4-e2b-iq2-native-tokenizer")
    assert evidence is not None
    expected_size = next(
        size for name, size, _ in evidence.tokenizer_assets if name == "tokenizer_config.json"
    )
    bad_hash = _write_test_archive(
        tmp_path,
        [_regular_member("source/tokenizer_config.json", b"x" * expected_size)],
    )
    _trust_test_archive(bad_hash, monkeypatch)
    with pytest.raises(
        ValueError,
        match=r"member 'source/tokenizer_config\.json' hash differs",
    ):
        load_qualification_inputs(bad_hash)

    limited = _write_test_archive(
        tmp_path,
        [_regular_member("source/config.json", b"")],
    )
    _trust_test_archive(limited, monkeypatch)
    monkeypatch.setattr(oracle_generator, "_QUALIFICATION_MAX_DECOMPRESSED_BYTES", 100)
    with pytest.raises(ValueError, match="exceeds decompression limit"):
        load_qualification_inputs(limited)


def _materialize_current_gemma4(
    tmp_path: Path,
    *,
    check_materialized_digest: bool,
) -> tuple[Tokenizer, dict]:
    fixture = _fixture()
    inputs = load_qualification_inputs(_QUALIFICATION_INPUTS)
    assert set(inputs) == {
        "gemma4.header",
        "source/chat_template.jinja",
        "source/config.json",
        "source/tokenizer.json",
        "source/tokenizer_config.json",
    }
    evidence = tokenizer_evidence("gemma4-e2b-iq2-native-tokenizer")
    assert evidence is not None
    assets = (evidence.source_config_asset, *evidence.tokenizer_assets)
    for name, size, digest in assets:
        payload = inputs[f"source/{name}"]
        assert len(payload) == size
        assert hashlib.sha256(payload).hexdigest() == digest

    header = tmp_path / "gemma4.header"
    header.write_bytes(inputs["gemma4.header"])
    metadata = _read_header(header, ROUTES[0])
    payloads = {
        name.removeprefix("source/"): payload
        for name, payload in inputs.items()
        if name.startswith("source/")
    }
    digest, native = _tokenizer._validate_pinned_tokenizer(
        metadata,
        payloads,
        reconstruct_gemma4_from_gguf=True,
    )
    if check_materialized_digest:
        assert digest == evidence.materialized_tokenizer_sha256
    return Tokenizer.from_str(native.decode("utf-8")), fixture


def _assert_current_gemma4_matches_oracle(
    tmp_path: Path,
    *,
    check_materialized_digest: bool = True,
) -> None:
    tokenizer, fixture = _materialize_current_gemma4(
        tmp_path,
        check_materialized_digest=check_materialized_digest,
    )
    route = next(item for item in fixture["routes"] if item["name"] == "gemma4")
    corpus = build_corpus("gemma4")
    cases = [(mode, text) for mode in MODES for text in corpus]
    assert len(cases) == len(route["expected_outputs"]) == 480
    for index, ((mode, text), (expected_ids, expected_hex)) in enumerate(
        zip(cases, route["expected_outputs"], strict=True)
    ):
        add_special = mode[0] == "add-special"
        tokenizer.encode_special_tokens = mode[1] == "no-parse-special"
        actual_ids = tokenizer.encode(text, add_special_tokens=add_special).ids
        assert actual_ids == expected_ids, f"tokenization output {index} differs"
        decode_ids = list(expected_ids)
        if add_special and decode_ids and decode_ids[0] == 2:
            decode_ids.pop(0)
        actual_hex = tokenizer.decode(decode_ids, skip_special_tokens=False).encode().hex()
        assert actual_hex == expected_hex, f"detokenization output {index} differs"


def test_current_gemma4_reconstruction_matches_all_llamacpp_outputs(tmp_path: Path) -> None:
    _assert_current_gemma4_matches_oracle(tmp_path)


def test_reconstruction_behavior_drift_fails_oracle_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replacements = tuple(
        item for item in _tokenizer._GEMMA4_CLEANUP_REPLACEMENTS if item != (" ?", "?")
    )
    monkeypatch.setattr(_tokenizer, "_GEMMA4_CLEANUP_REPLACEMENTS", replacements)
    with pytest.raises(AssertionError, match="detokenization output"):
        _assert_current_gemma4_matches_oracle(
            tmp_path,
            check_materialized_digest=False,
        )


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
