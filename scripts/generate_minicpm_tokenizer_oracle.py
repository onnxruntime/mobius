# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Regenerate bounded MiniCPM tokenizer mismatch evidence from immutable inputs.

This explicit evidence-refresh workflow consumes only the first 16 MiB of each
pinned GGUF plus the two pinned official ``tokenizer.json`` files. It creates
temporary tokenizer-only GGUFs, invokes a tokenizer tool built from the pinned
llama.cpp checkout, and compares all token IDs with the official tokenizers.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import struct
import subprocess
import tempfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mobius.integrations.gguf._tokenizer_mismatch_evidence import (
    GGUFTokenizerMismatchEvidence,
    tokenizer_mismatch_evidence,
)
from mobius.integrations.gguf._upstream import UPSTREAM_COMMIT

SEED = 648
RANDOM_COUNT = 128
RANDOM_LENGTH_STOP = 97
MODES = (
    ("no-add", "no-parse-special"),
    ("no-add", "parse-special"),
    ("add-special", "parse-special"),
)
FIXED_INPUTS = (
    "",
    "a",
    "Hello, world! 12345",
    "Hello, \u4e16\u754c! \U0001f469\u200d\U0001f4bb\n",
    "\u4f60\u597d\uff0c\u4e16\u754c\uff01",
    "\u041f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440",
    "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645",
    "\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e",
    "\u09ac\u09be\u0982\u09b2\u09be \u09ad\u09be\u09b7\u09be",
    "\u0e20\u0e32\u0e29\u0e32\u0e44\u0e17\u0e22",
    "\ud55c\uad6d\uc5b4 \ud14c\uc2a4\ud2b8",
    "\u65e5\u672c\u8a9e\u30c6\u30b9\u30c8",
    "e\u0301 caf\u00e9 \ufb01 \uff21",
    "  leading\tand  repeated\nspaces  ",
    "<s>literal</s>",
    "<\u7528\u6237>\u95ee\u9898<AI>",
    "<|im_start|>user\n\u5de5\u5177?<|im_end|>\n",
    "can't CAN'T we'll I'D",
    "1234567890 \u0661\u0662\u0663 \u0967\u0968\u0969",
    "\U0001f469\u200d\U0001f4bb\U0001f3f3\ufe0f\u200d\U0001f308\ufffd",
)
RANDOM_ALPHABET = tuple(
    "abcXYZ  \t\n.,!?'-_0123456789"
    "\u4f60\u597d\u4e16\u754c\u7528\u6237\u5de5\u5177\u6d4b\u8bd5"
    "\u00e9\u00df\u03a9\u0416\u0645\u0631\u062d\u0628\u0627"
    "\u0928\u092e\u0938\u094d\u0924\u0947"
    "\u0e20\u0e32\u0e29\u0e32\u0e44\u0e17\u0e22"
    "\ud55c\uad6d\uc5b4\u65e5\u672c\u8a9e"
    "\U0001f469\U0001f4bb\U0001f680"
)
_ROUTES = (
    "minicpm-2b-q2-k-tokenizer-mismatch",
    "minicpm3-4b-q4-k-m-tokenizer-mismatch",
)
_SCALAR_FORMATS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_corpus() -> tuple[str, ...]:
    """Return the exact fixed-plus-seeded corpus in oracle execution order."""
    generator = random.Random(SEED)
    random_inputs = tuple(
        "".join(
            generator.choice(RANDOM_ALPHABET)
            for _ in range(generator.randrange(0, RANDOM_LENGTH_STOP))
        )
        for _ in range(RANDOM_COUNT)
    )
    return FIXED_INPUTS + random_inputs


def summarize_results(
    llama_results: Sequence[Sequence[int]],
    source_results: Sequence[Sequence[int]],
    *,
    corpus: Sequence[str],
) -> dict[str, object]:
    """Hash ordered results and derive every mismatch witness."""
    expected_count = len(MODES) * len(corpus)
    if len(llama_results) != expected_count or len(source_results) != expected_count:
        raise ValueError(f"Expected {expected_count} ordered results per implementation")
    mismatch_indices_by_mode = []
    for mode_index in range(len(MODES)):
        start = mode_index * len(corpus)
        mismatch_indices_by_mode.append(
            [
                index
                for index in range(len(corpus))
                if tuple(llama_results[start + index]) != tuple(source_results[start + index])
            ]
        )
    if not mismatch_indices_by_mode[0]:
        raise ValueError("Oracle refresh unexpectedly found exact source parity")
    if any(indices != mismatch_indices_by_mode[0] for indices in mismatch_indices_by_mode[1:]):
        raise ValueError("Oracle mismatch input sets differ across tokenization modes")
    first_index = mismatch_indices_by_mode[0][0]
    return {
        "ordered_results_sha256": _sha256(_json_bytes(llama_results)),
        "source_results_sha256": _sha256(_json_bytes(source_results)),
        "mismatch_count": sum(len(indices) for indices in mismatch_indices_by_mode),
        "mismatch_count_by_mode": [len(indices) for indices in mismatch_indices_by_mode],
        "mismatch_input_indices": mismatch_indices_by_mode[0],
        "first_mismatch": {
            "mode": list(MODES[0]),
            "text": corpus[first_index],
            "llamacpp_ids": list(llama_results[first_index]),
            "source_ids": list(source_results[first_index]),
        },
    }


class _HeaderReader:
    def __init__(self, data: bytes):
        self.data = memoryview(data)
        self.offset = 0

    def unpack(self, value_format: str) -> Any:
        parser = struct.Struct("<" + value_format)
        if self.offset + parser.size > len(self.data):
            raise ValueError("Bounded GGUF header ends before metadata is complete")
        values = parser.unpack_from(self.data, self.offset)
        self.offset += parser.size
        return values[0] if len(values) == 1 else values

    def string(self) -> str:
        length = self.unpack("Q")
        end = self.offset + length
        if end > len(self.data):
            raise ValueError("Bounded GGUF header ends inside a string")
        value = bytes(self.data[self.offset : end]).decode("utf-8")
        self.offset = end
        return value

    def value(self, value_type: int) -> Any:
        if value_type == 8:
            return self.string()
        if value_type == 9:
            item_type = self.unpack("I")
            length = self.unpack("Q")
            if item_type == 8:
                return [self.string() for _ in range(length)]
            item_format = _SCALAR_FORMATS.get(item_type)
            if item_format is None:
                raise ValueError(f"Unsupported GGUF array type {item_type}")
            return [self.unpack(item_format) for _ in range(length)]
        value_format = _SCALAR_FORMATS.get(value_type)
        if value_format is None:
            raise ValueError(f"Unsupported GGUF metadata type {value_type}")
        return self.unpack(value_format)


def _read_bounded_header(
    path: Path, evidence: GGUFTokenizerMismatchEvidence
) -> dict[str, Any]:
    with path.open("rb") as stream:
        data = stream.read(evidence.bounded_header_bytes + 1)
    if len(data) != evidence.bounded_header_bytes:
        raise ValueError(
            f"{path} must contain exactly {evidence.bounded_header_bytes} bounded bytes"
        )
    if _sha256(data) != evidence.bounded_header_sha256:
        raise ValueError(f"{path} bounded-header SHA-256 differs from pinned evidence")

    reader = _HeaderReader(data)
    if bytes(reader.data[:4]) != b"GGUF":
        raise ValueError(f"{path} does not start with GGUF magic")
    reader.offset = 4
    version = reader.unpack("I")
    tensor_count = reader.unpack("Q")
    metadata_count = reader.unpack("Q")
    if version != 3 or tensor_count != evidence.tensor_count:
        raise ValueError(f"{path} GGUF header identity differs from pinned evidence")

    metadata: dict[str, Any] = {}
    for _ in range(metadata_count):
        key = reader.string()
        metadata[key] = reader.value(reader.unpack("I"))

    from gguf import GGMLQuantizationType

    qtypes: Counter[str] = Counter()
    embedding_rows = None
    for _ in range(tensor_count):
        name = reader.string()
        dimensions = tuple(reader.unpack("Q") for _ in range(reader.unpack("I")))
        qtype = GGMLQuantizationType(reader.unpack("I")).name
        reader.unpack("Q")
        qtypes[qtype] += 1
        if name == "token_embd.weight":
            embedding_rows = dimensions[1]

    tokenizer_metadata = {
        key: value for key, value in metadata.items() if key.startswith("tokenizer.")
    }
    checks = {
        "metadata": (
            _sha256(_json_bytes(dict(sorted(tokenizer_metadata.items())))),
            evidence.tokenizer_metadata_sha256,
        ),
        "vocabulary": (
            _sha256(_json_bytes(metadata["tokenizer.ggml.tokens"])),
            evidence.ordered_vocabulary_sha256,
        ),
        "scores": (
            _sha256(_json_bytes(metadata["tokenizer.ggml.scores"])),
            evidence.ordered_scores_sha256,
        ),
        "types": (
            _sha256(_json_bytes(metadata["tokenizer.ggml.token_type"])),
            evidence.ordered_token_types_sha256,
        ),
    }
    mismatch = [name for name, (actual, expected) in checks.items() if actual != expected]
    if mismatch:
        raise ValueError(f"{path} differs in pinned {', '.join(mismatch)} identity")
    if metadata.get("general.architecture") != evidence.architecture:
        raise ValueError(f"{path} architecture differs from pinned evidence")
    if metadata.get("tokenizer.ggml.model") != evidence.tokenizer_model:
        raise ValueError(f"{path} tokenizer model differs from pinned evidence")
    if metadata.get("tokenizer.ggml.pre") != evidence.pre_identifier:
        raise ValueError(f"{path} tokenizer pre differs from pinned evidence")
    if "tokenizer.ggml.merges" in metadata:
        raise ValueError(f"{path} unexpectedly serializes merges")
    if tuple(sorted(qtypes.items())) != evidence.tensor_qtypes:
        raise ValueError(f"{path} tensor qtype inventory differs from pinned evidence")
    if embedding_rows != evidence.embedding_vocabulary_size:
        raise ValueError(f"{path} embedding rows differ from pinned evidence")
    return metadata


def _write_tokenizer_only_gguf(
    path: Path, metadata: dict[str, Any], *, architecture: str
) -> None:
    from gguf import GGUFWriter

    writer = GGUFWriter(path, architecture)
    writer.add_tokenizer_model(metadata["tokenizer.ggml.model"])
    writer.add_tokenizer_pre(metadata["tokenizer.ggml.pre"])
    writer.add_token_list(metadata["tokenizer.ggml.tokens"])
    writer.add_token_scores(metadata["tokenizer.ggml.scores"])
    writer.add_token_types(metadata["tokenizer.ggml.token_type"])
    writer.add_bos_token_id(metadata["tokenizer.ggml.bos_token_id"])
    writer.add_eos_token_id(metadata["tokenizer.ggml.eos_token_id"])
    writer.add_unk_token_id(metadata["tokenizer.ggml.unknown_token_id"])
    writer.add_add_bos_token(metadata["tokenizer.ggml.add_bos_token"])
    writer.add_add_eos_token(metadata["tokenizer.ggml.add_eos_token"])
    writer.add_chat_template(metadata["tokenizer.chat_template"])
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.close()


def _run_llama(
    executable: Path,
    model: Path,
    corpus: Sequence[str],
    *,
    jobs: int,
) -> list[list[int]]:
    work = [(mode, text) for mode in MODES for text in corpus]

    def tokenize(item: tuple[tuple[str, str], str]) -> list[int]:
        mode, text = item
        command = [str(executable), "-m", str(model), "--stdin", "--ids"]
        if mode[0] == "no-add":
            command.append("--no-bos")
        if mode[1] == "no-parse-special":
            command.append("--no-parse-special")
        result = subprocess.run(
            command,
            input=text,
            check=True,
            capture_output=True,
            text=True,
        )
        outputs = [line for line in result.stdout.splitlines() if line.startswith("[")]
        if not outputs:
            raise ValueError("llama-tokenize emitted no Python-parseable ID list")
        token_ids = json.loads(outputs[-1])
        if not isinstance(token_ids, list) or any(type(value) is not int for value in token_ids):
            raise ValueError("llama-tokenize emitted a malformed ID list")
        return token_ids

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        return list(executor.map(tokenize, work))


def _source_results(path: Path, corpus: Sequence[str]) -> list[list[int]]:
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise RuntimeError("The tokenizer evidence refresh requires tokenizers") from error
    tokenizer = Tokenizer.from_file(str(path))
    results = []
    for mode in MODES:
        tokenizer.encode_special_tokens = mode[1] == "no-parse-special"
        results.extend(
            tokenizer.encode(text, add_special_tokens=mode[0] == "add-special").ids
            for text in corpus
        )
    return results


def _route_fixture(
    evidence: GGUFTokenizerMismatchEvidence,
    *,
    header: Path,
    tokenizer_json: Path,
    llama_tokenize: Path,
    corpus: Sequence[str],
    jobs: int,
) -> dict[str, object]:
    tokenizer_asset = next(
        asset for asset in evidence.tokenizer_assets if asset[0] == "tokenizer.json"
    )
    tokenizer_payload = tokenizer_json.read_bytes()
    if (
        len(tokenizer_payload) != tokenizer_asset[1]
        or _sha256(tokenizer_payload) != tokenizer_asset[2]
    ):
        raise ValueError(f"{tokenizer_json} differs from pinned official tokenizer.json")
    metadata = _read_bounded_header(header, evidence)
    with tempfile.TemporaryDirectory(prefix=f"{evidence.architecture}-tokenizer-oracle-") as temp:
        tokenizer_only = Path(temp) / "tokenizer-only.gguf"
        _write_tokenizer_only_gguf(
            tokenizer_only,
            metadata,
            architecture=evidence.architecture,
        )
        llama_results = _run_llama(
            llama_tokenize,
            tokenizer_only,
            corpus,
            jobs=jobs,
        )
    source_results = _source_results(tokenizer_json, corpus)
    return {
        "evidence_id": evidence.evidence_id,
        "artifact_sha256": evidence.lfs_sha256,
        "bounded_header_bytes": evidence.bounded_header_bytes,
        "bounded_header_sha256": evidence.bounded_header_sha256,
        **summarize_results(llama_results, source_results, corpus=corpus),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llamacpp-source", type=Path, required=True)
    parser.add_argument("--minicpm-header", type=Path, required=True)
    parser.add_argument("--minicpm-tokenizer-json", type=Path, required=True)
    parser.add_argument("--minicpm3-header", type=Path, required=True)
    parser.add_argument("--minicpm3-tokenizer-json", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/data/gguf_minicpm_tokenizer_oracle.json"),
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--jobs", type=int, default=8)
    return parser


def _build_llama_tokenize(source: Path, build: Path, *, jobs: int) -> Path:
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_EXAMPLES=OFF",
            "-DLLAMA_BUILD_SERVER=OFF",
            "-DGGML_NATIVE=OFF",
        ],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build), "--target", "llama-tokenize", f"-j{jobs}"],
        check=True,
    )
    candidates = (build / "bin/llama-tokenize", build / "bin/llama-tokenize.exe")
    executable = next((path for path in candidates if path.is_file()), None)
    if executable is None:
        raise ValueError("Fresh llama.cpp build did not produce llama-tokenize")
    return executable


def main() -> None:
    args = _parser().parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")
    source_commit = subprocess.run(
        ["git", "-C", str(args.llamacpp_source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if source_commit != UPSTREAM_COMMIT:
        raise ValueError(
            f"llama.cpp source must be pinned to {UPSTREAM_COMMIT}, got {source_commit}"
        )
    tracked_changes = subprocess.run(
        [
            "git",
            "-C",
            str(args.llamacpp_source),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if tracked_changes:
        raise ValueError("llama.cpp source must be a clean checkout of the pinned commit")

    corpus = build_corpus()
    route_inputs = (
        (_ROUTES[0], args.minicpm_header, args.minicpm_tokenizer_json),
        (_ROUTES[1], args.minicpm3_header, args.minicpm3_tokenizer_json),
    )
    routes = []
    with tempfile.TemporaryDirectory(prefix="llamacpp-tokenizer-build-") as build:
        executable = _build_llama_tokenize(
            args.llamacpp_source,
            Path(build),
            jobs=args.jobs,
        )
        for evidence_id, header, tokenizer_json in route_inputs:
            evidence = tokenizer_mismatch_evidence(evidence_id)
            if evidence is None:
                raise ValueError(f"Missing mismatch evidence {evidence_id!r}")
            routes.append(
                _route_fixture(
                    evidence,
                    header=header,
                    tokenizer_json=tokenizer_json,
                    llama_tokenize=executable,
                    corpus=corpus,
                    jobs=args.jobs,
                )
            )

    payload = {
        "generator": "scripts/generate_minicpm_tokenizer_oracle.py",
        "generator_sha256": _sha256(Path(__file__).read_bytes()),
        "serialization": "UTF-8 compact JSON arrays (ensure_ascii=false; separators=(',',':'))",
        "llamacpp_commit": UPSTREAM_COMMIT,
        "seed": SEED,
        "random_count": RANDOM_COUNT,
        "random_length_stop_exclusive": RANDOM_LENGTH_STOP,
        "fixed_count": len(FIXED_INPUTS),
        "fixed_inputs": list(FIXED_INPUTS),
        "random_alphabet": "".join(RANDOM_ALPHABET),
        "random_alphabet_sha256": _sha256("".join(RANDOM_ALPHABET).encode()),
        "corpus_sha256": _sha256(_json_bytes(corpus)),
        "modes": [list(mode) for mode in MODES],
        "case_count_per_route": len(MODES) * len(corpus),
        "routes": routes,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.check:
        if args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"{args.output} is stale; regenerate without --check")
    else:
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
