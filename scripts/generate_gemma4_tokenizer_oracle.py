# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Regenerate Gemma4 native and Granite fallback tokenizer oracle evidence.

The workflow consumes exact bounded GGUF header prefixes and immutable official
tokenizer assets. It builds a tokenizer-only GGUF for each route, invokes a
fresh tokenizer helper linked to the pinned llama.cpp checkout, and commits the
exact ordered token IDs and detokenized UTF-8 bytes used by offline tests.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import random
import shutil
import string
import subprocess
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from generate_minicpm_tokenizer_oracle import (
    _generator_sha256,
    _HeaderReader,
    _json_bytes,
    _sha256,
    _write_or_check_fixture,
)

from mobius.integrations.gguf import _tokenizer
from mobius.integrations.gguf._tokenizer_census import tokenizer_route_census
from mobius.integrations.gguf._tokenizer_evidence import tokenizer_evidence
from mobius.integrations.gguf._upstream import UPSTREAM_COMMIT

SEED = 650
RANDOM_COUNT = 128
RANDOM_LENGTH_STOP = 193
MODES = (
    ("no-add", "no-parse-special"),
    ("no-add", "parse-special"),
    ("add-special", "parse-special"),
)
FIXED_INPUT_PREFIX = (
    "",
    " ",
    "  ",
    "\n",
    "\r\n",
    "\t",
    "  spaced  text\n",
    "I'm we'd they'll can't I'M",
    "0 1 12 123 1234 12345 000001234567890",
    "!?. ,;:—/\\[]{}()\n",
    "你好，世界！",  # noqa: RUF001
    "漢字かなカナ 한글",
    "Café — κόσμος 🚀",
    "e\u0301 é",
    "🙂🚀👩\u200d💻🇺🇳",
    "\x00\x01\x7f",
    "A/B\r\nC",
    "<pad><bos><eos><unk><mask>",
)
ROUTE_FIXED_INPUTS = {
    "gemma4": (
        "<|turn>user\nhello<turn|>",
        "<|image|><|audio|><|video|>",
    ),
    "granite-fallback": (
        "<start_of_turn>user\nhello<end_of_turn>",
        "<image_soft_token>",
    ),
}
FIXED_INPUT_SUFFIX = (
    "<|channel>",
    "<channel|>",
    "<|tool_call>",
    "<tool_call|>",
    "<|tool_response>",
    "<tool_response|>",
    '<|"|>',
    "prefix<|tool_response>suffix",
    "<|channel><|tool_call>x<tool_call|><tool_response|>",
    "中文123 punctuation!? byte\x00 end",
    "▁ literal marker",
    "a" * 257,
)
RANDOM_ALPHABET = tuple(
    string.ascii_letters
    + string.digits
    + string.punctuation
    + " \t\n\r"
    + "你好世界漢字かなカナ한글é\u0301🙂🚀▁"
    + "".join(chr(index) for index in range(1, 32))
)
_GENERATOR_PATH = "scripts/generate_gemma4_tokenizer_oracle.py"
_SHARED_GENERATOR_PATH = "scripts/generate_minicpm_tokenizer_oracle.py"


@dataclasses.dataclass(frozen=True, slots=True)
class Route:
    name: str
    artifact_sha256: str
    bounded_header_bytes: int
    bounded_header_sha256: str
    architecture: str
    tensor_count: int
    tensor_qtypes: tuple[tuple[str, int], ...]
    embedding_vocabulary_size: int


ROUTES = (
    Route(
        name="gemma4",
        artifact_sha256="3d95ada2a122c9c0b42803317239b64b262ac9226a307ff895b3d87eec0c2acd",
        bounded_header_bytes=15_819_776,
        bounded_header_sha256=(
            "b3b56545efc7996330838dfa3d946db9cfe6801d963e60128a379e360cf1188a"
        ),
        architecture="gemma4",
        tensor_count=601,
        tensor_qtypes=(
            ("BF16", 1),
            ("F32", 353),
            ("IQ2_S", 116),
            ("IQ3_S", 40),
            ("IQ3_XXS", 15),
            ("IQ4_XS", 4),
            ("Q3_K", 1),
            ("Q4_K", 71),
        ),
        embedding_vocabulary_size=262_144,
    ),
    Route(
        name="granite-fallback",
        artifact_sha256="4cddb0ecb0ee45fcd1da37c007a608662568477a76ce5e12c14f7b34f002709e",
        bounded_header_bytes=15_773_152,
        bounded_header_sha256=(
            "3f1bf7afbde35f65cfef3af2ef2e4e312062dd28c4a7713a1f6e81cc7b5d9f89"
        ),
        architecture="modern-bert",
        tensor_count=134,
        tensor_qtypes=(
            ("F32", 45),
            ("IQ4_NL", 20),
            ("IQ4_XS", 66),
            ("Q5_1", 2),
            ("Q6_K", 1),
        ),
        embedding_vocabulary_size=262_152,
    ),
)

_ORACLE_SOURCE = r"""
#include "llama.h"
#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static std::string from_hex(const std::string & value) {
    if (value == "-") return {};
    std::string result;
    for (size_t index = 0; index < value.size(); index += 2) {
        result.push_back((char) std::stoi(value.substr(index, 2), nullptr, 16));
    }
    return result;
}

static std::string to_hex(const std::string & value) {
    static const char * digits = "0123456789abcdef";
    std::string result;
    for (unsigned char byte : value) {
        result.push_back(digits[byte >> 4]);
        result.push_back(digits[byte & 15]);
    }
    return result;
}

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    llama_backend_init();
    auto params = llama_model_default_params();
    params.vocab_only = true;
    auto * model = llama_model_load_from_file(argv[1], params);
    if (model == nullptr) return 3;
    const auto * vocab = llama_model_get_vocab(model);
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream input(line);
        int add_special = 0;
        int parse_special = 0;
        std::string encoded;
        input >> add_special >> parse_special >> encoded;
        const std::string text = from_hex(encoded);
        std::vector<llama_token> ids(std::max<size_t>(text.size() + 2, 8));
        int count = llama_tokenize(
            vocab, text.data(), text.size(), ids.data(), ids.size(),
            add_special, parse_special);
        if (count < 0) {
            ids.resize(-count);
            count = llama_tokenize(
                vocab, text.data(), text.size(), ids.data(), ids.size(),
                add_special, parse_special);
        }
        ids.resize(count);
        std::vector<char> decoded(std::max<size_t>(text.size() * 4 + 64, 256));
        int decoded_count = llama_detokenize(
            vocab, ids.data(), ids.size(), decoded.data(), decoded.size(),
            add_special, true);
        if (decoded_count < 0) {
            decoded.resize(-decoded_count);
            decoded_count = llama_detokenize(
                vocab, ids.data(), ids.size(), decoded.data(), decoded.size(),
                add_special, true);
        }
        for (size_t index = 0; index < ids.size(); ++index) {
            if (index != 0) std::cout << ',';
            std::cout << ids[index];
        }
        std::cout << '\t'
                  << to_hex(std::string(decoded.data(), decoded_count))
                  << '\n';
    }
    llama_model_free(model);
    llama_backend_free();
}
"""


def render_fixture(payload: object) -> bytes:
    """Render readable metadata with each large expected-output array compact."""
    rendered_payload = copy.deepcopy(payload)
    if not isinstance(rendered_payload, dict):
        raise TypeError("Gemma4 oracle fixture must contain a JSON object")
    routes = rendered_payload.get("routes")
    if not isinstance(routes, list):
        raise TypeError("Gemma4 oracle fixture must contain routes")
    replacements = {}
    for index, route in enumerate(routes):
        if not isinstance(route, dict) or "expected_outputs" not in route:
            raise TypeError("Gemma4 oracle route must contain expected outputs")
        marker = f"__EXPECTED_OUTPUTS_{index}__"
        replacements[json.dumps(marker)] = json.dumps(
            route["expected_outputs"],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        route["expected_outputs"] = marker
    rendered = json.dumps(rendered_payload, ensure_ascii=False, indent=2)
    for marker, expected_outputs in replacements.items():
        rendered = rendered.replace(marker, expected_outputs)
    return (rendered + "\n").encode("utf-8")


def build_corpus(route_name: str) -> tuple[str, ...]:
    """Return one route's exact fixed-plus-seeded corpus."""
    route_inputs = ROUTE_FIXED_INPUTS[route_name]
    generator = random.Random(SEED)
    random_inputs = tuple(
        "".join(
            generator.choice(RANDOM_ALPHABET)
            for _ in range(generator.randrange(0, RANDOM_LENGTH_STOP))
        )
        for _ in range(RANDOM_COUNT)
    )
    return FIXED_INPUT_PREFIX + route_inputs + FIXED_INPUT_SUFFIX + random_inputs


def _route_contract(route: Route) -> tuple[Any, tuple[tuple[str, int, str], ...]]:
    if route.name == "gemma4":
        evidence = tokenizer_evidence("gemma4-e2b-iq2-native-tokenizer")
        if evidence is None:
            raise ValueError("Missing Gemma4 tokenizer evidence")
        if (
            evidence.lfs_sha256 != route.artifact_sha256
            or evidence.architecture != route.architecture
            or evidence.tensor_count != route.tensor_count
            or evidence.tensor_qtypes != route.tensor_qtypes
            or evidence.embedding_vocabulary_size != route.embedding_vocabulary_size
        ):
            raise ValueError("Gemma4 generator route differs from tokenizer evidence")
        assets = tuple(sorted((evidence.source_config_asset, *evidence.tokenizer_assets)))
        return evidence, assets

    record = next(
        item
        for item in tokenizer_route_census()
        if item.identifier == "granite-embed-multi-311m"
    )
    if (
        record.evidence_id is not None
        or record.artifact_sha256 != route.artifact_sha256
        or record.artifact_architecture != route.architecture
        or record.declared_pre_identifier is not None
        or record.effective_pre_identifier != "gemma4"
    ):
        raise ValueError("Granite fallback generator route differs from deferred census")
    return record, record.tokenizer_assets


def _load_source_assets(
    route: Route, source_dir: Path
) -> tuple[dict[str, bytes], tuple[tuple[str, int, str], ...]]:
    _, assets = _route_contract(route)
    payloads = {}
    for filename, size, digest in assets:
        payload = (source_dir / filename).read_bytes()
        if len(payload) != size or _sha256(payload) != digest:
            raise ValueError(f"{source_dir / filename} differs from immutable evidence")
        payloads[filename] = payload
    return payloads, assets


def _read_header(path: Path, route: Route) -> dict[str, Any]:
    data = path.read_bytes()
    if len(data) != route.bounded_header_bytes or _sha256(data) != route.bounded_header_sha256:
        raise ValueError(f"{path} differs from the exact bounded GGUF header evidence")
    reader = _HeaderReader(data)
    if bytes(reader.data[:4]) != b"GGUF":
        raise ValueError(f"{path} does not start with GGUF magic")
    reader.offset = 4
    version = reader.unpack("I")
    tensor_count = reader.unpack("Q")
    metadata_count = reader.unpack("Q")
    if version != 3 or tensor_count != route.tensor_count:
        raise ValueError(f"{path} GGUF identity differs from the pinned route")
    metadata = {}
    for _ in range(metadata_count):
        key = reader.string()
        metadata[key] = reader.value(reader.unpack("I"))

    from gguf import GGMLQuantizationType

    qtypes: Counter[str] = Counter()
    embedding_rows = None
    for _ in range(tensor_count):
        name = reader.string()
        dimensions = tuple(reader.unpack("Q") for _ in range(reader.unpack("I")))
        qtypes[GGMLQuantizationType(reader.unpack("I")).name] += 1
        reader.unpack("Q")
        if name == "token_embd.weight":
            embedding_rows = dimensions[1]
    alignment = metadata.get("general.alignment", 32)
    data_start = (reader.offset + alignment - 1) // alignment * alignment
    if data_start != len(data):
        raise ValueError(f"{path} must end exactly at the aligned tensor-data boundary")
    if (
        metadata.get("general.architecture") != route.architecture
        or metadata.get("tokenizer.ggml.model") != "gemma4"
        or metadata.get("tokenizer.ggml.pre") is not None
        or tuple(sorted(qtypes.items())) != route.tensor_qtypes
        or embedding_rows != route.embedding_vocabulary_size
    ):
        raise ValueError(f"{path} tokenizer dispatch or tensor identity differs")
    return metadata


def _write_tokenizer_only_gguf(path: Path, metadata: Mapping[str, Any]) -> None:
    from gguf import GGUFWriter

    writer = GGUFWriter(path, metadata["general.architecture"])
    writer.add_tokenizer_model(metadata["tokenizer.ggml.model"])
    writer.add_token_list(metadata["tokenizer.ggml.tokens"])
    writer.add_token_scores(metadata["tokenizer.ggml.scores"])
    writer.add_token_types(metadata["tokenizer.ggml.token_type"])
    writer.add_token_merges(metadata["tokenizer.ggml.merges"])
    field_writers = (
        ("bos_token_id", writer.add_bos_token_id),
        ("eos_token_id", writer.add_eos_token_id),
        ("unknown_token_id", writer.add_unk_token_id),
        ("padding_token_id", writer.add_pad_token_id),
        ("seperator_token_id", writer.add_sep_token_id),
        ("mask_token_id", writer.add_mask_token_id),
        ("add_bos_token", writer.add_add_bos_token),
        ("add_eos_token", writer.add_add_eos_token),
        ("add_sep_token", writer.add_add_sep_token),
        ("add_space_prefix", writer.add_add_space_prefix),
    )
    for suffix, write in field_writers:
        key = f"tokenizer.ggml.{suffix}"
        if key in metadata:
            write(metadata[key])
    if "tokenizer.chat_template" in metadata:
        writer.add_chat_template(metadata["tokenizer.chat_template"])
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.close()


def _build_oracle(source: Path, build: Path, *, jobs: int) -> Path:
    source_commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if source_commit != UPSTREAM_COMMIT:
        raise ValueError(f"llama.cpp must be pinned to {UPSTREAM_COMMIT}, got {source_commit}")
    tracked_changes = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if tracked_changes:
        raise ValueError("llama.cpp source must be a clean pinned checkout")
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
        ["cmake", "--build", str(build), "--target", "llama", f"-j{jobs}"],
        check=True,
    )
    helper_source = build / "gemma4-tokenizer-oracle.cpp"
    helper_source.write_text(_ORACLE_SOURCE, encoding="utf-8", newline="\n")
    executable = build / "gemma4-tokenizer-oracle"
    compiler = os.environ.get("CXX") or shutil.which("c++")
    if compiler is None:
        raise RuntimeError("A C++ compiler is required to build the tokenizer oracle")
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            f"-I{source / 'include'}",
            f"-I{source / 'ggml/include'}",
            str(helper_source),
            f"-L{build / 'bin'}",
            f"-Wl,-rpath,{build / 'bin'}",
            "-lllama",
            "-o",
            str(executable),
        ],
        check=True,
    )
    return executable


def _run_oracle(
    executable: Path,
    model: Path,
    corpus: Sequence[str],
) -> list[tuple[list[int], str]]:
    process = subprocess.Popen(
        [str(executable), str(model)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("Failed to open tokenizer oracle pipes")
    for add_mode, parse_mode in MODES:
        add_special = add_mode == "add-special"
        parse_special = parse_mode == "parse-special"
        for text in corpus:
            process.stdin.write(
                f"{int(add_special)} {int(parse_special)} {text.encode().hex() or '-'}\n"
            )
    process.stdin.close()
    lines = process.stdout.readlines()
    if process.wait() != 0 or len(lines) != len(MODES) * len(corpus):
        raise ValueError("Pinned llama.cpp oracle emitted an incomplete result set")
    results = []
    for line in lines:
        ids_text, decoded_hex = line.rstrip("\n").split("\t")
        ids = [int(value) for value in ids_text.split(",") if value]
        bytes.fromhex(decoded_hex)
        results.append((ids, decoded_hex))
    return results


def _ordered_results(
    corpus: Sequence[str], outputs: Sequence[tuple[Sequence[int], str]]
) -> list[list[object]]:
    cases = [(mode, text) for mode in MODES for text in corpus]
    return [
        [
            [mode[0] == "add-special", mode[1] == "parse-special"],
            text.encode().hex(),
            list(ids),
            decoded_hex,
        ]
        for (mode, text), (ids, decoded_hex) in zip(cases, outputs, strict=True)
    ]


def ordered_results_sha256(
    corpus: Sequence[str], outputs: Sequence[tuple[Sequence[int], str]]
) -> str:
    """Recompute the historical ordered llama.cpp oracle digest."""
    normalized = [(list(item[0]), str(item[1])) for item in outputs]
    return _sha256(_json_bytes(_ordered_results(corpus, normalized)))


def _tokenizer_mismatch_counts(
    tokenizer_json: bytes,
    corpus: Sequence[str],
    outputs: Sequence[tuple[Sequence[int], str]],
    *,
    compare_decode: bool,
) -> tuple[int, int]:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_str(tokenizer_json.decode("utf-8"))
    token_mismatches = 0
    decode_mismatches = 0
    cases = [(mode, text) for mode in MODES for text in corpus]
    for (mode, text), (ids, decoded_hex) in zip(cases, outputs, strict=True):
        add_special = mode[0] == "add-special"
        tokenizer.encode_special_tokens = mode[1] == "no-parse-special"
        if tokenizer.encode(text, add_special_tokens=add_special).ids != list(ids):
            token_mismatches += 1
        if compare_decode:
            decode_ids = list(ids)
            if add_special and decode_ids and decode_ids[0] == 2:
                decode_ids.pop(0)
            decoded = tokenizer.decode(decode_ids, skip_special_tokens=False).encode()
            if decoded.hex() != decoded_hex:
                decode_mismatches += 1
    return token_mismatches, decode_mismatches


def _route_fixture(
    route: Route,
    *,
    header: Path,
    source_dir: Path,
    executable: Path,
    temp: Path,
) -> dict[str, object]:
    metadata = _read_header(header, route)
    payloads, _ = _load_source_assets(route, source_dir)
    if route.name == "granite-fallback":
        source_tokenizer = json.loads(payloads["tokenizer.json"])
        for token_id in range(262_145, 262_152):
            content = metadata["tokenizer.ggml.tokens"][token_id]
            source_tokenizer["added_tokens"].append(
                {
                    "id": token_id,
                    "content": content,
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": content == "<|tool_response>",
                }
            )
        source_tokenizer["added_tokens"] = [
            token for token in source_tokenizer["added_tokens"] if token["id"] != 105
        ]
        for old, new in _tokenizer._GEMMA4_CLEANUP_REPLACEMENTS:
            source_tokenizer["decoder"]["decoders"].append(
                {"type": "Replace", "pattern": {"String": old}, "content": new}
            )
        native_tokenizer = _json_bytes(source_tokenizer)
    else:
        _, native_tokenizer = _tokenizer._validate_pinned_tokenizer(
            metadata,
            payloads,
            reconstruct_gemma4_from_gguf=True,
        )
    tokenizer_only = temp / f"{route.name}.gguf"
    _write_tokenizer_only_gguf(tokenizer_only, metadata)
    corpus = build_corpus(route.name)
    outputs = _run_oracle(executable, tokenizer_only, corpus)
    native_token_mismatches, native_decode_mismatches = _tokenizer_mismatch_counts(
        native_tokenizer,
        corpus,
        outputs,
        compare_decode=True,
    )
    official_token_mismatches, _ = _tokenizer_mismatch_counts(
        payloads["tokenizer.json"],
        corpus,
        outputs,
        compare_decode=False,
    )
    if native_token_mismatches or native_decode_mismatches:
        raise ValueError(f"{route.name} native tokenizer differs from pinned llama.cpp")
    if official_token_mismatches == 0:
        raise ValueError(f"{route.name} official tokenizer unexpectedly has exact parity")
    result: dict[str, object] = {
        "name": route.name,
        "artifact_sha256": route.artifact_sha256,
        "bounded_header_bytes": route.bounded_header_bytes,
        "bounded_header_sha256": route.bounded_header_sha256,
        "case_count": len(outputs),
        "official_copy_tokenize_mismatch_count": official_token_mismatches,
        "native_tokenize_mismatch_count": native_token_mismatches,
        "native_detokenize_mismatch_count": native_decode_mismatches,
        "ordered_results_sha256": _sha256(_json_bytes(_ordered_results(corpus, outputs))),
        "expected_outputs": [[ids, decoded_hex] for ids, decoded_hex in outputs],
    }
    if route.name == "gemma4":
        result["evidence_id"] = "gemma4-e2b-iq2-native-tokenizer"
    else:
        result.update(
            {
                "requested_identifier": "granite-embed-multi-311m",
                "declared_pre_identifier": None,
                "effective_fallback_pre_identifier": "gemma4",
                "gguf_only_user_defined_tokens": [
                    ["<|channel>", 262_145],
                    ["<channel|>", 262_146],
                    ["<|tool_call>", 262_147],
                    ["<tool_call|>", 262_148],
                    ["<|tool_response>", 262_149],
                    ["<tool_response|>", 262_150],
                    ['<|"|>', 262_151],
                ],
                "forced_control_eog_ids": [262_149],
            }
        )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llamacpp-source", type=Path, required=True)
    parser.add_argument("--gemma4-header", type=Path, required=True)
    parser.add_argument("--gemma4-source-dir", type=Path, required=True)
    parser.add_argument("--granite-header", type=Path, required=True)
    parser.add_argument("--granite-source-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/data/gguf_gemma4_tokenizer_oracle.json"),
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--jobs", type=int, default=8)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")
    inputs = {
        "gemma4": (args.gemma4_header, args.gemma4_source_dir),
        "granite-fallback": (args.granite_header, args.granite_source_dir),
    }
    with tempfile.TemporaryDirectory(prefix="gemma4-tokenizer-oracle-") as temporary:
        temp = Path(temporary)
        executable = _build_oracle(args.llamacpp_source, temp / "build", jobs=args.jobs)
        routes = [
            _route_fixture(
                route,
                header=inputs[route.name][0],
                source_dir=inputs[route.name][1],
                executable=executable,
                temp=temp,
            )
            for route in ROUTES
        ]
    root = Path(__file__).resolve().parents[1]
    payload = {
        "generator_sources": [
            {
                "path": _GENERATOR_PATH,
                "sha256": _generator_sha256(root / _GENERATOR_PATH),
            },
            {
                "path": _SHARED_GENERATOR_PATH,
                "sha256": _generator_sha256(root / _SHARED_GENERATOR_PATH),
            },
        ],
        "serialization": (
            "UTF-8 compact JSON arrays (ensure_ascii=false; separators=(',',':')); "
            "fixture rendered as UTF-8 indented JSON with LF"
        ),
        "llamacpp_commit": UPSTREAM_COMMIT,
        "seed": SEED,
        "random_count": RANDOM_COUNT,
        "random_length_stop_exclusive": RANDOM_LENGTH_STOP,
        "fixed_input_prefix": list(FIXED_INPUT_PREFIX),
        "route_fixed_inputs": {
            name: list(values) for name, values in ROUTE_FIXED_INPUTS.items()
        },
        "fixed_input_suffix": list(FIXED_INPUT_SUFFIX),
        "random_alphabet": "".join(RANDOM_ALPHABET),
        "modes": [list(mode) for mode in MODES],
        "routes": routes,
    }
    _write_or_check_fixture(args.output, render_fixture(payload), check=args.check)


if __name__ == "__main__":
    main()
