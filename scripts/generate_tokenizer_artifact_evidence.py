# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Regenerate fail-closed evidence for the final tokenizer artifact alias groups.

The workflow consumes exact bounded GGUF headers and immutable official tokenizer
assets. It independently reconstructs deterministic GGUF padding, builds tokenizer-only
GGUFs, and compares pinned llama.cpp tokenization and detokenization with the official
tokenizers. It intentionally does not import Mobius tokenizer reconstruction code.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import lzma
import os
import random
import shutil
import struct
import subprocess
import tarfile
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

LLAMACPP_COMMIT = "8d9af256337d1a501250f9bbf4c0859a654bddd6"
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
    "\x00\r\n\u200b\u2060\ufe0f",
)
ROUTE_FIXED_INPUTS = {
    "bailingmoe": (
        "e\u0301",
        "\u00e9",
        "<|startoftext|><|endoftext|><|mask|>",
        "<|im_start|>user\nHello<|im_end|>",
        "[PAD157153]",
        "ordinary[PAD157183]text",
    ),
    "glm4": (
        "' \u597d",
        " .",
        "[gMASK]<|user|>hello<|assistant|>",
        "<|observation|><|endoftext|>",
        "[PAD154856]",
        "ordinary[PAD154879]text",
    ),
    "tiny_aya": (
        "\t 9",
        "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>",
        "hello<|END_OF_TURN_TOKEN|>",
        "<PAD><UNK>",
        "[PAD255032]",
        "ordinary[PAD262143]text",
    ),
}
RANDOM_ALPHABET = tuple(
    "abcXYZ  \t\n.,!?'-_0123456789"
    "\u4f60\u597d\u4e16\u754c\u7528\u6237\u5de5\u5177\u6d4b\u8bd5"
    "\u00e9\u00df\u03a9\u0416\u0645\u0631\u062d\u0628\u0627"
    "\u0928\u092e\u0938\u094d\u0924\u0947"
    "\u0e20\u0e32\u0e29\u0e32\u0e44\u0e17\u0e22"
    "\ud55c\uad6d\uc5b4\u65e5\u672c\u8a9e"
    "\U0001f469\U0001f4bb\U0001f680"
)
_GENERATOR_PATH = "scripts/generate_tokenizer_artifact_evidence.py"
_QUALIFICATION_INPUTS_PATH = "tests/data/gguf_tokenizer_artifact_inputs.tar.xz"
_QUALIFICATION_INPUTS_SIZE = 11_118_624
_QUALIFICATION_INPUTS_SHA256 = (
    "0795e448f2391d1e0bd255599ba3f6a4212e0abad6ea438037ec366c8eb661fb"
)
_QUALIFICATION_MAX_DECOMPRESSED_BYTES = 128 * 1024 * 1024
_DOWNLOAD_LIMIT_BYTES = 16 * 1024**3


@dataclasses.dataclass(frozen=True, slots=True)
class Asset:
    filename: str
    size: int
    sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class Route:
    name: str
    evidence_id: str
    identifiers: tuple[str, ...]
    declared_pre_identifier: str
    pre_type: str
    discriminator_pre_identifier: str
    artifact_repository: str
    artifact_revision: str
    artifact_filename: str
    artifact_size: int
    artifact_sha256: str
    bounded_header_bytes: int
    bounded_header_sha256: str
    architecture: str
    tensor_count: int
    tensor_qtypes: tuple[tuple[str, int], ...]
    embedding_vocabulary_size: int
    tokenizer_repository: str
    tokenizer_revision: str
    source_assets: tuple[Asset, ...]


ROUTES = (
    Route(
        name="bailingmoe",
        evidence_id="llada-moe-iq1-s-tokenizer-semantic-blocker",
        identifiers=("bailingmoe", "bailingmoe2", "llada-moe"),
        declared_pre_identifier="llada-moe",
        pre_type="BAILINGMOE",
        discriminator_pre_identifier="glm4",
        artifact_repository="mradermacher/LLaDA-MoE-7B-A1B-Instruct-i1-GGUF",
        artifact_revision="2ec29fbe69f07f382a864f93b40c4eecb45e6a0a",
        artifact_filename="LLaDA-MoE-7B-A1B-Instruct.i1-IQ1_S.gguf",
        artifact_size=1_717_318_112,
        artifact_sha256=("d711df4b4f819d9abd0e107469dd525eb12d3bc05ec173b6a8438c172f70f3de"),
        bounded_header_bytes=6_492_640,
        bounded_header_sha256=(
            "1d1fe0fcc1660d86157e99bc015910f14387a24029b9ba77820c674abdc9fc85"
        ),
        architecture="llada-moe",
        tensor_count=195,
        tensor_qtypes=(
            ("F32", 81),
            ("IQ1_S", 78),
            ("IQ2_XXS", 16),
            ("Q2_K", 3),
            ("Q4_K", 16),
            ("Q5_K", 1),
        ),
        embedding_vocabulary_size=157_184,
        tokenizer_repository="inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
        tokenizer_revision="67004f662901b09f729994d4b3c04201283941ba",
        source_assets=(
            Asset(
                "config.json",
                1_424,
                "59b6b803a1bf500b45249cb553b3fa0425e4f1a431ba9e7028ef6bda33c97586",
            ),
            Asset(
                "special_tokens_map.json",
                153,
                "f1fa4f8b8c24126a0c2a5d9b2de0fee32abbddf22f48c068e5cf42bc0a9b68ab",
            ),
            Asset(
                "tokenizer.json",
                7_663_358,
                "4dd5931b0a63e3f61cfc1bcde132cd0c314de2f8a011ac9dbf2ff5efc40d0cbd",
            ),
            Asset(
                "tokenizer_config.json",
                4_593,
                "ac03e164668db350d26b13bee7bc65fd7c4bc74595ac1fc3952ebd707b0b44a5",
            ),
        ),
    ),
    Route(
        name="glm4",
        evidence_id="glm4-7-flash-iq2-xxs-tokenizer-semantic-blocker",
        identifiers=("chatglm-bpe", "glm4"),
        declared_pre_identifier="glm4",
        pre_type="CHATGLM4",
        discriminator_pre_identifier="bailingmoe",
        artifact_repository="bartowski/zai-org_GLM-4.7-Flash-GGUF",
        artifact_revision="464d07505b441959737cd04d900f047469614c8d",
        artifact_filename="zai-org_GLM-4.7-Flash-IQ2_XXS.gguf",
        artifact_size=7_622_864_768,
        artifact_sha256=("b1f25d90e0da65587a5a8e359b40a9183c5a31b4908b3ee5ff370e05cc5e2ba4"),
        bounded_header_bytes=9_475_456,
        bounded_header_sha256=(
            "803a3d88b31f81b5ac0fc541758af3c233d9d4a8e1abe7a601f2abaa52c5b382"
        ),
        architecture="deepseek2",
        tensor_count=844,
        tensor_qtypes=(
            ("F32", 281),
            ("IQ1_M", 108),
            ("IQ2_XS", 30),
            ("IQ2_XXS", 119),
            ("IQ4_NL", 47),
            ("Q2_K", 48),
            ("Q4_K", 143),
            ("Q5_K", 1),
            ("Q6_K", 67),
        ),
        embedding_vocabulary_size=154_880,
        tokenizer_repository="zai-org/GLM-4.7-Flash",
        tokenizer_revision="a9308079ef95921451a690cd2d16cb572e564642",
        source_assets=(
            Asset(
                "chat_template.jinja",
                3_120,
                "d63ad536c3c81880043e22ec7fd08db42b4d8fb7c89c7138bc562bfa25281375",
            ),
            Asset(
                "config.json",
                1_070,
                "dc9b97c7c9bed726a2e6939da4234d5c43abb3edec8812068c9a1af1dbc13acb",
            ),
            Asset(
                "tokenizer.json",
                20_217_442,
                "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
            ),
            Asset(
                "tokenizer_config.json",
                7_226,
                "31a173e2797ddc8b72ac996803513e627fc28d7aad02cfcce321a431d865c86d",
            ),
        ),
    ),
    Route(
        name="tiny_aya",
        evidence_id="north-mini-code-iq1-s-tokenizer-semantic-blocker",
        identifiers=("cohere2moe", "tiny_aya"),
        declared_pre_identifier="cohere2moe",
        pre_type="TINY_AYA",
        discriminator_pre_identifier="glm4",
        artifact_repository="mradermacher/North-Mini-Code-1.0-i1-GGUF",
        artifact_revision="94d8eb17eaeb728f907639ee0eff457e3e274667",
        artifact_filename="North-Mini-Code-1.0.i1-IQ1_S.gguf",
        artifact_size=6_455_984_128,
        artifact_sha256=("660792f0dd77ef2e39e92549bd88bbb0f91734371a3763816648fe77f23fb4dc"),
        bounded_header_bytes=10_428_416,
        bounded_header_sha256=(
            "9f38c617b8cd6fb3481cb1d73981cbbbd112e0e745b7cbd2e29c568ec11e76bb"
        ),
        architecture="cohere2moe",
        tensor_count=442,
        tensor_qtypes=(
            ("F32", 98),
            ("IQ1_S", 239),
            ("IQ2_XXS", 49),
            ("Q2_K", 6),
            ("Q4_K", 49),
            ("Q5_K", 1),
        ),
        embedding_vocabulary_size=262_144,
        tokenizer_repository="CohereLabs/North-Mini-Code-1.0",
        tokenizer_revision="d11e61a842617a22dc328552fa5bb86231ee4f37",
        source_assets=(
            Asset(
                "chat_template.jinja",
                12_397,
                "d8366efb9f07c571da620ce6a924594fc52c80273a0fbb46a38b643972df95fd",
            ),
            Asset(
                "config.json",
                2_342,
                "0c987a88193e90c89a88a9dbeaba6844f5f24d00b728683338e2ace1476509a7",
            ),
            Asset(
                "tokenizer.json",
                28_217_141,
                "14bd1c49d7d11874921d324986713df4be21cd06060530c497dacef99919b7a5",
            ),
            Asset(
                "tokenizer_config.json",
                8_954,
                "1f45bd13ca86efccb5f74bf51a78c5e06f9066a5d4211499c7f81890f31d1da2",
            ),
        ),
    ),
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
_SPECIAL_WRITERS = {
    "bos_token_id": "add_bos_token_id",
    "eos_token_id": "add_eos_token_id",
    "eot_token_id": "add_eot_token_id",
    "eom_token_id": "add_eom_token_id",
    "unknown_token_id": "add_unk_token_id",
    "padding_token_id": "add_pad_token_id",
    "seperator_token_id": "add_sep_token_id",
    "mask_token_id": "add_mask_token_id",
    "add_bos_token": "add_add_bos_token",
    "add_eos_token": "add_add_eos_token",
    "add_sep_token": "add_add_sep_token",
}

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


def _json_bytes(value: object, *, sort_keys: bool = False) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=sort_keys,
    ).encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _generator_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return _sha256(text.replace("\r\n", "\n").replace("\r", "\n").encode())


def _render_fixture(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode()


def _write_or_check(path: Path, payload: bytes, *, check: bool) -> None:
    if check:
        if path.read_bytes() != payload:
            raise SystemExit(f"{path} is stale; regenerate without --check")
    else:
        path.write_bytes(payload)


def build_corpus(route_name: str) -> tuple[str, ...]:
    """Return the exact multilingual and seeded adversarial route corpus."""
    generator = random.Random(SEED)
    random_inputs = tuple(
        "".join(
            generator.choice(RANDOM_ALPHABET)
            for _ in range(generator.randrange(0, RANDOM_LENGTH_STOP))
        )
        for _ in range(RANDOM_COUNT)
    )
    return ROUTE_FIXED_INPUTS[route_name] + FIXED_INPUTS + random_inputs


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


def _read_header_bytes(data: bytes, route: Route) -> dict[str, Any]:
    if len(data) != route.bounded_header_bytes or _sha256(data) != route.bounded_header_sha256:
        raise ValueError(f"{route.name} bounded GGUF header identity differs")
    reader = _HeaderReader(data)
    if bytes(reader.data[:4]) != b"GGUF":
        raise ValueError(f"{route.name} bounded input does not start with GGUF magic")
    reader.offset = 4
    version = reader.unpack("I")
    tensor_count = reader.unpack("Q")
    metadata_count = reader.unpack("Q")
    if version != 3 or tensor_count != route.tensor_count:
        raise ValueError(f"{route.name} GGUF version or tensor count differs")
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
        raise ValueError(f"{route.name} bounded input does not end at tensor data")
    if any(key.startswith("split.") for key in metadata):
        raise ValueError(f"{route.name} evidence requires a complete unsharded GGUF")
    if (
        metadata.get("general.architecture") != route.architecture
        or metadata.get("tokenizer.ggml.model") != "gpt2"
        or metadata.get("tokenizer.ggml.pre") != route.declared_pre_identifier
        or tuple(sorted(qtypes.items())) != route.tensor_qtypes
        or embedding_rows != route.embedding_vocabulary_size
    ):
        raise ValueError(f"{route.name} GGUF route or tensor identity differs")
    return metadata


def _load_source_assets(route: Route, payloads: Mapping[str, bytes]) -> dict[str, bytes]:
    expected = {asset.filename: asset for asset in route.source_assets}
    if set(payloads) != set(expected):
        raise ValueError(f"{route.name} source asset inventory differs")
    result = {}
    for filename, asset in expected.items():
        payload = payloads[filename]
        if len(payload) != asset.size or _sha256(payload) != asset.sha256:
            raise ValueError(f"{route.name} official {filename} identity differs")
        result[filename] = payload
    return result


def _token_strings(tokenizer: Any) -> list[str | None]:
    return [
        tokenizer.id_to_token(index)
        for index in range(tokenizer.get_vocab_size(with_added_tokens=True))
    ]


def _added_token(value: Any, *, token_id: int, expected: str) -> dict[str, Any]:
    fields = {"content", "single_word", "lstrip", "rstrip", "normalized", "special"}
    if not isinstance(value, Mapping) or set(value) != fields or value["content"] != expected:
        raise ValueError(f"Official added token {token_id} differs from GGUF")
    if any(
        type(value[name]) is not bool
        for name in ("single_word", "lstrip", "rstrip", "normalized", "special")
    ):
        raise ValueError(f"Official added token {token_id} has non-boolean flags")
    return {"id": token_id, **value}


def _independent_materialization(
    metadata: Mapping[str, Any],
    payloads: Mapping[str, bytes],
) -> tuple[bytes, dict[str, Any]]:
    """Reconstruct typed GGUF padding without calling production materialization."""
    from tokenizers import Tokenizer

    tokenizer_json = json.loads(payloads["tokenizer.json"])
    tokenizer_config = json.loads(payloads["tokenizer_config.json"])
    if not isinstance(tokenizer_json, dict) or not isinstance(tokenizer_config, dict):
        raise TypeError("Official tokenizer assets must contain JSON objects")
    tokenizer = Tokenizer.from_str(payloads["tokenizer.json"].decode())
    actual_tokens = _token_strings(tokenizer)
    expected_tokens = metadata["tokenizer.ggml.tokens"]
    token_types = metadata["tokenizer.ggml.token_type"]
    if actual_tokens != expected_tokens[: len(actual_tokens)] or len(actual_tokens) >= len(
        expected_tokens
    ):
        raise ValueError("Official vocabulary is not a strict ordered GGUF prefix")
    model = tokenizer_json.get("model")
    if not isinstance(model, dict) or model.get("type") != "BPE":
        raise ValueError("Independent padding reconstruction requires BPE")
    vocab = model.get("vocab")
    if not isinstance(vocab, dict) or set(vocab.values()) != set(range(len(vocab))):
        raise ValueError("Official BPE vocabulary IDs are not contiguous")
    if any(
        vocab.get(token) != index for index, token in enumerate(expected_tokens[: len(vocab)])
    ):
        raise ValueError("Official BPE model vocabulary order differs from GGUF")

    decoder = tokenizer_config.get("added_tokens_decoder", {})
    if not isinstance(decoder, Mapping):
        raise TypeError("Official added_tokens_decoder must be an object")
    decoded = {}
    for raw_id, value in decoder.items():
        if not isinstance(raw_id, str) or not raw_id.isdecimal():
            raise ValueError("Official added token IDs must be decimal strings")
        token_id = int(raw_id)
        decoded[token_id] = _added_token(
            value,
            token_id=token_id,
            expected=expected_tokens[token_id],
        )
    added_tokens = tokenizer_json.get("added_tokens")
    if not isinstance(added_tokens, list):
        raise TypeError("Official tokenizer added_tokens must be a list")
    source_added = {
        token["id"]: token
        for token in added_tokens
        if isinstance(token, Mapping) and isinstance(token.get("id"), int)
    }
    for token_id, token in decoded.items():
        if token_id in source_added and source_added[token_id] != token:
            raise ValueError(f"Official tokenizer assets contradict token {token_id}")

    for token_id in range(len(vocab), len(expected_tokens)):
        token = expected_tokens[token_id]
        if token in vocab:
            raise ValueError(f"GGUF extension token {token_id} duplicates model vocabulary")
        vocab[token] = token_id
    for token_id in range(len(actual_tokens), len(expected_tokens)):
        token = decoded.get(token_id)
        if token is not None:
            if token_id not in source_added:
                added_tokens.append(token)
            continue
        if expected_tokens[token_id] != f"[PAD{token_id}]" or token_types[token_id] != 5:
            raise ValueError(f"GGUF extension token {token_id} is not typed padding")

    materialized = _json_bytes(tokenizer_json)
    reconstructed = Tokenizer.from_str(materialized.decode())
    if _token_strings(reconstructed) != expected_tokens:
        raise ValueError("Independent materialized vocabulary differs from GGUF")
    padding_samples = {
        len(actual_tokens),
        (len(actual_tokens) + len(expected_tokens) - 1) // 2,
        len(expected_tokens) - 1,
    }
    for token_id in padding_samples:
        token = expected_tokens[token_id]
        if token_id in reconstructed.encode(token, add_special_tokens=False).ids:
            raise ValueError(f"GGUF padding token {token_id} is matchable")
    return materialized, tokenizer_json


def _source_chat_template(payloads: Mapping[str, bytes]) -> str:
    config = json.loads(payloads["tokenizer_config.json"])
    config_template = config.get("chat_template")
    file_template = (
        payloads["chat_template.jinja"].decode() if "chat_template.jinja" in payloads else None
    )
    if config_template is not None and file_template is not None:
        if config_template != file_template:
            raise ValueError("Official tokenizer chat-template assets contradict each other")
    template = file_template if file_template is not None else config_template
    if not isinstance(template, str) or not template:
        raise ValueError("Official tokenizer requires one non-empty chat template")
    return template


def _semantic_inventory(
    metadata: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    materialized: bytes,
) -> dict[str, Any]:
    from tokenizers import Tokenizer

    tokenizer_json = json.loads(payloads["tokenizer.json"])
    tokenizer_config = json.loads(payloads["tokenizer_config.json"])
    source = Tokenizer.from_str(payloads["tokenizer.json"].decode())
    source_tokens = _token_strings(source)
    tokens = metadata["tokenizer.ggml.tokens"]
    model = tokenizer_json["model"]
    source_merges = [
        " ".join(merge) if isinstance(merge, list) else merge
        for merge in model.get("merges", ())
    ]
    if source_merges != metadata["tokenizer.ggml.merges"]:
        raise ValueError("Official ordered BPE merges differ from GGUF")
    if "tokenizer.ggml.scores" in metadata:
        raise ValueError("These BPE routes must not serialize token scores")
    token_types = metadata["tokenizer.ggml.token_type"]
    model_token_count = len(model["vocab"])
    padding_start = len(source_tokens)
    if any(
        token != f"[PAD{token_id}]" or token_types[token_id] != 5
        for token_id, token in enumerate(tokens[padding_start:], padding_start)
    ):
        raise ValueError("GGUF deterministic padding inventory differs")
    added_tokens = tokenizer_json.get("added_tokens", [])
    added_type_mismatches: set[int] = set()
    for token in added_tokens:
        token_id = token["id"]
        if token_id < model_token_count:
            continue
        expected_type = 3 if token["special"] else 4
        if token_types[token_id] != expected_type:
            added_type_mismatches.add(token_id)
    ordered_added_type_mismatches = sorted(added_type_mismatches)
    special_ids = {}
    for key, token_id in metadata.items():
        if not key.startswith("tokenizer.ggml.") or not key.endswith("_token_id"):
            continue
        if type(token_id) is not int:
            continue
        special_ids[key.removeprefix("tokenizer.ggml.")] = [
            tokens[token_id],
            token_id,
        ]
    source_template = _source_chat_template(payloads)
    gguf_template = metadata.get("tokenizer.chat_template")
    if source_template != gguf_template:
        raise ValueError("Official chat template differs from GGUF")
    pipeline = {
        name: tokenizer_json.get(name)
        for name in ("normalizer", "pre_tokenizer", "post_processor", "decoder")
    }
    return {
        "token_count": len(tokens),
        "source_token_count": len(source_tokens),
        "source_model_token_count": model_token_count,
        "embedding_vocabulary_size": len(tokens),
        "deterministic_padding_range": [padding_start, len(tokens) - 1],
        "ordered_vocabulary_sha256": _sha256(_json_bytes(tokens)),
        "source_vocabulary_sha256": _sha256(_json_bytes(source_tokens)),
        "merge_count": len(source_merges),
        "ordered_merges_sha256": _sha256(_json_bytes(source_merges)),
        "score_count": 0,
        "ordered_scores_sha256": None,
        "ordered_token_types_sha256": _sha256(_json_bytes(token_types)),
        "token_type_counts": [list(item) for item in sorted(Counter(token_types).items())],
        "source_added_token_count": len(added_tokens),
        "ordered_source_added_tokens_sha256": _sha256(_json_bytes(added_tokens)),
        "source_added_token_type_mismatch_count": len(ordered_added_type_mismatches),
        "source_added_token_type_mismatch_ids": ordered_added_type_mismatches,
        "special_token_ids": dict(sorted(special_ids.items())),
        "pipeline_sha256": _sha256(_json_bytes(pipeline, sort_keys=True)),
        "pipeline_component_sha256": {
            name: _sha256(_json_bytes(value, sort_keys=True))
            for name, value in pipeline.items()
        },
        "tokenizer_config_sha256": _sha256(_json_bytes(tokenizer_config, sort_keys=True)),
        "chat_template_sha256": _sha256(source_template.encode()),
        "normalizer": (
            "none"
            if pipeline["normalizer"] is None
            else str(pipeline["normalizer"].get("type"))
        ),
        "materialized_tokenizer_sha256": _sha256(materialized),
        "materialized_tokenizer_bytes": len(materialized),
    }


def _write_tokenizer_only_gguf(
    path: Path,
    metadata: Mapping[str, Any],
    *,
    pre_identifier: str,
) -> None:
    from gguf import GGUFWriter

    writer = GGUFWriter(path, metadata["general.architecture"])
    writer.add_tokenizer_model(metadata["tokenizer.ggml.model"])
    writer.add_tokenizer_pre(pre_identifier)
    writer.add_token_list(metadata["tokenizer.ggml.tokens"])
    if "tokenizer.ggml.scores" in metadata:
        writer.add_token_scores(metadata["tokenizer.ggml.scores"])
    writer.add_token_types(metadata["tokenizer.ggml.token_type"])
    writer.add_token_merges(metadata["tokenizer.ggml.merges"])
    for suffix, method_name in _SPECIAL_WRITERS.items():
        key = f"tokenizer.ggml.{suffix}"
        if key in metadata:
            getattr(writer, method_name)(metadata[key])
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
    if source_commit != LLAMACPP_COMMIT:
        raise ValueError(f"llama.cpp must be pinned to {LLAMACPP_COMMIT}, got {source_commit}")
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
    helper_source = build / "tokenizer-artifact-oracle.cpp"
    helper_source.write_text(_ORACLE_SOURCE, encoding="utf-8", newline="\n")
    executable = build / "tokenizer-artifact-oracle"
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
        for text in corpus:
            process.stdin.write(
                f"{int(add_mode == 'add-special')} "
                f"{int(parse_mode == 'parse-special')} "
                f"{text.encode().hex() or '-'}\n"
            )
    process.stdin.close()
    lines = process.stdout.readlines()
    if process.wait() != 0 or len(lines) != len(MODES) * len(corpus):
        raise ValueError("Pinned llama.cpp oracle emitted an incomplete result set")
    results = []
    for line in lines:
        ids_text, decoded_hex = line.rstrip("\n").split("\t")
        token_ids = [int(value) for value in ids_text.split(",") if value]
        bytes.fromhex(decoded_hex)
        results.append((token_ids, decoded_hex))
    return results


def _tokenizer_outputs(
    raw_tokenizer: bytes,
    corpus: Sequence[str],
    metadata: Mapping[str, Any],
) -> list[tuple[list[int], str]]:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_str(raw_tokenizer.decode())
    results = []
    for add_mode, parse_mode in MODES:
        add_special = add_mode == "add-special"
        tokenizer.encode_special_tokens = parse_mode == "no-parse-special"
        for text in corpus:
            ids = tokenizer.encode(text, add_special_tokens=add_special).ids
            decode_ids = list(ids)
            if add_special:
                bos_id = metadata.get("tokenizer.ggml.bos_token_id")
                if (
                    metadata.get("tokenizer.ggml.add_bos_token") is True
                    and decode_ids
                    and decode_ids[0] == bos_id
                ):
                    decode_ids.pop(0)
                eos_id = metadata.get("tokenizer.ggml.eos_token_id")
                if (
                    metadata.get("tokenizer.ggml.add_eos_token") is True
                    and decode_ids
                    and decode_ids[-1] == eos_id
                ):
                    decode_ids.pop()
            decoded = tokenizer.decode(decode_ids, skip_special_tokens=False).encode().hex()
            results.append((ids, decoded))
    return results


def _results_sha256(outputs: Sequence[tuple[Sequence[int], str]]) -> str:
    return _sha256(_json_bytes([[list(ids), decoded] for ids, decoded in outputs]))


def _mismatch_summary(
    corpus: Sequence[str],
    llama_outputs: Sequence[tuple[Sequence[int], str]],
    source_outputs: Sequence[tuple[Sequence[int], str]],
) -> dict[str, Any]:
    expected_count = len(MODES) * len(corpus)
    if len(llama_outputs) != expected_count or len(source_outputs) != expected_count:
        raise ValueError("Tokenizer oracle result counts differ")
    token_indices_by_mode = []
    decode_indices_by_mode = []
    for mode_index in range(len(MODES)):
        start = mode_index * len(corpus)
        token_indices_by_mode.append(
            [
                index
                for index in range(len(corpus))
                if list(llama_outputs[start + index][0])
                != list(source_outputs[start + index][0])
            ]
        )
        decode_indices_by_mode.append(
            [
                index
                for index in range(len(corpus))
                if llama_outputs[start + index][1] != source_outputs[start + index][1]
            ]
        )
    if not any(token_indices_by_mode):
        raise ValueError("Artifact evidence unexpectedly has exact source tokenization parity")

    first_token_mode = next(
        index for index, values in enumerate(token_indices_by_mode) if values
    )
    first_token_index = token_indices_by_mode[first_token_mode][0]
    token_offset = first_token_mode * len(corpus) + first_token_index
    first_decode = None
    if any(decode_indices_by_mode):
        first_decode_mode = next(
            index for index, values in enumerate(decode_indices_by_mode) if values
        )
        first_decode_index = decode_indices_by_mode[first_decode_mode][0]
        decode_offset = first_decode_mode * len(corpus) + first_decode_index
        first_decode = {
            "mode": list(MODES[first_decode_mode]),
            "text": corpus[first_decode_index],
            "token_ids": list(llama_outputs[decode_offset][0]),
            "llamacpp_hex": llama_outputs[decode_offset][1],
            "official_source_hex": source_outputs[decode_offset][1],
        }
    return {
        "tokenize_mismatch_count": sum(map(len, token_indices_by_mode)),
        "tokenize_mismatch_count_by_mode": list(map(len, token_indices_by_mode)),
        "tokenize_mismatch_input_indices_by_mode": token_indices_by_mode,
        "first_tokenize_mismatch": {
            "mode": list(MODES[first_token_mode]),
            "text": corpus[first_token_index],
            "llamacpp_ids": list(llama_outputs[token_offset][0]),
            "official_source_ids": list(source_outputs[token_offset][0]),
        },
        "detokenize_mismatch_count": sum(map(len, decode_indices_by_mode)),
        "detokenize_mismatch_count_by_mode": list(map(len, decode_indices_by_mode)),
        "detokenize_mismatch_input_indices_by_mode": decode_indices_by_mode,
        "first_detokenize_mismatch": first_decode,
    }


def _route_fixture(
    route: Route,
    *,
    header: bytes,
    payloads: Mapping[str, bytes],
    executable: Path,
    temporary: Path,
) -> dict[str, Any]:
    metadata = _read_header_bytes(header, route)
    assets = _load_source_assets(route, payloads)
    materialized, _ = _independent_materialization(metadata, assets)
    corpus = build_corpus(route.name)
    source_outputs = _tokenizer_outputs(assets["tokenizer.json"], corpus, metadata)
    materialized_outputs = _tokenizer_outputs(materialized, corpus, metadata)
    if source_outputs != materialized_outputs:
        raise ValueError("Independent deterministic padding changes tokenizer behavior")

    dispatch_outputs = {}
    for identifier in route.identifiers:
        model = temporary / f"{route.name}-{identifier}.gguf"
        _write_tokenizer_only_gguf(model, metadata, pre_identifier=identifier)
        dispatch_outputs[identifier] = _run_oracle(executable, model, corpus)
    declared_outputs = dispatch_outputs[route.declared_pre_identifier]
    if any(outputs != declared_outputs for outputs in dispatch_outputs.values()):
        raise ValueError(f"{route.name} aliases do not share pinned llama.cpp semantics")

    discriminator_model = temporary / f"{route.name}-discriminator.gguf"
    _write_tokenizer_only_gguf(
        discriminator_model,
        metadata,
        pre_identifier=route.discriminator_pre_identifier,
    )
    discriminator_outputs = _run_oracle(executable, discriminator_model, corpus)
    discriminator_mismatch_count = sum(
        left != right
        for left, right in zip(declared_outputs, discriminator_outputs, strict=True)
    )
    if discriminator_mismatch_count == 0:
        raise ValueError(f"{route.name} route discriminator did not change oracle results")

    tokenizer_metadata = {
        key: value for key, value in metadata.items() if key.startswith("tokenizer.")
    }
    return {
        "name": route.name,
        "evidence_id": route.evidence_id,
        "identifiers": list(route.identifiers),
        "pre_type": route.pre_type,
        "declared_pre_identifier": route.declared_pre_identifier,
        "artifact_repository": route.artifact_repository,
        "artifact_revision": route.artifact_revision,
        "artifact_filename": route.artifact_filename,
        "artifact_size": route.artifact_size,
        "artifact_sha256": route.artifact_sha256,
        "bounded_header_bytes": route.bounded_header_bytes,
        "bounded_header_sha256": route.bounded_header_sha256,
        "architecture": route.architecture,
        "tensor_count": route.tensor_count,
        "tensor_qtypes": [list(item) for item in route.tensor_qtypes],
        "tokenizer_repository": route.tokenizer_repository,
        "tokenizer_revision": route.tokenizer_revision,
        "tokenizer_assets": [
            [asset.filename, asset.size, asset.sha256] for asset in route.source_assets
        ],
        "tokenizer_metadata_sha256": _sha256(
            _json_bytes(dict(sorted(tokenizer_metadata.items())))
        ),
        "corpus_sha256": _sha256(_json_bytes(corpus)),
        "case_count": len(declared_outputs),
        "llamacpp_ordered_results_sha256": _results_sha256(declared_outputs),
        "official_source_ordered_results_sha256": _results_sha256(source_outputs),
        "materialized_ordered_results_sha256": _results_sha256(materialized_outputs),
        "dispatch_oracles": {
            identifier: _results_sha256(outputs)
            for identifier, outputs in sorted(dispatch_outputs.items())
        },
        "discriminator": {
            "pre_identifier": route.discriminator_pre_identifier,
            "ordered_results_sha256": _results_sha256(discriminator_outputs),
            "mismatch_count": discriminator_mismatch_count,
        },
        "inventory": _semantic_inventory(metadata, assets, materialized),
        **_mismatch_summary(corpus, declared_outputs, source_outputs),
    }


def _qualification_specs() -> dict[str, tuple[int, str]]:
    specs = {}
    for route in ROUTES:
        specs[f"{route.name}/header.gguf"] = (
            route.bounded_header_bytes,
            route.bounded_header_sha256,
        )
        for asset in route.source_assets:
            specs[f"{route.name}/source/{asset.filename}"] = (
                asset.size,
                asset.sha256,
            )
    return specs


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def render_qualification_inputs(
    route_inputs: Mapping[str, tuple[Path, Path]],
) -> bytes:
    """Render deterministic compressed evidence inputs from separately opened files."""
    members = {}
    for route in ROUTES:
        header_path, source_dir = route_inputs[route.name]
        header = header_path.read_bytes()
        _read_header_bytes(header, route)
        payloads = {
            asset.filename: (source_dir / asset.filename).read_bytes()
            for asset in route.source_assets
        }
        _load_source_assets(route, payloads)
        members[f"{route.name}/header.gguf"] = header
        members.update(
            {
                f"{route.name}/source/{filename}": payload
                for filename, payload in payloads.items()
            }
        )
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name, payload in sorted(members.items()):
            tar.addfile(_tar_info(name, len(payload)), io.BytesIO(payload))
    if len(archive.getvalue()) > _QUALIFICATION_MAX_DECOMPRESSED_BYTES:
        raise ValueError("Tokenizer qualification archive exceeds decompression limit")
    return lzma.compress(
        archive.getvalue(),
        format=lzma.FORMAT_XZ,
        check=lzma.CHECK_CRC64,
        preset=9 | lzma.PRESET_EXTREME,
    )


def _load_qualification_bytes(payload: bytes) -> dict[str, bytes]:
    raw = lzma.decompress(payload, format=lzma.FORMAT_XZ)
    if len(raw) > _QUALIFICATION_MAX_DECOMPRESSED_BYTES:
        raise ValueError("Tokenizer qualification archive exceeds decompression limit")
    specs = _qualification_specs()
    result = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tar:
        for member in tar:
            if not member.isreg():
                raise ValueError(
                    "Tokenizer qualification archive members must be regular files"
                )
            if member.name in result:
                raise ValueError(f"Tokenizer qualification archive duplicates {member.name!r}")
            expected = specs.get(member.name)
            if expected is None:
                raise ValueError(
                    f"Tokenizer qualification archive contains unexpected {member.name!r}"
                )
            expected_size, expected_sha256 = expected
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError(
                    f"Tokenizer qualification member {member.name!r} is unreadable"
                )
            member_payload = stream.read(expected_size + 1)
            if (
                member.size != expected_size
                or len(member_payload) != expected_size
                or _sha256(member_payload) != expected_sha256
            ):
                raise ValueError(
                    f"Tokenizer qualification member {member.name!r} identity differs"
                )
            result[member.name] = member_payload
    if set(result) != set(specs):
        raise ValueError("Tokenizer qualification archive member inventory differs")
    return result


def load_qualification_inputs(path: Path) -> dict[str, bytes]:
    """Verify and load the exact committed qualification archive."""
    if (
        path.stat().st_size != _QUALIFICATION_INPUTS_SIZE
        or _file_sha256(path) != _QUALIFICATION_INPUTS_SHA256
    ):
        raise ValueError("Tokenizer qualification archive compressed identity differs")
    return _load_qualification_bytes(path.read_bytes())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llamacpp-source", type=Path, required=True)
    for route in ROUTES:
        parser.add_argument(f"--{route.name}-header", type=Path, required=True)
        parser.add_argument(f"--{route.name}-source-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/data/gguf_tokenizer_artifact_evidence.json"),
    )
    parser.add_argument(
        "--qualification-inputs-output",
        type=Path,
        default=Path(_QUALIFICATION_INPUTS_PATH),
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--jobs", type=int, default=8)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")
    route_inputs = {
        route.name: (
            getattr(args, f"{route.name}_header"),
            getattr(args, f"{route.name}_source_dir"),
        )
        for route in ROUTES
    }
    qualification = render_qualification_inputs(route_inputs)
    _write_or_check(
        args.qualification_inputs_output,
        qualification,
        check=args.check,
    )
    inputs = _load_qualification_bytes(qualification)
    selected_artifact_bytes = sum(route.artifact_size for route in ROUTES)
    if selected_artifact_bytes > _DOWNLOAD_LIMIT_BYTES:
        raise ValueError("Selected tokenizer artifact evidence exceeds the 16 GiB cap")

    with tempfile.TemporaryDirectory(prefix="tokenizer-artifact-evidence-") as temp_dir:
        temporary = Path(temp_dir)
        executable = _build_oracle(
            args.llamacpp_source,
            temporary / "build",
            jobs=args.jobs,
        )
        fixtures = []
        for route in ROUTES:
            fixtures.append(
                _route_fixture(
                    route,
                    header=inputs[f"{route.name}/header.gguf"],
                    payloads={
                        asset.filename: inputs[f"{route.name}/source/{asset.filename}"]
                        for asset in route.source_assets
                    },
                    executable=executable,
                    temporary=temporary,
                )
            )
    root = Path(__file__).resolve().parents[1]
    payload = {
        "generator": _GENERATOR_PATH,
        "generator_sha256": _generator_sha256(root / _GENERATOR_PATH),
        "serialization": (
            "UTF-8 compact semantic arrays; fixture rendered as UTF-8 indented JSON with LF"
        ),
        "llamacpp_commit": LLAMACPP_COMMIT,
        "qualification_inputs": {
            "path": _QUALIFICATION_INPUTS_PATH,
            "size": len(qualification),
            "sha256": _sha256(qualification),
        },
        "download_budget": {
            "limit_bytes": _DOWNLOAD_LIMIT_BYTES,
            "selected_artifact_bytes": selected_artifact_bytes,
            "headroom_bytes": _DOWNLOAD_LIMIT_BYTES - selected_artifact_bytes,
            "selected_artifacts": [
                [
                    route.artifact_repository,
                    route.artifact_revision,
                    route.artifact_filename,
                    route.artifact_size,
                ]
                for route in ROUTES
            ],
        },
        "seed": SEED,
        "random_count": RANDOM_COUNT,
        "random_length_stop_exclusive": RANDOM_LENGTH_STOP,
        "fixed_inputs": list(FIXED_INPUTS),
        "route_fixed_inputs": {
            name: list(values) for name, values in ROUTE_FIXED_INPUTS.items()
        },
        "random_alphabet": "".join(RANDOM_ALPHABET),
        "modes": [list(mode) for mode in MODES],
        "routes": fixtures,
    }
    _write_or_check(args.output, _render_fixture(payload), check=args.check)


if __name__ == "__main__":
    main()
