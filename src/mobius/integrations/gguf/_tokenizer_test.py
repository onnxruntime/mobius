# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Closure and fail-closed tests for GGUF tokenizer handling."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from mobius.integrations.gguf._tokenizer import inspect_gguf_tokenizer
from mobius.integrations.gguf._tokenizer_registry import tokenizer_pre_policies

_PINNED_PRE_IDENTIFIERS = (
    "default",
    "minicpm5",
    "llama3",
    "llama-v3",
    "llama-bpe",
    "falcon3",
    "falcon-h1",
    "pixtral",
    "midm-2.0",
    "lfm2",
    "jina-v5-nano",
    "deepseek-llm",
    "deepseek-coder",
    "deepseek-v3",
    "youtu",
    "falcon",
    "mpt",
    "starcoder",
    "gpt-2",
    "phi-2",
    "jina-es",
    "jina-de",
    "gigachat",
    "jina-v2-es",
    "jina-v2-de",
    "a.x-4.0",
    "mellum",
    "modern-bert",
    "jais-2",
    "gemma4",
    "granite-embed-multi-311m",
    "sarvam-moe",
    "jina-v1-en",
    "jina-v2-code",
    "roberta-bpe",
    "whitespace",
    "refact",
    "command-r",
    "qwen2",
    "deepseek-r1-qwen",
    "kormo",
    "f2llmv2",
    "qwen35",
    "stablelm2",
    "olmo",
    "dbrx",
    "smaug-bpe",
    "poro-chat",
    "glm4",
    "chatglm-bpe",
    "viking",
    "jais",
    "tekken",
    "smollm",
    "codeshell",
    "bloom",
    "gpt3-finnish",
    "exaone",
    "exaone4",
    "exaone-moe",
    "chameleon",
    "minerva-7b",
    "megrez",
    "gpt-4o",
    "llama4",
    "kanana2",
    "talkie",
    "granite-embed-multi-97m",
    "tiny_aya",
    "cohere2moe",
    "superbpe",
    "trillion",
    "granite-docling",
    "bailingmoe",
    "bailingmoe2",
    "llada-moe",
    "seed-coder",
    "hunyuan",
    "hunyuan-dense",
    "joyai-llm",
    "kimi-k2",
    "grok-2",
    "afmoe",
    "laguna",
    "minimax-m2",
    "solar-open",
    "mellum2",
)


def _tokenizer_json(tokens: list[str]) -> str:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE

    return Tokenizer(
        BPE(vocab={token: index for index, token in enumerate(tokens)}, merges=[])
    ).to_str()


def _metadata(*, pre: str = "gpt-2", embedded: bool = False) -> dict:
    tokens = ["<pad>", "<eos>", "<bos>", "<unk>", "h", "i", "hi"]
    metadata = {
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": pre,
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.token_type": [3, 3, 3, 2, 1, 1, 1],
        "tokenizer.ggml.scores": [0.0] * len(tokens),
        "tokenizer.ggml.merges": ["h i"],
        "tokenizer.ggml.bos_token_id": 2,
        "tokenizer.ggml.eos_token_id": 1,
        "tokenizer.ggml.padding_token_id": 0,
        "tokenizer.ggml.unknown_token_id": 3,
        "tokenizer.ggml.add_bos_token": True,
        "tokenizer.ggml.add_eos_token": False,
    }
    if embedded:
        metadata["tokenizer.huggingface.json"] = _tokenizer_json(tokens)
    return metadata


def _write_gguf(path: Path, metadata: dict) -> None:
    from gguf import GGUFWriter

    writer = GGUFWriter(str(path), "llama")
    for key, value in metadata.items():
        if isinstance(value, str):
            writer.add_string(key, value)
        elif isinstance(value, bool):
            writer.add_bool(key, value)
        elif isinstance(value, int):
            writer.add_uint32(key, value)
        else:
            writer.add_array(key, value)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.close()


class TestTokenizerPreCensus:
    def test_exact_pinned_identifier_closure(self):
        policies = tokenizer_pre_policies()
        assert tuple(policies) == _PINNED_PRE_IDENTIFIERS
        assert len(policies) == len(set(policies)) == 87
        assert all(policy.default_route == "deferred" for policy in policies.values())

    def test_only_exact_upstream_aliases_share_canonical_names(self):
        policies = tokenizer_pre_policies()
        assert policies["llama-v3"].canonical == policies["llama3"].canonical
        assert policies["dbrx"].canonical != policies["llama3"].canonical
        assert policies["megrez"].canonical != policies["qwen2"].canonical
        assert policies["exaone4"].canonical == policies["gpt-2"].canonical

    def test_generated_documentation_table_matches_registry(self):
        policies = tokenizer_pre_policies()
        documentation = (
            Path(__file__).parents[4] / "docs" / "api" / "build_from_gguf.md"
        ).read_text(encoding="utf-8")
        table = documentation.split("| Canonical | Exact identifiers |", 1)[1].split(
            "### Metadata audit", 1
        )[0]
        documented_rows = re.findall(r"^\| `([^`]+)` \| (.+) \|$", table, re.MULTILINE)
        expected_groups: dict[str, list[str]] = {}
        for identifier, policy in policies.items():
            expected_groups.setdefault(policy.canonical, []).append(identifier)
        assert documented_rows == [
            (canonical, ", ".join(f"`{identifier}`" for identifier in identifiers))
            for canonical, identifiers in expected_groups.items()
        ]
        documented = {
            identifier: canonical
            for canonical, identifiers in documented_rows
            for identifier in re.findall(r"`([^`]+)`", identifiers)
        }
        assert documented == {
            identifier: policy.canonical for identifier, policy in policies.items()
        }


class TestInspectGgufTokenizer:
    def test_known_pre_without_complete_pipeline_is_deferred(self):
        verdict = inspect_gguf_tokenizer(_metadata(pre="hunyuan-dense"))
        assert verdict.route == "deferred"
        assert verdict.pre == "hunyuan-dense"
        assert "compiled llama.cpp behavior" in verdict.reason

    def test_exact_embedded_json_is_copy_route(self):
        verdict = inspect_gguf_tokenizer(_metadata(embedded=True))
        assert verdict.route == "copy"
        assert verdict.materialized
        assert verdict.tokenizer_sha256

    def test_pinned_undefined_token_type_is_accepted(self):
        metadata = _metadata()
        metadata["tokenizer.ggml.token_type"][4] = 0
        assert inspect_gguf_tokenizer(metadata).route == "deferred"

    def test_unknown_pre_rejects_without_generic_fallback(self):
        with pytest.raises(ValueError, match=r"unknown tokenizer\.ggml\.pre"):
            inspect_gguf_tokenizer(_metadata(pre="future-generic-bpe"))

    def test_missing_bpe_pre_rejects_without_default_fallback(self):
        metadata = _metadata()
        metadata.pop("tokenizer.ggml.pre")
        with pytest.raises(ValueError, match="quality-degrading generic default"):
            inspect_gguf_tokenizer(metadata)

    @pytest.mark.parametrize(
        ("mutation", "message"),
        [
            (lambda value: value["tokenizer.ggml.tokens"].append("h"), "duplicate token"),
            (lambda value: value["tokenizer.ggml.token_type"].pop(), "token_type length"),
            (lambda value: value["tokenizer.ggml.scores"].append(0.0), "scores length"),
            (
                lambda value: value.__setitem__("tokenizer.ggml.merges", ["missing h"]),
                "outside the vocabulary",
            ),
            (
                lambda value: value.__setitem__("tokenizer.ggml.bos_token_id", 100),
                r"bos_token_id.*\[0, 7\)",
            ),
            (
                lambda value: value.__setitem__("tokenizer.ggml.add_bos_token", 1),
                "must be a boolean",
            ),
            (
                lambda value: value.__setitem__(
                    "tokenizer.ggml.precompiled_charsmap", [0, 256]
                ),
                "byte values",
            ),
            (
                lambda value: value.__setitem__("tokenizer.ggml.byte_fallback", True),
                "not a pinned GGUF key",
            ),
            (
                lambda value: value.__setitem__("tokenizer.ggml.suppress_tokens", [0, 0]),
                "duplicate token ids",
            ),
        ],
    )
    def test_malformed_metadata_rejects(self, mutation, message):
        metadata = _metadata()
        mutation(metadata)
        with pytest.raises(ValueError, match=message):
            inspect_gguf_tokenizer(metadata)

    def test_embedded_json_must_match_ordered_gguf_vocab(self):
        metadata = _metadata()
        metadata["tokenizer.huggingface.json"] = _tokenizer_json(
            list(reversed(metadata["tokenizer.ggml.tokens"]))
        )
        with pytest.raises(ValueError, match="first mismatch at id 0"):
            inspect_gguf_tokenizer(metadata)

    def test_multiple_chat_templates_require_exact_deterministic_inventory(self):
        metadata = _metadata(embedded=True)
        metadata.update(
            {
                "tokenizer.chat_template": "default-template",
                "tokenizer.chat_templates": ["tool_use", "default"],
                "tokenizer.chat_template.tool_use": "tool-template",
            }
        )
        assert inspect_gguf_tokenizer(metadata).route == "copy"
        metadata["tokenizer.chat_templates"].append("missing")
        with pytest.raises(ValueError, match="does not exactly match"):
            inspect_gguf_tokenizer(metadata)


def test_write_exact_tokenizer_assets_preserves_templates_and_flags(tmp_path: Path):
    from mobius.integrations.gguf import write_gguf_tokenizer_json

    metadata = _metadata(embedded=True)
    metadata.update(
        {
            "tokenizer.chat_template": "default-template",
            "tokenizer.chat_templates": ["default", "tool_use"],
            "tokenizer.chat_template.tool_use": "tool-template",
        }
    )
    source = tmp_path / "model.gguf"
    _write_gguf(source, metadata)
    output = tmp_path / "out"

    result = write_gguf_tokenizer_json(source, output)

    assert result == str(output / "tokenizer.json")
    assert (output / "tokenizer.json").read_bytes() == metadata[
        "tokenizer.huggingface.json"
    ].encode("utf-8")
    config = json.loads((output / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert config["add_bos_token"] is True
    assert config["add_eos_token"] is False
    assert config["chat_template"] == {
        "default": "default-template",
        "tool_use": "tool-template",
    }
    assert (output / "chat_template.jinja").read_text(encoding="utf-8") == ("default-template")
    manifest = json.loads(
        (output / "gguf_tokenizer_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["route"] == "copy"
    assert manifest["pre"] == "gpt-2"
    assert manifest["metadata_sha256"]
    assert manifest["pipeline_semantics"] == "delegated_to_embedded_tokenizer_json"
    assert manifest["ort_genai_compatible"] == "delegated"


def test_write_exact_tokenizer_assets_removes_stale_default_template(tmp_path: Path):
    from mobius.integrations.gguf import write_gguf_tokenizer_json

    source = tmp_path / "tokenizer.gguf"
    _write_gguf(source, _metadata(embedded=True))
    output = tmp_path / "output"
    output.mkdir()
    stale = output / "chat_template.jinja"
    stale.write_text("stale-template", encoding="utf-8")

    write_gguf_tokenizer_json(source, output)

    assert not stale.exists()
