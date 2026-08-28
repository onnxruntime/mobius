# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Closure and fail-closed tests for GGUF tokenizer handling."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from mobius.integrations.gguf import _tokenizer
from mobius.integrations.gguf._tokenizer import (
    GGUFTokenizerAsset,
    GGUFTokenizerSource,
    inspect_gguf_tokenizer,
    materialize_gguf_tokenizer,
)
from mobius.integrations.gguf._tokenizer_census import tokenizer_route_census
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
        from mobius.integrations.gguf._docs import check_document

        assert check_document()


class TestInspectGgufTokenizer:
    @pytest.mark.parametrize(
        ("metadata", "route", "reason"),
        [
            ({}, "absent", "contains no tokenizer metadata"),
            (
                {"tokenizer.ggml.tokens": ["token"]},
                "absent",
                "without tokenizer.ggml.model",
            ),
            (
                {"tokenizer.ggml.model": "llama"},
                "llama",
                "no complete tokenizer token table",
            ),
        ],
    )
    def test_incomplete_metadata_has_authoritative_deferred_diagnostics(
        self,
        metadata: dict,
        route: str,
        reason: str,
    ) -> None:
        verdict = inspect_gguf_tokenizer(metadata)

        assert verdict.route == "deferred"
        assert verdict.route_identifier == route
        assert verdict.audit_status == "deferred-incomplete-pipeline"
        assert verdict.blocker_category == "serialized-tokenizer-pipeline-incomplete"
        assert reason in verdict.reason

    @pytest.mark.parametrize(
        ("metadata", "message"),
        [
            (
                {"tokenizer.ggml.tokens": ["duplicate", "duplicate"]},
                "duplicate token strings",
            ),
            (
                {"tokenizer.ggml.tokens": ["valid", 1]},
                "contain only UTF-8 strings",
            ),
            (
                {
                    "tokenizer.ggml.model": "llama",
                    "tokenizer.ggml.token_type": [7],
                },
                "token_type length",
            ),
            (
                {
                    "tokenizer.ggml.model": "llama",
                    "tokenizer.chat_templates": ["missing"],
                },
                "does not exactly match",
            ),
            (
                {
                    "tokenizer.ggml.model": 1,
                    "tokenizer.ggml.tokens": ["valid"],
                },
                "must be a non-empty string",
            ),
            (
                {
                    "tokenizer.ggml.model": "llama",
                    "tokenizer.ggml.pre": 1,
                },
                "pre must be a non-empty string",
            ),
            (
                {
                    "tokenizer.ggml.tokens": [],
                    "tokenizer.huggingface.json": 42,
                },
                "tokenizer.huggingface.json must be a string",
            ),
            (
                {
                    "tokenizer.ggml.model": "llama",
                    "tokenizer.huggingface.json": "{",
                },
                "not valid JSON",
            ),
            (
                {
                    "tokenizer.ggml.model": "llama",
                    "tokenizer.ggml.eos_token_id": -1,
                },
                "eos_token_id must be an integer",
            ),
            (
                {
                    "tokenizer.ggml.model": "llama",
                    "tokenizer.ggml.pre": "hunyuan-dense",
                },
                "pre for non-BPE model",
            ),
            (
                {"tokenizer.rwkv.world": 42},
                "tokenizer.rwkv.world must be a string",
            ),
        ],
    )
    def test_incomplete_pipeline_does_not_downgrade_present_field_corruption(
        self,
        metadata: dict,
        message: str,
    ) -> None:
        with pytest.raises((TypeError, ValueError), match=message):
            inspect_gguf_tokenizer(metadata)

    def test_known_pre_without_complete_pipeline_is_deferred(self):
        verdict = inspect_gguf_tokenizer(_metadata(pre="hunyuan-dense"))
        assert verdict.route == "deferred"
        assert verdict.pre == "hunyuan-dense"
        assert "compiled llama.cpp behavior" in verdict.reason
        assert verdict.audit_status == "deferred-compiled-semantics"
        assert verdict.blocker_category == "compiled-llama.cpp-semantic-dependency"

    @pytest.mark.parametrize(
        "identifier",
        [
            "bailingmoe",
            "bailingmoe2",
            "llada-moe",
            "chatglm-bpe",
            "glm4",
            "cohere2moe",
            "tiny_aya",
        ],
    )
    def test_known_alias_blocker_uses_exact_authoritative_evidence(
        self, identifier: str
    ) -> None:
        audit = next(
            record for record in tokenizer_route_census() if record.identifier == identifier
        )
        verdict = inspect_gguf_tokenizer(_metadata(pre=identifier))

        assert verdict.audit_status == "deferred-pinned-artifact-mismatch"
        assert verdict.blocker_category == "pinned-candidate-source-semantic-mismatch"
        assert verdict.evidence_id == audit.blocker_evidence_id
        assert verdict.reason == audit.candidate_disposition

    def test_validated_route_is_not_mislabeled_as_supported_tokenizer_output(self) -> None:
        verdict = inspect_gguf_tokenizer(_metadata(pre="gpt-2"))

        assert verdict.route == "deferred"
        assert verdict.blocker_category is None
        assert verdict.evidence_id is None

    def test_plamo2_legacy_default_pre_is_validated_but_deferred(self):
        metadata = _metadata(pre="default")
        metadata["tokenizer.ggml.model"] = "plamo2"
        metadata.pop("tokenizer.ggml.merges")
        verdict = inspect_gguf_tokenizer(metadata)
        assert verdict.route == "deferred"
        assert verdict.pre == "default"
        assert verdict.canonical_pre == "default"

    @pytest.mark.parametrize("architecture", ["minicpm", "minicpm3"])
    def test_minicpm_legacy_default_pre_is_architecture_scoped_and_deferred(
        self, architecture: str
    ):
        metadata = _metadata(pre="default")
        metadata["general.architecture"] = architecture
        metadata["tokenizer.ggml.model"] = "llama"
        metadata.pop("tokenizer.ggml.merges")

        verdict = inspect_gguf_tokenizer(metadata)

        assert verdict.route == "deferred"
        assert verdict.pre == "default"
        assert verdict.canonical_pre == "default"
        assert "exact ORT tokenizer materialization is unavailable" in verdict.reason

    def test_legacy_default_pre_remains_rejected_for_other_sentencepiece_architectures(self):
        metadata = _metadata(pre="default")
        metadata["general.architecture"] = "llama"
        metadata["tokenizer.ggml.model"] = "llama"
        metadata.pop("tokenizer.ggml.merges")

        with pytest.raises(ValueError, match="pre for non-BPE model"):
            inspect_gguf_tokenizer(metadata)

    def test_legacy_default_pre_rejects_non_string_architecture_cleanly(self):
        metadata = _metadata(pre="default")
        metadata["general.architecture"] = ["minicpm"]
        metadata["tokenizer.ggml.model"] = "llama"
        metadata.pop("tokenizer.ggml.merges")

        with pytest.raises(ValueError, match="pre for non-BPE model"):
            inspect_gguf_tokenizer(metadata)

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


def _pinned_payloads(metadata: dict) -> dict[str, bytes]:
    tokenizer = json.loads(_tokenizer_json(metadata["tokenizer.ggml.tokens"]))
    tokenizer["model"]["merges"] = metadata["tokenizer.ggml.merges"]
    tokenizer["normalizer"] = None
    tokenizer["pre_tokenizer"] = {
        "type": "Sequence",
        "pretokenizers": [
            {"type": "Digits", "individual_digits": True},
            {
                "type": "ByteLevel",
                "add_prefix_space": False,
                "trim_offsets": True,
                "use_regex": True,
            },
        ],
    }
    tokenizer["post_processor"] = None
    tokenizer["decoder"] = {
        "type": "ByteLevel",
        "add_prefix_space": True,
        "trim_offsets": True,
        "use_regex": True,
    }
    tokenizer["added_tokens"] = [
        {
            "id": index,
            "content": token,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        for index, (token, token_type) in enumerate(
            zip(
                metadata["tokenizer.ggml.tokens"],
                metadata["tokenizer.ggml.token_type"],
                strict=True,
            )
        )
        if token_type in {2, 3}
    ]
    config = {
        "bos_token": metadata["tokenizer.ggml.tokens"][
            metadata["tokenizer.ggml.bos_token_id"]
        ],
        "eos_token": metadata["tokenizer.ggml.tokens"][
            metadata["tokenizer.ggml.eos_token_id"]
        ],
        "unk_token": metadata["tokenizer.ggml.tokens"][
            metadata["tokenizer.ggml.unknown_token_id"]
        ],
        "pad_token": metadata["tokenizer.ggml.tokens"][
            metadata["tokenizer.ggml.padding_token_id"]
        ],
        "add_bos_token": metadata["tokenizer.ggml.add_bos_token"],
        "add_eos_token": metadata["tokenizer.ggml.add_eos_token"],
    }
    return {
        "special_tokens_map.json": json.dumps(
            {name: value for name, value in config.items() if name.endswith("_token")}
        ).encode(),
        "tokenizer.json": json.dumps(tokenizer).encode(),
        "tokenizer_config.json": json.dumps(config).encode(),
    }


def _pinned_source(metadata: dict, payloads: dict[str, bytes]) -> GGUFTokenizerSource:
    verdict = inspect_gguf_tokenizer(metadata, require_complete=True)
    return GGUFTokenizerSource(
        repository="owner/tokenizer",
        revision="a" * 40,
        metadata_sha256=str(verdict.metadata_sha256),
        assets=tuple(
            GGUFTokenizerAsset(name, len(payload), hashlib.sha256(payload).hexdigest())
            for name, payload in sorted(payloads.items())
        ),
    )


def test_pinned_source_rejects_mutable_revision_and_duplicate_assets() -> None:
    asset = GGUFTokenizerAsset("tokenizer.json", 2, hashlib.sha256(b"{}").hexdigest())
    with pytest.raises(ValueError, match="immutable 40-hex"):
        GGUFTokenizerSource("owner/tokenizer", "main", (asset,), "a" * 64)
    for repository in ("owner/", "/tokenizer"):
        with pytest.raises(ValueError, match="owner/repository"):
            GGUFTokenizerSource(repository, "a" * 40, (asset,), "a" * 64)
    with pytest.raises(ValueError, match="duplicate asset"):
        GGUFTokenizerSource("owner/tokenizer", "a" * 40, (asset, asset), "a" * 64)


def test_pinned_source_missing_tokenizer_json_rejects() -> None:
    payload = b"{}"
    asset = GGUFTokenizerAsset(
        "tokenizer_config.json", len(payload), hashlib.sha256(payload).hexdigest()
    )
    with pytest.raises(ValueError, match=r"must include tokenizer\.json"):
        GGUFTokenizerSource("owner/tokenizer", "a" * 40, (asset,), "a" * 64)


def test_missing_pinned_hub_asset_leaves_no_output(tmp_path: Path, monkeypatch) -> None:
    metadata = _metadata(pre="smollm")
    payloads = _pinned_payloads(metadata)
    source = _pinned_source(metadata, payloads)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        mock.Mock(side_effect=FileNotFoundError("missing tokenizer asset")),
    )
    output = tmp_path / "output"

    with pytest.raises(FileNotFoundError, match="missing tokenizer asset"):
        materialize_gguf_tokenizer(
            tmp_path / "model.gguf",
            output,
            source=source,
            metadata=metadata,
            local_files_only=True,
        )

    assert not output.exists()


def test_tokenizer_evidence_metadata_mismatch_rejects_before_download(
    tmp_path: Path, monkeypatch
) -> None:
    metadata = _metadata(pre="smollm")
    metadata.pop("tokenizer.ggml.scores")
    payloads = _pinned_payloads(metadata)
    source = _pinned_source(metadata, payloads)
    source = GGUFTokenizerSource(
        source.repository,
        source.revision,
        source.assets,
        "b" * 64,
    )
    download = mock.Mock()
    monkeypatch.setattr(_tokenizer, "_download_tokenizer_assets", download)

    with pytest.raises(ValueError, match="does not match GGUF tokenizer metadata"):
        materialize_gguf_tokenizer(
            tmp_path / "model.gguf",
            tmp_path / "output",
            source=source,
            metadata=metadata,
        )

    download.assert_not_called()


def test_semantic_mismatch_leaves_no_partial_output(tmp_path: Path, monkeypatch) -> None:
    metadata = _metadata(pre="smollm")
    payloads = _pinned_payloads(metadata)
    config = json.loads(payloads["tokenizer_config.json"])
    config["bos_token"] = "<eos>"
    payloads["tokenizer_config.json"] = json.dumps(config).encode()
    source = _pinned_source(metadata, payloads)
    monkeypatch.setattr(_tokenizer, "_download_tokenizer_assets", lambda *_a, **_k: payloads)
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="bos_token id differs"):
        materialize_gguf_tokenizer(
            tmp_path / "model.gguf",
            output,
            source=source,
            metadata=metadata,
        )

    assert not output.exists()


@pytest.mark.parametrize(
    ("materialized_sha256", "encodings", "message"),
    [
        ("b" * 64, (), "digest differs"),
        (None, (("hello", (999,)),), "representative encoding differs"),
    ],
)
def test_compact_evidence_mismatch_leaves_no_partial_output(
    tmp_path: Path,
    monkeypatch,
    materialized_sha256: str | None,
    encodings: tuple[tuple[str, tuple[int, ...]], ...],
    message: str,
) -> None:
    metadata = _metadata(pre="smollm")
    metadata.pop("tokenizer.ggml.scores")
    payloads = _pinned_payloads(metadata)
    source = _pinned_source(metadata, payloads)
    source = GGUFTokenizerSource(
        source.repository,
        source.revision,
        source.assets,
        source.metadata_sha256,
        materialized_sha256,
        encodings,
    )
    monkeypatch.setattr(_tokenizer, "_download_tokenizer_assets", lambda *_a, **_k: payloads)
    output = tmp_path / "output"

    with pytest.raises(ValueError, match=message):
        materialize_gguf_tokenizer(
            tmp_path / "model.gguf",
            output,
            source=source,
            metadata=metadata,
        )

    assert not output.exists()


def test_special_encoding_evidence_mismatch_leaves_no_partial_output(
    tmp_path: Path, monkeypatch
) -> None:
    metadata = _metadata(pre="smollm")
    metadata.pop("tokenizer.ggml.scores")
    payloads = _pinned_payloads(metadata)
    source = _pinned_source(metadata, payloads)
    source = GGUFTokenizerSource(
        source.repository,
        source.revision,
        source.assets,
        source.metadata_sha256,
        representative_special_encodings=(("hello", (999,)),),
    )
    monkeypatch.setattr(_tokenizer, "_download_tokenizer_assets", lambda *_a, **_k: payloads)
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="representative special encoding differs"):
        materialize_gguf_tokenizer(
            tmp_path / "model.gguf",
            output,
            source=source,
            metadata=metadata,
        )

    assert not output.exists()


def test_gpt2_add_sep_accepts_exact_roberta_post_processor() -> None:
    metadata = _metadata(pre="jina-v2-code")
    metadata.pop("tokenizer.ggml.scores")
    metadata["tokenizer.ggml.add_eos_token"] = True
    metadata["tokenizer.ggml.seperator_token_id"] = 1
    payloads = _pinned_payloads(metadata)
    config = json.loads(payloads["tokenizer_config.json"])
    config.pop("add_bos_token")
    config.pop("add_eos_token")
    tokenizer = json.loads(payloads["tokenizer.json"])
    tokenizer["post_processor"] = {
        "type": "RobertaProcessing",
        "sep": ["<eos>", 1],
        "cls": ["<bos>", 2],
        "trim_offsets": True,
        "add_prefix_space": False,
    }
    payloads["tokenizer.json"] = json.dumps(tokenizer).encode()
    payloads["tokenizer_config.json"] = json.dumps(config).encode()

    _tokenizer._validate_pinned_tokenizer(metadata, payloads)


def test_gpt2_add_sep_rejects_post_processor_without_exact_sep() -> None:
    metadata = _metadata(pre="roberta-bpe")
    metadata.pop("tokenizer.ggml.scores")
    metadata["tokenizer.ggml.add_eos_token"] = True
    metadata["tokenizer.ggml.seperator_token_id"] = 1
    payloads = _pinned_payloads(metadata)
    config = json.loads(payloads["tokenizer_config.json"])
    config.pop("add_bos_token")
    config.pop("add_eos_token")
    tokenizer = json.loads(payloads["tokenizer.json"])
    tokenizer["post_processor"] = {
        "type": "RobertaProcessing",
        "sep": ["<unk>", 3],
        "cls": ["<bos>", 2],
        "trim_offsets": True,
        "add_prefix_space": False,
    }
    payloads["tokenizer.json"] = json.dumps(tokenizer).encode()
    payloads["tokenizer_config.json"] = json.dumps(config).encode()

    with pytest.raises(ValueError, match="cannot prove GGUF add_eos_token"):
        _tokenizer._validate_pinned_tokenizer(metadata, payloads)


def _gpt4o_payloads(metadata: dict) -> dict[str, bytes]:
    payloads = _pinned_payloads(metadata)
    config = json.loads(payloads["tokenizer_config.json"])
    config.pop("add_bos_token")
    tokenizer = json.loads(payloads["tokenizer.json"])
    tokenizer["pre_tokenizer"] = _tokenizer._GPT4O_PRE_TOKENIZER
    tokenizer["post_processor"] = {
        "type": "Sequence",
        "processors": [
            _tokenizer._GPT4O_POST_BYTE_LEVEL,
            {
                "type": "TemplateProcessing",
                "single": [
                    {"SpecialToken": {"id": "<bos>", "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}},
                ],
                "pair": [
                    {"SpecialToken": {"id": "<bos>", "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}},
                    {"SpecialToken": {"id": "<bos>", "type_id": 1}},
                    {"Sequence": {"id": "B", "type_id": 1}},
                ],
                "special_tokens": {"<bos>": {"id": "<bos>", "ids": [2], "tokens": ["<bos>"]}},
            },
        ],
    }
    payloads["tokenizer.json"] = json.dumps(tokenizer).encode()
    payloads["tokenizer_config.json"] = json.dumps(config).encode()
    return payloads


def test_gpt4o_accepts_exact_pipeline_and_template_bos_insertion() -> None:
    metadata = _metadata(pre="kanana2")
    metadata.pop("tokenizer.ggml.scores")

    _tokenizer._validate_pinned_tokenizer(metadata, _gpt4o_payloads(metadata))


def test_gpt4o_canonicalizes_official_regex_to_pinned_llamacpp_regex() -> None:
    metadata = _metadata(pre="kanana2")
    metadata.pop("tokenizer.ggml.scores")
    payloads = _gpt4o_payloads(metadata)
    tokenizer = json.loads(payloads["tokenizer.json"])
    tokenizer["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = (
        _tokenizer._GPT4O_SOURCE_SPLIT_PATTERN
    )
    payloads["tokenizer.json"] = json.dumps(tokenizer).encode()

    _, materialized = _tokenizer._validate_pinned_tokenizer(metadata, payloads)

    canonical = json.loads(materialized)
    assert (
        canonical["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
        == _tokenizer._GPT4O_SPLIT_PATTERN
    )
    assert canonical["model"]["ignore_merges"] is False


def test_gpt4o_native_reconstruction_uses_exact_gguf_merge_order() -> None:
    metadata = _metadata(pre="talkie")
    metadata.pop("tokenizer.ggml.scores")
    payloads = _gpt4o_payloads(metadata)
    tokenizer = json.loads(payloads["tokenizer.json"])
    tokenizer["model"]["merges"] = []
    payloads["tokenizer.json"] = json.dumps(tokenizer).encode()

    with pytest.raises(ValueError, match="merge order differs"):
        _tokenizer._validate_pinned_tokenizer(metadata, payloads)

    _, materialized = _tokenizer._validate_pinned_tokenizer(
        metadata,
        payloads,
        reconstruct_gpt4o_from_gguf=True,
    )
    assert json.loads(materialized)["model"]["merges"] == ["h i"]


@pytest.mark.parametrize("mismatch", ["regex", "decoder", "bos"])
def test_gpt4o_rejects_pipeline_or_template_bos_mismatch(mismatch: str) -> None:
    metadata = _metadata(pre="kanana2")
    metadata.pop("tokenizer.ggml.scores")
    payloads = _gpt4o_payloads(metadata)
    tokenizer = json.loads(payloads["tokenizer.json"])
    if mismatch == "regex":
        tokenizer["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] += "x"
    elif mismatch == "decoder":
        tokenizer["decoder"]["trim_offsets"] = False
    else:
        tokenizer["post_processor"]["processors"][1]["single"].reverse()
    payloads["tokenizer.json"] = json.dumps(tokenizer).encode()

    with pytest.raises(ValueError, match="pipeline differs"):
        _tokenizer._validate_pinned_tokenizer(metadata, payloads)


def test_gpt4o_rejects_template_bos_when_gguf_disables_it() -> None:
    metadata = _metadata(pre="kanana2")
    metadata.pop("tokenizer.ggml.scores")
    metadata["tokenizer.ggml.add_bos_token"] = False

    with pytest.raises(ValueError, match="pipeline differs"):
        _tokenizer._validate_pinned_tokenizer(metadata, _gpt4o_payloads(metadata))


def _gemma4_payloads(metadata: dict) -> dict[str, bytes]:
    tokenizer = json.loads(_tokenizer_json(metadata["tokenizer.ggml.tokens"]))
    tokenizer["model"]["byte_fallback"] = True
    tokenizer["model"]["merges"] = [["h", "i"]]
    tokenizer["normalizer"] = _tokenizer._GEMMA4_NORMALIZER
    tokenizer["pre_tokenizer"] = _tokenizer._GEMMA4_PRE_TOKENIZER
    tokenizer["post_processor"] = _tokenizer._GEMMA4_SOURCE_POST_PROCESSOR
    tokenizer["decoder"] = _tokenizer._GEMMA4_SOURCE_DECODER
    tokenizer["added_tokens"] = [
        {
            "id": token_id,
            "content": token,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": token_type == 3,
        }
        for token_id, (token, token_type) in enumerate(
            zip(
                metadata["tokenizer.ggml.tokens"],
                metadata["tokenizer.ggml.token_type"],
                strict=True,
            )
        )
        if token_type in {3, 4}
    ]
    return {
        "tokenizer.json": json.dumps(tokenizer).encode(),
        "tokenizer_config.json": json.dumps(
            {
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "chat_template": "official template",
            }
        ).encode(),
        "special_tokens_map.json": b"{}",
        "chat_template.jinja": b"official template",
    }


def _gemma4_metadata() -> dict:
    metadata = _metadata(pre="gemma4")
    metadata.pop("tokenizer.ggml.pre")
    metadata["tokenizer.ggml.model"] = "gemma4"
    metadata["tokenizer.ggml.scores"] = [-1000.0] * 9
    metadata["tokenizer.ggml.tokens"] += ["<turn|>", "<user>"]
    metadata["tokenizer.ggml.token_type"] += [4, 4]
    metadata["tokenizer.ggml.eos_token_id"] = 7
    metadata["tokenizer.chat_template"] = "native template"
    return metadata


def test_gemma4_native_reconstruction_applies_llamacpp_semantics() -> None:
    metadata = _gemma4_metadata()
    _, materialized = _tokenizer._validate_pinned_tokenizer(
        metadata,
        _gemma4_payloads(metadata),
        reconstruct_gemma4_from_gguf=True,
    )

    tokenizer = json.loads(materialized)
    added = {token["id"]: token for token in tokenizer["added_tokens"]}
    assert added[7]["special"] is True
    assert added[8]["special"] is False
    assert tokenizer["post_processor"] == _tokenizer._gemma4_post_processor(metadata)
    assert tokenizer["decoder"]["decoders"][-1] == {
        "type": "Replace",
        "pattern": {"String": " 've"},
        "content": "'ve",
    }


@pytest.mark.parametrize("mismatch", ["score", "pipeline", "added-token"])
def test_gemma4_reconstruction_fails_closed_on_semantic_mismatch(mismatch: str) -> None:
    metadata = _gemma4_metadata()
    payloads = _gemma4_payloads(metadata)
    if mismatch == "score":
        metadata["tokenizer.ggml.scores"][-1] = 0.0
    else:
        tokenizer = json.loads(payloads["tokenizer.json"])
        if mismatch == "pipeline":
            tokenizer["normalizer"] = None
        else:
            tokenizer["added_tokens"][-1]["content"] = "different"
        payloads["tokenizer.json"] = json.dumps(tokenizer).encode()

    with pytest.raises(ValueError):
        _tokenizer._validate_pinned_tokenizer(
            metadata,
            payloads,
            reconstruct_gemma4_from_gguf=True,
        )


def test_pipeline_mismatch_leaves_no_partial_output(tmp_path: Path, monkeypatch) -> None:
    metadata = _metadata(pre="smollm")
    payloads = _pinned_payloads(metadata)
    tokenizer = json.loads(payloads["tokenizer.json"])
    tokenizer["pre_tokenizer"]["pretokenizers"].reverse()
    payloads["tokenizer.json"] = json.dumps(tokenizer).encode()
    source = _pinned_source(metadata, payloads)
    monkeypatch.setattr(_tokenizer, "_download_tokenizer_assets", lambda *_a, **_k: payloads)
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="pipeline differs"):
        materialize_gguf_tokenizer(
            tmp_path / "model.gguf",
            output,
            source=source,
            metadata=metadata,
        )

    assert not output.exists()


def test_non_smollm_pipeline_is_bound_by_exact_asset_hashes() -> None:
    metadata = _metadata(pre="qwen2")
    metadata.pop("tokenizer.ggml.scores")
    payloads = _pinned_payloads(metadata)
    tokenizer = json.loads(payloads["tokenizer.json"])
    tokenizer["pre_tokenizer"]["pretokenizers"].reverse()
    payloads["tokenizer.json"] = json.dumps(tokenizer).encode()

    tokenizer_sha256, materialized = _tokenizer._validate_pinned_tokenizer(metadata, payloads)

    assert hashlib.sha256(materialized).hexdigest() == tokenizer_sha256


def test_special_added_token_proves_legacy_config_omission() -> None:
    metadata = _metadata(pre="gpt-2")
    metadata.pop("tokenizer.ggml.scores")
    payloads = _pinned_payloads(metadata)
    config = json.loads(payloads["tokenizer_config.json"])
    config.pop("bos_token")
    config.pop("eos_token")
    payloads["tokenizer_config.json"] = json.dumps(config).encode()

    tokenizer_sha256, materialized = _tokenizer._validate_pinned_tokenizer(metadata, payloads)

    assert hashlib.sha256(materialized).hexdigest() == tokenizer_sha256


def test_deterministic_unused_padding_extends_bpe_vocab_without_matching_input() -> None:
    source_metadata = _metadata(pre="qwen2")
    source_metadata.pop("tokenizer.ggml.scores")
    payloads = _pinned_payloads(source_metadata)
    metadata = dict(source_metadata)
    metadata["tokenizer.ggml.tokens"] = [
        *source_metadata["tokenizer.ggml.tokens"],
        "[PAD7]",
        "[PAD8]",
    ]
    metadata["tokenizer.ggml.token_type"] = [
        *source_metadata["tokenizer.ggml.token_type"],
        5,
        5,
    ]

    _, materialized = _tokenizer._validate_pinned_tokenizer(metadata, payloads)

    from tokenizers import Tokenizer

    tokenizer_json = json.loads(materialized)
    tokenizer = Tokenizer.from_str(materialized.decode())
    assert tokenizer_json["model"]["vocab"]["[PAD7]"] == 7
    assert tokenizer_json["model"]["vocab"]["[PAD8]"] == 8
    assert all(
        token["content"] not in {"[PAD7]", "[PAD8]"}
        for token in tokenizer_json["added_tokens"]
    )
    assert [tokenizer.id_to_token(index) for index in range(9)] == metadata[
        "tokenizer.ggml.tokens"
    ]
    for text in ("[PAD7]", "prefix[PAD7]suffix", "[PAD8]", "prefix[PAD8]suffix"):
        assert not ({7, 8} & set(tokenizer.encode(text, add_special_tokens=False).ids))


def test_matchable_unused_padding_reconstruction_fails_closed() -> None:
    source_metadata = _metadata(pre="qwen2")
    source_metadata.pop("tokenizer.ggml.scores")
    payloads = _pinned_payloads(source_metadata)
    tokenizer_json = json.loads(payloads["tokenizer.json"])
    tokenizer_json["model"]["ignore_merges"] = True
    tokenizer_json["pre_tokenizer"] = {"type": "WhitespaceSplit"}
    payloads["tokenizer.json"] = json.dumps(tokenizer_json).encode()
    metadata = dict(source_metadata)
    metadata["tokenizer.ggml.tokens"] = [
        *source_metadata["tokenizer.ggml.tokens"],
        "[PAD7]",
    ]
    metadata["tokenizer.ggml.token_type"] = [
        *source_metadata["tokenizer.ggml.token_type"],
        5,
    ]

    with pytest.raises(ValueError, match="unused padding token 7 is matchable"):
        _tokenizer._validate_pinned_tokenizer(metadata, payloads)


def test_source_config_and_unused_padding_reconstruct_exact_ordered_vocabulary() -> None:
    source_metadata = _metadata(pre="qwen35")
    source_metadata.pop("tokenizer.ggml.scores")
    payloads = _pinned_payloads(source_metadata)
    metadata = dict(source_metadata)
    metadata["tokenizer.ggml.tokens"] = [
        *source_metadata["tokenizer.ggml.tokens"],
        "<|audio_start|>",
        "[PAD8]",
    ]
    metadata["tokenizer.ggml.token_type"] = [
        *source_metadata["tokenizer.ggml.token_type"],
        3,
        5,
    ]
    config = json.loads(payloads["tokenizer_config.json"])
    config["added_tokens_decoder"] = {
        "7": {
            "content": "<|audio_start|>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
    }
    payloads["tokenizer_config.json"] = json.dumps(config).encode()

    _, materialized = _tokenizer._validate_pinned_tokenizer(metadata, payloads)

    from tokenizers import Tokenizer

    tokenizer_json = json.loads(materialized)
    tokenizer = Tokenizer.from_str(materialized.decode())
    assert tokenizer_json["model"]["vocab"]["<|audio_start|>"] == 7
    assert tokenizer_json["model"]["vocab"]["[PAD8]"] == 8
    assert tokenizer_json["added_tokens"][-1] == {
        "id": 7,
        "content": "<|audio_start|>",
        "single_word": False,
        "lstrip": False,
        "rstrip": False,
        "normalized": False,
        "special": True,
    }
    assert [tokenizer.id_to_token(index) for index in range(9)] == metadata[
        "tokenizer.ggml.tokens"
    ]
    assert tokenizer.encode("<|audio_start|>", add_special_tokens=False).ids == [7]
    for text in ("[PAD8]", "prefix[PAD8]suffix"):
        assert 8 not in tokenizer.encode(text, add_special_tokens=False).ids


def test_source_config_added_token_must_match_exact_gguf_id() -> None:
    source_metadata = _metadata(pre="qwen35")
    source_metadata.pop("tokenizer.ggml.scores")
    payloads = _pinned_payloads(source_metadata)
    metadata = dict(source_metadata)
    metadata["tokenizer.ggml.tokens"] = [
        *source_metadata["tokenizer.ggml.tokens"],
        "<|audio_start|>",
    ]
    metadata["tokenizer.ggml.token_type"] = [
        *source_metadata["tokenizer.ggml.token_type"],
        3,
    ]
    config = json.loads(payloads["tokenizer_config.json"])
    config["added_tokens_decoder"] = {
        "7": {
            "content": "<|wrong|>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
    }
    payloads["tokenizer_config.json"] = json.dumps(config).encode()

    with pytest.raises(ValueError, match="added token 7 differs from GGUF"):
        _tokenizer._validate_pinned_tokenizer(metadata, payloads)


def test_evidenced_materializer_rejects_existing_destination_before_source_read(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    read_source = mock.Mock()
    monkeypatch.setattr("mobius.integrations.gguf._reader.GGUFModel", read_source)

    with pytest.raises(FileExistsError, match="non-atomic directory replacement"):
        _tokenizer.materialize_evidenced_gguf_tokenizer(tmp_path / "model.gguf", output)

    read_source.assert_not_called()


def test_evidenced_materializer_rejects_replaced_source_before_publication(
    tmp_path: Path, monkeypatch
) -> None:
    model = SimpleNamespace(
        metadata={},
        source_matches_path=mock.Mock(side_effect=(True, False)),
    )
    evidence = SimpleNamespace(
        source=object(),
        repository="owner/model",
        revision="a" * 40,
        filename="model.gguf",
        lfs_sha256="b" * 64,
    )
    monkeypatch.setattr("mobius.integrations.gguf._reader.GGUFModel", lambda _path: model)
    monkeypatch.setattr(
        "mobius.integrations.gguf._tokenizer.inspect_gguf_tokenizer",
        lambda *_a, **_k: SimpleNamespace(metadata_sha256="c" * 64),
    )
    monkeypatch.setattr(
        "mobius.integrations.gguf._tokenizer_evidence.matching_tokenizer_evidence",
        lambda *_a, **_k: evidence,
    )

    def materialize(_gguf_path, stage, **_kwargs):
        path = Path(stage) / "tokenizer.json"
        path.write_text("{}", encoding="utf-8")
        return str(path)

    monkeypatch.setattr(_tokenizer, "materialize_gguf_tokenizer", materialize)
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="changed while tokenizer assets"):
        _tokenizer.materialize_evidenced_gguf_tokenizer(tmp_path / "model.gguf", output)

    assert not output.exists()
    assert not list(tmp_path.glob(".output.*.tmp"))


def test_evidenced_materializer_atomically_publishes_complete_directory(
    tmp_path: Path, monkeypatch
) -> None:
    model = SimpleNamespace(
        metadata={},
        source_matches_path=mock.Mock(return_value=True),
    )
    evidence = SimpleNamespace(
        source=object(),
        repository="owner/model",
        revision="a" * 40,
        filename="model.gguf",
        lfs_sha256="b" * 64,
    )
    monkeypatch.setattr("mobius.integrations.gguf._reader.GGUFModel", lambda _path: model)
    monkeypatch.setattr(
        "mobius.integrations.gguf._tokenizer.inspect_gguf_tokenizer",
        lambda *_a, **_k: SimpleNamespace(metadata_sha256="c" * 64),
    )
    monkeypatch.setattr(
        "mobius.integrations.gguf._tokenizer_evidence.matching_tokenizer_evidence",
        lambda *_a, **_k: evidence,
    )

    def materialize(_gguf_path, stage, **_kwargs):
        path = Path(stage) / "tokenizer.json"
        path.write_text("{}", encoding="utf-8")
        (Path(stage) / "gguf_tokenizer_manifest.json").write_text("{}", encoding="utf-8")
        return str(path)

    monkeypatch.setattr(_tokenizer, "materialize_gguf_tokenizer", materialize)
    output = tmp_path / "output"

    result = _tokenizer.materialize_evidenced_gguf_tokenizer(tmp_path / "model.gguf", output)

    assert result == str(output / "tokenizer.json")
    assert sorted(path.name for path in output.iterdir()) == [
        "gguf_tokenizer_manifest.json",
        "tokenizer.json",
    ]
    assert model.source_matches_path.call_count == 2
    assert not list(tmp_path.glob(".output.*.tmp"))


@pytest.mark.parametrize(
    ("token", "token_type", "message"),
    [
        ("[PAD8]", 5, "vocabulary differs"),
        ("[PAD7]", 1, "padding must contain only unused tokens"),
    ],
)
def test_malformed_deterministic_padding_rejects(
    token: str, token_type: int, message: str
) -> None:
    source_metadata = _metadata(pre="qwen2")
    source_metadata.pop("tokenizer.ggml.scores")
    payloads = _pinned_payloads(source_metadata)
    metadata = dict(source_metadata)
    metadata["tokenizer.ggml.tokens"] = [*source_metadata["tokenizer.ggml.tokens"], token]
    metadata["tokenizer.ggml.token_type"] = [
        *source_metadata["tokenizer.ggml.token_type"],
        token_type,
    ]

    with pytest.raises(ValueError, match=message):
        _tokenizer._validate_pinned_tokenizer(metadata, payloads)


def test_gemma4_forced_control_token_exception_is_not_global() -> None:
    metadata = _metadata(pre="qwen2")
    metadata["tokenizer.ggml.tokens"].append("<|tool_response>")
    metadata["tokenizer.ggml.token_type"].append(4)
    metadata["tokenizer.ggml.scores"].append(0.0)
    payloads = _pinned_payloads(metadata)
    tokenizer_json = json.loads(payloads["tokenizer.json"])
    tokenizer_json["added_tokens"].append(
        {
            "id": 7,
            "content": "<|tool_response>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
    )
    payloads["tokenizer.json"] = json.dumps(tokenizer_json).encode()

    with pytest.raises(ValueError, match="user-defined/unused-token inventory"):
        _tokenizer._validate_pinned_tokenizer(metadata, payloads)


def test_post_processor_cannot_hide_matching_special_token_flags(
    tmp_path: Path, monkeypatch
) -> None:
    metadata = _metadata(pre="smollm")
    payloads = _pinned_payloads(metadata)
    tokenizer = json.loads(payloads["tokenizer.json"])
    tokenizer["post_processor"] = {
        "type": "TemplateProcessing",
        "single": [{"SpecialToken": {"id": "<bos>", "type_id": 0}}],
        "pair": [{"Sequence": {"id": "A", "type_id": 0}}],
        "special_tokens": {"<bos>": {"id": "<bos>", "ids": [2], "tokens": ["<bos>"]}},
    }
    payloads["tokenizer.json"] = json.dumps(tokenizer).encode()
    source = _pinned_source(metadata, payloads)
    monkeypatch.setattr(_tokenizer, "_download_tokenizer_assets", lambda *_a, **_k: payloads)

    with pytest.raises(ValueError, match="pipeline differs"):
        materialize_gguf_tokenizer(
            tmp_path / "model.gguf",
            tmp_path / "output",
            source=source,
            metadata=metadata,
        )


def test_cross_host_asset_request_strips_auth_and_rejects_redirect(monkeypatch) -> None:
    payload = b"{}"
    source = GGUFTokenizerSource(
        "owner/tokenizer",
        "a" * 40,
        (
            GGUFTokenizerAsset(
                "tokenizer.json", len(payload), hashlib.sha256(payload).hexdigest()
            ),
        ),
        "b" * 64,
    )
    response = SimpleNamespace(status_code=302)
    response.raise_for_status = lambda: None
    seen_headers: dict[str, str] = {}

    class _Stream:
        def __enter__(self):
            return response

        def __exit__(self, *_args):
            return None

    class _Session:
        def stream(self, _method, _url, *, headers, follow_redirects):
            assert follow_redirects is False
            seen_headers.update(headers)
            return _Stream()

    monkeypatch.setattr("huggingface_hub.hf_hub_url", lambda *_a, **_k: "https://hub/a")
    monkeypatch.setattr(
        "huggingface_hub.get_hf_file_metadata",
        lambda _url: SimpleNamespace(
            commit_hash="a" * 40,
            location="https://cdn/tokenizer.json",
        ),
    )
    monkeypatch.setattr("huggingface_hub.get_session", lambda: _Session())
    monkeypatch.setattr(
        "huggingface_hub.utils.build_hf_headers",
        lambda: {"Authorization": "Bearer secret", "user-agent": "test"},
    )

    with pytest.raises(ValueError, match="redirected after authorization policy"):
        _tokenizer._download_tokenizer_assets(source, local_files_only=False)

    assert not {name for name in seen_headers if name.lower() == "authorization"}


def test_local_asset_replacement_during_read_rejects(tmp_path: Path) -> None:
    path = tmp_path / "tokenizer.json"
    payload = b"{}"
    path.write_bytes(payload)
    expected = GGUFTokenizerAsset(
        "tokenizer.json", len(payload), hashlib.sha256(payload).hexdigest()
    )
    first = path.stat()
    changed = os.stat_result(
        (
            first.st_mode,
            first.st_ino + 1,
            first.st_dev,
            first.st_nlink,
            first.st_uid,
            first.st_gid,
            first.st_size,
            first.st_atime,
            first.st_mtime,
            first.st_ctime,
        )
    )
    with (
        mock.patch.object(_tokenizer.os, "fstat", side_effect=[first, changed]),
        pytest.raises(ValueError, match="changed while it was being read"),
    ):
        _tokenizer._read_regular_file(path, expected=expected)


def test_local_hub_cache_symlink_is_resolved_inside_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"{}"
    source = GGUFTokenizerSource(
        "owner/tokenizer",
        "a" * 40,
        (
            GGUFTokenizerAsset(
                "tokenizer.json", len(payload), hashlib.sha256(payload).hexdigest()
            ),
        ),
        "b" * 64,
    )
    cache = tmp_path / "hub"
    blob = cache / "models--owner--tokenizer" / "blobs" / ("c" * 64)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(payload)
    snapshot = (
        cache / "models--owner--tokenizer" / "snapshots" / source.revision / "tokenizer.json"
    )
    snapshot.parent.mkdir(parents=True)
    snapshot.symlink_to(blob)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **_kwargs: str(snapshot))

    assert _tokenizer._download_tokenizer_assets(source, local_files_only=True) == {
        "tokenizer.json": payload
    }


def test_local_hub_cache_symlink_escape_rejects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"{}"
    source = GGUFTokenizerSource(
        "owner/tokenizer",
        "a" * 40,
        (
            GGUFTokenizerAsset(
                "tokenizer.json", len(payload), hashlib.sha256(payload).hexdigest()
            ),
        ),
        "b" * 64,
    )
    cache = tmp_path / "hub"
    escaped = tmp_path / "escaped.json"
    escaped.write_bytes(payload)
    snapshot = (
        cache / "models--owner--tokenizer" / "snapshots" / source.revision / "tokenizer.json"
    )
    snapshot.parent.mkdir(parents=True)
    snapshot.symlink_to(escaped)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **_kwargs: str(snapshot))

    with pytest.raises(ValueError, match="outside the trusted Hub cache"):
        _tokenizer._download_tokenizer_assets(source, local_files_only=True)


def test_local_hub_cache_parent_symlink_escape_rejects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"{}"
    source = GGUFTokenizerSource(
        "owner/tokenizer",
        "a" * 40,
        (
            GGUFTokenizerAsset(
                "tokenizer.json", len(payload), hashlib.sha256(payload).hexdigest()
            ),
        ),
        "b" * 64,
    )
    cache = tmp_path / "hub"
    escaped = tmp_path / "escaped"
    escaped.mkdir()
    (escaped / "tokenizer.json").write_bytes(payload)
    snapshots = cache / "models--owner--tokenizer" / "snapshots"
    snapshots.parent.mkdir(parents=True)
    snapshots.symlink_to(escaped, target_is_directory=True)
    path = snapshots / "tokenizer.json"
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **_kwargs: str(path))

    with pytest.raises(ValueError, match="outside the trusted Hub cache"):
        _tokenizer._download_tokenizer_assets(source, local_files_only=True)
