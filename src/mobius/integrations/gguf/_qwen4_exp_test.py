# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from mobius._configs import Qwen4ExpConfig
from mobius.integrations.gguf._config_mapping import gguf_to_config
from mobius.integrations.gguf._qwen4_exp import (
    Qwen4ExpGGUFImportError,
    _expected_qtypes,
    _expected_shapes,
    validate_qwen4exp_tensor_contract,
)
from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names
from mobius.models.qwen4_exp import _build_layer_multipliers, _find_nth_prime_after

_EVIDENCE_REPO = "unsloth/Qwen3.8-Flash-Next-GGUF"
_EVIDENCE_REVISION = "d3bc75ee6ccef3efc1e228ec00a6cc2cdb1e2249"
_EVIDENCE_SHARDS = (
    SimpleNamespace(
        filename="UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf",
        tensor_count=0,
    ),
    SimpleNamespace(
        filename="UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00002-of-00003.gguf",
        tensor_count=595,
    ),
    SimpleNamespace(
        filename="UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00003-of-00003.gguf",
        tensor_count=629,
    ),
)


def _metadata() -> dict[str, object]:
    vocab_size = 248320
    ngram_size = 3
    heads_per_ngram = 8
    head_vocab_sizes = [
        _find_nth_prime_after(20_000_000 - 1, index + 1)
        for index in range((ngram_size - 1) * heads_per_ngram)
    ]
    head_offsets = []
    offset = 0
    for size in head_vocab_sizes:
        head_offsets.append(offset)
        offset += size
    return {
        "general.architecture": "qwen4exp",
        "general.type": "model",
        "general.name": "Qwen3.8 Flash Next",
        "general.description": "A Preview of the Qwen4 Architecture",
        "general.size_label": "512x56B",
        "qwen4exp.block_count": 48,
        "qwen4exp.context_length": 262144,
        "qwen4exp.embedding_length": 2560,
        "qwen4exp.vocab_size": vocab_size,
        "qwen4exp.attention.head_count": 24,
        "qwen4exp.attention.head_count_kv": 2,
        "qwen4exp.attention.key_length": 256,
        "qwen4exp.attention.value_length": 256,
        "qwen4exp.attention.layer_norm_rms_epsilon": float(np.float32(1e-6)),
        "qwen4exp.rope.dimension_count": 64,
        "qwen4exp.rope.dimension_sections": [11, 11, 10],
        "qwen4exp.rope.freq_base": 10_000_000.0,
        "qwen4exp.full_attention_interval": 4,
        "qwen4exp.ssm.conv_kernel": 4,
        "qwen4exp.ssm.state_size": 128,
        "qwen4exp.ssm.group_count": 16,
        "qwen4exp.ssm.time_step_rank": 48,
        "qwen4exp.ssm.inner_size": 6144,
        "qwen4exp.expert_count": 512,
        "qwen4exp.expert_used_count": 10,
        "qwen4exp.expert_feed_forward_length": 640,
        "qwen4exp.expert_shared_feed_forward_length": 640,
        "qwen4exp.hyper_connection.count": 4,
        "qwen4exp.hyper_connection.low_rank": 320,
        "qwen4exp.attention.indexer.head_count": 4,
        "qwen4exp.attention.indexer.key_length": 128,
        "qwen4exp.attention.indexer.top_k": 2048,
        "qwen4exp.attention.compress_ratios": [0, 0, 0, 4] * 12,
        "qwen4exp.ple.layers": [1],
        "qwen4exp.ple.ngram_size": ngram_size,
        "qwen4exp.ple.heads_per_ngram": heads_per_ngram,
        "qwen4exp.ple.conv_kernel": 4,
        "qwen4exp.ple.eos_token_id": 248044,
        "qwen4exp.embedding_length_per_layer_input": 160,
        "qwen4exp.ple.layer_multipliers": _build_layer_multipliers(
            vocab_size,
            ngram_size,
            0,
            1234,
        ).tolist(),
        "qwen4exp.ple.head_offsets": head_offsets,
        "qwen4exp.ple.head_vocab_sizes": head_vocab_sizes,
        "tokenizer.ggml.pre": "qwen35",
        "tokenizer.ggml.eos_token_id": 248044,
    }


class _HeaderFixture:
    architecture = "qwen4exp"

    def __init__(self, qtype_overrides: dict[str, str] | None = None) -> None:
        self.metadata = _metadata()
        self.shapes = _expected_shapes(self.metadata)
        self.qtypes = _expected_qtypes(self.metadata)
        self.tensor_names = list(self.shapes)
        self.manifest = SimpleNamespace(
            split_count=3,
            shards=[
                SimpleNamespace(tensor_count=shard.tensor_count) for shard in _EVIDENCE_SHARDS
            ],
        )
        self.qtype_overrides = qtype_overrides or {}

    def get_metadata(self, key, default=None):
        return self.metadata.get(key, default)

    def tensor_items_raw(self):
        for name, shape in self.shapes.items():
            qtype = self.qtype_overrides.get(name, self.qtypes[name])
            yield name, None, SimpleNamespace(name=qtype), shape


def _validate_fixture(
    fixture: _HeaderFixture,
    *,
    keep_quantized: bool | None = None,
) -> None:
    validate_qwen4exp_tensor_contract(
        fixture,
        source="header-fixture",
        keep_quantized=keep_quantized,
    )


def test_qwen4exp_config_mapping_is_exact():
    config = gguf_to_config(_HeaderFixture())

    assert isinstance(config, Qwen4ExpConfig)
    assert config.model_type == "qwen4_exp_text"
    assert config.layer_types == [
        layer_type
        for _ in range(12)
        for layer_type in (
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "qwen_sparse_attention",
        )
    ]
    assert config.hc_count == 4
    assert config.hc_lowrank == 320
    assert config.ple_layer_ids == [2]
    assert config.ple_embed_dim == 2560
    assert config.indexer_n_heads == 4
    assert config.indexer_kv_heads == 1
    assert config.indexer_head_dim == 128
    assert config.indexer_budget == 2048
    assert config.indexer_compress_ratio == 4
    assert config.output_gate_type == "sigmoid"
    assert config.linear_num_key_heads == 16
    assert config.linear_num_value_heads == 48
    assert config.linear_key_head_dim == 128
    assert config.linear_value_head_dim == 128
    assert config.mrope_section == [11, 11, 10]
    assert config.mrope_interleaved


def test_qwen4exp_rejects_unpinned_dense_ffn_metadata():
    fixture = _HeaderFixture()
    fixture.metadata["qwen4exp.feed_forward_length"] = 9999

    with pytest.raises(ValueError, match=r"feed_forward_length.*MoE-only"):
        validate_qwen4exp_tensor_contract(fixture, source="changed-header")


@pytest.mark.parametrize(
    ("gguf_name", "hf_name"),
    [
        (
            "output_hc_down.weight",
            "model.hyper_connection_mixer.input_mix_weight_down.weight",
        ),
        (
            "blk.0.hc_attn_inject.weight",
            "model.layers.0.attn_hyper_connection.block_inject_weight.weight",
        ),
        (
            "blk.1.ple_key.weight",
            "model.layers.1.ple.key_proj.weight",
        ),
        (
            "blk.3.indexer.q_proj.weight",
            "model.layers.3.self_attn.indexer.index_q_proj.weight",
        ),
        (
            "blk.3.indexer.k_proj.weight",
            "model.layers.3.self_attn.indexer.index_k_proj.weight",
        ),
        (
            "blk.47.ffn_down_exps.weight",
            "model.layers.47.mlp.experts.down_proj.weight",
        ),
    ],
)
def test_qwen4exp_exact_tensor_mapping(gguf_name: str, hf_name: str):
    assert map_gguf_to_hf_names(gguf_name, "qwen4exp") == hf_name


def test_qwen4exp_header_fixture_matches_pinned_evidence():
    fixture = _HeaderFixture()

    _validate_fixture(fixture)

    assert (
        float(fixture.metadata["qwen4exp.attention.layer_norm_rms_epsilon"]).hex()
        == float(np.float32(1e-6)).hex()
    )
    assert len(fixture.tensor_names) == 1224
    assert [shard.tensor_count for shard in fixture.manifest.shards] == [0, 595, 629]
    assert sum(shard.tensor_count for shard in fixture.manifest.shards) == 1224
    entries = [
        (name, tuple(shape), qtype.name)
        for name, _raw, qtype, shape in fixture.tensor_items_raw()
    ]
    lines = [
        f"{name}|{','.join(str(dim) for dim in shape)}|{qtype}"
        for name, shape, qtype in sorted(entries)
    ]
    assert hashlib.sha256("\n".join(lines).encode()).hexdigest() == (
        "25a1e6a2073caf19d3a3835dd23702a19fa09cc651506e11a13de7b48076359d"
    )


@pytest.mark.parametrize(
    ("keep_quantized", "message"),
    [
        (True, r"IQ4_NL embedding.*rank-3 routed experts.*may fall back to downloading"),
        (False, r"191 GiB.*bounded-memory route.*may fall back to downloading"),
    ],
)
def test_qwen4exp_payload_modes_fail_closed_before_raw_payload_access(
    keep_quantized: bool,
    message: str,
):
    with pytest.raises(Qwen4ExpGGUFImportError, match=message):
        _validate_fixture(_HeaderFixture(), keep_quantized=keep_quantized)


def test_qwen4exp_content_contract_rejects_unsupported_qtype():
    changed = _HeaderFixture({"blk.0.hc_attn_down.weight": "Q5_K"})

    with pytest.raises(ValueError, match="qtype mismatch"):
        validate_qwen4exp_tensor_contract(changed, source="changed-header")


def test_qwen4exp_hub_preflight_is_source_independent_and_forwards_revision(monkeypatch):
    from mobius.integrations.gguf import _builder

    hub_url = mock.Mock(return_value="https://huggingface.co/exact-file")
    monkeypatch.setattr(
        _builder,
        "hf_hub_url",
        hub_url,
    )
    monkeypatch.setattr(
        _builder,
        "get_hf_file_metadata",
        lambda _url: SimpleNamespace(
            commit_hash="a" * 40,
            location="https://cdn.example/model.gguf",
        ),
    )
    response = mock.MagicMock()
    bounded_response = b"complete-metadata-header|initial-tensor-bytes"
    response.iter_bytes.return_value = [bounded_response]
    response_context = mock.MagicMock()
    response_context.__enter__.return_value = response
    session = mock.MagicMock()
    session.stream.return_value = response_context
    monkeypatch.setattr(_builder, "get_session", lambda: session)
    inspected_ranges = []

    def inspect_range(data, **_kwargs):
        inspected_ranges.append(data)
        return _builder.GGUFHeaderInfo(
            architecture="qwen4exp",
            tensor_count=0,
            split_no=0,
            split_count=3,
            split_tensors_count=1224,
        )

    monkeypatch.setattr(
        _builder,
        "_gguf_header_info_from_header_prefix",
        inspect_range,
    )

    with pytest.raises(
        Qwen4ExpGGUFImportError,
        match=r"intentionally fail-closed.*Only bounded GGUF preflight range data",
    ):
        _builder._preflight_hf_gguf_file(
            "other/Qwen4Exp-GGUF",
            "renamed-00001-of-00003.gguf",
            revision="feature/revision",
        )
    assert inspected_ranges == [bounded_response]
    assert b"initial-tensor-bytes" in inspected_ranges[0]
    hub_url.assert_called_once_with(
        "other/Qwen4Exp-GGUF",
        "renamed-00001-of-00003.gguf",
        revision="feature/revision",
    )


def test_qwen4exp_split_fallback_does_not_claim_complete_payload_was_not_downloaded(
    monkeypatch,
):
    from mobius.integrations.gguf import _builder

    commit_hash = "b" * 40
    shards = [
        "model-00001-of-00002.gguf",
        "model-00002-of-00002.gguf",
    ]
    download_shards = mock.Mock(return_value="cached-primary.gguf")
    monkeypatch.setattr(
        _builder,
        "_preflight_hf_gguf_file",
        lambda *_args, **_kwargs: _builder._GGUFPreflightFallbackRevision(commit_hash),
    )
    api = mock.Mock()
    api.list_repo_files.return_value = shards
    monkeypatch.setattr(_builder, "HfApi", mock.Mock(return_value=api))
    monkeypatch.setattr(_builder, "_download_hf_gguf_shards", download_shards)

    assert (
        _builder._resolve_gguf_path(f"other/Qwen4Exp-GGUF:{shards[0]}")
        == "cached-primary.gguf"
    )
    download_shards.assert_called_once_with(
        api,
        repo_id="other/Qwen4Exp-GGUF",
        selected_filename=shards[0],
        shard_filenames=shards,
        revision=commit_hash,
    )

    with pytest.raises(Qwen4ExpGGUFImportError) as exc_info:
        _builder._validate_gguf_model(
            _HeaderFixture(),
            source="cached-primary.gguf",
            keep_quantized=True,
        )
    message = str(exc_info.value)
    assert "may fall back to downloading" in message
    assert "no complete GGUF file or shard payload was downloaded" not in message
