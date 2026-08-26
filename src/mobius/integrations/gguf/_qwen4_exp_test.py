# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from mobius._configs import Qwen4ExpConfig
from mobius.integrations.gguf._config_mapping import gguf_to_config
from mobius.integrations.gguf._qwen4_exp import (
    QWEN4EXP_GGUF_REPO,
    QWEN4EXP_GGUF_REVISION,
    QWEN4EXP_GGUF_SHARDS,
    Qwen4ExpGGUFImportError,
    _expected_shapes,
    validate_qwen4exp_hub_artifact,
    validate_qwen4exp_tensor_contract,
)
from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names
from mobius.models.qwen4_exp import _build_layer_multipliers, _find_nth_prime_after


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
        "qwen4exp.block_count": 48,
        "qwen4exp.context_length": 262144,
        "qwen4exp.embedding_length": 2560,
        "qwen4exp.feed_forward_length": 10240,
        "qwen4exp.vocab_size": vocab_size,
        "qwen4exp.attention.head_count": 24,
        "qwen4exp.attention.head_count_kv": 2,
        "qwen4exp.attention.key_length": 256,
        "qwen4exp.attention.value_length": 256,
        "qwen4exp.attention.layer_norm_rms_epsilon": 1e-6,
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

    def __init__(self) -> None:
        self.metadata = _metadata()
        self.shapes = _expected_shapes(self.metadata)
        self.tensor_names = list(self.shapes)
        self.manifest = SimpleNamespace(
            split_count=3,
            shards=[
                SimpleNamespace(tensor_count=shard.tensor_count)
                for shard in QWEN4EXP_GGUF_SHARDS
            ],
        )

    def get_metadata(self, key, default=None):
        return self.metadata.get(key, default)

    def tensor_items_raw(self):
        for name, shape in self.shapes.items():
            qtype = "BF16"
            if name == "per_layer_token_embd.weight" or name.endswith(
                ".ffn_down_exps.weight"
            ):
                qtype = "IQ4_NL"
            elif name.endswith((".ffn_gate_exps.weight", ".ffn_up_exps.weight")):
                qtype = "IQ1_S"
            yield name, None, SimpleNamespace(name=qtype), shape


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


def test_qwen4exp_header_fixture_proves_exact_three_shard_closure():
    fixture = _HeaderFixture()

    validate_qwen4exp_tensor_contract(fixture, source="header-fixture")

    assert len(fixture.tensor_names) == 1224
    assert [shard.tensor_count for shard in fixture.manifest.shards] == [0, 595, 629]
    assert sum(shard.tensor_count for shard in fixture.manifest.shards) == 1224


@pytest.mark.parametrize(
    ("keep_quantized", "message"),
    [
        (True, r"IQ4_NL embedding.*rank-3 routed experts.*No GGUF payload"),
        (False, r"95 GiB.*bounded-memory route.*No GGUF payload"),
    ],
)
def test_qwen4exp_payload_modes_fail_closed_before_raw_payload_access(
    keep_quantized: bool,
    message: str,
):
    with pytest.raises(Qwen4ExpGGUFImportError, match=message):
        validate_qwen4exp_tensor_contract(
            _HeaderFixture(),
            source="header-fixture",
            keep_quantized=keep_quantized,
        )


def _path_infos():
    return [
        SimpleNamespace(
            path=shard.filename,
            size=shard.size,
            lfs={"sha256": shard.lfs_sha256},
        )
        for shard in QWEN4EXP_GGUF_SHARDS
    ]


def test_qwen4exp_hub_identity_is_pinned_before_payload_download():
    api = mock.MagicMock()
    api.get_paths_info.return_value = _path_infos()

    with pytest.raises(Qwen4ExpGGUFImportError, match="intentionally fail-closed"):
        validate_qwen4exp_hub_artifact(
            api,
            repo_id=QWEN4EXP_GGUF_REPO,
            revision=QWEN4EXP_GGUF_REVISION,
            shard_filenames=[shard.filename for shard in QWEN4EXP_GGUF_SHARDS],
            keep_quantized=True,
        )

    api.get_paths_info.assert_called_once_with(
        QWEN4EXP_GGUF_REPO,
        [shard.filename for shard in QWEN4EXP_GGUF_SHARDS],
        revision=QWEN4EXP_GGUF_REVISION,
        expand=True,
    )


def test_qwen4exp_resolver_never_starts_hub_payload_download(monkeypatch):
    from mobius.integrations.gguf import _builder

    api = mock.MagicMock()
    filenames = [shard.filename for shard in QWEN4EXP_GGUF_SHARDS]
    api.list_repo_files.return_value = filenames
    api.get_paths_info.return_value = _path_infos()
    monkeypatch.setattr(_builder, "HfApi", lambda: api)
    monkeypatch.setattr(
        _builder,
        "_preflight_hf_gguf_file",
        lambda *_args, **_kwargs: QWEN4EXP_GGUF_REVISION,
    )
    download = mock.MagicMock(side_effect=AssertionError("payload download attempted"))
    monkeypatch.setattr(_builder, "hf_hub_download", download)

    source = (
        f"{QWEN4EXP_GGUF_REPO}@{QWEN4EXP_GGUF_REVISION}:"
        f"{QWEN4EXP_GGUF_SHARDS[0].filename}"
    )
    with pytest.raises(Qwen4ExpGGUFImportError, match="No GGUF payload was downloaded"):
        _builder._resolve_gguf_path(source)

    download.assert_not_called()
