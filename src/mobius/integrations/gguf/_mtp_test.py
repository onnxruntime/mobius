# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the Qwen3.5/3.8 MTP (multi-token-prediction) head GGUF export."""

from __future__ import annotations

import typing
from pathlib import Path

import numpy as np
import pytest

from mobius.integrations.gguf._mtp import (
    build_mtp_head_from_gguf,
    derive_mtp_config,
    has_mtp_head,
    map_gguf_mtp_to_hf_names,
)

# Tiny dimensions for a Qwen3.5-style GGUF that ships a single trailing MTP
# ("nextn") block. ``block_count`` includes the MTP block, so decoder=1 and
# the MTP head lives at block index 1.
_H = 64
_FFN = 128
_NH = 4
_NKV = 2
_HD = 16
_VOCAB = 100
_BLOCK_COUNT = 2  # 1 decoder layer + 1 nextn head block


def _write_qwen35_mtp_gguf(path: Path) -> None:
    """Write a tiny ``qwen35`` GGUF with a trailing ``blk.1.nextn.*`` MTP head."""
    from gguf import GGUFWriter

    writer = GGUFWriter(str(path), "qwen35")
    writer.add_context_length(512)
    writer.add_embedding_length(_H)
    writer.add_feed_forward_length(_FFN)
    writer.add_block_count(_BLOCK_COUNT)
    writer.add_head_count(_NH)
    writer.add_head_count_kv(_NKV)
    writer.add_rope_freq_base(10000.0)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_vocab_size(_VOCAB)
    # UINT32 == GGUFValueType 4.
    writer.add_key_value("qwen35.nextn_predict_layers", 1, 4)
    writer.add_key_value("qwen35.attention.key_length", _HD, 4)

    def f32(name: str, shape: tuple[int, ...]) -> None:
        writer.add_tensor(name, np.random.randn(*shape).astype(np.float32))

    def _add_decoder_block(idx: int) -> None:
        # Qwen3.5 uses doubled-Q output gating: q_proj emits 2 * num_heads *
        # head_dim rows (gate + query).
        f32(f"blk.{idx}.attn_q.weight", (2 * _NH * _HD, _H))
        f32(f"blk.{idx}.attn_k.weight", (_NKV * _HD, _H))
        f32(f"blk.{idx}.attn_v.weight", (_NKV * _HD, _H))
        f32(f"blk.{idx}.attn_output.weight", (_H, _NH * _HD))
        f32(f"blk.{idx}.attn_q_norm.weight", (_HD,))
        f32(f"blk.{idx}.attn_k_norm.weight", (_HD,))
        f32(f"blk.{idx}.attn_norm.weight", (_H,))
        f32(f"blk.{idx}.post_attention_norm.weight", (_H,))
        f32(f"blk.{idx}.ffn_gate.weight", (_FFN, _H))
        f32(f"blk.{idx}.ffn_up.weight", (_FFN, _H))
        f32(f"blk.{idx}.ffn_down.weight", (_H, _FFN))

    # Decoder layer 0.
    _add_decoder_block(0)

    # MTP head block (index 1): the nextn cross-conditioning tensors plus a
    # full attention/FFN sublayer.
    f32("blk.1.nextn.eh_proj.weight", (_H, 2 * _H))
    f32("blk.1.nextn.enorm.weight", (_H,))
    f32("blk.1.nextn.hnorm.weight", (_H,))
    f32("blk.1.nextn.shared_head_norm.weight", (_H,))
    _add_decoder_block(1)

    f32("token_embd.weight", (_VOCAB, _H))
    f32("output_norm.weight", (_H,))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.fixture
def qwen35_mtp_gguf(tmp_path: Path) -> Path:
    path = tmp_path / "tiny_qwen35_mtp.gguf"
    _write_qwen35_mtp_gguf(path)
    return path


class TestMtpNameMapping:
    def test_maps_nextn_tensors_to_head_stems(self):
        assert map_gguf_mtp_to_hf_names("blk.1.nextn.eh_proj.weight", 1) == "fc.weight"
        assert (
            map_gguf_mtp_to_hf_names("blk.1.nextn.enorm.weight", 1)
            == "pre_fc_norm_embedding.weight"
        )
        assert (
            map_gguf_mtp_to_hf_names("blk.1.nextn.hnorm.weight", 1)
            == "pre_fc_norm_hidden.weight"
        )
        assert (
            map_gguf_mtp_to_hf_names("blk.1.nextn.shared_head_norm.weight", 1)
            == "norm.weight"
        )

    def test_maps_attention_and_ffn_onto_layer_zero(self):
        assert (
            map_gguf_mtp_to_hf_names("blk.1.attn_q.weight", 1)
            == "layers.0.self_attn.q_proj.weight"
        )
        assert (
            map_gguf_mtp_to_hf_names("blk.1.ffn_down.weight", 1)
            == "layers.0.mlp.down_proj.weight"
        )
        assert (
            map_gguf_mtp_to_hf_names("blk.1.attn_norm.weight", 1)
            == "layers.0.input_layernorm.weight"
        )

    def test_returns_none_for_non_mtp_blocks(self):
        # Backbone decoder layer at a different index must not be captured.
        assert map_gguf_mtp_to_hf_names("blk.0.attn_q.weight", 1) is None
        # Non-block tensors (embeddings, tokenizer, output norm) are skipped.
        assert map_gguf_mtp_to_hf_names("token_embd.weight", 1) is None
        assert map_gguf_mtp_to_hf_names("output_norm.weight", 1) is None
        # Unknown stem inside the MTP block is skipped rather than mis-mapped.
        assert map_gguf_mtp_to_hf_names("blk.1.unknown_tensor.weight", 1) is None


class TestMtpConfigSurfacing:
    def test_config_exposes_mtp_block_indices(self, qwen35_mtp_gguf: Path):
        from mobius.integrations.gguf._config_mapping import gguf_to_config
        from mobius.integrations.gguf._reader import GGUFModel

        gguf_model = GGUFModel(str(qwen35_mtp_gguf))
        config = gguf_to_config(gguf_model)

        # The trailing nextn block is subtracted from the decoder count but its
        # index is surfaced for the head builder.
        assert config.num_hidden_layers == 1
        assert config._gguf_nextn_predict_layers == 1
        assert config._gguf_mtp_block_indices == [1]
        assert has_mtp_head(config)

    def test_derive_mtp_config_forces_single_full_attention_layer(
        self, qwen35_mtp_gguf: Path
    ):
        from mobius.integrations.gguf._config_mapping import gguf_to_config
        from mobius.integrations.gguf._reader import GGUFModel

        gguf_model = GGUFModel(str(qwen35_mtp_gguf))
        config = gguf_to_config(gguf_model)
        mtp_config = derive_mtp_config(config)

        assert mtp_config.num_hidden_layers == 1
        assert mtp_config.layer_types == ["full_attention"]
        # Dimensions are inherited from the backbone (never hardcoded).
        assert mtp_config.hidden_size == config.hidden_size
        assert mtp_config.num_attention_heads == config.num_attention_heads


class TestBuildMtpHead:
    def test_emits_head_sidecar_with_weights(self, qwen35_mtp_gguf: Path):
        from mobius.integrations.gguf._config_mapping import gguf_to_config
        from mobius.integrations.gguf._reader import GGUFModel

        gguf_model = GGUFModel(str(qwen35_mtp_gguf))
        config = gguf_to_config(gguf_model)

        head = build_mtp_head_from_gguf(
            gguf_model, config, preserve_quantization=False
        )
        assert head is not None

        model = head["model"]
        init_names = set(model.graph.initializers.keys())
        output_names = {v.name for v in model.graph.outputs}

        # The previously-dropped nextn tensors are now emitted. Linear weights
        # are stored transposed as ``*.weight_t`` by the mobius builder.
        assert "fc.weight_t" in init_names
        assert "pre_fc_norm_embedding.weight" in init_names
        assert "pre_fc_norm_hidden.weight" in init_names
        assert "norm.weight" in init_names
        assert "layers.0.self_attn.q_proj.weight_t" in init_names

        # The head threads its final hidden state forward for the LM head.
        assert "mtp_hidden" in output_names

        # Every initializer received backing weight data.
        missing = [
            name
            for name, value in model.graph.initializers.items()
            if value.const_value is None
        ]
        assert not missing, f"initializers missing weights: {missing}"

    def test_no_head_when_gguf_lacks_nextn(self):
        # A config with no MTP metadata must not build a head.
        class _NoMtpConfig:
            _gguf_mtp_block_indices: typing.ClassVar[list[int]] = []

        assert has_mtp_head(_NoMtpConfig()) is False
        assert (
            build_mtp_head_from_gguf(
                gguf_model=None,
                config=_NoMtpConfig(),
                preserve_quantization=False,
            )
            is None
        )
