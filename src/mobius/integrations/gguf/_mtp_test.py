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


def _write_qwen35_mtp_gguf(
    path: Path,
    *,
    architecture: str = "qwen35",
    quantized: bool = False,
    mtp_count: int = 1,
    mtp_aux_suffix: str | None = None,
) -> None:
    """Write a tiny Qwen3.5 GGUF with a trailing ``blk.1.nextn.*`` MTP head."""
    from gguf import GGMLQuantizationType, GGUFWriter

    writer = GGUFWriter(str(path), architecture)
    writer.add_context_length(512)
    writer.add_embedding_length(_H)
    writer.add_feed_forward_length(_FFN)
    writer.add_block_count(1 + mtp_count)
    writer.add_head_count(_NH)
    writer.add_head_count_kv(_NKV)
    writer.add_rope_freq_base(10000.0)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_vocab_size(_VOCAB)
    # UINT32 == GGUFValueType 4.
    writer.add_key_value(f"{architecture}.nextn_predict_layers", mtp_count, 4)
    writer.add_key_value(f"{architecture}.attention.key_length", _HD, 4)
    writer.add_array(f"{architecture}.attention.recurrent_layers", [False, False])
    writer.add_array(f"{architecture}.rope.dimension_sections", [4, 4, 0, 0])
    writer.add_ssm_conv_kernel(3)
    writer.add_ssm_inner_size(32)
    writer.add_ssm_state_size(16)
    writer.add_ssm_time_step_rank(2)
    writer.add_ssm_group_count(2)

    def f32(name: str, shape: tuple[int, ...]) -> None:
        writer.add_tensor(name, np.random.randn(*shape).astype(np.float32))

    def matrix(name: str, shape: tuple[int, ...]) -> None:
        if not quantized:
            f32(name, shape)
            return
        rows = int(np.prod(shape[:-1]))
        k_in = shape[-1]
        assert k_in % 32 == 0
        raw = np.zeros((rows, (k_in // 32) * 18), dtype=np.uint8)
        raw[:, 0::18] = 0x00
        raw[:, 1::18] = 0x3C
        writer.add_tensor(
            name,
            raw.reshape(*shape[:-1], -1),
            raw_dtype=GGMLQuantizationType.Q4_0,
        )

    def _add_decoder_block(idx: int) -> None:
        # Qwen3.5 uses doubled-Q output gating: q_proj emits 2 * num_heads *
        # head_dim rows (gate + query).
        matrix(f"blk.{idx}.attn_q.weight", (2 * _NH * _HD, _H))
        matrix(f"blk.{idx}.attn_k.weight", (_NKV * _HD, _H))
        matrix(f"blk.{idx}.attn_v.weight", (_NKV * _HD, _H))
        matrix(f"blk.{idx}.attn_output.weight", (_H, _NH * _HD))
        f32(f"blk.{idx}.attn_q_norm.weight", (_HD,))
        f32(f"blk.{idx}.attn_k_norm.weight", (_HD,))
        f32(f"blk.{idx}.attn_norm.weight", (_H,))
        f32(f"blk.{idx}.post_attention_norm.weight", (_H,))
        matrix(f"blk.{idx}.ffn_gate.weight", (_FFN, _H))
        matrix(f"blk.{idx}.ffn_up.weight", (_FFN, _H))
        matrix(f"blk.{idx}.ffn_down.weight", (_H, _FFN))

    # Decoder layer 0.
    _add_decoder_block(0)

    # MTP head blocks: each has cross-conditioning tensors plus a full
    # attention/FFN sublayer.
    for index in range(1, 1 + mtp_count):
        matrix(f"blk.{index}.nextn.eh_proj.weight", (_H, 2 * _H))
        if mtp_aux_suffix is not None:
            assert mtp_aux_suffix in {"scale", "input_scale"}
            f32(f"blk.{index}.nextn.eh_proj.{mtp_aux_suffix}", (1,))
        f32(f"blk.{index}.nextn.enorm.weight", (_H,))
        f32(f"blk.{index}.nextn.hnorm.weight", (_H,))
        f32(f"blk.{index}.nextn.shared_head_norm.weight", (_H,))
        _add_decoder_block(index)

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
            map_gguf_mtp_to_hf_names("blk.1.nextn.shared_head_norm.weight", 1) == "norm.weight"
        )
        assert map_gguf_mtp_to_hf_names("blk.1.nextn.eh_proj.bias", 1) == "fc.bias"
        assert map_gguf_mtp_to_hf_names("blk.1.nextn.eh_proj.scale", 1) == "fc.scale"
        assert (
            map_gguf_mtp_to_hf_names("blk.1.nextn.eh_proj.input_scale", 1) == "fc.input_scale"
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

    def test_derive_mtp_config_forces_single_full_attention_layer(self, qwen35_mtp_gguf: Path):
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

        head = build_mtp_head_from_gguf(gguf_model, config, preserve_quantization=False)
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

    def test_quantized_head_preserves_all_projection_weights(self, tmp_path: Path):
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "tiny_qwen35_quantized_mtp.gguf"
        _write_qwen35_mtp_gguf(path, quantized=True)

        pkg = build_from_gguf(str(path), keep_quantized=True)
        head = pkg.mtp_head["model"]
        op_types = [node.op_type for node in head.graph]

        assert op_types.count("MatMulNBits") == 8
        assert "fc.weight" in head.graph.initializers
        assert "fc.scales" in head.graph.initializers

    def test_multiple_mtp_heads_reject_before_any_package_is_built(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "tiny_qwen35_multi_mtp.gguf"
        _write_qwen35_mtp_gguf(path, mtp_count=2)

        built = False

        def _unexpected_build(*args, **kwargs):
            nonlocal built
            built = True
            raise AssertionError("graph construction must not start for multi-MTP input")

        monkeypatch.setattr(core_builder, "build_from_module", _unexpected_build)
        with pytest.raises(
            ValueError,
            match=r"nextn_predict_layers=2.*exactly one MTP sidecar head",
        ):
            build_from_gguf(str(path))
        assert built is False

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


class TestMtpAutoDetect:
    """MTP is emitted iff the source GGUF ships the nextn head (no user flag)."""

    @pytest.mark.parametrize("architecture", ["qwen35", "qwen35moe"])
    @pytest.mark.parametrize("quantized", [False, True], ids=["float", "quantized"])
    @pytest.mark.parametrize("suffix", ["scale", "input_scale"])
    def test_mtp_auxiliary_scales_are_rejected_before_any_graph_build(
        self,
        architecture: str,
        quantized: bool,
        suffix: str,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf
        from mobius.integrations.gguf._reader import GGUFModel

        path = tmp_path / f"{architecture}-{suffix}-{'q4' if quantized else 'f32'}.gguf"
        _write_qwen35_mtp_gguf(
            path,
            architecture=architecture,
            quantized=quantized,
            mtp_aux_suffix=suffix,
        )
        if quantized:
            assert any(
                qtype.name == "Q4_0" for _, _, qtype, _ in GGUFModel(path).tensor_items_raw()
            )

        graph_build_started = False

        def unexpected_graph_build(*args, **kwargs):
            nonlocal graph_build_started
            graph_build_started = True
            raise AssertionError("neither backbone nor MTP graph construction may start")

        monkeypatch.setattr(core_builder, "build_from_module", unexpected_graph_build)
        with pytest.raises(
            ValueError,
            match=rf"nextn\.eh_proj\.{suffix}.*cannot represent GGUF scale/input_scale",
        ):
            build_from_gguf(path)
        assert not graph_build_started

    def test_mtp_gguf_auto_emits_head_sidecar(self, qwen35_mtp_gguf: Path):
        from mobius.integrations.gguf import build_from_gguf

        # Source ships the nextn head -> sidecar is auto-attached, no opt-in.
        pkg = build_from_gguf(str(qwen35_mtp_gguf))
        head = getattr(pkg, "mtp_head", None)
        assert head is not None
        output_names = {v.name for v in head["model"].graph.outputs}
        assert "mtp_hidden" in output_names
        # The backbone must expose the final-layer hidden-state seed
        # (hidden_states.{num_hidden_layers-1}) that the head consumes.
        seed = pkg.config.num_hidden_layers - 1
        main_outputs = {v.name for v in pkg["model"].graph.outputs}
        assert f"hidden_states.{seed}" in main_outputs

    def test_moe_mtp_is_rejected_before_graph_build(self, tmp_path: Path, monkeypatch):
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "qwen35moe-mtp.gguf"
        _write_qwen35_mtp_gguf(path, architecture="qwen35moe")
        monkeypatch.setattr(
            core_builder,
            "build_from_module",
            lambda *_args, **_kwargs: pytest.fail("graph construction must not run"),
        )

        with pytest.raises(NotImplementedError, match="MTP blocks use routed experts"):
            build_from_gguf(path)

    def test_mtp_survives_dtype_replace(self, qwen35_mtp_gguf: Path):
        # Regression: an explicit dtype triggers dataclasses.replace on the
        # config, which drops the private _gguf_* attrs. The builder must
        # re-attach the MTP metadata so auto-detection still fires.
        from mobius.integrations.gguf import build_from_gguf

        pkg = build_from_gguf(str(qwen35_mtp_gguf), dtype="bf16")
        assert getattr(pkg, "mtp_head", None) is not None

    def test_text_only_gguf_omits_head_sidecar(self, tmp_path: Path):
        from mobius.integrations.gguf import _builder_test, build_from_gguf

        # A plain GGUF with no nextn block -> nothing to emit, byte-identical.
        path = tmp_path / "no_mtp.gguf"
        _builder_test._write_quantized_gguf(path, projection_quantization="f32")
        pkg = build_from_gguf(str(path))
        assert getattr(pkg, "mtp_head", None) is None
