# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for GGUF builder architecture cohorts."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest import mock

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

from mobius.integrations.gguf._builder_test_utils import (
    _write_encoder_gguf,
    _write_falcon_h1_gguf,
    _write_granitehybrid_moe_gguf,
    _write_jamba_gguf,
    _write_kimi_linear_gguf,
    _write_lfm2_gguf,
    _write_lfm2moe_gguf,
    _write_minimax_gguf,
    _write_nemotron_h_moe_gguf,
    _write_plamo2_gguf,
    _write_qwen35_gguf,
    _write_recurrent_gguf,
    _write_t5_gguf,
)


class TestLanguageDiffusionDispatch:
    @staticmethod
    def _patch_dream_reader(monkeypatch) -> None:
        class _DreamGGUF:
            architecture = "dream"
            metadata: ClassVar[dict] = {
                "general.architecture": "dream",
                "dream.embedding_length": 8,
                "dream.feed_forward_length": 16,
                "dream.block_count": 1,
                "dream.attention.head_count": 2,
                "dream.attention.head_count_kv": 1,
                "dream.context_length": 32,
                "dream.rope.dimension_count": 4,
                "dream.vocab_size": 16,
                "dream.attention.layer_norm_rms_epsilon": 1e-5,
                "tokenizer.ggml.mask_token_id": 15,
            }
            tensor_names: ClassVar[list[str]] = [
                "token_embd.weight",
                "output_norm.weight",
                "output.weight",
                "blk.0.attn_norm.weight",
                "blk.0.attn_q.weight",
                "blk.0.attn_k.weight",
                "blk.0.attn_v.weight",
                "blk.0.attn_output.weight",
                "blk.0.ffn_norm.weight",
                "blk.0.ffn_gate.weight",
                "blk.0.ffn_down.weight",
                "blk.0.ffn_up.weight",
            ]

            def __init__(self, _path):
                pass

            def get_metadata(self, key, default=None):
                return self.metadata.get(key, default)

        import mobius._builder
        import mobius.integrations.gguf._builder as builder
        import mobius.integrations.gguf._shard_set as shard_set

        monkeypatch.setattr(builder, "_resolve_gguf_path", lambda path, *_args: path)
        monkeypatch.setattr(builder, "_validate_gguf_model", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(builder, "_has_quantized_weights", lambda *_args: False)
        monkeypatch.setattr(shard_set, "open_gguf_model", _DreamGGUF)
        monkeypatch.setattr(
            mobius._builder,
            "build_from_module",
            lambda *_args, **_kwargs: pytest.fail("graph construction must not run"),
        )

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"task": "text-generation"}, "only supports task='masked-diffusion'"),
            ({"static_cache": True}, "has no KV cache"),
        ],
    )
    def test_invalid_generation_contract_is_rejected_before_graph_build(
        self, monkeypatch, kwargs, message
    ):
        from mobius.integrations.gguf import build_from_gguf

        self._patch_dream_reader(monkeypatch)
        with pytest.raises(ValueError, match=message):
            build_from_gguf("unused.gguf", **kwargs)


class TestEncoderGGUFBuild:
    """Encoder GGUFs dispatch to feature extraction and preserve token outputs."""

    @staticmethod
    def _run(model, sequence_length: int, masked: bool = False) -> np.ndarray:
        from mobius._testing.ort_inference import OnnxModelSession

        mask = np.ones((1, sequence_length), dtype=np.int64)
        if masked:
            mask[0, -1] = 0
        feeds = {
            "input_ids": np.arange(sequence_length, dtype=np.int64)[None, :],
            "attention_mask": mask,
            "token_type_ids": np.zeros((1, sequence_length), dtype=np.int64),
        }
        session = OnnxModelSession(model)
        try:
            outputs = session.run(feeds)
        finally:
            session.close()
        assert set(outputs) == {"last_hidden_state"}
        return outputs["last_hidden_state"]

    @pytest.mark.parametrize("architecture", ["bert", "modern-bert"])
    def test_float_build_save_load_and_variable_masks(
        self, tmp_path: Path, architecture: str
    ) -> None:
        from mobius._model_package import ModelPackage
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-f32.gguf"
        _write_encoder_gguf(path, architecture)
        package = build_from_gguf(path)
        model = package["model"]
        assert {value.name for value in model.graph.outputs} == {"last_hidden_state"}
        assert not any(
            marker in value.name
            for value in (*model.graph.inputs, *model.graph.outputs)
            for marker in ("past_key_values", "present", "logits")
        )

        saved = tmp_path / f"{architecture}-saved"
        package.save(saved, progress_bar=False)
        reloaded = ModelPackage.load(saved)
        for sequence_length in (1, 3, 7):
            output = self._run(reloaded["model"], sequence_length, masked=sequence_length > 1)
            assert output.shape == (1, sequence_length, 64)
            assert np.isfinite(output).all()

    @pytest.mark.parametrize("architecture", ["bert", "modern-bert"])
    def test_quantized_source_preserves_linears_but_dequantizes_embedding(
        self, tmp_path: Path, architecture: str
    ) -> None:
        from gguf import GGMLQuantizationType

        from mobius.integrations.gguf import build_from_gguf
        from mobius.integrations.gguf._reader import GGUFModel

        path = tmp_path / f"{architecture}-q4.gguf"
        _write_encoder_gguf(path, architecture, quantized=True)
        source = GGUFModel(path)
        token_qtype = next(
            qtype
            for name, _raw, qtype, _shape in source.tensor_items_raw()
            if name == "token_embd.weight"
        )
        assert token_qtype == GGMLQuantizationType.Q4_0

        preserved = build_from_gguf(path, keep_quantized=True)["model"]
        explicit_float = build_from_gguf(path, keep_quantized=False)["model"]

        op_types = {node.op_type for node in preserved.graph}
        assert "MatMulNBits" in op_types
        assert "GatherBlockQuantized" not in op_types
        actual = self._run(preserved, 5, masked=True)
        expected = self._run(explicit_float, 5, masked=True)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-2)

    def test_report_includes_mapped_biases_and_reconciles_source_bytes(
        self, tmp_path: Path
    ) -> None:
        from mobius.integrations.gguf import QuantizationDisposition, build_from_gguf

        path = tmp_path / "bert-records.gguf"
        _write_encoder_gguf(path, "bert", quantized=True)
        report = build_from_gguf(path).gguf_quantization_report

        bias_records = [
            record for record in report.tensor_records if record.name.endswith(".bias")
        ]
        assert bias_records
        assert {record.disposition for record in bias_records} == {
            QuantizationDisposition.SOURCE_FLOAT
        }
        assert sum(record.source_bytes for record in report.tensor_records) == sum(
            stat.source_bytes for stat in report.source_qtype_census
        )

    def test_float_fused_bert_qkv_splits_losslessly_and_runs(self, tmp_path: Path) -> None:
        import torch

        from mobius.integrations.gguf import build_from_gguf
        from mobius.integrations.gguf._builder import (
            _load_dequantized_state_dict,
            _normalize_gguf_weights,
        )
        from mobius.integrations.gguf._reader import GGUFModel

        path = tmp_path / "bert-fused-qkv.gguf"
        _write_encoder_gguf(path, "bert", fused_qkv=True)
        source = GGUFModel(path)
        mapped = _load_dequantized_state_dict(source, "bert")
        fused_weight = mapped["bert.encoder.layer.0.attention.self.qkv.weight"]
        fused_bias = mapped["bert.encoder.layer.0.attention.self.qkv.bias"]
        normalized = _normalize_gguf_weights(
            mapped,
            "bert",
            SimpleNamespace(hidden_size=64),
        )

        for index, projection in enumerate(("query", "key", "value")):
            stem = f"bert.encoder.layer.0.attention.self.{projection}"
            assert torch.equal(
                normalized[f"{stem}.weight"],
                fused_weight[index * 64 : (index + 1) * 64],
            )
            assert torch.equal(
                normalized[f"{stem}.bias"],
                fused_bias[index * 64 : (index + 1) * 64],
            )
        assert not any(".qkv." in name for name in normalized)

        model = build_from_gguf(path)["model"]
        output = self._run(model, 5, masked=True)
        assert output.shape == (1, 5, 64)
        assert np.isfinite(output).all()

    def test_float_fused_bert_qkv_fails_closed_in_mixed_quantized_file(
        self, tmp_path: Path
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "bert-fused-qkv-mixed.gguf"
        _write_encoder_gguf(
            path,
            "bert",
            quantized=True,
            fused_qkv=True,
            fused_qkv_float=True,
        )
        with pytest.raises(ValueError, match=r"would quantize a source-float tensor"):
            build_from_gguf(path, keep_quantized=True)

        explicit_float = build_from_gguf(path, keep_quantized=False)["model"]
        assert np.isfinite(self._run(explicit_float, 5, masked=True)).all()

    @pytest.mark.parametrize(
        ("quantized", "width_delta", "message"),
        [
            (False, 1, "invalid encoder tensor shape"),
            (True, 0, "cannot be split losslessly"),
        ],
    )
    def test_invalid_fused_bert_qkv_fails_before_graph_build(
        self,
        tmp_path: Path,
        quantized: bool,
        width_delta: int,
        message: str,
        monkeypatch,
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "bert-invalid-fused-qkv.gguf"
        _write_encoder_gguf(
            path,
            "bert",
            quantized=quantized,
            fused_qkv=True,
            fused_width_delta=width_delta,
        )
        monkeypatch.setattr(
            core_builder,
            "build_from_module",
            lambda *args, **kwargs: pytest.fail("graph construction must not start"),
        )
        with pytest.raises(ValueError, match=message):
            build_from_gguf(path)

    def test_fused_bert_qkv_requires_matching_bias(self, tmp_path: Path, monkeypatch) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "bert-fused-qkv-missing-bias.gguf"
        _write_encoder_gguf(
            path,
            "bert",
            fused_qkv=True,
            omit="blk.0.attn_qkv.bias",
        )
        monkeypatch.setattr(
            core_builder,
            "build_from_module",
            lambda *args, **kwargs: pytest.fail("graph construction must not start"),
        )
        with pytest.raises(ValueError, match=r"attn_qkv.bias"):
            build_from_gguf(path)

    @pytest.mark.parametrize("architecture", ["bert", "modern-bert"])
    @pytest.mark.parametrize("kv_heads", [None, 4])
    def test_encoder_kv_heads_default_to_query_heads_or_accept_equality(
        self, tmp_path: Path, architecture: str, kv_heads: int | None
    ) -> None:
        from mobius.integrations.gguf._config_mapping import gguf_to_config
        from mobius.integrations.gguf._reader import GGUFModel

        path = tmp_path / f"{architecture}-kv-{kv_heads}.gguf"
        _write_encoder_gguf(path, architecture, kv_heads=kv_heads)
        config = gguf_to_config(GGUFModel(path))
        assert config.num_attention_heads == 4
        assert config.num_key_value_heads == 4

    @pytest.mark.parametrize("architecture", ["bert", "modern-bert"])
    def test_encoder_gqa_rejected_before_graph_build(
        self, tmp_path: Path, architecture: str, monkeypatch
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-gqa.gguf"
        _write_encoder_gguf(path, architecture, kv_heads=2)
        monkeypatch.setattr(
            core_builder,
            "build_from_module",
            lambda *args, **kwargs: pytest.fail("graph construction must not start"),
        )
        with pytest.raises(ValueError, match="grouped-query attention is not supported"):
            build_from_gguf(path)

    @pytest.mark.parametrize("architecture", ["bert", "modern-bert"])
    def test_encoder_missing_geometry_is_actionable(self, architecture: str) -> None:
        from mobius.integrations.gguf._builder import (
            _raise_for_invalid_encoder_tensor_contract,
        )

        model = SimpleNamespace(
            architecture=architecture,
            metadata={
                f"{architecture}.context_length": 32,
                f"{architecture}.feed_forward_length": 64,
                f"{architecture}.block_count": 1,
                f"{architecture}.attention.head_count": 4,
            },
        )
        with pytest.raises(ValueError, match=r"missing required encoder metadata.*embedding"):
            _raise_for_invalid_encoder_tensor_contract(model)

    @pytest.mark.parametrize("architecture", ["bert", "modern-bert"])
    def test_encoder_hidden_size_must_divide_attention_heads(self, architecture: str) -> None:
        from mobius.integrations.gguf._builder import (
            _raise_for_invalid_encoder_tensor_contract,
        )

        model = SimpleNamespace(
            architecture=architecture,
            metadata={
                f"{architecture}.context_length": 32,
                f"{architecture}.embedding_length": 65,
                f"{architecture}.feed_forward_length": 64,
                f"{architecture}.block_count": 1,
                f"{architecture}.attention.head_count": 4,
            },
        )
        with pytest.raises(ValueError, match=r"embedding_length must be divisible"):
            _raise_for_invalid_encoder_tensor_contract(model)

    @pytest.mark.parametrize("architecture", ["bert", "modern-bert"])
    def test_pooling_classifier_and_task_overrides_are_rejected(
        self, tmp_path: Path, architecture: str
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        pooled = tmp_path / f"{architecture}-pooled.gguf"
        _write_encoder_gguf(pooled, architecture, pooling_type=1)
        with pytest.raises(ValueError, match="pooling_type"):
            build_from_gguf(pooled)

        headed = tmp_path / f"{architecture}-head.gguf"
        _write_encoder_gguf(headed, architecture, include_head=True)
        with pytest.raises(ValueError, match="silently discard encoder heads"):
            build_from_gguf(headed)

        plain = tmp_path / f"{architecture}-task.gguf"
        _write_encoder_gguf(plain, architecture)
        with pytest.raises(ValueError, match="feature-extraction"):
            build_from_gguf(plain, task="text-generation")
        with pytest.raises(ValueError, match="encoder-only"):
            build_from_gguf(plain, static_cache=True)

    @pytest.mark.parametrize("architecture", ["bert", "modern-bert"])
    @pytest.mark.parametrize("failure", ["missing", "rank"])
    def test_missing_and_malformed_tensors_fail_before_graph_build(
        self, tmp_path: Path, architecture: str, failure: str, monkeypatch
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        target = "blk.0.attn_q.weight" if architecture == "bert" else "blk.0.attn_qkv.weight"
        path = tmp_path / f"{architecture}-{failure}.gguf"
        _write_encoder_gguf(
            path,
            architecture,
            omit=target if failure == "missing" else None,
            malformed=target if failure == "rank" else None,
        )
        monkeypatch.setattr(
            core_builder,
            "build_from_module",
            lambda *args, **kwargs: pytest.fail("graph construction must not start"),
        )
        with pytest.raises(ValueError, match=r"missing required|invalid encoder tensor shape"):
            build_from_gguf(path)

    @pytest.mark.parametrize("architecture", ["bert", "modern-bert"])
    @pytest.mark.parametrize("suffix", ["scale", "input_scale"])
    def test_auxiliary_quantization_sidecars_are_never_dropped(
        self, tmp_path: Path, architecture: str, suffix: str, monkeypatch
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-{suffix}.gguf"
        _write_encoder_gguf(path, architecture, auxiliary_suffix=suffix)
        monkeypatch.setattr(
            core_builder,
            "build_from_module",
            lambda *args, **kwargs: pytest.fail("graph construction must not start"),
        )
        with pytest.raises(ValueError, match=r"scale/input_scale"):
            build_from_gguf(path)


class TestT5GGUFBuild:
    """T5 GGUFs preserve encoder-only and seq2seq task contracts."""

    @staticmethod
    def _run_encoder(model, sequence_length: int, masked: bool = False) -> np.ndarray:
        from mobius._testing.ort_inference import OnnxModelSession

        mask = np.ones((1, sequence_length), dtype=np.int64)
        if masked:
            mask[0, -1] = 0
        session = OnnxModelSession(model)
        try:
            outputs = session.run(
                {
                    "input_ids": np.arange(sequence_length, dtype=np.int64)[None, :],
                    "attention_mask": mask,
                }
            )
        finally:
            session.close()
        assert set(outputs) == {"last_hidden_state"}
        return outputs["last_hidden_state"]

    @staticmethod
    def _run_seq2seq(package) -> tuple[np.ndarray, np.ndarray]:
        from mobius._testing.ort_inference import OnnxModelSession

        encoder = OnnxModelSession(package["encoder"])
        decoder = OnnxModelSession(package["decoder"])
        try:
            encoder_mask = np.array([[1, 1, 0]], dtype=np.int64)
            encoder_outputs = encoder.run(
                {
                    "input_ids": np.array([[2, 3, 0]], dtype=np.int64),
                    "attention_mask": encoder_mask,
                }
            )
            hidden = encoder_outputs["last_hidden_state"]
            prefill_feeds = {
                "input_ids": np.array([[0]], dtype=np.int64),
                "encoder_hidden_states": hidden,
                "attention_mask": np.ones((1, 1), dtype=np.int64),
                "encoder_attention_mask": encoder_mask,
            }
            for layer in range(2):
                for attention in ("self", "cross"):
                    for kind in ("key", "value"):
                        prefill_feeds[f"past_key_values.{layer}.{attention}.{kind}"] = (
                            np.zeros((1, 4, 0, 8), dtype=np.float32)
                        )
            prefill = decoder.run(prefill_feeds)

            decode_feeds = {
                "input_ids": np.array([[1]], dtype=np.int64),
                "encoder_hidden_states": np.zeros((1, 0, 32), dtype=np.float32),
                "attention_mask": np.ones((1, 2), dtype=np.int64),
                "encoder_attention_mask": encoder_mask,
            }
            for layer in range(2):
                for attention in ("self", "cross"):
                    for kind in ("key", "value"):
                        decode_feeds[f"past_key_values.{layer}.{attention}.{kind}"] = prefill[
                            f"present.{layer}.{attention}.{kind}"
                        ]
            decode = decoder.run(decode_feeds)
        finally:
            encoder.close()
            decoder.close()

        assert prefill["present.1.self.key"].shape == (1, 4, 1, 8)
        assert prefill["present.1.cross.key"].shape == (1, 4, 3, 8)
        assert decode["present.1.self.key"].shape == (1, 4, 2, 8)
        assert decode["present.1.cross.key"].shape == (1, 4, 3, 8)
        return prefill["logits"], decode["logits"]

    def test_t5encoder_float_save_load_and_variable_masks(self, tmp_path: Path) -> None:
        from mobius._model_package import ModelPackage
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "t5encoder-f32.gguf"
        _write_t5_gguf(path, "t5encoder")
        package = build_from_gguf(path)
        assert set(package) == {"model"}
        assert {value.name for value in package["model"].graph.outputs} == {
            "last_hidden_state"
        }
        assert not any(
            marker in value.name
            for value in (*package["model"].graph.inputs, *package["model"].graph.outputs)
            for marker in ("past_key_values", "present", "logits")
        )

        saved = tmp_path / "t5encoder-saved"
        package.save(saved, progress_bar=False)
        reloaded = ModelPackage.load(saved)
        for sequence_length in (1, 3, 7):
            output = self._run_encoder(
                reloaded["model"], sequence_length, masked=sequence_length > 1
            )
            assert output.shape == (1, sequence_length, 32)
            assert np.isfinite(output).all()

    def test_t5encoder_quantized_projection_matches_dequantized(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "t5encoder-q4.gguf"
        _write_t5_gguf(path, "t5encoder", quantized=True)
        preserved = build_from_gguf(path, keep_quantized=True)["model"]
        explicit_float = build_from_gguf(path, keep_quantized=False)["model"]

        op_types = {node.op_type for node in preserved.graph}
        assert "MatMulNBits" in op_types
        assert "GatherBlockQuantized" not in op_types
        actual = self._run_encoder(preserved, 5, masked=True)
        expected = self._run_encoder(explicit_float, 5, masked=True)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-2)

    def test_relative_bias_values_keep_bucket_head_orientation(self, tmp_path: Path) -> None:
        import torch

        from mobius.integrations.gguf._builder import _load_dequantized_state_dict
        from mobius.integrations.gguf._reader import GGUFModel

        path = tmp_path / "t5encoder-relative-bias.gguf"
        _write_t5_gguf(path, "t5encoder")
        state_dict = _load_dequantized_state_dict(GGUFModel(path), "t5encoder")
        expected = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        assert torch.equal(
            state_dict["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"],
            expected,
        )

    @pytest.mark.parametrize("architecture", ["t5", "t5encoder"])
    def test_gated_activation_ambiguity_fails_before_graph_build(
        self, tmp_path: Path, architecture: str, monkeypatch
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-gated.gguf"
        _write_t5_gguf(path, architecture, gated=True)
        monkeypatch.setattr(
            core_builder,
            "build_from_module",
            lambda *args, **kwargs: pytest.fail("graph construction must not start"),
        )
        with pytest.raises(ValueError, match=r"gated FFNs are ambiguous"):
            build_from_gguf(path)

    def test_t5_builds_distinct_encoder_and_decoder_contracts(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "t5-f32.gguf"
        _write_t5_gguf(path, "t5", encoder_layers=1, decoder_layers=2)
        package = build_from_gguf(path)
        assert set(package) == {"encoder", "decoder"}
        decoder_inputs = {value.name for value in package["decoder"].graph.inputs}
        assert {"encoder_hidden_states", "encoder_attention_mask"} <= decoder_inputs
        assert "past_key_values.1.cross.key" in decoder_inputs
        assert {value.name for value in package["decoder"].graph.outputs} >= {
            "logits",
            "present.1.self.key",
            "present.1.cross.key",
        }
        prefill_logits, decode_logits = self._run_seq2seq(package)
        assert prefill_logits.shape == decode_logits.shape == (1, 1, 64)
        assert np.isfinite(prefill_logits).all()
        assert np.isfinite(decode_logits).all()

    def test_t5_quantized_prefill_and_decode_match_dequantized(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "t5-q4.gguf"
        _write_t5_gguf(
            path,
            "t5",
            quantized=True,
            encoder_layers=1,
            decoder_layers=2,
        )
        preserved = build_from_gguf(path, keep_quantized=True)
        explicit_float = build_from_gguf(path, keep_quantized=False)
        assert "MatMulNBits" in {
            node.op_type for model in preserved.values() for node in model.graph
        }

        actual_prefill, actual_decode = self._run_seq2seq(preserved)
        expected_prefill, expected_decode = self._run_seq2seq(explicit_float)
        np.testing.assert_allclose(actual_prefill, expected_prefill, rtol=0, atol=1e-2)
        np.testing.assert_allclose(actual_decode, expected_decode, rtol=0, atol=1e-2)

    @pytest.mark.parametrize("architecture", ["t5", "t5encoder"])
    def test_task_and_static_cache_misdispatch_is_rejected(
        self, tmp_path: Path, architecture: str
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-task.gguf"
        _write_t5_gguf(path, architecture)
        with pytest.raises(ValueError, match="only supports task"):
            build_from_gguf(path, task="text-generation")
        with pytest.raises(ValueError, match="static_cache"):
            build_from_gguf(path, static_cache=True)

    @pytest.mark.parametrize("architecture", ["t5", "t5encoder"])
    @pytest.mark.parametrize("failure", ["missing", "rank"])
    def test_missing_and_malformed_tensors_fail_before_graph_build(
        self,
        tmp_path: Path,
        architecture: str,
        failure: str,
        monkeypatch,
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        target = "enc.blk.0.attn_q.weight"
        path = tmp_path / f"{architecture}-{failure}.gguf"
        _write_t5_gguf(
            path,
            architecture,
            omit=target if failure == "missing" else None,
            malformed=target if failure == "rank" else None,
        )
        monkeypatch.setattr(
            core_builder,
            "build_from_module",
            lambda *args, **kwargs: pytest.fail("graph construction must not start"),
        )
        with pytest.raises(ValueError, match=r"missing required|invalid T5 tensor shape"):
            build_from_gguf(path)

    @pytest.mark.parametrize(
        ("architecture", "tensor_name"),
        [
            ("t5encoder", "enc.blk.1.attn_q.weight"),
            ("t5", "dec.blk.1.cross_attn_q.weight"),
            ("t5", "enc.blk.-1.attn_q.weight"),
            ("t5", "dec.blk.+1.cross_attn_q.weight"),
            ("t5encoder", "enc.blk.01.attn_q.weight"),
            ("t5encoder", "enc.blk.1\u0660.attn_q.weight"),
        ],
    )
    def test_invalid_layer_indices_fail_before_graph_build(
        self,
        tmp_path: Path,
        architecture: str,
        tensor_name: str,
        monkeypatch,
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-invalid-layer.gguf"
        _write_t5_gguf(path, architecture, extra_layer_tensor=tensor_name)
        monkeypatch.setattr(
            core_builder,
            "build_from_module",
            lambda *args, **kwargs: pytest.fail("graph construction must not start"),
        )
        with pytest.raises(ValueError, match=r"invalid T5 layer tensor index"):
            build_from_gguf(path)

    @pytest.mark.parametrize("architecture", ["t5", "t5encoder"])
    def test_exact_pinned_ignored_tensors_remain_accepted(
        self, tmp_path: Path, architecture: str, caplog
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-ignored.gguf"
        _write_t5_gguf(path, architecture, include_ignored_tensor=True)
        build_from_gguf(path)
        assert "Ignoring pinned llama.cpp T5 tensor" in caplog.text

    def test_cross_relative_bias_ignore_is_not_a_suffix_wildcard(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "t5-broad-ignore.gguf"
        _write_t5_gguf(
            path,
            "t5",
            extra_layer_tensor="dec.blk.0.extra.cross_attn_rel_b.weight",
        )
        monkeypatch.setattr(
            core_builder,
            "build_from_module",
            lambda *args, **kwargs: pytest.fail("graph construction must not start"),
        )
        with pytest.raises(ValueError, match=r"suffixes do not match"):
            build_from_gguf(path)

    @pytest.mark.parametrize("architecture", ["t5", "t5encoder"])
    @pytest.mark.parametrize("suffix", ["scale", "input_scale"])
    def test_auxiliary_quantization_sidecars_are_never_dropped(
        self, tmp_path: Path, architecture: str, suffix: str, monkeypatch
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-{suffix}.gguf"
        _write_t5_gguf(path, architecture, auxiliary_suffix=suffix)
        monkeypatch.setattr(
            core_builder,
            "build_from_module",
            lambda *args, **kwargs: pytest.fail("graph construction must not start"),
        )
        with pytest.raises(ValueError, match=r"scale/input_scale"):
            build_from_gguf(path)


class TestHybridGGUFBuild:
    @staticmethod
    def _run_lfm2(model) -> list[dict[str, np.ndarray]]:
        from mobius._testing.ort_inference import OnnxModelSession

        session = OnnxModelSession(model)
        states = {
            "past_key_values.0.conv_state": np.zeros((1, 32, 2), dtype=np.float32),
            "past_key_values.1.key": np.zeros((1, 2, 0, 8), dtype=np.float32),
            "past_key_values.1.value": np.zeros((1, 2, 0, 8), dtype=np.float32),
        }
        outputs = []
        try:
            for tokens in ([1, 2, 3], [4]):
                total = states["past_key_values.1.key"].shape[2] + len(tokens)
                output = session.run(
                    {
                        "input_ids": np.asarray([tokens], dtype=np.int64),
                        "attention_mask": np.ones((1, total), dtype=np.int64),
                        "position_ids": np.arange(total - len(tokens), total, dtype=np.int64)[
                            None
                        ],
                        **states,
                    }
                )
                outputs.append(output)
                states = {
                    "past_key_values.0.conv_state": output["present.0.conv_state"],
                    "past_key_values.1.key": output["present.1.key"],
                    "past_key_values.1.value": output["present.1.value"],
                }
        finally:
            session.close()
        return outputs

    @staticmethod
    def _run_qwen35(model) -> list[dict[str, np.ndarray]]:
        from mobius._testing.ort_inference import OnnxModelSession

        session = OnnxModelSession(model)
        states = {}
        for layer in range(3):
            states[f"past_key_values.{layer}.conv_state"] = np.zeros(
                (1, 512, 2), dtype=np.float32
            )
            states[f"past_key_values.{layer}.recurrent_state"] = np.zeros(
                (1, 4, 64, 64), dtype=np.float32
            )
        states["past_key_values.3.key"] = np.zeros((1, 2, 0, 8), dtype=np.float32)
        states["past_key_values.3.value"] = np.zeros((1, 2, 0, 8), dtype=np.float32)
        outputs = []
        try:
            for tokens in ([1, 2, 3], [4]):
                total = states["past_key_values.3.key"].shape[2] + len(tokens)
                output = session.run(
                    {
                        "input_ids": np.asarray([tokens], dtype=np.int64),
                        "attention_mask": np.ones((1, total), dtype=np.int64),
                        "position_ids": np.arange(total - len(tokens), total, dtype=np.int64)[
                            None
                        ],
                        **states,
                    }
                )
                outputs.append(output)
                for layer in range(3):
                    states[f"past_key_values.{layer}.conv_state"] = output[
                        f"present.{layer}.conv_state"
                    ]
                    states[f"past_key_values.{layer}.recurrent_state"] = output[
                        f"present.{layer}.recurrent_state"
                    ]
                states["past_key_values.3.key"] = output["present.3.key"]
                states["past_key_values.3.value"] = output["present.3.value"]
        finally:
            session.close()
        return outputs

    def test_lfm2_float_prefill_decode_threads_only_mixer_specific_state(
        self, tmp_path: Path
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "lfm2-f32.gguf"
        _write_lfm2_gguf(path, quantized=False)
        model = build_from_gguf(path)["model"]

        assert [value.name for value in model.graph.inputs if "past_" in value.name] == [
            "past_key_values.0.conv_state",
            "past_key_values.1.key",
            "past_key_values.1.value",
        ]
        outputs = self._run_lfm2(model)
        assert outputs[0]["present.0.conv_state"].shape == (1, 32, 2)
        assert outputs[0]["present.1.key"].shape == (1, 2, 3, 8)
        assert outputs[1]["present.1.key"].shape == (1, 2, 4, 8)
        assert all(np.isfinite(output["logits"]).all() for output in outputs)

    def test_lfm2_quantized_preservation_dequantizes_float_targets_and_executes(
        self, tmp_path: Path
    ) -> None:
        from mobius.integrations.gguf import QuantizationDisposition, build_from_gguf

        path = tmp_path / "lfm2-q4.gguf"
        _write_lfm2_gguf(path, quantized=True)
        preserved_package = build_from_gguf(path, keep_quantized=True)
        preserved = preserved_package["model"]
        assert "MatMulNBits" not in {node.op_type for node in preserved.graph}
        assert preserved_package.gguf_quantization_report.storage_quantized is False
        assert any(
            record.disposition is QuantizationDisposition.DEQUANTIZED_FLOAT
            for record in preserved_package.gguf_quantization_report.tensor_records
        )
        preserved_outputs = self._run_lfm2(preserved)
        assert preserved_outputs[0]["present.0.conv_state"].shape == (1, 32, 2)
        assert preserved_outputs[1]["present.1.key"].shape == (1, 2, 4, 8)
        assert all(np.isfinite(output["logits"]).all() for output in preserved_outputs)

        explicit_float = build_from_gguf(path, keep_quantized=False)["model"]

        assert "MatMulNBits" not in {node.op_type for node in explicit_float.graph}
        outputs = self._run_lfm2(explicit_float)
        assert outputs[0]["present.0.conv_state"].shape == (1, 32, 2)
        assert outputs[1]["present.1.key"].shape == (1, 2, 4, 8)
        assert all(np.isfinite(output["logits"]).all() for output in outputs)

    def test_lfm2moe_float_prefill_decode_threads_mixed_state(self, tmp_path: Path) -> None:
        from mobius._model_package import ModelPackage
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "lfm2moe-f32.gguf"
        _write_lfm2moe_gguf(path, quantized=False)
        package = build_from_gguf(path)
        model = package["model"]

        assert [value.name for value in model.graph.inputs if "past_" in value.name] == [
            "past_key_values.0.conv_state",
            "past_key_values.1.key",
            "past_key_values.1.value",
        ]
        outputs = self._run_lfm2(model)
        assert outputs[0]["present.0.conv_state"].shape == (1, 32, 2)
        assert outputs[0]["present.1.key"].shape == (1, 2, 3, 8)
        assert outputs[1]["present.1.key"].shape == (1, 2, 4, 8)
        assert all(np.isfinite(output["logits"]).all() for output in outputs)

        saved = tmp_path / "saved-lfm2moe"
        package.save(str(saved), progress_bar=False, check_weights=True)
        reloaded = ModelPackage.load(str(saved))["model"]
        reloaded_outputs = self._run_lfm2(reloaded)
        for actual, expected in zip(reloaded_outputs, outputs):
            for name in actual:
                np.testing.assert_allclose(actual[name], expected[name], rtol=0, atol=0)

    def test_lfm2moe_quantized_source_requires_explicit_dequantization(
        self, tmp_path: Path
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "lfm2moe-q4.gguf"
        _write_lfm2moe_gguf(path, quantized=True)
        with pytest.raises(ValueError, match="keep_quantized=False"):
            build_from_gguf(path)

        explicit_float = build_from_gguf(path, keep_quantized=False)["model"]
        outputs = self._run_lfm2(explicit_float)
        assert all(np.isfinite(output["logits"]).all() for output in outputs)

    def test_lfm2moe_state_rollback_and_batch_reorder(self, tmp_path: Path) -> None:
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "lfm2moe-state.gguf"
        _write_lfm2moe_gguf(path, quantized=False)
        session = OnnxModelSession(build_from_gguf(path)["model"])
        try:
            prefill = session.run(
                {
                    "input_ids": np.asarray([[1, 2], [3, 4]], dtype=np.int64),
                    "attention_mask": np.ones((2, 2), dtype=np.int64),
                    "position_ids": np.asarray([[0, 1], [0, 1]], dtype=np.int64),
                    "past_key_values.0.conv_state": np.zeros((2, 32, 2), dtype=np.float32),
                    "past_key_values.1.key": np.zeros((2, 2, 0, 8), dtype=np.float32),
                    "past_key_values.1.value": np.zeros((2, 2, 0, 8), dtype=np.float32),
                }
            )
            snapshot = {
                "past_key_values.0.conv_state": prefill["present.0.conv_state"],
                "past_key_values.1.key": prefill["present.1.key"],
                "past_key_values.1.value": prefill["present.1.value"],
            }

            def decode(tokens: list[list[int]], states: dict[str, np.ndarray]):
                return session.run(
                    {
                        "input_ids": np.asarray(tokens, dtype=np.int64),
                        "attention_mask": np.ones((2, 3), dtype=np.int64),
                        "position_ids": np.asarray([[2], [2]], dtype=np.int64),
                        **states,
                    }
                )

            first = decode([[5], [6]], snapshot)
            replayed = decode([[5], [6]], snapshot)
            reordered = decode(
                [[6], [5]],
                {name: value[[1, 0]] for name, value in snapshot.items()},
            )
        finally:
            session.close()

        for name in first:
            np.testing.assert_allclose(replayed[name], first[name], rtol=0, atol=0)
            np.testing.assert_allclose(reordered[name], first[name][[1, 0]], rtol=0, atol=0)

    def test_qwen35_float_and_quantized_prefill_decode_thread_mixed_state(
        self, tmp_path: Path
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        float_path = tmp_path / "qwen35-f32.gguf"
        quantized_path = tmp_path / "qwen35-q4.gguf"
        _write_qwen35_gguf(float_path, quantized=False)
        _write_qwen35_gguf(quantized_path, quantized=True)

        float_steps = self._run_qwen35(build_from_gguf(float_path)["model"])
        quantized_model = build_from_gguf(quantized_path, keep_quantized=True)["model"]
        quantized_steps = self._run_qwen35(quantized_model)

        assert "MatMulNBits" in {node.op_type for node in quantized_model.graph}
        for steps in (float_steps, quantized_steps):
            assert steps[0]["present.0.conv_state"].shape == (1, 512, 2)
            assert steps[0]["present.0.recurrent_state"].shape == (1, 4, 64, 64)
            assert steps[0]["present.3.key"].shape == (1, 2, 3, 8)
            assert steps[1]["present.3.key"].shape == (1, 2, 4, 8)
            assert all(np.isfinite(output["logits"]).all() for output in steps)

    def test_qwen35_tied_quantized_embedding_owns_head_and_saves(self, tmp_path: Path) -> None:
        from mobius._model_package import ModelPackage
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "qwen35-tied-embedding-q4.gguf"
        _write_qwen35_gguf(path, quantized=True, quantized_embedding=True)
        package = build_from_gguf(path, keep_quantized=True)
        model = package["model"]

        initializer_names = set(model.graph.initializers)
        assert {
            "model.embed_tokens.qweight",
            "model.embed_tokens.scales",
            "model.embed_tokens.zero_points",
        } <= initializer_names
        assert not any(name.startswith("lm_head.") for name in initializer_names)
        assert all(
            value.const_value is not None for value in model.graph.initializers.values()
        )
        op_types = {node.op_type for node in model.graph}
        assert "GatherBlockQuantized" in op_types
        assert "MatMulNBits" in op_types

        saved = tmp_path / "saved-qwen35"
        package.save(str(saved), progress_bar=False, check_weights=True)
        reloaded = ModelPackage.load(str(saved))["model"]
        assert set(reloaded.graph.initializers) == initializer_names
        assert all(
            value.const_value is not None for value in reloaded.graph.initializers.values()
        )

    def test_qwen35_packed_v_head_reorder_fails_closed_when_not_repackable(
        self, tmp_path: Path
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "qwen35-narrow-q4.gguf"
        _write_qwen35_gguf(path, quantized=True, inner_size=32)
        with pytest.raises(ValueError, match=r"cannot map .* quant blocks"):
            build_from_gguf(path, keep_quantized=True)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"static_cache": True}, "architecture-specific"),
            ({"task": "text-generation"}, "hybrid-text-generation"),
        ],
    )
    def test_lfm2_cache_task_misdispatch_is_rejected(
        self, tmp_path: Path, kwargs: dict, message: str
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "lfm2-misdispatch.gguf"
        _write_lfm2_gguf(path, quantized=False)
        with pytest.raises(ValueError, match=re.escape(message)):
            build_from_gguf(path, **kwargs)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"static_cache": True}, "architecture-specific"),
            ({"task": "text-generation"}, "hybrid-text-generation"),
        ],
    )
    def test_lfm2moe_cache_task_misdispatch_is_rejected(
        self, tmp_path: Path, kwargs: dict, message: str
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "lfm2moe-misdispatch.gguf"
        _write_lfm2moe_gguf(path, quantized=False)
        with pytest.raises(ValueError, match=re.escape(message)):
            build_from_gguf(path, **kwargs)


class TestRecurrentGGUFBuild:
    """Mamba GGUF imports preserve recurrent state and tensor-role semantics."""

    @staticmethod
    def _run_steps(model, architecture: str) -> list[dict[str, np.ndarray]]:
        from mobius._testing.ort_inference import OnnxModelSession

        conv_channels = 64 if architecture == "mamba" else 80
        ssm_shape = (1, 64, 8) if architecture == "mamba" else (1, 4, 8, 16)
        states = {
            "past_states.0.conv_state": np.zeros((1, conv_channels, 3), dtype=np.float32),
            "past_states.0.ssm_state": np.zeros(ssm_shape, dtype=np.float32),
        }
        token_groups = [[1], [2], [3], [4]]
        if architecture == "mamba2":
            token_groups = [[1, 2, 3], [4]]

        session = OnnxModelSession(model)
        outputs = []
        try:
            for tokens in token_groups:
                out = session.run(
                    {
                        "input_ids": np.asarray([tokens], dtype=np.int64),
                        **states,
                    }
                )
                outputs.append(out)
                states = {
                    "past_states.0.conv_state": out["present.0.conv_state"],
                    "past_states.0.ssm_state": out["present.0.ssm_state"],
                }
        finally:
            session.close()
        return outputs

    @pytest.mark.parametrize("architecture", ["mamba", "mamba2"])
    def test_float_prefill_decode_state_threading_and_save_load(
        self, tmp_path: Path, architecture: str
    ) -> None:
        from mobius._model_package import ModelPackage
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-f32.gguf"
        _write_recurrent_gguf(path, architecture, quantized=False)
        package = build_from_gguf(path)

        output_dir = tmp_path / f"{architecture}-saved"
        package.save(output_dir, progress_bar=False)
        reloaded = ModelPackage.load(output_dir)
        outputs = self._run_steps(reloaded["model"], architecture)

        assert all(np.isfinite(out["logits"]).all() for out in outputs)
        assert np.count_nonzero(outputs[-1]["present.0.conv_state"]) > 0
        assert np.count_nonzero(outputs[-1]["present.0.ssm_state"]) > 0

    @pytest.mark.parametrize("architecture", ["mamba", "mamba2"])
    def test_quantized_projection_preservation_is_rejected_and_float_import_executes(
        self, tmp_path: Path, architecture: str
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-q4.gguf"
        _write_recurrent_gguf(path, architecture, quantized=True)
        with pytest.raises(ValueError, match="does not support keep_quantized=True"):
            build_from_gguf(path, keep_quantized=True)

        # Current Mamba projection consumers are ordinary Linear modules.
        # Explicit float import dequantizes the GGUF source before stateful execution.
        explicit_float = build_from_gguf(path, keep_quantized=False)["model"]
        assert "MatMulNBits" not in {node.op_type for node in explicit_float.graph}
        float_steps = self._run_steps(explicit_float, architecture)
        assert all(np.isfinite(out["logits"]).all() for out in float_steps)
        assert np.count_nonzero(float_steps[-1]["present.0.conv_state"]) > 0
        assert np.count_nonzero(float_steps[-1]["present.0.ssm_state"]) > 0

    @pytest.mark.parametrize("architecture", ["mamba", "mamba2"])
    def test_static_kv_cache_is_rejected(self, tmp_path: Path, architecture: str) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-static.gguf"
        _write_recurrent_gguf(path, architecture, quantized=False)
        with pytest.raises(ValueError, match="conv_state and ssm_state"):
            build_from_gguf(path, static_cache=True)

    def test_mamba1_graph_rejects_multi_token_input(self, tmp_path: Path) -> None:
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "mamba-multitoken.gguf"
        _write_recurrent_gguf(path, "mamba", quantized=False)
        session = OnnxModelSession(build_from_gguf(path)["model"])
        try:
            with pytest.raises(
                ort.capi.onnxruntime_pybind11_state.InvalidArgument,
                match=r"dimension|Expected",
            ):
                session.run(
                    {
                        "input_ids": np.asarray([[1, 2]], dtype=np.int64),
                        "past_states.0.conv_state": np.zeros((1, 64, 3), np.float32),
                        "past_states.0.ssm_state": np.zeros((1, 64, 8), np.float32),
                    }
                )
        finally:
            session.close()

    @pytest.mark.parametrize("architecture", ["mamba", "mamba2"])
    def test_malformed_conv_shape_is_not_silently_loaded(
        self, tmp_path: Path, architecture: str
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-bad-conv.gguf"
        _write_recurrent_gguf(path, architecture, quantized=False, malformed_conv=True)
        with pytest.raises(ValueError, match="shape"):
            build_from_gguf(path)

    @pytest.mark.parametrize("architecture", ["mamba", "mamba2"])
    def test_malformed_recurrent_suffix_is_rejected_before_graph_build(
        self, tmp_path: Path, architecture: str, monkeypatch
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-bad-suffix.gguf"
        _write_recurrent_gguf(
            path,
            architecture,
            quantized=False,
            malformed_suffix=True,
        )

        graph_build_started = False

        def unexpected_graph_build(*args, **kwargs):
            nonlocal graph_build_started
            graph_build_started = True
            raise AssertionError("graph construction must not start")

        monkeypatch.setattr(core_builder, "build_from_module", unexpected_graph_build)
        with pytest.raises(ValueError, match="suffixes do not match"):
            build_from_gguf(path)
        assert not graph_build_started

    @pytest.mark.parametrize("architecture", ["mamba", "mamba2"])
    def test_auxiliary_quantization_sidecar_is_rejected_before_graph_build(
        self, tmp_path: Path, architecture: str, monkeypatch
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"{architecture}-sidecar.gguf"
        _write_recurrent_gguf(path, architecture, quantized=False, auxiliary_scale=True)
        graph_build_started = False

        def unexpected_graph_build(*args, **kwargs):
            nonlocal graph_build_started
            graph_build_started = True
            raise AssertionError("graph construction must not start")

        monkeypatch.setattr(core_builder, "build_from_module", unexpected_graph_build)
        with pytest.raises(ValueError, match="scale/input_scale"):
            build_from_gguf(path)
        assert not graph_build_started


class TestFalconH1GGUFBuild:
    """Falcon-H1 GGUF import preserves its parallel graph and four-state ABI."""

    @staticmethod
    def _initial_inputs(batch: int, tokens: list[list[int]]) -> dict[str, np.ndarray]:
        sequence = len(tokens[0])
        return {
            "input_ids": np.asarray(tokens, dtype=np.int64),
            "position_ids": np.broadcast_to(
                np.arange(sequence, dtype=np.int64),
                (batch, sequence),
            ).copy(),
            "attention_mask": np.ones((batch, sequence), dtype=np.int64),
            "past_key_values.0.key": np.zeros((batch, 2, 0, 8), np.float32),
            "past_key_values.0.value": np.zeros((batch, 2, 0, 8), np.float32),
            "past_key_values.0.conv_state": np.zeros((batch, 48, 3), np.float32),
            "past_key_values.0.ssm_state": np.zeros((batch, 4, 8, 8), np.float32),
        }

    def test_float_prefill_decode_replay_batch_reorder_and_save_load(
        self, tmp_path: Path
    ) -> None:
        from mobius._model_package import ModelPackage
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "falcon-h1-f32.gguf"
        _write_falcon_h1_gguf(path, quantized=False)
        package = build_from_gguf(path)
        output_dir = tmp_path / "saved"
        package.save(output_dir, progress_bar=False)
        model = ModelPackage.load(output_dir)["model"]
        assert [value.name for value in model.graph.outputs] == [
            "logits",
            "present.0.key",
            "present.0.value",
            "present.0.conv_state",
            "present.0.ssm_state",
        ]

        session = OnnxModelSession(model)
        try:
            prefill = session.run(self._initial_inputs(2, [[1, 2, 3], [4, 5, 6]]))
            decode_inputs = {
                "input_ids": np.asarray([[7], [8]], np.int64),
                "position_ids": np.asarray([[3], [3]], np.int64),
                "attention_mask": np.ones((2, 4), np.int64),
                **{
                    name.replace("present.", "past_key_values."): value
                    for name, value in prefill.items()
                    if name.startswith("present.")
                },
            }
            first = session.run(decode_inputs)
            replay = session.run(decode_inputs)
            np.testing.assert_array_equal(first["logits"], replay["logits"])

            reordered = {
                name: value[::-1].copy() if value.shape[0] == 2 else value
                for name, value in decode_inputs.items()
            }
            reordered["attention_mask"] = decode_inputs["attention_mask"][::-1].copy()
            swapped = session.run(reordered)
            np.testing.assert_allclose(swapped["logits"], first["logits"][::-1])
        finally:
            session.close()

    def test_quantized_source_dequantizes_and_keep_quantized_fails_closed(
        self, tmp_path: Path
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "falcon-h1-q4.gguf"
        _write_falcon_h1_gguf(path, quantized=True)
        model = build_from_gguf(path, keep_quantized=False)["model"]
        assert "MatMulNBits" not in {node.op_type for node in model.graph}
        with pytest.raises(ValueError, match="keep_quantized=True"):
            build_from_gguf(path, keep_quantized=True)

    def test_complete_optional_bias_families_select_the_biased_graph(
        self, tmp_path: Path
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "falcon-h1-biased.gguf"
        _write_falcon_h1_gguf(path, quantized=False, biases=True)
        model = build_from_gguf(path)["model"]
        assert "model.layers.0.self_attn.q_proj.bias" in model.graph.initializers
        assert "model.layers.0.self_attn.o_proj.bias" in model.graph.initializers
        assert "model.layers.0.feed_forward.gate_proj.bias" in model.graph.initializers

    def test_cli_builds_dequantized_graph_package(self, tmp_path: Path) -> None:
        from mobius.__main__ import main

        path = tmp_path / "falcon-h1-cli.gguf"
        output_dir = tmp_path / "falcon-h1-cli"
        _write_falcon_h1_gguf(path, quantized=True)
        main(
            [
                "build-gguf",
                str(path),
                "--output",
                str(output_dir),
                "--dequantize",
            ]
        )
        assert (output_dir / "model.onnx").is_file()

    def test_static_cache_and_incomplete_closure_fail_before_graph(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        complete = tmp_path / "falcon-h1-static.gguf"
        _write_falcon_h1_gguf(complete, quantized=False)
        with pytest.raises(ValueError, match="four-state"):
            build_from_gguf(complete, static_cache=True)

        incomplete = tmp_path / "falcon-h1-incomplete.gguf"
        _write_falcon_h1_gguf(
            incomplete,
            quantized=False,
            omit="blk.0.ssm_out.weight",
        )
        graph_build_started = False

        def unexpected_graph_build(*args, **kwargs):
            nonlocal graph_build_started
            graph_build_started = True
            raise AssertionError("graph construction must not start")

        monkeypatch.setattr(core_builder, "build_from_module", unexpected_graph_build)
        with pytest.raises(ValueError, match="tensor closure"):
            build_from_gguf(incomplete)
        assert not graph_build_started

        partial_bias = tmp_path / "falcon-h1-partial-bias.gguf"
        _write_falcon_h1_gguf(
            partial_bias,
            quantized=False,
            partial_bias=True,
        )
        with pytest.raises(ValueError, match="biases must be present"):
            build_from_gguf(partial_bias)
        assert not graph_build_started


class TestPlamo2GGUFBuild:
    """PLaMo2 GGUF import preserves exact fused tensors and mixed state."""

    @staticmethod
    def _inputs(batch: int) -> dict[str, np.ndarray]:
        return {
            "input_ids": np.asarray([[1, 2], [3, 4]][:batch], np.int64),
            "position_ids": np.asarray([[0, 1], [0, 1]][:batch], np.int64),
            "attention_mask": np.ones((batch, 2), np.int64),
            "past_key_values.0.conv_state": np.zeros((batch, 32, 3), np.float32),
            "past_key_values.0.recurrent_state": np.zeros((batch, 4, 8, 4), np.float32),
            "past_key_values.1.key": np.zeros((batch, 2, 0, 8), np.float32),
            "past_key_values.1.value": np.zeros((batch, 2, 0, 8), np.float32),
        }

    def test_float_import_executes_and_round_trips(self, tmp_path: Path) -> None:
        from gguf import GGUFReader

        from mobius._model_package import ModelPackage
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "plamo2-f32.gguf"
        _write_plamo2_gguf(path, quantized=False)
        package = build_from_gguf(path)
        model = package["model"]
        source_tensors = {tensor.name: tensor.data for tensor in GGUFReader(path).tensors}
        norm_pairs = {
            "output_norm.weight": "model.norm.weight",
            "blk.0.attn_norm.weight": "model.layers.0.pre_mixer_norm.weight",
            "blk.0.post_attention_norm": "model.layers.0.post_mixer_norm.weight",
            "blk.0.ffn_norm.weight": "model.layers.0.pre_mlp_norm.weight",
            "blk.0.post_ffw_norm": "model.layers.0.post_mlp_norm.weight",
        }
        for source_name, initializer_name in norm_pairs.items():
            np.testing.assert_array_equal(
                model.graph.initializers[initializer_name].const_value.numpy(),
                source_tensors[source_name],
            )
        assert model.metadata_props["mobius.runtime_support"] == (
            "ORT GenAI 0.15.2 state ABI; package requires GQA-specialized attention"
        )
        assert [value.name for value in model.graph.outputs] == [
            "logits",
            "present.0.conv_state",
            "present.0.recurrent_state",
            "present.1.key",
            "present.1.value",
        ]
        output_dir = tmp_path / "saved"
        package.save(output_dir, progress_bar=False)
        session = OnnxModelSession(ModelPackage.load(output_dir)["model"])
        outputs = session.run(self._inputs(2))
        assert outputs["logits"].shape == (2, 2, 64)
        assert outputs["present.0.recurrent_state"].dtype == np.float32

    def test_legacy_scalar_heads_infer_exact_tensor_schedule(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "plamo2-legacy-scalar-heads.gguf"
        _write_plamo2_gguf(path, quantized=False, legacy_scalar_heads=True)
        package = build_from_gguf(path)
        assert package.config.layer_types == ["mamba", "full_attention"]
        assert package.config.attention_head_counts == (0, 4)
        assert package.config.attention_kv_head_counts == (0, 2)

    def test_legacy_million_base_restores_reference_local_rope(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "plamo2-legacy-rope.gguf"
        _write_plamo2_gguf(path, quantized=False, rope_theta=1_000_000.0)
        package = build_from_gguf(path)
        assert package.config.rope_theta == pytest.approx(10_000.0)

    def test_quantized_source_preserves_exact_projection_roles(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "plamo2-q4.gguf"
        _write_plamo2_gguf(path, quantized=True)
        model = build_from_gguf(path, keep_quantized=True)["model"]
        assert sum(node.op_type == "MatMulNBits" for node in model.graph) == 9
        assert (
            model.graph.initializers["model.layers.0.mixer.A_log"].const_value.dtype
            == ir.DataType.FLOAT
        )
        assert (
            model.graph.initializers["model.layers.0.mixer.dt_proj.weight_t"].const_value.dtype
            == ir.DataType.FLOAT
        )
        assert (
            model.graph.initializers["model.layers.0.mixer.B_norm_weight"].const_value.dtype
            == ir.DataType.FLOAT
        )

    def test_quantized_tied_embedding_is_preserved_once(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "plamo2-q4-embedding.gguf"
        _write_plamo2_gguf(path, quantized=True, quantized_embedding=True)
        model = build_from_gguf(path, keep_quantized=True)["model"]
        op_types = [node.op_type for node in model.graph]
        assert op_types.count("GatherBlockQuantized") == 1
        assert op_types.count("MatMulNBits") == 10
        assert (
            sum(name.endswith("embed_tokens.qweight") for name in model.graph.initializers)
            == 1
        )

    def test_quantized_untied_output_head_is_preserved(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "plamo2-q4-output.gguf"
        _write_plamo2_gguf(path, quantized=True, include_output=True)
        model = build_from_gguf(path, keep_quantized=True)["model"]
        assert sum(node.op_type == "MatMulNBits" for node in model.graph) == 10
        assert "lm_head.weight" in model.graph.initializers
        assert "lm_head.scales" in model.graph.initializers

    def test_cli_builds_dequantized_package(self, tmp_path: Path) -> None:
        from mobius.__main__ import main

        source = tmp_path / "plamo2-cli.gguf"
        output = tmp_path / "plamo2-cli"
        _write_plamo2_gguf(source, quantized=True)
        main(["build-gguf", str(source), "--output", str(output), "--dequantize"])
        assert (output / "model.onnx").is_file()

    def test_static_cache_is_rejected(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "plamo2-static.gguf"
        _write_plamo2_gguf(path, quantized=False)
        with pytest.raises(ValueError, match="heterogeneous"):
            build_from_gguf(path, static_cache=True)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"omit": "blk.0.ssm_out.weight"}, "tensor closure"),
            ({"extra": "blk.0.unknown.weight"}, "tensor"),
            ({"invalid_decay": True}, "finite negative"),
            ({"group_count": 1}, "group_count=0"),
            ({"epsilon": 1e-5}, "rms_epsilon=1e-6"),
            ({"activation": "gelu"}, "'silu' or 'swiglu'"),
            ({"predefined_state": True}, "predefined initial state"),
            ({"head_counts": [4, 4]}, "head_count=4"),
            ({"kv_head_counts": [0]}, "match block_count"),
        ],
    )
    def test_malformed_sources_fail_before_graph(
        self,
        tmp_path: Path,
        monkeypatch,
        kwargs: dict[str, object],
        match: str,
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "plamo2-invalid.gguf"
        _write_plamo2_gguf(path, quantized=False, **kwargs)
        graph_build_started = False

        def unexpected_graph_build(*args, **kwargs):
            nonlocal graph_build_started
            graph_build_started = True
            raise AssertionError("graph construction must not start")

        monkeypatch.setattr(core_builder, "build_from_module", unexpected_graph_build)
        with pytest.raises(ValueError, match=match):
            build_from_gguf(path)
        assert not graph_build_started


class TestJambaGGUFBuild:
    """Jamba GGUF import preserves mixed mixers, routed experts, and state."""

    @staticmethod
    def _inputs(tokens: np.ndarray) -> dict[str, np.ndarray]:
        batch, sequence = tokens.shape
        return {
            "input_ids": tokens,
            "position_ids": np.broadcast_to(
                np.arange(sequence, dtype=np.int64),
                (batch, sequence),
            ).copy(),
            "attention_mask": np.ones((batch, sequence), np.int64),
            "past_key_values.0.conv_state": np.zeros((batch, 64, 3), np.float32),
            "past_key_values.0.ssm_state": np.zeros((batch, 64, 4), np.float32),
            "past_key_values.1.key": np.zeros((batch, 2, 0, 8), np.float32),
            "past_key_values.1.value": np.zeros((batch, 2, 0, 8), np.float32),
        }

    def test_float_import_preserves_expert_order_and_round_trips(self, tmp_path: Path) -> None:
        from mobius._model_package import ModelPackage
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "jamba-f32.gguf"
        _write_jamba_gguf(path, quantized=False)
        package = build_from_gguf(path)
        model = package["model"]
        assert model.metadata_props["mobius.runtime_support"].endswith(
            "onnxruntime/mobius/issues/605"
        )
        assert [value.name for value in model.graph.outputs] == [
            "logits",
            "present.0.conv_state",
            "present.0.ssm_state",
            "present.1.key",
            "present.1.value",
        ]
        for expert in range(2):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                value = model.graph.initializers[
                    f"model.layers.1.feed_forward.experts.{expert}.{projection}.weight_t"
                ].const_value.numpy()
                np.testing.assert_array_equal(value, expert + 1)
        output_dir = tmp_path / "saved-jamba"
        package.save(output_dir, progress_bar=False)
        session = OnnxModelSession(ModelPackage.load(output_dir)["model"])
        outputs = session.run(self._inputs(np.asarray([[1, 2], [3, 4]], np.int64)))
        assert outputs["logits"].shape == (2, 2, 64)
        assert outputs["present.0.ssm_state"].dtype == np.float32

    def test_prefill_decode_reorder_and_snapshot_replay(self, tmp_path: Path) -> None:
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "jamba-state.gguf"
        _write_jamba_gguf(path, quantized=False)
        session = OnnxModelSession(build_from_gguf(path)["model"])
        histories = np.asarray([[1, 2], [3, 4]], np.int64)
        prefill = session.run(self._inputs(histories))

        order = np.asarray([1, 0], np.int64)
        next_tokens = np.asarray([[5], [6]], np.int64)
        decode_inputs = {
            "input_ids": next_tokens,
            "position_ids": np.full((2, 1), 2, np.int64),
            "attention_mask": np.ones((2, 3), np.int64),
            "past_key_values.0.conv_state": prefill["present.0.conv_state"][order],
            "past_key_values.0.ssm_state": prefill["present.0.ssm_state"][order],
            "past_key_values.1.key": prefill["present.1.key"][order],
            "past_key_values.1.value": prefill["present.1.value"][order],
        }
        decoded = session.run(decode_inputs)
        replayed = session.run(decode_inputs)
        for name in decoded:
            np.testing.assert_array_equal(decoded[name], replayed[name])

        full_tokens = np.concatenate([histories[order], next_tokens], axis=1)
        full = session.run(self._inputs(full_tokens))
        np.testing.assert_allclose(
            decoded["logits"][:, -1],
            full["logits"][:, -1],
            atol=2e-5,
            rtol=2e-5,
        )

    def test_quantized_source_keeps_only_compatible_matmul_roles(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "jamba-q4.gguf"
        _write_jamba_gguf(path, quantized=True)
        model = build_from_gguf(path, keep_quantized=True)["model"]
        # 4 attention + 3 dense FFN + 2 experts * 3 expert projections.
        assert sum(node.op_type == "MatMulNBits" for node in model.graph) == 13
        for stem in (
            "model.layers.0.mamba.in_proj.weight_t",
            "model.layers.0.mamba.conv1d.weight",
            "model.layers.0.mamba.ssm.x_proj.weight_t",
            "model.layers.0.mamba.ssm.dt_proj.weight_t",
            "model.layers.0.mamba.out_proj.weight_t",
        ):
            assert model.graph.initializers[stem].const_value.dtype == ir.DataType.FLOAT

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"omit": "blk.1.ffn_up_exps.weight"}, "tensor closure"),
            ({"extra": "blk.0.ffn_gate_inp.weight"}, "tensor closure"),
            (
                {"malformed_shape": "blk.1.ffn_gate_exps.weight"},
                "tensor shape",
            ),
            ({"expert_count": 0, "expert_used_count": 1}, "expert_count"),
            ({"expert_count": 1, "expert_used_count": 1}, "not a routed-MoE"),
            ({"expert_count": 2, "expert_used_count": 3}, "expert_used_count"),
            ({"invalid_decay": True}, "finite negative"),
            ({"extra": "blk.0.ssm_in.scale"}, "auxiliary|tensor closure"),
        ],
    )
    def test_malformed_sources_fail_before_graph(
        self,
        tmp_path: Path,
        monkeypatch,
        kwargs: dict[str, object],
        match: str,
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "jamba-invalid.gguf"
        _write_jamba_gguf(path, quantized=False, **kwargs)
        graph_build_started = False

        def unexpected_graph_build(*args, **kwargs):
            nonlocal graph_build_started
            graph_build_started = True
            raise AssertionError("graph construction must not start")

        monkeypatch.setattr(core_builder, "build_from_module", unexpected_graph_build)
        with pytest.raises(ValueError, match=match):
            build_from_gguf(path)
        assert not graph_build_started


class TestMiniMaxGGUFBuild:
    """MiniMax-01 GGUF import preserves hybrid state and expert tensor order."""

    @staticmethod
    def _inputs(tokens: np.ndarray, states: dict[str, np.ndarray] | None = None):
        batch, sequence = tokens.shape
        if states is None:
            states = {
                "past_key_values.0.recurrent_state": np.zeros(
                    (batch, 4, 16, 16), dtype=np.float32
                ),
                "past_key_values.1.key": np.zeros((batch, 2, 0, 16), dtype=np.float32),
                "past_key_values.1.value": np.zeros((batch, 2, 0, 16), dtype=np.float32),
            }
        past = states["past_key_values.1.key"].shape[2]
        return {
            "input_ids": tokens,
            "attention_mask": np.ones((batch, past + sequence), dtype=np.int64),
            "position_ids": np.broadcast_to(
                np.arange(past, past + sequence, dtype=np.int64), (batch, sequence)
            ).copy(),
            **states,
        }

    def test_float_import_runtime_save_reload_and_expert_order(self, tmp_path: Path) -> None:
        from mobius._model_package import ModelPackage
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "minimax-f32.gguf"
        _write_minimax_gguf(path, quantized=False)
        package = build_from_gguf(path)
        model = package["model"]
        assert [value.name for value in model.graph.inputs if "past_" in value.name] == [
            "past_key_values.0.recurrent_state",
            "past_key_values.1.key",
            "past_key_values.1.value",
        ]
        for layer in range(2):
            for expert in range(2):
                prefix = f"model.layers.{layer}.mlp.experts.{expert}"
                np.testing.assert_array_equal(
                    model.graph.initializers[
                        f"{prefix}.gate_proj.weight_t"
                    ].const_value.numpy(),
                    11.0 + expert,
                )
                np.testing.assert_array_equal(
                    model.graph.initializers[f"{prefix}.up_proj.weight_t"].const_value.numpy(),
                    21.0 + expert,
                )
                np.testing.assert_array_equal(
                    model.graph.initializers[
                        f"{prefix}.down_proj.weight_t"
                    ].const_value.numpy(),
                    31.0 + expert,
                )

        session = OnnxModelSession(model)
        try:
            prefill = session.run(self._inputs(np.asarray([[1, 2], [3, 4]], np.int64)))
            snapshot = {
                "past_key_values.0.recurrent_state": prefill["present.0.recurrent_state"],
                "past_key_values.1.key": prefill["present.1.key"],
                "past_key_values.1.value": prefill["present.1.value"],
            }
            first = session.run(self._inputs(np.asarray([[5], [6]], np.int64), snapshot))
            replay = session.run(self._inputs(np.asarray([[5], [6]], np.int64), snapshot))
            reordered = session.run(
                self._inputs(
                    np.asarray([[6], [5]], np.int64),
                    {name: value[[1, 0]] for name, value in snapshot.items()},
                )
            )
        finally:
            session.close()
        for name in first:
            np.testing.assert_allclose(replay[name], first[name], rtol=0, atol=0)
            np.testing.assert_allclose(reordered[name], first[name][[1, 0]], rtol=0, atol=0)

        saved = tmp_path / "saved-minimax"
        package.save(saved, progress_bar=False, check_weights=True)
        reloaded = ModelPackage.load(saved)["model"]
        assert [value.name for value in reloaded.graph.outputs] == [
            value.name for value in model.graph.outputs
        ]

    def test_quantized_source_preserves_exact_projection_roles(self, tmp_path: Path) -> None:
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "minimax-q4.gguf"
        _write_minimax_gguf(path, quantized=True, quantized_embedding=True)
        model = build_from_gguf(path, keep_quantized=True)["model"]
        assert sum(node.op_type == "GatherBlockQuantized" for node in model.graph) == 1
        expected_projection_weights = {
            "model.layers.0.self_attn.qkv_proj.weight",
            "model.layers.0.self_attn.output_gate.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.1.self_attn.q_proj.weight",
            "model.layers.1.self_attn.k_proj.weight",
            "model.layers.1.self_attn.v_proj.weight",
            "model.layers.1.self_attn.o_proj.weight",
            *{
                f"model.layers.{layer}.mlp.experts.{expert}.{projection}_proj.weight"
                for layer in range(2)
                for expert in range(2)
                for projection in ("gate", "up", "down")
            },
        }
        assert {
            node.inputs[1].name for node in model.graph if node.op_type == "MatMulNBits"
        } == expected_projection_weights
        assert all(
            "norm" not in node.inputs[1].name
            for node in model.graph
            if node.op_type == "MatMulNBits"
        )
        session = OnnxModelSession(model)
        try:
            outputs = session.run(self._inputs(np.asarray([[1, 2]], np.int64)))
        finally:
            session.close()
        assert np.isfinite(outputs["logits"]).all()

    def test_tied_output_uses_embedding_storage(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "minimax-tied.gguf"
        _write_minimax_gguf(path, quantized=False, omit="output.weight")
        model = build_from_gguf(path)["model"]
        assert "lm_head.weight" not in model.graph.initializers
        assert "model.embed_tokens.weight" in model.graph.initializers

    def test_quantized_tied_output_uses_embedding_storage(self, tmp_path: Path) -> None:
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "minimax-q4-tied.gguf"
        _write_minimax_gguf(
            path,
            quantized=True,
            quantized_embedding=True,
            omit="output.weight",
        )
        model = build_from_gguf(path, keep_quantized=True)["model"]
        op_types = [node.op_type for node in model.graph]
        assert op_types.count("GatherBlockQuantized") == 1
        assert op_types.count("MatMulNBits") == 20
        assert (
            sum(name.endswith("embed_tokens.qweight") for name in model.graph.initializers)
            == 1
        )
        assert not any(name.startswith("lm_head.") for name in model.graph.initializers)
        tied_head = next(
            node for node in reversed(model.graph) if node.op_type == "MatMulNBits"
        )
        assert tied_head.inputs[2].name == "model.embed_tokens.scales"
        assert tied_head.inputs[3].name == "model.embed_tokens.zero_points"
        session = OnnxModelSession(model)
        try:
            outputs = session.run(self._inputs(np.asarray([[1, 2]], np.int64)))
        finally:
            session.close()
        assert np.isfinite(outputs["logits"]).all()

    def test_cli_build(self, tmp_path: Path) -> None:
        from mobius.__main__ import main
        from mobius._model_package import ModelPackage

        path = tmp_path / "minimax-cli.gguf"
        output = tmp_path / "minimax-cli-output"
        _write_minimax_gguf(path, quantized=False)
        main(["build-gguf", str(path), "--output", str(output), "--dequantize"])
        assert "model" in ModelPackage.load(output)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            (
                {"omit": "blk.0.attn_gate.weight"},
                "missing=.*blk.0.attn_gate.weight",
            ),
            (
                {"extra": "blk.0.attn_q.weight"},
                "unexpected=.*blk.0.attn_q.weight",
            ),
            (
                {"extra": "blk.2.attn_q.weight"},
                "out_of_range=.*blk.2.attn_q.weight",
            ),
            (
                {"malformed_shape": "blk.1.attn_k.weight"},
                "invalid tensor shape",
            ),
            (
                {"recurrent_layers": [True]},
                "must contain exactly 2 entries",
            ),
            (
                {"norm_eps": 0.0},
                "inconsistent architecture metadata",
            ),
            (
                {"rope_freq_base": float("nan")},
                "inconsistent architecture metadata",
            ),
        ],
    )
    def test_invalid_contract_rejected_before_graph(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        kwargs: dict,
        match: str,
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "minimax-invalid.gguf"
        _write_minimax_gguf(path, quantized=False, **kwargs)
        graph_build = mock.Mock(side_effect=AssertionError("graph construction reached"))
        monkeypatch.setattr(core_builder, "build_from_module", graph_build)
        with pytest.raises(ValueError, match=match):
            build_from_gguf(path)
        graph_build.assert_not_called()


class TestKimiLinearGGUFBuild:
    """Kimi Linear GGUF import preserves its heterogeneous state and tensor roles."""

    @staticmethod
    def _inputs(
        tokens: np.ndarray,
        states: dict[str, np.ndarray] | None = None,
        attention_mask: np.ndarray | None = None,
    ):
        batch, sequence = tokens.shape
        if states is None:
            states = {
                "past_key_values.0.q_conv_state": np.zeros((batch, 64, 3), dtype=np.float32),
                "past_key_values.0.k_conv_state": np.zeros((batch, 64, 3), dtype=np.float32),
                "past_key_values.0.v_conv_state": np.zeros((batch, 64, 3), dtype=np.float32),
                "past_key_values.0.recurrent_state": np.zeros(
                    (batch, 2, 32, 32), dtype=np.float32
                ),
                "past_key_values.1.key": np.zeros((batch, 2, 0, 48), dtype=np.float32),
                "past_key_values.1.value": np.zeros((batch, 2, 0, 32), dtype=np.float32),
            }
        past = states["past_key_values.1.key"].shape[2]
        return {
            "input_ids": tokens,
            "attention_mask": (
                np.ones((batch, past + sequence), dtype=np.int64)
                if attention_mask is None
                else attention_mask
            ),
            "position_ids": np.broadcast_to(
                np.arange(past, past + sequence, dtype=np.int64), (batch, sequence)
            ).copy(),
            **states,
        }

    def test_float_import_runtime_state_replay_reorder_and_roundtrip(
        self, tmp_path: Path
    ) -> None:
        from mobius._model_package import ModelPackage
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "kimi-linear-f32.gguf"
        _write_kimi_linear_gguf(path, quantized=False)
        package = build_from_gguf(path)
        model = package["model"]
        assert package.config.layer_types == ["kimi_linear_attention", "full_attention"]
        assert model.metadata_props["mobius.cache_abi"].startswith(
            "KDA:q_conv_state,k_conv_state,v_conv_state,recurrent_state"
        )
        assert [value.name for value in model.graph.outputs[1:]] == [
            "present.0.q_conv_state",
            "present.0.k_conv_state",
            "present.0.v_conv_state",
            "present.0.recurrent_state",
            "present.1.key",
            "present.1.value",
        ]

        session = OnnxModelSession(model)
        try:
            prefill = session.run(self._inputs(np.asarray([[1, 2], [3, 4]], np.int64)))
            snapshot = {
                name.replace("present.", "past_key_values."): value.copy()
                for name, value in prefill.items()
                if name.startswith("present.")
            }
            tokens = np.asarray([[5], [6]], np.int64)
            first = session.run(self._inputs(tokens, snapshot))
            replay = session.run(self._inputs(tokens, snapshot))
            np.testing.assert_array_equal(first["logits"], replay["logits"])

            reordered = {name: value[::-1].copy() for name, value in snapshot.items()}
            swapped = session.run(self._inputs(tokens[::-1].copy(), reordered))
            np.testing.assert_allclose(
                swapped["logits"], first["logits"][::-1], rtol=1e-5, atol=1e-5
            )
        finally:
            session.close()

        output = tmp_path / "roundtrip"
        package.save(output, progress_bar=False)
        reloaded = ModelPackage.load(str(output))
        assert set(reloaded) == {"model"}
        assert (
            reloaded["model"].metadata_props["mobius.cache_abi"]
            == model.metadata_props["mobius.cache_abi"]
        )

    @pytest.mark.parametrize(
        ("native_quantized", "qtype_name"),
        [(False, "Q4_0"), (True, "IQ4_NL")],
    )
    def test_quantized_mla_reshape_is_reported_as_lossy(
        self,
        tmp_path: Path,
        native_quantized: bool,
        qtype_name: str,
        caplog,
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / f"kimi-linear-{qtype_name.lower()}.gguf"
        _write_kimi_linear_gguf(
            path,
            quantized=True,
            native_quantized=native_quantized,
        )
        with caplog.at_level("WARNING"):
            package = build_from_gguf(path, keep_quantized=True)
        assert caplog.text.count("GGUF QUANTIZATION FIDELITY WARNING") == 1
        assert f"{qtype_name}:" in caplog.text
        assert package.gguf_quantization_report.source_fidelity is False

        model = build_from_gguf(path, keep_quantized=False)["model"]
        assert all(node.op_type != "MatMulNBits" for node in model.graph)

    def test_left_padding_does_not_change_valid_logits_or_states(self, tmp_path: Path) -> None:
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "kimi-linear-padding.gguf"
        _write_kimi_linear_gguf(path, quantized=False)
        session = OnnxModelSession(build_from_gguf(path)["model"])
        try:
            unpadded = session.run(self._inputs(np.asarray([[7, 8]], np.int64)))
            padded = session.run(
                self._inputs(
                    np.asarray([[0, 0, 7, 8]], np.int64),
                    attention_mask=np.asarray([[0, 0, 1, 1]], np.int64),
                )
            )
            np.testing.assert_allclose(
                padded["logits"][:, -2:], unpadded["logits"], rtol=1e-5, atol=1e-5
            )
            for name, expected in unpadded.items():
                if not name.startswith("present."):
                    continue
                actual = padded[name]
                if name.endswith((".key", ".value")):
                    actual = actual[:, :, -expected.shape[2] :]
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        finally:
            session.close()

    def test_right_padding_preserves_state_for_cached_decode(self, tmp_path: Path) -> None:
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "kimi-linear-right-padding.gguf"
        _write_kimi_linear_gguf(path, quantized=False)
        session = OnnxModelSession(build_from_gguf(path)["model"])
        try:
            unpadded = session.run(self._inputs(np.asarray([[7, 8]], np.int64)))
            padded = session.run(
                self._inputs(
                    np.asarray([[7, 8, 0, 0]], np.int64),
                    attention_mask=np.asarray([[1, 1, 0, 0]], np.int64),
                )
            )
            for suffix in (
                "q_conv_state",
                "k_conv_state",
                "v_conv_state",
                "recurrent_state",
            ):
                np.testing.assert_allclose(
                    padded[f"present.0.{suffix}"],
                    unpadded[f"present.0.{suffix}"],
                    rtol=1e-5,
                    atol=1e-5,
                )

            unpadded_states = {
                name.replace("present.", "past_key_values."): value
                for name, value in unpadded.items()
                if name.startswith("present.")
            }
            padded_states = {
                name.replace("present.", "past_key_values."): value
                for name, value in padded.items()
                if name.startswith("present.")
            }
            token = np.asarray([[9]], np.int64)
            expected = session.run(self._inputs(token, unpadded_states))
            actual = session.run(
                self._inputs(
                    token,
                    padded_states,
                    attention_mask=np.asarray([[1, 1, 0, 0, 1]], np.int64),
                )
            )
            np.testing.assert_allclose(
                actual["logits"], expected["logits"], rtol=1e-5, atol=1e-5
            )
        finally:
            session.close()

    def test_static_cache_is_rejected(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "kimi-linear-static.gguf"
        _write_kimi_linear_gguf(path, quantized=False)
        with pytest.raises(ValueError, match="does not support static cache"):
            build_from_gguf(path, static_cache=True)

    def test_generic_task_override_is_rejected(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "kimi-linear-task.gguf"
        _write_kimi_linear_gguf(path, quantized=False)
        with pytest.raises(ValueError, match="heterogeneous-state task"):
            build_from_gguf(path, task="text-generation")

    def test_cli_build(self, tmp_path: Path) -> None:
        from mobius.__main__ import main
        from mobius._model_package import ModelPackage

        path = tmp_path / "kimi-linear-cli.gguf"
        output = tmp_path / "kimi-linear-cli-output"
        _write_kimi_linear_gguf(path, quantized=False)
        main(["build-gguf", str(path), "--output", str(output), "--dequantize"])
        package = ModelPackage.load(output)
        assert "model" in package
        assert "KDA:q_conv_state" in package["model"].metadata_props["mobius.cache_abi"]

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"omit": "blk.0.ssm_a"}, "tensor closure"),
            ({"extra": "blk.2.attn_q.weight"}, "out_of_range"),
            (
                {"malformed_shape": "blk.1.attn_k_b.weight"},
                "invalid tensor shape",
            ),
            ({"kv_heads": [0, 0]}, "requires both KDA and MLA"),
            ({"gating": 0}, "must be SIGMOID"),
            ({"conv": 1}, "inconsistent pinned architecture metadata"),
        ],
    )
    def test_invalid_contract_fails_before_graph(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        kwargs: dict[str, object],
        match: str,
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "kimi-linear-invalid.gguf"
        _write_kimi_linear_gguf(path, quantized=False, **kwargs)
        graph_build = mock.Mock(side_effect=AssertionError("graph construction reached"))
        monkeypatch.setattr(core_builder, "build_from_module", graph_build)
        with pytest.raises((TypeError, ValueError), match=match):
            build_from_gguf(path)
        graph_build.assert_not_called()


class TestGraniteHybridMoEGGUFBuild:
    """GraniteHybrid GGUF import preserves mixed state and routed expert order."""

    @staticmethod
    def _inputs(tokens: np.ndarray) -> dict[str, np.ndarray]:
        batch, sequence = tokens.shape
        return {
            "input_ids": tokens,
            "position_ids": np.broadcast_to(
                np.arange(sequence, dtype=np.int64),
                (batch, sequence),
            ).copy(),
            "attention_mask": np.ones((batch, sequence), np.int64),
            "past_key_values.0.conv_state": np.zeros((batch, 72, 3), np.float32),
            "past_key_values.0.ssm_state": np.zeros((batch, 4, 4, 16), np.float32),
            "past_key_values.1.key": np.zeros((batch, 2, 0, 8), np.float32),
            "past_key_values.1.value": np.zeros((batch, 2, 0, 8), np.float32),
        }

    def test_float_import_preserves_expert_order_and_round_trips(self, tmp_path: Path) -> None:
        from mobius._model_package import ModelPackage
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "granitehybrid-moe-f32.gguf"
        _write_granitehybrid_moe_gguf(path, quantized=False)
        package = build_from_gguf(path)
        model = package["model"]
        assert [value.name for value in model.graph.outputs] == [
            "logits",
            "present.0.conv_state",
            "present.0.ssm_state",
            "present.1.key",
            "present.1.value",
        ]
        for layer in range(2):
            fused = model.graph.initializers[
                f"model.layers.{layer}.block_sparse_moe.input_linear.weight"
            ].const_value.numpy()
            down = model.graph.initializers[
                f"model.layers.{layer}.block_sparse_moe.output_linear.weight"
            ].const_value.numpy()
            for expert in range(2):
                np.testing.assert_array_equal(fused[expert, :32], 11.0 + expert)
                np.testing.assert_array_equal(fused[expert, 32:], 21.0 + expert)
                np.testing.assert_array_equal(down[expert], 31.0 + expert)

        output_dir = tmp_path / "saved-granitehybrid"
        package.save(output_dir, progress_bar=False)
        session = OnnxModelSession(ModelPackage.load(output_dir)["model"])
        outputs = session.run(self._inputs(np.asarray([[1, 2], [3, 4]], np.int64)))
        assert outputs["logits"].shape == (2, 2, 64)
        assert outputs["present.0.conv_state"].shape == (2, 72, 3)
        assert outputs["present.0.ssm_state"].shape == (2, 4, 4, 16)

    def test_prefill_decode_reorder_rollback_and_replay(self, tmp_path: Path) -> None:
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "granitehybrid-moe-state.gguf"
        _write_granitehybrid_moe_gguf(path, quantized=False)
        session = OnnxModelSession(build_from_gguf(path)["model"])
        histories = np.asarray([[1, 2], [3, 4]], np.int64)
        prefill = session.run(self._inputs(histories))
        snapshot = {name: np.array(value, copy=True) for name, value in prefill.items()}

        order = np.asarray([1, 0], np.int64)
        next_tokens = np.asarray([[5], [6]], np.int64)
        decode_inputs = {
            "input_ids": next_tokens,
            "position_ids": np.full((2, 1), 2, np.int64),
            "attention_mask": np.ones((2, 3), np.int64),
            "past_key_values.0.conv_state": snapshot["present.0.conv_state"][order],
            "past_key_values.0.ssm_state": snapshot["present.0.ssm_state"][order],
            "past_key_values.1.key": snapshot["present.1.key"][order],
            "past_key_values.1.value": snapshot["present.1.value"][order],
        }
        decoded = session.run(decode_inputs)
        replayed = session.run(decode_inputs)
        for name in decoded:
            np.testing.assert_array_equal(decoded[name], replayed[name])

        full_tokens = np.concatenate([histories[order], next_tokens], axis=1)
        full = session.run(self._inputs(full_tokens))
        np.testing.assert_allclose(
            decoded["logits"][:, -1],
            full["logits"][:, -1],
            atol=2e-5,
            rtol=2e-5,
        )

    def test_optional_shared_expert_can_be_absent(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "granitehybrid-moe-no-shared.gguf"
        _write_granitehybrid_moe_gguf(path, quantized=False, shared_width=0)
        model = build_from_gguf(path)["model"]
        assert not any("shared_mlp" in name for name in model.graph.initializers)

    def test_attention_biases_can_be_absent(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "granitehybrid-moe-no-attention-bias.gguf"
        _write_granitehybrid_moe_gguf(path, quantized=False, attention_biases=False)
        model = build_from_gguf(path)["model"]
        assert not any(
            name.startswith("model.layers.1.self_attn.")
            and name.endswith((".q_proj.bias", ".k_proj.bias", ".v_proj.bias"))
            for name in model.graph.initializers
        )

    def test_dense_ffn_biases_are_fused_in_gate_then_up_order(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "granitehybrid-dense-biased.gguf"
        _write_granitehybrid_moe_gguf(
            path,
            quantized=False,
            expert_count=0,
            expert_used_count=0,
            shared_width=0,
            mlp_biases=True,
        )

        model = build_from_gguf(path)["model"]
        for layer in range(2):
            prefix = f"model.layers.{layer}.shared_mlp."
            fused_bias = model.graph.initializers[
                prefix + "input_linear.bias"
            ].const_value.numpy()
            np.testing.assert_array_equal(fused_bias[:32], 41.0)
            np.testing.assert_array_equal(fused_bias[32:], 42.0)
            np.testing.assert_array_equal(
                model.graph.initializers[prefix + "output_linear.bias"].const_value.numpy(),
                43.0,
            )

    def test_quantized_source_requires_explicit_dequantization(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "granitehybrid-moe-q4.gguf"
        _write_granitehybrid_moe_gguf(path, quantized=True)
        with pytest.raises(ValueError, match=r"keep_quantized=False"):
            build_from_gguf(path, keep_quantized=True)

        model = build_from_gguf(path, keep_quantized=False)["model"]
        assert all(node.op_type != "MatMulNBits" for node in model.graph)
        assert model.graph.initializers[
            "model.layers.0.block_sparse_moe.input_linear.weight"
        ].shape == [2, 64, 32]

    def test_cli_builds_dequantized_package(self, tmp_path: Path) -> None:
        from mobius.__main__ import main
        from mobius._model_package import ModelPackage

        path = tmp_path / "granitehybrid-moe-cli.gguf"
        output_dir = tmp_path / "cli-output"
        _write_granitehybrid_moe_gguf(path, quantized=False)

        main(["build-gguf", str(path), "--output", str(output_dir), "--dequantize"])

        package = ModelPackage.load(output_dir)
        assert set(package) == {"model"}
        assert (output_dir / "model.onnx").is_file()

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"omit": "blk.1.ffn_up_exps.weight"}, "tensor closure"),
            ({"extra": "blk.0.ffn_gate.weight"}, "mixes dense and routed"),
            (
                {"malformed_shape": "blk.1.ffn_gate_exps.weight"},
                "tensor shape",
            ),
            ({"expert_count": 0, "expert_used_count": 1}, "both be zero or both positive"),
            ({"expert_count": 1, "expert_used_count": 1}, "not a routed-MoE"),
            ({"expert_count": 2, "expert_used_count": 3}, "must be in"),
            ({"omit": "blk.1.attn_q.bias"}, "partial attention Q/K/V projection"),
            ({"extra": "blk.0.ffn_gate_up_exps.weight"}, "unexpected|Malformed"),
            ({"extra": "blk.0.ffn_gate_exps.scale"}, "auxiliary quantization"),
            ({"extra": "blk.2.ffn_gate_inp.weight"}, "out_of_range"),
        ],
    )
    def test_malformed_sources_fail_before_graph(
        self,
        tmp_path: Path,
        monkeypatch,
        kwargs: dict[str, object],
        match: str,
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "granitehybrid-moe-invalid.gguf"
        _write_granitehybrid_moe_gguf(path, quantized=False, **kwargs)
        graph_build = mock.Mock(side_effect=AssertionError("graph construction reached"))
        monkeypatch.setattr(core_builder, "build_from_module", graph_build)
        with pytest.raises(ValueError, match=match):
            build_from_gguf(path)
        graph_build.assert_not_called()


class TestNemotronHMoEGGUFBuild:
    """Nemotron-H GGUF import preserves hybrid state and exact expert semantics."""

    @staticmethod
    def _inputs(tokens: np.ndarray) -> dict[str, np.ndarray]:
        batch, sequence = tokens.shape
        return {
            "input_ids": tokens,
            "position_ids": np.broadcast_to(
                np.arange(sequence, dtype=np.int64),
                (batch, sequence),
            ).copy(),
            "attention_mask": np.ones((batch, sequence), np.int64),
            "past_key_values.0.conv_state": np.zeros((batch, 72, 3), np.float32),
            "past_key_values.0.ssm_state": np.zeros((batch, 4, 4, 16), np.float32),
            "past_key_values.2.key": np.zeros((batch, 1, 0, 16), np.float32),
            "past_key_values.2.value": np.zeros((batch, 1, 0, 16), np.float32),
        }

    def test_float_import_preserves_expert_order_state_and_roundtrip(
        self, tmp_path: Path
    ) -> None:
        from mobius._model_package import ModelPackage
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "nemotron-h-moe-f32.gguf"
        _write_nemotron_h_moe_gguf(path, quantized=False)
        package = build_from_gguf(path)
        model = package["model"]
        runtime_support = model.metadata_props["mobius.runtime_support"]
        assert "discovers sparse KV and conv/recurrent slots" in runtime_support
        assert "derives recurrent_state where Mobius exports ssm_state" in runtime_support
        assert "does not beam-reorder recurrent state" in runtime_support
        assert "rejects nonzero recurrent-state rewind" in runtime_support
        assert [value.name for value in model.graph.outputs] == [
            "logits",
            "present.0.conv_state",
            "present.0.ssm_state",
            "present.2.key",
            "present.2.value",
        ]
        for expert in range(2):
            for projection in ("up_proj", "down_proj"):
                value = model.graph.initializers[
                    f"model.layers.1.moe.experts.{expert}.{projection}.weight_t"
                ].const_value.numpy()
                np.testing.assert_array_equal(value, expert + 1)
        np.testing.assert_array_equal(
            model.graph.initializers[
                "model.layers.2.self_attn.q_proj.weight_t"
            ].const_value.numpy(),
            np.arange(32 * 32, dtype=np.float32).reshape(32, 32).T,
        )
        np.testing.assert_array_equal(
            model.graph.initializers[
                "model.layers.2.self_attn.k_proj.weight_t"
            ].const_value.numpy(),
            np.arange(16 * 32, dtype=np.float32).reshape(16, 32).T,
        )

        output_dir = tmp_path / "saved-nemotron-h-moe"
        package.save(output_dir, progress_bar=False)
        session = OnnxModelSession(ModelPackage.load(output_dir)["model"])
        outputs = session.run(self._inputs(np.asarray([[1, 2], [3, 4]], np.int64)))
        assert outputs["logits"].shape == (2, 2, 64)
        assert outputs["present.0.conv_state"].shape == (2, 72, 3)
        assert outputs["present.0.ssm_state"].shape == (2, 4, 4, 16)
        session.close()

    def test_runtime_package_exports_with_unvalidated_state_contract(
        self, tmp_path: Path
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf, write_gguf_runtime_package

        path = tmp_path / "nemotron-h-moe-f32.gguf"
        _write_nemotron_h_moe_gguf(path, quantized=False)
        package = build_from_gguf(path)

        output = tmp_path / "runtime-package"
        artifacts = write_gguf_runtime_package(
            package,
            path,
            output,
            tokenizer_repository="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            tokenizer_revision="bf77c3174f68ad409e1c2aa60daeb46e32d1c606",
            runtime_version="0.15.2",
        )
        compatibility = json.loads(
            (output / "runtime_compatibility.json").read_text(encoding="utf-8")
        )
        assert compatibility["runtime_validation_status"] == "unvalidated"
        assert compatibility["gguf_architecture"] == "nemotron_h_moe"
        assert (output / "model.onnx").is_file()
        assert Path(artifacts["export_report"]).is_file()
        report = package.export_report
        assert report is not None
        model = report.component("model")
        runtime = report.component("runtime")
        assert model is not None
        assert runtime is not None
        assert model.output == "exported"
        assert runtime.output == "exported"
        assert runtime.runtime_validation_status == "unvalidated"
        assert runtime.blocker_category == "runtime-route-deferred"
        assert not (output / "tokenizer.json").exists()

    def test_runtime_unvalidated_fallback_rejects_source_mutation(
        self, tmp_path: Path
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf, write_gguf_runtime_package

        path = tmp_path / "nemotron-h-moe-mutated.gguf"
        _write_nemotron_h_moe_gguf(path, quantized=False)
        package = build_from_gguf(path)
        with path.open("r+b") as stream:
            stream.seek(-1, os.SEEK_END)
            value = stream.read(1)
            stream.seek(-1, os.SEEK_END)
            stream.write(bytes([value[0] ^ 0xFF]))

        output = tmp_path / "runtime-mutated"
        with pytest.raises(ValueError, match="exact artifact identity"):
            write_gguf_runtime_package(
                package,
                path,
                output,
                runtime_version="0.15.2",
            )
        assert not output.exists()
        assert not output.exists()

    def test_latent_projection_imports_and_executes(self, tmp_path: Path) -> None:
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "nemotron-h-moe-latent.gguf"
        _write_nemotron_h_moe_gguf(path, quantized=False, latent=True)
        model = build_from_gguf(path)["model"]
        assert model.graph.initializers[
            "model.layers.1.moe.fc1_latent_proj.weight_t"
        ].shape == [32, 32]
        session = OnnxModelSession(model)
        outputs = session.run(self._inputs(np.asarray([[1, 2]], np.int64)))
        assert outputs["logits"].shape == (1, 2, 64)
        session.close()

    def test_prefill_decode_threads_all_state_families(self, tmp_path: Path) -> None:
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "nemotron-h-moe-state.gguf"
        _write_nemotron_h_moe_gguf(path, quantized=False)
        session = OnnxModelSession(build_from_gguf(path)["model"])
        histories = np.asarray([[1, 2], [3, 4]], np.int64)
        prefill = session.run(self._inputs(histories))
        order = np.asarray([1, 0], np.int64)
        next_tokens = np.asarray([[5], [6]], np.int64)
        decode_inputs = {
            "input_ids": next_tokens,
            "position_ids": np.full((2, 1), 2, np.int64),
            "attention_mask": np.ones((2, 3), np.int64),
            "past_key_values.0.conv_state": prefill["present.0.conv_state"][order],
            "past_key_values.0.ssm_state": prefill["present.0.ssm_state"][order],
            "past_key_values.2.key": prefill["present.2.key"][order],
            "past_key_values.2.value": prefill["present.2.value"][order],
        }
        decoded = session.run(decode_inputs)
        replayed = session.run(decode_inputs)
        for name in decoded:
            np.testing.assert_array_equal(decoded[name], replayed[name])

        full_tokens = np.concatenate([histories[order], next_tokens], axis=1)
        full = session.run(self._inputs(full_tokens))
        np.testing.assert_allclose(
            decoded["logits"][:, -1],
            full["logits"][:, -1],
            atol=2e-5,
            rtol=2e-5,
        )
        session.close()

    def test_quantized_source_requires_explicit_dequantization(self, tmp_path: Path) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "nemotron-h-moe-q4.gguf"
        _write_nemotron_h_moe_gguf(path, quantized=True)
        with pytest.raises(ValueError, match=r"keep_quantized=False"):
            build_from_gguf(path, keep_quantized=True)

        model = build_from_gguf(path, keep_quantized=False)["model"]
        assert all(node.op_type != "MatMulNBits" for node in model.graph)
        assert (
            model.graph.initializers[
                "model.layers.1.moe.experts.0.up_proj.weight_t"
            ].const_value.dtype
            == ir.DataType.FLOAT
        )

    def test_quantized_expert_only_requires_explicit_dequantization(
        self, tmp_path: Path
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "nemotron-h-moe-expert-q4.gguf"
        _write_nemotron_h_moe_gguf(
            path,
            quantized=True,
            quantized_only="blk.1.ffn_up_exps.weight",
        )
        with pytest.raises(ValueError, match=r"keep_quantized=False"):
            build_from_gguf(path, keep_quantized=True)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"omit": "blk.1.ffn_up_exps.weight"}, "tensor closure"),
            ({"omit": "blk.1.exp_probs_b.bias"}, "tensor closure"),
            ({"extra": "blk.1.ffn_exp_probs_b.bias"}, "tensor closure"),
            ({"extra": "blk.1.ffn_up_exps.scale"}, "auxiliary quantization"),
            (
                {"latent": True, "omit": "blk.1.ffn_latent_up.weight"},
                "latent projection",
            ),
            (
                {"malformed_shape": "blk.1.ffn_down_exps.weight"},
                "tensor shape",
            ),
        ],
    )
    def test_malformed_sources_fail_before_graph(
        self,
        tmp_path: Path,
        monkeypatch,
        kwargs: dict[str, object],
        match: str,
    ) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "nemotron-h-moe-invalid.gguf"
        _write_nemotron_h_moe_gguf(path, quantized=False, **kwargs)
        graph_build = mock.Mock(side_effect=AssertionError("graph construction reached"))
        monkeypatch.setattr(core_builder, "build_from_module", graph_build)
        with pytest.raises(ValueError, match=match):
            build_from_gguf(path)
        graph_build.assert_not_called()

    def test_mtp_sidecar_fails_before_graph(self, tmp_path: Path, monkeypatch) -> None:
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "nemotron-h-moe-mtp.gguf"
        _write_nemotron_h_moe_gguf(path, quantized=False, mtp=True)
        graph_build = mock.Mock(side_effect=AssertionError("graph construction reached"))
        monkeypatch.setattr(core_builder, "build_from_module", graph_build)
        with pytest.raises(NotImplementedError, match="routed experts"):
            build_from_gguf(path)
        graph_build.assert_not_called()

    @pytest.mark.parametrize(
        ("options", "match"),
        [
            ({"static_cache": True}, "static_cache=True"),
            ({"task": "text-generation"}, "hybrid-text-generation"),
        ],
    )
    def test_incompatible_task_options_fail_closed(
        self,
        tmp_path: Path,
        options: dict[str, object],
        match: str,
    ) -> None:
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "nemotron-h-moe-options.gguf"
        _write_nemotron_h_moe_gguf(path, quantized=False)
        with pytest.raises(ValueError, match=match):
            build_from_gguf(path, **options)
