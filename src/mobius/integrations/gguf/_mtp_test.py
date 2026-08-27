# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the Qwen3.5/3.8 MTP (multi-token-prediction) head GGUF export."""

from __future__ import annotations

import json
import re
import typing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from mobius.integrations.gguf._mtp import (
    build_mtp_head_from_gguf,
    derive_mtp_config,
    has_mtp_head,
    map_gguf_mtp_to_hf_names,
    mtp_architecture_capabilities,
    validate_mtp_tensor_contract,
)
from mobius.integrations.gguf._spec import Support

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


@dataclass
class _ContractGGUF:
    architecture: str
    metadata: dict[str, object]
    tensor_names: list[str]


def _write_qwen35_mtp_gguf(
    path: Path,
    *,
    architecture: str = "qwen35",
    quantized: bool = False,
    mtp_count: int = 1,
    mtp_aux_suffix: str | None = None,
    dedicated_embedding: bool = False,
    dedicated_norm: bool = True,
    dedicated_head: bool = False,
    tied_output: bool = True,
    unknown_nextn: bool = False,
    quantized_backbone_embedding: bool = False,
    nextn_count: int | None = None,
    mtp_block_index: int = 1,
    omit_nextn_stem: str | None = None,
    metadata_key: str | None = None,
    extra_mtp_tensor: str | None = None,
    asymmetric_dedicated_head: bool = False,
    shape_overrides: dict[str, tuple[int, ...]] | None = None,
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
    writer.add_key_value(
        metadata_key or f"{architecture}.nextn_predict_layers",
        mtp_count if nextn_count is None else nextn_count,
        4,
    )
    writer.add_key_value(f"{architecture}.attention.key_length", _HD, 4)
    writer.add_array(
        f"{architecture}.attention.recurrent_layers",
        [False] * (1 + mtp_count),
    )
    writer.add_array(f"{architecture}.rope.dimension_sections", [4, 4, 0, 0])
    writer.add_ssm_conv_kernel(3)
    writer.add_ssm_inner_size(32)
    writer.add_ssm_state_size(16)
    writer.add_ssm_time_step_rank(2)
    writer.add_ssm_group_count(2)

    def f32(name: str, shape: tuple[int, ...]) -> None:
        if shape_overrides is not None:
            shape = shape_overrides.get(name, shape)
        writer.add_tensor(name, np.random.randn(*shape).astype(np.float32))

    def matrix(name: str, shape: tuple[int, ...], *, asymmetric: bool = False) -> None:
        if shape_overrides is not None:
            shape = shape_overrides.get(name, shape)
        if not quantized:
            f32(name, shape)
            return
        rows = int(np.prod(shape[:-1]))
        k_in = shape[-1]
        assert k_in % 32 == 0
        block_size = 20 if asymmetric else 18
        raw = np.zeros((rows, (k_in // 32) * block_size), dtype=np.uint8)
        if asymmetric:
            # Q4_1: fp16 scale, fp16 minimum, then 16 packed 4-bit pairs.
            scale = np.float16(0.25).tobytes()
            minimum = np.float16(-1.0).tobytes()
            for offset in range(0, raw.shape[1], block_size):
                raw[:, offset : offset + 2] = np.frombuffer(scale, dtype=np.uint8)
                raw[:, offset + 2 : offset + 4] = np.frombuffer(minimum, dtype=np.uint8)
                raw[:, offset + 4 : offset + block_size] = 0xD2
        else:
            raw[:, 0::block_size] = 0x00
            raw[:, 1::block_size] = 0x3C
        writer.add_tensor(
            name,
            raw.reshape(*shape[:-1], -1),
            raw_dtype=(GGMLQuantizationType.Q4_1 if asymmetric else GGMLQuantizationType.Q4_0),
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
    block_indices = range(1, 1 + mtp_count) if mtp_count > 1 else (mtp_block_index,)
    for index in block_indices:
        prefix = f"blk.{index}"
        if omit_nextn_stem != "eh_proj":
            matrix(f"{prefix}.nextn.eh_proj.weight", (_H, 2 * _H))
        if mtp_aux_suffix is not None:
            assert mtp_aux_suffix in {"scale", "input_scale"}
            f32(f"{prefix}.nextn.eh_proj.{mtp_aux_suffix}", (1,))
        if omit_nextn_stem != "enorm":
            f32(f"{prefix}.nextn.enorm.weight", (_H,))
        if omit_nextn_stem != "hnorm":
            f32(f"{prefix}.nextn.hnorm.weight", (_H,))
        if dedicated_embedding:
            matrix(f"{prefix}.nextn.embed_tokens.weight", (_VOCAB, _H))
        if dedicated_norm:
            f32(f"{prefix}.nextn.shared_head_norm.weight", (_H,))
        if unknown_nextn:
            f32(f"{prefix}.nextn.unknown.weight", (_H,))
        if extra_mtp_tensor is not None:
            f32(extra_mtp_tensor, (_H,))
        _add_decoder_block(index)
        if dedicated_head:
            matrix(
                f"{prefix}.nextn.shared_head_head.weight",
                (_VOCAB, _H),
                asymmetric=asymmetric_dedicated_head,
            )

    if quantized_backbone_embedding:
        matrix("token_embd.weight", (_VOCAB, _H))
    else:
        f32("token_embd.weight", (_VOCAB, _H))
    f32("output_norm.weight", (_H,))
    if not tied_output:
        matrix("output.weight", (_VOCAB, _H))

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
        assert (
            map_gguf_mtp_to_hf_names("blk.1.nextn.embed_tokens.weight", 1)
            == "embed_tokens.weight"
        )
        assert (
            map_gguf_mtp_to_hf_names("blk.1.nextn.shared_head_head.weight", 1)
            == "lm_head.weight"
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


class TestPinnedMtpCensus:
    def test_policy_closes_exact_pinned_loader_converter_union(self):
        pin_path = Path(__file__).parent / "_upstream_data" / "llamacpp_pin.json"
        census = json.loads(pin_path.read_text(encoding="utf-8"))["mtp_nextn"]
        upstream_union = set().union(
            census["loader_metadata_consumers"],
            census["loader_executed_sidecars"],
            census["loader_preserved_or_skipped"],
            census["standalone_assistant"],
            census["converter_metadata_emitters"],
            census["non_mtp_metadata_reemitters"],
        )
        capabilities = mtp_architecture_capabilities()

        assert set(capabilities) == upstream_union
        assert {
            architecture
            for architecture, capability in capabilities.items()
            if capability.support is Support.SUPPORTED
        } == {"hy_v3", "qwen35"}
        assert census["mobius_supported_sidecars"] == ["hy_v3", "qwen35"]
        assert census["mobius_runtime_status"] == "DEFERRED"

    def test_exact_pinned_metadata_and_tensor_names(self):
        pin_path = Path(__file__).parent / "_upstream_data" / "llamacpp_pin.json"
        census = json.loads(pin_path.read_text(encoding="utf-8"))["mtp_nextn"]

        assert census["metadata_key_template"] == "{arch}.nextn_predict_layers"
        assert census["metadata_type"] == "uint32"
        assert census["modern_tensor_names"] == [
            "blk.{bid}.nextn.eh_proj.weight",
            "blk.{bid}.nextn.embed_tokens.weight",
            "blk.{bid}.nextn.enorm.weight",
            "blk.{bid}.nextn.hnorm.weight",
            "blk.{bid}.nextn.shared_head_head.weight",
            "blk.{bid}.nextn.shared_head_norm.weight",
        ]
        assert census["legacy_tensor_names"] == [
            "nextn.pre_projection.weight",
            "nextn.post_projection.weight",
        ]
        assert census["generic_loader_extra_suffixes"] == ["scale", "input_scale"]

    @pytest.mark.parametrize(
        "architecture",
        sorted(
            architecture
            for architecture, capability in mtp_architecture_capabilities().items()
            if capability.support is not Support.SUPPORTED
        ),
    )
    def test_every_unsupported_pinned_architecture_rejects_specifically(
        self, architecture: str
    ):
        model = _ContractGGUF(
            architecture=architecture,
            metadata={
                f"{architecture}.nextn_predict_layers": 1,
                f"{architecture}.block_count": 2,
            },
            tensor_names=[
                "blk.1.nextn.eh_proj.weight",
                "blk.1.nextn.enorm.weight",
                "blk.1.nextn.hnorm.weight",
            ],
        )

        with pytest.raises(NotImplementedError, match=rf"^{re.escape(architecture)} GGUF MTP"):
            validate_mtp_tensor_contract(model)

    @pytest.mark.parametrize("invalid_count", [True, -1, 1.5, "1"])
    def test_head_count_metadata_requires_nonnegative_integer(self, invalid_count):
        model = _ContractGGUF(
            architecture="qwen35",
            metadata={
                "qwen35.nextn_predict_layers": invalid_count,
                "qwen35.block_count": 2,
            },
            tensor_names=[],
        )

        with pytest.raises(
            (TypeError, ValueError), match=r"must be (an integer|non-negative)"
        ):
            validate_mtp_tensor_contract(model)

    def test_tensor_without_positive_metadata_is_rejected(self):
        model = _ContractGGUF(
            architecture="qwen35",
            metadata={"qwen35.block_count": 2},
            tensor_names=["blk.1.nextn.eh_proj.weight"],
        )

        with pytest.raises(ValueError, match="does not declare a positive"):
            validate_mtp_tensor_contract(model)

    def test_foreign_architecture_metadata_namespace_is_rejected(self):
        model = _ContractGGUF(
            architecture="qwen35",
            metadata={
                "qwen35.block_count": 2,
                "qwen35moe.nextn_predict_layers": 1,
            },
            tensor_names=[],
        )

        with pytest.raises(ValueError, match="unsupported MTP metadata key"):
            validate_mtp_tensor_contract(model)


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
            match=r"nextn_predict_layers=2.*exactly one appended MTP block",
        ):
            build_from_gguf(str(path))
        assert built is False

    @pytest.mark.parametrize("dedicated_embedding", [False, True])
    @pytest.mark.parametrize("dedicated_norm", [False, True])
    @pytest.mark.parametrize("dedicated_head", [False, True])
    @pytest.mark.parametrize("tied_output", [False, True])
    @pytest.mark.parametrize("quantized", [False, True])
    def test_optional_tables_use_exact_dedicated_or_fallback_contract(
        self,
        tmp_path: Path,
        dedicated_embedding: bool,
        dedicated_norm: bool,
        dedicated_head: bool,
        tied_output: bool,
        quantized: bool,
    ) -> None:
        from mobius.integrations.gguf import QuantizationDisposition, build_from_gguf
        from mobius.integrations.gguf._reader import GGUFModel

        path = tmp_path / "mtp-tables.gguf"
        _write_qwen35_mtp_gguf(
            path,
            quantized=quantized,
            dedicated_embedding=dedicated_embedding,
            dedicated_norm=dedicated_norm,
            dedicated_head=dedicated_head,
            tied_output=tied_output,
        )
        preserve_quantization = quantized
        package = build_from_gguf(path, keep_quantized=preserve_quantization)
        if quantized:
            records = {
                record.name: record
                for record in package.gguf_quantization_report.tensor_records
            }
            if dedicated_embedding:
                assert (
                    records["blk.1.nextn.embed_tokens.weight"].disposition
                    is QuantizationDisposition.DEQUANTIZED_FLOAT
                )
            if dedicated_head:
                assert (
                    records["blk.1.nextn.shared_head_head.weight"].disposition
                    is QuantizationDisposition.DEQUANTIZED_FLOAT
                )
        head = package.mtp_head
        assert head is not None
        target_initializers = package["model"].graph.initializers
        assert "model.embed_tokens.weight" in target_initializers
        if tied_output:
            assert not any(name.startswith("lm_head.") for name in target_initializers)
        elif preserve_quantization:
            assert {
                "lm_head.weight",
                "lm_head.scales",
                "lm_head.zero_points",
            }.issubset(target_initializers)
        else:
            assert "lm_head.weight_t" in target_initializers
        model = head["model"]
        inputs = {value.name for value in model.graph.inputs}
        outputs = {value.name for value in model.graph.outputs}
        initializers = model.graph.initializers

        assert ("input_ids" in inputs) is dedicated_embedding
        assert ("inputs_embeds" in inputs) is not dedicated_embedding
        assert ("embed_tokens.weight" in initializers) is dedicated_embedding
        assert ("logits" in outputs) is dedicated_head
        assert ("mtp_hidden" in outputs) is not dedicated_head
        assert ("lm_head.weight_t" in initializers) is dedicated_head
        assert "norm.weight" in initializers
        assert all(value.const_value is not None for value in initializers.values())
        norm_source = (
            "blk.1.nextn.shared_head_norm.weight" if dedicated_norm else "output_norm.weight"
        )
        expected_norm = GGUFModel(path).get_tensor(norm_source) - 1.0
        np.testing.assert_allclose(
            initializers["norm.weight"].const_value.numpy(),
            expected_norm,
            rtol=0,
            atol=0,
        )

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

    def test_tied_quantized_target_head_uses_embedding_initializers(self, tmp_path):
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "tied-quantized-tables.gguf"
        _write_qwen35_mtp_gguf(
            path,
            quantized=True,
            tied_output=True,
            quantized_backbone_embedding=True,
        )
        package = build_from_gguf(path, keep_quantized=True)
        assert package.config.tie_word_embeddings
        assert package.config.quantization is not None
        assert package.config.quantization.quantize_embeddings
        assert package.config.quantization.quantize_lm_head
        initializers = package["model"].graph.initializers
        assert {
            "model.embed_tokens.qweight",
            "model.embed_tokens.scales",
            "model.embed_tokens.zero_points",
        }.issubset(initializers)
        assert not any(name.startswith("lm_head.") for name in initializers)

    def test_target_and_sidecar_save_with_weight_checks_and_reload(
        self, qwen35_mtp_gguf: Path, tmp_path: Path
    ):
        from mobius._model_package import ModelPackage
        from mobius.integrations.gguf import build_from_gguf

        package = build_from_gguf(qwen35_mtp_gguf)
        target_dir = tmp_path / "target"
        package.save(target_dir, progress_bar=False, check_weights=True)
        reloaded = ModelPackage.load(target_dir)
        target = reloaded["model"]
        assert reloaded.mtp_head is not None
        sidecar = reloaded.mtp_head["model"]
        assert all(
            value.const_value is not None for value in target.graph.initializers.values()
        )
        assert all(
            value.const_value is not None for value in sidecar.graph.initializers.values()
        )
        assert "model.embed_tokens.weight" in target.graph.initializers
        assert "embed_tokens.weight" not in sidecar.graph.initializers
        assert not any("nextn" in name for name in target.graph.initializers)

    def test_asymmetric_dedicated_head_is_reported_as_float(self, tmp_path: Path):
        import onnxruntime as ort

        from mobius.integrations.gguf import build_from_gguf
        from mobius.integrations.gguf._reader import GGUFModel

        shared_path = tmp_path / "shared-head-q4_1.gguf"
        dedicated_path = tmp_path / "dedicated-head-q4_1.gguf"
        np.random.seed(7)
        _write_qwen35_mtp_gguf(shared_path, quantized=True)
        np.random.seed(7)
        _write_qwen35_mtp_gguf(
            dedicated_path,
            quantized=True,
            dedicated_head=True,
            asymmetric_dedicated_head=True,
        )
        shared = build_from_gguf(shared_path, keep_quantized=True).mtp_head
        dedicated_package = build_from_gguf(dedicated_path, keep_quantized=True)
        dedicated = dedicated_package.mtp_head
        assert dedicated_package.gguf_quantization_report.source_fidelity is False
        assert any(
            record.name == "blk.1.nextn.shared_head_head.weight"
            for record in dedicated_package.gguf_quantization_report.explicit_float_tensors
        )
        shared_dir = tmp_path / "shared"
        dedicated_dir = tmp_path / "dedicated"
        shared.save(shared_dir, progress_bar=False, check_weights=True)
        dedicated.save(dedicated_dir, progress_bar=False, check_weights=True)

        rng = np.random.default_rng(11)
        feeds = {
            "inputs_embeds": rng.standard_normal((1, 2, _H), dtype=np.float32),
            "hidden_states": rng.standard_normal((1, 2, _H), dtype=np.float32),
            "attention_mask": np.ones((1, 2), dtype=np.int64),
            "position_ids": np.array([[0, 1]], dtype=np.int64),
            "past_key_values.0.key": np.empty((1, _NKV, 0, _HD), dtype=np.float32),
            "past_key_values.0.value": np.empty((1, _NKV, 0, _HD), dtype=np.float32),
        }
        shared_session = ort.InferenceSession(
            str(shared_dir / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        dedicated_session = ort.InferenceSession(
            str(dedicated_dir / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        mtp_hidden = shared_session.run(["mtp_hidden"], feeds)[0]
        logits = dedicated_session.run(["logits"], feeds)[0]

        # The dedicated Q4_1 table is intentionally asymmetric. The sidecar
        # dequantizes it for a mathematically ordinary MatMul.
        explicit_table = GGUFModel(dedicated_path).get_tensor(
            "blk.1.nextn.shared_head_head.weight"
        )
        assert abs(float(explicit_table.mean())) > 0.1
        expected = mtp_hidden @ explicit_table.T
        np.testing.assert_allclose(logits, expected, rtol=1e-5, atol=1e-5)


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
        # The ordinary layer capture remains pre-final-norm, while MTP gets a
        # dedicated post-final-norm seed.
        seed = pkg.config.num_hidden_layers - 1
        main_outputs = {v.name for v in pkg["model"].graph.outputs}
        assert f"hidden_states.{seed}" in main_outputs
        assert "mtp_seed" in main_outputs

    def test_mtp_seed_is_post_final_norm(self, qwen35_mtp_gguf: Path, tmp_path: Path):
        import onnxruntime as ort

        from mobius.integrations.gguf import build_from_gguf
        from mobius.integrations.gguf._reader import GGUFModel

        pkg = build_from_gguf(qwen35_mtp_gguf, keep_quantized=False)
        output_dir = tmp_path / "model"
        pkg.save(output_dir, progress_bar=False)
        session = ort.InferenceSession(
            str(output_dir / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        pre_norm, mtp_seed = session.run(
            ["hidden_states.0", "mtp_seed"],
            {
                "input_ids": np.array([[1, 2]], dtype=np.int64),
                "attention_mask": np.ones((1, 2), dtype=np.int64),
                "position_ids": np.array([[0, 1]], dtype=np.int64),
                "past_key_values.0.key": np.empty((1, _NKV, 0, _HD), dtype=np.float32),
                "past_key_values.0.value": np.empty((1, _NKV, 0, _HD), dtype=np.float32),
            },
        )
        norm_weight = GGUFModel(qwen35_mtp_gguf).get_tensor("output_norm.weight")
        expected = (
            pre_norm
            * norm_weight
            / np.sqrt(np.mean(np.square(pre_norm), axis=-1, keepdims=True) + 1e-5)
        )
        np.testing.assert_allclose(mtp_seed, expected, rtol=1e-5, atol=1e-5)
        assert not np.allclose(mtp_seed, pre_norm)

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

    def test_unknown_nextn_tensor_is_rejected_before_graph_build(
        self, tmp_path: Path, monkeypatch
    ):
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "unknown-nextn.gguf"
        _write_qwen35_mtp_gguf(path, unknown_nextn=True)
        monkeypatch.setattr(
            core_builder,
            "build_from_module",
            lambda *_args, **_kwargs: pytest.fail("graph construction must not run"),
        )

        with pytest.raises(ValueError, match="unsupported nextn tensor"):
            build_from_gguf(path)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"nextn_count": 2}, r"nextn_predict_layers=2"),
            ({"omit_nextn_stem": "hnorm"}, "missing required tensor stem"),
            ({"mtp_block_index": 2}, "trailing block 1"),
            (
                {"metadata_key": "qwen35.nextn.predict_layers"},
                "unsupported MTP metadata key",
            ),
            (
                {"extra_mtp_tensor": "nextn.pre_projection.weight"},
                "legacy global NextN tensors",
            ),
            ({"extra_mtp_tensor": "mtp.unknown.weight"}, "unsupported nextn tensor"),
            (
                {"extra_mtp_tensor": "blk.1.ssm_a"},
                "unsupported tensor",
            ),
            (
                {"extra_mtp_tensor": "blk.1.ffn_gate_exps.weight"},
                "unsupported tensor",
            ),
            (
                {"extra_mtp_tensor": "blk.1.attn_qkv.weight"},
                "unsupported tensor",
            ),
            (
                {"extra_mtp_tensor": "blk.1.unknown_state.weight"},
                "unsupported tensor",
            ),
        ],
    )
    def test_malformed_mtp_contracts_fail_before_graph_build(
        self,
        kwargs: dict[str, object],
        message: str,
        tmp_path: Path,
        monkeypatch,
    ):
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "malformed-mtp.gguf"
        _write_qwen35_mtp_gguf(path, **kwargs)
        monkeypatch.setattr(
            core_builder,
            "build_from_module",
            lambda *_args, **_kwargs: pytest.fail("graph construction must not run"),
        )

        with pytest.raises(ValueError, match=message):
            build_from_gguf(path)

    @pytest.mark.parametrize(
        ("tensor_name", "wrong_shape", "table_kwargs"),
        [
            ("blk.1.nextn.eh_proj.weight", (_H, _H), {}),
            ("blk.1.nextn.enorm.weight", (_H + 1,), {}),
            (
                "blk.1.nextn.embed_tokens.weight",
                (_VOCAB + 1, _H),
                {"dedicated_embedding": True},
            ),
            (
                "blk.1.nextn.shared_head_norm.weight",
                (_H + 1,),
                {"dedicated_norm": True},
            ),
            (
                "blk.1.nextn.shared_head_head.weight",
                (_VOCAB + 1, _H),
                {"dedicated_head": True},
            ),
            ("blk.1.attn_q.weight", (_NH * _HD, _H), {}),
            ("blk.1.ffn_down.weight", (_H, _FFN + 1), {}),
        ],
    )
    def test_malformed_mtp_shapes_fail_before_graph_build(
        self,
        tensor_name: str,
        wrong_shape: tuple[int, ...],
        table_kwargs: dict[str, bool],
        tmp_path: Path,
        monkeypatch,
    ):
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        path = tmp_path / "malformed-mtp-shape.gguf"
        _write_qwen35_mtp_gguf(
            path,
            shape_overrides={tensor_name: wrong_shape},
            **table_kwargs,
        )
        monkeypatch.setattr(
            core_builder,
            "build_from_module",
            lambda *_args, **_kwargs: pytest.fail("graph construction must not run"),
        )

        with pytest.raises(
            ValueError,
            match=rf"Invalid qwen35 GGUF MTP tensor shapes.*{re.escape(tensor_name)}",
        ):
            build_from_gguf(path)

    def test_static_cache_rejects_instead_of_omitting_sidecar(
        self, qwen35_mtp_gguf: Path, monkeypatch
    ):
        from mobius import _builder as core_builder
        from mobius.integrations.gguf import build_from_gguf

        monkeypatch.setattr(
            core_builder,
            "build_from_module",
            lambda *_args, **_kwargs: pytest.fail("graph construction must not run"),
        )

        with pytest.raises(ValueError, match=r"static_cache=True.*refusing to silently omit"):
            build_from_gguf(qwen35_mtp_gguf, static_cache=True)

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
