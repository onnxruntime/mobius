# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for GGUF import support.

Creates synthetic GGUF files using the ``gguf`` package's writer to
test the full pipeline without network downloads.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

try:
    import gguf as gguf_lib

    HAS_GGUF = True
except ImportError:
    HAS_GGUF = False

pytestmark = pytest.mark.skipif(not HAS_GGUF, reason="gguf not installed")


def _create_tiny_gguf(
    path,
    arch: str = "llama",
    hidden: int = 64,
    layers: int = 2,
    heads: int = 2,
    vocab: int = 256,
    ffn_size: int | None = None,
    float_type: str = "f32",
    tokenizer_pre: str | None = None,
) -> str:
    """Create a minimal GGUF file with fp32 weights for testing.

    Returns the string path to the created file.
    """
    if ffn_size is None:
        ffn_size = hidden * 4
    path = str(path)
    writer = gguf_lib.GGUFWriter(path, arch)

    # Metadata
    writer.add_block_count(layers)
    writer.add_embedding_length(hidden)
    writer.add_head_count(heads)
    writer.add_head_count_kv(heads)
    writer.add_context_length(128)
    writer.add_feed_forward_length(ffn_size)
    writer.add_vocab_size(vocab)
    if tokenizer_pre is not None:
        tokens = [f"token-{index}" for index in range(vocab)]
        writer.add_tokenizer_model("gpt2")
        writer.add_string("tokenizer.ggml.pre", tokenizer_pre)
        writer.add_token_list(tokens)
        writer.add_token_merges([f"{tokens[0]} {tokens[1]}"])

    # Tensors — unquantized random weights
    rng = np.random.default_rng(42)

    def add_float(name, shape):
        values = rng.standard_normal(shape, dtype=np.float32)
        if float_type == "bf16":
            raw = (values.view(np.uint32) >> 16).astype(np.uint16)
            writer.add_tensor(
                name,
                raw,
                raw_shape=shape,
                raw_dtype=gguf_lib.GGMLQuantizationType.BF16,
            )
        elif float_type == "f16":
            writer.add_tensor(name, values.astype(np.float16))
        else:
            writer.add_tensor(name, values)

    add_float("token_embd.weight", (vocab, hidden))
    add_float("output_norm.weight", (hidden,))
    add_float("output.weight", (vocab, hidden))

    for i in range(layers):
        prefix = f"blk.{i}"
        add_float(f"{prefix}.attn_q.weight", (hidden, hidden))
        add_float(f"{prefix}.attn_k.weight", (hidden, hidden))
        add_float(f"{prefix}.attn_v.weight", (hidden, hidden))
        add_float(f"{prefix}.attn_output.weight", (hidden, hidden))
        add_float(f"{prefix}.ffn_gate.weight", (ffn_size, hidden))
        add_float(f"{prefix}.ffn_up.weight", (ffn_size, hidden))
        add_float(f"{prefix}.ffn_down.weight", (hidden, ffn_size))
        add_float(f"{prefix}.attn_norm.weight", (hidden,))
        add_float(f"{prefix}.ffn_norm.weight", (hidden,))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


class TestGGUFReader:
    """Test GGUFModel metadata and tensor reading."""

    def test_architecture(self, tmp_path):
        """GGUFModel extracts the architecture name."""
        from mobius.integrations.gguf._reader import GGUFModel

        path = _create_tiny_gguf(tmp_path / "test.gguf")
        model = GGUFModel(path)
        assert model.architecture == "llama"

    def test_metadata_values(self, tmp_path):
        """GGUFModel parses metadata integer values."""
        from mobius.integrations.gguf._reader import GGUFModel

        path = _create_tiny_gguf(tmp_path / "test.gguf")
        model = GGUFModel(path)
        meta = model.metadata
        assert meta["llama.embedding_length"] == 64
        assert meta["llama.block_count"] == 2
        assert meta["llama.attention.head_count"] == 2
        assert meta["llama.attention.head_count_kv"] == 2
        assert meta["llama.context_length"] == 128
        assert meta["llama.feed_forward_length"] == 256

    def test_get_metadata(self, tmp_path):
        """get_metadata returns value for existing key, default otherwise."""
        from mobius.integrations.gguf._reader import GGUFModel

        path = _create_tiny_gguf(tmp_path / "test.gguf")
        model = GGUFModel(path)
        assert model.get_metadata("llama.block_count") == 2
        assert model.get_metadata("nonexistent.key", 42) == 42

    def test_tensor_names(self, tmp_path):
        """GGUFModel lists all tensor names."""
        from mobius.integrations.gguf._reader import GGUFModel

        path = _create_tiny_gguf(tmp_path / "test.gguf", layers=1)
        model = GGUFModel(path)
        names = model.tensor_names
        assert "token_embd.weight" in names
        assert "output.weight" in names
        assert "blk.0.attn_q.weight" in names
        assert "blk.0.ffn_gate.weight" in names
        # 3 global + 9 per layer = 12 tensors for 1 layer
        assert len(names) == 12

    def test_tensor_items_shapes(self, tmp_path):
        """tensor_items yields arrays preserving numpy shapes."""
        from mobius.integrations.gguf._reader import GGUFModel

        hidden, vocab = 64, 256
        path = _create_tiny_gguf(
            tmp_path / "test.gguf",
            hidden=hidden,
            vocab=vocab,
            layers=1,
        )
        model = GGUFModel(path)
        tensors = dict(model.tensor_items())

        # Shapes match what was written via GGUFWriter
        assert tensors["token_embd.weight"].shape == (vocab, hidden)
        assert tensors["output.weight"].shape == (vocab, hidden)
        assert tensors["output_norm.weight"].shape == (hidden,)
        assert tensors["blk.0.attn_q.weight"].shape == (hidden, hidden)
        assert tensors["blk.0.ffn_gate.weight"].shape == (hidden * 4, hidden)
        assert tensors["blk.0.ffn_down.weight"].shape == (hidden, hidden * 4)

    def test_get_tensor(self, tmp_path):
        """get_tensor returns a single dequantized tensor."""
        from mobius.integrations.gguf._reader import GGUFModel

        path = _create_tiny_gguf(tmp_path / "test.gguf", layers=1)
        model = GGUFModel(path)
        t = model.get_tensor("token_embd.weight")
        assert t.shape == (256, 64)  # (vocab, hidden)
        assert t.dtype == np.float32

    def test_get_tensor_missing_raises(self, tmp_path):
        """get_tensor raises KeyError for unknown tensor name."""
        from mobius.integrations.gguf._reader import GGUFModel

        path = _create_tiny_gguf(tmp_path / "test.gguf", layers=1)
        model = GGUFModel(path)
        with pytest.raises(KeyError, match="nonexistent"):
            model.get_tensor("nonexistent")

    def test_file_not_found_raises(self, tmp_path):
        """GGUFModel raises FileNotFoundError for missing file."""
        from mobius.integrations.gguf._reader import GGUFModel

        with pytest.raises(FileNotFoundError):
            GGUFModel(tmp_path / "does_not_exist.gguf")

    def test_repr(self, tmp_path):
        """GGUFModel repr includes path, arch, and tensor count."""
        from mobius.integrations.gguf._reader import GGUFModel

        path = _create_tiny_gguf(tmp_path / "test.gguf", layers=1)
        model = GGUFModel(path)
        r = repr(model)
        assert "llama" in r
        assert "12" in r  # tensor count


class TestGGUFConfigMapping:
    """Test GGUF metadata → ArchitectureConfig mapping."""

    def test_basic_config_fields(self, tmp_path):
        """gguf_to_config maps basic metadata to config fields."""
        from mobius.integrations.gguf._config_mapping import gguf_to_config
        from mobius.integrations.gguf._reader import GGUFModel

        path = _create_tiny_gguf(tmp_path / "test.gguf")
        model = GGUFModel(path)
        config = gguf_to_config(model)

        assert config.hidden_size == 64
        assert config.num_hidden_layers == 2
        assert config.num_attention_heads == 2
        assert config.num_key_value_heads == 2

    def test_head_dim_derived(self, tmp_path):
        """head_dim is derived from hidden_size / num_heads."""
        from mobius.integrations.gguf._config_mapping import gguf_to_config
        from mobius.integrations.gguf._reader import GGUFModel

        path = _create_tiny_gguf(tmp_path / "test.gguf", hidden=128, heads=4)
        model = GGUFModel(path)
        config = gguf_to_config(model)
        assert config.head_dim == 32

    def test_intermediate_size(self, tmp_path):
        """intermediate_size maps from feed_forward_length."""
        from mobius.integrations.gguf._config_mapping import gguf_to_config
        from mobius.integrations.gguf._reader import GGUFModel

        path = _create_tiny_gguf(tmp_path / "test.gguf", hidden=64, ffn_size=512)
        model = GGUFModel(path)
        config = gguf_to_config(model)
        assert config.intermediate_size == 512

    def test_tie_embeddings_false_when_output_present(self, tmp_path):
        """tie_word_embeddings is False when output.weight tensor exists."""
        from mobius.integrations.gguf._config_mapping import gguf_to_config
        from mobius.integrations.gguf._reader import GGUFModel

        path = _create_tiny_gguf(tmp_path / "test.gguf")
        model = GGUFModel(path)
        config = gguf_to_config(model)
        assert config.tie_word_embeddings is False

    def test_tie_embeddings_true_when_no_output(self, tmp_path):
        """tie_word_embeddings is True when no output.weight tensor."""
        from mobius.integrations.gguf._config_mapping import gguf_to_config
        from mobius.integrations.gguf._reader import GGUFModel

        # Create GGUF without output.weight
        path = str(tmp_path / "tied.gguf")
        writer = gguf_lib.GGUFWriter(path, "llama")
        writer.add_block_count(1)
        writer.add_embedding_length(32)
        writer.add_head_count(2)
        writer.add_head_count_kv(2)
        writer.add_context_length(64)
        writer.add_feed_forward_length(128)

        rng = np.random.default_rng(0)
        writer.add_tensor(
            "token_embd.weight",
            rng.standard_normal((64, 32), dtype=np.float32),
        )
        writer.add_tensor(
            "output_norm.weight",
            rng.standard_normal((32,), dtype=np.float32),
        )
        # No output.weight — embeddings are tied
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        model = GGUFModel(path)
        config = gguf_to_config(model)
        assert config.tie_word_embeddings is True

    def test_model_type_stored(self, tmp_path):
        """Config has _gguf_model_type attribute for registry lookup."""
        from mobius.integrations.gguf._config_mapping import gguf_to_config
        from mobius.integrations.gguf._reader import GGUFModel

        path = _create_tiny_gguf(tmp_path / "test.gguf")
        model = GGUFModel(path)
        config = gguf_to_config(model)
        assert getattr(config, "_gguf_model_type", None) == "llama"


class TestGGUFArchToModelType:
    """Test GGUF architecture → model_type mapping."""

    def test_known_architectures(self):
        """All expected GGUF architectures are mapped."""
        from mobius.integrations.gguf._config_mapping import (
            GGUF_ARCH_TO_MODEL_TYPE,
        )

        assert GGUF_ARCH_TO_MODEL_TYPE["llama"] == "llama"
        assert GGUF_ARCH_TO_MODEL_TYPE["mistral"] == "llama"
        assert GGUF_ARCH_TO_MODEL_TYPE["qwen2"] == "qwen2"
        assert GGUF_ARCH_TO_MODEL_TYPE["qwen35"] == "qwen3_5_text"
        assert GGUF_ARCH_TO_MODEL_TYPE["gemma2"] == "gemma2"
        assert GGUF_ARCH_TO_MODEL_TYPE["phi3"] == "phi3"
        assert GGUF_ARCH_TO_MODEL_TYPE["falcon"] == "falcon"
        assert GGUF_ARCH_TO_MODEL_TYPE["gpt2"] == "gpt2"
        assert GGUF_ARCH_TO_MODEL_TYPE["mamba"] == "mamba"


class TestGGUFTensorMapping:
    """Test GGUF → HF tensor name mapping."""

    def test_llama_global_tensors(self):
        """Global tensors map correctly for llama."""
        from mobius.integrations.gguf._tensor_mapping import (
            map_gguf_to_hf_names,
        )

        assert (
            map_gguf_to_hf_names("token_embd.weight", "llama") == "model.embed_tokens.weight"
        )
        assert map_gguf_to_hf_names("output.weight", "llama") == "lm_head.weight"
        assert map_gguf_to_hf_names("output_norm.weight", "llama") == "model.norm.weight"

    def test_llama_block_tensors(self):
        """Block-indexed tensors map correctly for llama."""
        from mobius.integrations.gguf._tensor_mapping import (
            map_gguf_to_hf_names,
        )

        assert (
            map_gguf_to_hf_names("blk.0.attn_q.weight", "llama")
            == "model.layers.0.self_attn.q_proj.weight"
        )
        assert (
            map_gguf_to_hf_names("blk.5.ffn_gate.weight", "llama")
            == "model.layers.5.mlp.gate_proj.weight"
        )
        assert (
            map_gguf_to_hf_names("blk.31.attn_output.weight", "llama")
            == "model.layers.31.self_attn.o_proj.weight"
        )

    def test_qwen35_hybrid_tensors(self):
        """Dense Qwen3.5 GGUF maps both DeltaNet and dense MLP tensors."""
        from mobius.integrations.gguf._tensor_mapping import (
            map_gguf_to_hf_names,
        )

        assert (
            map_gguf_to_hf_names("blk.0.attn_qkv.weight", "qwen35")
            == "model.layers.0.linear_attn.in_proj_qkv.weight"
        )
        assert (
            map_gguf_to_hf_names("blk.0.ssm_out.weight", "qwen35")
            == "model.layers.0.linear_attn.out_proj.weight"
        )
        assert (
            map_gguf_to_hf_names("blk.0.post_attention_norm.weight", "qwen35")
            == "model.layers.0.post_attention_layernorm.weight"
        )
        assert (
            map_gguf_to_hf_names("blk.0.ffn_gate.weight", "qwen35")
            == "model.layers.0.mlp.gate_proj.weight"
        )

    def test_skip_tokenizer_tensors(self):
        """Tokenizer tensors return None (should be skipped)."""
        from mobius.integrations.gguf._tensor_mapping import (
            map_gguf_to_hf_names,
        )

        assert map_gguf_to_hf_names("tokenizer.ggml.tokens", "llama") is None

    def test_skip_rope_freqs(self):
        """Rotary embedding tensors return None."""
        from mobius.integrations.gguf._tensor_mapping import (
            map_gguf_to_hf_names,
        )

        assert map_gguf_to_hf_names("rope_freqs.weight", "llama") is None

    def test_unsupported_arch_raises(self):
        """Unsupported architecture raises ValueError."""
        from mobius.integrations.gguf._tensor_mapping import (
            map_gguf_to_hf_names,
        )

        with pytest.raises(ValueError, match="Unsupported GGUF"):
            map_gguf_to_hf_names("token_embd.weight", "unknown_arch")

    def test_build_gguf_to_hf_map(self):
        """build_gguf_to_hf_map batch-maps tensor names."""
        from mobius.integrations.gguf._tensor_mapping import (
            build_gguf_to_hf_map,
        )

        gguf_names = [
            "token_embd.weight",
            "blk.0.attn_q.weight",
            "tokenizer.ggml.tokens",
        ]
        result = build_gguf_to_hf_map(gguf_names, "llama")
        assert "token_embd.weight" in result
        assert "blk.0.attn_q.weight" in result
        assert "tokenizer.ggml.tokens" not in result


class TestGGUFTensorProcessors:
    """Test architecture-specific tensor processors."""

    def test_no_op_for_unknown_model_type(self):
        """process_tensors returns state_dict unchanged for unknown types."""
        from mobius.integrations.gguf._tensor_processors import (
            process_tensors,
        )

        # Config with no model_type → passthrough
        class FakeConfig:
            model_type = "unknown_model_xyz"

        sd = {"layer.weight": torch.randn(4, 4)}
        result = process_tensors(sd, FakeConfig())
        assert result is sd

    def test_no_op_when_no_model_type(self):
        """process_tensors returns state_dict when config has no model_type."""
        from mobius.integrations.gguf._tensor_processors import (
            process_tensors,
        )

        sd = {"layer.weight": torch.randn(4, 4)}
        result = process_tensors(sd, object())
        assert result is sd

    def test_gemma_norm_offset(self):
        """Gemma processor subtracts 1 from GGUF norm weights."""
        from mobius.integrations.gguf._tensor_processors import (
            process_tensors,
        )

        class FakeConfig:
            model_type = "gemma2"

        sd = {
            "model.layers.0.input_layernorm.weight": torch.zeros(8),
            "model.layers.0.self_attn.q_proj.weight": torch.ones(8, 8),
        }
        result = process_tensors(sd, FakeConfig())
        # Norm weight should have 1 subtracted
        assert torch.allclose(
            result["model.layers.0.input_layernorm.weight"],
            -torch.ones(8),
        )
        # Non-norm weight unchanged
        assert torch.allclose(
            result["model.layers.0.self_attn.q_proj.weight"],
            torch.ones(8, 8),
        )


class TestCLIBuildGGUF:
    """Test the build-gguf CLI subcommand."""

    def test_partial_cli_status_distinguishes_deferred_support_from_omission(
        self, tmp_path, capsys
    ):
        from types import SimpleNamespace

        from mobius.__main__ import _print_gguf_export_status
        from mobius._export_report import (
            ComponentExportDisposition,
            ComponentExportReport,
        )

        report = ComponentExportReport.create(
            (
                ComponentExportDisposition(
                    name="model",
                    route="llama",
                    requested=True,
                    discovered=True,
                    support="supported",
                    output="exported",
                ),
                ComponentExportDisposition(
                    name="runtime",
                    route="ort-genai",
                    requested=True,
                    discovered=True,
                    support="deferred",
                    output="exported",
                    runtime_validation_status="unvalidated",
                    blocker_category="runtime-validation-unavailable",
                    reason="runtime evidence is pending",
                    impact="execution is unvalidated",
                    remediation="validate before production use",
                ),
                ComponentExportDisposition(
                    name="tokenizer",
                    route="embedded",
                    requested=True,
                    discovered=True,
                    support="supported",
                    output="exported",
                ),
            ),
            end_to_end_runnable=False,
        )

        _print_gguf_export_status(
            SimpleNamespace(export_report=report),
            str(tmp_path),
            runtime="ort-genai",
        )

        output = capsys.readouterr().out
        assert "all requested components were exported" in output
        assert "components were omitted" not in output

    def test_help_text(self, capsys):
        """build-gguf subcommand shows in help."""
        from mobius.__main__ import main

        with pytest.raises(SystemExit):
            main(["build-gguf", "--help"])
        out = capsys.readouterr().out
        assert "--dequantize" in out
        assert "--reuse-gguf-weights" in out
        assert "--output OUTPUT_DIR" in out
        assert "--keep-quantized" not in out, "the unread deprecated alias was removed"
        assert "--max-shard-size" in out, "shard sizing applies to GGUF builds too"

    def test_missing_gguf_path_errors(self):
        """build-gguf requires a gguf_path argument."""
        from mobius.__main__ import main

        with pytest.raises(SystemExit):
            main(["build-gguf"])

    @pytest.mark.parametrize("float_type", ["f32", "f16", "bf16"])
    def test_default_float_only_input_builds_and_reloads(self, tmp_path, float_type):
        """The quantized-by-default CLI accepts F32/F16/BF16-only GGUFs."""
        from mobius.__main__ import main
        from mobius._model_package import ModelPackage

        path = _create_tiny_gguf(tmp_path / f"test-{float_type}.gguf", float_type=float_type)
        output_dir = tmp_path / f"output-{float_type}"
        main(["build-gguf", path, "--output", str(output_dir)])

        package = ModelPackage.load(str(output_dir))
        op_types = {node.op_type for node in package["model"].graph}
        assert "MatMulNBits" not in op_types
        assert "BlockQuantizedMatMul" not in op_types

    def test_authoritative_tokenizer_blocker_exports_partial_model_package(
        self, tmp_path, caplog
    ):
        from mobius._model_package import ModelPackage
        from mobius.integrations.gguf import build_from_gguf
        from mobius.integrations.gguf._tokenizer_census import tokenizer_route_census

        path = _create_tiny_gguf(
            tmp_path / "blocked-tokenizer.gguf",
            tokenizer_pre="bailingmoe2",
        )
        with caplog.at_level("WARNING", logger="mobius.integrations.gguf._component_export"):
            package = build_from_gguf(path, keep_quantized=False)

        assert "model" in package
        assert package.export_report is not None
        assert package.export_report.status == "partial"
        assert package.export_report.runtime_validation_status == "unvalidated"
        tokenizer = package.export_report.component("tokenizer")
        assert tokenizer is not None
        audit = next(
            record for record in tokenizer_route_census() if record.identifier == "bailingmoe2"
        )
        assert tokenizer.route == "bailingmoe2"
        assert tokenizer.support == "blocked"
        assert tokenizer.output == "omitted"
        assert tokenizer.blocker_category == audit.blocker_category
        assert tokenizer.evidence_id == audit.blocker_evidence_id
        assert tokenizer.reason == audit.candidate_disposition
        assert "semantics are unverified" in tokenizer.remediation

        warnings = [
            record.getMessage()
            for record in caplog.records
            if record.getMessage().startswith("GGUF PARTIAL EXPORT WARNING:")
        ]
        assert len(warnings) == 1
        warning = json.loads(warnings[0].partition(": ")[2])
        assert warning["route"] == "bailingmoe2"
        assert warning["blocker_category"] == audit.blocker_category
        assert warning["evidence_id"] == audit.blocker_evidence_id
        assert warning["reason"] == audit.candidate_disposition
        assert "not end-to-end runnable" in warning["impact"]
        assert "Provide and validate a tokenizer" in warning["remediation"]

        first = tmp_path / "partial-a"
        second = tmp_path / "partial-b"
        package.save(first, progress_bar=False)
        package.save(second, progress_bar=False)
        assert (first / "model.onnx").is_file()
        assert (first / "export_report.json").read_bytes() == (
            second / "export_report.json"
        ).read_bytes()
        assert ModelPackage.load(first).export_report == package.export_report
        assert not {
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "gguf_tokenizer_manifest.json",
        }.intersection(path.name for path in first.iterdir())

    def test_missing_tokenizer_metadata_exports_model_with_one_structured_warning(
        self, tmp_path, caplog
    ):
        from mobius.integrations.gguf import build_from_gguf

        path = _create_tiny_gguf(tmp_path / "no-tokenizer.gguf")
        with caplog.at_level("WARNING", logger="mobius.integrations.gguf._component_export"):
            package = build_from_gguf(path, keep_quantized=False)

        tokenizer = package.export_report.component("tokenizer")
        assert tokenizer.route == "absent"
        assert tokenizer.support == "deferred"
        assert tokenizer.output == "omitted"
        assert tokenizer.blocker_category == "serialized-tokenizer-pipeline-incomplete"
        assert tokenizer.reason.endswith("contains no tokenizer metadata")
        assert "semantics are unverified" in tokenizer.remediation
        warnings = [
            record.getMessage()
            for record in caplog.records
            if record.getMessage().startswith("GGUF PARTIAL EXPORT WARNING:")
        ]
        assert len(warnings) == 1
        warning = json.loads(warnings[0].partition(": ")[2])
        assert warning["route"] == "absent"
        assert warning["blocker_category"] == ("serialized-tokenizer-pipeline-incomplete")
        assert warning["reason"].endswith("contains no tokenizer metadata")

    def test_unexpected_tokenizer_disposition_error_still_fails(
        self, tmp_path, monkeypatch, caplog
    ):
        from mobius.integrations.gguf import build_from_gguf

        path = _create_tiny_gguf(
            tmp_path / "unexpected-tokenizer.gguf",
            tokenizer_pre="bailingmoe",
        )

        def unexpected(*_args, **_kwargs):
            raise RuntimeError("unexpected tokenizer evidence failure")

        monkeypatch.setattr(
            "mobius.integrations.gguf._tokenizer_evidence.matching_tokenizer_blocker_evidence",
            unexpected,
        )
        with pytest.raises(RuntimeError, match="unexpected tokenizer evidence failure"):
            build_from_gguf(path, keep_quantized=False)
        assert "GGUF PARTIAL EXPORT WARNING" not in caplog.text

    def test_model_route_blocker_remains_a_hard_failure(self, tmp_path, caplog):
        from mobius.integrations.gguf import build_from_gguf
        from mobius.integrations.gguf._errors import UnsupportedGGUFArchitectureError

        path = _create_tiny_gguf(
            tmp_path / "unsupported-model.gguf",
            arch="future-model",
            tokenizer_pre="bailingmoe",
        )

        with pytest.raises(UnsupportedGGUFArchitectureError):
            build_from_gguf(path, keep_quantized=False)
        assert "GGUF PARTIAL EXPORT WARNING" not in caplog.text

    def test_exact_artifact_blocker_overrides_a_validated_route(
        self, tmp_path, monkeypatch, caplog
    ):
        from mobius.integrations.gguf import build_from_gguf
        from mobius.integrations.gguf._tokenizer_evidence import tokenizer_blocker_evidence

        blocker = tokenizer_blocker_evidence("plm-1.8b-instruct-q4-k-m-tokenizer-blocker")
        assert blocker is not None
        path = _create_tiny_gguf(
            tmp_path / "exact-artifact-blocker.gguf",
            tokenizer_pre="qwen2",
        )
        monkeypatch.setattr(
            "mobius.integrations.gguf._tokenizer_evidence.matching_tokenizer_blocker_evidence",
            lambda *_args, **_kwargs: blocker,
        )

        with caplog.at_level("WARNING", logger="mobius.integrations.gguf._component_export"):
            package = build_from_gguf(path, keep_quantized=False)

        tokenizer = package.export_report.component("tokenizer")
        assert tokenizer.route == "qwen2"
        assert tokenizer.evidence_id == blocker.evidence_id
        assert tokenizer.reason == blocker.disposition
        assert caplog.text.count("GGUF PARTIAL EXPORT WARNING:") == 1

    def test_multimodal_route_records_identity_and_exact_tokenizer_blocker(
        self, tmp_path, monkeypatch
    ):
        from mobius.integrations.gguf import build_from_gguf
        from mobius.integrations.gguf._tokenizer import GGUFTokenizerVerdict
        from mobius.integrations.gguf._tokenizer_evidence import tokenizer_blocker_evidence

        blocker = tokenizer_blocker_evidence("plm-1.8b-instruct-q4-k-m-tokenizer-blocker")
        assert blocker is not None
        package = mock.MagicMock()
        package.__iter__.return_value = iter(("model", "vision"))
        package.config = SimpleNamespace(model_type="test-vlm")
        package.gguf_source_path = str(tmp_path / "text.gguf")
        package.gguf_tokenizer_verdict = GGUFTokenizerVerdict(
            route="deferred",
            model="gpt2",
            pre="qwen2",
            canonical_pre="qwen2",
            reason="base deferred reason",
            token_count=2,
            metadata_sha256="a" * 64,
        )
        package.export_report = None
        text_model = SimpleNamespace(
            architecture="llama",
            metadata={},
            source_identity=(1, 2, 3),
            source_matches_path=lambda: True,
        )
        monkeypatch.setattr(
            "mobius.integrations.gguf._mmproj.build_vlm_from_gguf",
            lambda *_args, **_kwargs: package,
        )
        monkeypatch.setattr(
            "mobius.integrations.gguf._tokenizer_evidence.matching_tokenizer_blocker_evidence",
            lambda *_args, **_kwargs: blocker,
        )

        result = build_from_gguf(
            "text.gguf",
            mmproj="vision.gguf",
            _gguf_model=text_model,
        )

        assert result is package
        assert package.gguf_architecture == "llama"
        assert package.gguf_execution_provider == "default"
        assert package.gguf_source_filename == "text.gguf"
        route = json.loads(package.gguf_import_route)
        assert route["components"] == ["model", "vision"]
        assert route["multimodal_projector"] is True
        tokenizer = package.export_report.component("tokenizer")
        assert tokenizer.evidence_id == blocker.evidence_id
        assert tokenizer.reason == blocker.disposition

    def test_validated_tokenizer_route_preserves_existing_graph_package_files(
        self, tmp_path, caplog
    ):
        from mobius.integrations.gguf import build_from_gguf

        path = _create_tiny_gguf(
            tmp_path / "validated-tokenizer.gguf",
            tokenizer_pre="gpt-2",
        )
        package = build_from_gguf(path, keep_quantized=False)
        output = tmp_path / "validated-output"
        package.save(output, progress_bar=False)

        assert package.export_report is None
        assert not (output / "export_report.json").exists()
        assert "GGUF PARTIAL EXPORT WARNING" not in caplog.text

    def test_cli_runtime_request_preserves_model_for_tokenizer_blocker(
        self, tmp_path, caplog, capsys
    ):
        from mobius.__main__ import main
        from mobius._model_package import ModelPackage

        path = _create_tiny_gguf(
            tmp_path / "runtime-blocked-tokenizer.gguf",
            tokenizer_pre="cohere2moe",
        )
        output = tmp_path / "runtime-partial"
        with caplog.at_level("WARNING", logger="mobius.integrations.gguf._component_export"):
            main(
                [
                    "build-gguf",
                    path,
                    "--output",
                    str(output),
                    "--dequantize",
                    "--runtime",
                    "ort-genai",
                ]
            )

        stdout = capsys.readouterr().out
        package = ModelPackage.load(output)
        assert (output / "model.onnx").is_file()
        assert (output / "export_report.json").is_file()
        assert package.export_report is not None
        assert package.export_report.component("runtime").output == "exported"
        assert "Export status: PARTIAL" in stdout
        assert "requested runtime: ort-genai (unvalidated)" in stdout
        assert "omitted components: tokenizer" in stdout
        assert caplog.text.count("GGUF PARTIAL EXPORT WARNING:") == 1
        assert (output / "genai_config.json").is_file()
        assert not (output / "tokenizer.json").exists()
        assert not list(tmp_path.glob(".runtime-partial.*.tmp"))

    def test_cli_unmatched_runtime_version_exports_unvalidated_model(
        self, tmp_path, caplog, capsys
    ):
        from mobius.__main__ import main
        from mobius._model_package import ModelPackage

        path = _create_tiny_gguf(
            tmp_path / "runtime-unvalidated.gguf",
            tokenizer_pre="gpt-2",
        )
        output = tmp_path / "runtime-unvalidated"
        with caplog.at_level("WARNING", logger="mobius.integrations.gguf._component_export"):
            main(
                [
                    "build-gguf",
                    path,
                    "--output",
                    str(output),
                    "--dequantize",
                    "--runtime",
                    "ort-genai",
                    "--runtime-version",
                    "99.0",
                    "--tokenizer-repository",
                    "owner/tokenizer",
                    "--tokenizer-revision",
                    "a" * 40,
                ]
            )

        stdout = capsys.readouterr().out
        package = ModelPackage.load(output)
        assert (output / "model.onnx").is_file()
        assert package.export_report is not None
        assert package.export_report.export_status == "partial"
        assert package.export_report.runtime_validation_status == "unvalidated"
        runtime_component = package.export_report.component("runtime")
        assert runtime_component.blocker_category == "runtime-validation-unavailable"
        assert runtime_component.output == "exported"
        assert "Export status: PARTIAL" in stdout
        assert "Runtime validation: UNVALIDATED" in stdout
        assert "requested runtime: ort-genai (unvalidated)" in stdout
        assert (output / "genai_config.json").is_file()
        assert (output / "runtime_compatibility.json").is_file()
        assert not (output / "tokenizer.json").exists()

    @pytest.mark.parametrize(
        ("extra_args", "expected"),
        [
            ([], True),
            (["--dequantize"], False),
        ],
    )
    def test_quantization_flag_parsing(self, tmp_path, extra_args, expected):
        """Quantized target storage is default; ``--dequantize`` requests float.

        The ``--keep-quantized`` alias that used to be covered here was removed:
        it was never read (``keep_quantized = not args.dequantize``), so the case
        asserting it produced ``True`` was really just re-testing the default.
        """
        from mobius.__main__ import main

        package = mock.MagicMock()
        package.__iter__.return_value = iter(())
        package.mtp_head = None
        package.draft_manifest = None
        package.values.return_value = iter(())
        with mock.patch(
            "mobius.integrations.gguf.build_from_gguf",
            return_value=package,
        ) as build:
            main(
                [
                    "build-gguf",
                    str(tmp_path / "model.gguf"),
                    "--output",
                    str(tmp_path / "output"),
                    *extra_args,
                ]
            )

        assert build.call_args.kwargs["keep_quantized"] is expected

    def test_reuse_flag_is_forwarded(self, tmp_path):
        """The explicit CLI opt-in reaches the GGUF API unchanged."""
        from mobius.__main__ import main

        package = mock.MagicMock()
        package.__iter__.return_value = iter(())
        package.values.return_value = iter(())
        package.mtp_head = None
        package.draft_manifest = None
        with mock.patch(
            "mobius.integrations.gguf.build_from_gguf",
            return_value=package,
        ) as build:
            main(
                [
                    "build-gguf",
                    str(tmp_path / "model.gguf"),
                    "--output",
                    str(tmp_path / "output"),
                    "--reuse-gguf-weights",
                ]
            )

        assert build.call_args.kwargs["reuse_gguf_weights"] is True

    def test_reuse_with_ort_genai_preserves_runtime_unvalidated_model(self, tmp_path):
        from mobius.__main__ import main
        from mobius.integrations.gguf._component_export import (
            attach_runtime_unvalidated_report,
        )

        package = mock.MagicMock()
        package.export_report = None
        package.gguf_architecture = "llama"
        package.gguf_tokenizer_verdict = mock.Mock(
            materialized=False,
            route_identifier="gpt-2",
            reason="tokenizer materialization was not validated",
            blocker_category=None,
            evidence_id=None,
        )
        package.__iter__.return_value = iter(("model",))
        package.__len__.return_value = 1
        gguf_model = mock.Mock(metadata={}, architecture="llama")
        output = tmp_path / "output"

        def publish(pkg, _source, destination, **_kwargs):
            attach_runtime_unvalidated_report(
                pkg,
                "ort-genai",
                blocker_category="runtime-executor-limitation",
                reason="ORT GenAI cannot disable constant folding.",
            )
            destination = Path(destination)
            destination.mkdir()
            (destination / "model.onnx").write_bytes(b"model")
            pkg.export_report.write_json(destination / "export_report.json")
            return {"export_report": str(destination / "export_report.json")}

        with (
            mock.patch(
                "mobius.integrations.gguf._builder._resolve_gguf_path",
                return_value=str(tmp_path / "model.gguf"),
            ),
            mock.patch(
                "mobius.integrations.gguf._shard_set.open_gguf_model",
                return_value=gguf_model,
            ),
            mock.patch("mobius.integrations.gguf._builder._validate_gguf_model"),
            mock.patch(
                "mobius.integrations.gguf.build_from_gguf",
                return_value=package,
            ) as build,
            mock.patch(
                "mobius.integrations.gguf.write_gguf_runtime_package",
                side_effect=publish,
            ),
        ):
            main(
                [
                    "build-gguf",
                    str(tmp_path / "model.gguf"),
                    "--output",
                    str(output),
                    "--reuse-gguf-weights",
                    "--runtime",
                    "ort-genai",
                ]
            )

        assert build.call_args.kwargs["reuse_gguf_weights"] is True
        assert (output / "model.onnx").is_file()
        assert package.export_report.component("runtime").output == "exported"

    def test_ort_genai_runtime_is_forwarded_to_package_writer(self, tmp_path):
        """build-gguf forwards one canonically opened source to build and packaging."""
        from mobius.__main__ import main
        from mobius.integrations.gguf._spec import Support

        package = mock.MagicMock()
        package.__iter__.return_value = iter(())
        package.mtp_head = None
        package.draft_manifest = None
        gguf_model = mock.Mock(metadata={}, architecture="llama")
        output_dir = tmp_path / "output"
        with (
            mock.patch(
                "mobius.integrations.gguf._builder._resolve_gguf_path",
                return_value=str(tmp_path / "model.gguf"),
            ),
            mock.patch(
                "mobius.integrations.gguf._shard_set.open_gguf_model",
                return_value=gguf_model,
            ) as open_model,
            mock.patch("mobius.integrations.gguf._builder._validate_gguf_model"),
            mock.patch(
                "mobius.integrations.gguf._arch_registry.get_arch_spec",
                return_value=mock.Mock(
                    gguf_arch="llama",
                    runtime=Support.SUPPORTED,
                    reason=None,
                ),
            ),
            mock.patch(
                "mobius.integrations.gguf.build_from_gguf",
                return_value=package,
            ) as build,
            mock.patch(
                "mobius.integrations.gguf.write_gguf_runtime_package",
                return_value={"genai_config": str(output_dir / "genai_config.json")},
            ) as write_runtime,
        ):
            main(
                [
                    "build-gguf",
                    str(tmp_path / "model.gguf"),
                    "--output",
                    str(output_dir),
                    "--runtime",
                    "ort-genai",
                    "--tokenizer-repository",
                    "owner/tokenizer",
                    "--tokenizer-revision",
                    "a" * 40,
                ]
            )

        open_model.assert_called_once_with(str(tmp_path / "model.gguf"))
        assert build.call_args.kwargs["_gguf_model"] is gguf_model
        write_runtime.assert_called_once_with(
            package,
            str(tmp_path / "model.gguf"),
            str(output_dir),
            runtime="ort-genai",
            runtime_version=None,
            tokenizer_repository="owner/tokenizer",
            tokenizer_revision="a" * 40,
            local_files_only=False,
            external_data="onnx",
            max_shard_size_bytes=None,
            max_workers=8,
        )

    def test_runtime_without_pinned_tokenizer_exports_unvalidated_model(
        self, tmp_path, capsys
    ):
        from mobius.__main__ import main
        from mobius.integrations.gguf._component_export import (
            attach_runtime_unvalidated_report,
        )

        output = tmp_path / "output"
        package = mock.MagicMock()
        package.export_report = None
        package.gguf_architecture = "llama"
        package.gguf_tokenizer_verdict = mock.Mock(
            materialized=False,
            route_identifier="gpt-2",
            reason="tokenizer materialization was not validated",
            blocker_category=None,
            evidence_id=None,
        )
        package.__iter__.return_value = iter(("model",))
        package.__len__.return_value = 1

        def publish_unvalidated(pkg, _source, destination, **_kwargs):
            attach_runtime_unvalidated_report(
                pkg,
                "ort-genai",
                blocker_category="runtime-tokenizer-identity-unavailable",
                reason="No immutable tokenizer identity was supplied.",
            )
            destination = Path(destination)
            destination.mkdir()
            (destination / "model.onnx").write_bytes(b"model")
            pkg.export_report.write_json(destination / "export_report.json")
            return {"export_report": str(destination / "export_report.json")}

        with (
            mock.patch(
                "mobius.integrations.gguf._builder._resolve_gguf_path",
                return_value=str(tmp_path / "model.gguf"),
            ),
            mock.patch(
                "mobius.integrations.gguf._shard_set.open_gguf_model",
                return_value=mock.Mock(metadata={}, architecture="llama"),
            ),
            mock.patch("mobius.integrations.gguf._builder._validate_gguf_model"),
            mock.patch(
                "mobius.integrations.gguf.build_from_gguf",
                return_value=package,
            ) as build,
            mock.patch(
                "mobius.integrations.gguf.write_gguf_runtime_package",
                side_effect=publish_unvalidated,
            ),
        ):
            main(
                [
                    "build-gguf",
                    str(tmp_path / "model.gguf"),
                    "--output",
                    str(output),
                    "--runtime",
                    "ort-genai",
                ]
            )

        assert build.call_count == 1
        assert (output / "model.onnx").is_file()
        assert (output / "export_report.json").is_file()
        assert package.export_report.export_status == "partial"
        assert package.export_report.runtime_validation_status == "unvalidated"
        assert "requested runtime: ort-genai (unvalidated)" in capsys.readouterr().out

    @pytest.mark.parametrize(
        ("error", "message"),
        [
            (
                FileNotFoundError(
                    "Incomplete GGUF split set 'model': declared 2 shards but 1 missing"
                ),
                "1 missing",
            ),
            (ValueError("invalid GGUF header in continuation shard"), "invalid GGUF header"),
        ],
    )
    def test_runtime_shard_preflight_fails_before_build_or_output(
        self, tmp_path, capsys, error, message
    ):
        from mobius.__main__ import main

        output = tmp_path / "output"
        shard = tmp_path / "model-00001-of-00002.gguf"
        with (
            mock.patch(
                "mobius.integrations.gguf._builder._resolve_gguf_path",
                return_value=str(shard),
            ),
            mock.patch(
                "mobius.integrations.gguf._shard_set.open_gguf_model",
                side_effect=error,
            ),
            mock.patch("mobius.integrations.gguf.build_from_gguf") as build,
            pytest.raises(type(error), match=message),
        ):
            main(
                [
                    "build-gguf",
                    str(shard),
                    "--output",
                    str(output),
                    "--runtime",
                    "ort-genai",
                    "--runtime-version",
                    "0.15.2",
                    "--tokenizer-repository",
                    "owner/tokenizer",
                    "--tokenizer-revision",
                    "a" * 40,
                ]
            )

        build.assert_not_called()
        assert "Saved" not in capsys.readouterr().out
        assert not output.exists()

    def test_runtime_packaging_failure_emits_no_saved_output(self, tmp_path, capsys):
        from mobius.__main__ import main
        from mobius.integrations.gguf._spec import Support

        output = tmp_path / "output"
        package = mock.MagicMock()
        package.__iter__.return_value = iter(("model",))
        package.__len__.return_value = 1
        gguf_model = mock.Mock(metadata={}, architecture="llama")
        with (
            mock.patch(
                "mobius.integrations.gguf._builder._resolve_gguf_path",
                return_value=str(tmp_path / "model.gguf"),
            ),
            mock.patch(
                "mobius.integrations.gguf._shard_set.open_gguf_model",
                return_value=gguf_model,
            ),
            mock.patch("mobius.integrations.gguf._builder._validate_gguf_model"),
            mock.patch(
                "mobius.integrations.gguf._arch_registry.get_arch_spec",
                return_value=mock.Mock(
                    gguf_arch="llama",
                    runtime=Support.SUPPORTED,
                    reason=None,
                ),
            ),
            mock.patch(
                "mobius.integrations.gguf.build_from_gguf",
                return_value=package,
            ),
            mock.patch(
                "mobius.integrations.gguf.write_gguf_runtime_package",
                side_effect=ValueError("serialized package evidence mismatch"),
            ),
            pytest.raises(ValueError, match="evidence mismatch"),
        ):
            main(
                [
                    "build-gguf",
                    str(tmp_path / "model.gguf"),
                    "--output",
                    str(output),
                    "--runtime",
                    "ort-genai",
                    "--runtime-version",
                    "0.15.2",
                    "--tokenizer-repository",
                    "owner/tokenizer",
                    "--tokenizer-revision",
                    "a" * 40,
                ]
            )

        assert "Saved" not in capsys.readouterr().out
        assert not output.exists()

    def test_runtime_success_is_reported_only_after_publication(self, tmp_path):
        from mobius.__main__ import main
        from mobius.integrations.gguf._spec import Support

        output = tmp_path / "output"
        package = mock.MagicMock()
        package.__iter__.return_value = iter(("model",))
        package.__len__.return_value = 1

        def publish(*_args, **_kwargs):
            output.mkdir()
            (output / "model.onnx").write_bytes(b"model")
            return {"genai_config": str(output / "genai_config.json")}

        original_print = print

        def assert_published_before_print(*args, **kwargs):
            if args and str(args[0]).startswith("Saved"):
                assert (output / "model.onnx").is_file()
            original_print(*args, **kwargs)

        with (
            mock.patch(
                "mobius.integrations.gguf._builder._resolve_gguf_path",
                return_value=str(tmp_path / "model.gguf"),
            ),
            mock.patch(
                "mobius.integrations.gguf._shard_set.open_gguf_model",
                return_value=mock.Mock(metadata={}, architecture="llama"),
            ),
            mock.patch("mobius.integrations.gguf._builder._validate_gguf_model"),
            mock.patch(
                "mobius.integrations.gguf._arch_registry.get_arch_spec",
                return_value=mock.Mock(
                    gguf_arch="llama",
                    runtime=Support.SUPPORTED,
                    reason=None,
                ),
            ),
            mock.patch(
                "mobius.integrations.gguf.build_from_gguf",
                return_value=package,
            ),
            mock.patch(
                "mobius.integrations.gguf.write_gguf_runtime_package",
                side_effect=publish,
            ),
            mock.patch("builtins.print", side_effect=assert_published_before_print),
        ):
            main(
                [
                    "build-gguf",
                    str(tmp_path / "model.gguf"),
                    "--output",
                    str(output),
                    "--runtime",
                    "ort-genai",
                    "--runtime-version",
                    "0.15.2",
                    "--tokenizer-repository",
                    "owner/tokenizer",
                    "--tokenizer-revision",
                    "a" * 40,
                ]
            )
