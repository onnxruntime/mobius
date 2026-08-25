# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for GGUF import support.

Creates synthetic GGUF files using the ``gguf`` package's writer to
test the full pipeline without network downloads.
"""

from __future__ import annotations

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

    @pytest.mark.parametrize(
        ("extra_args", "expected"),
        [
            ([], True),
            (["--dequantize"], False),
        ],
    )
    def test_quantization_flag_parsing(self, tmp_path, extra_args, expected):
        """Quantization is preserved by default; ``--dequantize`` opts out.

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

    def test_reuse_rejects_ort_genai_runtime(self, tmp_path):
        from mobius.__main__ import main

        with pytest.raises(SystemExit, match="cannot be combined"):
            main(
                [
                    "build-gguf",
                    str(tmp_path / "model.gguf"),
                    "--output",
                    str(tmp_path / "output"),
                    "--reuse-gguf-weights",
                    "--runtime",
                    "ort-genai",
                ]
            )

    def test_ort_genai_runtime_is_forwarded_to_package_writer(self, tmp_path):
        """build-gguf forwards the selected runtime after saving the graph."""
        from mobius.__main__ import main

        package = mock.MagicMock()
        package.__iter__.return_value = iter(())
        package.mtp_head = None
        package.draft_manifest = None
        output_dir = tmp_path / "output"
        with (
            mock.patch(
                "mobius.integrations.gguf._builder._resolve_gguf_path",
                return_value=str(tmp_path / "model.gguf"),
            ),
            mock.patch(
                "mobius.integrations.gguf._reader.GGUFModel",
                return_value=mock.Mock(metadata={}),
            ),
            mock.patch("mobius.integrations.gguf._builder._validate_gguf_model"),
            mock.patch(
                "mobius.integrations.gguf._tokenizer.inspect_gguf_tokenizer",
                return_value=mock.Mock(materialized=True),
            ),
            mock.patch(
                "mobius.integrations.gguf.build_from_gguf",
                return_value=package,
            ),
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
                ]
            )

        write_runtime.assert_called_once_with(
            package,
            str(tmp_path / "model.gguf"),
            str(output_dir),
            runtime="ort-genai",
            external_data="onnx",
            max_shard_size_bytes=None,
            max_workers=8,
        )

    def test_deferred_runtime_tokenizer_fails_before_graph_or_output(self, tmp_path):
        from mobius.__main__ import main

        output = tmp_path / "output"
        with (
            mock.patch(
                "mobius.integrations.gguf._builder._resolve_gguf_path",
                return_value=str(tmp_path / "model.gguf"),
            ),
            mock.patch(
                "mobius.integrations.gguf._reader.GGUFModel",
                return_value=mock.Mock(metadata={}),
            ),
            mock.patch("mobius.integrations.gguf._builder._validate_gguf_model"),
            mock.patch(
                "mobius.integrations.gguf._tokenizer.inspect_gguf_tokenizer",
                return_value=mock.Mock(
                    materialized=False, reason="opaque pre-tokenizer is deferred"
                ),
            ),
            mock.patch("mobius.integrations.gguf.build_from_gguf") as build,
            pytest.raises(SystemExit, match="opaque pre-tokenizer is deferred"),
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
        build.assert_not_called()
        assert not output.exists()
