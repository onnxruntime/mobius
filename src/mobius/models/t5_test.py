# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for T5 model — relative position bias and weight renaming."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from mobius.models.t5 import _rename_t5_weight


def test_standard_t5_graph_contract_is_stable():
    """Ordinary T5 keeps the pre-GGUF deterministic graph size and cache shape."""
    from mobius._configs import ArchitectureConfig
    from mobius.models.t5 import T5ForConditionalGeneration
    from mobius.tasks import Seq2SeqTask

    config = ArchitectureConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
        num_decoder_layers=2,
        vocab_size=256,
        max_position_embeddings=64,
        hidden_act="gelu",
        rms_norm_eps=1e-6,
        rope_type="default",
        pad_token_id=0,
    )
    package = Seq2SeqTask().build(T5ForConditionalGeneration(config), config)
    decoder = package["decoder"]

    decoder_inputs = {value.name: value for value in decoder.graph.inputs}
    assert "encoder_attention_mask" not in decoder_inputs
    assert str(decoder_inputs["past_key_values.0.cross.key"].shape) == (
        "[batch,4,encoder_sequence_len,16]"
    )


def test_t5_encoder_and_decoder_use_independent_quantization():
    from mobius._builder import build_from_module
    from mobius._configs import ArchitectureConfig, QuantizationConfig
    from mobius.models.t5 import T5ForConditionalGeneration

    decoder = QuantizationConfig(
        bits=4,
        group_size=16,
        quant_method="olive",
        sym=True,
        quantize_embeddings=True,
    )
    config = ArchitectureConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=1,
        num_decoder_layers=1,
        vocab_size=256,
        max_position_embeddings=64,
        hidden_act="gelu",
        rms_norm_eps=1e-6,
        pad_token_id=0,
        quantization=decoder,
        component_quantization={
            "encoder": QuantizationConfig(
                bits=8,
                group_size=32,
                quant_method="olive",
                sym=True,
                quantize_embeddings=True,
            ),
            "decoder": decoder,
        },
    )

    package = build_from_module(
        T5ForConditionalGeneration(config),
        config,
        task="seq2seq",
    )

    def layouts(component: str) -> set[tuple[int, int]]:
        return {
            (
                node.attributes["bits"].as_int(),
                node.attributes["block_size"].as_int(),
            )
            for node in package[component].graph
            if node.op_type == "MatMulNBits"
        }

    assert layouts("encoder") == {(8, 32)}
    assert layouts("decoder") == {(4, 16)}

    def embedding_layout(component: str) -> tuple[int, int]:
        node = next(
            node for node in package[component].graph if node.op_type == "GatherBlockQuantized"
        )
        return (
            node.attributes["bits"].as_int(),
            node.attributes["block_size"].as_int(),
        )

    assert embedding_layout("encoder") == (8, 32)
    assert embedding_layout("decoder") == (4, 16)


def test_t5_component_preprocess_preserves_packed_shared_and_head_sidecars():
    from mobius._configs import ArchitectureConfig, QuantizationConfig
    from mobius.models.t5 import T5ForConditionalGeneration

    quantization = QuantizationConfig(
        bits=4,
        group_size=16,
        quant_method="olive",
        sym=True,
    )
    config = ArchitectureConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=1,
        num_decoder_layers=1,
        vocab_size=256,
        hidden_act="gelu",
        quantization=quantization,
        component_quantization={
            "encoder": quantization,
            "decoder": quantization,
        },
    )
    qweight = torch.zeros(256, 32, dtype=torch.uint8)
    scales = torch.ones(256, 4)

    result = T5ForConditionalGeneration(config).preprocess_weights(
        {
            "shared.weight_qweight": qweight,
            "shared.weight_scales": scales,
            "lm_head.weight_qweight": qweight,
            "lm_head.weight_scales": scales,
        }
    )

    assert result["encoder.embed_tokens.weight_qweight"] is qweight
    assert result["encoder.embed_tokens.weight_scales"] is scales
    assert result["decoder.embed_tokens.weight_qweight"] is qweight
    assert result["decoder.embed_tokens.weight_scales"] is scales
    assert result["decoder.lm_head.weight_qweight"] is qweight
    assert result["decoder.lm_head.weight_scales"] is scales


def test_t5_encoder_is_public_and_consumes_attention_mask():
    from mobius._configs import ArchitectureConfig
    from mobius.models import T5EncoderModel
    from mobius.tasks import T5TextEncoderTask

    config = ArchitectureConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        hidden_act="relu",
        relative_attention_num_buckets=8,
        relative_attention_max_distance=16,
    )
    graph = T5TextEncoderTask().build(T5EncoderModel(config), config)["model"].graph
    attention_mask = next(value for value in graph.inputs if value.name == "attention_mask")
    assert list(attention_mask.uses())
    assert any(node.op_type == "Unsqueeze" for node in graph)


class TestRelativePositionBucket:
    """Test T5 log-linear bucket computation against reference NumPy impl."""

    @staticmethod
    def _np_relative_position_bucket(
        relative_position: np.ndarray,
        *,
        bidirectional: bool,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> np.ndarray:
        """Reference NumPy implementation matching HuggingFace T5."""
        relative_buckets = np.zeros_like(relative_position, dtype=np.int64)
        if bidirectional:
            half = num_buckets // 2
            relative_buckets += (relative_position > 0).astype(np.int64) * half
            relative_position = np.abs(relative_position)
            effective_buckets = half
        else:
            relative_position = -np.minimum(
                relative_position, np.zeros_like(relative_position)
            )
            effective_buckets = num_buckets

        max_exact = effective_buckets // 2
        is_small = relative_position < max_exact
        rel_clamped = np.maximum(relative_position.astype(np.float32), 1.0)
        log_ratio = np.log(rel_clamped / max_exact)
        log_scale = math.log(max_distance / max_exact)
        bucket_float = max_exact + log_ratio * (effective_buckets - max_exact) / log_scale
        large_bucket = np.minimum(bucket_float.astype(np.int64), effective_buckets - 1)
        final_offset = np.where(is_small, relative_position, large_bucket)
        relative_buckets += final_offset
        return relative_buckets

    def test_bidirectional_4x4(self):
        """Encoder (bidirectional) bucket indices for 4x4."""
        ctx = np.arange(4)[:, None]
        mem = np.arange(4)[None, :]
        rel = mem - ctx
        buckets = self._np_relative_position_bucket(rel, bidirectional=True)
        expected = np.array(
            [
                [0, 17, 18, 19],
                [1, 0, 17, 18],
                [2, 1, 0, 17],
                [3, 2, 1, 0],
            ]
        )
        np.testing.assert_array_equal(buckets, expected)

    def test_unidirectional_4x4(self):
        """Decoder (unidirectional) bucket indices for 4x4."""
        ctx = np.arange(4)[:, None]
        mem = np.arange(4)[None, :]
        rel = mem - ctx
        buckets = self._np_relative_position_bucket(rel, bidirectional=False)
        expected = np.array(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [2, 1, 0, 0],
                [3, 2, 1, 0],
            ]
        )
        np.testing.assert_array_equal(buckets, expected)

    def test_decode_step_offset(self):
        """Decoder position bias for single-token decode at offset 3."""
        # query position = [3], key positions = [0, 1, 2, 3]
        ctx = np.array([[3]])
        mem = np.arange(4)[None, :]
        rel = mem - ctx  # [[-3, -2, -1, 0]]
        buckets = self._np_relative_position_bucket(rel, bidirectional=False)
        expected = np.array([[3, 2, 1, 0]])
        np.testing.assert_array_equal(buckets, expected)


class TestT5WeightRename:
    """Test weight name mapping from HuggingFace to ONNX."""

    def test_relative_attention_bias_keeps_encoder_layer_index(self):
        """Relative bias remains attached to its source encoder layer."""
        hf = "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
        assert _rename_t5_weight(hf) == "encoder.block.0.relative_attention_bias.weight"

    def test_relative_attention_bias_keeps_decoder_layer_index(self):
        """Relative bias remains attached to its source decoder layer."""
        hf = "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
        assert _rename_t5_weight(hf) == "decoder.block.0.relative_attention_bias.weight"

    def test_self_attention_projections(self):
        """Self-attention projections rename correctly."""
        hf = "encoder.block.2.layer.0.SelfAttention.q.weight"
        assert _rename_t5_weight(hf) == "encoder.block.2.self_attn.q_proj.weight"

    def test_preprocess_preserves_values_and_ties_shared_weights(self):
        from mobius._configs import ArchitectureConfig
        from mobius.models.t5 import T5ForConditionalGeneration

        config = ArchitectureConfig(
            vocab_size=8,
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=1,
            num_decoder_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=2,
            hidden_act="relu",
            tie_word_embeddings=True,
        )
        model = T5ForConditionalGeneration(config)
        shared = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        projection = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        relative_bias = torch.arange(64, dtype=torch.float32).reshape(32, 2)
        transformed = model.preprocess_weights(
            {
                "shared.weight": shared,
                "encoder.block.0.layer.0.SelfAttention.q.weight": projection,
                "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight": (
                    relative_bias
                ),
            }
        )

        assert transformed["encoder.block.0.self_attn.q_proj.weight"] is projection
        assert transformed["encoder.block.0.relative_attention_bias.weight"] is relative_bias
        assert transformed["encoder.embed_tokens.weight"] is shared
        assert transformed["decoder.embed_tokens.weight"] is shared
        assert transformed["decoder.lm_head.weight"] is shared

    @pytest.mark.parametrize("bias_layers", [[0], [0, 1]])
    def test_relative_bias_initializer_ownership_matches_source_layers(
        self, bias_layers: list[int]
    ):
        from mobius import build_from_module
        from mobius._configs import ArchitectureConfig
        from mobius.models.t5 import T5ForConditionalGeneration

        config = ArchitectureConfig(
            vocab_size=8,
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=2,
            num_decoder_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=2,
            hidden_act="relu",
            encoder_relative_attention_bias_layers=bias_layers,
            decoder_relative_attention_bias_layers=bias_layers,
        )
        package = build_from_module(T5ForConditionalGeneration(config), config, task="seq2seq")

        encoder_names = set(package["encoder"].graph.initializers)
        decoder_names = set(package["decoder"].graph.initializers)
        assert "encoder.block.0.relative_attention_bias.weight" in encoder_names
        assert "decoder.block.0.relative_attention_bias.weight" in decoder_names
        assert ("encoder.block.1.relative_attention_bias.weight" in encoder_names) == (
            1 in bias_layers
        )
        assert ("decoder.block.1.relative_attention_bias.weight" in decoder_names) == (
            1 in bias_layers
        )

    def test_cross_attention_projections(self):
        """Cross-attention projections rename correctly."""
        hf = "decoder.block.1.layer.1.EncDecAttention.k.weight"
        assert _rename_t5_weight(hf) == "decoder.block.1.cross_attn.k_proj.weight"

    def test_ffn_rename_encoder(self):
        """Encoder FFN weights rename correctly."""
        hf = "encoder.block.3.layer.1.DenseReluDense.wi.weight"
        assert _rename_t5_weight(hf) == "encoder.block.3.ffn.wi.weight"

    def test_ffn_rename_decoder(self):
        """Decoder FFN weights rename correctly."""
        hf = "decoder.block.0.layer.2.DenseReluDense.wo.weight"
        assert _rename_t5_weight(hf) == "decoder.block.0.ffn.wo.weight"

    @pytest.mark.parametrize(
        "hf_name",
        [
            "encoder.final_layer_norm.weight",
            "decoder.final_layer_norm.weight",
        ],
    )
    def test_final_layer_norm_unchanged(self, hf_name):
        """Final layer norms keep their names."""
        assert _rename_t5_weight(hf_name) == hf_name
