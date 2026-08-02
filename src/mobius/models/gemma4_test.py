# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Gemma4 preprocess_weights — MoE expert rename and router scale."""

from __future__ import annotations

import dataclasses

import onnx_ir as ir
import torch

from mobius._configs import AudioConfig, Gemma4AudioConfig, Gemma4Config, QuantizationConfig
from mobius.models.gemma4 import Gemma4CausalLMModel, Gemma4EmbeddingModel, Gemma4Model


def _tiny_gemma4_config(**overrides) -> Gemma4Config:
    """Create a minimal Gemma4Config for preprocess_weights tests."""
    from mobius._configs import VisionConfig

    defaults = dict(
        model_type="gemma4",
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        head_dim=16,
        global_head_dim=32,
        hidden_act="gelu",
        enable_moe_block=True,
        num_local_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        layer_types=["sliding_attention", "full_attention"],
        attention_k_eq_v=True,
        num_global_key_value_heads=1,
        vision=VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=16,
            patch_size=4,
        ),
    )
    defaults.update(overrides)
    return Gemma4Config(**defaults)


class TestGemma4CausalLMPreprocessWeights:
    """Test Gemma4CausalLMModel.preprocess_weights."""

    def test_expert_weight_rename(self):
        config = _tiny_gemma4_config()
        model = Gemma4CausalLMModel(config)

        fake_sd = {
            "model.layers.0.experts.gate_up_proj": torch.zeros(4, 64, 64),
            "model.layers.0.experts.down_proj": torch.zeros(4, 64, 32),
        }
        result = model.preprocess_weights(fake_sd)

        assert "model.layers.0.fc1_experts_weights" in result
        assert "model.layers.0.fc2_experts_weights" in result
        assert "model.layers.0.experts.gate_up_proj" not in result

    def test_router_scale_folding(self):
        config = _tiny_gemma4_config()
        model = Gemma4CausalLMModel(config)

        scale_val = torch.tensor([2.0])
        fake_sd = {"model.layers.0.router.scale": scale_val.clone()}
        result = model.preprocess_weights(fake_sd)

        expected = 2.0 * (64**-0.5)  # hidden_size=64
        assert abs(result["model.layers.0.router.scale"].item() - expected) < 1e-6


class TestGemma4ModelPreprocessWeights:
    """Test Gemma4Model.preprocess_weights (multimodal path)."""

    def test_expert_weight_rename(self):
        config = _tiny_gemma4_config()
        model = Gemma4Model(config)

        fake_sd = {
            "model.language_model.layers.0.experts.gate_up_proj": torch.zeros(4, 64, 64),
            "model.language_model.layers.0.experts.down_proj": torch.zeros(4, 64, 32),
        }
        result = model.preprocess_weights(fake_sd)

        assert "decoder.model.layers.0.fc1_experts_weights" in result
        assert "decoder.model.layers.0.fc2_experts_weights" in result

    def test_router_scale_folding(self):
        config = _tiny_gemma4_config()
        model = Gemma4Model(config)

        scale_val = torch.tensor([2.0])
        fake_sd = {"model.language_model.layers.0.router.scale": scale_val.clone()}
        result = model.preprocess_weights(fake_sd)

        expected = 2.0 * (64**-0.5)
        key = "decoder.model.layers.0.router.scale"
        assert key in result
        assert abs(result[key].item() - expected) < 1e-6

    def test_per_expert_scale_not_folded(self):
        """router.per_expert_scale should NOT be multiplied by scale_factor."""
        config = _tiny_gemma4_config()
        model = Gemma4Model(config)

        fake_sd = {
            "model.language_model.layers.0.router.per_expert_scale": torch.ones(4),
        }
        result = model.preprocess_weights(fake_sd)

        key = "decoder.model.layers.0.router.per_expert_scale"
        assert key in result
        assert torch.allclose(result[key], torch.ones(4))


class TestGemma4CompressedTensors:
    def test_uses_matmulnbits_for_packed_language_projections(self):
        from mobius.tasks._gemma4 import Gemma4Task

        config = _tiny_gemma4_config(
            enable_moe_block=False,
            attention_k_eq_v=False,
            num_kv_shared_layers=0,
            hidden_size_per_layer_input=32,
            vocab_size_per_layer_input=256,
            quantization=QuantizationConfig(
                bits=4,
                group_size=32,
                quant_method="compressed-tensors",
                sym=True,
                format="pack-quantized",
            ),
        )

        pkg = Gemma4Task().build(Gemma4Model(config), config)
        decoder_ops = [node.op_type for node in pkg["decoder"].graph]
        embedding_ops = [node.op_type for node in pkg["embedding"].graph]
        vision_ops = [node.op_type for node in pkg["vision_encoder"].graph]

        # Per decoder layer: Q/K/V/O (4), MLP (3), and PLE gate/proj (2).
        assert decoder_ops.count("MatMulNBits") == 2 * 9
        assert embedding_ops.count("MatMulNBits") == 1
        assert vision_ops.count("MatMulNBits") == 0
        assert "Compress" not in vision_ops


class TestGemma4Awq:
    @staticmethod
    def _config() -> Gemma4Config:
        return _tiny_gemma4_config(
            enable_moe_block=False,
            attention_k_eq_v=False,
            num_kv_shared_layers=0,
            hidden_size_per_layer_input=32,
            vocab_size_per_layer_input=256,
            quantization=QuantizationConfig(
                bits=4,
                group_size=32,
                quant_method="awq",
                sym=False,
            ),
        )

    def test_uses_matmulnbits_for_decoder_only(self):
        from mobius.tasks._gemma4 import Gemma4Task

        config = self._config()
        pkg = Gemma4Task().build(Gemma4Model(config), config)

        assert [node.op_type for node in pkg["decoder"].graph].count("MatMulNBits") == 2 * 9
        assert [node.op_type for node in pkg["embedding"].graph].count("MatMulNBits") == 0
        assert [node.op_type for node in pkg["vision_encoder"].graph].count("MatMulNBits") == 0

    def test_converts_and_routes_awq_decoder_weights(self):
        config = self._config()
        model = Gemma4Model(config)
        qweight = torch.randint(0, 255, (8, 64), dtype=torch.int32)
        scales = torch.randn(2, 64)
        qzeros = torch.full((1, 64), 0x05050505, dtype=torch.int32)
        vision_weight = torch.randn(32, 32)

        result = model.preprocess_weights(
            {
                "model.language_model.layers.0.self_attn.q_proj.qweight": qweight,
                "model.language_model.layers.0.self_attn.q_proj.scales": scales,
                "model.language_model.layers.0.self_attn.q_proj.qzeros": qzeros,
                "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight": vision_weight,
            }
        )

        prefix = "decoder.model.layers.0.self_attn.q_proj"
        assert result[f"{prefix}.weight"].shape == (64, 2, 16)
        assert result[f"{prefix}.weight"].dtype == torch.uint8
        assert result[f"{prefix}.scales"].shape == (64, 2)
        assert result[f"{prefix}.zero_points"].shape == (64, 1)
        assert (result[f"{prefix}.zero_points"] == 4).all()
        assert (
            result["vision_encoder.encoder.layers.0.self_attn.q_proj.weight"] is vision_weight
        )
        assert not any(key.endswith((".qweight", ".qzeros")) for key in result)

    def test_preserves_tied_float_embedding_and_lm_head(self):
        config = dataclasses.replace(self._config(), tie_word_embeddings=True)
        model = Gemma4Model(config)
        embedding = torch.randn(256, 64)

        result = model.preprocess_weights(
            {"model.language_model.embed_tokens.weight": embedding}
        )

        assert result["embedding.embed_tokens.weight"] is embedding
        assert result["decoder.model.embed_tokens.weight"] is embedding
        assert result["decoder.lm_head.weight"] is embedding


class TestGemma4QuarkAwq:
    @staticmethod
    def _config() -> Gemma4Config:
        return _tiny_gemma4_config(
            enable_moe_block=False,
            attention_k_eq_v=False,
            num_kv_shared_layers=0,
            hidden_size_per_layer_input=32,
            vocab_size_per_layer_input=256,
            quantization=QuantizationConfig(
                bits=4,
                group_size=32,
                quant_method="quark",
                sym=False,
            ),
        )

    def test_uses_matmulnbits_for_decoder_only(self):
        from mobius.tasks._gemma4 import Gemma4Task

        config = self._config()
        pkg = Gemma4Task().build(Gemma4Model(config), config)

        assert [node.op_type for node in pkg["decoder"].graph].count("MatMulNBits") == 2 * 9
        assert [node.op_type for node in pkg["embedding"].graph].count("MatMulNBits") == 0
        assert [node.op_type for node in pkg["vision_encoder"].graph].count("MatMulNBits") == 0

    def test_converts_quark_native_decoder_weights(self):
        config = self._config()
        model = Gemma4Model(config)
        qweight = torch.zeros(64, 8, dtype=torch.int32)
        scales = torch.randn(2, 64)
        qzeros = torch.zeros(2, 8, dtype=torch.int32)
        vision_weight = torch.randn(32, 32)

        result = model.preprocess_weights(
            {
                "model.language_model.layers.0.self_attn.q_proj.weight": qweight,
                "model.language_model.layers.0.self_attn.q_proj.weight_scale": scales,
                "model.language_model.layers.0.self_attn.q_proj.weight_zero_point": qzeros,
                "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight": vision_weight,
            }
        )

        prefix = "decoder.model.layers.0.self_attn.q_proj"
        assert result[f"{prefix}.weight"].shape == (64, 2, 16)
        assert result[f"{prefix}.weight"].dtype == torch.uint8
        assert result[f"{prefix}.scales"].shape == (64, 2)
        assert result[f"{prefix}.zero_points"].shape == (64, 1)
        assert (
            result["vision_encoder.encoder.layers.0.self_attn.q_proj.weight"] is vision_weight
        )


class TestGemma4OlivePacked:
    """Olive-native GPTQ checkpoints use ORT-oriented uint8 packed tensors."""

    @staticmethod
    def _config(
        *,
        quantize_lm_head: bool = False,
        quantize_embeddings: bool = False,
        tie_word_embeddings: bool = False,
        modules_to_not_convert: list[str] | None = None,
        audio: Gemma4AudioConfig | None = None,
    ) -> Gemma4Config:
        return _tiny_gemma4_config(
            enable_moe_block=False,
            attention_k_eq_v=False,
            num_kv_shared_layers=0,
            hidden_size_per_layer_input=32,
            vocab_size_per_layer_input=256,
            tie_word_embeddings=tie_word_embeddings,
            audio=audio,
            quantization=QuantizationConfig(
                bits=4,
                group_size=32,
                quant_method="olive",
                sym=False,
                quantize_embeddings=quantize_embeddings,
                quantize_lm_head=quantize_lm_head,
                modules_to_not_convert=modules_to_not_convert,
            ),
        )

    def test_uses_matmulnbits_for_decoder_projections(self):
        from mobius.tasks._gemma4 import Gemma4Task

        config = self._config()
        pkg = Gemma4Task().build(Gemma4Model(config), config)
        decoder_ops = [node.op_type for node in pkg["decoder"].graph]
        embedding_ops = [node.op_type for node in pkg["embedding"].graph]
        vision_ops = [node.op_type for node in pkg["vision_encoder"].graph]

        # Per decoder layer: Q/K/V/O (4), MLP (3), and PLE gate/proj (2).
        assert decoder_ops.count("MatMulNBits") == 2 * 9
        assert embedding_ops.count("MatMulNBits") == 0
        assert vision_ops.count("MatMulNBits") == 0

    def test_converts_and_routes_packed_decoder_weights(self):
        config = self._config()
        model = Gemma4Model(config)
        qweight = torch.randint(0, 255, (64, 32), dtype=torch.uint8)
        scales = torch.randn(64, 2)
        qzeros = torch.randint(0, 255, (64, 1), dtype=torch.uint8)

        result = model.preprocess_weights(
            {
                "model.language_model.layers.0.self_attn.q_proj.qweight": qweight,
                "model.language_model.layers.0.self_attn.q_proj.scales": scales,
                "model.language_model.layers.0.self_attn.q_proj.qzeros": qzeros,
            }
        )

        prefix = "decoder.model.layers.0.self_attn.q_proj"
        assert result[f"{prefix}.weight"].shape == (64, 2, 16)
        assert result[f"{prefix}.weight"].dtype == torch.uint8
        assert result[f"{prefix}.scales"] is scales
        assert result[f"{prefix}.zero_points"] is qzeros
        assert not any(key.endswith((".qweight", ".qzeros")) for key in result)

    def test_quantizes_lm_head_when_checkpoint_requests_it(self):
        from mobius.tasks._gemma4 import Gemma4Task

        config = self._config(quantize_lm_head=True)
        pkg = Gemma4Task().build(Gemma4Model(config), config)
        decoder_ops = [node.op_type for node in pkg["decoder"].graph]

        assert decoder_ops.count("MatMulNBits") == 2 * 9 + 1

    def test_routes_top_level_olive_packed_lm_head(self):
        config = self._config(quantize_lm_head=True)
        model = Gemma4Model(config)
        qweight = torch.randint(0, 255, (256, 32), dtype=torch.uint8)
        scales = torch.randn(256, 2)
        qzeros = torch.randint(0, 255, (256, 1), dtype=torch.uint8)

        result = model.preprocess_weights(
            {
                "lm_head.qweight": qweight,
                "lm_head.scales": scales,
                "lm_head.qzeros": qzeros,
            }
        )

        assert result["decoder.lm_head.weight"].shape == (256, 2, 16)
        assert result["decoder.lm_head.scales"] is scales
        assert result["decoder.lm_head.zero_points"] is qzeros

    def test_tied_quantized_lm_head_is_not_overwritten_by_float_embedding(self):
        config = self._config(quantize_lm_head=True, tie_word_embeddings=True)
        model = Gemma4Model(config)
        embed = torch.randn(256, 64)
        qweight = torch.randint(0, 255, (256, 32), dtype=torch.uint8)
        scales = torch.randn(256, 2)
        qzeros = torch.randint(0, 255, (256, 1), dtype=torch.uint8)

        result = model.preprocess_weights(
            {
                "model.language_model.embed_tokens.weight": embed,
                "model.language_model.lm_head.qweight": qweight,
                "model.language_model.lm_head.scales": scales,
                "model.language_model.lm_head.qzeros": qzeros,
            }
        )

        assert result["decoder.lm_head.weight"].shape == (256, 2, 16)
        assert result["decoder.lm_head.weight"].dtype == torch.uint8
        assert result["embedding.embed_tokens.weight"] is embed

    def test_rejects_quantized_token_embeddings(self):
        import pytest

        config = self._config(quantize_embeddings=True)
        with pytest.raises(NotImplementedError, match="quantized token embeddings"):
            Gemma4Model(config)

    def test_uses_matmulnbits_for_all_multimodal_linear_components(self):
        from mobius.tasks._gemma4 import Gemma4Task

        config = self._config(
            quantize_lm_head=True,
            modules_to_not_convert=[],
            audio=Gemma4AudioConfig(
                num_layers=1,
                hidden_size=32,
                output_proj_dims=64,
                subsampling_conv_channels=[8, 4],
                audio_token_id=240,
            ),
        )
        pkg = Gemma4Task().build(Gemma4Model(config), config)

        assert [node.op_type for node in pkg["decoder"].graph].count("MatMulNBits") == 19
        assert [node.op_type for node in pkg["embedding"].graph].count("MatMulNBits") == 1
        assert [node.op_type for node in pkg["vision_encoder"].graph].count("MatMulNBits") == 9
        assert [node.op_type for node in pkg["audio_encoder"].graph].count("MatMulNBits") == 14

    def test_component_exclusions_keep_multimodal_encoders_float(self):
        from mobius.tasks._gemma4 import Gemma4Task

        config = self._config(
            modules_to_not_convert=[
                "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear",
                "model.audio_tower.layers.0.self_attn.q_proj.linear",
                "model.language_model.per_layer_model_projection",
            ],
            audio=Gemma4AudioConfig(
                num_layers=1,
                hidden_size=32,
                output_proj_dims=64,
                subsampling_conv_channels=[8, 4],
                audio_token_id=240,
            ),
        )
        pkg = Gemma4Task().build(Gemma4Model(config), config)

        assert [node.op_type for node in pkg["decoder"].graph].count("MatMulNBits") == 18
        assert [node.op_type for node in pkg["embedding"].graph].count("MatMulNBits") == 0
        assert [node.op_type for node in pkg["vision_encoder"].graph].count("MatMulNBits") == 0
        assert [node.op_type for node in pkg["audio_encoder"].graph].count("MatMulNBits") == 0

    def test_converts_and_routes_packed_multimodal_weights(self):
        config = self._config(
            modules_to_not_convert=[],
            audio=Gemma4AudioConfig(
                num_layers=1,
                hidden_size=32,
                output_proj_dims=64,
                subsampling_conv_channels=[8, 4],
                audio_token_id=240,
            ),
        )
        model = Gemma4Model(config)
        qweight = torch.randint(0, 255, (32, 32), dtype=torch.uint8)
        scales = torch.randn(32, 2)
        qzeros = torch.randint(0, 255, (32, 1), dtype=torch.uint8)

        result = model.preprocess_weights(
            {
                "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.qweight": qweight,
                "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.scales": scales,
                "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.qzeros": qzeros,
                "model.audio_tower.layers.0.self_attn.q_proj.linear.qweight": qweight,
                "model.audio_tower.layers.0.self_attn.q_proj.linear.scales": scales,
                "model.audio_tower.layers.0.self_attn.q_proj.linear.qzeros": qzeros,
            }
        )

        for prefix in (
            "vision_encoder.encoder.layers.0.self_attn.q_proj",
            "audio_encoder.encoder.layers.0.self_attn.q_proj",
        ):
            assert result[f"{prefix}.weight"].shape == (32, 2, 16)
            assert result[f"{prefix}.scales"] is scales
            assert result[f"{prefix}.zero_points"] is qzeros


class TestGemma4EmbeddingModel:
    def test_reuses_token_id_fields_for_masking(self):
        config = _tiny_gemma4_config(
            image_token_id=200010,
            audio=AudioConfig(audio_token_id=200011),
        )
        model = Gemma4EmbeddingModel(config)

        assert model.image_token_id == 200010
        assert model.audio_token_id == 200011
        assert not hasattr(model, "_image_token_id_mask")
        assert not hasattr(model, "_audio_token_id_mask")


class TestScaleFreeRMSNormOverflow:
    """V norm should handle FP16 overflow from squaring large values."""

    def test_vnorm_fp16_no_nan(self):
        """Values ~888 overflow FP16 when squared (888²=788K > 65504).

        The scale-free RMSNorm must use stash_type=1 (float32 accumulation)
        to avoid inf/NaN from the variance computation.
        """
        import numpy as np

        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.models.gemma4 import _Gemma4ScaleFreeRMSNorm

        dim = 64
        norm = _Gemma4ScaleFreeRMSNorm(dim, eps=1e-6)

        # Build a minimal ONNX graph for the norm
        from mobius.tasks._base import _make_graph, _make_model

        graph, builder = _make_graph()
        op = builder.op
        x = builder.input("x", dtype=ir.DataType.FLOAT16, shape=[1, 4, dim])
        y = norm(op, x)
        builder.add_output(y, "y")
        model = _make_model(graph)

        session = OnnxModelSession(model, device="cpu")

        # Values that overflow FP16 when squared: 888² = 788,544 > 65504
        test_input = np.full((1, 4, dim), 888.0, dtype=np.float16)
        result = session.run({"x": test_input})
        output = result["y"]

        assert not np.any(np.isnan(output)), "V norm produced NaN for input 888"
        assert not np.any(np.isinf(output)), "V norm produced Inf for input 888"
        # RMSNorm of a constant vector: x/rms(x) = sign(x) ≈ 1.0
        np.testing.assert_allclose(
            output.astype(np.float32),
            np.ones_like(output, dtype=np.float32),
            atol=0.01,
        )


class TestGemma4BlockSequenceIds:
    """Vision-block bidirectional attention wiring (use_bidirectional_attention)."""

    def test_compute_block_sequence_ids_values(self):
        """``_compute_block_sequence_ids`` matches HF get_block_sequence_ids_for_mask."""
        import numpy as np

        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.models.gemma4 import _compute_block_sequence_ids
        from mobius.tasks._base import _make_graph, _make_model

        graph, builder = _make_graph()
        op = builder.op
        input_ids = builder.input("input_ids", dtype=ir.DataType.INT64, shape=[1, "S"])
        out = _compute_block_sequence_ids(op, input_ids, image_token_id=255)
        builder.add_output(out, "block_sequence_ids")
        session = OnnxModelSession(_make_model(graph), device="cpu")

        # text  img img  text  aud aud  text  img  text
        ids = np.array([[10, 255, 255, 11, 254, 254, 11, 255, 12]], dtype=np.int64)
        result = session.run({"input_ids": ids})["block_sequence_ids"]
        # Only image runs form blocks (groups 0, 1). Audio (254) and text -> -1,
        # matching HF where audio is token-type 3 and excluded from is_vision.
        expected = np.array([[-1, 0, 0, -1, -1, -1, -1, 1, -1]], dtype=np.int64)
        np.testing.assert_array_equal(result, expected)

    def test_unsupported_bidirectional_mode_raises(self):
        """``use_bidirectional_attention='all'`` is rejected, not silently causal."""
        import pytest

        from mobius.models.gemma4 import Gemma4TextModel

        config = _tiny_gemma4_config(use_bidirectional_attention="all")
        with pytest.raises(NotImplementedError, match="use_bidirectional_attention"):
            Gemma4TextModel(config)

    def test_audio_tokens_excluded_from_blocks(self):
        """Audio placeholders never join a vision block (HF parity).

        HF derives ``is_vision`` from ``mm_token_type_ids`` as ``(==1)|(==2)``
        (image or video); audio is token-type ``3`` and is excluded, so an audio
        run adjacent to an image run does NOT extend the block.
        """
        import numpy as np

        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.models.gemma4 import _compute_block_sequence_ids
        from mobius.tasks._base import _make_graph, _make_model

        graph, builder = _make_graph()
        op = builder.op
        input_ids = builder.input("input_ids", dtype=ir.DataType.INT64, shape=[1, "S"])
        out = _compute_block_sequence_ids(op, input_ids, image_token_id=255)
        builder.add_output(out, "block_sequence_ids")
        session = OnnxModelSession(_make_model(graph), device="cpu")

        ids = np.array([[10, 255, 255, 254, 254, 11]], dtype=np.int64)
        result = session.run({"input_ids": ids})["block_sequence_ids"]
        # Image run -> block 0; the adjacent audio run (254) stays -1 (causal).
        expected = np.array([[-1, 0, 0, -1, -1, -1]], dtype=np.int64)
        np.testing.assert_array_equal(result, expected)

    def test_package_wires_block_sequence_ids_end_to_end(self):
        """Decoder takes input_ids and derives the vision-block overlay itself.

        With ``use_bidirectional_attention='vision'`` the embedding no longer
        emits ``block_sequence_ids``; instead the decoder receives ``input_ids``
        (alongside ``inputs_embeds``) and computes the overlay internally. This
        avoids a cross-model tensor that onnxruntime-genai cannot forward.
        """
        from mobius.models.gemma4 import Gemma4Model
        from mobius.tasks._gemma4 import Gemma4Task

        config = _tiny_gemma4_config(use_bidirectional_attention="vision", image_token_id=255)
        pkg = Gemma4Task().build(Gemma4Model(config), config)

        emb_outputs = {o.name for o in pkg["embedding"].graph.outputs}
        dec_inputs = {i.name for i in pkg["decoder"].graph.inputs}
        assert "block_sequence_ids" not in emb_outputs
        assert "block_sequence_ids" not in dec_inputs
        assert "input_ids" in dec_inputs

        # Decoder attention must drop GQA and disable the op's built-in causal
        # mask (is_causal=0) so the baked blockwise bias is honored.
        dec = pkg["decoder"].graph
        assert not any(n.op_type == "GroupQueryAttention" for n in dec)
        attn_nodes = [n for n in dec if n.op_type == "Attention"]
        assert attn_nodes
        for n in attn_nodes:
            assert n.attributes["is_causal"].as_int() == 0

    def test_no_block_sequence_ids_when_causal(self):
        """Without bidirectional attention, no input_ids/overlay is wired."""
        from mobius.models.gemma4 import Gemma4Model
        from mobius.tasks._gemma4 import Gemma4Task

        config = _tiny_gemma4_config(use_bidirectional_attention=None)
        pkg = Gemma4Task().build(Gemma4Model(config), config)

        emb_outputs = {o.name for o in pkg["embedding"].graph.outputs}
        dec_inputs = {i.name for i in pkg["decoder"].graph.inputs}
        assert "block_sequence_ids" not in emb_outputs
        assert "block_sequence_ids" not in dec_inputs
        assert "input_ids" not in dec_inputs


def _tiny_gemma4_unified_config(**overrides) -> Gemma4Config:
    """Minimal gemma4_unified config (dense decoder + encoder-free embedders)."""
    from mobius._configs import Gemma4AudioConfig, VisionConfig

    base = dict(
        model_type="gemma4_unified",
        enable_moe_block=False,
        tie_word_embeddings=True,
        use_bidirectional_attention="vision",
        image_token_id=255,
        vision=VisionConfig(
            hidden_size=32,
            position_embedding_size=1120,
            patch_size=4,
            pooling_kernel_size=3,
            out_hidden_size=32,
            norm_eps=1e-6,
        ),
        audio=Gemma4AudioConfig(hidden_size=16, output_proj_dims=16, audio_token_id=254),
    )
    base.update(overrides)
    return _tiny_gemma4_config(**base)


class TestGemma4UnifiedPreprocessWeights:
    """Gemma4UnifiedModel.preprocess_weights — checkpoint name mapping."""

    def test_full_checkpoint_rename(self):
        from mobius.models.gemma4 import Gemma4UnifiedModel

        config = _tiny_gemma4_unified_config()
        model = Gemma4UnifiedModel(config)

        # pos_embedding stored as [posemb, 2, mm_embed_dim]; x-axis = [:, 0, :],
        # y-axis = [:, 1, :]. Use distinct constants to verify the split.
        pos_embedding = torch.empty(1120, 2, 32)
        pos_embedding[:, 0, :] = 1.0
        pos_embedding[:, 1, :] = 2.0

        fake_sd = {
            "model.language_model.embed_tokens.weight": torch.zeros(256, 64),
            "model.language_model.layers.0.input_layernorm.weight": torch.zeros(64),
            "model.vision_embedder.patch_ln1.weight": torch.zeros(432),
            "model.vision_embedder.patch_dense.weight": torch.zeros(32, 432),
            "model.vision_embedder.pos_embedding": pos_embedding,
            "model.vision_embedder.pos_norm.weight": torch.zeros(32),
            "model.embed_vision.embedding_projection.weight": torch.zeros(64, 32),
            "model.embed_audio.embedding_projection.weight": torch.zeros(64, 16),
            # Scale-free RMSNorms have no learnable weight in the checkpoint, but
            # assert they are dropped even if a stray key appears.
            "model.embed_vision.embedding_pre_projection_norm.weight": torch.zeros(32),
            "model.embed_audio.embedding_pre_projection_norm.weight": torch.zeros(16),
        }
        result = model.preprocess_weights(fake_sd)

        # Text backbone → decoder.model.* and tied embedding/lm_head.
        assert "decoder.model.embed_tokens.weight" in result
        assert "embedding.embed_tokens.weight" in result
        assert "decoder.lm_head.weight" in result  # synthesized from tied embed
        assert "decoder.model.layers.0.input_layernorm.weight" in result

        # Vision front-end → vision_encoder.*; pos_embedding split into x/y.
        assert "vision_encoder.patch_ln1.weight" in result
        assert "vision_encoder.patch_dense.weight" in result
        assert "vision_encoder.pos_norm.weight" in result
        assert result["vision_encoder.pos_emb_x.weight"].shape == (1120, 32)
        assert result["vision_encoder.pos_emb_y.weight"].shape == (1120, 32)
        assert torch.allclose(result["vision_encoder.pos_emb_x.weight"], torch.tensor(1.0))
        assert torch.allclose(result["vision_encoder.pos_emb_y.weight"], torch.tensor(2.0))

        # Projections → vision_encoder.projector / audio_encoder.projector.
        assert "vision_encoder.projector.weight" in result
        assert "audio_encoder.projector.weight" in result

        # Scale-free pre-projection norms must be dropped (no graph initializer).
        assert not any("embedding_pre_projection_norm" in k for k in result)
        # The raw checkpoint module prefixes must not leak through.
        assert not any(k.startswith("vision_embedder.") for k in result)
        assert not any(k.startswith("embed_vision.") for k in result)
        assert not any(k.startswith("language_model.") for k in result)

    def test_routes_olive_packed_per_layer_projection_to_embedding(self):
        from mobius.models.gemma4 import Gemma4UnifiedModel

        config = _tiny_gemma4_unified_config(
            hidden_size_per_layer_input=32,
            quantization=QuantizationConfig(
                bits=4,
                group_size=32,
                quant_method="olive",
                sym=False,
            ),
        )
        model = Gemma4UnifiedModel(config)
        qweight = torch.randint(0, 255, (64, 32), dtype=torch.uint8)
        scales = torch.randn(64, 2)
        qzeros = torch.randint(0, 255, (64, 1), dtype=torch.uint8)

        result = model.preprocess_weights(
            {
                "model.language_model.per_layer_model_projection.qweight": qweight,
                "model.language_model.per_layer_model_projection.scales": scales,
                "model.language_model.per_layer_model_projection.qzeros": qzeros,
            }
        )

        prefix = "embedding.per_layer_model_projection"
        assert result[f"{prefix}.weight"].shape == (64, 2, 16)
        assert result[f"{prefix}.scales"] is scales
        assert result[f"{prefix}.zero_points"] is qzeros

    def test_vision_embedder_preprocess_standalone(self):
        from mobius.models.gemma4 import _Gemma4UnifiedVisionEmbedderModel

        config = _tiny_gemma4_unified_config()
        embedder = _Gemma4UnifiedVisionEmbedderModel(config)

        pos_embedding = torch.empty(1120, 2, 32)
        pos_embedding[:, 0, :] = 3.0
        pos_embedding[:, 1, :] = 4.0
        fake_sd = {
            "vision_embedder.patch_ln1.weight": torch.zeros(432),
            "vision_embedder.pos_embedding": pos_embedding,
            "embed_vision.embedding_projection.weight": torch.zeros(64, 32),
            "embed_vision.embedding_pre_projection_norm.weight": torch.zeros(32),
        }
        result = embedder.preprocess_weights(fake_sd)

        assert "patch_ln1.weight" in result
        assert "projector.weight" in result
        assert torch.allclose(result["pos_emb_x.weight"], torch.tensor(3.0))
        assert torch.allclose(result["pos_emb_y.weight"], torch.tensor(4.0))
        assert not any("embedding_pre_projection_norm" in k for k in result)

    def test_audio_embedder_preprocess_standalone(self):
        from mobius.models.gemma4 import _Gemma4UnifiedAudioEmbedderModel

        config = _tiny_gemma4_unified_config()
        embedder = _Gemma4UnifiedAudioEmbedderModel(config)

        fake_sd = {
            "embed_audio.embedding_projection.weight": torch.zeros(64, 16),
            "embed_audio.embedding_pre_projection_norm.weight": torch.zeros(16),
        }
        result = embedder.preprocess_weights(fake_sd)

        assert result["projector.weight"].shape == (64, 16)
        assert not any("embedding_pre_projection_norm" in k for k in result)


class TestGemma4UnifiedConfigHooks:
    """Config extraction hooks map unified vision/audio sub-configs."""

    def test_vision_hook_maps_fields(self):
        from types import SimpleNamespace

        from mobius._configs.per_model._gemma4_unified_vision import (
            _gemma4_unified_vision,
        )

        composite = SimpleNamespace(
            model_type="gemma4_unified",
            image_token_id=258880,
            vision_config=SimpleNamespace(
                mm_embed_dim=3840,
                patch_size=16,
                pooling_kernel_size=3,
                mm_posemb_size=1120,
                output_proj_dims=3840,
                rms_norm_eps=1e-6,
            ),
        )
        fields: dict = {}
        _gemma4_unified_vision(composite, None, "gemma4_unified", fields)

        assert fields["hidden_size"] == 3840
        assert fields["patch_size"] == 16
        assert fields["pooling_kernel_size"] == 3
        assert fields["position_embedding_size"] == 1120
        assert fields["out_hidden_size"] == 3840
        assert fields["image_token_id"] == 258880

    def test_vision_hook_skips_unrelated_model(self):
        from types import SimpleNamespace

        from mobius._configs.per_model._gemma4_unified_vision import (
            _gemma4_unified_vision,
        )

        composite = SimpleNamespace(model_type="qwen2_vl", vision_config=object())
        fields: dict = {}
        result = _gemma4_unified_vision(composite, None, "qwen2_vl", fields)
        assert result is None
        assert fields == {}

    def test_audio_hook_maps_fields(self):
        from types import SimpleNamespace

        from mobius._configs.per_model._gemma4_unified_audio import (
            _gemma4_unified_audio,
        )

        composite = SimpleNamespace(
            model_type="gemma4_unified",
            audio_token_id=258881,
            audio_config=SimpleNamespace(audio_embed_dim=640),
        )
        result = _gemma4_unified_audio(composite, None, "gemma4_unified", {})

        assert result is not None
        audio_cfg = result["audio"]
        assert audio_cfg.hidden_size == 640
        assert audio_cfg.output_proj_dims == 640
        assert audio_cfg.audio_token_id == 258881
