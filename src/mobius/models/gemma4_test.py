# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Gemma4 preprocess_weights — MoE expert rename and router scale."""

from __future__ import annotations

import dataclasses

import onnx_ir as ir
import pytest
import torch

from mobius._configs import (
    AudioConfig,
    Gemma4AudioConfig,
    Gemma4Config,
    QuantizationConfig,
)
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

    def test_quantized_moe_experts_fail_closed(self):
        config = _tiny_gemma4_config(
            quantization=QuantizationConfig(
                bits=4,
                group_size=16,
                quant_method="olive",
                sym=True,
            )
        )
        state_dict = {
            "model.layers.0.experts.gate_up_proj_qweight": torch.zeros(
                4, 64, 32, dtype=torch.uint8
            )
        }

        with pytest.raises(NotImplementedError, match="Quantized Gemma4 MoE experts"):
            Gemma4CausalLMModel(config).preprocess_weights(state_dict)


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

    def test_quantized_moe_experts_fail_closed(self):
        config = _tiny_gemma4_config(
            quantization=QuantizationConfig(
                bits=4,
                group_size=16,
                quant_method="olive",
                sym=True,
            )
        )
        state_dict = {
            "model.language_model.layers.0.experts.down_proj.weight_scales": torch.ones(
                4, 64, 2
            )
        }

        with pytest.raises(NotImplementedError, match="Quantized Gemma4 MoE experts"):
            Gemma4Model(config).preprocess_weights(state_dict)

    def test_olive_quantized_decoder_sidecars_are_preprocessed(self):
        config = _tiny_gemma4_config(
            enable_moe_block=False,
            quantization=QuantizationConfig(
                bits=4,
                group_size=16,
                quant_method="olive",
                sym=True,
            ),
        )
        model = Gemma4Model(config)
        fake_sd = {
            "model.language_model.layers.0.self_attn.q_proj.weight_qweight": torch.zeros(
                64, 32, dtype=torch.uint8
            ),
            "model.language_model.layers.0.self_attn.q_proj.weight_scales": torch.ones(64, 4),
        }

        result = model.preprocess_weights(fake_sd)

        weight_key = "decoder.model.layers.0.self_attn.q_proj.weight"
        scales_key = "decoder.model.layers.0.self_attn.q_proj.scales"
        assert result[weight_key].shape == (64, 4, 8)
        assert result[weight_key].dtype == torch.uint8
        assert (
            result[scales_key]
            is fake_sd["model.language_model.layers.0.self_attn.q_proj.weight_scales"]
        )

    def test_olive_quantized_vision_sidecars_are_preprocessed(self):
        config = _tiny_gemma4_config(
            enable_moe_block=False,
            quantization=QuantizationConfig(
                bits=4,
                group_size=16,
                quant_method="olive",
                sym=True,
                quantize_vision=True,
            ),
        )
        model = Gemma4Model(config)
        fake_sd = {
            "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight_qweight": torch.zeros(
                32, 16, dtype=torch.uint8
            ),
            "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight_scales": torch.ones(
                32, 2
            ),
            "model.embed_vision.embedding_projection.weight_qweight": torch.zeros(
                64, 16, dtype=torch.uint8
            ),
            "model.embed_vision.embedding_projection.weight_scales": torch.ones(64, 2),
        }

        result = model.preprocess_weights(fake_sd)

        assert result["vision_encoder.encoder.layers.0.self_attn.q_proj.weight"].shape == (
            32,
            2,
            8,
        )
        assert result["vision_encoder.encoder.layers.0.self_attn.q_proj.scales"].shape == (
            32,
            2,
        )
        assert result["vision_encoder.projector.weight"].shape == (64, 2, 8)
        assert result["vision_encoder.projector.scales"].shape == (64, 2)

    def test_top_level_lm_head_routes_to_decoder(self):
        model = Gemma4Model(_tiny_gemma4_config(tie_word_embeddings=False))
        weight = torch.randn(256, 64)

        result = model.preprocess_weights({"lm_head.weight": weight})

        assert result["decoder.lm_head.weight"] is weight

    @pytest.mark.parametrize(
        ("quantize_embeddings", "quantize_lm_head", "state_dict"),
        [
            (
                True,
                False,
                {
                    "model.language_model.embed_tokens.weight_qweight": torch.zeros(
                        256, 32, dtype=torch.uint8
                    )
                },
            ),
            (
                False,
                True,
                {"lm_head.weight_qweight": torch.zeros(256, 32, dtype=torch.uint8)},
            ),
        ],
    )
    def test_quantized_embedding_or_lm_head_fails_closed(
        self,
        quantize_embeddings,
        quantize_lm_head,
        state_dict,
    ):
        config = _tiny_gemma4_config(
            tie_word_embeddings=False,
            quantization=QuantizationConfig(
                bits=4,
                group_size=16,
                quant_method="olive",
                sym=True,
                quantize_embeddings=quantize_embeddings,
                quantize_lm_head=quantize_lm_head,
            ),
        )

        with pytest.raises(
            NotImplementedError,
            match="Quantized embeddings and LM heads are not yet supported",
        ):
            Gemma4Model(config).preprocess_weights(state_dict)

    def test_gguf_quantized_tables_keep_canonical_weight_path(self):
        config = _tiny_gemma4_config(
            quantization=QuantizationConfig(
                bits=4,
                group_size=16,
                quant_method="gguf",
                sym=True,
                quantize_embeddings=True,
                quantize_lm_head=True,
            ),
        )

        assert Gemma4Model(config).preprocess_weights({}) == {}

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


class TestGemma4VisionQuantization:
    def test_quantize_vision_emits_matmulnbits_and_keeps_activation_clipping(self):
        from mobius.tasks._gemma4 import Gemma4Task

        config = _tiny_gemma4_config(
            enable_moe_block=False,
            vision=dataclasses.replace(
                _tiny_gemma4_config().vision,
                use_clipped_linears=True,
            ),
            quantization=QuantizationConfig(
                bits=4,
                group_size=16,
                quant_method="olive",
                sym=True,
                quantize_vision=True,
            ),
        )

        graph = Gemma4Task().build(Gemma4Model(config), config)["vision_encoder"].graph
        op_types = [node.op_type for node in graph]

        assert op_types.count("MatMulNBits") == 9
        assert op_types.count("Clip") >= 16

    def test_quantized_decoder_does_not_quantize_vision_by_default(self):
        from mobius.tasks._gemma4 import Gemma4Task

        config = _tiny_gemma4_config(
            enable_moe_block=False,
            quantization=QuantizationConfig(
                bits=4,
                group_size=16,
                quant_method="olive",
                sym=True,
            ),
        )

        graph = Gemma4Task().build(Gemma4Model(config), config)["vision_encoder"].graph

        assert not any(node.op_type == "MatMulNBits" for node in graph)

    def test_unrelated_module_plan_does_not_quantize_vision(self):
        from mobius.tasks._gemma4 import Gemma4Task

        config = _tiny_gemma4_config(
            enable_moe_block=False,
            quantization=QuantizationConfig(
                bits=4,
                group_size=16,
                quant_method="olive",
                sym=True,
                modules_to_not_convert=("lm_head",),
            ),
        )

        graph = Gemma4Task().build(Gemma4Model(config), config)["vision_encoder"].graph

        assert not any(node.op_type == "MatMulNBits" for node in graph)


class TestGemma4PerComponentQuantization:
    @staticmethod
    def _config() -> Gemma4Config:
        decoder = QuantizationConfig(
            bits=4,
            group_size=16,
            quant_method="olive",
            sym=True,
        )
        return _tiny_gemma4_config(
            enable_moe_block=False,
            hidden_size_per_layer_input=16,
            vocab_size_per_layer_input=256,
            audio=Gemma4AudioConfig(
                input_size=32,
                num_layers=1,
                hidden_size=32,
                output_proj_dims=64,
                subsampling_conv_channels=[16, 8],
                audio_token_id=254,
            ),
            quantization=decoder,
            component_quantization={
                "decoder": decoder,
                "vision_encoder": QuantizationConfig(
                    bits=8,
                    group_size=32,
                    quant_method="olive",
                    sym=True,
                ),
                "audio_encoder": QuantizationConfig(
                    bits=2,
                    group_size=16,
                    quant_method="olive",
                    sym=True,
                ),
                "embedding": QuantizationConfig(
                    bits=8,
                    group_size=16,
                    quant_method="olive",
                    sym=True,
                ),
            },
        )

    @staticmethod
    def _layouts(graph) -> set[tuple[int, int]]:
        return {
            (
                node.attributes["bits"].as_int(),
                node.attributes["block_size"].as_int(),
            )
            for node in graph
            if node.op_type == "MatMulNBits"
        }

    def test_each_component_emits_its_own_layout(self):
        from mobius.tasks._gemma4 import Gemma4Task

        config = self._config()
        package = Gemma4Task().build(Gemma4Model(config), config)

        assert self._layouts(package["decoder"].graph) == {(4, 16)}
        assert self._layouts(package["vision_encoder"].graph) == {(8, 32)}
        assert self._layouts(package["audio_encoder"].graph) == {(2, 16)}
        assert self._layouts(package["embedding"].graph) == {(8, 16)}

    def test_preprocesses_each_component_with_its_own_layout(self):
        model = Gemma4Model(self._config())
        state_dict = {
            "model.language_model.layers.0.self_attn.q_proj.weight_qweight": torch.zeros(
                64, 32, dtype=torch.uint8
            ),
            "model.language_model.layers.0.self_attn.q_proj.weight_scales": torch.ones(64, 4),
            "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight_qweight": torch.zeros(
                32, 32, dtype=torch.uint8
            ),
            "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight_scales": torch.ones(
                32, 1
            ),
            "model.audio_tower.layers.0.self_attn.q_proj.linear.weight_qweight": torch.zeros(
                32, 8, dtype=torch.uint8
            ),
            "model.audio_tower.layers.0.self_attn.q_proj.linear.weight_scales": torch.ones(
                32, 2
            ),
            "model.language_model.per_layer_model_projection.weight_qweight": torch.zeros(
                32, 64, dtype=torch.uint8
            ),
            "model.language_model.per_layer_model_projection.weight_scales": torch.ones(32, 4),
        }

        result = model.preprocess_weights(state_dict)

        assert result["decoder.model.layers.0.self_attn.q_proj.weight"].shape == (
            64,
            4,
            8,
        )
        assert result["vision_encoder.encoder.layers.0.self_attn.q_proj.weight"].shape == (
            32,
            1,
            32,
        )
        assert result["audio_encoder.encoder.layers.0.self_attn.q_proj.weight"].shape == (
            32,
            2,
            4,
        )
        assert result["embedding.per_layer_model_projection.weight"].shape == (
            32,
            4,
            16,
        )


class TestScaleFreeRMSNormOverflow:
    """V norm should handle FP16 overflow from squaring large values."""

    def test_vnorm_fp16_no_nan(self):
        """Values ~888 overflow FP16 when squared (888²=788K > 65504).

        The scale-free RMSNorm must use stash_type=1 (float32 accumulation)
        to avoid inf/NaN from the variance computation.
        """
        import numpy as np

        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.components import ScaleFreeRMSNorm

        dim = 64
        norm = ScaleFreeRMSNorm(dim, eps=1e-6)

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


def _tiny_gemma4_text_moe_config(**overrides) -> Gemma4Config:
    """Minimal text-only Gemma4 config with the parallel MoE block enabled.

    The graph builds on CPU, which has no fused ``com.microsoft::MoE`` op, so
    ``Gemma4DecoderLayer`` emits the vectorized unfused fallback — the path that
    applies the exact expert activation.
    """
    defaults = dict(
        model_type="gemma4",
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        vocab_size=256,
        rms_norm_eps=1e-6,
        hidden_act="gelu_pytorch_tanh",
        attn_qk_norm=True,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=8,
        global_head_dim=16,
        global_rope_theta=10_000.0,
        global_partial_rotary_factor=0.25,
        final_logit_softcapping=0.0,
        hidden_size_per_layer_input=0,
        pad_token_id=0,
        tie_word_embeddings=True,
        enable_moe_block=True,
        num_local_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
    )
    defaults.update(overrides)
    return Gemma4Config(**defaults)


def _build_gemma4_text(config: Gemma4Config) -> ir.Model:
    from mobius._registry import registry
    from mobius.integrations.transformers._config_resolver import _default_task_for_model
    from mobius.tasks import get_task

    module = registry.get("gemma4_text")(config)
    task = get_task(_default_task_for_model("gemma4_text"))
    return task.build(module, config)["model"]


class TestGemma4MoeFallbackActivation:
    """The unfused MoE fallback must apply the EXACT ``hidden_act`` (ACT2FN).

    ``gelu`` maps to the erf GELU while only the tanh/fast/new variants use the
    tanh approximation, so ``hidden_act="gelu"`` must NOT silently become
    ``Gelu(approximate="tanh")`` on the fallback path.
    """

    def _activation_nodes(self, model: ir.Model):
        acts = []
        for node in model.graph:
            if node.op_type in ("Gelu", "Swish"):
                attrs = {a.name: a.value for a in node.attributes.values()}
                acts.append((node.op_type, attrs.get("approximate")))
        return acts

    def test_silu_uses_swish(self):
        model = _build_gemma4_text(_tiny_gemma4_text_moe_config(hidden_act="silu"))
        acts = self._activation_nodes(model)
        assert acts, "MoE fallback must emit an activation op"
        assert all(op == "Swish" for op, _ in acts), acts

    def test_gelu_uses_erf_not_tanh(self):
        model = _build_gemma4_text(_tiny_gemma4_text_moe_config(hidden_act="gelu"))
        acts = self._activation_nodes(model)
        assert acts, "MoE fallback must emit an activation op"
        assert all(op == "Gelu" for op, _ in acts), acts
        # erf GELU => the ``approximate`` attribute must be absent/"none".
        assert all(approx in (None, "none") for _, approx in acts), acts

    def test_gelu_pytorch_tanh_uses_tanh_approx(self):
        model = _build_gemma4_text(
            _tiny_gemma4_text_moe_config(hidden_act="gelu_pytorch_tanh")
        )
        acts = self._activation_nodes(model)
        assert acts, "MoE fallback must emit an activation op"
        assert all(op == "Gelu" and approx == "tanh" for op, approx in acts), acts


class TestGemma4OutputLayerIndices:
    """``output_layer_indices`` taps the post-final-norm hidden as ``hidden_states.{idx}``.

    Gemma4 exposes ONLY the final post-final-norm hidden (== HF
    ``hidden_states[-1]``, the lm_head input), which a borrowed-KV speculative
    drafter consumes as its folded-carry seed. It is faithful only for the last
    decoder layer, so exactly one index == ``num_hidden_layers - 1`` is allowed;
    duplicate/out-of-range/negative/multiple/non-last indices are rejected.
    ``None``/empty preserves the legacy output contract (no extra port).
    """

    def test_none_keeps_legacy_output_contract(self):
        model = _build_gemma4_text(_tiny_gemma4_text_moe_config())
        names = {o.name for o in model.graph.outputs}
        assert "logits" in names
        assert not any(n.startswith("hidden_states.") for n in names)

    def test_empty_list_keeps_legacy_output_contract(self):
        model = _build_gemma4_text(_tiny_gemma4_text_moe_config(output_layer_indices=[]))
        names = {o.name for o in model.graph.outputs}
        assert "logits" in names
        assert not any(n.startswith("hidden_states.") for n in names)

    def test_last_index_emits_hidden_states_port(self):
        # Last decoder layer index (num_hidden_layers - 1) — the post-norm slot.
        config = _tiny_gemma4_text_moe_config(output_layer_indices=[1])
        model = _build_gemma4_text(config)
        names = {o.name for o in model.graph.outputs}
        assert "hidden_states.1" in names
        assert "logits" in names

    def test_hidden_states_port_is_identity_of_lm_head_input(self):
        """The emitted hidden is a DISTINCT Identity of the exact lm_head input.

        Distinctness prevents the in-place-rename collapse that would occur if a
        single shared Value were passed to ``add_output`` more than once, while
        the Identity's input being the lm_head input proves the post-final-norm
        faithfulness.
        """
        config = _tiny_gemma4_text_moe_config(output_layer_indices=[1])
        model = _build_gemma4_text(config)
        hs_out = next(o for o in model.graph.outputs if o.name == "hidden_states.1")
        logits_out = next(o for o in model.graph.outputs if o.name == "logits")
        lm_head_node = logits_out.producer()
        assert lm_head_node is not None
        # The output port is its own Identity node (unique name, no in-place
        # rename of the internal lm_head input value).
        id_node = hs_out.producer()
        assert id_node is not None and id_node.op_type == "Identity"
        # ...and its input is the exact post-final-norm value fed into lm_head.
        assert id_node.inputs[0] in set(lm_head_node.inputs), (
            "hidden_states port must be an Identity of the lm_head input (post-final-norm)"
        )

    def test_multiple_indices_rejected(self):
        with pytest.raises(ValueError, match="exactly"):
            Gemma4CausalLMModel(_tiny_gemma4_text_moe_config(output_layer_indices=[0, 1]))

    def test_non_last_index_rejected(self):
        # num_hidden_layers=2 => only [1] is valid; [0] is a non-last index.
        with pytest.raises(ValueError, match="exactly"):
            Gemma4CausalLMModel(_tiny_gemma4_text_moe_config(output_layer_indices=[0]))

    def test_duplicate_indices_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            Gemma4CausalLMModel(_tiny_gemma4_text_moe_config(output_layer_indices=[1, 1]))

    def test_out_of_range_index_rejected(self):
        # num_hidden_layers=2 => valid indices are [0, 2); 2 is out of range.
        with pytest.raises(ValueError, match="out of range"):
            Gemma4CausalLMModel(_tiny_gemma4_text_moe_config(output_layer_indices=[2]))

    def test_negative_index_rejected(self):
        with pytest.raises(ValueError, match="out of range"):
            Gemma4CausalLMModel(_tiny_gemma4_text_moe_config(output_layer_indices=[-1]))

    def test_valid_index_scales_with_num_hidden_layers(self):
        # The accepted index tracks the config (no hardcoded layer count/name).
        config = _tiny_gemma4_text_moe_config(
            num_hidden_layers=3,
            layer_types=["sliding_attention", "full_attention", "sliding_attention"],
            output_layer_indices=[2],
        )
        model = _build_gemma4_text(config)
        names = {o.name for o in model.graph.outputs}
        assert "hidden_states.2" in names


class TestGemma4PerExpertScalePartialStateDict:
    """Neutralization of ``router.per_expert_scale`` is scoped to folded prefixes.

    A partial state_dict that supplies ``router.per_expert_scale`` without the
    matching ``fc2_experts_weights`` must keep its scale intact (no silent drop).
    """

    def test_scale_preserved_when_fc2_absent(self):
        config = _tiny_gemma4_config()
        model = Gemma4CausalLMModel(config)
        pes = torch.tensor([1.0, 2.0, 3.0, 4.0])
        fake_sd = {"model.layers.0.router.per_expert_scale": pes.clone()}
        result = model.preprocess_weights(fake_sd)
        key = "model.layers.0.router.per_expert_scale"
        assert key in result
        assert torch.allclose(result[key], pes), (
            "per_expert_scale must be preserved when its fc2 pair is missing"
        )

    def test_scale_folded_and_neutralized_when_fc2_present(self):
        config = _tiny_gemma4_config()
        model = Gemma4CausalLMModel(config)
        pes = torch.tensor([1.0, 2.0, 3.0, 4.0])
        fc2 = torch.ones(4, 64, 32)
        fake_sd = {
            "model.layers.0.router.per_expert_scale": pes.clone(),
            "model.layers.0.experts.down_proj": fc2.clone(),
        }
        result = model.preprocess_weights(fake_sd)
        # Router copy neutralized to ones (scale now baked into fc2).
        neutralized = result["model.layers.0.router.per_expert_scale"]
        assert torch.allclose(neutralized, torch.ones(4))
        # fc2[e] scaled by per_expert_scale[e].
        folded = result["model.layers.0.fc2_experts_weights"]
        expected = fc2 * pes.reshape(-1, 1, 1)
        assert torch.allclose(folded, expected)
