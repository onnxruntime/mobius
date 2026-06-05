# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Gemma4 preprocess_weights — MoE expert rename and router scale."""

from __future__ import annotations

import onnx_ir as ir
import torch

from mobius._configs import AudioConfig, Gemma4Config
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
        out = _compute_block_sequence_ids(
            op, input_ids, image_token_id=255, audio_token_id=254
        )
        builder.add_output(out, "block_sequence_ids")
        session = OnnxModelSession(_make_model(graph), device="cpu")

        # text  img img  text  aud aud  text  img  text
        ids = np.array([[10, 255, 255, 11, 254, 254, 11, 255, 12]], dtype=np.int64)
        result = session.run({"input_ids": ids})["block_sequence_ids"]
        # 3 contiguous vision runs separated by text -> groups 0, 1, 2; text -> -1.
        expected = np.array([[-1, 0, 0, -1, 1, 1, -1, 2, -1]], dtype=np.int64)
        np.testing.assert_array_equal(result, expected)

    def test_adjacent_image_audio_same_block(self):
        """Image run immediately followed by audio run = single block (HF parity)."""
        import numpy as np

        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.models.gemma4 import _compute_block_sequence_ids
        from mobius.tasks._base import _make_graph, _make_model

        graph, builder = _make_graph()
        op = builder.op
        input_ids = builder.input("input_ids", dtype=ir.DataType.INT64, shape=[1, "S"])
        out = _compute_block_sequence_ids(
            op, input_ids, image_token_id=255, audio_token_id=254
        )
        builder.add_output(out, "block_sequence_ids")
        session = OnnxModelSession(_make_model(graph), device="cpu")

        ids = np.array([[10, 255, 255, 254, 254, 11]], dtype=np.int64)
        result = session.run({"input_ids": ids})["block_sequence_ids"]
        # No text gap between image and audio -> one contiguous block (id 0).
        expected = np.array([[-1, 0, 0, 0, 0, -1]], dtype=np.int64)
        np.testing.assert_array_equal(result, expected)

    def test_package_wires_block_sequence_ids_end_to_end(self):
        """Embedding emits block_sequence_ids and decoder declares it as input.

        With ``use_bidirectional_attention='vision'`` the overlay must be
        plumbed embedding -> decoder.
        """
        from mobius.models.gemma4 import Gemma4Model
        from mobius.tasks._gemma4 import Gemma4Task

        config = _tiny_gemma4_config(use_bidirectional_attention="vision", image_token_id=255)
        pkg = Gemma4Task().build(Gemma4Model(config), config)

        emb_outputs = {o.name for o in pkg["embedding"].graph.outputs}
        dec_inputs = {i.name for i in pkg["decoder"].graph.inputs}
        assert "block_sequence_ids" in emb_outputs
        assert "block_sequence_ids" in dec_inputs

        # Decoder attention must drop GQA and disable the op's built-in causal
        # mask (is_causal=0) so the baked blockwise bias is honored.
        dec = pkg["decoder"].graph
        assert not any(n.op_type == "GroupQueryAttention" for n in dec)
        attn_nodes = [n for n in dec if n.op_type == "Attention"]
        assert attn_nodes
        for n in attn_nodes:
            assert n.attributes["is_causal"].as_int() == 0

    def test_no_block_sequence_ids_when_causal(self):
        """Without bidirectional attention, no block_sequence_ids is wired."""
        from mobius.models.gemma4 import Gemma4Model
        from mobius.tasks._gemma4 import Gemma4Task

        config = _tiny_gemma4_config(use_bidirectional_attention=None)
        pkg = Gemma4Task().build(Gemma4Model(config), config)

        emb_outputs = {o.name for o in pkg["embedding"].graph.outputs}
        dec_inputs = {i.name for i in pkg["decoder"].graph.inputs}
        assert "block_sequence_ids" not in emb_outputs
        assert "block_sequence_ids" not in dec_inputs
