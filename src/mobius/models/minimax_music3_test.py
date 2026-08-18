# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import torch

from mobius import build_from_module
from mobius._diffusers_configs import (
    MINIMAX_MUSIC3_AUDIO_CODE_OFFSET,
    MINIMAX_MUSIC3_AUDIO_END_TOKEN_ID,
    MINIMAX_MUSIC3_FEEDBACK_SCALE,
    MINIMAX_MUSIC3_SEMANTIC_VOCAB_SIZE,
    MiniMaxMusic3ConditionConfig,
    MiniMaxMusic3LanguageConfig,
    MiniMaxMusic3RVQConfig,
    MiniMaxMusic3TransformerConfig,
    MiniMaxMusic3VocoderConfig,
)
from mobius._testing.ort_inference import OnnxModelSession
from mobius._weight_loading import apply_weights
from mobius.models.minimax_music3 import (
    MiniMaxMusic3ConditionEncoder,
    MiniMaxMusic3LanguageModel,
    MiniMaxMusic3RVQDepthDecoder,
    MiniMaxMusic3Transformer1DModel,
    MiniMaxMusic3Vocoder,
)


def _initialize(model) -> None:
    state = {
        name: torch.full(tuple(value.shape), 0.01, dtype=torch.float32)
        for name, value in model.graph.initializers.items()
        if value.const_value is None
    }
    apply_weights(model, state)


def test_rvq_depth_decoder_builds_all_pipeline_contracts():
    config = MiniMaxMusic3RVQConfig(
        hidden_size=32,
        num_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        audio_vocab_size=16,
        num_codebooks=3,
        max_position_embeddings=8,
    )
    package = build_from_module(
        MiniMaxMusic3RVQDepthDecoder(config), config, "minimax-music3-rvq"
    )
    assert set(package) == {
        "model",
        "projection",
        "embedding",
        "feedback_embedding",
        "heads",
    }

    for model in package.values():
        _initialize(model)
    session = OnnxModelSession(package["model"])
    for steps in (2, 8):
        output = session.run({"inputs_embeds": np.ones((1, steps, 32), np.float32)})
        assert output["hidden_states"].shape == (1, steps, 32)


def test_condition_encoder_runtime_uses_exact_upstream_length_rounding():
    config = MiniMaxMusic3ConditionConfig(
        condition_hidden_dim=4,
        num_condition_layers=8,
        out_dim=8,
    )
    package = build_from_module(
        MiniMaxMusic3ConditionEncoder(config), config, "minimax-music3-condition"
    )
    _initialize(package["model"])
    session = OnnxModelSession(package["model"])
    # Each frame is [global final hidden, seven growing-depth-step hiddens].
    output = session.run({"hidden_states": np.ones((1, 5, 32), np.float32)})
    # int(5 * 44100 / 24000 * 960 / 512) == 17
    assert output["encoder_hidden_states"].shape == (1, 17, 8)


def test_transformer_runtime_preserves_latent_shape():
    config = MiniMaxMusic3TransformerConfig(
        in_channels=4,
        condition_dim=8,
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        ff_inner_dim=32,
        rotary_dim=4,
        fourier_embedding_dim=8,
    )
    package = build_from_module(
        MiniMaxMusic3Transformer1DModel(config),
        config,
        "minimax-music3-denoising",
    )
    _initialize(package["model"])
    session = OnnxModelSession(package["model"])
    output = session.run(
        {
            "hidden_states": np.ones((1, 4, 3), np.float32),
            "timestep": np.array([0.5], np.float32),
            "encoder_hidden_states": np.ones((1, 3, 8), np.float32),
        }
    )
    assert output["sample"].shape == (1, 4, 3)
    assert np.isfinite(output["sample"]).all()


def test_transformer_bfloat16_partial_rope_has_no_mixed_mul_types():
    config = MiniMaxMusic3TransformerConfig(
        in_channels=4,
        condition_dim=8,
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        ff_inner_dim=32,
        rotary_dim=4,
        fourier_embedding_dim=8,
        dtype=ir.DataType.BFLOAT16,
    )
    model = build_from_module(
        MiniMaxMusic3Transformer1DModel(config),
        config,
        "minimax-music3-denoising",
    )["model"]
    rotary_muls = [
        node
        for node in model.graph
        if node.op_type == "Mul" and "/attn/Mul_node_" in node.name
    ]
    assert len(rotary_muls) == 4
    for node in rotary_muls:
        assert {value.dtype for value in node.inputs} == {ir.DataType.BFLOAT16}


def test_vocoder_runtime_decodes_stereo_and_upsamples():
    config = MiniMaxMusic3VocoderConfig(
        latent_channels=8,
        decoder_input_dim=16,
        decoder_hidden_dim=32,
        upsampling_ratios=(2, 2),
    )
    module = MiniMaxMusic3Vocoder(config)
    package = build_from_module(module, config, "minimax-music3-vocoder")
    _initialize(package["model"])
    session = OnnxModelSession(package["model"])
    output = session.run({"latents": np.ones((1, 8, 3), np.float32)})
    assert output["waveform"].shape == (1, 2, 12)
    assert np.max(np.abs(output["waveform"])) <= 1.0


def test_vocoder_preprocess_folds_checkpoint_weight_norm():
    module = MiniMaxMusic3Vocoder(
        MiniMaxMusic3VocoderConfig(
            latent_channels=8,
            decoder_input_dim=16,
            decoder_hidden_dim=32,
            upsampling_ratios=(2,),
        )
    )
    value = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) + 1
    scale = torch.tensor([[[2.0]], [[3.0]]])
    result = module.preprocess_weights(
        {"conv.weight_v": value, "conv.weight_g": scale, "conv.bias": torch.zeros(2)}
    )
    expected = value * scale / torch.linalg.vector_norm(value, dim=(1, 2), keepdim=True)
    torch.testing.assert_close(result["conv.weight"], expected)
    assert "conv.weight_v" not in result
    assert "conv.weight_g" not in result


def test_pinned_configs_parse_exact_architecture_values():
    condition = MiniMaxMusic3ConditionConfig.from_diffusers(
        {
            "condition_hidden_dim": 4096,
            "num_condition_layers": 8,
            "out_dim": 2048,
            "input_sampling_rate": 24000,
            "input_hop_length": 960,
            "output_sampling_rate": 44100,
            "output_hop_length": 512,
        }
    )
    transformer = MiniMaxMusic3TransformerConfig.from_diffusers(
        {
            "in_channels": 128,
            "condition_dim": 2048,
            "num_layers": 36,
            "num_attention_heads": 32,
            "attention_head_dim": 64,
            "ff_inner_dim": 8192,
            "rotary_dim": 32,
            "fourier_embedding_dim": 256,
        }
    )
    vocoder = MiniMaxMusic3VocoderConfig.from_diffusers({"upsampling_ratios": [8, 8, 4, 2]})
    assert condition.output_sampling_rate == 44100
    assert transformer.num_layers == 36
    assert transformer.rotary_dim == 32
    assert vocoder.upsampling_ratios == (8, 8, 4, 2)


def test_language_model_reuses_qwen3_and_exposes_music_pipeline_outputs():
    config = MiniMaxMusic3LanguageConfig.from_diffusers(
        {
            "model_type": "qwen3",
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 32,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {"rope_theta": 1_000_000, "rope_type": "default"},
            "tie_word_embeddings": False,
            "hidden_act": "silu",
        }
    )
    package = build_from_module(
        MiniMaxMusic3LanguageModel(config), config, "minimax-music3-language"
    )
    assert set(package) == {"model", "embedding", "semantic_embedding"}
    output_names = {value.name for value in package["model"].graph.outputs}
    assert {"logits", "last_hidden_state"} <= output_names
    assert "model.embed_tokens.weight" in package["embedding"].graph.initializers


def test_feedback_embedding_contract_uses_exact_offsets_and_scale():
    language_config = MiniMaxMusic3LanguageConfig.from_diffusers(
        {
            "model_type": "qwen3",
            "vocab_size": 151675 + 16384,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "max_position_embeddings": 16,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {"rope_theta": 1_000_000, "rope_type": "default"},
            "tie_word_embeddings": False,
            "hidden_act": "silu",
        }
    )
    language = build_from_module(
        MiniMaxMusic3LanguageModel(language_config),
        language_config,
        "minimax-music3-language",
    )
    rvq_config = MiniMaxMusic3RVQConfig(
        hidden_size=8,
        num_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        audio_vocab_size=4,
        num_codebooks=8,
        max_position_embeddings=8,
    )
    rvq = build_from_module(
        MiniMaxMusic3RVQDepthDecoder(rvq_config),
        rvq_config,
        "minimax-music3-rvq",
    )
    _initialize(language["semantic_embedding"])
    _initialize(rvq["feedback_embedding"])
    semantic = OnnxModelSession(language["semantic_embedding"]).run(
        {"semantic_codes": np.array([[0]], np.int64)}
    )["semantic_feedback_embedding"]
    acoustic = OnnxModelSession(rvq["feedback_embedding"]).run(
        {"acoustic_codes": np.zeros((1, 1, 7), np.int64)}
    )["acoustic_feedback_embedding"]
    np.testing.assert_allclose(
        semantic + acoustic,
        np.full((1, 1, 8), 0.08 * MINIMAX_MUSIC3_FEEDBACK_SCALE, np.float32),
    )
    assert MINIMAX_MUSIC3_AUDIO_CODE_OFFSET == 151675
    assert MINIMAX_MUSIC3_SEMANTIC_VOCAB_SIZE == 16384
    assert MINIMAX_MUSIC3_AUDIO_END_TOKEN_ID == 151670


def test_component_parameter_names_match_pinned_checkpoint_layouts():
    rvq_config = MiniMaxMusic3RVQConfig(
        hidden_size=32,
        num_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        audio_vocab_size=16,
        num_codebooks=3,
        max_position_embeddings=4,
    )
    rvq = build_from_module(
        MiniMaxMusic3RVQDepthDecoder(rvq_config),
        rvq_config,
        "minimax-music3-rvq",
    )
    names = set().union(*(model.graph.initializers for model in rvq.values()))
    assert "layers.0.attn.to_q.weight" in names
    assert "audio_embeddings.weight" in names
    assert "audio_heads.1.weight" in names
