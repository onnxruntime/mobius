# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Fun-ASR-Nano model: runtime edge cases and pipeline validation.

These tests exercise the 3-model split pipeline (audio_encoder → embedding
→ decoder) through ORT with random weights, covering edge cases that the
L1 graph-build tests in build_graph_test.py do not reach.

Test levels:
  - Extended L1/runtime: ORT execution with various input shapes and edge cases
  - Shape validation: Verify output dimensions across the pipeline
  - Edge cases: Temporal pooling boundary conditions, audio token alignment
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from mobius._configs import ArchitectureConfig, AudioConfig
from mobius._testing.ort_inference import OnnxModelSession
from mobius.models.fun_asr import FunASRForConditionalGeneration
from mobius.rewrite_rules._testing_utils import fill_random_weights
from mobius.tasks import FunASRSpeechLanguageTask

# Add tests/ to path so we can import the shared _base_config helper
_TESTS_DIR = Path(__file__).resolve().parents[3] / "tests"
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from _test_configs import TINY_HEADS, TINY_HIDDEN, TINY_INTERMEDIATE, TINY_VOCAB, _base_config  # noqa: E402, I001


def _tiny_config(**overrides):
    """Create a tiny Fun-ASR config with default RoPE and audio settings."""
    return _base_config(
        attn_qk_norm=True,
        hidden_act="silu",
        audio=AudioConfig(
            input_size=32,
            attention_dim=TINY_HIDDEN,
            attention_heads=TINY_HEADS,
            num_blocks=3,
            linear_units=TINY_INTERMEDIATE,
            kernel_size=5,
            tp_num_blocks=2,
            output_dim=TINY_HIDDEN,
            audio_token_id=100,
            adaptor_proj_dim=TINY_INTERMEDIATE,
            adaptor_num_blocks=2,
            adaptor_ffn_dim=32,
            adaptor_num_heads=TINY_HEADS,
        ),
        **overrides,
    )


def _build_package(config=None):
    """Build Fun-ASR 3-model package with random weights."""
    from mobius import build_from_module

    if config is None:
        config = _tiny_config()
    module = FunASRForConditionalGeneration(config)
    pkg = build_from_module(module, config, task=FunASRSpeechLanguageTask())
    for model in pkg.values():
        fill_random_weights(model)
    return pkg, config


def _run_pipeline(
    pkg,
    config: ArchitectureConfig,
    fbank: np.ndarray,
    prefix_ids: list[int] | None = None,
    suffix_ids: list[int] | None = None,
) -> dict[str, np.ndarray]:
    """Run the full 3-model pipeline: audio_encoder → embedding → decoder.

    Returns dict with audio_features, inputs_embeds, logits, and KV cache.
    """
    if prefix_ids is None:
        prefix_ids = [1, 2, 3]
    if suffix_ids is None:
        suffix_ids = [4, 5]

    # Step 1: Audio encoder
    enc_sess = OnnxModelSession(pkg["audio_encoder"])
    try:
        enc_out = enc_sess.run({"input_features": fbank})
    finally:
        enc_sess.close()
    audio_features = enc_out["audio_features"]
    num_audio_tokens = audio_features.shape[1]
    # Embedding expects (num_tokens, hidden_dim) — squeeze batch
    audio_features_2d = audio_features.reshape(-1, audio_features.shape[-1])

    # Step 2: Embedding (text + audio fusion)
    audio_token_id = config.audio.audio_token_id
    input_ids = np.array(
        [prefix_ids + [audio_token_id] * num_audio_tokens + suffix_ids],
        dtype=np.int64,
    )
    embed_sess = OnnxModelSession(pkg["embedding"])
    try:
        embed_out = embed_sess.run(
            {"input_ids": input_ids, "audio_features": audio_features_2d}
        )
    finally:
        embed_sess.close()
    inputs_embeds = embed_out["inputs_embeds"]

    # Step 3: Decoder
    seq_len = inputs_embeds.shape[1]
    past_kv = {}
    for i in range(config.num_hidden_layers):
        past_kv[f"past_key_values.{i}.key"] = np.zeros(
            (1, config.num_key_value_heads, 0, config.head_dim),
            dtype=np.float32,
        )
        past_kv[f"past_key_values.{i}.value"] = np.zeros(
            (1, config.num_key_value_heads, 0, config.head_dim),
            dtype=np.float32,
        )

    dec_sess = OnnxModelSession(pkg["decoder"])
    try:
        dec_out = dec_sess.run(
            {
                "inputs_embeds": inputs_embeds,
                "attention_mask": np.ones((1, seq_len), dtype=np.int64),
                "position_ids": np.arange(seq_len, dtype=np.int64).reshape(1, -1),
                **past_kv,
            }
        )
    finally:
        dec_sess.close()

    return {
        "audio_features": audio_features,
        "inputs_embeds": inputs_embeds,
        "input_ids": input_ids,
        **dec_out,
    }


class TestFunASRPipelineShapes:
    """Verify output shapes across the 3-model pipeline."""

    @pytest.fixture(scope="class")
    def package(self):
        pkg, config = _build_package()
        return pkg, config

    def test_audio_encoder_output_shape(self, package):
        """Audio encoder: (B, T, D_in) → (B, T, D_hidden).

        No temporal pooling — tp_encoders are refinement layers that
        preserve sequence length.
        """
        pkg, config = package
        input_dim = config.audio.input_size
        seq_len = 100
        fbank = np.random.randn(1, seq_len, input_dim).astype(np.float32)

        sess = OnnxModelSession(pkg["audio_encoder"])
        try:
            out = sess.run({"input_features": fbank})
        finally:
            sess.close()

        audio_features = out["audio_features"]
        assert audio_features.shape == (
            1,
            seq_len,
            config.audio.attention_dim,
        )

    def test_embedding_output_shape(self, package):
        """Embedding: input_ids + audio_features → (B, seq, hidden)."""
        pkg, config = package
        # Simulate audio encoder output
        num_audio_tokens = 10
        audio_features = np.random.randn(num_audio_tokens, config.audio.attention_dim).astype(
            np.float32
        )

        prefix = [1, 2]
        suffix = [3]
        audio_token_id = config.audio.audio_token_id
        input_ids = np.array(
            [prefix + [audio_token_id] * num_audio_tokens + suffix],
            dtype=np.int64,
        )
        total_len = len(prefix) + num_audio_tokens + len(suffix)

        sess = OnnxModelSession(pkg["embedding"])
        try:
            out = sess.run({"input_ids": input_ids, "audio_features": audio_features})
        finally:
            sess.close()

        inputs_embeds = out["inputs_embeds"]
        assert inputs_embeds.shape == (1, total_len, config.hidden_size)

    def test_decoder_output_shape(self, package):
        """Decoder: inputs_embeds → logits + KV cache."""
        pkg, config = package
        seq_len = 15
        inputs_embeds = np.random.randn(1, seq_len, config.hidden_size).astype(np.float32)

        past_kv = {}
        for i in range(config.num_hidden_layers):
            past_kv[f"past_key_values.{i}.key"] = np.zeros(
                (1, config.num_key_value_heads, 0, config.head_dim),
                dtype=np.float32,
            )
            past_kv[f"past_key_values.{i}.value"] = np.zeros(
                (1, config.num_key_value_heads, 0, config.head_dim),
                dtype=np.float32,
            )

        sess = OnnxModelSession(pkg["decoder"])
        try:
            out = sess.run(
                {
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": np.ones((1, seq_len), dtype=np.int64),
                    "position_ids": np.arange(seq_len, dtype=np.int64).reshape(1, -1),
                    **past_kv,
                }
            )
        finally:
            sess.close()

        assert out["logits"].shape == (1, seq_len, config.vocab_size)
        assert out["present.0.key"].shape == (
            1,
            config.num_key_value_heads,
            seq_len,
            config.head_dim,
        )


class TestFunASRSequenceLength:
    """Test audio encoder sequence length preservation.

    The tp_encoders are refinement layers, NOT temporal pooling.
    Input sequence length is preserved through the encoder.
    """

    @pytest.fixture(scope="class")
    def package(self):
        pkg, config = _build_package()
        return pkg, config

    @pytest.mark.parametrize("seq_len", [2, 4, 50, 100, 200])
    def test_sequence_lengths_preserved(self, package, seq_len):
        """Various sequence lengths should all pass through unchanged."""
        pkg, config = package
        input_dim = config.audio.input_size
        fbank = np.random.randn(1, seq_len, input_dim).astype(np.float32)

        sess = OnnxModelSession(pkg["audio_encoder"])
        try:
            out = sess.run({"input_features": fbank})
        finally:
            sess.close()

        audio_features = out["audio_features"]
        assert audio_features.shape[1] == seq_len
        assert not np.any(np.isnan(audio_features))

    def test_minimum_sequence_length(self, package):
        """Minimum viable sequence length (T=2)."""
        pkg, config = package
        input_dim = config.audio.input_size
        fbank = np.random.randn(1, 2, input_dim).astype(np.float32)

        sess = OnnxModelSession(pkg["audio_encoder"])
        try:
            out = sess.run({"input_features": fbank})
        finally:
            sess.close()

        assert out["audio_features"].shape[1] == 2


class TestFunASRFullPipeline:
    """End-to-end 3-model pipeline tests with various configurations."""

    @pytest.fixture(scope="class")
    def package(self):
        pkg, config = _build_package()
        return pkg, config

    def test_full_pipeline_produces_valid_logits(self, package):
        """Full pipeline: fbank → audio_features → embeds → logits."""
        pkg, config = package
        input_dim = config.audio.input_size
        fbank = np.random.randn(1, 100, input_dim).astype(np.float32)

        result = _run_pipeline(pkg, config, fbank)

        logits = result["logits"]
        assert logits.shape[0] == 1
        assert logits.shape[2] == config.vocab_size
        assert not np.any(np.isnan(logits))
        assert not np.any(np.isinf(logits))

    def test_pipeline_with_short_audio(self, package):
        """Short audio input (T=4, no pooling)."""
        pkg, config = package
        input_dim = config.audio.input_size
        fbank = np.random.randn(1, 4, input_dim).astype(np.float32)

        result = _run_pipeline(pkg, config, fbank, prefix_ids=[1], suffix_ids=[2])

        logits = result["logits"]
        # 4 audio tokens (no pooling) + 1 prefix + 1 suffix = 6
        assert logits.shape[1] == 6
        assert not np.any(np.isnan(logits))

    def test_pipeline_no_prefix_suffix(self, package):
        """Audio tokens only (no text prefix/suffix)."""
        pkg, config = package
        input_dim = config.audio.input_size
        fbank = np.random.randn(1, 20, input_dim).astype(np.float32)

        result = _run_pipeline(pkg, config, fbank, prefix_ids=[], suffix_ids=[])

        logits = result["logits"]
        # 20 audio tokens only (no pooling)
        assert logits.shape[1] == 20
        assert not np.any(np.isnan(logits))

    def test_pipeline_long_prefix(self, package):
        """Long text prefix before audio tokens."""
        pkg, config = package
        input_dim = config.audio.input_size
        fbank = np.random.randn(1, 20, input_dim).astype(np.float32)

        prefix = list(range(1, 51))  # 50 text tokens
        result = _run_pipeline(pkg, config, fbank, prefix_ids=prefix, suffix_ids=[51])

        logits = result["logits"]
        # 50 prefix + 20 audio (no pooling) + 1 suffix = 71
        assert logits.shape[1] == 71

    def test_decoder_step_with_kv_cache(self, package):
        """Verify decoder handles autoregressive step (seq_len=1 with cache)."""
        pkg, config = package

        # First: prefill
        prefill_len = 8
        inputs_embeds = np.random.randn(1, prefill_len, config.hidden_size).astype(np.float32)

        past_kv = {}
        for i in range(config.num_hidden_layers):
            past_kv[f"past_key_values.{i}.key"] = np.zeros(
                (1, config.num_key_value_heads, 0, config.head_dim),
                dtype=np.float32,
            )
            past_kv[f"past_key_values.{i}.value"] = np.zeros(
                (1, config.num_key_value_heads, 0, config.head_dim),
                dtype=np.float32,
            )

        dec_sess = OnnxModelSession(pkg["decoder"])
        try:
            prefill_out = dec_sess.run(
                {
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": np.ones((1, prefill_len), dtype=np.int64),
                    "position_ids": np.arange(prefill_len, dtype=np.int64).reshape(1, -1),
                    **past_kv,
                }
            )

            # Verify prefill produced cache
            assert prefill_out["present.0.key"].shape[2] == prefill_len

            # Decode step: seq_len=1 with populated KV cache
            step_embeds = np.random.randn(1, 1, config.hidden_size).astype(np.float32)
            step_kv = {}
            for i in range(config.num_hidden_layers):
                step_kv[f"past_key_values.{i}.key"] = prefill_out[f"present.{i}.key"]
                step_kv[f"past_key_values.{i}.value"] = prefill_out[f"present.{i}.value"]

            step_out = dec_sess.run(
                {
                    "inputs_embeds": step_embeds,
                    "attention_mask": np.ones((1, prefill_len + 1), dtype=np.int64),
                    "position_ids": np.array([[prefill_len]], dtype=np.int64),
                    **step_kv,
                }
            )
        finally:
            dec_sess.close()

        assert step_out["logits"].shape == (1, 1, config.vocab_size)
        assert step_out["present.0.key"].shape[2] == prefill_len + 1
        assert not np.any(np.isnan(step_out["logits"]))


class TestFunASRDeterminism:
    """Verify that the pipeline is deterministic (same input → same output)."""

    @pytest.fixture(scope="class")
    def package(self):
        pkg, config = _build_package()
        return pkg, config

    def test_audio_encoder_deterministic(self, package):
        """Same fbank input → same audio features."""
        pkg, config = package
        input_dim = config.audio.input_size
        rng = np.random.default_rng(42)
        fbank = rng.standard_normal((1, 50, input_dim)).astype(np.float32)

        sess = OnnxModelSession(pkg["audio_encoder"])
        try:
            out1 = sess.run({"input_features": fbank})
            out2 = sess.run({"input_features": fbank})
        finally:
            sess.close()

        np.testing.assert_array_equal(out1["audio_features"], out2["audio_features"])

    def test_full_pipeline_deterministic(self, package):
        """Full pipeline with same input → same logits."""
        pkg, config = package
        input_dim = config.audio.input_size
        rng = np.random.default_rng(99)
        fbank = rng.standard_normal((1, 20, input_dim)).astype(np.float32)

        result1 = _run_pipeline(pkg, config, fbank)
        result2 = _run_pipeline(pkg, config, fbank)

        np.testing.assert_array_equal(result1["logits"], result2["logits"])


class TestFunASRWeightNames:
    """Verify preprocess_weights maps all expected weight name patterns."""

    def test_audio_encoder_weight_routing(self):
        """Verify audio_encoder.* maps to audio_tower.*."""
        config = _tiny_config()
        module = FunASRForConditionalGeneration(config)
        import torch

        fake_sd = {
            "audio_encoder.encoders0.0.norm1.weight": torch.zeros(TINY_HIDDEN),
            "audio_encoder.encoders.0.norm1.weight": torch.zeros(TINY_HIDDEN),
            "audio_encoder.tp_encoders.0.norm1.weight": torch.zeros(TINY_HIDDEN),
            "audio_encoder.after_norm.weight": torch.zeros(TINY_HIDDEN),
            "audio_encoder.tp_norm.weight": torch.zeros(TINY_HIDDEN),
        }
        result = module.preprocess_weights(fake_sd)

        assert "audio_tower.encoders0.0.norm1.weight" in result
        assert "audio_tower.encoders.0.norm1.weight" in result
        assert "audio_tower.tp_encoders.0.norm1.weight" in result
        assert "audio_tower.after_norm.weight" in result
        assert "audio_tower.tp_norm.weight" in result

    def test_adaptor_weight_routing(self):
        """Verify audio_adaptor.* maps to audio_tower.adaptor.*."""
        config = _tiny_config()
        module = FunASRForConditionalGeneration(config)
        import torch

        fake_sd = {
            "audio_adaptor.linear1.weight": torch.zeros(TINY_INTERMEDIATE, TINY_HIDDEN),
            "audio_adaptor.blocks.0.norm1.weight": torch.zeros(config.hidden_size),
        }
        result = module.preprocess_weights(fake_sd)

        assert "audio_tower.adaptor.linear1.weight" in result
        assert "audio_tower.adaptor.blocks.0.norm1.weight" in result

    def test_decoder_weight_routing(self):
        """Verify llm.model.layers/norm/lm_head route to decoder.*."""
        config = _tiny_config()
        module = FunASRForConditionalGeneration(config)
        import torch

        fake_sd = {
            "llm.model.layers.0.self_attn.q_proj.weight": torch.zeros(
                TINY_HIDDEN, TINY_HIDDEN
            ),
            "llm.model.norm.weight": torch.zeros(TINY_HIDDEN),
            "llm.lm_head.weight": torch.zeros(TINY_VOCAB, TINY_HIDDEN),
            "llm.model.embed_tokens.weight": torch.zeros(TINY_VOCAB, TINY_HIDDEN),
        }
        result = module.preprocess_weights(fake_sd)

        assert "decoder.layers.0.self_attn.q_proj.weight" in result
        assert "decoder.norm.weight" in result
        assert "decoder.lm_head.weight" in result
        assert "embedding.embed_tokens.weight" in result

    def test_weight_tying(self):
        """Tied weights: embed_tokens.weight copied to lm_head.weight."""
        config = _tiny_config(tie_word_embeddings=True)
        module = FunASRForConditionalGeneration(config)
        import torch

        fake_sd = {
            "llm.model.embed_tokens.weight": torch.ones(TINY_VOCAB, TINY_HIDDEN),
        }
        result = module.preprocess_weights(fake_sd)

        assert "embedding.embed_tokens.weight" in result
        assert "decoder.lm_head.weight" in result
        # Both should be the same tensor
        assert torch.equal(
            result["embedding.embed_tokens.weight"],
            result["decoder.lm_head.weight"],
        )
