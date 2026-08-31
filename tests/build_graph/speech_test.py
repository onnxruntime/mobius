# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Speech, ASR, TTS, and diarization L1 graph-construction tests.

Run the complete L1 suite with ``pytest tests/build_graph``.
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest
from ._support import (
    _assert_outputs_have_shapes_and_dtypes,
    _make_params,
    _run_onnx_checker,
)
from _test_configs import (
    SPEECH_CONFIGS,
    TINY_HEAD_DIM,
    TINY_HEADS,
    TINY_HIDDEN,
    TINY_INTERMEDIATE,
    TINY_KV_HEADS,
    TINY_LAYERS,
    TINY_VOCAB,
    _base_config,
)

from mobius._builder import build_from_module
from mobius._configs import (
    AudioConfig,
    CodePredictorConfig,
    SpeakerEncoderConfig,
    TTSConfig,
)
from mobius._registry import registry
from mobius.integrations.transformers._config_resolver import _default_task_for_model
from mobius.tasks import (
    get_task,
)

_SPEECH_MODEL_PARAMS = _make_params(SPEECH_CONFIGS)
_SPEECH_TASK_KEYS: dict[str, set[str]] = {
    "speech-to-text": {"encoder", "decoder"},
    "speech-language": {"audio_encoder", "embedding", "decoder"},
    "codec": {"decoder", "encoder"},
    "audio-feature-extraction": {"model"},
    "feature-ctc-asr": {"model"},
    "speech-enhancement": {"model"},
}


class TestBuildGraphWhisper:
    """Verify Whisper encoder-decoder builds with SpeechToTextTask."""

    def _whisper_config(self, *, num_mel_bins=16):
        from mobius._configs import WhisperConfig

        return WhisperConfig(
            vocab_size=512,
            hidden_size=TINY_HIDDEN,
            intermediate_size=TINY_INTERMEDIATE,
            num_hidden_layers=TINY_LAYERS,
            num_attention_heads=TINY_HEADS,
            num_key_value_heads=TINY_HEADS,
            head_dim=TINY_HIDDEN // TINY_HEADS,
            hidden_act="gelu",
            pad_token_id=0,
            tie_word_embeddings=True,
            attn_qkv_bias=True,
            attn_o_bias=True,
            encoder_layers=TINY_LAYERS,
            encoder_attention_heads=TINY_HEADS,
            encoder_ffn_dim=TINY_INTERMEDIATE,
            num_mel_bins=num_mel_bins,
            max_source_positions=100,
            max_target_positions=50,
            scale_embedding=True,
        )

    def test_whisper_package_builds(self):
        """Build Whisper with SpeechToTextTask and verify encoder + decoder."""
        from mobius._builder import build_from_module
        from mobius.models.whisper import WhisperForConditionalGeneration
        from mobius.tasks import SpeechToTextTask

        config = self._whisper_config()
        module = WhisperForConditionalGeneration(config)
        task = SpeechToTextTask()
        pkg = build_from_module(module, config, task=task)

        assert "encoder" in pkg
        assert "decoder" in pkg

    def test_whisper_encoder_io(self):
        """Verify encoder inputs/outputs."""
        from mobius._builder import build_from_module
        from mobius.models.whisper import WhisperForConditionalGeneration
        from mobius.tasks import SpeechToTextTask

        config = self._whisper_config()
        module = WhisperForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=SpeechToTextTask())
        encoder = pkg["encoder"]

        input_names = {inp.name for inp in encoder.graph.inputs}
        output_names = {out.name for out in encoder.graph.outputs}
        assert "input_features" in input_names
        assert "encoder_hidden_states" in output_names

    def test_whisper_128_mel_encoder_input_shape(self):
        """Whisper large-v3/turbo graphs use their configured 128 mel channels."""
        from mobius._builder import build_from_module
        from mobius.models.whisper import WhisperForConditionalGeneration
        from mobius.tasks import SpeechToTextTask

        config = self._whisper_config(num_mel_bins=128)
        module = WhisperForConditionalGeneration(config)
        encoder = build_from_module(module, config, task=SpeechToTextTask())["encoder"]
        input_features = next(
            value for value in encoder.graph.inputs if value.name == "input_features"
        )

        assert config.encoder_input_channels == 128
        assert input_features.shape[1] == 128

    def test_whisper_decoder_io(self):
        """Verify decoder inputs/outputs including KV cache."""
        from mobius._builder import build_from_module
        from mobius.models.whisper import WhisperForConditionalGeneration
        from mobius.tasks import SpeechToTextTask

        config = self._whisper_config()
        module = WhisperForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=SpeechToTextTask())
        decoder = pkg["decoder"]

        input_names = {inp.name for inp in decoder.graph.inputs}
        output_names = {out.name for out in decoder.graph.outputs}

        assert "decoder_input_ids" in input_names
        assert "encoder_hidden_states" in input_names
        assert "position_ids" in input_names
        assert "logits" in output_names

        # KV cache inputs/outputs
        for i in range(TINY_LAYERS):
            assert f"past_key_values.{i}.key" in input_names
            assert f"past_key_values.{i}.value" in input_names
            assert f"present.{i}.key" in output_names
            assert f"present.{i}.value" in output_names

    def test_whisper_encoder_has_initializers(self):
        """Verify encoder has conv and layer norm initializers."""
        from mobius._builder import build_from_module
        from mobius.models.whisper import WhisperForConditionalGeneration
        from mobius.tasks import SpeechToTextTask

        config = self._whisper_config()
        module = WhisperForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=SpeechToTextTask())
        encoder = pkg["encoder"]

        init_names = list(encoder.graph.initializers)
        assert any("conv1" in n for n in init_names), "Should have conv1 initializers"
        assert any("conv2" in n for n in init_names), "Should have conv2 initializers"
        assert any("self_attn" in n for n in init_names), "Should have attention initializers"
        assert any("layer_norm" in n for n in init_names), "Should have LayerNorm initializer"

    def test_whisper_decoder_has_initializers(self):
        """Verify decoder has embedding, attention, cross-attention, and proj_out initializers."""
        from mobius._builder import build_from_module
        from mobius.models.whisper import WhisperForConditionalGeneration
        from mobius.tasks import SpeechToTextTask

        config = self._whisper_config()
        module = WhisperForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=SpeechToTextTask())
        decoder = pkg["decoder"]

        init_names = list(decoder.graph.initializers)
        assert any("embed_tokens" in n for n in init_names), "Should have token embeddings"
        assert any("embed_positions" in n for n in init_names), (
            "Should have position embeddings"
        )
        assert any("self_attn" in n for n in init_names), "Should have self-attention"
        assert any("encoder_attn" in n for n in init_names), "Should have cross-attention"
        assert any("proj_out" in n for n in init_names), "Should have proj_out"

    def test_whisper_registry_lookup(self):
        """Verify whisper model_type is properly registered."""
        model_cls = registry.get("whisper")
        from mobius.models.whisper import WhisperForConditionalGeneration

        assert model_cls is WhisperForConditionalGeneration


class TestBuildGraphMoonshine:
    """Verify Moonshine raw-audio encoder and cached decoder graphs."""

    def _moonshine_config(self):
        from mobius._configs import MoonshineConfig

        return MoonshineConfig(
            vocab_size=512,
            hidden_size=TINY_HIDDEN,
            intermediate_size=TINY_INTERMEDIATE,
            num_hidden_layers=TINY_LAYERS,
            num_attention_heads=TINY_HEADS,
            num_key_value_heads=TINY_HEADS,
            head_dim=TINY_HIDDEN // TINY_HEADS,
            hidden_act="silu",
            pad_token_id=2,
            tie_word_embeddings=True,
            max_position_embeddings=194,
            rope_type="default",
            rope_theta=10_000.0,
            partial_rotary_factor=0.75,
            rope_interleave=True,
            encoder_num_hidden_layers=TINY_LAYERS,
            encoder_num_attention_heads=TINY_HEADS,
            encoder_num_key_value_heads=TINY_HEADS,
        )

    def test_moonshine_package_and_io(self):
        from mobius._builder import build_from_module
        from mobius.models import MoonshineForConditionalGeneration
        from mobius.tasks import SpeechToTextTask

        config = self._moonshine_config()
        package = build_from_module(
            MoonshineForConditionalGeneration(config),
            config,
            task=SpeechToTextTask(),
        )

        assert set(package) == {"encoder", "decoder"}
        encoder_inputs = {value.name for value in package["encoder"].graph.inputs}
        encoder_outputs = {value.name for value in package["encoder"].graph.outputs}
        decoder_inputs = {value.name for value in package["decoder"].graph.inputs}
        assert encoder_inputs == {"input_values", "attention_mask"}
        assert encoder_outputs == {
            "encoder_hidden_states",
            "encoder_attention_mask",
        }
        assert "encoder_attention_mask" in decoder_inputs
        assert "position_ids" in decoder_inputs
        for layer_idx in range(TINY_LAYERS):
            assert f"past_key_values.{layer_idx}.key" in decoder_inputs

    def test_moonshine_architecture_initializers(self):
        from mobius._builder import build_from_module
        from mobius.models import MoonshineForConditionalGeneration
        from mobius.tasks import SpeechToTextTask

        config = self._moonshine_config()
        package = build_from_module(
            MoonshineForConditionalGeneration(config),
            config,
            task=SpeechToTextTask(),
        )
        encoder_initializers = set(package["encoder"].graph.initializers)
        decoder_initializers = set(package["decoder"].graph.initializers)

        assert "encoder.conv1.weight" in encoder_initializers
        assert "encoder.conv1.bias" not in encoder_initializers
        assert "encoder.groupnorm.weight" in encoder_initializers
        assert "encoder.layers.0.input_layernorm.weight" in encoder_initializers
        assert "encoder.layers.0.input_layernorm.bias" not in encoder_initializers
        assert "encoder.layers.0.self_attn.q_proj.weight" in encoder_initializers
        assert "encoder.layers.0.self_attn.q_proj.bias" not in encoder_initializers
        assert "encoder.layers.0.mlp.fc1.bias" in encoder_initializers
        assert "encoder.layers.0.encoder_attn.q_proj.weight" not in encoder_initializers

        assert "decoder.embed_tokens.weight" in decoder_initializers
        assert "decoder.layers.0.encoder_attn.q_proj.weight" in decoder_initializers
        assert "decoder.layers.0.mlp.fc1.weight" in decoder_initializers
        assert "decoder.proj_out.weight" in decoder_initializers
        assert "GroupNormalization" in {node.op_type for node in package["encoder"].graph}
        decoder_ops = [node.op_type for node in package["decoder"].graph]
        assert decoder_ops.count("Swish") == TINY_LAYERS
        assert "Sigmoid" not in decoder_ops


class TestBuildGraphGlmAsr:
    """Verify GLM-ASR's audio encoder, projector, embedding, and decoder split."""

    def _config(self):
        from mobius._configs import GlmAsrConfig

        return _base_config(
            _config_cls=GlmAsrConfig,
            audio_token_id=100,
            audio=AudioConfig(
                d_model=64,
                encoder_layers=2,
                encoder_attention_heads=4,
                encoder_ffn_dim=256,
                encoder_head_dim=16,
                encoder_num_key_value_heads=4,
                encoder_partial_rotary_factor=0.5,
                encoder_rope_theta=10_000.0,
                encoder_layer_norm_eps=1e-5,
                num_mel_bins=128,
                max_source_positions=256,
                output_dim=64,
                activation_function="gelu",
                audio_token_id=100,
            ),
        )

    def _build(self):
        from mobius.models.glm_asr import GlmAsrForConditionalGeneration
        from mobius.tasks import GlmAsrSpeechLanguageTask

        config = self._config()
        module = GlmAsrForConditionalGeneration(config)
        return (
            config,
            module,
            build_from_module(module, config, task=GlmAsrSpeechLanguageTask()),
        )

    def test_package_contract_and_attention(self):
        config, _, pkg = self._build()

        assert set(pkg) == {"audio_encoder", "embedding", "decoder"}
        audio = pkg["audio_encoder"]
        assert {value.name for value in audio.graph.inputs} == {
            "input_features",
            "input_features_mask",
        }
        assert {value.name for value in audio.graph.outputs} == {
            "audio_features",
            "audio_feature_lengths",
        }
        attention_nodes = [node for node in audio.graph if node.op_type == "Attention"]
        assert len(attention_nodes) == config.audio.encoder_layers
        assert all(node.attributes["is_causal"].value == 0 for node in attention_nodes)
        assert all(len(node.inputs) == 3 or node.inputs[3] is None for node in attention_nodes)
        assert all(len(node.outputs) == 1 for node in attention_nodes)

        decoder_inputs = {value.name for value in pkg["decoder"].graph.inputs}
        assert {"inputs_embeds", "attention_mask", "position_ids"} <= decoder_inputs
        assert "past_key_values.0.key" in decoder_inputs
        assert "past_key_values.0.value" in decoder_inputs

    def test_configured_audio_and_projector_activations(self):
        from mobius.models.glm_asr import GlmAsrForConditionalGeneration
        from mobius.tasks import GlmAsrSpeechLanguageTask

        config = self._config()
        assert config.audio is not None
        config.audio.activation_function = "relu"
        config.projector_hidden_act = "silu"
        package = build_from_module(
            GlmAsrForConditionalGeneration(config),
            config,
            task=GlmAsrSpeechLanguageTask(),
        )

        audio_ops = [node.op_type for node in package["audio_encoder"].graph]
        assert audio_ops.count("Relu") == config.audio.encoder_layers
        assert audio_ops.count("Swish") == 1

    def test_checkpoint_weight_routing(self):
        import torch

        _, module, _ = self._build()
        tensor = torch.ones(1)
        routed = module.preprocess_weights(
            {
                "audio_tower.conv1.weight": tensor,
                "multi_modal_projector.linear_1.weight": tensor,
                "language_model.model.embed_tokens.weight": tensor,
                "language_model.model.layers.0.self_attn.q_proj.weight": tensor,
                "language_model.model.norm.weight": tensor,
                "language_model.lm_head.weight": tensor,
            }
        )

        assert set(routed) == {
            "audio_encoder.audio_tower.conv1.weight",
            "audio_encoder.multi_modal_projector.linear_1.weight",
            "embedding.embed_tokens.weight",
            "decoder.layers.0.self_attn.q_proj.weight",
            "decoder.norm.weight",
            "decoder.lm_head.weight",
        }

    def test_cuda_build_preserves_standard_decoder_attention(self):
        from mobius.models.glm_asr import GlmAsrForConditionalGeneration
        from mobius.tasks import GlmAsrSpeechLanguageTask

        config = self._config()
        config.dtype = ir.DataType.FLOAT16
        package = build_from_module(
            GlmAsrForConditionalGeneration(config),
            config,
            task=GlmAsrSpeechLanguageTask(),
            execution_provider="cuda",
        )
        decoder_ops = [node.op_type for node in package["decoder"].graph]
        assert decoder_ops.count("Attention") == config.num_hidden_layers
        assert "GroupQueryAttention" not in decoder_ops

    def test_three_stage_pipeline_runs_with_ort(self):
        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.rewrite_rules._testing_utils import fill_random_weights

        config, _, pkg = self._build()
        for model in pkg.values():
            fill_random_weights(model)

        mel_sequence = 32
        audio_session = OnnxModelSession(pkg["audio_encoder"])
        audio_outputs = audio_session.run(
            {
                "input_features": np.random.default_rng(0)
                .standard_normal((1, 128, mel_sequence))
                .astype(np.float32),
                "input_features_mask": np.ones((1, mel_sequence), dtype=np.int64),
            }
        )
        audio_session.close()
        assert audio_outputs["audio_feature_lengths"].tolist() == [4]
        audio_features = audio_outputs["audio_features"].reshape(-1, config.hidden_size)

        input_ids = np.array(
            [[1, 2, *([config.audio_token_id] * audio_features.shape[0]), 3]],
            dtype=np.int64,
        )
        embedding_session = OnnxModelSession(pkg["embedding"])
        inputs_embeds = embedding_session.run(
            {"input_ids": input_ids, "audio_features": audio_features}
        )["inputs_embeds"]
        embedding_session.close()

        sequence_length = input_ids.shape[1]
        decoder_inputs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": np.ones_like(input_ids),
            "position_ids": np.arange(sequence_length, dtype=np.int64)[None, :],
        }
        for layer in range(config.num_hidden_layers):
            for cache_kind in ("key", "value"):
                decoder_inputs[f"past_key_values.{layer}.{cache_kind}"] = np.zeros(
                    (1, config.num_key_value_heads, 0, config.head_dim),
                    dtype=np.float32,
                )
        decoder_session = OnnxModelSession(pkg["decoder"])
        decoder_outputs = decoder_session.run(decoder_inputs)
        decoder_session.close()

        assert decoder_outputs["logits"].shape == (
            1,
            sequence_length,
            config.vocab_size,
        )
        assert decoder_outputs["present.0.key"].shape[2] == sequence_length

    def test_registry_lookup(self):
        from mobius.models.glm_asr import GlmAsrForConditionalGeneration

        assert registry.get("glmasr") is GlmAsrForConditionalGeneration
        assert _default_task_for_model("glmasr") == "glmasr-speech-language"


class TestBuildGraphQwen3ASR:
    """Verify Qwen3-ASR 3-model split with SpeechLanguageTask."""

    def _asr_config(self):
        return _base_config(
            attn_qk_norm=True,
            hidden_act="silu",
            mrope_section=[24, 20, 20],
            mrope_interleaved=True,
            audio=AudioConfig(
                d_model=64,
                encoder_layers=2,
                encoder_attention_heads=4,
                encoder_ffn_dim=128,
                num_mel_bins=128,
                max_source_positions=256,
                downsample_hidden_size=32,
                output_dim=64,
                audio_token_id=100,
            ),
        )

    def test_package_builds_3_models(self):
        """Build Qwen3-ASR and verify 3-model package."""
        from mobius.models.qwen3_asr import (
            Qwen3ASRForConditionalGeneration,
        )
        from mobius.tasks import SpeechLanguageTask

        config = self._asr_config()
        module = Qwen3ASRForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=SpeechLanguageTask())

        assert "audio_encoder" in pkg
        assert "embedding" in pkg
        assert "decoder" in pkg

    def test_audio_encoder_io(self):
        """Verify audio encoder inputs/outputs."""
        from mobius.models.qwen3_asr import (
            Qwen3ASRForConditionalGeneration,
        )
        from mobius.tasks import SpeechLanguageTask

        config = self._asr_config()
        module = Qwen3ASRForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=SpeechLanguageTask())
        encoder = pkg["audio_encoder"]

        input_names = {inp.name for inp in encoder.graph.inputs}
        assert "input_features" in input_names
        # feature_attention_mask is required so the encoder can ignore
        # padded mel frames; without it the LLM emits degenerate loops
        # on any input padded by the standard HF processor.
        assert "feature_attention_mask" in input_names

        output_names = {out.name for out in encoder.graph.outputs}
        assert "audio_features" in output_names
        # audio_feature_lengths exposes the valid token count after
        # the encoder's 8x time downsampling so downstream callers can
        # crop padding-derived rows out of audio_features before the
        # embedding gather.
        assert "audio_feature_lengths" in output_names

    def test_audio_encoder_attention_uses_mask(self):
        """The encoder's Attention ops must receive the mask input.

        Guards against accidentally dropping the mask wiring inside
        the encoder forward — the graph builds without it but the
        encoder behaves the same as the pre-fix version.
        """
        from mobius.models.qwen3_asr import (
            Qwen3ASRForConditionalGeneration,
        )
        from mobius.tasks import SpeechLanguageTask

        config = self._asr_config()
        module = Qwen3ASRForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=SpeechLanguageTask())
        encoder = pkg["audio_encoder"]

        attention_nodes = [n for n in encoder.graph if n.op_type == "Attention"]
        assert attention_nodes, "audio encoder must contain Attention ops"
        for node in attention_nodes:
            # 4th positional input on op.Attention is attn_mask; must
            # be a wired value, not None / empty.
            assert len(node.inputs) >= 4
            assert node.inputs[3] is not None

    def test_embedding_io(self):
        """Verify embedding model inputs/outputs."""
        from mobius.models.qwen3_asr import (
            Qwen3ASRForConditionalGeneration,
        )
        from mobius.tasks import SpeechLanguageTask

        config = self._asr_config()
        module = Qwen3ASRForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=SpeechLanguageTask())
        embedding = pkg["embedding"]

        input_names = {inp.name for inp in embedding.graph.inputs}
        assert "input_ids" in input_names
        assert "audio_features" in input_names

        output_names = {out.name for out in embedding.graph.outputs}
        assert "inputs_embeds" in output_names

    def test_decoder_io(self):
        """Verify decoder has MRoPE position_ids and KV cache."""
        from mobius.models.qwen3_asr import (
            Qwen3ASRForConditionalGeneration,
        )
        from mobius.tasks import SpeechLanguageTask

        config = self._asr_config()
        module = Qwen3ASRForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=SpeechLanguageTask())
        decoder = pkg["decoder"]

        input_names = {inp.name for inp in decoder.graph.inputs}
        assert "inputs_embeds" in input_names
        assert "attention_mask" in input_names
        assert "position_ids" in input_names

        output_names = {out.name for out in decoder.graph.outputs}
        assert "logits" in output_names
        assert "present.0.key" in output_names
        assert "present.0.value" in output_names

    def test_registry_lookup(self):
        """Verify qwen3_asr is registered with speech-language task."""
        model_cls = registry.get("qwen3_asr")
        from mobius.models.qwen3_asr import (
            Qwen3ASRForConditionalGeneration,
        )

        assert model_cls is Qwen3ASRForConditionalGeneration
        assert _default_task_for_model("qwen3_asr") == "speech-language"

    def test_qwen3_forced_aligner_alias_resolves(self):
        """Verify qwen3_forced_aligner alias resolves to same class as qwen3_asr."""
        from mobius.models.qwen3_asr import (
            Qwen3ASRForConditionalGeneration,
        )

        assert registry.get("qwen3_forced_aligner") is Qwen3ASRForConditionalGeneration
        assert registry.get("qwen3_forced_aligner") is registry.get("qwen3_asr")
        assert _default_task_for_model("qwen3_forced_aligner") == "speech-language"

    def test_3model_pipeline_runs_with_ort(self):
        """Run audio_encoder → embedding with ORT.

        Guards against audio token count mismatches: the number of
        AUDIO_TOKEN_ID positions in input_ids must equal the number of
        audio feature rows from the encoder, otherwise the embedding
        Gather goes out of bounds.
        """
        import numpy as np

        from mobius._testing.ort_inference import (
            OnnxModelSession,
        )
        from mobius.models.qwen3_asr import (
            Qwen3ASRForConditionalGeneration,
        )
        from mobius.rewrite_rules._testing_utils import (
            fill_random_weights,
        )
        from mobius.tasks import SpeechLanguageTask

        config = self._asr_config()
        module = Qwen3ASRForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=SpeechLanguageTask())

        for model in pkg.values():
            fill_random_weights(model)

        # Step 1: Audio encoder — random mel input
        enc_sess = OnnxModelSession(pkg["audio_encoder"])
        mel_seq = 100
        mel = np.random.randn(1, config.audio.num_mel_bins, mel_seq).astype(np.float32)
        # Mark the last 20 mel frames as padding to exercise the mask
        # path. The encoder must crop the corresponding audio rows so
        # they don't leak into the embedding's Gather.
        feature_attention_mask = np.ones((1, mel_seq), dtype=np.int64)
        feature_attention_mask[:, -20:] = 0
        enc_out = enc_sess.run(
            {
                "input_features": mel,
                "feature_attention_mask": feature_attention_mask,
            }
        )
        audio_features = enc_out["audio_features"]
        audio_feature_lengths = enc_out["audio_feature_lengths"]
        # Crop padding-derived rows before passing to the embedding —
        # this mirrors what production callers must do.
        valid_len = int(audio_feature_lengths[0])
        assert 0 < valid_len <= audio_features.shape[1]
        audio_features = audio_features[:, :valid_len, :]
        num_audio_tokens = audio_features.shape[1]
        # Flatten to 2D: (num_audio_tokens, output_dim)
        audio_features_2d = audio_features.reshape(-1, audio_features.shape[-1])
        enc_sess.close()

        # Step 2: Embedding — mix text + audio tokens
        # Build input_ids with exactly num_audio_tokens audio pad tokens
        # Use the config's audio_token_id (must be within vocab_size)
        audio_token_id = config.audio.audio_token_id
        prefix = [1, 2, 3]  # mock system/user tokens
        suffix = [4, 5]  # mock footer tokens
        input_ids = np.array(
            [prefix + [audio_token_id] * num_audio_tokens + suffix],
            dtype=np.int64,
        )

        embed_sess = OnnxModelSession(pkg["embedding"])
        embed_out = embed_sess.run(
            {
                "input_ids": input_ids,
                "audio_features": audio_features_2d,
            }
        )
        inputs_embeds = embed_out["inputs_embeds"]
        embed_sess.close()

        seq_len = inputs_embeds.shape[1]
        assert seq_len == input_ids.shape[1]
        assert inputs_embeds.shape[2] == config.hidden_size

        # Step 3: Decoder — single forward pass with MRoPE
        decoder_sess = OnnxModelSession(pkg["decoder"])
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

        pos = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
        # MRoPE: (3, 1, seq_len)
        position_ids = np.stack([pos, pos, pos])

        dec_out = decoder_sess.run(
            {
                "inputs_embeds": inputs_embeds,
                "attention_mask": np.ones((1, seq_len), dtype=np.int64),
                "position_ids": position_ids,
                **past_kv,
            }
        )
        decoder_sess.close()

        logits = dec_out["logits"]
        assert logits.shape[0] == 1
        assert logits.shape[1] == seq_len


class TestBuildGraphFunASR:
    """Verify Fun-ASR-Nano 3-model split with FunASRSpeechLanguageTask."""

    def _fun_asr_config(self):
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
        )

    def test_package_builds_3_models(self):
        """Build Fun-ASR and verify 3-model package."""
        from mobius.models.fun_asr import FunASRForConditionalGeneration
        from mobius.tasks import FunASRSpeechLanguageTask

        config = self._fun_asr_config()
        module = FunASRForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=FunASRSpeechLanguageTask())

        assert "audio_encoder" in pkg
        assert "embedding" in pkg
        assert "decoder" in pkg

    def test_audio_encoder_io(self):
        """Verify audio encoder inputs/outputs."""
        from mobius.models.fun_asr import FunASRForConditionalGeneration
        from mobius.tasks import FunASRSpeechLanguageTask

        config = self._fun_asr_config()
        module = FunASRForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=FunASRSpeechLanguageTask())
        encoder = pkg["audio_encoder"]

        input_names = {inp.name for inp in encoder.graph.inputs}
        assert "input_features" in input_names

        output_names = {out.name for out in encoder.graph.outputs}
        assert "audio_features" in output_names

    def test_embedding_io(self):
        """Verify embedding model inputs/outputs."""
        from mobius.models.fun_asr import FunASRForConditionalGeneration
        from mobius.tasks import FunASRSpeechLanguageTask

        config = self._fun_asr_config()
        module = FunASRForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=FunASRSpeechLanguageTask())
        embedding = pkg["embedding"]

        input_names = {inp.name for inp in embedding.graph.inputs}
        assert "input_ids" in input_names
        assert "audio_features" in input_names

        output_names = {out.name for out in embedding.graph.outputs}
        assert "inputs_embeds" in output_names

    def test_decoder_io(self):
        """Verify decoder has standard position_ids and KV cache."""
        from mobius.models.fun_asr import FunASRForConditionalGeneration
        from mobius.tasks import FunASRSpeechLanguageTask

        config = self._fun_asr_config()
        module = FunASRForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=FunASRSpeechLanguageTask())
        decoder = pkg["decoder"]

        input_names = {inp.name for inp in decoder.graph.inputs}
        assert "inputs_embeds" in input_names
        assert "attention_mask" in input_names
        assert "position_ids" in input_names

        output_names = {out.name for out in decoder.graph.outputs}
        assert "logits" in output_names
        assert "present.0.key" in output_names
        assert "present.0.value" in output_names

    def test_registry_lookup(self):
        """Verify fun_asr is registered with fun-asr-speech-language task."""
        model_cls = registry.get("fun_asr")
        from mobius.models.fun_asr import FunASRForConditionalGeneration

        assert model_cls is FunASRForConditionalGeneration
        assert _default_task_for_model("fun_asr") == "fun-asr-speech-language"

    def test_3model_pipeline_runs_with_ort(self):
        """Run audio_encoder → embedding → decoder with ORT."""
        import numpy as np

        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.models.fun_asr import FunASRForConditionalGeneration
        from mobius.rewrite_rules._testing_utils import fill_random_weights
        from mobius.tasks import FunASRSpeechLanguageTask

        config = self._fun_asr_config()
        module = FunASRForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=FunASRSpeechLanguageTask())

        for model in pkg.values():
            fill_random_weights(model)

        # Step 1: Audio encoder — random fbank input
        # Sequence length must be even (temporal pooling halves it)
        input_dim = config.audio.input_size
        enc_sess = OnnxModelSession(pkg["audio_encoder"])
        fbank = np.random.randn(1, 100, input_dim).astype(np.float32)
        enc_out = enc_sess.run({"input_features": fbank})
        audio_features = enc_out["audio_features"]
        num_audio_tokens = audio_features.shape[1]
        audio_features_2d = audio_features.reshape(-1, audio_features.shape[-1])
        enc_sess.close()

        # Step 2: Embedding — mix text + audio tokens
        audio_token_id = config.audio.audio_token_id
        prefix = [1, 2, 3]
        suffix = [4, 5]
        input_ids = np.array(
            [prefix + [audio_token_id] * num_audio_tokens + suffix],
            dtype=np.int64,
        )

        embed_sess = OnnxModelSession(pkg["embedding"])
        embed_out = embed_sess.run(
            {
                "input_ids": input_ids,
                "audio_features": audio_features_2d,
            }
        )
        inputs_embeds = embed_out["inputs_embeds"]
        embed_sess.close()

        seq_len = inputs_embeds.shape[1]
        assert seq_len == input_ids.shape[1]
        assert inputs_embeds.shape[2] == config.hidden_size

        # Step 3: Decoder — single forward pass
        decoder_sess = OnnxModelSession(pkg["decoder"])
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

        pos = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
        dec_out = decoder_sess.run(
            {
                "inputs_embeds": inputs_embeds,
                "attention_mask": np.ones((1, seq_len), dtype=np.int64),
                "position_ids": pos,
                **past_kv,
            }
        )
        decoder_sess.close()

        logits = dec_out["logits"]
        assert logits.shape[0] == 1
        assert logits.shape[1] == seq_len


class TestBuildGraphQwen3TTS:
    """Verify Qwen3-TTS 4-model split with TTSTask."""

    def _tts_config(self):
        """Tiny config mimicking 0.6B: hidden_size=64 but text_hidden_size=128."""
        return _base_config(
            attn_qk_norm=True,
            hidden_act="silu",
            rope_scaling={
                "rope_type": "default",
                "mrope_section": [24, 20, 20],
            },
            mrope_interleaved=True,
            tts=TTSConfig(
                text_hidden_size=TINY_INTERMEDIATE,  # 128 (larger than hidden)
                text_vocab_size=TINY_VOCAB,
                num_code_groups=4,  # Fewer groups for testing
                code_predictor=CodePredictorConfig(
                    hidden_size=TINY_HIDDEN,
                    intermediate_size=TINY_INTERMEDIATE,
                    num_hidden_layers=2,
                    num_attention_heads=TINY_HEADS,
                    num_key_value_heads=TINY_KV_HEADS,
                    head_dim=TINY_HEAD_DIM,
                    vocab_size=TINY_VOCAB,
                    num_code_groups=4,
                ),
                speaker_encoder=SpeakerEncoderConfig(
                    mel_dim=32,
                    enc_dim=TINY_HIDDEN,
                    enc_channels=[16, 16, 16, 16, 48],
                    enc_kernel_sizes=[5, 3, 3, 3, 1],
                    enc_dilations=[1, 2, 3, 4, 1],
                    enc_attention_channels=16,
                    enc_res2net_scale=2,
                    enc_se_channels=16,
                ),
            ),
        )

    def test_package_builds_4_models(self):
        """Build Qwen3-TTS and verify 4-model package."""
        from mobius.models.qwen3_tts import (
            Qwen3TTSForConditionalGeneration,
        )
        from mobius.tasks import TTSTask

        config = self._tts_config()
        module = Qwen3TTSForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=TTSTask())

        assert "talker" in pkg
        assert "code_predictor" in pkg
        assert "embedding" in pkg
        assert "speaker_encoder" in pkg

    def test_talker_io(self):
        """Verify talker has inputs_embeds, logits, last_hidden_state, KV cache."""
        from mobius.models.qwen3_tts import (
            Qwen3TTSForConditionalGeneration,
        )
        from mobius.tasks import TTSTask

        config = self._tts_config()
        module = Qwen3TTSForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=TTSTask())
        talker = pkg["talker"]

        input_names = {inp.name for inp in talker.graph.inputs}
        assert "inputs_embeds" in input_names
        assert "attention_mask" in input_names
        assert "position_ids" in input_names
        assert "past_key_values.0.key" in input_names

        output_names = {out.name for out in talker.graph.outputs}
        assert "logits" in output_names
        assert "last_hidden_state" in output_names
        assert "present.0.key" in output_names

    def test_code_predictor_io(self):
        """Verify code predictor takes inputs_embeds and step_index."""
        from mobius.models.qwen3_tts import (
            Qwen3TTSForConditionalGeneration,
        )
        from mobius.tasks import TTSTask

        config = self._tts_config()
        module = Qwen3TTSForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=TTSTask())
        cp = pkg["code_predictor"]

        input_names = {inp.name for inp in cp.graph.inputs}
        assert "inputs_embeds" in input_names
        assert "step_index" in input_names
        assert "position_ids" in input_names
        assert "attention_mask" in input_names

        output_names = {out.name for out in cp.graph.outputs}
        assert "logits" in output_names

        # Verify 2D position_ids (1D RoPE, not 3D MRoPE)
        pos_input = next(i for i in cp.graph.inputs if i.name == "position_ids")
        assert len(pos_input.shape) == 2  # (batch, seq_len)

    def test_embedding_io(self):
        """Verify embedding model takes text_ids + codec_ids."""
        from mobius.models.qwen3_tts import (
            Qwen3TTSForConditionalGeneration,
        )
        from mobius.tasks import TTSTask

        config = self._tts_config()
        module = Qwen3TTSForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=TTSTask())
        embedding = pkg["embedding"]

        input_names = {inp.name for inp in embedding.graph.inputs}
        assert "text_ids" in input_names
        assert "codec_ids" in input_names

        output_names = {out.name for out in embedding.graph.outputs}
        assert "text_embeds" in output_names
        assert "codec_embeds" in output_names

    def test_speaker_encoder_io(self):
        """Verify speaker encoder takes mel_input."""
        from mobius.models.qwen3_tts import (
            Qwen3TTSForConditionalGeneration,
        )
        from mobius.tasks import TTSTask

        config = self._tts_config()
        module = Qwen3TTSForConditionalGeneration(config)
        pkg = build_from_module(module, config, task=TTSTask())
        se = pkg["speaker_encoder"]

        input_names = {inp.name for inp in se.graph.inputs}
        assert "mel_input" in input_names

        output_names = {out.name for out in se.graph.outputs}
        assert "speaker_embedding" in output_names

    def test_registry_lookup(self):
        """Verify qwen3_tts is registered with tts task."""
        model_cls = registry.get("qwen3_tts")
        from mobius.models.qwen3_tts import (
            Qwen3TTSForConditionalGeneration,
        )

        assert model_cls is Qwen3TTSForConditionalGeneration
        assert _default_task_for_model("qwen3_tts") == "tts"


@pytest.mark.parametrize("model_type,config_overrides", _SPEECH_MODEL_PARAMS)
class TestBuildSpeechGraph:
    """Verify speech/TTS/codec models build valid multi-model packages."""

    def test_package_builds(self, model_type: str, config_overrides: dict):
        """Build a speech model and verify expected sub-models."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task_name = _default_task_for_model(model_type)
        task = get_task(task_name)
        pkg = task.build(module, config)

        expected = _SPEECH_TASK_KEYS.get(task_name, set())
        for key in expected:
            assert key in pkg, f"{model_type} ({task_name}) should produce '{key}'"

        # Every sub-model should have a valid graph
        for name, model in pkg.items():
            assert model.graph is not None, f"{model_type}/{name} graph is None"
            assert len(model.graph.inputs) > 0, f"{model_type}/{name} has no inputs"
            assert len(model.graph.outputs) > 0, f"{model_type}/{name} has no outputs"

    def test_has_initializers(self, model_type: str, config_overrides: dict):
        """Verify all sub-models have non-empty initializers."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task_name = _default_task_for_model(model_type)
        task = get_task(task_name)
        pkg = task.build(module, config)

        for name, model in pkg.items():
            init_names = list(model.graph.initializers)
            assert len(init_names) > 0, f"{model_type}/{name} should have initializers"

    def test_onnx_checker_passes(self, model_type: str, config_overrides: dict):
        """Run ONNX CheckerPass on all sub-models."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task = get_task(_default_task_for_model(model_type))
        pkg = task.build(module, config)
        _run_onnx_checker(pkg, model_type)

    def test_outputs_have_shapes_and_dtypes(self, model_type: str, config_overrides: dict):
        """Verify shape inference populates all output shapes and dtypes."""
        config = _base_config(**config_overrides)
        model_cls = registry.get(model_type)
        module = model_cls(config)
        task = get_task(_default_task_for_model(model_type))
        pkg = task.build(module, config)
        _assert_outputs_have_shapes_and_dtypes(pkg, model_type)


class TestBuildGraphSortformer:
    """Verify Sortformer diarization builds with DiarizationTask."""

    def _sortformer_config(self):
        from mobius.models.sortformer import SortformerConfig

        # Tiny config: reduced widths/layers, structure identical to the real
        # nvidia/diar_streaming_sortformer_4spk model.
        return SortformerConfig(
            feat_in=32,
            fc_d_model=64,
            fc_num_layers=2,
            fc_num_heads=4,
            fc_ff_expansion=4,
            fc_conv_kernel=9,
            fc_subsampling_conv_channels=16,
            fc_subsampling_factor=8,
            tf_d_model=32,
            tf_num_layers=2,
            tf_num_heads=4,
            tf_inner_size=64,
            num_spks=4,
        )

    def test_package_builds(self):
        """Build Sortformer and verify a single 'model' component."""
        from mobius.models.sortformer import SortformerDiarizationModel
        from mobius.tasks import DiarizationTask

        config = self._sortformer_config()
        module = SortformerDiarizationModel(config)
        pkg = build_from_module(module, config, task=DiarizationTask())

        assert "model" in pkg

    def test_model_io(self):
        """Verify diarization input/output names and shapes."""
        from mobius.models.sortformer import SortformerDiarizationModel
        from mobius.tasks import DiarizationTask

        config = self._sortformer_config()
        module = SortformerDiarizationModel(config)
        pkg = build_from_module(module, config, task=DiarizationTask())
        model = pkg["model"]

        input_names = {inp.name for inp in model.graph.inputs}
        output_names = {out.name for out in model.graph.outputs}
        assert "input_features" in input_names
        assert "speaker_probs" in output_names

        # input_features: [batch, feat_in, time]
        feat_dim = model.graph.inputs[0].shape[1]
        assert feat_dim == config.feat_in
        # speaker_probs last dim == num_spks
        spk_dim = model.graph.outputs[0].shape[2]
        assert spk_dim == config.num_spks

    def test_has_initializers(self):
        """Verify encoder / transformer / head initializers are present."""
        from mobius.models.sortformer import SortformerDiarizationModel
        from mobius.tasks import DiarizationTask

        config = self._sortformer_config()
        module = SortformerDiarizationModel(config)
        pkg = build_from_module(module, config, task=DiarizationTask())
        init_names = list(pkg["model"].graph.initializers)

        assert any(n.startswith("encoder.") for n in init_names)
        assert any(n.startswith("transformer_encoder.") for n in init_names)
        assert any(n.startswith("sortformer_modules.") for n in init_names)

    def test_task_registry_lookup(self):
        """Verify the 'diarization' task resolves to DiarizationTask."""
        from mobius.tasks import DiarizationTask, get_task

        assert isinstance(get_task("diarization"), DiarizationTask)

    def test_runs_with_random_weights(self):
        """Fill random weights and run a forward pass through ORT."""
        import os
        import tempfile

        import onnxruntime as ort

        from mobius.models.sortformer import SortformerDiarizationModel
        from mobius.tasks import DiarizationTask

        config = self._sortformer_config()
        module = SortformerDiarizationModel(config)
        pkg = build_from_module(module, config, task=DiarizationTask())
        model = pkg["model"]

        # Fill each empty initializer with small random values.  Batch-norm
        # running variance must stay positive to avoid NaNs.
        for init in model.graph.initializers.values():
            if init.const_value is not None:
                continue
            shape = [d if isinstance(d, int) else 1 for d in init.shape]
            if "running_var" in init.name:
                arr = np.ones(shape, dtype=np.float32)
            elif "running_mean" in init.name:
                arr = np.zeros(shape, dtype=np.float32)
            else:
                arr = (np.random.randn(*shape) * 0.02).astype(np.float32)
            init.const_value = ir.tensor(arr, name=init.name)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.onnx")
            ir.save(model, path, external_data="model.onnx.data")
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            n_time = 40  # multiple of subsampling factor (8) -> 5 output frames
            feats = np.random.randn(1, config.feat_in, n_time).astype(np.float32)
            out = sess.run(None, {sess.get_inputs()[0].name: feats})[0]

        assert out.shape == (1, n_time // config.fc_subsampling_factor, config.num_spks)
        # Sigmoid output must lie in [0, 1].
        assert out.min() >= 0.0 and out.max() <= 1.0
