# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for GenaiConfigGenerator."""

from __future__ import annotations

import dataclasses
import json
import os

import pytest

from mobius.integrations.ort_genai.genai_config import (
    GenaiConfigGenerator,
)


class TestGenaiConfigGeneratorLLM:
    """Test genai_config generation for decoder-only LLMs."""

    def test_minimal_llm_config(self):
        """Generates a valid config with required fields only."""
        gen = GenaiConfigGenerator(
            "llama",
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
        )
        config = gen.generate()

        assert config["model"]["type"] == "llama"
        assert config["model"]["vocab_size"] == 32000
        assert config["model"]["context_length"] == 4096

        decoder = config["model"]["decoder"]
        assert decoder["hidden_size"] == 4096
        assert decoder["num_hidden_layers"] == 32
        assert decoder["num_attention_heads"] == 32
        assert decoder["num_key_value_heads"] == 8
        assert decoder["head_size"] == 128
        assert decoder["component"] == "decoder"

    def test_llm_decoder_inputs_have_input_ids(self):
        """LLM decoders receive input_ids, not inputs_embeds."""
        gen = GenaiConfigGenerator(
            "qwen2",
            vocab_size=151936,
            hidden_size=896,
            num_hidden_layers=24,
            num_attention_heads=14,
            num_key_value_heads=2,
            head_dim=64,
        )
        config = gen.generate()
        inputs = config["model"]["decoder"]["inputs"]
        assert "input_ids" in inputs
        assert "inputs_embeds" not in inputs
        assert inputs["past_key_names"] == "past_key_values.%d.key"
        assert inputs["past_value_names"] == "past_key_values.%d.value"

    def test_llm_decoder_outputs(self):
        """Decoder outputs include logits and present KV names."""
        gen = GenaiConfigGenerator(
            "llama",
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
        )
        config = gen.generate()
        outputs = config["model"]["decoder"]["outputs"]
        assert outputs["logits"] == "logits"
        assert outputs["present_key_names"] == "present.%d.key"
        assert outputs["present_value_names"] == "present.%d.value"

    def test_token_ids_included_when_set(self):
        """Token IDs are included in the model section."""
        gen = GenaiConfigGenerator(
            "llama",
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            bos_token_id=1,
            eos_token_id=[2, 3],
            pad_token_id=0,
        )
        config = gen.generate()
        assert config["model"]["bos_token_id"] == 1
        assert config["model"]["eos_token_id"] == [2, 3]
        assert config["model"]["pad_token_id"] == 0

    def test_token_ids_omitted_when_none(self):
        """Token IDs are not in the config when not provided."""
        gen = GenaiConfigGenerator(
            "llama",
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
        )
        config = gen.generate()
        assert "bos_token_id" not in config["model"]
        assert "eos_token_id" not in config["model"]
        assert "pad_token_id" not in config["model"]

    def test_search_params_defaults(self):
        """Search section has EP-agnostic defaults."""
        gen = GenaiConfigGenerator(
            "llama",
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            context_length=8192,
        )
        config = gen.generate()
        search = config["search"]
        # Sampling is enabled by default for chat/generation use cases
        assert search["do_sample"] is True
        assert search["num_beams"] == 1
        assert search["temperature"] == pytest.approx(1.0)
        assert search["top_k"] == 1
        assert search["top_p"] == pytest.approx(1.0)
        # max_length tracks the model's context window — not capped at any
        # EP-specific KV-buffer ceiling. EP-specific overrides land via the
        # variant's consumer_metadata.genai_config_overlay.
        assert search["max_length"] == 8192
        # past_present_share_buffer is an EP-specific knob and must NOT
        # appear in the package-level base genai_config.
        assert "past_present_share_buffer" not in search

    def test_search_params_no_ep_cap_for_large_context(self):
        """Large context_length is not capped — base config is EP-agnostic."""
        gen = GenaiConfigGenerator(
            "qwen2",
            vocab_size=151936,
            hidden_size=896,
            num_hidden_layers=24,
            num_attention_heads=14,
            num_key_value_heads=2,
            head_dim=64,
            context_length=131072,
        )
        config = gen.generate()
        # No WebGPU-style 4096 cap — packagers add EP-specific limits via
        # variant overlays.
        assert config["search"]["max_length"] == 131072

    def test_decoder_component_default(self):
        """Decoder block carries a 'component' field, default 'decoder'."""
        gen = GenaiConfigGenerator(
            "llama",
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
        )
        config = gen.generate()
        assert config["model"]["decoder"]["component"] == "decoder"

    def test_decoder_component_custom(self):
        """Custom decoder component name is emitted unchanged."""
        gen = GenaiConfigGenerator(
            "llama",
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            decoder_component="text_decoder",
        )
        config = gen.generate()
        assert config["model"]["decoder"]["component"] == "text_decoder"

    def test_decoder_block_has_no_filename_or_session_options(self):
        """Package-level base genai_config must not carry EP-coupled keys."""
        gen = GenaiConfigGenerator(
            "llama",
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
        )
        decoder = gen.generate()["model"]["decoder"]
        assert "filename" not in decoder
        assert "session_options" not in decoder
        assert "provider_options" not in decoder


class TestGenaiConfigGeneratorVLM:
    """Test genai_config generation for vision-language models."""

    def _make_vlm_gen(self) -> GenaiConfigGenerator:
        return GenaiConfigGenerator(
            "qwen2_5_vl",
            vocab_size=151936,
            hidden_size=3584,
            num_hidden_layers=28,
            num_attention_heads=28,
            num_key_value_heads=4,
            head_dim=128,
            bos_token_id=151643,
            eos_token_id=[151645, 151643],
            pad_token_id=151643,
        ).with_vision(
            image_token_id=151655,
            vision_start_token_id=151652,
            video_token_id=151656,
        )

    def test_vlm_has_vision_section(self):
        """VLM config includes vision model section."""
        config = self._make_vlm_gen().generate()
        vision = config["model"]["vision"]
        assert vision["component"] == "vision_encoder"
        assert vision["spatial_merge_size"] == 2
        assert vision["inputs"]["pixel_values"] == "pixel_values"
        assert vision["outputs"]["image_features"] == "image_features"

    def test_vlm_has_embedding_section(self):
        """VLM config includes embedding model section."""
        config = self._make_vlm_gen().generate()
        emb = config["model"]["embedding"]
        assert emb["component"] == "embedding"
        assert emb["inputs"]["input_ids"] == "input_ids"
        assert emb["inputs"]["image_features"] == "image_features"
        assert emb["outputs"]["inputs_embeds"] == "inputs_embeds"

    def test_vlm_decoder_uses_inputs_embeds(self):
        """VLM decoder receives inputs_embeds, not input_ids."""
        config = self._make_vlm_gen().generate()
        inputs = config["model"]["decoder"]["inputs"]
        assert "inputs_embeds" in inputs
        assert "input_ids" not in inputs

    def test_vlm_decoder_has_decoder_component(self):
        """VLM decoder block carries component='decoder'."""
        config = self._make_vlm_gen().generate()
        decoder = config["model"]["decoder"]
        assert decoder["component"] == "decoder"

    def test_vlm_blocks_have_no_filename_or_session_options(self):
        """VLM vision/embedding blocks must not carry EP-coupled keys."""
        config = self._make_vlm_gen().generate()
        for role in ("vision", "embedding"):
            block = config["model"][role]
            assert "filename" not in block
            assert "session_options" not in block
            assert "provider_options" not in block

    def test_vlm_token_ids_at_model_level(self):
        """VLM-specific token IDs are at the model level."""
        config = self._make_vlm_gen().generate()
        model = config["model"]
        assert model["image_token_id"] == 151655
        assert model["vision_start_token_id"] == 151652
        assert model["video_token_id"] == 151656

    def test_vlm_without_video_token(self):
        """VLM without video_token_id omits the field."""
        gen = GenaiConfigGenerator(
            "gemma3",
            vocab_size=262144,
            hidden_size=2048,
            num_hidden_layers=26,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=256,
        ).with_vision(image_token_id=255999)
        config = gen.generate()
        assert config["model"]["image_token_id"] == 255999
        assert "video_token_id" not in config["model"]

    def test_any_model_type_with_vision_uses_inputs_embeds(self):
        """with_vision() controls decoder input, not the model_type."""
        gen = GenaiConfigGenerator(
            "llama",  # LLM model type, but used as VLM decoder
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
        ).with_vision(image_token_id=128256)
        config = gen.generate()
        inputs = config["model"]["decoder"]["inputs"]
        assert "inputs_embeds" in inputs
        assert "input_ids" not in inputs

    def test_image_token_id_required(self):
        """with_vision() requires image_token_id."""
        gen = GenaiConfigGenerator(
            "gemma3",
            vocab_size=262144,
            hidden_size=2048,
            num_hidden_layers=26,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=256,
        )
        with pytest.raises(TypeError):
            gen.with_vision()  # missing image_token_id

    def test_custom_embedding_input_names(self):
        """embedding_input_names overrides the default embedding inputs."""
        gen = GenaiConfigGenerator(
            "gemma4",
            vocab_size=262144,
            hidden_size=2048,
            num_hidden_layers=26,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=256,
        ).with_vision(
            image_token_id=255999,
            embedding_input_names={
                "input_ids": "input_ids",
                "image_features": "image_features",
                "custom_input": "custom_input",
            },
        )
        config = gen.generate()
        emb = config["model"]["embedding"]
        assert emb["inputs"]["custom_input"] == "custom_input"
        assert emb["inputs"]["input_ids"] == "input_ids"


class TestGenaiConfigFromConfig:
    """Test from_config() factory method."""

    def test_from_dataclass_config(self):
        """Creates generator from a config-like dataclass."""

        @dataclasses.dataclass
        class FakeConfig:
            vocab_size: int = 32000
            hidden_size: int = 4096
            num_hidden_layers: int = 32
            num_attention_heads: int = 32
            num_key_value_heads: int = 8
            head_dim: int = 128
            pad_token_id: int = 0
            max_position_embeddings: int = 8192

        cfg = FakeConfig()
        gen = GenaiConfigGenerator.from_config(cfg, "llama")
        config = gen.generate()
        assert config["model"]["vocab_size"] == 32000
        assert config["model"]["pad_token_id"] == 0
        # context_length picks up max_position_embeddings
        assert config["model"]["context_length"] == 8192

    def test_sentinel_pad_token_id_ignored(self):
        """pad_token_id == -42 (DEFAULT_INT sentinel) is ignored."""

        @dataclasses.dataclass
        class FakeConfig:
            vocab_size: int = 32000
            hidden_size: int = 4096
            num_hidden_layers: int = 32
            num_attention_heads: int = 32
            num_key_value_heads: int = 8
            head_dim: int = 128
            pad_token_id: int = -42

        cfg = FakeConfig()
        gen = GenaiConfigGenerator.from_config(cfg, "llama")
        config = gen.generate()
        assert "pad_token_id" not in config["model"]

    def test_context_length_default_when_no_max_pos(self):
        """Uses default 4096 when max_position_embeddings not present."""

        @dataclasses.dataclass
        class FakeConfig:
            vocab_size: int = 32000
            hidden_size: int = 4096
            num_hidden_layers: int = 32
            num_attention_heads: int = 32
            num_key_value_heads: int = 8
            head_dim: int = 128

        cfg = FakeConfig()
        gen = GenaiConfigGenerator.from_config(cfg, "llama")
        config = gen.generate()
        assert config["model"]["context_length"] == 4096


class TestGenaiConfigWrite:
    """Test writing genai_config.json to disk."""

    def test_write_creates_valid_json(self, tmp_path):
        """write() produces a valid JSON file."""
        gen = GenaiConfigGenerator(
            "llama",
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
        )
        path = gen.write(str(tmp_path))
        assert os.path.isfile(path)
        assert path.endswith("genai_config.json")

        with open(path) as f:
            loaded = json.load(f)
        assert loaded["model"]["type"] == "llama"
        assert "search" in loaded

    def test_write_roundtrips_vlm(self, tmp_path):
        """VLM config survives write + read roundtrip."""
        gen = GenaiConfigGenerator(
            "qwen2_5_vl",
            vocab_size=151936,
            hidden_size=3584,
            num_hidden_layers=28,
            num_attention_heads=28,
            num_key_value_heads=4,
            head_dim=128,
        ).with_vision(
            image_token_id=151655,
            spatial_merge_size=2,
        )
        path = gen.write(str(tmp_path))
        with open(path) as f:
            loaded = json.load(f)
        assert "vision" in loaded["model"]
        assert "embedding" in loaded["model"]
        assert loaded["model"]["image_token_id"] == 151655


class TestExplicitDecoderInputs:
    """Test decoder_inputs parameter overrides defaults."""

    def test_explicit_decoder_inputs_used(self):
        """When decoder_inputs is provided, it replaces the defaults."""
        decoder_inputs = {
            "inputs_embeds": "inputs_embeds",
            "input_ids": "input_ids",
            "attention_mask": "attention_mask",
            "position_ids": "position_ids",
            "past_key_names": "past_key_values.%d.key",
            "past_value_names": "past_key_values.%d.value",
        }
        gen = GenaiConfigGenerator(
            "gemma4",
            vocab_size=262144,
            hidden_size=2048,
            num_hidden_layers=26,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=256,
            decoder_inputs=decoder_inputs,
        ).with_vision(
            image_token_id=255999,
            spatial_merge_size=None,
        )
        config = gen.generate()
        result = config["model"]["decoder"]["inputs"]
        assert "input_ids" in result
        assert "inputs_embeds" in result
        assert result["past_key_names"] == "past_key_values.%d.key"

    def test_default_used_when_decoder_inputs_none(self):
        """When decoder_inputs is None, default mapping is used."""
        gen = GenaiConfigGenerator(
            "llama",
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
        )
        config = gen.generate()
        inputs = config["model"]["decoder"]["inputs"]
        # LLM default: input_ids, not inputs_embeds
        assert "input_ids" in inputs
        assert "inputs_embeds" not in inputs


class TestGenaiConfigGeneratorMultimodal:
    """Test genai_config generation for multimodal (vision + speech)."""

    def _make_phi4mm_gen(self) -> GenaiConfigGenerator:
        return (
            GenaiConfigGenerator(
                "phi4mm",
                vocab_size=200064,
                hidden_size=3072,
                num_hidden_layers=32,
                num_attention_heads=24,
                num_key_value_heads=8,
                head_dim=128,
                context_length=131072,
                bos_token_id=199999,
                eos_token_id=[200020, 199999],
                pad_token_id=199999,
            )
            .with_vision(
                image_token_id=200010,
                spatial_merge_size=None,
                config_filename="image_processor.json",
                input_names={
                    "pixel_values": "pixel_values",
                    "image_sizes": "image_sizes",
                },
            )
            .with_audio(
                audio_token_id=200011,
            )
        )

    def test_multimodal_has_all_four_sections(self):
        """Phi4MM config has decoder, vision, speech, and embedding."""
        config = self._make_phi4mm_gen().generate()
        model = config["model"]
        assert "decoder" in model
        assert "vision" in model
        assert "speech" in model
        assert "embedding" in model

    def test_audio_section_has_correct_inputs(self):
        """Audio section has audio_embeds, audio_sizes, mode."""
        config = self._make_phi4mm_gen().generate()
        audio = config["model"]["speech"]
        assert audio["component"] == "audio_encoder"
        assert audio["config_filename"] == "audio_processor.json"
        assert audio["inputs"]["audio_embeds"] == "audio_embeds"
        assert audio["inputs"]["audio_sizes"] == "audio_sizes"
        assert audio["inputs"]["audio_projection_mode"] == "audio_projection_mode"
        assert audio["outputs"]["audio_features"] == "audio_features"

    def test_audio_block_has_no_filename_or_session_options(self):
        """Audio block must not carry EP-coupled keys."""
        config = self._make_phi4mm_gen().generate()
        speech = config["model"]["speech"]
        assert "filename" not in speech
        assert "session_options" not in speech
        assert "provider_options" not in speech

    def test_boa_token_id_in_audio_config(self):
        """boa_token_id is included when passed to with_audio()."""
        gen = GenaiConfigGenerator(
            "gemma4",
            vocab_size=256,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=16,
        )
        gen.with_vision(image_token_id=200010)
        gen.with_audio(audio_token_id=200011, boa_token_id=256000)
        config = gen.generate()
        assert config["model"]["boa_token_id"] == 256000

    def test_boa_token_id_absent_when_not_set(self):
        """boa_token_id is omitted when not provided."""
        gen = GenaiConfigGenerator(
            "gemma4",
            vocab_size=256,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=16,
        )
        gen.with_vision(image_token_id=200010)
        gen.with_audio(audio_token_id=200011)
        config = gen.generate()
        assert "boa_token_id" not in config["model"]

    def test_vision_custom_inputs(self):
        """Vision section uses custom input names (no image_grid_thw)."""
        config = self._make_phi4mm_gen().generate()
        vision = config["model"]["vision"]
        assert vision["inputs"]["pixel_values"] == "pixel_values"
        assert vision["inputs"]["image_sizes"] == "image_sizes"
        assert "image_grid_thw" not in vision["inputs"]
        assert "spatial_merge_size" not in vision

    def test_embedding_includes_audio_features(self):
        """Embedding inputs include audio_features when speech enabled."""
        config = self._make_phi4mm_gen().generate()
        emb = config["model"]["embedding"]
        assert emb["inputs"]["input_ids"] == "input_ids"
        assert emb["inputs"]["image_features"] == "image_features"
        assert emb["inputs"]["audio_features"] == "audio_features"

    def test_embedding_no_audio_without_audio(self):
        """Embedding inputs don't have audio_features without audio."""
        gen = GenaiConfigGenerator(
            "qwen2_5_vl",
            vocab_size=151936,
            hidden_size=3584,
            num_hidden_layers=28,
            num_attention_heads=28,
            num_key_value_heads=4,
            head_dim=128,
        ).with_vision(image_token_id=151655)
        config = gen.generate()
        emb = config["model"]["embedding"]
        assert "audio_features" not in emb["inputs"]

    def test_audio_token_id_at_model_level(self):
        """audio_token_id is set at the model level."""
        config = self._make_phi4mm_gen().generate()
        assert config["model"]["audio_token_id"] == 200011

    def test_decoder_uses_inputs_embeds(self):
        """Multimodal decoder receives inputs_embeds."""
        config = self._make_phi4mm_gen().generate()
        inputs = config["model"]["decoder"]["inputs"]
        assert "inputs_embeds" in inputs
        assert "input_id" not in inputs

    def test_audio_only_uses_inputs_embeds(self):
        """Audio-only (no vision) still uses inputs_embeds."""
        gen = GenaiConfigGenerator(
            "whisper",
            vocab_size=51865,
            hidden_size=512,
            num_hidden_layers=6,
            num_attention_heads=8,
            num_key_value_heads=8,
            head_dim=64,
        ).with_audio()
        config = gen.generate()
        inputs = config["model"]["decoder"]["inputs"]
        assert "inputs_embeds" in inputs

    def test_chaining_returns_self(self):
        """with_vision() and with_audio() return self for chaining."""
        gen = GenaiConfigGenerator(
            "phi4mm",
            vocab_size=200064,
            hidden_size=3072,
            num_hidden_layers=32,
            num_attention_heads=24,
            num_key_value_heads=8,
            head_dim=128,
        )
        result = gen.with_vision(image_token_id=200010).with_audio()
        assert result is gen
