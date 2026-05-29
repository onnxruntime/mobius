# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for :func:`mobius.list_components`.

These tests mock the network calls so they run offline. They verify the
component-name discovery logic against the real ``mobius`` task registry,
diffusers class map, and component specs.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar
from unittest import mock

import pytest

import mobius
from mobius._introspection import (
    _components_for_task,
    _resolve_effective_model_type,
    list_components,
)


class TestComponentsForTask:
    """``_components_for_task`` returns the names declared by each task."""

    def test_single_component_task_returns_model(self):
        assert _components_for_task("text-generation") == ["model"]

    def test_seq2seq_returns_encoder_and_decoder(self):
        assert sorted(_components_for_task("seq2seq")) == ["decoder", "encoder"]

    def test_vae_returns_encoder_and_decoder(self):
        assert sorted(_components_for_task("vae")) == ["decoder", "encoder"]

    def test_multimodal_task_returns_all_components(self):
        # Phi4MM has vision_encoder, audio_encoder, embedding, decoder
        names = set(_components_for_task("phi4mm-multimodal"))
        assert {"vision_encoder", "audio_encoder", "embedding", "decoder"} <= names

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            _components_for_task("does-not-exist")


class TestResolveEffectiveModelType:
    """``_resolve_effective_model_type`` mirrors ``build()`` dispatch."""

    def test_plain_model_type_returned_as_is(self):
        cfg = SimpleNamespace(model_type="llama")
        assert _resolve_effective_model_type(cfg) == "llama"

    def test_talker_config_does_not_override_model_type(self):
        cfg = SimpleNamespace(model_type="qwen2_5_omni", talker_config=object())
        assert _resolve_effective_model_type(cfg) == "qwen2_5_omni"

    def test_qwen3_5_moe_with_vision_overrides_to_vl(self):
        cfg = SimpleNamespace(
            model_type="qwen3_5_moe",
            text_config=object(),
            vision_config=object(),
        )
        assert _resolve_effective_model_type(cfg) == "qwen3_5_moe_vl"

    def test_qwen3_5_moe_without_vision_stays_text_only(self):
        cfg = SimpleNamespace(
            model_type="qwen3_5_moe",
            text_config=object(),
            vision_config=None,
        )
        assert _resolve_effective_model_type(cfg) == "qwen3_5_moe"


class TestListComponentsTransformer:
    """Transformer-model code path of ``list_components``."""

    def _patch_autoconfig(self, hf_config):
        return mock.patch(
            "transformers.AutoConfig.from_pretrained",
            return_value=hf_config,
        )

    def test_causal_lm_returns_single_model(self):
        cfg = SimpleNamespace(model_type="llama")
        with self._patch_autoconfig(cfg):
            assert list_components("any/llama-id") == ["model"]

    def test_seq2seq_returns_encoder_decoder(self):
        cfg = SimpleNamespace(model_type="t5")
        with self._patch_autoconfig(cfg):
            assert sorted(list_components("any/t5-id")) == ["decoder", "encoder"]

    def test_explicit_task_overrides_default(self):
        cfg = SimpleNamespace(model_type="llama")
        with self._patch_autoconfig(cfg):
            assert set(list_components("any/llama-id", task="seq2seq")) == {
                "encoder",
                "decoder",
            }

    def test_qwen3_5_moe_with_vision_returns_vl_components(self):
        cfg = SimpleNamespace(
            model_type="qwen3_5_moe",
            text_config=object(),
            vision_config=object(),
        )
        with self._patch_autoconfig(cfg):
            names = set(list_components("any/qwen-vl-id"))
        assert {"decoder", "vision_encoder", "embedding"} <= names

    def test_unsupported_model_type_falls_back_to_diffusers(self):
        # AutoConfig succeeds with a model_type that isn't in registry
        # and isn't a CausalLM fallback candidate; we should attempt the
        # diffusers path (which then raises since model_index.json is
        # absent).
        cfg = SimpleNamespace(
            model_type="something-unsupported",
            is_encoder_decoder=False,
            architectures=[],
        )
        with (
            self._patch_autoconfig(cfg),
            mock.patch(
                "mobius._diffusers_builder._load_diffusers_pipeline_index",
                return_value=None,
            ),
            pytest.raises(ValueError, match="not a diffusers pipeline"),
        ):
            list_components("any/unknown-id")


class TestListComponentsDiffusers:
    """Diffusers-pipeline code path of ``list_components``."""

    SD3_INDEX: ClassVar[dict] = {
        "_class_name": "StableDiffusion3Pipeline",
        "_diffusers_version": "0.30.0",
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "CLIPTextModelWithProjection"],
        "text_encoder_2": ["transformers", "CLIPTextModelWithProjection"],
        "text_encoder_3": ["transformers", "T5EncoderModel"],
        "tokenizer": ["transformers", "CLIPTokenizer"],
        "tokenizer_2": ["transformers", "CLIPTokenizer"],
        "tokenizer_3": ["transformers", "T5TokenizerFast"],
        "transformer": ["diffusers", "SD3Transformer2DModel"],
        "vae": ["diffusers", "AutoencoderKL"],
    }

    def test_sd3_pipeline_returns_supported_components(self):
        with (
            mock.patch(
                "transformers.AutoConfig.from_pretrained",
                side_effect=OSError("no config.json"),
            ),
            mock.patch(
                "mobius._config_resolver._try_load_config_json",
                return_value=None,
            ),
            mock.patch(
                "mobius._diffusers_builder._load_diffusers_pipeline_index",
                return_value=self.SD3_INDEX,
            ),
        ):
            names = list_components("stabilityai/sd3-mock")

        # text_encoders and tokenizers from transformers are not in
        # _DIFFUSERS_CLASS_MAP and therefore skipped; only the transformer
        # and vae (single-component each) remain.
        assert "transformer" in names
        # VAE task is multi-component (encoder/decoder) so it gets flattened
        # to vae_encoder + vae_decoder.
        assert {"vae_encoder", "vae_decoder"} <= set(names)
        assert "text_encoder" not in names  # CLIPTextModelWithProjection not registered
        assert "scheduler" not in names

    def test_diffusers_skips_private_entries(self):
        index = {
            "_class_name": "TestPipeline",
            "_skip_me": ["x", "y"],
            "transformer": ["diffusers", "SD3Transformer2DModel"],
        }
        with (
            mock.patch(
                "transformers.AutoConfig.from_pretrained",
                side_effect=OSError("no config.json"),
            ),
            mock.patch(
                "mobius._config_resolver._try_load_config_json",
                return_value=None,
            ),
            mock.patch(
                "mobius._diffusers_builder._load_diffusers_pipeline_index",
                return_value=index,
            ),
        ):
            assert list_components("test/mock") == ["transformer"]

    def test_diffusers_pipeline_with_no_supported_components_raises(self):
        index = {
            "_class_name": "EmptyPipeline",
            "scheduler": ["diffusers", "DDIMScheduler"],
            "tokenizer": ["transformers", "CLIPTokenizer"],
        }
        with (
            mock.patch(
                "transformers.AutoConfig.from_pretrained",
                side_effect=OSError("no config.json"),
            ),
            mock.patch(
                "mobius._config_resolver._try_load_config_json",
                return_value=None,
            ),
            mock.patch(
                "mobius._diffusers_builder._load_diffusers_pipeline_index",
                return_value=index,
            ),
            pytest.raises(ValueError, match="No supported components"),
        ):
            list_components("test/empty")

    def test_no_model_index_and_not_a_supported_model_raises(self):
        with (
            mock.patch(
                "transformers.AutoConfig.from_pretrained",
                side_effect=OSError("no config.json"),
            ),
            mock.patch(
                "mobius._config_resolver._try_load_config_json",
                return_value=None,
            ),
            mock.patch(
                "mobius._diffusers_builder._load_diffusers_pipeline_index",
                return_value=None,
            ),
            pytest.raises(ValueError, match="not a diffusers pipeline"),
        ):
            list_components("nonexistent/model")


class TestPublicAPI:
    """``list_components`` is exported at the package root."""

    def test_exposed_from_top_level(self):
        assert mobius.list_components is list_components

    def test_in_dunder_all(self):
        assert "list_components" in mobius.__all__
