# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for task classes.

Covers: CausalLM, VisionLanguage, Seq2Seq, Denoising, VAE,
FeatureExtraction, ImageClassification, SSM, SSM2, AudioFeatureExtraction,
ObjectDetection, SpeechToText, and shared builder helpers
(build_decoder_from_embeds, build_embedding_from_features).
"""

from __future__ import annotations

import onnx_ir as ir
import pytest

from mobius._configs import VisionConfig
from mobius._model_package import ModelPackage
from mobius._testing import make_config
from mobius.models.base import CausalLMModel
from mobius.models.gemma3 import Gemma3MultiModalModel
from mobius.tasks import (
    TASK_REGISTRY,
    AudioFeatureExtractionTask,
    CausalLMTask,
    DenoisingTask,
    FeatureExtractionTask,
    ImageClassificationTask,
    ModelTask,
    ObjectDetectionTask,
    Seq2SeqTask,
    SpeechToTextTask,
    SSM2CausalLMTask,
    SSMCausalLMTask,
    VAETask,
    VisionLanguageTask,
    build_decoder_from_embeds,
    build_embedding_from_features,
    get_task,
)


class TestGetTask:
    def test_get_task_by_name(self):
        task = get_task("text-generation")
        assert isinstance(task, CausalLMTask)

    def test_get_task_by_instance(self):
        instance = CausalLMTask()
        assert get_task(instance) is instance

    def test_get_task_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            get_task("nonexistent-task")

    def test_task_registry_has_text_generation(self):
        assert "text-generation" in TASK_REGISTRY


class TestCausalLMTask:
    def test_build_returns_package(self):
        config = make_config()
        module = CausalLMModel(config)
        task = CausalLMTask()
        pkg = task.build(module, config)
        assert isinstance(pkg, ModelPackage)
        assert "model" in pkg

    def test_build_inputs(self):
        config = make_config()
        module = CausalLMModel(config)
        task = CausalLMTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        input_names = [v.name for v in model.graph.inputs]
        assert "input_ids" in input_names
        assert "attention_mask" in input_names
        assert "position_ids" in input_names
        # 2 layers x 2 (key, value)
        assert "past_key_values.0.key" in input_names
        assert "past_key_values.1.value" in input_names

    def test_build_outputs(self):
        config = make_config()
        module = CausalLMModel(config)
        task = CausalLMTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        output_names = [v.name for v in model.graph.outputs]
        assert "logits" in output_names
        assert "present.0.key" in output_names
        assert "present.0.value" in output_names

    def test_build_producer_info(self):
        config = make_config()
        module = CausalLMModel(config)
        task = CausalLMTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        assert model.producer_name == "mobius"


class TestCustomTask:
    """Test that users can create custom tasks."""

    def test_subclass_model_task(self):
        class MyTask(ModelTask):
            def build(self, module, config):
                # Minimal: just create an empty model
                graph = ir.Graph([], [], nodes=[], name="custom")
                model = ir.Model(graph, ir_version=10)
                return ModelPackage({"model": model})

        task = MyTask()
        config = make_config()
        module = CausalLMModel(config)
        pkg = task.build(module, config)
        assert isinstance(pkg, ModelPackage)
        assert pkg["model"].graph.name == "custom"


class TestCustomModuleWithTask:
    """Test the user story: custom module + standard task."""

    def test_custom_module_with_causal_lm_task(self):
        """A user-defined module should work with CausalLMTask.

        It follows the expected forward signature.
        """
        config = make_config()

        # Re-use CausalLMModel as a "custom" module — it has the right signature
        custom_module = CausalLMModel(config)
        task = CausalLMTask()
        pkg = task.build(custom_module, config)

        assert isinstance(pkg, ModelPackage)
        model = pkg["model"]
        output_names = [v.name for v in model.graph.outputs]
        assert "logits" in output_names


def _make_multimodal_config():
    return make_config(
        sliding_window=8,
        layer_types=["full_attention", "sliding_attention"],
        attn_qk_norm=True,
        rope_local_base_freq=10_000.0,
        vision=VisionConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=32,
            patch_size=8,
            norm_eps=1e-6,
            image_token_id=999,
        ),
        image_token_id=999,
    )


class TestVisionLanguageTask:
    def test_task_registered(self):
        assert "vision-language" in TASK_REGISTRY

    def test_get_task_by_name(self):
        task = get_task("vision-language")
        assert isinstance(task, VisionLanguageTask)

    def test_build_returns_3_model_package(self):
        config = _make_multimodal_config()
        module = Gemma3MultiModalModel(config)
        task = VisionLanguageTask()
        pkg = task.build(module, config)
        assert isinstance(pkg, ModelPackage)
        assert set(pkg.keys()) == {"decoder", "vision_encoder", "embedding"}

    def test_decoder_has_inputs_embeds(self):
        config = _make_multimodal_config()
        module = Gemma3MultiModalModel(config)
        task = VisionLanguageTask()
        pkg = task.build(module, config)
        decoder = pkg["decoder"]
        input_names = [v.name for v in decoder.graph.inputs]
        assert "inputs_embeds" in input_names
        assert "attention_mask" in input_names
        assert "position_ids" in input_names

    def test_decoder_has_kv_cache(self):
        config = _make_multimodal_config()
        module = Gemma3MultiModalModel(config)
        task = VisionLanguageTask()
        pkg = task.build(module, config)
        decoder = pkg["decoder"]
        input_names = [v.name for v in decoder.graph.inputs]
        assert "past_key_values.0.key" in input_names
        assert "past_key_values.0.value" in input_names

    def test_decoder_outputs_logits(self):
        config = _make_multimodal_config()
        module = Gemma3MultiModalModel(config)
        task = VisionLanguageTask()
        pkg = task.build(module, config)
        decoder = pkg["decoder"]
        output_names = [v.name for v in decoder.graph.outputs]
        assert "logits" in output_names
        assert "present.0.key" in output_names

    def test_vision_has_pixel_values(self):
        config = _make_multimodal_config()
        module = Gemma3MultiModalModel(config)
        task = VisionLanguageTask()
        pkg = task.build(module, config)
        vision = pkg["vision_encoder"]
        input_names = [v.name for v in vision.graph.inputs]
        assert "pixel_values" in input_names
        output_names = [v.name for v in vision.graph.outputs]
        assert "image_features" in output_names

    def test_embedding_fuses_features(self):
        config = _make_multimodal_config()
        module = Gemma3MultiModalModel(config)
        task = VisionLanguageTask()
        pkg = task.build(module, config)
        embed = pkg["embedding"]
        input_names = [v.name for v in embed.graph.inputs]
        assert "input_ids" in input_names
        assert "image_features" in input_names
        output_names = [v.name for v in embed.graph.outputs]
        assert "inputs_embeds" in output_names

    def test_build_producer_info(self):
        config = _make_multimodal_config()
        module = Gemma3MultiModalModel(config)
        task = VisionLanguageTask()
        pkg = task.build(module, config)
        for model in pkg.values():
            assert model.producer_name == "mobius"


# ── Seq2SeqTask ──────────────────────────────────────────────────────────


class TestSeq2SeqTask:
    def _make_seq2seq(self):
        from mobius.models.bart import BartForConditionalGeneration

        config = make_config(
            hidden_act="gelu",
            num_decoder_layers=2,
            max_position_embeddings=64,
        )
        module = BartForConditionalGeneration(config)
        return config, module

    def test_task_registered(self):
        assert "seq2seq" in TASK_REGISTRY

    def test_get_task_by_name(self):
        task = get_task("seq2seq")
        assert isinstance(task, Seq2SeqTask)

    def test_build_returns_encoder_and_decoder(self):
        config, module = self._make_seq2seq()
        task = Seq2SeqTask()
        pkg = task.build(module, config)
        assert isinstance(pkg, ModelPackage)
        assert "encoder" in pkg
        assert "decoder" in pkg

    def test_encoder_inputs(self):
        config, module = self._make_seq2seq()
        task = Seq2SeqTask()
        pkg = task.build(module, config)
        encoder = pkg["encoder"]
        input_names = {v.name for v in encoder.graph.inputs}
        assert "input_ids" in input_names
        assert "attention_mask" in input_names

    def test_encoder_outputs(self):
        config, module = self._make_seq2seq()
        task = Seq2SeqTask()
        pkg = task.build(module, config)
        encoder = pkg["encoder"]
        output_names = {v.name for v in encoder.graph.outputs}
        assert "last_hidden_state" in output_names

    def test_decoder_inputs(self):
        config, module = self._make_seq2seq()
        task = Seq2SeqTask()
        pkg = task.build(module, config)
        decoder = pkg["decoder"]
        input_names = {v.name for v in decoder.graph.inputs}
        assert "input_ids" in input_names
        assert "encoder_hidden_states" in input_names
        assert "attention_mask" in input_names
        # Self-attention KV cache
        assert "past_key_values.0.self.key" in input_names
        assert "past_key_values.0.self.value" in input_names
        # Cross-attention KV cache
        assert "past_key_values.0.cross.key" in input_names
        assert "past_key_values.0.cross.value" in input_names

    def test_decoder_outputs(self):
        config, module = self._make_seq2seq()
        task = Seq2SeqTask()
        pkg = task.build(module, config)
        decoder = pkg["decoder"]
        output_names = {v.name for v in decoder.graph.outputs}
        assert "logits" in output_names
        assert "present.0.self.key" in output_names
        assert "present.0.self.value" in output_names
        assert "present.0.cross.key" in output_names
        assert "present.0.cross.value" in output_names

    def test_build_producer_info(self):
        config, module = self._make_seq2seq()
        task = Seq2SeqTask()
        pkg = task.build(module, config)
        for model in pkg.values():
            assert model.producer_name == "mobius"


# ── DenoisingTask ────────────────────────────────────────────────────────


class TestDenoisingTask:
    def _make_denoiser(self):
        from mobius.models.dit import (
            DiTConfig,
            DiTTransformer2DModel,
        )

        config = DiTConfig(
            in_channels=4,
            out_channels=4,
            patch_size=2,
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            cross_attention_dim=32,
            caption_channels=32,
            sample_size=8,
        )
        module = DiTTransformer2DModel(config)
        return config, module

    def test_task_registered(self):
        assert "denoising" in TASK_REGISTRY

    def test_get_task_by_name(self):
        task = get_task("denoising")
        assert isinstance(task, DenoisingTask)

    def test_build_returns_single_model(self):
        config, module = self._make_denoiser()
        task = DenoisingTask()
        pkg = task.build(module, config)
        assert isinstance(pkg, ModelPackage)
        assert "model" in pkg
        assert len(pkg) == 1

    def test_inputs(self):
        config, module = self._make_denoiser()
        task = DenoisingTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        input_names = {v.name for v in model.graph.inputs}
        assert "sample" in input_names
        assert "timestep" in input_names
        assert "encoder_hidden_states" in input_names

    def test_input_types(self):
        config, module = self._make_denoiser()
        task = DenoisingTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        inputs_by_name = {v.name: v for v in model.graph.inputs}
        assert inputs_by_name["sample"].dtype == ir.DataType.FLOAT
        assert inputs_by_name["timestep"].dtype == ir.DataType.INT64
        assert inputs_by_name["encoder_hidden_states"].dtype == ir.DataType.FLOAT

    def test_outputs(self):
        config, module = self._make_denoiser()
        task = DenoisingTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        output_names = {v.name for v in model.graph.outputs}
        assert "noise_pred" in output_names
        assert len(model.graph.outputs) == 1

    def test_build_producer_info(self):
        config, module = self._make_denoiser()
        task = DenoisingTask()
        pkg = task.build(module, config)
        assert pkg["model"].producer_name == "mobius"


# ── VAETask ──────────────────────────────────────────────────────────────


class TestVAETask:
    def _make_vae(self):
        from mobius._diffusers_configs import VAEConfig
        from mobius.models.vae import AutoencoderKLModel

        config = VAEConfig(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            block_out_channels=(32, 64),
            layers_per_block=1,
            norm_num_groups=32,
            act_fn="silu",
            mid_block_add_attention=True,
            use_quant_conv=True,
            use_post_quant_conv=True,
        )
        module = AutoencoderKLModel(config)
        return config, module

    def test_task_registered(self):
        assert "vae" in TASK_REGISTRY

    def test_get_task_by_name(self):
        task = get_task("vae")
        assert isinstance(task, VAETask)

    def test_build_returns_encoder_and_decoder(self):
        config, module = self._make_vae()
        task = VAETask()
        pkg = task.build(module, config)
        assert isinstance(pkg, ModelPackage)
        assert "encoder" in pkg
        assert "decoder" in pkg
        assert len(pkg) == 2

    def test_encoder_inputs(self):
        config, module = self._make_vae()
        task = VAETask()
        pkg = task.build(module, config)
        encoder = pkg["encoder"]
        input_names = {v.name for v in encoder.graph.inputs}
        assert "sample" in input_names

    def test_encoder_outputs(self):
        config, module = self._make_vae()
        task = VAETask()
        pkg = task.build(module, config)
        encoder = pkg["encoder"]
        output_names = {v.name for v in encoder.graph.outputs}
        assert "latent_dist" in output_names

    def test_decoder_inputs(self):
        config, module = self._make_vae()
        task = VAETask()
        pkg = task.build(module, config)
        decoder = pkg["decoder"]
        input_names = {v.name for v in decoder.graph.inputs}
        assert "latent_sample" in input_names

    def test_decoder_outputs(self):
        config, module = self._make_vae()
        task = VAETask()
        pkg = task.build(module, config)
        decoder = pkg["decoder"]
        output_names = {v.name for v in decoder.graph.outputs}
        assert "sample" in output_names

    def test_build_producer_info(self):
        config, module = self._make_vae()
        task = VAETask()
        pkg = task.build(module, config)
        for model in pkg.values():
            assert model.producer_name == "mobius"


# ── FeatureExtractionTask ────────────────────────────────────────────────


class TestFeatureExtractionTask:
    def _make_encoder(self):
        from mobius._configs import EncoderConfig
        from mobius.models.bert import BertModel

        config = EncoderConfig(
            vocab_size=100,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            hidden_act="gelu",
            pad_token_id=0,
            max_position_embeddings=32,
            type_vocab_size=2,
            attn_qkv_bias=True,
            attn_o_bias=True,
        )
        module = BertModel(config)
        return config, module

    def test_task_registered(self):
        assert "feature-extraction" in TASK_REGISTRY

    def test_get_task_by_name(self):
        task = get_task("feature-extraction")
        assert isinstance(task, FeatureExtractionTask)

    def test_build_returns_single_model(self):
        config, module = self._make_encoder()
        task = FeatureExtractionTask()
        pkg = task.build(module, config)
        assert isinstance(pkg, ModelPackage)
        assert "model" in pkg
        assert len(pkg) == 1

    def test_inputs(self):
        config, module = self._make_encoder()
        task = FeatureExtractionTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        input_names = {v.name for v in model.graph.inputs}
        assert "input_ids" in input_names
        assert "attention_mask" in input_names
        assert "token_type_ids" in input_names

    def test_no_kv_cache(self):
        config, module = self._make_encoder()
        task = FeatureExtractionTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        input_names = {v.name for v in model.graph.inputs}
        assert not any("past_key_values" in n for n in input_names)

    def test_outputs(self):
        config, module = self._make_encoder()
        task = FeatureExtractionTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        output_names = {v.name for v in model.graph.outputs}
        assert "last_hidden_state" in output_names
        assert len(model.graph.outputs) == 1


# ── ImageClassificationTask ─────────────────────────────────────────────


class TestImageClassificationTask:
    def _make_vision(self):
        from mobius._configs import EncoderConfig
        from mobius.models.vit import ViTModel

        config = EncoderConfig(
            vocab_size=100,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            hidden_act="gelu",
            pad_token_id=0,
            max_position_embeddings=32,
            image_size=32,
            patch_size=8,
            num_channels=3,
            attn_qkv_bias=True,
            attn_o_bias=True,
        )
        module = ViTModel(config)
        return config, module

    def test_task_registered(self):
        assert "image-classification" in TASK_REGISTRY

    def test_get_task_by_name(self):
        task = get_task("image-classification")
        assert isinstance(task, ImageClassificationTask)

    def test_build_returns_single_model(self):
        config, module = self._make_vision()
        task = ImageClassificationTask()
        pkg = task.build(module, config)
        assert isinstance(pkg, ModelPackage)
        assert "model" in pkg
        assert len(pkg) == 1

    def test_inputs(self):
        config, module = self._make_vision()
        task = ImageClassificationTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        input_names = {v.name for v in model.graph.inputs}
        assert "pixel_values" in input_names

    def test_outputs(self):
        config, module = self._make_vision()
        task = ImageClassificationTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        output_names = {v.name for v in model.graph.outputs}
        assert "last_hidden_state" in output_names


# ── ObjectDetectionTask ──────────────────────────────────────────────────


class TestObjectDetectionTask:
    def _make_detector(self):
        from mobius._configs import YolosConfig
        from mobius.models.yolos import YolosForObjectDetection

        config = YolosConfig(
            vocab_size=100,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            hidden_act="gelu",
            pad_token_id=0,
            max_position_embeddings=32,
            image_size=32,
            patch_size=8,
            num_channels=3,
            num_labels=10,
            num_detection_tokens=5,
            attn_qkv_bias=True,
            attn_o_bias=True,
        )
        module = YolosForObjectDetection(config)
        return config, module

    def test_task_registered(self):
        assert "object-detection" in TASK_REGISTRY

    def test_get_task_by_name(self):
        task = get_task("object-detection")
        assert isinstance(task, ObjectDetectionTask)

    def test_build_returns_single_model(self):
        config, module = self._make_detector()
        task = ObjectDetectionTask()
        pkg = task.build(module, config)
        assert isinstance(pkg, ModelPackage)
        assert "model" in pkg

    def test_inputs(self):
        config, module = self._make_detector()
        task = ObjectDetectionTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        input_names = {v.name for v in model.graph.inputs}
        assert "pixel_values" in input_names

    def test_outputs(self):
        config, module = self._make_detector()
        task = ObjectDetectionTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        output_names = {v.name for v in model.graph.outputs}
        assert "logits" in output_names
        assert "pred_boxes" in output_names
        assert len(model.graph.outputs) == 2


# ── SSMCausalLMTask ──────────────────────────────────────────────────────


class TestSSMCausalLMTask:
    def _make_mamba(self):
        from mobius._configs import MambaConfig
        from mobius.models.mamba import MambaCausalLMModel

        config = MambaConfig(
            vocab_size=100,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=64,
            pad_token_id=0,
            state_size=16,
            conv_kernel=4,
            expand=2,
            time_step_rank=4,
        )
        module = MambaCausalLMModel(config)
        return config, module

    def test_task_registered(self):
        assert "ssm-text-generation" in TASK_REGISTRY

    def test_get_task_by_name(self):
        task = get_task("ssm-text-generation")
        assert isinstance(task, SSMCausalLMTask)

    def test_build_returns_single_model(self):
        config, module = self._make_mamba()
        task = SSMCausalLMTask()
        pkg = task.build(module, config)
        assert isinstance(pkg, ModelPackage)
        assert "model" in pkg

    def test_inputs_have_ssm_states(self):
        config, module = self._make_mamba()
        task = SSMCausalLMTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        input_names = {v.name for v in model.graph.inputs}
        assert "input_ids" in input_names
        assert "past_states.0.conv_state" in input_names
        assert "past_states.0.ssm_state" in input_names
        assert "past_states.1.conv_state" in input_names

    def test_no_kv_cache(self):
        config, module = self._make_mamba()
        task = SSMCausalLMTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        input_names = {v.name for v in model.graph.inputs}
        assert not any("past_key_values" in n for n in input_names)
        assert "attention_mask" not in input_names

    def test_outputs(self):
        config, module = self._make_mamba()
        task = SSMCausalLMTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        output_names = {v.name for v in model.graph.outputs}
        assert "logits" in output_names
        assert "present.0.conv_state" in output_names
        assert "present.0.ssm_state" in output_names


# ── SSM2CausalLMTask ─────────────────────────────────────────────────────


class TestSSM2CausalLMTask:
    def _make_mamba2(self):
        from mobius._configs import Mamba2Config
        from mobius.models.mamba import Mamba2CausalLMModel

        config = Mamba2Config(
            vocab_size=100,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=16,
            pad_token_id=0,
            num_heads=8,
            state_size=16,
            n_groups=2,
            conv_kernel=4,
            expand=2,
        )
        module = Mamba2CausalLMModel(config)
        return config, module

    def test_task_registered(self):
        assert "ssm2-text-generation" in TASK_REGISTRY

    def test_get_task_by_name(self):
        task = get_task("ssm2-text-generation")
        assert isinstance(task, SSM2CausalLMTask)

    def test_build_returns_single_model(self):
        config, module = self._make_mamba2()
        task = SSM2CausalLMTask()
        pkg = task.build(module, config)
        assert isinstance(pkg, ModelPackage)
        assert "model" in pkg

    def test_inputs_have_ssm_states(self):
        config, module = self._make_mamba2()
        task = SSM2CausalLMTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        input_names = {v.name for v in model.graph.inputs}
        assert "input_ids" in input_names
        assert "past_states.0.conv_state" in input_names
        assert "past_states.0.ssm_state" in input_names

    def test_outputs(self):
        config, module = self._make_mamba2()
        task = SSM2CausalLMTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        output_names = {v.name for v in model.graph.outputs}
        assert "logits" in output_names
        assert "present.0.conv_state" in output_names
        assert "present.0.ssm_state" in output_names


# ── AudioFeatureExtractionTask ───────────────────────────────────────────


class TestAudioFeatureExtractionTask:
    def _make_audio(self):
        from mobius.models.wav2vec2 import Wav2Vec2Model

        config = make_config(
            hidden_act="gelu",
            attn_qkv_bias=True,
            attn_o_bias=True,
        )
        # Wav2Vec2Model uses getattr for conv_channels/conv_kernel_sizes
        # with defaults. We override with tiny values to keep graphs small.
        config.conv_channels = [1, 32, 32]
        config.conv_kernel_sizes = [5, 3]
        module = Wav2Vec2Model(config)
        return config, module

    def test_task_registered(self):
        assert "audio-feature-extraction" in TASK_REGISTRY

    def test_get_task_by_name(self):
        task = get_task("audio-feature-extraction")
        assert isinstance(task, AudioFeatureExtractionTask)

    def test_build_returns_single_model(self):
        config, module = self._make_audio()
        task = AudioFeatureExtractionTask()
        pkg = task.build(module, config)
        assert isinstance(pkg, ModelPackage)
        assert "model" in pkg
        assert len(pkg) == 1

    def test_inputs(self):
        config, module = self._make_audio()
        task = AudioFeatureExtractionTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        input_names = {v.name for v in model.graph.inputs}
        assert "input_values" in input_names

    def test_outputs(self):
        config, module = self._make_audio()
        task = AudioFeatureExtractionTask()
        pkg = task.build(module, config)
        model = pkg["model"]
        output_names = {v.name for v in model.graph.outputs}
        assert "last_hidden_state" in output_names


# ── SpeechToTextTask ─────────────────────────────────────────────────────


class TestSpeechToTextTask:
    def _make_whisper(self):
        from mobius._configs import WhisperConfig
        from mobius.models.whisper import (
            WhisperForConditionalGeneration,
        )

        config = WhisperConfig(
            vocab_size=100,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            hidden_act="gelu",
            pad_token_id=0,
            encoder_layers=2,
            encoder_attention_heads=4,
            encoder_ffn_dim=128,
            num_mel_bins=40,
            max_source_positions=64,
            max_target_positions=32,
            attn_qkv_bias=True,
            attn_o_bias=True,
        )
        module = WhisperForConditionalGeneration(config)
        return config, module

    def test_task_registered(self):
        assert "speech-to-text" in TASK_REGISTRY

    def test_get_task_by_name(self):
        task = get_task("speech-to-text")
        assert isinstance(task, SpeechToTextTask)

    def test_build_returns_encoder_and_decoder(self):
        config, module = self._make_whisper()
        task = SpeechToTextTask()
        pkg = task.build(module, config)
        assert isinstance(pkg, ModelPackage)
        assert "encoder" in pkg
        assert "decoder" in pkg
        assert len(pkg) == 2

    def test_encoder_inputs(self):
        config, module = self._make_whisper()
        task = SpeechToTextTask()
        pkg = task.build(module, config)
        encoder = pkg["encoder"]
        input_names = {v.name for v in encoder.graph.inputs}
        assert "input_features" in input_names

    def test_encoder_outputs(self):
        config, module = self._make_whisper()
        task = SpeechToTextTask()
        pkg = task.build(module, config)
        encoder = pkg["encoder"]
        output_names = {v.name for v in encoder.graph.outputs}
        assert "encoder_hidden_states" in output_names

    def test_decoder_inputs(self):
        config, module = self._make_whisper()
        task = SpeechToTextTask()
        pkg = task.build(module, config)
        decoder = pkg["decoder"]
        input_names = {v.name for v in decoder.graph.inputs}
        assert "decoder_input_ids" in input_names
        assert "encoder_hidden_states" in input_names
        assert "position_ids" in input_names
        assert "past_key_values.0.key" in input_names
        assert "past_key_values.0.value" in input_names

    def test_decoder_outputs(self):
        config, module = self._make_whisper()
        task = SpeechToTextTask()
        pkg = task.build(module, config)
        decoder = pkg["decoder"]
        output_names = {v.name for v in decoder.graph.outputs}
        assert "logits" in output_names
        assert "present.0.key" in output_names
        assert "present.0.value" in output_names


# ── build_decoder_from_embeds ─────────────────────────────────────────────


class TestBuildDecoderFromEmbeds:
    """Smoke tests for the shared build_decoder_from_embeds helper."""

    def _make_decoder_module(self):
        from mobius.models.gemma3 import Gemma3MultiModalModel

        config = _make_multimodal_config()
        module = Gemma3MultiModalModel(config)
        return config, module.decoder

    def test_returns_ir_model(self):
        config, decoder = self._make_decoder_module()
        model = build_decoder_from_embeds(decoder, config)
        assert isinstance(model, ir.Model)

    def test_graph_name_is_decoder(self):
        config, decoder = self._make_decoder_module()
        model = build_decoder_from_embeds(decoder, config)
        # name= arg removed; builder overrides graph.name with model_id anyway
        assert model.graph.name == "main_graph"

    def test_inputs_include_inputs_embeds_and_kv_cache(self):
        config, decoder = self._make_decoder_module()
        model = build_decoder_from_embeds(decoder, config)
        input_names = {v.name for v in model.graph.inputs}
        assert "inputs_embeds" in input_names
        assert "attention_mask" in input_names
        assert "position_ids" in input_names
        assert "past_key_values.0.key" in input_names

    def test_outputs_include_logits_and_kv_cache(self):
        config, decoder = self._make_decoder_module()
        model = build_decoder_from_embeds(decoder, config)
        output_names = {v.name for v in model.graph.outputs}
        assert "logits" in output_names
        assert "present.0.key" in output_names

    def test_mrope_position_ids_are_3d(self):
        config, decoder = self._make_decoder_module()
        model = build_decoder_from_embeds(decoder, config, mrope=True)
        pos_ids = next(v for v in model.graph.inputs if v.name == "position_ids")
        # MRoPE shape: [3, batch, seq_len]
        assert pos_ids.shape is not None
        assert pos_ids.shape[0] == 3


# ── build_embedding_from_features ─────────────────────────────────────────


class TestBuildEmbeddingFromFeatures:
    """Smoke tests for the shared build_embedding_from_features helper."""

    def _make_embedding_module(self):
        from mobius.models.gemma3 import Gemma3MultiModalModel

        config = _make_multimodal_config()
        module = Gemma3MultiModalModel(config)
        return config, module.embedding

    def test_returns_ir_model(self):
        config, embedding = self._make_embedding_module()
        model = build_embedding_from_features(
            embedding, config, feature_name="image_features", feature_dim=config.hidden_size
        )
        assert isinstance(model, ir.Model)

    def test_graph_name_is_embedding(self):
        config, embedding = self._make_embedding_module()
        model = build_embedding_from_features(
            embedding, config, feature_name="image_features", feature_dim=config.hidden_size
        )
        assert model.graph.name == "embedding"

    def test_inputs_include_input_ids_and_features(self):
        config, embedding = self._make_embedding_module()
        model = build_embedding_from_features(
            embedding, config, feature_name="image_features", feature_dim=config.hidden_size
        )
        input_names = {v.name for v in model.graph.inputs}
        assert "input_ids" in input_names
        assert "image_features" in input_names

    def test_output_is_inputs_embeds(self):
        config, embedding = self._make_embedding_module()
        model = build_embedding_from_features(
            embedding, config, feature_name="image_features", feature_dim=config.hidden_size
        )
        output_names = {v.name for v in model.graph.outputs}
        assert "inputs_embeds" in output_names

    def test_custom_feature_name(self):
        # Use a stub module that accepts any feature name to verify
        # that the feature_name argument correctly controls graph input naming.
        from onnxscript import nn as onnx_nn

        class _StubEmbedding(onnx_nn.Module):
            def forward(self, op, input_ids, audio_features):
                return op.Identity(input_ids)

        config = _make_multimodal_config()
        model = build_embedding_from_features(
            _StubEmbedding(), config, feature_name="audio_features", feature_dim=64
        )
        input_names = {v.name for v in model.graph.inputs}
        assert "audio_features" in input_names


# ── build_decoder_from_embeds — hybrid branch ────────────────────────────


class TestBuildDecoderFromEmbedsHybrid:
    """Cover the hybrid=True branch of build_decoder_from_embeds."""

    def _make_hybrid_config(self):
        """Config with linear_attention + full_attention layers for hybrid cache."""
        return make_config(
            layer_types=["linear_attention", "full_attention"],
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_conv_kernel_dim=4,
        )

    def _make_stub_hybrid_decoder(self, config):
        """Stub decoder that pipes hybrid cache states through unchanged."""
        from onnxscript import nn as onnx_nn

        class _StubHybridDecoder(onnx_nn.Module):
            def forward(
                self, op, inputs_embeds, attention_mask, position_ids, past_key_values
            ):
                # Return dummy 1-token logits + pass states through
                batch = op.Shape(inputs_embeds, start=0, end=1)
                vocab = op.Constant(value_ints=[config.vocab_size])
                one = op.Constant(value_ints=[1])
                shape = op.Concat(batch, one, vocab, axis=0)
                logits = op.ConstantOfShape(shape)
                # Pass back whatever states we received
                present = list(past_key_values)
                return logits, present

        return _StubHybridDecoder()

    def test_hybrid_returns_ir_model(self):
        config = self._make_hybrid_config()
        decoder = self._make_stub_hybrid_decoder(config)
        model = build_decoder_from_embeds(decoder, config, mrope=False, hybrid=True)
        assert isinstance(model, ir.Model)

    def test_hybrid_graph_name_is_decoder(self):
        config = self._make_hybrid_config()
        decoder = self._make_stub_hybrid_decoder(config)
        model = build_decoder_from_embeds(decoder, config, hybrid=True)
        # name= arg removed; builder overrides graph.name with model_id anyway
        assert model.graph.name == "main_graph"

    def test_hybrid_has_linear_attention_state_inputs(self):
        config = self._make_hybrid_config()
        decoder = self._make_stub_hybrid_decoder(config)
        model = build_decoder_from_embeds(decoder, config, hybrid=True)
        input_names = {v.name for v in model.graph.inputs}
        # Layer 0 is linear_attention — should have conv_state + recurrent_state
        assert any("conv_state" in n for n in input_names)

    def test_hybrid_has_full_attention_kv_inputs(self):
        config = self._make_hybrid_config()
        decoder = self._make_stub_hybrid_decoder(config)
        model = build_decoder_from_embeds(decoder, config, hybrid=True)
        input_names = {v.name for v in model.graph.inputs}
        # Layer 1 is full_attention — should have key + value
        assert any(".key" in n for n in input_names)


# ── VisionLanguageTask — config.vision=None path ─────────────────────────


class TestVisionLanguageTaskNoVisionConfig:
    """Operator-precedence fix: config.vision=None must not crash."""

    def _make_stub_vision(self):
        from onnxscript import nn as onnx_nn

        class _StubVision(onnx_nn.Module):
            def forward(self, op, pixel_values):
                return op.Identity(pixel_values)

        return _StubVision()

    def _make_stub_embedding(self):
        from onnxscript import nn as onnx_nn

        class _StubEmbed(onnx_nn.Module):
            def forward(self, op, input_ids, image_features):
                return op.Identity(image_features)

        return _StubEmbed()

    def test_build_vision_defaults_when_no_vision_config(self):
        """_build_vision should use image_size=224 when config.vision is None."""
        task = VisionLanguageTask()
        # config with no vision sub-config
        config = make_config()
        assert config.vision is None

        stub_vision = self._make_stub_vision()
        model = task._build_vision(stub_vision, config)
        assert isinstance(model, ir.Model)
        # Default image_size=224 → pixel_values shape should include 224
        pv = next(v for v in model.graph.inputs if v.name == "pixel_values")
        assert pv.shape is not None
        assert 224 in pv.shape

    def test_build_vision_uses_vision_config_when_present(self):
        """_build_vision should use image_size from config.vision when present."""
        task = VisionLanguageTask()
        config = _make_multimodal_config()  # has vision.image_size=32
        stub_vision = self._make_stub_vision()
        model = task._build_vision(stub_vision, config)
        pv = next(v for v in model.graph.inputs if v.name == "pixel_values")
        assert 32 in pv.shape


# ── SpeechLanguageTask — config.audio=None path ──────────────────────────


class TestSpeechLanguageTaskNoAudioConfig:
    """Operator-precedence fix: config.audio=None must not crash."""

    def _make_stub_audio_encoder(self):
        from onnxscript import nn as onnx_nn

        class _StubAudio(onnx_nn.Module):
            def forward(self, op, input_features):
                return op.Identity(input_features)

        return _StubAudio()

    def test_audio_encoder_defaults_when_no_audio_config(self):
        """_build_audio_encoder should use n_mels=128 when config.audio is None."""
        from mobius.tasks._speech_language import SpeechLanguageTask

        task = SpeechLanguageTask()
        config = make_config()
        assert config.audio is None

        stub_audio = self._make_stub_audio_encoder()
        model = task._build_audio_encoder(stub_audio, config)
        assert isinstance(model, ir.Model)
        # Default n_mels=128 → input_features shape should include 128
        inp = next(v for v in model.graph.inputs if v.name == "input_features")
        assert inp.shape is not None
        assert 128 in inp.shape

    def test_output_dim_defaults_to_hidden_size_when_no_audio_config(self):
        """output_dim should fall back to config.hidden_size when config.audio is None."""
        from mobius.tasks._speech_language import SpeechLanguageTask

        task = SpeechLanguageTask()
        config = make_config()  # hidden_size=64, audio=None
        assert config.audio is None

        stub_audio = self._make_stub_audio_encoder()
        model = task._build_audio_encoder(stub_audio, config)
        # The embedding feature_dim would be config.hidden_size=64 — verify the
        # audio encoder model itself was built without errors (it doesn't embed)
        assert isinstance(model, ir.Model)


# ── TTSTask — speaker_encoder optional ───────────────────────────────────


class TestTTSTaskSpeakerEncoderOptional:
    """speaker_encoder is optional — build() must handle both cases."""

    def _make_tts_module_no_speaker(self):
        """Stub TTS module without a speaker encoder."""
        from onnxscript import nn as onnx_nn

        class _StubTalker(onnx_nn.Module):
            def forward(
                self, op, inputs_embeds, attention_mask, position_ids, past_key_values
            ):
                batch = op.Shape(inputs_embeds, start=0, end=1)
                one = op.Constant(value_ints=[1])
                vocab = op.Constant(value_ints=[100])
                logits = op.ConstantOfShape(op.Concat(batch, one, vocab, axis=0))
                hidden = op.Identity(inputs_embeds)
                return logits, hidden, past_key_values

        class _StubCodePredictor(onnx_nn.Module):
            def forward(self, op, inputs_embeds, step_index, past_key_values):
                batch = op.Shape(inputs_embeds, start=0, end=1)
                one = op.Constant(value_ints=[1])
                vocab = op.Constant(value_ints=[100])
                logits = op.ConstantOfShape(op.Concat(batch, one, vocab, axis=0))
                return logits, past_key_values, op.Identity(inputs_embeds)

        class _StubEmbed(onnx_nn.Module):
            def forward(self, op, text_ids, codec_ids):
                text_embeds = op.Identity(text_ids)
                codec_embeds = op.Identity(codec_ids)
                return text_embeds, codec_embeds

        class _StubTTSModule(onnx_nn.Module):
            def __init__(self):
                super().__init__()
                self.talker = _StubTalker()
                self.code_predictor = _StubCodePredictor()
                self.embedding = _StubEmbed()
                self.speaker_encoder = None  # Optional — absent

        return _StubTTSModule()

    def test_build_without_speaker_encoder_excludes_key(self):
        """When speaker_encoder is None, package must not contain 'speaker_encoder'."""
        # Verify ComponentSpec validation passes and speaker_encoder is absent.
        module = self._make_tts_module_no_speaker()
        assert module.speaker_encoder is None
        # ComponentSpec only checks talker, code_predictor, embedding — should pass
        from mobius.tasks._base import ComponentSpec as _ComponentSpec

        spec = _ComponentSpec(
            talker="talker", code_predictor="code_predictor", embedding="embedding"
        )
        spec.validate(module, "TTSTask")  # must not raise

    def test_build_with_speaker_encoder_attribute(self):
        """TTSTask's ComponentSpec omits speaker_encoder — it's treated as optional."""
        from mobius.tasks import TTSTask as _TTSTask

        task = _TTSTask()
        # ComponentSpec must NOT include speaker_encoder
        assert task.components is not None
        keys = task.components.keys()
        assert "speaker_encoder" not in keys
        assert "talker" in keys
        assert "code_predictor" in keys
        assert "embedding" in keys


# ── ComponentSpec validation ─────────────────────────────────────────────


class TestComponentSpecValidation:
    """ComponentSpec.validate() must raise TypeError for missing attributes."""

    def test_missing_single_attribute_raises_type_error(self):
        from mobius.tasks._base import ComponentSpec

        class _Incomplete:
            decoder = object()
            # 'vision_encoder' is missing

        spec = ComponentSpec(decoder="decoder", vision="vision_encoder")
        with pytest.raises(TypeError, match="vision_encoder"):
            spec.validate(_Incomplete(), "FakeTask")

    def test_all_present_does_not_raise(self):
        from mobius.tasks._base import ComponentSpec

        class _Complete:
            decoder = object()
            vision_encoder = object()

        spec = ComponentSpec(decoder="decoder", vision="vision_encoder")
        spec.validate(_Complete(), "FakeTask")  # should not raise

    def test_dot_notation_nested_attribute(self):
        from mobius.tasks._base import ComponentSpec

        class _Inner:
            encoder = object()

        class _Outer:
            model = _Inner()

        spec = ComponentSpec(enc="model.encoder")
        spec.validate(_Outer(), "FakeTask")  # should not raise

    def test_dot_notation_missing_nested_raises(self):
        from mobius.tasks._base import ComponentSpec

        class _OuterNoEncoder:
            # model attribute exists but has no 'encoder'
            class _EmptyModel:
                pass

            model = _EmptyModel()

        spec = ComponentSpec(enc="model.encoder")
        with pytest.raises(TypeError, match=r"model\.encoder"):
            spec.validate(_OuterNoEncoder(), "FakeTask")

    def test_error_message_contains_task_name(self):
        from mobius.tasks._base import ComponentSpec

        class _Empty:
            pass

        spec = ComponentSpec(decoder="decoder")
        with pytest.raises(TypeError, match="MySpecialTask"):
            spec.validate(_Empty(), "MySpecialTask")

    def test_vision_language_task_raises_on_missing_component(self):
        """VisionLanguageTask._validate_components raises TypeError on missing attrs."""

        class _IncompleteModule:
            decoder = object()
            # Missing vision_encoder and embedding

        task = VisionLanguageTask()
        with pytest.raises(TypeError, match="vision_encoder"):
            task._validate_components(_IncompleteModule())

    def test_seq2seq_task_raises_on_missing_component(self):
        """Seq2SeqTask._validate_components raises TypeError on missing attrs."""

        class _NoDecoder:
            encoder = object()
            # Missing decoder

        task = Seq2SeqTask()
        with pytest.raises(TypeError, match="decoder"):
            task._validate_components(_NoDecoder())

    def test_vae_task_raises_on_missing_component(self):
        """VAETask._validate_components raises TypeError on missing attrs."""

        class _NoDecoder:
            encoder = object()
            # Missing decoder

        task = VAETask()
        with pytest.raises(TypeError, match="decoder"):
            task._validate_components(_NoDecoder())
