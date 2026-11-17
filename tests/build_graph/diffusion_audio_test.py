# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Diffusion, audio encoder, VAE, codec, and audio-language L1 tests.

Run the complete L1 suite with ``pytest tests/build_graph``.
"""

from __future__ import annotations

import pytest
from _test_configs import _base_config

from mobius._builder import build_from_module
from mobius._configs import ArchitectureConfig, MMSConfig
from mobius._registry import registry
from mobius.integrations.transformers._config_resolver import _default_task_for_model


class TestBuildVAEGraph:
    """Verify VAE (AutoencoderKL) graph construction."""

    def _vae_config(self):
        from mobius.integrations.diffusers._configs import VAEConfig

        return VAEConfig(
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

    def test_decoder_graph_builds(self):
        from mobius.models.vae import AutoencoderKLModel
        from mobius.tasks import VAETask

        config = self._vae_config()
        module = AutoencoderKLModel(config)
        task = VAETask()
        pkg = task.build(module, config)
        model = pkg["decoder"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "latent_sample" in input_names

        output_names = {out.name for out in model.graph.outputs}
        assert "sample" in output_names

    def test_package_has_encoder_and_decoder(self):
        from mobius.models.vae import AutoencoderKLModel
        from mobius.tasks import VAETask

        config = self._vae_config()
        module = AutoencoderKLModel(config)
        task = VAETask()
        pkg = task.build(module, config)

        assert "encoder" in pkg
        assert "decoder" in pkg

        # Encoder: sample → latent_dist
        enc_inputs = {inp.name for inp in pkg["encoder"].graph.inputs}
        enc_outputs = {out.name for out in pkg["encoder"].graph.outputs}
        assert "sample" in enc_inputs
        assert "latent_dist" in enc_outputs

        # Decoder: latent_sample → sample
        dec_inputs = {inp.name for inp in pkg["decoder"].graph.inputs}
        dec_outputs = {out.name for out in pkg["decoder"].graph.outputs}
        assert "latent_sample" in dec_inputs
        assert "sample" in dec_outputs

    def test_decoder_has_initializers(self):
        from mobius.models.vae import AutoencoderKLModel
        from mobius.tasks import VAETask

        config = self._vae_config()
        module = AutoencoderKLModel(config)
        task = VAETask()
        pkg = task.build(module, config)
        model = pkg["decoder"]

        init_names = list(model.graph.initializers)
        assert len(init_names) > 0
        has_conv = any("conv" in n for n in init_names)
        has_norm = any("norm" in n for n in init_names)
        assert has_conv, "Should have conv initializers"
        assert has_norm, "Should have norm initializers"


class TestBuildAudioGraph:
    """Verify audio encoder-only models build valid ONNX graphs."""

    def test_wav2vec2_graph_builds(self):
        from mobius.models.wav2vec2 import Wav2Vec2Model
        from mobius.tasks import AudioFeatureExtractionTask

        config = _base_config()
        module = Wav2Vec2Model(config)
        task = AudioFeatureExtractionTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "input_values" in input_names

        output_names = {out.name for out in model.graph.outputs}
        assert "last_hidden_state" in output_names

    def test_wav2vec2_has_initializers(self):
        from mobius.models.wav2vec2 import Wav2Vec2Model
        from mobius.tasks import AudioFeatureExtractionTask

        config = _base_config()
        module = Wav2Vec2Model(config)
        task = AudioFeatureExtractionTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        init_names = list(model.graph.initializers)
        has_feature_extractor = any("feature_extractor" in n for n in init_names)
        has_attention = any("attention" in n for n in init_names)
        assert has_feature_extractor, "Should have feature extractor initializers"
        assert has_attention, "Should have attention initializers"

    def test_audio_aliases_build(self):
        """Audio model aliases (hubert, wavlm, musicgen, etc.) all build."""
        from mobius.tasks import AudioFeatureExtractionTask

        config = _base_config()
        task = AudioFeatureExtractionTask()
        for model_type in (
            "data2vec-audio",
            "hubert",
            "wavlm",
            "mctct",
            "musicgen",
            "seamless_m4t",
            "seamless_m4t_v2",
            "sew",
            "sew-d",
            "speecht5",
            "unispeech",
            "unispeech-sat",
            "voxtral_encoder",
            "wav2vec2",
            "wav2vec2-bert",
            "wav2vec2-conformer",
        ):
            model_cls = registry.get(model_type)
            module = model_cls(config)
            pkg = task.build(module, config)
            model = pkg["model"]
            assert model.graph is not None, f"{model_type} graph should build"

            input_names = {inp.name for inp in model.graph.inputs}
            assert "input_values" in input_names, f"{model_type} missing input_values"

            output_names = {out.name for out in model.graph.outputs}
            assert "last_hidden_state" in output_names, (
                f"{model_type} missing last_hidden_state"
            )

            init_names = list(model.graph.initializers)
            assert len(init_names) > 0, f"{model_type} should have initializers"


class TestBuildMMSGraph:
    """Verify MMS (Massively Multilingual Speech) CTC model builds correctly.

    Tests both the base wav2vec2 encoder + CTC head, and with the per-language
    adapter (``add_adapter=True``) that enables language switching in MMS-1b-all.
    """

    def _mms_config(self, add_adapter: bool = False):
        """Tiny CTC config: hidden=64, 2 layers, 10 vocab labels."""
        return _base_config(
            config_cls=MMSConfig,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            vocab_size=10,
            add_adapter=add_adapter,
            output_hidden_size=64,
            adapter_kernel_size=3,
            adapter_stride=2,
            num_adapter_layers=2,
        )

    def test_package_builds(self):
        """Build MMS ONNX model and verify single-model package."""
        from mobius.models.wav2vec2_ctc import Wav2Vec2ForCTCModel
        from mobius.tasks import CTCAsrTask

        config = self._mms_config()
        module = Wav2Vec2ForCTCModel(config)
        pkg = CTCAsrTask().build(module, config)

        assert "model" in pkg

    def test_io_contract(self):
        """Verify input/output names of the CTC model."""
        from mobius.models.wav2vec2_ctc import Wav2Vec2ForCTCModel
        from mobius.tasks import CTCAsrTask

        config = self._mms_config()
        module = Wav2Vec2ForCTCModel(config)
        pkg = CTCAsrTask().build(module, config)
        model = pkg["model"]

        input_names = {inp.name for inp in model.graph.inputs}
        assert "input_values" in input_names
        assert "attention_mask" in input_names

        output_names = {out.name for out in model.graph.outputs}
        assert "logits" in output_names

    def test_has_ctc_head_initializers(self):
        """Verify the CTC lm_head parameters are present."""
        from mobius.models.wav2vec2_ctc import Wav2Vec2ForCTCModel
        from mobius.tasks import CTCAsrTask

        config = self._mms_config()
        module = Wav2Vec2ForCTCModel(config)
        pkg = CTCAsrTask().build(module, config)
        model = pkg["model"]

        init_names = list(model.graph.initializers)
        assert any("lm_head" in n for n in init_names), "Should have lm_head params"
        assert any("feature_extractor" in n for n in init_names), (
            "Should have feature_extractor params"
        )
        assert any("encoder" in n for n in init_names), "Should have encoder params"

    def test_adapter_variant_builds(self):
        """Build with adapter enabled (MMS-1b-all language adapter path)."""
        from mobius.models.wav2vec2_ctc import Wav2Vec2ForCTCModel
        from mobius.tasks import CTCAsrTask

        config = self._mms_config(add_adapter=True)
        module = Wav2Vec2ForCTCModel(config)
        pkg = CTCAsrTask().build(module, config)
        model = pkg["model"]

        init_names = list(model.graph.initializers)
        assert any("adapter" in n for n in init_names), (
            "Should have adapter params when add_adapter=True"
        )

    def test_registry_lookup(self):
        """Verify 'mms' is registered with ctc-asr task."""
        from mobius.models.wav2vec2_ctc import Wav2Vec2ForCTCModel

        assert registry.get("mms") is Wav2Vec2ForCTCModel
        assert _default_task_for_model("mms") == "ctc-asr"

    def test_ort_inference(self):
        """Build and run MMS through OnnxRuntime end-to-end."""
        import numpy as np

        from mobius._testing.ort_inference import OnnxModelSession
        from mobius.models.wav2vec2_ctc import Wav2Vec2ForCTCModel
        from mobius.rewrite_rules._testing_utils import fill_random_weights
        from mobius.tasks import CTCAsrTask

        config = self._mms_config()
        module = Wav2Vec2ForCTCModel(config)
        pkg = CTCAsrTask().build(module, config)
        fill_random_weights(pkg["model"])

        sess = OnnxModelSession(pkg["model"])
        num_samples = 8000  # 0.5 sec at 16 kHz
        waveform = np.random.randn(1, num_samples).astype(np.float32)
        attention_mask = np.ones((1, num_samples), dtype=np.int64)

        out = sess.run({"input_values": waveform, "attention_mask": attention_mask})
        sess.close()

        logits = out["logits"]
        assert logits.shape[0] == 1  # batch
        assert logits.shape[1] > 0  # num_frames (after CNN downsampling)
        assert logits.shape[2] == config.vocab_size  # CTC vocab

    # UNet2DConditionModel graph construction.

    def _unet_config(self):
        from mobius.integrations.diffusers._configs import UNet2DConfig

        return UNet2DConfig(
            in_channels=4,
            out_channels=4,
            block_out_channels=(32, 64),
            layers_per_block=1,
            norm_num_groups=32,
            cross_attention_dim=32,
            attention_head_dim=8,
        )

    def test_unet_graph_builds(self):
        from mobius.models.unet import UNet2DConditionModel
        from mobius.tasks import DenoisingTask

        config = self._unet_config()
        module = UNet2DConditionModel(config)
        task = DenoisingTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "sample" in input_names
        assert "timestep" in input_names
        assert "encoder_hidden_states" in input_names

        output_names = {out.name for out in model.graph.outputs}
        assert "noise_pred" in output_names

    def test_unet_has_initializers(self):
        from mobius.models.unet import UNet2DConditionModel
        from mobius.tasks import DenoisingTask

        config = self._unet_config()
        module = UNet2DConditionModel(config)
        task = DenoisingTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        init_names = list(model.graph.initializers)
        assert len(init_names) > 0
        has_time_emb = any("time_embedding" in n for n in init_names)
        has_conv = any("conv" in n for n in init_names)
        has_mid = any("mid_block" in n for n in init_names)
        assert has_time_emb, "Should have time embedding initializers"
        assert has_conv, "Should have conv initializers"
        assert has_mid, "Should have mid block initializers"


class TestBuildDiTGraph:
    """Verify DiT transformer denoiser graph construction."""

    def test_dit_graph_builds(self):
        from mobius.models.dit import DiTConfig, DiTTransformer2DModel
        from mobius.tasks import DenoisingTask

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
        task = DenoisingTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "sample" in input_names
        assert "timestep" in input_names
        assert "encoder_hidden_states" in input_names

        output_names = {out.name for out in model.graph.outputs}
        assert "noise_pred" in output_names


class TestBuildHunyuanDiTGraph:
    """Verify HunyuanDiT transformer denoiser graph construction."""

    def test_hunyuan_dit_graph_builds(self):
        from mobius.models.hunyuan_dit import (
            HunyuanDiT2DModel,
            HunyuanDiTConfig,
        )
        from mobius.tasks import DenoisingTask

        config = HunyuanDiTConfig(
            in_channels=4,
            patch_size=2,
            hidden_size=64,
            num_layers=4,
            num_attention_heads=4,
            cross_attention_dim=32,
            mlp_ratio=4.0,
            learn_sigma=True,
            sample_size=8,
        )
        module = HunyuanDiT2DModel(config)
        task = DenoisingTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "sample" in input_names
        assert "timestep" in input_names
        assert "encoder_hidden_states" in input_names

        output_names = {out.name for out in model.graph.outputs}
        assert "noise_pred" in output_names


class TestBuildControlNetGraph:
    """Verify ControlNet model graph construction."""

    def test_controlnet_graph_builds(self):
        from mobius.models.controlnet import ControlNetConfig, ControlNetModel
        from mobius.tasks import ControlNetTask

        config = ControlNetConfig(
            in_channels=4,
            conditioning_channels=3,
            block_out_channels=(32, 64),
            layers_per_block=1,
            norm_num_groups=32,
            cross_attention_dim=32,
            attention_head_dim=8,
        )
        module = ControlNetModel(config)
        task = ControlNetTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "sample" in input_names
        assert "timestep" in input_names
        assert "encoder_hidden_states" in input_names
        assert "controlnet_cond" in input_names

        output_names = {out.name for out in model.graph.outputs}
        assert "mid_block_res" in output_names
        down_res = [n for n in output_names if n.startswith("down_block_res_")]
        assert len(down_res) > 0, "Should have down block residual outputs"


class TestBuildVideoVAEGraph:
    """Verify Video VAE (3D autoencoder) graph construction."""

    def test_video_decoder_graph_builds(self):
        from mobius.models.video_vae import VideoAutoencoderModel, VideoVAEConfig
        from mobius.tasks import VAETask

        config = VideoVAEConfig(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            block_out_channels=(32, 64),
            layers_per_block=1,
            norm_num_groups=32,
        )
        module = VideoAutoencoderModel(config)
        task = VAETask()
        pkg = task.build(module, config)
        model = pkg["decoder"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "latent_sample" in input_names

        output_names = {out.name for out in model.graph.outputs}
        assert "sample" in output_names

        init_names = list(model.graph.initializers)
        assert len(init_names) > 0
        has_conv = any("conv" in n for n in init_names)
        assert has_conv, "Should have 3D conv initializers"


class TestBuildSD3Graph:
    """Verify SD3 (MMDiT) transformer denoiser graph construction."""

    def test_sd3_graph_builds(self):
        from mobius.models.flux_sd3 import SD3Config, SD3Transformer2DModel
        from mobius.tasks import DenoisingTask

        config = SD3Config(
            in_channels=4,
            out_channels=4,
            patch_size=2,
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            joint_attention_dim=32,
            caption_projection_dim=32,
            cross_attention_dim=32,
            sample_size=8,
        )
        module = SD3Transformer2DModel(config)
        task = DenoisingTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "sample" in input_names
        assert "timestep" in input_names
        assert "encoder_hidden_states" in input_names
        assert "noise_pred" in {out.name for out in model.graph.outputs}


class TestBuildFluxGraph:
    """Verify Flux transformer denoiser graph construction."""

    def test_flux_graph_builds(self):
        from mobius.models.flux_sd3 import FluxConfig, FluxTransformer2DModel
        from mobius.tasks import DenoisingTask

        config = FluxConfig(
            in_channels=4,
            out_channels=4,
            patch_size=2,
            hidden_size=64,
            num_layers=1,
            num_single_layers=2,
            num_attention_heads=4,
            joint_attention_dim=32,
            cross_attention_dim=32,
            sample_size=8,
        )
        module = FluxTransformer2DModel(config)
        task = DenoisingTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "sample" in input_names
        assert "timestep" in input_names
        assert "encoder_hidden_states" in input_names
        assert "noise_pred" in {out.name for out in model.graph.outputs}


class TestBuildCogVideoXGraph:
    """Verify CogVideoX 3D video transformer graph construction."""

    def test_cogvideox_graph_builds(self):
        from mobius.integrations.diffusers._configs import CogVideoXConfig
        from mobius.models.cogvideox import CogVideoXTransformer3DModel
        from mobius.tasks import VideoDenoisingTask

        config = CogVideoXConfig(
            num_attention_heads=2,
            attention_head_dim=32,
            in_channels=4,
            out_channels=4,
            time_embed_dim=64,
            text_embed_dim=32,
            num_layers=2,
            patch_size=2,
            sample_height=8,
            sample_width=8,
            sample_frames=9,
            temporal_compression_ratio=4,
            max_text_seq_length=8,
            spatial_interpolation_scale=1.0,
            temporal_interpolation_scale=1.0,
            norm_eps=1e-5,
            cross_attention_dim=32,
        )
        module = CogVideoXTransformer3DModel(config)
        task = VideoDenoisingTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "sample" in input_names
        assert "timestep" in input_names
        assert "encoder_hidden_states" in input_names
        assert "noise_pred" in {out.name for out in model.graph.outputs}

        # Verify 5D sample shape
        sample_input = next(inp for inp in model.graph.inputs if inp.name == "sample")
        assert len(sample_input.shape) == 5


class TestBuildCogVideoXVAEGraph:
    """Verify the CogVideoX causal 3D VAE decoder graph construction."""

    @staticmethod
    def _config():
        from mobius.models.cogvideox_vae import CogVideoXVAEConfig

        return CogVideoXVAEConfig(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            block_out_channels=(8, 8, 8, 8),
            layers_per_block=1,
            norm_num_groups=2,
            temporal_compression_ratio=4,
            scaling_factor=1.15258426,
        )

    def test_video_vae_graph_builds_with_paired_conv_caches(self):
        from mobius.models.cogvideox_vae import AutoencoderKLCogVideoXModel
        from mobius.tasks import VideoVAETask
        from mobius.tasks._video_vae import (
            CONV_CACHE_INPUT_PREFIX,
            CONV_CACHE_OUTPUT_PREFIX,
            CONV_CACHE_SCALE_METADATA,
        )

        config = self._config()
        model = VideoVAETask().build(AutoencoderKLCogVideoXModel(config), config)["decoder"]

        assert model.graph is not None
        latent = next(v for v in model.graph.inputs if v.name == "latent_sample")
        # [B, C, T, H, W]: the temporal axis is explicit, and the frame count is
        # a free dimension rather than a baked clip length.
        assert len(latent.shape) == 5
        assert str(latent.shape[2]) == "latent_frames"

        sample = next(v for v in model.graph.outputs if v.name == "sample")
        assert len(sample.shape) == 5
        assert int(sample.shape[1]) == config.out_channels

        cache_inputs = {
            v.name[len(CONV_CACHE_INPUT_PREFIX) :]
            for v in model.graph.inputs
            if v.name.startswith(CONV_CACHE_INPUT_PREFIX)
        }
        cache_outputs = {
            v.name[len(CONV_CACHE_OUTPUT_PREFIX) :]
            for v in model.graph.outputs
            if v.name.startswith(CONV_CACHE_OUTPUT_PREFIX)
        }
        # Every cached convolution has to be readable and writable, or a clip
        # decoded in chunks would silently lose the frames before each chunk.
        assert cache_inputs
        assert cache_inputs == cache_outputs
        for name in cache_inputs:
            key = f"{CONV_CACHE_SCALE_METADATA}{CONV_CACHE_INPUT_PREFIX}{name}"
            assert key in model.metadata_props

    def test_video_vae_cache_spec_matches_upsampled_resolutions(self):
        from mobius.models.cogvideox_vae import AutoencoderKLCogVideoXModel

        config = self._config()
        module = AutoencoderKLCogVideoXModel(config)
        scales = {entry.name: entry.spatial_scale for entry in module.conv_cache_spec()}
        assert scales["conv_in"] == 1
        # Three upsampling stages for four blocks, so the last cached
        # convolutions live at the full frame resolution.
        assert scales["conv_out"] == 2 ** (len(config.block_out_channels) - 1)


class TestBuildAdapterGraph:
    """Verify T2I-Adapter and IP-Adapter graph construction."""

    def test_t2i_adapter_graph_builds(self):
        from mobius.models.adapters import T2IAdapterConfig, T2IAdapterModel
        from mobius.tasks import AdapterTask

        config = T2IAdapterConfig(in_channels=3, channels=(32, 64), num_res_blocks=1)
        module = T2IAdapterModel(config)
        task = AdapterTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "condition" in input_names
        output_names = {out.name for out in model.graph.outputs}
        assert any(n.startswith("feature_") for n in output_names)

    def test_ip_adapter_graph_builds(self):
        from mobius.models.adapters import IPAdapterConfig, IPAdapterModel
        from mobius.tasks import AdapterTask

        config = IPAdapterConfig(image_embed_dim=32, cross_attention_dim=64, num_tokens=4)
        module = IPAdapterModel(config)
        task = AdapterTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "image_embeds" in input_names
        output_names = {out.name for out in model.graph.outputs}
        assert "adapter_output" in output_names


class TestBuildQwenImageGraph:
    """Verify QwenImage transformer denoiser graph construction."""

    def test_qwen_image_transformer_graph_builds(self):
        from mobius.integrations.diffusers._configs import QwenImageConfig
        from mobius.models.qwen_image import QwenImageTransformer2DModel
        from mobius.tasks import QwenImageDenoisingTask

        config = QwenImageConfig(
            in_channels=4,
            out_channels=4,
            patch_size=2,
            num_layers=2,
            attention_head_dim=32,
            num_attention_heads=2,
            joint_attention_dim=64,
            cross_attention_dim=64,
        )
        module = QwenImageTransformer2DModel(config)
        task = QwenImageDenoisingTask()
        pkg = task.build(module, config)
        model = pkg["model"]

        assert model.graph is not None
        input_names = {inp.name for inp in model.graph.inputs}
        assert "sample" in input_names
        assert "timestep" in input_names
        assert "encoder_hidden_states" in input_names
        assert "encoder_hidden_states_mask" in input_names
        assert "image_rotary_cos" in input_names
        assert "target_sequence_length" in input_names
        assert "noise_pred" in {out.name for out in model.graph.outputs}

    def test_qwen_image_vae_encoder_decoder_graphs_build(self):
        from mobius.integrations.diffusers._configs import QwenImageVAEConfig
        from mobius.models.qwen_image_vae import AutoencoderKLQwenImageModel
        from mobius.tasks import QwenImageVAETask

        config = QwenImageVAEConfig(
            base_dim=8,
            z_dim=4,
            dim_mult=(1, 2),
            num_res_blocks=1,
            temperal_downsample=(False,),
        )
        module = AutoencoderKLQwenImageModel(config)
        task = QwenImageVAETask()
        pkg = task.build(module, config)

        enc = pkg["encoder"]
        assert enc.graph is not None
        assert "sample" in {inp.name for inp in enc.graph.inputs}
        assert "latent_dist" in {out.name for out in enc.graph.outputs}

        dec = pkg["decoder"]
        assert dec.graph is not None
        assert "latent_sample" in {inp.name for inp in dec.graph.inputs}
        assert "sample" in {out.name for out in dec.graph.outputs}


class TestBuildMimiCodec:
    """Verify the Mimi codec (nvidia/personaplex-7b-v1) graph construction."""

    def test_package_builds_2_models(self):
        """Build the Mimi codec and verify a 2-model (encoder+decoder) package."""
        from mobius.models.mimi import MimiModel, _mimi_default_config
        from mobius.tasks import CodecTask

        config = _mimi_default_config()
        module = MimiModel(config)
        pkg = build_from_module(module, config, task=CodecTask())

        assert "encoder" in pkg
        assert "decoder" in pkg

    def test_encoder_io(self):
        """Verify the Mimi encoder I/O contract: waveform -> codes."""
        from mobius.models.mimi import MimiModel, _mimi_default_config
        from mobius.tasks import CodecTask

        config = _mimi_default_config()
        pkg = build_from_module(MimiModel(config), config, task=CodecTask())
        encoder = pkg["encoder"]

        assert "waveform" in {inp.name for inp in encoder.graph.inputs}
        assert "codes" in {out.name for out in encoder.graph.outputs}

    def test_decoder_io(self):
        """Verify the Mimi decoder I/O contract: codes -> waveform."""
        from mobius.models.mimi import MimiModel, _mimi_default_config
        from mobius.tasks import CodecTask

        config = _mimi_default_config()
        pkg = build_from_module(MimiModel(config), config, task=CodecTask())
        decoder = pkg["decoder"]

        assert "codes" in {inp.name for inp in decoder.graph.inputs}
        assert "waveform" in {out.name for out in decoder.graph.outputs}


class TestBuildMoshiLM:
    """Verify the Moshi LM (nvidia/personaplex-7b-v1) graph construction."""

    @staticmethod
    def _temporal_tiny_config():
        import dataclasses

        from mobius.models.moshi import _moshi_temporal_config

        cfg = _moshi_temporal_config()
        return dataclasses.replace(
            cfg,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            intermediate_size=128,
            max_position_embeddings=256,
        )

    def test_temporal_io(self):
        """Temporal model I/O: 17-channel frame -> hidden + text_logits + KV."""
        from mobius.models.moshi import MoshiTemporalModel
        from mobius.tasks import MoshiTemporalTask

        config = self._temporal_tiny_config()
        pkg = build_from_module(MoshiTemporalModel(config), config, task=MoshiTemporalTask())
        model = pkg["model"]
        inputs = {inp.name for inp in model.graph.inputs}
        outputs = {out.name for out in model.graph.outputs}
        assert "input_frame" in inputs
        assert "position_ids" in inputs
        assert "hidden" in outputs
        assert "text_logits" in outputs
        assert "present.0.key" in outputs

    def test_temporal_gqa_emits_sliding_window(self):
        """Temporal GQA nodes carry Moshi's sliding window as local_window_size.

        On the GQA (fp16/cuda) path, the temporal transformer's uniform sliding
        window (Moshi ``context``) must reach every GroupQueryAttention node as
        ``local_window_size``.  PersonaPlex deploys this fp16 path, so without it
        long streams would silently run full causal attention.
        """
        import dataclasses

        import onnx_ir as ir

        from mobius.models.moshi import MoshiTemporalModel, _moshi_temporal_config
        from mobius.tasks import MoshiTemporalTask

        full_window = _moshi_temporal_config().sliding_window
        assert full_window and full_window > 0, "Moshi temporal must be sliding"

        config = dataclasses.replace(self._temporal_tiny_config(), dtype=ir.DataType.FLOAT16)
        pkg = build_from_module(
            MoshiTemporalModel(config),
            config,
            task=MoshiTemporalTask(),
            execution_provider="cuda",
        )
        gqa_nodes = [n for n in pkg["model"].graph if n.op_type == "GroupQueryAttention"]
        assert len(gqa_nodes) == config.num_hidden_layers
        for node in gqa_nodes:
            assert node.attributes["local_window_size"].value == full_window

    @pytest.mark.parametrize("dep_q", [8, 16])
    def test_depformer_io(self, dep_q):
        """Depformer model I/O: hidden + prev_token + substep_index -> logits."""
        from mobius.models.moshi import MoshiDepformerModel, _moshi_depformer_config
        from mobius.tasks import MoshiDepformerTask

        config = _moshi_depformer_config(dep_q=dep_q)
        pkg = build_from_module(MoshiDepformerModel(config), config, task=MoshiDepformerTask())
        model = pkg["model"]
        inputs = {inp.name for inp in model.graph.inputs}
        outputs = {out.name for out in model.graph.outputs}
        assert "hidden" in inputs
        assert "prev_token" in inputs
        assert "substep_index" in inputs
        assert "logits" in outputs
        assert "present.0.key" in outputs
        assert model.graph.initializers["depformer_in.weight"].shape[0] == dep_q

    @pytest.mark.parametrize("dep_q", [0, 1, 2, 7, 9, 17])
    def test_depformer_rejects_unsupported_width(self, dep_q):
        from mobius.models.moshi import _moshi_depformer_config

        with pytest.raises(ValueError, match=r"dep_q must be 8 .* or 16"):
            _moshi_depformer_config(dep_q)


class TestBuildCodecGraph:
    """Verify codec tokenizer (Qwen3-TTS-Tokenizer-12Hz) graph construction."""

    @staticmethod
    def _codec_config():
        from mobius._configs import (
            CodecDecoderConfig,
            CodecEncoderConfig,
        )

        return ArchitectureConfig(
            # Use decoder's transformer dims as top-level (from exporter)
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=8,
            intermediate_size=64,
            vocab_size=256,
            max_position_embeddings=128,
            rms_norm_eps=1e-5,
            codec_decoder=CodecDecoderConfig(
                codebook_dim=32,
                codebook_size=64,
                latent_dim=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=4,
                head_dim=8,
                rms_norm_eps=1e-5,
                rope_theta=10000.0,
                max_position_embeddings=128,
                decoder_dim=96,
                num_quantizers=4,
                upsample_rates=[2, 2, 2, 2],
                upsampling_ratios=[2, 2],
            ),
            codec_encoder=CodecEncoderConfig(
                codebook_dim=16,
                codebook_size=64,
                hidden_size=32,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=4,
                head_dim=8,
                rope_theta=10000.0,
                max_position_embeddings=128,
                num_quantizers=8,
                num_semantic_quantizers=1,
                # Narrow conv stack (1->4->8->16->32->64->32) but the same
                # depth as the real checkpoint, so `layers.*` numbering and
                # weight names match what the real model relies on.
                num_filters=4,
            ),
        )

    def test_package_builds_2_models(self):
        """Build codec tokenizer and verify 2-model package."""
        from mobius.models.qwen3_tts_tokenizer import (
            Qwen3TTSTokenizerV2Model,
        )
        from mobius.tasks import CodecTask

        config = self._codec_config()
        module = Qwen3TTSTokenizerV2Model(config)
        pkg = build_from_module(module, config, task=CodecTask())

        assert "decoder" in pkg
        assert "encoder" in pkg

    def test_decoder_io(self):
        """Verify decoder: codes → waveform."""
        from mobius.models.qwen3_tts_tokenizer import (
            Qwen3TTSTokenizerV2Model,
        )
        from mobius.tasks import CodecTask

        config = self._codec_config()
        module = Qwen3TTSTokenizerV2Model(config)
        pkg = build_from_module(module, config, task=CodecTask())
        decoder = pkg["decoder"]

        input_names = {inp.name for inp in decoder.graph.inputs}
        assert "codes" in input_names

        output_names = {out.name for out in decoder.graph.outputs}
        assert "waveform" in output_names

    def test_encoder_io(self):
        """Verify encoder: waveform → codes."""
        from mobius.models.qwen3_tts_tokenizer import (
            Qwen3TTSTokenizerV2Model,
        )
        from mobius.tasks import CodecTask

        config = self._codec_config()
        module = Qwen3TTSTokenizerV2Model(config)
        pkg = build_from_module(module, config, task=CodecTask())
        encoder = pkg["encoder"]

        input_names = {inp.name for inp in encoder.graph.inputs}
        assert "waveform" in input_names

        output_names = {out.name for out in encoder.graph.outputs}
        assert "codes" in output_names

    @staticmethod
    def _conv_encoder_weights(encoder_config):
        """Return {param name: shape} for the conv stack of an encoder config."""
        from mobius.models.qwen3_tts_tokenizer import Qwen3TTSCodecEncoderModel

        config = ArchitectureConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=8,
            intermediate_size=64,
            vocab_size=256,
            max_position_embeddings=128,
            codec_encoder=encoder_config,
        )
        module = Qwen3TTSCodecEncoderModel(config)
        return {
            name: tuple(param.shape)
            for name, param in module.named_parameters()
            if name.startswith("encoder.layers") and name.endswith("conv.weight")
        }

    def test_default_config_conv_stack_matches_checkpoint(self):
        """Default config must reproduce the real checkpoint's conv stack.

        Both names and shapes are asserted: the ``layers.*`` indices are
        derived from ``upsampling_ratios``/``num_residual_layers``, so a
        drift in indexing would only surface at weight-load time.
        """
        from mobius._configs import CodecEncoderConfig

        weights = self._conv_encoder_weights(CodecEncoderConfig())

        # Matches Qwen/Qwen3-TTS-Tokenizer-12Hz encoder.encoder.layers.*
        assert weights == {
            "encoder.layers.0.conv.weight": (64, 1, 7),
            "encoder.layers.1.block.1.conv.weight": (32, 64, 3),
            "encoder.layers.1.block.3.conv.weight": (64, 32, 1),
            "encoder.layers.3.conv.weight": (128, 64, 8),
            "encoder.layers.4.block.1.conv.weight": (64, 128, 3),
            "encoder.layers.4.block.3.conv.weight": (128, 64, 1),
            "encoder.layers.6.conv.weight": (256, 128, 10),
            "encoder.layers.7.block.1.conv.weight": (128, 256, 3),
            "encoder.layers.7.block.3.conv.weight": (256, 128, 1),
            "encoder.layers.9.conv.weight": (512, 256, 12),
            "encoder.layers.10.block.1.conv.weight": (256, 512, 3),
            "encoder.layers.10.block.3.conv.weight": (512, 256, 1),
            "encoder.layers.12.conv.weight": (1024, 512, 16),
            "encoder.layers.14.conv.weight": (512, 1024, 3),
        }

    def test_tiny_config_keeps_checkpoint_layer_names(self):
        """The tiny test config must keep the checkpoint's layer numbering."""
        from mobius._configs import CodecEncoderConfig

        tiny = self._conv_encoder_weights(self._codec_config().codec_encoder)
        default = self._conv_encoder_weights(CodecEncoderConfig())

        assert set(tiny) == set(default)
        # Only the widths shrink: 1 -> 4 -> 8 -> 16 -> 32 -> 64 -> 32.
        assert tiny["encoder.layers.0.conv.weight"] == (4, 1, 7)
        assert tiny["encoder.layers.12.conv.weight"] == (64, 32, 16)
        assert tiny["encoder.layers.14.conv.weight"] == (32, 64, 3)

    def test_conv_stack_indices_follow_config(self):
        """Layer indices are derived from ratios and residual-layer count."""
        from mobius._configs import CodecEncoderConfig

        # 2 ratios -> final conv lands at layers.8
        two_ratios = self._conv_encoder_weights(
            CodecEncoderConfig(hidden_size=8, num_filters=2, upsampling_ratios=[4, 2])
        )
        assert "encoder.layers.8.conv.weight" in two_ratios
        assert two_ratios["encoder.layers.8.conv.weight"] == (8, 8, 3)

        # 4 ratios with 2 residual layers each -> final conv at layers.18
        deep = self._conv_encoder_weights(
            CodecEncoderConfig(hidden_size=8, num_filters=2, num_residual_layers=2)
        )
        assert "encoder.layers.18.conv.weight" in deep

    def test_hidden_size_drives_conv_output_width(self):
        """The final conv width follows ``hidden_size`` (no hardcoded 512).

        Regression guard: a hardcoded final width silently produced a
        malformed graph (LayerNormalization over a mismatched width).
        """
        from mobius._configs import CodecEncoderConfig

        weights = self._conv_encoder_weights(CodecEncoderConfig(hidden_size=32, num_filters=4))
        assert weights["encoder.layers.14.conv.weight"][0] == 32

    def test_default_config_rvq_projections_match_checkpoint(self):
        """Encoder RVQ projections must match the real checkpoint's shapes.

        The RVQ consumes ``hidden_size``-wide features and projects to
        ``codebook_dim``, mirroring HF ``MimiResidualVectorQuantizer``.
        Deriving these from ``codebook_dim`` alone produced projections
        that could not load the checkpoint's weights at all.
        """
        from mobius._configs import CodecEncoderConfig
        from mobius.models.qwen3_tts_tokenizer import Qwen3TTSCodecEncoderModel

        config = ArchitectureConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=8,
            intermediate_size=64,
            vocab_size=256,
            max_position_embeddings=128,
            codec_encoder=CodecEncoderConfig(),
        )
        module = Qwen3TTSCodecEncoderModel(config)
        shapes = {
            name: tuple(param.shape)
            for name, param in module.named_parameters()
            if "_proj.weight" in name and "quantizer" in name
        }

        # Matches Qwen/Qwen3-TTS-Tokenizer-12Hz encoder.quantizer.* weights.
        for prefix in (
            "quantizer.semantic_residual_vector_quantizer",
            "quantizer.acoustic_residual_vector_quantizer",
        ):
            assert shapes[f"{prefix}.input_proj.weight"] == (256, 512, 1)
            assert shapes[f"{prefix}.output_proj.weight"] == (512, 256, 1)

        # Codebooks are codebook_dim-wide, matching codebook.embed_sum.
        codebooks = {
            tuple(param.shape)
            for name, param in module.named_parameters()
            if name.endswith("codebook.embedding")
        }
        assert codebooks == {(2048, 256)}

    def test_encoder_input_declares_configured_audio_channels(self):
        """The graph input channel count must match the first conv.

        ``audio_channels`` sizes ``encoder.layers.0.conv``, so a task that
        always declared a mono input would feed a 1-channel tensor into a
        conv expecting more.
        """
        import dataclasses

        from mobius.models.qwen3_tts_tokenizer import Qwen3TTSTokenizerV2Model
        from mobius.tasks import CodecTask

        config = self._codec_config()
        config.codec_encoder = dataclasses.replace(config.codec_encoder, audio_channels=2)
        module = Qwen3TTSTokenizerV2Model(config)
        pkg = build_from_module(module, config, task=CodecTask())

        waveform = next(inp for inp in pkg["encoder"].graph.inputs if inp.name == "waveform")
        assert waveform.shape[1] == 2

    def test_registry_lookup(self):
        """Verify qwen3_tts_tokenizer_12hz is registered with codec task."""
        model_cls = registry.get("qwen3_tts_tokenizer_12hz")
        from mobius.models.qwen3_tts_tokenizer import (
            Qwen3TTSTokenizerV2Model,
        )

        assert model_cls is Qwen3TTSTokenizerV2Model
        assert _default_task_for_model("qwen3_tts_tokenizer_12hz") == "codec"
