# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the Wan 3D causal video VAE (``diffusers.AutoencoderKLWan``).

Layers:

* Config parsing / validation against the real ``nvidia/Cosmos3-Nano/vae``
  ``config.json`` (embedded verbatim below — no network access).
* L1 graph construction with a tiny config: I/O contract, node construction and
  HuggingFace weight-name alignment for both the Wan 2.2 residual architecture
  and the Wan 2.1 flat architecture.
* Numerical parity against ``diffusers`` PyTorch for the causal convolution, the
  residual block, both temporal resampling modes (whole-sequence vs upstream's
  chunked ``feat_cache`` loop) and the full encoder/decoder.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import onnx_ir as ir
import pytest

from mobius._configs._wan_vae import WanVAEConfig
from mobius.models.wan_vae import AutoencoderKLWanModel
from mobius.tasks._base import _make_graph, _make_model
from mobius.tasks._wan_vae import WanVAETask

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

#: Verbatim ``vae/config.json`` of ``nvidia/Cosmos3-Nano`` (== Wan2.2-TI2V-5B).
COSMOS3_VAE_CONFIG: dict = {
    "_class_name": "AutoencoderKLWan",
    "_diffusers_version": "0.37.1",
    "_name_or_path": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "attn_scales": [],
    "base_dim": 160,
    "clip_output": False,
    "decoder_base_dim": 256,
    "dim_mult": [1, 2, 4, 4],
    "dropout": 0.0,
    "in_channels": 12,
    "is_residual": True,
    "latents_mean": [
        -0.2289,
        -0.0052,
        -0.1323,
        -0.2339,
        -0.2799,
        0.0174,
        0.1838,
        0.1557,
        -0.1382,
        0.0542,
        0.2813,
        0.0891,
        0.157,
        -0.0098,
        0.0375,
        -0.1825,
        -0.2246,
        -0.1207,
        -0.0698,
        0.5109,
        0.2665,
        -0.2108,
        -0.2158,
        0.2502,
        -0.2055,
        -0.0322,
        0.1109,
        0.1567,
        -0.0729,
        0.0899,
        -0.2799,
        -0.123,
        -0.0313,
        -0.1649,
        0.0117,
        0.0723,
        -0.2839,
        -0.2083,
        -0.052,
        0.3748,
        0.0152,
        0.1957,
        0.1433,
        -0.2944,
        0.3573,
        -0.0548,
        -0.1681,
        -0.0667,
    ],
    "latents_std": [
        0.4765,
        1.0364,
        0.4514,
        1.1677,
        0.5313,
        0.499,
        0.4818,
        0.5013,
        0.8158,
        1.0344,
        0.5894,
        1.0901,
        0.6885,
        0.6165,
        0.8454,
        0.4978,
        0.5759,
        0.3523,
        0.7135,
        0.6804,
        0.5833,
        1.4146,
        0.8986,
        0.5659,
        0.7069,
        0.5338,
        0.4889,
        0.4917,
        0.4069,
        0.4999,
        0.6866,
        0.4093,
        0.5709,
        0.6065,
        0.6415,
        0.4944,
        0.5726,
        1.2042,
        0.5458,
        1.6887,
        0.3971,
        1.06,
        0.3943,
        0.5537,
        0.5444,
        0.4089,
        0.7468,
        0.7744,
    ],
    "num_res_blocks": 2,
    "out_channels": 12,
    "patch_size": 2,
    "scale_factor_spatial": 16,
    "scale_factor_temporal": 4,
    "temperal_downsample": [False, True, True],
    "z_dim": 48,
}

#: Tiny Wan 2.2 (residual) config: 2 resample stages, 1 of them temporal.
TINY_WAN22: dict = {
    "_class_name": "AutoencoderKLWan",
    "attn_scales": [],
    "base_dim": 8,
    "clip_output": False,
    "decoder_base_dim": 8,
    "dim_mult": [1, 2, 2],
    "dropout": 0.0,
    "in_channels": 12,
    "is_residual": True,
    "latents_mean": [0.1, -0.2, 0.3, -0.4],
    "latents_std": [1.1, 0.9, 1.2, 0.8],
    "num_res_blocks": 1,
    "out_channels": 12,
    "patch_size": 2,
    "scale_factor_spatial": 8,
    "scale_factor_temporal": 2,
    "temperal_downsample": [False, True],
    "z_dim": 4,
}

#: Tiny Wan 2.1 (flat, non-residual, un-patchified) config.
TINY_WAN21: dict = {
    "_class_name": "AutoencoderKLWan",
    "attn_scales": [],
    "base_dim": 8,
    "dim_mult": [1, 2, 4],
    "dropout": 0.0,
    "in_channels": 3,
    "is_residual": False,
    "latents_mean": [0.1, -0.2, 0.3, -0.4],
    "latents_std": [1.1, 0.9, 1.2, 0.8],
    "num_res_blocks": 1,
    "out_channels": 3,
    "scale_factor_spatial": 4,
    "scale_factor_temporal": 2,
    "temperal_downsample": [False, True],
    "z_dim": 4,
}


def _build(config_dict: dict):
    """Parse a config, instantiate the module and build both graphs."""
    config = WanVAEConfig.from_diffusers(config_dict)
    module = AutoencoderKLWanModel(config)
    return config, module, WanVAETask().build(module, config)


def _op_types(model: ir.Model) -> set[str]:
    return {node.op_type for node in model.graph}


def _run(model: ir.Model, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Serialise *model* and run it through onnxruntime."""
    import onnxruntime as ort

    # Windows keeps the ORT model file mapped; ignore cleanup errors.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        path = Path(temp_dir) / "model.onnx"
        ir.save(model, path)
        session = ort.InferenceSession(path)
        return session.run(None, feeds)


# ---------------------------------------------------------------------------
# Config parsing and validation
# ---------------------------------------------------------------------------


class TestConfig:
    """Parsing and validation of the public ``AutoencoderKLWan`` config."""

    def test_parses_real_cosmos3_config(self):
        config = WanVAEConfig.from_diffusers(COSMOS3_VAE_CONFIG)

        assert config.z_dim == 48
        assert config.in_channels == 12
        assert config.out_channels == 12
        assert config.patch_size == 2
        assert config.base_dim == 160
        assert config.decoder_base_dim == 256
        assert config.dim_mult == (1, 2, 4, 4)
        assert config.num_res_blocks == 2
        assert config.attn_scales == ()
        assert config.dropout == pytest.approx(0.0)
        assert config.is_residual is True
        assert config.temporal_downsample == (False, True, True)
        assert config.scale_factor_spatial == 16
        assert config.scale_factor_temporal == 4
        assert len(config.latents_mean) == 48
        assert len(config.latents_std) == 48
        assert config.latents_mean[0] == pytest.approx(-0.2289)
        assert config.latents_std[-1] == pytest.approx(0.7744)
        assert config.dtype == ir.DataType.FLOAT

    def test_derived_shapes_for_cosmos3(self):
        config = WanVAEConfig.from_diffusers(COSMOS3_VAE_CONFIG)

        # in_channels counts patchified channels: 3 RGB * 2 * 2.
        assert config.video_channels == 3
        assert config.decoded_video_channels == 3
        assert config.encoder_dims == (160, 160, 320, 640, 640)
        assert config.decoder_dims == (1024, 1024, 1024, 512, 256)
        assert config.temporal_upsample == (True, True, False)

    def test_misspelled_temperal_downsample_is_accepted_and_aliased(self):
        config = WanVAEConfig.from_diffusers(COSMOS3_VAE_CONFIG)

        # Upstream's misspelling is the parsed key; the dataclass exposes the
        # corrected name and keeps the misspelled one as a read-only alias.
        assert config.temporal_downsample == (False, True, True)
        assert config.temperal_downsample == config.temporal_downsample

    def test_corrected_spelling_is_also_accepted(self):
        raw = dict(COSMOS3_VAE_CONFIG)
        raw.pop("temperal_downsample")
        raw["temporal_downsample"] = [False, True, True]

        assert WanVAEConfig.from_diffusers(raw).temporal_downsample == (False, True, True)

    def test_clip_output_is_parsed_but_never_honoured(self):
        # ``clip_output`` is not a parameter of AutoencoderKLWan.__init__, so
        # diffusers drops it and always clamps. The graph must clamp too.
        config = WanVAEConfig.from_diffusers(COSMOS3_VAE_CONFIG)
        assert config.clip_output is False

        _, _, package = _build(TINY_WAN22)
        assert "Clip" in _op_types(package["decoder"])

    def test_decoder_base_dim_defaults_to_base_dim(self):
        raw = dict(TINY_WAN21)
        assert "decoder_base_dim" not in raw

        assert WanVAEConfig.from_diffusers(raw).decoder_base_dim == raw["base_dim"]

    @pytest.mark.parametrize(
        ("dtype", "expected"),
        [
            (None, ir.DataType.FLOAT),
            ("auto", ir.DataType.FLOAT),
            ("float32", ir.DataType.FLOAT),
            ("float16", ir.DataType.FLOAT16),
            ("torch.bfloat16", ir.DataType.BFLOAT16),
        ],
    )
    def test_dtype_resolution(self, dtype, expected):
        raw = dict(TINY_WAN22, dtype=dtype)

        assert WanVAEConfig.from_diffusers(raw).dtype == expected

    def test_rejects_unknown_dtype(self):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            WanVAEConfig.from_diffusers(dict(TINY_WAN22, dtype="int8"))

    def test_rejects_foreign_class_name(self):
        raw = dict(TINY_WAN22, _class_name="AutoencoderKLQwenImage")

        with pytest.raises(ValueError, match="AutoencoderKLWan"):
            WanVAEConfig.from_diffusers(raw)

    @pytest.mark.parametrize(
        ("override", "message"),
        [
            ({"temperal_downsample": [False, True, True]}, "len\\(dim_mult\\) - 1"),
            ({"latents_mean": [0.0, 0.0]}, "latents_mean must have z_dim"),
            ({"latents_std": [1.0, 1.0]}, "latents_std must have z_dim"),
            ({"latents_std": [1.0, 0.0, 1.0, 1.0]}, "non-zero"),
            ({"scale_factor_spatial": 4}, "scale_factor_spatial"),
            ({"scale_factor_temporal": 4}, "scale_factor_temporal"),
            ({"in_channels": 10}, "divisible by patch_size"),
            ({"z_dim": 0}, "z_dim must be positive"),
            ({"num_res_blocks": 0}, "num_res_blocks must be positive"),
            ({"base_dim": 0}, "base_dim must be positive"),
        ],
    )
    def test_validation_rejects_inconsistent_fields(self, override, message):
        with pytest.raises(ValueError, match=message):
            WanVAEConfig.from_diffusers(dict(TINY_WAN22, **override))

    def test_validation_rejects_bad_residual_shortcut_widths(self):
        # AvgDown3D groups in_dim * factor channels into out_dim groups; a
        # non-divisible pair cannot be expressed as a grouped mean.
        raw = dict(TINY_WAN22, dim_mult=[1, 3, 3], scale_factor_spatial=8)

        with pytest.raises(ValueError, match="AvgDown3D"):
            WanVAEConfig.from_diffusers(raw)


# ---------------------------------------------------------------------------
# L1 graph construction
# ---------------------------------------------------------------------------


class TestGraphConstruction:
    """Graph I/O contract, node construction and weight-name alignment."""

    def test_encoder_graph_io(self):
        config, _, package = _build(TINY_WAN22)
        graph = package["encoder"].graph

        assert [i.name for i in graph.inputs] == ["sample"]
        sample = graph.inputs[0]
        assert sample.dtype == ir.DataType.FLOAT
        # Explicit 5D video: patchification happens inside the graph, so the
        # input carries pixel-space (3-channel) video, not the 12 patch channels.
        assert [str(d) for d in sample.shape] == [
            "batch",
            str(config.video_channels),
            "frames",
            "height",
            "width",
        ]

        assert [o.name for o in graph.outputs] == [
            "latent_mean",
            "latent_logvar",
            "latent",
        ]
        for output in graph.outputs:
            assert output.dtype == ir.DataType.FLOAT
            assert [str(d) for d in output.shape] == [
                "batch",
                str(config.z_dim),
                "latent_frames",
                "latent_height",
                "latent_width",
            ]

    def test_decoder_graph_io(self):
        config, _, package = _build(TINY_WAN22)
        graph = package["decoder"].graph

        assert [i.name for i in graph.inputs] == ["latent"]
        latent = graph.inputs[0]
        assert latent.dtype == ir.DataType.FLOAT
        assert [str(d) for d in latent.shape] == [
            "batch",
            str(config.z_dim),
            "latent_frames",
            "latent_height",
            "latent_width",
        ]

        assert [o.name for o in graph.outputs] == ["sample"]
        sample = graph.outputs[0]
        assert sample.dtype == ir.DataType.FLOAT
        assert [str(d) for d in sample.shape] == [
            "batch",
            str(config.decoded_video_channels),
            "frames",
            "height",
            "width",
        ]

    def test_package_keys_and_roles(self):
        _, _, package = _build(TINY_WAN22)

        assert sorted(package.keys()) == ["decoder", "encoder"]
        assert WanVAETask.model_roles == {"encoder": "encoder", "decoder": "decoder"}

    def test_graph_metadata_and_opset(self):
        from mobius._constants import OPSET_VERSION

        _, _, package = _build(TINY_WAN22)

        for model in package.values():
            assert model.producer_name == "mobius"
            assert model.graph.opset_imports[""] == OPSET_VERSION

    def test_encoder_node_construction(self):
        _, _, package = _build(TINY_WAN22)
        ops = _op_types(package["encoder"])

        # Causal 3D convolution (Pad + Conv), spatial downsample (ZeroPad2d),
        # AvgDown3D shortcut (Pad/Reshape/Transpose/ReduceMean), RMS norm
        # (ReduceL2/Max/Div), SiLU (Sigmoid/Mul), single-head attention, and the
        # posterior split with the logvar clamp.
        assert {"Conv", "Pad", "Reshape", "Transpose", "Concat"} <= ops
        assert {"ReduceL2", "Max", "Div", "Sigmoid", "Mul", "Add"} <= ops
        assert "ReduceMean" in ops  # AvgDown3D grouped mean
        assert "Attention" in ops  # mid-block single-head attention
        assert {"Split", "Clip"} <= ops  # mean/logvar split + logvar clamp
        # No sampling in the graph: the posterior is returned deterministically.
        assert "RandomNormalLike" not in ops
        assert "RandomNormal" not in ops

    def test_decoder_node_construction(self):
        _, _, package = _build(TINY_WAN22)
        ops = _op_types(package["decoder"])

        assert "Resize" in ops  # nearest-exact 2x spatial upsample
        assert "Expand" in ops  # DupUp3D channel repeat_interleave
        assert "Slice" in ops  # first-frame passthrough / first_chunk drop
        assert "Clip" in ops  # unconditional clamp to [-1, 1]
        assert "Attention" in ops

    def test_nearest_upsample_uses_floor_semantics(self):
        _, _, package = _build(TINY_WAN22)

        resizes = [n for n in package["decoder"].graph if n.op_type == "Resize"]
        assert resizes
        for node in resizes:
            assert node.attributes["mode"].as_string() == "nearest"
            assert node.attributes["nearest_mode"].as_string() == "floor"
            assert (
                node.attributes["coordinate_transformation_mode"].as_string() == "asymmetric"
            )

    def test_latent_statistics_are_graph_level_constants(self):
        config, _, package = _build(TINY_WAN22)

        for key in ("encoder", "decoder"):
            initializers = package[key].graph.initializers
            assert "latents_mean" in initializers
            assert "latents_std" in initializers
            mean = initializers["latents_mean"].const_value
            assert mean is not None
            assert tuple(mean.shape) == (1, config.z_dim, 1, 1, 1)
            np.testing.assert_allclose(
                np.asarray(mean.numpy()).reshape(-1), config.latents_mean, rtol=1e-6
            )

    def test_graph_dtype_follows_config(self):
        from mobius._builder import _cast_module_dtype

        config = WanVAEConfig.from_diffusers(dict(TINY_WAN22, dtype="float16"))
        module = AutoencoderKLWanModel(config)
        # ``build_from_module`` casts parameters before building; do the same so
        # ONNX type inference sees a uniformly half-precision graph.
        _cast_module_dtype(module, config.dtype)
        package = WanVAETask().build(module, config)

        for model in package.values():
            graph = model.graph
            assert graph.inputs[0].dtype == ir.DataType.FLOAT16
            assert graph.outputs[0].dtype == ir.DataType.FLOAT16
            assert graph.initializers["latents_mean"].dtype == ir.DataType.FLOAT16
            assert graph.initializers["latents_std"].dtype == ir.DataType.FLOAT16

    def test_weight_names_match_huggingface_layout(self):
        _, _, package = _build(TINY_WAN22)
        encoder = set(package["encoder"].graph.initializers)
        decoder = set(package["decoder"].graph.initializers)

        # Encoder: residual down block with a temporal downsampler.
        assert "encoder.conv_in.weight" in encoder
        assert "encoder.down_blocks.0.resnets.0.norm1.gamma" in encoder
        assert "encoder.down_blocks.0.resnets.0.conv1.weight" in encoder
        # nn.Sequential(ZeroPad2d, Conv2d) -> the conv is at index 1.
        assert "encoder.down_blocks.0.downsampler.resample.1.weight" in encoder
        assert "encoder.down_blocks.1.downsampler.time_conv.weight" in encoder
        assert "encoder.mid_block.attentions.0.to_qkv.weight" in encoder
        assert "encoder.mid_block.resnets.1.conv2.bias" in encoder
        assert "encoder.norm_out.gamma" in encoder
        assert "encoder.conv_out.weight" in encoder
        assert "quant_conv.weight" in encoder

        # Decoder: residual up block uses the singular ``upsampler`` attribute.
        assert "post_quant_conv.weight" in decoder
        assert "decoder.conv_in.weight" in decoder
        assert "decoder.up_blocks.0.upsampler.resample.1.weight" in decoder
        assert "decoder.up_blocks.0.upsampler.time_conv.weight" in decoder
        assert "decoder.norm_out.gamma" in decoder
        assert "decoder.conv_out.bias" in decoder

        # Parameter-free shortcuts must not create initializers.
        assert not [n for n in encoder | decoder if "avg_shortcut" in n]

    def test_shortcut_conv_only_when_channels_change(self):
        _, _, package = _build(TINY_WAN22)
        encoder = set(package["encoder"].graph.initializers)

        # Stage 0 keeps 8 channels (dim_mult[0] == 1) -> nn.Identity upstream.
        assert "encoder.down_blocks.0.resnets.0.conv_shortcut.weight" not in encoder
        # Stage 1 widens 8 -> 16 -> a 1x1x1 causal conv shortcut.
        assert "encoder.down_blocks.1.resnets.0.conv_shortcut.weight" in encoder

    def test_wan21_non_residual_layout(self):
        config, _, package = _build(TINY_WAN21)
        encoder = set(package["encoder"].graph.initializers)
        decoder = set(package["decoder"].graph.initializers)

        assert config.is_residual is False
        assert config.patch_size is None
        assert config.video_channels == 3
        # Wan 2.1 flattens residual blocks and resamplers into ``down_blocks``.
        assert "encoder.down_blocks.0.conv1.weight" in encoder
        assert "encoder.down_blocks.1.resample.1.weight" in encoder
        # Wan 2.1 decoder stores upsamplers in a one-element ModuleList.
        assert "decoder.up_blocks.0.upsamplers.0.resample.1.weight" in decoder
        assert "decoder.up_blocks.0.upsamplers.0.time_conv.weight" in decoder
        assert not [n for n in decoder if "up_blocks.0.upsampler." in n]

    def test_attn_scales_insert_encoder_attention_blocks(self):
        # Wan 2.1 inserts an attention block after each residual block whose
        # spatial scale is listed in ``attn_scales`` (1.0 == full resolution).
        _, _, package = _build(dict(TINY_WAN21, attn_scales=[1.0]))
        encoder = set(package["encoder"].graph.initializers)

        assert "encoder.down_blocks.1.to_qkv.weight" in encoder
        assert "encoder.down_blocks.1.proj.weight" in encoder

    def test_preprocess_weights_is_identity(self):
        _, module, _ = _build(TINY_WAN22)
        state_dict = {"encoder.conv_in.weight": object()}

        assert module.preprocess_weights(state_dict) is state_dict

    def test_task_rejects_module_without_components(self):
        from onnxscript import nn

        config = WanVAEConfig.from_diffusers(TINY_WAN22)

        with pytest.raises(TypeError, match="encoder"):
            WanVAETask().build(nn.Module(), config)


# ---------------------------------------------------------------------------
# Numerical parity against diffusers
# ---------------------------------------------------------------------------


def _wan_module(name: str):
    """Import a private class from ``diffusers.models.autoencoders.autoencoder_kl_wan``."""
    module = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_wan")
    return getattr(module, name)


def _single_block_graph(block, torch_input, dtype=ir.DataType.FLOAT):
    """Build a one-block ONNX model with ``x`` as its only input/output."""
    graph, builder = _make_graph(name="block")
    x = builder.input("x", dtype=dtype, shape=list(torch_input.shape))
    builder.add_output(block(builder.op, x), "y")
    return _make_model(graph)


class TestParity:
    """Compare individual blocks and the full VAE against diffusers PyTorch."""

    def test_causal_conv3d_matches_diffusers(self):
        pytest.importorskip("onnxruntime")
        torch = pytest.importorskip("torch")
        from mobius._weight_loading import apply_weights
        from mobius.models.wan_vae import _WanCausalConv3d

        wan_causal_conv3d = _wan_module("WanCausalConv3d")
        torch.manual_seed(0)
        reference = wan_causal_conv3d(3, 5, 3, padding=1).eval()
        block = _WanCausalConv3d(3, 5, 3, padding=1)

        x = torch.randn(1, 3, 4, 6, 6)
        with torch.no_grad():
            expected = reference(x).numpy()

        model = _single_block_graph(block, x)
        apply_weights(model, dict(reference.state_dict()))
        actual = _run(model, {"x": x.numpy()})[0]

        # Causal padding must keep the frame count and never look ahead in time.
        assert actual.shape == expected.shape
        assert np.abs(actual - expected).max() < 1e-5

    def test_residual_block_matches_diffusers(self):
        pytest.importorskip("onnxruntime")
        torch = pytest.importorskip("torch")
        from mobius._weight_loading import apply_weights
        from mobius.models.wan_vae import _WanResidualBlock

        wan_residual_block = _wan_module("WanResidualBlock")
        torch.manual_seed(0)
        reference = wan_residual_block(4, 6).eval()
        block = _WanResidualBlock(4, 6)

        x = torch.randn(1, 4, 3, 5, 5)
        with torch.no_grad():
            expected = reference(x).numpy()

        model = _single_block_graph(block, x)
        apply_weights(model, dict(reference.state_dict()))
        actual = _run(model, {"x": x.numpy()})[0]

        assert np.abs(actual - expected).max() < 1e-5

    def test_downsample3d_matches_chunked_diffusers(self):
        """Whole-sequence ``downsample3d`` == upstream's cached chunk loop."""
        pytest.importorskip("onnxruntime")
        torch = pytest.importorskip("torch")
        from mobius._weight_loading import apply_weights
        from mobius.models.wan_vae import _WanResample

        wan_resample = _wan_module("WanResample")
        torch.manual_seed(0)
        reference = wan_resample(4, mode="downsample3d").eval()
        block = _WanResample(4, mode="downsample3d")

        # The encoder feeds frame 0 alone, then groups of four.
        x = torch.randn(1, 4, 9, 8, 8)
        chunks = [x[:, :, :1], x[:, :, 1:5], x[:, :, 5:9]]
        feat_cache: list = [None]
        with torch.no_grad():
            expected = torch.cat(
                [reference(c, feat_cache=feat_cache, feat_idx=[0]) for c in chunks], dim=2
            ).numpy()

        model = _single_block_graph(block, x)
        apply_weights(model, dict(reference.state_dict()))
        actual = _run(model, {"x": x.numpy()})[0]

        assert actual.shape == expected.shape
        assert np.abs(actual - expected).max() < 1e-5

    def test_upsample3d_matches_chunked_diffusers(self):
        """Whole-sequence ``upsample3d`` == upstream's per-frame cached loop."""
        pytest.importorskip("onnxruntime")
        torch = pytest.importorskip("torch")
        from mobius._weight_loading import apply_weights
        from mobius.models.wan_vae import _WanResample

        wan_resample = _wan_module("WanResample")
        torch.manual_seed(0)
        reference = wan_resample(4, mode="upsample3d", upsample_out_dim=4).eval()
        block = _WanResample(4, mode="upsample3d", upsample_out_dim=4)

        # The decoder feeds one latent frame at a time.
        x = torch.randn(1, 4, 3, 4, 4)
        feat_cache: list = [None]
        with torch.no_grad():
            expected = torch.cat(
                [
                    reference(x[:, :, i : i + 1], feat_cache=feat_cache, feat_idx=[0])
                    for i in range(x.shape[2])
                ],
                dim=2,
            ).numpy()

        model = _single_block_graph(block, x)
        apply_weights(model, dict(reference.state_dict()))
        actual = _run(model, {"x": x.numpy()})[0]

        # Frame 0 is not temporally doubled: 3 latent frames -> 2 * 3 - 1 = 5.
        assert actual.shape[2] == 2 * x.shape[2] - 1
        assert actual.shape == expected.shape
        assert np.abs(actual - expected).max() < 1e-5

    @pytest.mark.parametrize("config_dict", [TINY_WAN22, TINY_WAN21])
    def test_encoder_and_decoder_match_diffusers(self, config_dict):
        pytest.importorskip("onnxruntime")
        torch = pytest.importorskip("torch")
        from mobius._weight_loading import apply_weights

        autoencoder_kl_wan = _wan_module("AutoencoderKLWan")
        torch.manual_seed(0)
        kwargs = {k: v for k, v in config_dict.items() if not k.startswith("_")}
        kwargs.pop("clip_output", None)
        reference = autoencoder_kl_wan(**kwargs).eval()
        state_dict = dict(reference.state_dict())

        config, _, package = _build(config_dict)
        encoder_names = set(package["encoder"].graph.initializers)
        decoder_names = set(package["decoder"].graph.initializers)

        # Weight-name alignment: the module-scoped initializers of the two graphs
        # are exactly the checkpoint keys — no renaming, nothing missing or extra.
        module_prefixes = ("encoder.", "quant_conv.", "decoder.", "post_quant_conv.")
        graph_parameters = {
            name for name in encoder_names | decoder_names if name.startswith(module_prefixes)
        }
        assert graph_parameters == set(state_dict)

        apply_weights(
            package["encoder"],
            {k: v for k, v in state_dict.items() if k in encoder_names},
        )
        apply_weights(
            package["decoder"],
            {k: v for k, v in state_dict.items() if k in decoder_names},
        )

        latents_mean = np.asarray(config.latents_mean, dtype=np.float32).reshape(
            1, -1, 1, 1, 1
        )
        latents_std = np.asarray(config.latents_std, dtype=np.float32).reshape(1, -1, 1, 1, 1)

        # 5 video frames == 4 * 1 + 1; two spatial stages plus patchification.
        video = torch.randn(1, config.video_channels, 5, 16, 16)
        with torch.no_grad():
            posterior = reference.encode(video).latent_dist
        mean, logvar, latent = _run(package["encoder"], {"sample": video.numpy()})

        assert np.abs(mean - posterior.mean.numpy()).max() < 1e-4
        assert np.abs(logvar - posterior.logvar.numpy()).max() < 1e-4
        # The normalisation lives at the graph boundary, not inside ``encoder``.
        assert np.abs(latent - (mean - latents_mean) / latents_std).max() < 1e-5

        latent_frames = mean.shape[2]
        z = torch.randn(1, config.z_dim, latent_frames, *mean.shape[3:])
        with torch.no_grad():
            expected = reference.decode(z).sample.numpy()
        normalized = ((z.numpy() - latents_mean) / latents_std).astype(np.float32)
        actual = _run(package["decoder"], {"latent": normalized})[0]

        assert actual.shape == expected.shape
        assert actual.shape[1] == config.decoded_video_channels
        # T_video = scale_factor_temporal * (T_latent - 1) + 1
        assert actual.shape[2] == config.scale_factor_temporal * (latent_frames - 1) + 1
        assert np.abs(actual - expected).max() < 1e-4

    def test_longer_sequence_matches_diffusers(self):
        """A 9-frame clip spans three encode chunks, exercising the cache seam."""
        pytest.importorskip("onnxruntime")
        torch = pytest.importorskip("torch")
        from mobius._weight_loading import apply_weights

        autoencoder_kl_wan = _wan_module("AutoencoderKLWan")
        torch.manual_seed(0)
        kwargs = {k: v for k, v in TINY_WAN22.items() if not k.startswith("_")}
        kwargs.pop("clip_output")
        reference = autoencoder_kl_wan(**kwargs).eval()
        state_dict = dict(reference.state_dict())

        config, _, package = _build(TINY_WAN22)
        encoder_names = set(package["encoder"].graph.initializers)
        decoder_names = set(package["decoder"].graph.initializers)
        apply_weights(
            package["encoder"],
            {k: v for k, v in state_dict.items() if k in encoder_names},
        )
        apply_weights(
            package["decoder"],
            {k: v for k, v in state_dict.items() if k in decoder_names},
        )

        video = torch.randn(1, config.video_channels, 9, 16, 16)
        with torch.no_grad():
            posterior = reference.encode(video).latent_dist
        mean, _, latent = _run(package["encoder"], {"sample": video.numpy()})

        # T_latent = (T_video - 1) / scale_factor_temporal + 1
        assert mean.shape[2] == (9 - 1) // config.scale_factor_temporal + 1
        assert np.abs(mean - posterior.mean.numpy()).max() < 1e-4

        with torch.no_grad():
            expected = reference.decode(posterior.mean).sample.numpy()
        actual = _run(package["decoder"], {"latent": latent})[0]

        assert actual.shape == video.shape
        assert actual.shape == expected.shape
        assert np.abs(actual - expected).max() < 1e-4
