# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Numerical and execution-provider tests for Qwen Image."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import onnx_ir as ir
import pytest

from mobius.integrations._weight_loading import apply_weights
from mobius.integrations.diffusers._configs import QwenImageConfig
from mobius.models.qwen_image import (
    QwenImageTransformer2DModel,
    prepare_qwen_image_rotary_embeddings,
)
from mobius.tasks import QwenImageDenoisingTask


def _tiny_config(dtype: ir.DataType = ir.DataType.FLOAT) -> QwenImageConfig:
    return QwenImageConfig(
        in_channels=4,
        out_channels=4,
        patch_size=2,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=8,
        cross_attention_dim=8,
        axes_dims_rope=(2, 2, 4),
        dtype=dtype,
    )


def _create_model_and_feeds():
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(7)
    hf_model = diffusers.QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=4,
        out_channels=4,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=8,
        axes_dims_rope=(2, 2, 4),
    ).eval()
    config = _tiny_config()
    module = QwenImageTransformer2DModel(config)
    model = QwenImageDenoisingTask().build(module, config)["model"]
    apply_weights(model, module.preprocess_weights(dict(hf_model.state_dict())))

    # Four target tokens followed by four nonzero source-image tokens.
    generator = torch.Generator().manual_seed(11)
    target = torch.randn((1, 4, 4), generator=generator)
    source = torch.linspace(-0.75, 0.75, 16).reshape(1, 4, 4)
    sample = torch.cat([target, source], dim=1)
    text = torch.randn((1, 3, 8), generator=generator)
    text_mask = torch.tensor([[True, False, True]])
    timestep = torch.tensor([0.5])
    image_shapes = [(1, 2, 2), (1, 2, 2)]
    image_cos, image_sin, text_cos, text_sin = prepare_qwen_image_rotary_embeddings(
        image_shapes, text.shape[1], config.axes_dims_rope
    )
    feeds = {
        "sample": sample.numpy(),
        "timestep": timestep.numpy(),
        "encoder_hidden_states": text.numpy(),
        "encoder_hidden_states_mask": text_mask.numpy(),
        "image_rotary_cos": image_cos,
        "image_rotary_sin": image_sin,
        "text_rotary_cos": text_cos,
        "text_rotary_sin": text_sin,
        "target_sequence_length": np.array([4], dtype=np.int64),
    }
    return hf_model, model, feeds, image_shapes


def test_rotary_embeddings_match_diffusers():
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    hf_model = diffusers.QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=4,
        out_channels=4,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=8,
        axes_dims_rope=(2, 2, 4),
    )
    image_shapes = [(1, 2, 4), (1, 4, 2)]
    actual = prepare_qwen_image_rotary_embeddings(image_shapes, 5, (2, 2, 4))
    image_freqs, text_freqs = hf_model.pos_embed(
        [image_shapes], max_txt_seq_len=5, device=torch.device("cpu")
    )
    expected = (
        image_freqs.real.numpy(),
        image_freqs.imag.numpy(),
        text_freqs.real.numpy(),
        text_freqs.imag.numpy(),
    )
    for actual_value, expected_value in zip(actual, expected):
        np.testing.assert_allclose(actual_value, expected_value, rtol=1e-6, atol=1e-6)


def test_transformer_weight_names_match_diffusers():
    diffusers = pytest.importorskip("diffusers")
    hf_model = diffusers.QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=4,
        out_channels=4,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=8,
        axes_dims_rope=(2, 2, 4),
    )
    module = QwenImageTransformer2DModel(_tiny_config())
    assert {name for name, _ in module.named_parameters()} == set(hf_model.state_dict())


def test_vae_weight_names_match_diffusers():
    diffusers = pytest.importorskip("diffusers")

    from mobius.integrations.diffusers._configs import QwenImageVAEConfig
    from mobius.models.qwen_image_vae import AutoencoderKLQwenImageModel

    hf_vae = diffusers.AutoencoderKLQwenImage(
        base_dim=8,
        z_dim=4,
        dim_mult=[1, 2],
        num_res_blocks=1,
        temperal_downsample=[False],
    )
    module = AutoencoderKLQwenImageModel(
        QwenImageVAEConfig(
            base_dim=8,
            z_dim=4,
            dim_mult=(1, 2),
            num_res_blocks=1,
            temperal_downsample=(False,),
        )
    )
    assert {name for name, _ in module.named_parameters()} == set(hf_vae.state_dict())


def test_edit_vae_rejects_wrong_latent_statistics_length():
    from mobius.integrations.diffusers._configs import QwenImageVAEConfig
    from mobius.models.qwen_image_vae import AutoencoderKLQwenImageModel
    from mobius.tasks import QwenImageEditVAETask

    config = QwenImageVAEConfig(
        base_dim=8,
        z_dim=4,
        dim_mult=(1, 2),
        num_res_blocks=1,
        temperal_downsample=(False,),
        latents_mean=(0.0, 0.0, 0.0),
        latents_std=(1.0, 1.0, 1.0, 1.0),
    )
    with pytest.raises(ValueError, match="latents_mean must contain 4 values"):
        QwenImageEditVAETask().build(AutoencoderKLQwenImageModel(config), config)


@pytest.mark.integration
@pytest.mark.integration_fast
def test_transformer_matches_diffusers_with_source_image_and_mask():
    import onnxruntime as ort

    hf_model, model, feeds, image_shapes = _create_model_and_feeds()
    torch = pytest.importorskip("torch")
    with torch.no_grad():
        expected = hf_model(
            hidden_states=torch.from_numpy(feeds["sample"]),
            timestep=torch.from_numpy(feeds["timestep"]),
            encoder_hidden_states=torch.from_numpy(feeds["encoder_hidden_states"]),
            encoder_hidden_states_mask=torch.from_numpy(feeds["encoder_hidden_states_mask"]),
            img_shapes=[image_shapes],
        ).sample[:, :4]

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "model.onnx")
        ir.save(model, path)
        actual = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(
            None, feeds
        )[0]
    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-3, atol=1e-3)


@pytest.mark.integration
def test_cuda_matches_cpu():
    ort = pytest.importorskip("onnxruntime")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("CUDAExecutionProvider is not available")
    ort.preload_dlls()
    _, model, feeds, _ = _create_model_and_feeds()
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "model.onnx")
        ir.save(model, path)
        cpu = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, feeds)[
            0
        ]
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.profile_file_prefix = os.path.join(directory, "cuda_profile")
        session = ort.InferenceSession(
            path, sess_options=options, providers=["CUDAExecutionProvider"]
        )
        if session.get_providers()[0] != "CUDAExecutionProvider":
            session.end_profiling()
            del session
            pytest.skip("CUDAExecutionProvider could not initialize on this host")
        cuda = session.run(None, feeds)[0]
        profile_path = session.end_profiling()
        with open(profile_path, encoding="utf-8") as profile_file:
            profile = json.load(profile_file)
        assert any(
            event.get("args", {}).get("provider") == "CUDAExecutionProvider"
            for event in profile
        )
    np.testing.assert_allclose(cuda, cpu, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "dtype",
    [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16],
)
def test_all_supported_dtypes_build(dtype: ir.DataType):
    config = _tiny_config(dtype)
    model = QwenImageDenoisingTask().build(QwenImageTransformer2DModel(config), config)[
        "model"
    ]
    assert model.graph.inputs[0].dtype == dtype
    assert not any(node.op_type == "Identity" for node in model.graph)


def test_transformer_graph_is_fused_and_post_weight_optimized():
    from collections import Counter

    from mobius import build_from_module
    from mobius._optimizations import fold_initializers_after_weights

    config = QwenImageConfig(
        in_channels=16,
        out_channels=4,
        patch_size=2,
        num_layers=2,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=8,
        cross_attention_dim=8,
        axes_dims_rope=(2, 2, 4),
        dtype=ir.DataType.FLOAT16,
    )
    model = build_from_module(
        QwenImageTransformer2DModel(config),
        config,
        task="qwen-image-denoising",
        execution_provider="cuda",
    )["model"]
    counts = Counter(node.op_type for node in model.graph.all_nodes())

    # Main's fused-Swish path plus CSE shares the common timestep modulation
    # across image/text streams and transformer blocks.
    assert counts["Swish"] == 2
    assert counts["Sigmoid"] == 0
    assert counts["SkipLayerNormalization"] == 6
    assert counts["Gelu"] == 3
    assert counts["Identity"] == 0
    assert counts["Attention"] == config.num_layers

    rng = np.random.default_rng(17)
    for initializer in model.graph.initializers.values():
        if initializer.const_value is None:
            initializer.const_value = ir.tensor(
                rng.standard_normal(list(initializer.shape)).astype(initializer.dtype.numpy())
            )
    transpose_count = counts["Transpose"]
    fold_initializers_after_weights(model)
    optimized_counts = Counter(node.op_type for node in model.graph.all_nodes())

    # All per-weight Linear transposes fold into the initializers; only dynamic
    # data-layout transposes for RoPE/attention remain.
    assert transpose_count == 39
    assert optimized_counts["Transpose"] == 8
    assert sum(optimized_counts.values()) < sum(counts.values())


@pytest.mark.integration
@pytest.mark.integration_fast
def test_edit_vae_matches_diffusers_on_real_source_image():
    ort = pytest.importorskip("onnxruntime")
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    image_module = pytest.importorskip("PIL.Image")

    from mobius.integrations.diffusers._configs import QwenImageVAEConfig
    from mobius.models.qwen_image_vae import AutoencoderKLQwenImageModel
    from mobius.tasks import QwenImageEditVAETask

    means = [-0.2, -0.1, 0.1, 0.2]
    stds = [1.1, 1.2, 1.3, 1.4]
    torch.manual_seed(13)
    hf_vae = diffusers.AutoencoderKLQwenImage(
        base_dim=8,
        z_dim=4,
        dim_mult=[1, 2],
        num_res_blocks=1,
        temperal_downsample=[False],
        latents_mean=means,
        latents_std=stds,
    ).eval()
    config = QwenImageVAEConfig(
        base_dim=8,
        z_dim=4,
        dim_mult=(1, 2),
        num_res_blocks=1,
        temperal_downsample=(False,),
        latents_mean=tuple(means),
        latents_std=tuple(stds),
    )
    module = AutoencoderKLQwenImageModel(config)
    package = QwenImageEditVAETask().build(module, config)
    weights = module.preprocess_weights(dict(hf_vae.state_dict()))
    apply_weights(package["encoder"], weights)
    apply_weights(package["decoder"], weights)

    image = image_module.open(Path("testdata") / "pipeline-cat-chonk.jpeg").convert("RGB")
    image = image.resize((16, 16))
    pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1.0
    pixels = pixels[None, :, None, :, :]
    with torch.no_grad():
        moments = hf_vae._encode(torch.from_numpy(pixels))
        expected_latents = (
            moments.chunk(2, dim=1)[0] - torch.tensor(means)[None, :, None, None, None]
        )
        expected_latents = expected_latents / torch.tensor(stds)[None, :, None, None, None]
        denormalized = (
            expected_latents * torch.tensor(stds)[None, :, None, None, None]
            + torch.tensor(means)[None, :, None, None, None]
        )
        expected_image = hf_vae.decode(denormalized).sample.numpy()

    with tempfile.TemporaryDirectory() as directory:
        encoder_path = os.path.join(directory, "encoder.onnx")
        decoder_path = os.path.join(directory, "decoder.onnx")
        ir.save(package["encoder"], encoder_path)
        ir.save(package["decoder"], decoder_path)
        actual_latents = ort.InferenceSession(
            encoder_path, providers=["CPUExecutionProvider"]
        ).run(None, {"sample": pixels})[0]
        actual_image = ort.InferenceSession(
            decoder_path, providers=["CPUExecutionProvider"]
        ).run(None, {"latent_sample": actual_latents})[0]

    np.testing.assert_allclose(actual_latents, expected_latents.numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(actual_image, expected_image, rtol=1e-4, atol=1e-4)


def _temporal_vae(dtype: ir.DataType = ir.DataType.FLOAT):
    """Build a tiny edit VAE whose resamplers own temporal convolutions.

    ``temperal_downsample=(True,)`` is what the real Qwen Image Edit checkpoint
    uses, and it is the only configuration that instantiates ``time_conv`` in the
    down/upsample blocks -- the code path the image (single frame) case must skip.
    """
    from mobius._diffusers_configs import QwenImageVAEConfig
    from mobius.models.qwen_image_vae import AutoencoderKLQwenImageModel
    from mobius.tasks import QwenImageEditVAETask

    config = QwenImageVAEConfig(
        base_dim=8,
        z_dim=4,
        dim_mult=(1, 2),
        num_res_blocks=1,
        temperal_downsample=(True,),
        latents_mean=(-0.2, -0.1, 0.1, 0.2),
        latents_std=(1.1, 1.2, 1.3, 1.4),
        dtype=dtype,
    )
    module = AutoencoderKLQwenImageModel(config)
    return config, module, QwenImageEditVAETask().build(module, config)


def test_edit_vae_skips_temporal_convolutions_for_single_frame_images():
    """A T=1 image must not run the temporal resampling branch.

    ``QwenImageResample.forward`` only applies ``time_conv`` (and the temporal
    pixel shuffle) from the *second* cached chunk onward, so a lone image chunk
    keeps its frame count. Applying it unconditionally produced a graph whose
    Conv output dimension collapsed to zero and could not even be loaded, so
    this pins both the graph shape and the numerical agreement with diffusers.
    """
    ort = pytest.importorskip("onnxruntime")
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")

    means = [-0.2, -0.1, 0.1, 0.2]
    stds = [1.1, 1.2, 1.3, 1.4]
    torch.manual_seed(17)
    hf_vae = diffusers.AutoencoderKLQwenImage(
        base_dim=8,
        z_dim=4,
        dim_mult=[1, 2],
        num_res_blocks=1,
        temperal_downsample=[True],
        latents_mean=means,
        latents_std=stds,
    ).eval()
    _, module, package = _temporal_vae()
    weights = module.preprocess_weights(dict(hf_vae.state_dict()))
    apply_weights(package["encoder"], weights)
    apply_weights(package["decoder"], weights)

    generator = torch.Generator().manual_seed(19)
    pixels = torch.randn((1, 3, 1, 16, 16), generator=generator)
    with torch.no_grad():
        moments = hf_vae._encode(pixels)
        mean = moments.chunk(2, dim=1)[0]
        scale = torch.tensor(stds)[None, :, None, None, None]
        offset = torch.tensor(means)[None, :, None, None, None]
        expected_latents = (mean - offset) / scale
        expected_image = hf_vae.decode(expected_latents * scale + offset).sample.numpy()

    with tempfile.TemporaryDirectory() as directory:
        encoder_path = os.path.join(directory, "encoder.onnx")
        decoder_path = os.path.join(directory, "decoder.onnx")
        ir.save(package["encoder"], encoder_path)
        ir.save(package["decoder"], decoder_path)
        actual_latents = ort.InferenceSession(
            encoder_path, providers=["CPUExecutionProvider"]
        ).run(None, {"sample": pixels.numpy()})[0]
        actual_image = ort.InferenceSession(
            decoder_path, providers=["CPUExecutionProvider"]
        ).run(None, {"latent_sample": actual_latents})[0]

    # The frame axis survives encode and decode: a single image stays a single image.
    assert actual_latents.shape[2] == 1
    assert actual_image.shape == (1, 3, 1, 16, 16)
    np.testing.assert_allclose(actual_latents, expected_latents.numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(actual_image, expected_image, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("part", ["encoder", "decoder"])
def test_edit_vae_bfloat16_graph_has_only_kernel_backed_ops(part):
    """bfloat16 VAE graphs must avoid ops onnxruntime has no bfloat16 kernel for.

    onnxruntime ships no bfloat16 ``ReduceL2``, ``Resize`` or ``Clip`` kernel on
    any execution provider, and a single unassignable node aborts session
    creation outright. The RMS norm reduces in float32, ``Resize`` is sandwiched
    between casts, and the bfloat16 lowering pass rewrites ``Clip`` into
    ``Min``/``Max``, so no bfloat16-typed instance of those ops may survive.
    """
    from mobius._optimizations import optimize_model

    package = _temporal_vae(ir.DataType.BFLOAT16)[2]
    model = package[part]
    optimize_model(model, ep="cuda", dtype=ir.DataType.BFLOAT16, model_role="encoder")
    unsupported = {"ReduceL2", "Resize", "Clip"}
    offenders = [
        node.op_type
        for node in ir.traversal.RecursiveGraphIterator(model.graph)
        if node.op_type in unsupported
        and any(
            value is not None and value.dtype == ir.DataType.BFLOAT16
            for value in (*node.inputs, *node.outputs)
        )
    ]
    assert offenders == []


@pytest.mark.integration
@pytest.mark.integration_fast
def test_deterministic_l4_l5_image_edit_golden():
    """Compare an independent diffusers edit chain with the full Mobius ONNX chain."""
    ort = pytest.importorskip("onnxruntime")
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    image_module = pytest.importorskip("PIL.Image")

    from mobius.integrations.diffusers._configs import QwenImageVAEConfig
    from mobius.models.qwen_image_vae import AutoencoderKLQwenImageModel
    from mobius.tasks import QwenImageEditVAETask

    golden_path = Path("testdata/golden/diffusion/qwen-image-edit-2509.json")
    with open(golden_path, encoding="utf-8") as golden_file:
        golden = json.load(golden_file)

    means = (-0.2, -0.1, 0.1, 0.2)
    stds = (1.1, 1.2, 1.3, 1.4)
    torch.manual_seed(golden["seed"])
    hf_vae = diffusers.AutoencoderKLQwenImage(
        base_dim=8,
        z_dim=4,
        dim_mult=[1, 2],
        num_res_blocks=1,
        temperal_downsample=[False],
        latents_mean=list(means),
        latents_std=list(stds),
    ).eval()
    hf_transformer = diffusers.QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=16,
        out_channels=4,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=8,
        axes_dims_rope=(2, 2, 4),
    ).eval()
    transformer_config = QwenImageConfig(
        in_channels=16,
        out_channels=4,
        patch_size=2,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=8,
        cross_attention_dim=8,
        axes_dims_rope=(2, 2, 4),
    )
    vae_config = QwenImageVAEConfig(
        base_dim=8,
        z_dim=4,
        dim_mult=(1, 2),
        num_res_blocks=1,
        temperal_downsample=(False,),
        latents_mean=means,
        latents_std=stds,
    )
    transformer_module = QwenImageTransformer2DModel(transformer_config)
    transformer_model = QwenImageDenoisingTask().build(transformer_module, transformer_config)[
        "model"
    ]
    transformer_weights = transformer_module.preprocess_weights(
        dict(hf_transformer.state_dict())
    )
    apply_weights(transformer_model, transformer_weights)
    vae_module = AutoencoderKLQwenImageModel(vae_config)
    vae_package = QwenImageEditVAETask().build(vae_module, vae_config)
    vae_weights = vae_module.preprocess_weights(dict(hf_vae.state_dict()))
    apply_weights(vae_package["encoder"], vae_weights)
    apply_weights(vae_package["decoder"], vae_weights)

    image = image_module.open(Path("testdata") / golden["source_image"]).convert("RGB")
    source_pixels = np.asarray(image.resize((16, 16)), dtype=np.float32)
    source_pixels = source_pixels.transpose(2, 0, 1) / 127.5 - 1.0
    source_pixels = source_pixels[None, :, None, :, :]

    generator = torch.Generator().manual_seed(golden["seed"] + 1)
    initial_target_latents = torch.randn((1, 4, 1, 8, 8), generator=generator)
    prompt_embeds = torch.randn((1, 3, 8), generator=generator)
    prompt_mask = torch.tensor([[True, False, True]])
    mean_tensor = torch.tensor(means).view(1, 4, 1, 1, 1)
    std_tensor = torch.tensor(stds).view(1, 4, 1, 1, 1)

    def pack_latents(latents):
        batch, channels, frames, height, width = latents.shape
        assert frames == 1 and height % 2 == 0 and width % 2 == 0
        return (
            latents.reshape(batch, channels, height // 2, 2, width // 2, 2)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch, (height // 2) * (width // 2), channels * 4)
        )

    def unpack_latents(latents):
        batch, _, packed_channels = latents.shape
        channels = packed_channels // 4
        return (
            latents.reshape(batch, 4, 4, channels, 2, 2)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(batch, channels, 1, 8, 8)
        )

    def pack_numpy_latents(latents):
        batch, channels, frames, height, width = latents.shape
        assert frames == 1 and height % 2 == 0 and width % 2 == 0
        return (
            latents.reshape(batch, channels, height // 2, 2, width // 2, 2)
            .transpose(0, 2, 4, 1, 3, 5)
            .reshape(batch, (height // 2) * (width // 2), channels * 4)
        )

    def unpack_numpy_latents(latents):
        batch, _, packed_channels = latents.shape
        channels = packed_channels // 4
        return (
            latents.reshape(batch, 4, 4, channels, 2, 2)
            .transpose(0, 3, 1, 4, 2, 5)
            .reshape(batch, channels, 1, 8, 8)
        )

    with torch.no_grad():
        reference_moments = hf_vae._encode(torch.from_numpy(source_pixels))
        reference_source_latents = reference_moments.chunk(2, dim=1)[0] - mean_tensor
        reference_source_latents = reference_source_latents / std_tensor
        reference_source_tokens = pack_latents(reference_source_latents)
        initial_target_tokens = pack_latents(initial_target_latents)

    image_shapes = [(1, 4, 4), (1, 4, 4)]
    image_cos, image_sin, text_cos, text_sin = prepare_qwen_image_rotary_embeddings(
        image_shapes, 3, (2, 2, 4)
    )

    steps = golden["num_inference_steps"]
    sigmas = np.linspace(1.0, 1 / steps, steps)
    slope = (0.9 - 0.5) / (8192 - 256)
    mu = initial_target_tokens.shape[1] * slope + (0.5 - slope * 256)

    def create_scheduler():
        scheduler = diffusers.FlowMatchEulerDiscreteScheduler(
            base_image_seq_len=256,
            base_shift=0.5,
            max_image_seq_len=8192,
            max_shift=0.9,
            shift=1.0,
            shift_terminal=0.02,
            time_shift_type="exponential",
            use_dynamic_shifting=True,
        )
        scheduler.set_timesteps(steps, sigmas=sigmas, mu=mu)
        return scheduler

    # Independent reference: real image -> reduced diffusers VAE -> packed
    # source conditioning -> three reduced diffusers transformer/scheduler steps
    # -> reduced diffusers VAE decode.
    reference_scheduler = create_scheduler()
    reference_target_tokens = initial_target_tokens.clone()
    reference_first_noise = None
    with torch.no_grad():
        for timestep in reference_scheduler.timesteps:
            reference_noise = hf_transformer(
                hidden_states=torch.cat(
                    [reference_target_tokens, reference_source_tokens], dim=1
                ),
                timestep=timestep.expand(1).to(reference_target_tokens.dtype) / 1000,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_mask=prompt_mask,
                img_shapes=[image_shapes],
            ).sample[:, : reference_target_tokens.shape[1]]
            if reference_first_noise is None:
                reference_first_noise = reference_noise.clone()
            reference_target_tokens = reference_scheduler.step(
                reference_noise,
                timestep,
                reference_target_tokens,
                return_dict=False,
            )[0]
        reference_final_latents = unpack_latents(reference_target_tokens)
        reference_final_image = hf_vae.decode(
            reference_final_latents * std_tensor + mean_tensor
        ).sample.clip(-1.0, 1.0)

    np.testing.assert_allclose(
        reference_first_noise.numpy().reshape(-1),
        golden["l4_noise_pred"],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        reference_target_tokens.numpy().reshape(-1),
        golden["l5_final_latents"],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        reference_final_image.numpy().reshape(-1),
        golden["l5_final_image"],
        rtol=1e-6,
        atol=1e-6,
    )

    with tempfile.TemporaryDirectory() as directory:
        transformer_path = os.path.join(directory, "transformer.onnx")
        encoder_path = os.path.join(directory, "vae_encoder.onnx")
        decoder_path = os.path.join(directory, "vae_decoder.onnx")
        ir.save(transformer_model, transformer_path)
        ir.save(vae_package["encoder"], encoder_path)
        ir.save(vae_package["decoder"], decoder_path)
        transformer_session = ort.InferenceSession(
            transformer_path, providers=["CPUExecutionProvider"]
        )
        encoder_session = ort.InferenceSession(
            encoder_path, providers=["CPUExecutionProvider"]
        )
        decoder_session = ort.InferenceSession(
            decoder_path, providers=["CPUExecutionProvider"]
        )
        onnx_source_latents = encoder_session.run(None, {"sample": source_pixels})[0]
        np.testing.assert_allclose(
            onnx_source_latents,
            reference_source_latents.numpy(),
            rtol=1e-4,
            atol=1e-4,
        )
        onnx_source_tokens = pack_numpy_latents(onnx_source_latents)
        np.testing.assert_allclose(
            onnx_source_tokens,
            reference_source_tokens.numpy(),
            rtol=1e-4,
            atol=1e-4,
        )
        onnx_target_tokens = pack_numpy_latents(initial_target_latents.numpy())
        onnx_scheduler = create_scheduler()
        onnx_first_noise = None
        for timestep in onnx_scheduler.timesteps.numpy():
            feeds = {
                "sample": np.concatenate([onnx_target_tokens, onnx_source_tokens], axis=1),
                "timestep": np.array([timestep / 1000], dtype=np.float32),
                "encoder_hidden_states": prompt_embeds.numpy(),
                "encoder_hidden_states_mask": prompt_mask.numpy(),
                "image_rotary_cos": image_cos,
                "image_rotary_sin": image_sin,
                "text_rotary_cos": text_cos,
                "text_rotary_sin": text_sin,
                "target_sequence_length": np.array(
                    [onnx_target_tokens.shape[1]], dtype=np.int64
                ),
            }
            onnx_noise = transformer_session.run(None, feeds)[0]
            if onnx_first_noise is None:
                onnx_first_noise = onnx_noise.copy()
            onnx_target_tokens = onnx_scheduler.step(
                torch.from_numpy(onnx_noise),
                torch.tensor(timestep),
                torch.from_numpy(onnx_target_tokens),
                return_dict=False,
            )[0].numpy()
        onnx_final_latents = unpack_numpy_latents(onnx_target_tokens)
        onnx_final_image = decoder_session.run(None, {"latent_sample": onnx_final_latents})[0]

    np.testing.assert_allclose(
        onnx_first_noise,
        reference_first_noise.numpy(),
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        onnx_target_tokens,
        reference_target_tokens.numpy(),
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        onnx_final_image,
        reference_final_image.numpy(),
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("dtype", "torch_dtype", "numpy_dtype"),
    [
        (ir.DataType.FLOAT16, "float16", np.float16),
        (ir.DataType.BFLOAT16, "bfloat16", None),
    ],
)
def test_cuda_low_precision_matches_diffusers(dtype, torch_dtype, numpy_dtype):
    ort = pytest.importorskip("onnxruntime")
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    if (
        not torch.cuda.is_available()
        or "CUDAExecutionProvider" not in ort.get_available_providers()
    ):
        pytest.skip("CUDA is not available")

    from mobius import build_from_module

    torch.manual_seed(23)
    torch_type = getattr(torch, torch_dtype)
    hf_model = diffusers.QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=16,
        out_channels=4,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=8,
        axes_dims_rope=(2, 2, 4),
    ).to(device="cuda", dtype=torch_type)
    config = QwenImageConfig(
        in_channels=16,
        out_channels=4,
        patch_size=2,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=8,
        cross_attention_dim=8,
        axes_dims_rope=(2, 2, 4),
        dtype=dtype,
    )
    module = QwenImageTransformer2DModel(config)
    model = build_from_module(
        module,
        config,
        task="qwen-image-denoising",
        execution_provider="cuda",
    )["model"]
    weights = {
        name: value.to(torch_type).cpu() for name, value in hf_model.state_dict().items()
    }
    apply_weights(model, module.preprocess_weights(weights))

    generator = torch.Generator().manual_seed(29)
    sample = torch.randn((1, 8, 16), generator=generator)
    prompt_embeds = torch.randn((1, 3, 8), generator=generator)
    prompt_mask = torch.tensor([[True, False, True]])
    timestep = torch.tensor([0.5])
    image_shapes = [(1, 2, 2), (1, 2, 2)]
    with torch.no_grad():
        expected = (
            hf_model(
                hidden_states=sample.to(device="cuda", dtype=torch_type),
                timestep=timestep.to(device="cuda", dtype=torch_type),
                encoder_hidden_states=prompt_embeds.to(device="cuda", dtype=torch_type),
                encoder_hidden_states_mask=prompt_mask.to("cuda"),
                img_shapes=[image_shapes],
            )
            .sample[:, :4]
            .float()
            .cpu()
            .numpy()
        )
    image_cos, image_sin, text_cos, text_sin = prepare_qwen_image_rotary_embeddings(
        image_shapes, 3, (2, 2, 4)
    )
    values = {
        "sample": sample.numpy(),
        "timestep": timestep.numpy(),
        "encoder_hidden_states": prompt_embeds.numpy(),
        "encoder_hidden_states_mask": prompt_mask.numpy(),
        "image_rotary_cos": image_cos,
        "image_rotary_sin": image_sin,
        "text_rotary_cos": text_cos,
        "text_rotary_sin": text_sin,
        "target_sequence_length": np.array([4], dtype=np.int64),
    }

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "model.onnx")
        ir.save(model, path)
        session = ort.InferenceSession(path, providers=["CUDAExecutionProvider"])
        if session.get_providers()[0] != "CUDAExecutionProvider":
            pytest.skip("CUDAExecutionProvider could not initialize on this host")
        if numpy_dtype is not None:
            feeds = {
                name: (
                    value.astype(numpy_dtype)
                    if name in {"sample", "timestep", "encoder_hidden_states"}
                    else value
                )
                for name, value in values.items()
            }
            actual = session.run(None, feeds)[0].astype(np.float32)
        else:
            feeds = {}
            for name, value in values.items():
                if name in {"sample", "timestep", "encoder_hidden_states"}:
                    tensor = torch.from_numpy(value).to(device="cuda", dtype=torch.bfloat16)
                    feeds[name] = ort.OrtValue.from_dlpack(tensor)
                else:
                    feeds[name] = ort.OrtValue.ortvalue_from_numpy(value)
            output = session.run_with_ort_values(["noise_pred"], feeds)[0]
            actual = torch.from_dlpack(output).float().cpu().numpy()

    np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-2)
