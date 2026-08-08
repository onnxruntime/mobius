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

from mobius._diffusers_configs import QwenImageConfig
from mobius._weight_loading import apply_weights
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

    from mobius._diffusers_configs import QwenImageVAEConfig
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
    from mobius._diffusers_configs import QwenImageVAEConfig
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


@pytest.mark.integration
@pytest.mark.integration_fast
def test_edit_vae_matches_diffusers_on_real_source_image():
    ort = pytest.importorskip("onnxruntime")
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    image_module = pytest.importorskip("PIL.Image")

    from mobius._diffusers_configs import QwenImageVAEConfig
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


@pytest.mark.integration
@pytest.mark.integration_fast
def test_deterministic_l4_l5_image_edit_golden():
    ort = pytest.importorskip("onnxruntime")
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    image_module = pytest.importorskip("PIL.Image")

    golden_path = Path("testdata/golden/diffusion/qwen-image-edit-2509.json")
    with open(golden_path, encoding="utf-8") as golden_file:
        golden = json.load(golden_file)

    torch.manual_seed(golden["seed"])
    hf_model = diffusers.QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=16,
        out_channels=4,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=8,
        axes_dims_rope=(2, 2, 4),
    ).eval()
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
    )
    module = QwenImageTransformer2DModel(config)
    model = QwenImageDenoisingTask().build(module, config)["model"]
    apply_weights(model, module.preprocess_weights(dict(hf_model.state_dict())))

    rng = np.random.default_rng(golden["seed"])
    latents = rng.normal(size=(1, 4, 16)).astype(np.float32)
    image = image_module.open(Path("testdata") / golden["source_image"]).convert("RGB")
    rgb = np.asarray(image.resize((2, 2)), dtype=np.float32).reshape(4, 3) / 127.5 - 1.0
    luminance = rgb.mean(axis=1, keepdims=True)
    source_latents = np.concatenate(
        [
            rgb,
            luminance,
            rgb**2,
            np.sin(rgb),
            np.cos(rgb),
            rgb[:, 0:1] * rgb[:, 1:2],
            rgb[:, 1:2] * rgb[:, 2:3],
            luminance**2,
        ],
        axis=1,
    )[None].astype(np.float32)
    prompt_embeds = rng.normal(size=(1, 3, 8)).astype(np.float32)
    prompt_mask = np.array([[True, False, True]])
    image_shapes = [(1, 2, 2), (1, 2, 2)]
    image_cos, image_sin, text_cos, text_sin = prepare_qwen_image_rotary_embeddings(
        image_shapes, 3, (2, 2, 4)
    )

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
    steps = golden["num_inference_steps"]
    sigmas = np.linspace(1.0, 1 / steps, steps)
    slope = (0.9 - 0.5) / (8192 - 256)
    mu = 4 * slope + (0.5 - slope * 256)
    scheduler.set_timesteps(steps, sigmas=sigmas, mu=mu)

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "model.onnx")
        ir.save(model, path)
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        first_noise = None
        for timestep in scheduler.timesteps.numpy():
            feeds = {
                "sample": np.concatenate([latents, source_latents], axis=1),
                "timestep": np.array([timestep / 1000], dtype=np.float32),
                "encoder_hidden_states": prompt_embeds,
                "encoder_hidden_states_mask": prompt_mask,
                "image_rotary_cos": image_cos,
                "image_rotary_sin": image_sin,
                "text_rotary_cos": text_cos,
                "text_rotary_sin": text_sin,
                "target_sequence_length": np.array([4], dtype=np.int64),
            }
            noise_pred = session.run(None, feeds)[0]
            if first_noise is None:
                first_noise = noise_pred.copy()
            latents = scheduler.step(
                torch.from_numpy(noise_pred),
                torch.tensor(timestep),
                torch.from_numpy(latents),
                return_dict=False,
            )[0].numpy()

    np.testing.assert_allclose(
        first_noise.reshape(-1), golden["l4_noise_pred"], rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        latents.reshape(-1), golden["l5_final_latents"], rtol=1e-4, atol=1e-4
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
