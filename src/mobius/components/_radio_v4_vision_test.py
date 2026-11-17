# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tiny independent PyTorch parity for C-RADIOv4, without downloads.

Architecture and patch/CPE behavior were checked against ``modeling_radio.py``
and ``configuration_radio.py`` at revision
``c6cc2e8ffa353403213d6b9b18aaf8eb684d705a``. Reference parameters originate in
PyTorch, not in ONNX initializers. MultiheadAttention independently implements
the source's biased, bidirectional scaled-dot-product attention.
"""

from __future__ import annotations

import json
import pathlib
import uuid

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch
from torch import nn
from torch.nn import functional as torch_functional

from mobius._builder import _cast_module_dtype
from mobius._testing import create_test_builder, create_test_input
from mobius._testing.ort_inference import _numpy_to_ort_value, _ort_value_to_numpy
from mobius.components import RadioV4VisionModel
from mobius.components._radio_vision import RadioVisionModel


class _TorchBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(32, eps=1e-6)
        self.attn = nn.MultiheadAttention(32, 4, dropout=0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(32, eps=1e-6)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 32)
        with torch.no_grad():
            for norm in (self.norm1, self.norm2):
                norm.weight.uniform_(0.8, 1.2)
                norm.bias.uniform_(-0.1, 0.1)
            self.attn.in_proj_bias.normal_(0, 0.05)
            self.attn.out_proj.bias.normal_(0, 0.05)

    def forward(self, x):
        norm = self.norm1(x)
        x = x + self.attn(norm, norm, norm, need_weights=False)[0]
        return x + self.fc2(torch_functional.gelu(self.fc1(self.norm2(x)), approximate="none"))


class _TorchRadioV4(nn.Module):
    """Independently specified two-layer, patch-2, 4x4-CPE-grid fixture."""

    def __init__(self):
        super().__init__()
        self.image_projection = nn.Linear(12, 32, bias=False)
        self.video_projection = nn.Linear(24, 32, bias=False)
        self.position = nn.Parameter(torch.randn(1, 16, 32) * 0.1)
        self.prefix = nn.Parameter(torch.randn(10, 32) * 0.1)
        self.layers = nn.ModuleList([_TorchBlock(), _TorchBlock()])

    def forward(self, pixels, video=False, summary_idxs=(0, 1)):
        batch, channels, height, width = pixels.shape
        rows, cols = height // 2, width // 2
        # HF's patch order is row, col, packed-channel, patch-row, patch-col.
        patches = pixels.reshape(batch, channels, rows, 2, cols, 2)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(batch, rows * cols, -1)
        patches = (self.video_projection if video else self.image_projection)(patches)
        position = self.position.reshape(1, 4, 4, 32).permute(0, 3, 1, 2)
        position = torch_functional.interpolate(
            position.float(), (max(rows, cols),) * 2, mode="bilinear", align_corners=False
        ).to(patches.dtype)
        position = position[:, :, :rows, :cols].flatten(2).transpose(1, 2)
        x = torch.cat([self.prefix.unsqueeze(0).expand(batch, -1, -1), patches + position], 1)
        for layer in self.layers:
            x = layer(x)
        return x[:, summary_idxs].flatten(1), x[:, 10:]

    def checkpoint_state(self):
        # Explicit release-header -> independent reference mapping, including
        # fused QKV. No graph-derived parameter shapes or names generate weights.
        state = {
            "patch_generator.embedder.weight": self.image_projection.weight,
            "patch_generator.video_embedder.weight": self.video_projection.weight,
            "patch_generator.pos_embed": self.position,
            "patch_generator.cls_token.token": self.prefix,
        }
        for index, block in enumerate(self.layers):
            prefix = f"blocks.{index}"
            state.update(
                {
                    f"{prefix}.norm1.weight": block.norm1.weight,
                    f"{prefix}.norm1.bias": block.norm1.bias,
                    f"{prefix}.attn.qkv.weight": block.attn.in_proj_weight,
                    f"{prefix}.attn.qkv.bias": block.attn.in_proj_bias,
                    f"{prefix}.attn.proj.weight": block.attn.out_proj.weight,
                    f"{prefix}.attn.proj.bias": block.attn.out_proj.bias,
                    f"{prefix}.norm2.weight": block.norm2.weight,
                    f"{prefix}.norm2.bias": block.norm2.bias,
                    f"{prefix}.mlp.fc1.weight": block.fc1.weight,
                    f"{prefix}.mlp.fc1.bias": block.fc1.bias,
                    f"{prefix}.mlp.fc2.weight": block.fc2.weight,
                    f"{prefix}.mlp.fc2.bias": block.fc2.bias,
                }
            )
        return {name: value.detach().cpu().numpy() for name, value in state.items()}


def _component(**overrides):
    kwargs = dict(
        patch_size=2,
        max_grid_size=4,
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_heads=4,
        video_temporal_patch_size=2,
    )
    kwargs.update(overrides)
    return RadioV4VisionModel(**kwargs)


@pytest.fixture
def cuda_provider():
    if (
        "CUDAExecutionProvider" not in ort.get_available_providers()
        or not torch.cuda.is_available()
    ):
        pytest.skip("CUDA PyTorch and ONNX Runtime are required for CUDA parity")
    return "CUDAExecutionProvider"


@pytest.fixture(params=["CPUExecutionProvider", "CUDAExecutionProvider"])
def provider(request):
    if request.param == "CUDAExecutionProvider":
        return request.getfixturevalue("cuda_provider")
    return request.param


def _session(component, state, provider, *, video=False, dtype=ir.DataType.FLOAT):
    parameters = dict(component.named_parameters())
    assert set(parameters) == set(state)
    for name, value in state.items():
        assert tuple(parameters[name].shape) == value.shape
        parameters[name].const_value = ir.tensor(value)
    _cast_module_dtype(component, dtype)
    builder, op, graph = create_test_builder()
    pixels = create_test_input(
        builder, "pixel_values", ["batch", 6 if video else 3, "height", "width"], dtype
    )
    summary, features = component(op, pixels, video)
    for name, value, dimensions in (
        ("summary", summary, ["batch", len(component._summary_idxs) * 32]),
        ("features", features, ["batch", "patches", 32]),
    ):
        value.name = name
        value.type = ir.TensorType(dtype)
        value.shape = ir.Shape(dimensions)
        graph.outputs.append(value)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    if provider == "CUDAExecutionProvider":
        options.enable_profiling = True
        options.profile_file_prefix = str(
            pathlib.Path.cwd() / f".radio-v4-profile-{uuid.uuid4().hex}"
        )
    # Only ONNX IR handles serialization; scratch files stay in the worktree
    # and are removed even when session creation fails.
    path = pathlib.Path.cwd() / f".radio-v4-{uuid.uuid4().hex}.onnx"
    try:
        ir.save(ir.Model(graph, ir_version=11), path)
        session = ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=[(provider, {"use_tf32": "0"})] if "CUDA" in provider else [provider],
        )
    finally:
        path.unlink(missing_ok=True)
    assert session.get_providers()[0] == provider
    return session


def _assert_cuda_execution(session):
    profile = pathlib.Path(session.end_profiling())
    try:
        events = json.loads(profile.read_text())
    finally:
        profile.unlink(missing_ok=True)
    kernels = [event for event in events if "provider" in event.get("args", {})]
    gpu_ops = {
        event["args"]["op_name"]
        for event in kernels
        if event["args"]["provider"] == "CUDAExecutionProvider"
    }
    assert {"Attention", "MatMul", "Resize"} <= gpu_ops
    assert gpu_ops & {"LayerNormalization", "SkipLayerNormalization"}
    # ORT intentionally keeps shape arithmetic on CPU; unlike a session-level
    # fallback ban, this gate permits only those nodes, never backbone compute.
    for event in kernels:
        if event["args"]["provider"] == "CPUExecutionProvider":
            assert event["name"].startswith("patch_generator/")
            assert event["args"]["op_name"] in {"Div", "ReduceMax", "Concat", "Slice"}


@pytest.mark.parametrize("video", [False, True], ids=["image", "packed-video"])
def test_dynamic_rectangular_parity(provider, video):
    torch.manual_seed(752)
    reference = _TorchRadioV4().eval()
    session = _session(_component(), reference.checkpoint_state(), provider, video=video)
    device = "cuda" if provider == "CUDAExecutionProvider" else "cpu"
    reference.to(device)
    rng = np.random.default_rng(942)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        # One exported graph covers square, portrait, landscape, learned-grid
        # size, upsampling beyond that grid, and singleton output dimensions.
        for batch, height, width in [
            (1, 4, 6),
            (2, 6, 4),
            (1, 8, 6),
            (2, 10, 6),
            (1, 2, 2),
            (2, 2, 6),
            (1, 6, 6),
        ]:
            pixels = rng.normal(size=(batch, 6 if video else 3, height, width)).astype(
                np.float32
            )
            with torch.inference_mode():
                expected = reference(torch.from_numpy(pixels).to(device), video)
            actual = session.run(None, {"pixel_values": pixels})
            assert actual[0].shape == (batch, 64)
            assert actual[1].shape == (batch, height * width // 4, 32)
            for result, target in zip(actual, expected):
                np.testing.assert_allclose(
                    result, target.cpu().numpy(), rtol=1e-4, atol=2e-6, strict=True
                )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        if provider == "CUDAExecutionProvider":
            _assert_cuda_execution(session)


def test_summary_selection_and_spatial_prefix_boundary():
    torch.manual_seed(753)
    reference = _TorchRadioV4().eval()
    component = _component(summary_idxs=(2, 0, 2))
    session = _session(component, reference.checkpoint_state(), "CPUExecutionProvider")
    pixels = np.arange(72, dtype=np.float32).reshape(1, 3, 4, 6) / 72
    with torch.inference_mode():
        expected = reference(torch.from_numpy(pixels), summary_idxs=(2, 0, 2))
    actual = session.run(None, {"pixel_values": pixels})
    assert actual[0].shape == (1, 96)
    assert actual[1].shape == (1, 6, 32)
    np.testing.assert_array_equal(actual[0][:, :32], actual[0][:, 64:])
    for result, target in zip(actual, expected):
        np.testing.assert_allclose(result, target.numpy(), rtol=1e-4, atol=2e-6)


@pytest.mark.parametrize("video", [False, True], ids=["image", "packed-video"])
@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
def test_reduced_precision_cuda_parity(cuda_provider, dtype, video):
    torch.manual_seed(756)
    reference = _TorchRadioV4().eval()
    session = _session(
        _component(), reference.checkpoint_state(), cuda_provider, video=video, dtype=dtype
    )
    torch_dtype = torch.float16 if dtype == ir.DataType.FLOAT16 else torch.bfloat16
    reference.to(device="cuda", dtype=torch_dtype)
    rng = np.random.default_rng(943)
    try:
        for height, width in [(4, 6), (10, 6), (2, 2)]:
            pixels = rng.normal(size=(2, 6 if video else 3, height, width)).astype(
                dtype.numpy()
            )
            with torch.inference_mode():
                expected = reference(
                    torch.from_numpy(pixels.astype(np.float32)).to(
                        device="cuda", dtype=torch_dtype
                    ),
                    video,
                )
            actual = session.run_with_ort_values(
                None, {"pixel_values": _numpy_to_ort_value(pixels)}
            )
            for result, target in zip(actual, expected):
                np.testing.assert_allclose(
                    _ort_value_to_numpy(result).astype(np.float32),
                    target.float().cpu().numpy(),
                    rtol=1e-2,
                    atol=2e-2 if dtype == ir.DataType.BFLOAT16 else 2e-3,
                )
    finally:
        _assert_cuda_execution(session)


def test_release_header_parameter_contract_and_v2_unchanged():
    component = RadioV4VisionModel(
        patch_size=16,
        max_grid_size=128,
        hidden_size=1280,
        intermediate_size=5120,
        num_layers=32,
        num_heads=16,
        video_temporal_patch_size=2,
    )
    parameters = dict(component.named_parameters())
    assert list(parameters["patch_generator.embedder.weight"].shape) == [1280, 768]
    assert list(parameters["patch_generator.video_embedder.weight"].shape) == [1280, 1536]
    assert list(parameters["patch_generator.pos_embed"].shape) == [1, 16384, 1280]
    assert list(parameters["patch_generator.cls_token.token"].shape) == [10, 1280]
    assert list(parameters["blocks.0.attn.qkv.weight"].shape) == [3840, 1280]
    assert list(parameters["blocks.31.mlp.fc2.weight"].shape) == [1280, 5120]
    assert len(parameters) == 4 + 32 * 12
    assert not any("layer_scale" in name or "final" in name for name in parameters)

    v2 = RadioVisionModel(
        image_height=4,
        image_width=8,
        patch_size=2,
        max_grid_size=4,
        hidden_size=32,
        intermediate_size=64,
        num_layers=1,
        num_heads=4,
    )
    assert list(dict(v2.named_parameters())["patch_generator.cls_token.token"].shape) == [
        8,
        32,
    ]
    with pytest.raises(ValueError, match="longest patch-grid"):
        RadioVisionModel(
            image_height=4,
            image_width=6,
            patch_size=2,
            max_grid_size=4,
            hidden_size=32,
            intermediate_size=64,
            num_layers=1,
            num_heads=4,
        )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"hidden_size": 33}, "divisible"),
        ({"patch_size": 0}, "positive"),
        ({"max_grid_size": 0}, "positive"),
        ({"num_layers": -1}, "Invalid"),
        ({"num_cls_tokens": 0}, "Invalid"),
        ({"num_registers": -1}, "Invalid"),
        ({"norm_eps": 0}, "Invalid"),
        ({"summary_idxs": ()}, "summary_idxs"),
        ({"summary_idxs": (3,)}, "summary_idxs"),
        ({"summary_idxs": (-1,)}, "summary_idxs"),
        ({"video_temporal_patch_size": 0}, "positive"),
    ],
)
def test_invalid_constructor(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _component(**kwargs)


@pytest.mark.parametrize("shape", [[1, 3, 3, 4], [1, 3, 4, 0], [1, 6, 4, 4], [3, 4, 4]])
def test_invalid_static_image_shape(shape):
    builder, op, _ = create_test_builder()
    pixels = create_test_input(builder, "pixel_values", shape)
    with pytest.raises(ValueError, match="RADIOv4"):
        _component()(op, pixels)


def test_video_projection_must_be_enabled_and_export_time():
    builder, op, _ = create_test_builder()
    pixels = create_test_input(builder, "pixel_values", [1, 6, 4, 4])
    with pytest.raises(ValueError, match="video_temporal_patch_size"):
        _component(video_temporal_patch_size=None)(op, pixels, True)
    with pytest.raises(TypeError, match="export-time bool"):
        _component()(op, pixels, 1)


def test_dynamic_invalid_shape_does_not_drop_pixels():
    torch.manual_seed(754)
    reference = _TorchRadioV4().eval()
    session = _session(_component(), reference.checkpoint_state(), "CPUExecutionProvider")
    for shape in [(1, 3, 5, 6), (1, 6, 4, 6)]:
        with pytest.raises(
            (
                ort.capi.onnxruntime_pybind11_state.Fail,
                ort.capi.onnxruntime_pybind11_state.InvalidArgument,
            ),
            match=r"reshap|dimensions",
        ):
            session.run(None, {"pixel_values": np.zeros(shape, dtype=np.float32)})
