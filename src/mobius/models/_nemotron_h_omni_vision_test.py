# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Independent PyTorch oracles for Omni shuffle, projection, video packing and EVS.

No source checkout, production weights, or remote reference dependencies are
needed. Source semantics are pinned to c6cc2e8ffa353403213d6b9b18aaf8eb684d705a.
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
from onnxscript import nn as onnx_nn
from torch import nn
from torch.nn import functional as torch_functional

from mobius._builder import _cast_module_dtype
from mobius._testing import create_test_builder, create_test_input
from mobius._testing.ort_inference import _numpy_to_ort_value, _ort_value_to_numpy
from mobius.components import RadioV4VisionModel
from mobius.models._nemotron_h_omni_vision import (
    NemotronHOmniVisionProjector,
    build_omni_image_features,
    build_omni_video_features,
    compact_omni_video_features,
    pack_omni_video_frames,
)


class _TorchProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.linspace(0.6, 1.4, 32))
        self.linear1 = nn.Linear(32, 16, bias=False)
        self.linear2 = nn.Linear(16, 6, bias=False)

    def forward(self, features, rows, cols):
        batch = features.shape[0]
        # Independent block-gather formulation; unlike the ONNX implementation,
        # this never swaps the full spatial axes during the shuffle.
        grid = features.reshape(batch, rows // 2, 2, cols // 2, 2, 8)
        merged = grid.permute(0, 1, 3, 2, 4, 5).reshape(batch, (rows // 2) * (cols // 2), 32)
        normalized = merged.float()
        normalized = normalized * torch.rsqrt(
            normalized.square().mean(-1, keepdim=True) + 1e-5
        )
        normalized = (normalized * self.gamma.float()).to(features.dtype)
        return self.linear2(torch_functional.relu(self.linear1(normalized)).square())

    def checkpoint_state(self):
        return {
            "mlp1.0.weight": self.gamma.detach().float().cpu().numpy(),
            "mlp1.1.weight": self.linear1.weight.detach().float().cpu().numpy(),
            "mlp1.3.weight": self.linear2.weight.detach().float().cpu().numpy(),
        }


class _TorchVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.image = nn.Linear(12, 8, bias=False)
        self.video = nn.Linear(24, 8, bias=False)
        self.position = nn.Parameter(torch.randn(1, 16, 8) * 0.1)
        self.prefix = nn.Parameter(torch.randn(10, 8) * 0.1)
        self.layer = nn.TransformerEncoderLayer(
            d_model=8,
            nhead=2,
            dim_feedforward=16,
            dropout=0,
            activation="gelu",
            layer_norm_eps=1e-6,
            batch_first=True,
            norm_first=True,
        )

    def forward(self, pixels, video=False):
        rows, cols = pixels.shape[-2] // 2, pixels.shape[-1] // 2
        patches = torch_functional.unfold(pixels, kernel_size=2, stride=2).transpose(1, 2)
        patches = (self.video if video else self.image)(patches)
        position = self.position.reshape(1, 4, 4, 8).permute(0, 3, 1, 2)
        position = torch_functional.interpolate(
            position.float(), (max(rows, cols),) * 2, mode="bilinear", align_corners=False
        ).to(patches.dtype)
        position = position[:, :, :rows, :cols].flatten(2).transpose(1, 2)
        prefix = self.prefix.unsqueeze(0).expand(pixels.shape[0], -1, -1)
        return self.layer(torch.cat([prefix, patches + position], dim=1))[:, 10:]

    def checkpoint_state(self):
        layer = self.layer
        state = {
            "patch_generator.embedder.weight": self.image.weight,
            "patch_generator.video_embedder.weight": self.video.weight,
            "patch_generator.pos_embed": self.position,
            "patch_generator.cls_token.token": self.prefix,
            "blocks.0.norm1.weight": layer.norm1.weight,
            "blocks.0.norm1.bias": layer.norm1.bias,
            "blocks.0.norm2.weight": layer.norm2.weight,
            "blocks.0.norm2.bias": layer.norm2.bias,
            "blocks.0.attn.qkv.weight": layer.self_attn.in_proj_weight,
            "blocks.0.attn.qkv.bias": layer.self_attn.in_proj_bias,
            "blocks.0.attn.proj.weight": layer.self_attn.out_proj.weight,
            "blocks.0.attn.proj.bias": layer.self_attn.out_proj.bias,
            "blocks.0.mlp.fc1.weight": layer.linear1.weight,
            "blocks.0.mlp.fc1.bias": layer.linear1.bias,
            "blocks.0.mlp.fc2.weight": layer.linear2.weight,
            "blocks.0.mlp.fc2.bias": layer.linear2.bias,
        }
        return {name: value.detach().float().cpu().numpy() for name, value in state.items()}


def _torch_pack(pixels):
    if pixels.shape[0] % 2:
        pixels = torch.cat([pixels, pixels[-1:]], 0)
    return pixels.reshape(pixels.shape[0] // 2, 6, *pixels.shape[-2:])


def _torch_evs(features, q):
    temporal, patches, hidden = features.shape
    similarity = torch_functional.cosine_similarity(features[1:], features[:-1], dim=-1)
    scores = torch.cat([torch.full_like(features[:1, :, 0], 255), 1 - similarity], 0).flatten()
    count = max(patches, int(temporal * patches * (1 - q)))
    selected = torch.argsort(scores, descending=True, stable=True)[:count]
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask[selected] = True
    indices = mask.nonzero().flatten()
    return features.reshape(temporal * patches, hidden)[mask], mask, indices


class _Pipeline(onnx_nn.Module):
    def __init__(self, video, q=0.5):
        super().__init__()
        self.video = video
        self.q = q
        self.vision_model = RadioV4VisionModel(
            patch_size=2,
            max_grid_size=4,
            hidden_size=8,
            intermediate_size=16,
            num_layers=1,
            num_heads=2,
            video_temporal_patch_size=2,
        )
        self.vision_projector = NemotronHOmniVisionProjector(8, 16, 6)

    def forward(self, op, pixels):
        kwargs = dict(
            vision_model=self.vision_model,
            projector=self.vision_projector,
            patch_size=2,
        )
        if self.video:
            return build_omni_video_features(op, pixels, **kwargs, pruning_rate=self.q)
        return build_omni_image_features(op, pixels, **kwargs)


@pytest.fixture(autouse=True)
def torch_eager_attention():
    fastpath = torch.backends.mha.get_fastpath_enabled()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            yield
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
        torch.backends.cuda.matmul.allow_tf32 = tf32


@pytest.fixture
def cuda_provider():
    if (
        "CUDAExecutionProvider" not in ort.get_available_providers()
        or not torch.cuda.is_available()
    ):
        pytest.skip("CUDA PyTorch and ONNX Runtime are required")
    return "CUDAExecutionProvider"


@pytest.fixture(params=["CPUExecutionProvider", "CUDAExecutionProvider"])
def provider(request):
    if request.param == "CUDAExecutionProvider":
        return request.getfixturevalue("cuda_provider")
    return request.param


@pytest.fixture(
    params=[
        ("CPUExecutionProvider", ir.DataType.FLOAT),
        ("CUDAExecutionProvider", ir.DataType.FLOAT),
        ("CUDAExecutionProvider", ir.DataType.FLOAT16),
        ("CUDAExecutionProvider", ir.DataType.BFLOAT16),
    ]
)
def execution(request):
    provider, dtype = request.param
    if provider == "CUDAExecutionProvider":
        request.getfixturevalue("cuda_provider")
    return provider, dtype


def _session(
    build,
    inputs,
    outputs,
    provider,
    *,
    module=None,
    state=None,
    dtype=ir.DataType.FLOAT,
    profile=False,
):
    if module is not None:
        parameters = dict(module.named_parameters())
        assert set(parameters) == set(state)
        for name, value in state.items():
            assert tuple(parameters[name].shape) == value.shape
            parameters[name].const_value = ir.tensor(value)
        _cast_module_dtype(module, dtype)
    builder, op, graph = create_test_builder()
    values = [
        create_test_input(builder, name, shape, input_dtype)
        for name, shape, input_dtype in inputs
    ]
    results = build(op, *values)
    if not isinstance(results, tuple):
        results = (results,)
    for value, (name, shape, output_dtype) in zip(results, outputs):
        value.name = name
        value.type = ir.TensorType(output_dtype)
        value.shape = ir.Shape(shape)
        graph.outputs.append(value)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    path = pathlib.Path.cwd() / f".omni-vision-{uuid.uuid4().hex}.onnx"
    if profile:
        options.enable_profiling = True
        options.profile_file_prefix = str(path.with_suffix(""))
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


def _run(session, feeds):
    return [
        _ort_value_to_numpy(value)
        for value in session.run_with_ort_values(
            None, {name: _numpy_to_ort_value(value) for name, value in feeds.items()}
        )
    ]


def _torch_type(dtype):
    return {
        ir.DataType.FLOAT: torch.float32,
        ir.DataType.FLOAT16: torch.float16,
        ir.DataType.BFLOAT16: torch.bfloat16,
    }[dtype]


def _compare(actual, expected, dtype):
    np.testing.assert_allclose(
        actual.astype(np.float32),
        expected.detach().float().cpu().numpy(),
        rtol=1e-4 if dtype == ir.DataType.FLOAT else 1e-2,
        atol={ir.DataType.FLOAT: 2e-6, ir.DataType.FLOAT16: 2e-3, ir.DataType.BFLOAT16: 2e-2}[
            dtype
        ],
    )


def test_pixel_shuffle_v2_nonuniform_sentinels(provider):
    projector = NemotronHOmniVisionProjector(3, 5, 7)
    session = _session(
        projector.pixel_shuffle,
        [
            ("features", ["batch", "patches", 3], ir.DataType.FLOAT),
            ("grid", [2], ir.DataType.INT64),
        ],
        [("merged", ["batch", "merged", 12], ir.DataType.FLOAT)],
        provider,
    )
    for batch, rows, cols in [(2, 4, 6), (1, 6, 4), (2, 2, 2), (0, 4, 6)]:
        grid = np.fromfunction(
            lambda b, r, c, d: b * 1000 + r * 100 + c * 10 + d,
            (batch, rows, cols, 3),
        ).astype(np.float32)
        actual = _run(
            session,
            {"features": grid.reshape(batch, rows * cols, 3), "grid": np.array([rows, cols])},
        )[0]
        expected = np.stack(
            [
                grid[:, 0::2, 0::2],
                grid[:, 0::2, 1::2],
                grid[:, 1::2, 0::2],
                grid[:, 1::2, 1::2],
            ],
            axis=-2,
        ).reshape(batch, rows * cols // 4, 12)
        np.testing.assert_array_equal(actual, expected)
        if batch:
            np.testing.assert_array_equal(
                actual[0, 0], [0, 1, 2, 10, 11, 12, 100, 101, 102, 110, 111, 112]
            )


def test_projector_native_dtype_parity_and_empty_batch(execution):
    provider, dtype = execution
    torch.manual_seed(810)
    reference = _TorchProjector().eval()
    projector = NemotronHOmniVisionProjector(8, 16, 6)
    session = _session(
        projector,
        [("features", ["batch", "patches", 8], dtype), ("grid", [2], ir.DataType.INT64)],
        [("projected", ["batch", "patches_out", 6], dtype)],
        provider,
        module=projector,
        state=reference.checkpoint_state(),
        dtype=dtype,
    )
    device = "cuda" if "CUDA" in provider else "cpu"
    reference.to(device=device, dtype=_torch_type(dtype))
    rng = np.random.default_rng(811)
    for batch, rows, cols in [(2, 4, 6), (1, 6, 4), (2, 2, 2), (0, 4, 6)]:
        features = rng.normal(size=(batch, rows * cols, 8)).astype(dtype.numpy())
        with torch.inference_mode():
            expected = reference(
                torch.from_numpy(features.astype(np.float32)).to(
                    device=device, dtype=_torch_type(dtype)
                ),
                rows,
                cols,
            )
        actual = _run(session, {"features": features, "grid": np.array([rows, cols])})[0]
        assert actual.shape == (batch, rows * cols // 4, 6)
        _compare(actual, expected, dtype)


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
def test_projector_gamma_is_applied_before_native_rounding(cuda_provider, dtype):
    torch.manual_seed(817)
    reference = _TorchProjector()
    projector = NemotronHOmniVisionProjector(8, 16, 6)
    session = _session(
        lambda op, features: projector.mlp1[0](op, features),
        [("features", [2, 4, 32], dtype)],
        [("normalized", [2, 4, 32], dtype)],
        cuda_provider,
        module=projector,
        state=reference.checkpoint_state(),
        dtype=dtype,
    )
    native = _torch_type(dtype)
    values = torch.randn(2, 4, 32, device="cuda").to(native)
    gamma = reference.gamma.detach().to(device="cuda", dtype=native)
    normalized = values.float() * torch.rsqrt(
        values.float().square().mean(-1, keepdim=True) + 1e-5
    )
    expected = (normalized * gamma.float()).to(native)
    premature_round = normalized.to(native) * gamma
    assert torch.any(expected != premature_round)
    actual = _run(session, {"features": values.float().cpu().numpy().astype(dtype.numpy())})[0]
    np.testing.assert_array_equal(actual.astype(np.float32), expected.float().cpu().numpy())


def test_temporal_padding_and_channel_order(provider):
    session = _session(
        pack_omni_video_frames,
        [("pixels", ["frames", 3, 4, 6], ir.DataType.FLOAT)],
        [("packed", ["groups", 6, 4, 6], ir.DataType.FLOAT)],
        provider,
    )
    for count in [0, 1, 2, 3, 4, 7]:
        pixels = np.fromfunction(
            lambda t, c, h, w: 1000 * t + 100 * c + 10 * h + w,
            (count, 3, 4, 6),
        ).astype(np.float32)
        actual = _run(session, {"pixels": pixels})[0]
        expected = _torch_pack(torch.from_numpy(pixels)).numpy()
        np.testing.assert_array_equal(actual, expected)
        if count % 2:
            np.testing.assert_array_equal(actual[-1, :3], actual[-1, 3:])


@pytest.mark.parametrize("q", [0.0, 0.5, 0.7, 0.8, 1.0])
def test_evs_order_ties_first_frame_and_empty(execution, q):
    provider, dtype = execution
    session = _session(
        lambda op, features: compact_omni_video_features(op, features, q),
        [("features", ["frames", "patches", 3], dtype)],
        [
            ("compacted", ["kept", 3], dtype),
            ("retention_mask", ["total"], ir.DataType.BOOL),
            ("kept_indices", ["kept"], ir.DataType.INT64),
        ],
        provider,
    )
    cases = [
        np.array([[[1, 0, 0], [0, 1, 0]], [[-1, 0, 0], [0, 1, 0]], [[-1, 0, 0], [0, -1, 0]]]),
        np.ones((15, 2, 3)),
        np.ones((1, 4, 3)),
        np.ones((0, 2, 3)),
    ]
    if dtype != ir.DataType.FLOAT16:
        cases.append(np.zeros((3, 2, 3)))
    device = "cuda" if "CUDA" in provider else "cpu"
    for data in cases:
        features = data.astype(dtype.numpy())
        tensor = torch.from_numpy(features.astype(np.float32)).to(
            device=device, dtype=_torch_type(dtype)
        )
        expected = _torch_evs(tensor, q)
        actual = _run(session, {"features": features})
        _compare(actual[0], expected[0], dtype)
        np.testing.assert_array_equal(actual[1], expected[1].cpu().numpy())
        np.testing.assert_array_equal(actual[2], expected[2].cpu().numpy())
        np.testing.assert_array_equal(actual[0], features.reshape(-1, 3)[actual[1]])
        assert (
            np.all(actual[1][: features.shape[1]]) if features.shape[0] else not actual[1].size
        )
        if q in {0.8} and data.shape == (15, 2, 3):
            np.testing.assert_array_equal(actual[2], np.arange(5))


@pytest.mark.parametrize("video", [False, True], ids=["image", "video"])
def test_radio_projector_pipeline_numerical_parity(execution, video):
    provider, dtype = execution
    torch.manual_seed(812)
    vision = _TorchVision().eval()
    reference = _TorchProjector().eval()
    state = {
        **{f"vision_model.{name}": value for name, value in vision.checkpoint_state().items()},
        **{
            f"vision_projector.{name}": value
            for name, value in reference.checkpoint_state().items()
        },
    }
    pipeline = _Pipeline(video)
    outputs = (
        [
            ("image_features", ["kept", 6], dtype),
            ("retention_mask", ["total"], ir.DataType.BOOL),
            ("kept_indices", ["kept"], ir.DataType.INT64),
        ]
        if video
        else [("image_features", ["images", "patches", 6], dtype)]
    )
    session = _session(
        pipeline,
        [("pixels", ["media", 3, "height", "width"], dtype)],
        outputs,
        provider,
        module=pipeline,
        state=state,
        dtype=dtype,
    )
    device = "cuda" if "CUDA" in provider else "cpu"
    vision.to(device=device, dtype=_torch_type(dtype))
    reference.to(device=device, dtype=_torch_type(dtype))
    rng = np.random.default_rng(813)
    shapes = (
        [(1, 8, 8), (3, 8, 8), (5, 8, 8), (8, 12, 12)]
        if video
        else [(2, 8, 12), (1, 12, 8), (2, 4, 4)]
    )
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for count, height, width in shapes:
            pixels = rng.normal(size=(count, 3, height, width)).astype(dtype.numpy())
            tensor = torch.from_numpy(pixels.astype(np.float32)).to(
                device=device, dtype=_torch_type(dtype)
            )
            with torch.inference_mode():
                spatial = vision(_torch_pack(tensor) if video else tensor, video)
                projected = reference(spatial, height // 2, width // 2)
                expected = _torch_evs(projected, 0.5) if video else (projected,)
            actual = _run(session, {"pixels": pixels})
            _compare(actual[0], expected[0], dtype)
            if video:
                np.testing.assert_array_equal(actual[1], expected[1].cpu().numpy())
                np.testing.assert_array_equal(actual[2], expected[2].cpu().numpy())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


@pytest.mark.parametrize("q", [0.0, 0.5])
def test_source_wrapper_non_square_video_patch_count(provider, q):
    torch.manual_seed(818)
    vision = _TorchVision().eval()
    reference = _TorchProjector().eval()
    state = {
        **{f"vision_model.{name}": value for name, value in vision.checkpoint_state().items()},
        **{
            f"vision_projector.{name}": value
            for name, value in reference.checkpoint_state().items()
        },
    }
    pipeline = _Pipeline(True, q)
    session = _session(
        pipeline,
        [("pixels", ["media", 3, "height", "width"], ir.DataType.FLOAT)],
        [
            ("image_features", ["kept", 6], ir.DataType.FLOAT),
            ("retention_mask", ["total"], ir.DataType.BOOL),
            ("kept_indices", ["kept"], ir.DataType.INT64),
        ],
        provider,
        module=pipeline,
        state=state,
    )
    pixels = np.random.default_rng(819).normal(size=(3, 3, 8, 12)).astype(np.float32)
    if q > 0:
        with pytest.raises(ort.capi.onnxruntime_pybind11_state.Fail, match="reshap"):
            _run(session, {"pixels": pixels})
    else:
        actual = _run(session, {"pixels": pixels})
        assert actual[0].shape == (12, 6)
        np.testing.assert_array_equal(actual[1], np.ones(12, dtype=np.bool_))
        np.testing.assert_array_equal(actual[2], np.arange(12))


def test_retention_metadata_maps_distinct_streams_across_prompt_rows(provider):
    session = _session(
        lambda op, features: compact_omni_video_features(op, features, 0.5),
        [("features", [4, 2, 3], ir.DataType.FLOAT)],
        [
            ("image_features", ["kept", 3], ir.DataType.FLOAT),
            ("retention_mask", [8], ir.DataType.BOOL),
            ("kept_indices", ["kept"], ir.DataType.INT64),
        ],
        provider,
    )
    video = np.array(
        [
            [[10, 1, 0], [20, 0, 1]],
            [[-30, 1, 0], [40, 0, 1]],
            [[-50, 1, 0], [-60, 0, 1]],
            [[70, 1, 0], [80, 0, 1]],
        ],
        dtype=np.float32,
    )
    compacted, mask, indices = _run(session, {"features": video})
    # These are separate caller-owned streams, not an assertion that the source
    # wrapper provides distinct token IDs (it uses the image ID for video too).
    image_id, video_id = 77, 88
    ids = np.array(
        [
            [1, image_id, video_id, video_id, video_id, video_id, 2],
            [3, video_id, image_id, video_id, video_id, video_id, 4],
        ]
    )
    attention = np.arange(ids.size).reshape(ids.shape)
    token_mask = np.ones_like(ids, dtype=np.bool_)
    token_mask[ids == video_id] = mask
    embeddings = np.zeros((*ids.shape, 3), dtype=np.float32)
    images = np.array([[101, 102, 103], [201, 202, 203]], dtype=np.float32)
    embeddings[ids == image_id] = images
    embeddings[ids == video_id] = video.reshape(-1, 3)
    kept_embeddings = embeddings[token_mask]
    kept_ids = ids[token_mask]
    np.testing.assert_array_equal(kept_embeddings[kept_ids == image_id], images)
    np.testing.assert_array_equal(kept_embeddings[kept_ids == video_id], compacted)
    np.testing.assert_array_equal(np.flatnonzero(mask), indices)
    np.testing.assert_array_equal(attention[token_mask], np.flatnonzero(token_mask))
    # Caller must use one flattened media offset across rows, not restart row 2.
    assert token_mask[1, 1] == mask[4]


@pytest.mark.parametrize(
    "dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16]
)
def test_cuda_projector_and_evs_compute_is_on_gpu(cuda_provider, dtype):
    torch.manual_seed(814)
    reference = _TorchProjector()
    projector = NemotronHOmniVisionProjector(8, 16, 6)
    session = _session(
        lambda op, features: compact_omni_video_features(
            op, projector(op, features, op.Constant(value_ints=[4, 4])), 0.5
        ),
        [("features", ["frames", 16, 8], dtype)],
        [
            ("image_features", ["kept", 6], dtype),
            ("retention_mask", ["total"], ir.DataType.BOOL),
            ("kept_indices", ["kept"], ir.DataType.INT64),
        ],
        cuda_provider,
        module=projector,
        state=reference.checkpoint_state(),
        dtype=dtype,
        profile=True,
    )
    try:
        features = np.random.default_rng(815).normal(size=(4, 16, 8)).astype(dtype.numpy())
        _run(session, {"features": features})
    finally:
        path = pathlib.Path(session.end_profiling())
        try:
            events = json.loads(path.read_text())
        finally:
            path.unlink(missing_ok=True)
    kernels = [event for event in events if "provider" in event.get("args", {})]
    gpu_ops = {
        event["args"]["op_name"]
        for event in kernels
        if event["args"]["provider"] == cuda_provider
    }
    assert {
        "MatMul",
        "RMSNormalization",
        "TopK",
        "ReduceL2",
        "ScatterElements",
        "NonZero",
    } <= gpu_ops
    for event in kernels:
        args = event["args"]
        if args["provider"] == "CPUExecutionProvider":
            # Python-double keep counts and small shape vectors may use CPU,
            # but projected features and cosine arithmetic must stay on GPU.
            for specification in args["input_type_shape"]:
                for type_name, shape in specification.items():
                    assert type_name in {"int64", "double"}
                    assert np.prod(shape) <= 4
        if args["op_name"] in {"MemcpyFromHost", "MemcpyToHost"}:
            # The fail-closed score guard exchanges scalar shape metadata only.
            assert args["input_type_shape"] == [{"int64": [1]}]


def test_pinned_projector_parameter_names_and_shapes():
    projector = NemotronHOmniVisionProjector(
        1280,
        20480,
        4096,
        ps_version="v2",
        downsample_ratio=0.5,
        vision_final_layernorm=False,
    )
    assert {
        name: list(parameter.shape) for name, parameter in projector.named_parameters()
    } == {
        "mlp1.0.weight": [5120],
        "mlp1.1.weight": [20480, 5120],
        "mlp1.3.weight": [4096, 20480],
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"downsample_ratio": 0.3},
        {"downsample_ratio": 0.25},
        {"downsample_ratio": 0},
        {"downsample_ratio": float("nan")},
        {"ps_version": "v1"},
        {"vision_final_layernorm": True},
        {"vision_final_layernorm": None},
        {"norm_eps": 0},
    ],
)
def test_unsupported_projector_configuration(kwargs):
    with pytest.raises(ValueError):
        NemotronHOmniVisionProjector(8, 16, 6, **kwargs)


@pytest.mark.parametrize("q", [-0.1, 1.1, float("nan")])
def test_invalid_pruning_rate(q):
    builder, op, _ = create_test_builder()
    features = create_test_input(builder, "features", [2, 4, 3])
    with pytest.raises(ValueError, match="pruning_rate"):
        compact_omni_video_features(op, features, q)


def test_nonfinite_cosine_scores_fail_explicitly(cuda_provider):
    session = _session(
        lambda op, features: compact_omni_video_features(op, features, 0.5),
        [("features", [3, 2, 3], ir.DataType.FLOAT16)],
        [
            ("image_features", ["kept", 3], ir.DataType.FLOAT16),
            ("retention_mask", [6], ir.DataType.BOOL),
            ("kept_indices", ["kept"], ir.DataType.INT64),
        ],
        cuda_provider,
    )
    for value in [0, float("inf"), float("nan")]:
        features = np.full((3, 2, 3), value, dtype=np.float16)
        with pytest.raises(ort.capi.onnxruntime_pybind11_state.Fail, match="reshap"):
            _run(session, {"features": features})
