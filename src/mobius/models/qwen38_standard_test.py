# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Offline Qwen3.8-27B standard export and reduced text/multimodal parity.

The pinned official config exercises the public text-only and full VLM builders.
Numerical tests use random weights and reduced dimensions, not the 27B checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch
import transformers

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.integrations._weight_loading import apply_weights
from mobius.integrations.transformers._builder import build_transformers_model

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "testdata/configs/qwen3_8-27b.json"


def _hf_config(*, reduced: bool = False):
    config = transformers.Qwen3_5Config.from_dict(json.loads(_CONFIG_PATH.read_text()))
    if reduced:
        text = config.text_config
        text.hidden_size = 64
        text.intermediate_size = 128
        text.num_hidden_layers = 4
        text.layer_types = text.layer_types[:4]
        text.num_attention_heads = 6
        text.num_key_value_heads = 1
        text.head_dim = 32
        text.vocab_size = 256
        text.linear_num_key_heads = 2
        text.linear_num_value_heads = 6
        text.linear_key_head_dim = 8
        text.linear_value_head_dim = 12
        text.max_position_embeddings = 128
        text.rope_parameters["mrope_section"] = [2, 1, 1]
        text.bos_token_id = 1
        text.eos_token_id = 2
    return config


def _assert_standard(model: ir.Model) -> None:
    assert not model.functions, "standard-only export must inline local functions"
    nodes = list(model.graph.all_nodes())
    for node in nodes:
        assert node.domain in {"", "ai.onnx"}, (node.domain, node.op_type)
        onnx.defs.get_schema(
            node.op_type, model.opset_imports.get(node.domain, model.opset_imports[""]), ""
        )


def _build(config, path: Path, dtype: str, *, text_only: bool = True):
    config.save_pretrained(path)
    package = build_transformers_model(
        str(path),
        text_only=text_only,
        load_weights=False,
        dtype=dtype,
        execution_provider="onnx-standard",
    )
    if text_only:
        assert set(package) == {"model"}
        assert package.config.model_type == "qwen3_5_text"
        decoder = package["model"]
    else:
        assert set(package) == {"decoder", "vision_encoder", "embedding"}
        assert package.config.vision is not None
        decoder = package["decoder"]
    assert any(node.op_type == "Scan" for node in decoder.graph.all_nodes())
    for model in package.values():
        _assert_standard(model)
    return package


def _hf_vl_config():
    config = _hf_config(reduced=True)
    config.image_token_id = 240
    config.video_token_id = 241
    config.vision_start_token_id = 242
    config.vision_end_token_id = 243
    vision = config.vision_config
    vision.depth = 2
    vision.hidden_size = 32
    vision.intermediate_size = 64
    vision.num_heads = 2
    vision.out_hidden_size = config.text_config.hidden_size
    vision.num_position_embeddings = 16
    return config


def test_qwen38_official_text_config():
    config = ArchitectureConfig.from_transformers(_hf_config().text_config)
    assert config.num_hidden_layers == 64
    assert (
        config.layer_types
        == ["linear_attention"] * 3
        + ["full_attention"] * 1
        + (["linear_attention"] * 3 + ["full_attention"]) * 15
    )
    assert (config.hidden_size, config.intermediate_size, config.vocab_size) == (
        5120,
        17408,
        248320,
    )
    assert (config.num_attention_heads, config.num_key_value_heads, config.head_dim) == (
        24,
        4,
        256,
    )
    assert config.mrope_interleaved
    assert config.mrope_section == [11, 11, 10]
    assert getattr(config, "mamba_ssm_dtype", None) == ir.DataType.FLOAT


@pytest.mark.parametrize("dtype", ["f32", "f16"])
@pytest.mark.parametrize("text_config_only", [False, True])
def test_qwen38_standard_text_only_build(tmp_path: Path, dtype: str, text_config_only: bool):
    config = _hf_config(reduced=True)
    if text_config_only:
        config = config.text_config
    package = _build(config, tmp_path / "config", dtype)
    graph = package["model"].graph
    assert len([value for value in graph.inputs if "past_key_values." in value.name]) == 8
    for value in [*graph.inputs, *graph.outputs]:
        if value.name.endswith(".recurrent_state"):
            assert value.dtype == ir.DataType.FLOAT


@pytest.mark.parametrize("invalid_dtype", ["not-a-dtype", "int64"])
def test_qwen38_rejects_invalid_state_dtype(invalid_dtype: str):
    config = _hf_config(reduced=True).text_config
    config.mamba_ssm_dtype = invalid_dtype
    with pytest.raises(ValueError, match="mamba_ssm_dtype"):
        ArchitectureConfig.from_transformers(config)


@pytest.mark.parametrize("state_dtype", [None, "auto"])
def test_qwen38_unspecified_state_dtype_preserves_existing_behavior(
    tmp_path: Path, state_dtype: str | None
):
    config = _hf_config(reduced=True)
    if state_dtype is None:
        del config.text_config.mamba_ssm_dtype
    else:
        config.text_config.mamba_ssm_dtype = state_dtype
    package = _build(config, tmp_path / "config", "f16")
    assert package.config.mamba_ssm_dtype is None
    states = [
        value
        for value in package["model"].graph.inputs
        if value.name.endswith(".recurrent_state")
    ]
    assert states
    assert all(value.dtype == ir.DataType.FLOAT16 for value in states)


def test_qwen38_standard_check_rejects_nested_custom_op(tmp_path: Path):
    model = _build(_hf_config(reduced=True), tmp_path / "config", "f32")["model"]
    scan = next(node for node in model.graph if node.op_type == "Scan")
    body = scan.attributes["body"].value
    next(node for node in body if node.op_type == "MatMul").domain = "com.microsoft"
    with pytest.raises(AssertionError, match=r"com\.microsoft"):
        _assert_standard(model)


@pytest.mark.integration
@pytest.mark.integration_fast
@pytest.mark.parametrize("dtype", ["f32", "f16"])
def test_qwen38_full_config_standard_build(tmp_path: Path, dtype: str):
    """L2 config/graph evidence only: all 64 layers, without weight allocation."""
    package = _build(_hf_config(), tmp_path / "config", dtype)
    graph = package["model"].graph
    assert len([value for value in graph.inputs if value.name.endswith(".key")]) == 16
    assert len([value for value in graph.inputs if value.name.endswith(".conv_state")]) == 48


@pytest.mark.parametrize("dtype", ["f32", "f16"])
def test_qwen38_standard_vl_build_roundtrip(tmp_path: Path, dtype: str):
    package = _build(_hf_vl_config(), tmp_path / "config", dtype, text_only=False)
    assert {value.name for value in package["embedding"].graph.inputs} == {
        "input_ids",
        "image_features",
    }
    assert package["vision_encoder"].graph.inputs[0].dtype == ir.DataType.FLOAT
    positions = next(
        value for value in package["decoder"].graph.inputs if value.name == "position_ids"
    )
    assert positions.shape[0] == 3
    for model in package.values():
        for value in model.graph.initializers.values():
            if value.const_value is None:
                value.const_value = ir.Tensor(
                    np.zeros(tuple(value.shape), dtype=value.dtype.numpy())
                )
    package.save(str(tmp_path / "package"), progress_bar=False)
    reloaded = ModelPackage.load(str(tmp_path / "package"))
    assert set(reloaded) == set(package)
    for model in reloaded.values():
        _assert_standard(model)


@pytest.mark.parametrize("with_media", [True, False])
def test_qwen3_split_deepstack_global_media_order(tmp_path: Path, with_media: bool):
    """Shared DeepStack maps use the same global mixed-media indices as main features."""
    from mobius.models.qwen_vl import Qwen3VLEmbeddingModel
    from mobius.tasks._base import build_embedding_from_features

    hf_config = _hf_vl_config()
    config = ArchitectureConfig.from_transformers(hf_config.text_config)
    config.image_token_id = hf_config.image_token_id
    config.video_token_id = hf_config.video_token_id
    config.dtype = ir.DataType.FLOAT
    config.deepstack_visual_indexes = [0, 1]
    model = build_embedding_from_features(
        Qwen3VLEmbeddingModel(config),
        config,
        feature_name="image_features",
        feature_dim=config.hidden_size,
        deepstack=True,
    )
    _assert_standard(model)
    assert {
        name for name, value in model.graph.initializers.items() if value.const_value is None
    } == {"embed_tokens.weight"}
    table = model.graph.initializers["embed_tokens.weight"]
    table.const_value = ir.Tensor(np.zeros(tuple(table.shape), dtype=np.float32))
    tokens = np.asarray([[3, 240, 241, 5], [4, 241, 240, 6]], dtype=np.int64)
    main = np.repeat(np.arange(1, 5, dtype=np.float32)[:, None], config.hidden_size, axis=1)
    packed = np.concatenate((main, main + 100, main + 200), axis=1)
    if not with_media:
        tokens = np.asarray([[3, 4, 5, 6], [7, 8, 9, 10]], dtype=np.int64)
        packed = packed[:0]
    path = tmp_path / "deepstack-embedding.onnx"
    ir.save(model, path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = dict(
        zip(
            (value.name for value in session.get_outputs()),
            session.run(None, {"input_ids": tokens, "image_features": packed}),
            strict=True,
        )
    )
    expected = np.zeros((2, 4, config.hidden_size), dtype=np.float32)
    expected_deepstack = np.zeros((2, 4, 2 * config.hidden_size), dtype=np.float32)
    if with_media:
        expected[:, 1:3] = main.reshape(2, 2, config.hidden_size)
        expected_deepstack[:, 1:3] = packed[:, config.hidden_size :].reshape(
            2, 2, 2 * config.hidden_size
        )
    np.testing.assert_array_equal(actual["inputs_embeds"], expected)
    np.testing.assert_array_equal(actual["per_layer_inputs"], expected_deepstack)


@pytest.mark.integration
@pytest.mark.integration_fast
@pytest.mark.parametrize("dtype", ["f32", "f16"])
def test_qwen38_full_vl_config_standard_build(tmp_path: Path, dtype: str):
    """L2: all 64 decoder and 27 vision layers, without allocating weights."""
    package = _build(_hf_config(), tmp_path / "config", dtype, text_only=False)
    graph = package["decoder"].graph
    assert len([value for value in graph.inputs if value.name.endswith(".key")]) == 16
    assert len([value for value in graph.inputs if value.name.endswith(".conv_state")]) == 48
    assert all(
        value.dtype == ir.DataType.FLOAT
        for value in graph.inputs
        if value.name.endswith(".recurrent_state")
    )
    assert package.config.vision.num_hidden_layers == 27
    assert package.config.vision.out_hidden_size == package.config.hidden_size == 5120
    assert {value.name for value in package["embedding"].graph.inputs} == {
        "input_ids",
        "image_features",
    }


def _initial_feeds(model: ir.Model, batch: int) -> dict[str, np.ndarray]:
    feeds = {}
    for value in model.graph.inputs:
        if not value.name.startswith("past_key_values."):
            continue
        shape = [
            dim if isinstance(dim, int) else batch if i == 0 else 0
            for i, dim in enumerate(value.shape)
        ]
        feeds[value.name] = np.zeros(shape, dtype=value.dtype.numpy())
    return feeds


def _compare_states(actual, cache, config, state_feeds, *, tolerance: float, step: int):
    for layer_idx, layer_type in enumerate(config.layer_types):
        layer = cache.layers[layer_idx]
        if layer_type == "linear_attention":
            expected_states = {
                "conv_state": layer.conv_states[0][
                    ..., -(config.linear_conv_kernel_dim - 1) :
                ],
                "recurrent_state": layer.recurrent_states[0],
            }
        else:
            expected_states = {"key": layer.keys, "value": layer.values}
        for role, value in expected_states.items():
            name = f"present.{layer_idx}.{role}"
            np.testing.assert_allclose(
                actual[name],
                value.cpu().numpy(),
                rtol=tolerance,
                atol=tolerance,
                err_msg=f"{name} step {step}",
                strict=True,
            )
            state_feeds[f"past_key_values.{layer_idx}.{role}"] = actual[name]


def _assert_cuda_matrix(profile, role: str):
    assert any(
        event.get("args", {}).get("provider") == "CUDAExecutionProvider"
        and event["args"].get("op_name") in {"MatMul", "FusedMatMul", "Gemm", "FusedGemm"}
        for event in profile
    ), f"{role} must execute CUDA matrix operations"


@pytest.mark.integration
@pytest.mark.integration_fast
@pytest.mark.parametrize("dtype", ["f32", "f16"])
@pytest.mark.parametrize("batch,prefill", [(1, 1), (2, 6)])
def test_qwen38_standard_cuda_hf_parity(tmp_path: Path, dtype: str, batch: int, prefill: int):
    """Compare full logits and every carried state at prefill and four decode steps."""
    if (
        "CUDAExecutionProvider" not in ort.get_available_providers()
        or not torch.cuda.is_available()
    ):
        pytest.skip("CUDA ORT and PyTorch are required")
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    torch.manual_seed(38)
    config = _hf_config(reduced=True)
    config.text_config._attn_implementation = "eager"
    torch_dtype = torch.float32 if dtype == "f32" else torch.float16
    reference = (
        Qwen3_5ForCausalLM._from_config(config.text_config, dtype=torch_dtype).eval().cuda()
    )
    package = _build(config, tmp_path / "config", dtype)
    model = package["model"]
    processed = reference.state_dict()
    # Exercise the actual standalone model's HF name adapter.
    from mobius.models import Qwen35CausalLMModel

    processed = Qwen35CausalLMModel(package.config).preprocess_weights(dict(processed))
    apply_weights(model, processed)
    _assert_standard(model)
    package.save(str(tmp_path / "package"), progress_bar=False)
    reloaded = ModelPackage.load(str(tmp_path / "package"))
    model = reloaded["model"]
    _assert_standard(model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.enable_profiling = True
    options.profile_file_prefix = str(tmp_path / "cuda-profile")
    session = ort.InferenceSession(
        str(tmp_path / "package/model.onnx"),
        sess_options=options,
        providers=[("CUDAExecutionProvider", {"use_tf32": "0"})],
    )
    assert session.get_providers()[0] == "CUDAExecutionProvider"
    rng = np.random.default_rng(38)
    tokens = rng.integers(3, 256, size=(batch, prefill), dtype=np.int64)
    state_feeds = _initial_feeds(model, batch)
    cache = None
    seen = 0
    output_names = [value.name for value in model.graph.outputs]
    tolerance = 1e-3 if dtype == "f32" else 1e-2
    for step in range(5):
        length = tokens.shape[1]
        mask = np.ones((batch, seen + length), dtype=np.int64)
        positions = np.broadcast_to(np.arange(seen, seen + length), tokens.shape).copy()
        with torch.no_grad():
            expected = reference(
                input_ids=torch.from_numpy(tokens).cuda(),
                attention_mask=torch.from_numpy(mask).cuda(),
                position_ids=torch.from_numpy(positions).cuda(),
                past_key_values=cache,
                use_cache=True,
            )
        cache = expected.past_key_values
        actual = dict(
            zip(
                output_names,
                session.run(
                    None,
                    {
                        "input_ids": tokens,
                        "attention_mask": mask,
                        "position_ids": positions,
                        **state_feeds,
                    },
                ),
                strict=True,
            )
        )
        target_logits = expected.logits.cpu().numpy()
        np.testing.assert_allclose(
            actual["logits"],
            target_logits,
            rtol=tolerance,
            atol=tolerance,
            err_msg=f"logits step {step}",
            strict=True,
        )
        expected_tokens = expected.logits[:, -1].argmax(-1).cpu().numpy()
        np.testing.assert_array_equal(actual["logits"][:, -1].argmax(-1), expected_tokens)
        _compare_states(
            actual,
            cache,
            config.text_config,
            state_feeds,
            tolerance=tolerance,
            step=step,
        )
        seen += length
        tokens = expected_tokens[:, None]
    profile = json.loads(Path(session.end_profiling()).read_text())
    _assert_cuda_matrix(profile, "decoder")


def _vl_inputs(config, case: str, batch: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(38)
    rows = []
    image_grids = []
    video_grids = []
    for row in range(batch):
        blocks = []
        if case in {"images", "mixed"}:
            image_grids.append([1, 4, 6])
            blocks.append([config.image_token_id] * 6)
        if case == "images":
            image_grids.append([1, 6, 4])
            blocks.append([config.image_token_id] * 6)
        if case in {"video", "mixed"}:
            video_grids.append([2, 4, 4])
            blocks.extend([[config.video_token_id] * 4 for _ in range(2)])
        if row % 2:
            blocks.reverse()
        tokens = [3]
        for block in blocks:
            tokens.extend(
                [config.vision_start_token_id, *block, config.vision_end_token_id, 4]
            )
        tokens.append(5)
        rows.append(tokens)
    result = {"input_ids": np.asarray(rows, dtype=np.int64)}
    result["attention_mask"] = np.ones_like(result["input_ids"])
    result["mm_token_type_ids"] = np.where(
        result["input_ids"] == config.image_token_id,
        1,
        np.where(result["input_ids"] == config.video_token_id, 2, 0),
    ).astype(np.int32)
    vision = config.vision_config
    patch_dim = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
    for grids, grid_name, pixel_name in (
        (image_grids, "image_grid_thw", "pixel_values"),
        (video_grids, "video_grid_thw", "pixel_values_videos"),
    ):
        if grids:
            grid = np.asarray(grids, dtype=np.int64)
            result[grid_name] = grid
            result[pixel_name] = rng.standard_normal(
                (int(np.prod(grid, axis=1).sum()), patch_dim)
            ).astype(np.float32)
    return result


def _pack_vl_features(tokens, features, config):
    """Pack actual ONNX image/video rows in flattened placeholder-token order."""
    streams = {
        config.image_token_id: features["image_features"],
        config.video_token_id: features["video_features"],
    }
    offsets = dict.fromkeys(streams, 0)
    packed = []
    for token in tokens.flat:
        if token in streams:
            packed.append(streams[token][offsets[token]])
            offsets[token] += 1
    assert all(offsets[token] == len(stream) for token, stream in streams.items())
    return np.stack(packed) if packed else features["image_features"]


@pytest.mark.integration
@pytest.mark.integration_fast
@pytest.mark.parametrize("dtype", ["f32", "f16"])
@pytest.mark.parametrize(
    "case,batch",
    [
        ("text", 1),
        ("images", 1),
        ("images", 2),
        ("video", 1),
        ("video", 2),
        ("mixed", 1),
        ("mixed", 2),
    ],
)
def test_qwen38_standard_vl_cuda_hf_parity(tmp_path: Path, dtype: str, case: str, batch: int):
    """L3: actual ONNX vision -> fusion -> decoder, then four reused-state steps.

    Nonzero packed patches are synthetic, not official processor outputs. HF
    supplies the graph's explicit MRoPE positions; host preprocessing/position
    generation and the official checkpoint need separate qualification.
    """
    if (
        "CUDAExecutionProvider" not in ort.get_available_providers()
        or not torch.cuda.is_available()
    ):
        pytest.skip("CUDA ORT and PyTorch are required")
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForConditionalGeneration,
    )

    from mobius.models import Qwen35VL3ModelCausalLMModel

    torch.manual_seed(38)
    config = _hf_vl_config()
    config.text_config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"
    torch_dtype = torch.float32 if dtype == "f32" else torch.float16
    reference = (
        Qwen3_5ForConditionalGeneration._from_config(config, dtype=torch_dtype).eval().cuda()
    )
    package = _build(config, tmp_path / "config", dtype, text_only=False)
    weights = Qwen35VL3ModelCausalLMModel(package.config).preprocess_weights(
        dict(reference.state_dict())
    )
    for model in package.values():
        apply_weights(model, weights)
        _assert_standard(model)
    package.save(str(tmp_path / "package"), progress_bar=False)
    reloaded = ModelPackage.load(str(tmp_path / "package"))
    sessions = {}
    for name, model in reloaded.items():
        _assert_standard(model)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.enable_profiling = True
        options.profile_file_prefix = str(tmp_path / f"cuda-{name}")
        sessions[name] = ort.InferenceSession(
            str(tmp_path / f"package/{name}/model.onnx"),
            sess_options=options,
            providers=[("CUDAExecutionProvider", {"use_tf32": "0"})],
        )
        assert sessions[name].get_providers()[0] == "CUDAExecutionProvider"

    def run(name, feeds):
        session = sessions[name]
        return dict(
            zip(
                (value.name for value in session.get_outputs()),
                session.run(None, feeds),
                strict=True,
            )
        )

    inputs = _vl_inputs(config, case, batch)
    hf_inputs = {name: torch.from_numpy(value).cuda() for name, value in inputs.items()}
    tolerance = 1e-3 if dtype == "f32" else 1e-2
    empty = np.zeros((0, config.text_config.hidden_size), dtype=package.config.dtype.numpy())
    features = {"image_features": empty, "video_features": empty}
    with torch.no_grad():
        expected_embeds = reference.get_input_embeddings()(hf_inputs["input_ids"])
        for pixel_name, grid_name, feature_name, token_id in (
            ("pixel_values", "image_grid_thw", "image_features", config.image_token_id),
            ("pixel_values_videos", "video_grid_thw", "video_features", config.video_token_id),
        ):
            if pixel_name not in inputs:
                continue
            expected_features = reference.model.visual(
                hf_inputs[pixel_name].to(torch_dtype), grid_thw=hf_inputs[grid_name]
            ).pooler_output
            features[feature_name] = run(
                "vision_encoder",
                {
                    "pixel_values": inputs[pixel_name],
                    "image_grid_thw": inputs[grid_name],
                },
            )["image_features"]
            np.testing.assert_allclose(
                features[feature_name],
                expected_features.cpu().numpy(),
                rtol=tolerance,
                atol=tolerance,
                strict=True,
            )
            mask = (hf_inputs["input_ids"] == token_id).unsqueeze(-1)
            expected_embeds = expected_embeds.masked_scatter(mask, expected_features)
        positions, deltas = reference.model.get_rope_index(
            hf_inputs["input_ids"],
            hf_inputs["mm_token_type_ids"],
            image_grid_thw=hf_inputs.get("image_grid_thw"),
            video_grid_thw=hf_inputs.get("video_grid_thw"),
            attention_mask=hf_inputs["attention_mask"],
        )
    tokens = inputs["input_ids"]
    positions = positions.cpu().numpy()
    deltas = deltas.cpu().numpy()
    state_feeds = _initial_feeds(reloaded["decoder"], batch)
    cache = None
    seen = 0
    for step in range(5):
        mask = np.ones((batch, seen + tokens.shape[1]), dtype=np.int64)
        embeds = run(
            "embedding",
            {
                "input_ids": tokens,
                "image_features": _pack_vl_features(tokens, features, config),
            },
        )["inputs_embeds"]
        with torch.no_grad():
            if step == 0:
                np.testing.assert_allclose(
                    embeds,
                    expected_embeds.cpu().numpy(),
                    rtol=tolerance,
                    atol=tolerance,
                    strict=True,
                )
            expected = reference(
                **(hf_inputs if step == 0 else {"input_ids": torch.from_numpy(tokens).cuda()}),
                position_ids=torch.from_numpy(positions).cuda(),
                past_key_values=cache,
                use_cache=True,
                **({"attention_mask": torch.from_numpy(mask).cuda()} if step else {}),
            )
        cache = expected.past_key_values
        actual = run(
            "decoder",
            {
                "inputs_embeds": embeds,
                "attention_mask": mask,
                "position_ids": positions,
                **state_feeds,
            },
        )
        np.testing.assert_allclose(
            actual["logits"],
            expected.logits.cpu().numpy(),
            rtol=tolerance,
            atol=tolerance,
            err_msg=f"logits step {step}",
            strict=True,
        )
        expected_tokens = expected.logits[:, -1].argmax(-1).cpu().numpy()
        np.testing.assert_array_equal(actual["logits"][:, -1].argmax(-1), expected_tokens)
        _compare_states(
            actual, cache, config.text_config, state_feeds, tolerance=tolerance, step=step
        )
        seen += tokens.shape[1]
        tokens = expected_tokens[:, None]
        positions = np.broadcast_to((seen + deltas)[None], (3, batch, 1)).copy()
        features = {"image_features": empty, "video_features": empty}
    profiles = {
        name: json.loads(Path(session.end_profiling()).read_text())
        for name, session in sessions.items()
    }
    for role in ["decoder", *([] if case == "text" else ["vision_encoder"])]:
        _assert_cuda_matrix(profiles[role], role)
