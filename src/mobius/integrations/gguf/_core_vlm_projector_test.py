# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import dataclasses

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch
from onnxscript import GraphBuilder

from mobius._builder import build_from_module
from mobius._configs import Gemma4AudioConfig, VisionConfig
from mobius._constants import OPSET_VERSION
from mobius.components import Gemma4AudioEncoder
from mobius.integrations.gguf._core_vlm_projector import (
    _Gemma4ProjectorConfig,
    _load_core_vlm_projector_weights,
    core_vlm_projector_fingerprint,
    map_core_vlm_projector_tensor,
    read_core_vlm_projector_config,
)
from mobius.models.gguf_core_projector import CoreVLMProjectorModel
from mobius.tasks._gguf_core_projector import CoreVLMProjectorTask
from mobius.tasks._gguf_projector import GGUFVisionProjectorModel, GGUFVisionProjectorTask


class _Sidecar:
    def __init__(self, metadata, shapes, tensors=None):
        self.metadata = metadata
        self._shapes = shapes
        self._tensors = tensors or {}
        self.tensor_names = tuple(self._tensors or self._shapes)

    def get_tensor_shape(self, name):
        return tuple(self._shapes[name])

    def get_tensor(self, name):
        return self._tensors[name]


def _vision_metadata(
    *,
    hidden=8,
    intermediate=16,
    layers=1,
    heads=2,
    image=8,
    patch=2,
    projection=6,
    merge=2,
):
    return {
        "clip.has_vision_encoder": True,
        "clip.vision.embedding_length": hidden,
        "clip.vision.feed_forward_length": intermediate,
        "clip.vision.block_count": layers,
        "clip.vision.projection_dim": projection,
        "clip.vision.attention.head_count": heads,
        "clip.vision.attention.layer_norm_epsilon": 1e-6,
        "clip.vision.image_size": image,
        "clip.vision.patch_size": patch,
        "clip.vision.projector.scale_factor": merge,
    }


@pytest.mark.parametrize(
    ("projector_type", "expected_tokens"),
    [
        ("idefics3", 4),
        ("internvl", 4),
        ("llama4", 4),
        ("pixtral", 16),
    ],
)
def test_vision_config_recovers_feature_cardinality(projector_type, expected_tokens):
    metadata = _vision_metadata()
    shapes = {"mm.model.mlp.1.weight": (7, 32)}
    if projector_type == "pixtral":
        metadata.pop("clip.vision.projector.scale_factor")
        metadata["clip.vision.spatial_merge_size"] = 1

    config = read_core_vlm_projector_config(
        _Sidecar(metadata, shapes),
        projector_type,
    )

    assert config.hidden_size == 6
    assert config.vision.mm_tokens_per_image == expected_tokens
    assert config.vision.spatial_merge_size == (1 if projector_type == "pixtral" else 2)
    if projector_type == "llama4":
        assert config.vision.projector_intermediate_size == 7


def test_gemma3n_roles_keep_distinct_fixed_cardinalities():
    metadata = {
        **_vision_metadata(
            hidden=2048,
            intermediate=8192,
            layers=128,
            heads=8,
            image=768,
            patch=3,
            projection=2048,
            merge=1,
        ),
        "clip.audio.embedding_length": 1536,
        "clip.audio.feed_forward_length": 6144,
        "clip.audio.block_count": 12,
        "clip.audio.projection_dim": 2048,
        "clip.audio.attention.head_count": 8,
        "clip.audio.num_mel_bins": 128,
    }
    shapes = {
        "mm.embedding.weight": (128, 2048),
        "mm.a.embedding.weight": (128, 1536),
        "a.conv1d.0.weight": (128, 1, 3, 3),
        "a.conv1d.1.weight": (32, 128, 3, 3),
    }
    sidecar = _Sidecar(metadata, shapes)

    vision = read_core_vlm_projector_config(sidecar, "gemma3nv")
    audio = read_core_vlm_projector_config(sidecar, "gemma3na")

    assert vision.vision_soft_tokens_per_image == 256
    assert vision.vision.image_size == 768
    assert vision.audio is None
    assert audio.audio_soft_tokens_per_image == 188
    assert audio.audio.input_feat_size == 128
    assert audio.vision is None


def test_gemma4_nonunified_and_unified_audio_configs_do_not_alias():
    nonunified_md = {
        "clip.audio.embedding_length": 16,
        "clip.audio.feed_forward_length": 64,
        "clip.audio.block_count": 1,
        "clip.audio.projection_dim": 8,
        "clip.audio.attention.head_count": 2,
        "clip.audio.num_mel_bins": 8,
    }
    nonunified = read_core_vlm_projector_config(
        _Sidecar(
            nonunified_md,
            {
                "a.conv1d.0.weight": (4, 1, 3, 3),
                "a.conv1d.1.weight": (4, 4, 3, 3),
                "a.pre_encode.out.weight": (8, 16),
            },
        ),
        "gemma4a",
    )
    unified_md = {
        "clip.audio.embedding_length": 640,
        "clip.audio.feed_forward_length": 0,
        "clip.audio.block_count": 0,
        "clip.audio.projection_dim": 3840,
        "clip.audio.attention.head_count": 1,
        "clip.audio.num_mel_bins": 128,
    }
    unified = read_core_vlm_projector_config(
        _Sidecar(unified_md, {}),
        "gemma4ua",
    )

    assert nonunified.audio.input_size == 8
    assert nonunified.audio.num_layers == 1
    assert unified.audio.hidden_size == 640
    assert unified.audio.num_layers == 0


def test_gemma4_unified_vision_uses_effective_48px_patches():
    metadata = {
        **_vision_metadata(
            hidden=3840,
            intermediate=0,
            layers=0,
            heads=1,
            image=224,
            patch=16,
            projection=3840,
            merge=1,
        )
    }
    sidecar = _Sidecar(
        metadata,
        {"v.position_embd.weight": (2, 1120, 3840)},
    )

    config = read_core_vlm_projector_config(sidecar, "gemma4uv")

    assert config.vision.patch_size == 16
    assert config.vision.pooling_kernel_size == 3
    assert config.vision.patch_size * config.vision.pooling_kernel_size == 48
    assert config.vision.position_embedding_size == 1120


@pytest.mark.parametrize(
    ("projector_type", "source", "target"),
    [
        (
            "gemma3nv",
            "v.blk.2.3.layer_scale.gamma",
            "vision_encoder.encoder.blocks.2.3.layer_scale.gamma",
        ),
        (
            "gemma3na",
            "a.blk.4.linear_pos.weight",
            (
                "audio_encoder.encoder.conformer.4.attention.attn."
                "relative_position_embedding.pos_proj.weight"
            ),
        ),
        (
            "idefics3",
            "mm.model.fc.weight",
            "vision_encoder.projector.model_fc.weight",
        ),
        (
            "internvl",
            "mm.model.mlp.3.bias",
            "vision_encoder.projector.mlp.3.bias",
        ),
        (
            "llama4",
            "mm.model.mlp.2.weight",
            "vision_encoder.projector.model_mlp_2.weight",
        ),
        (
            "pixtral",
            "v.token_embd.img_break",
            "vision_encoder.projector.image_break",
        ),
        (
            "gemma4uv",
            "v.patch_norm.3.weight",
            "vision_encoder.pos_norm.weight",
        ),
        (
            "gemma4ua",
            "mm.a.input_projection.weight",
            "audio_encoder.projector.weight",
        ),
    ],
)
def test_tensor_routes_are_role_and_architecture_specific(projector_type, source, target):
    assert map_core_vlm_projector_tensor(source, projector_type) == target


def test_similar_projectors_do_not_alias_tensor_families():
    assert map_core_vlm_projector_tensor("mm.model.fc.weight", "idefics3")
    assert map_core_vlm_projector_tensor("mm.model.fc.weight", "internvl") is None
    assert map_core_vlm_projector_tensor("v.class_embd", "internvl")
    assert map_core_vlm_projector_tensor("v.class_embd", "idefics3") is None
    assert map_core_vlm_projector_tensor("a.blk.0.attn_q.weight", "gemma4a")
    assert map_core_vlm_projector_tensor("a.blk.0.attn_q.weight", "gemma4ua") is None
    assert map_core_vlm_projector_tensor("v.blk.0.attn_q.weight", "llama4")
    assert map_core_vlm_projector_tensor("v.blk.0.attn_q.weight", "pixtral")
    assert (
        map_core_vlm_projector_tensor("v.blk.0.attn_q.bias", "pixtral") is None
    )


def test_gemma3n_audio_loader_reverses_baked_softplus():
    raw = np.array([-1.0, 0.25], dtype=np.float32)
    baked = np.log1p(np.exp(raw)).astype(np.float32)
    sidecar = _Sidecar(
        {},
        {"a.blk.0.per_dim_scale": baked.shape},
        {"a.blk.0.per_dim_scale": baked},
    )

    state = _load_core_vlm_projector_weights(sidecar, "gemma3na")

    np.testing.assert_allclose(
        state["audio_encoder.encoder.conformer.0.attention.attn.per_dim_scale"].numpy(),
        raw,
        rtol=1e-6,
        atol=1e-6,
    )


def test_unified_vision_loader_restores_hf_patch_order():
    patch = 3
    indices = np.arange(patch * patch * 3)
    channels = indices // (patch * patch)
    rows = (indices % (patch * patch)) // patch
    columns = indices % patch
    permutation = rows * patch * 3 + columns * 3 + channels
    original = np.arange(2 * 27, dtype=np.float32).reshape(2, 27)
    converted = original[:, permutation]
    position = np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2)
    sidecar = _Sidecar(
        {"clip.vision.patch_size": 1},
        {
            "v.patch_embd.weight": converted.shape,
            "v.position_embd.weight": position.shape,
        },
        {
            "v.patch_embd.weight": converted,
            "v.position_embd.weight": position,
        },
    )
    state = _load_core_vlm_projector_weights(sidecar, "gemma4uv")

    np.testing.assert_array_equal(
        state["vision_encoder.patch_dense.weight"].numpy(),
        original,
    )
    np.testing.assert_array_equal(
        state["vision_encoder.pos_emb_x.weight"].numpy(),
        position[0],
    )


def test_route_fingerprint_includes_projector_discriminator():
    config = _vision_metadata()
    sidecar = _Sidecar(config, {"mm.model.mlp.1.weight": (7, 32)})
    idefics = read_core_vlm_projector_config(sidecar, "idefics3")
    internvl = dataclasses.replace(idefics, model_type="internvl")

    assert core_vlm_projector_fingerprint(idefics, "idefics3") != (
        core_vlm_projector_fingerprint(internvl, "internvl")
    )


def _layer_norm(x, weight, bias, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    variance = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + eps) * weight + bias


def _rms_norm(x, eps=1e-6):
    return x / np.sqrt((x**2).mean(axis=-1, keepdims=True) + eps)


def test_gemma4_unified_audio_projector_matches_closed_form(tmp_path):
    config = _Gemma4ProjectorConfig(
        hidden_size=4,
        rms_norm_eps=1e-6,
        audio=Gemma4AudioConfig(
            hidden_size=3,
            num_layers=0,
            output_proj_dims=3,
            rms_norm_eps=1e-6,
        ),
    )
    module = CoreVLMProjectorModel(config, "gemma4ua")
    weight = np.arange(12, dtype=np.float32).reshape(4, 3) / 20
    module.audio_encoder.projector.weight.const_value = ir.tensor(weight)
    package = build_from_module(
        module,
        config,
        task=CoreVLMProjectorTask("gemma4ua"),
    )
    path = tmp_path / "gemma4ua.onnx"
    ir.save(package["audio_encoder"], path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    features = np.array([[[1.0, -2.0, 0.5], [4.0, 5.0, 6.0]]], np.float32)
    mask = np.array([[True, False]])

    (actual,) = session.run(
        ["audio_features"],
        {"input_features": features, "input_features_mask": mask},
    )
    expected = _rms_norm(features[:, :1]).reshape(1, 3) @ weight.T

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    assert actual.shape == (1, 4)


def test_gemma4_unified_vision_projector_matches_closed_form(tmp_path):
    config = _Gemma4ProjectorConfig(
        hidden_size=5,
        rms_norm_eps=1e-6,
        vision=VisionConfig(
            hidden_size=4,
            intermediate_size=0,
            num_hidden_layers=0,
            num_attention_heads=0,
            image_size=2,
            patch_size=1,
            pooling_kernel_size=1,
            position_embedding_size=2,
            out_hidden_size=4,
            norm_eps=1e-6,
        ),
    )
    sidecar = CoreVLMProjectorModel(config, "gemma4uv")
    encoder = sidecar.vision_encoder
    rng = np.random.default_rng(17)
    state = {}
    for name, parameter in encoder.named_parameters():
        if parameter.const_value is not None:
            continue
        shape = tuple(int(dim) for dim in parameter.shape)
        if name.endswith(".weight") and ("ln" in name or "norm" in name):
            values = (1.0 + rng.normal(0, 0.1, shape)).astype(np.float32)
        else:
            values = rng.normal(0, 0.1, shape).astype(np.float32)
        parameter.const_value = ir.tensor(values)
        state[name] = values
    package = build_from_module(
        GGUFVisionProjectorModel(encoder),
        config,
        task=GGUFVisionProjectorTask(),
    )
    path = tmp_path / "gemma4uv.onnx"
    ir.save(package["vision_encoder"], path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    pixels = np.array([[[0.2, 0.4, 0.8], [0.3, 0.1, 0.7]]], np.float32)
    positions = np.array([[[0, 0], [1, 0]]], np.int64)

    (actual,) = session.run(
        None,
        {"pixel_values": pixels, "pixel_position_ids": positions},
    )
    expected = _layer_norm(
        pixels,
        state["patch_ln1.weight"],
        state["patch_ln1.bias"],
    )
    expected = expected @ state["patch_dense.weight"].T + state["patch_dense.bias"]
    expected = _layer_norm(
        expected,
        state["patch_ln2.weight"],
        state["patch_ln2.bias"],
    )
    expected += state["pos_emb_x.weight"][positions[..., 0]]
    expected += state["pos_emb_y.weight"][positions[..., 1]]
    expected = _layer_norm(
        expected,
        state["pos_norm.weight"],
        state["pos_norm.bias"],
    )
    expected = _rms_norm(expected) @ state["projector.weight"].T

    np.testing.assert_allclose(
        actual,
        expected.reshape(-1, 5),
        rtol=1e-5,
        atol=1e-5,
    )


def test_gemma4_audio_projector_matches_reference():
    modeling = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
    configuration = pytest.importorskip("transformers.models.gemma4.configuration_gemma4")
    torch.manual_seed(4)
    hf_config = configuration.Gemma4AudioConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        subsampling_conv_channels=[8, 4],
        conv_kernel_size=3,
        attention_chunk_size=4,
        attention_context_left=5,
        output_proj_dims=8,
        use_clipped_linears=True,
    )
    reference = modeling.Gemma4AudioModel(hf_config).eval()
    for parameter in reference.parameters():
        parameter.data.normal_(0, 0.1)

    state = {}
    for name, tensor in reference.state_dict().items():
        name = name.replace(
            "subsample_conv_projection.layer0.conv.",
            "subsample_conv_projection.conv0.",
        )
        name = name.replace(
            "subsample_conv_projection.layer0.norm.",
            "subsample_conv_projection.norm0.",
        )
        name = name.replace(
            "subsample_conv_projection.layer1.conv.",
            "subsample_conv_projection.conv1.",
        )
        name = name.replace(
            "subsample_conv_projection.layer1.norm.",
            "subsample_conv_projection.norm1.",
        )
        state[name.replace(".linear.", ".")] = tensor.detach().numpy().astype(np.float32)

    encoder = Gemma4AudioEncoder(
        input_size=8,
        hidden_size=16,
        num_heads=2,
        num_layers=1,
        conv_kernel_size=3,
        conv_channels=[8, 4],
        attention_context_left=5,
        output_proj_dims=8,
    )
    for name, parameter in encoder.named_parameters():
        if name in state:
            parameter.const_value = ir.tensor(state[name])

    inputs = [
        ir.Value(
            name="input_features",
            shape=ir.Shape([1, "time", 8]),
            type=ir.TensorType(ir.DataType.FLOAT),
        ),
        ir.Value(
            name="input_features_mask",
            shape=ir.Shape([1, "time"]),
            type=ir.TensorType(ir.DataType.BOOL),
        ),
    ]
    graph = ir.Graph(
        inputs=inputs,
        outputs=[],
        nodes=[],
        name="gemma4_audio_parity",
        opset_imports={"": OPSET_VERSION},
    )
    builder = GraphBuilder(graph)
    output, output_mask = encoder(builder.op, *inputs)
    output.name = "output"
    output_mask.name = "output_mask"
    graph.outputs.extend((output, output_mask))
    session = ort.InferenceSession(
        ir.serde.serialize_model(ir.Model(graph, ir_version=11)).SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    features = np.random.default_rng(2).normal(size=(1, 20, 8)).astype(np.float32)
    mask = np.zeros((1, 20), dtype=bool)
    mask[:, :17] = True

    with torch.no_grad():
        expected = reference(
            torch.from_numpy(features),
            attention_mask=torch.from_numpy(mask),
        )
    actual, actual_mask = session.run(
        None,
        {"input_features": features, "input_features_mask": mask},
    )

    np.testing.assert_allclose(
        actual,
        expected.last_hidden_state.numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_array_equal(actual_mask, expected.attention_mask.numpy())
