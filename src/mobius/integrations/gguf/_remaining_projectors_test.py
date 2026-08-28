# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx_ir as ir
import pytest

from mobius._configs import ArchitectureConfig
from mobius._testing.ort_inference import OnnxModelSession
from mobius.integrations.gguf._mmproj import build_mmproj_from_gguf
from mobius.integrations.gguf._mmproj_mapping import (
    map_mmproj_audio_projector_to_onnx,
)
from mobius.integrations.gguf._remaining_projectors import (
    create_remaining_vision_projector,
    remaining_projector_state_dict,
)
from mobius.models.gguf_audio_projector import create_gguf_audio_projector
from mobius.tasks import (
    GGUFAudioProjectorTask,
    GGUFVisionProjectorModel,
    GGUFVisionProjectorTask,
    get_task,
)


def _write_tiny_janus(path: Path) -> None:
    from gguf import GGUFWriter

    rng = np.random.default_rng(17)
    hidden, intermediate, output = 8, 16, 6
    writer = GGUFWriter(str(path), "clip")
    writer.add_string("general.type", "mmproj")
    writer.add_bool("clip.has_vision_encoder", True)
    writer.add_string("clip.projector_type", "janus_pro")
    writer.add_bool("clip.use_gelu", True)
    writer.add_uint32("clip.vision.embedding_length", hidden)
    writer.add_uint32("clip.vision.feed_forward_length", intermediate)
    writer.add_uint32("clip.vision.block_count", 1)
    writer.add_uint32("clip.vision.projection_dim", output)
    writer.add_uint32("clip.vision.attention.head_count", 2)
    writer.add_float32("clip.vision.attention.layer_norm_epsilon", 1e-6)
    writer.add_uint32("clip.vision.image_size", 4)
    writer.add_uint32("clip.vision.patch_size", 2)
    writer.add_array("clip.vision.image_mean", [0.5, 0.5, 0.5])
    writer.add_array("clip.vision.image_std", [0.5, 0.5, 0.5])

    def add(name: str, shape: tuple[int, ...]) -> None:
        writer.add_tensor(name, rng.normal(0.0, 0.2, shape).astype(np.float32))

    add("v.patch_embd.weight", (hidden, 3, 2, 2))
    add("v.patch_embd.bias", (hidden,))
    add("v.position_embd.weight", (4, hidden))
    add("v.post_ln.weight", (hidden,))
    add("v.post_ln.bias", (hidden,))
    add("mm.0.weight", (output, hidden))
    add("mm.0.bias", (output,))
    add("mm.1.weight", (output, output))
    add("mm.1.bias", (output,))
    prefix = "v.blk.0."
    for norm in ("ln1", "ln2"):
        add(prefix + norm + ".weight", (hidden,))
        add(prefix + norm + ".bias", (hidden,))
    for projection in ("attn_q", "attn_k", "attn_v", "attn_out"):
        add(prefix + projection + ".weight", (hidden, hidden))
        add(prefix + projection + ".bias", (hidden,))
    add(prefix + "ffn_up.weight", (intermediate, hidden))
    add(prefix + "ffn_up.bias", (intermediate,))
    add(prefix + "ffn_down.weight", (hidden, intermediate))
    add(prefix + "ffn_down.bias", (hidden,))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _write_tiny_meralion(path: Path) -> None:
    from gguf import GGUFWriter

    rng = np.random.default_rng(23)
    writer = GGUFWriter(str(path), "clip")
    writer.add_string("general.type", "mmproj")
    writer.add_bool("clip.has_audio_encoder", True)
    writer.add_string("clip.projector_type", "meralion")
    writer.add_uint32("clip.audio.embedding_length", 4)
    writer.add_uint32("clip.audio.feed_forward_length", 8)
    writer.add_uint32("clip.audio.block_count", 1)
    writer.add_uint32("clip.audio.projection_dim", 5)
    writer.add_uint32("clip.audio.attention.head_count", 1)
    writer.add_float32("clip.audio.attention.layer_norm_epsilon", 1e-5)
    writer.add_uint32("clip.audio.num_mel_bins", 128)
    writer.add_uint32("clip.audio.projector.stack_factor", 15)

    def add(name: str, shape: tuple[int, ...]) -> None:
        writer.add_tensor(name, rng.normal(0.0, 0.2, shape).astype(np.float32))

    add("a.conv1d.1.weight", (4, 128, 3))
    add("a.conv1d.1.bias", (4,))
    add("a.conv1d.2.weight", (4, 4, 3))
    add("a.conv1d.2.bias", (4,))
    add("a.position_embd.weight", (1500, 4))
    add("a.post_ln.weight", (4,))
    add("a.post_ln.bias", (4,))
    prefix = "a.blk.0."
    for norm in ("ln1", "ln2"):
        add(prefix + norm + ".weight", (4,))
        add(prefix + norm + ".bias", (4,))
    for projection in ("attn_q", "attn_k", "attn_v", "attn_out"):
        add(prefix + projection + ".weight", (4, 4))
        if projection != "attn_k":
            add(prefix + projection + ".bias", (4,))
    add(prefix + "ffn_up.weight", (8, 4))
    add(prefix + "ffn_up.bias", (8,))
    add(prefix + "ffn_down.weight", (4, 8))
    add(prefix + "ffn_down.bias", (4,))
    add("mm.a.norm_pre.weight", (4,))
    add("mm.a.norm_pre.bias", (4,))
    add("mm.a.mlp.0.weight", (9, 60))
    add("mm.a.mlp.0.bias", (9,))
    for index in (1, 2):
        add(f"mm.a.mlp.{index}.weight", (9, 9))
        add(f"mm.a.mlp.{index}.bias", (9,))
    add("mm.a.mlp.3.weight", (5, 9))
    add("mm.a.mlp.3.bias", (5,))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_public_standalone_dispatch_builds_and_runs_janus(tmp_path: Path) -> None:
    path = tmp_path / "janus.gguf"
    _write_tiny_janus(path)
    package = build_mmproj_from_gguf(
        path,
        projector_type="janus_pro",
        target_architecture="llama",
        dtype="f32",
    )
    assert set(package) == {"vision_encoder"}
    assert package.gguf_projector_type == "janus_pro"
    assert "not validated" in package.gguf_runtime_warning
    assert package["vision_encoder"].metadata_props["mobius.runtime_support"] == (
        "standalone-sidecar-only; paired multimodal runtime unvalidated"
    )
    input_schema = json.loads(
        package["vision_encoder"].metadata_props["mobius.gguf_input_schema"]
    )
    assert [entry["name"] for entry in input_schema] == ["pixel_values"]

    session = OnnxModelSession(package["vision_encoder"])
    output = session.run(
        {"pixel_values": np.random.default_rng(3).normal(size=(1, 3, 4, 4)).astype(np.float32)}
    )["image_features"]
    session.close()
    assert output.shape == (4, 6)
    assert np.isfinite(output).all()


def test_source_only_minimax_component_builds_with_explicit_positions() -> None:
    metadata = {
        "clip.vision.embedding_length": 160,
        "clip.vision.feed_forward_length": 320,
        "clip.vision.block_count": 1,
        "clip.vision.attention.head_count": 2,
        "clip.vision.attention.layer_norm_epsilon": 1e-5,
        "clip.vision.image_size": 4,
        "clip.vision.patch_size": 2,
        "clip.vision.spatial_merge_size": 2,
        "clip.vision.projection_dim": 64,
    }
    shapes = {
        "v.patch_embd.weight": (160, 3, 2, 2),
        "v.patch_embd.weight.1": (160, 3, 2, 2),
        "mm.1.weight": (192, 160),
        "mm.1.bias": (192,),
        "mm.2.weight": (160, 192),
        "mm.2.bias": (160,),
        "mm.merger.fc1.weight": (256, 640),
        "mm.merger.fc1.bias": (256,),
        "mm.merger.fc2.weight": (64, 256),
        "mm.merger.fc2.bias": (64,),
    }
    module = create_remaining_vision_projector("minimax_m3", metadata, shapes)
    config = ArchitectureConfig(
        model_type="gguf_minimax_m3",
        vocab_size=1,
        hidden_size=160,
        intermediate_size=320,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=80,
        max_position_embeddings=1,
    )
    package = GGUFVisionProjectorTask().build(GGUFVisionProjectorModel(module), config)
    assert set(package) == {"vision_encoder"}
    assert package["vision_encoder"].graph.num_nodes() > 0


def test_public_meralion_package_persists_processor_contract(tmp_path: Path) -> None:
    path = tmp_path / "meralion.gguf"
    _write_tiny_meralion(path)
    package = build_mmproj_from_gguf(
        path,
        projector_type="meralion",
        target_architecture="gemma2",
        dtype="f32",
    )
    assert set(package) == {"audio_encoder"}
    metadata = json.loads(
        package["audio_encoder"].metadata_props["mobius.gguf_audio_processor_abi"]
    )
    assert metadata["sample_rate"] == 16_000
    assert metadata["chunk_seconds"] == 30
    assert metadata["graph_layout"] == "float32[3000,128]"

    session = OnnxModelSession(package["audio_encoder"])
    output = session.run(
        {
            "input_features": np.random.default_rng(5)
            .normal(size=(3000, 128))
            .astype(np.float32)
        }
    )["audio_features"]
    session.close()
    assert output.shape == (100, 5)
    assert (
        package["audio_encoder"].graph.metadata_props["mobius.pipeline.when_present"]
        == "audio"
    )
    assert np.isfinite(output).all()


def test_meralion_default_task_is_registered() -> None:
    assert isinstance(get_task("gguf-audio-projector"), GGUFAudioProjectorTask)


def test_lfm2_contract_accepts_processor_native_int32_mask() -> None:
    metadata = {
        "clip.vision.embedding_length": 4,
        "clip.vision.feed_forward_length": 8,
        "clip.vision.block_count": 1,
        "clip.vision.projection_dim": 4,
        "clip.vision.attention.head_count": 1,
        "clip.vision.attention.layer_norm_epsilon": 1e-6,
        "clip.vision.patch_size": 2,
        "clip.vision.projector.scale_factor": 2,
    }
    module = create_remaining_vision_projector(
        "lfm2",
        metadata,
        {"v.position_embd.weight": (4, 4), "mm.1.weight": (8, 16)},
    )
    assert module.input_schema[1][1] == ir.DataType.INT32


@pytest.mark.parametrize(
    ("projector_type", "source_name", "target_name"),
    [
        ("cogvlm", "v.patch_embd.weight", "vision_encoder.patch_embedding.weight"),
        ("nemotron_v2_vl", "v.patch_embd.weight", "vision_encoder.patch_embedding.weight"),
        ("kimik25", "v.patch_embd.weight", "vision_encoder.patch_embed.proj"),
        ("kimivl", "v.patch_embd.weight", "vision_encoder.patch_embed.proj"),
        ("exaone4_5", "v.patch_embd.weight", "vision_encoder.patch_embed.weight_0"),
        ("mimovl", "v.patch_embd.weight", "vision_encoder.patch_embed.weight_0"),
        ("minimax_m3", "v.patch_embd.weight", "vision_encoder.patch_embed.weight_0"),
    ],
)
def test_production_patch_weight_mapping_preserves_values(
    projector_type: str,
    source_name: str,
    target_name: str,
) -> None:
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2)

    class _Reader:
        def __init__(self) -> None:
            self.tensor_names = (source_name,)
            self.metadata = {"clip.vision.block_count": 1}

        def get_tensor(self, name: str) -> np.ndarray:
            if name == source_name:
                return values
            if name.endswith(".weight"):
                return np.arange(4, dtype=np.float32).reshape(2, 2)
            return np.arange(2, dtype=np.float32)

    state = remaining_projector_state_dict(_Reader(), projector_type)
    assert target_name in state
    np.testing.assert_array_equal(state[target_name].numpy(), values)


def test_minicpm_rejects_unknown_projector_scale() -> None:
    metadata = {
        "clip.vision.embedding_length": 4,
        "clip.vision.projection_dim": 4,
        "clip.vision.patch_size": 2,
        "clip.vision.wa_layer_indexes": [0],
        "clip.vision.projector.scale_factor": 3,
    }
    with pytest.raises(ValueError, match="scale_factor must be 2 or 4"):
        create_remaining_vision_projector(
            "minicpmv4_6",
            metadata,
            {"v.position_embd.weight": (4, 4)},
        )


def test_meralion_factory_and_mapping_preserve_stack_before_norm_contract() -> None:
    metadata = {
        "clip.audio.embedding_length": 4,
        "clip.audio.projection_dim": 5,
        "clip.audio.feed_forward_length": 8,
        "clip.audio.block_count": 1,
        "clip.audio.attention.head_count": 1,
        "clip.audio.attention.layer_norm_epsilon": 1e-5,
        "clip.audio.num_mel_bins": 128,
        "clip.audio.projector.stack_factor": 15,
    }
    shapes = {
        "a.position_embd.weight": (1500, 4),
        "mm.a.mlp.0.weight": (9, 60),
        "mm.a.mlp.1.weight": (9, 9),
        "mm.a.mlp.2.weight": (9, 9),
        "mm.a.mlp.3.weight": (5, 9),
    }
    module = create_gguf_audio_projector("meralion", metadata, shapes)
    assert module.input_schema == (("input_features", ir.DataType.FLOAT, (3000, 128)),)
    assert (
        map_mmproj_audio_projector_to_onnx("mm.a.mlp.3.bias", "meralion")
        == "audio_encoder.projector.linear3.bias"
    )


def test_meralion_rejects_projection_width_metadata_mismatch() -> None:
    metadata = {
        "clip.audio.embedding_length": 4,
        "clip.audio.projection_dim": 6,
        "clip.audio.feed_forward_length": 8,
        "clip.audio.block_count": 1,
        "clip.audio.attention.head_count": 1,
        "clip.audio.attention.layer_norm_epsilon": 1e-5,
        "clip.audio.num_mel_bins": 128,
        "clip.audio.projector.stack_factor": 15,
    }
    shapes = {
        "a.position_embd.weight": (1500, 4),
        "mm.a.mlp.0.weight": (9, 60),
        "mm.a.mlp.1.weight": (9, 9),
        "mm.a.mlp.2.weight": (9, 9),
        "mm.a.mlp.3.weight": (5, 9),
    }
    with pytest.raises(ValueError, match="gated adapter contract"):
        create_gguf_audio_projector("meralion", metadata, shapes)
