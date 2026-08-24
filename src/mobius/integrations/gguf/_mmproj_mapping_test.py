# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for the GGUF ``clip`` mmproj → HF tensor-name mapping."""

from __future__ import annotations

import pytest

from mobius.integrations.gguf._mmproj_mapping import (
    is_mmproj_stat_tensor,
    map_mmproj_audio_to_hf,
    map_mmproj_vision_to_hf,
)


class TestStatTensorDetection:
    @pytest.mark.parametrize(
        "name",
        [
            "v.blk.0.attn_q.weight.input_max",
            "v.blk.0.attn_q.weight.input_min",
            "v.blk.0.attn_q.weight.output_max",
            "v.blk.0.attn_q.weight.output_min",
            "a.blk.3.ffn_up.weight.output_min",
        ],
    )
    def test_stat_tensors_detected(self, name: str):
        assert is_mmproj_stat_tensor(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            "v.blk.0.attn_q.weight",
            "v.patch_embd.weight",
            "mm.input_projection.weight",
            "a.conv1d.0.weight",
        ],
    )
    def test_real_weights_not_flagged(self, name: str):
        assert is_mmproj_stat_tensor(name) is False


class TestVisionMapping:
    @pytest.mark.parametrize(
        ("gguf_name", "expected"),
        [
            (
                "v.blk.0.attn_q.weight",
                "vision_tower.encoder.layers.0.self_attn.q_proj.weight",
            ),
            (
                "v.blk.7.attn_k.weight",
                "vision_tower.encoder.layers.7.self_attn.k_proj.weight",
            ),
            (
                "v.blk.2.attn_v.weight",
                "vision_tower.encoder.layers.2.self_attn.v_proj.weight",
            ),
            (
                "v.blk.2.attn_out.weight",
                "vision_tower.encoder.layers.2.self_attn.o_proj.weight",
            ),
            (
                "v.blk.1.attn_q_norm.weight",
                "vision_tower.encoder.layers.1.self_attn.q_norm.weight",
            ),
            (
                "v.blk.1.attn_k_norm.weight",
                "vision_tower.encoder.layers.1.self_attn.k_norm.weight",
            ),
            (
                "v.blk.0.ln1.weight",
                "vision_tower.encoder.layers.0.input_layernorm.weight",
            ),
            (
                "v.blk.0.ln2.weight",
                "vision_tower.encoder.layers.0.pre_feedforward_layernorm.weight",
            ),
            (
                "v.blk.0.attn_post_norm.weight",
                "vision_tower.encoder.layers.0.post_attention_layernorm.weight",
            ),
            (
                "v.blk.0.ffn_post_norm.weight",
                "vision_tower.encoder.layers.0.post_feedforward_layernorm.weight",
            ),
            (
                "v.blk.4.ffn_gate.weight",
                "vision_tower.encoder.layers.4.mlp.gate_proj.weight",
            ),
            (
                "v.blk.4.ffn_up.weight",
                "vision_tower.encoder.layers.4.mlp.up_proj.weight",
            ),
            (
                "v.blk.4.ffn_down.weight",
                "vision_tower.encoder.layers.4.mlp.down_proj.weight",
            ),
            (
                "v.patch_embd.weight",
                "vision_tower.patch_embedder.input_proj.weight",
            ),
            (
                "v.position_embd.weight",
                "vision_tower.patch_embedder.position_embedding_table",
            ),
            (
                "mm.input_projection.weight",
                "embed_vision.embedding_projection.weight",
            ),
        ],
    )
    def test_vision_names(self, gguf_name: str, expected: str):
        assert map_mmproj_vision_to_hf(gguf_name) == expected

    def test_clipping_bounds_are_mapped(self):
        assert (
            map_mmproj_vision_to_hf("v.blk.0.attn_q.input_max")
            == "vision_tower.encoder.layers.0.self_attn.q_proj.input_max"
        )

    def test_unknown_stem_skipped(self):
        assert map_mmproj_vision_to_hf("v.blk.0.mystery.weight") is None

    def test_audio_tensor_not_mapped_by_vision(self):
        assert map_mmproj_vision_to_hf("a.blk.0.attn_q.weight") is None


class TestAudioMapping:
    @pytest.mark.parametrize(
        ("gguf_name", "expected"),
        [
            (
                "a.blk.0.attn_q.weight",
                "audio_tower.layers.0.self_attn.q_proj.weight",
            ),
            (
                "a.blk.0.per_dim_scale.weight",
                "audio_tower.layers.0.self_attn.per_dim_scale",
            ),
            (
                "a.blk.0.conv_dw.weight",
                "audio_tower.layers.0.lconv1d.depthwise_conv1d.weight",
            ),
            (
                "a.conv1d.0.weight",
                "audio_tower.subsample_conv_projection.conv0.weight",
            ),
            (
                "a.input_projection.weight",
                "audio_tower.subsample_conv_projection.input_proj_linear.weight",
            ),
            (
                "mm.a.input_projection.weight",
                "embed_audio.embedding_projection.weight",
            ),
        ],
    )
    def test_audio_names(self, gguf_name: str, expected: str):
        assert map_mmproj_audio_to_hf(gguf_name) == expected

    def test_stat_tensors_skipped(self):
        assert map_mmproj_audio_to_hf("a.blk.0.attn_q.weight.output_min") is None


class TestMuseGlimmerVisionMapping:
    """Muse Glimmer ``clip`` mmproj → HF vision names.

    Verified against ``unsloth/Muse-Glimmer-30B-GGUF``'s
    ``mmproj-Muse-Glimmer-30B-BF16.gguf``: all 809 tensors map, and the mapped
    names are exactly the 809 parameters of the published vision encoder graph.
    """

    def test_block_tensors_land_on_the_vision_tower_layers(self) -> None:
        from mobius.integrations.gguf._mmproj_mapping import (
            map_mmproj_muse_glimmer_vision_to_hf as convert,
        )

        assert (
            convert("v.blk.7.attn_q.weight")
            == "model.vision_tower.layers.7.attn.q_proj.weight"
        )
        assert convert("v.blk.7.attn_q.bias") == "model.vision_tower.layers.7.attn.q_proj.bias"
        assert (
            convert("v.blk.7.attn_out.weight")
            == "model.vision_tower.layers.7.attn.proj.weight"
        )
        assert convert("v.blk.7.ln1.weight") == "model.vision_tower.layers.7.norm1.weight"
        assert convert("v.blk.7.ln2.bias") == "model.vision_tower.layers.7.norm2.bias"
        assert convert("v.blk.7.ffn_up.weight") == "model.vision_tower.layers.7.mlp.fc1.weight"
        assert convert("v.blk.7.ffn_down.bias") == "model.vision_tower.layers.7.mlp.fc2.bias"

    def test_stem_tensors_and_the_three_projector_matrices(self) -> None:
        from mobius.integrations.gguf._mmproj_mapping import (
            map_mmproj_muse_glimmer_vision_to_hf as convert,
        )

        assert (
            convert("v.patch_embd.weight")
            == "model.vision_tower.patch_embedder.patch_embedding.weight"
        )
        assert (
            convert("v.position_embd.weight")
            == "model.vision_tower.patch_embedder.position_embedding_table.weight"
        )
        assert convert("v.pre_ln.bias") == "model.vision_tower.ln_pre.bias"
        assert convert("v.post_ln.weight") == "model.vision_tower.ln_post.weight"
        # mm.0/mm.1 are the pixel-shuffle adapter, mm.2 the text projection.
        assert convert("mm.0.weight") == "model.vision_adapter.fc1.weight"
        assert convert("mm.1.weight") == "model.vision_adapter.fc2.weight"
        assert convert("mm.2.weight") == "model.vision_projection.weight"

    def test_stats_and_unknown_tensors_are_skipped(self) -> None:
        from mobius.integrations.gguf._mmproj_mapping import (
            map_mmproj_muse_glimmer_vision_to_hf as convert,
        )

        assert convert("v.blk.0.attn_q.input_max") is None
        # Muse Glimmer's tower has no SwiGLU gate and no QK norms; a file
        # carrying them is not this architecture.
        assert convert("v.blk.0.ffn_gate.weight") is None
        assert convert("v.blk.0.attn_q_norm.weight") is None
        assert convert("a.blk.0.attn_q.weight") is None
