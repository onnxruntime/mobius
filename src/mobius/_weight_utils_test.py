# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for _weight_utils.py — shared weight preprocessing utilities."""

from __future__ import annotations

import pytest
import torch

from mobius._weight_utils import (
    infer_compressed_tensors_group_size,
    merge_lora_weights,
    preprocess_awq_weights,
    preprocess_compressed_tensors_weights,
    preprocess_gptq_weights,
    unwrap_gptq_observer_modules,
    preprocess_olive_weights,
    preprocess_quark_weights,
    rename_weight_keys,
    split_codegen_qkv,
    split_fused_qkv,
    split_gate_up_proj,
    split_interleaved_qkv,
    strip_prefix,
    tie_word_embeddings,
    vlm_decoder_weights,
    vlm_embedding_weights,
    vlm_vision_weights,
)


class TestPreprocessCompressedTensorsWeights:
    """Tests for compressed-tensors ``pack-quantized`` import."""

    BITS = 4
    GROUP_SIZE = 32
    N = 2
    K = 64
    N_BLOCKS = K // GROUP_SIZE
    BLOB_SIZE = GROUP_SIZE * BITS // 8

    @staticmethod
    def _packed_weight() -> tuple[torch.Tensor, torch.Tensor]:
        # Signed INT4 -8..7 maps to unsigned codes 0..15. Compressed-tensors
        # stores each earlier code in the lower nibble, exactly like ORT.
        codes = torch.arange(16, dtype=torch.uint8).repeat(4)
        packed_bytes = (codes[0::2] | (codes[1::2] << 4)).repeat(2, 1)
        return packed_bytes.contiguous().view(torch.int32), packed_bytes

    def _state_dict(self) -> dict[str, torch.Tensor]:
        packed, _ = self._packed_weight()
        return {
            "q_proj.weight_packed": packed,
            "q_proj.weight_shape": torch.tensor([self.N, self.K]),
            "q_proj.weight_scale": torch.arange(
                self.N * self.N_BLOCKS, dtype=torch.float16
            ).reshape(self.N, self.N_BLOCKS),
            "q_proj.input_min": torch.tensor(-1.0),
            "q_proj.input_max": torch.tensor(1.0),
            "norm.weight": torch.ones(self.K),
        }

    def test_infers_group_size_from_scale_shape(self):
        assert (
            infer_compressed_tensors_group_size(self._state_dict(), bits=self.BITS)
            == self.GROUP_SIZE
        )

    def test_configured_group_size_allows_partial_final_block(self):
        k = 48
        n_blocks = 2
        state_dict = {
            "q_proj.weight_packed": torch.zeros(
                self.N, k * self.BITS // 32, dtype=torch.int32
            ),
            "q_proj.weight_shape": torch.tensor([self.N, k]),
            "q_proj.weight_scale": torch.ones(self.N, n_blocks),
        }

        assert (
            infer_compressed_tensors_group_size(
                state_dict,
                bits=self.BITS,
                group_size=self.GROUP_SIZE,
            )
            == self.GROUP_SIZE
        )
        assert (
            infer_compressed_tensors_group_size(state_dict, bits=self.BITS) == self.GROUP_SIZE
        )

    def test_repack_preserves_bytes_and_scales(self):
        state_dict = self._state_dict()
        _, expected_bytes = self._packed_weight()

        result = preprocess_compressed_tensors_weights(
            state_dict, bits=self.BITS, group_size=self.GROUP_SIZE
        )

        assert result["q_proj.weight"].shape == (
            self.N,
            self.N_BLOCKS,
            self.BLOB_SIZE,
        )
        assert torch.equal(result["q_proj.weight"].reshape(self.N, -1), expected_bytes)
        assert (
            result["q_proj.scales"].data_ptr() == state_dict["q_proj.weight_scale"].data_ptr()
        )

    def test_drops_shape_and_preserves_clipping_metadata(self):
        result = preprocess_compressed_tensors_weights(
            self._state_dict(), bits=self.BITS, group_size=self.GROUP_SIZE
        )

        assert "q_proj.weight_shape" not in result
        assert "q_proj.input_min" in result
        assert "q_proj.input_max" in result
        assert "norm.weight" in result

    def test_rejects_asymmetric_zero_point(self):
        state_dict = self._state_dict()
        state_dict["q_proj.weight_zero_point"] = torch.zeros(self.N, self.N_BLOCKS)

        with pytest.raises(ValueError, match="Asymmetric"):
            preprocess_compressed_tensors_weights(
                state_dict, bits=self.BITS, group_size=self.GROUP_SIZE
            )

    def test_rejects_activation_order(self):
        state_dict = self._state_dict()
        state_dict["q_proj.weight_g_idx"] = torch.arange(self.K)

        with pytest.raises(ValueError, match="Activation-ordered"):
            preprocess_compressed_tensors_weights(
                state_dict, bits=self.BITS, group_size=self.GROUP_SIZE
            )


class TestSplitFusedQKV:
    """Tests for split_fused_qkv."""

    def test_mha_equal_heads(self):
        """MHA: num_heads == num_kv_heads, all sizes equal."""
        num_heads = 4
        num_kv_heads = 4
        head_dim = 8
        hidden = 16
        total = (num_heads + 2 * num_kv_heads) * head_dim
        weight = torch.arange(total * hidden).reshape(total, hidden).float()

        q, k, v = split_fused_qkv(weight, num_heads, num_kv_heads, head_dim)

        assert q.shape == (num_heads * head_dim, hidden)
        assert k.shape == (num_kv_heads * head_dim, hidden)
        assert v.shape == (num_kv_heads * head_dim, hidden)
        # Values are contiguous slices of the original
        torch.testing.assert_close(q, weight[: num_heads * head_dim])

    def test_gqa_fewer_kv_heads(self):
        """GQA: num_kv_heads < num_heads."""
        num_heads = 8
        num_kv_heads = 2
        head_dim = 4
        q_size = num_heads * head_dim  # 32
        kv_size = num_kv_heads * head_dim  # 8
        total = q_size + 2 * kv_size  # 48
        weight = torch.randn(total)

        q, k, v = split_fused_qkv(weight, num_heads, num_kv_heads, head_dim)

        assert q.shape == (q_size,)
        assert k.shape == (kv_size,)
        assert v.shape == (kv_size,)
        # Verify the splits reconstruct the original
        torch.testing.assert_close(torch.cat([q, k, v]), weight)

    def test_mqa_single_kv_head(self):
        """MQA: num_kv_heads == 1."""
        num_heads = 16
        num_kv_heads = 1
        head_dim = 64
        hidden = 32
        total = (num_heads + 2) * head_dim
        weight = torch.randn(total, hidden)

        q, k, v = split_fused_qkv(weight, num_heads, num_kv_heads, head_dim)

        assert q.shape == (num_heads * head_dim, hidden)
        assert k.shape == (head_dim, hidden)
        assert v.shape == (head_dim, hidden)

    def test_splits_are_views(self):
        """Splits should be views of the original tensor (no copy)."""
        weight = torch.randn(32, 16)
        q, _k, _v = split_fused_qkv(weight, 4, 2, 4)
        assert q.data_ptr() == weight.data_ptr()

    def test_1d_bias_tensor(self):
        """Works with 1D bias tensors too."""
        # num_heads=4, num_kv_heads=2, head_dim=4 → total = 16+8+8 = 32
        bias = torch.randn(32)
        q, k, v = split_fused_qkv(bias, 4, 2, 4)
        assert q.shape == (16,)
        assert k.shape == (8,)
        assert v.shape == (8,)


class TestSplitGateUpProj:
    """Tests for split_gate_up_proj."""

    def test_basic_split(self):
        """Splits a 2D weight at intermediate_size."""
        intermediate_size = 64
        hidden = 32
        weight = torch.randn(2 * intermediate_size, hidden)

        gate, up = split_gate_up_proj(weight, intermediate_size)

        assert gate.shape == (intermediate_size, hidden)
        assert up.shape == (intermediate_size, hidden)
        torch.testing.assert_close(torch.cat([gate, up]), weight)

    def test_1d_bias(self):
        """Works with 1D bias tensors."""
        intermediate_size = 128
        bias = torch.randn(2 * intermediate_size)

        gate, up = split_gate_up_proj(bias, intermediate_size)

        assert gate.shape == (intermediate_size,)
        assert up.shape == (intermediate_size,)

    def test_splits_are_views(self):
        """Splits should be views of the original tensor."""
        weight = torch.randn(256, 64)
        gate, _up = split_gate_up_proj(weight, 128)
        assert gate.data_ptr() == weight.data_ptr()


class TestStripPrefix:
    """Tests for strip_prefix."""

    def test_strips_prefix(self):
        """Basic prefix stripping."""
        state_dict = {
            "model.layers.0.weight": torch.tensor(1.0),
            "model.layers.1.weight": torch.tensor(2.0),
            "model.embed.weight": torch.tensor(3.0),
        }
        result = strip_prefix(state_dict, "model")
        assert set(result.keys()) == {
            "layers.0.weight",
            "layers.1.weight",
            "embed.weight",
        }

    def test_drops_non_matching_keys(self):
        """Keys without the prefix are excluded."""
        state_dict = {
            "model.weight": torch.tensor(1.0),
            "other.weight": torch.tensor(2.0),
        }
        result = strip_prefix(state_dict, "model")
        assert list(result.keys()) == ["weight"]

    def test_trailing_dot_optional(self):
        """Prefix with or without trailing dot gives same result."""
        state_dict = {"foo.bar": torch.tensor(1.0)}
        result_no_dot = strip_prefix(state_dict, "foo")
        result_with_dot = strip_prefix(state_dict, "foo.")
        assert result_no_dot == result_with_dot

    def test_empty_state_dict(self):
        """Empty input returns empty output."""
        result = strip_prefix({}, "model")
        assert result == {}

    def test_preserves_tensor_values(self):
        """Tensor values are preserved (not copied)."""
        t = torch.randn(3, 4)
        result = strip_prefix({"prefix.key": t}, "prefix")
        assert result["key"].data_ptr() == t.data_ptr()


class TestRenameWeightKeys:
    """Tests for rename_weight_keys."""

    def test_applies_replacements(self):
        """Each (old, new) pair is applied to every key."""
        state_dict = {
            "model.layers.0.attention_layernorm.weight": torch.tensor(1.0),
            "model.layers.0.feedforward_layernorm.weight": torch.tensor(2.0),
            "model.embed.weight": torch.tensor(3.0),
        }
        result = rename_weight_keys(
            state_dict,
            [
                (".attention_layernorm.", ".input_layernorm."),
                (".feedforward_layernorm.", ".post_attention_layernorm."),
            ],
        )
        assert set(result.keys()) == {
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.embed.weight",
        }

    def test_replacements_are_ordered_and_cascade(self):
        """Replacements apply to the progressively updated key, in order."""
        state_dict = {"a.weight": torch.tensor(1.0)}
        result = rename_weight_keys(state_dict, [("a.", "b."), ("b.", "c.")])
        assert list(result.keys()) == ["c.weight"]

    def test_no_replacements_is_identity(self):
        """An empty replacement list returns a key-equivalent copy."""
        state_dict = {"x.weight": torch.tensor(1.0)}
        result = rename_weight_keys(state_dict, [])
        assert list(result.keys()) == ["x.weight"]
        assert result is not state_dict

    def test_shares_tensor_values(self):
        """Tensor values are shared, not cloned."""
        t = torch.randn(2, 3)
        result = rename_weight_keys({"old.k": t}, [("old.", "new.")])
        assert result["new.k"].data_ptr() == t.data_ptr()

    def test_collision_raises(self):
        """Two source keys mapping to the same renamed key raises.

        The error message must name both the colliding key and the original
        producer key to aid debugging in large checkpoints.
        """
        state_dict = {
            "a.weight": torch.tensor(1.0),
            "b.weight": torch.tensor(2.0),
        }
        with pytest.raises(ValueError, match="collision") as exc_info:
            rename_weight_keys(state_dict, [("a.", "x."), ("b.", "x.")])
        message = str(exc_info.value)
        # The producer ("a.weight", processed first) and the colliding key
        # ("b.weight") must both appear.
        assert "a.weight" in message
        assert "b.weight" in message

    def test_empty_state_dict(self):
        """Empty input returns empty output."""
        assert rename_weight_keys({}, [("a", "b")]) == {}


class TestSplitFusedQkvValidation:
    """Tests for QKV dimension mismatch detection."""

    def test_wrong_dim_raises(self):
        weight = torch.randn(100, 64)  # wrong size
        with pytest.raises(ValueError, match="QKV weight dim 0 is 100"):
            split_fused_qkv(weight, num_heads=8, num_kv_heads=2, head_dim=8)

    def test_too_small_raises(self):
        # Expected: 8*8 + 2*2*8 = 96, but provide 80
        weight = torch.randn(80, 64)
        with pytest.raises(ValueError, match="expected 96"):
            split_fused_qkv(weight, num_heads=8, num_kv_heads=2, head_dim=8)


class TestSplitGateUpProjValidation:
    """Tests for gate_up dimension mismatch detection."""

    def test_wrong_dim_raises(self):
        weight = torch.randn(100, 64)  # wrong size
        with pytest.raises(ValueError, match="gate_up weight dim 0 is 100"):
            split_gate_up_proj(weight, intermediate_size=64)

    def test_odd_size_raises(self):
        # Expected: 2*32 = 64, but provide 63
        weight = torch.randn(63, 16)
        with pytest.raises(ValueError, match="expected 64"):
            split_gate_up_proj(weight, intermediate_size=32)


class TestTieWordEmbeddings:
    """Tests for tie_word_embeddings."""

    def test_copies_embed_to_head(self):
        """If lm_head is missing, copies embed_tokens."""
        t = torch.randn(4, 8)
        sd = {"model.embed_tokens.weight": t}
        tie_word_embeddings(sd)
        assert "lm_head.weight" in sd
        assert sd["lm_head.weight"].data_ptr() == t.data_ptr()

    def test_copies_head_to_embed(self):
        """If embed_tokens is missing, copies lm_head."""
        t = torch.randn(4, 8)
        sd = {"lm_head.weight": t}
        tie_word_embeddings(sd)
        assert "model.embed_tokens.weight" in sd
        assert sd["model.embed_tokens.weight"].data_ptr() == t.data_ptr()

    def test_both_present_no_change(self):
        """If both present, no change."""
        t1 = torch.randn(4, 8)
        t2 = torch.randn(4, 8)
        sd = {"model.embed_tokens.weight": t1, "lm_head.weight": t2}
        tie_word_embeddings(sd)
        assert sd["model.embed_tokens.weight"].data_ptr() == t1.data_ptr()
        assert sd["lm_head.weight"].data_ptr() == t2.data_ptr()

    def test_neither_present_raises(self):
        """Raise ValueError if neither key is found (catches key mismatches)."""
        import pytest

        sd = {"other.weight": torch.randn(4)}
        with pytest.raises(ValueError, match=r"neither.*found"):
            tie_word_embeddings(sd)

    def test_custom_keys(self):
        """Custom embed/head keys."""
        t = torch.randn(4, 8)
        sd = {"enc.embed.weight": t}
        tie_word_embeddings(sd, embed_key="enc.embed.weight", head_key="dec.head.weight")
        assert "dec.head.weight" in sd


class TestVlmDecoderWeights:
    """Tests for vlm_decoder_weights."""

    def test_strips_prefix_and_ties(self):
        """Strips language_model. prefix and ties embeddings."""
        embed = torch.randn(4, 8)
        sd = {
            "language_model.model.layers.0.weight": torch.randn(4),
            "language_model.model.embed_tokens.weight": embed,
            "language_model.model.norm.weight": torch.randn(4),
            "vision_model.encoder.weight": torch.randn(4),
        }
        result = vlm_decoder_weights(sd, tie=True)
        assert "model.layers.0.weight" in result
        assert "model.embed_tokens.weight" in result
        assert "lm_head.weight" in result
        # Vision weights are excluded
        assert not any("vision" in k for k in result)

    def test_no_tie(self):
        """Without tie, lm_head is not added."""
        sd = {
            "language_model.model.embed_tokens.weight": torch.randn(4, 8),
        }
        result = vlm_decoder_weights(sd, tie=False)
        assert "lm_head.weight" not in result

    def test_returns_new_dict(self):
        """Returns a new dict, doesn't modify input."""
        sd = {"language_model.w": torch.randn(4)}
        result = vlm_decoder_weights(sd)
        assert "w" in result
        assert "language_model.w" in sd  # original unchanged

    def test_custom_prefix(self):
        """Works with a non-default prefix."""
        sd = {"decoder.layers.0.weight": torch.randn(4)}
        result = vlm_decoder_weights(sd, prefix="decoder.")
        assert "layers.0.weight" in result


class TestVlmEmbeddingWeights:
    """Tests for vlm_embedding_weights."""

    def test_filters_and_strips(self):
        """Filters embed_tokens and strips prefixes."""
        t = torch.randn(4, 8)
        sd = {
            "language_model.model.embed_tokens.weight": t,
            "language_model.model.layers.0.weight": torch.randn(4),
            "vision_model.encoder.weight": torch.randn(4),
        }
        result = vlm_embedding_weights(sd)
        assert list(result.keys()) == ["embed_tokens.weight"]
        assert result["embed_tokens.weight"].data_ptr() == t.data_ptr()

    def test_strips_shorter_prefix(self):
        """Falls through to shorter prefix."""
        t = torch.randn(4, 8)
        sd = {"language_model.embed_tokens.weight": t}
        result = vlm_embedding_weights(sd)
        assert list(result.keys()) == ["embed_tokens.weight"]

    def test_no_prefix_match(self):
        """Keys without matching prefix kept as-is."""
        t = torch.randn(4, 8)
        sd = {"embed_tokens.weight": t}
        result = vlm_embedding_weights(sd)
        assert list(result.keys()) == ["embed_tokens.weight"]

    def test_empty_when_no_keyword(self):
        """Returns empty dict if no keys match keyword."""
        sd = {"language_model.layers.0.weight": torch.randn(4)}
        result = vlm_embedding_weights(sd)
        assert result == {}

    def test_custom_keyword(self):
        """Custom keyword filter."""
        t = torch.randn(4, 8)
        sd = {
            "model.word_embedding.weight": t,
            "model.layers.0.weight": torch.randn(4),
        }
        result = vlm_embedding_weights(sd, keyword="word_embedding", prefixes=("model.",))
        assert list(result.keys()) == ["word_embedding.weight"]


class TestVlmVisionWeights:
    """Tests for vlm_vision_weights."""

    def test_filters_and_renames(self):
        """Keeps prefixed keys and renames fc1/fc2."""
        fc1 = torch.randn(4, 8)
        fc2 = torch.randn(8, 4)
        sd = {
            "vision_tower.encoder.layers.0.mlp.fc1.weight": fc1,
            "vision_tower.encoder.layers.0.mlp.fc2.weight": fc2,
            "multi_modal_projector.linear.weight": torch.randn(4),
            "language_model.model.layers.0.weight": torch.randn(4),
        }
        result = vlm_vision_weights(sd, ("vision_tower.", "multi_modal_projector."))
        assert set(result.keys()) == {
            "vision_tower.encoder.layers.0.mlp.up_proj.weight",
            "vision_tower.encoder.layers.0.mlp.down_proj.weight",
            "multi_modal_projector.linear.weight",
        }
        assert result["vision_tower.encoder.layers.0.mlp.up_proj.weight"].data_ptr() == (
            fc1.data_ptr()
        )

    def test_single_prefix(self):
        """Works with a single-element prefix tuple."""
        sd = {
            "vision_model.layers.0.mlp.fc1.weight": torch.randn(2),
            "other.weight": torch.randn(2),
        }
        result = vlm_vision_weights(sd, ("vision_model.",))
        assert list(result.keys()) == ["vision_model.layers.0.mlp.up_proj.weight"]

    def test_empty_when_no_match(self):
        """Returns empty dict when no key matches the prefixes."""
        sd = {"language_model.layers.0.weight": torch.randn(2)}
        assert vlm_vision_weights(sd, ("vision_tower.",)) == {}


class TestPreprocessGptqWeights:
    """Tests for GPTQ weight preprocessing.

    Uses realistic GPTQ shapes for INT4, group_size=32:
    K=256, N=128 → K_packed=32, n_groups=8, blob_size=16.
    """

    K = 256
    N = 128
    BITS = 4
    GROUP_SIZE = 32
    K_PACKED = K * BITS // 32  # 32
    N_GROUPS = K // GROUP_SIZE  # 8
    BLOB_SIZE = GROUP_SIZE * BITS // 8  # 16
    # qzeros packs (32 // BITS)=8 *output channels* per int32, so the
    # packed axis is N and the group axis stays unpacked.
    N_PACKED = N * BITS // 32  # 16

    def test_qweight_renamed_to_weight(self):
        sd = {
            "q_proj.qweight": torch.randint(0, 255, (self.K_PACKED, self.N), dtype=torch.int32)
        }
        result = preprocess_gptq_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)
        assert "q_proj.weight" in result
        assert "q_proj.qweight" not in result

    def test_qweight_shape_3d(self):
        """Weight must be [N, n_blocks, blob_size] for MatMulNBits."""
        sd = {
            "q_proj.qweight": torch.randint(0, 255, (self.K_PACKED, self.N), dtype=torch.int32)
        }
        result = preprocess_gptq_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)
        w = result["q_proj.weight"]
        assert w.shape == (self.N, self.N_GROUPS, self.BLOB_SIZE)
        assert w.dtype == torch.uint8

    def test_qzeros_renamed_to_zero_points(self):
        sd = {
            "q_proj.qweight": torch.randint(
                0, 255, (self.K_PACKED, self.N), dtype=torch.int32
            ),
            "q_proj.qzeros": torch.randint(
                0, 255, (self.N_GROUPS, self.N_PACKED), dtype=torch.int32
            ),
        }
        result = preprocess_gptq_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)
        assert "q_proj.zero_points" in result
        assert "q_proj.qzeros" not in result

    def test_g_idx_dropped(self):
        sd = {
            "q_proj.g_idx": torch.arange(self.K),
            "q_proj.scales": torch.randn(self.N_GROUPS, self.N),
        }
        result = preprocess_gptq_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)
        assert not any(k.endswith(".g_idx") for k in result)
        assert "q_proj.scales" in result

    def test_g_idx_nontrivial_warns(self, caplog):
        """Non-trivial g_idx (desc_act) should emit a warning."""
        import logging

        sd = {
            "q_proj.g_idx": torch.tensor([7, 3, 0, 1]),  # not sequential
        }
        with caplog.at_level(logging.WARNING):
            result = preprocess_gptq_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)
        assert not any(k.endswith(".g_idx") for k in result)
        assert "desc_act" in caplog.text

    def test_scales_transposed(self):
        sd = {"q_proj.scales": torch.randn(self.N_GROUPS, self.N)}
        result = preprocess_gptq_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)
        assert result["q_proj.scales"].shape == (self.N, self.N_GROUPS)

    def test_observer_wrapped_linear_weights_are_unwrapped(self):
        """Unquantized layers keep a `.linear.` infix that must be stripped.

        GPTQModel wraps targeted `nn.Linear` modules in an observer. Layers
        it leaves in floating point end up as `<path>.linear.weight`. That
        name does not exist in the built graph, so without unwrapping they
        never bind and the component silently runs on uninitialized weights.
        """
        weight = torch.randn(8, 8)
        sd = {
            "vision_tower.layers.0.mlp.down_proj.linear.weight": weight,
            "vision_tower.layers.0.mlp.down_proj.linear.bias": torch.randn(8),
            "vision_tower.layers.0.mlp.down_proj.input_min": torch.tensor(0.0),
        }
        result = unwrap_gptq_observer_modules(sd)

        # Infix stripped; activation observers preserved because the built
        # graph declares initializers for them.
        assert set(result) == {
            "vision_tower.layers.0.mlp.down_proj.weight",
            "vision_tower.layers.0.mlp.down_proj.bias",
            "vision_tower.layers.0.mlp.down_proj.input_min",
        }
        torch.testing.assert_close(
            result["vision_tower.layers.0.mlp.down_proj.weight"], weight
        )

    def test_qzeros_round_trip_applies_gptq_plus_one_offset(self):
        """Zero points must be transposed, repacked, and biased by +1.

        GPTQ packs zero points along N while MatMulNBits packs them along
        the block axis, so the conversion has to unpack, transpose and
        repack rather than reinterpret the int32 buffer as bytes.

        GPTQ also stores ``zero - 1``: its dequantization is
        ``scale * (q - (z + 1))``. MatMulNBits applies
        ``scale * (q - zero_point)`` with no bias of its own, so the +1 must
        be folded into the stored zero point here.
        """
        pack_factor = 32 // self.BITS
        maxq = (1 << self.BITS) - 1
        torch.manual_seed(0)
        zeros = torch.randint(0, 2**self.BITS, (self.N_GROUPS, self.N), dtype=torch.int32)

        # Pack along N exactly as GPTQModel's packer does.
        qzeros = torch.zeros((self.N_GROUPS, self.N_PACKED), dtype=torch.int32)
        for col in range(self.N_PACKED):
            for j in range(pack_factor):
                qzeros[:, col] |= zeros[:, col * pack_factor + j] << (self.BITS * j)

        sd = {
            "q_proj.qweight": torch.randint(
                0, 255, (self.K_PACKED, self.N), dtype=torch.int32
            ),
            "q_proj.qzeros": qzeros,
        }
        result = preprocess_gptq_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)
        packed = result["q_proj.zero_points"]

        zp_cols = self.N_GROUPS * self.BITS // 8
        assert packed.shape == (self.N, zp_cols)
        assert packed.dtype == torch.uint8

        per_byte = 8 // self.BITS
        recovered = torch.zeros((self.N, self.N_GROUPS), dtype=torch.int32)
        for n in range(self.N):
            for g in range(self.N_GROUPS):
                byte = int(packed[n, g // per_byte])
                recovered[n, g] = (byte >> (self.BITS * (g % per_byte))) & maxq

        expected = (zeros + 1).clamp(0, maxq).transpose(0, 1).contiguous()
        torch.testing.assert_close(recovered, expected)

    def test_expert_major_tensors_preserve_expert_axis(self):
        num_experts = 64
        sd = {
            "experts.qweight": torch.randint(
                0,
                255,
                (num_experts, self.K_PACKED, self.N),
                dtype=torch.int32,
            ),
            "experts.qzeros": torch.randint(
                0,
                255,
                (num_experts, self.N_GROUPS, self.N_PACKED),
                dtype=torch.int32,
            ),
            "experts.scales": torch.randn(num_experts, self.N_GROUPS, self.N),
        }
        result = preprocess_gptq_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)

        assert result["experts.weight"].shape == (
            num_experts,
            self.N,
            self.N_GROUPS,
            self.BLOB_SIZE,
        )
        assert result["experts.zero_points"].shape == (
            num_experts,
            self.N,
            self.N_GROUPS * self.BITS // 8,
        )
        assert result["experts.scales"].shape == (
            num_experts,
            self.N,
            self.N_GROUPS,
        )

    def test_non_gptq_keys_pass_through(self):
        t = torch.randn(4, 8)
        sd = {"model.embed_tokens.weight": t, "lm_head.weight": t.clone()}
        result = preprocess_gptq_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)
        assert "model.embed_tokens.weight" in result
        assert "lm_head.weight" in result
        assert torch.equal(result["model.embed_tokens.weight"], t)

    def test_missing_qweight_raises(self):
        """Qzeros without matching qweight raises ValueError."""
        sd = {"q_proj.qzeros": torch.zeros(1, self.N, dtype=torch.int32)}
        with pytest.raises(ValueError, match=r"Missing q_proj\.qweight"):
            preprocess_gptq_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)


class TestPreprocessOliveWeights:
    """Tests for Olive uint8-packed weight preprocessing."""

    K = 256
    N = 128
    BITS = 4
    GROUP_SIZE = 32
    PACKED_K = K * BITS // 8
    N_BLOCKS = K // GROUP_SIZE
    BLOB_SIZE = GROUP_SIZE * BITS // 8

    def test_qweight_renamed_and_reshaped_to_matmulnbits_weight(self):
        sd = {
            "q_proj.qweight": torch.randint(0, 255, (self.N, self.PACKED_K), dtype=torch.uint8)
        }

        result = preprocess_olive_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)

        assert "q_proj.weight" in result
        assert "q_proj.qweight" not in result
        assert result["q_proj.weight"].shape == (self.N, self.N_BLOCKS, self.BLOB_SIZE)
        assert result["q_proj.weight"].dtype == torch.uint8

    def test_scales_orientation_is_preserved(self):
        scales = torch.randn(self.N, self.N_BLOCKS)

        result = preprocess_olive_weights(
            {"q_proj.scales": scales}, bits=self.BITS, group_size=self.GROUP_SIZE
        )

        assert result["q_proj.scales"] is scales

    def test_qzeros_renamed_to_zero_points_without_reshape(self):
        qzeros = torch.randint(0, 255, (self.N, self.N_BLOCKS // 2), dtype=torch.uint8)

        result = preprocess_olive_weights(
            {"q_proj.qzeros": qzeros}, bits=self.BITS, group_size=self.GROUP_SIZE
        )

        assert "q_proj.zero_points" in result
        assert "q_proj.qzeros" not in result
        assert torch.equal(result["q_proj.zero_points"], qzeros)

    def test_non_quantized_keys_pass_through(self):
        weight = torch.randn(4, 8)

        result = preprocess_olive_weights(
            {"model.embed_tokens.weight": weight}, bits=self.BITS, group_size=self.GROUP_SIZE
        )

        assert result["model.embed_tokens.weight"] is weight

    def test_non_uint8_qweight_raises(self):
        sd = {
            "q_proj.qweight": torch.randint(0, 255, (self.N, self.PACKED_K), dtype=torch.int32)
        }

        with pytest.raises(ValueError, match="Olive qweight must be uint8"):
            preprocess_olive_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)

    def test_non_uint8_qzeros_raises(self):
        sd = {
            "q_proj.qzeros": torch.randint(
                0, 255, (self.N, self.N_BLOCKS // 2), dtype=torch.int32
            )
        }
        with pytest.raises(ValueError, match="Olive qzeros must be uint8"):
            preprocess_olive_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)

    # --- Quantized embedding (GatherBlockQuantized) ---

    V = 64  # vocab size for embedding tests

    def test_embed_qweight_kept_2d(self):
        """embed_tokens.qweight targets GatherBlockQuantized: keep 2-D uint8."""
        qweight = torch.randint(0, 255, (self.V, self.PACKED_K), dtype=torch.uint8)
        result = preprocess_olive_weights(
            {"model.embed_tokens.qweight": qweight},
            bits=self.BITS,
            group_size=self.GROUP_SIZE,
            quantize_embeddings=True,
        )
        out = result["model.embed_tokens.qweight"]
        assert out.shape == (self.V, self.PACKED_K)
        assert out.dtype == torch.uint8

    def test_embed_qzeros_renamed_to_zero_points(self):
        qzeros = torch.randint(0, 255, (self.V, self.N_BLOCKS // 2), dtype=torch.uint8)
        result = preprocess_olive_weights(
            {"model.embed_tokens.qzeros": qzeros},
            bits=self.BITS,
            group_size=self.GROUP_SIZE,
            quantize_embeddings=True,
        )
        assert "model.embed_tokens.zero_points" in result
        assert "model.embed_tokens.qzeros" not in result

    def test_embed_non_uint8_qweight_raises(self):
        sd = {
            "model.embed_tokens.qweight": torch.randint(
                0, 255, (self.V, self.PACKED_K), dtype=torch.int32
            )
        }
        with pytest.raises(ValueError, match="Olive embedding qweight must be uint8"):
            preprocess_olive_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)

    def test_embed_non_uint8_qzeros_raises(self):
        sd = {
            "model.embed_tokens.qzeros": torch.randint(
                0, 255, (self.V, self.N_BLOCKS // 2), dtype=torch.int32
            )
        }
        with pytest.raises(ValueError, match="Olive embedding qzeros must be uint8"):
            preprocess_olive_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)

    # --- Tied LM head synthesis ---

    def test_tied_quantized_head_produces_no_lm_head_keys(self):
        """A tied *quantized* head shares the embed Parameters in the module.

        So preprocessing must NOT emit duplicate lm_head.* initializers.
        """
        qweight = torch.randint(0, 255, (self.V, self.PACKED_K), dtype=torch.uint8)
        scales = torch.randn(self.V, self.N_BLOCKS)
        qzeros = torch.randint(0, 255, (self.V, self.N_BLOCKS // 2), dtype=torch.uint8)
        result = preprocess_olive_weights(
            {
                "model.embed_tokens.qweight": qweight,
                "model.embed_tokens.scales": scales,
                "model.embed_tokens.qzeros": qzeros,
            },
            bits=self.BITS,
            group_size=self.GROUP_SIZE,
            quantize_embeddings=True,
            quantize_lm_head=True,
            tie_word_embeddings=True,
        )
        assert not any(k.startswith("lm_head.") for k in result)
        assert "model.embed_tokens.qweight" in result
        assert "model.embed_tokens.zero_points" in result

    def test_float_tie_fallback(self):
        """Unquantized embed + tie -> lm_head shares the float embedding table."""
        embed = torch.randn(self.V, self.K)
        result = preprocess_olive_weights(
            {"model.embed_tokens.weight": embed},
            bits=self.BITS,
            group_size=self.GROUP_SIZE,
            tie_word_embeddings=True,
        )
        assert result["lm_head.weight"] is embed

    def test_untied_quantized_lm_head_not_overwritten(self):
        """A present lm_head.qweight must not be replaced by embed synthesis."""
        lm_head = torch.randint(0, 255, (self.V, self.PACKED_K), dtype=torch.uint8)
        embed = torch.randint(0, 255, (self.V, self.PACKED_K), dtype=torch.uint8)
        result = preprocess_olive_weights(
            {
                "lm_head.qweight": lm_head,
                "model.embed_tokens.qweight": embed,
            },
            bits=self.BITS,
            group_size=self.GROUP_SIZE,
            quantize_embeddings=True,
            quantize_lm_head=True,
            tie_word_embeddings=False,
        )
        assert torch.equal(
            result["lm_head.weight"],
            lm_head.reshape(self.V, -1, self.BLOB_SIZE),
        )


class TestPreprocessAwqWeights:
    """Tests for AWQ weight preprocessing.

    Uses same realistic shapes as GPTQ tests: INT4, group_size=32,
    K=256, N=128.
    """

    K = 256
    N = 128
    BITS = 4
    GROUP_SIZE = 32
    K_PACKED = K * BITS // 32  # 32
    N_GROUPS = K // GROUP_SIZE  # 8
    BLOB_SIZE = GROUP_SIZE * BITS // 8  # 16
    N_GROUPS_PACKED = N_GROUPS * BITS // 32  # 1
    ZERO_POINT_COLUMNS = (N_GROUPS * BITS + 7) // 8  # 4

    def test_qweight_renamed_to_weight(self):
        sd = {
            "q_proj.qweight": torch.randint(0, 255, (self.K_PACKED, self.N), dtype=torch.int32)
        }
        result = preprocess_awq_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)
        assert "q_proj.weight" in result
        assert "q_proj.qweight" not in result

    def test_qweight_shape_3d(self):
        """Weight must be [N, n_blocks, blob_size] for MatMulNBits."""
        sd = {
            "q_proj.qweight": torch.randint(0, 255, (self.K_PACKED, self.N), dtype=torch.int32)
        }
        result = preprocess_awq_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)
        w = result["q_proj.weight"]
        assert w.shape == (self.N, self.N_GROUPS, self.BLOB_SIZE)
        assert w.dtype == torch.uint8

    def test_qzeros_renamed_to_zero_points(self):
        sd = {
            "q_proj.qweight": torch.randint(
                0, 255, (self.K_PACKED, self.N), dtype=torch.int32
            ),
            "q_proj.qzeros": torch.randint(
                0,
                255,
                (self.N_GROUPS_PACKED, self.N),
                dtype=torch.int32,
            ),
        }
        result = preprocess_awq_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)
        assert "q_proj.zero_points" in result
        assert "q_proj.qzeros" not in result
        assert result["q_proj.zero_points"].shape == (
            self.N,
            self.ZERO_POINT_COLUMNS,
        )

    def test_zero_point_offset_subtracted(self):
        """AWQ zero points have +1 offset that must be subtracted."""
        # 0x55555555 sets every 4-bit nibble to 5. (0x05050505 would only
        # set alternating nibbles, since one byte holds two zero points.)
        sd = {
            "q_proj.qweight": torch.randint(
                0, 255, (self.K_PACKED, self.N), dtype=torch.int32
            ),
            "q_proj.qzeros": torch.full(
                (self.N_GROUPS_PACKED, self.N),
                0x55555555,
                dtype=torch.int32,
            ),
        }
        result = preprocess_awq_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)
        zp = result["q_proj.zero_points"]
        # Every zero point was 5; after the -1 offset each nibble is 4, so
        # each output byte packs two 4s as 0x44.
        assert (zp == 0x44).all()
        assert ((zp & 0x0F) == 4).all()
        assert (((zp >> 4) & 0x0F) == 4).all()

    def test_no_g_idx_handling(self):
        """AWQ does not have g_idx — unknown keys should pass through."""
        sd = {
            "q_proj.scales": torch.randn(self.N_GROUPS, self.N),
            "q_proj.some_extra": torch.randn(4),
        }
        result = preprocess_awq_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)
        assert "q_proj.some_extra" in result

    def test_scales_transposed(self):
        sd = {"q_proj.scales": torch.randn(self.N_GROUPS, self.N)}
        result = preprocess_awq_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)
        assert result["q_proj.scales"].shape == (self.N, self.N_GROUPS)

    def test_same_qweight_reshape_as_gptq(self):
        """AWQ and GPTQ should produce identical qweight reshapes."""
        qweight = torch.randint(0, 255, (self.K_PACKED, self.N), dtype=torch.int32)
        gptq_result = preprocess_gptq_weights(
            {"p.qweight": qweight.clone(), "p.scales": torch.randn(self.N_GROUPS, self.N)},
            bits=self.BITS,
            group_size=self.GROUP_SIZE,
        )
        awq_result = preprocess_awq_weights(
            {"p.qweight": qweight.clone(), "p.scales": torch.randn(self.N_GROUPS, self.N)},
            bits=self.BITS,
            group_size=self.GROUP_SIZE,
        )
        assert torch.equal(gptq_result["p.weight"], awq_result["p.weight"])

    def test_zero_valued_qzeros_no_underflow(self):
        """When raw qzeros bytes are 0, subtraction must not underflow uint8."""
        sd = {
            "q_proj.qweight": torch.randint(
                0, 255, (self.K_PACKED, self.N), dtype=torch.int32
            ),
            "q_proj.qzeros": torch.zeros(
                self.N_GROUPS_PACKED, self.N, dtype=torch.int32
            ),
        }
        result = preprocess_awq_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)
        zp = result["q_proj.zero_points"]
        # 0 - 1 clamped to 0; no uint8 wrap-around to 255
        assert zp.dtype == torch.uint8
        assert (zp == 0).all()

    def test_zero_point_offset_per_nibble(self):
        """AWQ -1 offset must apply per-nibble, not per-byte.

        0x88 = low nibble 8, high nibble 8.  After per-nibble -1:
        low = 7, high = 7 → repacked = 0x77.
        A byte-level subtract would give 0x87 (wrong).
        """
        sd = {
            "q_proj.qweight": torch.randint(
                0, 255, (self.K_PACKED, self.N), dtype=torch.int32
            ),
            "q_proj.qzeros": torch.full(
                (self.N_GROUPS_PACKED, self.N),
                -2004318072,  # 0x88888888 as signed int32
                dtype=torch.int32,
            ),
        }
        result = preprocess_awq_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)
        zp = result["q_proj.zero_points"]
        # Both nibbles decremented: 0x88 → 0x77
        assert (zp == 0x77).all()

    def test_missing_qweight_raises(self):
        """Qzeros without matching qweight raises ValueError."""
        sd = {"q_proj.qzeros": torch.zeros(1, self.N, dtype=torch.int32)}
        with pytest.raises(ValueError, match=r"Missing q_proj\.qweight"):
            preprocess_awq_weights(sd, bits=self.BITS, group_size=self.GROUP_SIZE)


class TestPreprocessQuarkWeights:
    BITS = 4
    GROUP_SIZE = 32
    K = 64
    N = 32
    GROUPS = K // GROUP_SIZE

    @staticmethod
    def _pack_int32_last(values: torch.Tensor) -> torch.Tensor:
        order = [0, 2, 4, 6, 1, 3, 5, 7]
        values = values.reshape(*values.shape[:-1], -1, 8)[..., order].flatten(-2)
        packed = torch.zeros(*values.shape[:-1], values.shape[-1] // 8, dtype=torch.int32)
        for index in range(8):
            packed |= values[..., index::8].to(torch.int32) << (index * 4)
        return packed

    def test_converts_quark_native_layout(self):
        codes = torch.arange(self.K * self.N, dtype=torch.uint8).reshape(self.K, self.N) % 16
        zero_codes = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8).repeat(1, self.N // 2)
        scales = torch.randn(self.GROUPS, self.N)
        media_weight = torch.randn(8, 8)
        state_dict = {
            "q_proj.weight": self._pack_int32_last(codes),
            "q_proj.weight_scale": scales,
            "q_proj.weight_zero_point": self._pack_int32_last(zero_codes),
            "vision.weight": media_weight,
        }

        result = preprocess_quark_weights(
            state_dict,
            bits=self.BITS,
            group_size=self.GROUP_SIZE,
        )

        expected_weight = codes.T[:, 0::2] | (codes.T[:, 1::2] << 4)
        expected_weight = expected_weight.reshape(self.N, self.GROUPS, 16)
        expected_zeros = zero_codes.T[:, 0::2] | (zero_codes.T[:, 1::2] << 4)
        assert torch.equal(result["q_proj.weight"], expected_weight)
        assert torch.equal(result["q_proj.scales"], scales.T)
        assert torch.equal(result["q_proj.zero_points"], expected_zeros)
        assert result["vision.weight"] is media_weight
        assert "q_proj.weight_scale" not in result
        assert "q_proj.weight_zero_point" not in result


class TestMergeLoraWeights:
    """Tests for merge_lora_weights()."""

    def test_basic_merge(self):
        """LoRA delta is added to base weight: W' = W + (alpha/rank) * B @ A."""
        base = {"layer.weight": torch.zeros(4, 8)}
        lora = {
            "layer.lora_A.weight": torch.ones(2, 8),  # rank=2
            "layer.lora_B.weight": torch.ones(4, 2),
        }
        result = merge_lora_weights(base, lora, default_alpha=2.0)
        # scale = alpha/rank = 2/2 = 1.0
        # delta = B @ A = ones(4,2) @ ones(2,8) = 2*ones(4,8)
        expected = torch.full((4, 8), 2.0)
        torch.testing.assert_close(result["layer.weight"], expected)

    def test_alpha_from_tensor(self):
        """Per-layer alpha tensor overrides default_alpha."""
        base = {"layer.weight": torch.zeros(4, 8)}
        lora = {
            "layer.lora_A.weight": torch.eye(2, 8),
            "layer.lora_B.weight": torch.eye(4, 2),
            "layer.lora_A.alpha": torch.tensor(4.0),
        }
        result = merge_lora_weights(base, lora, default_alpha=1.0)
        # scale = alpha/rank = 4/2 = 2.0
        # delta = B @ A = eye(4,2) @ eye(2,8) -> (4,8) with 1s in top-left 2x2
        # scaled delta: 2.0 * delta
        w = result["layer.weight"]
        assert w[0, 0].item() == pytest.approx(2.0)
        assert w[2, 0].item() == pytest.approx(0.0)

    def test_default_alpha_equals_rank(self):
        """When no alpha provided, scale = rank/rank = 1.0."""
        base = {"layer.weight": torch.zeros(4, 8)}
        lora = {
            "layer.lora_A.weight": torch.ones(2, 8),
            "layer.lora_B.weight": torch.ones(4, 2),
        }
        result = merge_lora_weights(base, lora)
        # scale = rank/rank = 1.0, delta = 2*ones(4,8)
        expected = torch.full((4, 8), 2.0)
        torch.testing.assert_close(result["layer.weight"], expected)

    def test_preserves_base_dtype(self):
        """Merged result keeps the base weight's dtype (e.g. float16)."""
        base = {"layer.weight": torch.zeros(4, 8, dtype=torch.float16)}
        lora = {
            "layer.lora_A.weight": torch.ones(2, 8),
            "layer.lora_B.weight": torch.ones(4, 2),
        }
        result = merge_lora_weights(base, lora, default_alpha=2.0)
        assert result["layer.weight"].dtype == torch.float16

    def test_multiple_layers(self):
        """Merges LoRA for multiple layers independently."""
        base = {
            "attn.q_proj.weight": torch.zeros(4, 4),
            "attn.v_proj.weight": torch.ones(4, 4),
        }
        lora = {
            "attn.q_proj.lora_A.weight": torch.eye(1, 4),
            "attn.q_proj.lora_B.weight": torch.eye(4, 1),
            "attn.v_proj.lora_A.weight": torch.eye(1, 4),
            "attn.v_proj.lora_B.weight": torch.eye(4, 1),
        }
        result = merge_lora_weights(base, lora, default_alpha=1.0)
        # Both get identity delta (rank=1, alpha=1, scale=1)
        assert result["attn.q_proj.weight"][0, 0].item() == pytest.approx(1.0)
        assert result["attn.v_proj.weight"][0, 0].item() == pytest.approx(2.0)

    def test_missing_base_key_warns(self, caplog):
        """Warns when LoRA targets a weight not in base model."""
        import logging

        base = {"other.weight": torch.zeros(4, 4)}
        lora = {
            "missing.lora_A.weight": torch.ones(2, 4),
            "missing.lora_B.weight": torch.ones(4, 2),
        }
        with caplog.at_level(logging.WARNING):
            result = merge_lora_weights(base, lora)
        assert "not found in base model" in caplog.text
        # base unchanged
        torch.testing.assert_close(result["other.weight"], torch.zeros(4, 4))

    def test_orphan_lora_a_warns(self, caplog):
        """Warns when lora_A has no matching lora_B."""
        import logging

        base = {"layer.weight": torch.zeros(4, 4)}
        lora = {"layer.lora_A.weight": torch.ones(2, 4)}
        with caplog.at_level(logging.WARNING):
            merge_lora_weights(base, lora)
        assert "without matching lora_B" in caplog.text

    def test_non_lora_keys_ignored(self):
        """Non-LoRA keys in lora_state_dict are silently ignored."""
        base = {"layer.weight": torch.zeros(4, 4)}
        lora = {"some_other_key": torch.ones(4, 4)}
        result = merge_lora_weights(base, lora)
        torch.testing.assert_close(result["layer.weight"], torch.zeros(4, 4))

    def test_modifies_base_in_place(self):
        """Returns the same dict object (modified in-place)."""
        base = {"layer.weight": torch.zeros(4, 8)}
        lora = {
            "layer.lora_A.weight": torch.ones(2, 8),
            "layer.lora_B.weight": torch.ones(4, 2),
        }
        result = merge_lora_weights(base, lora)
        assert result is base


class TestSplitInterleavedQKV:
    """Tests for split_interleaved_qkv (GPT-NeoX / Persimmon layout)."""

    def test_2d_weight_mha(self):
        """MHA weight with interleaved [h0_q, h0_k, h0_v, h1_q, ...] layout."""
        num_heads, head_dim, hidden = 4, 8, 32
        # Build a known interleaved pattern so we can verify the split
        qs, ks, vs = [], [], []
        for h in range(num_heads):
            qs.append(torch.full((head_dim, hidden), float(h)))
            ks.append(torch.full((head_dim, hidden), float(h) + 0.1))
            vs.append(torch.full((head_dim, hidden), float(h) + 0.2))
        # Interleave: [q0, k0, v0, q1, k1, v1, ...]
        parts = []
        for h in range(num_heads):
            parts.extend([qs[h], ks[h], vs[h]])
        fused = torch.cat(parts, dim=0)  # [3*hidden, hidden]

        q, k, v = split_interleaved_qkv(fused, num_heads, num_heads, head_dim)

        assert q.shape == (num_heads * head_dim, hidden)
        assert k.shape == (num_heads * head_dim, hidden)
        assert v.shape == (num_heads * head_dim, hidden)
        torch.testing.assert_close(q, torch.cat(qs))
        torch.testing.assert_close(k, torch.cat(ks))
        torch.testing.assert_close(v, torch.cat(vs))

    def test_1d_bias(self):
        """Bias vector (1D) with interleaved layout."""
        num_heads, head_dim = 4, 8
        bias = torch.arange(num_heads * 3 * head_dim, dtype=torch.float32)
        q, k, v = split_interleaved_qkv(bias, num_heads, num_heads, head_dim)

        assert q.shape == (num_heads * head_dim,)
        assert k.shape == (num_heads * head_dim,)
        assert v.shape == (num_heads * head_dim,)
        # Reconstruct from known interleaved pattern
        reshaped = bias.reshape(num_heads, 3, head_dim)
        torch.testing.assert_close(q, reshaped[:, 0].reshape(-1))
        torch.testing.assert_close(k, reshaped[:, 1].reshape(-1))
        torch.testing.assert_close(v, reshaped[:, 2].reshape(-1))

    def test_wrong_dim_raises(self):
        """Wrong leading dimension raises ValueError."""
        with pytest.raises(ValueError, match="expected 96"):
            split_interleaved_qkv(
                torch.zeros(100, 32), num_heads=4, num_kv_heads=4, head_dim=8
            )

    def test_gqa_raises(self):
        """GQA (num_kv_heads != num_heads) is not supported."""
        with pytest.raises(ValueError, match="requires MHA"):
            split_interleaved_qkv(torch.zeros(96, 32), num_heads=4, num_kv_heads=2, head_dim=8)


class TestSplitCodegenQKV:
    """Tests for split_codegen_qkv (QVK model-parallel layout)."""

    def test_basic_split(self):
        """Verify Q/V/K extraction from mp-interleaved layout."""
        num_heads, head_dim, hidden, mp_num = 4, 8, 32, 2
        local_dim = num_heads * head_dim // mp_num  # 16

        # Build known pattern: each mp-block has [q, v, k] chunks
        q_full = torch.ones(num_heads * head_dim, hidden) * 1.0
        v_full = torch.ones(num_heads * head_dim, hidden) * 2.0
        k_full = torch.ones(num_heads * head_dim, hidden) * 3.0

        # Interleave by mp blocks: [q_mp0, v_mp0, k_mp0, q_mp1, v_mp1, k_mp1]
        parts = []
        for mp in range(mp_num):
            s = mp * local_dim
            e = s + local_dim
            parts.extend([q_full[s:e], v_full[s:e], k_full[s:e]])
        fused = torch.cat(parts, dim=0)  # [3*hidden, hidden]

        q, k, v = split_codegen_qkv(fused, num_heads, head_dim, mp_num)

        assert q.shape == (num_heads * head_dim, hidden)
        assert k.shape == (num_heads * head_dim, hidden)
        assert v.shape == (num_heads * head_dim, hidden)
        torch.testing.assert_close(q, q_full)
        torch.testing.assert_close(k, k_full)
        torch.testing.assert_close(v, v_full)

    def test_wrong_dim_raises(self):
        """Wrong leading dimension raises ValueError."""
        with pytest.raises(ValueError, match="expected 96"):
            split_codegen_qkv(torch.zeros(100, 32), num_heads=4, head_dim=8)

    def test_indivisible_mp_num_raises(self):
        """mp_num that doesn't divide hidden raises ValueError."""
        with pytest.raises(ValueError, match="divisible"):
            split_codegen_qkv(torch.zeros(96, 32), num_heads=4, head_dim=8, mp_num=3)
