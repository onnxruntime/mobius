# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Gemma4 audio encoder components."""

from __future__ import annotations

from mobius._testing import create_test_builder, create_test_input
from mobius.components._gemma4_audio import (
    Gemma4AudioEncoder,
    Gemma4CausalChunkedAttention,
    Gemma4ConformerEncoderLayer,
    Gemma4ConvSubsampling,
)

_DIM = 32
_HEADS = 2
_INNER = 64
_KERNEL = 3
_CTX_LEFT = 4  # attention_context_left for tests
_BATCH = 1
_TIME = 16
_INPUT_SIZE = 16


class TestGemma4ConvSubsampling:
    def test_has_parameters(self):
        sub = Gemma4ConvSubsampling(_INPUT_SIZE, conv_channels=[8, 4], hidden_size=_DIM)
        param_names = [n for n, _ in sub.named_parameters()]
        assert any("conv0" in n for n in param_names)
        assert any("conv1" in n for n in param_names)
        assert any("out" in n for n in param_names)

    def test_forward(self):
        sub = Gemma4ConvSubsampling(_INPUT_SIZE, conv_channels=[8, 4], hidden_size=_DIM)
        builder, op, _graph = create_test_builder()
        x = create_test_input(builder, "x", [_BATCH, _TIME, _INPUT_SIZE])
        out = sub(op, x)
        assert out is not None

    def test_channel_progression(self):
        # Stage-0 weight: [c0, 1, 3, 3], stage-1 weight: [c1, c0, 3, 3]
        sub = Gemma4ConvSubsampling(_INPUT_SIZE, conv_channels=[8, 4], hidden_size=_DIM)
        param_shapes = {n: p.shape for n, p in sub.named_parameters()}
        assert param_shapes["conv0.weight"][0] == 8  # c0 output channels
        assert param_shapes["conv1.weight"][0] == 4  # c1 output channels


class TestGemma4CausalChunkedAttention:
    def test_has_parameters(self):
        attn = Gemma4CausalChunkedAttention(_DIM, _HEADS, _CTX_LEFT)
        param_names = [n for n, _ in attn.named_parameters()]
        assert any("linear_q" in n for n in param_names)
        assert any("linear_k" in n for n in param_names)
        assert any("linear_v" in n for n in param_names)
        assert any("linear_out" in n for n in param_names)

    def test_forward(self):
        attn = Gemma4CausalChunkedAttention(_DIM, _HEADS, _CTX_LEFT)
        builder, op, _graph = create_test_builder()
        x = create_test_input(builder, "x", [_BATCH, _TIME, _DIM])
        out = attn(op, x)
        assert out is not None


class TestGemma4ConformerEncoderLayer:
    def test_has_parameters(self):
        layer = Gemma4ConformerEncoderLayer(
            _DIM, _HEADS, _INNER, _KERNEL, _CTX_LEFT
        )
        param_names = [n for n, _ in layer.named_parameters()]
        assert any("self_attn" in n for n in param_names)
        assert any("feed_forward_in" in n for n in param_names)
        assert any("feed_forward_out" in n for n in param_names)
        assert any("conv" in n for n in param_names)
        assert any("layer_norm" in n for n in param_names)

    def test_forward(self):
        layer = Gemma4ConformerEncoderLayer(
            _DIM, _HEADS, _INNER, _KERNEL, _CTX_LEFT
        )
        builder, op, _graph = create_test_builder()
        x = create_test_input(builder, "x", [_BATCH, _TIME, _DIM])
        out = layer(op, x)
        assert out is not None

    def test_uses_rms_norm(self):
        """Verify attention pre-norm and final norm are RMSNorm (not LayerNorm)."""
        from mobius.components._rms_norm import RMSNorm

        layer = Gemma4ConformerEncoderLayer(
            _DIM, _HEADS, _INNER, _KERNEL, _CTX_LEFT
        )
        assert isinstance(layer.layer_norm_att, RMSNorm)
        assert isinstance(layer.layer_norm, RMSNorm)


class TestGemma4AudioEncoder:
    def test_has_parameters(self):
        enc = Gemma4AudioEncoder(
            input_size=_INPUT_SIZE,
            hidden_size=_DIM,
            num_heads=_HEADS,
            num_layers=1,
            ffn_inner_size=_INNER,
            conv_kernel_size=_KERNEL,
            conv_channels=[8, 4],
            attention_context_left=_CTX_LEFT,
            output_proj_dims=_DIM,
        )
        param_names = [n for n, _ in enc.named_parameters()]
        assert any("subsampling" in n for n in param_names)
        assert any("encoders" in n for n in param_names)
        assert any("output_projection" in n for n in param_names)

    def test_forward(self):
        enc = Gemma4AudioEncoder(
            input_size=_INPUT_SIZE,
            hidden_size=_DIM,
            num_heads=_HEADS,
            num_layers=1,
            ffn_inner_size=_INNER,
            conv_kernel_size=_KERNEL,
            conv_channels=[8, 4],
            attention_context_left=_CTX_LEFT,
            output_proj_dims=_DIM,
        )
        builder, op, _graph = create_test_builder()
        x = create_test_input(builder, "x", [_BATCH, _TIME, _INPUT_SIZE])
        out = enc(op, x)
        assert out is not None

    def test_default_config_matches_gemma4(self):
        """Default args match google/gemma-4-E2B-it audio config."""
        enc = Gemma4AudioEncoder()
        assert len(enc.encoders) == 12
        assert enc.output_projection.weight.shape == [1536, 1024]
        # Check context window on first layer's attention
        assert enc.encoders[0].self_attn._attention_context_left == 13

    def test_layer_count(self):
        enc = Gemma4AudioEncoder(
            input_size=_INPUT_SIZE,
            hidden_size=_DIM,
            num_heads=_HEADS,
            num_layers=3,
            ffn_inner_size=_INNER,
            conv_kernel_size=_KERNEL,
            conv_channels=[8, 4],
            attention_context_left=_CTX_LEFT,
            output_proj_dims=_DIM,
        )
        assert len(enc.encoders) == 3
