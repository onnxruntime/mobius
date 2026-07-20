# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Unit tests for Gemma4AudioEncoder components.

All tests build ONNX graphs without weights.  They verify:
- graph builds without exception
- output nodes are present
- expected parameter names exist
- parameter counts match architecture expectations
"""

from __future__ import annotations

import onnx_ir as ir

from mobius._testing import create_test_builder, create_test_input
from mobius.components._gemma4_audio import (
    ClippableLinear,
    Gemma4Attention,
    Gemma4AudioEncoder,
    Gemma4AudioLayer,
    Gemma4ConvSubsampling,
    Gemma4FeedForward,
    Gemma4LightConv1d,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HIDDEN = 64
HEADS = 4
HEAD_DIM = HIDDEN // HEADS  # 16
CTX_LEFT = 5


class TestClippableLinear:
    def test_bfloat16_clips_in_float32(self):
        comp = ClippableLinear(HIDDEN, HIDDEN)
        for parameter in comp.parameters():
            parameter.type = ir.TensorType(ir.DataType.BFLOAT16)

        b, op, graph = create_test_builder()
        x = create_test_input(b, "x", [1, 2, HIDDEN], ir.DataType.BFLOAT16)
        result = comp(op, x)
        b._adapt_outputs([result], "")

        clips = [node for node in graph if node.op_type == "Clip"]
        assert len(clips) == 2
        assert all(node.inputs[0].dtype == ir.DataType.FLOAT for node in clips)
        assert result.dtype == ir.DataType.BFLOAT16


def _build(comp, inputs):
    """Build graph and return the last output value."""
    b, op, graph = create_test_builder()
    result = comp(op, *inputs(b))
    b._adapt_outputs([result], "")
    return result, graph


# ---------------------------------------------------------------------------
# Gemma4ConvSubsampling
# ---------------------------------------------------------------------------


class TestGemma4ConvSubsampling:
    def test_graph_builds(self):
        comp = Gemma4ConvSubsampling(input_size=32, conv_channels=[16, 8], hidden_size=HIDDEN)
        b, op, graph = create_test_builder()
        x = create_test_input(b, "x", [1, 20, 32])
        result, _mask = comp(op, x)
        b._adapt_outputs([result], "")
        assert graph.num_nodes() > 0

    def test_parameter_names(self):
        comp = Gemma4ConvSubsampling(input_size=32, conv_channels=[16, 8], hidden_size=HIDDEN)
        names = {n for n, _ in comp.named_parameters()}
        assert "conv0.weight" in names
        assert "conv1.weight" in names
        assert "norm0.weight" in names
        assert "norm1.weight" in names
        assert "input_proj_linear.weight" in names

    def test_mask_input(self):
        """Passing input_features_mask returns a mask and adds Mul/Slice nodes."""
        comp = Gemma4ConvSubsampling(input_size=32, conv_channels=[16, 8], hidden_size=HIDDEN)
        b, op, graph = create_test_builder()
        x = create_test_input(b, "x", [1, 20, 32])
        mask = create_test_input(b, "mask", [1, 20])
        result, out_mask = comp(op, x, input_features_mask=mask)
        b._adapt_outputs([result], "")
        assert out_mask is not None
        op_types = {node.op_type for node in graph}
        assert "Mul" in op_types, "Mask application should produce Mul nodes"
        assert "Slice" in op_types, "Mask downsampling should produce Slice nodes"

    def test_no_bias_convolutions(self):
        """Subsampling conv layers have no bias (matches HF Conv2dNoBias)."""
        comp = Gemma4ConvSubsampling(input_size=32, conv_channels=[16, 8], hidden_size=HIDDEN)
        names = {n for n, _ in comp.named_parameters()}
        assert "conv0.bias" not in names
        assert "conv1.bias" not in names

    def test_no_norm_bias(self):
        """LayerNorm layers have no bias (matches HF bias=False)."""
        comp = Gemma4ConvSubsampling(input_size=32, conv_channels=[16, 8], hidden_size=HIDDEN)
        names = {n for n, _ in comp.named_parameters()}
        assert "norm0.bias" not in names
        assert "norm1.bias" not in names

    def test_proj_input_dim(self):
        """Linear projection input size = (freq_after_2_strides * c1)."""
        # input_size=32, 2 stride-2 convs with pad=1:
        # 32 → (32-1)//2+1 = 16 → (16-1)//2+1 = 8; then c1=8 → 8*8=64
        comp = Gemma4ConvSubsampling(input_size=32, conv_channels=[16, 8], hidden_size=HIDDEN)
        params = dict(comp.named_parameters())
        in_proj = params["input_proj_linear.weight"]
        assert in_proj.shape == (64, HIDDEN)


# ---------------------------------------------------------------------------
# Gemma4FeedForward
# ---------------------------------------------------------------------------


class TestGemma4FeedForward:
    def test_graph_builds(self):
        comp = Gemma4FeedForward(hidden_size=HIDDEN)
        b, op, graph = create_test_builder()
        x = create_test_input(b, "x", [1, 10, HIDDEN])
        result = comp(op, x)
        b._adapt_outputs([result], "")
        assert graph.num_nodes() > 0

    def test_parameter_names(self):
        comp = Gemma4FeedForward(hidden_size=HIDDEN)
        names = {n for n, _ in comp.named_parameters()}
        assert "pre_layer_norm.weight" in names
        assert "post_layer_norm.weight" in names
        assert "ffw_layer_1.weight" in names
        assert "ffw_layer_2.weight" in names

    def test_no_bias(self):
        """Feed-forward linears have bias=False (HF Gemma4ClippableLinear has no bias in checkpoint)."""
        comp = Gemma4FeedForward(hidden_size=HIDDEN)
        names = {n for n, _ in comp.named_parameters()}
        assert "ffw_layer_1.bias" not in names
        assert "ffw_layer_2.bias" not in names

    def test_linear_shapes(self):
        """FF1 expands h→4h, FF2 contracts 4h→h. ONNX weight is (out, in)."""
        comp = Gemma4FeedForward(hidden_size=HIDDEN)
        params = dict(comp.named_parameters())
        assert params["ffw_layer_1.weight"].shape == (HIDDEN * 4, HIDDEN)
        assert params["ffw_layer_2.weight"].shape == (HIDDEN, HIDDEN * 4)


# ---------------------------------------------------------------------------
# Gemma4LightConv1d
# ---------------------------------------------------------------------------


class TestGemma4LightConv1d:
    def test_graph_builds(self):
        comp = Gemma4LightConv1d(hidden_size=HIDDEN, conv_kernel_size=5)
        b, op, graph = create_test_builder()
        x = create_test_input(b, "x", [1, 10, HIDDEN])
        result = comp(op, x)
        b._adapt_outputs([result], "")
        assert graph.num_nodes() > 0

    def test_parameter_names(self):
        comp = Gemma4LightConv1d(hidden_size=HIDDEN, conv_kernel_size=5)
        names = {n for n, _ in comp.named_parameters()}
        assert "pre_layer_norm.weight" in names
        assert "linear_start.weight" in names
        assert "linear_end.weight" in names
        assert "depthwise_conv1d.weight" in names
        assert "conv_norm.weight" in names

    def test_glu_expansion(self):
        """linear_start doubles hidden size for GLU split. ONNX weight is (out, in)."""
        comp = Gemma4LightConv1d(hidden_size=HIDDEN, conv_kernel_size=5)
        params = dict(comp.named_parameters())
        assert params["linear_start.weight"].shape == (HIDDEN * 2, HIDDEN)

    def test_depthwise_weight_shape(self):
        """Depthwise conv weight: [channels, 1, kernel_size]."""
        comp = Gemma4LightConv1d(hidden_size=HIDDEN, conv_kernel_size=5)
        params = dict(comp.named_parameters())
        assert params["depthwise_conv1d.weight"].shape == (HIDDEN, 1, 5)

    def test_no_conv_bias(self):
        """Depthwise conv has no bias (HF CausalConv1d uses bias=False)."""
        comp = Gemma4LightConv1d(hidden_size=HIDDEN, conv_kernel_size=5)
        names = {n for n, _ in comp.named_parameters()}
        assert "depthwise_conv1d.bias" not in names


# ---------------------------------------------------------------------------
# Gemma4Attention
# ---------------------------------------------------------------------------


class TestGemma4Attention:
    def test_graph_builds(self):
        comp = Gemma4Attention(
            hidden_size=HIDDEN, num_heads=HEADS, attention_context_left=CTX_LEFT
        )
        b, op, graph = create_test_builder()
        x = create_test_input(b, "x", [1, 10, HIDDEN])
        result = comp(op, x)
        b._adapt_outputs([result], "")
        assert graph.num_nodes() > 0

    def test_parameter_names(self):
        comp = Gemma4Attention(
            hidden_size=HIDDEN, num_heads=HEADS, attention_context_left=CTX_LEFT
        )
        names = {n for n, _ in comp.named_parameters()}
        assert "q_proj.weight" in names
        assert "k_proj.weight" in names
        assert "v_proj.weight" in names
        assert "post.weight" in names
        assert "per_dim_scale" in names
        assert "relative_k_proj.weight" in names
        assert "pos_embed" in names

    def test_per_dim_scale_shape(self):
        """per_dim_scale is [head_dim] = [hidden_size / num_heads]."""
        comp = Gemma4Attention(
            hidden_size=HIDDEN, num_heads=HEADS, attention_context_left=CTX_LEFT
        )
        params = dict(comp.named_parameters())
        assert params["per_dim_scale"].shape == (HEAD_DIM,)

    def test_pos_embed_shape(self):
        """pos_embed is [context_left, hidden_size]."""
        comp = Gemma4Attention(
            hidden_size=HIDDEN, num_heads=HEADS, attention_context_left=CTX_LEFT
        )
        params = dict(comp.named_parameters())
        assert params["pos_embed"].shape == (CTX_LEFT, HIDDEN)

    def test_pos_embed_is_precomputed(self):
        """pos_embed has a precomputed constant value (not random)."""
        comp = Gemma4Attention(
            hidden_size=HIDDEN, num_heads=HEADS, attention_context_left=CTX_LEFT
        )
        params = dict(comp.named_parameters())
        p = params["pos_embed"]
        # If constant value is set, const_value is not None
        assert p.const_value is not None

    def test_no_qkv_bias(self):
        """Q/K/V projections and output projection all have no bias.

        (HF checkpoint has no bias for any self_attn linear in audio tower).
        """
        comp = Gemma4Attention(
            hidden_size=HIDDEN, num_heads=HEADS, attention_context_left=CTX_LEFT
        )
        names = {n for n, _ in comp.named_parameters()}
        assert "q_proj.bias" not in names
        assert "k_proj.bias" not in names
        assert "v_proj.bias" not in names
        assert "relative_k_proj.bias" not in names
        assert "post.bias" not in names  # HF checkpoint has no self_attn.post bias


# ---------------------------------------------------------------------------
# Gemma4AudioLayer
# ---------------------------------------------------------------------------


class TestGemma4AudioLayer:
    def test_graph_builds(self):
        comp = Gemma4AudioLayer(
            hidden_size=HIDDEN,
            num_heads=HEADS,
            conv_kernel_size=5,
            attention_context_left=CTX_LEFT,
        )
        b, op, graph = create_test_builder()
        x = create_test_input(b, "x", [1, 10, HIDDEN])
        result = comp(op, x)
        b._adapt_outputs([result], "")
        assert graph.num_nodes() > 0

    def test_parameter_names(self):
        comp = Gemma4AudioLayer(
            hidden_size=HIDDEN,
            num_heads=HEADS,
            conv_kernel_size=5,
            attention_context_left=CTX_LEFT,
        )
        names = {n for n, _ in comp.named_parameters()}
        # Feed-forward blocks
        assert "feed_forward1.pre_layer_norm.weight" in names
        assert "feed_forward2.pre_layer_norm.weight" in names
        # Attention
        assert "self_attn.q_proj.weight" in names
        # LightConv1d
        assert "lconv1d.depthwise_conv1d.weight" in names
        # Per-layer norms
        assert "norm_pre_attn.weight" in names
        assert "norm_post_attn.weight" in names
        assert "norm_out.weight" in names

    def test_submodule_types(self):
        comp = Gemma4AudioLayer(
            hidden_size=HIDDEN,
            num_heads=HEADS,
            conv_kernel_size=5,
            attention_context_left=CTX_LEFT,
        )
        assert isinstance(comp.feed_forward1, Gemma4FeedForward)
        assert isinstance(comp.feed_forward2, Gemma4FeedForward)
        assert isinstance(comp.self_attn, Gemma4Attention)
        assert isinstance(comp.lconv1d, Gemma4LightConv1d)


# ---------------------------------------------------------------------------
# Gemma4AudioEncoder
# ---------------------------------------------------------------------------


class TestGemma4AudioEncoder:
    def _make_encoder(self, num_layers=2):
        return Gemma4AudioEncoder(
            input_size=32,
            hidden_size=HIDDEN,
            num_heads=HEADS,
            num_layers=num_layers,
            conv_kernel_size=5,
            conv_channels=[16, 8],
            attention_context_left=CTX_LEFT,
            output_proj_dims=96,
        )

    def test_graph_builds(self):
        enc = self._make_encoder()
        b, op, graph = create_test_builder()
        x = create_test_input(b, "input_features", [1, 20, 32])
        result, mask = enc(op, x)
        b._adapt_outputs([result], "")
        assert graph.num_nodes() > 0
        assert mask is None  # No mask provided → no mask returned

    def test_layer_count(self):
        enc = self._make_encoder(num_layers=3)
        assert len(enc.layers) == 3

    def test_output_proj_has_bias(self):
        """Output projection has bias=True (matches HF nn.Linear(..., bias=True))."""
        enc = self._make_encoder()
        names = {n for n, _ in enc.named_parameters()}
        assert "output_proj.bias" in names

    def test_subsampling_submodule(self):
        enc = self._make_encoder()
        assert isinstance(enc.subsample_conv_projection, Gemma4ConvSubsampling)

    def test_layer_types(self):
        enc = self._make_encoder(num_layers=2)
        for layer in enc.layers:
            assert isinstance(layer, Gemma4AudioLayer)

    def test_parameter_namespacing(self):
        enc = self._make_encoder(num_layers=1)
        names = {n for n, _ in enc.named_parameters()}
        # Subsampling
        assert "subsample_conv_projection.conv0.weight" in names
        # First (and only) layer
        assert "layers.0.self_attn.q_proj.weight" in names
        # Output proj with bias
        assert "output_proj.weight" in names
        assert "output_proj.bias" in names

    def test_exports_from_components(self):
        """Gemma4AudioEncoder is importable from mobius.components."""
        from mobius.components import Gemma4AudioEncoder as _Enc

        assert _Enc is Gemma4AudioEncoder
