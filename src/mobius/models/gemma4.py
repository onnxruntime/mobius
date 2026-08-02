# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Gemma 4 model implementations.

Architecture variants:
- **Gemma4CausalLMModel**: Text-only causal LM (model_type ``gemma4_text``).
- **Gemma4Model**: Multimodal model — supports Image-Text-to-Text (26B-A4B, 31B)
  and Any-to-Any audio+vision+text (E2B, E4B) via ``Gemma4Task``.

Key architectural differences from Gemma3:
- Standard ``RMSNorm`` throughout (no ``OffsetRMSNorm``).
- Dual head_dim: local sliding-window layers use ``config.head_dim``; global
  full-attention layers use ``config.global_head_dim``.
- Dual RoPE: different ``rope_theta`` and ``partial_rotary_factor`` per layer type.
- Per-layer input gating (disabled when ``hidden_size_per_layer_input == 0``).
- Vision encoder: pre-patchified input ``[B, N, 3*P^2]`` with 2D position lookup,
  bidirectional attention, 4-norm structure, and scale-then-project pooling.
- Vision projector: scale-free RMSNorm -> Linear (matches ``embed_vision`` weights).
- KV sharing: last ``num_kv_shared_layers`` layers borrow K,V from earlier layers
  and have no k_proj/v_proj weights of their own.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import numpy as np
import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._build_context import ep_capabilities
from mobius._configs import ArchitectureConfig, Gemma4Config
from mobius._weight_utils import (
    preprocess_awq_weights,
    preprocess_olive_weights,
    preprocess_quark_weights,
    vlm_decoder_weights,
    vlm_embedding_weights,
)
from mobius.components import (
    MLP,
    ClippableLinear,
    Embedding,
    LayerNorm,
    Linear,
    QuantizedEmbedding,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
    make_clippable_quantized_linear_factory,
    make_quantized_linear_factory,
)
from mobius.components._activations import get_activation
from mobius.components._gemma4_audio import Gemma4AudioEncoder
from mobius.components._mlp import GatedMLP
from mobius.models.base import CausalLMModel
from mobius.models.gemma3_text import Gemma3TextScaledWordEmbedding

if TYPE_CHECKING:
    from mobius.components._attention import GQAContext, StaticCacheState


# ---------------------------------------------------------------------------
# Mixed-precision quantization helpers
#
# Gemma4 is a multimodal model whose text decoder, token embeddings, vision,
# and audio components may be weight-quantized independently. The helpers
# below centralize checkpoint-level quantization and component exclusions.
# ---------------------------------------------------------------------------


def _text_quantization_config(config: Gemma4Config):
    """Return the active weight-quantization config, or ``None`` when off."""
    quantization_config = getattr(config, "quantization", None)
    if quantization_config is None or quantization_config.quant_method == "none":
        return None
    return quantization_config


def _text_linear_class(config: Gemma4Config) -> type | None:
    """Return a QuantizedLinear factory for text projections, or ``None``.

    ``None`` means "use a plain float :class:`Linear`". Only the text decoder
    projections (attention Q/K/V/O + MLP + LM head) use this; vision and audio
    linears are always float.
    """
    quantization_config = _text_quantization_config(config)
    if quantization_config is None:
        return None
    if quantization_config.quant_method == "olive" and quantization_config.quantize_embeddings:
        raise NotImplementedError(
            "Gemma 4 does not yet support Olive-packed quantized token embeddings."
        )
    if quantization_config.group_size <= 0:
        raise ValueError(
            f"{quantization_config.quant_method} group_size must be resolved "
            "before building Gemma 4."
        )
    zero_point_dtype = (
        config.dtype
        if getattr(quantization_config, "float_zero_point", False)
        else ir.DataType.UINT8
    )
    return make_quantized_linear_factory(
        bits=quantization_config.bits,
        block_size=quantization_config.group_size,
        has_zero_point=not quantization_config.sym,
        zero_point_dtype=zero_point_dtype,
    )


def _text_embeddings_quantized(config: Gemma4Config) -> bool:
    """Whether the text token-embedding tables use GatherBlockQuantized."""
    quantization_config = _text_quantization_config(config)
    return quantization_config is not None and bool(
        getattr(quantization_config, "quantize_embeddings", False)
    )


def _text_lm_head_quantized(config: Gemma4Config) -> bool:
    """Whether the text LM head projection uses MatMulNBits."""
    quantization_config = _text_quantization_config(config)
    return quantization_config is not None and bool(
        getattr(quantization_config, "quantize_lm_head", False)
    )


def _make_scaled_word_embedding(
    config: Gemma4Config,
    num_embeddings: int,
    embedding_dim: int,
    embed_scale: float,
):
    """Build a scaled token embedding, quantized when the config requests it.

    Returns a :class:`Gemma4ScaledQuantizedWordEmbedding` (GatherBlockQuantized
    lookup) when embedding quantization is enabled and the embedding dimension
    is block-aligned, otherwise a float :class:`Gemma3TextScaledWordEmbedding`.
    """
    quantization_config = _text_quantization_config(config)
    if (
        _text_embeddings_quantized(config)
        and embedding_dim % quantization_config.group_size == 0
    ):
        return Gemma4ScaledQuantizedWordEmbedding(
            num_embeddings,
            embedding_dim,
            config.pad_token_id,
            embed_scale=embed_scale,
            bits=quantization_config.bits,
            block_size=quantization_config.group_size,
            has_zero_point=not quantization_config.sym,
        )
    return Gemma3TextScaledWordEmbedding(
        num_embeddings,
        embedding_dim,
        config.pad_token_id,
        embed_scale=embed_scale,
    )


def _make_lm_head(config: Gemma4Config) -> nn.Module:
    """Build the LM head projection, quantized (MatMulNBits) when requested.

    The multimodal decoder and embedding live in separate ONNX graphs, so a
    quantized head cannot share the embedding's packed table — it always gets
    its own independent MatMulNBits weight (populated from the same repacked
    token-embedding data at load time).
    """
    if _text_lm_head_quantized(config):
        linear_cls = _text_linear_class(config)
        if linear_cls is not None:
            return linear_cls(config.hidden_size, config.vocab_size, bias=False)
    return Linear(config.hidden_size, config.vocab_size, bias=False)


class Gemma4ScaledQuantizedWordEmbedding(QuantizedEmbedding):
    """GatherBlockQuantized token embedding scaled by ``embed_scale``.

    Quantized counterpart of :class:`Gemma3TextScaledWordEmbedding`: the packed
    embedding rows are gathered and dequantized by ``GatherBlockQuantized`` and
    then multiplied by the Gemma ``sqrt(hidden_size)`` (or per-layer) scale.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        embed_scale: float = 1.0,
        *,
        bits: int = 4,
        block_size: int = 32,
        has_zero_point: bool = True,
    ):
        super().__init__(
            num_embeddings,
            embedding_dim,
            bits=bits,
            block_size=block_size,
            has_zero_point=has_zero_point,
            padding_idx=padding_idx,
        )
        self.embed_scale = embed_scale

    def forward(self, op: OpBuilder, input_ids: ir.Value) -> ir.Value:
        embeddings = super().forward(op, input_ids)
        return op.Mul(embeddings, self.embed_scale)


_PER_LAYER_EMBEDDING_PREFIXES = (
    "embed_tokens_per_layer.",
    "per_layer_model_projection.",
    "per_layer_projection_norm.",
)


def _module_is_excluded(config: Gemma4Config, module_name: str) -> bool:
    quantization = config.quantization
    return quantization is not None and any(
        module_name == name or module_name.startswith(f"{name}.")
        for name in (quantization.modules_to_not_convert or [])
    )


def _component_is_quantized(config: Gemma4Config, source_prefixes: tuple[str, ...]) -> bool:
    """Return whether an Olive-packed component is uniformly quantized."""
    quantization = config.quantization
    if quantization is None or quantization.quant_method != "olive":
        return False
    if quantization.modules_to_not_convert is None:
        return False
    return not any(
        any(name == prefix or name.startswith(f"{prefix}.") for prefix in source_prefixes)
        for name in (quantization.modules_to_not_convert or [])
    )


def _component_linear_classes(
    config: Gemma4Config,
    source_prefixes: tuple[str, ...],
) -> tuple[type, type]:
    if not _component_is_quantized(config, source_prefixes):
        return Linear, ClippableLinear
    quantization = config.quantization
    zero_point_dtype = config.dtype if quantization.float_zero_point else ir.DataType.UINT8
    kwargs = {
        "bits": quantization.bits,
        "block_size": quantization.group_size,
        "has_zero_point": not quantization.sym,
        "zero_point_dtype": zero_point_dtype,
    }
    return (
        make_quantized_linear_factory(**kwargs),
        make_clippable_quantized_linear_factory(**kwargs),
    )


def _per_layer_model_projection_class(config: Gemma4Config) -> type:
    """Keep Olive GPTQ's uncalibrated pre-decoder projection in floating point."""
    quantization = config.quantization
    if quantization is not None and quantization.quant_method == "compressed-tensors":
        return _text_linear_class(config) or Linear
    if quantization is not None and quantization.quant_method == "olive":
        if quantization.modules_to_not_convert is not None and not _module_is_excluded(
            config,
            "model.language_model.per_layer_model_projection",
        ):
            return _text_linear_class(config) or Linear
    return Linear


def _preprocess_quantized_checkpoint(
    state_dict: dict[str, torch.Tensor],
    config: Gemma4Config,
) -> dict[str, torch.Tensor]:
    """Convert renamed packed tensors to Gemma 4 ONNX initializer layouts."""
    quantization = config.quantization
    if quantization is None:
        return state_dict
    if quantization.quant_method == "awq":
        return preprocess_awq_weights(
            state_dict,
            bits=quantization.bits,
            group_size=quantization.group_size,
        )
    if quantization.quant_method == "quark":
        return preprocess_quark_weights(
            state_dict,
            bits=quantization.bits,
            group_size=quantization.group_size,
            reorder=quantization.format != "order",
        )
    if quantization.quant_method != "olive":
        return state_dict
    return preprocess_olive_weights(
        state_dict,
        bits=quantization.bits,
        group_size=quantization.group_size,
        quantize_embeddings=quantization.quantize_embeddings,
        quantize_lm_head=quantization.quantize_lm_head,
        # Gemma 4 synthesizes and routes tied weights before this conversion.
        tie_word_embeddings=False,
    )


def _route_split_per_layer_weights(
    state_dict: dict[str, torch.Tensor],
    config: Gemma4Config,
) -> None:
    """Move oversized per-layer embedding weights into the decoder for WebGPU."""
    if not config.split_per_layer_embedding:
        return

    fused_key = "embedding.embed_tokens_per_layer.weight"
    if fused_key in state_dict:
        num_layers = config.num_hidden_layers
        per_layer_dim = config.hidden_size_per_layer_input
        fused = state_dict.pop(fused_key)
        assert fused.shape[1] == num_layers * per_layer_dim, (
            f"{fused_key} dim 1 expected {num_layers * per_layer_dim} "
            f"({num_layers} layers x {per_layer_dim} per_layer_dim), "
            f"got {fused.shape[1]}"
        )
        for i, chunk in enumerate(fused.chunk(num_layers, dim=1)):
            state_dict[f"decoder.model.embed_tokens_per_layer_split.{i}.weight"] = chunk

    for key in list(state_dict):
        if key.startswith(
            (
                "embedding.per_layer_model_projection.",
                "embedding.per_layer_projection_norm.",
            )
        ):
            state_dict[key.replace("embedding.", "decoder.model.", 1)] = state_dict.pop(key)


def _dtype_safe_compress(
    op: OpBuilder, data: ir.Value, condition: ir.Value, *, axis: int
) -> ir.Value:
    """Row-select ``data`` by a one-dimensional boolean ``condition``.

    ``Compress`` does not support bfloat16 and its CUDA graph transformer can
    introduce provider-less helper nodes. Convert the mask to ordered indices
    and use ``Gather`` in float32 instead. The float16/bfloat16 round-trip
    through float32 is lossless.
    """
    data_f32 = op.Cast(data, to=ir.DataType.FLOAT)
    indices = op.Squeeze(op.NonZero(condition), [0])
    selected = op.Gather(data_f32, indices, axis=axis)
    return op.CastLike(selected, data)


# ---------------------------------------------------------------------------
# Shared weight preprocessing helpers
# ---------------------------------------------------------------------------


def _remap_moe_expert_weights(
    state_dict: dict[str, torch.Tensor],
    config: Gemma4Config,
) -> None:
    """Rename HF MoE expert weights and fold router scale in-place.

    Shared by ``Gemma4CausalLMModel`` and ``Gemma4Model`` to avoid
    duplicating the rename/fold logic.
    """
    # experts.gate_up_proj → fc1_experts_weights
    # experts.down_proj    → fc2_experts_weights
    for key in list(state_dict.keys()):
        if ".experts.gate_up_proj" in key:
            new_key = key.replace(".experts.gate_up_proj", ".fc1_experts_weights")
            state_dict[new_key] = state_dict.pop(key)
        elif ".experts.down_proj" in key:
            new_key = key.replace(".experts.down_proj", ".fc2_experts_weights")
            state_dict[new_key] = state_dict.pop(key)

    # Fold hidden_size^-0.5 into router.scale
    if config.enable_moe_block:
        scale_factor = float(config.hidden_size**-0.5)
        for key in list(state_dict.keys()):
            if ".router.scale" in key and ".per_expert_scale" not in key:
                state_dict[key] = state_dict[key] * scale_factor


# ---------------------------------------------------------------------------
# Scale-free RMSNorm (Gemma4RMSNorm with with_scale=False)
# ---------------------------------------------------------------------------


class _Gemma4ScaleFreeRMSNorm(nn.Module):
    """RMSNorm with a constant all-ones scale (no learnable parameter).

    Matches ``Gemma4RMSNorm(with_scale=False)`` in HuggingFace.
    Used for V norms in the vision encoder, the vision projector pre-norm,
    and the audio pre-projection norm.

    Uses ``op.RMSNormalization`` with ``stash_type=1`` (float32 accumulation)
    to handle FP16 overflow: values > 256 squared exceed the FP16 max (65504).
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # Constant all-ones scale (not a learnable parameter from HF).
        self.weight = nn.Parameter([dim], data=ir.Tensor(np.ones(dim, dtype=np.float32)))

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        # stash_type=1 means accumulate variance in float32, avoiding
        # FP16 overflow when squaring large values.
        # CastLike ensures the weight matches the input dtype.
        scale = op.CastLike(self.weight, hidden_states)
        return op.RMSNormalization(
            hidden_states,
            scale,
            axis=-1,
            epsilon=self.eps,
            stash_type=1,
        )


# ---------------------------------------------------------------------------
# Gemma4 vision encoder
# ---------------------------------------------------------------------------


class Gemma4VisionSelfAttention(nn.Module):
    """Bidirectional multi-head self-attention for the Gemma4 vision encoder.

    Differences from standard text Attention:
    - No causal mask (bidirectional attention via ``op.Attention`` without ``is_causal``).
    - Per-head QK norms (``RMSNorm`` with learned scale).
    - Per-head V norm (scale-free RMSNorm, no learned parameter).
    - Scale = 1.0 matching HF ``Gemma4VisionAttention``.
    - 2D RoPE: first ``head_dim//2`` dims rotated by x-coord, last by y-coord.
      Uses ``rope_theta=100.0`` and a position lookup table of size ``max_position``.

    Weight names align with HF after stripping the ``.linear.`` infix from
    ``Gemma4ClippableLinear`` module attributes.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        norm_eps: float = 1e-6,
        rope_theta: float = 100.0,
        max_position: int = 128,
        use_clipped_linears: bool = True,
        linear_class: type | None = None,
        clippable_linear_class: type | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        linear_class = (
            clippable_linear_class or ClippableLinear
            if use_clipped_linears
            else linear_class or Linear
        )
        self.q_proj = linear_class(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = linear_class(hidden_size, num_heads * self.head_dim, bias=False)
        self.v_proj = linear_class(hidden_size, num_heads * self.head_dim, bias=False)
        self.o_proj = linear_class(num_heads * self.head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=norm_eps)
        self.v_norm = _Gemma4ScaleFreeRMSNorm(self.head_dim, eps=norm_eps)

        # Precompute 2D RoPE cos/sin lookup tables: shape [max_position, head_dim//2].
        # HF computes inv_freq over the *spatial half* (head_dim//2) as the denominator,
        # then concatenates to fill both halves of each half-head dimension:
        #   inv_freq[i] = 1 / (rope_theta ^ (2*i / spatial_dim))  for i in 0..spatial_dim//2-1
        #   angles_base [max_pos, spatial_dim//2] → concat with itself → [max_pos, spatial_dim]
        # For vision, rope_theta=100.0 (much smaller than text's 10000).
        spatial_dim = self.head_dim // 2  # each spatial dimension (x or y) gets this many dims
        inv_freq = (
            1.0
            / (rope_theta ** (np.arange(0, spatial_dim, 2, dtype=np.float32) / spatial_dim))
        ).astype(np.float32)  # [spatial_dim//2]
        positions = np.arange(max_position, dtype=np.float32)  # [max_position]
        angles_base = np.outer(positions, inv_freq).astype(
            np.float32
        )  # [max_pos, spatial_dim//2]
        # Duplicate frequencies: each freq is used for both the lower and upper quarter
        angles = np.concatenate([angles_base, angles_base], axis=-1)  # [max_pos, spatial_dim]
        self.cos_cache = nn.Parameter(
            list(angles.shape),
            name="cos_cache",
            data=ir.tensor(np.cos(angles)),
        )
        self.sin_cache = nn.Parameter(
            list(angles.shape),
            name="sin_cache",
            data=ir.tensor(np.sin(angles)),
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None = None,
        pixel_position_ids: ir.Value | None = None,
    ) -> ir.Value:
        # [B, N, hidden] -> project and reshape for per-head norms
        q = self.q_proj(op, hidden_states)
        k = self.k_proj(op, hidden_states)
        v = self.v_proj(op, hidden_states)

        # Reshape to [B, N, num_heads, head_dim] for per-head norms
        q = op.Reshape(q, [0, 0, -1, self.head_dim])
        k = op.Reshape(k, [0, 0, -1, self.head_dim])
        v = op.Reshape(v, [0, 0, -1, self.head_dim])
        q = self.q_norm(op, q)
        k = self.k_norm(op, k)
        v = self.v_norm(op, v)

        # Apply 2D RoPE: first head_dim//2 dims use x-coordinate RoPE,
        # last head_dim//2 dims use y-coordinate RoPE.
        # Matches HF apply_multidimensional_rope with ndim=2, rope_theta=100.0.
        if pixel_position_ids is not None:
            half = self.head_dim // 2
            quarter = self.head_dim // 4

            # Extract and clamp x/y coords (clamping turns -1 padding into 0)
            x_coords = op.Clip(
                op.Gather(pixel_position_ids, op.Constant(value_int=0), axis=2),
                op.Constant(value_int=0),
            )  # [B, N]
            y_coords = op.Clip(
                op.Gather(pixel_position_ids, op.Constant(value_int=1), axis=2),
                op.Constant(value_int=0),
            )  # [B, N]

            # Gather cos/sin from lookup table
            cos_x = op.Gather(self.cos_cache, x_coords, axis=0)  # [B, N, half]
            sin_x = op.Gather(self.sin_cache, x_coords, axis=0)  # [B, N, half]
            cos_y = op.Gather(self.cos_cache, y_coords, axis=0)  # [B, N, half]
            sin_y = op.Gather(self.sin_cache, y_coords, axis=0)  # [B, N, half]

            # Unsqueeze for num_heads broadcast: [B, N, half] -> [B, N, 1, half]
            cos_x = op.Unsqueeze(cos_x, [2])
            sin_x = op.Unsqueeze(sin_x, [2])
            cos_y = op.Unsqueeze(cos_y, [2])
            sin_y = op.Unsqueeze(sin_y, [2])

            def apply_rope_half(x_half, cos, sin):
                """Apply RoPE to a head slice of size `half`: x_half [B, N, nh, half]."""
                # rotate_half: swap lower and upper quarters with negation on upper
                # rotate_half([a, b]) = [-b, a] where a, b each have `quarter` dims
                lower = op.Slice(x_half, [0], [quarter], [3])  # [B, N, nh, quarter]
                upper = op.Slice(x_half, [quarter], [half], [3])  # [B, N, nh, quarter]
                neg_upper = op.Neg(upper)
                rotated = op.Concat(neg_upper, lower, axis=3)  # [B, N, nh, half]
                # x_rot = x * cos + rotate_half(x) * sin  (standard HF RoPE formula)
                return op.Add(op.Mul(x_half, cos), op.Mul(rotated, sin))

            # Split q and k into first/second halves of head_dim
            q_first = op.Slice(q, [0], [half], [3])  # [B, N, nh, half]
            q_second = op.Slice(q, [half], [self.head_dim], [3])
            k_first = op.Slice(k, [0], [half], [3])
            k_second = op.Slice(k, [half], [self.head_dim], [3])

            q = op.Concat(
                apply_rope_half(q_first, cos_x, sin_x),
                apply_rope_half(q_second, cos_y, sin_y),
                axis=3,
            )
            k = op.Concat(
                apply_rope_half(k_first, cos_x, sin_x),
                apply_rope_half(k_second, cos_y, sin_y),
                axis=3,
            )

        # Transpose to [B, num_heads, N, head_dim] for manual attention.
        # The ONNX Attention op has a bug for N > ~2430 (ORT core dump / NaN),
        # so we implement Q@K^T/softmax/V explicitly.
        q = op.Transpose(q, perm=[0, 2, 1, 3])  # [B, nh, N, hd]
        k = op.Transpose(k, perm=[0, 2, 1, 3])
        v = op.Transpose(v, perm=[0, 2, 1, 3])

        # Scaled dot-product attention with scale=1.0 (matches HF Gemma4VisionAttention).
        # scores [B, nh, N, N] = q @ k^T (scale=1.0 so no division needed)
        k_t = op.Transpose(k, perm=[0, 1, 3, 2])  # [B, nh, hd, N]
        scores = op.MatMul(q, k_t)  # [B, nh, N, N]

        # Add additive attention bias (masks padding key positions with -1e9)
        if attention_bias is not None:
            # attention_bias: [B, 1, 1, N] broadcast over heads and queries
            scores = op.Add(scores, attention_bias)

        # Softmax over the key dimension
        attn_weights = op.Softmax(scores, axis=-1)  # [B, nh, N, N]

        # Weighted sum over values
        attn_output = op.MatMul(attn_weights, v)  # [B, nh, N, hd]

        # Reshape back to [B, N, num_heads * head_dim]
        attn_output = op.Transpose(attn_output, perm=[0, 2, 1, 3])  # [B, N, nh, hd]
        attn_output = op.Reshape(attn_output, [0, 0, -1])  # [B, N, nh*hd]

        return self.o_proj(op, attn_output)


class Gemma4VisionEncoderLayer(nn.Module):
    """Gemma4 vision transformer encoder layer.

    4-norm structure matching HF ``Gemma4VisionEncoderLayer``:
    pre-attn norm -> attention -> post-attn norm -> residual ->
    pre-MLP norm -> gated MLP -> post-MLP norm -> residual.

    Uses standard ``RMSNorm`` throughout (not ``OffsetRMSNorm``).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        norm_eps: float,
        hidden_act: str = "gelu_pytorch_tanh",
        rope_theta: float = 100.0,
        max_position: int = 128,
        use_clipped_linears: bool = True,
        linear_class: type | None = None,
        clippable_linear_class: type | None = None,
    ):
        super().__init__()
        self.self_attn = Gemma4VisionSelfAttention(
            hidden_size,
            num_heads,
            norm_eps,
            rope_theta=rope_theta,
            max_position=max_position,
            use_clipped_linears=use_clipped_linears,
            linear_class=linear_class,
            clippable_linear_class=clippable_linear_class,
        )
        self.input_layernorm = RMSNorm(hidden_size, eps=norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(hidden_size, eps=norm_eps)
        self.post_feedforward_layernorm = RMSNorm(hidden_size, eps=norm_eps)
        # Gated MLP: activation(gate_proj) * up_proj -> down_proj (SwiGLU/GEGLU style)
        # HF uses gelu_pytorch_tanh (GELU with tanh approximation); read from config.
        # Use ClippableLinear only when the checkpoint has clipping weights.
        linear_class = (
            clippable_linear_class or ClippableLinear
            if use_clipped_linears
            else linear_class or Linear
        )
        self.mlp = MLP(
            ArchitectureConfig(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=hidden_act,
                rms_norm_eps=norm_eps,
            ),
            linear_class=linear_class,
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None = None,
        pixel_position_ids: ir.Value | None = None,
    ) -> ir.Value:
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        hidden_states = self.self_attn(op, hidden_states, attention_bias, pixel_position_ids)
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = self.post_feedforward_layernorm(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)
        return hidden_states


class Gemma4VisionPooler(nn.Module):
    """Position-based spatial pooling for Gemma4 vision features.

    Replicates HF ``Gemma4VisionPooler._avg_pool_by_positions``:

    1. Mask padding patches (``pixel_position_ids == (-1,-1)``) to zero.
    2. For each patch at position ``(x, y)``, assign it to output bucket
       ``floor(x/k) + w_out * floor(y/k)`` where ``w_out = (max_x+1) // k``.
    3. One-hot the bucket indices to a FIXED ``length = T // k²`` grid and
       average-pool (``1/k²`` weights) via ``[B, length, T] @ [B, T, D]``.
    4. Return the pooled ``[B, length, D]`` features (scaled by
       ``sqrt(hidden_size)``) plus a ``[B, length]`` validity mask marking the
       occupied bins (bucket index ``< w_out * h_out``).

    Pooling to a fixed ``length`` (HF's ``output_length``) — rather than a
    per-image ``w_out * h_out`` — keeps the OneHot depth independent of the
    image content, so a batch of images with different aspect ratios can be
    pooled in a single forward pass.  The caller drops the empty trailing bins
    using the mask so the soft-token count equals the per-image placeholder
    count.  For a 57x42 image (2520 patches, k=3): ``length = 2520 // 9 = 280``
    with ``19 * 14 = 266`` valid bins.
    """

    def __init__(self, hidden_size: int, kernel_size: int = 3):
        super().__init__()
        self._kernel_size = kernel_size
        self._pooler_scale = float(hidden_size**0.5)

    def forward(
        self,
        op: OpBuilder,
        vision_features: ir.Value,
        pixel_position_ids: ir.Value,
    ) -> tuple[ir.Value, ir.Value]:
        # vision_features:      [B, T, D]
        # pixel_position_ids:   [B, T, 2]  — (x, y); (-1,-1) marks padding/CLS patches
        k = self._kernel_size
        k2 = k * k

        # --- 1. Mask padding patches to zero ---------------------------------
        # padding if BOTH x AND y are -1
        neg_one = op.Constant(value_int=-1)
        x_pos = op.Gather(pixel_position_ids, op.Constant(value_int=0), axis=2)  # [B, T]
        y_pos = op.Gather(pixel_position_ids, op.Constant(value_int=1), axis=2)  # [B, T]
        is_padding = op.And(op.Equal(x_pos, neg_one), op.Equal(y_pos, neg_one))  # [B, T] bool

        # zero out padding hidden states: features * (1 - is_padding)
        not_padding_f = op.CastLike(
            op.Cast(op.Not(is_padding), to=ir.DataType.FLOAT), vision_features
        )
        not_padding_f = op.Unsqueeze(not_padding_f, [2])  # [B, T, 1]
        vision_features = op.Mul(vision_features, not_padding_f)  # [B, T, D]

        # --- 2. Clamp positions to [0, ∞) so padding doesn't break Div ------
        clamped_x = op.Max(x_pos, op.Constant(value_int=0))  # [B, T]
        clamped_y = op.Max(y_pos, op.Constant(value_int=0))  # [B, T]

        # max valid x coord (padding patches are clamped to 0 but zeroed above).
        # Only max_x is needed: HF derives the bucket row stride from
        # ``max_x // k`` (= w_out); max_y is not used by the bucket index.
        # max_x = max(x_coord for non-padding) + 1 → [B, 1]
        max_x = op.Add(
            op.ReduceMax(clamped_x, op.Constant(value_ints=[1]), keepdims=1),
            op.Constant(value_int=1),
        )  # [B, 1]

        # --- 3. Fixed pooled length = T // k² (HF VisionPooler output_length) -
        # HF pools every image to a FIXED ``length = input_seq_len // k²`` grid
        # and returns a validity mask; using the fixed length (rather than a
        # per-image ``w_out * h_out``) keeps the OneHot depth independent of the
        # batch, so images of different aspect ratios pool together in one pass.
        # ``length`` is derived at runtime from the (padded) patch count T, so
        # it needs no static config value and stays a single scalar for any B.
        k_c = op.Constant(value_ints=[k])
        k2_c = op.Constant(value_ints=[k2])
        num_patches = op.Shape(vision_features, start=1, end=2)  # [1] = T
        length = op.Squeeze(
            op.Div(num_patches, k2_c), op.Constant(value_ints=[0])
        )  # scalar = T // k²

        # w_out is still per-image; it only feeds the bucket index below, never
        # the OneHot depth, so no scalar Squeeze (and no B=1 assumption) is needed.
        w_out = op.Div(max_x, k_c)  # [B, 1]  pooled width per image

        # --- 4. Kernel bucket index per patch --------------------------------
        # floor(x/k) + w_out * floor(y/k)  in range [0, w_out*h_out) ⊆ [0, length)
        kx = op.Div(clamped_x, k_c)  # [B, T]
        ky = op.Div(clamped_y, k_c)  # [B, T]
        kernel_idxs = op.Add(kx, op.Mul(w_out, ky))  # [B, T]

        # --- 5. One-hot weight matrix [B, T, length] -------------------------
        # weights[b, t, j] = 1/k² if patch t maps to bucket j, else 0.
        # Keep values as float32 — OneHot doesn't support bfloat16.
        on_val = 1.0 / float(k2)
        one_hot_vals = op.Constant(value_floats=[0.0, on_val])
        weights = op.OneHot(kernel_idxs, length, one_hot_vals)  # [B, T, length] f32

        # --- 6. Validity mask: a bin is valid iff some patch maps to it ------
        # HF: ``mask = logical_not((weights == 0).all(dim=1))`` over the patch
        # axis.  Empty trailing bins (index ≥ w_out*h_out for that image) get no
        # patch; the caller drops them so the soft-token count matches the
        # per-image placeholder count.  Computed on the float32 one-hot.
        valid_mask = op.Greater(
            op.ReduceMax(weights, op.Constant(value_ints=[1]), keepdims=0),  # [B, length]
            op.Constant(value_float=0.0),
        )  # [B, length] bool

        # --- 7. Weighted sum: [B, length, T] @ [B, T, D] -> [B, length, D] ---
        weights = op.CastLike(weights, vision_features)  # cast to model dtype
        weights_t = op.Transpose(weights, perm=[0, 2, 1])  # [B, length, T]
        output = op.MatMul(weights_t, vision_features)  # [B, length, D]

        # --- 8. Scale by sqrt(hidden_size) matching HF VisionPooler ----------
        scale = op.CastLike(op.Constant(value_float=self._pooler_scale), output)
        return op.Mul(output, scale), valid_mask


class _Gemma4VisionPatchEmbedder(nn.Module):
    """Gemma4 patch embedder: linear projection + 2D position lookup.

    Inputs:
    - ``pixel_values [B, N, 3*patch_size^2]``: pre-patchified, normalized to ``[-1, 1]``
    - ``pixel_position_ids [B, N, 2]``: (x, y) coordinates for each patch

    Output: ``[B, N, hidden_size]``

    Weight names match HF ``Gemma4VisionPatchEmbedder``:
    - ``input_proj.weight``
    - ``position_embedding_table`` (Parameter ``[2, pos_emb_size, hidden_size]``)
    """

    def __init__(
        self,
        patch_size: int,
        hidden_size: int,
        position_embedding_size: int,
        linear_class: type = Linear,
    ):
        super().__init__()
        self.input_proj = linear_class(3 * patch_size * patch_size, hidden_size, bias=False)
        # Position embedding table: [2, pos_emb_size, hidden] — x and y tables
        self.position_embedding_table = nn.Parameter([2, position_embedding_size, hidden_size])
        self.position_embedding_size = position_embedding_size

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        pixel_position_ids: ir.Value,
    ) -> tuple[ir.Value, ir.Value]:
        """Return ``(hidden_states [B, N, hidden], is_padding [B, N bool])``.

        Padding patches are indicated by ``pixel_position_ids == -1``.  Their
        position embeddings are zeroed exactly as HF does in
        ``Gemma4VisionPatchEmbedder._position_embeddings``.
        """
        # pixel_values in [0,1] -> normalize to [-1, 1]: 2*(v - 0.5) = 2v - 1
        two = op.CastLike(op.Constant(value_float=2.0), pixel_values)
        one = op.CastLike(op.Constant(value_float=1.0), pixel_values)
        pixel_values = op.Sub(op.Mul(pixel_values, two), one)
        hidden_states = self.input_proj(op, pixel_values)  # [B, N, hidden]

        # Detect padding patches: x-coord == -1 means the patch is padding.
        # pixel_position_ids [B, N, 2]; gather x-coord [B, N].
        x_raw = op.Gather(pixel_position_ids, op.Constant(value_int=0), axis=2)  # [B, N]
        is_padding = op.Equal(x_raw, op.Constant(value_int=-1))  # [B, N] bool

        # Clamp to ≥0 before embedding lookup (padding → 0 temporarily)
        clamped = op.Clip(pixel_position_ids, op.Constant(value_int=0))

        # Extract x and y coordinates: each [B, N]
        x_coords = op.Gather(clamped, op.Constant(value_int=0), axis=2)  # [B, N]
        y_coords = op.Gather(clamped, op.Constant(value_int=1), axis=2)  # [B, N]

        # Look up position embeddings from table
        x_table = op.Gather(self.position_embedding_table, op.Constant(value_int=0), axis=0)
        y_table = op.Gather(self.position_embedding_table, op.Constant(value_int=1), axis=0)
        x_emb = op.Gather(x_table, x_coords, axis=0)  # [B, N, hidden]
        y_emb = op.Gather(y_table, y_coords, axis=0)  # [B, N, hidden]

        # Zero position embeddings for padding patches.
        pos_emb = op.Add(x_emb, y_emb)  # [B, N, hidden]
        zero = op.CastLike(0.0, pos_emb)
        not_pad = op.Not(is_padding)  # [B, N]
        not_pad_3d = op.Unsqueeze(not_pad, [2])  # [B, N, 1]
        pos_emb = op.Where(not_pad_3d, pos_emb, zero)

        return op.Add(hidden_states, pos_emb), is_padding  # ([B, N, hidden], [B, N])


class _Gemma4VisionEncoderCore(nn.Module):
    """Gemma4 full vision encoder: patch embedding + transformer blocks.

    Accepts pre-patchified pixel values ``[B, N, 3*P^2]`` and position IDs
    ``[B, N, 2]``.  Returns patch features ``[B, N, vision_hidden]``.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        vc = config.vision  # VisionConfig for the SigLIP encoder
        linear_class, clippable_linear_class = _component_linear_classes(
            config,
            ("model.vision_tower", "model.embed_vision"),
        )
        self.patch_embedder = _Gemma4VisionPatchEmbedder(
            patch_size=vc.patch_size or 16,
            hidden_size=vc.hidden_size,
            position_embedding_size=vc.position_embedding_size or 128,
            linear_class=linear_class,
        )
        self.layers = nn.ModuleList(
            [
                Gemma4VisionEncoderLayer(
                    hidden_size=vc.hidden_size,
                    intermediate_size=vc.intermediate_size,
                    num_heads=vc.num_attention_heads,
                    norm_eps=vc.norm_eps,
                    hidden_act=vc.hidden_act or "gelu_pytorch_tanh",
                    rope_theta=vc.rope_theta or 100.0,
                    max_position=vc.position_embedding_size or 128,
                    use_clipped_linears=vc.use_clipped_linears,
                    linear_class=linear_class,
                    clippable_linear_class=clippable_linear_class,
                )
                for _ in range(vc.num_hidden_layers)
            ]
        )
        # No post-encoder norm: HF Gemma4VisionEncoder has none.
        # The scale-free RMSNorm (embedding_pre_projection_norm) lives in
        # _Gemma4VisionEncoderModel.projector_norm, applied before the projector.

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        pixel_position_ids: ir.Value,
    ) -> ir.Value:
        # Returns hidden_states [B, N, hidden] and is_padding [B, N bool]
        hidden_states, is_padding = self.patch_embedder(op, pixel_values, pixel_position_ids)

        # Build additive attention bias [B, 1, 1, N] masking out padding columns.
        # Valid positions get 0 (no effect), padding columns get -1e9 (suppressed).
        neg_inf = op.CastLike(-1e9, hidden_states)
        zero = op.CastLike(0.0, hidden_states)
        attn_bias = op.Where(is_padding, neg_inf, zero)  # [B, N]
        attn_bias = op.Unsqueeze(attn_bias, [1, 2])  # [B, 1, 1, N]

        for layer in self.layers:
            hidden_states = layer(op, hidden_states, attn_bias, pixel_position_ids)

        # Zero padding patches after all encoder blocks (matching HF pooler masked_fill).
        is_pad_expanded = op.Unsqueeze(is_padding, [2])  # [B, N, 1]
        zero = op.CastLike(op.Constant(value_float=0.0), hidden_states)
        hidden_states = op.Where(is_pad_expanded, zero, hidden_states)

        return hidden_states  # [B, N, vision_hidden]


# ---------------------------------------------------------------------------
# Gemma4 text decoder layers
# ---------------------------------------------------------------------------


class Gemma4TextAttention(nn.Module):
    """Gemma4 text multi-head attention with per-head QKV norms and KV sharing.

    Key differences from standard Attention:
    - Fixed scale=1.0 (HF hardcodes this)
    - Q and K normalized per-head with learnable RMSNorm
    - V normalized per-head with parameterless RMS (no learnable scale)
    - head_dim and rotary_embedding_dim differ between sliding/full layers
    - KV-shared layers borrow K,V from a source layer (no k/v projections)
    - Attention logit softcapping via the native ONNX Attention ``softcap``
      attribute (opset 24): ``tanh(qk / cap) * cap`` is applied after the
      QK dot-product and before softmax. The value is taken from
      ``config.attn_logit_softcapping`` (50.0 for Gemma4; 0.0 = disabled).

    Args:
        config: Gemma4Config.
        layer_idx: Index of this layer (0-based).
        layer_types: Full list of layer types for all layers.
        first_kv_shared_layer_idx: First layer index where KV sharing starts.
        head_dim: Head dimension (differs per layer type).
        rotary_embedding_dim: Dims to rotate (0 = full rotation).
    """

    def __init__(
        self,
        config: Gemma4Config,
        layer_idx: int,
        layer_types: list[str],
        first_kv_shared_layer_idx: int,
        head_dim: int,
        rotary_embedding_dim: int,
    ):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = head_dim
        self.scaling = 1.0
        # attn_logit_softcapping maps directly to the ONNX Attention op's
        # native ``softcap`` attribute (opset 24). No manual Tanh/scale ops needed.
        self.softcap = config.attn_logit_softcapping
        self._v_norm_eps = config.rms_norm_eps
        self.rotary_embedding_dim = rotary_embedding_dim
        self._rope_interleave = config.rope_interleave
        self.layer_idx = layer_idx

        # Alternative attention: full_attention layers with k_eq_v share V=K
        # (no separate v_proj). Uses fewer KV heads (num_global_key_value_heads).
        is_sliding = layer_types[layer_idx] == "sliding_attention"
        self._use_alternative_attention = (
            getattr(config, "attention_k_eq_v", False) and not is_sliding
        )
        # Full-attention layers use num_global_key_value_heads when set,
        # independent of the k_eq_v flag.
        if not is_sliding and config.num_global_key_value_heads is not None:
            self.num_key_value_heads = config.num_global_key_value_heads
        else:
            self.num_key_value_heads = config.num_key_value_heads

        # KV sharing: layers >= first_kv_shared_layer_idx borrow K,V from source
        self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        prev_layers = layer_types[:first_kv_shared_layer_idx]
        if self.is_kv_shared_layer:
            # Reverse-scan prev_layers to find the last non-shared layer with the
            # same type — KV-shared layers borrow K,V from that source layer.
            self.kv_shared_layer_index = (
                len(prev_layers) - 1 - prev_layers[::-1].index(layer_types[layer_idx])
            )
            self.provides_shared_kv = False
        else:
            self.kv_shared_layer_index = None
            # True for the last non-shared layer of each type that has downstream
            # KV-shared layers depending on it — it must store its K,V for reuse.
            self.provides_shared_kv = first_kv_shared_layer_idx > 0 and (
                layer_idx
                == len(prev_layers) - 1 - prev_layers[::-1].index(layer_types[layer_idx])
            )

        # Text-decoder projections are quantized (MatMulNBits) when the config
        # requests it; otherwise plain float Linear. Attention norms stay float.
        linear_class = _text_linear_class(config) or Linear

        # All layers have Q projection + Q norm + output projection
        self.q_proj = linear_class(
            config.hidden_size, config.num_attention_heads * head_dim, bias=False
        )
        self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.o_proj = linear_class(
            config.num_attention_heads * head_dim, config.hidden_size, bias=False
        )

        # KV-shared layers borrow K,V — no projections needed
        if not self.is_kv_shared_layer:
            self.k_proj = linear_class(
                config.hidden_size, self.num_key_value_heads * head_dim, bias=False
            )
            # Alternative attention (k_eq_v): V = K, no separate v_proj
            if not self._use_alternative_attention:
                self.v_proj = linear_class(
                    config.hidden_size, self.num_key_value_heads * head_dim, bias=False
                )
            self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | GQAContext,
        position_embeddings: tuple | None = None,
        shared_kv_states: dict | None = None,
        past_key_value: tuple | None = None,
        is_causal: int = 1,
        static_cache: StaticCacheState | None = None,
        static_kv_seqlen: ir.Value | None = None,
    ):
        from mobius.components._attention import (
            GQAContext,
            _apply_attention,
            apply_rotary_pos_emb,
        )

        use_gqa = isinstance(attention_bias, GQAContext)
        # Static-cache mode: fixed-width pre-allocated KV buffers (TensorScatter
        # in place). Signalled by static_kv_seqlen being set. Cache-owning layers
        # additionally receive their own StaticCacheState via static_cache; KV-shared
        # layers receive static_cache=None and read the source layer's full buffer.
        is_static = static_kv_seqlen is not None

        # Q projection + per-head Q norm
        # For GQA, skip manual RoPE — the op applies it internally.
        query_states = self.q_proj(op, hidden_states)
        query_states = op.Reshape(query_states, [0, 0, -1, self.head_dim])
        query_states = self.q_norm(op, query_states)
        query_states = op.Reshape(query_states, [0, 0, -1])

        if not use_gqa and position_embeddings is not None:
            query_states = apply_rotary_pos_emb(
                op,
                x=query_states,
                position_embeddings=position_embeddings,
                num_heads=self.num_attention_heads,
                rotary_embedding_dim=self.rotary_embedding_dim,
                interleaved=self._rope_interleave,
            )

        if self.is_kv_shared_layer:
            # KV-shared layers borrow K,V from a source layer (no own KV cache).
            src_key, src_value = shared_kv_states[self.kv_shared_layer_index][:2]

            if is_static:
                # Static cache: the source layer scattered its RoPE'd K,V into a
                # fixed-width buffer and stored the full 3D [B, max_seq, kv_hidden]
                # present. This layer's query attends over that buffer directly.
                # attention_bias is the static-cache float bias (causal + sliding +
                # padding baked in), so is_causal=0. nonpad_kv_seqlen bounds the
                # valid prefix. No scatter here — this layer produces no cache.
                attn_output, present_key, present_value = op.Attention(
                    query_states,
                    src_key,
                    src_value,
                    attention_bias,
                    None,  # no past_key
                    None,  # no past_value
                    static_kv_seqlen,
                    q_num_heads=self.num_attention_heads,
                    kv_num_heads=self.num_key_value_heads,
                    scale=self.scaling,
                    softcap=self.softcap,
                    is_causal=0,
                    _outputs=3,
                )
                attn_output = self.o_proj(op, attn_output)
                return attn_output, (present_key, present_value)

            # The borrowed K,V are the source layer's ``present`` outputs. In the
            # standard (non-GQA) path those come from the opset-24 ``Attention``
            # op, whose ``present_key``/``present_value`` outputs have NO shape
            # inference in ORT — they arrive here rank-unknown. Left unannotated,
            # the downstream Transpose/Reshape lose the head dimension and this
            # layer's ``Attention`` infers a zero-width output, which then makes
            # ``o_proj``'s MatMul fail shape inference at model-load time. The
            # source shares the same KV-head/head-dim configuration as this
            # layer, so pin the known 4D BNSH shape to restore inference.
            for _kv in (src_key, src_value):
                if _kv.shape is None or len(_kv.shape) != 4:
                    _kv.shape = ir.Shape(
                        [
                            "batch",
                            self.num_key_value_heads,
                            "kv_sequence_length",
                            self.head_dim,
                        ]
                    )

            if use_gqa:
                # GQA path for shared KV: pass empty K/V tensors and wire the
                # source layer's present_key/value as past_key/past_value.
                # The shared buffer is already in BNSH format, so GQA reads it
                # directly — no Transpose/Reshape needed.
                gqa_ctx = attention_bias

                # Create empty K/V tensors with kv_sequence_length=0.
                # Shape: [batch, 0, kv_heads * head_dim]
                batch_dim = op.Shape(query_states, start=0, end=1)
                kv_hidden = self.num_key_value_heads * self.head_dim
                empty_shape = op.Concat(
                    batch_dim,
                    op.Constant(value_ints=[0, kv_hidden]),
                    axis=0,
                )
                empty_kv = op.CastLike(op.ConstantOfShape(empty_shape), query_states)

                gqa_attrs: dict = {
                    "num_heads": self.num_attention_heads,
                    "kv_num_heads": self.num_key_value_heads,
                    "scale": self.scaling,
                    "do_rotary": 1,
                    "rotary_interleaved": int(self._rope_interleave),
                }
                if self.softcap:
                    gqa_attrs["softcap"] = self.softcap
                if self.rotary_embedding_dim:
                    gqa_attrs["rotary_embedding_dim"] = self.rotary_embedding_dim
                if gqa_ctx.local_window_size > 0:
                    gqa_attrs["local_window_size"] = gqa_ctx.local_window_size

                attn_output, present_key, present_value = op.GroupQueryAttention(
                    query_states,
                    empty_kv,  # key: empty (kv_sequence_length=0)
                    empty_kv,  # value: empty (kv_sequence_length=0)
                    src_key,  # past_key: shared KV in BNSH
                    src_value,  # past_value: shared KV in BNSH
                    gqa_ctx.seqlens_k,
                    gqa_ctx.total_seq_len,
                    gqa_ctx.cos_cache,
                    gqa_ctx.sin_cache,
                    _domain="com.microsoft",
                    _outputs=3,
                    **gqa_attrs,
                )
            else:
                # Fallback Attention path: transpose shared KV from BNSH to 3D.
                # Source K/V shape depends on whether the source layer uses
                # static or dynamic cache:
                #   Dynamic: [B, kv_heads, total_seq, head_dim] (4D present)
                #   Static:  [B, max_seq, kv_heads*head_dim]    (3D updated cache)
                # Static cache sources are already 3D — skip reshape.
                is_static_source = src_key.shape is not None and len(src_key.shape) == 3
                if not is_static_source:
                    shared_kv_hidden = self.num_key_value_heads * self.head_dim
                    src_key = op.Transpose(src_key, perm=[0, 2, 1, 3])
                    src_key = op.Reshape(src_key, [0, 0, shared_kv_hidden])
                    src_value = op.Transpose(src_value, perm=[0, 2, 1, 3])
                    src_value = op.Reshape(src_value, [0, 0, shared_kv_hidden])

                # KV-shared layers always use the dynamic Attention path with
                # mask (attention_bias). Even when the source layer uses static
                # cache, the KV-shared layer's Attention uses the source's full
                # cache as K/V with past_key=None (no own KV concat).
                # The nonpad_kv_seqlen path is NOT used here because ORT's
                # is_causal=0 + nonpad_kv_seqlen triggers a CUDA kernel issue
                # for KV-shared decode (S_q=1, S_kv=max_seq).
                attn_output, present_key, present_value = _apply_attention(
                    op,
                    query_states,
                    src_key,
                    src_value,
                    attention_bias,
                    past_key=None,
                    past_value=None,
                    num_attention_heads=self.num_attention_heads,
                    num_key_value_heads=self.num_key_value_heads,
                    scale=self.scaling,
                    softcap=self.softcap,
                    is_causal=0,
                )
        elif use_gqa:
            # GQA path: emit com.microsoft.GroupQueryAttention directly.
            # The op fuses RoPE + attention + KV cache into a single op,
            # with optional sliding-window via local_window_size.
            gqa_ctx = attention_bias

            # K projection + per-head K norm (no manual RoPE — GQA does it)
            key_raw = self.k_proj(op, hidden_states)
            key_states = op.Reshape(key_raw, [0, 0, -1, self.head_dim])
            key_states = self.k_norm(op, key_states)
            key_states = op.Reshape(key_states, [0, 0, -1])

            # V: separate projection, or V=K (alternative attention)
            if self._use_alternative_attention:
                value_raw = key_raw
            else:
                value_raw = self.v_proj(op, hidden_states)
            # Parameterless per-head V normalisation (FP32 accumulation to
            # prevent FP16 overflow when squaring values > 256).
            value_states = op.Reshape(
                value_raw,
                op.Constant(value_ints=[0, 0, self.num_key_value_heads, self.head_dim]),
            )
            v_f32 = op.Cast(value_states, to=ir.DataType.FLOAT)
            sq = op.Mul(v_f32, v_f32)
            mean_sq = op.ReduceMean(sq, [-1], keepdims=1)
            eps = op.Constant(value_floats=[self._v_norm_eps])
            rms = op.Sqrt(op.Add(mean_sq, eps))
            value_states = op.CastLike(op.Div(v_f32, rms), value_states)
            value_states = op.Reshape(value_states, [0, 0, -1])

            # Build GQA attributes
            past_key = past_key_value[0] if past_key_value is not None else None
            past_value = past_key_value[1] if past_key_value is not None else None

            gqa_attrs: dict = {
                "num_heads": self.num_attention_heads,
                "kv_num_heads": self.num_key_value_heads,
                "scale": self.scaling,
                "do_rotary": 1,
                "rotary_interleaved": int(self._rope_interleave),
            }
            if self.softcap:
                gqa_attrs["softcap"] = self.softcap
            if self.rotary_embedding_dim:
                gqa_attrs["rotary_embedding_dim"] = self.rotary_embedding_dim
            if gqa_ctx.local_window_size > 0:
                gqa_attrs["local_window_size"] = gqa_ctx.local_window_size

            attn_output, present_key, present_value = op.GroupQueryAttention(
                query_states,
                key_states,
                value_states,
                past_key,
                past_value,
                gqa_ctx.seqlens_k,
                gqa_ctx.total_seq_len,
                gqa_ctx.cos_cache,
                gqa_ctx.sin_cache,
                _domain="com.microsoft",
                _outputs=3,
                **gqa_attrs,
            )

            # Source layers store K,V for downstream KV-shared layers.
            if self.provides_shared_kv and shared_kv_states is not None:
                shared_kv_states[self.layer_idx] = (
                    present_key,
                    present_value,
                    None,  # no nonpad_kv_seqlen for GQA path
                )
        else:
            # K projection + per-head K norm + optional RoPE
            key_raw = self.k_proj(op, hidden_states)
            key_states = op.Reshape(key_raw, [0, 0, -1, self.head_dim])
            key_states = self.k_norm(op, key_states)
            key_states = op.Reshape(key_states, [0, 0, -1])

            if position_embeddings is not None:
                key_states = apply_rotary_pos_emb(
                    op,
                    x=key_states,
                    position_embeddings=position_embeddings,
                    num_heads=self.num_key_value_heads,
                    rotary_embedding_dim=self.rotary_embedding_dim,
                    interleaved=self._rope_interleave,
                )

            # V: separate projection, or V=K (alternative attention)
            if self._use_alternative_attention:
                value_raw = key_raw
            else:
                value_raw = self.v_proj(op, hidden_states)
            # Parameterless per-head V normalisation (FP32 accumulation to
            # prevent FP16 overflow when squaring values > 256).
            value_states = op.Reshape(
                value_raw,
                op.Constant(value_ints=[0, 0, self.num_key_value_heads, self.head_dim]),
            )
            v_f32 = op.Cast(value_states, to=ir.DataType.FLOAT)
            sq = op.Mul(v_f32, v_f32)
            mean_sq = op.ReduceMean(sq, [-1], keepdims=1)
            # Use op.Constant to create a 1D tensor node (not a scalar initializer).
            # Scalar Python floats use a type-keyed cache that can fail when upstream
            # type information is missing (e.g., after custom ops like com.microsoft.MoE).
            eps = op.Constant(value_floats=[self._v_norm_eps])
            rms = op.Sqrt(op.Add(mean_sq, eps))
            value_states = op.CastLike(op.Div(v_f32, rms), value_states)
            value_states = op.Reshape(value_states, [0, 0, -1])

            attn_output, present_key, present_value = _apply_attention(
                op,
                query_states,
                key_states,
                value_states,
                attention_bias,
                None
                if static_cache is not None
                else (past_key_value[0] if past_key_value is not None else None),
                None
                if static_cache is not None
                else (past_key_value[1] if past_key_value is not None else None),
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
                scale=self.scaling,
                softcap=self.softcap,
                static_cache=static_cache,
                is_causal=is_causal,
            )

            # Source layers store K,V for downstream KV-shared layers.
            # Include nonpad_kv_seqlen for static cache sources so
            # KV-shared layers can pass it to the Attention op.
            if self.provides_shared_kv and shared_kv_states is not None:
                nonpad = static_cache.nonpad_kv_seqlen if static_cache else None
                shared_kv_states[self.layer_idx] = (
                    present_key,
                    present_value,
                    nonpad,
                )

        attn_output = self.o_proj(op, attn_output)
        return attn_output, (present_key, present_value)


# ---------------------------------------------------------------------------
# Gemma4 MoE router
# ---------------------------------------------------------------------------


class _Gemma4MoeRouter(nn.Module):
    """Gemma4 MoE router: scale-free RMSNorm → learned scale → linear → softmax.

    Matches ``Gemma4TextRouter`` in HuggingFace.

    Parameters (aligned with HF state_dict, with one weight-fold):
    - ``scale`` [hidden_size]: ``router.scale * hidden_size^-0.5`` (folded during
      ``preprocess_weights`` — see ``Gemma4CausalLMModel.preprocess_weights``)
    - ``proj.weight`` [num_experts, hidden_size]: router logits (no bias)
    - ``per_expert_scale`` [num_experts]: per-expert output weight scaling

    ``router.norm`` is scale-free (no learnable parameter, omitted from state_dict).

    The ``hidden_size^-0.5`` scale factor is absorbed into ``self.scale`` at weight-load
    time to avoid graph-level float-constant collisions across multiple decoder layers.

    Args:
        hidden_size: Model hidden dimension.
        num_experts: Total number of experts.
        rms_norm_eps: Epsilon for the scale-free RMSNorm.
    """

    def __init__(self, hidden_size: int, num_experts: int, rms_norm_eps: float = 1e-6):
        super().__init__()
        self._eps = rms_norm_eps
        self._hidden_size = hidden_size
        # ``scale`` stores router.scale * hidden_size^-0.5 (folded in preprocess_weights)
        self.scale = nn.Parameter([hidden_size])
        self.proj = Linear(hidden_size, num_experts, bias=False)
        self.per_expert_scale = nn.Parameter([num_experts])

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        """Compute router probabilities over all experts.

        Args:
            hidden_states: [num_tokens, hidden_size] (batch-sequence flattened)

        Returns:
            router_probs: [num_tokens, num_experts] full softmax probabilities
        """
        # Scale-free RMSNorm using RMSNormalization op (epsilon as attribute avoids
        # graph-level float constant name collisions across multiple decoder layers).
        h_shape = op.Shape(hidden_states, start=1, end=2)  # [1] containing hidden_size
        ones = op.CastLike(
            op.ConstantOfShape(h_shape, value=ir.tensor(np.ones(1, dtype=np.float32))),
            hidden_states,
        )
        x_normed = op.RMSNormalization(hidden_states, ones, epsilon=self._eps, axis=-1)
        # Scale: x_normed * self.scale  (hidden_size^-0.5 already folded into scale)
        x_scaled = op.Mul(x_normed, self.scale)
        # Linear projection then softmax -> router_probs [num_tokens, num_experts]
        expert_scores = self.proj(op, x_scaled)
        return op.Softmax(expert_scores, axis=-1)


class Gemma4DecoderLayer(nn.Module):
    """Gemma4 text decoder layer with 4 norms, layer_scalar, and optional per-layer input.

    Architecture (dense path):
        h = residual + post_attn_norm(attn(input_layernorm(h)))
        h = residual + post_ff_norm(mlp(pre_ff_norm(h)))
        if per_layer_input:
            h += post_per_layer_norm(project(act(gate(h)) * per_layer_input))
        h = h * layer_scalar   # LAST step, after per-layer input

    When ``enable_moe_block=True`` (Gemma4 26B-A4B, 31B), the MLP block uses a
    parallel architecture — a dense MLP and a MoE block run independently, then
    their outputs are summed before the final post-FF norm::

        h = pre_ff_norm(h)                                    # pre_feedforward_layernorm
        h_dense  = post_ff_norm_dense(mlp(h))                 # post_feedforward_layernorm_1
        h_moe    = post_ff_norm_moe(                          # post_feedforward_layernorm_2
            experts(pre_ff_norm_moe(residual), router(residual))  # pre_feedforward_layernorm_2
        )
        h = post_ff_norm(h_dense + h_moe)                    # post_feedforward_layernorm
        h = residual + h

    Uses standard RMSNorm (not OffsetRMSNorm). KV-shared layers use double-wide
    MLP when config.use_double_wide_mlp=True.
    """

    # Marks this layer as implementing the StaticCacheState dispatch in forward()
    # so CausalLMTask(static_cache=True) accepts Gemma4 (see
    # _validate_static_cache_support). Gemma4 has KV-shared + sliding/full layers
    # with dual head_dim, handled by static_kv_cache_specs() on the text model.
    _supports_static_cache = True

    def __init__(self, config: Gemma4Config, layer_idx: int):
        super().__init__()
        layer_types = config.layer_types or ["sliding_attention"] * config.num_hidden_layers
        if len(layer_types) != config.num_hidden_layers:
            raise ValueError(
                f"Gemma4Config.layer_types length ({len(layer_types)}) "
                f"must match num_hidden_layers ({config.num_hidden_layers})"
            )
        first_kv_shared = config.num_hidden_layers - config.num_kv_shared_layers
        layer_type = layer_types[layer_idx]
        is_full = layer_type == "full_attention"

        head_dim = (config.global_head_dim or config.head_dim) if is_full else config.head_dim
        # ProportionalRope handles partial rotation via zero-padded cos/sin (full head_dim
        # coverage). The ONNX RotaryEmbedding op must see rotary_embedding_dim=0 (full)
        # so it pairs dims using the split-half convention over the entire head_dim.
        # For sliding (DefaultRope, full rotation), rotary_embedding_dim=0 is also correct.
        rotary_dim = 0

        self.self_attn = Gemma4TextAttention(
            config,
            layer_idx=layer_idx,
            layer_types=layer_types,
            first_kv_shared_layer_idx=first_kv_shared,
            head_dim=head_dim,
            rotary_embedding_dim=rotary_dim,
        )

        is_kv_shared = layer_idx >= first_kv_shared > 0
        intermediate_size = config.intermediate_size * (
            2 if (config.use_double_wide_mlp and is_kv_shared) else 1
        )
        linear_class = _text_linear_class(config) or Linear
        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            activation=config.hidden_act,
            bias=config.mlp_bias,
            linear_class=linear_class,
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # layer_scalar: learned or constant scalar applied at end of layer
        self.layer_scalar = nn.Parameter([1])

        self._per_layer_dim = config.hidden_size_per_layer_input
        if self._per_layer_dim > 0:
            self.per_layer_input_gate = linear_class(
                config.hidden_size, self._per_layer_dim, bias=False
            )
            self.per_layer_projection = linear_class(
                self._per_layer_dim, config.hidden_size, bias=False
            )
            self.post_per_layer_input_norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.act_fn = get_activation(config.hidden_act)

        self._enable_moe_block = config.enable_moe_block
        if config.enable_moe_block:
            assert config.num_local_experts is not None, "num_local_experts required for MoE"
            assert config.num_experts_per_tok is not None, (
                "num_experts_per_tok required for MoE"
            )
            assert config.moe_intermediate_size is not None, (
                "moe_intermediate_size required for MoE"
            )

            self._top_k = config.num_experts_per_tok
            self._num_experts = config.num_local_experts
            self._moe_intermediate_size = config.moe_intermediate_size
            self._hidden_size = config.hidden_size
            moe_inter = config.moe_intermediate_size

            self.router = _Gemma4MoeRouter(
                config.hidden_size, config.num_local_experts, config.rms_norm_eps
            )
            # Expert weights stored as 3D tensors (all experts stacked).
            # fc1_experts_weights: gate+up combined [E, 2*moe_inter, hidden].
            # fc2_experts_weights: down projection [E, hidden, moe_inter].
            # These map to HF experts.gate_up_proj and experts.down_proj.
            self.fc1_experts_weights = nn.Parameter(
                [config.num_local_experts, 2 * moe_inter, config.hidden_size]
            )
            self.fc2_experts_weights = nn.Parameter(
                [config.num_local_experts, config.hidden_size, moe_inter]
            )
            # MoE-specific norms (in addition to the shared pre/post_feedforward norms)
            self.pre_feedforward_layernorm_2 = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_feedforward_layernorm_1 = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_feedforward_layernorm_2 = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | GQAContext,
        position_embeddings: tuple | None,
        shared_kv_states: dict,
        per_layer_input: ir.Value | None,
        past_key_value: tuple | StaticCacheState | None,
        is_causal: int = 1,
        static_kv_seqlen: ir.Value | None = None,
    ):
        from mobius.components._attention import StaticCacheState

        # Static-cache dispatch: a StaticCacheState arrives via past_key_value for
        # cache-owning layers; unpack it into static_cache and clear past_key_value.
        # KV-shared layers get past_key_value=None but still run in static mode,
        # signalled by static_kv_seqlen (shared across all layers).
        static_cache: StaticCacheState | None = None
        if isinstance(past_key_value, StaticCacheState):
            static_cache = past_key_value
            past_key_value = None

        # Attention block: pre-norm -> attn -> post-norm -> residual
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        attn_output, present_key_value = self.self_attn(
            op,
            hidden_states=hidden_states,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            shared_kv_states=shared_kv_states,
            past_key_value=past_key_value,
            static_cache=static_cache,
            is_causal=is_causal,
            static_kv_seqlen=static_kv_seqlen,
        )
        hidden_states = self.post_attention_layernorm(op, attn_output)
        hidden_states = op.Add(residual, hidden_states)

        # MLP block: dense MLP path runs always; parallel MoE path added when enabled.
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)

        if self._enable_moe_block:
            # Hybrid dense+MoE architecture:
            #   dense path output: post_ff_norm_1(mlp(pre_ff_norm(h)))
            #   moe   path output: post_ff_norm_2(experts(pre_ff_norm_2(residual)))
            #   combined: post_ff_norm(dense + moe) + residual
            dense_out = self.post_feedforward_layernorm_1(op, hidden_states)

            # MoE input is the pre-attention residual (flattened to 2D for routing).
            batch_size = op.Shape(residual, start=0, end=1)  # [1] scalar
            seq_len = op.Shape(residual, start=1, end=2)  # [1] scalar
            hidden_size = op.Shape(residual, start=2, end=3)  # [1] scalar
            num_tokens = op.Mul(batch_size, seq_len)  # [1]
            flat_shape = op.Concat(num_tokens, hidden_size, axis=0)  # [2]
            residual_flat = op.Reshape(residual, flat_shape)  # [B*S, H]

            # router_probs: [B*S, E] full softmax over all experts
            router_probs = self.router(op, residual_flat)  # [B*S, E]

            # Norm residual before experts
            normed_flat = self.pre_feedforward_layernorm_2(op, residual_flat)  # [B*S, H]

            caps = ep_capabilities()
            if caps.supports_fused_moe:
                # Fused com.microsoft::MoE op handles top-k selection +
                # expert dispatch internally. Requires ORT main (post
                # microsoft/onnxruntime#28467, MoE GEMM Refactor) which
                # plumbs the SwiGLU schema attributes to the kernel.
                #
                # Gemma 4 SwiGLU semantics (vs GPT-OSS):
                #   activation_alpha=1.0   — silu(gate) (no GPT-OSS alpha=1.702)
                #   activation_beta=0.0    — linear * gate (no GPT-OSS "+1" bias)
                #   swiglu_limit=inf       — no clipping (≤0 disables the clamp)
                #   swiglu_fusion=1        — interleaved layout
                #                            [g_0, u_0, g_1, u_1, ...].
                #
                # mobius stores ``fc1_experts_weights`` chunked as
                # ``[E, 2*inter, H]`` (first ``inter`` rows = gate,
                # next ``inter`` = up) because that matches HuggingFace
                # ``experts.gate_up_proj``. The fused op needs the
                # interleaved layout, so reshape ``[E, 2, inter, H]`` →
                # transpose to ``[E, inter, 2, H]`` → flatten back to
                # ``[E, 2*inter, H]``. The whole chain operates on a
                # constant initializer so ORT folds it into a single
                # static tensor at session load.
                #
                # ``swiglu_fusion=1`` is required because the CPU MoE
                # kernel still only supports the interleaved layout
                # (``contrib_ops/cpu/moe/moe_cpu.cc:27``); the new CUDA
                # kernel accepts either.
                #
                # CastLike restores the input dtype because op.MoE is a
                # custom op with type=None on its output; without the
                # cast downstream type inference cannot share scalar
                # initializers in bf16/fp16 graphs.
                e_dim = self._num_experts
                inter = self._moe_intermediate_size
                hidden = self._hidden_size
                fc1_interleaved = op.Reshape(
                    op.Transpose(
                        op.Reshape(
                            self.fc1_experts_weights,
                            op.Constant(value_ints=[e_dim, 2, inter, hidden]),
                        ),
                        perm=[0, 2, 1, 3],
                    ),
                    op.Constant(value_ints=[e_dim, 2 * inter, hidden]),
                )  # [E, 2*inter, H] interleaved
                moe_out_flat = op.CastLike(
                    op.MoE(  # type: ignore[attr-defined]
                        normed_flat,
                        router_probs,
                        fc1_interleaved,
                        None,  # fc1_experts_bias (slot 3, optional)
                        self.fc2_experts_weights,
                        activation_type="swiglu",
                        k=self._top_k,
                        normalize_routing_weights=1,
                        activation_alpha=1.0,
                        activation_beta=0.0,
                        swiglu_limit=float("inf"),
                        swiglu_fusion=1,
                        _domain="com.microsoft",
                    ),
                    normed_flat,  # match input dtype (bf16/fp16/fp32)
                )  # [B*S, H]
            else:
                # EPs without fused MoE support fall back to a static
                # per-expert unroll.
                moe_out_flat = self._dispatch_moe_fallback(op, normed_flat, router_probs)

            moe_out = op.Reshape(moe_out_flat, op.Shape(residual))  # [B, S, H]
            moe_out = self.post_feedforward_layernorm_2(op, moe_out)
            ff_out = op.Add(dense_out, moe_out)
            hidden_states = self.post_feedforward_layernorm(op, ff_out)
            hidden_states = op.Add(residual, hidden_states)
        else:
            hidden_states = self.post_feedforward_layernorm(op, hidden_states)
            hidden_states = op.Add(residual, hidden_states)

        # Per-layer input gating (skip when disabled)
        if self._per_layer_dim > 0 and per_layer_input is not None:
            residual = hidden_states
            gated = self.per_layer_input_gate(op, hidden_states)
            gated = self.act_fn(op, gated)
            gated = op.Mul(gated, per_layer_input)
            projected = self.per_layer_projection(op, gated)
            projected = self.post_per_layer_input_norm(op, projected)
            hidden_states = op.Add(residual, projected)

        # Layer scalar LAST (after per-layer input contribution)
        hidden_states = op.Mul(hidden_states, self.layer_scalar)

        return hidden_states, present_key_value

    def _dispatch_moe_fallback(
        self,
        op: OpBuilder,
        normed_flat: ir.Value,
        router_probs: ir.Value,
    ) -> ir.Value:
        """Fallback expert dispatch (static unroll) when fused MoE op is unavailable.

        Iterates over each expert (outer loop), accumulates the routing weight for
        all k slots that select that expert (inner loop), applies the SwiGLU MLP,
        and accumulates into the output tensor.

        This produces O(E x K) ONNX nodes (e.g. Gemma4 26B: 256 experts x 2 top-k =
        512 per layer).  The primary path uses a fused MoE op for supported EPs
        (CUDA/DML); this fallback is only invoked for CPU/other EPs where the fused
        op is absent.

        Args:
            normed_flat: [T, H] — pre-normed input for the experts (T = B*S).
            router_probs: [T, E] — full softmax router probabilities.

        Returns:
            [T, H] — weighted sum of expert outputs.
        """
        # Top-K selection: top_weights/top_indices both [T, K]
        top_weights_raw, top_indices = op.TopK(
            router_probs, op.Constant(value_ints=[self._top_k]), axis=-1, _outputs=2
        )
        # Arithmetic normalisation: weights sum to 1 (matches HF: top_k_weights /= top_k_weights.sum(-1, keepdim=True))
        top_weights = op.Div(
            top_weights_raw, op.ReduceSum(top_weights_raw, [1], keepdims=1)
        )  # [T, K]

        # Scale routing weights by per_expert_scale for each selected expert
        # Gather from [E] using [T, K] indices → result [T, K]
        pes_topk = op.Gather(self.router.per_expert_scale, top_indices, axis=0)  # [T, K]
        top_weights = op.Mul(top_weights, op.CastLike(pes_topk, top_weights))  # [T, K]

        # Build output accumulator, matching dtype of the input
        out_shape = op.Shape(normed_flat)  # [T, H] shape vec
        output = op.CastLike(
            op.ConstantOfShape(out_shape, value=ir.tensor(np.zeros(1, dtype=np.float32))),
            normed_flat,
        )

        # T_shape: 1-D tensor [T] used for per-expert weight accumulation
        t_shape = op.Shape(normed_flat, start=0, end=1)  # shape vec of length 1

        for e_idx in range(self._num_experts):
            # Gather this expert's weight matrices
            fc1 = op.Squeeze(
                op.Gather(self.fc1_experts_weights, [e_idx], axis=0), [0]
            )  # [2*moe_inter, H]
            fc2 = op.Squeeze(
                op.Gather(self.fc2_experts_weights, [e_idx], axis=0), [0]
            )  # [H, moe_inter]

            # Gated SiLU MLP: fc1 produces gate+up concatenated → SwiGLU
            proj = op.MatMul(normed_flat, op.Transpose(fc1))  # [T, 2*moe_inter]
            half = op.Shape(fc2, start=1, end=2)  # [moe_inter]
            gate = op.Slice(proj, [0], half, [1])  # [T, moe_inter] first half
            up = op.Slice(proj, half, op.Shape(proj, start=1, end=2), [1])  # second half
            # SiLU(gate) = gate * sigmoid(gate)
            expert_out = op.MatMul(
                op.Mul(op.Mul(gate, op.Sigmoid(gate)), up),  # [T, moe_inter]
                op.Transpose(fc2),  # → [T, H]
            )

            # Accumulate routing weight for expert e_idx across all k slots
            e_weight = op.CastLike(
                op.ConstantOfShape(t_shape, value=ir.tensor(np.zeros(1, dtype=np.float32))),
                top_weights,
            )
            for k_idx in range(self._top_k):
                # idx_k: [T] — which expert was selected at slot k_idx
                idx_k = op.Squeeze(op.Slice(top_indices, [k_idx], [k_idx + 1], [1]), [1])
                # w_k: [T] — routing weight for slot k_idx
                w_k = op.Squeeze(op.Slice(top_weights, [k_idx], [k_idx + 1], [1]), [1])
                # Add w_k only for tokens routed to expert e_idx at this slot
                is_expert = op.CastLike(op.Equal(idx_k, op.Constant(value_int=e_idx)), w_k)
                e_weight = op.Add(e_weight, op.Mul(is_expert, w_k))

            # Weight expert output by aggregated routing weight and add to output
            e_weight_2d = op.Reshape(
                e_weight,
                op.Concat(t_shape, op.Constant(value_ints=[1]), axis=0),  # [T, 1]
            )
            output = op.Add(output, op.Mul(expert_out, op.CastLike(e_weight_2d, expert_out)))

        return output


# ---------------------------------------------------------------------------
# Gemma4 text model
# ---------------------------------------------------------------------------


def _compute_block_sequence_ids(
    op: OpBuilder,
    input_ids: ir.Value,
    *,
    image_token_id: int,
) -> ir.Value:
    """Compute Gemma4 ``block_sequence_ids`` [B, S] from ``input_ids``.

    Mirrors HuggingFace ``get_block_sequence_ids_for_mask``: each contiguous
    run of image placeholder tokens gets a unique, monotonically increasing
    block id (``>= 0``); every other position (text **and audio**) gets ``-1``.
    Tokens within the same block may attend to each other bidirectionally.

    Only image tokens form blocks. HF derives ``is_vision`` from
    ``mm_token_type_ids`` as ``(== 1) | (== 2)`` (image or video); audio is
    token-type ``3`` and is deliberately excluded, so audio placeholders keep
    plain causal attention. gemma4_unified has no video modality, so this
    reduces to image tokens alone.

    A new block starts only on a non-image -> image transition.

    Returns an INT64 tensor of shape ``[B, S]``.
    """
    # is_vision [B, S] BOOL: token is an image placeholder. Audio tokens are
    # intentionally NOT included (HF block mask covers image/video only).
    is_vision = op.Equal(input_ids, op.Constant(value_int=image_token_id))

    # is_prev_vision: is_vision shifted right by one along the sequence axis,
    # with position 0 forced to False. Implemented without ConstantOfShape
    # (which blocks ONNX shape inference): left-pad the int mask with one
    # zero column, then drop the last column.
    is_vision_int = op.Cast(is_vision, to=ir.DataType.INT64)
    padded = op.Pad(
        is_vision_int,
        op.Constant(value_ints=[0, 1, 0, 0]),  # prepend 1 col on axis 1
        op.Constant(value_int=0),
    )  # [B, S + 1]
    prev_int = op.Slice(
        padded,
        op.Constant(value_ints=[0]),
        op.Constant(value_ints=[-1]),
        op.Constant(value_ints=[1]),
    )  # [B, S]
    is_prev_vision = op.Cast(prev_int, to=ir.DataType.BOOL)

    # new_vision_starts = is_vision AND NOT is_prev_vision
    new_starts = op.And(is_vision, op.Not(is_prev_vision))
    # vision_group_ids = cumsum(new_starts) - 1  (along sequence axis)
    group_ids = op.Sub(
        op.CumSum(op.Cast(new_starts, to=ir.DataType.INT64), op.Constant(value_int=1)),
        op.Constant(value_int=1),
    )
    # block_sequence_ids = where(is_vision, group_ids, -1)
    return op.Where(is_vision, group_ids, op.Constant(value_int=-1))


class Gemma4TextModel(nn.Module):
    """Gemma4 text transformer with hybrid local/global attention.

    Key differences from Gemma3TextModel:
    - Standard ``RMSNorm`` (no ``OffsetRMSNorm``).
    - Dual head_dim: local layers use ``config.head_dim``, global layers use
      ``config.global_head_dim``.
    - Dual RoPE: separate ``rotary_emb_local`` and ``rotary_emb_global`` instances
      with different theta and partial_rotary_factor.
    - Optional per-layer input embeddings (disabled when
      ``hidden_size_per_layer_input == 0``).

    Inputs may be ``input_ids`` (text-only) or ``inputs_embeds`` (VL decoder path).
    """

    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.config = config
        self._dtype = config.dtype

        embed_scale = math.sqrt(config.hidden_size)
        self.embed_tokens = _make_scaled_word_embedding(
            config,
            config.vocab_size,
            config.hidden_size,
            embed_scale,
        )

        layer_types = config.layer_types or ["sliding_attention"] * config.num_hidden_layers
        if len(layer_types) != config.num_hidden_layers:
            raise ValueError(
                f"Gemma4Config.layer_types length ({len(layer_types)}) must match "
                f"num_hidden_layers ({config.num_hidden_layers})"
            )
        self.layer_types = layer_types
        self.sliding_window = config.sliding_window
        # Bidirectional attention mode (None | "vision"). When "vision",
        # contiguous image-token blocks attend bidirectionally; the overlay is
        # derived from input_ids at runtime via ``block_sequence_ids`` and forces
        # the float-bias attention path (``is_causal=0``). HF also defines an
        # "all" mode (every token bidirectional, no causal mask); it is not used
        # by any supported Gemma4 checkpoint and not implemented here, so reject
        # it explicitly rather than silently falling back to causal attention.
        if config.use_bidirectional_attention not in (None, "vision"):
            raise NotImplementedError(
                "Gemma4 use_bidirectional_attention="
                f"{config.use_bidirectional_attention!r} is not supported; only "
                "None (fully causal) and 'vision' (image-block bidirectional) "
                "are implemented."
            )
        self._use_bidirectional_attention = config.use_bidirectional_attention

        # Local (sliding window) config — full rotation, local rope_theta
        local_config = dataclasses.replace(
            config,
            rope_type="default",
            rope_scaling=None,
            partial_rotary_factor=1.0,
        )
        # Global (full attention) config — larger head_dim, proportional RoPE
        # (partial rotation via zero-padded inv_freq to cover full head_dim)
        global_head_dim = config.global_head_dim or config.head_dim
        global_config = dataclasses.replace(
            config,
            head_dim=global_head_dim,
            rope_theta=config.global_rope_theta,
            partial_rotary_factor=config.global_partial_rotary_factor,
            rope_type="proportional",
            rope_scaling=None,
            sliding_window=None,
        )

        self.layers = nn.ModuleList(
            [Gemma4DecoderLayer(config, layer_idx=i) for i in range(len(layer_types))]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.rotary_emb_local = initialize_rope(local_config)
        self.rotary_emb_global = initialize_rope(global_config)

        # Per-layer input dimension (used by decoder layers).
        # In VLM 3-model split, per-layer inputs are precomputed by the
        # embedding model and passed as per_layer_inputs. In single-model
        # (text-only) mode, they are computed here from input_ids.
        self._per_layer_dim = getattr(config, "hidden_size_per_layer_input", 0)
        self._hidden_size = config.hidden_size
        self._image_token_id: int = config.image_token_id or 0
        # The vision-block overlay keys on image_token_id. A 0/None id means the
        # model has no image-placeholder token (e.g. the text-only backbone
        # split), so no image tokens can appear — keep pure causal attention
        # (GQA-eligible) and never build the overlay. This also avoids the
        # footgun of a 0 fallback marking real token-0 positions as vision.
        self._has_image_token: bool = bool(config.image_token_id)
        self._audio_token_id: int | None = (
            config.audio.audio_token_id if config.audio is not None else None
        )
        if self._per_layer_dim:
            self._num_layers = config.num_hidden_layers
            vocab_per_layer = getattr(config, "vocab_size_per_layer_input", 0)
            # Fused [V, L*D] table — used when split_per_layer_embedding is False.
            # Requires ORT >= 1.27 for CUDA Gather int64 index support (onnxruntime#28107).
            self.embed_tokens_per_layer = _make_scaled_word_embedding(
                config,
                vocab_per_layer,
                self._num_layers * self._per_layer_dim,
                float(self._per_layer_dim**0.5),
            )
            # Split [V, D] tables — used when split_per_layer_embedding is True
            # (i.e. the fused table exceeds the EP's max_buffer_size, e.g. WebGPU's
            # 256 MiB limit; ~128 MiB each vs ~4.7 GB fused).
            # Only the table actually called in forward() is realized as an
            # ONNX initializer, so the unused one adds no graph weight.
            self.embed_tokens_per_layer_split = nn.ModuleList(
                [
                    Gemma3TextScaledWordEmbedding(
                        vocab_per_layer,
                        self._per_layer_dim,
                        config.pad_token_id,
                        embed_scale=float(self._per_layer_dim**0.5),
                    )
                    for _ in range(self._num_layers)
                ]
            )
            self.per_layer_model_projection = _per_layer_model_projection_class(config)(
                config.hidden_size,
                config.num_hidden_layers * self._per_layer_dim,
                bias=False,
            )
            self.per_layer_projection_norm = RMSNorm(
                self._per_layer_dim, eps=config.rms_norm_eps
            )

    def static_kv_cache_specs(self) -> list[tuple[int, int]]:
        """Return ``(num_kv_heads, head_dim)`` for each cache-OWNING layer.

        Consumed by :class:`~mobius.tasks.CausalLMTask` in static-cache mode to
        allocate the fixed-width KV buffers. Gemma4 needs a per-layer spec (not a
        single uniform head_dim) because:

        * only the first ``num_hidden_layers - num_kv_shared_layers`` layers own a
          cache — the trailing KV-shared layers borrow K,V from a source layer;
        * sliding (local) layers use ``config.head_dim`` while full (global)
          layers use ``config.global_head_dim`` (E2B: 256 vs 512).

        The order matches the iterator consumption in :meth:`forward`
        (``next(kv_iter)`` for each non-shared layer), so the i-th spec here maps
        to the i-th cache-owning layer.
        """
        config = self.config
        first_kv_shared = config.num_hidden_layers - config.num_kv_shared_layers
        global_head_dim = config.global_head_dim or config.head_dim
        specs: list[tuple[int, int]] = []
        for idx, layer_type in enumerate(self.layer_types):
            if first_kv_shared > 0 and idx >= first_kv_shared:
                continue  # KV-shared layer: borrows K,V, owns no cache
            is_full = layer_type == "full_attention"
            head_dim = global_head_dim if is_full else config.head_dim
            if is_full and config.num_global_key_value_heads is not None:
                num_kv_heads = config.num_global_key_value_heads
            else:
                num_kv_heads = config.num_key_value_heads
            specs.append((num_kv_heads, head_dim))
        return specs

    def _compute_per_layer_inputs(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        inputs_embeds: ir.Value,
    ) -> list[ir.Value]:
        """Compute per-layer input embeddings for single-model (text-only) mode."""
        proj = self.per_layer_model_projection(op, inputs_embeds)
        proj = op.Mul(proj, float(self._hidden_size**-0.5))
        proj = op.Reshape(
            proj, op.Constant(value_ints=[0, 0, self._num_layers, self._per_layer_dim])
        )
        proj = self.per_layer_projection_norm(op, proj)

        pad = op.Constant(value_int=0)
        masked_ids = input_ids
        if self._image_token_id:
            masked_ids = op.Where(
                op.Equal(masked_ids, op.Constant(value_int=self._image_token_id)),
                pad,
                masked_ids,
            )
        if self._audio_token_id is not None:
            masked_ids = op.Where(
                op.Equal(masked_ids, op.Constant(value_int=self._audio_token_id)),
                pad,
                masked_ids,
            )

        if self.config.split_per_layer_embedding:
            # L separate Gathers on [V, D] tables — each fits within the EP's
            # max_buffer_size (e.g. WebGPU's 256 MiB limit).
            per_layer_embs = [
                op.Unsqueeze(self.embed_tokens_per_layer_split[i](op, masked_ids), [2])
                for i in range(self._num_layers)
            ]
            fused_emb = op.Concat(*per_layer_embs, axis=2)  # [B, S, L, D]
        else:
            fused_emb = self.embed_tokens_per_layer(op, masked_ids)
            fused_emb = op.Reshape(
                fused_emb,
                op.Constant(value_ints=[0, 0, self._num_layers, self._per_layer_dim]),
            )

        combined = op.Add(proj, fused_emb)
        combined = op.Mul(combined, float(0.5**0.5))

        return [
            op.Gather(combined, op.Constant(value_int=i), axis=2)
            for i in range(self._num_layers)
        ]

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value | None,
        attention_mask: ir.Value | None,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        inputs_embeds: ir.Value | None = None,
        per_layer_inputs: ir.Value | None = None,
        block_sequence_ids: ir.Value | None = None,
    ) -> tuple[ir.Value, list]:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(op, input_ids)

        # Unpack precomputed per_layer_inputs [B, S, L*D] (VLM split),
        # or compute from input_ids (text-only single-model).
        per_layer_list: list[ir.Value] | None = None
        if self._per_layer_dim and per_layer_inputs is not None:
            # VLM split: unpack precomputed per-layer inputs
            num_layers = len(self.layers)
            per_layer_4d = op.Reshape(
                per_layer_inputs,
                op.Constant(value_ints=[0, 0, num_layers, self._per_layer_dim]),
            )
            per_layer_list = [
                op.Squeeze(op.Slice(per_layer_4d, starts=[i], ends=[i + 1], axes=[2]), [2])
                for i in range(num_layers)
            ]
        elif self._per_layer_dim and input_ids is not None:
            # Text-only: compute per-layer inputs from input_ids
            per_layer_list = self._compute_per_layer_inputs(op, input_ids, hidden_states)

        # Determine whether to emit GroupQueryAttention directly.
        # GQA fuses RoPE + attention + KV cache into a single op, and
        # supports local_window_size for sliding-window layers.
        # KV-shared layers fall back to standard Attention because they
        # borrow K,V from another layer (no own KV cache).
        from mobius._build_context import get_build_dtype
        from mobius.components._attention import GQAContext, StaticCacheState

        caps = ep_capabilities()
        dtype = get_build_dtype()
        # Bidirectional vision-block overlay (Gemma4 larger models). When
        # active, contiguous vision-token blocks attend bidirectionally on
        # BOTH full and sliding layers. This cannot be expressed by the
        # GroupQueryAttention op (causal / local-window only), so we force
        # the float-bias Attention path with ``is_causal=0`` and bake the
        # full mask (causal + sliding + padding + blockwise OR) into the bias.
        #
        # ``block_sequence_ids`` [B, S] identifies contiguous image token
        # blocks. It is derived from ``input_ids`` (image token spans only;
        # audio keeps causal attention, matching HF). In the multimodal
        # 3/4-model split the decoder receives ``input_ids`` alongside
        # ``inputs_embeds`` and computes the overlay here, so it does not need a
        # separate cross-model tensor (onnxruntime-genai forwards ``input_ids``
        # to the decoder but cannot forward an arbitrary int tensor). When a
        # caller supplies ``block_sequence_ids`` directly it is used as-is.
        #
        # Trade-off: the overlay is a *static* graph choice — once a model is
        # built with ``use_bidirectional_attention == "vision"`` the decoder
        # always takes the float-bias (``is_causal=0``) path and forgoes GQA,
        # for text-only prompts and every decode step too. This is unavoidable
        # in a single static graph: image-token presence is data-dependent at
        # runtime, and the same graph serves both image prefill and text decode.
        # It stays numerically correct everywhere — when there are no image
        # tokens the block ids are all ``-1`` and the ``q_group >= 0`` guard in
        # ``create_attention_bias`` makes the overlay a pure no-op (plain causal,
        # matching HuggingFace). Only the fused/GQA fast path is given up.
        bidirectional = self._use_bidirectional_attention == "vision" and self._has_image_token
        if bidirectional and block_sequence_ids is None and input_ids is not None:
            block_sequence_ids = _compute_block_sequence_ids(
                op,
                input_ids,
                image_token_id=self._image_token_id,
            )
        use_block_overlay = bidirectional and block_sequence_ids is not None

        # GQA is available when attention_mask exists and the EP supports it.
        # In hybrid mode, sliding layers use GQA while full-attention layers
        # use the static Attention path with TensorScatter.
        use_gqa = (
            attention_mask is not None
            and dtype in caps.gqa_dtypes
            and caps.supports_fused_rope
            and not use_block_overlay
        )
        # When the blockwise overlay is active the Attention op must NOT
        # re-apply its built-in causal mask (it would cancel the
        # future-position unmasking baked into the float bias).
        attn_is_causal = 0 if use_block_overlay else 1

        if use_gqa:
            # Calling forward() on the RoPE modules materializes their
            # cos_cache / sin_cache nn.Parameters as ONNX graph initializers.
            # GQA references these caches directly; without the call the
            # parameters are never emitted into the graph.
            _ = self.rotary_emb_local(op, position_ids)
            global_pos_emb = self.rotary_emb_global(op, position_ids)

            # seqlens_k[b] = sum(attention_mask[b]) - 1  (last valid KV idx)
            # total_seq_len = attention_mask.shape[1]     (past + current)
            one_i32 = op.Constant(value_int=1)
            reduce_sum = op.ReduceSum(attention_mask, [1], keepdims=0)
            seqlens_k = op.Cast(
                op.Sub(reduce_sum, one_i32),
                to=ir.DataType.INT32,
            )
            if caps.requires_graph_capture_rewrite:
                # Support graph capture for shared-KV layer models on WebGPU EP.
                # Derive total_seq_len from reduce_sum (already computed) as a
                # scalar INT32 via Gather index 0 (valid because graph capture
                # requires batch=1).
                total_seq_len = op.Gather(
                    op.Cast(reduce_sum, to=ir.DataType.INT32),
                    op.Constant(value_int=0),
                )
            else:
                total_seq_len = op.Cast(
                    op.Gather(op.Shape(attention_mask), 1),
                    to=ir.DataType.INT32,
                )

            # Per-layer-type GQA contexts with appropriate cos/sin caches
            # and local_window_size for sliding layers.
            gqa_ctx_dict: dict[str, GQAContext] = {
                "sliding_attention": GQAContext(
                    seqlens_k=seqlens_k,
                    total_seq_len=total_seq_len,
                    cos_cache=self.rotary_emb_local.cos_cache,
                    sin_cache=self.rotary_emb_local.sin_cache,
                    local_window_size=self.sliding_window or -1,
                ),
                "full_attention": GQAContext(
                    seqlens_k=seqlens_k,
                    total_seq_len=total_seq_len,
                    cos_cache=self.rotary_emb_global.cos_cache,
                    sin_cache=self.rotary_emb_global.sin_cache,
                ),
            }
            position_embeddings_dict: dict = {
                "sliding_attention": None,
                "full_attention": None,
            }
            # In hybrid mode, static-cache full-attention layers need RoPE
            # embeddings for the standard Attention path (not GQA).
            if past_key_values is not None and any(
                isinstance(kv, StaticCacheState) for kv in past_key_values if kv is not None
            ):
                position_embeddings_dict["full_attention"] = global_pos_emb
        else:
            position_embeddings_dict = {
                "sliding_attention": self.rotary_emb_local(op, position_ids),
                "full_attention": self.rotary_emb_global(op, position_ids),
            }

        # Fallback attention bias for non-GQA layers (used when use_gqa is False).
        # Two flavours:
        #   * static-cache mode (attention_mask is None, StaticCacheState past):
        #     a fixed-width [B,1,S_q,max_seq] bias keyed on absolute positions;
        #   * dynamic mode: the growing [B,1,S_q,total] bias from attention_mask.
        from mobius.components._attention import StaticCacheState

        query_input = input_ids if input_ids is not None else hidden_states
        fallback_bias_dict: dict[str, ir.Value | None] = {}
        need_fallback = not use_gqa
        is_static = (
            past_key_values is not None
            and len(past_key_values) > 0
            and isinstance(past_key_values[0], StaticCacheState)
        )
        static_kv_seqlen: ir.Value | None = None
        if need_fallback and is_static:
            from mobius.components import create_static_cache_attention_bias

            first_cache = past_key_values[0]
            static_kv_seqlen = first_cache.nonpad_kv_seqlen
            max_seq_len = first_cache.key_cache.shape[1]
            if not isinstance(max_seq_len, int):
                raise TypeError(
                    "Gemma4 static cache requires a concrete key_cache KV "
                    f"dimension (axis 1), got symbolic {max_seq_len!r}."
                )
            # S_q at dim 1 of hidden_states ([B, S_q, hidden]); works for both
            # input_ids and inputs_embeds entry points.
            seq_len_t = op.Shape(hidden_states, start=1, end=2)
            fallback_bias_dict = {
                "sliding_attention": create_static_cache_attention_bias(
                    op,
                    write_indices=first_cache.write_indices,
                    seq_len=seq_len_t,
                    nonpad_kv_seqlen=first_cache.nonpad_kv_seqlen,
                    max_seq_len=max_seq_len,
                    sliding_window=self.sliding_window,
                    dtype=self._dtype,
                ),
                "full_attention": create_static_cache_attention_bias(
                    op,
                    write_indices=first_cache.write_indices,
                    seq_len=seq_len_t,
                    nonpad_kv_seqlen=first_cache.nonpad_kv_seqlen,
                    max_seq_len=max_seq_len,
                    sliding_window=None,
                    dtype=self._dtype,
                ),
            }
            fallback_pos_dict = position_embeddings_dict
        elif need_fallback:
            # All fallback layers use float additive bias masks encoding
            # causal + sliding window + padding constraints. Float bias
            # works with both unfused and MEA kernel paths on CUDA EP.
            fallback_bias_dict = {
                "sliding_attention": create_attention_bias(
                    op,
                    input_ids=query_input,
                    attention_mask=attention_mask,
                    sliding_window=self.sliding_window,
                    dtype=self._dtype,
                    block_sequence_ids=block_sequence_ids if use_block_overlay else None,
                ),
                "full_attention": create_attention_bias(
                    op,
                    input_ids=query_input,
                    attention_mask=attention_mask,
                    dtype=self._dtype,
                    block_sequence_ids=block_sequence_ids if use_block_overlay else None,
                ),
            }
            fallback_pos_dict = position_embeddings_dict
        else:
            fallback_pos_dict = {}

        # shared_kv_states: source layers populate it, shared layers consume it
        shared_kv_states: dict = {}
        present_key_values = []

        # Build per-layer past_kv list for all num_hidden_layers layers.
        # past_key_values has only num_kv_layers entries (no entry for KV-shared
        # layers). Expand it to a full per-layer list so we can zip over all
        # layers without truncation.
        if past_key_values is not None:
            kv_iter = iter(past_key_values)
            past_kvs: list = [
                None if layer.self_attn.is_kv_shared_layer else next(kv_iter)
                for layer in self.layers
            ]
        else:
            past_kvs = [None] * len(self.layers)

        for i, (layer, layer_type, past_kv) in enumerate(
            zip(self.layers, self.layer_types, past_kvs)
        ):
            per_layer_input = per_layer_list[i] if per_layer_list is not None else None

            # Per-layer cache/attention dispatch:
            # - StaticCacheState → static path (TensorScatter + Attention)
            # - Dynamic tuple → GQA path (with local_window_size)
            # - None (no cache) → fallback Attention path
            is_layer_static = isinstance(past_kv, StaticCacheState)

            if is_layer_static:
                attn_bias = None
                pos_emb = position_embeddings_dict[layer_type]
            elif use_gqa:
                attn_bias = gqa_ctx_dict[layer_type]
                pos_emb = None
            elif fallback_bias_dict:
                attn_bias = fallback_bias_dict[layer_type]
                pos_emb = fallback_pos_dict[layer_type]
            else:
                attn_bias = None
                pos_emb = position_embeddings_dict.get(layer_type)

            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attn_bias,
                position_embeddings=pos_emb,
                shared_kv_states=shared_kv_states,
                per_layer_input=per_layer_input,
                past_key_value=past_kv,
                is_causal=attn_is_causal,
                static_kv_seqlen=static_kv_seqlen,
            )
            # KV-shared layers borrow K,V from source layers — exclude from
            # present_key_values so the output has exactly num_kv_layers entries.
            if not layer.self_attn.is_kv_shared_layer:
                present_key_values.append(present_kv)

        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values


# ---------------------------------------------------------------------------
# Gemma4CausalLMModel (text-only)
# ---------------------------------------------------------------------------


class Gemma4CausalLMModel(CausalLMModel):
    """Gemma 4 text-only causal language model.

    Wraps :class:`Gemma4TextModel` (transformer backbone) with an ``lm_head``
    linear projection to produce next-token logits.  Corresponds to HF
    ``Gemma4ForCausalLM``, which similarly wraps ``Gemma4Model`` (our
    :class:`Gemma4TextModel`) and adds the language-model head on top.

    Registered as ``gemma4_text`` in the model registry.  Uses hybrid
    local/global attention, standard ``RMSNorm``, and optional per-layer
    input embeddings.
    """

    config_class: type = Gemma4Config
    category: str = "Text"
    default_task: str = "gemma4-text-generation"

    def __init__(self, config: Gemma4Config):
        nn.Module.__init__(self)
        self.config = config
        self.model = Gemma4TextModel(config)
        self.lm_head = _make_lm_head(config)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        inputs_embeds: ir.Value | None = None,
    ) -> tuple[ir.Value, list]:
        hidden_states, present_key_values = self.model(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )
        logits = self.lm_head(op, hidden_states)
        # Optional final logit soft-capping (tanh scaled): logit_cap * tanh(x / logit_cap)
        if self.config.final_logit_softcapping:
            cap = op.CastLike(
                self.config.final_logit_softcapping,
                logits,
            )
            logits = op.Mul(op.Tanh(op.Div(logits, cap)), cap)
        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # Strip optional 'language_model.' prefix from multimodal checkpoints
        for key in list(state_dict.keys()):
            if "language_model." in key:
                new_key = key.replace("language_model.", "")
                state_dict[new_key] = state_dict.pop(key)
            elif "vision_tower" in key or "embed_vision" in key:
                state_dict.pop(key, None)
        # HF's model.embed_tokens_per_layer.weight [V, L*D] maps directly
        # to our fused embedding table — no splitting needed.
        # (For WebGPU, splitting is handled by _Gemma4DecoderModel.preprocess_weights.)
        # Map HF expert weight names and fold router scale
        _remap_moe_expert_weights(state_dict, self.config)
        return super().preprocess_weights(state_dict)

    def static_kv_cache_specs(self) -> list[tuple[int, int]]:
        """Per-cache-layer ``(num_kv_heads, head_dim)`` for static-cache mode."""
        assert isinstance(self.model, Gemma4TextModel)
        return self.model.static_kv_cache_specs()


# ---------------------------------------------------------------------------
# Gemma4 multimodal sub-models
# ---------------------------------------------------------------------------


class _Gemma4DecoderModel(nn.Module):
    """Gemma4 text decoder sub-model accepting ``inputs_embeds``.

    When ``hidden_size_per_layer_input > 0`` (e.g. Gemma4 E2B), per-layer input
    embeddings are precomputed by the embedding sub-model and passed as
    ``per_layer_inputs`` (shape ``[B, S, L*D]``).  The decoder unpacks them
    and feeds one ``[B, S, D]`` slice to each decoder layer's gating mechanism.
    """

    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.config = config
        self.model = Gemma4TextModel(config)
        self.lm_head = _make_lm_head(config)

    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        per_layer_inputs: ir.Value | None = None,
        past_key_values: list | None = None,
        block_sequence_ids: ir.Value | None = None,
        input_ids: ir.Value | None = None,
    ) -> tuple[ir.Value, list]:
        # ``input_ids`` is forwarded alongside ``inputs_embeds`` so the text
        # model can derive the bidirectional vision-block overlay internally
        # (see Gemma4TextModel.forward). ``inputs_embeds`` still takes
        # precedence for the actual token embeddings.
        hidden_states, present_key_values = self.model(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            block_sequence_ids=block_sequence_ids,
        )
        logits = self.lm_head(op, hidden_states)
        # Gemma4 applies final logit soft-capping: logit_cap * tanh(x / logit_cap)
        if self.config.final_logit_softcapping:
            cap = op.CastLike(
                self.config.final_logit_softcapping,
                logits,
            )
            logits = op.Mul(op.Tanh(op.Div(logits, cap)), cap)
        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        state_dict = vlm_decoder_weights(state_dict, tie=self.config.tie_word_embeddings)
        # For WebGPU: split the fused [V, L*D] per-layer embedding into L separate [V, D] tables.
        per_layer_dim = self.config.hidden_size_per_layer_input
        if per_layer_dim and self.config.split_per_layer_embedding:
            fused_key = "model.embed_tokens_per_layer.weight"
            if fused_key in state_dict:
                num_layers = self.config.num_hidden_layers
                fused = state_dict.pop(fused_key)
                assert fused.shape[1] == num_layers * per_layer_dim, (
                    f"{fused_key} dim 1 expected {num_layers * per_layer_dim} "
                    f"({num_layers} layers x {per_layer_dim} per_layer_dim), "
                    f"got {fused.shape[1]}"
                )
                chunks = fused.chunk(num_layers, dim=1)
                for i, chunk in enumerate(chunks):
                    state_dict[f"model.embed_tokens_per_layer_split.{i}.weight"] = chunk
        return state_dict


class _Gemma4VisionEncoderModel(nn.Module):
    """Gemma4 vision encoder sub-model: pre-patchified input -> projected features.

    Pipeline:
    1. ``encoder``: patch embedding + N transformer blocks + final norm
    2. Scale by ``sqrt(vision_hidden)`` (HF ``VisionPooler`` scaling step)
    3. ``projector_norm``: scale-free RMSNorm (HF ``embedding_pre_projection_norm``)
    4. ``projector``: Linear to text hidden size (HF ``embedding_projection``)

    Weight name mapping strips:
    - ``vision_tower.`` prefix -> ``encoder.``
    - ``.linear.`` infix from ``Gemma4ClippableLinear`` wrapper
    - ``embed_vision.embedding_projection.*`` -> ``projector.*``
    - ``embed_vision.embedding_pre_projection_norm.*`` -> skip (scale-free, no weight)
    """

    def __init__(self, config: Gemma4Config):
        super().__init__()
        vc = config.vision  # VisionConfig for the SigLIP encoder
        self._text_hidden_size = config.hidden_size
        self._vision_hidden_size = vc.hidden_size
        self.encoder = _Gemma4VisionEncoderCore(config)
        # Gemma4VisionPooler: 3x3 spatial average pooling + sqrt(hidden) scaling.
        # Reduces N patches to N/9 before projection.
        self.pooler = Gemma4VisionPooler(vc.hidden_size, vc.pooling_kernel_size or 3)
        self.projector_norm = _Gemma4ScaleFreeRMSNorm(vc.hidden_size, eps=vc.norm_eps)
        linear_class, _ = _component_linear_classes(
            config,
            ("model.vision_tower", "model.embed_vision"),
        )
        self.projector = linear_class(vc.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        pixel_position_ids: ir.Value,
    ) -> ir.Value:
        # [B, N, 3*P^2] -> [B, N, vision_hidden]
        vision_features = self.encoder(op, pixel_values, pixel_position_ids)

        # Position-based pooling to a FIXED [B, length, D] grid (length = N // k²)
        # plus a [B, length] validity mask marking the occupied soft-token bins
        # of each image.  Pooling at a fixed length keeps the op batch-independent
        # so a batch of images with different aspect ratios pools in one pass.
        pooled, valid_mask = self.pooler(op, vision_features, pixel_position_ids)

        # Flatten batch+token dims and drop the empty trailing bins so the number
        # of soft tokens equals the per-image placeholder count (sum of
        # w_out*h_out over the batch).  [B, length, D] -> [B*length, D]
        # --Compress--> [num_valid, D].  For a single image this yields exactly
        # the same valid tokens as before; for a multi-image batch it produces
        # the concatenation of each image's (variable-length) pooled tokens.
        pooled = op.Reshape(
            pooled, op.Constant(value_ints=[-1, self._vision_hidden_size])
        )  # [B*length, vision_hidden]
        valid_mask = op.Reshape(valid_mask, op.Constant(value_ints=[-1]))  # [B*length]
        vision_features = _dtype_safe_compress(
            op, pooled, valid_mask, axis=0
        )  # [num_valid, vision_hidden]

        # Scale-free norm + linear projection -> [num_valid, text_hidden]
        vision_features = self.projector_norm(op, vision_features)
        vision_features = self.projector(op, vision_features)

        return vision_features  # [num_valid, text_hidden]

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("vision_tower."):
                new_key = "encoder." + key[len("vision_tower.") :]
                # HF Gemma4VisionModel wraps its layer list in a Gemma4VisionEncoder
                # submodule, adding an extra "encoder." level. Our _Gemma4VisionEncoderCore
                # exposes its layers as "layers" directly, so strip the extra prefix.
                new_key = new_key.replace("encoder.encoder.", "encoder.", 1)
                # Flatten Gemma4ClippableLinear's .linear. wrapper
                new_key = new_key.replace(".linear.weight", ".weight")
                new_key = new_key.replace(".linear.bias", ".bias")
                renamed[new_key] = value
            elif key.startswith("embed_vision.embedding_projection."):
                suffix = key[len("embed_vision.embedding_projection.") :]
                renamed["projector." + suffix] = value
            elif key.startswith("embed_vision.embedding_pre_projection_norm."):
                pass  # Scale-free RMSNorm: normalizes without a learnable scale, no parameter
        return renamed


class Gemma4EmbeddingModel(nn.Module):
    """Gemma4 embedding sub-model: scaled token lookup + multimodal feature fusion.

    Always scatters vision features at image-token positions.  When the model
    has audio support (``config.audio is not None``), ``forward()`` also accepts
    ``audio_features`` and scatters them at audio-token positions.

    When ``hidden_size_per_layer_input > 0``, also computes per-layer input
    embeddings that condition each decoder layer.  This moves the per-layer
    computation out of the decoder so the decoder no longer needs ``input_ids``.

    Inputs (image-only variant):
    - ``input_ids [B, S]`` INT64
    - ``image_features [num_img_tokens, hidden_size]``

    Inputs (image + audio variant):
    - ``input_ids [B, S]`` INT64
    - ``image_features [num_img_tokens, hidden_size]``
    - ``audio_features [num_aud_tokens, hidden_size]``

    Outputs:
    - ``inputs_embeds [B, S, hidden_size]``
    - ``per_layer_inputs [B, S, L*D]`` (only when ``hidden_size_per_layer_input > 0``)
    """

    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.config = config
        embed_scale = math.sqrt(config.hidden_size)
        self.embed_tokens = _make_scaled_word_embedding(
            config,
            config.vocab_size,
            config.hidden_size,
            embed_scale,
        )
        self.image_token_id = config.image_token_id or 0
        # Audio token ID is only set when the model has an audio encoder.
        self.audio_token_id: int | None = config.audio.audio_token_id if config.audio else None

        # Per-layer input embedding components (moved from the decoder).
        self._per_layer_dim = getattr(config, "hidden_size_per_layer_input", 0)
        self._hidden_size = config.hidden_size
        if self._per_layer_dim:
            self._num_layers = config.num_hidden_layers
            vocab_per_layer = getattr(config, "vocab_size_per_layer_input", 0)
            # Single fused [V, L*D] embedding table matching HuggingFace's
            # ``embed_tokens_per_layer.weight`` shape.  The ORT CUDA Gather
            # int32 overflow (onnxruntime#28107) is now fixed.
            self.embed_tokens_per_layer = _make_scaled_word_embedding(
                config,
                vocab_per_layer,
                self._num_layers * self._per_layer_dim,
                float(self._per_layer_dim**0.5),
            )
            self.per_layer_model_projection = _per_layer_model_projection_class(config)(
                config.hidden_size,
                config.num_hidden_layers * self._per_layer_dim,
                bias=False,
            )
            self.per_layer_projection_norm = RMSNorm(
                self._per_layer_dim, eps=config.rms_norm_eps
            )

    def _scatter_features(
        self,
        op: OpBuilder,
        hidden: ir.Value,
        input_ids: ir.Value,
        token_id: int,
        features: ir.Value,
    ) -> ir.Value:
        """Scatter ``features`` into ``hidden`` at positions matching ``token_id``.

        Appends a dummy zero row to ``features`` before Gather so that ORT's
        eager evaluation of the Where branches never faults on an empty tensor
        during text-only / decode steps.
        """
        mask = op.Equal(input_ids, op.Constant(value_int=token_id))
        mask_3d = op.Unsqueeze(mask, [-1])

        # CumSum → sub-1 → clip gives 0-based index into features for each token
        mask_int = op.Cast(mask, to=7)  # INT64
        cumsum = op.CumSum(mask_int, op.Constant(value_int=1))
        indices = op.Clip(op.Sub(cumsum, op.Constant(value_int=1)), op.Constant(value_int=0))

        # One-row dummy prevents empty-tensor Gather faults during decode steps.
        # Use Constant (static) + Unsqueeze to avoid ConstantOfShape, whose
        # dynamic-shape input blocks ONNX shape inference.
        dummy_row = op.Unsqueeze(
            op.CastLike(
                op.Constant(value_floats=[0.0] * self.config.hidden_size),
                features,
            ),
            [0],
        )  # [1, hidden_size]
        features_safe = op.Concat(features, dummy_row, axis=0)
        gathered = op.Gather(features_safe, indices, axis=0)
        return op.Where(mask_3d, gathered, hidden)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        image_features: ir.Value,
        audio_features: ir.Value | None = None,
    ) -> dict[str, ir.Value]:
        """Return a dict of named embedding outputs.

        Always contains ``inputs_embeds``. Contains ``per_layer_inputs`` when
        ``hidden_size_per_layer_input > 0``.

        The vision-block bidirectional attention overlay is NOT emitted here:
        the decoder derives it from ``input_ids`` directly (see
        ``Gemma4TextModel.forward``), which avoids a cross-model tensor that
        onnxruntime-genai cannot forward between sub-models.
        """
        # [B, S] → [B, S, hidden]
        hidden = self.embed_tokens(op, input_ids)

        # Scatter image features at image-token positions
        hidden = self._scatter_features(
            op, hidden, input_ids, self.image_token_id, image_features
        )

        # Scatter audio features at audio-token positions (only for AnyToAny models)
        if audio_features is not None:
            assert self.audio_token_id is not None, (
                "Gemma4EmbeddingModel received audio_features but audio_token_id is not set. "
                "Ensure config.audio is provided."
            )
            hidden = self._scatter_features(
                op, hidden, input_ids, self.audio_token_id, audio_features
            )

        outputs: dict[str, ir.Value] = {"inputs_embeds": hidden}

        if not self._per_layer_dim:
            return outputs

        # When split_per_layer_embedding is set, the per-layer computation runs
        # inside the decoder using split [V, D] tables.  The embedding model only
        # emits inputs_embeds in that case.
        if self.config.split_per_layer_embedding:
            return outputs

        # Compute per-layer input embeddings (moved from the decoder).
        # 1. Project hidden states → [B, S, L*D] and scale by hidden_size**-0.5
        proj = self.per_layer_model_projection(op, hidden)
        proj = op.Mul(proj, float(self._hidden_size**-0.5))
        # Reshape to [B, S, L, D] for per-layer RMSNorm
        proj = op.Reshape(
            proj, op.Constant(value_ints=[0, 0, self._num_layers, self._per_layer_dim])
        )
        proj = self.per_layer_projection_norm(op, proj)

        # 2. Mask multimodal token IDs → configured pad_token_id before per-layer lookup
        pad = op.Constant(value_int=self.config.pad_token_id)
        masked_ids = input_ids
        if self.image_token_id:
            masked_ids = op.Where(
                op.Equal(masked_ids, op.Constant(value_int=self.image_token_id)),
                pad,
                masked_ids,
            )
        if self.audio_token_id is not None:
            masked_ids = op.Where(
                op.Equal(masked_ids, op.Constant(value_int=self.audio_token_id)),
                pad,
                masked_ids,
            )

        # 3. Single Gather on fused [V, L*D] table → reshape to [B, S, L, D]
        fused_emb = self.embed_tokens_per_layer(op, masked_ids)
        # fused_emb: [B, S, L*D] → [B, S, L, D]
        fused_emb = op.Reshape(
            fused_emb,
            op.Constant(value_ints=[0, 0, self._num_layers, self._per_layer_dim]),
        )

        # 4. Combine: (proj + emb) * 0.707 per layer, then flatten back
        combined = op.Add(proj, fused_emb)  # [B, S, L, D]
        combined = op.Mul(combined, float(0.5**0.5))
        # Flatten L*D → single per_layer_inputs output: [B, S, L*D]
        per_layer_inputs = op.Reshape(
            combined,
            op.Constant(value_ints=[0, 0, self._num_layers * self._per_layer_dim]),
        )
        outputs["per_layer_inputs"] = per_layer_inputs

        return outputs

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return vlm_embedding_weights(state_dict)


# ---------------------------------------------------------------------------
# _Gemma4AudioEncoderModel — Conformer encoder + projector sub-model
# ---------------------------------------------------------------------------


class _Gemma4AudioEncoderModel(nn.Module):
    """Gemma4 Conformer audio encoder sub-model.

    Wraps :class:`Gemma4AudioEncoder` and applies a learned linear projector
    (``embed_audio.embedding_projection`` in HF) that maps from the encoder
    output dimension to the text model's hidden size.

    Inputs:
    - ``input_features [B, T, input_size]``: mel-spectrogram features

    Output:
    - ``audio_features [B, T//4, text_hidden_size]``: projected audio tokens
    """

    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.config = config
        linear_class, clippable_linear_class = _component_linear_classes(
            config,
            ("model.audio_tower", "model.embed_audio"),
        )
        ac = config.audio  # Gemma4AudioConfig (guaranteed non-None when used)
        input_size = (ac.input_size if ac else None) or 128
        hidden_size = (ac.hidden_size if ac else None) or 1024
        num_layers = (ac.num_layers if ac else None) or 12
        # HF config field is output_proj_dims (not output_dim)
        output_proj_dims = (
            getattr(ac, "output_proj_dims", None) if ac else None
        ) or config.hidden_size
        conv_channels = ac.subsampling_conv_channels if ac else None
        rms_norm_eps = config.rms_norm_eps or 1e-6

        self.encoder = Gemma4AudioEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_heads=8,  # fixed per Gemma4 audio_config
            num_layers=num_layers,
            conv_kernel_size=5,  # fixed per Gemma4 audio_config
            conv_channels=conv_channels,
            attention_context_left=13,  # fixed per Gemma4 audio_config
            output_proj_dims=output_proj_dims,
            rms_norm_eps=rms_norm_eps,
            linear_class=linear_class,
            clippable_linear_class=clippable_linear_class,
        )
        # Scale-free RMSNorm applied before the projection (HF embed_audio.embedding_pre_projection_norm).
        # with_scale=False in HF → no learnable weight → no checkpoint key, no ONNX initializer.
        # NOTE: We inline the RMSNorm in forward() using manual ops to prevent
        # ORT from fusing Add(output_proj.bias) + RMSNormalization into
        # SkipSimplifiedLayerNormalization (CUDA rejects 1D skip).
        self._rms_norm_eps = rms_norm_eps
        # Learned projection from encoder output space → text hidden size.
        # Corresponds to HF's embed_audio.embedding_projection (no bias).
        self.projector = linear_class(output_proj_dims, config.hidden_size, bias=False)

    def forward(
        self,
        op: OpBuilder,
        input_features: ir.Value,
        input_features_mask: ir.Value | None = None,
    ) -> tuple[ir.Value, ir.Value | None]:
        # [B, T, input_size] → encoder → [B, T//4, output_proj_dims]
        audio_features, downsampled_mask = self.encoder(
            op, input_features, input_features_mask=input_features_mask
        )
        # Scale-free RMSNorm before projection (HF embed_audio.embedding_pre_projection_norm).
        # Use manual primitive ops instead of op.RMSNormalization to prevent
        # ORT from fusing Add(output_proj.bias) + RMSNorm into
        # SkipSimplifiedLayerNormalization with a 1D bias as skip input
        # (CUDA kernel rejects 1D skip, CPU kernel accepts it).
        x_f32 = op.Cast(audio_features, to=ir.DataType.FLOAT)
        sq = op.Mul(x_f32, x_f32)
        mean_sq = op.ReduceMean(sq, op.Constant(value_ints=[-1]), keepdims=1)
        eps = op.Constant(value_float=self._rms_norm_eps)
        rms = op.Sqrt(op.Add(mean_sq, eps))
        audio_features = op.CastLike(op.Div(x_f32, rms), audio_features)
        # → projector → [B, T//4, text_hidden_size]
        return self.projector(op, audio_features), downsampled_mask


# ---------------------------------------------------------------------------
# gemma4_unified (gemma-4-12B) encoder-free vision / audio embedders
# ---------------------------------------------------------------------------


class _F32Linear(Linear):
    """Linear that computes its MatMul in float32 regardless of model dtype.

    Used by the gemma4_unified vision embedder's ``patch_dense`` projection
    **only when the model dtype is float16**, whose output magnitude (~77000)
    exceeds the float16 range (65504). The weights are stored in the model dtype;
    activations and weights are upcast to float32 for the MatMul so the result
    does not overflow to +inf. The output stays float32 (the following
    ``_F32LayerNorm`` normalizes it back into a float16-safe range). bfloat16 and
    float32 models have the range natively and use a plain :class:`Linear`.
    """

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        w_t = op.Cast(
            op.Transpose(self.weight, perm=[1, 0]), to=ir.DataType.FLOAT
        )  # [in_features, out_features]
        result = op.MatMul(op.Cast(x, to=ir.DataType.FLOAT), w_t)
        if self.bias is not None:
            result = op.Add(result, op.Cast(self.bias, to=ir.DataType.FLOAT))
        return result  # float32


class _F32LayerNorm(LayerNorm):
    """LayerNorm that computes in float32 and returns a float32 output.

    Pairs with :class:`_F32Linear` in the gemma4_unified vision embedder, **only
    for float16 models**, so the large (out-of-float16-range) ``patch_dense``
    output is normalized in float32 before being cast back to the model dtype.
    bfloat16 and float32 models use a plain :class:`LayerNorm`.
    """

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        return op.LayerNormalization(
            op.Cast(hidden_states, to=ir.DataType.FLOAT),
            op.Cast(self.weight, to=ir.DataType.FLOAT),
            op.Cast(self.bias, to=ir.DataType.FLOAT),
            epsilon=self.eps,
            axis=-1,
        )  # float32


class _Gemma4UnifiedVisionEmbedderModel(nn.Module):
    """Encoder-free vision embedder for ``gemma4_unified`` (gemma-4-12B).

    Unlike gemma4's SigLIP tower, the unified model has **no vision encoder**.
    Raw merged pixel patches are projected directly into language-model space.

    Replicates HF ``Gemma4UnifiedVisionEmbedder``:

        patch_ln1 (LayerNorm, patch_dim)
        → patch_dense (Linear patch_dim → mm_embed_dim)
        → patch_ln2 (LayerNorm, mm_embed_dim)
        → + factorized 2D positional embedding
        → pos_norm (LayerNorm, mm_embed_dim)
        → embedding_pre_projection_norm (scale-free RMSNorm)
        → embedding_projection (Linear mm_embed_dim → text_hidden)

    ``patch_dim = (patch_size * pooling_kernel_size)^2 * 3`` (48*48*3 = 6912).

    Inputs:
    - ``pixel_values [B, N, patch_dim]``: raw merged pixel patches.
    - ``pixel_position_ids [B, N, 2]``: integer (x, y) patch coordinates;
      ``(-1, -1)`` marks padding patch slots.

    Output:
    - ``image_features [num_valid_patches, text_hidden_size]``: padding
      patches (position == -1) are stripped so the output rows align 1:1 with
      image placeholder tokens in the text sequence (matches HF, which selects
      ``vision_outputs[~padding_mask]``).
    """

    def __init__(self, config: Gemma4Config):
        super().__init__()
        vc = config.vision  # VisionConfig populated by the gemma4_unified hook
        patch_size = (vc.patch_size if vc else None) or 16
        pooling = (vc.pooling_kernel_size if vc else None) or 3
        model_patch_size = patch_size * pooling
        patch_dim = 3 * model_patch_size * model_patch_size
        mm_embed_dim = (vc.hidden_size if vc else None) or config.hidden_size
        posemb_size = (vc.position_embedding_size if vc else None) or 1120
        out_proj_dim = (vc.out_hidden_size if vc else None) or mm_embed_dim
        eps = (vc.norm_eps if vc else None) or config.rms_norm_eps or 1e-6
        self._text_hidden_size = config.hidden_size

        self.patch_ln1 = LayerNorm(patch_dim, eps=eps)
        # patch_dense produces activations whose magnitude (~77000, measured) is
        # outside the float16 range (max 65504); HF runs this embedder in bfloat16
        # (max ~3.4e38). Only float16 actually overflows, so we upcast the dense
        # projection + the following LayerNorm to float32 *only* when the model
        # dtype is float16. bfloat16 and float32 have the range natively and keep
        # their dtype (matching HF for bfloat16). See _F32Linear / _F32LayerNorm.
        if config.dtype == ir.DataType.FLOAT16:
            self.patch_dense = _F32Linear(patch_dim, mm_embed_dim, bias=True)
            self.patch_ln2 = _F32LayerNorm(mm_embed_dim, eps=eps)
        else:
            self.patch_dense = Linear(patch_dim, mm_embed_dim, bias=True)
            self.patch_ln2 = LayerNorm(mm_embed_dim, eps=eps)
        # Factorized 2D positional embedding: HF stores a single
        # [posemb_size, 2, mm_embed_dim] table looked up per axis. We split it
        # into two [posemb_size, mm_embed_dim] tables (x and y) in
        # preprocess_weights so each axis is a plain Gather.
        self.pos_emb_x = Embedding(posemb_size, mm_embed_dim)
        self.pos_emb_y = Embedding(posemb_size, mm_embed_dim)
        self.pos_norm = LayerNorm(mm_embed_dim, eps=eps)
        # Scale-free RMSNorm before the projection (HF
        # embed_vision.multimodal_embedder.embedding_pre_projection_norm).
        # The projection consumes the post-position-norm activations, whose
        # last dim is mm_embed_dim (HF embedding_projection: mm_embed_dim →
        # text_hidden). out_proj_dim is retained only as a sanity check.
        assert out_proj_dim == mm_embed_dim, (
            "gemma4_unified vision projector expects output_proj_dims == "
            f"mm_embed_dim, got {out_proj_dim} != {mm_embed_dim}"
        )
        self.projector_norm = _Gemma4ScaleFreeRMSNorm(mm_embed_dim, eps=eps)
        # HF embed_vision.multimodal_embedder.embedding_projection (no bias).
        self.projector = Linear(mm_embed_dim, config.hidden_size, bias=False)

    def forward(
        self,
        op: OpBuilder,
        pixel_values: ir.Value,
        pixel_position_ids: ir.Value,
    ) -> ir.Value:
        # Patch embedding: LN → Dense → LN.  [B, N, patch_dim] → [B, N, mm_embed_dim]
        # For float16 models patch_dense + patch_ln2 run in float32 (see __init__
        # and _F32Linear / _F32LayerNorm): the dense projection produces activations
        # whose magnitude exceeds the float16 range (measured absmax ~77000 > 65504;
        # HF runs this embedder in bfloat16), so an f16 intermediate would overflow
        # to +inf and patch_ln2 would emit NaN. patch_ln2 normalizes the result back
        # into a float16-safe range. For bfloat16 / float32 the projection stays in
        # the model dtype and CastLike below is a no-op.
        h = self.patch_ln1(op, pixel_values)
        h = self.patch_dense(op, h)  # f16 model: f16 in → f32 out; else native dtype
        h = self.patch_ln2(op, h)  # f16 model: f32 in → f32 out (normalized)
        h = op.CastLike(h, pixel_values)  # back to the model dtype (no-op unless upcast)

        # Factorized positional embedding.  Split (x, y) coords on the last axis.
        x_ids = op.Gather(pixel_position_ids, op.Constant(value_int=0), axis=-1)  # [B, N]
        y_ids = op.Gather(pixel_position_ids, op.Constant(value_int=1), axis=-1)  # [B, N]
        # Padding patches carry -1: clamp to a valid index for the Gather, then
        # zero their contribution via the validity mask.
        zero = op.Constant(value_int=0)
        clamped_x = op.Max(x_ids, zero)
        clamped_y = op.Max(y_ids, zero)
        neg_one = op.Constant(value_int=-1)
        valid_x = op.Unsqueeze(
            op.CastLike(op.Not(op.Equal(x_ids, neg_one)), h), [-1]
        )  # [B, N, 1]
        valid_y = op.Unsqueeze(op.CastLike(op.Not(op.Equal(y_ids, neg_one)), h), [-1])
        pos_x = op.Mul(self.pos_emb_x(op, clamped_x), valid_x)  # [B, N, mm_embed_dim]
        pos_y = op.Mul(self.pos_emb_y(op, clamped_y), valid_y)
        h = op.Add(h, op.Add(pos_x, pos_y))
        h = self.pos_norm(op, h)

        # Scale-free RMSNorm → projection to text hidden size.
        h = self.projector_norm(op, h)
        h = self.projector(op, h)  # [B, N, text_hidden]

        # Strip padding patches so output rows align 1:1 with placeholder tokens.
        h_flat = op.Reshape(h, op.Constant(value_ints=[-1, self._text_hidden_size]))
        keep = op.Reshape(
            op.Not(op.Equal(x_ids, neg_one)), op.Constant(value_ints=[-1])
        )  # [B*N] BOOL
        return _dtype_safe_compress(op, h_flat, keep, axis=0)  # [num_valid, text_hidden]

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("vision_embedder.pos_embedding"):
                # HF [posemb_size, 2, mm_embed_dim] → two [posemb_size, mm_embed_dim]
                renamed["pos_emb_x.weight"] = value[:, 0, :].contiguous()
                renamed["pos_emb_y.weight"] = value[:, 1, :].contiguous()
            elif key.startswith("vision_embedder."):
                renamed[key[len("vision_embedder.") :]] = value
            elif key.startswith("embed_vision.embedding_projection."):
                renamed["projector." + key[len("embed_vision.embedding_projection.") :]] = (
                    value
                )
            # embed_vision.*.embedding_pre_projection_norm: scale-free, no weight.
        return renamed


class _Gemma4UnifiedAudioEmbedderModel(nn.Module):
    """Encoder-free audio embedder for ``gemma4_unified`` (gemma-4-12B).

    The unified model has **no Conformer audio tower**.  Raw waveform-frame
    features are projected directly into language-model space.

    Replicates HF ``Gemma4UnifiedMultimodalEmbedder`` (the ``embed_audio``
    branch):

        embedding_pre_projection_norm (scale-free RMSNorm, audio_embed_dim)
        → embedding_projection (Linear audio_embed_dim → text_hidden)

    Inputs:
    - ``input_features [B, T, audio_embed_dim]``: raw waveform-frame features.
    - ``input_features_mask [B, T]``: BOOL mask, ``True`` for valid frames.

    Output:
    - ``audio_features [num_valid_frames, text_hidden_size]``: padding frames
      are stripped so output rows align 1:1 with audio placeholder tokens
      (matches HF, which selects ``audio_features[audio_mask]``).
    """

    def __init__(self, config: Gemma4Config):
        super().__init__()
        ac = config.audio  # Gemma4AudioConfig populated by the gemma4_unified hook
        audio_embed_dim = (ac.hidden_size if ac else None) or 640
        # Prefer the audio config's own eps; fall back to the text decoder's.
        eps = (ac.rms_norm_eps if ac else None) or config.rms_norm_eps or 1e-6
        self._text_hidden_size = config.hidden_size
        # HF embed_audio.embedding_pre_projection_norm (scale-free RMSNorm).
        self.projector_norm = _Gemma4ScaleFreeRMSNorm(audio_embed_dim, eps=eps)
        # HF embed_audio.embedding_projection (no bias).
        self.projector = Linear(audio_embed_dim, config.hidden_size, bias=False)

    def forward(
        self,
        op: OpBuilder,
        input_features: ir.Value,
        input_features_mask: ir.Value | None = None,
    ) -> tuple[ir.Value, ir.Value | None]:
        # [B, T, audio_embed_dim] → scale-free RMSNorm → [B, T, text_hidden]
        h = self.projector_norm(op, input_features)
        h = self.projector(op, h)
        if input_features_mask is None:
            return h, None
        # Strip padding frames so output rows align 1:1 with placeholder tokens.
        h_flat = op.Reshape(h, op.Constant(value_ints=[-1, self._text_hidden_size]))
        keep = op.Reshape(input_features_mask, op.Constant(value_ints=[-1]))  # [B*T]
        return _dtype_safe_compress(op, h_flat, keep, axis=0), None  # [num_valid, text_hidden]

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("embed_audio.embedding_projection."):
                renamed["projector." + key[len("embed_audio.embedding_projection.") :]] = value
            # embed_audio.embedding_pre_projection_norm: scale-free, no weight.
        return renamed


# ---------------------------------------------------------------------------
# Gemma4Model — unified vision-language (+ optional audio) model
# ---------------------------------------------------------------------------


class Gemma4Model(nn.Module):
    """Unified Gemma4 multimodal model (3- or 4-model split).

    Builds three or four separate ONNX models depending on whether the config
    includes an audio sub-config:

    Always produced:
    - ``decoder``: Gemma4 text decoder taking ``inputs_embeds``
    - ``vision``: SigLIP-style encoder + projector
    - ``embedding``: scaled word embedding + multimodal feature fusion

    Added when ``config.audio is not None``:
    - ``speech``: Conformer audio encoder + projection to text hidden size

    Covers all Gemma4 variants:
    - Vision-language (26B-A4B, 31B): ``audio=None``
    - Any-to-Any (E2B-it, E4B-it): ``audio=Gemma4AudioConfig(...)``

    Registered as ``gemma4`` (and ``gemma4_any_to_any`` for back-compat).
    """

    default_task: str = "gemma4"
    category: str = "Multimodal"

    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.config = config
        self.decoder = _Gemma4DecoderModel(config)
        self.vision_encoder = _Gemma4VisionEncoderModel(config)
        self.embedding = Gemma4EmbeddingModel(config)
        self.audio_encoder: _Gemma4AudioEncoderModel | None = (
            _Gemma4AudioEncoderModel(config) if config.audio is not None else None
        )

    def forward(self, op: OpBuilder, **kwargs):
        raise NotImplementedError(
            "Gemma4Model is a multi-model split; Gemma4Task builds each sub-module "
            "(decoder, vision_encoder, embedding, and optionally audio_encoder) separately."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Rename HuggingFace weight keys to ONNX initializer names.

        HF multimodal checkpoints prefix every key with ``model.``
        (e.g. ``model.language_model.*``, ``model.vision_tower.*``).

        Mapping (after stripping the leading ``model.``):

        - ``language_model.lm_head.*`` → ``decoder.lm_head.*``
        - ``language_model.*`` → ``decoder.model.*``
        - ``language_model.embed_tokens.weight`` also → ``embedding.embed_tokens.weight``
        - ``vision_tower.*`` → ``vision_encoder.encoder.*``
          (strips HF's extra ``encoder.`` level and ``.linear.`` infix from
          ``Gemma4ClippableLinear``)
        - ``embed_vision.embedding_projection.*`` → ``vision_encoder.projector.*``
        - ``embed_vision.embedding_pre_projection_norm.*`` → skip (scale-free, no weight)
        - ``audio_tower.*`` → ``audio_encoder.encoder.*``
          (strips ``.linear.`` infix; renames ``subsample_conv_projection.layerN.{conv,norm}``
          to ``{conv,norm}N``)
        - ``embed_audio.embedding_projection.*`` → ``audio_encoder.projector.*``
        - ``embed_audio.embedding_pre_projection_norm.*`` → skip (``Gemma4RMSNorm`` with
          ``with_scale=False``; normalises activations but has no learnable ``weight``
          parameter and produces no checkpoint key — verified against google/gemma-4-E2B-it)

        Note: the decoder sub-model takes ``inputs_embeds`` rather than
        ``input_ids``, so ``embed_tokens`` is not a decoder initializer — the
        token embedding lives only in the ``embedding`` sub-model.
        """
        # Strip top-level "model." prefix used by HF multimodal checkpoints.
        state_dict = {
            (key[len("model.") :] if key.startswith("model.") else key): value
            for key, value in state_dict.items()
        }

        # Synthesize lm_head from embed_tokens when weights are tied. For a
        # float checkpoint this copies ``embed_tokens.weight``; for a quantized
        # checkpoint the tied MatMulNBits head tensors (weight/scales/
        # zero_points) are emitted directly by the loader, so nothing to do here.
        quantization = self.config.quantization
        if self.config.tie_word_embeddings and not (
            quantization is not None and quantization.quantize_lm_head
        ):
            embed_key = "language_model.embed_tokens.weight"
            head_key = "language_model.lm_head.weight"
            if head_key not in state_dict and embed_key in state_dict:
                state_dict[head_key] = state_dict[embed_key]

        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("lm_head."):
                renamed["decoder." + key] = value

            elif key.startswith("language_model."):
                suffix = key[len("language_model.") :]
                if suffix.startswith("lm_head"):
                    # lm_head lives directly under decoder (not decoder.model)
                    renamed["decoder." + suffix] = value
                elif any(suffix.startswith(p) for p in _PER_LAYER_EMBEDDING_PREFIXES):
                    # Per-layer embedding weights → embedding sub-model
                    renamed["embedding." + suffix] = value
                else:
                    # All other text weights nest under decoder.model.*
                    onnx_key = "decoder.model." + suffix
                    renamed[onnx_key] = value
                    if suffix.startswith("embed_tokens."):
                        # Token embedding is shared with the embedding sub-model.
                        # The suffix tail (``weight`` for float, or ``qweight`` /
                        # ``scales`` / ``zero_points`` for a GatherBlockQuantized
                        # table) is preserved so both float and quantized
                        # embeddings route correctly.
                        renamed["embedding." + suffix] = value

            elif key.startswith("vision_tower."):
                new_key = "vision_encoder.encoder." + key[len("vision_tower.") :]
                # HF wraps encoder layers under an extra "encoder." sub-module; strip it
                new_key = new_key.replace(
                    "vision_encoder.encoder.encoder.", "vision_encoder.encoder.", 1
                )
                # HF uses Gemma4ClippableLinear which adds a ".linear." infix;
                # strip it for float and packed qweight/scales/qzeros tensors.
                new_key = new_key.replace(".linear.", ".")
                renamed[new_key] = value

            elif key.startswith("embed_vision.embedding_projection."):
                suffix = key[len("embed_vision.embedding_projection.") :]
                renamed["vision_encoder.projector." + suffix] = value

            elif key.startswith("embed_vision.embedding_pre_projection_norm."):
                pass  # Scale-free RMSNorm: normalizes without a learnable scale, no weight

            elif key.startswith("audio_tower."):
                new_key = "audio_encoder.encoder." + key[len("audio_tower.") :]
                # HF Conformer linears use a ".linear." infix; strip it for
                # float and packed qweight/scales/qzeros tensors.
                new_key = new_key.replace(".linear.", ".")
                # HF subsample_conv_projection uses "layerN.conv" / "layerN.norm" names;
                # our ONNX module uses "convN" / "normN" directly.
                new_key = new_key.replace(
                    ".subsample_conv_projection.layer0.conv.",
                    ".subsample_conv_projection.conv0.",
                )
                new_key = new_key.replace(
                    ".subsample_conv_projection.layer0.norm.",
                    ".subsample_conv_projection.norm0.",
                )
                new_key = new_key.replace(
                    ".subsample_conv_projection.layer1.conv.",
                    ".subsample_conv_projection.conv1.",
                )
                new_key = new_key.replace(
                    ".subsample_conv_projection.layer1.norm.",
                    ".subsample_conv_projection.norm1.",
                )
                renamed[new_key] = value

            elif key.startswith("embed_audio.embedding_projection."):
                # Learned audio-to-text projector (embed_audio.embedding_projection in HF)
                suffix = key[len("embed_audio.embedding_projection.") :]
                renamed["audio_encoder.projector." + suffix] = value

            elif key.startswith("embed_audio."):
                # embed_audio.embedding_pre_projection_norm.* — Gemma4RMSNorm(with_scale=False):
                # normalises activations but carries no learnable weight.  No checkpoint key
                # is saved, so nothing to map here.  The norm IS applied in the ONNX forward
                # pass via _Gemma4AudioEncoderModel.pre_projection_norm.
                pass

            else:
                renamed[key] = value

        # Map HF expert weight names and fold router scale
        _remap_moe_expert_weights(renamed, self.config)

        # For WebGPU, split the fused per-layer embedding and move its projections
        # into the decoder, which owns the split tables in that configuration.
        _route_split_per_layer_weights(renamed, self.config)

        return _preprocess_quantized_checkpoint(renamed, self.config)


# ---------------------------------------------------------------------------
# Gemma4UnifiedModel — gemma-4-12B encoder-free multimodal model
# ---------------------------------------------------------------------------


class Gemma4UnifiedModel(nn.Module):
    """Unified gemma-4-12B (``gemma4_unified``) multimodal model.

    Encoder-free counterpart to :class:`Gemma4Model`: it shares the gemma4
    text decoder and the multimodal-fusion embedding sub-model, but replaces
    the SigLIP vision tower and Conformer audio tower with the lightweight
    encoder-free embedders (:class:`_Gemma4UnifiedVisionEmbedderModel` and
    :class:`_Gemma4UnifiedAudioEmbedderModel`).

    Builds a 3- or 4-model package (built by :class:`~mobius.tasks.Gemma4UnifiedTask`):

    Always produced:
    - ``decoder``: gemma4 text decoder taking ``inputs_embeds`` (and
      ``input_ids`` for the vision-block bidirectional mask, which it derives
      internally; dual head_dim, k_eq_v)
    - ``vision_encoder``: raw-patch vision embedder
    - ``embedding``: scaled word embedding + multimodal feature fusion

    Added when ``config.audio is not None``:
    - ``audio_encoder``: raw-frame audio embedder

    Registered as ``gemma4_unified``.
    """

    default_task: str = "gemma4-unified"
    category: str = "Multimodal"

    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.config = config
        self.decoder = _Gemma4DecoderModel(config)
        self.vision_encoder = _Gemma4UnifiedVisionEmbedderModel(config)
        self.embedding = Gemma4EmbeddingModel(config)
        self.audio_encoder: _Gemma4UnifiedAudioEmbedderModel | None = (
            _Gemma4UnifiedAudioEmbedderModel(config) if config.audio is not None else None
        )

    def forward(self, op: OpBuilder, **kwargs):
        raise NotImplementedError(
            "Gemma4UnifiedModel is a multi-model split; Gemma4UnifiedTask builds "
            "each sub-module (decoder, vision_encoder, embedding, and optionally "
            "audio_encoder) separately."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Rename HuggingFace ``gemma4_unified`` checkpoint keys to ONNX names.

        After stripping the leading ``model.`` prefix:

        - ``language_model.lm_head.*`` → ``decoder.lm_head.*``
        - ``language_model.*`` → ``decoder.model.*`` (token embedding is also
          shared with ``embedding.embed_tokens.weight``)
        - ``vision_embedder.*`` → ``vision_encoder.*`` (``pos_embedding`` is
          split into ``pos_emb_x``/``pos_emb_y``)
        - ``embed_vision.embedding_projection.*`` → ``vision_encoder.projector.*``
        - ``embed_audio.embedding_projection.*`` → ``audio_encoder.projector.*``
        - ``embed_{vision,audio}.*.embedding_pre_projection_norm.*`` → skip
          (scale-free RMSNorm, no learnable weight)
        """
        # Strip top-level "model." prefix used by HF multimodal checkpoints.
        state_dict = {
            (key[len("model.") :] if key.startswith("model.") else key): value
            for key, value in state_dict.items()
        }

        # Synthesize lm_head from embed_tokens when weights are tied.
        quantization = self.config.quantization
        if self.config.tie_word_embeddings and not (
            quantization is not None and quantization.quantize_lm_head
        ):
            embed_key = "language_model.embed_tokens.weight"
            head_key = "language_model.lm_head.weight"
            if head_key not in state_dict and embed_key in state_dict:
                state_dict[head_key] = state_dict[embed_key]

        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("language_model."):
                suffix = key[len("language_model.") :]
                if suffix.startswith("lm_head"):
                    renamed["decoder." + suffix] = value
                elif any(suffix.startswith(p) for p in _PER_LAYER_EMBEDDING_PREFIXES):
                    renamed["embedding." + suffix] = value
                else:
                    renamed["decoder.model." + suffix] = value
                    if suffix == "embed_tokens.weight":
                        renamed["embedding.embed_tokens.weight"] = value

            elif key.startswith("vision_embedder.pos_embedding"):
                # [posemb_size, 2, mm_embed_dim] → two [posemb_size, mm_embed_dim]
                renamed["vision_encoder.pos_emb_x.weight"] = value[:, 0, :].contiguous()
                renamed["vision_encoder.pos_emb_y.weight"] = value[:, 1, :].contiguous()

            elif key.startswith("vision_embedder."):
                renamed["vision_encoder." + key[len("vision_embedder.") :]] = value

            elif key.startswith("embed_vision.embedding_projection."):
                suffix = key[len("embed_vision.embedding_projection.") :]
                renamed["vision_encoder.projector." + suffix] = value

            elif key.startswith("embed_audio.embedding_projection."):
                suffix = key[len("embed_audio.embedding_projection.") :]
                renamed["audio_encoder.projector." + suffix] = value

            elif key.startswith(("embed_vision.", "embed_audio.")):
                # *.embedding_pre_projection_norm.*: scale-free RMSNorm, no weight.
                pass

            else:
                renamed[key] = value

        _route_split_per_layer_weights(renamed, self.config)
        return _preprocess_quantized_checkpoint(renamed, self.config)
