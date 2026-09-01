# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""GPT-OSS causal language model with Mixture-of-Experts and attention sinks.

GPT-OSS (openai/gpt-oss-20b) features:
- Alternating sliding/full attention layers (config.layer_types)
- Attention sinks: learnable per-head scalar appended to the softmax denominator,
  allowing each head to "discard" tokens into a virtual null position
- GQA with attention projection biases (config.attention_bias=True)
- YaRN RoPE
- MoE FFN: top-k routing with softmax scores and additive router bias
- Custom gated activation: (up.clamp(-L,L) + 1) * silu_alpha(gate.clamp(max=L))
  where silu_alpha(x) = x * sigmoid(alpha * x), alpha=1.702, L=7.0

HuggingFace reference: ``GptOssForCausalLM``.
"""

from __future__ import annotations

import math

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._build_context import ep_capabilities, get_build_dtype
from mobius._configs import ArchitectureConfig, QuantizedWeightFormat
from mobius.components import (
    Embedding,
    Linear,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
)
from mobius.components._moe import (
    _flatten_to_2d,
    _realize_gate_and_get_qmoe_routing,
)
from mobius.components._rotary_embedding import apply_rotary_pos_emb
from mobius.models.base import CausalLMModel

_MXFP4_BLOCK_SIZE = 32
_MXFP4_REPACK_OUTPUT_PAIRS_PER_CHUNK = 64
_MXFP4_SCALE_VALIDATION_ELEMENTS_PER_CHUNK = 1 << 20


def repack_gptoss_mxfp4_blocks(blocks: torch.Tensor) -> torch.Tensor:
    """Losslessly repack GPT-OSS MXFP4 codes for QMoE.

    The checkpoint layout is ``[E, N, K/32, 16]``. Each byte stores two
    adjacent K-axis E2M1 codes (even K in the low nibble). QMoE's native FP4
    input is the legacy column-major layout ``[E, K, N/2]`` where each byte
    stores adjacent N-axis codes (even N in the low nibble).

    Only nibble placement changes; no FP4 value is decoded or requantized.
    """
    if blocks.dtype != torch.uint8:
        raise TypeError(f"GPT-OSS MXFP4 blocks must be uint8, got {blocks.dtype}")
    if blocks.ndim != 4:
        raise ValueError(
            "GPT-OSS MXFP4 blocks must have rank 4 with shape "
            f"[E, N, K/32, 16], got {tuple(blocks.shape)}"
        )
    if blocks.shape[-1] != _MXFP4_BLOCK_SIZE // 2:
        raise ValueError(
            "GPT-OSS MXFP4 blocks must have 16 packed bytes per block, "
            f"got shape {tuple(blocks.shape)}"
        )
    if blocks.shape[1] % 2:
        raise ValueError(
            f"GPT-OSS MXFP4 output dimension N must be even, got N={blocks.shape[1]}"
        )

    num_experts, output_size, num_blocks, _ = blocks.shape
    output_pairs = output_size // 2
    destination = torch.empty(
        (num_experts, num_blocks * _MXFP4_BLOCK_SIZE, output_pairs),
        dtype=torch.uint8,
        device=blocks.device,
    )
    # Expose K as [block, packed-K-byte, K-nibble] so each assignment writes
    # directly into the final storage. Only one bounded scratch plane is used;
    # in particular, there are no projection-sized even/odd/stack/contiguous
    # intermediates while repacking 120B expert banks.
    destination_planes = destination.view(
        num_experts,
        num_blocks,
        _MXFP4_BLOCK_SIZE // 2,
        2,
        output_pairs,
    )
    chunk_pairs = max(1, min(_MXFP4_REPACK_OUTPUT_PAIRS_PER_CHUNK, output_pairs))
    scratch = torch.empty(
        (num_experts, num_blocks, _MXFP4_BLOCK_SIZE // 2, chunk_pairs),
        dtype=torch.uint8,
        device=blocks.device,
    )
    for pair_start in range(0, output_pairs, chunk_pairs):
        pair_end = min(pair_start + chunk_pairs, output_pairs)
        chunk_size = pair_end - pair_start
        even_n = blocks[:, 2 * pair_start : 2 * pair_end : 2].permute(0, 2, 3, 1)
        odd_n = blocks[:, 2 * pair_start + 1 : 2 * pair_end : 2].permute(0, 2, 3, 1)
        workspace = scratch[..., :chunk_size]

        # even K code: low nibble = even N, high nibble = odd N.
        target = destination_planes[..., 0, pair_start:pair_end]
        target.copy_(odd_n)
        target.bitwise_and_(0x0F)
        target.bitwise_left_shift_(4)
        workspace.copy_(even_n)
        workspace.bitwise_and_(0x0F)
        target.bitwise_or_(workspace)

        # odd K code: the corresponding high checkpoint nibbles.
        target = destination_planes[..., 1, pair_start:pair_end]
        target.copy_(odd_n)
        target.bitwise_and_(0xF0)
        workspace.copy_(even_n)
        workspace.bitwise_right_shift_(4)
        target.bitwise_or_(workspace)

    return destination


def _validate_mxfp4_scale_bytes(scales: torch.Tensor, scale_key: str) -> None:
    """Validate raw E8M0 bytes using bounded scratch.

    ONNX FLOAT8E8M0 reserves byte ``0xff`` for NaN, which is not a valid MXFP4
    block scale. Validation is chunked so checking a 120B checkpoint does not
    allocate another projection-sized boolean tensor.
    """
    flattened = scales.reshape(-1)
    for start in range(0, flattened.numel(), _MXFP4_SCALE_VALIDATION_ELEMENTS_PER_CHUNK):
        chunk = flattened[start : start + _MXFP4_SCALE_VALIDATION_ELEMENTS_PER_CHUNK]
        if torch.any(chunk == 0xFF).item():
            raise ValueError(
                f"GPT-OSS MXFP4 scales {scale_key!r} contain invalid raw E8M0 "
                "byte 0xff (NaN); valid scale bytes are 0x00 through 0xfe."
            )


def _reinterpret_mxfp4_scales_unchecked(scales: torch.Tensor) -> torch.Tensor:
    """Reinterpret prevalidated raw scale bytes without numeric conversion."""
    if not scales.is_contiguous():
        scales = scales.contiguous()
    # Same-size dtype views preserve the raw storage and exponent bytes.
    return scales.view(torch.float8_e8m0fnu)


def _reinterpret_valid_mxfp4_scales(scales: torch.Tensor, scale_key: str) -> torch.Tensor:
    """Validate and reinterpret raw E8M0 bytes without a numeric copy."""
    _validate_mxfp4_scale_bytes(scales, scale_key)
    return _reinterpret_mxfp4_scales_unchecked(scales)


def _native_mxfp4_projection_specs(
    config: ArchitectureConfig,
) -> dict[str, dict[str, tuple[tuple[int, ...], str]]]:
    """Return deterministic source geometry and graph targets for every MoE layer."""
    num_experts = config.num_local_experts
    if num_experts is None:
        raise ValueError("GPT-OSS MXFP4 requires num_local_experts")
    return {
        f"model.layers.{layer_index}.mlp": {
            "gate_up_proj": (
                (
                    num_experts,
                    2 * config.intermediate_size,
                    config.hidden_size // _MXFP4_BLOCK_SIZE,
                    _MXFP4_BLOCK_SIZE // 2,
                ),
                "fc1",
            ),
            "down_proj": (
                (
                    num_experts,
                    config.hidden_size,
                    config.intermediate_size // _MXFP4_BLOCK_SIZE,
                    _MXFP4_BLOCK_SIZE // 2,
                ),
                "fc2",
            ),
        }
        for layer_index in range(config.num_hidden_layers)
    }


def _validate_native_mxfp4_state(
    state_dict: dict[str, torch.Tensor],
    config: ArchitectureConfig,
    mxfp4_blocks: dict[str, str],
    mxfp4_scales: dict[str, str],
) -> dict[str, dict[str, tuple[tuple[int, ...], str]]]:
    """Preflight the complete native expert set without mutating caller state."""
    specs = _native_mxfp4_projection_specs(config)
    expected_bases = {
        f"{mlp_root}.experts.{projection}"
        for mlp_root, projections in specs.items()
        for projection in projections
    }
    block_bases = set(mxfp4_blocks)
    scale_bases = set(mxfp4_scales)
    if block_bases != expected_bases or scale_bases != expected_bases:
        missing_blocks = sorted(expected_bases - block_bases)
        missing_scales = sorted(expected_bases - scale_bases)
        unexpected_blocks = sorted(block_bases - expected_bases)
        unexpected_scales = sorted(scale_bases - expected_bases)
        raise ValueError(
            "Malformed GPT-OSS MXFP4 checkpoint: every expected MoE layer must "
            "contain exactly gate_up_proj/down_proj block tensors with a "
            "matching scale tensor. "
            f"Missing blocks: {missing_blocks}; missing scales: {missing_scales}; "
            f"unexpected blocks: {unexpected_blocks}; unexpected scales: "
            f"{unexpected_scales}."
        )

    num_experts = config.num_local_experts
    assert num_experts is not None
    for mlp_root in sorted(specs):
        for projection, (expected_block_shape, _target) in specs[mlp_root].items():
            base = f"{mlp_root}.experts.{projection}"
            block_key = mxfp4_blocks[base]
            scale_key = mxfp4_scales[base]
            blocks = state_dict[block_key]
            scales = state_dict[scale_key]
            if blocks.dtype != torch.uint8:
                raise TypeError(
                    f"GPT-OSS MXFP4 blocks {block_key!r} must be uint8, got {blocks.dtype}"
                )
            if tuple(blocks.shape) != expected_block_shape:
                raise ValueError(
                    f"GPT-OSS MXFP4 blocks {block_key!r} must have shape "
                    f"{expected_block_shape}, got {tuple(blocks.shape)}"
                )
            expected_scale_shape = expected_block_shape[:-1]
            if scales.dtype != torch.uint8:
                raise TypeError(
                    f"GPT-OSS MXFP4 scales {scale_key!r} must contain raw "
                    f"E8M0 bytes as uint8, got {scales.dtype}"
                )
            if tuple(scales.shape) != expected_scale_shape:
                raise ValueError(
                    f"GPT-OSS MXFP4 scales {scale_key!r} must have shape "
                    f"{expected_scale_shape}, got {tuple(scales.shape)}"
                )
            _validate_mxfp4_scale_bytes(scales, scale_key)

        tensor_specs = {
            f"{mlp_root}.experts.gate_up_proj_bias": (
                (num_experts, 2 * config.intermediate_size),
                "expert bias",
            ),
            f"{mlp_root}.experts.down_proj_bias": (
                (num_experts, config.hidden_size),
                "expert bias",
            ),
            f"{mlp_root}.router.weight": (
                (num_experts, config.hidden_size),
                "router tensor",
            ),
            f"{mlp_root}.router.bias": ((num_experts,), "router tensor"),
        }
        for key, (expected_shape, description) in tensor_specs.items():
            tensor = state_dict.get(key)
            if tensor is None:
                raise ValueError(
                    f"Native GPT-OSS MXFP4 QMoE requires {description} tensor {key!r}."
                )
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"GPT-OSS {description} {key!r} must have shape "
                    f"{expected_shape}, got {tuple(tensor.shape)}"
                )
            if not tensor.is_floating_point():
                raise TypeError(
                    f"GPT-OSS {description} {key!r} must be floating point, got {tensor.dtype}"
                )
    return specs


class _GptOssGate(nn.Module):
    """Top-k router with additive bias for GPT-OSS.

    HF ``GptOssTopKRouter``: router_logits = hidden @ weight.T + bias,
    then top-k selection followed by softmax over the selected scores.
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.weight = nn.Parameter([num_experts, hidden_size])
        self.bias = nn.Parameter([num_experts])

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        # router_logits: [B, S, N_exp] = [B, S, H] @ [H, N_exp] + [N_exp]
        weight_t = op.Transpose(self.weight, perm=[1, 0])
        router_logits = op.MatMul(hidden_states, weight_t)
        router_logits = op.Add(router_logits, self.bias)
        k = op.Constant(value_ints=[self.top_k])
        routing_weights, selected_experts = op.TopK(router_logits, k, axis=-1, _outputs=2)
        # Softmax over top-k logits only
        routing_weights = op.Softmax(routing_weights, axis=-1)
        return routing_weights, selected_experts

    def qmoe_routing(self, op: OpBuilder, hidden_states: ir.Value):
        """Return biased logits for QMoE's internal top-k and normalization."""
        weight_t = op.Transpose(self.weight, perm=[1, 0])
        router_logits = op.Add(op.MatMul(hidden_states, weight_t), self.bias)
        return op.CastLike(router_logits, hidden_states), None, True, 1.0


class _GptOssExpertMLP(nn.Module):
    """Expert MLP with GPT-OSS custom gated activation and projection biases.

    Implements:
        gate = gate_proj(x)                   # [B, S, inter]
        up   = up_proj(x)                     # [B, S, inter]
        gate_clamped = min(gate, L)           # clamp from above
        up_clamped   = clip(up, -L, L)       # symmetric clamp
        glu  = gate_clamped * sigmoid(alpha * gate_clamped)   # SiLU with alpha
        out  = (up_clamped + 1) * glu         # gated output
        return down_proj(out)

    HF ``GptOssExperts._apply_gate`` uses alpha=1.702 and L=7.0 (``swiglu_limit``).
    Note: in HF gate/up are interleaved in a packed weight; here they are split
    in ``preprocess_weights`` into separate ``gate_proj``/``up_proj`` parameters.
    """

    _ALPHA: float = 1.702
    _LIMIT: float = 7.0

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=True)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=True)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        gate = self.gate_proj(op, hidden_states)  # [B, S, inter]
        up = self.up_proj(op, hidden_states)  # [B, S, inter]

        # gate.clamp(max=limit): min(gate, 7.0)
        gate_clamped = op.Min(gate, self._LIMIT)
        # up.clamp(-limit, limit)
        up_clamped = op.Clip(up, -self._LIMIT, self._LIMIT)

        # glu = gate * sigmoid(alpha * gate)  — SiLU with custom alpha
        glu = op.Mul(gate_clamped, op.Sigmoid(op.Mul(self._ALPHA, gate_clamped)))

        # gated_output = (up + 1) * glu
        gated = op.Mul(op.Add(up_clamped, 1.0), glu)
        return self.down_proj(op, gated)  # [B, S, hidden]


class _GptOssMoELayer(nn.Module):
    """MoE layer for GPT-OSS with biased router and custom expert MLPs."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        assert config.num_local_experts is not None
        assert config.num_experts_per_tok is not None
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.gate = _GptOssGate(config.hidden_size, self.num_experts, self.top_k)
        quantization = config.quantization
        native_mxfp4 = bool(
            quantization is not None
            and quantization.weight_format is QuantizedWeightFormat.MXFP4
        )
        if native_mxfp4:
            if config.disable_qmoe:
                raise ValueError(
                    "Native GPT-OSS MXFP4 weights require com.microsoft::QMoE; "
                    "disable_qmoe=True has no lossless dense fallback."
                )
            self.experts = None
            self._init_native_mxfp4_parameters(config)
        else:
            self.experts = nn.ModuleList(
                [
                    _GptOssExpertMLP(config.hidden_size, config.intermediate_size)
                    for _ in range(self.num_experts)
                ]
            )

    def _init_native_mxfp4_parameters(self, config: ArchitectureConfig) -> None:
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        if hidden_size % _MXFP4_BLOCK_SIZE or intermediate_size % _MXFP4_BLOCK_SIZE:
            raise ValueError(
                "Native GPT-OSS MXFP4 QMoE requires hidden_size and "
                f"intermediate_size divisible by 32, got {hidden_size} and {intermediate_size}"
            )

        # QMoE's native FP4 initializer is column-major [E, K, N/2].
        self.fc1_experts_weights = nn.Parameter(
            [self.num_experts, hidden_size, intermediate_size],
            dtype=ir.DataType.UINT8,
        )
        self.fc1_scales = nn.Parameter(
            [
                self.num_experts,
                2 * intermediate_size,
                hidden_size // _MXFP4_BLOCK_SIZE,
            ],
            dtype=ir.DataType.FLOAT8E8M0,
        )
        self.fc1_experts_bias = nn.Parameter([self.num_experts, 2 * intermediate_size])
        self.fc1_global_scales = nn.Parameter(
            [self.num_experts],
            dtype=ir.DataType.FLOAT,
        )
        self.fc1_global_scales._keep_float32 = True  # type: ignore[attr-defined]

        self.fc2_experts_weights = nn.Parameter(
            [self.num_experts, intermediate_size, hidden_size // 2],
            dtype=ir.DataType.UINT8,
        )
        self.fc2_scales = nn.Parameter(
            [
                self.num_experts,
                hidden_size,
                intermediate_size // _MXFP4_BLOCK_SIZE,
            ],
            dtype=ir.DataType.FLOAT8E8M0,
        )
        self.fc2_experts_bias = nn.Parameter([self.num_experts, hidden_size])
        self.fc2_global_scales = nn.Parameter(
            [self.num_experts],
            dtype=ir.DataType.FLOAT,
        )
        self.fc2_global_scales._keep_float32 = True  # type: ignore[attr-defined]

    def _native_mxfp4_forward(self, op: OpBuilder, hidden_states: ir.Value):
        """Emit the native op only for a CUDA FP4-QMoE FP16/BF16 build."""
        target_ep = ep_capabilities().name
        if target_ep != "cuda":
            raise NotImplementedError(
                "Native GPT-OSS MXFP4 requires the CUDA FP4-QMoE runtime/build; "
                f"got execution_provider={target_ep!r}. Export with "
                "--execution-provider cuda and --dtype f16 (or bf16). "
                "CPU/default cannot execute FLOAT8E8M0-scaled FP4 QMoE, and no "
                "lossless dense fallback exists. ORT must be built with FP4 QMoE "
                "enabled (CUDA >=12.8); pre-Blackwell GPUs such as A100 may require "
                "the available SM80 fallback/runtime configuration."
            )
        build_dtype = get_build_dtype()
        if build_dtype not in {
            ir.DataType.FLOAT16,
            ir.DataType.BFLOAT16,
        }:
            raise ValueError(
                "Native GPT-OSS MXFP4 requires an FP16 or BF16 CUDA FP4-QMoE "
                f"runtime/build, got build dtype {build_dtype}. Export with "
                "--execution-provider cuda and --dtype f16 (or bf16). ORT must "
                "be built with FP4 QMoE enabled (CUDA >=12.8); pre-Blackwell "
                "GPUs such as A100 may require the available SM80 "
                "fallback/runtime configuration."
            )

        router_probs, _, normalize, _ = _realize_gate_and_get_qmoe_routing(
            op, self.gate, hidden_states
        )

        return op.QMoE(
            hidden_states,
            _flatten_to_2d(op, router_probs),
            self.fc1_experts_weights,
            self.fc1_scales,
            self.fc1_experts_bias,
            self.fc2_experts_weights,
            self.fc2_scales,
            self.fc2_experts_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            self.fc1_global_scales,
            self.fc2_global_scales,
            activation_type="swiglu",
            activation_alpha=1.702,
            activation_beta=1.0,
            swiglu_limit=7.0,
            normalize_routing_weights=int(normalize),
            k=self.top_k,
            expert_weight_bits=4,
            block_size=_MXFP4_BLOCK_SIZE,
            swiglu_fusion=1,
            quant_type="fp4",
            _domain="com.microsoft",
        )

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        if self.experts is None:
            return self._native_mxfp4_forward(op, hidden_states)

        routing_weights, selected_experts = self.gate(op, hidden_states)

        # Loop over experts: mask-and-accumulate dispatch
        result = None
        for expert_idx, expert in enumerate(self.experts):
            expert_output = expert(op, hidden_states)
            expert_id = op.Constant(value_int=expert_idx)
            match = op.Equal(selected_experts, expert_id)
            match_float = op.Cast(match, to=1)  # FLOAT
            weighted = op.Mul(routing_weights, match_float)
            weight = op.ReduceSum(weighted, [-1], keepdims=True)
            contribution = op.Mul(expert_output, weight)
            if result is None:
                result = contribution
            else:
                result = op.Add(result, contribution)

        return result


class _GptOssAttention(nn.Module):
    """GQA attention with learned per-head sinks for GPT-OSS.

    HF ``eager_attention_forward`` appends one extra logit per token per head
    (the learnable ``sinks`` value) to the attention scores before softmax.
    This lets each head "discard" a token's weight into a virtual null position:

        combined = cat([attn_weights, sinks_expanded], dim=-1)  # [B, H, S, S_kv+1]
        combined = combined - max(combined)                      # numerical stability
        probs    = softmax(combined, dim=-1)[..., :-1]           # drop sink, [B, H, S, S_kv]
        out      = probs @ V

    Implements this manually (cannot use fused op.Attention with sinks).
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_kv_groups = config.num_attention_heads // config.num_key_value_heads
        self.scale = config.head_dim**-0.5
        partial_rotary_factor = config.partial_rotary_factor
        if partial_rotary_factor is None:
            raise ValueError("GPT-OSS requires partial_rotary_factor")
        self._rotary_embedding_dim = (
            0
            if math.isclose(partial_rotary_factor, 1.0)
            else int(self.head_dim * partial_rotary_factor)
        )
        self._rope_interleave = config.rope_interleave

        # QKV projections with bias (attention_bias=True for GPT-OSS)
        self.q_proj = Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=config.attn_qkv_bias,
        )
        self.k_proj = Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attn_qkv_bias,
        )
        self.v_proj = Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attn_qkv_bias,
        )
        self.o_proj = Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=config.attn_o_bias,
        )

        # Learnable sink logit: one scalar per attention head [num_heads]
        self.sinks = nn.Parameter([config.num_attention_heads])

    def _expand_kv_for_gqa(
        self,
        op: OpBuilder,
        kv: ir.Value,
        batch_1d: ir.Value,
        kv_len_1d: ir.Value,
    ) -> ir.Value:
        """Expand KV from [B, kv_heads, S, d] to [B, q_heads, S, d] for GQA.

        Uses unsqueeze+expand+reshape to replicate each KV head ``num_kv_groups``
        times consecutively: [kv0]*g, [kv1]*g, ..., which is what ``repeat_kv`` does.
        """
        # [B, kv_heads, S, d] → [B, kv_heads, 1, S, d]
        kv_5d = op.Unsqueeze(kv, [2])
        # Expand to [B, kv_heads, num_kv_groups, S, d]
        expand_shape = op.Concat(
            batch_1d,
            [self.num_key_value_heads, self.num_kv_groups],
            kv_len_1d,
            [self.head_dim],
            axis=0,
        )
        kv_exp = op.Expand(kv_5d, expand_shape)
        # Flatten to [B, q_heads, S, d]
        flat_shape = op.Concat(
            batch_1d,
            [self.num_attention_heads],
            kv_len_1d,
            [self.head_dim],
            axis=0,
        )
        return op.Reshape(kv_exp, flat_shape)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple | None = None,
        past_key_value: tuple | None = None,
    ):
        # hidden_states: [B, S, H]
        batch_1d = op.Shape(hidden_states, start=0, end=1)  # 1D tensor containing B
        seq_1d = op.Shape(hidden_states, start=1, end=2)  # 1D tensor containing S

        # QKV projections: [B, S, heads * d]
        query = self.q_proj(op, hidden_states)
        key = self.k_proj(op, hidden_states)
        value = self.v_proj(op, hidden_states)

        # Apply RoPE on 3D packed format [B, S, heads * d]
        if position_embeddings is not None:
            query = apply_rotary_pos_emb(
                op,
                x=query,
                position_embeddings=position_embeddings,
                num_heads=self.num_attention_heads,
                rotary_embedding_dim=self._rotary_embedding_dim,
                interleaved=self._rope_interleave,
            )
            key = apply_rotary_pos_emb(
                op,
                x=key,
                position_embeddings=position_embeddings,
                num_heads=self.num_key_value_heads,
                rotary_embedding_dim=self._rotary_embedding_dim,
                interleaved=self._rope_interleave,
            )

        # Reshape to 4D and transpose: [B, S, heads, d] → [B, heads, S, d]
        query = op.Transpose(
            op.Reshape(query, [0, 0, self.num_attention_heads, self.head_dim]),
            perm=[0, 2, 1, 3],
        )  # [B, q_heads, S, d]
        key = op.Transpose(
            op.Reshape(key, [0, 0, self.num_key_value_heads, self.head_dim]),
            perm=[0, 2, 1, 3],
        )  # [B, kv_heads, S, d]
        value = op.Transpose(
            op.Reshape(value, [0, 0, self.num_key_value_heads, self.head_dim]),
            perm=[0, 2, 1, 3],
        )  # [B, kv_heads, S, d]

        # KV cache: prepend past tokens
        if past_key_value is not None:
            key = op.Concat(past_key_value[0], key, axis=2)  # [B, kv_heads, past+S, d]
            value = op.Concat(past_key_value[1], value, axis=2)  # [B, kv_heads, past+S, d]
        present_key_value = (key, value)

        # Total KV sequence length (after cache concatenation)
        kv_len_1d = op.Shape(key, start=2, end=3)  # [total_S]

        # GQA: expand key/value from kv_heads to q_heads
        if self.num_kv_groups > 1:
            key_exp = self._expand_kv_for_gqa(op, key, batch_1d, kv_len_1d)
            value_exp = self._expand_kv_for_gqa(op, value, batch_1d, kv_len_1d)
        else:
            key_exp = key
            value_exp = value

        # Attention scores: [B, q_heads, S_q, S_kv]
        # query @ key.T: [B, q_heads, S_q, d] @ [B, q_heads, d, S_kv]
        key_t = op.Transpose(key_exp, perm=[0, 1, 3, 2])  # [B, q_heads, d, S_kv]
        attn_scores = op.MatMul(query, key_t)  # [B, q_heads, S_q, S_kv]
        attn_scores = op.Mul(attn_scores, self.scale)

        # Add causal+sliding_window+padding mask (float additive bias)
        if attention_bias is not None:
            # attention_bias: [B, 1, S_q, S_kv] — broadcasts over q_heads
            attn_scores = op.Add(attn_scores, attention_bias)

        # Append sinks column: [q_heads] → [B, q_heads, S_q, 1]
        sinks_4d = op.Reshape(
            self.sinks,
            [1, self.num_attention_heads, 1, 1],
        )
        expand_shape = op.Concat(
            batch_1d,
            [self.num_attention_heads],
            seq_1d,
            [1],
            axis=0,
        )
        sinks_expanded = op.Expand(sinks_4d, expand_shape)  # [B, q_heads, S_q, 1]
        # combined: [B, q_heads, S_q, S_kv+1]
        combined = op.Concat(attn_scores, sinks_expanded, axis=-1)

        # Numerical stability: subtract per-row max before softmax
        row_max = op.ReduceMax(combined, [-1], keepdims=True)  # [B, q_heads, S_q, 1]
        combined = op.Sub(combined, row_max)

        # Softmax over the extended sequence (S_kv + 1) dimension
        probs = op.Softmax(combined, axis=-1)  # [B, q_heads, S_q, S_kv+1]

        # Drop sink column: slice axis=3 from 0 to -1 (all but last)
        scores = op.Slice(probs, [0], [-1], [3])  # [B, q_heads, S_q, S_kv]

        # Weighted sum with value: [B, q_heads, S_q, d]
        attn_out = op.MatMul(scores, value_exp)

        # Transpose and flatten heads: [B, q_heads, S_q, d] → [B, S_q, q_heads*d]
        attn_out = op.Transpose(attn_out, perm=[0, 2, 1, 3])  # [B, S_q, q_heads, d]
        attn_out = op.Reshape(attn_out, [0, 0, -1])  # [B, S_q, q_heads*d]

        # Output projection
        attn_out = self.o_proj(op, attn_out)
        return attn_out, present_key_value


class _GptOssDecoderLayer(nn.Module):
    """GPT-OSS decoder layer: pre-norm attention (with sinks) + pre-norm MoE FFN."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.self_attn = _GptOssAttention(config)
        self.mlp = _GptOssMoELayer(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple,
        past_key_value: tuple | None,
    ):
        # Pre-norm attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        attn_out, present_kv = self.self_attn(
            op, hidden_states, attention_bias, position_embeddings, past_key_value
        )
        hidden_states = op.Add(residual, attn_out)

        # Pre-norm MoE FFN block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, present_kv


class _GptOssTextModel(nn.Module):
    """GPT-OSS text backbone with alternating sliding/full attention layers."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype
        self._layer_types = config.layer_types  # list of 'sliding_attention'/'full_attention'
        self._sliding_window = config.sliding_window
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [_GptOssDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        rotary_emb = initialize_rope(config)
        if rotary_emb is None:
            raise ValueError("GPT-OSS requires rotary position embeddings")
        self.rotary_emb = rotary_emb

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value | None,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states = self.embed_tokens(op, input_ids)
        position_embeddings = self.rotary_emb(op, position_ids)

        # Create attention biases (full and sliding window) for dynamic dispatch
        full_attn_bias = (
            create_attention_bias(
                op,
                input_ids=input_ids,
                attention_mask=attention_mask,
                dtype=self._dtype,
            )
            if attention_mask is not None
            else None
        )
        sliding_attn_bias = None
        if self._sliding_window is not None and attention_mask is not None:
            sliding_attn_bias = create_attention_bias(
                op,
                input_ids=input_ids,
                attention_mask=attention_mask,
                sliding_window=self._sliding_window,
                dtype=self._dtype,
            )

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for i, (layer, past_kv) in enumerate(zip(self.layers, past_kvs)):
            # Select bias: sliding window for 'sliding_attention' layers, full for others
            layer_type = self._layer_types[i] if self._layer_types else "full_attention"
            if layer_type == "sliding_attention" and sliding_attn_bias is not None:
                attn_bias = sliding_attn_bias
            else:
                attn_bias = full_attn_bias

            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attn_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values


class GPTOSSCausalLMModel(CausalLMModel):
    """GPT-OSS causal language model with MoE FFN and attention sinks.

    Architecture highlights:
    - Alternating sliding/full attention layers (``config.layer_types``)
    - Attention sinks: learned per-head scalar extends softmax denominator
    - GQA with YaRN RoPE and attention projection biases
    - MoE FFN: top-k routing with softmax scores and additive router bias
    - Custom activation: (up+1) * silu_alpha(gate), alpha=1.702

    HuggingFace model_type: ``gpt_oss``.
    """

    category: str = "Mixture of Experts"

    def __init__(self, config: ArchitectureConfig):
        nn.Module.__init__(self)
        self.config = config
        self.model = _GptOssTextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HF GPT-OSS weight names to our ONNX parameter names.

        Key transformations:
        - ``mlp.router.{weight,bias}`` → ``mlp.gate.{weight,bias}``
        - ``mlp.experts.gate_up_proj [N, hidden, 2*inter]``: de-interleave and
          transpose to per-expert ``gate_proj.weight`` and ``up_proj.weight [inter, hidden]``
          (HF stores transposed packed weights; gate/up interleaved at every other column)
        - ``mlp.experts.gate_up_proj_bias [N, 2*inter]``: de-interleave to per-expert
          ``gate_proj.bias`` and ``up_proj.bias [inter]``
        - ``mlp.experts.down_proj [N, inter, hidden]``: transpose per-expert to
          ``down_proj.weight [hidden, inter]``
        - ``mlp.experts.down_proj_bias [N, hidden]``: split to per-expert ``down_proj.bias``

        Native MXFP4 checkpoints keep expert ``_blocks`` and ``_scales`` packed.
        Their E2M1 nibbles are losslessly rearranged for QMoE, while E8M0 scale
        bytes are reinterpreted as FLOAT8E8M0 without numerical conversion.
        """
        native_mxfp4 = bool(
            self.config.quantization is not None
            and self.config.quantization.weight_format is QuantizedWeightFormat.MXFP4
        )
        mxfp4_blocks = {
            key.removesuffix("_blocks"): key
            for key in state_dict
            if key.endswith("_blocks") and ".mlp.experts." in key
        }
        mxfp4_scales = {
            key.removesuffix("_scales"): key
            for key in state_dict
            if key.endswith("_scales") and ".mlp.experts." in key
        }
        mxfp4_bases = set(mxfp4_blocks) | set(mxfp4_scales)
        if mxfp4_bases and not native_mxfp4:
            raise ValueError(
                "GPT-OSS MXFP4 expert blocks were found, but the model does not "
                "declare native MXFP4 quantization. A lossless dense fallback is "
                "not available; load the checkpoint's quantization_config with "
                "quant_method='mxfp4' and export com.microsoft::QMoE."
            )

        native_weights: dict[str, torch.Tensor] = {}
        if native_mxfp4:
            num_experts = self.config.num_local_experts
            assert num_experts is not None
            # Validate the complete checkpoint before popping a single source
            # or allocating any repacked destination. A late bad layer must
            # leave the caller's state dict byte-for-byte intact.
            specs = _validate_native_mxfp4_state(
                state_dict,
                self.config,
                mxfp4_blocks,
                mxfp4_scales,
            )
            for mlp_root in sorted(specs):
                for projection, (_expected_block_shape, target) in specs[mlp_root].items():
                    base = f"{mlp_root}.experts.{projection}"
                    block_key = mxfp4_blocks[base]
                    scale_key = mxfp4_scales[base]

                    # Drop both dictionary references before allocating the
                    # repacked destination. Across a multi-layer 120B checkpoint,
                    # this lets each original expert bank be reclaimed while
                    # already-transformed destinations accumulate.
                    blocks = state_dict.pop(block_key)
                    scales = state_dict.pop(scale_key)
                    native_weights[f"{mlp_root}.{target}_scales"] = (
                        _reinterpret_mxfp4_scales_unchecked(scales)
                    )
                    del scales
                    native_weights[f"{mlp_root}.{target}_experts_weights"] = (
                        repack_gptoss_mxfp4_blocks(blocks)
                    )
                    del blocks
                    native_weights[f"{mlp_root}.{target}_global_scales"] = torch.ones(
                        num_experts,
                        dtype=torch.float32,
                    )

            for mlp_root in sorted(specs):
                bias_specs = {
                    f"{mlp_root}.experts.gate_up_proj_bias": f"{mlp_root}.fc1_experts_bias",
                    f"{mlp_root}.experts.down_proj_bias": f"{mlp_root}.fc2_experts_bias",
                }
                for source, target in bias_specs.items():
                    native_weights[target] = state_dict.pop(source)

        # Split fused/stacked weights into per-expert tensors for the full
        # precision path. Native MXFP4 expert tensors use the QMoE names above.
        result: dict[str, torch.Tensor] = dict(native_weights)
        for name, tensor in state_dict.items():
            # Router weight/bias rename
            if "mlp.router.weight" in name:
                result[name.replace("mlp.router.weight", "mlp.gate.weight")] = tensor
            elif "mlp.router.bias" in name:
                result[name.replace("mlp.router.bias", "mlp.gate.bias")] = tensor

            # Expert fused gate+up bias: [N, 2*inter] — interleaved gate/up
            elif "mlp.experts.gate_up_proj_bias" in name:
                prefix = name.replace(".mlp.experts.gate_up_proj_bias", "")
                n_exp = tensor.shape[0]
                for i in range(n_exp):
                    result[f"{prefix}.mlp.experts.{i}.gate_proj.bias"] = tensor[
                        i, ::2
                    ].contiguous()
                    result[f"{prefix}.mlp.experts.{i}.up_proj.bias"] = tensor[
                        i, 1::2
                    ].contiguous()

            # Expert fused gate+up weight: [N, hidden, 2*inter] — transposed + interleaved
            elif name.endswith("mlp.experts.gate_up_proj"):
                prefix = name.replace(".mlp.experts.gate_up_proj", "")
                n_exp = tensor.shape[0]
                for i in range(n_exp):
                    w = tensor[i]  # [hidden, 2*inter]
                    # Gate at even columns, up at odd columns; transpose for nn.Linear format
                    result[f"{prefix}.mlp.experts.{i}.gate_proj.weight"] = w[
                        :, ::2
                    ].T.contiguous()  # [inter, hidden]
                    result[f"{prefix}.mlp.experts.{i}.up_proj.weight"] = w[
                        :, 1::2
                    ].T.contiguous()  # [inter, hidden]

            # Expert down projection bias: [N, hidden] → per-expert
            elif "mlp.experts.down_proj_bias" in name:
                prefix = name.replace(".mlp.experts.down_proj_bias", "")
                n_exp = tensor.shape[0]
                for i in range(n_exp):
                    result[f"{prefix}.mlp.experts.{i}.down_proj.bias"] = tensor[i].contiguous()

            # Expert down projection weight: [N, inter, hidden] — transposed
            elif name.endswith("mlp.experts.down_proj"):
                prefix = name.replace(".mlp.experts.down_proj", "")
                n_exp = tensor.shape[0]
                for i in range(n_exp):
                    w = tensor[i]  # [inter, hidden]
                    result[f"{prefix}.mlp.experts.{i}.down_proj.weight"] = (
                        w.T.contiguous()  # [hidden, inter]
                    )

            else:
                result[name] = tensor

        return super().preprocess_weights(result)
