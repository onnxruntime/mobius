# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Mixture of Experts (MoE) components."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components._mlp import MLP

if TYPE_CHECKING:
    pass


class TopKGate(nn.Module):
    """Standard top-k expert routing gate.

    Selects top-k experts by logit value and normalizes routing weights
    with softmax over the selected experts.
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.weight = nn.Parameter([num_experts, hidden_size])

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        weight_t = op.Transpose(self.weight, perm=[1, 0])
        router_logits = op.MatMul(hidden_states, weight_t)
        k = op.Constant(value_ints=[self.top_k])
        routing_weights, selected_experts = op.TopK(router_logits, k, axis=-1, _outputs=2)
        routing_weights = op.Softmax(routing_weights, axis=-1)
        return routing_weights, selected_experts


class SoftmaxTopKGate(nn.Module):
    """Softmax-first top-k expert routing gate (Qwen3-Next style).

    Applies softmax over all expert logits first, then selects top-k.
    Optionally renormalizes the selected weights to sum to 1.
    """

    def __init__(
        self, hidden_size: int, num_experts: int, top_k: int, *, norm_topk_prob: bool = True
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.weight = nn.Parameter([num_experts, hidden_size])

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        weight_t = op.Transpose(self.weight, perm=[1, 0])
        router_logits = op.MatMul(hidden_states, weight_t)
        # Softmax over all experts first
        routing_probs = op.Softmax(router_logits, axis=-1)
        k = op.Constant(value_ints=[self.top_k])
        routing_weights, selected_experts = op.TopK(routing_probs, k, axis=-1, _outputs=2)
        if self.norm_topk_prob:
            # Renormalize selected weights to sum to 1
            weight_sum = op.ReduceSum(routing_weights, [-1], keepdims=True)
            routing_weights = op.Div(routing_weights, weight_sum)
        return routing_weights, selected_experts


class SigmoidTopKGate(nn.Module):
    """Sigmoid-first top-k expert routing gate (GLM4-MoE style).

    Applies element-wise sigmoid over all expert logits, selects top-k,
    and optionally renormalizes the selected weights to sum to 1.
    Used by GLM4-MoE where group routing collapses to standard top-k
    when n_group=1 (all experts in one group).
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        *,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.weight = nn.Parameter([num_experts, hidden_size])

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        weight_t = op.Transpose(self.weight, perm=[1, 0])
        router_logits = op.MatMul(hidden_states, weight_t)
        # Sigmoid instead of softmax: each expert scored independently
        routing_probs = op.Sigmoid(router_logits)
        k = op.Constant(value_ints=[self.top_k])
        routing_weights, selected_experts = op.TopK(routing_probs, k, axis=-1, _outputs=2)
        if self.norm_topk_prob:
            # Renormalize selected weights to sum to 1 (prevents vanishing gradients)
            weight_sum = op.ReduceSum(routing_weights, [-1], keepdims=True)
            routing_weights = op.Div(routing_weights, op.Add(weight_sum, 1e-9))
        if self.routed_scaling_factor != 1.0:  # noqa: RUF069
            routing_weights = op.Mul(routing_weights, self.routed_scaling_factor)
        return routing_weights, selected_experts


class SparseMixerGate(nn.Module):
    """Sparsemixer-style routing gate (used by PhiMoE).

    Implements the inference-mode routing from HuggingFace PhiMoE:
    experts are selected sequentially with a threshold mask that filters
    out experts whose logits are relatively far from the maximum. For
    each round, softmax is computed over the non-masked experts, and the
    weight of the selected expert is its softmax probability.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        jitter_eps: float = 0.01,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_eps = jitter_eps
        self.weight = nn.Parameter([num_experts, hidden_size])

    def _threshold_mask_and_select(self, op, scores, jitter_eps):
        """Apply threshold mask and select the top expert."""
        max_score = op.ReduceMax(scores, [-1], keepdims=True)
        abs_scores = op.Abs(scores)
        factor = op.Max(abs_scores, max_score)
        diff = op.Sub(max_score, scores)
        ratio = op.Div(diff, factor)
        threshold = 2.0 * jitter_eps
        mask = op.Greater(ratio, threshold)
        # op.CastLike with Python literal: reuses a single constant, avoids cache-key
        # collision that would occur if -1e30 were used as a plain literal in both
        # op.Where (auto-cast to typed constant) and op.Expand (unbound → FLOAT).
        neg_inf = op.CastLike(-1e30, scores)
        masked_scores = op.Where(mask, neg_inf, scores)
        weights = op.Softmax(masked_scores, axis=-1)
        k_one = op.Constant(value_ints=[1])
        _top_val, expert_idx = op.TopK(masked_scores, k_one, axis=-1, _outputs=2)
        expert_weight = op.GatherElements(weights, expert_idx, axis=-1)
        return expert_weight, expert_idx

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        weight_t = op.Transpose(self.weight, perm=[1, 0])
        router_logits = op.MatMul(hidden_states, weight_t)

        all_weights = []
        all_experts = []
        current_scores = router_logits

        for _k in range(self.top_k):
            weight_k, expert_k = self._threshold_mask_and_select(
                op, current_scores, self.jitter_eps
            )
            all_weights.append(weight_k)
            all_experts.append(expert_k)
            neg_inf = op.CastLike(-1e30, current_scores)
            current_scores = op.ScatterElements(
                current_scores,
                expert_k,
                op.Expand(neg_inf, op.Shape(expert_k)),
                axis=-1,
            )

        routing_weights = op.Concat(*all_weights, axis=-1)
        selected_experts = op.Concat(*all_experts, axis=-1)
        return routing_weights, selected_experts


class MoELayer(nn.Module):
    """Mixture of Experts layer.

    Routes each token to top-k experts via a gating mechanism, applies
    each expert MLP, and accumulates weighted results.

    Uses loop-over-experts dispatch: each expert processes all tokens,
    then results are masked and weighted.
    """

    def __init__(
        self,
        config: ArchitectureConfig,
        gate: nn.Module | None = None,
        linear_class: type | None = None,
    ):
        super().__init__()
        assert config.num_local_experts is not None
        assert config.num_experts_per_tok is not None
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        if gate is not None:
            self.gate = gate
        else:
            self.gate = TopKGate(config.hidden_size, self.num_experts, self.top_k)
        # Use moe_intermediate_size for experts when specified (Qwen2-MoE, Qwen3-MoE).
        expert_config = (
            dataclasses.replace(config, intermediate_size=config.moe_intermediate_size)
            if config.moe_intermediate_size is not None
            else config
        )
        self.experts = nn.ModuleList(
            [MLP(expert_config, linear_class=linear_class) for _ in range(self.num_experts)]
        )

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        routing_weights, selected_experts = self.gate(op, hidden_states)

        result = None
        for expert_idx, expert in enumerate(self.experts):
            expert_output = expert(op, hidden_states)
            expert_id = op.Constant(value_int=expert_idx)
            match = op.Equal(selected_experts, expert_id)
            match_float = op.CastLike(match, routing_weights)
            weighted = op.Mul(routing_weights, match_float)
            weight = op.ReduceSum(weighted, [-1], keepdims=True)
            contribution = op.Mul(expert_output, weight)
            if result is None:
                result = contribution
            else:
                result = op.Add(result, contribution)

        return result


class FusedQuantizedMoE(nn.Module):
    """Routed MoE experts emitted as a single fused ``com.microsoft::QMoE`` op.

    Replaces the per-expert ``MatMulNBits`` unroll (:class:`MoELayer`) for
    weight-only int-quantized MoE. All routed experts are packed into
    expert-major integer weight tensors laid out exactly as the ORT contrib
    ``QMoE`` kernel expects::

        fc1_experts_weights: [E, 2*inter, hidden // pack_size]   uint8
        fc1_scales:          [E, 2*inter, hidden // block_size]  float32
        fc2_experts_weights: [E, hidden, inter // pack_size]     uint8
        fc2_scales:          [E, hidden, inter // block_size]    float32

    where ``pack_size = 8 // bits``. ``fc1`` fuses the SwiGLU gate/up
    projections in the **interleaved** layout ``[g_0, u_0, g_1, u_1, ...]`` and is
    consumed with ``swiglu_fusion=1`` (the only SwiGLU layout the ORT CPU QMoE
    kernel supports), so the kernel computes ``silu(gate) * up`` — matching
    GLM/DeepSeek's SiLU-gated experts (``activation_alpha=1``, ``activation_beta=0``,
    ``swiglu_limit=inf``, all ORT defaults).

    Routing is delegated to ``gate.route_for_qmoe`` (GLM sigmoid + noaux_tc /
    DeepSeek softmax group-limited). The kernel re-derives top-k selection from
    ``router_probs`` (the gate's ``scores_for_choice``); the exact combine
    weights are scattered into a dense ``[rows, E]`` aggregation tensor and fed
    through the optional ``router_weights`` input with
    ``normalize_routing_weights=0``, so the fused op reproduces the per-expert
    path's routing bit-for-bit.

    Symmetric int quantization is used (no zero-points): the kernel defaults the
    per-block zero-point to ``1 << (bits - 1)``.
    """

    _MICROSOFT_DOMAIN = "com.microsoft"

    def __init__(
        self,
        config: ArchitectureConfig,
        gate: nn.Module,
    ):
        super().__init__()
        assert config.num_local_experts is not None
        assert config.num_experts_per_tok is not None
        assert config.moe_intermediate_size is not None
        assert config.quantization is not None
        if not hasattr(gate, "route_for_qmoe"):
            raise TypeError(
                f"gate {type(gate).__name__} does not support the fused QMoE path "
                "(missing route_for_qmoe); use MoELayer instead"
            )

        qc = config.quantization
        bits = qc.bits
        block_size = qc.group_size
        if bits not in (1, 2, 4, 8):
            raise ValueError(f"QMoE expert_weight_bits must be 1/2/4/8, got {bits}")
        if block_size < 16 or (block_size & (block_size - 1)):
            raise ValueError(f"QMoE block_size must be a power of 2 >= 16, got {block_size}")

        self.gate = gate
        self._num_experts = config.num_local_experts
        self._top_k = config.num_experts_per_tok
        self._hidden = config.hidden_size
        self._inter = config.moe_intermediate_size
        self._bits = bits
        self._block_size = block_size

        if self._hidden % block_size != 0:
            raise ValueError(
                f"hidden_size {self._hidden} must be divisible by block_size {block_size}"
            )
        if self._inter % block_size != 0:
            raise ValueError(
                f"moe_intermediate_size {self._inter} must be divisible by "
                f"block_size {block_size}"
            )

        pack_size = 8 // bits
        e = self._num_experts
        fc1_out = 2 * self._inter  # interleaved [g_0, u_0, ...] for swiglu_fusion=1
        # fc1: [E, 2*inter, hidden] quantized along hidden (K)
        self.fc1_experts_weights = nn.Parameter(
            [e, fc1_out, self._hidden // pack_size],
            dtype=ir.DataType.UINT8,
        )
        self.fc1_scales = nn.Parameter(
            [e, fc1_out, self._hidden // block_size],
            dtype=ir.DataType.FLOAT,
        )
        # fc2: [E, hidden, inter] quantized along inter (K)
        self.fc2_experts_weights = nn.Parameter(
            [e, self._hidden, self._inter // pack_size],
            dtype=ir.DataType.UINT8,
        )
        self.fc2_scales = nn.Parameter(
            [e, self._hidden, self._inter // block_size],
            dtype=ir.DataType.FLOAT,
        )

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        hidden = self._hidden

        # QMoE requires 2-D router_probs, so flatten [B, S, H] -> [rows, H].
        orig_shape = op.Shape(hidden_states)
        flat = op.Reshape(hidden_states, op.Constant(value_ints=[-1, hidden]))
        # QMoE input/router_probs must be float32.
        flat_f32 = op.Cast(flat, to=1)

        scores_for_choice, routing_weights, selected_experts = self._route(op, flat_f32)

        # Dense [rows, E] aggregation weights: combine weight at each selected
        # expert position, 0 elsewhere. QMoE reads these at its own top-k picks.
        zeros = op.Mul(scores_for_choice, 0.0)
        aggregation = op.ScatterElements(
            zeros, selected_experts, routing_weights, axis=-1
        )

        moe_out = op.QMoE(
            flat_f32,  # 0: input
            scores_for_choice,  # 1: router_probs (selection logits)
            self.fc1_experts_weights,  # 2
            self.fc1_scales,  # 3
            None,  # 4: fc1_experts_bias
            self.fc2_experts_weights,  # 5
            self.fc2_scales,  # 6
            None,  # 7: fc2_experts_bias
            None,  # 8: fc3_experts_weights
            None,  # 9: fc3_scales
            None,  # 10: fc3_experts_bias
            None,  # 11: fc1_zero_points
            None,  # 12: fc2_zero_points
            None,  # 13: fc3_zero_points
            aggregation,  # 14: router_weights (explicit combine weights)
            activation_type="swiglu",
            k=self._top_k,
            normalize_routing_weights=0,
            swiglu_fusion=1,
            expert_weight_bits=self._bits,
            block_size=self._block_size,
            quant_type="int",
            _domain=self._MICROSOFT_DOMAIN,
        )
        moe_out = op.CastLike(moe_out, hidden_states)
        return op.Reshape(moe_out, orig_shape)

    def _route(self, op: OpBuilder, hidden_states: ir.Value):
        """Invoke ``gate.route_for_qmoe`` under the gate's module scope.

        ``route_for_qmoe`` is a plain method (not ``forward``), so it never goes
        through :meth:`nn.Module.__call__`. We replicate the parameter-realization
        step here so the gate's ``weight`` / ``e_score_correction_bias`` are
        registered as graph initializers under the ``...moe.gate`` scope (matching
        the per-expert path) instead of dangling as unqualified names.
        """
        builder = op.builder
        module_name = self.gate._name or "gate"
        class_name = type(self.gate).__qualname__
        builder.push_module(module_name, class_name)
        try:
            for param in self.gate._parameters.values():
                param._realize(builder)
            return self.gate.route_for_qmoe(op, hidden_states)
        finally:
            builder.pop_module()
