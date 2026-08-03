# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Mixture of Experts (MoE) components."""

from __future__ import annotations

import dataclasses
import math
import re
from typing import TYPE_CHECKING

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._build_context import ep_capabilities
from mobius._configs import ArchitectureConfig, QuantizationConfig
from mobius._weight_utils import preprocess_awq_weights, preprocess_gptq_weights
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

    def qmoe_routing(self, op: OpBuilder, hidden_states: ir.Value):
        """Return selection and aggregation tensors for QMoE.

        Raw router logits are intentionally passed as ``router_probs`` with
        ``router_weights=None`` and ``normalize_routing_weights=1``. QMoE
        selects the top-k by raw value (monotonic activations preserve that
        selection) and, on this path, gives the selected logits softmax
        weights. This matches ``TopKGate.forward``:
        ``Softmax(TopK(router_logits))`` on both CPU and CUDA EPs.

        ORT's CPU QMoE honors Input 14 (``router_weights``) by gathering it at
        the selected experts, but CUDA QMoE ignores Input 14 and always uses
        softmax-top-k on ``router_probs``; see
        ``contrib_ops/cpu/moe/moe_quantization_cpu.cc`` and
        ``contrib_ops/cuda/moe/moe_quantization.cc``. Thus ``None`` is correct
        here on both EPs, while activated probabilities used by Softmax/Sigmoid
        gates are correct on CPU but double-activated on CUDA.
        """
        weight_t = op.Transpose(self.weight, perm=[1, 0])
        router_logits = op.MatMul(hidden_states, weight_t)
        router_logits = op.Cast(router_logits, to=ir.DataType.FLOAT.value)
        return router_logits, None, True, 1.0


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

    def qmoe_routing(self, op: OpBuilder, hidden_states: ir.Value):
        """Return selection and aggregation tensors for QMoE.

        ``router_probs`` (QMoE input 1) is the raw float32 logits and the
        pre-softmaxed probabilities are passed as ``router_weights`` (input
        14). CUDA QMoE ignores ``router_weights`` and applies softmax-top-k on
        ``router_probs``, so feeding logits reproduces ``forward`` exactly
        (``Softmax`` then top-k then renormalize); feeding pre-softmaxed probs
        here would double-softmax on CUDA. CPU QMoE selects the top-k over
        ``router_probs`` (softmax is monotonic, so logits give the same
        selection) and gathers ``router_weights`` at the selected experts,
        renormalizing when ``normalize_routing_weights=1`` -- which equals
        ``forward``'s renormalized top-k of the full softmax.
        """
        weight_t = op.Transpose(self.weight, perm=[1, 0])
        router_logits = op.MatMul(hidden_states, weight_t)
        router_logits = op.Cast(router_logits, to=ir.DataType.FLOAT.value)
        routing_probs = op.Softmax(router_logits, axis=-1)
        return router_logits, routing_probs, self.norm_topk_prob, 1.0


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

    def qmoe_routing(self, op: OpBuilder, hidden_states: ir.Value):
        """Return selection and aggregation tensors for QMoE.

        The sigmoid-activated probabilities are passed as ``router_weights``
        (QMoE input 14) while the raw float32 logits are passed as
        ``router_probs`` (input 1). CPU QMoE selects the top-k over
        ``router_probs`` (sigmoid is monotonic, so logits give the same
        selection) and gathers ``router_weights`` at the selected experts,
        renormalizing per ``normalize_routing_weights``. Passing logits (rather
        than the sigmoid probs) as ``router_probs`` avoids a softmax-of-sigmoid
        on CUDA, which ignores ``router_weights``.
        """
        weight_t = op.Transpose(self.weight, perm=[1, 0])
        router_logits = op.MatMul(hidden_states, weight_t)
        router_logits = op.Cast(router_logits, to=ir.DataType.FLOAT.value)
        routing_probs = op.Sigmoid(router_logits)
        return (
            router_logits,
            routing_probs,
            self.norm_topk_prob,
            self.routed_scaling_factor,
        )


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

    Two dispatch paths are selected at construction time:

    - Loop-over-experts (default): each expert ``MLP`` processes all
      tokens, then results are masked and weighted. Used when the model
      is unquantized or the gate has no ``qmoe_routing`` hook.
    - Fused ``com.microsoft::QMoE`` (``experts=None``): used when the
      quantization config matches the native QMoE ABI
      (:func:`_supported_qmoe_quantization`) and the gate implements
      ``qmoe_routing``. Expert weights are packed into quantized
      ``fc1``/``fc2`` parameters instead of per-expert ``MLP`` modules.

    The loop-over-experts path is the portable dense fallback representation:
    it uses only standard ONNX operators, evaluates every expert for every
    token, then masks and weights each contribution. It is a correctness oracle
    and compatibility path, not the grouped-expert performance representation.
    """

    def __init__(
        self,
        config: ArchitectureConfig,
        gate: nn.Module | None = None,
        linear_class: type | None = None,
        enable_qmoe: bool = True,
    ):
        super().__init__()
        assert config.num_local_experts is not None
        assert config.num_experts_per_tok is not None
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self._qmoe_quantization = _supported_qmoe_quantization(config.quantization)
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
        if (
            enable_qmoe
            and self._qmoe_quantization is not None
            and hasattr(self.gate, "qmoe_routing")
        ):
            self.experts = None
            self._init_qmoe_parameters(expert_config)
        else:
            self.experts = nn.ModuleList(
                [
                    MLP(expert_config, linear_class=linear_class)
                    for _ in range(self.num_experts)
                ]
            )

    def _init_qmoe_parameters(self, expert_config: ArchitectureConfig) -> None:
        quantization = self._qmoe_quantization
        assert quantization is not None
        hidden_size = expert_config.hidden_size
        intermediate_size = expert_config.intermediate_size
        block_size = quantization.group_size
        bits = quantization.bits
        fc1_out = 2 * intermediate_size
        if hidden_size % block_size or intermediate_size % block_size:
            raise ValueError(
                "QMoE dimensions must be divisible by the quantization group size"
            )

        self.fc1_experts_weights = nn.Parameter(
            [self.num_experts, fc1_out, hidden_size * bits // 8],
            dtype=ir.DataType.UINT8,
        )
        self.fc1_scales = nn.Parameter([self.num_experts, fc1_out, hidden_size // block_size])
        self.fc1_scales._keep_float32 = True
        self.fc2_experts_weights = nn.Parameter(
            [self.num_experts, hidden_size, intermediate_size * bits // 8],
            dtype=ir.DataType.UINT8,
        )
        self.fc2_scales = nn.Parameter(
            [self.num_experts, hidden_size, intermediate_size // block_size]
        )
        self.fc2_scales._keep_float32 = True
        if quantization.sym:
            self.fc1_experts_zero_points = None
            self.fc2_experts_zero_points = None
        else:
            self.fc1_experts_zero_points = nn.Parameter(
                [
                    self.num_experts,
                    fc1_out,
                    math.ceil((hidden_size // block_size) * bits / 8),
                ],
                dtype=ir.DataType.UINT8,
            )
            self.fc2_experts_zero_points = nn.Parameter(
                [
                    self.num_experts,
                    hidden_size,
                    math.ceil((intermediate_size // block_size) * bits / 8),
                ],
                dtype=ir.DataType.UINT8,
            )

    def _qmoe_forward(self, op: OpBuilder, hidden_states: ir.Value):
        quantization = self._qmoe_quantization
        assert quantization is not None
        router_probs, router_weights, normalize, output_scale = self.gate.qmoe_routing(
            op, hidden_states
        )
        result = op.QMoE(
            hidden_states,
            router_probs,
            self.fc1_experts_weights,
            self.fc1_scales,
            None,
            self.fc2_experts_weights,
            self.fc2_scales,
            None,
            None,
            None,
            None,
            self.fc1_experts_zero_points,
            self.fc2_experts_zero_points,
            None,
            router_weights,
            activation_type="swiglu",
            normalize_routing_weights=int(normalize),
            k=self.top_k,
            expert_weight_bits=quantization.bits,
            block_size=quantization.group_size,
            swiglu_fusion=2,
            _domain="com.microsoft",
        )
        if output_scale != 1.0:  # noqa: RUF069
            result = op.Mul(result, op.CastLike(output_scale, result))
        return result

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        if self.experts is None:
            return self._qmoe_forward(op, hidden_states)

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

    Symmetric quantization omits zero-point inputs and uses the kernel's implicit
    midpoint. Asymmetric quantization supplies packed per-block zero-points.
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
        if qc.sym:
            self.fc1_experts_zero_points = None
            self.fc2_experts_zero_points = None
        else:
            self.fc1_experts_zero_points = nn.Parameter(
                [
                    e,
                    fc1_out,
                    math.ceil((self._hidden // block_size) * bits / 8),
                ],
                dtype=ir.DataType.UINT8,
            )
            self.fc2_experts_zero_points = nn.Parameter(
                [
                    e,
                    self._hidden,
                    math.ceil((self._inter // block_size) * bits / 8),
                ],
                dtype=ir.DataType.UINT8,
            )

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        if ep_capabilities().name == "cuda":
            raise ValueError(
                "FusedQuantizedMoE is disabled for CUDA because ORT CUDA QMoE "
                "currently ignores router_weights (input 14), which is required "
                "for GLM/DeepSeek group-limited routing. Use the decomposed "
                "MatMulNBits export path for CUDA."
            )
        hidden = self._hidden

        # QMoE requires 2-D router_probs, so flatten [B, S, H] -> [rows, H].
        orig_shape = op.Shape(hidden_states)
        flat = op.Reshape(hidden_states, op.Constant(value_ints=[-1, hidden]))
        # Router math must be float32, but QMoE activation input stays in the
        # model dtype: CUDA QMoE kernels are registered for fp16/bf16 inputs.
        flat_f32 = op.Cast(flat, to=1)

        scores_for_choice, routing_weights, selected_experts = self._route(op, flat_f32)

        # Dense [rows, E] aggregation weights: combine weight at each selected
        # expert position, 0 elsewhere. QMoE reads these at its own top-k picks.
        zeros = op.Mul(scores_for_choice, 0.0)
        aggregation = op.ScatterElements(zeros, selected_experts, routing_weights, axis=-1)

        moe_out = op.QMoE(
            flat,  # 0: input
            scores_for_choice,  # 1: router_probs (selection logits)
            self.fc1_experts_weights,  # 2
            op.Cast(self.fc1_scales, to=1),  # 3
            None,  # 4: fc1_experts_bias
            self.fc2_experts_weights,  # 5
            op.Cast(self.fc2_scales, to=1),  # 6
            None,  # 7: fc2_experts_bias
            None,  # 8: fc3_experts_weights
            None,  # 9: fc3_scales
            None,  # 10: fc3_experts_bias
            self.fc1_experts_zero_points,  # 11
            self.fc2_experts_zero_points,  # 12
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


_PER_EXPERT_RE = re.compile(
    r"^(?P<prefix>.*\.mlp)\.experts\.(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\."
    r"(?P<kind>weight|scales|zero_points)$"
)
_FUSED_EXPERT_RE = re.compile(
    r"^(?P<prefix>.*\.mlp)\.experts\."
    r"(?P<projection>gate_up_proj|down_proj)\."
    r"(?P<kind>qweight|qzeros|g_idx|weight|scales|zero_points)$"
)


def pack_fused_quantized_moe_weights(
    state_dict: dict[str, torch.Tensor],
    config: ArchitectureConfig,
) -> dict[str, torch.Tensor]:
    """Pack routed expert tensors into the expert-major QMoE initializer ABI.

    Accepts either fused 3-D GPTQ/AWQ checkpoint tensors or already-repacked
    per-expert GGUF/MatMulNBits tensors. Gate/up rows are interleaved for
    ``swiglu_fusion=1``; shared-expert tensors are left untouched.
    """
    quantization = config.quantization
    if quantization is None:
        raise ValueError("Fused QMoE packing requires quantization settings")

    result: dict[str, torch.Tensor] = {}
    fused: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    per_expert: dict[str, dict[int, dict[str, dict[str, torch.Tensor]]]] = {}
    for key, value in state_dict.items():
        match = _FUSED_EXPERT_RE.match(key)
        if match is not None:
            fused.setdefault(match["prefix"], {}).setdefault(match["projection"], {})[
                match["kind"]
            ] = value
            continue
        match = _PER_EXPERT_RE.match(key)
        if match is not None:
            expert = int(match["expert"])
            per_expert.setdefault(match["prefix"], {}).setdefault(expert, {}).setdefault(
                match["projection"], {}
            )[match["kind"]] = value
            continue
        result[key] = value

    prefixes = set(fused) | set(per_expert)
    for prefix in prefixes:
        if prefix in fused and prefix in per_expert:
            raise ValueError(f"Mixed fused and per-expert weights under {prefix}")
        if prefix in fused:
            gate_up = _prepare_fused_projection(
                fused[prefix].get("gate_up_proj"),
                quantization.quant_method,
                quantization.bits,
                quantization.group_size,
            )
            down = _prepare_fused_projection(
                fused[prefix].get("down_proj"),
                quantization.quant_method,
                quantization.bits,
                quantization.group_size,
            )
            fc1 = {kind: _interleave_gate_up(value) for kind, value in gate_up.items()}
            fc2 = down
        else:
            experts = per_expert[prefix]
            expected = set(range(config.num_local_experts or 0))
            if set(experts) != expected:
                raise ValueError(
                    f"{prefix} has expert indices {sorted(experts)}, "
                    f"expected {sorted(expected)}"
                )
            fc1, fc2 = _stack_per_expert_projections(experts)

        _store_qmoe_projection(
            result,
            f"{prefix}.moe.fc1",
            fc1,
            symmetric=quantization.sym,
        )
        _store_qmoe_projection(
            result,
            f"{prefix}.moe.fc2",
            fc2,
            symmetric=quantization.sym,
        )
    return result


def _prepare_fused_projection(
    tensors: dict[str, torch.Tensor] | None,
    quant_method: str,
    bits: int,
    block_size: int,
) -> dict[str, torch.Tensor]:
    if tensors is None:
        raise ValueError("Missing fused routed-expert projection")
    if "qweight" in tensors:
        preprocess = (
            preprocess_gptq_weights if quant_method == "gptq" else preprocess_awq_weights
        )
        num_experts = tensors["qweight"].shape[0]
        processed: dict[str, list[torch.Tensor]] = {}
        for expert in range(num_experts):
            expert_state = {}
            for kind in {"qweight", "qzeros", "scales", "g_idx"} & tensors.keys():
                value = tensors[kind]
                expert_state[f"projection.{kind}"] = (
                    value if kind == "g_idx" and value.ndim == 1 else value[expert]
                )
            for key, value in preprocess(
                expert_state, bits=bits, group_size=block_size
            ).items():
                processed.setdefault(key.rsplit(".", 1)[-1], []).append(value)
        return {kind: torch.stack(values) for kind, values in processed.items()}
    required = {"weight", "scales"}
    if not required.issubset(tensors):
        raise ValueError(f"Missing fused projection tensors: {required - tensors.keys()}")
    return tensors


def _stack_per_expert_projections(
    experts: dict[int, dict[str, dict[str, torch.Tensor]]],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    fc1: dict[str, list[torch.Tensor]] = {}
    fc2: dict[str, list[torch.Tensor]] = {}
    for expert in sorted(experts):
        projections = experts[expert]
        for name in ("gate_proj", "up_proj", "down_proj"):
            if name not in projections:
                raise ValueError(f"Expert {expert} is missing {name}")
        kinds = set(projections["gate_proj"]) | set(projections["up_proj"])
        for kind in kinds:
            if kind not in projections["gate_proj"] or kind not in projections["up_proj"]:
                raise ValueError(f"Expert {expert} gate/up {kind} tensors are incomplete")
            gate = projections["gate_proj"][kind]
            up = projections["up_proj"][kind]
            fc1.setdefault(kind, []).append(torch.stack((gate, up), dim=1).flatten(0, 1))
        for kind, value in projections["down_proj"].items():
            fc2.setdefault(kind, []).append(value)
    return (
        {kind: torch.stack(values) for kind, values in fc1.items()},
        {kind: torch.stack(values) for kind, values in fc2.items()},
    )


def _interleave_gate_up(value: torch.Tensor) -> torch.Tensor:
    if value.shape[1] % 2:
        raise ValueError(f"gate_up projection has odd output size {value.shape[1]}")
    intermediate = value.shape[1] // 2
    return (
        value.reshape(value.shape[0], 2, intermediate, *value.shape[2:])
        .transpose(1, 2)
        .flatten(1, 2)
    )


def _store_qmoe_projection(
    result: dict[str, torch.Tensor],
    stem: str,
    tensors: dict[str, torch.Tensor],
    *,
    symmetric: bool,
) -> None:
    weight = tensors["weight"]
    if weight.ndim == 4:
        weight = weight.flatten(-2)
    result[f"{stem}_experts_weights"] = weight
    result[f"{stem}_scales"] = tensors["scales"].float()
    zero_points = tensors.get("zero_points")
    if not symmetric:
        if zero_points is None:
            raise ValueError(f"Asymmetric QMoE projection {stem} requires zero-points")
        result[f"{stem}_experts_zero_points"] = zero_points


def _supported_qmoe_quantization(
    quantization: QuantizationConfig | None,
) -> QuantizationConfig | None:
    """Return quantization settings when they match the native QMoE ABI."""
    if (
        quantization is None
        or quantization.bits != 4
        or quantization.float_zero_point
        or quantization.quant_method not in {"gptq", "awq"}
    ):
        return None
    return quantization
