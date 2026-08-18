# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Mixture of Experts (MoE) components."""

from __future__ import annotations

import dataclasses
import math

import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius._weight_utils import supported_qmoe_quantization
from mobius.components._mlp import MLP

_supported_qmoe_quantization = supported_qmoe_quantization


def _interleave_gate_up_rows(
    op: OpBuilder, tensor: ir.Value, num_experts: int, fc1_out: int
) -> ir.Value:
    """Reorder fc1 gate/up rows from HF-concatenated to QMoE-interleaved layout.

    mobius stores ``fc1_*`` parameters chunked as ``[E, 2*inter, ...]`` with
    the first ``inter`` rows holding the gate projection and the next
    ``inter`` rows holding the up projection (matching HuggingFace
    ``experts.gate_up_proj``). ``swiglu_fusion=1`` requires the interleaved
    layout ``[g_0, u_0, g_1, u_1, ...]`` instead -- it is the only layout the
    CPU QMoE kernel supports (``contrib_ops/cpu/moe/moe_quantization_cpu.cc``).
    Reshape/transpose/reshape the row dimension to reorder it; this operates
    on a constant initializer so ORT folds it away at session load (mirrors
    the unquantized-MoE interleave in ``gemma4.py``).
    """
    half = fc1_out // 2
    trailing = op.Shape(tensor, start=2)
    split_shape = op.Concat(op.Constant(value_ints=[num_experts, 2, half]), trailing, axis=0)
    reshaped = op.Reshape(tensor, split_shape)
    transposed = op.Transpose(reshaped, perm=[0, 2, 1, 3])
    merge_shape = op.Concat(op.Constant(value_ints=[num_experts, fc1_out]), trailing, axis=0)
    return op.Reshape(transposed, merge_shape)


def _flatten_to_2d(op: OpBuilder, tensor: ir.Value) -> ir.Value:
    """Flatten leading (batch, sequence, ...) dims into one, keeping the last.

    QMoE's ``router_probs`` and ``router_weights`` inputs require a strictly
    2D ``(num_tokens, num_experts)`` shape (unlike ``input``/hidden_states,
    which accepts 2D or 3D) -- see ``contrib_defs.cc``. Router logits are
    computed from a possibly-3D ``hidden_states`` (``batch, seq, hidden``),
    so collapse everything but the last dim.
    """
    last_dim = op.Shape(tensor, start=-1)
    flat_shape = op.Concat(op.Constant(value_ints=[-1]), last_dim, axis=0)
    return op.Reshape(tensor, flat_shape)


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
        # QMoE's ``router_probs`` input shares type constraint "T" with
        # ``input`` (see contrib_defs.cc), so it must match hidden_states'
        # dtype rather than a fixed float32.
        router_logits = op.CastLike(router_logits, hidden_states)
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

        ``router_probs`` (QMoE input 1) is the raw logits (cast to hidden_states's dtype) and the
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
        # QMoE's ``router_probs`` input shares type constraint "T" with
        # ``input`` (see contrib_defs.cc), so it must match hidden_states'
        # dtype rather than a fixed float32.
        router_logits = op.CastLike(router_logits, hidden_states)
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
        (QMoE input 14) while the raw logits (cast to hidden_states's dtype) are passed as
        ``router_probs`` (input 1). CPU QMoE selects the top-k over
        ``router_probs`` (sigmoid is monotonic, so logits give the same
        selection) and gathers ``router_weights`` at the selected experts,
        renormalizing per ``normalize_routing_weights``. Passing logits (rather
        than the sigmoid probs) as ``router_probs`` avoids a softmax-of-sigmoid
        on CUDA, which ignores ``router_weights``.
        """
        weight_t = op.Transpose(self.weight, perm=[1, 0])
        router_logits = op.MatMul(hidden_states, weight_t)
        # QMoE's ``router_probs`` input shares type constraint "T" with
        # ``input`` (see contrib_defs.cc), so it must match hidden_states'
        # dtype rather than a fixed float32.
        router_logits = op.CastLike(router_logits, hidden_states)
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
        if self._qmoe_quantization is not None and hasattr(self.gate, "qmoe_routing"):
            self.experts = None
            self._init_qmoe_parameters(expert_config)
        else:
            # ``linear_class`` also drives the dense loop-over-experts
            # fallback (e.g. when the quantization config doesn't match the
            # native QMoE ABI). Without threading it here, this fallback
            # path would silently lose quantization for every per-expert
            # MLP even though the caller requested a quantized linear_class.
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
        self._fc1_out = fc1_out
        if hidden_size % block_size or intermediate_size % block_size:
            raise ValueError(
                "QMoE dimensions must be divisible by the quantization group size"
            )

        self.fc1_experts_weights = nn.Parameter(
            [self.num_experts, fc1_out, hidden_size * bits // 8],
            dtype=ir.DataType.UINT8,
        )
        self.fc1_scales = nn.Parameter([self.num_experts, fc1_out, hidden_size // block_size])
        self.fc2_experts_weights = nn.Parameter(
            [self.num_experts, hidden_size, intermediate_size * bits // 8],
            dtype=ir.DataType.UINT8,
        )
        self.fc2_scales = nn.Parameter(
            [self.num_experts, hidden_size, intermediate_size // block_size]
        )
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
        # ``qmoe_routing`` is called directly (not via ``self.gate(op, ...)``)
        # because it returns multiple routing tensors rather than following
        # the standard forward() signature. ``Module.__call__`` only realizes
        # parameters (qualifying their initializer names, e.g.
        # "<prefix>.gate.weight") for the module actually invoked through
        # ``__call__`` -- see onnxscript/nn/_module.py -- so bypassing it here
        # would leave ``self.gate.weight`` registered as a bare, unqualified
        # "weight" initializer. Manually push the gate's module scope and
        # realize its parameters first, mirroring the pattern used for the
        # sparse LM head in gemma4_assistant.py.
        op.builder.push_module(self.gate.name or "gate", type(self.gate).__qualname__)
        try:
            # ``recurse=False``: mirror ``Module.__call__``'s direct-only
            # realization. Gate classes are currently leaf modules with no
            # child modules, but recursing here would double-register any
            # nested module's parameters under this pushed scope if one were
            # ever added.
            for param in self.gate.parameters(recurse=False):
                param._realize(op.builder)  # pylint: disable=protected-access
            router_probs, router_weights, normalize, output_scale = self.gate.qmoe_routing(
                op, hidden_states
            )
        finally:
            op.builder.pop_module()

        router_probs = _flatten_to_2d(op, router_probs)
        if router_weights is not None:
            router_weights = _flatten_to_2d(op, router_weights)

        fc1_experts_weights = _interleave_gate_up_rows(
            op, self.fc1_experts_weights, self.num_experts, self._fc1_out
        )
        fc1_scales = _interleave_gate_up_rows(
            op, self.fc1_scales, self.num_experts, self._fc1_out
        )
        fc1_experts_zero_points = (
            _interleave_gate_up_rows(
                op, self.fc1_experts_zero_points, self.num_experts, self._fc1_out
            )
            if self.fc1_experts_zero_points is not None
            else None
        )
        result = op.QMoE(
            hidden_states,
            router_probs,
            fc1_experts_weights,
            fc1_scales,
            None,
            self.fc2_experts_weights,
            self.fc2_scales,
            None,
            None,
            None,
            None,
            fc1_experts_zero_points,
            self.fc2_experts_zero_points,
            None,
            router_weights,
            activation_type="swiglu",
            normalize_routing_weights=int(normalize),
            k=self.top_k,
            expert_weight_bits=quantization.bits,
            block_size=quantization.group_size,
            swiglu_fusion=1,
            quant_type="int",
            weights_prepacked=0,
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
