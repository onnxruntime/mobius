# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""DeepSeek-V4 export with native compressed sparse attention and MTP sidecar.

The released V4 architecture replaces V3 MLA with compressed sparse attention
and adds Hyper-Connections. This module implements the V4 projections,
Hyper-Connections, hash/sqrt-softplus MoE routing, and optional native
``CompressedSparseAttention``. The official per-layer compression schedule is
represented by live compressor/indexer projections, while non-native exports
retain the sink-aware dense fallback. The checkpoint's MTP block is exported as
a standalone sidecar.
"""

from __future__ import annotations

import dataclasses
import logging
import math

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._build_context import ep_capabilities, get_build_dtype
from mobius._configs import ArchitectureConfig
from mobius._weight_utils import (
    pack_qmoe_expert_weights,
    stack_per_expert_moe_weights,
    supported_qmoe_quantization,
)
from mobius.components import (
    Embedding,
    Linear,
    MoELayer,
    PlanarBlockQuantizedLinear,
    QuantizedEmbedding,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
    make_quantized_linear_factory,
)
from mobius.components._moe import _scatter_selected_to_full
from mobius.components._rotary_embedding import apply_rotary_pos_emb
from mobius.models._deepseek_v4_csa import (
    CsaLayerPlan,
    NativeCsaExportError,
    assert_native_runtime_supports_block_quant,
    emit_csa_attention,
    plan_native_csa,
)
from mobius.models.base import CausalLMModel

logger = logging.getLogger(__name__)


def _cast_to_f32(op: OpBuilder, value: ir.Value, model_dtype: ir.DataType) -> ir.Value:
    """Cast a model-dtype activation to FLOAT for the f32-only CSA op inputs.

    No-op when the build dtype is already FLOAT so the common f32 export
    stays Cast-free (and byte-identical to the pre-CSA graph on that path).
    """
    if model_dtype == ir.DataType.FLOAT:
        return value
    return op.Cast(value, to=ir.DataType.FLOAT)


def _use_fused_gqa() -> bool:
    """Whether the active EP/build dtype supports direct ``GroupQueryAttention`` emission.

    Mirrors ``TextModel.forward``'s ``use_gqa`` gate (``mobius/models/base.py``):
    the active EP must declare GQA support for the current build dtype via
    :func:`~mobius._build_context.ep_capabilities`. Unlike that generic path,
    V4 always applies RoPE explicitly in-graph (``do_rotary=0``), so this
    doesn't also need to check ``caps.supports_fused_rope`` -- V4's fused path
    never asks ``GroupQueryAttention`` to rotate internally.

    EPs such as ``"default"``, ``"onnx-standard"``, ``"qnn"``, and
    ``"openvino"`` declare an empty ``gqa_dtypes`` specifically so their
    exported graphs stay free of ``com.microsoft`` ops; V4 must fall back to
    the portable decomposed attention path for those, exactly like every
    other model in this codebase.
    """
    caps = ep_capabilities()
    return get_build_dtype() in caps.gqa_dtypes


def _gqa_kv_lengths(op: OpBuilder, attention_mask: ir.Value) -> tuple[ir.Value, ir.Value]:
    """Derive ``GroupQueryAttention``'s required ``seqlens_k``/``total_sequence_length``.

    Same formula ``TextModel`` uses to build a ``GQAContext``
    (``mobius/models/base.py``): ``seqlens_k[b] = sum(attention_mask[b]) - 1``
    (last valid KV index per batch entry) and
    ``total_seq_len = attention_mask.shape[1]`` (past + current length).
    Computed directly here (rather than via ``GQAContext``, which also
    bundles ``cos_cache``/``sin_cache`` for the fused-rotary ``do_rotary=1``
    path that V4 does not use -- RoPE is applied explicitly in-graph).
    """
    one_i32 = op.Constant(value_int=1)
    seqlens_k = op.Cast(
        op.Sub(op.ReduceSum(attention_mask, [1], keepdims=0), one_i32),
        to=ir.DataType.INT32,
    )
    total_seq_len = op.Cast(
        op.Gather(op.Shape(attention_mask), 1),
        to=ir.DataType.INT32,
    )
    return seqlens_k, total_seq_len


def _projection_class(config: ArchitectureConfig):
    scheme = config.block_quant_scheme
    if scheme is not None:
        if not scheme.is_block_scaled_fp8 or len(scheme.weight_block_size) != 2:
            raise NativeCsaExportError(
                "native DeepSeek-V4 block-quant projections require block-scaled "
                f"FP8 with a 2-D block geometry; got {scheme}"
            )
        block_out, block_in = scheme.weight_block_size

        def planar_projection(in_features: int, out_features: int, bias: bool = False):
            return PlanarBlockQuantizedLinear(
                in_features,
                out_features,
                format="block_fp8",
                block_size_out=block_out,
                block_size_in=block_in,
                model_dtype=config.dtype,
                bias=bias,
            )

        return planar_projection
    quantization = config.quantization
    if quantization is None or quantization.quant_method == "none":
        return Linear
    zero_point_dtype = config.dtype if quantization.float_zero_point else ir.DataType.UINT8
    return make_quantized_linear_factory(
        bits=quantization.bits,
        block_size=quantization.group_size,
        has_zero_point=not quantization.sym,
        zero_point_dtype=zero_point_dtype,
    )


def _shape_anchor(op: OpBuilder, parameters: list[ir.Value]) -> ir.Value:
    """Reference one element of each deferred-runtime tensor and produce zero."""
    total = None
    for parameter in parameters:
        first = op.Gather(op.Reshape(parameter, [-1]), [0])
        present = op.Cast(
            op.Equal(first, first),
            to=ir.DataType.INT64,
        )
        total = present if total is None else op.Add(total, present)
    assert total is not None
    return op.ReduceSum(op.Mul(total, 0), [0], keepdims=False)


class DeepSeekV4DeferredProjection(nn.Module):
    """Projection parameters exported for a runtime path not yet executed."""

    def __init__(
        self,
        config: ArchitectureConfig,
        in_features: int,
        out_features: int,
    ):
        super().__init__()
        quantization = config.quantization
        self._gguf_quantized_linear = (
            quantization is not None and quantization.quant_method != "none"
        )
        if quantization is None or quantization.quant_method == "none":
            self.weight = nn.Parameter([out_features, in_features])
            self.scales = None
            self.zero_points = None
            return

        n_blocks = (in_features + quantization.group_size - 1) // quantization.group_size
        blob_size = quantization.group_size * quantization.bits // 8
        self.weight = nn.Parameter(
            [out_features, n_blocks, blob_size],
            dtype=ir.DataType.UINT8,
        )
        self.scales = nn.Parameter([out_features, n_blocks])
        if quantization.sym:
            self.zero_points = None
        elif quantization.float_zero_point:
            self.zero_points = nn.Parameter([out_features, n_blocks], dtype=config.dtype)
        else:
            packed_zero_points = (n_blocks * quantization.bits + 7) // 8
            self.zero_points = nn.Parameter(
                [out_features, packed_zero_points],
                dtype=ir.DataType.UINT8,
            )

    def forward(self, op: OpBuilder, x: ir.Value | None = None) -> ir.Value:
        """Zero-valued shape anchor (dense fallback) or the real projection.

        With ``x is None`` (the deferred dense-fallback path) this emits a
        zero-valued shape anchor that keeps the projection weight live in the
        graph without executing it. With ``x`` supplied (the native-CSA
        dataflow) it computes the real ``x @ weight.T`` projection so the
        compressor activations feeding ``pkg.nxrt::CompressedSparseAttention``
        are genuine.

        Both paths run through ``forward`` (invoked via ``Module.__call__``) on
        purpose: ``__call__`` realizes this module's parameters -- qualifies
        each name and registers it as a graph initializer -- *before* calling
        ``forward``. A bare helper method that used ``self.weight`` directly
        would leave the weight unrealized (dangling, unregistered) and break
        graph serialization. Only the unquantized weight layout is supported
        for the real projection; a quantized compressor weight raises so
        native CSA never silently drops the quantization.
        """
        if x is None:
            return _shape_anchor(
                op,
                [
                    value
                    for value in (self.weight, self.scales, self.zero_points)
                    if value is not None
                ],
            )
        if self._gguf_quantized_linear:
            raise NativeCsaExportError(
                "native CSA C1 cannot project a quantized compressor weight; "
                "the frozen op consumes f32 compressor activations. Quantized "
                "compressor dequant is a follow-up slice"
            )
        return op.MatMul(x, op.Transpose(self.weight, perm=[1, 0]))


class DeepSeekV4DeferredNorm(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter([hidden_size])

    def forward(self, op: OpBuilder, *, raw: bool = False) -> ir.Value:
        """Zero-valued shape anchor (dense fallback) or the realized weight.

        ``__call__`` realizes ``self.weight`` before ``forward`` runs, so with
        ``raw=True`` (native-CSA dataflow) the already-registered weight is
        returned directly for the op's ``compressor_norm`` input; otherwise a
        zero-valued shape anchor keeps the weight live without executing it.
        """
        if raw:
            return self.weight
        return _shape_anchor(op, [self.weight])


def _validate_hash_routing_tables(state_dict: dict[str, torch.Tensor]) -> None:
    """Fail fast if a hash-routing table names a duplicate expert for a token.

    ``_scatter_selected_to_full`` (used by ``DeepSeekV4Gate.qmoe_routing`` to
    drive QMoE) requires ``top_k`` *distinct* experts per token:
    ``ScatterElements`` overwrites rather than accumulates duplicate indices,
    so a repeated expert would silently drop one of its contributions instead
    of raising an error. Real hash tables have no reason to route a token to
    the same expert twice, but this checks the actual checkpoint data rather
    than relying on that assumption alone.
    """
    for key, table in state_dict.items():
        if not key.endswith(".mlp.moe.gate.tid2eid") or table.shape[-1] <= 1:
            continue
        sorted_table, _ = torch.sort(table, dim=-1)
        duplicate_rows = torch.any(sorted_table[..., 1:] == sorted_table[..., :-1], dim=-1)
        if duplicate_rows.any():
            bad_tokens = torch.nonzero(duplicate_rows, as_tuple=False).flatten()[:5].tolist()
            raise ValueError(
                f"{key} routes token id(s) {bad_tokens} (showing up to 5 of "
                f"{int(duplicate_rows.sum())}) to a duplicate expert; QMoE "
                "export requires top_k distinct experts per token (see "
                "mobius.components._moe._scatter_selected_to_full)."
            )


def _pack_planar_expert_weights(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Stack per-expert FP4 weight/scale planes into canonical nxrt banks."""
    grouped: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    passthrough: dict[str, torch.Tensor] = {}
    marker = ".mlp.experts."
    for key, value in state_dict.items():
        if marker not in key:
            passthrough[key] = value
            continue
        prefix, suffix = key.split(marker, 1)
        parts = suffix.split(".")
        if (
            len(parts) != 3
            or parts[1]
            not in {
                "gate_proj",
                "up_proj",
                "down_proj",
            }
            or parts[2] not in {"weight", "scale"}
        ):
            raise NativeCsaExportError(f"malformed planar routed-expert tensor name {key!r}")
        try:
            expert = int(parts[0])
        except ValueError as exc:
            raise NativeCsaExportError(f"invalid routed-expert index in {key!r}") from exc
        grouped.setdefault(prefix, {}).setdefault(expert, {})[f"{parts[1]}.{parts[2]}"] = value

    required = {
        "gate_proj.weight",
        "gate_proj.scale",
        "up_proj.weight",
        "up_proj.scale",
        "down_proj.weight",
        "down_proj.scale",
    }
    for prefix, experts in grouped.items():
        ids = sorted(experts)
        if ids != list(range(len(ids))):
            raise NativeCsaExportError(
                f"{prefix} planar expert ids must be contiguous 0..E-1, got {ids}"
            )
        for expert, tensors in experts.items():
            missing = sorted(required - set(tensors))
            extra = sorted(set(tensors) - required)
            if missing or extra:
                raise NativeCsaExportError(
                    f"{prefix} expert {expert} malformed planar tensor set: "
                    f"missing={missing}, extra={extra}"
                )
        passthrough[f"{prefix}.mlp.moe.fc1_experts_weights"] = torch.stack(
            [experts[i]["gate_proj.weight"] for i in ids]
        )
        passthrough[f"{prefix}.mlp.moe.fc2_experts_weights"] = torch.stack(
            [experts[i]["down_proj.weight"] for i in ids]
        )
        passthrough[f"{prefix}.mlp.moe.fc3_experts_weights"] = torch.stack(
            [experts[i]["up_proj.weight"] for i in ids]
        )
        passthrough[f"{prefix}.mlp.moe.fc1_experts_aux_scale"] = torch.stack(
            [experts[i]["gate_proj.scale"] for i in ids]
        )
        passthrough[f"{prefix}.mlp.moe.fc2_experts_aux_scale"] = torch.stack(
            [experts[i]["down_proj.scale"] for i in ids]
        )
        passthrough[f"{prefix}.mlp.moe.fc3_experts_aux_scale"] = torch.stack(
            [experts[i]["up_proj.scale"] for i in ids]
        )
    return passthrough


def _map_checkpoint_weight_name(key: str) -> str:
    """Map one official DeepSeek-V4 tensor name to the exported module name."""
    new_key = key
    if new_key == "embed.weight":
        return "model.embed_tokens.weight"
    if new_key == "head.weight":
        return "lm_head.weight"
    if new_key == "norm.weight":
        return "model.norm.weight"
    if new_key.startswith("hc_head_"):
        return f"model.{new_key}.weight" if new_key == "hc_head_fn" else f"model.{new_key}"
    if new_key.startswith("layers."):
        new_key = f"model.{new_key}"
    if new_key.startswith(("model.layers.", "mtp.")):
        new_key = new_key.replace(".attn.wq_a.", ".self_attn.q_a_proj.")
        new_key = new_key.replace(".attn.q_norm.", ".self_attn.q_a_layernorm.")
        new_key = new_key.replace(".attn.wq_b.", ".self_attn.q_b_proj.")
        new_key = new_key.replace(".attn.wkv.", ".self_attn.kv_proj.")
        new_key = new_key.replace(".attn.kv_norm.", ".self_attn.kv_layernorm.")
        new_key = new_key.replace(".attn.wo_a.", ".self_attn.o_a_proj.")
        new_key = new_key.replace(".attn.wo_b.", ".self_attn.o_b_proj.")
        new_key = new_key.replace(".attn.", ".self_attn.")
        new_key = new_key.replace(".attn_norm.", ".input_layernorm.")
        new_key = new_key.replace(".ffn_norm.", ".post_attention_layernorm.")
        new_key = new_key.replace(".ffn.gate.", ".mlp.moe.gate.")
        new_key = new_key.replace(".ffn.experts.", ".mlp.experts.")
        new_key = new_key.replace(".ffn.shared_experts.", ".mlp.shared_experts.")
        new_key = new_key.replace(".w1.", ".gate_proj.")
        new_key = new_key.replace(".w2.", ".down_proj.")
        new_key = new_key.replace(".w3.", ".up_proj.")
        if ".hc_" in new_key and new_key.endswith("_fn"):
            new_key = f"{new_key}.weight"
    return new_key


class DeepSeekV4Gate(nn.Module):
    """V4 sqrt-softplus router with hash routing for the first layers."""

    def __init__(self, config: ArchitectureConfig, layer_id: int):
        super().__init__()
        assert config.num_local_experts is not None
        assert config.num_experts_per_tok is not None
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.route_scale = config.routed_scaling_factor
        self.score_func = config.scoring_func
        self.hash_routing = layer_id < config.num_hash_layers
        self.weight = nn.Parameter([self.num_experts, config.hidden_size])
        # Hash-routed layers do not consume this bias, but DeepSeek V4 GGUF
        # checkpoints provide it for every layer.
        self.bias = nn.Parameter([self.num_experts])
        if self.hash_routing:
            self.tid2eid = nn.Parameter(
                [config.vocab_size, self.top_k], dtype=ir.DataType.INT32
            )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        input_ids: ir.Value,
    ):
        logits = op.MatMul(
            op.Cast(hidden_states, to=ir.DataType.FLOAT.value),
            op.Transpose(self.weight, perm=[1, 0]),
        )
        if self.score_func == "softmax":
            scores = op.Softmax(logits, axis=-1)
        elif self.score_func == "sigmoid":
            scores = op.Sigmoid(logits)
        else:
            scores = op.Sqrt(op.Softplus(logits))

        if self.hash_routing:
            selected_experts = op.Cast(
                op.Gather(self.tid2eid, input_ids, axis=0),
                to=ir.DataType.INT64.value,
            )
        else:
            choice_scores = op.Add(scores, self.bias)
            _, selected_experts = op.TopK(
                choice_scores,
                op.Constant(value_ints=[self.top_k]),
                axis=-1,
                _outputs=2,
            )

        routing_weights = op.GatherElements(scores, selected_experts, axis=-1)
        if self.score_func != "softmax":
            weight_sum = op.ReduceSum(routing_weights, [-1], keepdims=True)
            routing_weights = op.Div(routing_weights, op.Add(weight_sum, 1e-20))
        if self.route_scale != 1.0:  # noqa: RUF069
            routing_weights = op.Mul(routing_weights, self.route_scale)
        return routing_weights, selected_experts

    def qmoe_routing(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        input_ids: ir.Value,
    ):
        """Adapt the already-selected (routing_weights, selected_experts) pair for QMoE.

        This does not alter their computation.

        Hash routing (``tid2eid``) is not expressible as "top-k of a
        per-expert score" (QMoE's only selection ABI), so this reuses
        ``forward()`` verbatim -- covering both hash and learned top-k
        routing identically -- and scatters its output into the
        full-``num_experts``-width tensors QMoE requires. See
        ``_scatter_selected_to_full`` for why this preserves the exact
        selection and weights, and for why -- like V3's ``DeepSeekMoEGate``
        -- this path is CPU-EP-correct only: CUDA QMoE ignores the gathered
        ``router_weights`` this adapter relies on (learned layers select via
        ``scores + bias`` but weight by ``scores`` alone, so raw-logit
        passthrough can't make CUDA's forced internal recompute agree either).
        """
        routing_weights, selected_experts = self.forward(op, hidden_states, input_ids)
        # QMoE's router_probs/router_weights share type constraint "T" with
        # hidden_states; routing_weights is computed in float32 (matching the
        # reference sqrt-softplus/softmax/sigmoid scoring), so cast back
        # before scattering.
        routing_weights = op.CastLike(routing_weights, hidden_states)
        router_probs, router_weights = _scatter_selected_to_full(
            op, routing_weights, selected_experts, self.num_experts
        )
        # route_scale (and, for non-softmax scoring, weight_sum renormalization)
        # is already folded into routing_weights above, so QMoE must not
        # renormalize or rescale again.
        return router_probs, router_weights, False, 1.0


class _DeepSeekV4Expert(nn.Module):
    def __init__(self, config: ArchitectureConfig, intermediate_size: int):
        super().__init__()
        projection = _projection_class(config)
        self.gate_proj = projection(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = projection(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = projection(intermediate_size, config.hidden_size, bias=False)
        self.limit = config.swiglu_limit

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        gate = self.gate_proj(op, hidden_states)
        up = self.up_proj(op, hidden_states)
        if self.limit > 0:
            gate = op.Clip(gate, None, self.limit)
            up = op.Clip(up, -self.limit, self.limit)
        return self.down_proj(op, op.Mul(op.Swish(gate), up))


class _DeepSeekV4PlanarMoE(nn.Module):
    """Sparse routed experts using the canonical 12-input nxrt v1 ABI."""

    def __init__(self, config: ArchitectureConfig, gate: DeepSeekV4Gate):
        super().__init__()
        assert config.num_local_experts is not None
        assert config.num_experts_per_tok is not None
        assert config.moe_intermediate_size is not None
        scheme = config.block_quant_scheme
        if scheme is None or not scheme.has_packed_fp4_experts:
            raise NativeCsaExportError(
                "planar DeepSeek-V4 MoE requires packed FP4 routed experts"
            )
        self.gate = gate
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.model_dtype = config.dtype
        self.swiglu_limit = config.swiglu_limit

        packed_hidden = config.hidden_size // 2
        packed_intermediate = config.moe_intermediate_size // 2
        scale_hidden = config.hidden_size // 32
        scale_intermediate = config.moe_intermediate_size // 32
        if config.hidden_size % 32 or config.moe_intermediate_size % 32:
            raise NativeCsaExportError(
                "fp4_planar routed experts require hidden/intermediate widths "
                "divisible by 32; got "
                f"{config.hidden_size}/{config.moe_intermediate_size}"
            )
        self.fc1_experts_weights = nn.Parameter(
            [self.num_experts, self.intermediate_size, packed_hidden],
            dtype=ir.DataType.INT8,
        )
        self.fc2_experts_weights = nn.Parameter(
            [self.num_experts, self.hidden_size, packed_intermediate],
            dtype=ir.DataType.INT8,
        )
        self.fc3_experts_weights = nn.Parameter(
            [self.num_experts, self.intermediate_size, packed_hidden],
            dtype=ir.DataType.INT8,
        )
        self.fc1_experts_aux_scale = nn.Parameter(
            [self.num_experts, self.intermediate_size, scale_hidden],
            dtype=ir.DataType.FLOAT8E8M0,
        )
        self.fc2_experts_aux_scale = nn.Parameter(
            [self.num_experts, self.hidden_size, scale_intermediate],
            dtype=ir.DataType.FLOAT8E8M0,
        )
        self.fc3_experts_aux_scale = nn.Parameter(
            [self.num_experts, self.intermediate_size, scale_hidden],
            dtype=ir.DataType.FLOAT8E8M0,
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        input_ids: ir.Value,
    ) -> ir.Value:
        routing_weights, selected_experts = self.gate(op, hidden_states, input_ids)
        routing_weights = op.CastLike(routing_weights, hidden_states)
        router_logits, router_weights = _scatter_selected_to_full(
            op, routing_weights, selected_experts, self.num_experts
        )
        router_logits = op.Cast(
            op.Reshape(router_logits, [-1, self.num_experts]),
            to=ir.DataType.FLOAT,
        )
        router_weights = op.Cast(
            op.Reshape(router_weights, [-1, self.num_experts]),
            to=ir.DataType.FLOAT,
        )
        activation = (
            hidden_states
            if hidden_states.dtype == ir.DataType.FLOAT
            else op.Cast(hidden_states, to=ir.DataType.FLOAT)
        )
        op.builder.graph.opset_imports["pkg.nxrt"] = 1
        result = op.BlockQuantizedMoE(
            activation,
            router_logits,
            self.fc1_experts_weights,
            None,
            self.fc2_experts_weights,
            None,
            self.fc3_experts_weights,
            None,
            router_weights,
            self.fc1_experts_aux_scale,
            self.fc2_experts_aux_scale,
            self.fc3_experts_aux_scale,
            k=self.top_k,
            activation_type="swiglu",
            normalize_routing_weights=0,
            swiglu_fusion=0,
            swiglu_limit=(self.swiglu_limit if self.swiglu_limit > 0 else float("inf")),
            fc1_format="fp4_planar",
            fc2_format="fp4_planar",
            fc3_format="fp4_planar",
            fc1_block_size_out=1,
            fc1_block_size_in=32,
            fc2_block_size_out=1,
            fc2_block_size_in=32,
            fc3_block_size_out=1,
            fc3_block_size_in=32,
            block_layout_version=1,
            _domain="pkg.nxrt",
        )
        result.dtype = ir.DataType.FLOAT
        result.shape = hidden_states.shape
        if self.model_dtype != ir.DataType.FLOAT:
            result = op.Cast(result, to=self.model_dtype)
            result.dtype = self.model_dtype
            result.shape = hidden_states.shape
        return result


class DeepSeekV4MoE(nn.Module):
    def __init__(self, config: ArchitectureConfig, layer_id: int):
        super().__init__()
        assert config.num_local_experts is not None
        assert config.moe_intermediate_size is not None
        gate = DeepSeekV4Gate(config, layer_id)
        # QMoE's clipped-SwiGLU maps exactly onto _DeepSeekV4Expert's
        # activation (plain SiLU: alpha=1.0, beta=0.0). config.swiglu_limit<=0
        # means "no clipping" in _DeepSeekV4Expert.forward, but QMoE treats
        # swiglu_limit=0.0 as "clip to zero" -- math.inf is required to
        # disable clipping at the op level.
        swiglu_limit = config.swiglu_limit if config.swiglu_limit > 0 else math.inf
        if config.block_quant_scheme is not None:
            self.moe = _DeepSeekV4PlanarMoE(config, gate)
        else:
            self.moe = MoELayer(
                config,
                gate=gate,
                expert_factory=lambda expert_config, _linear_class: _DeepSeekV4Expert(
                    expert_config, expert_config.intermediate_size
                ),
                activation_alpha=1.0,
                activation_beta=0.0,
                swiglu_limit=swiglu_limit,
            )
        shared_size = config.moe_intermediate_size * (config.n_shared_experts or 1)
        self.shared_experts = _DeepSeekV4Expert(config, shared_size)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        input_ids: ir.Value,
    ):
        moe_output = self.moe(op, hidden_states, input_ids)
        return op.Add(moe_output, self.shared_experts(op, hidden_states))


class DeepSeekV4CompressorTensors(nn.Module):
    """Official learned compressor tensors retained for sparse-runtime handoff."""

    def __init__(self, config: ArchitectureConfig, compress_ratio: int, head_dim: int):
        super().__init__()
        overlap_factor = 2 if compress_ratio == 4 else 1
        self.ape = nn.Parameter([compress_ratio, overlap_factor * head_dim])
        self.wkv = DeepSeekV4DeferredProjection(
            config, config.hidden_size, overlap_factor * head_dim
        )
        self.wgate = DeepSeekV4DeferredProjection(
            config, config.hidden_size, overlap_factor * head_dim
        )
        self.norm = DeepSeekV4DeferredNorm(head_dim)

    def forward(self, op: OpBuilder, hidden_states: ir.Value | None = None):
        """Zero-valued shape anchor (dense fallback) or live compressor tensors.

        With ``hidden_states is None`` (deferred dense fallback) this emits the
        zero-valued shape anchor that keeps ``ape``/``wkv``/``wgate``/``norm``
        live in the graph. With ``hidden_states`` supplied (native-CSA
        dataflow) it returns ``(compressor_kv, compressor_gate, ape,
        norm_weight)``: ``compressor_kv``/``compressor_gate`` are the projected
        ``W·x`` activations (``[B, S, overlap*head_dim]``) and ``ape``/the norm
        weight are the learned parameters. Every parameter is realized here by
        routing through each child's ``__call__`` (``self.wkv(op, ...)``,
        ``self.norm(op, raw=True)``) and by ``__call__`` realizing ``self.ape``
        before ``forward`` runs.
        """
        if hidden_states is None:
            anchor = _shape_anchor(
                op,
                [
                    self.ape,
                ],
            )
            anchor = op.Add(anchor, self.wkv(op))
            anchor = op.Add(anchor, self.wgate(op))
            return op.Add(anchor, self.norm(op))
        compressor_kv = self.wkv(op, hidden_states)
        compressor_gate = self.wgate(op, hidden_states)
        return compressor_kv, compressor_gate, self.ape, self.norm(op, raw=True)


class DeepSeekV4IndexerTensors(nn.Module):
    """Official ratio-4 sparse indexer tensors retained in the dense graph."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        assert config.q_lora_rank is not None
        assert config.index_n_heads is not None
        assert config.index_head_dim is not None
        self.index_n_heads = config.index_n_heads
        self.index_head_dim = config.index_head_dim
        if config.block_quant_scheme is not None:
            self.wq_b = _projection_class(config)(
                config.q_lora_rank,
                config.index_n_heads * config.index_head_dim,
                bias=False,
            )
        else:
            self.wq_b = DeepSeekV4DeferredProjection(
                config,
                config.q_lora_rank,
                config.index_n_heads * config.index_head_dim,
            )
        self.weights_proj = DeepSeekV4DeferredProjection(
            config, config.hidden_size, config.index_n_heads
        )
        self.compressor = DeepSeekV4CompressorTensors(
            config, compress_ratio=4, head_dim=config.index_head_dim
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value | None = None,
        query_lora: ir.Value | None = None,
    ):
        """Zero-valued shape anchor (dense fallback) or live indexer tensors.

        With ``hidden_states``/``query_lora`` both ``None`` (deferred dense
        fallback) this emits the zero-valued shape anchor that keeps the
        indexer weights live. With both supplied (native-CSA ratio-4 dataflow)
        it returns the six frozen-op index inputs:
        ``(index_query, index_weight, index_compressor_kv,
        index_compressor_gate, index_compressor_ape, index_compressor_norm)``.

        ``index_query`` is the *raw* ``wq_b(query_lora)`` projection reshaped to
        ``[B, S, index_n_heads, index_head_dim]`` -- the frozen op applies the
        compressed RoPE + Hadamard + FP4 round-trip internally
        (``finalize_index_query``), so no rotation happens here. ``index_weight``
        is the raw ``weights_proj(hidden_states)`` projection
        ``[B, S, index_n_heads]``; the index compressor mirrors the attention
        compressor (raw ``W·x`` activations + learned ``ape``/norm).
        """
        if hidden_states is None and query_lora is None:
            own = op.Add(self.wq_b(op), self.weights_proj(op))
            return op.Add(own, self.compressor(op))
        if hidden_states is None or query_lora is None:
            raise NativeCsaExportError(
                "native CSA ratio-4 indexer requires both hidden_states and "
                "query_lora for its live dataflow; got "
                f"hidden_states={'set' if hidden_states is not None else 'None'}, "
                f"query_lora={'set' if query_lora is not None else 'None'}"
            )
        index_query = op.Reshape(
            self.wq_b(op, query_lora),
            [0, 0, self.index_n_heads, self.index_head_dim],
        )
        index_weight = self.weights_proj(op, hidden_states)
        (
            index_compressor_kv,
            index_compressor_gate,
            index_ape,
            index_norm,
        ) = self.compressor(op, hidden_states)
        return (
            index_query,
            index_weight,
            index_compressor_kv,
            index_compressor_gate,
            index_ape,
            index_norm,
        )


class DeepSeekV4Attention(nn.Module):
    """V4 MQA projections using sink-aware dense causal attention."""

    def __init__(self, config: ArchitectureConfig, layer_id: int, *, is_mtp: bool = False):
        super().__init__()
        assert config.q_lora_rank is not None
        assert config.qk_rope_head_dim is not None
        assert config.o_lora_rank is not None
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.nope_dim = self.head_dim - self.rope_dim
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        assert self.num_heads % self.o_groups == 0
        projection = _projection_class(config)

        self.q_a_proj = projection(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = projection(
            config.q_lora_rank,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.kv_proj = projection(config.hidden_size, self.head_dim, bias=False)
        self.kv_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        group_width = self.num_heads * self.head_dim // self.o_groups
        self.o_a_proj = projection(
            group_width,
            self.o_groups * self.o_lora_rank,
            bias=False,
        )
        self.o_b_proj = projection(
            self.o_groups * self.o_lora_rank,
            config.hidden_size,
            bias=False,
        )
        self.eps = config.rms_norm_eps
        self.scale = self.head_dim**-0.5
        self.rope_interleave = config.rope_interleave
        self._dtype = config.dtype
        # Official reference (`inference/model.py::Attention.forward`,
        # `get_window_topk_idxs`) unconditionally restricts every layer -
        # regardless of `compress_ratio` - to a circular-buffer local window
        # of `config.sliding_window` (DeepSeek-V4-Flash: 128) most-recent
        # positions; ratio>0 layers additionally union in compressed/indexed
        # positions (not modeled here, see `DeepSeekV4CompressorTensors`/
        # `DeepSeekV4IndexerTensors`), but the window restriction itself
        # applies to every layer, not just ratio-0. `-1` is GQA's documented
        # sentinel for "no local window" (matches `_gqa_local_window_size` in
        # `models/base.py`), used only if the config genuinely has no window.
        sliding_window = config.sliding_window
        self.local_window_size = (
            int(sliding_window) if sliding_window and sliding_window > 0 else -1
        )
        ratios = config.compress_ratios or []
        self.compress_ratio = ratios[layer_id] if layer_id < len(ratios) else 0
        if self.compress_ratio not in (0, 4, 128):
            raise ValueError(
                "DeepSeek-V4 supports compression ratios 0, 4, and 128; "
                f"layer {layer_id} requested {self.compress_ratio}"
            )
        self.attn_sink = nn.Parameter([self.num_heads], dtype=ir.DataType.FLOAT)
        self.compressor = (
            DeepSeekV4CompressorTensors(config, self.compress_ratio, self.head_dim)
            if self.compress_ratio
            else None
        )
        self.indexer = DeepSeekV4IndexerTensors(config) if self.compress_ratio == 4 else None
        # Property gate (default off). Resolves to a CsaLayerPlan for a
        # ratio-128 (HCA) or ratio-4 (CSA) layer whose config matches the
        # frozen op contract, and raises NativeCsaExportError (never silent
        # dense) for MTP/unknown-ratio/unsupported-quant when native CSA is
        # requested. See mobius.models._deepseek_v4_csa.
        self.csa_plan: CsaLayerPlan | None = plan_native_csa(config, layer_id, is_mtp=is_mtp)

    def _rotate(
        self,
        op: OpBuilder,
        value: ir.Value,
        position_embeddings: tuple,
        num_heads: int,
        *,
        inverse: bool = False,
    ):
        value = op.Reshape(value, [0, 0, num_heads, self.head_dim])
        nope, rope = op.Split(value, [self.nope_dim, self.rope_dim], axis=-1, _outputs=2)
        rope = op.Reshape(rope, [0, 0, -1])
        if inverse:
            position_embeddings = (position_embeddings[0], op.Neg(position_embeddings[1]))
        rope = apply_rotary_pos_emb(
            op,
            rope,
            position_embeddings,
            num_heads=num_heads,
            rotary_embedding_dim=0,
            interleaved=self.rope_interleave,
        )
        rope = op.Reshape(rope, [0, 0, num_heads, self.rope_dim])
        return op.Reshape(op.Concat(nope, rope, axis=-1), [0, 0, -1])

    def _expand_kv(
        self,
        op: OpBuilder,
        value: ir.Value,
        batch: ir.Value,
        sequence_length: ir.Value,
    ) -> ir.Value:
        """Broadcast the single MQA key/value head across all query heads.

        Only used by the portable decomposed path below (EPs whose
        ``gqa_dtypes`` is empty). The fused ``GroupQueryAttention`` path does
        this broadcast internally via its own ``kv_num_heads=1`` handling, so
        it never calls this helper.
        """
        value = op.Unsqueeze(value, [2])
        value = op.Expand(
            value,
            op.Concat(
                batch,
                [1, self.num_heads],
                sequence_length,
                [self.head_dim],
                axis=0,
            ),
        )
        return op.Reshape(
            value,
            op.Concat(
                batch,
                [self.num_heads],
                sequence_length,
                [self.head_dim],
                axis=0,
            ),
        )

    def _forward_native_csa(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        query: ir.Value,
        query_lora: ir.Value,
        kv: ir.Value,
        past_key_value: tuple | None,
        past_compressed: tuple | None,
        csa_lengths: tuple | None,
    ) -> tuple[ir.Value, ir.Value, ir.Value, tuple[ir.Value, ...]]:
        """Emit the frozen CSA/HCA op for this layer (native-CSA path).

        Handles both ratios the schedule interleaves: ratio-128 (HCA,
        compressor-only, 11-in/3-out) and ratio-4 (CSA, compressor + learned
        FP4 indexer, 19-in/6-out). ``query``/``kv`` arrive already RoPE-rotated
        (compressed rope theta) by the caller, exactly as the dense paths
        receive them; the op has no cos/sin/position inputs and only rotates
        its *compressed records* (and, for ratio-4, its index query/records)
        internally, so the dense-window ``query``/``current_kv`` must be
        pre-rotated here and the shared inverse ``_rotate`` still runs on the
        op's ``Y`` after this returns. Builds:

        * ``query`` as rank-4 BSND ``[B, S, num_heads, head_dim]``;
        * ``current_kv`` ``[B, K, head_dim]`` -- the dense sliding-window ring
          reuses the existing dense KV IO (``past_key_values.{i}`` ->
          ``present.{i}``), stored BNSD ``[B, 1, K, head_dim]`` (MQA, one kv
          head) like the decomposed path, then squeezed for the op;
        * real, *unrotated* compressor activations (the op pools then rotates
          the compressed records internally at their block positions);
        * for ratio-4, the raw learned-indexer activations (``index_query`` from
          ``wq_b(query_lora)``, ``index_weight``, and the index compressor) --
          the op applies the index RoPE/Hadamard/FP4 internally;
        * the threaded ``past_* -> present_*`` compressed (and, for ratio-4,
          index) state.

        Returns ``(output, present_key, present_value, present_compressed)``
        with ``output`` flattened ``[B, S, num_heads*head_dim]`` in the rotated
        frame so the caller's shared inverse ``_rotate`` + output projection
        apply unchanged. ``present_compressed`` is the ratio-shaped state tuple:
        ``(present_compressed_kv, present_compression_carry)`` for ratio-128, or
        ``(present_compressed_kv, present_compression_carry, present_index_key,
        present_index_carry, selected_indices)`` for ratio-4.
        """
        plan = self.csa_plan
        assert plan is not None
        if csa_lengths is None:
            raise NativeCsaExportError(
                f"native CSA layer {plan.layer_id} requires seqlens_k/"
                "total_sequence_length; csa_lengths was not threaded to the "
                "attention layer"
            )
        if past_compressed is None:
            raise NativeCsaExportError(
                f"native CSA layer {plan.layer_id} requires past_compressed_kv/"
                "past_compression_carry state; none was threaded to the layer"
            )
        seqlens_k, total_seq_len = csa_lengths
        expected_state = 4 if plan.is_ratio4 else 2
        if len(past_compressed) != expected_state:
            raise NativeCsaExportError(
                f"native CSA layer {plan.layer_id} (ratio "
                f"{plan.compression_ratio}) requires {expected_state} threaded "
                f"past-state tensors, got {len(past_compressed)}"
            )

        query_4d = op.Reshape(query, [0, 0, self.num_heads, self.head_dim])

        # Dense sliding-window ring: present KV stored BNSD [B, 1, K, head_dim]
        # (one MQA kv head) exactly like the decomposed path -- key and value
        # carry identical data (MQA) but must be *distinct* graph values so the
        # present.{i}.key/value cache outputs don't collapse onto one tensor.
        # ``current_kv`` squeezes the head axis to [B, K, head_dim] for the op.
        new_kv = op.Transpose(op.Reshape(kv, [0, 0, 1, self.head_dim]), perm=[0, 2, 1, 3])
        if past_key_value is not None:
            present_key = op.Concat(past_key_value[0], new_kv, axis=2)
            present_value = op.Concat(past_key_value[1], new_kv, axis=2)
        else:
            present_key = new_kv
            present_value = op.Identity(new_kv)
        current_kv = op.Reshape(present_key, [0, -1, self.head_dim])

        # Real, unrotated compressor activations replace the zero-valued shape
        # anchor: wkv/wgate are live W·x projections, ape/norm are the learned
        # parameters. Routed through the compressor's ``__call__`` so every
        # compressor parameter is realized (named + registered as an
        # initializer) rather than left dangling.
        assert self.compressor is not None
        compressor_kv, compressor_gate, ape, norm_weight = self.compressor(op, hidden_states)

        present_compressed: tuple
        common = dict(
            query=_cast_to_f32(op, query_4d, self._dtype),
            current_kv=_cast_to_f32(op, current_kv, self._dtype),
            compressor_kv=_cast_to_f32(op, compressor_kv, self._dtype),
            compressor_gate=_cast_to_f32(op, compressor_gate, self._dtype),
            compressor_ape=_cast_to_f32(op, ape, self._dtype),
            compressor_norm=_cast_to_f32(op, norm_weight, self._dtype),
            past_compressed_kv=past_compressed[0],
            past_compression_carry=past_compressed[1],
            seqlens_k=seqlens_k,
            total_sequence_length=op.Cast(total_seq_len, to=ir.DataType.INT64),
            # ``head_sink`` is declared FLOAT but ``_cast_module_dtype`` folds
            # it to the model dtype in a non-float build; the frozen op's
            # ``head_sink`` is f32, so cast it back up when needed.
            head_sink=_cast_to_f32(op, self.attn_sink, self._dtype),
        )

        if plan.is_ratio4:
            assert self.indexer is not None
            # Raw learned-indexer activations (op rotates/packs internally).
            (
                index_query,
                index_weight,
                index_compressor_kv,
                index_compressor_gate,
                index_ape,
                index_norm,
            ) = self.indexer(op, hidden_states, query_lora)
            # ``past_index_key`` is packed uint8 state -- passed through as-is;
            # the index carry and all index activations are f32.
            outputs = emit_csa_attention(
                op,
                plan,
                index_query=_cast_to_f32(op, index_query, self._dtype),
                index_weight=_cast_to_f32(op, index_weight, self._dtype),
                index_compressor_kv=_cast_to_f32(op, index_compressor_kv, self._dtype),
                index_compressor_gate=_cast_to_f32(op, index_compressor_gate, self._dtype),
                index_compressor_ape=_cast_to_f32(op, index_ape, self._dtype),
                index_compressor_norm=_cast_to_f32(op, index_norm, self._dtype),
                past_index_key=past_compressed[2],
                past_index_carry=past_compressed[3],
                **common,
            )
            (
                y,
                present_compressed_kv,
                present_compression_carry,
                present_index_key,
                present_index_carry,
                selected_indices,
            ) = outputs
            present_compressed = (
                present_compressed_kv,
                present_compression_carry,
                present_index_key,
                present_index_carry,
                selected_indices,
            )
        else:
            y, present_compressed_kv, present_compression_carry = emit_csa_attention(
                op, plan, **common
            )
            present_compressed = (present_compressed_kv, present_compression_carry)

        # ``Y`` is [B, S, N, D] FLOAT; flatten to [B, S, N*D] and cast back to
        # the model dtype for the shared inverse-RoPE + output projection.
        output = op.Reshape(y, [0, 0, -1])
        if self._dtype != ir.DataType.FLOAT:
            output = op.Cast(output, to=self._dtype)
        return (
            output,
            present_key,
            present_value,
            present_compressed,
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None = None,
        attention_bias: ir.Value | None = None,
        seqlens_k: ir.Value | None = None,
        total_seq_len: ir.Value | None = None,
        past_compressed: tuple | None = None,
        csa_lengths: tuple | None = None,
    ):
        query_lora = self.q_a_layernorm(op, self.q_a_proj(op, hidden_states))
        query = self.q_b_proj(op, query_lora)
        query_4d = op.Reshape(query, [0, 0, self.num_heads, self.head_dim])
        query_rms = op.Sqrt(
            op.Add(
                op.ReduceMean(op.Mul(query_4d, query_4d), [-1], keepdims=True),
                self.eps,
            )
        )
        query = op.Reshape(op.Div(query_4d, query_rms), [0, 0, -1])
        query = self._rotate(op, query, position_embeddings, self.num_heads)

        kv = self.kv_layernorm(op, self.kv_proj(op, hidden_states))
        kv = self._rotate(op, kv, position_embeddings, 1)

        present_compressed = None
        if self.csa_plan is not None:
            # Native CSA/HCA path (default-off ``config.native_csa`` opt-in).
            # Emits the frozen ``pkg.nxrt::CompressedSparseAttention`` op
            # (ratio-128 HCA or ratio-4 CSA) instead of the dense correctness
            # fallback; the compressor/indexer tensors are consumed as real
            # dataflow here rather than as a zero-valued shape anchor below.
            # ``output`` comes back flattened in the rotated frame, so the
            # shared inverse ``_rotate`` and the output projection run unchanged.
            output, present_key, present_value, present_compressed = self._forward_native_csa(
                op,
                hidden_states,
                query,
                query_lora,
                kv,
                past_key_value,
                past_compressed,
                csa_lengths,
            )
        elif seqlens_k is not None:
            # Fused causal attention core, used only when the active EP's
            # `ep_capabilities().gqa_dtypes` includes the build dtype (see
            # `_use_fused_gqa`). `query`/`kv` are already RoPE-rotated
            # (trailing rope slice only, applied explicitly above via
            # `_rotate`, the same convention V2/V3 MLA uses with the plain
            # `Attention` op), so `GroupQueryAttention` only needs
            # `do_rotary=0` (its default): cos_cache/sin_cache/position_ids
            # stay unset. V4 is MQA (`kv_num_heads=1`): key and value are
            # literally the same rotated `kv` tensor -- broadcasting
            # `kv_num_heads=1` across all query heads is the fused op's own
            # job, so `_expand_kv` isn't needed on this path. No explicit
            # `attention_bias` is passed: causal + padding is carried by
            # `seqlens_k`/`total_seq_len` (no KV-sharing across layers, no
            # per-layer dual head_dim), same as the "direct GQA" convention
            # `Attention._forward_gqa` (mobius/components/_attention.py)
            # already uses for every other simple-causal model in this
            # codebase (see the `attention-optimization` skill: "Use Contrib
            # GQA when you don't need attention bias"). The official
            # reference (`inference/model.py::Attention.forward`,
            # `get_window_topk_idxs`) unconditionally restricts every layer
            # to the most recent `local_window_size` positions -- this is
            # NOT optional/ratio-dependent -- so `local_window_size` is
            # passed as a first-class GQA attribute (`-1` sentinel when the
            # config declares no window) instead of baking it into an
            # explicit float bias as the decomposed path below does via
            # `create_attention_bias(sliding_window=...)`. `head_sink`
            # carries the learned per-head attention sink exactly as the
            # decomposed path's `Concat(scores, sinks) -> Softmax -> slice`
            # sequence does, now as a first-class op input instead of a
            # manually broadcast Concat.
            past_key = past_key_value[0] if past_key_value is not None else None
            past_value = past_key_value[1] if past_key_value is not None else None
            gqa_attrs: dict = {
                "num_heads": self.num_heads,
                "kv_num_heads": 1,
                "scale": self.scale,
            }
            # Match the shared `Attention._forward_gqa` convention (see
            # `mobius/components/_attention.py`): only emit the attribute
            # when a window is actually configured, so a model with no
            # window (local_window_size == -1) keeps a graph identical to
            # before this attribute existed.
            if self.local_window_size > 0:
                gqa_attrs["local_window_size"] = self.local_window_size
            output, present_key, present_value = op.GroupQueryAttention(
                query,
                kv,
                kv,
                past_key,
                past_value,
                seqlens_k,
                total_seq_len,
                None,  # cos_cache: unused, RoPE already applied above
                None,  # sin_cache: unused, RoPE already applied above
                None,  # position_ids: unused, do_rotary=0
                None,  # attention_bias: unused, plain causal + seqlens_k suffices
                op.CastLike(self.attn_sink, query),  # head_sink
                _domain="com.microsoft",
                _outputs=3,
                **gqa_attrs,
            )
        else:
            # Portable decomposed path: used whenever the active EP declares
            # no GQA support for the build dtype (e.g. `"default"`,
            # `"onnx-standard"`, `"qnn"`, `"openvino"` -- see
            # `_use_fused_gqa`). Manually replicates the exact same
            # sink-aware dense causal attention using only standard ONNX
            # ops, so these EPs' exported graphs stay free of
            # `com.microsoft` custom ops.
            batch = op.Shape(query, start=0, end=1)
            query_length = op.Shape(query, start=1, end=2)
            query = op.Transpose(
                op.Reshape(query, [0, 0, self.num_heads, self.head_dim]),
                perm=[0, 2, 1, 3],
            )
            key = op.Transpose(op.Reshape(kv, [0, 0, 1, self.head_dim]), perm=[0, 2, 1, 3])
            value = key
            if past_key_value is not None:
                key = op.Concat(past_key_value[0], key, axis=2)
                value = op.Concat(past_key_value[1], value, axis=2)
            present_key, present_value = key, value

            kv_length = op.Shape(key, start=2, end=3)
            key = self._expand_kv(op, key, batch, kv_length)
            value = self._expand_kv(op, value, batch, kv_length)
            scores = op.Mul(
                op.MatMul(query, op.Transpose(key, perm=[0, 1, 3, 2])),
                self.scale,
            )
            scores = op.Add(scores, attention_bias)
            sinks = op.Expand(
                op.Reshape(
                    op.CastLike(self.attn_sink, scores),
                    [1, self.num_heads, 1, 1],
                ),
                op.Concat(batch, [self.num_heads], query_length, [1], axis=0),
            )
            probabilities = op.Softmax(op.Concat(scores, sinks, axis=-1), axis=-1)
            probabilities = op.Slice(probabilities, [0], [-1], [3])
            output = op.Reshape(
                op.Transpose(op.MatMul(probabilities, value), perm=[0, 2, 1, 3]),
                [0, 0, -1],
            )

        output = self._rotate(op, output, position_embeddings, self.num_heads, inverse=True)

        if self.compressor is not None and self.csa_plan is None:
            # Zero-valued shape anchor for the dense correctness fallback,
            # keeping the deferred compressor/indexer weights live in the
            # graph. Skipped on the native-CSA path, where those same weights
            # are consumed as real dataflow inside ``_forward_native_csa`` --
            # so a native-CSA layer emits no dead anchor arithmetic.
            anchor = self.compressor(op)
            if self.indexer is not None:
                anchor = op.Add(anchor, self.indexer(op))
            output = op.Add(output, op.CastLike(anchor, output))

        group_width = self.num_heads * self.head_dim // self.o_groups
        groups = op.Split(
            output,
            [group_width] * self.o_groups,
            axis=-1,
            _outputs=self.o_groups,
        )
        projected_groups = []
        for group_idx, group in enumerate(groups):
            projected = self.o_a_proj(op, group)
            projected_groups.append(
                op.Slice(
                    projected,
                    [group_idx * self.o_lora_rank],
                    [(group_idx + 1) * self.o_lora_rank],
                    [-1],
                )
            )
        output = self.o_b_proj(op, op.Concat(*projected_groups, axis=-1))
        return output, (present_key, present_value), present_compressed


class DeepSeekV4DecoderLayer(nn.Module):
    def __init__(self, config: ArchitectureConfig, layer_id: int, *, is_mtp: bool = False):
        super().__init__()
        self.self_attn = DeepSeekV4Attention(config, layer_id, is_mtp=is_mtp)
        self.mlp = DeepSeekV4MoE(config, layer_id)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_mult = config.hc_mult
        self.hc_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        hc_dim = config.hc_mult * config.hidden_size
        mix_dim = (2 + config.hc_mult) * config.hc_mult
        self.hc_attn_fn = Linear(hc_dim, mix_dim, bias=False)
        self.hc_attn_base = nn.Parameter([mix_dim])
        self.hc_attn_scale = nn.Parameter([3])
        self.hc_ffn_fn = Linear(hc_dim, mix_dim, bias=False)
        self.hc_ffn_base = nn.Parameter([mix_dim])
        self.hc_ffn_scale = nn.Parameter([3])

    def _hc_pre(self, op, states, fn, scale, base):
        flat = op.Reshape(states, [0, 0, -1])
        rms = op.Sqrt(op.Add(op.ReduceMean(op.Mul(flat, flat), [-1], keepdims=True), 1e-6))
        mixes = op.Div(fn(op, flat), rms)
        pre_raw, post_raw, comb_raw = op.Split(
            mixes,
            [self.hc_mult, self.hc_mult, self.hc_mult * self.hc_mult],
            axis=-1,
            _outputs=3,
        )
        base_pre, base_post, base_comb = op.Split(
            base,
            [self.hc_mult, self.hc_mult, self.hc_mult * self.hc_mult],
            axis=-1,
            _outputs=3,
        )
        scale_pre, scale_post, scale_comb = op.Split(scale, [1, 1, 1], axis=-1, _outputs=3)
        pre = op.Add(
            op.Sigmoid(op.Add(op.Mul(pre_raw, scale_pre), base_pre)),
            self.hc_eps,
        )
        post = op.Mul(op.Sigmoid(op.Add(op.Mul(post_raw, scale_post), base_post)), 2.0)
        comb = op.Reshape(
            op.Add(op.Mul(comb_raw, scale_comb), base_comb),
            [0, 0, self.hc_mult, self.hc_mult],
        )
        comb = op.Add(op.Softmax(comb, axis=-1), self.hc_eps)
        comb = op.Div(
            comb,
            op.Add(op.ReduceSum(comb, [-2], keepdims=True), self.hc_eps),
        )
        for _ in range(max(self.hc_iters - 1, 0)):
            comb = op.Div(
                comb,
                op.Add(op.ReduceSum(comb, [-1], keepdims=True), self.hc_eps),
            )
            comb = op.Div(
                comb,
                op.Add(op.ReduceSum(comb, [-2], keepdims=True), self.hc_eps),
            )
        reduced = op.ReduceSum(op.Mul(op.Unsqueeze(pre, [-1]), states), [2], keepdims=False)
        return reduced, post, comb

    @staticmethod
    def _hc_post(op, value, residual, post, comb):
        injected = op.Mul(op.Unsqueeze(post, [-1]), op.Unsqueeze(value, [-2]))
        mixed = op.ReduceSum(
            op.Mul(op.Unsqueeze(comb, [-1]), op.Unsqueeze(residual, [-2])),
            [2],
            keepdims=False,
        )
        return op.Add(injected, mixed)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        input_ids: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None = None,
        attention_bias: ir.Value | None = None,
        seqlens_k: ir.Value | None = None,
        total_seq_len: ir.Value | None = None,
        past_compressed: tuple | None = None,
        csa_lengths: tuple | None = None,
    ):
        residual = hidden_states
        value, post, comb = self._hc_pre(
            op,
            hidden_states,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
        )
        value, present, present_compressed = self.self_attn(
            op,
            self.input_layernorm(op, value),
            position_embeddings,
            past_key_value,
            attention_bias,
            seqlens_k,
            total_seq_len,
            past_compressed,
            csa_lengths,
        )
        hidden_states = self._hc_post(op, value, residual, post, comb)

        residual = hidden_states
        value, post, comb = self._hc_pre(
            op,
            hidden_states,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
        )
        value = self.mlp(op, self.post_attention_layernorm(op, value), input_ids)
        hidden_states = self._hc_post(op, value, residual, post, comb)
        return hidden_states, present, present_compressed


class DeepSeekV4TextModel(nn.Module):
    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        if config.quantization is not None and config.quantization.quantize_embeddings:
            self.embed_tokens = QuantizedEmbedding(
                config.vocab_size,
                config.hidden_size,
                bits=config.quantization.bits,
                block_size=config.quantization.group_size,
                has_zero_point=not config.quantization.sym,
                padding_idx=config.pad_token_id,
            )
        self.layers = nn.ModuleList(
            [
                DeepSeekV4DecoderLayer(config, layer_id)
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        hc_dim = config.hc_mult * config.hidden_size
        self.hc_head_fn = Linear(hc_dim, config.hc_mult, bias=False)
        self.hc_head_base = nn.Parameter([config.hc_mult])
        self.hc_head_scale = nn.Parameter([1])
        rope_config = config
        if config.qk_rope_head_dim is not None:
            rope_config = dataclasses.replace(config, head_dim=config.qk_rope_head_dim)
        self.rotary_emb = initialize_rope(
            dataclasses.replace(
                rope_config,
                rope_type="default",
                rope_scaling=None,
                original_max_position_embeddings=None,
            )
        )
        self.compressed_rotary_emb = initialize_rope(
            dataclasses.replace(
                rope_config,
                rope_theta=config.compress_rope_theta or config.rope_theta,
            )
        )
        self._dtype = config.dtype
        self.native_csa = config.native_csa
        # See `DeepSeekV4Attention.local_window_size`: the reference always
        # restricts attention to this window, so the decomposed path's
        # shared bias (built once here for every layer) must bake it in too.
        self.sliding_window = config.sliding_window

    def _hc_head(self, op, states):
        flat = op.Reshape(states, [0, 0, -1])
        rms = op.Sqrt(op.Add(op.ReduceMean(op.Mul(flat, flat), [-1], keepdims=True), 1e-6))
        mixes = op.Div(self.hc_head_fn(op, flat), rms)
        weights = op.Add(
            op.Sigmoid(
                op.Add(
                    op.Mul(mixes, self.hc_head_scale),
                    self.hc_head_base,
                )
            ),
            self.hc_eps,
        )
        return op.ReduceSum(op.Mul(op.Unsqueeze(weights, [-1]), states), [2], keepdims=False)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        inputs_embeds: ir.Value | None = None,
        past_compressed_states: list | None = None,
    ):
        hidden_states = (
            inputs_embeds if inputs_embeds is not None else self.embed_tokens(op, input_ids)
        )
        hidden_states = op.Expand(
            op.Unsqueeze(hidden_states, [-2]),
            [1, 1, self.hc_mult, 1],
        )
        position_embeddings = self.rotary_emb(op, position_ids)
        compressed_position_embeddings = self.compressed_rotary_emb(op, position_ids)
        if _use_fused_gqa():
            attention_bias = None
            seqlens_k, total_seq_len = _gqa_kv_lengths(op, attention_mask)
        else:
            attention_bias = create_attention_bias(
                op,
                input_ids=hidden_states if input_ids is None else input_ids,
                attention_mask=attention_mask,
                sliding_window=self.sliding_window,
                dtype=self._dtype,
            )
            seqlens_k, total_seq_len = None, None
        # The native-CSA op needs seqlens_k/total_sequence_length regardless of
        # the dense path's fused-GQA gate. Compute them only when the feature
        # is enabled, and reuse the fused-path lengths when that path already
        # built them so a GQA-capable EP emits no duplicate length nodes. This
        # intentionally does NOT feed the dense `seqlens_k` gate above: leaving
        # it None on non-GQA EPs keeps non-CSA layers on the portable
        # decomposed path (no stray `com.microsoft` ops on a `default` EP).
        csa_lengths = None
        if self.native_csa:
            if seqlens_k is not None:
                csa_lengths = (seqlens_k, total_seq_len)
            else:
                csa_lengths = _gqa_kv_lengths(op, attention_mask)
        presents = []
        present_compressed_states = []
        past_kvs = past_key_values or [None] * len(self.layers)
        past_compressed = past_compressed_states or [None] * len(self.layers)
        for layer, past_kv, past_comp in zip(self.layers, past_kvs, past_compressed):
            layer_position_embeddings = (
                compressed_position_embeddings
                if layer.self_attn.compress_ratio
                else position_embeddings
            )
            hidden_states, present, present_compressed = layer(
                op,
                hidden_states,
                input_ids,
                layer_position_embeddings,
                past_kv,
                attention_bias,
                seqlens_k,
                total_seq_len,
                past_comp,
                csa_lengths,
            )
            presents.append(present)
            present_compressed_states.append(present_compressed)
        return (
            self.norm(op, self._hc_head(op, hidden_states)),
            presents,
            present_compressed_states,
            hidden_states,
        )


class DeepSeekV4Mtp(DeepSeekV4DecoderLayer):
    """Official single MTP block, sharing target embeddings and LM head externally."""

    def __init__(self, config: ArchitectureConfig, layer_id: int):
        super().__init__(config, layer_id, is_mtp=True)
        projection = _projection_class(config)
        self.e_proj = projection(config.hidden_size, config.hidden_size, bias=False)
        self.h_proj = projection(config.hidden_size, config.hidden_size, bias=False)
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_head_fn = Linear(
            config.hc_mult * config.hidden_size, config.hc_mult, bias=False
        )
        self.hc_head_base = nn.Parameter([config.hc_mult])
        self.hc_head_scale = nn.Parameter([1])
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        rope_config = dataclasses.replace(
            config,
            head_dim=config.qk_rope_head_dim,
            rope_type="default",
            rope_scaling=None,
            original_max_position_embeddings=None,
        )
        self.rotary_emb = initialize_rope(rope_config)
        self._dtype = config.dtype
        # See `DeepSeekV4Attention.local_window_size`: the MTP layer is a
        # regular ratio-0 (or configured-ratio) DeepSeekV4DecoderLayer, so it
        # is bound by the same mandatory window as the backbone.
        self.sliding_window = config.sliding_window

    def _hc_head(self, op: OpBuilder, states: ir.Value) -> ir.Value:
        flat = op.Reshape(states, [0, 0, -1])
        rms = op.Sqrt(op.Add(op.ReduceMean(op.Mul(flat, flat), [-1], keepdims=True), 1e-6))
        mixes = op.Div(self.hc_head_fn(op, flat), rms)
        weights = op.Add(
            op.Sigmoid(op.Add(op.Mul(mixes, self.hc_head_scale), self.hc_head_base)),
            self.hc_eps,
        )
        return op.ReduceSum(op.Mul(op.Unsqueeze(weights, [-1]), states), [2], keepdims=False)

    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: ir.Value,
        hidden_states: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_value: tuple | None = None,
    ):
        hidden_states = op.Add(
            op.Unsqueeze(self.e_proj(op, self.enorm(op, inputs_embeds)), [2]),
            self.h_proj(op, self.hnorm(op, hidden_states)),
        )
        position_embeddings = self.rotary_emb(op, position_ids)
        if _use_fused_gqa():
            attention_bias = None
            seqlens_k, total_seq_len = _gqa_kv_lengths(op, attention_mask)
        else:
            attention_bias = create_attention_bias(
                op,
                input_ids=inputs_embeds,
                attention_mask=attention_mask,
                sliding_window=self.sliding_window,
                dtype=self._dtype,
            )
            seqlens_k, total_seq_len = None, None
        hidden_states, present, _present_compressed = super().forward(
            op,
            hidden_states,
            None,
            position_embeddings,
            past_key_value,
            attention_bias,
            seqlens_k,
            total_seq_len,
        )
        return self.norm(op, self._hc_head(op, hidden_states)), present


class DeepSeekV4CausalLMModel(CausalLMModel):
    """DeepSeek-V4 Causal LM with dense CSA fallback and an MTP sidecar."""

    default_task: str = "deepseek-v4"
    category: str = "Mixture of Experts"

    def __init__(self, config: ArchitectureConfig):
        nn.Module.__init__(self)
        if any(config.compress_ratios or ()) and not config.native_csa:
            logger.warning(
                "DeepSeek-V4 sparse cache execution requires runtime support; "
                "exporting sink-aware dense attention with CSA/HCA tensors retained."
            )
        self.config = config
        self.model = DeepSeekV4TextModel(config)
        if config.quantization is not None and config.quantization.quantize_lm_head:
            self.lm_head = _projection_class(config)(
                config.hidden_size, config.vocab_size, bias=False
            )
        else:
            self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.num_nextn_predict_layers not in (0, 1):
            raise ValueError("DeepSeek-V4 MTP export supports exactly one MTP layer")
        self.mtp = nn.ModuleList(
            [
                DeepSeekV4Mtp(config, config.num_hidden_layers + index)
                for index in range(config.num_nextn_predict_layers)
            ]
        )

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states, presents, _present_compressed, _hc_states = self.model(
            op,
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
        )
        return self.lm_head(op, hidden_states), presents

    def build_block_quant_streaming_plan(
        self,
        component_name: str,
        key_index: dict[str, tuple[str, list[int], str]],
        initializers: dict[str, ir.Value],
    ):
        """Build a complete header-validated native planar weight plan."""
        from mobius.integrations._block_quant import (
            QuantKind,
            build_descriptors,
        )
        from mobius.integrations._weight_loading import (
            StreamingExpertBankSource,
            StreamingWeightPlan,
            StreamingWeightSource,
        )

        assert_native_runtime_supports_block_quant(self.config)
        if self.config.block_quant_scheme is None:
            raise NativeCsaExportError(
                "native block-quant streaming requested without a block-quant scheme"
            )
        if component_name not in {"model", "mtp"}:
            raise NativeCsaExportError(
                f"unknown DeepSeek-V4 package component {component_name!r}"
            )
        descriptors = build_descriptors(
            {name: (dtype, tuple(shape)) for name, (_path, shape, dtype) in key_index.items()},
            self.config.block_quant_scheme,
        )
        malformed = [
            descriptor
            for descriptor in descriptors.values()
            if descriptor.kind is QuantKind.UNSUPPORTED
        ]
        if malformed:
            sample = malformed[0]
            raise NativeCsaExportError(
                f"malformed block-quant checkpoint tensor {sample.name!r}: "
                f"{sample.unsupported_reason}"
            )

        mtp_component = component_name == "mtp"
        targets: dict[str, StreamingWeightSource | StreamingExpertBankSource] = {}
        ignored: dict[str, str] = {}

        def owned(source_name: str) -> bool:
            return source_name.startswith("mtp.") == mtp_component

        def native(source_name: str) -> StreamingWeightSource:
            if source_name not in key_index:
                raise NativeCsaExportError(
                    f"missing required planar checkpoint tensor {source_name!r}"
                )
            return StreamingWeightSource(source_name, mode="native")

        expert_prefixes: set[str] = set()
        for source_name in key_index:
            if not owned(source_name):
                ignored[source_name] = (
                    "target decoder tensor" if mtp_component else "MTP sidecar tensor"
                )
                continue
            if ".ffn.experts." in source_name:
                expert_prefixes.add(source_name.split(".ffn.experts.", 1)[0])
                continue

            target_name = _map_checkpoint_weight_name(source_name)
            if target_name not in initializers:
                # Hash-routed layers consume the gate matrix to compute routing
                # weights, but their checkpoint bias is intentionally unused.
                if (
                    source_name.endswith(".ffn.gate.bias")
                    and source_name.startswith("layers.")
                    and int(source_name.split(".", 2)[1]) < self.config.num_hash_layers
                ):
                    ignored[source_name] = "unused hash-router bias"
                    continue
                raise NativeCsaExportError(
                    f"checkpoint tensor {source_name!r} maps to missing "
                    f"{component_name} initializer {target_name!r}"
                )
            source_dtype = key_index[source_name][2]
            if source_dtype in {"F8_E4M3", "F8_E8M0", "I8"}:
                targets[target_name] = StreamingWeightSource(source_name, mode="native")
            else:
                targets[target_name] = StreamingWeightSource(source_name, mode="direct")

        experts = self.config.num_local_experts
        if experts is None or experts <= 0:
            raise NativeCsaExportError(
                f"invalid routed expert count {experts} for native planar export"
            )
        for source_prefix in sorted(expert_prefixes):
            target_prefix = _map_checkpoint_weight_name(
                f"{source_prefix}.ffn.experts.0.w1.weight"
            ).split(".mlp.experts.", 1)[0]
            bank_sources = {
                f"{target_prefix}.mlp.moe.fc1_experts_weights": (("w1", "weight"),),
                f"{target_prefix}.mlp.moe.fc2_experts_weights": (("w2", "weight"),),
                f"{target_prefix}.mlp.moe.fc3_experts_weights": (("w3", "weight"),),
                f"{target_prefix}.mlp.moe.fc1_experts_aux_scale": (("w1", "scale"),),
                f"{target_prefix}.mlp.moe.fc2_experts_aux_scale": (("w2", "scale"),),
                f"{target_prefix}.mlp.moe.fc3_experts_aux_scale": (("w3", "scale"),),
            }
            for target_name, projections in bank_sources.items():
                if target_name not in initializers:
                    raise NativeCsaExportError(
                        f"planar expert bank maps to missing {component_name} "
                        f"initializer {target_name!r}"
                    )
                targets[target_name] = StreamingExpertBankSource(
                    experts=tuple(
                        tuple(
                            native(f"{source_prefix}.ffn.experts.{expert}.{projection}.{kind}")
                            for projection, kind in projections
                        )
                        for expert in range(experts)
                    )
                )

        missing_targets = sorted(
            name
            for name, initializer in initializers.items()
            if initializer.const_value is None and name not in targets
        )
        if missing_targets:
            raise NativeCsaExportError(
                f"{len(missing_targets)} {component_name} initializer(s) have no "
                f"checkpoint mapping (e.g. {missing_targets[:5]})"
            )

        return StreamingWeightPlan(
            targets=targets,
            ignored=ignored,
            report={
                "output_weight_format": "native_planar_block_quant",
                "native_fp8": True,
                "native_planar_fp4": True,
                "runtime_execution_proven": False,
                "runtime_dependency": "justinchuby/onnx-genai#2321",
                "component": component_name,
            },
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map the official DeepSeek checkpoint names to mobius modules."""
        # Full-export runtime-capability gate. When a native-CSA export deferred
        # #602's block-quant reject (so graph construction could progress), the
        # canonical planar producer is representable before any weight is
        # mapped/assigned. No-op on every ordinary path.
        assert_native_runtime_supports_block_quant(self.config)
        # Same predicate as MoELayer/_supported_qmoe_quantization so the
        # repacked weights and the emitted graph never disagree.
        use_qmoe = supported_qmoe_quantization(self.config.quantization) is not None
        use_planar = self.config.block_quant_scheme is not None
        renamed: dict[str, torch.Tensor] = {}
        skipped = 0
        for key, value in state_dict.items():
            if key.startswith("mtp.") and len(self.mtp) == 0:
                skipped += 1
                continue
            if key.startswith("mtp."):
                try:
                    mtp_index = int(key.split(".", 2)[1])
                except (IndexError, ValueError):
                    skipped += 1
                    continue
                if mtp_index >= len(self.mtp):
                    skipped += 1
                    continue
            new_key = _map_checkpoint_weight_name(key)
            # Dense fallback (unquantized or non-QMoE-eligible): experts are a
            # ModuleList under ``mlp.moe.experts``. Fused QMoE/planar paths
            # consume the intermediate ``mlp.experts`` names as banks.
            if not use_qmoe and not use_planar:
                new_key = new_key.replace(".mlp.experts.", ".mlp.moe.experts.")
            renamed[new_key] = value
        if skipped:
            logger.warning(
                "Skipped %d DeepSeek-V4 MTP tensors outside the configured "
                "num_nextn_predict_layers.",
                skipped,
            )
        processed = super().preprocess_weights(renamed)
        if use_planar:
            _validate_hash_routing_tables(processed)
            processed = _pack_planar_expert_weights(processed)
        elif use_qmoe:
            _validate_hash_routing_tables(processed)
            # DeepSeek-V4 checkpoints store routed experts per-index
            # (".mlp.experts.{i}.gate_proj/up_proj/down_proj.*"), unlike
            # DeepSeek-V3's already-fused HF format. Bridge to the fused
            # expert-major layout pack_qmoe_expert_weights expects.
            processed = stack_per_expert_moe_weights(processed, qmoe_target_path=".mlp")
            processed = pack_qmoe_expert_weights(processed)
        return processed
