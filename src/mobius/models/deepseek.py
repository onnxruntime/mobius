# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""DeepSeek-V3 model with Multi-head Latent Attention (MLA) and MoE.

Reference: DeepSeek-V3 paper, HuggingFace DeepseekV3ForCausalLM.
"""

from __future__ import annotations

import dataclasses

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius._weight_utils import pack_qmoe_expert_weights
from mobius.components import (
    MLP,
    Attention,
    Linear,
    MoELayer,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
)
from mobius.components._deepseek_mla import DeepSeekMLA
from mobius.components._moe import _supported_qmoe_quantization
from mobius.components._paged_mla import (
    PagedLatentMLA,
    absorb_mla_weights,
    mla_paged_geometry,
)
from mobius.components._quantized_linear import make_quantized_linear_factory
from mobius.models.base import CausalLMModel, embedding_for_config


def _linear_factory(config: ArchitectureConfig):
    quantization = config.quantization
    if quantization is None or quantization.quant_method == "none":
        return None
    zero_point_dtype = config.dtype if quantization.float_zero_point else ir.DataType.UINT8
    return make_quantized_linear_factory(
        bits=quantization.bits,
        block_size=quantization.group_size,
        has_zero_point=not quantization.sym,
        zero_point_dtype=zero_point_dtype,
    )


class DeepSeekMoEGate(nn.Module):
    """Expert routing gate for DeepSeek-V2/V3 MoE.

    Supports two scoring modes:
    - sigmoid (V3): sigmoid scoring + correction bias + group TopK
    - softmax (V2/V2-Lite): softmax scoring + simple or group-limited TopK

    Selection method is controlled by topk_method config:
    - "greedy": simple TopK (V2-Lite)
    - "group_limited_greedy": group-based selection with softmax (V2)
    - "noaux_tc": sigmoid + correction bias + group TopK (V3)
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        assert config.num_local_experts is not None
        assert config.num_experts_per_tok is not None
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob
        self.scoring_func = config.scoring_func
        self.topk_method = config.topk_method
        explicit_expert_bias = getattr(config, "use_expert_bias", None)
        self.use_expert_bias = (
            self.scoring_func == "sigmoid"
            if explicit_expert_bias is None
            else explicit_expert_bias
        )
        if self.n_group <= 0 or self.num_experts % self.n_group:
            raise ValueError(
                "DeepSeek grouped routing requires num_local_experts to be evenly "
                f"divisible by n_group, got {self.num_experts} and {self.n_group}"
            )
        if self.topk_group <= 0 or self.topk_group > self.n_group:
            raise ValueError(
                "DeepSeek grouped routing requires 1 <= topk_group <= n_group, got "
                f"{self.topk_group} and {self.n_group}"
            )

        self.weight = nn.Parameter([self.num_experts, config.hidden_size])
        # Correction bias changes selection only and is valid for both the
        # softmax and sigmoid Dots1 routes.
        if self.use_expert_bias:
            self.e_score_correction_bias = nn.Parameter([self.num_experts])

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        _, scores, scores_for_choice = self._routing_scores(op, hidden_states)

        # Select top-k experts
        k_val = op.Constant(value_ints=[self.top_k])
        _, selected_experts = op.TopK(scores_for_choice, k_val, axis=-1, _outputs=2)

        # Gather original scores (without bias) for selected experts
        routing_weights = op.GatherElements(scores, selected_experts, axis=-1)

        # Normalize weights (V3 with norm_topk_prob=True)
        if self.norm_topk_prob:
            weight_sum = op.ReduceSum(routing_weights, [-1], keepdims=True)
            eps = 1e-20
            routing_weights = op.Div(routing_weights, op.Add(weight_sum, eps))

        # Apply routing scale
        routing_weights = op.Mul(routing_weights, float(self.routed_scaling_factor))

        return routing_weights, selected_experts

    def qmoe_routing(self, op: OpBuilder, hidden_states: ir.Value):
        """Return the best QMoE encoding available for this routing contract.

        QMoE must receive raw logits as ``router_probs`` because its CUDA
        kernel ignores ``router_weights`` and applies softmax internally.
        This is exact for ungrouped softmax routing without a selection
        correction bias. Existing grouped or bias-corrected users retain the
        CPU-correct activated-score encoding; their configs must set
        ``disable_qmoe`` when targeting CUDA.
        """
        router_logits, scores, scores_for_choice = self._routing_scores(op, hidden_states)
        exact_cuda_route = (
            self.scoring_func == "softmax" and not self.use_expert_bias and self.n_group == 1
        )
        router_probs = router_logits if exact_cuda_route else scores_for_choice
        # QMoE's router_probs/router_weights inputs share type constraint "T"
        # with hidden_states (see contrib_defs.cc). _routing_scores computes
        # in float32 for numerical stability (matching HF's fp32 routing), so
        # cast back to hidden_states' dtype before returning -- otherwise
        # QMoE rejects a fp16/bf16 model with a mismatched-T type error.
        return (
            op.CastLike(router_probs, hidden_states),
            op.CastLike(scores, hidden_states),
            self.norm_topk_prob,
            float(self.routed_scaling_factor),
        )

    def _routing_scores(self, op: OpBuilder, hidden_states: ir.Value):
        weight_t = op.Transpose(self.weight, perm=[1, 0])
        router_logits = op.MatMul(
            op.Cast(hidden_states, to=1),
            op.Cast(weight_t, to=1),
        )

        # Score computation depends on scoring function
        if self.scoring_func == "sigmoid":
            scores = op.Sigmoid(router_logits)  # (B*S, num_experts)
        else:
            # Softmax scoring (V2)
            scores = op.Softmax(router_logits, axis=-1)
        # Aggregation always gathers unbiased probabilities. Some Dots1 files
        # omit this optional selection-only correction tensor.
        scores_for_choice = (
            op.Add(scores, self.e_score_correction_bias) if self.use_expert_bias else scores
        )

        # Expert selection: group-based or simple TopK
        if self.n_group > 1 and self.topk_method != "greedy":
            scores_for_choice = self._group_topk_selection(op, scores_for_choice)
        return router_logits, scores, scores_for_choice

    def _group_topk_selection(self, op, scores_for_choice):
        """Group-based expert selection: pick topk_group groups first."""
        experts_per_group = self.num_experts // self.n_group

        # Flatten batch*seq dims: (B, S, n_experts) → (B*S, n_experts)
        # so group reshaping operates on a 2D tensor (token x expert).
        orig_shape = op.Shape(scores_for_choice)
        flat = op.Reshape(scores_for_choice, [-1, self.num_experts])

        # Reshape to groups: (B*S, n_group, experts_per_group)
        scores_grouped = op.Reshape(flat, [0, self.n_group, experts_per_group])
        if self.topk_method == "noaux_tc":
            # Bias-corrected routing scores groups by their two strongest experts.
            k_two = op.Constant(value_ints=[min(2, experts_per_group)])
            group_top2, _ = op.TopK(scores_grouped, k_two, axis=-1, _outputs=2)
            group_scores = op.ReduceSum(group_top2, [-1], keepdims=False)
        else:
            # Group-limited greedy routing scores each group by its strongest expert.
            group_scores = op.ReduceMax(scores_grouped, [-1], keepdims=False)

        # Select top groups
        k_groups = op.Constant(value_ints=[self.topk_group])
        _, group_indices = op.TopK(group_scores, k_groups, axis=-1, _outputs=2)

        # Create mask for selected groups
        group_mask = op.OneHot(
            group_indices,
            self.n_group,
            op.Constant(value_floats=[0.0, 1.0]),
            axis=-1,
        )  # (B*S, topk_group, n_group)
        # Reduce to (B*S, n_group) — 1 if group selected
        group_mask = op.ReduceMax(group_mask, [1], keepdims=False)
        # Expand to per-expert: (B*S, n_group, 1) → (B*S, n_group, experts_per_group)
        group_mask_expanded = op.Reshape(group_mask, [0, self.n_group, 1])
        group_mask_expanded = op.Expand(
            group_mask_expanded,
            [1, 1, experts_per_group],
        )
        # Flatten back: (B*S, num_experts)
        expert_mask = op.Reshape(group_mask_expanded, [0, self.num_experts])
        # Mask rather than multiply by zero: correction biases may make valid
        # selected-group scores negative, while zero would let excluded experts win TopK.
        selected = op.Greater(expert_mask, 0.0)
        neg_inf = op.CastLike(float("-inf"), flat)
        masked_scores = op.Where(selected, flat, neg_inf)
        return op.Reshape(masked_scores, orig_shape)


class DeepSeekMLADecoderLayer(nn.Module):
    """Decoder layer using Multi-head Latent Attention.

    Forward signature matches DecoderLayer for compatibility with TextModel.
    """

    def __init__(self, config: ArchitectureConfig, is_moe: bool = False):
        super().__init__()
        linear_class = _linear_factory(config)
        if config.export_paged_attention:
            # Feature-on: emit LATENT PagedAttention. An ineligible geometry
            # raises here (typed reason) rather than silently falling back.
            self.self_attn = PagedLatentMLA(config, linear_class=linear_class)
        else:
            self.self_attn = DeepSeekMLA(config, linear_class=linear_class)
        if is_moe:
            gate = DeepSeekMoEGate(config)
            self.mlp = _DeepSeekMoEFFN(config, gate, linear_class=linear_class)
        else:
            self.mlp = MLP(config, linear_class=linear_class)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None = None,
    ):
        # Self attention with pre-norm
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        if isinstance(self.self_attn, PagedLatentMLA):
            hidden_states, present_kv = self.self_attn(
                op,
                hidden_states=hidden_states,
                cache=past_key_value,
            )
        else:
            hidden_states, present_kv = self.self_attn(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_key_value,
            )
        hidden_states = op.Add(residual, hidden_states)

        # FFN with pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, present_kv


class _DeepSeekStandardDecoderLayer(nn.Module):
    """Decoder layer using standard attention (no MLA) with optional MoE.

    Used by DeepSeek-V2 models with use_mla=false (e.g. DeepSeek-OCR-2 LLM).
    Forward signature matches DeepSeekMLADecoderLayer for compatibility.
    """

    def __init__(self, config: ArchitectureConfig, is_moe: bool = False):
        super().__init__()
        linear_class = _linear_factory(config)
        self.self_attn = Attention(config, linear_class=linear_class)
        if is_moe:
            gate = DeepSeekMoEGate(config)
            self.mlp = _DeepSeekMoEFFN(config, gate, linear_class=linear_class)
        else:
            self.mlp = MLP(config, linear_class=linear_class)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        past_key_value: tuple | None = None,
    ):
        # Self attention with pre-norm
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)
        hidden_states, present_kv = self.self_attn(
            op,
            hidden_states=hidden_states,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
        )
        hidden_states = op.Add(residual, hidden_states)

        # FFN with pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, present_kv


class _DeepSeekMoEFFN(nn.Module):
    """MoE FFN with shared expert for DeepSeek-V3.

    Combines routed experts with a shared expert that processes all tokens.
    Output = moe_routed_output + shared_expert_output.
    """

    def __init__(
        self, config: ArchitectureConfig, gate: nn.Module, linear_class: type | None = None
    ):
        super().__init__()
        assert config.num_local_experts is not None
        assert config.moe_intermediate_size is not None
        # ``linear_class`` must reach the routed-expert dense-loop fallback
        # too (not just the shared expert below): otherwise a quantized
        # config quantizes every other linear in the model (attention,
        # dense FFN, shared expert) but silently leaves the routed MoE
        # experts as plain float `MatMul`, which both loses quantization
        # and breaks the `fuse_dense_moe_to_qmoe` post-hoc rewrite (it
        # only matches a quantized `MatMulNBits` dense-fallback pattern).
        self.moe = MoELayer(config, gate=gate, linear_class=linear_class)
        # Shared expert uses moe_intermediate_size * n_shared_experts
        n_shared = config.n_shared_experts or 1
        shared_intermediate = config.moe_intermediate_size * n_shared
        self.shared_experts = _SharedExpertMLP(
            config, shared_intermediate, linear_class=linear_class
        )

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        moe_output = self.moe(op, hidden_states)
        shared_output = self.shared_experts(op, hidden_states)
        return op.Add(moe_output, shared_output)


class _SharedExpertMLP(nn.Module):
    """Shared expert MLP (same architecture as gate/up/down SiLU MLP)."""

    def __init__(
        self,
        config: ArchitectureConfig,
        intermediate_size: int,
        linear_class: type | None = None,
    ):
        super().__init__()
        if linear_class is None:
            linear_class = Linear
        self.gate_proj = linear_class(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = linear_class(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = linear_class(intermediate_size, config.hidden_size, bias=False)

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        gate_out = self.gate_proj(op, hidden_states)
        gate = op.Swish(gate_out)
        up = self.up_proj(op, hidden_states)
        return self.down_proj(op, op.Mul(gate, up))


class DeepSeekV3TextModel(nn.Module):
    """Text model for DeepSeek-V2/V3 with optional MLA and MoE.

    Architecture:
    - Embedding → N layers → RMSNorm
    - First `first_k_dense_replace` layers use standard MLP
    - Remaining layers use MoE FFN (sigmoid/softmax routing, shared expert)
    - When qk_nope_head_dim > 0: uses Multi-head Latent Attention (MLA)
    - When qk_nope_head_dim == 0 or None: uses standard attention (GQA)
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.embed_tokens = embedding_for_config(config)
        self._dtype = config.dtype

        # Detect MLA vs standard attention
        use_mla = config.qk_nope_head_dim is not None and config.qk_nope_head_dim > 0
        LayerClass = DeepSeekMLADecoderLayer if use_mla else _DeepSeekStandardDecoderLayer  # noqa: N806

        # Build layers: dense for first k, MoE for rest
        first_k = config.first_k_dense_replace
        # If no experts are configured, force all layers to be dense
        if not config.num_local_experts:
            first_k = config.num_hidden_layers
        self.layers = nn.ModuleList(
            [
                LayerClass(config, is_moe=(i >= first_k))
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._export_paged_attention = config.export_paged_attention

        # For MLA, RoPE applies only to the qk_rope_head_dim portion of Q and K,
        # not the full head_dim. Create a modified config so the cos/sin cache
        # has the correct dimensionality: (max_pos, qk_rope_head_dim/2).
        if use_mla and config.qk_rope_head_dim is not None and config.qk_rope_head_dim > 0:
            rope_config = dataclasses.replace(config, head_dim=config.qk_rope_head_dim)
        else:
            rope_config = config
        self.rotary_emb = initialize_rope(rope_config)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        inputs_embeds: ir.Value | None = None,
    ):
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(op, input_ids)

        if self._export_paged_attention:
            return self._forward_paged(op, hidden_states, past_key_values)

        position_embeddings = self.rotary_emb(op, position_ids)
        attention_bias = create_attention_bias(
            op,
            input_ids=hidden_states if input_ids is None else input_ids,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )

        present_key_values = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer, past_kv in zip(self.layers, past_kvs):
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)

        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values

    def _forward_paged(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        past_key_values: list | None,
    ):
        """Paged LATENT forward: RoPE is applied inside ``PagedAttention``.

        The RoPE cos/sin tables come from this model's ``rotary_emb`` parameters
        (cast to the compute dtype) and are injected into every layer's
        caller-owned :class:`PagedCacheState`.
        """
        if past_key_values is None:
            raise ValueError(
                "export_paged_attention requires caller-owned paged cache state; "
                "use CausalLMTask(paged_cache=True)."
            )
        cos = self.rotary_emb.cos_cache
        sin = self.rotary_emb.sin_cache
        if self._dtype != ir.DataType.FLOAT:
            cos = op.Cast(cos, to=self._dtype)
            sin = op.Cast(sin, to=self._dtype)

        present_key_values = []
        for layer, cache in zip(self.layers, past_key_values):
            cache = dataclasses.replace(cache, cos_cache=cos, sin_cache=sin)
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=None,
                position_embeddings=None,
                past_key_value=cache,
            )
            present_key_values.append(present_kv)

        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values


class DeepSeekV3CausalLMModel(CausalLMModel):
    """DeepSeek-V3 Causal LM with MLA + MoE.

    model_type: deepseek_v3
    """

    default_task: str = "text-generation"
    category: str = "Mixture of Experts"

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        self._replace_text_model(DeepSeekV3TextModel(config))

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Remap HuggingFace weight names to ONNX parameter names.

        Key mappings:
        - MLA attention projections align already (q_a_proj, q_b_proj, etc.)
        - MoE gate: mlp.gate.weight → mlp.moe.gate.weight
        - Shared expert weights align already (mlp.shared_experts.* → same)
        - MoE expert weights: HF stores all experts in a single fused tensor
            experts.gate_up_proj: (n_experts, 2*intermediate, hidden)
            experts.down_proj:    (n_experts, hidden, intermediate)
          These are split into per-expert ONNX weights:
            moe.experts.{i}.gate_proj.weight: (intermediate, hidden)
            moe.experts.{i}.up_proj.weight:   (intermediate, hidden)
            moe.experts.{i}.down_proj.weight: (hidden, intermediate)
        """
        renamed = {}
        # Same predicate as MoELayer/_supported_qmoe_quantization so the
        # repacked weights and the emitted graph never disagree.
        use_qmoe = (
            not self.config.disable_qmoe
            and _supported_qmoe_quantization(self.config.quantization) is not None
        )
        for key, value in state_dict.items():
            new_key = key

            # Remap MoE layer names: mlp.gate.* → mlp.moe.gate.*
            new_key = new_key.replace(".mlp.gate.", ".mlp.moe.gate.")
            # GGUF expert-major tensors are expanded before model preprocessing.
            if ".mlp.experts." in new_key:
                expert_suffix = new_key.split(".mlp.experts.", 1)[1]
                if expert_suffix.split(".", 1)[0].isdigit():
                    new_key = new_key.replace(".mlp.experts.", ".mlp.moe.experts.")

            # HF stores all routed experts in fused tensors:
            # layers.N.mlp.experts.gate_up_proj  (n_experts, 2*mid, hidden)
            # layers.N.mlp.experts.down_proj      (n_experts, hidden, mid)
            # Split into per-expert weights for our ModuleList.
            if not use_qmoe and new_key.endswith(".mlp.experts.gate_up_proj"):
                prefix = new_key[: -len(".mlp.experts.gate_up_proj")]
                mid = value.shape[1] // 2
                for i in range(value.shape[0]):
                    renamed[f"{prefix}.mlp.moe.experts.{i}.gate_proj.weight"] = value[i, :mid]
                    renamed[f"{prefix}.mlp.moe.experts.{i}.up_proj.weight"] = value[i, mid:]
                continue
            if not use_qmoe and new_key.endswith(".mlp.experts.down_proj"):
                prefix = new_key[: -len(".mlp.experts.down_proj")]
                for i in range(value.shape[0]):
                    renamed[f"{prefix}.mlp.moe.experts.{i}.down_proj.weight"] = value[i]
                continue

            renamed[new_key] = value

        # Absorb kv_b_proj into the query/output projections for the opt-in
        # LATENT PagedAttention export (feature-off leaves weights untouched).
        if self.config.export_paged_attention:
            renamed = self._absorb_paged_mla_weights(renamed)

        # Handle weight tying and GPTQ/AWQ conversion before flattening the
        # expert-major MatMulNBits blobs into the QMoE ABI.
        processed = super().preprocess_weights(renamed)
        if use_qmoe:
            processed = pack_qmoe_expert_weights(processed)
        return processed

    def _absorb_paged_mla_weights(
        self, weights: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Fold each layer's ``kv_b_proj`` into ``q_(b_)proj`` and ``o_proj``.

        Produces the absorbed shapes consumed by :class:`PagedLatentMLA` and
        drops ``kv_b_proj`` (fully absorbed). Matches the numeric contract in
        :func:`mobius.components._paged_mla.absorb_mla_weights`.
        """
        # Validate eligibility (raises a typed reason if ineligible).
        mla_paged_geometry(self.config)
        out = dict(weights)
        kv_b_suffix = ".self_attn.kv_b_proj.weight"
        for key in list(weights):
            if not key.endswith(kv_b_suffix):
                continue
            prefix = key[: -len(kv_b_suffix)]
            q_key = f"{prefix}.self_attn.q_b_proj.weight"
            if q_key not in weights:
                q_key = f"{prefix}.self_attn.q_proj.weight"
            o_key = f"{prefix}.self_attn.o_proj.weight"
            if q_key not in weights or o_key not in weights:
                raise ValueError(
                    f"Cannot absorb paged MLA weights for '{prefix}': missing "
                    f"query or output projection."
                )
            q_t, o_t = weights[q_key], weights[o_key]
            absorbed = absorb_mla_weights(
                {q_key: q_t, key: weights[key], o_key: o_t},
                self.config,
                q_key=q_key,
                kv_b_key=key,
                o_key=o_key,
            )
            out[q_key] = torch.as_tensor(absorbed[q_key]).to(q_t.dtype)
            out[o_key] = torch.as_tensor(absorbed[o_key]).to(o_t.dtype)
            del out[key]  # kv_b_proj fully absorbed
        return out
