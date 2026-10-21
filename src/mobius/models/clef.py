# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Clef-Flash's Qwen3.5 backbone and joint schema decision head.

Replicates Cloudflare's ``JointSchemaHead``: final backbone hidden states,
question/option token spans, and output-embedding lexical features produce
one logit per allowed option. This is a decision model, not a generative LM.
"""

from __future__ import annotations

import dataclasses
import math
from typing import ClassVar

import onnx_ir as ir
import torch
from onnxscript import nn

from mobius._configs import ArchitectureConfig
from mobius.components import Embedding, LayerNorm, Linear
from mobius.models.qwen35 import Qwen35VL3ModelCausalLMModel, Qwen35VLDecoderModel
from mobius.models.qwen_vl import Qwen25VLEmbeddingModel

CLEF_FLASH_MODEL_ID = "Cloudflare/clef-flash"
CLEF_FLASH_REVISION = "fde727a287004204b7518dcc983fe64379776712"


@dataclasses.dataclass
class ClefConfig(ArchitectureConfig):
    """Backbone config plus the separately published joint-head topology."""

    head_width: int = 1024
    head_routing_layers: int = 2
    head_layers: int = 4
    head_heads: int = 16
    head_feedforward: int = 4096
    # Qwen3.5's FLA-compatible L2 norm adds epsilon to the squared norm.
    linear_qk_l2norm_eps: float = 1e-6
    mamba_ssm_dtype: ir.DataType = ir.DataType.FLOAT

    @classmethod
    def from_transformers(cls, config, parent_config=None):
        result = super().from_transformers(config, parent_config)
        # HF's Qwen3.5 rotary implementation reads rope_parameters, even when
        # legacy top-level defaults disagree with the serialized values.
        parameters = getattr(config, "rope_parameters", None) or {}
        if result.rope is not None:
            rope = dataclasses.replace(
                result.rope,
                rope_theta=parameters.get("rope_theta", result.rope.rope_theta),
                partial_rotary_factor=parameters.get(
                    "partial_rotary_factor", result.rope.partial_rotary_factor
                ),
            )
            result = dataclasses.replace(
                result,
                rope=rope,
                rope_theta=rope.rope_theta,
                partial_rotary_factor=rope.partial_rotary_factor,
            )
        return result

    def __post_init__(self):
        if (
            min(self.head_width, self.head_heads, self.head_feedforward) < 1
            or min(self.head_routing_layers, self.head_layers) < 0
            or self.head_width % self.head_heads
        ):
            raise ValueError("Invalid Clef joint-head dimensions or layer counts")


class _HeadAttention(nn.Module):
    """Noncausal PyTorch MultiheadAttention, with split biased Q/K/V."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.q_proj = Linear(width, width, bias=True)
        self.k_proj = Linear(width, width, bias=True)
        self.v_proj = Linear(width, width, bias=True)
        self.out_proj = Linear(width, width, bias=True)

    def forward(self, op, queries, memory):
        attended = op.Attention(
            self.q_proj(op, queries),
            self.k_proj(op, memory),
            self.v_proj(op, memory),
            q_num_heads=self.heads,
            kv_num_heads=self.heads,
        )
        return self.out_proj(op, attended)


class _EvidenceLayer(nn.Module):
    def __init__(self, config: ClefConfig):
        super().__init__()
        width = config.head_width
        self.query_norm = LayerNorm(width, eps=1e-5)
        self.memory_norm = LayerNorm(width, eps=1e-5)
        self.attention = _HeadAttention(width, config.head_heads)
        self.feedforward_norm = LayerNorm(width, eps=1e-5)
        # Keep the dropout slots so checkpoint indices remain 0 and 3.
        self.feedforward = nn.ModuleList(
            [
                Linear(width, config.head_feedforward),
                nn.Module(),
                nn.Module(),
                Linear(config.head_feedforward, width),
            ]
        )

    def forward(self, op, queries, memory):
        queries = op.Add(
            queries,
            self.attention(op, self.query_norm(op, queries), self.memory_norm(op, memory)),
        )
        ff = self.feedforward[0](op, self.feedforward_norm(op, queries))
        ff = self.feedforward[3](op, op.Gelu(ff))
        return op.Add(queries, ff)


class _FieldLayer(nn.Module):
    """Pre-norm, noncausal TransformerDecoderLayer over schema fields."""

    def __init__(self, config: ClefConfig):
        super().__init__()
        width = config.head_width
        self.self_attn = _HeadAttention(width, config.head_heads)
        self.multihead_attn = _HeadAttention(width, config.head_heads)
        self.norm1 = LayerNorm(width, eps=1e-5)
        self.norm2 = LayerNorm(width, eps=1e-5)
        self.norm3 = LayerNorm(width, eps=1e-5)
        self.linear1 = Linear(width, config.head_feedforward)
        self.linear2 = Linear(config.head_feedforward, width)

    def forward(self, op, fields, memory):
        normalized = self.norm1(op, fields)
        fields = op.Add(fields, self.self_attn(op, normalized, normalized))
        fields = op.Add(fields, self.multihead_attn(op, self.norm2(op, fields), memory))
        return op.Add(
            fields, self.linear2(op, op.Gelu(self.linear1(op, self.norm3(op, fields))))
        )


def _span_mean(op, values, spans):
    # Prefix sums avoid a sequence-length-by-option-count pooling mask.
    # Accumulate in FP32: FP16/BF16 prefixes lose short spans late in a record.
    accumulated = op.Cast(values, to=ir.DataType.FLOAT)
    zero = op.Mul(op.Slice(accumulated, [0], [1], axes=[0]), op.CastLike(0.0, accumulated))
    prefix = op.Concat(zero, op.CumSum(accumulated, op.Constant(value_int=0)), axis=0)
    start = op.Gather(spans, op.Constant(value_int=0), axis=1)
    end = op.Gather(spans, op.Constant(value_int=1), axis=1)
    total = op.Sub(op.Gather(prefix, end), op.Gather(prefix, start))
    mean = op.Div(total, op.Unsqueeze(op.CastLike(op.Sub(end, start), accumulated), [1]))
    return op.CastLike(mean, values)


def _normalize(op, value, epsilon):
    norm = op.Sqrt(op.ReduceSum(op.Mul(value, value), [-1], keepdims=1))
    return op.Div(value, op.Max(norm, op.CastLike(epsilon, value)))


class ClefJointHead(nn.Module):
    """Joint decision head for one unpadded record with ragged flat options.

    Inputs are ``hidden_states [1,S,H]``, ``input_ids [1,S]``,
    ``question_spans [Q,2]``, ``option_spans [O,2]``,
    ``option_question_ids [O]`` and ``question_types [Q]``. Spans are
    nonempty half-open intervals into the complete encoded prompt.
    Returns flat ``logits [O]`` and per-question ``probabilities [O]``.
    """

    def __init__(self, config: ClefConfig):
        super().__init__()
        hidden, width = config.hidden_size, config.head_width
        self.width = width
        self.hidden_norm = LayerNorm(hidden, eps=1e-5)
        self.memory_projection = Linear(hidden, width, bias=False)
        self.question_projection = Linear(hidden, width, bias=False)
        self.option_question_projection = Linear(hidden, width, bias=False)
        self.global_projection = Linear(hidden, width, bias=False)
        self.option_context_projection = Linear(hidden, width, bias=False)
        self.option_lexical_projection = Linear(hidden, width, bias=False)
        self.type_embedding = Embedding(3, width)
        self.evidence_layers = nn.ModuleList(
            [_EvidenceLayer(config) for _ in range(config.head_routing_layers)]
        )
        self.option_summary_norm = LayerNorm(width, eps=1e-5)
        self.layers = nn.ModuleList([_FieldLayer(config) for _ in range(config.head_layers)])
        self.field_norm = LayerNorm(width, eps=1e-5)
        self.option_norm = LayerNorm(width, eps=1e-5)
        self.residual_scorer = nn.ModuleList(
            [Linear(width * 4, width), nn.Module(), nn.Module(), Linear(width, 1)]
        )
        self.prior_logit_scale = nn.Parameter([])
        self.joint_logit_scale = nn.Parameter([])
        self.residual_gate = nn.Parameter([])
        # This is the output LM table, not the input token embedding table.
        self.output_embedding = Embedding(config.vocab_size, hidden)

    def forward(
        self,
        op,
        hidden_states,
        input_ids,
        question_spans,
        option_spans,
        option_question_ids,
        question_types,
    ):
        hidden = op.Squeeze(self.hidden_norm(op, hidden_states), [0])  # (S,H)
        lexical = op.Squeeze(self.output_embedding(op, input_ids), [0])
        questions = _span_mean(op, hidden, question_spans)  # (Q,H)
        contexts = _span_mean(op, hidden, option_spans)  # (O,H)
        lexical = _span_mean(op, lexical, option_spans)
        global_vector = op.Squeeze(op.Slice(hidden, [-1], [2**63 - 1], axes=[0]), [0])
        memory = op.Unsqueeze(self.memory_projection(op, hidden), [0])
        option_questions = op.Gather(questions, option_question_ids)
        routed = op.Add(
            op.Add(
                self.option_context_projection(op, contexts),
                self.option_lexical_projection(op, lexical),
            ),
            self.option_question_projection(op, option_questions),
        )
        routed = op.Unsqueeze(routed, [0])
        for layer in self.evidence_layers:
            routed = layer(op, routed, memory)
        routed = op.Squeeze(routed, [0])  # (O,W)
        base_fields = self.question_projection(op, questions)
        question_indices = op.Range(
            op.Constant(value_int=0),
            op.Squeeze(op.Shape(questions, start=0, end=1)),
            op.Constant(value_int=1),
        )
        membership = op.Equal(
            op.Unsqueeze(question_indices, [1]), op.Unsqueeze(option_question_ids, [0])
        )  # (Q,O), all questions have at least one option
        routing_logits = op.Div(
            op.MatMul(base_fields, op.Transpose(routed, perm=[1, 0])),
            op.CastLike(math.sqrt(self.width), routed),
        )
        routing_weights = op.Softmax(
            op.Where(membership, routing_logits, op.CastLike(float("-inf"), routed)),
            axis=-1,
        )
        fields = op.Add(
            op.Add(
                base_fields,
                self.option_summary_norm(op, op.MatMul(routing_weights, routed)),
            ),
            op.Add(
                self.global_projection(op, global_vector),
                self.type_embedding(op, question_types),
            ),
        )
        fields = op.Unsqueeze(fields, [0])
        for layer in self.layers:
            fields = layer(op, fields, memory)
        fields = op.Gather(op.Squeeze(self.field_norm(op, fields), [0]), option_question_ids)
        options = self.option_norm(op, routed)
        # PyTorch normalize and cosine_similarity use different epsilon values.
        prior = op.ReduceSum(
            op.Mul(
                _normalize(op, op.Add(option_questions, global_vector), 1e-12),
                _normalize(op, lexical, 1e-12),
            ),
            [-1],
            keepdims=0,
        )
        cosine = op.ReduceSum(
            op.Mul(_normalize(op, fields, 1e-8), _normalize(op, options, 1e-8)),
            [-1],
            keepdims=0,
        )
        features = op.Concat(
            fields,
            options,
            op.Mul(fields, options),
            op.Abs(op.Sub(fields, options)),
            axis=-1,
        )
        residual = self.residual_scorer[0](op, features)
        residual = op.Squeeze(self.residual_scorer[3](op, op.Gelu(residual)), [-1])
        prior_scale = op.Exp(
            op.Min(
                self.prior_logit_scale, op.CastLike(math.log(100.0), self.prior_logit_scale)
            )
        )
        joint_scale = op.Exp(
            op.Min(
                self.joint_logit_scale, op.CastLike(math.log(100.0), self.joint_logit_scale)
            )
        )
        logits = op.Add(
            op.Mul(prior_scale, prior),
            op.Mul(
                op.Sigmoid(self.residual_gate),
                op.Add(op.Mul(joint_scale, cosine), residual),
            ),
        )
        grouped = op.Softmax(
            op.Where(
                membership, op.Unsqueeze(logits, [0]), op.CastLike(float("-inf"), logits)
            ),
            axis=-1,
        )
        return logits, op.ReduceSum(grouped, [0], keepdims=0)


def clef_head_source_name(name: str) -> tuple[str, int | None]:
    """Map split attention parameters back to PyTorch's fused QKV tensors."""
    for index, projection in enumerate(("q_proj", "k_proj", "v_proj")):
        for suffix in ("weight", "bias"):
            ending = f".{projection}.{suffix}"
            if name.endswith(ending):
                return name[: -len(ending)] + f".in_proj_{suffix}", index
    return name, None


class ClefFlashModel(Qwen35VL3ModelCausalLMModel):
    """Clef-Flash: multimodal Qwen3.5-9B plus a joint schema decision head.

    Exports a full-record hidden-state backbone, vision encoder, embedding
    mixer, and joint decision head. Host schema encoding and grouped
    probabilities replace autoregressive text generation.
    """

    default_task = "clef-decision"
    category = "Multimodal"
    config_class = ClefConfig
    HF_COMPONENT_SOURCES: ClassVar[dict[str, tuple[str, ...]]] = {
        **Qwen35VL3ModelCausalLMModel.HF_COMPONENT_SOURCES,
        "decision_head": ("head", "lm_head"),
    }

    def __init__(self, config: ClefConfig):
        if config.deepstack_visual_indexes:
            raise ValueError("Clef-Flash does not use DeepStack vision features")
        super().__init__(config)
        self.decoder = _ClefBackbone(config)
        self.embedding = Qwen25VLEmbeddingModel(config)
        self.decision_head = ClefJointHead(config)

    def preprocess_weights(self, state_dict: dict[str, torch.Tensor]):
        backbone = {k: v for k, v in state_dict.items() if not k.startswith("head.")}
        result = super().preprocess_weights(backbone)
        if "decoder.lm_head.weight" in result:
            result["decision_head.output_embedding.weight"] = result.pop(
                "decoder.lm_head.weight"
            )
        for name, _ in self.decision_head.named_parameters():
            if name == "output_embedding.weight":
                continue
            source, split = clef_head_source_name(name)
            key = f"head.{source}"
            if key in state_dict:
                tensor = state_dict[key]
                if split is not None:
                    tensor = tensor.chunk(3, dim=0)[split]
                result[f"decision_head.{name}"] = tensor
        return result


class _ClefBackbone(Qwen35VLDecoderModel):
    def forward(
        self,
        op,
        inputs_embeds,
        attention_mask,
        position_ids,
        past_key_values=None,
    ):
        return self.model(
            op,
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
