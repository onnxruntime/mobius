# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""GraniteSWA: Granite with sliding-window attention and learnable attention sinks.

GraniteSWA (``ibm-granite/granite-swash-2b``) is the Granite decoder-only
architecture plus three changes:

* **Mixed attention span.** ``config.layer_types`` selects, per layer, either
  full causal attention or a ``config.sliding_window``-wide local window.
* **Learnable per-head attention sinks.** Every layer owns a
  ``sinks[num_attention_heads]`` parameter that adds one extra logit to the
  softmax denominator, letting a head shed probability mass instead of being
  forced to distribute it over real tokens.
* **Per-layer RoPE base.** ``config.layer_rope_theta[i]`` gives layer ``i`` its
  own RoPE base frequency; a value of ``0`` marks a NoPE layer that receives no
  positional rotation at all.

Everything else follows Granite, including the four scaling multipliers
(``embedding_multiplier``, ``attention_multiplier``, ``logits_scaling``,
``residual_multiplier``).

HuggingFace reference: ``GraniteSWAForCausalLM``.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig, GraniteSwaConfig
from mobius.components import (
    Float32SinkAttention,
    Float32SlidingWindowSinkAttention,
    RMSNorm,
    create_attention_bias,
    create_decoder_layer,
    initialize_rope,
)
from mobius.models.base import CausalLMModel, embedding_for_config, linear_class_for_config

if TYPE_CHECKING:
    import onnx_ir as ir


def resolve_layer_rope_theta(config: ArchitectureConfig) -> list[float | int]:
    """Return the per-layer RoPE base frequency, with ``0`` marking NoPE.

    ``layer_rope_theta`` only exists on :class:`~mobius._configs.GraniteSwaConfig`,
    and even there it is optional, so a plain :class:`ArchitectureConfig` (used
    by the tiny graph-build fixtures) must still resolve to something sensible.
    ``no_rope_layers`` names the same set of layers, so prefer it before falling
    back to rotating every layer at the global ``rope_theta``.
    """
    layer_rope_theta = getattr(config, "layer_rope_theta", None)
    if layer_rope_theta:
        return list(layer_rope_theta)

    no_rope_layers = set(config.no_rope_layers or ())
    return [
        0 if index in no_rope_layers else (config.rope_theta or 0)
        for index in range(config.num_hidden_layers)
    ]


class GraniteSwaTextModel(nn.Module):
    """GraniteSWA backbone: mixed full/sliding attention with per-layer RoPE.

    Replicates HuggingFace ``GraniteSWAModel``: scale the embeddings by
    ``embedding_multiplier``, build one causal mask per attention span, build one
    ``(cos, sin)`` table per distinct non-zero RoPE base, and dispatch both
    per layer.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self._dtype = config.dtype
        self._layer_types = config.layer_types or [
            "full_attention" if index % 4 == 0 else "sliding_attention"
            for index in range(config.num_hidden_layers)
        ]
        self._layer_rope_theta = resolve_layer_rope_theta(config)
        self._sliding_window = config.sliding_window
        self.embedding_multiplier = config.embedding_multiplier
        self.output_layer_indices: list[int] | None = (
            list(config.output_layer_indices or []) or None
        )

        self.embed_tokens = embedding_for_config(config)
        linear_class = linear_class_for_config(config)
        self.layers = nn.ModuleList(
            [
                # Float32SinkAttention, not SinkAttention: GraniteSWA's eager
                # kernel forces the sink scaling and the softmax to float32.
                create_decoder_layer(
                    config,
                    attention_class=(
                        Float32SlidingWindowSinkAttention
                        if layer_type == "sliding_attention"
                        else Float32SinkAttention
                    ),
                    linear_class=linear_class,
                )
                for layer_type in self._layer_types
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # One rotary module per distinct non-zero base theta, mirroring HF's
        # ``GraniteSWAModel.rotary_embs``.  ``sorted`` keeps construction (and
        # therefore initializer naming) deterministic across builds.
        self._rope_thetas = sorted({theta for theta in self._layer_rope_theta if theta})
        self.rotary_embs = nn.ModuleList(
            [
                initialize_rope(dataclasses.replace(config, rope_theta=float(theta)))
                for theta in self._rope_thetas
            ]
        )
        # ``TextModel``-style single-rope attribute, kept so that generic
        # tooling that expects ``model.rotary_emb`` (metadata emitters, GQA
        # rewrite guards) sees the common case.  ``None`` when every layer is
        # NoPE.
        self.rotary_emb = self.rotary_embs[0] if self.rotary_embs else None

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value | None,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        inputs_embeds: ir.Value | None = None,
    ):
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(op, input_ids)
        # Granite embedding multiplier, applied before the first decoder layer.
        hidden_states = op.Mul(hidden_states, self.embedding_multiplier)

        # SinkAttention builds the score matrix explicitly, so the bias must
        # carry the FULL mask (causal + window + padding); there is no
        # ``is_causal`` flag to fall back on.
        mask_source = position_ids if input_ids is None else input_ids
        full_attention_bias = create_attention_bias(
            op,
            input_ids=mask_source,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )  # (B, 1, S_q, S_kv)
        sliding_attention_bias = full_attention_bias
        if self._sliding_window and "sliding_attention" in self._layer_types:
            # Sliding layers retain only ``window - 1`` past entries. Trim the
            # input mask to that cache plus the current query chunk so the
            # additive-bias key axis exactly matches their local KV tensor.
            query_len = op.Shape(mask_source, start=1, end=2)
            local_length = op.Add(query_len, self._sliding_window - 1)
            mask_length = op.Shape(attention_mask, start=1, end=2)
            local_attention_mask = op.Slice(
                attention_mask, op.Neg(local_length), mask_length, [1]
            )
            sliding_attention_bias = create_attention_bias(
                op,
                input_ids=mask_source,
                attention_mask=local_attention_mask,
                sliding_window=self._sliding_window,
                dtype=self._dtype,
            )  # (B, 1, S_q, S_kv)

        # (cos, sin) per distinct non-zero base theta; NoPE layers get None.
        position_embeddings_by_theta = {
            theta: rotary_emb(op, position_ids)
            for theta, rotary_emb in zip(self._rope_thetas, self.rotary_embs)
        }

        present_key_values = []
        if self.output_layer_indices is not None:
            num_layers = len(self.layers)
            if len(set(self.output_layer_indices)) != len(self.output_layer_indices):
                raise ValueError(
                    "output_layer_indices must not contain duplicates: "
                    f"{self.output_layer_indices}"
                )
            out_of_range = [
                index for index in self.output_layer_indices if not 0 <= index < num_layers
            ]
            if out_of_range:
                raise ValueError(
                    f"output_layer_indices {out_of_range} out of range [0, {num_layers})"
                )
        capture_set = set(self.output_layer_indices or ())
        captured_by_index: dict[int, ir.Value] = {}
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_kvs)):
            is_sliding = self._layer_types[layer_idx] == "sliding_attention"
            theta = self._layer_rope_theta[layer_idx]
            hidden_states, present_key_value = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=(sliding_attention_bias if is_sliding else full_attention_bias),
                position_embeddings=position_embeddings_by_theta.get(theta),
                past_key_value=past_key_value,
            )
            present_key_values.append(present_key_value)
            if layer_idx in capture_set:
                # Capture the post-residual layer output before final RMSNorm,
                # matching TextModel and the draft-target hidden-state contract.
                captured_by_index[layer_idx] = hidden_states

        hidden_states = self.norm(op, hidden_states)
        if self.output_layer_indices is not None:
            ordered = [captured_by_index[index] for index in self.output_layer_indices]
            return hidden_states, present_key_values, ordered
        return hidden_states, present_key_values


class GraniteSwaCausalLMModel(CausalLMModel):
    """GraniteSWA causal LM: Granite scaling + sliding windows + attention sinks.

    Extends the Granite architecture with per-layer sliding-window attention,
    a learnable per-head attention sink folded into the softmax denominator,
    and a per-layer RoPE base frequency (``0`` = NoPE).  The four Granite
    scaling multipliers still apply:

    - ``embedding_multiplier``: scales embeddings after lookup
    - ``attention_multiplier``: replaces ``1/sqrt(head_dim)`` as attention scale
    - ``residual_multiplier``: scales attention/MLP outputs before residual add
    - ``logits_scaling``: divides the final logits

    HuggingFace model_type: ``granite_swa`` (``GraniteSWAForCausalLM``).
    """

    # Declared here as well as on the registry entry: the registry-consistency
    # check requires the two to agree, and inheriting CausalLMConfig would
    # silently drop ``layer_rope_theta``.
    config_class: type = GraniteSwaConfig
    # SinkAttention needs an explicit score matrix to include the virtual sink
    # in its softmax denominator; the external static-cache Attention path
    # cannot represent that extra logit.
    _supports_static_cache: bool = False

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        self._replace_text_model(GraniteSwaTextModel(config))
        self.logits_scaling = config.logits_scaling

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        logits, *rest = super().forward(
            op, input_ids, attention_mask, position_ids, past_key_values
        )
        # Granite logits scaling while preserving optional hidden-state outputs.
        return op.Div(logits, self.logits_scaling), *rest
