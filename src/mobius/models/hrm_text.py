# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""HRM-Text hierarchical recurrent text models (``sapientinc/HRM-Text-1B``).

Replicates HuggingFace's ``HrmTextForCausalLM``.  HRM-Text is *not* a plain
stack of decoder layers: it owns two independently-weighted transformer
stacks — a fast "low" stack (``L_module``) and a slow "high" stack
(``H_module``) — and runs them in a fixed recurrence::

    z_H = embed(input_ids) * embedding_scale
    z_L = z_L_init                                  # broadcast over (B, S, H)
    for h in range(H_cycles):
        for l in range(L_cycles):
            z_L = L_module(z_L + z_H)
        z_H = H_module(z_H + z_L)
    logits = lm_head(z_H)

Every stack invocation runs all ``num_layers_per_stack`` blocks and therefore
performs its own attention with its own KV-cache slots.  Upstream reflects that
by inflating ``config.num_hidden_layers`` to
``num_layers_per_stack * H_cycles * (L_cycles + 1)`` so ``DynamicCache``
allocates one slot per unique attention invocation; the exported ONNX graph
uses the same inflated count and emits the slots in the same order, so the
graph's ``past_key_values.{i}`` / ``present.{i}`` indices line up 1:1 with
HuggingFace's cache layout.

Architectural differences from the standard :class:`CausalLMModel`:

* **Parameterless RMSNorm** — ``HrmTextRMSNorm`` has no learnable scale and
  normalises in float32, matching :class:`ScaleFreeRMSNorm`.
* **Gated attention output** — an extra ``gate_proj`` produces a per-element
  sigmoid gate applied to the attention result before ``o_proj``.
* **Embedding scaling** — token embeddings are multiplied by
  ``config.embedding_scale`` (``1 / initializer_range`` when unset).
* **MHA, not GQA** — ``k_proj``/``v_proj`` are sized by ``num_attention_heads``.
* **No trailing model norm** — each stack ends with its own ``final_norm``, so
  the last ``H_module`` output feeds ``lm_head`` directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig, HrmTextConfig
from mobius._weight_utils import split_gate_up_proj
from mobius.components import (
    Attention,
    DecoderLayer,
    Embedding,
    Linear,
    ScaleFreeRMSNorm,
    create_attention_bias,
    initialize_rope,
)
from mobius.models.base import CausalLMModel, TextModel, _retain_last_sequence_token

if TYPE_CHECKING:
    import onnx_ir as ir

# Order of the four equally-sized chunks packed into the checkpoint's fused
# ``attn.gqkv_proj`` weight, matching HuggingFace's ``hrm_text`` entry in
# ``transformers.conversion_mapping`` (``Chunk(dim=0)`` over gate/q/k/v).
_FUSED_GQKV_PARTS: tuple[str, ...] = ("gate_proj", "q_proj", "k_proj", "v_proj")

_FUSED_GQKV_SUFFIX = ".attn.gqkv_proj.weight"
_FUSED_GATE_UP_SUFFIX = ".mlp.gate_up_proj.weight"


class HrmTextAttention(Attention):
    """Multi-head attention with a sigmoid output gate.

    Identical to the base :class:`~mobius.components.Attention` except for an
    extra ``gate_proj`` that is driven by the *same* (already normalised)
    layer input as Q/K/V.  Upstream applies the gate on the per-head
    ``(B, S, num_heads, head_dim)`` view; because both tensors use that exact
    layout, the elementwise product is identical on the flattened
    ``(B, S, num_heads * head_dim)`` view used here.
    """

    def __init__(self, config: ArchitectureConfig, linear_class: type | None = None):
        super().__init__(config, linear_class=linear_class)
        gate_linear = linear_class if linear_class is not None else Linear
        self.gate_proj = gate_linear(
            self.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=config.attn_qkv_bias,
        )
        # Under PrefixLM the model feeds a float additive bias that already
        # bakes in causal + padding + the bidirectional prefix overlay, so the
        # Attention op must not re-apply its own causal mask (that would cancel
        # the future-position unmasking). Mirrors Gemma4's vision-block overlay.
        if getattr(config, "prefix_lm", False):
            self._is_causal = 0

    def _post_attention(
        self,
        op: OpBuilder,
        attn_output: ir.Value,
        hidden_states: ir.Value,
    ) -> ir.Value:
        # gate: (B, S, num_heads * head_dim); attn_output has the same shape.
        gate = self.gate_proj(op, hidden_states)
        return op.Mul(attn_output, op.Sigmoid(gate))


class HrmTextDecoderLayer(DecoderLayer):
    """Pre-norm decoder layer with parameterless norms and gated attention.

    Replicates HuggingFace's ``HrmTextDecoderLayer``: ``input_layernorm`` →
    gated self-attention → residual → ``post_attention_layernorm`` → SwiGLU
    MLP → residual.  Both norms are scale-free RMSNorms, so the checkpoint
    ships no weights for them.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config, norm_class=ScaleFreeRMSNorm)
        # Replace the plain attention built by DecoderLayer with the gated
        # variant; the discarded module's parameters are dropped with it.
        self.self_attn = HrmTextAttention(config)


class HrmTextStack(nn.Module):
    """One HRM transformer stack (used twice: as ``L_module`` and ``H_module``).

    Replicates HuggingFace's ``HrmTextStack``: ``num_layers_per_stack``
    decoder layers followed by a parameterless ``final_norm``.  The same
    instance is invoked several times per forward pass, once per recurrence
    step, so its parameters appear exactly once in the ONNX graph while its
    ops are unrolled per invocation.
    """

    def __init__(self, config: ArchitectureConfig, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([HrmTextDecoderLayer(config) for _ in range(num_layers)])
        self.final_norm = ScaleFreeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias,
        position_embeddings: tuple | None,
        past_key_values: list,
    ):
        """Run every block in the stack over one recurrence step.

        Args:
            past_key_values: Exactly ``len(self.layers)`` cache entries — the
                slice of the global cache belonging to *this* invocation.

        Returns:
            ``(hidden_states, present_key_values)`` where ``hidden_states`` is
            ``(B, S, hidden)`` after ``final_norm``.
        """
        if len(past_key_values) != len(self.layers):
            raise ValueError(
                f"HrmTextStack expected {len(self.layers)} cache entries for this "
                f"invocation, got {len(past_key_values)}"
            )
        present_key_values = []
        for layer, past_kv in zip(self.layers, past_key_values):
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)
        return self.final_norm(op, hidden_states), present_key_values


class HrmTextModel(TextModel):
    """HRM-Text backbone: embeddings + the H/L recurrence over two stacks.

    Replicates HuggingFace's ``HrmTextModel``.  Subclasses
    :class:`~mobius.models.base.TextModel` purely to reuse its EP-aware
    attention-context construction (GQA fusion vs. bool padding mask vs.
    static-cache bias); the layer schedule itself is completely different, so
    ``forward`` is overridden and ``self.layers`` is intentionally absent.
    """

    def __init__(self, config: ArchitectureConfig):
        # Deliberately skip TextModel.__init__: it would build a flat
        # ``self.layers`` stack and a trailing ``self.norm`` that HRM-Text
        # does not have.
        nn.Module.__init__(self)
        self.config = config
        self._dtype = config.dtype
        self.output_layer_indices = None
        # HRM-Text uses full causal attention; no sliding window. Required by
        # the inherited ``_maybe_static_cache_bias`` / ``_gqa_local_window_size``.
        self._sliding_window = None

        self._h_cycles = int(config.H_cycles)
        self._l_cycles = int(config.L_cycles)
        self._layers_per_stack = _resolve_layers_per_stack(config)
        self._embedding_scale = float(config.embedding_scale)
        self._prefix_lm = bool(getattr(config, "prefix_lm", False))

        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.rotary_emb = initialize_rope(config)
        self.L_module = HrmTextStack(config, self._layers_per_stack)
        self.H_module = HrmTextStack(config, self._layers_per_stack)
        # Frozen initial low-cycle state, stored as a (hidden,) vector and
        # broadcast against (B, S, hidden) — equivalent to HF's ``expand_as``.
        self.z_L_init = nn.Parameter([config.hidden_size], dtype=config.dtype)

    def _build_prefix_lm_context(
        self,
        op: OpBuilder,
        *,
        input_ids: ir.Value,
        attention_mask: ir.Value | None,
        position_ids: ir.Value,
        token_type_ids: ir.Value | None,
    ) -> tuple[ir.Value, tuple]:
        """Build the PrefixLM float attention bias and the RoPE embeddings.

        Reproduces upstream ``HrmTextModel.forward``, which converts
        ``token_type_ids`` into ``block_sequence_ids = where(tt == 1, 0, -1)``
        and hands it to ``create_causal_mask``.  HuggingFace's
        ``blockwise_overlay`` unmasks a ``(q, kv)`` pair when
        ``(q_group == kv_group) & (q_group >= 0)``, OR-ed onto the causal mask;
        :func:`~mobius.components.create_attention_bias` implements exactly
        that, so prompt tokens (``token_type_ids == 1``, block ``0``) attend to
        each other bidirectionally while everything else stays causal.

        Upstream gates the overlay on ``is_first_iteration``.  The exported
        graph reproduces that behaviour *by data* rather than by branching:
        a generated token is fed with ``token_type_ids == 0`` -> block ``-1``,
        and the ``q_group >= 0`` guard makes the overlay a pure no-op for it,
        leaving plain causal attention on every decode step.  Passing all-zero
        ``token_type_ids`` therefore reproduces HuggingFace's
        ``token_type_ids=None`` (fully causal) path exactly, so one graph
        serves both contracts.

        The bias bakes in causal + padding + prefix-block masking, so
        :class:`HrmTextAttention` runs with ``is_causal=0``; that also rules out
        ``GroupQueryAttention``, whose mask model is causal/local-window only.
        """
        if attention_mask is None:
            # Static cache passes attention_mask=None and relies on
            # is_causal=1 + nonpad_kv_seqlen, which cannot express the
            # bidirectional prefix block.
            raise NotImplementedError(
                "HRM-Text PrefixLM masking is not supported with the static KV "
                "cache: the bidirectional prefix overlay needs the dynamic "
                "attention_mask to build its float additive bias. Build with "
                "the default dynamic cache, or set config.prefix_lm=False for a "
                "fully causal export."
            )
        if token_type_ids is None:
            raise ValueError(
                "HRM-Text was built with config.prefix_lm=True, which requires a "
                "token_type_ids input. The task supplies it via the module's "
                "requires_token_type_ids hook; a caller invoking forward() "
                "directly must pass token_type_ids explicitly."
            )

        position_embeddings = self.rotary_emb(op, position_ids)

        # block_sequence_ids: (B, S_q) int64 — 0 for prefix tokens, -1 otherwise.
        prefix_marker = op.Constant(value_ints=[1])
        block_id = op.Constant(value_ints=[0])
        non_block_id = op.Constant(value_ints=[-1])
        block_sequence_ids = op.Where(
            op.Equal(token_type_ids, prefix_marker), block_id, non_block_id
        )

        # (B, 1, S_q, past + S_q) additive float bias.
        attention_bias = create_attention_bias(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            dtype=self._dtype,
            block_sequence_ids=block_sequence_ids,
        )
        return attention_bias, position_embeddings

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value | None,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        inputs_embeds: ir.Value | None = None,
        token_type_ids: ir.Value | None = None,
    ):
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(op, input_ids)
        # Token-embedding multiplier (1 / initializer_range for this family).
        # z_H — slow / high-level state: (B, S, hidden)
        z_high = op.Mul(hidden_states, self._embedding_scale)

        if self._prefix_lm:
            attention_bias, position_embeddings = self._build_prefix_lm_context(
                op,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
            )
        else:
            attention_bias, position_embeddings = self._build_attention_context(
                op,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                hidden_states=z_high,
                past_key_values=past_key_values,
            )

        num_invocations = self._h_cycles * (self._l_cycles + 1)
        expected_slots = num_invocations * self._layers_per_stack
        cache = past_key_values if past_key_values is not None else [None] * expected_slots
        if len(cache) != expected_slots:
            raise ValueError(
                f"HRM-Text expects {expected_slots} KV-cache slots "
                f"({num_invocations} stack invocations x {self._layers_per_stack} "
                f"layers), got {len(cache)}"
            )

        # z_L — fast / low-level state. Starts as the frozen (hidden,) vector
        # and is broadcast by the first Add against z_high.
        z_low: ir.Value = self.z_L_init
        present_key_values: list = []
        # Cache slots are consumed strictly in invocation order, which
        # reproduces upstream's
        #   slot(h, l, layer) = (h * (L_cycles + 1) + l) * layers_per_stack + layer
        # (the trailing H invocation of cycle h uses l == L_cycles).
        slot = 0
        for _ in range(self._h_cycles):
            for _ in range(self._l_cycles):
                next_slot = slot + self._layers_per_stack
                z_low, presents = self.L_module(
                    op,
                    hidden_states=op.Add(z_low, z_high),
                    attention_bias=attention_bias,
                    position_embeddings=position_embeddings,
                    past_key_values=cache[slot:next_slot],
                )
                present_key_values.extend(presents)
                slot = next_slot

            next_slot = slot + self._layers_per_stack
            z_high, presents = self.H_module(
                op,
                hidden_states=op.Add(z_high, z_low),
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_values=cache[slot:next_slot],
            )
            present_key_values.extend(presents)
            slot = next_slot

        # No trailing model-level norm: H_module.final_norm already ran.
        return z_high, present_key_values


class HrmTextCausalLMModel(CausalLMModel):
    """Causal LM head over the HRM-Text hierarchical recurrent backbone.

    Replicates HuggingFace's ``HrmTextForCausalLM`` (``sapientinc/HRM-Text-1B``):
    two recurrently-invoked transformer stacks with parameterless RMSNorm,
    sigmoid-gated attention output, and scaled token embeddings.

    Inputs: ``input_ids``, ``attention_mask``, ``position_ids``,
    ``past_key_values``, plus ``token_type_ids`` when ``config.prefix_lm`` is
    set (the pre-training PrefixLM mask: ``1`` marks a prompt position that
    attends bidirectionally within the prefix block, anything else stays
    causal).  Outputs: ``logits`` and one present KV pair per unique attention
    invocation of the recurrence.
    """

    config_class: type = HrmTextConfig
    # Class-level default so test/golden harnesses can tell, without building
    # the model, that this architecture's graph carries a ``token_type_ids``
    # input. ``__init__`` narrows it per instance from ``config.prefix_lm``.
    requires_token_type_ids: bool = True

    def __init__(self, config: ArchitectureConfig):
        nn.Module.__init__(self)
        self.config = config
        self.model = HrmTextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        # Task hook: ask CausalLMTask for the extra ``token_type_ids`` graph
        # input. Only meaningful under PrefixLM; a purely causal HRM-Text
        # export keeps the standard input set.
        self.requires_token_type_ids = bool(getattr(config, "prefix_lm", False))

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value | None,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        token_type_ids: ir.Value | None = None,
    ):
        hidden_states, present_key_values = self.model(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            token_type_ids=token_type_ids,
        )
        # Honour prefill-prefix pruning exactly like CausalLMModel.forward.
        hidden_states = _retain_last_sequence_token(op, hidden_states)
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HRM-Text checkpoint keys onto this module tree.

        Handles both layouts that reach us:

        * The **raw checkpoint** (``sapientinc/HRM-Text-1B``), which packs
          ``attn.gqkv_proj`` (gate/q/k/v concatenated on dim 0) and
          ``mlp.gate_up_proj``, and names the attention submodule ``attn``.
        * An **already-converted** HuggingFace ``state_dict``, whose keys equal
          our ONNX parameter names — that path is a pure identity mapping.
        """
        head_width = self.config.num_attention_heads * self.config.head_dim
        intermediate_size = self.config.intermediate_size
        converted: dict[str, torch.Tensor] = {}

        for key, value in state_dict.items():
            if key.endswith(_FUSED_GQKV_SUFFIX):
                prefix = key[: -len(_FUSED_GQKV_SUFFIX)]
                expected = len(_FUSED_GQKV_PARTS) * head_width
                if value.shape[0] != expected:
                    raise ValueError(
                        f"{key}: fused gqkv_proj dim 0 is {value.shape[0]}, expected "
                        f"{expected} ({len(_FUSED_GQKV_PARTS)} x "
                        f"num_attention_heads * head_dim = {head_width})"
                    )
                for index, part in enumerate(_FUSED_GQKV_PARTS):
                    start = index * head_width
                    converted[f"{prefix}.self_attn.{part}.weight"] = value[
                        start : start + head_width
                    ]
                continue

            if key.endswith(_FUSED_GATE_UP_SUFFIX):
                prefix = key[: -len(_FUSED_GATE_UP_SUFFIX)]
                gate, up = split_gate_up_proj(value, intermediate_size)
                converted[f"{prefix}.mlp.gate_proj.weight"] = gate
                converted[f"{prefix}.mlp.up_proj.weight"] = up
                continue

            # ``.attn.`` never matches the already-converted ``.self_attn.``
            # spelling, so this rename is idempotent.
            converted[key.replace(".attn.", ".self_attn.")] = value

        return super().preprocess_weights(converted)


def _resolve_layers_per_stack(config: ArchitectureConfig) -> int:
    """Return the number of transformer blocks inside each H / L stack.

    ``config.num_hidden_layers`` is the *inflated* per-invocation count, so the
    real per-stack depth comes from ``num_layers_per_stack`` when the config
    carries it and is otherwise derived back out of the inflated total.
    """
    per_stack = getattr(config, "num_layers_per_stack", None)
    h_cycles = int(config.H_cycles)
    l_cycles = int(config.L_cycles)
    invocations = h_cycles * (l_cycles + 1)
    if per_stack is None:
        per_stack, remainder = divmod(int(config.num_hidden_layers), invocations)
        if remainder or per_stack <= 0:
            raise ValueError(
                f"HRM-Text num_hidden_layers ({config.num_hidden_layers}) is not a "
                f"positive multiple of H_cycles * (L_cycles + 1) = {invocations}"
            )
        return per_stack

    per_stack = int(per_stack)
    expected_total = per_stack * invocations
    if int(config.num_hidden_layers) != expected_total:
        raise ValueError(
            f"HRM-Text num_hidden_layers ({config.num_hidden_layers}) must equal "
            f"num_layers_per_stack * H_cycles * (L_cycles + 1) = {expected_total}"
        )
    return per_stack
