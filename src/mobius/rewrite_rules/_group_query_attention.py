# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Rewrite rules for fusing Attention into GroupQueryAttention.

Two rules are provided, tried in order:

1. **RotaryAttentionToGQA** — the optimal path.  Matches the pattern
   ``RotaryEmbedding(q) + RotaryEmbedding(k) + Attention`` where cos/sin
   are gathered from a position-indexed cache table.  Produces GQA with
   ``do_rotary=1`` so the rotary op is fused into the kernel.

2. **AttentionToGQA** — universal fallback.  Matches any ``Attention``
   node that has KV-cache inputs (decoder attention), regardless of how
   Q/K position embeddings are computed.  This covers models with
   non-standard rotary implementations (e.g. Qwen3.5 3D mRoPE via
   ``Where`` nodes).  Produces GQA with ``do_rotary=0``; the existing
   position-embedding nodes remain in the graph and feed directly into
   the GQA ``query``/``key`` slots.

Two rules (``PackQKVForGQA`` and ``PackQKVWithBiasForGQA``) run after
the GQA fusion and consolidate separate Q, K, V projection MatMuls into
a single packed MatMul when they share the same hidden_states input.
``PackQKVForGQA`` handles models without bias (Llama, Gemma) and
``PackQKVWithBiasForGQA`` handles models with QKV bias (Qwen2.5, Phi).
The packed QKV tensor is passed in the ``query`` slot of
``GroupQueryAttention`` with ``key`` and ``value`` set to ``None``.
Models with QK norm (e.g. Qwen3) are unaffected because the Q/K
projections are followed by a normalization op, so the pattern does not
match.

These rules are applied automatically by
:func:`~mobius._optimizations.optimize_model` when the EP's ``gqa_dtypes``
or ``qkv_pack_dtypes`` includes the current model dtype (decoder role only).
They can also be applied manually::

    from mobius.rewrite_rules import group_query_attention_rules
    from onnxscript.rewriter import rewrite

    model = build("Qwen/Qwen3-0.6B")
    rewrite(model, pattern_rewrite_rules=group_query_attention_rules())
"""

from __future__ import annotations

from onnxscript.rewriter._basics import MatchFailureError, MatchResult
from onnxscript.rewriter._rewrite_rule import (
    RewriteRuleClassBase,
    RewriteRuleSet,
)

# CUDA EP's GroupQueryAttention kernel historically enforced MAX_HEAD_SIZE = 256.
# Latest ORT (unreleased) supports head_dim=512 for Gemma4 full-attention layers.
# The rewrite rule (Attention → GQA) still skips head_dim > 512 to avoid
# generating nodes that would fail on older runtimes.
_MAX_GQA_HEAD_DIM = 512


def _head_dim_exceeds_gqa_limit(past_key) -> int | None:
    """Return the head_dim if it exceeds the GQA kernel limit, else None.

    ``past_key`` is expected to have shape ``(batch, kv_heads, seq, head_dim)``.
    """
    if past_key is None or past_key.shape is None or len(past_key.shape) < 4:
        return None
    hd = past_key.shape[3]
    if isinstance(hd, int) and hd > _MAX_GQA_HEAD_DIM:
        return hd
    return None


class RotaryAttentionToGQA(RewriteRuleClassBase):
    """Replace RotaryEmbedding + Attention with GroupQueryAttention.

    **Matched pattern:**

    .. code-block:: text

        q_rot = RotaryEmbedding(q_pre, cos, sin)
        k_rot = RotaryEmbedding(k_pre, cos, sin)
        attn_out, present_key, present_value = Attention(
            q_rot, k_rot, v, attention_bias, past_key, past_value,
        )

    Where ``past_key`` and ``past_value`` are graph inputs (decoder
    attention with KV cache), and ``cos`` / ``sin`` are position-gathered
    from rotary cache tables via ``Gather``.

    **Replacement:**

    .. code-block:: text

        seqlens_k = Cast(ReduceSum(attention_mask, axis=1) - 1, INT32)
        total_seq_len = Cast(Shape(attention_mask)[1], INT32)
        attn_out, present_key, present_value = GroupQueryAttention(
            q_pre, k_pre, v, past_key, past_value,
            seqlens_k, total_seq_len, cos_cache, sin_cache,
            num_heads=..., kv_num_heads=..., do_rotary=1,
        )

    ``cos_cache`` and ``sin_cache`` are the original rotary embedding
    tables (traced back through the ``Gather`` nodes).
    """

    def __init__(self):
        super().__init__()
        # Cached graph-level values shared across all GQA replacements
        self._seqlens_k = None
        self._total_seq_len = None
        self._cos_cache = None
        self._sin_cache = None

    # ------------------------------------------------------------------ pattern

    def pattern(self, op, q_pre, k_pre, v, attention_bias, past_key, past_value, cos, sin):
        q_rot = op.RotaryEmbedding(
            q_pre,
            cos,
            sin,
            _allow_other_attributes=True,
        )
        k_rot = op.RotaryEmbedding(
            k_pre,
            cos,
            sin,
            _allow_other_attributes=True,
        )
        return op.Attention(
            q_rot,
            k_rot,
            v,
            attention_bias,
            past_key,
            past_value,
            _allow_other_attributes=True,
            _outputs=["attn_out", "present_key", "present_value"],
        )

    # ------------------------------------------------------------------ check

    def check(self, context, attn_out, cos, sin, past_key, past_value, **_):
        result = MatchResult()

        attn = attn_out.producer()
        if attn.attributes.get_float("scale", None) is None:
            return result.fail("Missing scale attribute on Attention")
        if attn.attributes.get_int("q_num_heads", None) is None:
            return result.fail("Missing q_num_heads on Attention")
        if attn.attributes.get_int("kv_num_heads", None) is None:
            return result.fail("Missing kv_num_heads on Attention")

        # cos/sin must come from Gather (position-indexed cache tables)
        cos_prod = cos.producer()
        sin_prod = sin.producer()
        if cos_prod is None or cos_prod.op_type != "Gather":
            return result.fail("cos must be Gather-produced")
        if sin_prod is None or sin_prod.op_type != "Gather":
            return result.fail("sin must be Gather-produced")

        # past_key/past_value must be graph inputs (not None, not computed)
        # This distinguishes decoder attention from vision-encoder attention
        if past_key is None or past_value is None:
            return result.fail("No KV cache inputs")
        if past_key.producer() is not None:
            return result.fail("past_key is not a graph input")
        if past_value.producer() is not None:
            return result.fail("past_value is not a graph input")

        # Skip when head_dim exceeds CUDA GQA MAX_HEAD_SIZE (256).
        hd = _head_dim_exceeds_gqa_limit(past_key)
        if hd is not None:
            return result.fail(f"head_dim={hd} exceeds GQA MAX_HEAD_SIZE={_MAX_GQA_HEAD_DIM}")

        return result

    # ------------------------------------------------------------------ rewrite

    def rewrite(
        self,
        op,
        q_pre,
        k_pre,
        v,
        attention_bias,
        past_key,
        past_value,
        cos,
        sin,
        attn_out,
        present_key,
        present_value,
        **_,
    ):
        attn = attn_out.producer()
        scale = attn.attributes.get_float("scale")
        q_num_heads = attn.attributes.get_int("q_num_heads")
        kv_num_heads = attn.attributes.get_int("kv_num_heads")
        # Preserve softcap (Gemma2 logit soft-capping). Default 0.0 = disabled.
        softcap = attn.attributes.get_float("softcap", 0.0)

        # Read interleaved from the Q RotaryEmbedding op (0=half-split, 1=interleaved).
        # GLM4/ChatGLM use interleaved=1; most models use 0. Must not hardcode.
        q_rope_node = attn.inputs[0].producer()  # RotaryEmbedding producing q_rot
        rotary_interleaved = q_rope_node.attributes.get_int("interleaved", 0)

        # Trace cos/sin back through Gather to the cache table initializers
        if self._cos_cache is None:
            self._cos_cache = cos.producer().inputs[0]
            self._sin_cache = sin.producer().inputs[0]

        # Build seqlens_k and total_sequence_length once (shared)
        if self._seqlens_k is None:
            graph = attn.graph
            attention_mask = None
            for gi in graph.inputs:
                if gi.name == "attention_mask":
                    attention_mask = gi
                    break

            # seqlens_k = Cast(ReduceSum(attention_mask, axis=1) - 1, INT32)
            axis = op.Constant(value_ints=[1])
            reduce_sum = op.ReduceSum(attention_mask, axis)
            one = op.Constant(value_ints=[1])
            self._seqlens_k = op.Cast(
                op.Sub(reduce_sum, one),
                to=6,
            )

            # total_seq_len = Cast(Gather(Shape(attention_mask), 1), INT32)
            mask_shape = op.Shape(attention_mask)
            idx_1 = op.Constant(value_int=1)
            self._total_seq_len = op.Cast(
                op.Gather(mask_shape, idx_1),
                to=6,
            )

        # Create GroupQueryAttention (3 outputs match Attention's 3)
        gqa_attrs: dict[str, int | float] = {
            "num_heads": q_num_heads,
            "kv_num_heads": kv_num_heads,
            "scale": scale,
            "do_rotary": 1,
            "rotary_interleaved": rotary_interleaved,
        }
        if softcap:
            gqa_attrs["softcap"] = softcap
        outputs = op.op_multi_out(
            "GroupQueryAttention",
            inputs=[
                q_pre,
                k_pre,
                v,
                past_key,
                past_value,
                self._seqlens_k,
                self._total_seq_len,
                self._cos_cache,
                self._sin_cache,
            ],
            domain="com.microsoft",
            attributes=gqa_attrs,
            num_outputs=3,
        )

        return outputs[0], outputs[1], outputs[2]


# ====================================================================
# PackQKVForGQA — consolidates 3 separate MatMuls into 1 packed MatMul
# ====================================================================


class PackQKVForGQA(RewriteRuleClassBase):
    """Pack separate Q/K/V projections into a single MatMul for GQA.

    This rule runs **after** ``RotaryAttentionToGQA`` and looks for
    ``GroupQueryAttention`` nodes whose Q, K, V inputs each come from a
    separate ``Transpose+MatMul`` projection that shares the same
    ``hidden_states`` input.

    **Matched pattern:**

    .. code-block:: text

        q_wt = Transpose(W_q, perm=[1, 0])
        k_wt = Transpose(W_k, perm=[1, 0])
        v_wt = Transpose(W_v, perm=[1, 0])
        q    = MatMul(hidden, q_wt)
        k    = MatMul(hidden, k_wt)
        v    = MatMul(hidden, v_wt)
        out, pkey, pval = GroupQueryAttention(q, k, v, ...)

    Where ``W_q``, ``W_k``, ``W_v`` are graph parameters (initializers).

    **Replacement:**

    .. code-block:: text

        W_qkv  = Concat(W_q, W_k, W_v, axis=0)   # (q_out+k_out+v_out, hidden)
        packed  = MatMul(hidden, Transpose(W_qkv))
        out, pkey, pval = GroupQueryAttention(packed, None, None, ...)

    The packed weight is expressed as a ``Concat`` graph node so that
    the rule fires without requiring actual weight data (weights may be
    applied to the model after optimization).  A subsequent
    ``FoldConstantsPass`` collapses the ``Concat`` into a single
    initializer once weights are loaded.

    Models with QK norm (e.g. Qwen3) are unaffected because the Q/K
    projections are followed by a normalization op, so the pattern does
    not match.
    """

    _pack_counter: int

    def __init__(self):
        super().__init__()
        self._pack_counter = 0

    # ------------------------------------------------------------------ pattern

    def pattern(self, op, hidden, q_w, k_w, v_w):
        q_wt = op.Transpose(q_w, perm=[1, 0])
        k_wt = op.Transpose(k_w, perm=[1, 0])
        v_wt = op.Transpose(v_w, perm=[1, 0])
        q = op.MatMul(hidden, q_wt)
        k = op.MatMul(hidden, k_wt)
        v = op.MatMul(hidden, v_wt)

        return op.GroupQueryAttention(
            q,
            k,
            v,
            _domain="com.microsoft",
            _allow_other_attributes=True,
            _allow_other_inputs=True,
            _outputs=["gqa_out", "present_key", "present_value"],
        )

    # ------------------------------------------------------------------ check

    def check(self, context, q_w, k_w, v_w, **_):
        # Verify all three projection weights are traceable to graph parameters.
        # Raises MatchFailureError if any weight is a computed (non-parameter) value.
        if q_w.producer() is not None:
            raise MatchFailureError(f"q_w {q_w.name!r} is not a graph parameter")
        if k_w.producer() is not None:
            raise MatchFailureError(f"k_w {k_w.name!r} is not a graph parameter")
        if v_w.producer() is not None:
            raise MatchFailureError(f"v_w {v_w.name!r} is not a graph parameter")
        return True

    # ------------------------------------------------------------------ rewrite

    def rewrite(
        self,
        op,
        hidden,
        q_w,
        k_w,
        v_w,
        gqa_out,
        present_key,
        present_value,
        **_,
    ):
        # Concat along output dimension: (q_out + k_out + v_out, hidden)
        # This is a graph-level op so it works without actual weight data.
        # A FoldConstantsPass after apply_weights() collapses it to one initializer.
        self._pack_counter += 1
        packed_w = op.Concat(q_w, k_w, v_w, axis=0)
        # Transpose packed weight: (q_out+k_out+v_out, hidden) → (hidden, q_out+k_out+v_out)
        packed_wt = op.Transpose(packed_w, perm=[1, 0])
        packed_qkv = op.MatMul(hidden, packed_wt)

        # Recover remaining GQA inputs and attributes from the matched node
        gqa_node = gqa_out.producer()
        attrs = {key: gqa_node.attributes[key].value for key in gqa_node.attributes}

        outputs = op.op_multi_out(
            "GroupQueryAttention",
            inputs=[
                packed_qkv,
                None,
                None,
                *gqa_node.inputs[3:],
            ],
            domain="com.microsoft",
            attributes=attrs,
            num_outputs=3,
        )

        return outputs[0], outputs[1], outputs[2]


class PackQKVWithBiasForGQA(RewriteRuleClassBase):
    """Pack Q/K/V projections (with bias) into a single MatMul+Add for GQA.

    Handles models where each QKV projection is ``Linear(bias=True)``,
    producing ``Transpose+MatMul → Add(bias)`` per projection — e.g. Qwen2.5, Phi3/4.
    The existing :class:`PackQKVForGQA` only matches the no-bias form.

    **Matched pattern:**

    .. code-block:: text

        q_wt = Transpose(W_q, perm=[1, 0])
        k_wt = Transpose(W_k, perm=[1, 0])
        v_wt = Transpose(W_v, perm=[1, 0])
        q = Add(MatMul(hidden, q_wt), bias_q)
        k = Add(MatMul(hidden, k_wt), bias_k)
        v = Add(MatMul(hidden, v_wt), bias_v)
        out, pkey, pval = GroupQueryAttention(q, k, v, ...)

    Where ``W_q``, ``W_k``, ``W_v``, ``bias_q``, ``bias_k``, ``bias_v``
    are graph parameters (initializers).

    **Replacement:**

    .. code-block:: text

        W_qkv    = Concat(W_q, W_k, W_v, axis=0)
        bias_qkv = Concat(bias_q, bias_k, bias_v, axis=0)
        packed   = Add(MatMul(hidden, Transpose(W_qkv)), bias_qkv)
        out, pkey, pval = GroupQueryAttention(packed, None, None, ...)

    Both the weight ``Concat`` and the bias ``Concat`` are graph-level
    ops so the rule fires without requiring actual weight data.  A
    subsequent ``FoldConstantsPass`` collapses them once weights are
    loaded.

    GQA does not have a dedicated bias input, so the ``Add`` must stay
    in the graph as the value that feeds the ``query`` slot.
    """

    _pack_counter: int

    def __init__(self):
        super().__init__()
        self._pack_counter = 0

    # ------------------------------------------------------------------ pattern

    def pattern(self, op, hidden, q_w, bias_q, k_w, bias_k, v_w, bias_v):
        q_wt = op.Transpose(q_w, perm=[1, 0])
        k_wt = op.Transpose(k_w, perm=[1, 0])
        v_wt = op.Transpose(v_w, perm=[1, 0])
        q = op.Add(op.MatMul(hidden, q_wt), bias_q)
        k = op.Add(op.MatMul(hidden, k_wt), bias_k)
        v = op.Add(op.MatMul(hidden, v_wt), bias_v)

        return op.GroupQueryAttention(
            q,
            k,
            v,
            _domain="com.microsoft",
            _allow_other_attributes=True,
            _allow_other_inputs=True,
            _outputs=["gqa_out", "present_key", "present_value"],
        )

    # ------------------------------------------------------------------ check

    def check(self, context, q_w, bias_q, k_w, bias_k, v_w, bias_v, **_):
        # Projection weights must be direct graph parameters.
        if q_w.producer() is not None:
            raise MatchFailureError(f"q_w {q_w.name!r} is not a graph parameter")
        if k_w.producer() is not None:
            raise MatchFailureError(f"k_w {k_w.name!r} is not a graph parameter")
        if v_w.producer() is not None:
            raise MatchFailureError(f"v_w {v_w.name!r} is not a graph parameter")
        # Bias terms must be direct graph parameters (no producer node).
        for bias in (bias_q, bias_k, bias_v):
            if bias is None or bias.producer() is not None:
                raise MatchFailureError(f"bias {bias!r} is not a graph parameter")
        return True

    # ------------------------------------------------------------------ rewrite

    def rewrite(
        self,
        op,
        hidden,
        q_w,
        bias_q,
        k_w,
        bias_k,
        v_w,
        bias_v,
        gqa_out,
        present_key,
        present_value,
        **_,
    ):
        # Concat weights along output dimension: (q_out + k_out + v_out, hidden)
        self._pack_counter += 1
        packed_w = op.Concat(q_w, k_w, v_w, axis=0)
        # Transpose packed weight: (q_out+k_out+v_out, hidden) → (hidden, q_out+k_out+v_out)
        packed_wt = op.Transpose(packed_w, perm=[1, 0])
        packed_mm = op.MatMul(hidden, packed_wt)

        # Concat biases: (q_out + k_out + v_out,)
        packed_bias = op.Concat(bias_q, bias_k, bias_v, axis=0)
        # GQA has no bias input — the Add stays in the graph.
        packed_qkv = op.Add(packed_mm, packed_bias)

        gqa_node = gqa_out.producer()
        attrs = {key: gqa_node.attributes[key].value for key in gqa_node.attributes}

        outputs = op.op_multi_out(
            "GroupQueryAttention",
            inputs=[
                packed_qkv,
                None,
                None,
                *gqa_node.inputs[3:],
            ],
            domain="com.microsoft",
            attributes=attrs,
            num_outputs=3,
        )

        return outputs[0], outputs[1], outputs[2]


class AttentionToGQA(RewriteRuleClassBase):
    """Universal fallback: replace Attention with GroupQueryAttention (do_rotary=0).

    This rule matches any decoder ``Attention`` node (past_key/past_value
    are graph inputs) regardless of how Q/K rotary embeddings are applied.
    It is tried **after** :class:`RotaryAttentionToGQA`, so it only fires
    for models where that rule does not match — e.g. Qwen3.5, which uses
    3D multimodal RoPE implemented with ``Where`` nodes rather than
    standard ``RotaryEmbedding`` ops.

    **Matched pattern:**

    .. code-block:: text

        attn_out, present_key, present_value = Attention(
            q, k, v, attention_bias, past_key, past_value,
        )

    Where ``past_key`` and ``past_value`` are graph inputs.  The Q/K
    values may already have RoPE applied via any graph ops.

    **Replacement:**

    .. code-block:: text

        seqlens_k = Cast(ReduceSum(attention_mask, axis=1) - 1, INT32)
        total_seq_len = Cast(Shape(attention_mask)[1], INT32)
        attn_out, present_key, present_value = GroupQueryAttention(
            q, k, v, past_key, past_value, seqlens_k, total_seq_len,
            num_heads=..., kv_num_heads=..., do_rotary=0,
        )

    ``do_rotary=0`` means the position embeddings are already baked into
    the Q/K inputs — the GQA kernel does not apply additional RoPE.
    """

    def __init__(self):
        super().__init__()
        self._seqlens_k = None
        self._total_seq_len = None

    # ------------------------------------------------------------------ pattern

    def pattern(self, op, q, k, v, attention_bias, past_key, past_value):
        return op.Attention(
            q,
            k,
            v,
            attention_bias,
            past_key,
            past_value,
            _allow_other_attributes=True,
            _outputs=["attn_out", "present_key", "present_value"],
        )

    # ------------------------------------------------------------------ check

    def check(self, context, attn_out, past_key, past_value, **_):
        result = MatchResult()
        attn = attn_out.producer()

        if attn.attributes.get_float("scale", None) is None:
            return result.fail("Missing scale attribute on Attention")
        if attn.attributes.get_int("q_num_heads", None) is None:
            return result.fail("Missing q_num_heads on Attention")
        if attn.attributes.get_int("kv_num_heads", None) is None:
            return result.fail("Missing kv_num_heads on Attention")

        # past_key/past_value must be graph inputs (decoder attention, not encoder)
        if past_key is None or past_value is None:
            return result.fail("No KV cache inputs")
        if past_key.producer() is not None:
            return result.fail("past_key is not a graph input")
        if past_value.producer() is not None:
            return result.fail("past_value is not a graph input")

        # attention_mask must be a graph input — needed to build seqlens_k.
        graph = attn.graph
        if not any(gi.name == "attention_mask" for gi in graph.inputs):
            return result.fail("No attention_mask graph input — cannot build seqlens_k")

        # Skip when head_dim exceeds CUDA GQA MAX_HEAD_SIZE (256).
        hd = _head_dim_exceeds_gqa_limit(past_key)
        if hd is not None:
            return result.fail(f"head_dim={hd} exceeds GQA MAX_HEAD_SIZE={_MAX_GQA_HEAD_DIM}")

        return result

    # ------------------------------------------------------------------ rewrite

    def rewrite(
        self,
        op,
        q,
        k,
        v,
        attention_bias,
        past_key,
        past_value,
        attn_out,
        present_key,
        present_value,
        **_,
    ):
        attn = attn_out.producer()
        scale = attn.attributes.get_float("scale")
        q_num_heads = attn.attributes.get_int("q_num_heads")
        kv_num_heads = attn.attributes.get_int("kv_num_heads")
        softcap = attn.attributes.get_float("softcap", 0.0)

        # Build seqlens_k and total_seq_len from attention_mask (computed once).
        if self._seqlens_k is None:
            graph = attn.graph
            attention_mask = next(gi for gi in graph.inputs if gi.name == "attention_mask")
            axis = op.Constant(value_ints=[1])
            reduce_sum = op.ReduceSum(attention_mask, axis)
            one = op.Constant(value_ints=[1])
            self._seqlens_k = op.Cast(op.Sub(reduce_sum, one), to=6)
            mask_shape = op.Shape(attention_mask)
            idx_1 = op.Constant(value_int=1)
            self._total_seq_len = op.Cast(op.Gather(mask_shape, idx_1), to=6)

        # do_rotary=0: Q/K already have position embeddings baked in.
        gqa_attrs: dict[str, int | float] = {
            "num_heads": q_num_heads,
            "kv_num_heads": kv_num_heads,
            "scale": scale,
            "do_rotary": 0,
        }
        if softcap:
            gqa_attrs["softcap"] = softcap

        outputs = op.op_multi_out(
            "GroupQueryAttention",
            inputs=[q, k, v, past_key, past_value, self._seqlens_k, self._total_seq_len],
            domain="com.microsoft",
            attributes=gqa_attrs,
            num_outputs=3,
        )
        return outputs[0], outputs[1], outputs[2]


def group_query_attention_rules() -> RewriteRuleSet:
    """Return rules that fuse Attention (+ optional RotaryEmbedding) into GQA.

    Two rules are included, applied in order:

    1. :class:`RotaryAttentionToGQA` — optimal path (``do_rotary=1``).
       Matches ``RotaryEmbedding + Attention`` where cos/sin come from
       Gather nodes.  Fuses rotary into the GQA kernel.

    2. :class:`AttentionToGQA` — universal fallback (``do_rotary=0``).
       Matches any decoder ``Attention`` node.  Used for models whose
       position embeddings are not expressed as standard ``RotaryEmbedding``
       ops (e.g. Qwen3.5 3D mRoPE via ``Where`` nodes).

    QKV packing is a separate optional pass; use
    :func:`pack_qkv_for_gqa_rules` for that.

    Returns:
        :class:`RewriteRuleSet` containing the GQA fusion rules.
    """
    return RewriteRuleSet([RotaryAttentionToGQA().rule(), AttentionToGQA().rule()])


def pack_qkv_for_gqa_rules() -> RewriteRuleSet:
    """Return rules that pack Q/K/V projections into a single MatMul (±Add bias).

    Two rules are included; both run **after** :func:`group_query_attention_rules`:

    1. :class:`PackQKVForGQA` — no-bias models (Llama, Gemma).  Matches
       ``MatMul → GQA`` and packs the three weight matrices into one.

    2. :class:`PackQKVWithBiasForGQA` — bias models (Qwen2.5, Phi3/4).
       Matches ``MatMul → Add(bias) → GQA`` and packs both the weights and
       the biases, emitting ``MatMul(packed_w) → Add(packed_bias) → GQA``.

    Only applies when the EP's ``qkv_pack_dtypes`` includes the current
    model dtype.  The caller is responsible for gating on that condition.

    Returns:
        :class:`RewriteRuleSet` containing the ``PackQKVForGQA`` and
        ``PackQKVWithBiasForGQA`` rules.
    """
    return RewriteRuleSet([PackQKVForGQA().rule(), PackQKVWithBiasForGQA().rule()])
