# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Paged, absorbed Multi-head Latent Attention (LATENT ``PagedAttention``).

This module emits the ``com.microsoft::PagedAttention`` v1 operator in its
``kv_cache_layout="LATENT"`` (absorbed-MLA) mode for *dense* MLA models whose
geometry is expressible by the operator (DeepSeek-V2/V3, GLM-5.2
``--glm-full-attention``). It is an opt-in export path
(``config.export_paged_attention``); the default dense-MLA export
(:class:`~mobius.components._deepseek_mla.DeepSeekMLA`) is byte-identical when
the flag is off.

Eligibility is decided purely from semantic geometry (see
:func:`paged_attention_rejection`), never from model names. Query-dependent
sparse selection (GLM DSA/IndexShare, DeepSeek-V4 CSA/HCA) is *not* expressible
because the operator has no query-dependent sparse-index input, so those modes
are typed-rejected rather than silently falling back to dense.

Absorbed-MLA contract (mirrors the onnx-genai oracle
``crates/onnx-genai-paged-attention/tests/equivalence.rs``):

* ``l`` = ``kv_lora_rank`` = absorbed content width = ``v_head_size``.
* ``r`` = ``qk_rope_head_dim`` = decoupled RoPE width.
* ``head_size = l + r``; RoPE covers the suffix ``[l, l + r)``.
* ``d`` = ``qk_nope_head_dim`` (folded away), ``dv`` = ``v_head_dim`` (folded
  into ``o_proj``).
* Per head, ``W_UK^h`` (``[d, l]``, the k-nope block of ``kv_b_proj``) folds
  into the query, ``W_UV^h`` (``[dv, l]``, the value block of ``kv_b_proj``)
  folds into ``o_proj``. ``kv_b_proj`` is fully absorbed and dropped.

The operator consumes/mutates caller-owned page buffers in place; Mobius never
allocates or manages pages (``onnx-genai-kv`` remains the sole cache authority).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components._common import Linear
from mobius.components._rms_norm import RMSNorm
from mobius.components._rotary_embedding import yarn_apply_mscale

if TYPE_CHECKING:
    import onnx_ir as ir

DOMAIN = "com.microsoft"

# ``com.microsoft::PagedAttention`` LATENT cache-write sentinel + defaults. Kept
# in sync with the onnx-genai native ABI (``crates/onnx-genai-kv``):
# ``slot = page_id * block_size + offset``; ``-1`` skips the write.
PAGED_SLOT_EMPTY = -1
PAGED_BLOCK_TABLE_PAD = 0
# Minimum page/block size: power-of-two and >= 16 (never divisible by 256).
DEFAULT_PAGED_BLOCK_SIZE = 16


def _supported_dtypes() -> frozenset:
    import onnx_ir as ir

    return frozenset({ir.DataType.FLOAT16, ir.DataType.BFLOAT16})


@dataclass(frozen=True)
class PagedMlaGeometry:
    """Resolved LATENT geometry for the ``PagedAttention`` operator.

    All fields are derived from :class:`ArchitectureConfig` and validated
    against the onnx-genai-kv LATENT geometry constraints.
    """

    num_heads: int
    """``num_heads`` attribute (query heads)."""
    kv_lora_rank: int
    """``l`` -- latent content width; also ``v_head_size``."""
    qk_nope_head_dim: int
    """``d`` -- folded-away nope width of the decomposed query/key."""
    qk_rope_head_dim: int
    """``r`` -- decoupled RoPE width (``rotary_dim``)."""
    v_head_dim: int
    """``dv`` -- decomposed value width, folded into ``o_proj``."""
    scale: float
    """Explicit softmax scale (REQUIRED because ``v_head_size != head_size``)."""
    rope_interleaved: bool
    """RoPE layout for the ``do_rotary`` suffix."""

    @property
    def head_size(self) -> int:
        """Absorbed head width ``l + r``."""
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def v_head_size(self) -> int:
        """LATENT context width ``l`` (== ``kv_lora_rank``)."""
        return self.kv_lora_rank

    @property
    def rotary_dim(self) -> int:
        """RoPE width ``r`` (== ``qk_rope_head_dim``)."""
        return self.qk_rope_head_dim

    @property
    def rotary_offset(self) -> int:
        """RoPE suffix start ``l`` (== ``kv_lora_rank``)."""
        return self.kv_lora_rank

    @property
    def qk_head_dim(self) -> int:
        """Decomposed query/key width ``d + r`` (drives the scale)."""
        return self.qk_nope_head_dim + self.qk_rope_head_dim


def _mla_scale(config: ArchitectureConfig) -> float:
    """Softmax scale identical to the decomposed :class:`DeepSeekMLA`.

    Uses ``1 / sqrt(qk_head_dim)`` with the YaRN ``mscale^2`` correction when
    active, so the absorbed LATENT logits match the decomposed oracle exactly.
    """
    qk_head_dim = (config.qk_nope_head_dim or 0) + (config.qk_rope_head_dim or 0)
    return yarn_apply_mscale(config.rope_type, config.rope_scaling, qk_head_dim**-0.5)


def mla_paged_geometry(
    config: ArchitectureConfig, scale: float | None = None
) -> PagedMlaGeometry:
    """Resolve the LATENT geometry for ``config``.

    Raises:
        ValueError: If ``config`` is not an eligible dense-MLA geometry; the
            message is the typed rejection reason from
            :func:`paged_attention_rejection`.
    """
    reason = paged_attention_rejection(config)
    if reason is not None:
        raise ValueError(reason)
    return PagedMlaGeometry(
        num_heads=config.num_attention_heads,
        kv_lora_rank=int(config.kv_lora_rank),
        qk_nope_head_dim=int(config.qk_nope_head_dim),
        qk_rope_head_dim=int(config.qk_rope_head_dim),
        v_head_dim=int(config.v_head_dim),
        scale=scale if scale is not None else _mla_scale(config),
        rope_interleaved=config.rope_interleave,
    )


def paged_attention_rejection(config: ArchitectureConfig) -> str | None:
    """Return a typed rejection reason, or ``None`` if LATENT-eligible.

    Eligibility is purely geometric. A non-``None`` return means the model is
    *not* expressible as ``com.microsoft::PagedAttention`` LATENT and the
    feature-on export must error rather than silently emit a dense graph.
    """
    ir_dtypes = _supported_dtypes()

    def _has(name: str) -> bool:
        v = getattr(config, name, None)
        return v is not None and v > 0

    # --- Must be MLA at all. ---
    if not (_has("kv_lora_rank") and _has("qk_nope_head_dim") and _has("qk_rope_head_dim")):
        return (
            "PagedAttention LATENT requires an MLA geometry "
            "(kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim > 0); "
            "this config is not MLA."
        )
    if not _has("v_head_dim"):
        return "PagedAttention LATENT requires v_head_dim > 0."

    # --- Query-dependent sparse selection is not expressible. ---
    # The operator has no query-dependent sparse-index input. GLM's DSA indexer
    # (index_topk / indexer_types / index_n_heads / index_head_dim) is consumed
    # *only* when use_dsa is active; --glm-full-attention (use_dsa=False) drops
    # every indexer weight and builds plain dense MLA, so those vestigial config
    # fields must NOT reject on their own. DeepSeek-V4 CSA/HCA is discriminated
    # by its own always-on fields (compress_ratios / o_lora_rank / hc_mult).
    if getattr(config, "use_dsa", False):
        return (
            "PagedAttention LATENT cannot express GLM DeepSeek Sparse Attention "
            "(DSA/IndexShare): query-dependent sparse indices have no operator "
            "input. Export dense MLA (--glm-full-attention) to opt in."
        )
    if getattr(config, "compress_ratios", None):
        return (
            "PagedAttention LATENT cannot express DeepSeek-V4 compressed sparse "
            "attention (CSA/HCA): query-dependent compression is not an operator "
            "input."
        )
    if getattr(config, "o_lora_rank", None) or getattr(config, "o_groups", 1) not in (0, 1):
        return (
            "PagedAttention LATENT cannot express DeepSeek-V4 grouped/low-rank "
            "output projection (o_groups/o_lora_rank)."
        )
    if getattr(config, "hc_mult", 1) not in (0, 1):
        return "PagedAttention LATENT cannot express Hyper-Connections (hc_mult > 1)."

    # --- Multi-token prediction is out of scope. ---
    if getattr(config, "num_nextn_predict_layers", 0) > 0:
        return (
            "PagedAttention LATENT export does not cover Multi-Token Prediction "
            "(num_nextn_predict_layers > 0)."
        )

    # --- Optional operator modes not implemented in this slice. ---
    if getattr(config, "attention_sink", False) or getattr(config, "head_sink", False):
        return "PagedAttention LATENT slice does not implement the head_sink input."
    if getattr(config, "use_qk_norm", False) or getattr(config, "qk_layernorm", False):
        return "PagedAttention LATENT slice does not implement q_norm/k_norm inputs."
    window = getattr(config, "sliding_window", None)
    if window is not None and window > 0:
        return (
            "PagedAttention LATENT slice does not implement windowed attention "
            f"(sliding_window={window})."
        )

    # --- Cache dtype: fp16/bf16 only; quantized cache rejected this slice. ---
    if config.dtype not in ir_dtypes:
        return (
            "PagedAttention LATENT slice supports float16/bfloat16 only; "
            f"got dtype {config.dtype!r}. Quantized cache modes are rejected."
        )

    # --- onnx-genai-kv LATENT geometry constraints. ---
    l = int(config.kv_lora_rank)  # noqa: E741  (matches oracle naming)
    r = int(config.qk_rope_head_dim)
    head_size = l + r
    if head_size % 8 != 0:
        return f"PagedAttention LATENT requires head_size % 8 == 0; got {head_size}."
    if l % 8 != 0:
        return f"PagedAttention LATENT requires latent_dim (kv_lora_rank) % 8 == 0; got {l}."
    if not (1 <= l <= head_size):
        return f"PagedAttention LATENT requires 1 <= v_head_size <= head_size; got {l}."
    if r != 0 and r % 16 != 0:
        return f"PagedAttention LATENT requires rotary_dim % 16 == 0 (or 0); got {r}."
    # rotary_offset == l by construction; enforce the ORT sibling constraint.
    if l % 8 != 0:
        return f"PagedAttention LATENT requires rotary_offset % 8 == 0; got {l}."
    if l + r > head_size:
        return "PagedAttention LATENT requires rotary_offset + rotary_dim <= head_size."
    return None


def paged_attention_eligible(config: ArchitectureConfig) -> bool:
    """Whether ``config`` can be exported as ``PagedAttention`` LATENT."""
    return paged_attention_rejection(config) is None


# ---------------------------------------------------------------------------
# Weight absorption
# ---------------------------------------------------------------------------


def _to_numpy(array) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    # torch.Tensor (deferred import to keep numpy-only tests light).
    return array.detach().to("cpu").float().numpy()


def absorb_mla_weights(
    weights: dict[str, np.ndarray],
    config: ArchitectureConfig,
    *,
    q_key: str,
    kv_b_key: str,
    o_key: str,
) -> dict[str, np.ndarray]:
    """Fold ``kv_b_proj`` into the query and output projections.

    Given the decomposed MLA weights for a single layer, returns replacement
    weights for the absorbed LATENT path:

    * ``q_key``  ``[nh * (d + r), in]``  -> ``[nh * (l + r), in]``
    * ``o_key``  ``[hidden, nh * dv]``   -> ``[hidden, nh * l]``
    * ``kv_b_key`` is dropped (fully absorbed).

    Args:
        weights: Mapping containing at least ``q_key``, ``kv_b_key``, ``o_key``.
        config: MLA architecture config.
        q_key: Name of the query up-projection weight (``q_b_proj`` or
            ``q_proj``), stored ``[out, in]``.
        kv_b_key: Name of the ``kv_b_proj`` weight, stored
            ``[nh * (d + dv), l]``.
        o_key: Name of the output projection weight, stored ``[hidden, nh*dv]``.

    Returns:
        A dict with the absorbed ``q_key`` and ``o_key`` (and no ``kv_b_key``).
    """
    geom = mla_paged_geometry(config)
    nh = geom.num_heads
    d = geom.qk_nope_head_dim
    r = geom.qk_rope_head_dim
    dv = geom.v_head_dim
    l = geom.kv_lora_rank  # noqa: E741

    w_q = _to_numpy(weights[q_key]).astype(np.float64)
    w_kvb = _to_numpy(weights[kv_b_key]).astype(np.float64)
    w_o = _to_numpy(weights[o_key]).astype(np.float64)

    qk_head_dim = d + r
    if w_q.shape[0] != nh * qk_head_dim:
        raise ValueError(f"{q_key} rows {w_q.shape[0]} != num_heads*(d+r) {nh * qk_head_dim}")
    if w_kvb.shape != (nh * (d + dv), l):
        raise ValueError(
            f"{kv_b_key} shape {w_kvb.shape} != (num_heads*(d+dv), l) {(nh * (d + dv), l)}"
        )
    if w_o.shape[1] != nh * dv:
        raise ValueError(f"{o_key} cols {w_o.shape[1]} != num_heads*dv {nh * dv}")

    in_dim = w_q.shape[1]
    hidden = w_o.shape[0]
    head_size = l + r
    q_absorbed = np.empty((nh * head_size, in_dim), dtype=np.float64)
    o_absorbed = np.empty((hidden, nh * l), dtype=np.float64)
    for h in range(nh):
        w_uk = w_kvb[h * (d + dv) : h * (d + dv) + d, :]  # [d, l]
        w_uv = w_kvb[h * (d + dv) + d : (h + 1) * (d + dv), :]  # [dv, l]
        qb_nope = w_q[h * qk_head_dim : h * qk_head_dim + d, :]  # [d, in]
        qb_rope = w_q[h * qk_head_dim + d : (h + 1) * qk_head_dim, :]  # [r, in]
        # Absorbed query nope: W_UK^T @ qb_nope -> [l, in].
        q_absorbed[h * head_size : h * head_size + l, :] = w_uk.T @ qb_nope
        q_absorbed[h * head_size + l : (h + 1) * head_size, :] = qb_rope
        # Absorbed o_proj: W_O^h @ W_UV^h -> [hidden, l].
        w_o_head = w_o[:, h * dv : (h + 1) * dv]  # [hidden, dv]
        o_absorbed[:, h * l : (h + 1) * l] = w_o_head @ w_uv

    out_dtype = _to_numpy(weights[q_key]).dtype
    return {
        q_key: q_absorbed.astype(out_dtype),
        o_key: o_absorbed.astype(out_dtype),
    }


# ---------------------------------------------------------------------------
# Component
# ---------------------------------------------------------------------------


@dataclass
class PagedCacheState:
    """Caller-owned page/cache tensors bound to the LATENT operator.

    Mobius does not allocate or manage any of these; they are graph inputs
    owned by the onnx-genai-kv page manager. ``key_cache`` is mutated in place
    and aliased to the operator's ``key_cache`` output.
    """

    key_cache: ir.Value
    """LATENT cache ``[num_blocks, block_size, 1, head_size]`` (in place)."""
    block_table: ir.Value
    """``[batch, max_blocks_per_seq]`` int32 page indices."""
    slot_mapping: ir.Value
    """``[token_count]`` int32 flattened slots; ``-1`` skips the write."""
    cumulative_sequence_length: ir.Value
    """``[batch + 1]`` int32 cumulative query lengths."""
    past_seqlens: ir.Value
    """``[batch]`` int32 already-cached lengths per sequence."""
    cos_cache: ir.Value | None = None
    """``[max_pos, rotary_dim/2]`` RoPE cosine cache (filled by the model)."""
    sin_cache: ir.Value | None = None
    """``[max_pos, rotary_dim/2]`` RoPE sine cache (filled by the model)."""


class PagedLatentMLA(nn.Module):
    """Absorbed dense MLA that emits ``com.microsoft::PagedAttention`` LATENT.

    Structurally mirrors :class:`~mobius.components._deepseek_mla.DeepSeekMLA`
    for the Q/KV projections, but folds ``kv_b_proj`` into the query and output
    projections (see :func:`absorb_mla_weights`) and replaces the decomposed
    ``op.Attention`` with a single LATENT ``PagedAttention`` node.

    The module's ``q_b_proj``/``q_proj`` and ``o_proj`` parameters are the
    *absorbed* shapes; :func:`absorb_mla_weights` must be applied to the raw
    checkpoint before loading.
    """

    def __init__(
        self,
        config: ArchitectureConfig,
        scale: float | None = None,
        linear_class: type | None = None,
    ):
        super().__init__()
        if linear_class is None:
            linear_class = Linear
        self.geom = mla_paged_geometry(config, scale=scale)
        geom = self.geom
        self.num_heads = geom.num_heads
        self.q_lora_rank = config.q_lora_rank
        self.head_size = geom.head_size
        self.kv_lora_rank = geom.kv_lora_rank
        self.qk_rope_head_dim = geom.qk_rope_head_dim

        # Q path: absorbed up-projection produces [nh * head_size].
        if self.q_lora_rank is not None and self.q_lora_rank > 0:
            self.q_a_proj = linear_class(config.hidden_size, self.q_lora_rank, bias=False)
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = linear_class(
                self.q_lora_rank, self.num_heads * self.head_size, bias=False
            )
        else:
            self.q_proj = linear_class(
                config.hidden_size, self.num_heads * self.head_size, bias=False
            )

        # KV path: joint projection for the single-head latent row.
        self.kv_a_proj_with_mqa = linear_class(
            config.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)

        # Absorbed output projection consumes the [nh * l] LATENT context.
        self.o_proj = linear_class(
            self.num_heads * self.kv_lora_rank, config.hidden_size, bias=False
        )

        self.scaling = geom.scale
        self._rope_interleave = geom.rope_interleaved

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        cache: PagedCacheState,
    ):
        geom = self.geom
        # --- Absorbed query: [B, S, nh * head_size] -> [tokens, nh * head_size]. ---
        if self.q_lora_rank is not None and self.q_lora_rank > 0:
            q = self.q_a_proj(op, hidden_states)
            q = self.q_a_layernorm(op, q)
            q = self.q_b_proj(op, q)
        else:
            q = self.q_proj(op, hidden_states)
        query = op.Reshape(q, [-1, self.num_heads * self.head_size])

        # --- Latent key row: [B, S, head_size] -> [tokens, head_size]. ---
        compressed_kv = self.kv_a_proj_with_mqa(op, hidden_states)
        k_pass, k_rope = op.Split(
            compressed_kv,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            axis=-1,
            _outputs=2,
        )
        k_pass = self.kv_a_layernorm(op, k_pass)
        latent_key = op.Concat(k_pass, k_rope, axis=-1)
        key = op.Reshape(latent_key, [-1, self.head_size])

        # --- Single LATENT PagedAttention node (17 in / 3 out). ---
        # Absent optional inputs are passed as ``None`` to preserve positions.
        output, key_cache_out, _value_cache_out = op.PagedAttention(
            query,  # 0 query [tokens, nh * head_size]
            key,  # 1 key (latent row) [tokens, head_size]
            None,  # 2 value (absent in LATENT)
            cache.key_cache,  # 3 key_cache [num_blocks, block_size, 1, head_size]
            None,  # 4 value_cache (absent in LATENT)
            cache.cumulative_sequence_length,  # 5 [batch + 1] int32
            cache.past_seqlens,  # 6 [batch] int32
            cache.block_table,  # 7 [batch, max_blocks_per_seq] int32
            cache.cos_cache,  # 8 [max_pos, rotary_dim/2]
            cache.sin_cache,  # 9 [max_pos, rotary_dim/2]
            cache.slot_mapping,  # 10 [tokens] int32 (-1 skips write)
            None,  # 11 head_sink (absent)
            None,  # 12 q_norm (absent)
            None,  # 13 k_norm (absent)
            None,  # 14 k_scale (absent)
            None,  # 15 v_scale (absent)
            None,  # 16 attention_metadata (absent)
            num_heads=self.num_heads,
            kv_num_heads=1,
            scale=self.scaling,
            kv_cache_layout="LATENT",
            v_head_size=geom.v_head_size,
            rotary_offset=geom.rotary_offset,
            do_rotary=1,
            rotary_interleaved=int(self._rope_interleave),
            _domain=DOMAIN,
            _outputs=3,
        )

        # --- Absorbed output projection: reshape [tokens, nh*l] back to the
        # [batch, seq, nh*l] rank of the input, then project to hidden. ---
        hidden_shape = op.Shape(hidden_states)
        batch_seq = op.Slice(hidden_shape, [0], [2], [0])
        out_target = op.Concat(batch_seq, op.Constant(value_ints=[-1]), axis=0)
        output = op.Reshape(output, out_target)
        attn_output = self.o_proj(op, output)
        return attn_output, (key_cache_out,)
