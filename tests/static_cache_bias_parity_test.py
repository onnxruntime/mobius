# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Static-cache *bias* numerical-parity tests (mobius #366, Part of #349).

These tests guard the bias-aware external-KV static-cache attention combination
that ``flags.static_cache_bias`` enables: ONNX ``Attention`` with ``is_causal=0``
+ a float additive ``attn_mask`` (Attention input #3) + ``nonpad_kv_seqlen``
(input #6), fed by two ``TensorScatter`` writes into a pre-allocated KV cache.
The additive bias is built in-graph by
:func:`mobius.components.create_static_cache_attention_bias` and carries the
FULL mask geometry (causal + sliding window + Gemma4 block overlay + padding),
keyed on absolute query positions with KV validity ``slot < nonpad_kv_seqlen``.

Unlike the maskless ``is_causal=1`` static-cache path (``static_cache_parity_test``)
— which is Flash-eligible but rejected by CPU/pre-#28958 kernels for
``S_q != total_kv`` with no ``past_key`` — the **bias** path routes ORT to the
MEA external-cache combination, which **runs on the CPU EP today** (no genai, no
Flash, no onnxruntime#28958).  So these tests run unconditionally on CPU.

The reference is an independent NumPy *dense* attention that gathers the valid
cache slots and applies the SAME causal + sliding + block-overlay + padding mask
the in-graph bias encodes.  Because the bias geometry (absolute query positions
vs. dense cache slot ids, GQA head sharing, the bottom-right contract) is the
only thing under test, an independent re-derivation of the mask in NumPy is the
authoritative guard (mirrors ``create_attention_bias``'s parity strategy).

Two cases:

* prefill chunk — ``write_indices=0``, ``S_q=N``, ``nonpad=N`` (the whole chunk
  is the valid region).
* decode step — ``write_indices=N``, ``S_q=1``, ``nonpad=N+1`` against a
  pre-populated cache (the single query attends the ``N`` past slots + itself).

Plus a padding-clamp guard (risk ii): an under-clamped ``nonpad < S_q`` feed
masks the cache tail; because this bias keys query positions on
``write_indices + arange(S_q)`` (not the bottom-right contract), every row still
attends its diagonal slot, so the output stays finite (no all-``dtype.min`` NaN)
and matches the dense reference.

Run::

    pytest tests/static_cache_bias_parity_test.py -v
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest
from onnxscript import GraphBuilder

from mobius._constants import OPSET_VERSION
from mobius._flags import override_flags
from mobius._testing.ort_inference import OnnxModelSession
from mobius.components import create_static_cache_attention_bias

# ---------------------------------------------------------------------------
# Standalone bias-attention graph builder
# ---------------------------------------------------------------------------


def _build_static_cache_bias_graph(
    *,
    batch: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_seq_len: int,
    query_len: int,
    sliding_window: int | None,
    use_block_overlay: bool,
    dtype: ir.DataType = ir.DataType.FLOAT,
) -> ir.Model:
    """Build a standalone graph mirroring the bias static-cache attention branch.

    Replicates ``_apply_attention``'s static *bias* path: scatter ``key`` /
    ``value`` into the pre-allocated caches via ``TensorScatter``, build the
    ``(B, 1, S_q, max_seq)`` additive bias via
    :func:`create_static_cache_attention_bias`, then run ``Attention`` with that
    bias as input #3, ``is_causal=0``, and ``nonpad_kv_seqlen`` as input #6.

    Inputs:  ``query`` ``[B, S_q, num_q_heads*head_dim]``; ``key`` / ``value``
             ``[B, S_q, num_kv_heads*head_dim]`` (GQA); ``key_cache`` /
             ``value_cache`` ``[B, max_seq_len, num_kv_heads*head_dim]``;
             ``write_indices`` and ``nonpad_kv_seqlen`` ``[B]`` int64; optionally
             ``block_sequence_ids`` ``[B, S_q]`` int64.
    Outputs: ``attn_output`` ``[B, S_q, num_q_heads*head_dim]``, the updated
             caches, and ``bias`` (for debugging / reference cross-check).
    """
    q_hidden = num_q_heads * head_dim
    kv_hidden = num_kv_heads * head_dim

    def _value(name: str, dims: list[int], dt: ir.DataType) -> ir.Value:
        return ir.Value(name=name, shape=ir.Shape(dims), type=ir.TensorType(dt))

    query = _value("query", [batch, query_len, q_hidden], dtype)
    key = _value("key", [batch, query_len, kv_hidden], dtype)
    value = _value("value", [batch, query_len, kv_hidden], dtype)
    key_cache = _value("key_cache", [batch, max_seq_len, kv_hidden], dtype)
    value_cache = _value("value_cache", [batch, max_seq_len, kv_hidden], dtype)
    write_indices = _value("write_indices", [batch], ir.DataType.INT64)
    nonpad_kv_seqlen = _value("nonpad_kv_seqlen", [batch], ir.DataType.INT64)

    inputs = [query, key, value, key_cache, value_cache, write_indices, nonpad_kv_seqlen]
    block_sequence_ids = None
    if use_block_overlay:
        block_sequence_ids = _value(
            "block_sequence_ids", [batch, query_len], ir.DataType.INT64
        )
        inputs.append(block_sequence_ids)

    graph = ir.Graph(
        inputs=inputs,
        outputs=[],
        nodes=[],
        name="static_cache_bias_probe",
        opset_imports={"": OPSET_VERSION},
    )
    op = GraphBuilder(graph).op

    # Scatter new K/V into the pre-allocated cache: cache[b, write[b] + t] = upd[b, t].
    updated_k = op.TensorScatter(key_cache, key, write_indices, axis=1)
    updated_v = op.TensorScatter(value_cache, value, write_indices, axis=1)

    # Build the additive bias in-graph (the unit under test).
    seq_len = op.Constant(value_ints=[query_len])  # (1,) int64 == [S_q]
    bias = create_static_cache_attention_bias(
        op,
        write_indices=write_indices,
        seq_len=seq_len,
        nonpad_kv_seqlen=nonpad_kv_seqlen,
        max_seq_len=max_seq_len,
        sliding_window=sliding_window,
        block_sequence_ids=block_sequence_ids,
        dtype=dtype,
    )

    scale = 1.0 / np.sqrt(head_dim)
    # is_causal=0 STRICTLY paired with the bias (the bias already encodes
    # causality; is_causal=1 would double-apply it).
    attn_output, _, _ = op.Attention(
        query,
        updated_k,
        updated_v,
        bias,  # attn_mask input #3 — the additive float bias
        None,  # no past_key (full cache is provided)
        None,  # no past_value
        nonpad_kv_seqlen,  # input #6 — drives the fully-masked-row zero guard
        q_num_heads=num_q_heads,
        kv_num_heads=num_kv_heads,
        scale=float(scale),
        is_causal=0,
        _outputs=3,
    )

    attn_output.name = "attn_output"
    updated_k.name = "updated_key_cache"
    updated_v.name = "updated_value_cache"
    bias.name = "bias"
    graph.outputs.extend([attn_output, updated_k, updated_v, bias])

    return ir.Model(graph, ir_version=10)


# ---------------------------------------------------------------------------
# Independent NumPy dense-attention reference
# ---------------------------------------------------------------------------


def _dense_reference(
    query: np.ndarray,
    full_key_cache: np.ndarray,
    full_value_cache: np.ndarray,
    *,
    write_indices: np.ndarray,
    nonpad_kv_seqlen: np.ndarray,
    max_seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    sliding_window: int | None,
    block_sequence_ids: np.ndarray | None,
) -> np.ndarray:
    """Dense reference applying the SAME mask the in-graph bias encodes.

    ``full_*_cache`` are the *post-scatter* caches (current chunk already
    written).  For each (batch, query, head) we build the boolean mask in the
    exact rule order ``create_static_cache_attention_bias`` uses — causal,
    AND sliding window, OR block overlay, AND padding validity — gather the
    valid cache slots (with GQA head sharing) and softmax over them.  A query
    row with no valid slot yields exactly ``0`` (the kernel's zero guard).
    """
    batch, query_len, q_hidden = query.shape
    scale = 1.0 / np.sqrt(head_dim)
    group = num_q_heads // num_kv_heads
    out = np.zeros((batch, query_len, q_hidden), dtype=np.float64)

    kv_slots = np.arange(max_seq_len)
    for b in range(batch):
        wi = int(write_indices[b])
        npad = int(nonpad_kv_seqlen[b])
        q_abs = wi + np.arange(query_len)  # absolute query positions

        # Per-slot KV block ids: -1 buffer with the current chunk scattered in
        # (mirrors the in-graph TensorScatter into a -1 buffer).
        kv_group = np.full((max_seq_len,), -1, dtype=np.int64)
        if block_sequence_ids is not None:
            kv_group[wi : wi + query_len] = block_sequence_ids[b]

        # Cache reshaped to (max_seq, num_kv_heads, head_dim) for GQA gather.
        k_heads = full_key_cache[b].reshape(max_seq_len, num_kv_heads, head_dim)
        v_heads = full_value_cache[b].reshape(max_seq_len, num_kv_heads, head_dim)
        q_heads = query[b].reshape(query_len, num_q_heads, head_dim)

        for t in range(query_len):
            mask = q_abs[t] >= kv_slots  # causal
            if sliding_window is not None:
                mask &= (q_abs[t] - kv_slots) < sliding_window  # local window
            if block_sequence_ids is not None and block_sequence_ids[b, t] >= 0:
                mask |= kv_group == block_sequence_ids[b, t]  # bidirectional block
            mask &= kv_slots < npad  # padding validity
            idx = np.nonzero(mask)[0]
            if idx.size == 0:
                continue  # structurally empty -> exactly 0 (zero guard)
            for h in range(num_q_heads):
                kv_h = h // group  # GQA: query head -> shared kv head
                kk = k_heads[idx, kv_h]  # (n_valid, head_dim)
                vv = v_heads[idx, kv_h]
                scores = (q_heads[t, h] @ kk.T) * scale
                scores = scores - scores.max()
                weights = np.exp(scores)
                weights = weights / weights.sum()
                out[b, t, h * head_dim : (h + 1) * head_dim] = weights @ vv
    return out


# ---------------------------------------------------------------------------
# Shared tiny bias-decoder geometry (hidden=64, GQA kv<heads)
# ---------------------------------------------------------------------------

_BATCH = 1
_NUM_Q_HEADS = 4
_NUM_KV_HEADS = 2  # GQA: 2 query heads share each kv head
_HEAD_DIM = 16  # hidden = 4 * 16 = 64
_MAX_SEQ_LEN = 16
_SLIDING_WINDOW = 4


def _scatter_into_cache(cache: np.ndarray, chunk: np.ndarray, write_idx: int) -> np.ndarray:
    """NumPy mirror of ``TensorScatter(cache, chunk, [write_idx], axis=1)``."""
    out = cache.copy()
    out[:, write_idx : write_idx + chunk.shape[1], :] = chunk
    return out


def _run_and_compare(
    *,
    query_len: int,
    write_idx: int,
    nonpad: int,
    use_block_overlay: bool,
    prefilled_valid_len: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build + run the bias graph and the dense reference; return (onnx, ref)."""
    rng = np.random.default_rng(seed)
    q_hidden = _NUM_Q_HEADS * _HEAD_DIM
    kv_hidden = _NUM_KV_HEADS * _HEAD_DIM

    query = rng.standard_normal((_BATCH, query_len, q_hidden)).astype(np.float32)
    key = rng.standard_normal((_BATCH, query_len, kv_hidden)).astype(np.float32)
    value = rng.standard_normal((_BATCH, query_len, kv_hidden)).astype(np.float32)

    # Pre-populate the cache slots [0, prefilled_valid_len) with random K/V so
    # the decode query has real past keys to attend (zeros for prefill).
    key_cache = np.zeros((_BATCH, _MAX_SEQ_LEN, kv_hidden), dtype=np.float32)
    value_cache = np.zeros((_BATCH, _MAX_SEQ_LEN, kv_hidden), dtype=np.float32)
    if prefilled_valid_len > 0:
        key_cache[:, :prefilled_valid_len, :] = rng.standard_normal(
            (_BATCH, prefilled_valid_len, kv_hidden)
        ).astype(np.float32)
        value_cache[:, :prefilled_valid_len, :] = rng.standard_normal(
            (_BATCH, prefilled_valid_len, kv_hidden)
        ).astype(np.float32)

    write_indices = np.full((_BATCH,), write_idx, dtype=np.int64)
    nonpad_kv_seqlen = np.full((_BATCH,), nonpad, dtype=np.int64)

    feeds: dict[str, np.ndarray] = {
        "query": query,
        "key": key,
        "value": value,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "write_indices": write_indices,
        "nonpad_kv_seqlen": nonpad_kv_seqlen,
    }
    block_sequence_ids = None
    if use_block_overlay:
        # Synthetic overlay: first ceil(S_q/2) query tokens share block 0, rest
        # are text (-1).  Exercises the bidirectional same-block OR.
        block_sequence_ids = np.full((_BATCH, query_len), -1, dtype=np.int64)
        block_sequence_ids[:, : (query_len + 1) // 2] = 0
        feeds["block_sequence_ids"] = block_sequence_ids

    model = _build_static_cache_bias_graph(
        batch=_BATCH,
        num_q_heads=_NUM_Q_HEADS,
        num_kv_heads=_NUM_KV_HEADS,
        head_dim=_HEAD_DIM,
        max_seq_len=_MAX_SEQ_LEN,
        query_len=query_len,
        sliding_window=_SLIDING_WINDOW,
        use_block_overlay=use_block_overlay,
    )

    session = OnnxModelSession(model)  # CPU EP (MEA external-cache path)
    try:
        out = session.run(feeds)
    finally:
        session.close()

    onnx_attn = out["attn_output"]
    full_k = _scatter_into_cache(key_cache, key, write_idx)
    full_v = _scatter_into_cache(value_cache, value, write_idx)
    ref = _dense_reference(
        query,
        full_k,
        full_v,
        write_indices=write_indices,
        nonpad_kv_seqlen=nonpad_kv_seqlen,
        max_seq_len=_MAX_SEQ_LEN,
        num_q_heads=_NUM_Q_HEADS,
        num_kv_heads=_NUM_KV_HEADS,
        head_dim=_HEAD_DIM,
        sliding_window=_SLIDING_WINDOW,
        block_sequence_ids=block_sequence_ids,
    )
    return onnx_attn, ref


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_prefill_chunk_bias_matches_dense_reference():
    """Prefill chunk (write=0, S_q=N, nonpad=N) matches the dense reference.

    The whole chunk is the valid region; query row ``i`` attends slots within
    ``[max(0, i - sliding_window + 1), i]``, OR-ed with same-block tokens.
    """
    query_len = 6
    onnx_attn, ref = _run_and_compare(
        query_len=query_len,
        write_idx=0,
        nonpad=query_len,
        use_block_overlay=True,
        prefilled_valid_len=0,
        seed=0,
    )
    assert onnx_attn.shape == (_BATCH, query_len, _NUM_Q_HEADS * _HEAD_DIM)
    assert np.isfinite(onnx_attn).all(), "attention output contains NaN/Inf"
    np.testing.assert_allclose(onnx_attn, ref, atol=1e-4, rtol=1e-4)


def test_decode_step_bias_matches_dense_reference():
    """Decode step (write=N, S_q=1, nonpad=N+1) matches the dense reference.

    The single query at absolute position ``N`` attends the ``N`` pre-populated
    past slots plus the just-scattered slot ``N``, clipped by the sliding
    window and padding validity.
    """
    valid_len = 5
    onnx_attn, ref = _run_and_compare(
        query_len=1,
        write_idx=valid_len,
        nonpad=valid_len + 1,
        use_block_overlay=False,
        prefilled_valid_len=valid_len,
        seed=1,
    )
    assert onnx_attn.shape == (_BATCH, 1, _NUM_Q_HEADS * _HEAD_DIM)
    assert np.isfinite(onnx_attn).all(), "attention output contains NaN/Inf"
    np.testing.assert_allclose(onnx_attn, ref, atol=1e-4, rtol=1e-4)


def test_nonpad_padding_clamp_matches_dense_reference():
    """A ``nonpad < S_q`` padding clamp excludes the masked tail (finite + parity).

    Risk (ii) note: unlike the maskless *bottom-right* static-cache path (where
    the chunk aligns to ``nonpad`` and the top ``S_q - nonpad`` query rows can
    fall below the causal frontier and become structurally empty), this bias
    path keys query positions on ``write_indices + arange(S_q)`` (top-left from
    the write slot).  With a contract-consistent feed (``nonpad == write + S_q``)
    every query row therefore attends at least its own diagonal slot, so no row
    is ever fully masked and the all-``dtype.min``-row NaN edge cannot arise in
    normal operation.

    This test still stresses the padding term directly with an *under-clamped*
    ``nonpad`` (``nonpad < S_q`` at ``write=0``): cache slots ``>= nonpad`` are
    masked out, every row falls back to the valid prefix ``[0, nonpad)``, and we
    assert the output stays finite (no NaN from the masked tail) and matches the
    independent dense reference.
    """
    query_len, nonpad = 4, 2  # slots >= 2 are masked out for every query row
    onnx_attn, ref = _run_and_compare(
        query_len=query_len,
        write_idx=0,
        nonpad=nonpad,
        use_block_overlay=False,
        prefilled_valid_len=0,
        seed=2,
    )
    assert np.isfinite(onnx_attn).all(), "masked-tail attention produced NaN/Inf"
    np.testing.assert_allclose(onnx_attn, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("sliding_window", [None, 2, 4])
def test_prefill_sliding_window_variants(sliding_window):
    """Prefill parity across no-window and tight/loose sliding windows.

    Sweeps the local-window geometry directly (the bias's ``Less(dist, w)``
    term) against the dense reference, with no block overlay.
    """
    query_len = 6
    rng = np.random.default_rng(7)
    q_hidden = _NUM_Q_HEADS * _HEAD_DIM
    kv_hidden = _NUM_KV_HEADS * _HEAD_DIM
    query = rng.standard_normal((_BATCH, query_len, q_hidden)).astype(np.float32)
    key = rng.standard_normal((_BATCH, query_len, kv_hidden)).astype(np.float32)
    value = rng.standard_normal((_BATCH, query_len, kv_hidden)).astype(np.float32)
    key_cache = np.zeros((_BATCH, _MAX_SEQ_LEN, kv_hidden), dtype=np.float32)
    value_cache = np.zeros((_BATCH, _MAX_SEQ_LEN, kv_hidden), dtype=np.float32)
    write_indices = np.zeros((_BATCH,), dtype=np.int64)
    nonpad_kv_seqlen = np.full((_BATCH,), query_len, dtype=np.int64)

    model = _build_static_cache_bias_graph(
        batch=_BATCH,
        num_q_heads=_NUM_Q_HEADS,
        num_kv_heads=_NUM_KV_HEADS,
        head_dim=_HEAD_DIM,
        max_seq_len=_MAX_SEQ_LEN,
        query_len=query_len,
        sliding_window=sliding_window,
        use_block_overlay=False,
    )
    session = OnnxModelSession(model)
    try:
        out = session.run(
            {
                "query": query,
                "key": key,
                "value": value,
                "key_cache": key_cache,
                "value_cache": value_cache,
                "write_indices": write_indices,
                "nonpad_kv_seqlen": nonpad_kv_seqlen,
            }
        )
    finally:
        session.close()

    full_k = _scatter_into_cache(key_cache, key, 0)
    full_v = _scatter_into_cache(value_cache, value, 0)
    ref = _dense_reference(
        query,
        full_k,
        full_v,
        write_indices=write_indices,
        nonpad_kv_seqlen=nonpad_kv_seqlen,
        max_seq_len=_MAX_SEQ_LEN,
        num_q_heads=_NUM_Q_HEADS,
        num_kv_heads=_NUM_KV_HEADS,
        head_dim=_HEAD_DIM,
        sliding_window=sliding_window,
        block_sequence_ids=None,
    )
    assert np.isfinite(out["attn_output"]).all()
    np.testing.assert_allclose(out["attn_output"], ref, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Graph-wiring tests (deliverables C + D): models/base.py + flags.static_cache_bias
# ---------------------------------------------------------------------------


def _build_static_cache_model_graph(model_type: str, *, static_cache_bias: bool, **overrides):
    """Build a real tiny static-cache model graph under a flag setting.

    Imports the shared tiny-config helper lazily so this module stays importable
    without the ``tests`` package on ``sys.path`` for the pure-ORT tests above.
    """
    import sys
    from pathlib import Path

    tests_dir = str(Path(__file__).resolve().parent)
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    from _test_configs import _base_config

    from mobius import registry
    from mobius.tasks import CausalLMTask

    config = _base_config(**overrides)
    module = registry.get(model_type)(config)
    with override_flags(static_cache_bias=static_cache_bias):
        pkg = CausalLMTask(static_cache=True, max_seq_len=64).build(module, config)
    return pkg["model"]


def _attention_nodes(model):
    return [n for n in model.graph if n.op_type == "Attention"]


def test_flag_off_emits_maskless_static_cache():
    """flags.static_cache_bias=False: sliding-window model stays maskless.

    Deliverable (D) default-off guarantee: every static-cache ``Attention`` node
    keeps ``is_causal=1`` with no ``attn_mask`` input — the emission is byte-for-
    byte the maskless path, so no shipped model changes when the flag is off.
    """
    model = _build_static_cache_model_graph(
        "mistral", static_cache_bias=False, sliding_window=8
    )
    attns = _attention_nodes(model)
    assert attns, "no Attention nodes found in static-cache graph"
    for node in attns:
        assert node.attributes.get("is_causal").value == 1
        # Attention input #3 (attn_mask) must be absent (None).
        assert node.inputs[3] is None


def test_flag_on_threads_bias_for_sliding_window_model():
    """flags.static_cache_bias=True + sliding_window set: bias threaded, is_causal=0.

    Deliverables (C)+(D): the sliding-window model now emits a float additive
    bias as Attention input #3, STRICTLY paired with ``is_causal=0`` (never
    ``is_causal=1`` + bias — risk iii).  ``nonpad_kv_seqlen`` stays as input #6.
    """
    model = _build_static_cache_model_graph(
        "mistral", static_cache_bias=True, sliding_window=8
    )
    attns = _attention_nodes(model)
    assert attns, "no Attention nodes found in static-cache graph"
    for node in attns:
        assert node.attributes.get("is_causal").value == 0, (
            "bias-present Attention must use is_causal=0 (risk iii double-causal)"
        )
        assert node.inputs[3] is not None, "additive bias must be Attention input #3"
        # nonpad_kv_seqlen preserved as input #6.
        assert len(node.inputs) >= 7 and node.inputs[6] is not None


def test_flag_on_non_sliding_model_stays_maskless():
    """flags.static_cache_bias=True but no bias need: full-attention model maskless.

    Deliverable (C) gate: a model that declares no bias need (``sliding_window``
    is ``None``, e.g. Llama) must keep the maskless ``is_causal=1`` emission even
    when the flag is on — the bias is opt-in *and* capability-gated, so standard
    decoders are unaffected.
    """
    model = _build_static_cache_model_graph("llama", static_cache_bias=True)
    attns = _attention_nodes(model)
    assert attns, "no Attention nodes found in static-cache graph"
    for node in attns:
        assert node.attributes.get("is_causal").value == 1
        assert node.inputs[3] is None
