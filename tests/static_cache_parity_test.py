# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Static-cache numerical-parity tests (mobius #329).

These tests guard the maskless static-cache attention combination that mobius
emits today — ONNX ``Attention`` with ``is_causal=1`` + ``nonpad_kv_seqlen``
(Attention input #6) fed by two ``TensorScatter`` writes into a pre-allocated
KV cache (``src/mobius/components/_attention.py`` static branch).  That graph is
Flash-eligible but, until onnx/onnx#8068 (the ``is_causal`` *bottom-right*
errata) and microsoft/onnxruntime#28958 (the CUDA kernel that removes the
``causal_cross_no_past`` reject and feeds ``nonpad_kv_seqlen`` into the Flash
``seqlens_k`` frontier) ship in a pinned ORT release, the kernel rejects the
combination with ``NOT_IMPLEMENTED`` whenever ``S_q != total_kv`` with no
``past_key`` — which is *both* prefill and decode against a pre-allocated cache.

Two tests:

* :func:`TestStaticCacheParity.test_static_vs_dynamic_vs_hf_decode` — the
  end-to-end parity guard #329 asks for.  Builds a real tiny causal LM twice
  from the *same* weights (static cache and dynamic cache) and drives N decode
  steps, asserting the per-step token IDs and last-token logits agree across
  **static cache**, **dynamic cache**, and **HuggingFace** PyTorch.

* :func:`test_chunked_prefill_structurally_empty_rows_are_zero` — the
  highest-value correctness check.  It exercises a chunked-prefill step where
  ``q_seq > 1`` and ``nonpad_kv_seqlen[b] < q_seq`` for the batch row, so the
  bottom-right causal frontier leaves the *top* query rows with **no** valid
  keys.  The onnx#8068/ort#28958 fix requires those structurally-empty rows to
  be **exactly 0** (the ``LaunchZeroFullyMaskedRows`` guard), *not* NaN and
  *not* mean-of-V.  Upstream's Python/numpy bottom-right *parity* reference is
  deferred (ort#28958 §2E — correctness is locked only by the C++ goldens), so
  the **exact values** of the surviving (non-empty) rows in this degenerate
  ``nonpad < q_seq`` regime are not yet authoritatively defined; we therefore
  assert only the structural guarantees there (empty rows exactly 0, all
  outputs finite, valid rows non-zero) — this is mobius's own guard for that
  edge.

* :func:`test_prefill_chunk_causal_triangle_matches_reference` — the q_seq>1
  *math* check in the well-defined regime.  A multi-token prefill chunk with
  ``nonpad_kv_seqlen == q_seq`` (write at index 0) has standard bottom-right =
  top-left causal alignment, so query row ``i`` attends keys ``[0, i]``.  We
  verify the kernel reproduces that exact causal triangle against a NumPy
  scaled-dot-product reference, validating the bottom-right causal arithmetic
  for ``q_seq > 1`` (not just single-token decode).

Skip mechanism (documented contract)
-------------------------------------
Both tests require the *fixed* ORT kernel.  Rather than pin to an opaque ORT
version number, we reuse the shared **functional capability probe**
:func:`mobius._testing.ort_capabilities.supports_static_cache_flash` (also used by
the e2e Flash-dispatch test): it builds a tiny static-cache model and runs
prefill on the CUDA Execution Provider.  Pre-#28958 ORT raises
``NOT_IMPLEMENTED`` (→ skip); post-#28958 ORT runs.  It also short-circuits when
no CUDA EP is registered (``TensorScatter`` and the external-cache ``Attention``
path are CUDA-only).  The probe is cached so it runs at most once per process.
This is authoritative (it tests the actual kernel, not a version string) and
self-enables the moment mobius bumps its ORT pin past onnxruntime#28958.

Run::

    pytest tests/static_cache_parity_test.py -m integration -v
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest
import transformers
from mobius._testing.ort_capabilities import CUDA_AVAILABLE, supports_static_cache_flash

from mobius import build
from mobius._configs import ArchitectureConfig
from mobius._constants import OPSET_VERSION
from mobius._testing.comparison import assert_logits_close
from mobius._testing.ort_inference import OnnxModelSession
from mobius._testing.torch_reference import load_torch_model, torch_forward
from mobius.tasks import CausalLMTask

pytestmark = pytest.mark.integration

# Tiny real causal LM used as the parity oracle.  Qwen2.5 is a plain
# ``DecoderLayer`` ``CausalLMModel`` (llama/mistral/qwen2 family), which is
# explicitly supported by the static-cache eligibility guard
# (``_validate_static_cache_support``).  0.5B keeps CI fast.
_PARITY_MODEL_ID = "Qwen/Qwen2.5-0.5B"

# Static cache buffer length.  Must comfortably exceed prompt_len + decode steps.
_MAX_SEQ_LEN = 64

# Number of greedy decode steps to compare.
_DECODE_STEPS = 6


def _require_static_cache_attention() -> None:
    """Skip the calling test unless the fixed static-cache kernel is present.

    Delegates to the shared functional probe
    :func:`mobius._testing.ort_capabilities.supports_static_cache_flash` (cached), which
    runs the maskless ``is_causal=1`` + ``nonpad_kv_seqlen`` + ``TensorScatter``
    combination on the CUDA EP — the exact thing onnxruntime#28958 enables.
    """
    if not supports_static_cache_flash():
        if not CUDA_AVAILABLE:
            pytest.skip(
                "CUDA Execution Provider not available; static-cache "
                "TensorScatter + external-cache Attention is CUDA-only."
            )
        pytest.skip(
            "Installed ORT cannot run maskless is_causal=1 + nonpad_kv_seqlen "
            "+ TensorScatter on CUDA (needs onnxruntime#28958)."
        )


# ---------------------------------------------------------------------------
# Standalone attention-graph builder (for the graph-level edge tests)
# ---------------------------------------------------------------------------


def _build_static_cache_attention_graph(
    *,
    batch: int,
    num_heads: int,
    head_dim: int,
    max_seq_len: int,
    query_len: int,
    dtype: ir.DataType,
) -> ir.Model:
    """Build a standalone graph mirroring the static-cache attention branch.

    Replicates ``_apply_attention``'s static path exactly: scatter ``key`` /
    ``value`` into pre-allocated caches via ``TensorScatter``, then run a
    maskless ``Attention`` with ``is_causal=1`` and ``nonpad_kv_seqlen``
    (input #6).  No ``attn_mask`` / ``past_key`` — the Flash-eligible form.

    Inputs:  query/key/value ``[B, query_len, num_heads*head_dim]`` (3-D, the
             format mobius emits, reshaped internally via ``q_num_heads`` /
             ``kv_num_heads``), ``key_cache`` / ``value_cache``
             ``[B, max_seq_len, num_heads*head_dim]``, ``write_indices`` and
             ``nonpad_kv_seqlen`` ``[B]`` int64.
    Outputs: ``attn_output`` ``[B, query_len, num_heads*head_dim]`` plus the
             updated caches.
    """
    from onnxscript import GraphBuilder

    hidden = num_heads * head_dim

    def _value(name: str, dims: list[int], dt: ir.DataType) -> ir.Value:
        return ir.Value(
            name=name,
            shape=ir.Shape(dims),
            type=ir.TensorType(dt),
        )

    query = _value("query", [batch, query_len, hidden], dtype)
    key = _value("key", [batch, query_len, hidden], dtype)
    value = _value("value", [batch, query_len, hidden], dtype)
    key_cache = _value("key_cache", [batch, max_seq_len, hidden], dtype)
    value_cache = _value("value_cache", [batch, max_seq_len, hidden], dtype)
    write_indices = _value("write_indices", [batch], ir.DataType.INT64)
    nonpad_kv_seqlen = _value("nonpad_kv_seqlen", [batch], ir.DataType.INT64)

    graph = ir.Graph(
        inputs=[
            query,
            key,
            value,
            key_cache,
            value_cache,
            write_indices,
            nonpad_kv_seqlen,
        ],
        outputs=[],
        nodes=[],
        name="static_cache_attention_probe",
        opset_imports={"": OPSET_VERSION},
    )
    op = GraphBuilder(graph).op

    # Scatter new K/V into the pre-allocated cache: cache[b, write[b] + t] = upd[b, t].
    updated_k = op.TensorScatter(key_cache, key, write_indices, axis=1)
    updated_v = op.TensorScatter(value_cache, value, write_indices, axis=1)

    scale = 1.0 / np.sqrt(head_dim)
    attn_output, _, _ = op.Attention(
        query,
        updated_k,
        updated_v,
        None,  # no attn_mask — is_causal handles masking
        None,  # no past_key (full cache is provided)
        None,  # no past_value
        nonpad_kv_seqlen,
        q_num_heads=num_heads,
        kv_num_heads=num_heads,
        scale=float(scale),
        is_causal=1,
        _outputs=3,
    )

    attn_output.name = "attn_output"
    updated_k.name = "updated_key_cache"
    updated_v.name = "updated_value_cache"
    graph.outputs.extend([attn_output, updated_k, updated_v])

    return ir.Model(graph, ir_version=10)


# ---------------------------------------------------------------------------
# Helpers — feeds for the two ONNX cache variants
# ---------------------------------------------------------------------------


def _get_config(model_id: str) -> ArchitectureConfig:
    hf_config = transformers.AutoConfig.from_pretrained(model_id)
    return ArchitectureConfig.from_transformers(hf_config)


def _dynamic_prefill_feeds(config, input_ids, attention_mask, position_ids):
    feeds = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    for i in range(config.num_hidden_layers):
        feeds[f"past_key_values.{i}.key"] = np.zeros(
            (1, config.num_key_value_heads, 0, config.head_dim), dtype=np.float32
        )
        feeds[f"past_key_values.{i}.value"] = np.zeros(
            (1, config.num_key_value_heads, 0, config.head_dim), dtype=np.float32
        )
    return feeds


def _dynamic_decode_feeds(config, input_ids, attention_mask, position_ids, prev_out):
    feeds = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    for i in range(config.num_hidden_layers):
        feeds[f"past_key_values.{i}.key"] = prev_out[f"present.{i}.key"]
        feeds[f"past_key_values.{i}.value"] = prev_out[f"present.{i}.value"]
    return feeds


def _empty_static_caches(config) -> dict[str, np.ndarray]:
    kv_hidden = config.num_key_value_heads * config.head_dim
    caches: dict[str, np.ndarray] = {}
    for i in range(config.num_hidden_layers):
        caches[f"key_cache.{i}"] = np.zeros((1, _MAX_SEQ_LEN, kv_hidden), dtype=np.float32)
        caches[f"value_cache.{i}"] = np.zeros((1, _MAX_SEQ_LEN, kv_hidden), dtype=np.float32)
    return caches


def _static_feeds(config, input_ids, position_ids, write_indices, nonpad, caches):
    feeds: dict[str, np.ndarray] = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "write_indices": write_indices,
        "nonpad_kv_seqlen": nonpad,
    }
    feeds.update(caches)
    return feeds


def _carry_static_caches(config, out) -> dict[str, np.ndarray]:
    caches: dict[str, np.ndarray] = {}
    for i in range(config.num_hidden_layers):
        caches[f"key_cache.{i}"] = out[f"updated_key_cache.{i}"]
        caches[f"value_cache.{i}"] = out[f"updated_value_cache.{i}"]
    return caches


# ---------------------------------------------------------------------------
# Test 1 — end-to-end static vs dynamic vs HuggingFace decode parity (#329)
# ---------------------------------------------------------------------------


@pytest.mark.integration_fast
class TestStaticCacheParity:
    """Static-cache decode output must match dynamic cache and HuggingFace."""

    def test_static_vs_dynamic_vs_hf_decode(self):
        """N greedy decode steps agree across static / dynamic / HF.

        The same HuggingFace-argmax token drives all three decoders in
        lockstep so any divergence reflects a real op-level difference, not
        compounding sampling drift.  We assert both the chosen token IDs and
        the last-token logits agree at every step.
        """
        _require_static_cache_attention()

        config = _get_config(_PARITY_MODEL_ID)

        # Two ONNX exports from the SAME HuggingFace weights.
        dynamic_pkg = build(_PARITY_MODEL_ID, dtype="f32", load_weights=True)
        static_pkg = build(
            _PARITY_MODEL_ID,
            task=CausalLMTask(static_cache=True, max_seq_len=_MAX_SEQ_LEN),
            dtype="f32",
            load_weights=True,
        )
        torch_model, tokenizer = load_torch_model(_PARITY_MODEL_ID)

        prompt = "The capital of France is"
        tokens = tokenizer(prompt, return_tensors="np")
        prompt_ids = tokens["input_ids"].astype(np.int64)
        prompt_len = prompt_ids.shape[1]
        assert prompt_len + _DECODE_STEPS < _MAX_SEQ_LEN

        dynamic_session = OnnxModelSession(dynamic_pkg, device="cuda")
        static_session = OnnxModelSession(static_pkg, device="cuda")
        try:
            # --- Prefill ---
            position_ids = np.arange(prompt_len, dtype=np.int64)[np.newaxis, :]
            attention_mask = np.ones((1, prompt_len), dtype=np.int64)

            hf_logits, hf_kv = torch_forward(
                torch_model, prompt_ids, attention_mask, position_ids
            )

            dyn_out = dynamic_session.run(
                _dynamic_prefill_feeds(config, prompt_ids, attention_mask, position_ids)
            )

            # Static prefill: write the whole prompt starting at index 0.  The
            # bottom-right contract makes nonpad_kv_seqlen the number of valid
            # cache entries AFTER the scatter, i.e. write_indices + query_len.
            static_caches = _empty_static_caches(config)
            write_indices = np.zeros((1,), dtype=np.int64)
            nonpad = np.full((1,), prompt_len, dtype=np.int64)
            stat_out = static_session.run(
                _static_feeds(
                    config, prompt_ids, position_ids, write_indices, nonpad, static_caches
                )
            )

            self._assert_step_parity(
                step="prefill",
                hf_logits=hf_logits[:, -1, :],
                dyn_logits=dyn_out["logits"][:, -1, :],
                stat_logits=stat_out["logits"][:, -1, :],
            )

            static_caches = _carry_static_caches(config, stat_out)
            valid_len = prompt_len

            # --- Decode steps ---
            for step in range(_DECODE_STEPS):
                # HF argmax drives every decoder identically.
                next_token = np.argmax(hf_logits[:, -1, :], axis=-1, keepdims=True)
                decode_ids = next_token.astype(np.int64)
                decode_pos = np.array([[valid_len]], dtype=np.int64)

                hf_mask = np.ones((1, valid_len + 1), dtype=np.int64)
                hf_logits, hf_kv = torch_forward(
                    torch_model, decode_ids, hf_mask, decode_pos, past_key_values=hf_kv
                )

                dyn_mask = np.ones((1, valid_len + 1), dtype=np.int64)
                dyn_out = dynamic_session.run(
                    _dynamic_decode_feeds(config, decode_ids, dyn_mask, decode_pos, dyn_out)
                )

                write_indices = np.full((1,), valid_len, dtype=np.int64)
                nonpad = np.full((1,), valid_len + 1, dtype=np.int64)
                stat_out = static_session.run(
                    _static_feeds(
                        config,
                        decode_ids,
                        decode_pos,
                        write_indices,
                        nonpad,
                        static_caches,
                    )
                )
                static_caches = _carry_static_caches(config, stat_out)
                valid_len += 1

                self._assert_step_parity(
                    step=f"decode-{step}",
                    hf_logits=hf_logits[:, -1, :],
                    dyn_logits=dyn_out["logits"][:, -1, :],
                    stat_logits=stat_out["logits"][:, -1, :],
                )
        finally:
            dynamic_session.close()
            static_session.close()

    @staticmethod
    def _assert_step_parity(*, step, hf_logits, dyn_logits, stat_logits):
        """Static logits must match both the dynamic ONNX and HF references."""
        hf_tok = int(np.argmax(hf_logits, axis=-1)[0])
        dyn_tok = int(np.argmax(dyn_logits, axis=-1)[0])
        stat_tok = int(np.argmax(stat_logits, axis=-1)[0])
        assert stat_tok == dyn_tok == hf_tok, (
            f"[{step}] token mismatch: static={stat_tok} dynamic={dyn_tok} hf={hf_tok}"
        )
        # Static vs dynamic share the same ONNX op set / dtype → tight.
        assert_logits_close(stat_logits, dyn_logits, rtol=1e-3, atol=1e-3)
        # Static vs HuggingFace → integration-grade tolerance.
        assert_logits_close(stat_logits, hf_logits, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Test 2 — chunked-prefill: structurally-empty bottom-right rows must be 0
# ---------------------------------------------------------------------------


def _causal_triangle_reference_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    *,
    num_heads: int,
    head_dim: int,
) -> np.ndarray:
    """NumPy bottom-right causal attention for the ``nonpad == q_seq`` regime.

    When a prefill chunk of ``S_q`` tokens is written at cache index 0 and the
    full chunk is valid (``nonpad_kv_seqlen == S_q``), bottom-right and top-left
    causal alignment coincide: query row ``i`` attends keys ``[0, i]``.  This is
    the well-defined, authoritative reference (unlike the degenerate
    ``nonpad < q_seq`` regime, whose exact values upstream defers to the C++
    goldens — ort#28958 §2E).
    """
    batch, query_len, hidden = query.shape
    assert batch == 1
    scale = 1.0 / np.sqrt(head_dim)
    out = np.zeros((batch, query_len, hidden), dtype=np.float64)

    q = query.reshape(query_len, num_heads, head_dim).astype(np.float64)
    k = key.reshape(query_len, num_heads, head_dim).astype(np.float64)
    v = value.reshape(query_len, num_heads, head_dim).astype(np.float64)

    for row in range(query_len):
        valid = np.arange(0, row + 1)  # causal: attend keys [0, row]
        for head in range(num_heads):
            scores = (q[row, head] @ k[valid, head].T) * scale
            scores = scores - scores.max()
            weights = np.exp(scores)
            weights = weights / weights.sum()
            out[0, row, head * head_dim : (head + 1) * head_dim] = weights @ v[valid, head]
    return out


def test_chunked_prefill_structurally_empty_rows_are_zero():
    """q_seq>1 with nonpad<q_seq: empty bottom-right rows are exactly 0.

    This is the onnx#8068 / onnxruntime#28958 edge: a chunked-prefill step
    where the valid KV length is shorter than the query chunk.  With
    bottom-right alignment the *top* ``q_seq - nonpad`` query rows fall entirely
    below the causal frontier and have no key to attend; the kernel must emit
    exactly ``0`` for those rows (``LaunchZeroFullyMaskedRows``) — never NaN,
    never mean-of-V.

    The exact values of the surviving rows in this degenerate regime are
    governed by ORT's C++ goldens (upstream's bottom-right *parity* numpy
    reference is deferred — ort#28958 §2E), so we assert only the structural
    guarantees the fix promises: empty rows exactly 0, everything finite, and
    the valid rows genuinely participated (non-zero).  The exact bottom-right
    causal arithmetic for ``q_seq > 1`` is checked in the well-defined
    ``nonpad == q_seq`` regime by
    :func:`test_prefill_chunk_causal_triangle_matches_reference`.
    """
    _require_static_cache_attention()

    batch, num_heads, head_dim = 1, 2, 8
    max_seq_len, query_len, nonpad_value = 8, 4, 2
    hidden = num_heads * head_dim
    rng = np.random.default_rng(0)

    query = rng.standard_normal((batch, query_len, hidden)).astype(np.float32)
    key = rng.standard_normal((batch, query_len, hidden)).astype(np.float32)
    value = rng.standard_normal((batch, query_len, hidden)).astype(np.float32)

    model = _build_static_cache_attention_graph(
        batch=batch,
        num_heads=num_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        query_len=query_len,
        dtype=ir.DataType.FLOAT,
    )
    feeds = {
        "query": query,
        "key": key,
        "value": value,
        "key_cache": np.zeros((batch, max_seq_len, hidden), dtype=np.float32),
        "value_cache": np.zeros((batch, max_seq_len, hidden), dtype=np.float32),
        # Write the whole chunk at position 0 but mark only `nonpad_value`
        # entries valid, so the top (query_len - nonpad_value) rows are empty.
        "write_indices": np.zeros((batch,), dtype=np.int64),
        "nonpad_kv_seqlen": np.full((batch,), nonpad_value, dtype=np.int64),
    }

    session = OnnxModelSession(model, device="cuda")
    try:
        out = session.run(feeds)
    finally:
        session.close()

    attn = out["attn_output"]
    assert attn.shape == (batch, query_len, hidden)
    assert np.isfinite(attn).all(), "attention output contains NaN/Inf"

    num_empty = query_len - nonpad_value  # rows [0, num_empty) are structurally empty
    empty_rows = attn[0, :num_empty, :]
    assert np.array_equal(empty_rows, np.zeros_like(empty_rows)), (
        "structurally-empty bottom-right rows must be exactly 0, got "
        f"max|row|={np.abs(empty_rows).max()}"
    )

    # The surviving rows must have actually attended something (non-zero,
    # finite).  We deliberately do NOT assert their exact values: upstream's
    # bottom-right numpy parity reference for nonpad < q_seq is deferred.
    valid_rows = attn[0, num_empty:, :]
    assert np.isfinite(valid_rows).all()
    assert np.abs(valid_rows).max() > 0.0, "valid rows unexpectedly all-zero"


def test_prefill_chunk_causal_triangle_matches_reference():
    """q_seq>1 prefill (nonpad==q_seq): kernel matches the causal-triangle ref.

    Validates the bottom-right causal arithmetic for a multi-token chunk in the
    well-defined regime where every query position has at least one valid key.
    Query row ``i`` must attend exactly keys ``[0, i]`` and match a NumPy
    scaled-dot-product reference.
    """
    _require_static_cache_attention()

    batch, num_heads, head_dim = 1, 2, 8
    max_seq_len, query_len = 8, 4
    hidden = num_heads * head_dim
    rng = np.random.default_rng(1)

    query = rng.standard_normal((batch, query_len, hidden)).astype(np.float32)
    key = rng.standard_normal((batch, query_len, hidden)).astype(np.float32)
    value = rng.standard_normal((batch, query_len, hidden)).astype(np.float32)

    model = _build_static_cache_attention_graph(
        batch=batch,
        num_heads=num_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        query_len=query_len,
        dtype=ir.DataType.FLOAT,
    )
    feeds = {
        "query": query,
        "key": key,
        "value": value,
        "key_cache": np.zeros((batch, max_seq_len, hidden), dtype=np.float32),
        "value_cache": np.zeros((batch, max_seq_len, hidden), dtype=np.float32),
        # Full chunk valid: write at 0, nonpad == query_len → standard causal.
        "write_indices": np.zeros((batch,), dtype=np.int64),
        "nonpad_kv_seqlen": np.full((batch,), query_len, dtype=np.int64),
    }

    session = OnnxModelSession(model, device="cuda")
    try:
        out = session.run(feeds)
    finally:
        session.close()

    attn = out["attn_output"]
    assert attn.shape == (batch, query_len, hidden)
    assert np.isfinite(attn).all(), "attention output contains NaN/Inf"

    reference = _causal_triangle_reference_attention(
        query, key, value, num_heads=num_heads, head_dim=head_dim
    )
    np.testing.assert_allclose(
        attn,
        reference,
        rtol=1e-3,
        atol=1e-3,
        err_msg="prefill chunk diverges from causal-triangle reference",
    )
