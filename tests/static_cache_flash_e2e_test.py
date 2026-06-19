# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""End-to-end CUDA regression test for the static (TensorScatter) KV cache.

mobius ``--static-cache`` decoders emit the opset-24 ONNX-domain ``Attention``
op with ``is_causal=1`` + ``nonpad_kv_seqlen`` (no ``attn_mask``, no
``past_key``) right after two ``TensorScatter`` writes that update a
pre-allocated KV cache in place.  Historically the CUDA kernel rejected that
combination (``NOT_IMPLEMENTED``) for both prefill (``S_q = N``) and decode
(``S_q = 1``) against a cache pre-sized to ``max_seq_len``, because the ONNX
``is_causal`` spec was top-left aligned (wrong for cached decode / chunked
prefill against an external cache).

microsoft/onnxruntime#28958 (with the onnx/onnx#8068 errata) corrects
``is_causal`` to bottom-right alignment, removes the reject, and makes the
combination **Flash-eligible** — feeding ``nonpad_kv_seqlen[b]`` into Flash's
per-batch ``seqlens_k`` so the bottom-right frontier is native.

This test re-adds the e2e coverage dropped with the aborted PR #340, adapted to
the current **maskless** graph on ``main``.  It does two things on the CUDA EP:

1. :func:`test_static_cache_prefill_and_decode_run_on_cuda` — runs prefill then
   a decode step and asserts the outputs are finite and the in-place cache
   advanced.  Reverting ORT to the pre-#28958 kernel makes this raise
   ``NOT_IMPLEMENTED``.

2. :func:`test_static_cache_attention_dispatches_flash_on_cuda` — asserts the
   ONNX-domain ``Attention`` actually runs on the **Flash** kernel (not
   Memory-Efficient / unfused) for fp16, via the kernel's own ``VERBOSE``
   dispatch log (see :mod:`_static_cache_support`).

Both skip when the installed ORT lacks #28958 (functional probe), so the file
is committed now and self-enables once a fixed ORT is pinned.  The Flash
assertion additionally skips under ``onnxruntime_QUICK_BUILD`` (Flash compiled
for head_dim 128 only there → false negatives).
"""

from __future__ import annotations

import tempfile

import numpy as np
import onnxruntime as ort
import pytest
from _static_cache_support import (
    CUDA_AVAILABLE,
    DEFAULT_MAX_SEQ_LEN,
    attention_kernels_from_log,
    build_static_cache_model,
    captured_attention_dispatch,
    carry_caches,
    empty_caches,
    quick_build_enabled,
    static_cache_cuda_supported,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not CUDA_AVAILABLE,
        reason="static-cache TensorScatter / external-cache Attention are CUDA-only",
    ),
    pytest.mark.skipif(
        not static_cache_cuda_supported(),
        reason=(
            "installed ORT lacks microsoft/onnxruntime#28958 "
            "(is_causal=1 + nonpad_kv_seqlen on the CUDA Attention kernel)"
        ),
    ),
]


def _build_session(tmp_dir: str) -> tuple[ort.InferenceSession, object]:
    """Build the tiny fp16 static-cache graph and load it on CUDA."""
    model_path, config = build_static_cache_model(tmp_dir)
    session = ort.InferenceSession(
        model_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    assert "CUDAExecutionProvider" in session.get_providers(), (
        "static-cache e2e test must run on the CUDA Execution Provider"
    )
    return session, config


def _prefill(session: ort.InferenceSession, config: object, prefill_len: int):
    """Run prefill (write from slot 0, ``S_q = prefill_len``) and return outputs."""
    output_names = [out.name for out in session.get_outputs()]
    rng = np.random.default_rng(1)
    feeds: dict[str, np.ndarray] = {
        "input_ids": rng.integers(0, config.vocab_size, size=(1, prefill_len), dtype=np.int64),
        "position_ids": np.arange(prefill_len, dtype=np.int64)[None, :],
        "write_indices": np.array([0], dtype=np.int64),
        "nonpad_kv_seqlen": np.array([prefill_len], dtype=np.int64),
    }
    feeds.update(empty_caches(config))
    return output_names, dict(zip(output_names, session.run(output_names, feeds)))


def _decode_feeds(config: object, prefill_out: dict[str, np.ndarray], step: int):
    """Single-token decode feed at slot ``prefill_len + step`` carrying the cache."""
    prefill_len = 4
    write_index = prefill_len + step
    rng = np.random.default_rng(100 + step)
    feeds: dict[str, np.ndarray] = {
        "input_ids": rng.integers(0, config.vocab_size, size=(1, 1), dtype=np.int64),
        "position_ids": np.array([[write_index]], dtype=np.int64),
        "write_indices": np.array([write_index], dtype=np.int64),
        "nonpad_kv_seqlen": np.array([write_index + 1], dtype=np.int64),
    }
    feeds.update(carry_caches(prefill_out, config.num_hidden_layers))
    return feeds


def test_static_cache_prefill_and_decode_run_on_cuda():
    """Prefill (``S_q > 1``) and decode (``S_q = 1``) both run without errors.

    Loads a real static-cache graph and runs both phases on the CUDA EP.  The
    pre-#28958 kernel raises ``NOT_IMPLEMENTED`` here; the fixed kernel runs and
    advances the in-place KV cache.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        session, config = _build_session(tmp_dir)
        kv_hidden = config.num_key_value_heads * config.head_dim
        vocab = config.vocab_size

        # --- Prefill: write N tokens from slot 0 (S_q = N != max_seq). ---
        prefill_len = 4
        output_names, prefill_out = _prefill(session, config, prefill_len)
        prefill_logits = prefill_out["logits"]
        assert prefill_logits.shape == (1, prefill_len, vocab)
        assert np.isfinite(prefill_logits).all(), "prefill logits must be finite"

        # --- Decode two tokens one at a time (S_q = 1 each), carrying cache. ---
        carried = prefill_out
        for step in range(2):
            decode_feeds = _decode_feeds(config, carried, step)
            decode_out = dict(zip(output_names, session.run(output_names, decode_feeds)))
            decode_logits = decode_out["logits"]
            assert decode_logits.shape == (1, 1, vocab)
            assert np.isfinite(decode_logits).all(), (
                f"decode logits at step {step} must be finite"
            )

            # The decode step must have scattered its key into the previously
            # empty tail slot, confirming the in-place cache advanced.
            write_index = prefill_len + step
            advanced_key = decode_out["updated_key_cache.0"]
            assert advanced_key.shape == (1, DEFAULT_MAX_SEQ_LEN, kv_hidden)
            assert np.any(advanced_key[0, write_index] != 0), (
                f"decode step {step} should scatter the new key into cache slot {write_index}"
            )
            carried = decode_out


@pytest.mark.skipif(
    quick_build_enabled(),
    reason=(
        "onnxruntime_QUICK_BUILD compiles the CUDA Flash kernel for head_dim 128 "
        "only; the tiny head_dim 64 geometry would route to MEA (false negative)"
    ),
)
def test_static_cache_attention_dispatches_flash_on_cuda():
    """The fp16 maskless static-cache Attention must run on the **Flash** kernel.

    Flash — not Memory-Efficient or unfused attention — is the whole point of
    the ``is_causal=1 + nonpad_kv_seqlen`` (no ``attn_mask``) external-cache
    path enabled by microsoft/onnxruntime#28958.  This asserts the dispatch
    directly: it captures the ONNX-domain Attention kernel's ``VERBOSE`` log
    (``ONNX Attention: using Flash Attention``) emitted from
    ``onnxruntime/core/providers/cuda/llm/attention.cc`` during both a prefill
    (``q_seq > 1``) and a decode (``q_seq = 1``) run.

    Eligibility (and therefore Flash selection) holds because the fp16 graph
    has ``head_size = 64`` (<= 256, multiple of 8), ``head_size == v_head_size``,
    ``attn_mask == nullptr``, no ``past_key`` (full cache supplied), and the
    test requires an Ampere+ (SM >= 8.0) device implicitly — on pre-SM80 the
    kernel falls back and this assertion correctly fails, flagging that Flash is
    unavailable on that hardware.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        session, config = _build_session(tmp_dir)
        prefill_len = 4

        # Prefill (q_seq = prefill_len) under dispatch capture.
        with captured_attention_dispatch() as prefill_log:
            output_names, prefill_out = _prefill(session, config, prefill_len)
        prefill_kernels = attention_kernels_from_log(prefill_log)
        assert prefill_kernels == {"flash"}, (
            "prefill ONNX-domain Attention did not dispatch Flash on CUDA "
            f"(observed kernels: {prefill_kernels or 'none — no dispatch log captured'}; "
            "expected {'flash'})"
        )

        # Decode (q_seq = 1) under dispatch capture — the cached-decode path.
        decode_feeds = _decode_feeds(config, prefill_out, step=0)
        with captured_attention_dispatch() as decode_log:
            session.run(output_names, decode_feeds)
        decode_kernels = attention_kernels_from_log(decode_log)
        assert decode_kernels == {"flash"}, (
            "decode ONNX-domain Attention did not dispatch Flash on CUDA "
            f"(observed kernels: {decode_kernels or 'none — no dispatch log captured'}; "
            "expected {'flash'})"
        )
