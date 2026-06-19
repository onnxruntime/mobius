# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Functional ORT capability probes for mobius tests.

mobius ``--static-cache`` decoders emit the opset-24 ONNX-domain ``Attention``
op with ``is_causal=1`` + ``nonpad_kv_seqlen`` (no ``attn_mask``, no
``past_key``) right after two ``TensorScatter`` writes that update a
pre-allocated KV cache in place.  Running that graph on the CUDA Execution
Provider requires an ORT build that contains microsoft/onnxruntime#28958, which
removes the historical ``NOT_IMPLEMENTED`` reject for ``is_causal=1`` +
``nonpad_kv_seqlen`` (when ``S_q != total_kv`` with no ``past_key``) and makes
the combination Flash-eligible.

:func:`supports_static_cache_flash` is the single, canonical gate every
static-cache CUDA test shares.  It is a *functional* probe — it builds a minimal
``TensorScatter`` + ``Attention`` graph and actually runs it on the CUDA EP —
rather than an ORT version-string check, because the fix landed on ``main``
before any tagged release, so the exact enabling version is build-dependent.

The probe is **fail-closed**: any unexpected failure (no CUDA EP registered,
``NOT_IMPLEMENTED`` on a pre-#28958 build, a serialization or session error,
anything else) returns ``False`` so callers can use it directly as a
``pytest.skip`` predicate and the suite stays green on arbitrary infrastructure.
"""

from __future__ import annotations

import functools
import tempfile
from pathlib import Path

import numpy as np
import onnx_ir as ir
import onnxruntime as ort

from mobius._constants import OPSET_VERSION

__all__ = [
    "CUDA_AVAILABLE",
    "static_cache_flash_skip_reason",
    "supports_static_cache_flash",
]

CUDA_AVAILABLE = "CUDAExecutionProvider" in ort.get_available_providers()

# Minimal probe geometry.  A single attention head with ``head_dim == 64``
# (in ORT's compiled Flash kernel set) is enough to exercise the external-cache
# ``is_causal=1`` + ``nonpad_kv_seqlen`` path.  ``query_len < max_seq_len`` with
# ``write_indices == 0`` reproduces the ``S_q != total_kv`` (no ``past_key``)
# regime that pre-#28958 ORT rejects — both prefill and decode hit it.
_PROBE_BATCH = 1
_PROBE_NUM_HEADS = 1
_PROBE_HEAD_DIM = 64
_PROBE_MAX_SEQ_LEN = 8
_PROBE_QUERY_LEN = 2
_PROBE_HIDDEN = _PROBE_NUM_HEADS * _PROBE_HEAD_DIM

# Flash is fp16/bf16 only on every CUDA path; fp32 routes to MEA/unfused.
_PROBE_DTYPE = ir.DataType.FLOAT16
_PROBE_NP_DTYPE = np.float16


def _build_static_cache_attention_probe() -> ir.Model:
    """Build a standalone graph mirroring the static-cache attention branch.

    Replicates ``_apply_attention``'s static path: scatter ``key`` / ``value``
    into pre-allocated caches via ``TensorScatter``, then run a maskless
    ``Attention`` with ``is_causal=1`` and ``nonpad_kv_seqlen`` (input #6).  No
    ``attn_mask`` / ``past_key`` — the Flash-eligible form #28958 enables.
    """
    from onnxscript import GraphBuilder

    def _value(name: str, dims: list[int], dt: ir.DataType) -> ir.Value:
        return ir.Value(name=name, shape=ir.Shape(dims), type=ir.TensorType(dt))

    query = _value("query", [_PROBE_BATCH, _PROBE_QUERY_LEN, _PROBE_HIDDEN], _PROBE_DTYPE)
    key = _value("key", [_PROBE_BATCH, _PROBE_QUERY_LEN, _PROBE_HIDDEN], _PROBE_DTYPE)
    value = _value("value", [_PROBE_BATCH, _PROBE_QUERY_LEN, _PROBE_HIDDEN], _PROBE_DTYPE)
    key_cache = _value(
        "key_cache", [_PROBE_BATCH, _PROBE_MAX_SEQ_LEN, _PROBE_HIDDEN], _PROBE_DTYPE
    )
    value_cache = _value(
        "value_cache", [_PROBE_BATCH, _PROBE_MAX_SEQ_LEN, _PROBE_HIDDEN], _PROBE_DTYPE
    )
    write_indices = _value("write_indices", [_PROBE_BATCH], ir.DataType.INT64)
    nonpad_kv_seqlen = _value("nonpad_kv_seqlen", [_PROBE_BATCH], ir.DataType.INT64)

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

    scale = 1.0 / np.sqrt(_PROBE_HEAD_DIM)
    attn_output, _, _ = op.Attention(
        query,
        updated_k,
        updated_v,
        None,  # no attn_mask — is_causal handles masking
        None,  # no past_key (full cache is provided)
        None,  # no past_value
        nonpad_kv_seqlen,
        q_num_heads=_PROBE_NUM_HEADS,
        kv_num_heads=_PROBE_NUM_HEADS,
        scale=float(scale),
        is_causal=1,
        _outputs=3,
    )

    attn_output.name = "attn_output"
    updated_k.name = "updated_key_cache"
    updated_v.name = "updated_value_cache"
    graph.outputs.extend([attn_output, updated_k, updated_v])

    return ir.Model(graph, ir_version=10)


def _probe_feeds() -> dict[str, np.ndarray]:
    """Concrete inputs for the probe graph (write from slot 0, ``S_q = query_len``)."""
    rng = np.random.default_rng(0)

    def _rand(shape: tuple[int, ...]) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.1).astype(_PROBE_NP_DTYPE)

    return {
        "query": _rand((_PROBE_BATCH, _PROBE_QUERY_LEN, _PROBE_HIDDEN)),
        "key": _rand((_PROBE_BATCH, _PROBE_QUERY_LEN, _PROBE_HIDDEN)),
        "value": _rand((_PROBE_BATCH, _PROBE_QUERY_LEN, _PROBE_HIDDEN)),
        "key_cache": np.zeros(
            (_PROBE_BATCH, _PROBE_MAX_SEQ_LEN, _PROBE_HIDDEN), dtype=_PROBE_NP_DTYPE
        ),
        "value_cache": np.zeros(
            (_PROBE_BATCH, _PROBE_MAX_SEQ_LEN, _PROBE_HIDDEN), dtype=_PROBE_NP_DTYPE
        ),
        "write_indices": np.array([0], dtype=np.int64),
        "nonpad_kv_seqlen": np.array([_PROBE_QUERY_LEN], dtype=np.int64),
    }


@functools.lru_cache(maxsize=1)
def supports_static_cache_flash() -> bool:
    """Return ``True`` when the installed ORT runs the static-cache graph on CUDA.

    Builds the minimal ``TensorScatter`` + maskless ``Attention`` (``is_causal=1``
    + ``nonpad_kv_seqlen``) graph and runs it on the CUDA EP.  ORT builds lacking
    microsoft/onnxruntime#28958 raise ``NOT_IMPLEMENTED`` for the combination →
    ``False``; post-fix ORT runs it → ``True``.  The result is cached for the
    test session (the probe runs at most once per process).

    **Fail-closed:** returns ``False`` rather than raising on *any* failure — no
    CUDA EP, ``NOT_IMPLEMENTED``, or any other exception — so it is safe to call
    directly as a skip predicate on arbitrary infrastructure.
    """
    if not CUDA_AVAILABLE:
        return False
    try:
        model = _build_static_cache_attention_probe()
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = str(Path(tmp_dir) / "probe.onnx")
            ir.save(model, model_path)
            session = ort.InferenceSession(
                model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            if "CUDAExecutionProvider" not in session.get_providers():
                return False
            session.run(None, _probe_feeds())
        return True
    except Exception:
        return False


def static_cache_flash_skip_reason() -> str | None:
    """Return ``None`` when the static-cache Flash path runs here, else a reason.

    A convenience wrapper over :func:`supports_static_cache_flash` for tests that
    skip via ``pytest.skip(...)`` inside a fixture/helper rather than a
    module-level ``skipif`` marker::

        reason = static_cache_flash_skip_reason()
        if reason:
            pytest.skip(reason)

    The returned string distinguishes the two skip causes — no CUDA Execution
    Provider registered vs. a CUDA build that predates microsoft/onnxruntime#28958
    — so failures-to-run are self-explanatory in the test report.
    """
    if supports_static_cache_flash():
        return None
    if not CUDA_AVAILABLE:
        return (
            "CUDA Execution Provider not available; static-cache TensorScatter "
            "+ external-cache Attention is CUDA-only."
        )
    return (
        "installed ORT cannot run maskless is_causal=1 + nonpad_kv_seqlen "
        "+ TensorScatter on CUDA (needs microsoft/onnxruntime#28958)."
    )
