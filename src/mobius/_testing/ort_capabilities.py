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

The probe is **fail-closed but not fail-silent**: it never raises, so callers
can use it directly as a ``pytest.skip`` predicate and the suite stays green on
arbitrary infrastructure.  But it distinguishes the *expected* pre-#28958 kernel
reject (logged at debug) from an *unexpected* failure — a transient CUDA OOM,
onnxscript/ORT drift, or a broken install (logged at warning with a traceback).
Without that distinction a broken probe would silently cache ``False`` and make
the entire static-cache suite skip with green CI and zero coverage, while
:func:`static_cache_flash_skip_reason` misreported the cause as "needs #28958".
"""

from __future__ import annotations

import enum
import functools
import logging
import tempfile
from pathlib import Path

import numpy as np
import onnx_ir as ir
import onnxruntime as ort

from mobius._constants import OPSET_VERSION

logger = logging.getLogger(__name__)

# ORT raises one of these from ``session.run`` when the CUDA Attention kernel
# rejects the maskless ``is_causal=1`` + ``nonpad_kv_seqlen`` (no ``past_key``)
# combination on a build predating microsoft/onnxruntime#28958 — the *expected*
# "needs the fix" signal, raised from ``ComputeInternal`` (proven by ORT's own
# deleted reject test).  Treating only these as expected keeps build/serialize/
# session-create failures (the symptoms of a broken probe) loud.  Imported
# defensively so a pybind layout change cannot break import; falls back to the
# common base class of every ORT pybind error.
try:
    from onnxruntime.capi import onnxruntime_pybind11_state as _ort_capi

    _EXPECTED_REJECT_ERRORS: tuple[type[BaseException], ...] = (
        _ort_capi.NotImplemented,
        _ort_capi.Fail,
    )
except Exception:  # pragma: no cover - defensive against pybind layout drift
    _EXPECTED_REJECT_ERRORS = (RuntimeError,)

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


class _ProbeOutcome(enum.Enum):
    """Classification of the static-cache Flash probe — drives the skip reason.

    Separating ``NEEDS_FIX`` (the expected pre-#28958 kernel reject) from
    ``PROBE_ERROR`` (an unexpected failure) is what lets the probe stay
    fail-closed without being fail-silent.
    """

    SUPPORTED = "supported"
    NO_CUDA = "no_cuda"
    NEEDS_FIX = "needs_fix"  # expected: ORT lacks microsoft/onnxruntime#28958
    PROBE_ERROR = "probe_error"  # unexpected: must not silently disable the gate


@functools.lru_cache(maxsize=1)
def _probe_static_cache_flash() -> _ProbeOutcome:
    """Run the functional capability probe once and classify the outcome.

    Builds the minimal ``TensorScatter`` + maskless ``Attention`` (``is_causal=1``
    + ``nonpad_kv_seqlen``) graph and runs it on the CUDA EP.  The result is
    cached for the process (the probe runs at most once).

    Fail-closed (never raises) but NOT fail-silent: an *expected* kernel reject
    (:data:`_EXPECTED_REJECT_ERRORS` from ``session.run``) is logged at debug and
    maps to :attr:`_ProbeOutcome.NEEDS_FIX`, while any *unexpected* failure — a
    build/serialize/session error, a non-reject ``session.run`` error, or the
    CUDA EP silently dropping out of the session — is logged at warning with a
    traceback and maps to :attr:`_ProbeOutcome.PROBE_ERROR` so a broken probe
    cannot silently disable the static-cache gate.
    """
    if not CUDA_AVAILABLE:
        return _ProbeOutcome.NO_CUDA
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
                logger.warning(
                    "Static-cache Flash probe: CUDA EP is registered globally but "
                    "absent from the probe session providers (%s); treating the "
                    "static-cache CUDA path as unsupported.",
                    session.get_providers(),
                )
                return _ProbeOutcome.PROBE_ERROR
            try:
                session.run(None, _probe_feeds())
            except _EXPECTED_REJECT_ERRORS as exc:
                logger.debug(
                    "Static-cache Flash probe: CUDA Attention kernel rejected the "
                    "maskless is_causal=1 + nonpad_kv_seqlen combination — expected "
                    "without microsoft/onnxruntime#28958 (%s).",
                    exc,
                )
                return _ProbeOutcome.NEEDS_FIX
    except Exception:
        logger.warning(
            "Static-cache Flash capability probe failed unexpectedly; treating the "
            "static-cache CUDA path as unsupported so dependent tests SKIP rather "
            "than error. This is fail-closed but may mask a real regression "
            "(transient CUDA OOM, onnxscript/ORT drift, broken install) — "
            "investigate if this GPU is expected to support the path.",
            exc_info=True,
        )
        return _ProbeOutcome.PROBE_ERROR
    return _ProbeOutcome.SUPPORTED


def supports_static_cache_flash() -> bool:
    """Return ``True`` when the installed ORT runs the static-cache graph on CUDA.

    Thin boolean view over :func:`_probe_static_cache_flash` (which is cached and
    handles all logging).  ORT builds lacking microsoft/onnxruntime#28958 reject
    the maskless ``is_causal=1`` + ``nonpad_kv_seqlen`` combination → ``False``;
    post-fix ORT runs it → ``True``.

    **Fail-closed:** returns ``False`` rather than raising on *any* failure — no
    CUDA EP, an expected kernel reject, or an unexpected probe error — so it is
    safe to call directly as a skip predicate on arbitrary infrastructure.  Use
    :func:`static_cache_flash_skip_reason` when the *cause* matters.
    """
    return _probe_static_cache_flash() is _ProbeOutcome.SUPPORTED


def static_cache_flash_skip_reason() -> str | None:
    """Return ``None`` when the static-cache Flash path runs here, else a reason.

    A convenience wrapper for tests that skip via ``pytest.skip(...)`` inside a
    fixture/helper rather than a module-level ``skipif`` marker::

        reason = static_cache_flash_skip_reason()
        if reason:
            pytest.skip(reason)

    The returned string names the *true* cause — no CUDA EP, a build predating
    microsoft/onnxruntime#28958, or an unexpected probe failure — so a broken
    probe is never misattributed to the missing fix (the message points at the
    logged warning instead).
    """
    outcome = _probe_static_cache_flash()
    if outcome is _ProbeOutcome.SUPPORTED:
        return None
    if outcome is _ProbeOutcome.NO_CUDA:
        return (
            "CUDA Execution Provider not available; static-cache TensorScatter "
            "+ external-cache Attention is CUDA-only."
        )
    if outcome is _ProbeOutcome.NEEDS_FIX:
        return (
            "installed ORT cannot run maskless is_causal=1 + nonpad_kv_seqlen "
            "+ TensorScatter on CUDA (needs microsoft/onnxruntime#28958)."
        )
    return (
        "static-cache Flash capability probe failed unexpectedly (see the logged "
        "warning for the traceback); treating the path as unsupported."
    )
