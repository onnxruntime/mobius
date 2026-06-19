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
   dispatch log (captured by :func:`captured_attention_dispatch` below).

Both skip when the installed ORT lacks #28958 (functional probe), so the file
is committed now and self-enables once a fixed ORT is pinned.  The Flash
assertion additionally skips under ``onnxruntime_QUICK_BUILD`` (Flash compiled
for head_dim 128 only there → false negatives) and on pre-Ampere GPUs with
compute capability < 8.0 (:func:`_flash_capable_gpu`), where the kernel routes
to Memory-Efficient Attention instead of Flash.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

from mobius import ArchitectureConfig, build_from_module
from mobius._registry import registry
from mobius._testing.ort_capabilities import CUDA_AVAILABLE, supports_static_cache_flash
from mobius.tasks import CausalLMTask

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not CUDA_AVAILABLE,
        reason="static-cache TensorScatter / external-cache Attention are CUDA-only",
    ),
    pytest.mark.skipif(
        not supports_static_cache_flash(),
        reason=(
            "installed ORT lacks microsoft/onnxruntime#28958 "
            "(is_causal=1 + nonpad_kv_seqlen on the CUDA Attention kernel)"
        ),
    ),
]


# ---------------------------------------------------------------------------
# Tiny model geometry
# ---------------------------------------------------------------------------
# qwen2 is a plain GQA causal LM with no softcapping/quirks — a clean carrier
# for the static-cache Attention path.  head_dim=64 sits in ORT's compiled
# Flash kernel set (multiples of 8, <= 256) on a normal build; QUICK_BUILD
# compiles head_dim 128 only (see ``quick_build_enabled``).  num_kv_heads <
# num_heads exercises the external-cache *GQA* Flash path.
_MODEL_TYPE = "qwen2"
DEFAULT_MAX_SEQ_LEN = 16
# Prefill writes this many tokens from slot 0, so ``S_q = _PREFILL_LEN !=
# total_kv`` (no ``past_key``) — the regime the static-cache Flash path targets.
# Decode then writes single tokens into the tail slots that follow.
_PREFILL_LEN = 4
_FLASH_HEAD_DIM = 64
_FLASH_NUM_HEADS = 2
_FLASH_NUM_KV_HEADS = 1
_FLASH_HIDDEN = _FLASH_HEAD_DIM * _FLASH_NUM_HEADS

# Flash is fp16/bf16 only on every CUDA path; fp32 routes to MEA/unfused.
FLASH_CACHE_DTYPE = np.float16

# Substrings of the ``ONNX Attention: using ...`` VERBOSE log, keyed by the
# canonical kernel name reported by :func:`attention_kernels_from_log`.
_KERNEL_LOG_MARKERS = {
    "flash": "using Flash Attention",
    "memory_efficient": "using Memory Efficient Attention",
    "unfused": "using Unfused Attention",
}


def quick_build_enabled() -> bool:
    """Return ``True`` when ORT was built with ``onnxruntime_QUICK_BUILD=ON``.

    There is no runtime API to detect a QUICK_BUILD ORT, so CI that runs against
    such a build must export ``onnxruntime_QUICK_BUILD=1``.  Under QUICK_BUILD
    the CUDA Flash kernel is compiled for ``head_dim == 128`` only, so other
    shapes silently route to Memory-Efficient Attention and a Flash-dispatch
    assertion would be a false negative.
    """
    return os.environ.get("onnxruntime_QUICK_BUILD", "").strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }


def _cuda_compute_capability() -> tuple[int, int] | None:
    """Return the active CUDA device's ``(major, minor)`` compute capability.

    Returns ``None`` when it cannot be determined (torch missing, no CUDA device
    visible, or a query error) so callers treat "unknown" as "not Flash-capable"
    and skip conservatively.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_capability(0)
    except Exception:
        return None


def _flash_capable_gpu() -> bool:
    """``True`` only on Ampere+ (SM >= 8.0), where the CUDA Flash kernel exists.

    The ONNX-domain Attention Flash kernel requires compute capability >= 8.0;
    on pre-SM80 GPUs (e.g. T4/Volta, SM 7.x) the same graph routes to
    Memory-Efficient Attention, so a Flash-dispatch assertion would hard-fail
    rather than skip.  Used to guard
    :func:`test_static_cache_attention_dispatches_flash_on_cuda`.
    """
    capability = _cuda_compute_capability()
    return capability is not None and capability >= (8, 0)


def _fill_random_weights(model: ir.Model, rng: np.random.Generator) -> None:
    """Fill empty initializers with small random values of their own dtype.

    The graph is built without real weights; ORT still needs concrete
    initializers to run.  Small values keep logits finite and well-scaled.
    """
    for initializer in model.graph.initializers.values():
        if initializer.const_value is not None:
            continue
        shape = initializer.shape
        dims = [d if isinstance(d, int) else 1 for d in shape] if shape else [1]
        dtype = initializer.dtype or ir.DataType.FLOAT
        np_dtype = dtype.numpy()
        if np.issubdtype(np_dtype, np.floating):
            data = (rng.standard_normal(dims) * 0.02).astype(np_dtype)
        else:
            data = np.zeros(dims, dtype=np_dtype)
        initializer.const_value = ir.Tensor(data)


def build_static_cache_model(
    tmp_dir: str,
    *,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    dtype: ir.DataType = ir.DataType.FLOAT16,
    seed: int = 0,
) -> tuple[str, ArchitectureConfig]:
    """Build a tiny static-cache qwen2 graph and serialise it to ``tmp_dir``.

    Built with ``execution_provider="default"`` so the static-cache
    ``Attention`` stays an ONNX-domain op (no CUDA GQA fusion) — that is the op
    microsoft/onnxruntime#28958 fixes and the one whose kernel selection this
    test asserts.  The model is loaded later on the CUDA EP at session time.

    Returns the on-disk model path and the :class:`ArchitectureConfig` used
    (callers need ``num_hidden_layers`` / ``num_key_value_heads`` / ``head_dim``
    / ``vocab_size`` to shape the feeds).
    """
    from _test_configs import _base_config

    config = _base_config(
        num_attention_heads=_FLASH_NUM_HEADS,
        num_key_value_heads=_FLASH_NUM_KV_HEADS,
        head_dim=_FLASH_HEAD_DIM,
        hidden_size=_FLASH_HIDDEN,
        dtype=dtype,
    )
    module = registry.get(_MODEL_TYPE)(config)
    task = CausalLMTask(static_cache=True, max_seq_len=max_seq_len)
    model = build_from_module(module, config, task=task, execution_provider="default")["model"]
    _fill_random_weights(model, np.random.default_rng(seed))

    model_path = str(Path(tmp_dir) / "model.onnx")
    ir.save(model, model_path, external_data="model.onnx.data")
    return model_path, config


def empty_caches(
    config: ArchitectureConfig,
    *,
    batch: int = 1,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    dtype: type[np.floating] = FLASH_CACHE_DTYPE,
) -> dict[str, np.ndarray]:
    """Zeroed ``[batch, max_seq_len, kv_hidden]`` cache buffers for every layer."""
    kv_hidden = config.num_key_value_heads * config.head_dim
    feeds: dict[str, np.ndarray] = {}
    for layer in range(config.num_hidden_layers):
        zeros = np.zeros((batch, max_seq_len, kv_hidden), dtype=dtype)
        feeds[f"key_cache.{layer}"] = zeros.copy()
        feeds[f"value_cache.{layer}"] = zeros.copy()
    return feeds


def carry_caches(outputs: dict[str, np.ndarray], num_layers: int) -> dict[str, np.ndarray]:
    """Feed the previous step's ``updated_*`` caches back in as the next inputs."""
    feeds: dict[str, np.ndarray] = {}
    for layer in range(num_layers):
        feeds[f"key_cache.{layer}"] = outputs[f"updated_key_cache.{layer}"]
        feeds[f"value_cache.{layer}"] = outputs[f"updated_value_cache.{layer}"]
    return feeds


@contextlib.contextmanager
def captured_attention_dispatch() -> Iterator[list[str]]:
    """Capture the ONNX-domain Attention kernel's ``VERBOSE`` dispatch log.

    This ORT build exposes neither CUPTI ``"Kernel"``-category profiling events
    nor a per-op ``AttentionKernelDebugInfo`` print for the ONNX-domain
    ``Attention`` kernel.  The authoritative, build-portable signal is the
    kernel's own ``LOGS_DEFAULT(VERBOSE)`` line emitted from
    ``onnxruntime/core/providers/cuda/llm/attention.cc``::

        ONNX Attention: using Flash Attention (...)
        ONNX Attention: using Memory Efficient Attention (...)
        ONNX Attention: using Unfused Attention (...)

    Within the ``with`` block the ORT *default* logger is set to ``VERBOSE``
    (the kernel logs via ``LOGS_DEFAULT``, not the session logger) and the
    process' native ``stderr`` (file descriptor 2) is redirected to a temp file
    — the message originates in C++ and bypasses Python's ``sys.stderr``.  On
    exit the captured lines are appended to the yielded list and the previous
    logger severity / ``stderr`` are restored.

    Usage::

        with captured_attention_dispatch() as log_lines:
            session.run(output_names, feeds)
        kernels = attention_kernels_from_log(log_lines)
        assert kernels == {"flash"}
    """
    captured: list[str] = []
    # ORT's default logger severity defaults to WARNING (2); restore it after.
    ort.set_default_logger_severity(0)
    saved_stderr_fd = os.dup(2)
    capture_file = tempfile.TemporaryFile(mode="w+b")
    try:
        os.dup2(capture_file.fileno(), 2)
        yield captured
    finally:
        # Flush C++ stderr, restore the real fd, then read back what we caught.
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stderr_fd)
        ort.set_default_logger_severity(2)
        capture_file.seek(0)
        text = capture_file.read().decode("utf-8", errors="replace")
        capture_file.close()
        captured.extend(text.splitlines())


def attention_kernels_from_log(log_lines: list[str]) -> set[str]:
    """Return the set of ONNX-domain Attention kernels named in *log_lines*.

    Values are drawn from :data:`_KERNEL_LOG_MARKERS` keys: ``"flash"``,
    ``"memory_efficient"``, ``"unfused"``.  An empty set means no
    ``ONNX Attention: using ...`` line was captured (e.g. the kernel did not log,
    or the run never reached the attention op).
    """
    text = "\n".join(log_lines)
    return {kernel for kernel, marker in _KERNEL_LOG_MARKERS.items() if marker in text}


def _build_session(tmp_dir: str) -> tuple[ort.InferenceSession, ArchitectureConfig]:
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


def _prefill(session: ort.InferenceSession, config: ArchitectureConfig, prefill_len: int):
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


def _decode_feeds(config: ArchitectureConfig, prefill_out: dict[str, np.ndarray], step: int):
    """Single-token decode feed at slot ``_PREFILL_LEN + step`` carrying the cache."""
    write_index = _PREFILL_LEN + step
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
        prefill_len = _PREFILL_LEN
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
@pytest.mark.skipif(
    not _flash_capable_gpu(),
    reason=(
        "CUDA Flash attention requires compute capability >= 8.0 (Ampere+); "
        "pre-SM80 GPUs (e.g. T4/Volta) route to Memory-Efficient Attention, so "
        "the Flash-dispatch assertion is only meaningful on Ampere or newer"
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
    ``attn_mask == nullptr``, and no ``past_key`` (full cache supplied).  Flash
    additionally requires an Ampere+ (SM >= 8.0) device; on pre-SM80 hardware the
    kernel routes to Memory-Efficient Attention, so the ``_flash_capable_gpu``
    guard above skips this test there rather than letting the dispatch assertion
    hard-fail.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        session, config = _build_session(tmp_dir)
        prefill_len = _PREFILL_LEN

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
