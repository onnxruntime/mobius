# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Shared helpers for static-cache CUDA tests (capability probe + Flash dispatch).

mobius ``--static-cache`` decoders emit the opset-24 ONNX-domain ``Attention``
op with ``is_causal=1`` + ``nonpad_kv_seqlen`` (no ``attn_mask``, no
``past_key``) directly after two ``TensorScatter`` writes that update a
pre-allocated KV cache in place.  Running that graph on the CUDA Execution
Provider requires an ORT build that contains microsoft/onnxruntime#28958, which
removes the historical ``NOT_IMPLEMENTED`` reject for
``is_causal=1 + nonpad_kv_seqlen`` (when ``S_q != total_kv`` with no
``past_key``) and makes the combination Flash-eligible.

The *functional capability probe* (does the installed ORT run the static-cache
graph on CUDA at all?) lives in the shared, importable
:func:`mobius._testing.ort_capabilities.supports_static_cache_flash`; both the
e2e Flash-dispatch test and the numerical-parity test import it from there.  This
module adds the e2e-specific machinery on top of that gate:

* :func:`build_static_cache_model` — builds the tiny fp16 static-cache model the
  e2e test actually drives through prefill + decode.

* :func:`quick_build_enabled` — guards the Flash-*dispatch* assertion.  Under
  ``onnxruntime_QUICK_BUILD=ON`` the CUDA Flash kernel is compiled for
  ``head_dim == 128`` only, so the tiny ``head_dim == 64`` geometry below
  silently routes to Memory-Efficient Attention and a Flash assertion would be
  a false negative.

**Flash-dispatch assertion mechanism.**  This ORT build exposes neither
CUPTI ``"Kernel"``-category profiling events nor the per-op
``AttentionKernelDebugInfo`` print for the ONNX-domain ``Attention`` kernel.
The authoritative, build-portable signal is the kernel's own ``VERBOSE`` log
line emitted from ``onnxruntime/core/providers/cuda/llm/attention.cc``::

    ONNX Attention: using Flash Attention (batch=..., q_seq=..., ...)
    ONNX Attention: using Memory Efficient Attention (...)
    ONNX Attention: using Unfused Attention (...)

:func:`captured_attention_dispatch` enables that log (``set_default_logger_severity(0)``),
captures the kernel's native ``stderr`` (file descriptor 2 — the message comes
from C++, not Python's ``sys.stderr``), and :func:`attention_kernels_from_log`
parses out which kernel(s) ran.
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

from mobius import ArchitectureConfig, build_from_module
from mobius._registry import registry
from mobius._testing.ort_capabilities import (
    CUDA_AVAILABLE as CUDA_AVAILABLE,
)
from mobius._testing.ort_capabilities import (
    supports_static_cache_flash,
)
from mobius.tasks import CausalLMTask

# Migration bridge: the functional capability probe was renamed and moved to
# ``mobius._testing.ort_capabilities.supports_static_cache_flash``.  This alias
# keeps tests that still import the old name collectable; remove it once every
# test imports the canonical name directly.
static_cache_cuda_supported = supports_static_cache_flash

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
_FLASH_HEAD_DIM = 64
_FLASH_NUM_HEADS = 2
_FLASH_NUM_KV_HEADS = 1
_FLASH_HIDDEN = _FLASH_HEAD_DIM * _FLASH_NUM_HEADS

# Flash is fp16/bf16 only on every CUDA path; fp32 routes to MEA/unfused.
FLASH_CACHE_DTYPE = np.float16

# Substrings of the ``ONNX Attention: using ...`` VERBOSE log, keyed by the
# canonical kernel name this module reports.
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
    module asserts.  The model is loaded later on the CUDA EP at session time.

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

    Within the ``with`` block the ORT *default* logger is set to ``VERBOSE``
    (the kernel logs via ``LOGS_DEFAULT(VERBOSE)``) and the process' native
    ``stderr`` (file descriptor 2) is redirected to a temp file — the message
    originates in C++ and bypasses Python's ``sys.stderr``.  On exit the
    captured lines are appended to the yielded list and the previous logger
    severity / ``stderr`` are restored.

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
