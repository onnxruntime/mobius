# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Runtime feature flags for mobius.

Flags control experimental or environment-specific behaviour. Each flag can be
set via an environment variable (``MOBIUS_<FLAG_NAME>``) or programmatically
by assigning to the :data:`flags` singleton.

Environment variable values are read each time a :class:`_Flags` instance is
constructed. The global :data:`flags` singleton is constructed at import time,
so env vars should be set before importing mobius. Valid truthy strings are
``1``, ``true``, ``yes``; falsy are ``0``, ``false``, ``no``
(case-insensitive). Any other value falls back to the field default.

**Adding new flags:** add a field to :class:`_Flags` with a
``dataclasses.field(default_factory=...)`` that calls :func:`_env_bool`,
plus a docstring string literal immediately after the field for documentation
generation.

Example::

    from mobius import flags, override_flags

    # Check a flag
    if flags.suppress_dedup_warning:
        ...

    # Programmatic override (persists until changed)
    flags.suppress_dedup_warning = False

    # Scoped override for tests
    with override_flags(suppress_dedup_warning=False):
        ...
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Iterator
from contextlib import contextmanager


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean from an environment variable.

    Returns *default* if the variable is unset or has an unrecognised value.
    """
    val = os.environ.get(name, "")
    if val.lower() in ("1", "true", "yes"):
        return True
    if val.lower() in ("0", "false", "no"):
        return False
    return default


@dataclasses.dataclass
class _Flags:
    """Runtime feature flags singleton.

    Each flag maps to a ``MOBIUS_<FLAG_NAME>`` environment variable read when
    a :class:`_Flags` instance is constructed. The global :data:`flags`
    singleton is constructed at import time. Flags can be overridden
    programmatically at any point or scoped temporarily with
    :func:`override_flags`.

    **Available flags**

    .. list-table::
       :header-rows: 1

       * - Flag
         - Env var
         - Default
         - Description
       * - ``suppress_dedup_warning``
         - ``MOBIUS_SUPPRESS_DEDUP_WARNING``
         - ``True``
         - Suppress "has no constant value" warnings from the initializer
           deduplication pass.
       * - ``ort_lower_opset_for_ep``
         - ``MOBIUS_ORT_LOWER_OPSET_FOR_EP``
         - ``True``
         - Lower the ONNX opset declaration to 23 for non-CPU EPs
           (ORT ≤1.24.x workaround).
       * - ``use_gqa_for_kv_shared``
         - ``MOBIUS_USE_GQA_FOR_KV_SHARED``
         - ``False``
         - Use GQA for KV-shared layers on CUDA EP. Requires ORT GQA
           new_kv_length=0 support.
       * - ``prune_lm_head``
         - ``MOBIUS_PRUNE_LM_HEAD``
         - ``False``
         - Insert ``Gather(axis=1, indices=[-1])`` before the LM head MatMul
           so logits are only computed for the last token. Speeds up prefill
           for chat-only workloads but breaks logprob scoring and
           speculative decoding.
    """

    suppress_dedup_warning: bool = dataclasses.field(
        default_factory=lambda: _env_bool("MOBIUS_SUPPRESS_DEDUP_WARNING", True)
    )
    """Suppress "has no constant value" warnings from the initializer-deduplication pass.

    These warnings are expected noise when optimisation passes run before weights
    are loaded. Set ``MOBIUS_SUPPRESS_DEDUP_WARNING=0`` to see all warnings.
    """

    ort_cuda_grouped_rmsnorm_workaround: bool = dataclasses.field(
        default_factory=lambda: _env_bool("MOBIUS_ORT_CUDA_GROUPED_RMSNORM_WORKAROUND", False)
    )
    """Decompose grouped RMSNormalization into basic ops to work around an
    ORT ≤1.24.4 CUDA kernel bug that produces wrong results when scale is 2D.
    Set ``MOBIUS_ORT_CUDA_GROUPED_RMSNORM_WORKAROUND=1`` when targeting CUDA.
    """

    use_gqa_for_kv_shared: bool = dataclasses.field(
        default_factory=lambda: _env_bool("MOBIUS_USE_GQA_FOR_KV_SHARED", False)
    )
    """Use GroupQueryAttention for KV-shared layers on CUDA EP.

    When ``False`` (default), KV-shared layers that borrow K/V from a
    source layer use standard ONNX Attention (which falls back to unfused
    attention on CUDA EP).  This avoids a CUTLASS MEA crash with the
    aligned kernel for certain sequence lengths when ``past_key`` is
    ``nullptr``.

    Set ``MOBIUS_USE_GQA_FOR_KV_SHARED=1`` when the ORT GQA kernel
    supports ``new_kv_length=0`` (KV-shared pattern), which will enable
    fused attention for these layers.
    """

    ort_lower_opset_for_ep: bool = dataclasses.field(
        default_factory=lambda: _env_bool("MOBIUS_ORT_LOWER_OPSET_FOR_EP", True)
    )
    """Lower the ONNX default-domain opset declaration to 23 when creating
    ORT sessions on non-CPU execution providers (CUDA, TRT, etc.).

    ORT ≤1.24.x EPs don't register kernels for opset 24 standard ops
    (Squeeze, Reshape, etc.) even though the semantics are unchanged.
    Lowering the import declaration lets the EP find its existing kernels.
    Set ``MOBIUS_ORT_LOWER_OPSET_FOR_EP=0`` to disable once ORT adds
    opset 24 kernel support.
    """

    prune_lm_head: bool = dataclasses.field(
        default_factory=lambda: _env_bool("MOBIUS_PRUNE_LM_HEAD", False)
    )
    """Insert ``Gather(axis=1, indices=[-1])`` before the LM head MatMul to
    select only the last token's hidden state.

    When ``False`` (default), the LM head computes logits for every token
    in the sequence (output shape ``[B, S, vocab]``). This preserves all
    use cases including logprob scoring, perplexity evaluation,
    speculative-decoding verification, and multi-token-at-a-time generation.

    When ``True``, only the last token's logits are computed (output shape
    ``[B, 1, vocab]``), avoiding the full ``[B, S, vocab]`` MatMul during
    prefill.  This is a meaningful prefill speedup for chat / single-token
    generation workloads on models with large vocabularies (e.g. Qwen at
    151936 tokens), but **breaks** any workflow that needs per-token logits.

    Mirrors the ``prune_lm_head`` extra option in onnxruntime-genai's
    Model Builder.  Set ``MOBIUS_PRUNE_LM_HEAD=1`` to enable for chat-only
    deployments.

    .. note::
        **Compatibility:** This flag only takes effect for models that use
        the base :class:`~mobius.models.base.CausalLMModel.forward`
        implementation — Llama, Mistral, Qwen2 / 2.5 / 3, and other
        standard decoder-only architectures.  Models that override
        ``forward()`` (GPT-J, CodeGen, Phi, NanoChat, DOGE, Gemma3n,
        Apertus, GPT-2 family, Mamba, InternLM2, Qwen3.5, etc.) **silently
        ignore this flag** because they assemble logits via their own code
        paths.  Coverage will be extended in follow-up work as needed.
    """


# Global singleton — import and use this directly.
flags = _Flags()


def list_flags() -> dict[str, object]:
    """Return the current value of all flags as a plain dict snapshot."""
    return dataclasses.asdict(flags)


@contextmanager
def override_flags(**kwargs: bool) -> Iterator[None]:
    """Temporarily override one or more flags within a ``with`` block.

    Restores the original values on exit, even if an exception is raised.
    Intended for use in tests.

    .. note::
        **Thread safety:** ``override_flags`` is not thread-safe — concurrent
        calls in different threads may interleave the save/restore cycle
        (TOCTOU). For pytest, this is safe when running with ``-n auto``
        because xdist spawns separate worker *processes* (not threads), so
        each worker has its own copy of the flag singleton.

    Raises:
        ValueError: If any key in *kwargs* is not a known flag name.

    Example::

        with override_flags(suppress_dedup_warning=False):
            build(model_id)
    """
    valid = {f.name for f in dataclasses.fields(_Flags)}
    unknown = sorted(set(kwargs) - valid)
    if unknown:
        available = ", ".join(sorted(valid))
        raise ValueError(
            f"Unknown flag name(s): {', '.join(unknown)}. Available flags: {available}"
        )
    old = {k: getattr(flags, k) for k in kwargs}
    for k, v in kwargs.items():
        setattr(flags, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(flags, k, v)
