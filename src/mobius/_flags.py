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
         - ``False``
         - Lower the ONNX opset declaration to 23 for non-CPU EPs
           (ORT <=1.24.x workaround). Disabled by default.
       * - ``tencent_q1_0_use_native_2bit``
         - ``MOBIUS_TENCENT_Q1_0_USE_NATIVE_2BIT``
         - ``False``
         - Use native ``MatMulNBits bits=2`` for Tencent SEQ Q1_0
           (smaller, semantically faithful, but ~20x slower on CPU EP
           pending an MLAS fast path).
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
    ORT <=1.24.4 CUDA kernel bug that produces wrong results when scale is 2D.
    Set ``MOBIUS_ORT_CUDA_GROUPED_RMSNORM_WORKAROUND=1`` when targeting CUDA.
    """

    ort_lower_opset_for_ep: bool = dataclasses.field(
        default_factory=lambda: _env_bool("MOBIUS_ORT_LOWER_OPSET_FOR_EP", False)
    )
    """Lower the ONNX default-domain opset declaration to 23 when creating
    ORT sessions on non-CPU execution providers (CUDA, TRT, etc.).

    ORT <=1.24.x EPs didn't register kernels for opset 24 standard ops
    (Squeeze, Reshape, etc.) even though the semantics are unchanged.
    Lowering the import declaration lets the EP find its existing kernels.
    Set ``MOBIUS_ORT_LOWER_OPSET_FOR_EP=1`` to re-enable if running on
    an older ORT build without opset 24 kernel registration.
    """

    tencent_q1_0_use_native_2bit: bool = dataclasses.field(
        default_factory=lambda: _env_bool("MOBIUS_TENCENT_Q1_0_USE_NATIVE_2BIT", False)
    )
    """Emit Tencent custom Q1_0 (2-bit SEQ) tensors using native
    ``MatMulNBits bits=2`` + float ``zero_point = 1.5`` instead of the
    ``bits=4`` inflation that defaults today.

    Pros (when set to ``True``):
        Halves the on-disk weight bytes (2 bpw vs 4 bpw inflated).
        Semantically faithful to the source quantization layout.

    Cons (default ``False``):
        ORT's CPU ``bits=2`` + float-zp dequant path is currently a
        naive scalar fallback (~20x slower than the ``bits=4`` packed
        path on the same weights). See
        `microsoft/onnxruntime#28552
        <https://github.com/microsoft/onnxruntime/issues/28552>`_.
        Also requires ORT >=1.27 (the float-zp path was added in
        `microsoft/onnxruntime#28354
        <https://github.com/microsoft/onnxruntime/pull/28354>`_).

    The ``bits=4`` default inflates each 2-bit code ``c in {0..3}`` to
    a 4-bit slot ``2c in {0,2,4,6}`` paired with integer ``zero_point=3``;
    dequant gives the same SEQ codebook values, just at twice the
    weight storage. Set ``MOBIUS_TENCENT_Q1_0_USE_NATIVE_2BIT=1`` to
    opt in to the smaller native form once kernel performance lands.
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
