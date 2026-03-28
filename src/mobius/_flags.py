# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Runtime feature flags for mobius.

Flags control experimental or environment-specific behaviour. Each flag can be
set via an environment variable (``MOBIUS_<FLAG_NAME>``) or programmatically
by assigning to the :data:`flags` singleton.

Environment variable values are read once at import time. Valid truthy strings
are ``1``, ``true``, ``yes``; falsy are ``0``, ``false``, ``no`` (case-insensitive).

Example::

    from mobius import flags, override_flags

    # Check a flag
    if flags.mmap_loading:
        ...

    # Programmatic override (persists)
    flags.mmap_loading = True

    # Scoped override for tests
    with override_flags(mmap_loading=True):
        ...
"""

from __future__ import annotations

import dataclasses
import os
from contextlib import contextmanager
from typing import Generator


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean from an environment variable."""
    val = os.environ.get(name, "")
    if val.lower() in ("1", "true", "yes"):
        return True
    if val.lower() in ("0", "false", "no"):
        return False
    return default


@dataclasses.dataclass
class Flags:
    """Runtime feature flags singleton.

    All flags are read from environment variables at import time and can be
    overridden programmatically at any point. Use :func:`override_flags` for
    temporary scoped overrides (e.g., in tests).
    """

    # Weight loading: use memory-mapped I/O instead of eager safetensors loading.
    # Default OFF — mmap loading is experimental; set MOBIUS_MMAP_LOADING=1 to enable.
    mmap_loading: bool = dataclasses.field(
        default_factory=lambda: _env_bool("MOBIUS_MMAP_LOADING", False)
    )

    # When mmap_loading is enabled, return MmapTensorDescriptor (lazy) objects
    # that defer materialisation to serialisation time.
    # Default ON; set MOBIUS_LAZY_CAST=0 to force eager materialisation.
    lazy_cast: bool = dataclasses.field(
        default_factory=lambda: _env_bool("MOBIUS_LAZY_CAST", True)
    )

    # Suppress spurious "has no constant value" warnings emitted by the
    # initializer-deduplication pass (expected noise before weights are loaded).
    # Default ON; set MOBIUS_SUPPRESS_DEDUP_WARNING=0 to see all warnings.
    suppress_dedup_warning: bool = dataclasses.field(
        default_factory=lambda: _env_bool("MOBIUS_SUPPRESS_DEDUP_WARNING", True)
    )


# Global singleton — import and use this directly.
flags = Flags()


def list_flags() -> dict[str, bool]:
    """Return the current value of all flags as a plain dict."""
    return dataclasses.asdict(flags)


@contextmanager
def override_flags(**kwargs: bool) -> Generator[None, None, None]:
    """Temporarily override one or more flags within a ``with`` block.

    Restores the original values on exit, even if an exception is raised.
    Intended for use in tests.

    Example::

        with override_flags(mmap_loading=True, lazy_cast=False):
            result = build(model_id)
    """
    old = {k: getattr(flags, k) for k in kwargs}
    for k, v in kwargs.items():
        setattr(flags, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(flags, k, v)
