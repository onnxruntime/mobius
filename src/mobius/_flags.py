# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Runtime feature flags for mobius.

Flags control experimental or environment-specific behaviour. Each flag can be
set via an environment variable (``MOBIUS_<FLAG_NAME>``) or programmatically
by assigning to the :data:`flags` singleton.

Environment variable values are read once at import time. Valid truthy strings
are ``1``, ``true``, ``yes``; falsy are ``0``, ``false``, ``no``
(case-insensitive). Any other value falls back to the default.

**Adding new flags:** add a field to :class:`Flags` with a
``dataclasses.field(default_factory=...)`` that calls :func:`_env_bool`.

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
from contextlib import contextmanager
from typing import Iterator


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
class Flags:
    """Runtime feature flags singleton.

    Each flag maps to a ``MOBIUS_<FLAG_NAME>`` environment variable read at
    import time. Flags can be overridden programmatically at any point or
    scoped temporarily with :func:`override_flags`.
    """

    # Suppress spurious "has no constant value" warnings from the
    # initializer-deduplication pass.  These are expected noise when
    # optimisation passes run before weights are loaded.
    # Set MOBIUS_SUPPRESS_DEDUP_WARNING=0 to see all warnings.
    suppress_dedup_warning: bool = dataclasses.field(
        default_factory=lambda: _env_bool("MOBIUS_SUPPRESS_DEDUP_WARNING", True)
    )


# Global singleton — import and use this directly.
flags = Flags()


def list_flags() -> dict[str, bool]:
    """Return the current value of all flags as a plain dict snapshot."""
    return dataclasses.asdict(flags)


@contextmanager
def override_flags(**kwargs: bool) -> Iterator[None]:
    """Temporarily override one or more flags within a ``with`` block.

    Restores the original values on exit, even if an exception is raised.
    Intended for use in tests.

    Example::

        with override_flags(suppress_dedup_warning=False):
            build(model_id)
    """
    old = {k: getattr(flags, k) for k in kwargs}
    for k, v in kwargs.items():
        setattr(flags, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(flags, k, v)
