# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Exception types for the GGUF importer.

Every exception here subclasses the type that the corresponding code path
raised before the support registry existed, so existing callers and tests that
catch :class:`ValueError` or :class:`NotImplementedError` keep working:

=============================== =========================
Exception                       Base
=============================== =========================
``UnsupportedGGUFArchitectureError``  ``ValueError``
``UnsupportedGGUFQuantizationError``  ``ValueError``
``DisabledGGUFArchitectureError``     ``NotImplementedError``
``ShardedGGUFNotSupportedError``      ``NotImplementedError``
=============================== =========================
"""

from __future__ import annotations

__all__ = [
    "DisabledGGUFArchitectureError",
    "ShardedGGUFNotSupportedError",
    "UnsupportedGGUFArchitectureError",
    "UnsupportedGGUFQuantizationError",
    "VibeASRBitNetGGUFImportError",
]


class UnsupportedGGUFArchitectureError(ValueError):
    """A GGUF ``general.architecture`` that mobius cannot import.

    Raised when the architecture is unknown to the registry, or when it is
    known but one of its capabilities is not ``SUPPORTED``.
    """


class UnsupportedGGUFQuantizationError(ValueError):
    """A stored GGML tensor type that mobius cannot read or preserve."""


class VibeASRBitNetGGUFImportError(UnsupportedGGUFQuantizationError):
    """A VibeASR.cpp-native GGUF whose execution contract has no ORT equivalent."""


class DisabledGGUFArchitectureError(NotImplementedError):
    """A GGUF architecture whose conversion is deliberately turned off.

    Distinct from :class:`UnsupportedGGUFArchitectureError`: the pieces exist but
    conversion is known to build a wrong graph, so it is blocked on purpose.
    """


class ShardedGGUFNotSupportedError(NotImplementedError):
    """A split/sharded GGUF input, which the single-file reader cannot assemble."""
