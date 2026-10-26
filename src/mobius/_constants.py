# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Package-wide constants."""

from __future__ import annotations

# Default ONNX opset version used for all graph construction.
# Separated into its own module to avoid circular imports between
# tasks, components, and the top-level package.
OPSET_VERSION = 24
QKV_PROJECTION_SOURCES_METADATA_KEY = "pkg.mobius.qkv_projection_sources"
"""Versioned source names and widths retained on projection-packing Split nodes."""

LEGACY_QKV_PROJECTION_SOURCES_METADATA_KEY = "mobius.qkv_projection_sources"
"""Read-only key for restoring packed adapters from pre-namespace preview exports."""

LAZY_SOURCE_FILES_META_KEY = "mobius.lazy_source_files"
"""Backing-file paths retained by lazy tensors and their model in ``.meta``.

Folding can remove an ExternalTensor from the initializer registry while a lazy
callback still needs its file. ModelPackage checks these paths before saving so
it cannot overwrite that backing file before the callback reads it. This is
transient Python bookkeeping, not serialized ``metadata_props``.
"""

# ---------------------------------------------------------------------------
# Static KV cache ABI
# ---------------------------------------------------------------------------
# A static-cache export scatters each step's keys and values into pre-allocated,
# fixed-capacity buffers instead of concatenating a growing cache. The two
# control ports below are plain integer vectors, so they are *shape-indistin-
# guishable* from one another and from every other per-row integer input: no
# consumer can recover their roles from the graph. They are therefore a declared
# ABI, minted here once and read back by the metadata producers, rather than
# names any consumer is expected to guess.
STATIC_CACHE_WRITE_INDICES = "write_indices"
"""Per-row destination of this step's scatter, along the cache sequence axis."""

STATIC_CACHE_KV_SEQUENCE_LENGTH = "nonpad_kv_seqlen"
"""Per-row count of valid cache entries *after* this step's scatter."""

STATIC_CACHE_SEQUENCE_AXIS = 1
"""Cache axis the scatter addresses: buffers are ``[batch, capacity, kv_hidden]``."""

STATIC_CACHE_LAYOUT = "bsh"
"""Element layout of a static cache buffer: batch, sequence slot, packed KV hidden."""
