# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Temporary back-compat bridge for the static-cache capability probe.

The functional capability probe that asks "can the installed ORT run the
maskless ``is_causal=1`` + ``nonpad_kv_seqlen`` + ``TensorScatter`` static-cache
graph on the CUDA EP (needs microsoft/onnxruntime#28958)?" now lives — as the
single source of truth — in
:func:`mobius._testing.ort_capabilities.supports_static_cache_flash`.

This module only re-exports it under the historical name so any test that has
not yet migrated its import keeps collecting.  It is scheduled for deletion the
moment every test imports the canonical module directly; do not add new code
here.
"""

from __future__ import annotations

from mobius._testing.ort_capabilities import CUDA_AVAILABLE as CUDA_AVAILABLE
from mobius._testing.ort_capabilities import supports_static_cache_flash

static_cache_cuda_supported = supports_static_cache_flash
