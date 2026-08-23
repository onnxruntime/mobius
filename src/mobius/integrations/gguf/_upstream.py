# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Pinned llama.cpp support census.

The JSON payload in ``_upstream_data/llamacpp_pin.json`` was extracted
mechanically from llama.cpp at commit
``8d9af256337d1a501250f9bbf4c0859a654bddd6``: 147 real ``llm_arch`` entries and
all 43 ``ggml_type`` slots.

This data exists to make coverage *measurable*, not to make claims. Nothing
here implies mobius supports an architecture — support lives in
:mod:`mobius.integrations.gguf._arch_registry`, and the census is used only to

* validate that every architecture mobius registers is a real upstream
  architecture string rather than a mobius ``model_type`` that leaked into the
  architecture namespace,
* validate that the quantization registry covers all 43 slots with the pinned
  block geometry, and
* turn an unrecognized architecture into an actionable message that names the
  upstream cohort it belongs to.

This module is an import leaf: it depends on nothing else in the package.
"""

from __future__ import annotations

__all__ = [
    "UPSTREAM_COMMIT",
    "UpstreamArchitecture",
    "UpstreamQuantType",
    "upstream_architecture",
    "upstream_architectures",
    "upstream_quant_types",
]

import dataclasses
import functools
import json

# Importing from ``importlib.resources`` keeps the payload readable from a
# zipped wheel as well as a source checkout.
from importlib import resources
from typing import Any

UPSTREAM_COMMIT = "8d9af256337d1a501250f9bbf4c0859a654bddd6"

_DATA_PACKAGE = "mobius.integrations.gguf._upstream_data"
_DATA_FILE = "llamacpp_pin.json"


@dataclasses.dataclass(frozen=True, slots=True)
class UpstreamArchitecture:
    """One ``LLM_ARCH_NAMES`` entry as it exists at the pinned commit.

    Attributes:
        gguf_arch: The ``general.architecture`` string.
        cohort: Survey cohort, e.g. ``"C01-dense-transformer"``.
        rope_type: ``llama_model_rope_type`` for the architecture.
        topology: ``"dense"``, ``"moe"``, ``"recurrent"``, and similar.
        moe_mode: Whether tensor shapes switch on ``expert_count``.
        recurrent: Whether the architecture uses a recurrent state cache.
        hybrid: Whether attention and SSM blocks interleave per layer.
        cpp_loader: Whether a ``llama_model_*`` class exists. ``gptj`` is the
            one architecture where this is ``False``.
        converter: Whether a ``convert_hf_to_gguf.py`` registration exists.
    """

    gguf_arch: str
    cohort: str
    rope_type: str
    topology: str
    moe_mode: str
    recurrent: bool
    hybrid: bool
    cpp_loader: bool
    converter: bool


@dataclasses.dataclass(frozen=True, slots=True)
class UpstreamQuantType:
    """One ``enum ggml_type`` slot as it exists at the pinned commit.

    Block geometry was read from a compiled ``libggml-base`` via
    ``ggml_get_type_traits()``, so ``block_elements``/``block_bytes`` are exact.

    Attributes:
        ggml_type_id: Numeric enum value.
        enum_name: C enum name, e.g. ``"GGML_TYPE_Q4_K"``.
        name: Lower-case type name, empty for removed slots.
        status: ``"active"`` or ``"removed"``.
        role: Upstream storage role string.
        block_elements: ``blck_size``. ``0`` for removed slots.
        block_bytes: ``type_size``. ``0`` for removed slots.
        readable: Whether the GGUF parse layer accepts the type.
        has_to_float: Whether ggml can expand the type to float.
        dequant_impl: Whether ``gguf-py`` ships a Python dequantizer, which is
            what the mobius float import path actually calls.
        removed_note: Upstream deprecation note, when present.
    """

    ggml_type_id: int
    enum_name: str
    name: str
    status: str
    role: str
    block_elements: int
    block_bytes: int
    readable: bool
    has_to_float: bool
    dequant_impl: bool
    removed_note: str


@functools.lru_cache(maxsize=1)
def _payload() -> dict[str, Any]:
    """Load and cache the pinned census payload."""
    text = resources.files(_DATA_PACKAGE).joinpath(_DATA_FILE).read_text(encoding="utf-8")
    data: dict[str, Any] = json.loads(text)
    if data.get("commit") != UPSTREAM_COMMIT:
        raise ValueError(
            f"{_DATA_FILE} records commit {data.get('commit')!r} but this module pins "
            f"{UPSTREAM_COMMIT!r}. Re-extract the census or update the pin."
        )
    return data


@functools.lru_cache(maxsize=1)
def upstream_architectures() -> dict[str, UpstreamArchitecture]:
    """Return every pinned upstream architecture keyed by GGUF architecture string."""
    return {
        name: UpstreamArchitecture(gguf_arch=name, **fields)
        for name, fields in _payload()["architectures"].items()
    }


@functools.lru_cache(maxsize=1)
def upstream_quant_types() -> dict[int, UpstreamQuantType]:
    """Return every pinned ``ggml_type`` slot keyed by numeric type id."""
    return {
        int(type_id): UpstreamQuantType(
            ggml_type_id=int(type_id),
            enum_name=fields["enum"],
            name=fields["name"],
            status=fields["status"],
            role=fields["role"],
            block_elements=fields["block_elements"],
            block_bytes=fields["block_bytes"],
            readable=fields["readable"],
            has_to_float=fields["has_to_float"],
            dequant_impl=fields["dequant_impl"],
            removed_note=fields["removed_note"],
        )
        for type_id, fields in _payload()["ggml_types"].items()
    }


def upstream_architecture(gguf_arch: str) -> UpstreamArchitecture | None:
    """Return the pinned entry for *gguf_arch*, or ``None`` if it is not upstream."""
    return upstream_architectures().get(gguf_arch)
