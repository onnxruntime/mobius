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

    Only the facts the registry reasons about are vendored; the full 24-column
    survey stays out of the repository.

    Attributes:
        gguf_arch: The ``general.architecture`` string.
        cohort: Survey cohort, e.g. ``"C01-dense-transformer"``. Used to tell a
            caller which family an unimported architecture belongs to.
        cpp_loader: Whether a ``llama_model_*`` class exists. ``gptj`` is the
            one architecture where this is ``False``, so no tool can load it.
        dual_moe: Whether tensor shapes switch on ``expert_count`` rather than
            on the architecture name. True for 47 architectures.
        tensor_families: Exact entries from the pinned
            ``gguf-py/gguf/constants.py::MODEL_TENSORS`` table. Vendored for
            architectures whose tensor-map support mobius claims and for audited
            cohorts whose explicit deferral depends on proving the graph mismatch.
        tensor_names: Exact full tensor names, including suffixes, created by
            architecture-specific pinned C++ tensor creation sites.
        converter_extra_tensor_names: Exact full tensor names emitted by a pinned
            converter but not created by the architecture-specific C++ loader.
            These are semantic sidecars, not part of the loader-required closure.
        expert_tensor_suffixes: Exact suffixes created for routed expert tensors
            by the pinned C++ generic loader pass.
    """

    gguf_arch: str
    cohort: str
    cpp_loader: bool
    dual_moe: bool
    tensor_families: tuple[str, ...] = ()
    tensor_names: tuple[str, ...] = ()
    converter_extra_tensor_names: tuple[str, ...] = ()
    expert_tensor_suffixes: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class UpstreamQuantType:
    """One ``enum ggml_type`` slot as it exists at the pinned commit.

    Block geometry was read from a compiled ``libggml-base`` via
    ``ggml_get_type_traits()``, so ``block_elements``/``block_bytes`` are exact.

    Attributes:
        ggml_type_id: Numeric enum value.
        enum_name: C enum name, e.g. ``"GGML_TYPE_Q4_K"``. Removed slots have no
            type name, so this is what identifies them in a message.
        name: Lower-case type name, empty for removed slots.
        role: Upstream storage role string.
        block_elements: ``blck_size``. ``0`` for removed slots.
        block_bytes: ``type_size``. ``0`` for removed slots.
        readable: Whether the GGUF parse layer accepts the type.
        dequant_impl: Whether ``gguf-py`` ships a Python dequantizer, which is
            what the mobius float import path actually calls.
        removed_note: Upstream deprecation note, when present.
    """

    ggml_type_id: int
    enum_name: str
    name: str
    role: str
    block_elements: int
    block_bytes: int
    readable: bool
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
        name: UpstreamArchitecture(
            gguf_arch=name,
            cohort=fields["cohort"],
            cpp_loader=fields["cpp_loader"],
            dual_moe=fields["dual_moe"],
            tensor_families=tuple(fields.get("tensor_families", ())),
            tensor_names=tuple(fields.get("tensor_names", ())),
            converter_extra_tensor_names=tuple(fields.get("converter_extra_tensor_names", ())),
            expert_tensor_suffixes=tuple(fields.get("expert_tensor_suffixes", ())),
        )
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
            role=fields["role"],
            block_elements=fields["block_elements"],
            block_bytes=fields["block_bytes"],
            readable=fields["readable"],
            dequant_impl=fields["dequant_impl"],
            removed_note=fields["removed_note"],
        )
        for type_id, fields in _payload()["ggml_types"].items()
    }


def upstream_architecture(gguf_arch: str) -> UpstreamArchitecture | None:
    """Return the pinned entry for *gguf_arch*, or ``None`` if it is not upstream."""
    return upstream_architectures().get(gguf_arch)
