# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Single source of truth for GGUF stored-quantization support.

Every ``ggml_type`` slot at the pinned llama.cpp commit gets exactly one
:class:`~mobius.integrations.gguf._spec.GGUFQuantSpec`. Block geometry
(``block_elements`` / ``block_bytes`` / readability) is taken from the vendored
census rather than typed by hand, so it cannot drift from upstream. Only the
*mobius* capabilities below are declared here:

``_NATIVE_BLOCK_FORMATS``
    Types whose serialized blocks the runtime consumes byte-for-byte.
``_AFFINE_REPACK_TARGETS``
    Types repacked into a ``MatMulNBits`` affine layout.
``_LM_HEAD_PRESERVE``
    Types an untied output head may stay quantized in.

Before this module existed, those three facts were spread over five
hand-synchronized tables in ``_repacker.py`` and ``_builder.py``. A comment at
``_builder.py`` even documented the hazard: a type repackable in one table but
missing from another raised ``KeyError`` at build time, after the download.
Deriving both modules from here turns that into an import-time invariant.

This module is an import leaf: it depends only on ``_spec`` and ``_upstream``.
"""

from __future__ import annotations

__all__ = [
    "affine_repack_target",
    "explicit_zero_point_type_names",
    "float_storage_type_ids",
    "get_quant_spec",
    "iter_quant_specs",
    "lm_head_preserve_type_names",
    "native_block_format",
    "quant_spec_by_name",
]

import functools
from types import MappingProxyType

from mobius.integrations.gguf._spec import (
    AffineRepackSpec,
    GGUFQuantSpec,
    NativeBlockSpec,
    StorageRole,
    Support,
)
from mobius.integrations.gguf._upstream import upstream_quant_types

# ---------------------------------------------------------------------------
# mobius capabilities, keyed by upper-case ggml type name
# ---------------------------------------------------------------------------

#: Types the runtime reads in their serialized GGUF block layout. The block
#: geometry is *not* repeated here — it is asserted against the pinned census
#: when the spec is constructed, so a wrong byte count is a test failure.
_NATIVE_BLOCK_FORMATS: MappingProxyType[str, str] = MappingProxyType(
    {
        "MXFP4": "mxfp4",
        "IQ4_NL": "iq4_nl",
        "IQ4_XS": "iq4_xs",
        "IQ3_S": "iq3_s",
        "IQ3_XXS": "iq3_xxs",
        "IQ2_XXS": "iq2_xxs",
        "IQ2_XS": "iq2_xs",
        "IQ2_S": "iq2_s",
        "IQ1_S": "iq1_s",
        "IQ1_M": "iq1_m",
    }
)

#: ``MatMulNBits`` representation produced for each repackable type.
#:
#: ``omit_zero_points`` is a property of the *target*, never of the source.
#: Q4_0 and Q8_0 are symmetric on disk, but their dequantization formulas are
#: still ``(q - 8) * scale`` and ``(q - 128) * scale``, and
#: ``GatherBlockQuantized`` has diverging CPU/CUDA defaults when the input is
#: omitted — which corrupts embeddings on CUDA before the first decoder layer
#: runs. Q6_K is symmetric around 32 yet requantizes through the asymmetric
#: affine path. So every target here emits zero points explicitly.
_AFFINE_REPACK_TARGETS: MappingProxyType[str, AffineRepackSpec] = MappingProxyType(
    {
        "Q4_0": AffineRepackSpec(bits=4, block_size=32, omit_zero_points=False),
        "Q4_1": AffineRepackSpec(bits=4, block_size=32, omit_zero_points=False),
        "Q8_0": AffineRepackSpec(bits=8, block_size=32, omit_zero_points=False),
        "Q4_K": AffineRepackSpec(bits=4, block_size=32, omit_zero_points=False),
        "Q6_K": AffineRepackSpec(bits=4, block_size=32, omit_zero_points=False),
        # Mainline Q1_0 is 1-bit binary over 128-element blocks, repacked into
        # 2-bit MatMulNBits with zp=1. Tencent's custom Q1_0 reuses the same
        # type id with a different on-disk layout and is handled separately in
        # ``_tencent_q1_0``.
        "Q1_0": AffineRepackSpec(bits=2, block_size=128, omit_zero_points=False),
    }
)

#: Types whose presence forces the shared graph scaffolding to carry explicit
#: ``zero_points``, even when the file is otherwise made of runtime-native
#: blocks. These are the formats whose GGUF dequantization uses a non-zero
#: offset that the affine scaffolding has to represent.
#:
#: Q3_K, Q5_0, and Q6_K also dequantize with an offset but are deliberately
#: absent, preserving the behavior this set had before it moved here. That
#: asymmetry is a known rough edge, recorded rather than silently normalized.
_REQUIRES_EXPLICIT_ZERO_POINT: frozenset[str] = frozenset(
    {
        "Q1_0",
        "Q2_K",
        "Q4_0",
        "Q4_1",
        "Q4_K",
        "Q5_1",
        "Q5_K",
        "Q8_0",
    }
)

#: Types an untied ``lm_head`` may stay quantized in. Broader than the union of
#: the two tables above because the head is also allowed to ride the generic
#: requantization path.
_LM_HEAD_PRESERVE: frozenset[str] = frozenset(
    {
        "Q1_0",
        "Q2_K",
        "Q3_K",
        "Q4_0",
        "Q4_1",
        "Q4_K",
        "Q5_0",
        "Q5_1",
        "Q5_K",
        "Q6_K",
        "Q8_0",
        "MXFP4",
        "IQ4_NL",
        "IQ4_XS",
        "IQ3_S",
        "IQ3_XXS",
        "IQ2_XXS",
        "IQ2_XS",
        "IQ2_S",
        "IQ1_S",
        "IQ1_M",
    }
)

#: Upstream ``gguf_role`` strings → :class:`StorageRole`. The census stores the
#: role as free text, so this is the one place it is interpreted.
_ROLE_PREFIXES: tuple[tuple[str, StorageRole], ...] = (
    ("storage-float", StorageRole.FLOAT),
    ("storage-quantized", StorageRole.QUANTIZED),
    ("aux-nonweight", StorageRole.AUX),
    ("compute-only", StorageRole.COMPUTE_ONLY),
    ("removed-unreadable", StorageRole.REMOVED),
)


def _role_of(upstream_role: str) -> StorageRole:
    """Map an upstream ``gguf_role`` string onto a :class:`StorageRole`."""
    for prefix, role in _ROLE_PREFIXES:
        if upstream_role.startswith(prefix):
            return role
    raise ValueError(f"Unrecognized upstream ggml role {upstream_role!r}")


def _display_name(enum_name: str, name: str) -> str:
    """Return an upper-case type name, falling back to the retired enum name.

    Removed slots have an empty ``ggml_type_name`` upstream, but their identity
    still has to appear in rejection messages, so the ``GGML_TYPE_`` prefix is
    stripped from the enum instead.
    """
    return (name or enum_name.removeprefix("GGML_TYPE_")).upper()


def _dequantize_verdict(
    role: StorageRole, dequant_impl: bool, removed_note: str
) -> tuple[Support, str | None]:
    """Decide whether block data of this type can be expanded to float."""
    if role is StorageRole.REMOVED:
        note = removed_note or "retired upstream"
        return (
            Support.REJECTED,
            (
                f"Removed ggml type ({note}). Its block size is 0, so the GGUF parse "
                "layer rejects the file before any architecture logic runs. "
                "Re-quantize the model with a current llama.cpp."
            ),
        )
    if role is StorageRole.COMPUTE_ONLY:
        return (
            Support.REJECTED,
            (
                "Compute-only vec_dot intermediate with no to_float conversion. It "
                "is never valid weight storage, so a GGUF containing it is malformed."
            ),
        )
    if role is StorageRole.QUANTIZED and not dequant_impl:
        return (
            Support.DEFERRED,
            (
                "gguf-py ships no Python dequantizer for this type at the pinned "
                "llama.cpp commit, so the float import path cannot expand it."
            ),
        )
    return (Support.SUPPORTED, None)


@functools.lru_cache(maxsize=1)
def _specs() -> MappingProxyType[int, GGUFQuantSpec]:
    """Build the quantization registry from the pinned census."""
    specs: dict[int, GGUFQuantSpec] = {}
    for type_id, upstream in sorted(upstream_quant_types().items()):
        role = _role_of(upstream.role)
        name = _display_name(upstream.enum_name, upstream.name)
        verdict, reason = _dequantize_verdict(
            role, upstream.dequant_impl, upstream.removed_note
        )

        native_format = _NATIVE_BLOCK_FORMATS.get(name)
        native = (
            None
            if native_format is None
            else NativeBlockSpec(
                format=native_format,
                elements=upstream.block_elements,
                bytes=upstream.block_bytes,
            )
        )
        specs[type_id] = GGUFQuantSpec(
            ggml_type_id=type_id,
            name=name,
            role=role,
            block_elements=upstream.block_elements,
            block_bytes=upstream.block_bytes,
            dequantize=verdict,
            native_preserve=native,
            affine_repack=_AFFINE_REPACK_TARGETS.get(name),
            requires_explicit_zero_point=name in _REQUIRES_EXPLICIT_ZERO_POINT,
            lm_head_preserve=name in _LM_HEAD_PRESERVE,
            reason=reason,
        )
    return MappingProxyType(specs)


@functools.lru_cache(maxsize=1)
def _by_name() -> MappingProxyType[str, GGUFQuantSpec]:
    """Index the registry by upper-case type name."""
    return MappingProxyType({spec.name: spec for spec in _specs().values()})


def iter_quant_specs() -> tuple[GGUFQuantSpec, ...]:
    """Return every ggml type slot, ordered by numeric type id."""
    return tuple(_specs().values())


def get_quant_spec(ggml_type: object) -> GGUFQuantSpec | None:
    """Return the spec for a ggml type id or ``gguf.GGMLQuantizationType``.

    Args:
        ggml_type: A numeric type id, or any object with a ``value`` attribute
            holding one (which is how ``gguf.GGMLQuantizationType`` behaves).

    Returns:
        The matching spec, or ``None`` when the id is outside the pinned enum.
    """
    type_id = getattr(ggml_type, "value", ggml_type)
    if not isinstance(type_id, int):
        return None
    return _specs().get(type_id)


def quant_spec_by_name(name: str) -> GGUFQuantSpec | None:
    """Return the spec for an upper-case ggml type name such as ``"Q4_K"``."""
    return _by_name().get(name.upper())


def native_block_format(ggml_type: object) -> str | None:
    """Return the runtime-native block format for a type, if it has one."""
    spec = get_quant_spec(ggml_type)
    if spec is None or spec.native_preserve is None:
        return None
    return spec.native_preserve.format


def affine_repack_target(ggml_type: object) -> AffineRepackSpec | None:
    """Return the ``MatMulNBits`` repack target for a type, if it has one."""
    spec = get_quant_spec(ggml_type)
    return None if spec is None else spec.affine_repack


@functools.lru_cache(maxsize=1)
def float_storage_type_ids() -> frozenset[int]:
    """Return type ids that hold plain float data rather than quantized blocks.

    ``f64`` is classified ``aux-nonweight`` upstream, but a tensor stored in it
    is still unquantized float data, so the builder's "is anything quantized?"
    check has always counted it alongside ``f32``/``f16``/``bf16``.
    """
    return frozenset(
        spec.ggml_type_id
        for spec in iter_quant_specs()
        if spec.role is StorageRole.FLOAT or spec.name == "F64"
    )


@functools.lru_cache(maxsize=1)
def lm_head_preserve_type_names() -> frozenset[str]:
    """Return the type names an untied output head may stay quantized in."""
    return frozenset(spec.name for spec in iter_quant_specs() if spec.lm_head_preserve)


@functools.lru_cache(maxsize=1)
def explicit_zero_point_type_names() -> frozenset[str]:
    """Return type names that force explicit ``zero_points`` on the shared graph."""
    return frozenset(
        spec.name for spec in iter_quant_specs() if spec.requires_explicit_zero_point
    )
