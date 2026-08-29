# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Block-scaled FP8 / packed-FP4 quantized weight loading contract.

Some recent checkpoints (e.g. DeepSeek-V4) publish *mixed-precision* weights
whose real numeric layout is not captured by the coarse
:class:`~mobius._configs.QuantizationConfig` (which only distinguishes int4
GPTQ/AWQ-style block quant, GGUF, and per-tensor fp8). Three distinct weight
families coexist in one checkpoint and must be told apart **by tensor
properties, never by model name**:

1. **Ordinary** ``bf16`` / ``f16`` / ``f32`` tensors (router gate, norms,
   biases, sinks). No paired scale.
2. **Block-FP8 projections** — an ``F8_E4M3`` weight of *logical* shape paired
   with an ``F8_E8M0`` (UE8M0, exponent-only) 2D block-scale of shape
   ``[ceil(out / bs0), ceil(in / bs1)]`` (``weight_block_size`` from the HF
   ``quantization_config``, e.g. ``[128, 128]``).
3. **FP4-packed routed experts** — an ``I8`` weight that stores **two E2M1
   (fp4) nibbles per byte**, so its *packed* shape is the logical shape with the
   last dim halved, paired with an ``F8_E8M0`` micro-scale of shape
   ``[out, logical_in / 32]`` (one UE8M0 exponent per output row per 32 logical
   input elements). Numerically this is **MXFP4** (E2M1 + block-32 + E8M0), not
   NVFP4 (block-16 + E4M3 block-scale + FP32 global scale).

This module is the *clean, breaking* descriptor + load contract for those
families. It intentionally does **not** dequantize to float, does not copy
weights, and preserves raw bytes. It classifies and validates tensors by
property, loads their raw bytes lazily (bounded to one tensor at a time), and
exposes a byte-exact expert-major bank-stacking primitive.

Crucially, the routed-expert *emission gate*
(:func:`plan_routed_expert_bank`) proves whether the onnx-genai ``nxrt``
runtime can represent these banks and, when it cannot, **fails closed with a
typed :class:`BlockQuantExportError` naming the exact ABI gap** rather than
emitting an unrunnable node. As of this writing the ``nxrt`` block-quant ABI
(``crates/onnx-runtime-ep-cpu/src/kernels/block_quantized_{matmul,moe}.rs``)
accepts only the interleaved llama.cpp ``block_mxfp4`` layout and the ``iq*``
GGUF formats — it has **no block-FP8 format** and **no planar-FP4 bank
layout** — so both quantized families here are typed-rejected.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import pathlib
import struct

__all__ = [
    "QuantKind",
    "BlockQuantScheme",
    "QuantizedTensorDescriptor",
    "BlockQuantError",
    "BlockQuantValidationError",
    "BlockQuantExportError",
    "PackedExpertBank",
    "SAFETENSORS_DTYPE_BYTES",
    "classify_tensor",
    "validate_descriptor",
    "pair_weight_scales",
    "build_descriptors",
    "read_safetensors_header",
    "raw_tensor_span",
    "read_raw_tensor_bytes",
    "LazyRawTensor",
    "stack_expert_bank",
    "runtime_representation_gap",
    "plan_routed_expert_bank",
    "NXRT_BLOCK_FORMATS",
]


# ---------------------------------------------------------------------------
# Typed errors
# ---------------------------------------------------------------------------


class BlockQuantError(Exception):
    """Base class for block-quant load/emit failures."""


class BlockQuantValidationError(BlockQuantError, ValueError):
    """A tensor's declared metadata is internally inconsistent.

    Raised on a logical-vs-packed shape contradiction, a missing/duplicate or
    mismatched scale, or a scale grid that does not match the weight's block
    geometry. This is a hard reject — never a silently-reinterpreted tensor.
    """


class BlockQuantExportError(BlockQuantError, NotImplementedError):
    """A validated bank cannot be represented by the target runtime ABI.

    Raised by the emission gate instead of emitting an unrunnable node. The
    message names the exact ABI gap (which format/layout the runtime lacks) so
    the blocker is actionable rather than a confusing downstream shape error.
    """


# ---------------------------------------------------------------------------
# safetensors dtype table + tiny header reader (byte-exact, no data read)
# ---------------------------------------------------------------------------

#: Bytes per stored element for each safetensors dtype string. ``I8`` counts a
#: single byte even though an FP4-packed tensor stores *two* logical E2M1 codes
#: per byte — that 2x is captured by the packed-vs-logical shape, not here.
SAFETENSORS_DTYPE_BYTES: dict[str, int] = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
}

_FLOAT_DTYPES = frozenset({"F64", "F32", "F16", "BF16"})


def _num_elements(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def read_safetensors_header(path: str | pathlib.Path) -> dict:
    """Read a safetensors file's JSON header only (no tensor data).

    Returns the raw header dict mapping each tensor name to
    ``{"dtype", "shape", "data_offsets"}`` plus the ``__metadata__`` block.
    Only the 8-byte length prefix and the header JSON are read from disk.
    """
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(header_len))


def raw_tensor_span(
    path: str | pathlib.Path, key: str
) -> tuple[str, tuple[int, ...], int, int]:
    """Return ``(dtype, shape, abs_start, abs_end)`` for one tensor's raw bytes.

    ``abs_start``/``abs_end`` are absolute byte offsets into *path* (the
    safetensors ``data_offsets`` are relative to the end of the header, so the
    ``8 + header_len`` base is added). Only the header is read.
    """
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    if key not in header:
        raise BlockQuantValidationError(f"Tensor {key!r} not found in {path!s}")
    entry = header[key]
    base = 8 + header_len
    start, end = entry["data_offsets"]
    return entry["dtype"], tuple(int(d) for d in entry["shape"]), base + start, base + end


def read_raw_tensor_bytes(path: str | pathlib.Path, key: str) -> bytes:
    """Read exactly one tensor's raw on-disk bytes, byte-for-byte.

    No dtype interpretation, no cast, no copy-to-float. The returned ``bytes``
    are identical to the tensor's storage in the shard, so a subsequent write
    preserves the quantized payload exactly.
    """
    _dtype, _shape, start, end = raw_tensor_span(path, key)
    with open(path, "rb") as f:
        f.seek(start)
        return f.read(end - start)


@dataclasses.dataclass(frozen=True)
class LazyRawTensor:
    """A bounded, byte-preserving handle to one tensor in one shard.

    Holds only the shard path + key + header metadata; the payload is read on
    demand via :meth:`read`, so peak resident memory is one tensor, not the
    whole checkpoint. ``num_bytes`` is known from the header without reading
    data.
    """

    path: str
    key: str
    dtype: str
    shape: tuple[int, ...]
    num_bytes: int

    @classmethod
    def open(cls, path: str | pathlib.Path, key: str) -> LazyRawTensor:
        dtype, shape, start, end = raw_tensor_span(path, key)
        return cls(path=str(path), key=key, dtype=dtype, shape=shape, num_bytes=end - start)

    def read(self) -> bytes:
        """Read and return this tensor's raw bytes (byte-exact)."""
        data = read_raw_tensor_bytes(self.path, self.key)
        if len(data) != self.num_bytes:
            raise BlockQuantValidationError(
                f"Tensor {self.key!r} in {self.path}: header declared "
                f"{self.num_bytes} bytes but read {len(data)}"
            )
        return data


# ---------------------------------------------------------------------------
# Quantization scheme parsed from the HF quantization_config (by properties)
# ---------------------------------------------------------------------------


class QuantKind(enum.Enum):
    """Property-classified quantization family of a single tensor."""

    ORDINARY = "ordinary"  # bf16/f16/f32, no scale
    BLOCK_FP8 = "block_fp8"  # F8_E4M3 weight + 2D UE8M0 block scale
    FP4_PACKED = "fp4_packed"  # I8-packed E2M1 nibbles + 1D UE8M0 micro-scale
    UNSUPPORTED = "unsupported"  # recognized-but-unhandled / malformed


#: Micro-scale block length (logical input elements per UE8M0 exponent) that
#: marks an MXFP4-style fp4 tensor. NVFP4 would instead use 16 + an E4M3 scale.
MXFP4_MICROSCALE_BLOCK = 32


@dataclasses.dataclass(frozen=True)
class BlockQuantScheme:
    """The checkpoint-wide quantization scheme, parsed by properties.

    Derived from the HF ``quantization_config`` plus the top-level
    ``expert_dtype`` — never from the model name. ``weight_block_size`` is the
    block-FP8 projection block geometry; an empty tuple means per-tensor fp8
    (which this contract does not own — that stays ``QuantizationConfig`` None).
    """

    quant_method: str
    weight_fmt: str | None = None  # e.g. "e4m3"
    scale_fmt: str | None = None  # e.g. "ue8m0"
    weight_block_size: tuple[int, ...] = ()
    activation_scheme: str | None = None
    expert_dtype: str | None = None  # e.g. "fp4"

    @property
    def is_block_scaled_fp8(self) -> bool:
        """True for block-scaled fp8 (a non-empty ``weight_block_size``)."""
        return self.quant_method == "fp8" and len(self.weight_block_size) > 0

    @property
    def has_packed_fp4_experts(self) -> bool:
        """True when routed experts are packed fp4 (``expert_dtype`` fp4-like)."""
        return (self.expert_dtype or "").lower() in {"fp4", "nvfp4", "mxfp4"}

    @property
    def is_owned(self) -> bool:
        """True when this scheme is one this block-quant contract handles."""
        return self.is_block_scaled_fp8 or self.has_packed_fp4_experts

    @classmethod
    def from_quantization_config(
        cls, qc: dict | None, *, expert_dtype: str | None = None
    ) -> BlockQuantScheme | None:
        """Parse a ``quantization_config`` dict; ``None`` if not owned here.

        Returns ``None`` for absent configs, per-tensor fp8 (no
        ``weight_block_size``), and non-fp8 methods without fp4 experts.
        """
        if not isinstance(qc, dict):
            if expert_dtype and str(expert_dtype).lower() in {"fp4", "nvfp4", "mxfp4"}:
                return cls(quant_method="none", expert_dtype=str(expert_dtype).lower())
            return None
        block = qc.get("weight_block_size")
        block_t: tuple[int, ...] = (
            tuple(int(x) for x in block) if isinstance(block, (list, tuple)) else ()
        )
        scheme = cls(
            quant_method=str(qc.get("quant_method", "none")),
            weight_fmt=(str(qc["fmt"]).lower() if qc.get("fmt") is not None else None),
            scale_fmt=(
                str(qc["scale_fmt"]).lower() if qc.get("scale_fmt") is not None else None
            ),
            weight_block_size=block_t,
            activation_scheme=(
                str(qc["activation_scheme"])
                if qc.get("activation_scheme") is not None
                else None
            ),
            expert_dtype=(str(expert_dtype).lower() if expert_dtype else None),
        )
        return scheme if scheme.is_owned else None

    @classmethod
    def from_hf_config(cls, hf_config) -> BlockQuantScheme | None:
        """Parse from a HF config object (reads ``quantization_config``)."""
        qc = getattr(hf_config, "quantization_config", None)
        if qc is not None and hasattr(qc, "to_dict"):
            qc = qc.to_dict()
        return cls.from_quantization_config(
            qc, expert_dtype=getattr(hf_config, "expert_dtype", None)
        )


# ---------------------------------------------------------------------------
# The breaking per-tensor descriptor
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class QuantizedTensorDescriptor:
    """A complete, self-describing load contract for one quantized tensor.

    Carries the *logical* shape (what the math sees), the *packed* on-disk
    shape (what the bytes are), the weight/scale dtypes and names, the block
    geometry in logical-element units, the exact byte counts, and the tensor's
    structural role. This is deliberately verbose and breaking: consumers must
    read explicit fields, never re-derive layout from a name.
    """

    name: str
    kind: QuantKind
    weight_dtype: str
    logical_shape: tuple[int, ...]
    packed_shape: tuple[int, ...]
    weight_num_bytes: int
    is_routed_expert: bool
    is_shared_expert: bool
    block_shape: tuple[int, ...] | None = None
    scale_name: str | None = None
    scale_dtype: str | None = None
    scale_shape: tuple[int, ...] | None = None
    scale_layout: str | None = None
    scale_num_bytes: int | None = None
    microscale_kind: str | None = None  # "mxfp4" | "nvfp4" | None
    unsupported_reason: str | None = None

    @property
    def pack_factor(self) -> int:
        """Logical elements per stored element (2 for nibble-packed fp4)."""
        stored = _num_elements(self.packed_shape)
        return _num_elements(self.logical_shape) // stored if stored else 1


def _expert_role(name: str) -> tuple[bool, bool]:
    """Return ``(is_routed_expert, is_shared_expert)`` from the module path.

    This is a *structural* graph property (the standard HF MoE naming:
    ``...experts.<i>...`` routed, ``...shared_experts...`` shared), not a
    model-name allowlist — the numeric *kind* is classified separately from
    dtype/scale properties.
    """
    is_shared = "shared_expert" in name
    is_routed = (".experts." in name) and not is_shared
    return is_routed, is_shared


def classify_tensor(
    name: str,
    weight_dtype: str,
    weight_shape: tuple[int, ...],
    *,
    scale_dtype: str | None = None,
    scale_shape: tuple[int, ...] | None = None,
    scale_name: str | None = None,
    scheme: BlockQuantScheme | None = None,
) -> QuantizedTensorDescriptor:
    """Classify one tensor into a :class:`QuantizedTensorDescriptor` by property.

    Uses only the weight dtype/shape and the paired scale dtype/shape (plus the
    scheme's block size for block-FP8). Never inspects the model name for the
    numeric family. Unrecognized combinations return ``kind=UNSUPPORTED`` with a
    reason instead of guessing.
    """
    weight_shape = tuple(int(d) for d in weight_shape)
    scale_shape = tuple(int(d) for d in scale_shape) if scale_shape is not None else None
    is_routed, is_shared = _expert_role(name)
    w_elem_bytes = SAFETENSORS_DTYPE_BYTES.get(weight_dtype)
    scale_elem_bytes = SAFETENSORS_DTYPE_BYTES.get(scale_dtype) if scale_dtype else None
    weight_num_bytes = _num_elements(weight_shape) * (w_elem_bytes or 0)
    scale_num_bytes = (
        _num_elements(scale_shape) * scale_elem_bytes
        if scale_shape is not None and scale_elem_bytes is not None
        else None
    )

    def _desc(
        kind: QuantKind,
        *,
        logical: tuple[int, ...],
        block: tuple[int, ...] | None = None,
        layout: str | None = None,
        microscale: str | None = None,
        reason: str | None = None,
    ) -> QuantizedTensorDescriptor:
        return QuantizedTensorDescriptor(
            name=name,
            kind=kind,
            weight_dtype=weight_dtype,
            logical_shape=logical,
            packed_shape=weight_shape,
            weight_num_bytes=weight_num_bytes,
            is_routed_expert=is_routed,
            is_shared_expert=is_shared,
            block_shape=block,
            scale_name=scale_name,
            scale_dtype=scale_dtype,
            scale_shape=scale_shape,
            scale_layout=layout,
            scale_num_bytes=scale_num_bytes,
            microscale_kind=microscale,
            unsupported_reason=reason,
        )

    # Ordinary float tensors carry no scale.
    if weight_dtype in _FLOAT_DTYPES and scale_dtype is None:
        return _desc(QuantKind.ORDINARY, logical=weight_shape)

    # FP4-packed: I8 nibbles + UE8M0 micro-scale.
    if weight_dtype == "I8" and scale_dtype == "F8_E8M0":
        if len(weight_shape) < 1:
            return _desc(
                QuantKind.UNSUPPORTED, logical=weight_shape, reason="scalar I8 weight"
            )
        logical = (*weight_shape[:-1], weight_shape[-1] * 2)  # last dim halved on disk
        # MXFP4 pins the micro-scale block to 32 logical input elements per row;
        # this is a format property, not something inferred from a (possibly
        # wrong) scale grid — validate_descriptor checks the scale against it.
        block_len = MXFP4_MICROSCALE_BLOCK
        return _desc(
            QuantKind.FP4_PACKED,
            logical=logical,
            block=(1, block_len),
            layout=f"microscale_1x{block_len}_ue8m0",
            microscale="mxfp4",
        )

    # Block-FP8: E4M3 weight + 2D UE8M0 block scale.
    if weight_dtype == "F8_E4M3" and scale_dtype == "F8_E8M0":
        bs = scheme.weight_block_size if scheme and scheme.weight_block_size else None
        if (
            bs is None
            and scale_shape is not None
            and len(weight_shape) == len(scale_shape) == 2
        ):
            # Infer square block geometry from the ceil-divided scale grid.
            bs0 = -(-weight_shape[0] // scale_shape[0]) if scale_shape[0] else 0
            bs1 = -(-weight_shape[1] // scale_shape[1]) if scale_shape[1] else 0
            bs = (bs0, bs1)
        block = tuple(int(x) for x in bs) if bs else None
        layout = (
            f"block{block[0]}x{block[1]}_ue8m0"
            if block and len(block) == 2
            else "block2d_ue8m0"
        )
        return _desc(QuantKind.BLOCK_FP8, logical=weight_shape, block=block, layout=layout)

    # Recognized-but-unhandled or malformed combinations fail closed.
    if weight_dtype.startswith("F8") and scale_dtype is None:
        return _desc(
            QuantKind.UNSUPPORTED,
            logical=weight_shape,
            reason=f"{weight_dtype} weight without a paired block scale",
        )
    if weight_dtype == "I8" and scale_dtype is None:
        return _desc(
            QuantKind.UNSUPPORTED,
            logical=weight_shape,
            reason="I8 weight without a paired UE8M0 micro-scale (cannot be fp4)",
        )
    return _desc(
        QuantKind.UNSUPPORTED,
        logical=weight_shape,
        reason=f"unrecognized weight/scale dtype pair ({weight_dtype}, {scale_dtype})",
    )


def validate_descriptor(desc: QuantizedTensorDescriptor) -> None:
    """Validate a descriptor's internal shape/scale consistency; raise on error.

    Checks (per kind): logical-vs-packed shape relation, scale presence + dtype,
    scale grid vs block geometry, and byte counts. Raises
    :class:`BlockQuantValidationError` on any contradiction.
    """
    if desc.kind is QuantKind.UNSUPPORTED:
        raise BlockQuantValidationError(
            f"{desc.name}: unsupported tensor ({desc.unsupported_reason})"
        )

    if desc.kind is QuantKind.ORDINARY:
        if desc.scale_name is not None or desc.scale_shape is not None:
            raise BlockQuantValidationError(
                f"{desc.name}: ordinary float tensor must not carry a scale"
            )
        if desc.logical_shape != desc.packed_shape:
            raise BlockQuantValidationError(
                f"{desc.name}: ordinary tensor logical {desc.logical_shape} != "
                f"packed {desc.packed_shape}"
            )
        return

    # Both quantized kinds require a paired scale.
    if desc.scale_shape is None or desc.scale_dtype is None:
        raise BlockQuantValidationError(
            f"{desc.name}: {desc.kind.value} tensor has no paired scale"
        )
    if desc.scale_dtype != "F8_E8M0":
        raise BlockQuantValidationError(
            f"{desc.name}: expected UE8M0 (F8_E8M0) scale, got {desc.scale_dtype}"
        )

    if desc.kind is QuantKind.FP4_PACKED:
        if len(desc.packed_shape) != 2 or len(desc.logical_shape) != 2:
            raise BlockQuantValidationError(f"{desc.name}: fp4 experts must be 2D")
        out_l, in_l = desc.logical_shape
        out_p, in_p = desc.packed_shape
        if out_l != out_p:
            raise BlockQuantValidationError(
                f"{desc.name}: fp4 output dim mismatch logical {out_l} vs packed {out_p}"
            )
        if in_l != in_p * 2:
            raise BlockQuantValidationError(
                f"{desc.name}: fp4 packed input {in_p} must be logical {in_l} / 2 "
                f"(two E2M1 nibbles per int8 byte)"
            )
        block_len = desc.block_shape[1] if desc.block_shape else MXFP4_MICROSCALE_BLOCK
        if block_len <= 0 or in_l % block_len != 0:
            raise BlockQuantValidationError(
                f"{desc.name}: logical input {in_l} not divisible by micro-scale block {block_len}"
            )
        expected_scale = (out_l, in_l // block_len)
        if desc.scale_shape != expected_scale:
            raise BlockQuantValidationError(
                f"{desc.name}: fp4 scale shape {desc.scale_shape} != expected "
                f"{expected_scale} (out, logical_in / {block_len})"
            )
        return

    # BLOCK_FP8
    if len(desc.packed_shape) != 2 or len(desc.scale_shape) != 2:
        raise BlockQuantValidationError(
            f"{desc.name}: block-fp8 weight and scale must both be 2D"
        )
    if desc.logical_shape != desc.packed_shape:
        raise BlockQuantValidationError(
            f"{desc.name}: block-fp8 logical {desc.logical_shape} must equal packed "
            f"{desc.packed_shape} (E4M3 is not sub-byte packed)"
        )
    if not desc.block_shape or len(desc.block_shape) != 2:
        raise BlockQuantValidationError(f"{desc.name}: block-fp8 needs a 2D block_shape")
    out_d, in_d = desc.logical_shape
    bs0, bs1 = desc.block_shape
    if bs0 <= 0 or bs1 <= 0:
        raise BlockQuantValidationError(f"{desc.name}: invalid block_shape {desc.block_shape}")
    expected_scale = (-(-out_d // bs0), -(-in_d // bs1))
    if desc.scale_shape != expected_scale:
        raise BlockQuantValidationError(
            f"{desc.name}: block-fp8 scale grid {desc.scale_shape} != expected "
            f"{expected_scale} (ceil(out/{bs0}), ceil(in/{bs1}))"
        )


# ---------------------------------------------------------------------------
# Index-level pairing + classification
# ---------------------------------------------------------------------------

_WEIGHT_SUFFIX = ".weight"
_SCALE_SUFFIX = ".scale"


def pair_weight_scales(
    header_index: dict[str, tuple[str, tuple[int, ...]]],
) -> dict[str, str | None]:
    """Pair each ``<prefix>.weight`` with its ``<prefix>.scale`` sibling.

    ``header_index`` maps ``name -> (dtype, shape)``. Returns ``{weight_name:
    scale_name | None}`` for every ``.weight`` key. Raises
    :class:`BlockQuantValidationError` on an *orphan* scale (a ``.scale`` whose
    ``.weight`` sibling is absent) — a duplicate/misnamed scale that would
    otherwise be silently dropped.
    """
    weights = {k for k in header_index if k.endswith(_WEIGHT_SUFFIX)}
    scales = {k for k in header_index if k.endswith(_SCALE_SUFFIX)}
    pairing: dict[str, str | None] = {}
    for w in sorted(weights):
        sibling = w[: -len(_WEIGHT_SUFFIX)] + _SCALE_SUFFIX
        pairing[w] = sibling if sibling in scales else None
    orphans = sorted(
        s for s in scales if (s[: -len(_SCALE_SUFFIX)] + _WEIGHT_SUFFIX) not in weights
    )
    if orphans:
        raise BlockQuantValidationError(
            f"{len(orphans)} orphan scale tensor(s) with no matching .weight: {orphans[:5]}"
        )
    return pairing


def build_descriptors(
    header_index: dict[str, tuple[str, tuple[int, ...]]],
    scheme: BlockQuantScheme | None = None,
    *,
    validate: bool = True,
) -> dict[str, QuantizedTensorDescriptor]:
    """Classify every ``.weight`` tensor in a header index into a descriptor.

    ``header_index`` maps ``name -> (dtype, shape)`` (the header-only triple a
    caller gets from a safetensors reader). Scale tensors are consumed as pairs
    and not returned standalone. When *validate* is True each descriptor is run
    through :func:`validate_descriptor`, so a malformed group raises atomically.
    """
    pairing = pair_weight_scales(header_index)
    descriptors: dict[str, QuantizedTensorDescriptor] = {}
    for weight_name, scale_name in pairing.items():
        w_dtype, w_shape = header_index[weight_name]
        if scale_name is not None:
            s_dtype, s_shape = header_index[scale_name]
        else:
            s_dtype, s_shape = None, None
        desc = classify_tensor(
            weight_name,
            w_dtype,
            w_shape,
            scale_dtype=s_dtype,
            scale_shape=s_shape,
            scale_name=scale_name,
            scheme=scheme,
        )
        if validate and desc.kind is not QuantKind.UNSUPPORTED:
            validate_descriptor(desc)
        descriptors[weight_name] = desc
    return descriptors


# ---------------------------------------------------------------------------
# Byte-exact expert-major bank stacking (reusable lowering primitive)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PackedExpertBank:
    """A byte-exact expert-major concatenation of per-expert packed weights.

    ``data`` is ``num_experts`` equal-length payloads laid out expert-major
    (expert 0's bytes, then expert 1's, ...). No re-quantization or re-ordering
    within an expert occurs, so the original per-expert bytes are recoverable
    by slicing. This is the reusable primitive a future planar->interleaved
    transcode or a native bank emitter consumes.
    """

    num_experts: int
    per_expert_num_bytes: int
    per_expert_packed_shape: tuple[int, ...]
    weight_dtype: str
    data: bytes

    def expert_bytes(self, i: int) -> bytes:
        """Return expert ``i``'s original bytes (byte-exact slice)."""
        if not 0 <= i < self.num_experts:
            raise IndexError(f"expert {i} out of range [0, {self.num_experts})")
        off = i * self.per_expert_num_bytes
        return self.data[off : off + self.per_expert_num_bytes]


def stack_expert_bank(
    per_expert_bytes: list[bytes],
    *,
    per_expert_packed_shape: tuple[int, ...],
    weight_dtype: str,
) -> PackedExpertBank:
    """Concatenate per-expert packed bytes expert-major, byte-for-byte.

    Every expert must contribute exactly the same number of bytes (a ragged
    bank is a hard error, never zero-padded). The result preserves each
    expert's payload verbatim.
    """
    if not per_expert_bytes:
        raise BlockQuantValidationError("cannot stack an empty expert bank")
    n0 = len(per_expert_bytes[0])
    for i, b in enumerate(per_expert_bytes):
        if len(b) != n0:
            raise BlockQuantValidationError(
                f"ragged expert bank: expert 0 has {n0} bytes but expert {i} has {len(b)}"
            )
    return PackedExpertBank(
        num_experts=len(per_expert_bytes),
        per_expert_num_bytes=n0,
        per_expert_packed_shape=tuple(int(d) for d in per_expert_packed_shape),
        weight_dtype=weight_dtype,
        data=b"".join(per_expert_bytes),
    )


# ---------------------------------------------------------------------------
# Runtime (nxrt) emission gate — prove representability or typed-reject
# ---------------------------------------------------------------------------

#: Block formats accepted by the canonical onnx-genai ``pkg.nxrt`` v1 ABI.
#: The planar formats use the dedicated auxiliary-scale input; the remaining
#: formats are self-describing interleaved blocks.
NXRT_BLOCK_FORMATS: frozenset[str] = frozenset(
    {
        "block_fp8",
        "fp4_planar",
        "mxfp4",
        "iq4_nl",
        "iq4_xs",
        "iq3_s",
        "iq3_xxs",
        "iq2_s",
        "iq2_xs",
        "iq2_xxs",
        "iq1_s",
        "iq1_m",
    }
)


def runtime_representation_gap(
    desc: QuantizedTensorDescriptor, *, runtime: str = "nxrt"
) -> str | None:
    """Return a precise ABI-gap string if *desc* is not runtime-representable.

    Returns ``None`` when the tensor could be emitted for *runtime* today.
    Purely a property check — it never emits a node. Only ``nxrt`` is modelled.
    """
    if runtime != "nxrt":
        return f"unknown runtime {runtime!r}; only 'nxrt' representability is modelled"

    if desc.kind is QuantKind.ORDINARY:
        return None

    if desc.kind in (QuantKind.BLOCK_FP8, QuantKind.FP4_PACKED):
        return None

    return f"{desc.name}: unsupported tensor ({desc.unsupported_reason})"


def plan_routed_expert_bank(
    expert_descriptors: list[QuantizedTensorDescriptor],
    *,
    runtime: str = "nxrt",
    per_expert_bytes: list[bytes] | None = None,
) -> PackedExpertBank:
    """Prove a routed-expert bank is runtime-representable, else typed-reject.

    Validates that every routed expert shares one packed shape / dtype / scale
    layout, then checks :func:`runtime_representation_gap`. The caller supplies
    byte-exact per-expert payloads; no dense fallback or dequantization occurs.
    """
    if not expert_descriptors:
        raise BlockQuantValidationError("cannot plan an empty routed-expert bank")

    non_routed = [d.name for d in expert_descriptors if not d.is_routed_expert]
    if non_routed:
        raise BlockQuantValidationError(
            f"plan_routed_expert_bank got non-routed tensor(s): {non_routed[:5]}"
        )

    first = expert_descriptors[0]
    for d in expert_descriptors:
        validate_descriptor(d)
        if (d.kind, d.packed_shape, d.weight_dtype, d.scale_layout) != (
            first.kind,
            first.packed_shape,
            first.weight_dtype,
            first.scale_layout,
        ):
            raise BlockQuantValidationError(
                "mixed expert bank: all routed experts must share kind/packed_shape/"
                f"dtype/scale layout; {d.name} differs from {first.name}"
            )

    gap = runtime_representation_gap(first, runtime=runtime)
    if gap is not None:
        raise BlockQuantExportError(
            f"Routed-expert bank ({len(expert_descriptors)} x {first.kind.value}) is not "
            f"representable by the {runtime!r} runtime, so no BlockQuantizedMoE node is "
            f"emitted (fail closed, no dense fallback, no dequantization). ABI gap: {gap} "
            "Resolving this needs a runtime ABI extension or a proven byte-exact "
            "layout conversion primitive."
        )

    # Build the bank only from caller-supplied byte-exact payloads.
    if per_expert_bytes is None:
        raise BlockQuantValidationError(
            "runtime can represent the bank but no per_expert_bytes were supplied to pack it"
        )
    if len(per_expert_bytes) != len(expert_descriptors):
        raise BlockQuantValidationError(
            f"per_expert_bytes count {len(per_expert_bytes)} != "
            f"{len(expert_descriptors)} expert descriptors"
        )
    return stack_expert_bank(
        per_expert_bytes,
        per_expert_packed_shape=first.packed_shape,
        weight_dtype=first.weight_dtype,
    )
