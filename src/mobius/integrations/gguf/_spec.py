# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Immutable capability specs for GGUF import support.

This module is the vocabulary for "what does mobius support?". It holds only
dataclasses, enums, and their validation — it imports nothing else from
:mod:`mobius.integrations.gguf`, so every other module in the package can
import it without creating a cycle.

Two ideas are kept deliberately separate:

* :class:`GGUFArchitectureSpec` answers *architecture* questions, and splits
  them into four independent verdicts rather than one boolean, because they
  genuinely differ per architecture:

  ``config``
      Can GGUF metadata be turned into an ``ArchitectureConfig``?
  ``tensor_map``
      Can GGUF tensor names be mapped onto HuggingFace names?
  ``graph``
      Does mobius have a registered model class that can build the graph?
  ``runtime``
      Can a loadable runtime package (tokenizer + inference metadata) be written?

* :class:`GGUFQuantSpec` answers *stored-quantization* questions, and likewise
  splits them:

  ``readable``
      Can the GGUF parse layer even read the tensor? (Removed ggml types have
      ``block_elements == 0`` and are rejected before any architecture logic.)
  ``dequantize``
      Can the block data be expanded to float?
  ``native_preserve``
      Can the block data be handed to the runtime byte-for-byte?
  ``affine_repack``
      Can the block data be repacked into a ``MatMulNBits`` affine layout?

A verdict that is not :attr:`Support.SUPPORTED` must carry a ``reason``. That
is what stops the registry from silently implying capability: listing an
architecture or a quantization type is never itself a support claim.

Behavior is referenced **by name**, never by callable. ``config_postprocessor``,
``tensor_processor``, ``tensor_map_recipe`` and ``vlm_builder`` are string keys
resolved by the module that owns the implementation. That keeps this module and
:mod:`mobius.integrations.gguf._arch_registry` free of imports on the
implementation modules, and makes an unreferenced implementation a test failure
rather than dead weight.
"""

from __future__ import annotations

__all__ = [
    "AffineRepackSpec",
    "GGUFArchitectureSpec",
    "GGUFQuantSpec",
    "QuantImportRoute",
    "RepackExactness",
    "NativeBlockSpec",
    "StorageRole",
    "Support",
    "TensorRole",
]

import dataclasses
import enum

from mobius.integrations.gguf._runtime_evidence import (
    validate_quant_runtime_evidence_ids,
    validate_runtime_evidence_ids,
)


class Support(enum.Enum):
    """Verdict for a single capability.

    Attributes:
        SUPPORTED: Implemented and covered by a test.
        DEFERRED: Not implemented yet, but no known blocker. Needs a ``reason``.
        REJECTED: Deliberately refused. Needs a ``reason``.
    """

    SUPPORTED = "supported"
    DEFERRED = "deferred"
    REJECTED = "rejected"


class StorageRole(enum.Enum):
    """How a ggml type slot may appear in a GGUF file.

    Attributes:
        FLOAT: Unquantized weight storage (``f32``, ``f16``, ``bf16``).
        QUANTIZED: Block-quantized weight storage.
        AUX: Valid tensor type that never holds model weights (``i8``..``f64``).
        COMPUTE_ONLY: Quantized but has no ``to_float``; a ``vec_dot``
            intermediate only (``q8_1``, ``q8_K``). Seeing one stored in a GGUF
            means the file is malformed.
        REMOVED: Slot retired upstream. ``blck_size`` is 0, so the GGUF parse
            layer rejects the file before any architecture logic runs.
    """

    FLOAT = "storage-float"
    QUANTIZED = "storage-quantized"
    AUX = "aux-nonweight"
    COMPUTE_ONLY = "compute-only"
    REMOVED = "removed-unreadable"


class QuantImportRoute(enum.Enum):
    """How a stored quantized tensor reaches the ONNX graph."""

    NATIVE_BYTES = "native byte-preserved"
    AFFINE_REPACK = "affine repack"
    DEQUANTIZE_REQUANTIZE = "dequantize/requantize"
    DEQUANTIZE_FLOAT = "dequantize to float"
    REJECTED = "rejected"


class RepackExactness(enum.Enum):
    """Whether an affine conversion preserves every represented source value."""

    EXACT = "exact"
    LOSSY = "lossy"


class TensorRole(enum.Enum):
    """Runtime ABI selected for a mapped GGUF tensor."""

    PROJECTION = "projection"
    AFFINE_PROJECTION = "projection (affine-only graph)"
    OUTPUT = "output"
    EMBEDDING = "embedding"
    EXPERT = "expert-major"
    NON_MATMUL = "non-MatMul"


@dataclasses.dataclass(frozen=True, slots=True)
class NativeBlockSpec:
    """Serialized GGUF block layout matching the custom operator input ABI.

    Attributes:
        format: Runtime format string (e.g. ``"mxfp4"``).
        elements: Weights per block.
        bytes: Serialized bytes per block.
    """

    format: str
    elements: int
    bytes: int

    def __post_init__(self) -> None:
        if not self.format:
            raise ValueError("NativeBlockSpec.format must be non-empty")
        if self.elements <= 0 or self.bytes <= 0:
            raise ValueError(
                f"NativeBlockSpec {self.format!r} must have positive elements/bytes, "
                f"got elements={self.elements} bytes={self.bytes}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class AffineRepackSpec:
    """``MatMulNBits`` representation produced for a repackable GGUF type.

    Attributes:
        bits: Quantization bit width of the repacked form.
        block_size: Weights per block in the repacked form.
        omit_zero_points: Whether the repacked form may drop the ``zero_points``
            input. This is a property of the *target*, not of the source: Q6_K
            is symmetric on disk but requantizes through the asymmetric affine
            path, so it still needs zero points.
        lossless: Whether repacking preserves the source dequantized values
            without requantization.
    """

    bits: int
    block_size: int
    omit_zero_points: bool = False
    lossless: bool = False

    def __post_init__(self) -> None:
        if self.bits <= 0 or self.block_size <= 0:
            raise ValueError(
                f"AffineRepackSpec needs positive bits/block_size, got "
                f"bits={self.bits} block_size={self.block_size}"
            )

    def as_params(self) -> tuple[int, int]:
        """Return the legacy ``(bits, block_size)`` tuple."""
        return (self.bits, self.block_size)


def _require_reason(label: str, verdicts: dict[str, Support], reason: str | None) -> None:
    """Raise unless every non-``SUPPORTED`` verdict is justified."""
    unsupported = sorted(
        name for name, verdict in verdicts.items() if verdict is not Support.SUPPORTED
    )
    if unsupported and not reason:
        raise ValueError(
            f"{label}: capabilities {unsupported} are not SUPPORTED but no reason was "
            "given. Every non-SUPPORTED verdict must say why, so that rejections stay "
            "actionable and support is never implied by omission."
        )


@dataclasses.dataclass(frozen=True, slots=True)
class GGUFArchitectureSpec:
    """Everything mobius knows about one GGUF ``general.architecture`` value.

    Attributes:
        gguf_arch: The canonical architecture string as llama.cpp writes it.
            Must be one of the architectures in the pinned upstream census.
        aliases: Additional strings accepted for this architecture. Used for
            defensive spellings that llama.cpp does not currently emit.
        model_type: The mobius registry ``model_type`` this maps to.
        module_type: Optional internal registry key when the GGUF runtime graph
            intentionally differs from the HuggingFace graph for ``model_type``.
        config: Whether GGUF metadata can be turned into a config.
        tensor_map: Whether GGUF tensor names can be mapped to HuggingFace names.
        graph: Whether mobius can build the graph for ``model_type``.
        runtime: Whether a loadable runtime package can be written.
        quantized_import: Whether the graph exposes packed projection modules
            that can consume a preserved GGUF quantization route.
        reason: Why any non-``SUPPORTED`` verdict is what it is. Required
            whenever at least one verdict is not ``SUPPORTED``.
        config_key_map: Name of the architecture-specific GGUF-key → config-field
            map owned by ``_config_mapping``, or ``None`` for the default map.
        config_postprocessor: Name of the config postprocessor owned by
            ``_config_mapping``, or ``None``.
        tensor_map_recipe: Ordered names of the tensor-name mapping tables owned
            by ``_tensor_mapping``. Later entries override earlier ones.
        required_metadata: Architecture-scoped GGUF metadata suffixes that must
            be present before config extraction can claim success.
        tensor_processor: Name of the weight-value processor owned by
            ``_tensor_processors``, or ``None``.
        vlm_builder: Name of the multimodal assembly entry point owned by
            ``_mmproj``, or ``None`` for text-only architectures.
        llama_qk_permute: Whether llama.cpp stores Q/K with the interleaved-rope
            permutation, which the importer has to undo.
        offset_norm: Whether the converter bakes ``+1`` into ``*norm.weight``.
        v_head_reorder: Whether the converter tiles Gated-DeltaNet V-heads.
        rope_interleave: Whether the architecture rotates adjacent pairs
            (GPT-J style) rather than split-half (NEOX style).
        preflight_only: Whether exact header/config/tensor evidence exists while
            graph import is intentionally unavailable.
    """

    gguf_arch: str
    model_type: str | None = None
    module_type: str | None = None
    aliases: frozenset[str] = frozenset()
    config: Support = Support.SUPPORTED
    tensor_map: Support = Support.SUPPORTED
    graph: Support = Support.SUPPORTED
    runtime: Support = Support.DEFERRED
    quantized_import: Support = Support.SUPPORTED
    reason: str | None = None
    runtime_evidence_ids: tuple[str, ...] = ()
    config_key_map: str | None = None
    config_postprocessor: str | None = None
    tensor_map_recipe: tuple[str, ...] = ()
    required_metadata: tuple[str, ...] = ()
    tensor_processor: str | None = None
    vlm_builder: str | None = None
    llama_qk_permute: bool = False
    offset_norm: bool = False
    v_head_reorder: bool = False
    rope_interleave: bool = False
    preflight_only: bool = False

    def __post_init__(self) -> None:
        if not self.gguf_arch:
            raise ValueError("GGUFArchitectureSpec.gguf_arch must be non-empty")
        if self.gguf_arch in self.aliases:
            raise ValueError(f"{self.gguf_arch!r}: canonical name must not repeat in aliases")
        _require_reason(
            f"GGUF architecture {self.gguf_arch!r}", self.capabilities, self.reason
        )
        if self.runtime is Support.SUPPORTED:
            validate_runtime_evidence_ids(self.gguf_arch, self.runtime_evidence_ids)
        if self.runtime is not Support.SUPPORTED and self.runtime_evidence_ids:
            raise ValueError(
                f"{self.gguf_arch!r}: runtime evidence cannot accompany "
                f"runtime={self.runtime.value}"
            )
        if self.graph is Support.SUPPORTED and not self.model_type:
            raise ValueError(
                f"{self.gguf_arch!r}: graph=SUPPORTED requires a model_type to build with"
            )
        if self.module_type is not None and self.graph is not Support.SUPPORTED:
            raise ValueError(f"{self.gguf_arch!r}: module_type requires graph=SUPPORTED")
        if self.tensor_map is Support.SUPPORTED and not self.tensor_map_recipe:
            raise ValueError(
                f"{self.gguf_arch!r}: tensor_map=SUPPORTED requires a non-empty "
                "tensor_map_recipe"
            )
        if self.tensor_map is not Support.SUPPORTED and self.tensor_map_recipe:
            raise ValueError(
                f"{self.gguf_arch!r}: tensor_map is {self.tensor_map.value} but a "
                "tensor_map_recipe was given, which would imply it works"
            )
        if self.config is not Support.SUPPORTED and self.required_metadata:
            raise ValueError(
                f"{self.gguf_arch!r}: config is {self.config.value} but required "
                "metadata was declared, which would imply config extraction works"
            )
        if self.preflight_only and (
            self.config is not Support.SUPPORTED
            or self.tensor_map is not Support.SUPPORTED
            or self.graph is Support.SUPPORTED
        ):
            raise ValueError(
                f"{self.gguf_arch!r}: preflight_only requires supported config/tensor "
                "evidence and an unavailable graph route"
            )

    @property
    def verdicts(self) -> dict[str, Support]:
        """The core float-import verdicts, keyed by capability name."""
        return {
            "config": self.config,
            "tensor_map": self.tensor_map,
            "graph": self.graph,
            "runtime": self.runtime,
        }

    @property
    def capabilities(self) -> dict[str, Support]:
        """All architecture verdicts, including quantized-import reachability."""
        return {
            **self.verdicts,
            "quantized_import": self.quantized_import,
        }

    @property
    def names(self) -> frozenset[str]:
        """The canonical name together with every accepted alias."""
        return frozenset({self.gguf_arch}) | self.aliases

    @property
    def is_importable(self) -> bool:
        """Whether config, tensor mapping, and graph build are all supported."""
        return (
            self.config is Support.SUPPORTED
            and self.tensor_map is Support.SUPPORTED
            and self.graph is Support.SUPPORTED
        )


@dataclasses.dataclass(frozen=True, slots=True)
class GGUFQuantSpec:
    """Everything mobius knows about one ``ggml_type`` slot.

    Attributes:
        ggml_type_id: The numeric ``enum ggml_type`` value.
        name: Upper-case type name (e.g. ``"Q4_K"``). Removed slots keep the
            retired name so error messages can still identify them.
        role: How the slot may appear in a GGUF file.
        block_elements: Weights per block upstream. ``0`` for removed slots,
            which is exactly why the GGUF parse layer rejects them.
        block_bytes: Serialized bytes per block upstream. ``0`` for removed slots.
        dequantize: Whether mobius can expand the block data to float.
        native_preserve: Runtime-native block layout, when the bytes can be
            handed to the runtime unchanged.
        affine_repack: ``MatMulNBits`` target, when the block data can be
            repacked into an affine layout.
        import_route: Primary importer route for projection/output weights.
        repack_exactness: Whether ``affine_repack`` preserves represented values.
        runtime: Runtime execution verdict. Graph construction or ABI matching
            alone is not runtime evidence.
        runtime_evidence_ids: Immutable full-logit/stateful evidence records that
            qualify this exact stored qtype for runtime execution.
        requires_explicit_zero_point: Whether a file containing this type forces
            the shared graph scaffolding to carry explicit ``zero_points``.
        lm_head_preserve: Whether an untied output head stored in this type may
            stay quantized.
        reason: Why ``dequantize`` is not ``SUPPORTED``. Required in that case.
    """

    ggml_type_id: int
    name: str
    role: StorageRole
    block_elements: int
    block_bytes: int
    dequantize: Support = Support.SUPPORTED
    native_preserve: NativeBlockSpec | None = None
    affine_repack: AffineRepackSpec | None = None
    import_route: QuantImportRoute = QuantImportRoute.REJECTED
    repack_exactness: RepackExactness | None = None
    runtime: Support = Support.DEFERRED
    runtime_reason: str = "No real-weight ONNX Runtime execution evidence is recorded."
    runtime_evidence_ids: tuple[str, ...] = ()
    requires_explicit_zero_point: bool = False
    lm_head_preserve: bool = False
    reason: str | None = None

    def __post_init__(self) -> None:
        label = f"GGML type {self.name!r} (id {self.ggml_type_id})"
        if not self.name:
            raise ValueError(f"GGML type id {self.ggml_type_id} must have a name")
        _require_reason(label, {"dequantize": self.dequantize}, self.reason)
        _require_reason(label, {"runtime": self.runtime}, self.runtime_reason)
        if self.runtime is Support.SUPPORTED:
            validate_quant_runtime_evidence_ids(self.name, self.runtime_evidence_ids)
        if self.runtime is not Support.SUPPORTED and self.runtime_evidence_ids:
            raise ValueError(
                f"{label}: runtime evidence cannot accompany runtime={self.runtime.value}"
            )
        # Upstream rejects a tensor whose type has blck_size == 0 in gguf.cpp
        # before any model logic runs, so readability is not an independent
        # choice — it is a consequence of the pinned block size.
        if self.readable != (self.block_elements > 0):
            raise ValueError(
                f"{label}: readable={self.readable} contradicts "
                f"block_elements={self.block_elements}"
            )
        if not self.readable and self.dequantize is Support.SUPPORTED:
            raise ValueError(
                f"{label}: unreadable slots cannot be dequantized; the GGUF parse "
                "layer rejects them first"
            )

        if self.native_preserve is not None and self.affine_repack is not None:
            raise ValueError(
                f"{label}: a type is either preserved natively or repacked into an "
                "affine layout, never both — two paths would silently disagree"
            )
        if self.import_route is QuantImportRoute.NATIVE_BYTES and self.native_preserve is None:
            raise ValueError(f"{label}: native-byte route requires native_preserve")
        if self.import_route is QuantImportRoute.AFFINE_REPACK and self.affine_repack is None:
            raise ValueError(f"{label}: affine route requires affine_repack")
        if (
            self.import_route is QuantImportRoute.DEQUANTIZE_REQUANTIZE
            and self.dequantize is not Support.SUPPORTED
        ):
            raise ValueError(f"{label}: dequantize/requantize route requires a dequantizer")
        if self.import_route is QuantImportRoute.REJECTED and (
            self.native_preserve is not None or self.affine_repack is not None
        ):
            raise ValueError(f"{label}: rejected route cannot expose a preservation path")
        if self.affine_repack is None and self.repack_exactness is not None:
            raise ValueError(f"{label}: repack exactness requires an affine repack target")
        if self.affine_repack is not None and self.repack_exactness is None:
            raise ValueError(f"{label}: affine repack target must declare exact or lossy")
        if self.native_preserve is not None:
            if self.native_preserve.elements != self.block_elements:
                raise ValueError(
                    f"{label}: native block elements "
                    f"({self.native_preserve.elements}) must match the upstream "
                    f"block size ({self.block_elements})"
                )
            if self.native_preserve.bytes != self.block_bytes:
                raise ValueError(
                    f"{label}: native block bytes ({self.native_preserve.bytes}) must "
                    f"match the upstream type size ({self.block_bytes})"
                )
        if self.lm_head_preserve and not self.is_quantized_storage:
            raise ValueError(
                f"{label}: lm_head_preserve only applies to quantized storage types"
            )
        if self.requires_explicit_zero_point and not self.is_quantized_storage:
            raise ValueError(
                f"{label}: requires_explicit_zero_point only applies to quantized "
                "storage types"
            )

    @property
    def preserves_values(self) -> bool:
        """Whether a quantized graph can consume this type without requantization."""
        return self.native_preserve is not None or (
            self.affine_repack is not None and self.affine_repack.lossless
        )

    @property
    def readable(self) -> bool:
        """Whether the GGUF parse layer accepts a tensor of this type."""
        return self.role is not StorageRole.REMOVED

    @property
    def is_quantized_storage(self) -> bool:
        """Whether this type is block-quantized weight storage."""
        return self.role is StorageRole.QUANTIZED

    @property
    def repack_params(self) -> tuple[int, int] | None:
        """The legacy ``(bits, block_size)`` repack target, if any."""
        return None if self.affine_repack is None else self.affine_repack.as_params()
