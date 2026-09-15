# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Weight-quantization configuration parsed from HuggingFace configs."""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Mapping


class QuantizedWeightFormat(str, enum.Enum):
    """Storage semantics of quantized checkpoint weights.

    ``INTEGER_AFFINE`` is the existing GPTQ/AWQ/Olive representation. ``MXFP4``
    denotes native E2M1 codes with one E8M0 scale per 32-value block; it must
    never be interpreted as affine INT4.
    """

    INTEGER_AFFINE = "integer_affine"
    MXFP4 = "mxfp4"


@dataclasses.dataclass(frozen=True)
class QuantizationOverride:
    """Per-module overrides emitted by Olive mixed-precision quantization."""

    bits: int | None = None
    group_size: int | None = None
    sym: bool | None = None

    @classmethod
    def from_value(cls, value: object) -> QuantizationOverride:
        """Parse one serialized Olive override."""
        if isinstance(value, cls):
            return value
        if hasattr(value, "to_dict"):
            value = value.to_dict()
        if not isinstance(value, Mapping):
            raise TypeError(
                f"quantization override must be a mapping, got {type(value).__name__}"
            )
        return cls(
            bits=value.get("bits"),
            group_size=value.get("group_size"),
            sym=value.get("sym", value.get("symmetric")),
        )

    def apply(self, config: QuantizationConfig) -> QuantizationConfig:
        """Return *config* with this override applied."""
        updates = {
            name: value
            for name, value in dataclasses.asdict(self).items()
            if value is not None
        }
        return dataclasses.replace(config, **updates)


@dataclasses.dataclass
class QuantizationConfig:
    """Weight quantization parameters parsed from HuggingFace configs.

    Captures the settings from ``quantization_config`` in HuggingFace model
    configs (GPTQ, AWQ, etc.) so models can decide whether to use
    :class:`~mobius.components.QuantizedLinear` instead of
    :class:`~mobius.components.Linear`.
    """

    bits: int = 4
    group_size: int = 128
    quant_method: str = "none"
    sym: bool = True
    # When True, zero_points is a per-block float tensor rather than a
    # bit-packed uint8. Required for codebooks with non-integer offsets
    # (e.g. Tencent SEQ uses 1.5).
    float_zero_point: bool = False
    # When True, the input embedding table is block-wise quantized and is
    # looked up with GatherBlockQuantized instead of a plain Gather. Used by
    # Olive RTN exports and quantized GGUF imports.
    quantize_embeddings: bool = False
    # When True, the LM head projection is block-wise quantized (MatMulNBits).
    # Used by Olive RTN exports and quantized GGUF imports.
    quantize_lm_head: bool = False
    # When True, multimodal vision projections are block-wise quantized.
    # Olive RTN records this when ``quantize_vision`` is enabled.
    quantize_vision: bool = False
    # When True, the input embedding and LM head share one weight table. Olive
    # RTN records this in its own config (``tie_word_embeddings``) and may clear
    # the model's top-level flag, so it is tracked here independently.
    tie_word_embeddings: bool = False
    # Root-relative HuggingFace module names left in floating point by Olive's
    # mixed-precision planner. ``None`` means no component plan was recorded;
    # an empty tuple means the planner explicitly quantized every eligible
    # module with the default configuration.
    modules_to_not_convert: tuple[str, ...] | None = None
    # Olive per-module precision overrides. Mobius can collapse uniform
    # overrides beneath one component into that component's effective config.
    overrides: dict[str, QuantizationOverride] = dataclasses.field(default_factory=dict)
    # Keep this field last: QuantizationConfig has historically supported
    # positional construction, so inserting a field earlier would silently
    # change the meaning of existing callers' arguments.
    weight_format: QuantizedWeightFormat = QuantizedWeightFormat.INTEGER_AFFINE

    def __post_init__(self) -> None:
        """Normalize serialized enum values without inferring storage semantics."""
        if isinstance(self.weight_format, str):
            try:
                self.weight_format = QuantizedWeightFormat(self.weight_format)
            except ValueError as exc:
                raise ValueError(
                    f"Unknown quantized weight format {self.weight_format!r}; "
                    f"expected one of {[item.value for item in QuantizedWeightFormat]}"
                ) from exc
        elif not isinstance(self.weight_format, QuantizedWeightFormat):
            raise TypeError(
                "weight_format must be a QuantizedWeightFormat or its serialized "
                f"string value, got {type(self.weight_format).__name__}"
            )

    @classmethod
    def from_value(
        cls,
        value: object,
        *,
        expert_dtype: object | None = None,
    ) -> QuantizationConfig | None:
        """Parse a serialized HuggingFace quantization-config value."""
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if hasattr(value, "to_dict"):
            value = value.to_dict()
        if not isinstance(value, Mapping):
            return None
        qc = dict(value)
        method = qc.get("quant_method", "none")
        # NVIDIA ModelOpt NVFP4/FP8 checkpoints (e.g. quantized Qwen3.6) encode
        # weights as packed E2M1 (fp4) / float8 with block + global scales — a
        # layout the INT4 ``QuantizedLinear``/``MatMulNBits`` path below would
        # silently mis-dequantize. Checked before the ``quant_method == "none"``
        # early-return because ModelOpt may name its scheme only via
        # ``quant_algo``/``quant_cfg`` (no ``quant_method``). Fail loudly: the
        # reconstruction math is available in :mod:`mobius.integrations.modelopt`,
        # but the full weight-load integration + native routed-expert NVFP4 QMoE
        # emission (CUDA/Blackwell, ``onnxruntime_USE_FP4_QMOE=ON``) are not wired.
        from mobius.integrations.modelopt import is_modelopt_quant_config

        if is_modelopt_quant_config(qc):
            raise NotImplementedError(
                "NVIDIA ModelOpt NVFP4/FP8 checkpoints are not yet fully "
                "supported for export. The weight-reconstruction core lives in "
                "mobius.integrations.modelopt (dequantize_nvfp4 / dequantize_fp8); "
                "remaining work is checkpoint weight-loading and native "
                "routed-expert NVFP4 QMoE emission (CUDA/Blackwell, "
                "onnxruntime_USE_FP4_QMOE=ON). Export the unquantized (bf16) "
                "checkpoint instead, or quantize the bf16 export via Olive."
            )
        # compressed-tensors is a container format, not an INT4 method. Its
        # ``bits``/``group_size`` values live inside ordered config groups and
        # may describe FP8 and NVFP4 simultaneously. Parsing it as this coarse
        # config would emit integer MatMulNBits nodes for FP4 E2M1 bytes.
        from mobius.integrations.compressed_tensors import (
            CompressedTensorsConfig,
            is_compressed_tensors_config,
        )

        if is_compressed_tensors_config(qc):
            CompressedTensorsConfig.parse(qc)
            return None
        # Block-scaled fp8 (E4M3 weight + 2D UE8M0 block scale) and packed-fp4
        # routed experts (I8-packed E2M1 nibbles + UE8M0 micro-scale) are a
        # mixed-precision layout this INT4/per-tensor path cannot load — the
        # packed [out, in/2] fp4 expert vs its logical [out, in] initializer
        # produces a confusing "Weight shape mismatch". Detect it by property
        # (not model name, not the ``quant_method`` string) and fail closed with
        # a typed, actionable blocker naming the real layout + the runtime ABI
        # gap. Checked before the ``quant_method == "none"`` early-return because
        # a checkpoint can advertise fp4 experts via top-level ``expert_dtype``
        # while leaving ``quant_method`` unset.
        from mobius.integrations._block_quant import BlockQuantScheme

        scheme = BlockQuantScheme.from_quantization_config(
            qc,
            expert_dtype=expert_dtype,
        )
        if scheme is not None:
            from mobius.integrations._block_quant import BlockQuantExportError

            raise BlockQuantExportError(
                "Block-scaled FP8 / packed-FP4 checkpoint is not loadable by the "
                "INT4/per-tensor quantization path. Detected "
                f"quant_method={scheme.quant_method!r}, "
                f"weight_block_size={list(scheme.weight_block_size) or None}, "
                f"expert_dtype={scheme.expert_dtype!r}: block-FP8 projections "
                "(E4M3 weight + 2D UE8M0 block scale) and/or FP4-packed routed "
                "experts (I8-packed E2M1 nibbles + UE8M0 micro-scale). Parse and "
                "validate these by property with mobius.integrations._block_quant "
                "(BlockQuantScheme / classify_tensor / QuantizedTensorDescriptor); "
                "the routed-expert emission gate (plan_routed_expert_bank) reports "
                "the exact onnx-genai nxrt ABI gap. Native export is blocked until "
                "the runtime gains a block-FP8 / planar-FP4 BlockFormat."
            )
        if method == "none":
            return None
        # Per-tensor fp8 (float8_e4m3fn + a scalar scale) is handled by dtype
        # casting in _assign_weight(), so it returns None here. (Block-scaled
        # fp8 was already routed to the typed blocker above.)
        if method == "fp8":
            return None
        if method == "mxfp4":
            return cls(
                bits=4,
                group_size=32,
                quant_method="mxfp4",
                sym=True,
                weight_format=QuantizedWeightFormat.MXFP4,
            )
        raw_modules_to_not_convert = qc.get("modules_to_not_convert")
        if raw_modules_to_not_convert is not None and not isinstance(
            raw_modules_to_not_convert, (list, tuple)
        ):
            raise TypeError(
                "quantization_config.modules_to_not_convert must be a list or tuple"
            )
        raw_overrides = qc.get("overrides") or {}
        if not isinstance(raw_overrides, Mapping):
            raise TypeError("quantization_config.overrides must be a mapping")
        return cls(
            bits=qc.get("bits", 4),
            group_size=qc.get("group_size", 128),
            quant_method=method,
            sym=qc.get("sym", qc.get("symmetric", True)),
            float_zero_point=bool(qc.get("float_zero_point")),
            quantize_embeddings=bool(qc.get("embeds")),
            quantize_lm_head=bool(qc.get("lm_head")),
            quantize_vision=bool(qc.get("quantize_vision")),
            tie_word_embeddings=bool(qc.get("tie_word_embeddings")),
            modules_to_not_convert=(
                tuple(str(name) for name in raw_modules_to_not_convert)
                if raw_modules_to_not_convert is not None
                else None
            ),
            overrides={
                str(name): QuantizationOverride.from_value(override)
                for name, override in raw_overrides.items()
            },
        )

    @classmethod
    def from_transformers(cls, hf_config) -> QuantizationConfig | None:
        """Parse ``quantization_config`` from a HuggingFace config.

        Returns ``None`` when no quantization is configured.
        """
        return cls.from_value(
            getattr(hf_config, "quantization_config", None),
            expert_dtype=getattr(hf_config, "expert_dtype", None),
        )

    @property
    def has_module_plan(self) -> bool:
        """Whether Olive recorded component-selection metadata."""
        return self.modules_to_not_convert is not None or bool(self.overrides)

    def for_source_paths(
        self,
        source_paths: tuple[str, ...],
        *,
        component: str,
    ) -> QuantizationConfig | None:
        """Collapse a uniform Olive module plan into one component config.

        A component-level ONNX graph cannot represent different quantization
        layouts for individual projections without constructing each projection
        separately. Uniform overrides are therefore accepted, fully excluded
        components stay float, and mixed layouts fail loudly.
        """
        if not source_paths:
            raise ValueError(
                f"Cannot derive component quantization for {component!r}: "
                "the model declares no HuggingFace source paths."
            )

        def targets_component(name: str) -> bool:
            return any(
                name == prefix
                or name.startswith(f"{prefix}.")
                or prefix.startswith(f"{name}.")
                for prefix in source_paths
            )

        def covers_source_path(name: str, source_path: str) -> bool:
            return name == source_path or source_path.startswith(f"{name}.")

        regex_exclusions = [
            name for name in self.modules_to_not_convert or () if name.startswith("re:")
        ]
        regex_overrides = [name for name in self.overrides if name.startswith("re:")]
        if regex_exclusions or regex_overrides:
            raise ValueError(
                f"Cannot derive component quantization for {component!r} from "
                "regex module rules. Store an explicit component_quantization "
                "mapping in the checkpoint config instead."
            )

        exclusions = [
            name for name in self.modules_to_not_convert or () if targets_component(name)
        ]
        component_overrides = [
            (name, override)
            for name, override in self.overrides.items()
            if targets_component(name)
        ]
        if exclusions and component_overrides:
            raise ValueError(
                f"Component {component!r} mixes excluded and quantized modules; "
                "Mobius requires one quantization configuration per component."
            )
        if exclusions:
            fully_excluded = all(
                any(covers_source_path(name, source_path) for name in exclusions)
                for source_path in source_paths
            )
            if fully_excluded:
                return None
            raise ValueError(
                f"Component {component!r} has partial module exclusions; "
                "Mobius requires one quantization configuration per component."
            )

        base_layout = (self.bits, self.group_size, self.sym)
        effective = [
            (
                override.bits if override.bits is not None else self.bits,
                (override.group_size if override.group_size is not None else self.group_size),
                override.sym if override.sym is not None else self.sym,
            )
            for _, override in component_overrides
        ]
        different_layouts = {layout for layout in effective if layout != base_layout}
        if len(different_layouts) > 1:
            raise ValueError(
                f"Component {component!r} has multiple quantization layouts "
                f"{sorted(different_layouts | {base_layout})!r}; "
                "Mobius requires one per component."
            )

        config = self
        if different_layouts:
            target_layout = next(iter(different_layouts))
            fully_overridden = all(
                any(
                    covers_source_path(name, source_path) and effective[index] == target_layout
                    for index, (name, _) in enumerate(component_overrides)
                )
                for source_path in source_paths
            )
            if not fully_overridden:
                raise ValueError(
                    f"Component {component!r} mixes the default layout "
                    f"{base_layout!r} with override layout {target_layout!r}; "
                    "store an explicit component_quantization mapping instead."
                )
            override = next(
                override
                for index, (_, override) in enumerate(component_overrides)
                if effective[index] == target_layout
            )
            config = override.apply(self)
        return dataclasses.replace(
            config,
            modules_to_not_convert=None,
            overrides={},
        )

    def for_components(
        self,
        component_sources: Mapping[str, tuple[str, ...]],
    ) -> dict[str, QuantizationConfig]:
        """Collapse an Olive module plan into package-component layouts."""
        result: dict[str, QuantizationConfig] = {}
        for component, source_paths in component_sources.items():
            if not source_paths:
                continue
            effective_paths = source_paths
            if component in {"decoder", "model"}:
                # LM-head and token-table selection already have dedicated
                # QuantizationConfig flags. Their exclusions must not turn an
                # otherwise quantized decoder backbone into a float component.
                effective_paths = tuple(
                    path
                    for path in source_paths
                    if not path.endswith(("lm_head", "embed_tokens"))
                )
            if not effective_paths:
                effective_paths = source_paths
            quantization = self.for_source_paths(
                effective_paths,
                component=component,
            )
            if quantization is not None:
                result[component] = quantization
        return result
