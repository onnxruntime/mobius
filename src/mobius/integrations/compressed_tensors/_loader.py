# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Storage-preserving loading for the compressed-tensors 0.17.2 mixed format.

The default route emits Microsoft's block-quantized FP8/NVFP4 weight MatMuls
around the exact source payloads. The NVFP4 op carries an inlineable standard
ONNX Function for portable weight-only fallback. Dense reconstruction remains
available only as an explicit fallback.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from collections.abc import Callable, Mapping

import onnx_ir as ir
import torch
from onnx_ir import tensor_adapters
from safetensors import safe_open

from mobius._model_package import ModelPackage
from mobius._optimizations import fold_initializers_after_weights
from mobius.integrations._weight_loading import _resolve_shard_paths, _shard_key_index

logger = logging.getLogger(__name__)

_SUPPORTED_VERSION_PREFIX = "0.17.2"
_FP8_DTYPES = frozenset(
    {
        "F8_E4M3",
        "F8_E4M3FNUZ",
        "F8_E5M2",
        "F8_E5M2FNUZ",
    }
)
_CONFIG_METADATA = "mobius.compressed_tensors.config"
_STORAGE_METADATA = "mobius.compressed_tensors.storage"
_RUNTIME_METADATA = "mobius.compressed_tensors.runtime_support"
_SOURCE_METADATA = "mobius.compressed_tensors.source"
_ROLE_METADATA = "mobius.compressed_tensors.role"
_AUX_SUFFIXES = (
    ".weight_scale",
    ".weight_global_scale",
    ".input_global_scale",
    ".weight_zero_point",
    ".k_scale",
    ".v_scale",
)
_FP4_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


class CompressedTensorsError(ValueError):
    """A compressed-tensors checkpoint cannot be represented faithfully."""


@dataclasses.dataclass(frozen=True)
class QuantizationArgs:
    """Validated subset of compressed-tensors quantization arguments."""

    num_bits: int
    type: str
    strategy: str
    symmetric: bool
    group_size: int | None
    dynamic: bool | str
    scale_dtype: str | None

    @classmethod
    def parse(cls, value: object, *, where: str) -> QuantizationArgs:
        if not isinstance(value, Mapping):
            raise CompressedTensorsError(f"{where} must be an object.")
        required = ("num_bits", "type", "strategy", "symmetric")
        missing = [field for field in required if field not in value]
        if missing:
            raise CompressedTensorsError(f"Invalid {where}: missing {missing}.")
        num_bits = value["num_bits"]
        type_name = value["type"]
        strategy = value["strategy"]
        symmetric = value["symmetric"]
        group_size = value.get("group_size")
        dynamic = value.get("dynamic", False)
        scale_dtype = value.get("scale_dtype")
        if not isinstance(num_bits, int) or isinstance(num_bits, bool):
            raise CompressedTensorsError(f"Invalid {where}: num_bits must be an integer.")
        if not isinstance(type_name, str) or not isinstance(strategy, str):
            raise CompressedTensorsError(
                f"Invalid {where}: type and strategy must be strings."
            )
        if not isinstance(symmetric, bool):
            raise CompressedTensorsError(f"Invalid {where}: symmetric must be boolean.")
        if group_size is not None and (
            not isinstance(group_size, int) or isinstance(group_size, bool)
        ):
            raise CompressedTensorsError(
                f"Invalid {where}: group_size must be an integer or null."
            )
        if not isinstance(dynamic, (bool, str)):
            raise CompressedTensorsError(
                f"Invalid {where}: dynamic must be boolean or a string."
            )
        if scale_dtype is not None and not isinstance(scale_dtype, str):
            raise CompressedTensorsError(
                f"Invalid {where}: scale_dtype must be a string or null."
            )
        return cls(
            num_bits=num_bits,
            type=type_name,
            strategy=strategy,
            symmetric=symmetric,
            group_size=group_size,
            dynamic=dynamic,
            scale_dtype=scale_dtype,
        )


@dataclasses.dataclass(frozen=True)
class CompressionGroup:
    """One ordered compressed-tensors target group."""

    name: str
    format: str
    targets: tuple[str, ...]
    weights: QuantizationArgs
    input_activations: QuantizationArgs


@dataclasses.dataclass(frozen=True)
class CompressedTensorsConfig:
    """Validated compressed-tensors 0.17.2 mixed FP8/NVFP4 configuration."""

    version: str
    groups: tuple[CompressionGroup, ...]
    ignore: tuple[str, ...]
    kv_cache_scheme: QuantizationArgs | None

    @classmethod
    def from_hf_config(cls, hf_config: object) -> CompressedTensorsConfig | None:
        value = _normalize_config_value(getattr(hf_config, "quantization_config", None))
        if not is_compressed_tensors_config(value):
            return None
        return cls.parse(value)

    @classmethod
    def parse(cls, value: object) -> CompressedTensorsConfig:
        value = _normalize_config_value(value)
        if not isinstance(value, Mapping):
            raise CompressedTensorsError("quantization_config must be an object.")
        if str(value.get("quant_method", "")).lower() != "compressed-tensors":
            raise CompressedTensorsError("quant_method must be 'compressed-tensors'.")
        version = str(value.get("version", ""))
        if not (
            version == _SUPPORTED_VERSION_PREFIX
            or version.startswith(f"{_SUPPORTED_VERSION_PREFIX}.")
        ):
            raise CompressedTensorsError(
                "Only compressed-tensors 0.17.2 checkpoint metadata is supported; "
                f"got version {version!r}."
            )
        if value.get("format") != "mixed-precision":
            raise CompressedTensorsError(
                "Only compressed-tensors format='mixed-precision' is supported; "
                f"got {value.get('format')!r}."
            )
        if value.get("quantization_status") != "compressed":
            raise CompressedTensorsError(
                "compressed-tensors quantization_status must be 'compressed'."
            )

        raw_groups = value.get("config_groups")
        if not isinstance(raw_groups, Mapping) or not raw_groups:
            raise CompressedTensorsError("config_groups must be a non-empty object.")
        groups = tuple(_parse_group(str(name), group) for name, group in raw_groups.items())
        formats = {group.format for group in groups}
        required_formats = {"float-quantized", "nvfp4-pack-quantized"}
        if formats != required_formats:
            raise CompressedTensorsError(
                "The supported mixed format requires exactly float-quantized and "
                f"nvfp4-pack-quantized groups; got {sorted(formats)}."
            )

        ignore = value.get("ignore", [])
        if not isinstance(ignore, list) or not all(isinstance(item, str) for item in ignore):
            raise CompressedTensorsError("ignore must be a list of module-name patterns.")
        _validate_patterns(
            (*ignore, *(target for group in groups for target in group.targets))
        )

        raw_kv = value.get("kv_cache_scheme")
        kv_cache = (
            QuantizationArgs.parse(raw_kv, where="kv_cache_scheme")
            if raw_kv is not None
            else None
        )
        if kv_cache is not None and (
            kv_cache.num_bits != 8
            or kv_cache.type != "float"
            or kv_cache.strategy != "tensor"
            or not kv_cache.symmetric
            or kv_cache.dynamic is not False
        ):
            raise CompressedTensorsError(
                "Only static symmetric per-tensor FP8 KV-cache metadata is supported."
            )
        return cls(
            version=version,
            groups=groups,
            ignore=tuple(ignore),
            kv_cache_scheme=kv_cache,
        )

    def resolve(self, module_name: str) -> CompressionGroup | None:
        """Resolve a module using compressed-tensors' exact-before-regex precedence."""
        if any(_match_name(module_name, pattern) for pattern in self.ignore):
            return None
        # compressed-tensors first builds an ordered target map. Assigning a
        # duplicate target replaces its scheme while retaining the target's
        # position, so the last config group owning that exact target wins.
        target_groups: dict[str, CompressionGroup] = {}
        for group in self.groups:
            for target in group.targets:
                target_groups[target] = group
        if exact_group := target_groups.get(module_name):
            return exact_group
        for target, group in target_groups.items():
            if target.startswith("re:") and _match_name(module_name, target):
                return group
        return None

    def metadata(self) -> dict[str, object]:
        """Return the ordered recipe persisted into each ONNX component."""
        return {
            "version": self.version,
            "groups": [
                {
                    "name": group.name,
                    "format": group.format,
                    "targets": list(group.targets),
                    "weights": dataclasses.asdict(group.weights),
                    "input_activations": dataclasses.asdict(group.input_activations),
                }
                for group in self.groups
            ],
            "ignore": list(self.ignore),
            "kv_cache_scheme": (
                dataclasses.asdict(self.kv_cache_scheme)
                if self.kv_cache_scheme is not None
                else None
            ),
        }


@dataclasses.dataclass(frozen=True)
class CompressedTensorsLoadReport:
    """Storage and runtime facts for a completed checkpoint load.

    ``preserved_weight_formats`` describes serialized source storage, while
    ``native_weight_formats`` is reserved for compute kernels proven available
    in the active runtime. A preserved W4A16 custom-op graph therefore does not
    claim W4A4 activation execution or universal native runtime support.
    """

    storage_policy: str
    preserved_weight_formats: tuple[str, ...]
    native_weight_formats: tuple[str, ...]
    dequantized_weight_formats: tuple[str, ...]
    activation_quantization: str
    runtime_support: str
    kv_cache: str
    output_is_nvfp4: bool
    assigned_initializers: int


def is_compressed_tensors_config(value: object) -> bool:
    """Return whether *value* declares the compressed-tensors method."""
    value = _normalize_config_value(value)
    return isinstance(value, Mapping) and (
        str(value.get("quant_method", "")).lower() == "compressed-tensors"
    )


def _normalize_config_value(value: object) -> object:
    """Convert HuggingFace mapping, attribute, and ``to_dict`` configs to plain data."""
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
    if isinstance(value, Mapping):
        return {key: _normalize_config_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_config_value(item) for item in value]
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, Mapping):
        return {
            key: _normalize_config_value(item)
            for key, item in attributes.items()
            if not key.startswith("_")
        }
    return value


def _parse_group(name: str, value: object) -> CompressionGroup:
    if not isinstance(value, Mapping):
        raise CompressedTensorsError(f"config_groups.{name} must be an object.")
    format_name = str(value.get("format", ""))
    targets = value.get("targets")
    if (
        not isinstance(targets, list)
        or not targets
        or not all(isinstance(target, str) for target in targets)
    ):
        raise CompressedTensorsError(f"config_groups.{name}.targets must be strings.")
    if value.get("output_activations") is not None:
        raise CompressedTensorsError(
            f"config_groups.{name}.output_activations must be null; output "
            "activation quantization is not represented by the dense fallback."
        )
    weights = QuantizationArgs.parse(
        value.get("weights"), where=f"config_groups.{name}.weights"
    )
    inputs = QuantizationArgs.parse(
        value.get("input_activations"),
        where=f"config_groups.{name}.input_activations",
    )
    if format_name == "float-quantized":
        valid = (
            weights.num_bits == 8
            and weights.type == "float"
            and weights.strategy == "channel"
            and weights.symmetric
            and weights.dynamic is False
            and inputs.num_bits == 8
            and inputs.type == "float"
            and inputs.strategy == "token"
            and inputs.symmetric
            and inputs.dynamic is True
        )
    elif format_name == "nvfp4-pack-quantized":
        valid = (
            weights.num_bits == 4
            and weights.type == "float"
            and weights.strategy == "tensor_group"
            and weights.group_size == 16
            and weights.symmetric
            and weights.dynamic is False
            and weights.scale_dtype == "torch.float8_e4m3fn"
            and inputs.num_bits == 4
            and inputs.type == "float"
            and inputs.strategy == "tensor_group"
            and inputs.group_size == 16
            and inputs.symmetric
            and inputs.dynamic == "local"
            and inputs.scale_dtype == "torch.float8_e4m3fn"
        )
    else:
        raise CompressedTensorsError(
            f"Unsupported compressed-tensors group format {format_name!r}."
        )
    if not valid:
        raise CompressedTensorsError(
            f"config_groups.{name} does not match the supported {format_name} ABI."
        )
    return CompressionGroup(
        name=name,
        format=format_name,
        targets=tuple(targets),
        weights=weights,
        input_activations=inputs,
    )


def _validate_patterns(patterns: tuple[str, ...]) -> None:
    for pattern in patterns:
        if pattern.startswith("re:"):
            try:
                re.compile(pattern[3:])
            except re.error as error:
                raise CompressedTensorsError(
                    f"Invalid compressed-tensors regex {pattern!r}: {error}"
                ) from error


def _match_name(name: str, target: str) -> bool:
    if target.startswith("re:"):
        return re.match(target[3:], name) is not None
    return name == target


def _load_tensors(
    key_index: Mapping[str, tuple[str, list[int], str]], keys: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    keys_by_path: dict[str, list[str]] = {}
    for key in keys:
        try:
            path = key_index[key][0]
        except KeyError as error:
            raise CompressedTensorsError(
                f"Missing required checkpoint tensor {key!r}."
            ) from error
        keys_by_path.setdefault(path, []).append(key)

    tensors: dict[str, torch.Tensor] = {}
    for path, path_keys in keys_by_path.items():
        with safe_open(path, framework="pt", device="cpu") as handle:
            tensors.update((key, handle.get_tensor(key)) for key in path_keys)
    return tensors


def _dequantize_fp8(
    key_index: Mapping[str, tuple[str, list[int], str]],
    module_name: str,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    weight_key = f"{module_name}.weight"
    scale_key = f"{module_name}.weight_scale"
    tensors = _load_tensors(key_index, (weight_key, scale_key))
    weight = tensors[weight_key]
    scale = tensors[scale_key]
    return (weight.float() * scale.float()).to(target_dtype)


def _dequantize_nvfp4(
    key_index: Mapping[str, tuple[str, list[int], str]],
    module_name: str,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    packed_key = f"{module_name}.weight_packed"
    scale_key = f"{module_name}.weight_scale"
    global_scale_key = f"{module_name}.weight_global_scale"
    tensors = _load_tensors(key_index, (packed_key, scale_key, global_scale_key))
    packed = tensors[packed_key]
    scale = tensors[scale_key]
    global_scale = tensors[global_scale_key]
    if packed.dtype != torch.uint8 or packed.ndim != 2:
        raise CompressedTensorsError(
            f"{module_name}.weight_packed must be uint8 [N, K/2], got "
            f"{packed.dtype} {tuple(packed.shape)}."
        )
    low = packed & 0x0F
    high = packed >> 4
    codes = torch.empty(
        packed.shape[0], packed.shape[1] * 2, dtype=torch.uint8, device=packed.device
    )
    codes[:, 0::2] = low
    codes[:, 1::2] = high
    values = _FP4_VALUES[(codes & 0x7).long()]
    values = torch.where((codes & 0x8) != 0, -values, values)
    values = values.unflatten(1, (-1, 16))
    divisor = global_scale.float().reshape(-1)
    if divisor.numel() != 1 or not torch.isfinite(divisor).all() or divisor.item() == 0:
        raise CompressedTensorsError(
            f"{module_name}.weight_global_scale must be one finite non-zero divisor."
        )
    result = values * scale.float().unsqueeze(-1) / divisor.item()
    return result.flatten(1).to(target_dtype)


def _validate_module_layout(
    key_index: Mapping[str, tuple[str, list[int], str]],
    module_name: str,
    group: CompressionGroup,
) -> tuple[list[int], str]:
    if group.format == "float-quantized":
        weight_key = f"{module_name}.weight"
        scale_key = f"{module_name}.weight_scale"
        _require_metadata(key_index, weight_key, dtypes=_FP8_DTYPES, rank=2)
        weight_shape = key_index[weight_key][1]
        _require_metadata(key_index, scale_key, dtypes={"BF16", "F16", "F32"}, rank=2)
        if key_index[scale_key][1] != [weight_shape[0], 1]:
            raise CompressedTensorsError(
                f"{scale_key} must have channel shape [{weight_shape[0]}, 1], got "
                f"{key_index[scale_key][1]}."
            )
        return list(weight_shape), weight_key

    packed_key = f"{module_name}.weight_packed"
    scale_key = f"{module_name}.weight_scale"
    global_key = f"{module_name}.weight_global_scale"
    _require_metadata(key_index, packed_key, dtypes={"U8"}, rank=2)
    packed_shape = key_index[packed_key][1]
    logical_shape = [packed_shape[0], packed_shape[1] * 2]
    if logical_shape[1] % 16:
        raise CompressedTensorsError(
            f"{packed_key} logical K={logical_shape[1]} is not divisible by 16."
        )
    _require_metadata(key_index, scale_key, dtypes={"F8_E4M3"}, rank=2)
    expected_scale = [logical_shape[0], logical_shape[1] // 16]
    if key_index[scale_key][1] != expected_scale:
        raise CompressedTensorsError(
            f"{scale_key} must have block-16 shape {expected_scale}, got "
            f"{key_index[scale_key][1]}."
        )
    _require_metadata(key_index, global_key, dtypes={"F32"}, rank=1)
    if key_index[global_key][1] != [1]:
        raise CompressedTensorsError(f"{global_key} must be an F32 scalar stored as [1].")
    _require_metadata(
        key_index,
        f"{module_name}.input_global_scale",
        dtypes={"F32"},
        rank=1,
    )
    if key_index[f"{module_name}.input_global_scale"][1] != [1]:
        raise CompressedTensorsError(
            f"{module_name}.input_global_scale must be an F32 scalar stored as [1]."
        )
    return logical_shape, packed_key


def _require_metadata(
    key_index: Mapping[str, tuple[str, list[int], str]],
    key: str,
    *,
    dtypes: set[str] | frozenset[str],
    rank: int,
) -> None:
    try:
        _path, shape, dtype = key_index[key]
    except KeyError as error:
        raise CompressedTensorsError(f"Missing required checkpoint tensor {key!r}.") from error
    if dtype not in dtypes or len(shape) != rank:
        raise CompressedTensorsError(
            f"{key} must have dtype in {sorted(dtypes)} and rank {rank}, got {dtype} {shape}."
        )


def _torch_dtype(dtype: str) -> torch.dtype:
    mapping = {
        "BOOL": torch.bool,
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        "F32": torch.float32,
        "F64": torch.float64,
        "F8_E4M3": torch.float8_e4m3fn,
        "F8_E4M3FNUZ": torch.float8_e4m3fnuz,
        "F8_E5M2": torch.float8_e5m2,
        "F8_E5M2FNUZ": torch.float8_e5m2fnuz,
        "I8": torch.int8,
        "I16": torch.int16,
        "I32": torch.int32,
        "I64": torch.int64,
        "U8": torch.uint8,
    }
    try:
        return mapping[dtype]
    except KeyError as error:
        raise CompressedTensorsError(f"Unsupported safetensors dtype {dtype!r}.") from error


def _ir_dtype(dtype: str) -> ir.DataType:
    mapping = {
        "BF16": ir.DataType.BFLOAT16,
        "F16": ir.DataType.FLOAT16,
        "F32": ir.DataType.FLOAT,
        "F8_E4M3": ir.DataType.FLOAT8E4M3FN,
        "F8_E5M2": ir.DataType.FLOAT8E5M2,
        "I64": ir.DataType.INT64,
        "U8": ir.DataType.UINT8,
    }
    try:
        return mapping[dtype]
    except KeyError as error:
        raise CompressedTensorsError(
            f"Unsupported preserved safetensors dtype {dtype!r}."
        ) from error


def _map_source_weight(
    preprocess_weights: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None,
    source_name: str,
    shape: list[int],
    dtype: torch.dtype,
) -> tuple[str, ...]:
    sentinel = torch.empty(tuple(shape), dtype=dtype, device="meta")
    if preprocess_weights is None:
        return (source_name,)
    mapped = preprocess_weights({source_name: sentinel})
    for target_name, value in mapped.items():
        if value is not sentinel:
            raise CompressedTensorsError(
                "Streaming compressed-tensors loading only supports name-only weight "
                f"preprocessing; {source_name!r} was transformed for {target_name!r}."
            )
    return tuple(mapped)


def _find_initializer(
    package: ModelPackage, mapped_name: str
) -> tuple[str, ir.Graph, ir.Value] | None:
    for component_name, model in package.items():
        if mapped_name in model.graph.initializers:
            return (
                f"{component_name}.{mapped_name}",
                model.graph,
                model.graph.initializers[mapped_name],
            )
        prefix = f"{component_name}."
        if mapped_name.startswith(prefix):
            local_name = mapped_name[len(prefix) :]
            if local_name in model.graph.initializers:
                return mapped_name, model.graph, model.graph.initializers[local_name]
    return None


def _lazy_source_initializer(
    graph: ir.Graph,
    key_index: Mapping[str, tuple[str, list[int], str]],
    source_key: str,
    name: str,
    role: str,
    *,
    dtype: ir.DataType | None = None,
    shape: list[int] | None = None,
    raw_bytes: bool = False,
) -> ir.Value:
    _path, source_shape, source_dtype = key_index[source_key]
    value_dtype = dtype or _ir_dtype(source_dtype)
    value_shape = shape or source_shape
    metadata = {_SOURCE_METADATA: source_key, _ROLE_METADATA: role}

    def tensor_func() -> ir.TensorProtocol:
        tensor = _load_tensors(key_index, (source_key,))[source_key]
        if raw_bytes:
            tensor = tensor.view(torch.uint8)
        return tensor_adapters.TorchTensor(tensor, name=name)

    value = ir.Value(
        name=name,
        shape=ir.Shape(value_shape),
        type=ir.TensorType(value_dtype),
        const_value=ir.LazyTensor(
            tensor_func,
            dtype=value_dtype,
            shape=ir.Shape(value_shape),
            name=name,
            metadata_props=metadata,
        ),
        metadata_props=metadata,
    )
    graph.register_initializer(value)
    return value


def _projection_sites(
    graph: ir.Graph,
    initializer: ir.Value,
) -> list[tuple[ir.Node, ir.Node]]:
    """Return the exact ``Transpose(weight) -> MatMul`` sites for a Linear."""
    sites: list[tuple[ir.Node, ir.Node]] = []
    for transpose, input_index in initializer.uses():
        if (
            input_index != 0
            or transpose.domain != ""
            or transpose.op_type != "Transpose"
            or tuple(transpose.attributes.get_ints("perm", ())) != (1, 0)
        ):
            raise CompressedTensorsError(
                f"Quantized weight {initializer.name!r} must feed only a 2-D "
                "Transpose(perm=[1, 0]) before MatMul."
            )
        transposed = transpose.outputs[0]
        uses = list(transposed.uses())
        if not uses:
            raise CompressedTensorsError(
                f"Quantized weight transpose {transpose.name!r} has no MatMul consumer."
            )
        for matmul, matmul_input_index in uses:
            if matmul_input_index != 1 or matmul.domain != "" or matmul.op_type != "MatMul":
                raise CompressedTensorsError(
                    f"Quantized weight transpose {transpose.name!r} must feed only "
                    "the B input of MatMul."
                )
            sites.append((transpose, matmul))
    if not sites:
        raise CompressedTensorsError(
            f"Quantized initializer {initializer.name!r} has no Linear projection."
        )
    return sites


def _replace_initializer_with_native_projection(
    graph: ir.Graph,
    initializer: ir.Value,
    key_index: Mapping[str, tuple[str, list[int], str]],
    module_name: str,
    group: CompressionGroup,
    logical_shape: list[int],
) -> None:
    """Replace a dense Linear projection with the pinned Microsoft block op."""
    sites = _projection_sites(graph, initializer)
    assert initializer.name is not None
    prefix = f"{initializer.name}.compressed_tensors"

    if group.format == "nvfp4-pack-quantized":
        weight = _lazy_source_initializer(
            graph,
            key_index,
            f"{module_name}.weight_packed",
            f"{prefix}.weight",
            "weight_packed",
        )
        scale = _lazy_source_initializer(
            graph,
            key_index,
            f"{module_name}.weight_scale",
            f"{prefix}.weight_scale",
            "weight_scale",
            dtype=ir.DataType.UINT8,
            raw_bytes=True,
        )
        global_scale = _lazy_source_initializer(
            graph,
            key_index,
            f"{module_name}.weight_global_scale",
            f"{prefix}.weight_global_scale",
            "weight_global_scale",
        )
        earliest_transpose = min(
            (transpose for transpose, _ in sites),
            key=graph.index,
        )
        one = ir.Value(
            name=f"{prefix}.one",
            shape=ir.Shape([1]),
            type=ir.TensorType(ir.DataType.FLOAT),
            const_value=ir.tensor(
                torch.tensor([1.0], dtype=torch.float32),
                name=f"{prefix}.one",
            ),
        )
        graph.register_initializer(one)
        weight_scale_2 = ir.Value(
            name=f"{prefix}.weight_scale_2",
            shape=ir.Shape([1]),
            type=ir.TensorType(ir.DataType.FLOAT),
        )
        graph.insert_before(
            earliest_transpose,
            ir.Node(
                "",
                "Div",
                inputs=[one, global_scale],
                outputs=[weight_scale_2],
                name=f"{prefix}.ReciprocalGlobalScale",
            ),
        )
        op_type = "MatMulBlockQuantizedFp4Weight"
        raw_inputs = [weight, scale, weight_scale_2]
        attributes = {"block_size": 16}
    else:
        source_dtype = key_index[f"{module_name}.weight"][2]
        if source_dtype != "F8_E4M3":
            raise CompressedTensorsError(
                "MatMulBlockQuantizedFp8Weight requires an exact "
                "FLOAT8E4M3FN weight payload, but "
                f"{module_name}.weight uses {source_dtype}. Set "
                "keep_quantized=False (--dequantize) for dense reconstruction."
            )
        weight = _lazy_source_initializer(
            graph,
            key_index,
            f"{module_name}.weight",
            f"{prefix}.weight",
            "weight",
        )
        scale = _lazy_source_initializer(
            graph,
            key_index,
            f"{module_name}.weight_scale",
            f"{prefix}.weight_scale",
            "weight_scale",
        )
        if scale.dtype == ir.DataType.FLOAT:
            native_scale = scale
        else:
            native_scale = ir.Value(
                name=f"{prefix}.weight_scale_float",
                shape=scale.shape,
                type=ir.TensorType(ir.DataType.FLOAT),
            )
            earliest_transpose = min(
                (transpose for transpose, _ in sites),
                key=graph.index,
            )
            graph.insert_before(
                earliest_transpose,
                ir.Node(
                    "",
                    "Cast",
                    inputs=[scale],
                    outputs=[native_scale],
                    attributes=ir.convenience.convert_attributes({"to": ir.DataType.FLOAT}),
                    name=f"{prefix}.CastWeightScale",
                ),
            )
        op_type = "MatMulBlockQuantizedFp8Weight"
        raw_inputs = [weight, native_scale]
        attributes = {"block_size": logical_shape[1]}

    graph.opset_imports["com.microsoft"] = 1
    removed_transposes: set[ir.Node] = set()
    for index, (transpose, matmul) in enumerate(sites):
        activation = matmul.inputs[0]
        matmul_output = matmul.outputs[0]
        assert activation is not None
        if activation.dtype != ir.DataType.FLOAT16:
            raise CompressedTensorsError(
                f"{op_type} requires FLOAT16 activations, but projection "
                f"{matmul.name!r} has {activation.dtype}. Build with dtype='f16' "
                "or set keep_quantized=False."
            )
        output = ir.Value(
            name=matmul_output.name or f"{prefix}.output_{index}",
            shape=matmul_output.shape,
            type=matmul_output.type,
        )
        node = ir.Node(
            "com.microsoft",
            op_type,
            inputs=[activation, *raw_inputs],
            outputs=[output],
            attributes=ir.convenience.convert_attributes(attributes),
            name=f"{prefix}.{op_type}_{index}",
        )
        graph.insert_before(transpose, node)
        matmul_output.replace_all_uses_with(output, replace_graph_outputs=True)
        graph.remove(matmul, safe=True)
        removed_transposes.add(transpose)

    for transpose in removed_transposes:
        graph.remove(transpose, safe=True)
    del graph.initializers[initializer.name]


def _has_fp8_kv_cache(package: ModelPackage) -> bool:
    """Return whether the built decoder actually carries the FP8 GQA cache ABI."""
    gqa_nodes = [
        node
        for model in package.values()
        for node in model.graph
        if node.domain == "com.microsoft" and node.op_type == "GroupQueryAttention"
    ]
    if not gqa_nodes:
        return False
    return all(
        node.attributes.get_int("kv_cache_bit_width", 0) == 8
        and node.attributes.get_string("k_quant_type", "") == "PER_TENSOR"
        and node.attributes.get_string("v_quant_type", "") == "PER_TENSOR"
        for node in gqa_nodes
    )


def stream_compressed_tensors_to_package(
    package: ModelPackage,
    model_id: str,
    config: CompressedTensorsConfig,
    *,
    preprocess_weights: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]
    | None = None,
    revision: str | None = None,
    fp8_kv_cache: bool = False,
    keep_quantized: bool = True,
) -> CompressedTensorsLoadReport:
    """Bind a mixed FP8/NVFP4 checkpoint through lazy source tensors.

    With ``keep_quantized=True`` (the default), source codes and scales are
    serialized unchanged and Microsoft's native block-weight MatMul ops consume
    them. The NVFP4 op carries a registered standard-ONNX Function fallback.
    ``keep_quantized=False`` explicitly selects bounded dense reconstruction.
    """
    from mobius.functions import register_function_bodies

    paths = _resolve_shard_paths(model_id, revision)
    key_index = _shard_key_index(paths)

    module_names = {
        key[: -len(suffix)]
        for key in key_index
        for suffix in (
            ".weight",
            ".weight_packed",
            ".weight_scale",
            ".weight_global_scale",
            ".input_global_scale",
        )
        if key.endswith(suffix)
    }
    layouts: dict[str, tuple[CompressionGroup, list[int], str]] = {}
    for module_name in sorted(module_names):
        group = config.resolve(module_name)
        has_compression_params = any(
            f"{module_name}{suffix}" in key_index
            for suffix in (
                ".weight_packed",
                ".weight_scale",
                ".weight_global_scale",
                ".input_global_scale",
            )
        )
        if group is None:
            if has_compression_params:
                raise CompressedTensorsError(
                    f"Orphan quantization parameters for untargeted/ignored module "
                    f"{module_name!r}."
                )
            continue
        if group.format == "nvfp4-pack-quantized":
            logical_shape, root_key = _validate_module_layout(key_index, module_name, group)
            if f"{module_name}.weight" in key_index:
                raise CompressedTensorsError(
                    f"{module_name!r} resolves to NVFP4 but stores an ordinary weight."
                )
        else:
            logical_shape, root_key = _validate_module_layout(key_index, module_name, group)
            if f"{module_name}.weight_packed" in key_index:
                raise CompressedTensorsError(
                    f"{module_name!r} resolves to FP8 but stores weight_packed."
                )
        layouts[module_name] = (group, logical_shape, root_key)

    for key, (_path, _shape, dtype) in key_index.items():
        if not key.endswith(".weight") or dtype not in _FP8_DTYPES:
            continue
        module_name = key[: -len(".weight")]
        if module_name not in layouts:
            raise CompressedTensorsError(
                f"FP8 weight {key!r} is not covered by a float-quantized target."
            )

    root_to_layout = {
        root: (module, group, shape) for module, (group, shape, root) in layouts.items()
    }
    assigned: set[int] = set()
    target_sources: dict[int, str] = {}

    for source_key, (_path, source_shape, source_dtype) in sorted(key_index.items()):
        if source_key.endswith(_AUX_SUFFIXES):
            continue
        layout = root_to_layout.get(source_key)
        if layout is not None:
            module_name, group, logical_shape = layout
            logical_name = f"{module_name}.weight"
            logical_dtype = torch.bfloat16
        else:
            logical_name = source_key
            logical_shape = list(source_shape)
            logical_dtype = _torch_dtype(source_dtype)
            group = None
            module_name = source_key.rsplit(".", 1)[0]

        mapped_names = _map_source_weight(
            preprocess_weights, logical_name, logical_shape, logical_dtype
        )
        for mapped_name in mapped_names:
            located = _find_initializer(package, mapped_name)
            if located is None:
                continue
            qualified_name, graph, initializer = located
            identity = id(initializer)
            prior = target_sources.get(identity)
            if prior is not None and prior != source_key:
                raise CompressedTensorsError(
                    f"Initializer {qualified_name!r} maps from both {prior!r} and "
                    f"{source_key!r}."
                )
            expected = [int(dim) for dim in initializer.shape]
            if expected != logical_shape:
                raise CompressedTensorsError(
                    f"Weight shape mismatch for {qualified_name!r}: graph expects "
                    f"{expected}, checkpoint reconstructs {logical_shape}."
                )
            onnx_dtype = initializer.dtype
            assert onnx_dtype is not None
            if keep_quantized and group is not None:
                _replace_initializer_with_native_projection(
                    graph,
                    initializer,
                    key_index,
                    module_name,
                    group,
                    logical_shape,
                )
            else:
                target_dtype = tensor_adapters.to_torch_dtype(onnx_dtype)

                def tensor_func(
                    *,
                    source=source_key,
                    module=module_name,
                    selected_group=group,
                    output_dtype=target_dtype,
                    output_name=qualified_name,
                ) -> tensor_adapters.TorchTensor:
                    if selected_group is None:
                        tensor = _load_tensors(key_index, (source,))[source]
                        if tensor.dtype != output_dtype:
                            tensor = tensor.to(output_dtype)
                    elif selected_group.format == "float-quantized":
                        tensor = _dequantize_fp8(key_index, module, output_dtype)
                    else:
                        tensor = _dequantize_nvfp4(key_index, module, output_dtype)
                    return tensor_adapters.TorchTensor(tensor, name=output_name)

                initializer.const_value = ir.LazyTensor(
                    tensor_func,
                    dtype=onnx_dtype,
                    shape=ir.Shape(logical_shape),
                    name=qualified_name,
                )
            assigned.add(identity)
            target_sources[identity] = source_key

    missing = [
        f"{component}.{name}"
        for component, model in package.items()
        for name, initializer in model.graph.initializers.items()
        if initializer.const_value is None and id(initializer) not in assigned
    ]
    if missing:
        raise CompressedTensorsError(
            f"{len(missing)} graph initializer(s) have no checkpoint mapping, e.g. "
            f"{missing[:5]}."
        )

    # Fold remaining ordinary float initializers without materializing the full
    # checkpoint. Native block projections no longer have a dense initializer.
    for model in package.values():
        if keep_quantized:
            register_function_bodies(model)
        fold_initializers_after_weights(model)
        model.metadata_props[_CONFIG_METADATA] = json.dumps(
            config.metadata(),
            separators=(",", ":"),
        )
        model.metadata_props[_STORAGE_METADATA] = (
            "preserved-native-block" if keep_quantized else "dense-dequantized"
        )
        model.metadata_props[_RUNTIME_METADATA] = (
            "faithful W4A16/W8A16 block-weight custom-op package; native execution "
            "requires a sufficiently new/custom ONNX Runtime, and the registered "
            "NVFP4 Function is the portable weight-only fallback"
            if keep_quantized
            else "dense floating-point fallback; source quantized compute is not preserved"
        )

    actual_fp8_kv_cache = fp8_kv_cache and _has_fp8_kv_cache(package)
    kv_cache = (
        "FP8 preserved through the explicit EP-gated GQA cache feature"
        if actual_fp8_kv_cache
        else "dequantized/float graph cache; checkpoint FP8 cache scheme not enabled"
    )
    loaded_format_set = {group.format for group, _shape, _root in layouts.values()}
    loaded_formats = tuple(
        dict.fromkeys(
            group.format for group in config.groups if group.format in loaded_format_set
        )
    )
    report = CompressedTensorsLoadReport(
        storage_policy=("preserved-native-block" if keep_quantized else "dense-dequantized"),
        preserved_weight_formats=loaded_formats if keep_quantized else (),
        native_weight_formats=(),
        dequantized_weight_formats=() if keep_quantized else loaded_formats,
        activation_quantization=(
            "recipe metadata preserved; dynamic token/local activation quantization "
            "is not part of the default W4A16/W8A16 graph"
        ),
        runtime_support=(
            "native W4A16/W8A16 custom-op compute requires a sufficiently new/custom "
            "ONNX Runtime; NVFP4 can be inlined to a portable weight-only function"
            if keep_quantized
            else "dense fallback is executable where the original float graph is supported"
        ),
        kv_cache=kv_cache,
        output_is_nvfp4=keep_quantized and "nvfp4-pack-quantized" in loaded_format_set,
        assigned_initializers=len(assigned),
    )
    if keep_quantized:
        logger.warning(
            "Preserved compressed-tensors FP8/NVFP4 storage through native W4A16/W8A16 "
            "block-weight ops. Native execution requires a sufficiently new/custom "
            "ONNX Runtime; dynamic activation quantization is metadata-only. "
            "KV cache: %s.",
            kv_cache,
        )
    else:
        logger.warning(
            "Loaded compressed-tensors weights through explicit bounded dense "
            "reconstruction. The output is no longer NVFP4/FP8 weight-quantized. "
            "KV cache: %s.",
            kv_cache,
        )
    package.quantization_report = report
    return report
