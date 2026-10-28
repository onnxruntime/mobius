# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unqualified, opt-in Drama 3 CUDA 13.0 consumer graphs.

This builder-only adapter is not registered as a task or an EP capability.
It preserves FishOmniModel's ports and learned parameter layout. Callers must
validate shared, contiguous, unpadded dynamic-cache inputs before every call.
Emitted graphs need the separately built SM90 runtime; CPU execution, capture,
source-bit parity and full-model/performance admission are NOT established.

QMoE uses FP8 storage and dynamic activation quantization with register-local
FP16 MMA, NOT native hardware FP8 MMA. Portable ordered reductions and Softplus
remain in the graph: neither native ordered operators nor Log1p are implied.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import onnx_ir as ir
from onnxscript import OpBuilder

from mobius._build_context import ep_capabilities
from mobius._configs.fish_omni import FishOmniConfig
from mobius._model_package import ModelPackage
from mobius.components import chunk_gated_delta_rule
from mobius.generation._fish_omni_context import (
    SOURCE_CONTEXT_CONTRACT,
    SOURCE_CONTEXT_METADATA,
    validate_source_context,
)
from mobius.models.fish_omni import (
    FishOmniAudioDecoder,
    FishOmniModel,
    _FishDecoderLayer,
    _FishFP8Experts,
    _FishSourceAttention,
    _FishSourceGatedDeltaNet,
    _FishSourceTowerAttention,
    _FishTextModel,
)
from mobius.tasks._fish_omni import FishOmniGenerationTask, FishOmniTask

PROFILE = "drama3-cuda13-fp16-compatible-candidate-v1"
PROFILE_METADATA = "mobius.fish_omni.cuda13_candidate"


@dataclasses.dataclass(frozen=True)
class _RightCausalContract:
    """Construction token, never an arbitrary additive/padding mask."""

    attention_mask: ir.Value | None = None


_RIGHT_CAUSAL = _RightCausalContract()


def _tensor_dtype(value: ir.Value | None) -> ir.DataType | None:
    """Read existing type or a small, schema-proven type-preserving producer chain."""
    if value is None:
        return None
    if value.dtype is not None:
        return value.dtype
    node = value.producer()
    if node is None or node.domain:
        return None
    if node.op_type == "Cast":
        return ir.DataType(node.attributes["to"].as_int())
    if node.op_type == "CastLike":
        return _tensor_dtype(node.inputs[1])
    if node.op_type in {
        "Reshape",
        "Transpose",
        "Gemm",
        "Gather",
        "Identity",
        "Mul",
        "Split",
        "Concat",
        "Slice",
    }:
        return _tensor_dtype(node.inputs[0])
    return None


def _require_tensor(value: ir.Value, dtype: ir.DataType, rank: int) -> ir.Shape:
    if value.dtype != dtype or value.shape is None or len(value.shape) != rank:
        raise ValueError(f"Expected {dtype.name} rank-{rank} tensor with known shape")
    return value.shape


def _compatible_dimensions(
    left: ir.Shape | Sequence[object], right: ir.Shape | Sequence[object]
) -> bool:
    # Unknown dimensions need caller validation; contradictory known ones never qualify.
    return len(left) == len(right) and all(
        a == b or _unknown_dimension(a) or _unknown_dimension(b) for a, b in zip(left, right)
    )


def _unknown_dimension(dim: object) -> bool:
    return dim is None or (isinstance(dim, ir.SymbolicDim) and dim.value is None)


def unit_lower_triangular_solve(op: OpBuilder, system: ir.Value, rhs: ir.Value) -> ir.Value:
    """Emit the FLOAT, exact-prefix, no-broadcast provider boundary."""
    if system.shape is None or len(system.shape) < 2:
        raise ValueError("UnitLowerTriangularSolve requires rank >= 2")
    shape = _require_tensor(system, ir.DataType.FLOAT, len(system.shape))
    other = _require_tensor(rhs, ir.DataType.FLOAT, len(shape))
    # Require explicit equal dimensions, not merely potentially broadcastable shapes.
    if (
        any(_unknown_dimension(dim) for dim in [*shape, *other])
        or shape[-1] != shape[-2]
        or list(shape[:-1]) != list(other[:-1])
    ):
        raise ValueError("UnitLowerTriangularSolve requires square A and exact RHS prefix")
    output = op.UnitLowerTriangularSolve(system, rhs, _domain="com.microsoft")
    output.type, output.shape = rhs.type, rhs.shape
    return output


def causal_right_attention(
    op: OpBuilder,
    query: ir.Value,
    key: ir.Value,
    value: ir.Value,
    bias: object,
    past_key: ir.Value | None,
    past_value: ir.Value | None,
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
    scale: float,
    dtype: ir.DataType,
) -> tuple[ir.Value, ir.Value, ir.Value]:
    """Adapt packed Q/new KV and unexpanded rank-4 caches to native attention.

    Only the scoped model's explicit right-causal construction token qualifies.
    The runtime input validator enforces shared positions, no padding, and the
    B/S/T bounds when dimensions are symbolic. No arbitrary mask is discarded.
    """
    if not isinstance(bias, _RightCausalContract):
        raise TypeError("CausalRightAttention requires the explicit unpadded contract")
    if (
        dtype not in (ir.DataType.FLOAT16, ir.DataType.BFLOAT16)
        or not 1 <= num_attention_heads <= 32
        or not 1 <= num_key_value_heads <= num_attention_heads
        or num_attention_heads % num_key_value_heads
        or not math.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("Unsupported CausalRightAttention dtype, heads or scale")
    shapes = [_require_tensor(x, dtype, 3) for x in (query, key, value)]
    for shape, heads in zip(
        shapes, (num_attention_heads, num_key_value_heads, num_key_value_heads)
    ):
        if shape[-1] != heads * 256 or not _compatible_dimensions(shape[:2], shapes[0][:2]):
            raise ValueError("CausalRightAttention requires packed D256 Q/new KV")
    if past_key is None or past_value is None:
        raise ValueError("CausalRightAttention requires explicit dynamic KV caches")
    caches = [_require_tensor(x, dtype, 4) for x in (past_key, past_value)]
    if not _compatible_dimensions(caches[0], caches[1]):
        raise ValueError("CausalRightAttention requires matching KV caches")
    for shape in caches:
        if (
            shape[1] != num_key_value_heads
            or shape[3] != 256
            or not _compatible_dimensions(shape[:1], shapes[0][:1])
        ):
            raise ValueError("CausalRightAttention requires unexpanded [B,KVH,P,256] caches")
    batch, length, past = shapes[0][0], shapes[0][1], caches[0][2]
    if isinstance(batch, int) and not 1 <= batch <= 8:
        raise ValueError("CausalRightAttention requires 1 <= batch <= 8")
    if isinstance(length, int) and not 1 <= length <= 256:
        raise ValueError("CausalRightAttention requires 1 <= S <= T <= 256")
    if isinstance(past, int) and (
        past < 0 or (isinstance(length, int) and past + length > 256)
    ):
        raise ValueError("CausalRightAttention requires 1 <= S <= T <= 256")

    def unpack(x, heads):
        return op.Transpose(op.Reshape(x, [0, 0, heads, 256]), perm=[0, 2, 1, 3])

    q = unpack(query, num_attention_heads)  # [B,H,S,256]
    # Concatenate REAL history before repeating heads. The returned caches keep KVH.
    present_key = op.Concat(past_key, unpack(key, num_key_value_heads), axis=2)
    present_value = op.Concat(past_value, unpack(value, num_key_value_heads), axis=2)
    total = length + past if isinstance(length, int) and isinstance(past, int) else None
    for tensor in (present_key, present_value):
        tensor.type = query.type
        tensor.shape = ir.Shape([batch, num_key_value_heads, total, 256])
    indices = op.Constant(
        value=ir.tensor(
            np.repeat(
                np.arange(num_key_value_heads, dtype=np.int64),
                num_attention_heads // num_key_value_heads,
            )
        )
    )
    k, v = [op.Gather(x, indices, axis=1) for x in (present_key, present_value)]
    q.type, q.shape = query.type, ir.Shape([batch, num_attention_heads, length, 256])
    for tensor in (k, v):
        tensor.type, tensor.shape = (
            query.type,
            ir.Shape([batch, num_attention_heads, total, 256]),
        )
    output = op.CausalRightAttention(q, k, v, scale=scale, _domain="com.microsoft")
    output.type, output.shape = q.type, q.shape
    packed_shape = op.Shape(query)
    if bias.attention_mask is not None:
        # Retain the existing mask port without altering Q/K/V bits. Caller
        # validation proves its values; the batch dimension participates in I/O.
        packed_shape = op.Concat(
            op.Shape(bias.attention_mask, start=0, end=1),
            op.Shape(query, start=1),
            axis=0,
        )
    output = op.Reshape(op.Transpose(output, perm=[0, 2, 1, 3]), packed_shape)  # [B,S,H*256]
    output.type, output.shape = query.type, query.shape
    return output, present_key, present_value


class _Cuda13Attention:
    def _attend(self, op, *args, **kwargs):
        # The producer's packed shapes are known from the architecture, even if
        # upstream generic op inference has not filled in their Value metadata.
        query, key, value, bias, pk, pv = args
        dtype = kwargs["dtype"]
        for tensor, heads in (
            (query, kwargs["num_attention_heads"]),
            (key, kwargs["num_key_value_heads"]),
            (value, kwargs["num_key_value_heads"]),
        ):
            if _tensor_dtype(tensor) != dtype:
                raise ValueError(
                    "CUDA13 attention requires proven low-dtype projections: "
                    f"{tensor.name}, {_tensor_dtype(tensor)}"
                )
            if tensor.shape is not None and (
                len(tensor.shape) != 3
                or (isinstance(tensor.shape[-1], int) and tensor.shape[-1] != heads * 256)
            ):
                raise ValueError("CUDA13 attention producer has incompatible packed geometry")
            tensor.type = ir.TensorType(dtype)
            if tensor.shape is None:
                tensor.shape = ir.Shape(["batch", "sequence_len", heads * 256])
        return causal_right_attention(op, query, key, value, bias, pk, pv, **kwargs)


class _Cuda13BackboneAttention(_Cuda13Attention, _FishSourceAttention):
    pass


class _Cuda13TowerAttention(_Cuda13Attention, _FishSourceTowerAttention):
    pass


class _Cuda13Experts(_FishFP8Experts):
    def _execution_attributes(self, hidden_states, router_logits):
        for tensor in (hidden_states, router_logits):
            if _tensor_dtype(tensor) != self.dtype:
                raise ValueError(
                    f"fp16_compatible QMoE requires {self.dtype.name} X and logits, "
                    f"got {tensor.dtype}: {tensor.name}"
                )
        return {"fp8_execution_profile": "fp16_compatible"}

    def _qmoe_inputs(self, op, hidden_states, router_logits):
        self._execution_attributes(hidden_states, router_logits)
        values = super()._qmoe_inputs(op, hidden_states, router_logits)
        for value, width in zip(values, (self.hidden_size, self.num_experts)):
            value.type = ir.TensorType(self.dtype)
            value.shape = ir.Shape([None, width])
        return values


class _Cuda13DeltaNet(_FishSourceGatedDeltaNet):
    def _chunk_rule(self, op, *args, **kwargs):
        return chunk_gated_delta_rule(
            op, *args, unit_lower_solver=unit_lower_triangular_solve, **kwargs
        )


class _Cuda13Bias:
    def _attention_bias(self, op, **kwargs):
        return _RightCausalContract(kwargs["attention_mask"])


class _Cuda13TextModel(_Cuda13Bias, _FishTextModel):
    def __init__(self, config):
        super().__init__(config)
        assert config.tower is not None
        for layer in self.layers:
            assert isinstance(layer, _FishDecoderLayer)
            layer.mlp.packed_experts = _Cuda13Experts(config)
            if layer.layer_type == "linear_attention":
                layer.linear_attn = _Cuda13DeltaNet(config)
            else:
                layer.self_attn = _Cuda13BackboneAttention(config)


class _Cuda13AudioDecoder(_Cuda13Bias, FishOmniAudioDecoder):
    def __init__(self, config):
        super().__init__(config)
        tower = dataclasses.replace(config.tower, dtype=config.dtype)
        for layer in [*self.shared_layers, *self.residual_layers]:
            layer.transformer_layer.attention = _Cuda13TowerAttention(tower)


def _validate_config(config: FishOmniConfig) -> None:
    if (
        config.dtype not in (ir.DataType.FLOAT16, ir.DataType.BFLOAT16)
        or not config.native_fp8_experts
        or not config.fp8_experts
        or config.num_local_experts != 256
        or config.num_experts_per_tok != 8
        or not config.norm_topk_prob
        or config.moe_intermediate_size is None
        or config.moe_intermediate_size % 128
        or config.hidden_size % 128
    ):
        raise ValueError(
            "CUDA13 candidate requires F16/BF16 FP8 E256/top8 and H/I128 alignment"
        )
    if config.tower is None:
        raise ValueError("CUDA13 candidate requires a temporal tower")
    for architecture in (config, config.tower):
        if (
            architecture.head_dim != 256
            or not 1 <= architecture.num_attention_heads <= 32
            or not 1 <= architecture.num_key_value_heads <= architecture.num_attention_heads
            or architecture.num_attention_heads % architecture.num_key_value_heads
        ):
            raise ValueError("CUDA13 candidate requires D256 and H1..32 divisible by KV heads")


class Cuda13QualificationModel(FishOmniModel):
    """Builder-only candidate; intentionally not exported from the model registry."""

    def __init__(self, config: FishOmniConfig):
        _validate_config(config)
        super().__init__(config)
        self.decoder.model = _Cuda13TextModel(config)
        self.audio_decoder = _Cuda13AudioDecoder(config)


def validate_cuda13_context(
    position_ids: np.ndarray,
    attention_mask: np.ndarray,
    inputs: np.ndarray,
    caches: Mapping[str, np.ndarray],
    *,
    cuda_version: str,
    sm: int,
    capture: bool,
) -> None:
    """Caller guard, not a runtime qualification certificate.

    The pure runtime harness consumes the exported contract instead of importing
    this builder module (which imports Mobius). CUDA identity must be measured by
    that harness, not inferred from this function accepting a supplied string.
    """
    if cuda_version != "13.0" or sm != 90 or capture:
        raise ValueError("CUDA13 candidate requires exact CUDA13.0, SM90 and capture off")
    if inputs.dtype not in (ir.DataType.FLOAT16.numpy(), ir.DataType.BFLOAT16.numpy()):
        raise ValueError("CUDA13 candidate requires true F16/BF16 activations")
    validate_source_context(
        position_ids, attention_mask, inputs.shape, {k: v.shape for k, v in caches.items()}
    )
    if inputs.shape[0] > 8 or attention_mask.shape[1] > 256:
        raise ValueError("CUDA13 candidate requires B1..8 and 1 <= S <= T <= 256")
    for name, value in caches.items():
        if not name.startswith("past_key_values.") or not name.endswith(
            (".key", ".value", ".conv_state", ".recurrent_state")
        ):
            raise ValueError(f"Unknown CUDA13 candidate cache input: {name}")
        if name.endswith((".key", ".value")) and (
            value.dtype != inputs.dtype
            or not 1 <= value.shape[1] <= 32
            or value.shape[3] != 256
        ):
            raise ValueError(f"CUDA13 candidate requires matching low-dtype D256 KV: {name}")
        if name.endswith(".conv_state") and (value.dtype != inputs.dtype or value.ndim != 3):
            raise ValueError(f"CUDA13 candidate requires low-dtype rank-3 conv state: {name}")
        if name.endswith(".recurrent_state") and (
            value.dtype != np.float32 or value.ndim != 4
        ):
            raise ValueError(f"CUDA13 candidate requires FLOAT rank-4 recurrent state: {name}")


def _inventory(model: ir.Model) -> dict[str, Any]:
    nodes = list(model.graph.all_nodes())
    for function in model.functions.values():
        nodes.extend(function.graph.all_nodes())
    counts = Counter(f"{n.domain or 'ai.onnx'}::{n.op_type}" for n in nodes)
    features = set()
    for node in nodes:
        if node.domain:
            feature = f"{node.domain}::{node.op_type}"
            if node.domain == "com.microsoft" and node.op_type == "QMoE":
                attr = node.attributes.get("fp8_execution_profile")
                feature += f"[fp8_execution_profile={attr.as_string() if attr else 'native'}]"
            features.add(feature)
    return {
        "operators": dict(sorted(counts.items())),
        "required_features": sorted(features),
        "opset_imports": dict(model.graph.opset_imports),
    }


def native_candidate_manifest(package: Mapping[str, ir.Model]) -> dict[str, Any]:
    """JSON inventory of actually emitted ops; not saved-artifact/runtime proof."""
    inventory = {name: _inventory(model) for name, model in package.items()}
    features = {
        feature for model in inventory.values() for feature in model["required_features"]
    }
    qmoe = "not emitted"
    if "com.microsoft::QMoE[fp8_execution_profile=fp16_compatible]" in features:
        qmoe = "FP8 storage/dynamic activation quantization; register-local FP16 MMA"
    elif any(feature.startswith("com.microsoft::QMoE[") for feature in features):
        qmoe = "native profile; compute unqualified"
    return {
        "schema_version": 1,
        "profile": PROFILE,
        "readiness": False,
        "qualification": "unqualified",
        "cuda_version": "13.0",
        "sm": 90,
        "capture": False,
        "context_contract": SOURCE_CONTEXT_CONTRACT,
        "expected_checkpoint_revision": "b2b5ee998034b2975720906c815e7fb0e9f007bd",
        "expected_source_revision": "541566932abbaebcb0a40e562a1f846738383f38",
        "limits": {"batch": [1, 8], "query_length": [1, 256], "total_length": [1, 256]},
        "activation_dtypes": ["FLOAT16", "BFLOAT16"],
        "qmoe_compute": qmoe,
        "native_ordered_reductions": bool(
            features
            & {
                "com.microsoft::OrderedRowReduce",
                "com.microsoft::OrderedMatrixReduce128",
            }
        ),
        "softplus": "portable; Log1p/source-bit qualification pending",
        "optimization_level": "ORT_ENABLE_ALL",
        "disabled_optimizers": [],
        "historical_optimizer_risk": (
            "RotaryEmbeddingFusion may change source rounding; measure actual CUDA13 "
            "optimized graph/vectors before selecting any disabled optimizer"
        ),
        "models": inventory,
    }


def saved_native_candidate_manifest(graph_paths: Mapping[str, Path]) -> dict[str, Any]:
    """Inventory reloaded saved graphs, including every external tensor file hash.

    Does not load external tensors into RAM or claim that weights/runtime vectors
    are qualified. Keep this inventory separate from immutable source DATA and
    the harness's measured runtime/provider selection.
    """
    models = {name: ir.load(path) for name, path in graph_paths.items()}
    manifest = native_candidate_manifest(models)

    def digest(path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    for name, model in models.items():
        if graph_paths[name].is_symlink():
            raise ValueError("Candidate graph must not be a symlink")
        path = graph_paths[name].resolve()
        external = set()
        for graph in [model.graph, *model.graph.subgraphs()]:
            for value in graph.initializers.values():
                if isinstance(value.const_value, ir.ExternalTensor):
                    external.add(str(value.const_value.location))
        entry = manifest["models"][name]
        entry["path"], entry["sha256"] = str(path), digest(path)
        entry["external_data"] = []
        for location in sorted(external):
            external_path = path.parent / location
            if (
                Path(location).is_absolute()
                or ".." in Path(location).parts
                or external_path.is_symlink()
                or not external_path.resolve().is_relative_to(path.parent)
            ):
                raise ValueError(f"External data escapes candidate component: {location}")
            entry["external_data"].append(
                {
                    "path": str(external_path),
                    "sha256": digest(external_path),
                    "bytes": external_path.stat().st_size,
                }
            )
        floating = next(
            (
                value.dtype
                for value in model.graph.outputs
                if value.dtype
                in (ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16)
            ),
            ir.DataType.INT64,
        )
        entry["dtype"] = {
            ir.DataType.FLOAT: "f32",
            ir.DataType.FLOAT16: "f16",
            ir.DataType.BFLOAT16: "bf16",
            ir.DataType.INT64: "int64",
        }[floating]
    manifest["components"] = {
        name: {
            "graph": {
                "path": entry["path"],
                "sha256": entry["sha256"],
                "bytes": graph_paths[name].stat().st_size,
            },
            "external_data": entry["external_data"],
            "dtype": entry["dtype"],
            "required_runtime_features": models[name].metadata_props.get(
                "mobius.required_runtime_features", ""
            ),
        }
        for name, entry in manifest["models"].items()
    }
    return manifest


class _QualificationTask(FishOmniTask):
    def _validate_precision(self, config):
        _validate_config(config)
        if ep_capabilities().name != "cuda":
            raise ValueError("CUDA13 candidate requires execution_provider='cuda'")

    def build(self, module, config) -> ModelPackage:
        if not isinstance(module, Cuda13QualificationModel) or module.config != config:
            raise ValueError("CUDA13 task requires its explicit qualification model/config")
        package = super().build(module, config)
        assert config.tower is not None
        # Native schemas may not exist in the builder's installed ORT. Declare
        # public output contracts from architecture/cache inputs, not guessed
        # inference through unpublished operators.
        output_shapes = {
            "inputs_embeds": ["batch", "sequence_len", config.hidden_size],
            "shared_input": ["batch", "sequence_len", config.hidden_size],
            "logits": ["batch", "sequence_len", config.vocab_size],
            "hidden_states": ["batch", "sequence_len", config.hidden_size],
            "codebook_logits": [
                "batch",
                "sequence_len",
                config.num_codebooks - 1,
                config.tower.vocab_size,
            ],
        }
        for model in package.values():
            inputs = {v.name: v for v in model.graph.inputs}
            for output in model.graph.outputs:
                if output.name in output_shapes:
                    output.type = ir.TensorType(config.dtype)
                    output.shape = ir.Shape(output_shapes[output.name])
                elif output.name is not None and output.name.startswith("present."):
                    cache = inputs["past_key_values." + output.name.removeprefix("present.")]
                    output.type = cache.type
                    assert cache.shape is not None
                    shape = list(cache.shape)
                    if output.name.endswith((".key", ".value")):
                        shape[2] = ir.SymbolicDim("total_sequence_len")
                    output.shape = ir.Shape(shape)
        manifest = native_candidate_manifest(package)
        for name, model in package.items():
            inventory = manifest["models"][name]
            # Generic node counts change in build_from_module's later optimizations.
            # Metadata stores the profile/requirements, never a stale graph inventory.
            metadata = {k: v for k, v in manifest.items() if k != "models"}
            metadata["required_features"] = inventory["required_features"]
            model.metadata_props[PROFILE_METADATA] = json.dumps(metadata, sort_keys=True)
            model.metadata_props["mobius.precision_validation"] = (
                "unqualified-cuda13-candidate"
            )
            model.metadata_props[SOURCE_CONTEXT_METADATA] = SOURCE_CONTEXT_CONTRACT
            model.metadata_props.pop("mobius.required_runtime_features", None)
            if inventory["required_features"]:
                model.metadata_props["mobius.required_runtime_features"] = (
                    "CUDA13.0; SM90; " + "; ".join(inventory["required_features"])
                )
            if name == "decoder":
                model.metadata_props["mobius.fish_omni.gdn_arithmetic"] = (
                    "fp32-chunk64; com.microsoft::UnitLowerTriangularSolve; "
                    "portable-ordered-reductions; unqualified-cuda13-candidate"
                )
        return package


class Cuda13QualificationTask(_QualificationTask):
    """Explicit teacher-forced candidate; never a public admission override."""


class Cuda13GenerationQualificationTask(_QualificationTask, FishOmniGenerationTask):
    """Explicit generation candidate with the existing sampling/cache ABI."""
