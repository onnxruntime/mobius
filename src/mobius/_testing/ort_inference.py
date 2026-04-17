# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""ONNX Runtime inference session wrapper for ir.Model objects.

Uses ``onnxruntime-easy`` which handles bfloat16 and other non-standard
dtypes transparently.
"""

from __future__ import annotations

import logging
import math
import tempfile
from pathlib import Path

import numpy as np
import onnx_ir as ir
import onnxruntime_easy as ort_easy

from mobius._flags import flags
from mobius._model_package import ModelPackage

logger = logging.getLogger(__name__)

# Maximum default-domain ONNX opset that ORT ≤1.24.x CUDA/TRT EPs
# register kernels for.  Models built with a higher opset can be
# loaded after lowering the declared import — the op semantics have
# not changed, only the version label.
_MAX_EP_OPSET = 23

# ORT ≤1.24.x CUDA Gather kernel uses int32 for element offset
# computation (CUDA_LONG = int32_t).  Tensors with >2^31 elements
# cause integer overflow → cudaErrorIllegalAddress.
# See: https://github.com/microsoft/onnxruntime/issues/28107
_MAX_GATHER_ELEMENTS = 2**31 - 1


def _should_lower_opset(model: ir.Model, device: str) -> bool:
    """Return True when lowering the opset import is safe and needed.

    Lowering is only attempted when
    :attr:`~mobius._flags._Flags.ort_lower_opset_for_ep` is enabled and the
    target device is non-CPU with a default-domain opset exceeding
    ``_MAX_EP_OPSET``.

    Lowering is *not* safe when the graph contains ops that were first
    introduced in a post-23 opset (e.g. ``TensorScatter``).  In that
    case the model requires genuine opset 24+ support and lowering
    would produce an invalid model.
    """
    if not flags.ort_lower_opset_for_ep:
        return False
    if device == "cpu":
        return False
    current_opset = model.opset_imports.get("", 0)
    if current_opset <= _MAX_EP_OPSET:
        return False

    # Ops that were *introduced* in opset 24 and have no opset 23
    # equivalent.  If any appear in the graph, lowering is unsafe.
    opset_24_only_ops = {"TensorScatter"}
    for node in model.graph:
        if node.domain == "" and node.op_type in opset_24_only_ops:
            return False
    return True


def _split_large_gathers(model: ir.Model) -> None:
    """Split Gather ops whose data tensor exceeds the CUDA int32 limit.

    ORT ≤1.24.x CUDA Gather kernel uses ``int32_t`` for element offset
    computation.  When ``row_index * row_stride + col_offset`` exceeds
    ``INT32_MAX``, the kernel produces an illegal memory access
    (cudaError 700).  This affects any rank-2 initializer-backed
    embedding table with more than ~2.15 billion elements.

    This transform shards the data tensor along axis 0 into chunks
    that each stay under the int32 limit, then routes indices to the
    correct shard using ``Less`` / ``Where`` / ``Sub`` ops and merges
    the results.

    The model is mutated **in-place**.  Only axis-0 Gather on rank-2
    initializers is handled — other cases are left untouched.

    See: https://github.com/microsoft/onnxruntime/issues/28107
    """
    graph = model.graph
    nodes_to_replace: list[ir.Node] = []

    for node in graph:
        if node.op_type != "Gather" or node.domain not in ("", None):
            continue
        axis = node.attributes.get("axis")
        axis_val = axis.as_int() if axis is not None else 0
        if axis_val != 0:
            continue
        data_val = node.inputs[0]
        if data_val is None or data_val.shape is None or len(data_val.shape) != 2:
            continue
        rows, cols = data_val.shape
        if not isinstance(rows, int) or not isinstance(cols, int):
            continue
        if rows * cols <= _MAX_GATHER_ELEMENTS:
            continue
        # Only handle initializer-backed data (embedding weights)
        if data_val.const_value is None:
            continue
        nodes_to_replace.append(node)

    if not nodes_to_replace:
        return

    for node in nodes_to_replace:
        data_val = node.inputs[0]
        indices_val = node.inputs[1]
        output_val = node.outputs[0]
        rows, cols = data_val.shape  # type: ignore[misc]

        # Determine number of shards so each has ≤ _MAX_GATHER_ELEMENTS
        max_rows_per_shard = _MAX_GATHER_ELEMENTS // cols
        num_shards = math.ceil(rows / max_rows_per_shard)
        shard_size = math.ceil(rows / num_shards)

        logger.info(
            "Splitting Gather %r: data [%d, %d] (%s elems) into %d shards of ≤%d rows each",
            node.name,
            rows,
            cols,
            f"{rows * cols:,}",
            num_shards,
            shard_size,
        )

        # Shard the initializer data along axis 0
        weight_np = data_val.const_value.numpy()
        shard_values: list[ir.Value] = []
        boundaries: list[int] = []
        for i in range(num_shards):
            start = i * shard_size
            end = min(start + shard_size, rows)
            boundaries.append(start)
            shard_np = weight_np[start:end]
            shard_tensor = ir.Tensor(shard_np, name=f"{data_val.name}_shard{i}")
            shard_val = ir.Value(
                name=f"{data_val.name}_shard{i}",
                type=ir.TensorType(data_val.dtype),
                shape=ir.Shape(shard_np.shape),
                const_value=shard_tensor,
            )
            graph.register_initializer(shard_val)
            shard_values.append(shard_val)

        # Build routing subgraph:
        # For each shard i with boundary[i]:
        #   is_shard_i = (boundary[i] <= indices) & (indices < boundary[i+1])
        #   local_idx = indices - boundary[i]
        #   shard_result = Gather(shard_data, local_idx)
        # Final result = nested Where(is_shard_0, shard_0_result, Where(...))

        idx_dtype = indices_val.dtype if indices_val.dtype is not None else ir.DataType.INT64

        # Start from the last shard and work backwards with Where
        result: ir.Value | None = None
        for i in reversed(range(num_shards)):
            boundary = boundaries[i]

            # local_idx = indices - boundary (or just indices for shard 0)
            if boundary == 0:
                local_idx = indices_val
            else:
                boundary_const = _make_scalar_constant(
                    graph,
                    boundary,
                    idx_dtype,
                    f"{node.name}_boundary{i}",
                )
                sub_node = ir.Node(
                    "",
                    "Sub",
                    inputs=[indices_val, boundary_const],
                    num_outputs=1,
                    name=f"{node.name}_sub_shard{i}",
                )
                graph.append(sub_node)
                local_idx = sub_node.outputs[0]

            # Gather from this shard
            gather_node = ir.Node(
                "",
                "Gather",
                inputs=[shard_values[i], local_idx],
                attributes=[ir.Attr("axis", ir.AttributeType.INT, 0)],
                num_outputs=1,
                name=f"{node.name}_gather_shard{i}",
            )
            graph.append(gather_node)
            shard_result = gather_node.outputs[0]

            if result is None:
                # Last shard — this is the fallback
                result = shard_result
            else:
                # is_this_shard = indices < boundary[i+1]
                next_boundary = boundaries[i + 1]
                bound_const = _make_scalar_constant(
                    graph,
                    next_boundary,
                    idx_dtype,
                    f"{node.name}_bound{i}",
                )
                less_node = ir.Node(
                    "",
                    "Less",
                    inputs=[indices_val, bound_const],
                    num_outputs=1,
                    name=f"{node.name}_less_shard{i}",
                )
                graph.append(less_node)
                cond = less_node.outputs[0]

                # Unsqueeze condition for broadcasting: [B, S] → [B, S, 1]
                neg_one = _make_scalar_constant(
                    graph,
                    -1,
                    ir.DataType.INT64,
                    f"{node.name}_neg1_shard{i}",
                )
                unsq_node = ir.Node(
                    "",
                    "Unsqueeze",
                    inputs=[cond, neg_one],
                    num_outputs=1,
                    name=f"{node.name}_unsq_shard{i}",
                )
                graph.append(unsq_node)

                where_node = ir.Node(
                    "",
                    "Where",
                    inputs=[unsq_node.outputs[0], shard_result, result],
                    num_outputs=1,
                    name=f"{node.name}_where_shard{i}",
                )
                graph.append(where_node)
                result = where_node.outputs[0]

        # Replace all uses of the original Gather output
        assert result is not None
        output_val.replace_all_uses_with(result)
        graph.remove(node, safe=True)

    logger.info("Split %d large Gather node(s)", len(nodes_to_replace))


def _make_scalar_constant(
    graph: ir.Graph,
    value: int,
    dtype: ir.DataType,
    name: str,
) -> ir.Value:
    """Create a scalar constant value and register it as an initializer."""
    if dtype == ir.DataType.INT64:
        np_val = np.array(value, dtype=np.int64)
    elif dtype == ir.DataType.INT32:
        np_val = np.array(value, dtype=np.int32)
    else:
        np_val = np.array(value, dtype=np.int64)
    tensor = ir.Tensor(np_val, name=name)
    val = ir.Value(
        name=name,
        type=ir.TensorType(dtype),
        shape=ir.Shape(np_val.shape),
        const_value=tensor,
    )
    graph.register_initializer(val)
    return val


class OnnxModelSession:
    """Wraps an ``onnxruntime_easy.EasySession`` for an ``ir.Model``.

    Serializes the model to a temporary file and creates an ORT session.
    Provides a simple ``run()`` interface that accepts and returns numpy arrays.

    Example::

        from mobius._testing.ort_inference import OnnxModelSession

        session = OnnxModelSession(model)
        outputs = session.run({"input_ids": np.array([[1, 2, 3]])})
        logits = outputs["logits"]
    """

    def __init__(
        self,
        model: ir.Model | ModelPackage,
        **load_kwargs,
    ):
        if isinstance(model, ModelPackage):
            if len(model) != 1:
                raise ValueError(
                    f"ModelPackage has {len(model)} models; pass a "
                    f"single ir.Model or index into the package."
                )
            model = next(iter(model.values()))

        # Workaround: ORT ≤1.24.x CUDA/TRT EPs don't register kernels
        # for ONNX opset 24 standard ops (Squeeze, Reshape, etc.).  The
        # op semantics are identical to opset 23, so lowering the import
        # declaration lets the EP find its existing kernels.  The model
        # object is restored to its original opset after saving.
        device = load_kwargs.get("device", "cpu")
        lower_opset = _should_lower_opset(model, device)
        original_opset = model.opset_imports.get("", 0) if lower_opset else 0
        if lower_opset:
            logger.info(
                "Lowering default-domain opset from %d to %d for %s",
                original_opset,
                _MAX_EP_OPSET,
                device,
            )
            model.opset_imports[""] = _MAX_EP_OPSET

        # Workaround: ORT ≤1.24.x CUDA Gather kernel uses int32 for
        # element offset computation.  Split oversized Gather ops so
        # each shard stays under INT32_MAX elements.
        if device == "cuda":
            _split_large_gathers(model)

        self._tmpdir = tempfile.TemporaryDirectory()
        self._model_path = str(Path(self._tmpdir.name) / "model.onnx")
        try:
            ir.save(model, self._model_path, external_data="model.onnx.data")
        finally:
            if lower_opset:
                model.opset_imports[""] = original_opset

        self._session = ort_easy.load(self._model_path, **load_kwargs)
        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._output_names = [out.name for out in self._session.get_outputs()]

    @property
    def input_names(self) -> list[str]:
        return self._input_names

    def get_input_shape(self, name: str) -> list[int | str] | None:
        """Return the declared shape of an input, or ``None`` if not found.

        Shape elements may be ``int`` (static) or ``str`` (symbolic).
        """
        for inp in self._session.get_inputs():
            if inp.name == name:
                return list(inp.shape)
        return None

    @property
    def output_names(self) -> list[str]:
        return self._output_names

    def run(self, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference and return outputs as a name→array dict.

        Args:
            feeds: Input name → numpy array mapping. Only inputs present
                in the model are used; extra keys are ignored.

        Returns:
            Dict mapping output names to numpy arrays.
        """
        # Filter to only inputs the model expects and convert to OrtValues
        ort_feeds = {}
        for k, v in feeds.items():
            if k not in self._input_names:
                continue
            # Ensure contiguous layout for ORT. Skip 0-d scalars
            # because np.ascontiguousarray promotes them to 1-d.
            if v.ndim > 0:
                v = np.ascontiguousarray(v)
            ort_feeds[k] = ort_easy.ort_value(v)
        raw_outputs = self._session(**ort_feeds)
        return dict(zip(self._output_names, (o.numpy() for o in raw_outputs)))

    def close(self) -> None:
        self._tmpdir.cleanup()

    def __del__(self) -> None:
        self.close()
