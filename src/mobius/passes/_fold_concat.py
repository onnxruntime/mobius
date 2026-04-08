# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Pass that folds Concat(init_0, init_1, ...) into a single pre-packed initializer.

When all inputs to a ``Concat`` node are graph initializers the concatenation
result is statically known at weight-load time.  This pass materialises the
result as a new initializer, removing the runtime ``Concat`` node and its
operands.

:class:`onnx_ir.LazyTensor` is used so the actual numpy concatenation is
deferred until the tensor data is first accessed (e.g. during ONNX
serialization), avoiding memory spikes from eagerly materialising all
concatenated weights during the pass itself.

Primary use case: QKV weight packing.  After the GQA rewrite rules produce
``Concat(q_weight_t, k_weight_t, v_weight_t, axis=0)`` and the Transpose
folding pass replaces each ``*_weight_t`` with a pre-transposed initializer,
this pass folds the resulting all-initializer Concat into a single packed
``qkv_weight`` initializer.
"""

from __future__ import annotations

import logging

import numpy as np
import onnx_ir as ir

logger = logging.getLogger(__name__)


class FoldConcatInitializersPass(ir.passes.InPlacePass):
    """Fold ``Concat(init_0, init_1, ...)`` into a single packed initializer.

    For each ``Concat`` node whose **every** input is a graph initializer:

    1. A new initializer ``{name_0}__{name_1}__axis_{axis}__concat`` is registered whose
       :class:`~onnx_ir.LazyTensor` value lazily concatenates the inputs along
       the Concat's ``axis`` attribute.
    2. All consumers of the ``Concat`` output are rewired to the new
       initializer.
    3. The ``Concat`` node is removed from the graph.

    The original initializers are left in place; a subsequent
    :class:`~onnx_ir.passes.common.RemoveUnusedNodesPass` (or the
    initializer-dedup pass) will prune them if they have no remaining uses.
    """

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        modified = False
        folded_count = 0

        for node in list(model.graph.all_nodes()):
            if node.op_type != "Concat":
                continue

            inputs = node.inputs
            if not inputs:
                continue

            # Only fold when ALL inputs are graph initializers.
            if not all(
                v is not None and v.name is not None and v.name in model.graph.initializers
                for v in inputs
            ):
                continue

            axis_attr = node.attributes.get("axis")
            axis: int = axis_attr.value if axis_attr is not None else 0

            out_val = node.outputs[0]
            out_shape = out_val.shape

            # Require uniform dtype — mixed-dtype concat is unusual and likely
            # a modelling error; skip and warn rather than silently produce
            # a wrong result.
            dtypes = [v.dtype for v in inputs]  # type: ignore[union-attr]
            if len(set(dtypes)) > 1:
                logger.warning(
                    "FoldConcatInitializersPass: skipping Concat with mixed dtypes %s"
                    " — inputs must all share the same dtype to be folded.",
                    dtypes,
                )
                continue

            # Build a name for the packed initializer from the input names and axis.
            # The concatenated value depends on both, so include axis to avoid
            # collisions when the same inputs are concatenated along different axes.
            input_names = [v.name for v in inputs]  # type: ignore[union-attr]
            packed_name = "__".join(input_names) + f"__axis_{axis}__concat"

            # If an equivalent packed initializer already exists, reuse it and still
            # remove this Concat node so all eligible Concat nodes are folded.
            if packed_name in model.graph.initializers:
                existing_val = model.graph.initializers[packed_name]
                out_val.replace_all_uses_with(existing_val, replace_graph_outputs=True)
                model.graph.remove(node)
                folded_count += 1
                modified = True

                logger.debug(
                    "FoldConcatInitializers: reused [%s] → %r (axis=%d, shape=%s)",
                    ", ".join(input_names),
                    packed_name,
                    axis,
                    out_shape,
                )
                continue

            captured_inputs = list(inputs)  # capture for closure

            def _make_packed(
                parts: list[ir.Value] = captured_inputs, ax: int = axis
            ) -> ir.TensorProtocol:
                arrays = []
                for v in parts:
                    assert v.const_value is not None, (
                        f"Initializer {v.name!r} has no const_value. "
                        "FoldConcatInitializersPass must run after weights are loaded."
                    )
                    arrays.append(v.const_value.numpy())
                return ir.tensor(np.concatenate(arrays, axis=ax))

            dtype = inputs[0].dtype or ir.DataType.FLOAT  # type: ignore[union-attr]
            new_val = ir.Value(name=packed_name, shape=out_shape, type=out_val.type)
            new_val.const_value = ir.LazyTensor(_make_packed, dtype=dtype, shape=out_shape)
            model.graph.register_initializer(new_val)

            out_val.replace_all_uses_with(new_val, replace_graph_outputs=True)
            model.graph.remove(node)
            folded_count += 1
            modified = True

            logger.debug(
                "FoldConcatInitializers: packed [%s] → %r (axis=%d, shape=%s)",
                ", ".join(input_names),
                packed_name,
                axis,
                out_shape,
            )

        if modified:
            logger.debug("FoldConcatInitializersPass: folded %d Concat nodes", folded_count)

        return ir.passes.PassResult(model, modified=modified)
