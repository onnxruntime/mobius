# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Pass that folds Concat(init_0, init_1, ...) into a single pre-packed initializer.

When all inputs to a ``Concat`` node are graph initializers the concatenation
result is statically known at weight-load time.  This pass materialises the
result as a new initializer, removing the runtime ``Concat`` node.

If all inputs have tensor data (``const_value is not None``), the new
initializer uses ``ir.LazyTensor`` so the concatenation is deferred until
serialisation, avoiding holding a duplicate copy of all weight slices in memory.

If any input has no tensor data (``const_value is None``), the new initializer
is registered with ``const_value=None`` and source information is stored in
``metadata_props`` so that
:func:`~mobius._optimizations.fold_initializers_after_weights` can materialise
the value once weights are loaded.

Primary use case: QKV weight packing.  After the GQA rewrite rules produce
``Concat(q_weight, k_weight, v_weight, axis=0)`` and the Transpose folding pass
replaces each ``*_weight_t`` with a pre-transposed initializer, this pass folds
the resulting all-initializer Concat into a single packed initializer.
"""

from __future__ import annotations

import logging

import numpy as np
import onnx_ir as ir

logger = logging.getLogger(__name__)


class FoldConcatInitializersPass(ir.passes.InPlacePass):
    """Fold ``Concat(init_0, init_1, ...)`` into a single packed initializer.

    For each ``Concat`` node whose **every** input is a graph initializer:

    1. A new initializer ``{name_0}__{name_1}__axis_{axis}__concat`` is
       registered.

       * If all sources have tensor data, the new initializer uses
         ``ir.LazyTensor`` so the concatenation is deferred until serialisation,
         avoiding holding duplicate copies of all weight slices in memory.
       * If any source has no tensor data (``const_value is None``),
         ``const_value`` is left as ``None`` and the source names are stored in
         ``metadata_props`` for later materialisation by
         :func:`~mobius._optimizations.fold_initializers_after_weights`.

    2. All consumers of the ``Concat`` output are rewired to the new
       initializer.
    3. The ``Concat`` node is removed from the graph.

    The original initializers are left in place; a subsequent
    :class:`~onnx_ir.passes.common.RemoveUnusedNodesPass` (called after
    :func:`~mobius._optimizations.fold_initializers_after_weights` materialises
    any deferred values) will prune them once they have no remaining uses.
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
            # If shape inference hasn't set the Concat output shape, compute it
            # from the input shapes so folded initializers have usable metadata.
            if out_shape is None:
                input_shapes = [v.shape for v in inputs]  # type: ignore[union-attr]
                if all(s is not None for s in input_shapes):
                    dims = list(input_shapes[0])  # type: ignore[arg-type]
                    dims[axis] = sum(int(s[axis]) for s in input_shapes)  # type: ignore[index]
                    out_shape = ir.Shape(dims)

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

            new_val = ir.Value(name=packed_name, shape=out_shape, type=out_val.type)

            # Check if all sources have data available.
            all_have_data = all(v.const_value is not None for v in inputs)  # type: ignore[union-attr]
            if all_have_data:
                # Use a LazyTensor to defer the concatenation until serialization,
                # avoiding holding a second copy of all weights in memory.
                captured_inputs = list(inputs)
                captured_axis = axis
                new_val.const_value = ir.LazyTensor(
                    lambda ins=captured_inputs, ax=captured_axis: ir.tensor(
                        np.concatenate([v.const_value.numpy() for v in ins], axis=ax)  # type: ignore[union-attr]
                    ),
                    dtype=dtypes[0] or ir.DataType.FLOAT,
                    shape=out_shape,
                    name=new_val.name,
                )
            else:
                # No data yet — leave const_value=None and record the source names
                # so fold_initializers_after_weights() can fill it in later.
                new_val.metadata_props["_fold_sources"] = ",".join(input_names)
                new_val.metadata_props["_fold_axis"] = str(axis)

            model.graph.initializers[new_val.name] = new_val

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
