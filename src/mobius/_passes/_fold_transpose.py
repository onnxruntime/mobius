# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Pass that folds Transpose(initializer, perm=[1, 0]) into a pre-transposed weight.

After :class:`~mobius.components.Linear` emits ``Transpose(weight, perm=[1, 0])
→ MatMul(x, w_t)``, this pass pre-computes the transposition and stores the
result as a new initializer named ``{original_name}_t``.  The runtime
Transpose node is then removed, eliminating per-inference overhead.

:class:`onnx_ir.LazyTensor` is used so the actual numpy transposition is
deferred until the tensor data is first accessed (e.g. during ONNX
serialization), avoiding memory spikes from eagerly materializing all
transposed weights during the pass itself.

.. important::
    This pass must run **after** weights are loaded (i.e. after
    :func:`~mobius._optimizations.fold_initializers_after_weights` is called).
    The ``LazyTensor`` closures capture the source initializer and will fail
    if ``const_value`` is still ``None`` when the model is serialized.
"""

from __future__ import annotations

import logging

import onnx_ir as ir

logger = logging.getLogger(__name__)


class FoldTransposedInitializerPass(ir.passes.InPlacePass):
    """Fold ``Transpose(initializer, perm=[1, 0])`` into a pre-transposed weight.

    For each ``Transpose`` node whose sole input is a graph initializer that
    has **exactly one use** (the Transpose node itself) and whose ``perm``
    attribute is ``[1, 0]``:

    1. A new initializer ``{original_name}_t`` is registered whose
       :class:`~onnx_ir.LazyTensor` value lazily transposes the original data.
    2. All consumers of the ``Transpose`` output are rewired to the new
       initializer.
    3. The ``Transpose`` node is removed from the graph.

    Initializers with multiple consumers are skipped — renaming or removing a
    shared initializer would silently break the other consumers.

    The original initializer is left in place; a subsequent
    :class:`~onnx_ir.passes.common.RemoveUnusedNodesPass` (or the
    initializer-dedup pass) will prune it if it has no remaining consumers.
    """

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        # Map from original initializer name → pre-transposed ir.Value, so
        # multiple Transpose nodes using the same initializer share one result.
        folded: dict[str, ir.Value] = {}
        folded_nodes = 0  # total Transpose nodes removed (may exceed len(folded))
        modified = False

        for node in list(model.graph.all_nodes()):
            if node.op_type != "Transpose":
                continue

            if len(node.inputs) != 1:
                continue

            inp = node.inputs[0]
            if inp is None or inp.name is None:
                continue

            # Only fold initializer inputs (not graph inputs or computed values).
            if not inp.is_initializer():
                continue

            # Skip if the initializer is shared by multiple consumers — folding
            # would rename/remove it and silently break the other users.
            if len(inp.uses()) != 1:
                continue

            perm_attr = node.attributes.get("perm")
            if perm_attr is None or list(perm_attr.value) != [1, 0]:
                continue

            out_val = node.outputs[0]

            if inp.name not in folded:
                new_name = f"{inp.name}_t"

                # Derive shape of the transposed tensor from the Transpose output.
                # Fall back to computing from the input shape if shape inference
                # did not propagate to this new node (e.g. after stage-2 rewrites).
                t_shape = out_val.shape
                if t_shape is None and inp.shape is not None:
                    perm = list(perm_attr.value)
                    t_shape = ir.Shape([inp.shape[p] for p in perm])

                new_val = ir.Value(name=new_name, shape=t_shape, type=inp.type)

                # Create a LazyTensor that transposes the original data on demand.
                # The actual numpy transposition is deferred until serialization,
                # avoiding holding a second copy of the weight in memory.
                src = inp  # captured for the closure below
                new_val.const_value = ir.LazyTensor(
                    lambda s=src: ir.tensor(s.const_value.numpy().T),
                    dtype=inp.dtype or ir.DataType.FLOAT,
                    shape=t_shape,
                    name=new_val.name,
                )

                model.graph.initializers[new_val.name] = new_val
                folded[inp.name] = new_val
                logger.debug(
                    "FoldTransposedInitializer: registered %r (shape %s)",
                    new_val.name,
                    t_shape,
                )

            replacement = folded[inp.name]
            out_val.replace_all_uses_with(replacement, replace_graph_outputs=True)
            model.graph.remove(node)
            folded_nodes += 1
            modified = True

        if modified:
            logger.debug(
                "FoldTransposedInitializerPass: folded %d Transpose nodes"
                " (%d unique initializers pre-transposed)",
                folded_nodes,
                len(folded),
            )

        return ir.passes.PassResult(model, modified=modified)
