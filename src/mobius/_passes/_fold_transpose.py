# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Pass that folds Transpose(initializer, perm=[1, 0]) into a pre-transposed weight.

After :class:`~mobius.components.Linear` emits ``Transpose(weight, perm=[1, 0])
→ MatMul(x, w_t)``, this pass pre-computes the transposition and stores the
result as a new initializer named ``{original_name}_t``.  The runtime
Transpose node is then removed, eliminating per-inference overhead.

When the source initializer has tensor data (``const_value is not None``), the
folded initializer uses ``ir.LazyTensor`` so the transposition is deferred until
serialisation, avoiding holding a duplicate copy of the weight in memory.

When the source initializer has no tensor data (``const_value is None``), the
folded initializer is registered with ``const_value=None`` and the source name
is stored in ``metadata_props["_fold_source"]`` so that
:func:`~mobius._optimizations.fold_initializers_after_weights` can materialise
the value once weights are loaded.
"""

from __future__ import annotations

import logging

import onnx_ir as ir

logger = logging.getLogger(__name__)


class FoldTransposedInitializerPass(ir.passes.InPlacePass):
    """Fold ``Transpose(initializer, perm=[1, 0])`` into a pre-transposed weight.

    For each ``Transpose`` node whose sole input is a graph initializer and
    whose ``perm`` attribute is ``[1, 0]``:

    1. A new initializer ``{original_name}_t`` is registered.

       * If the source has tensor data, the new initializer uses
         ``ir.LazyTensor`` so the transposition is deferred until serialisation,
         avoiding a duplicate in-memory copy.
       * If the source has no tensor data (``const_value is None``), the
         initializer is left with ``const_value=None`` and
         ``metadata_props["_fold_source"]`` records the source name for later
         materialisation by :func:`~mobius._optimizations.fold_initializers_after_weights`.

    2. All consumers of the ``Transpose`` output are rewired to the new
       initializer.
    3. The ``Transpose`` node is removed from the graph.

    If multiple ``Transpose`` nodes reference the same initializer, the
    transposed initializer is created only once and shared among all consumers.

    The original initializer is left in place; a subsequent
    :class:`~onnx_ir.passes.common.RemoveUnusedNodesPass` (called after
    :func:`~mobius._optimizations.fold_initializers_after_weights` materialises
    any deferred values) will prune it once it has no remaining consumers.
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
            if inp.name not in model.graph.initializers:
                continue

            perm_attr = node.attributes.get("perm")
            if perm_attr is None or list(perm_attr.value) != [1, 0]:
                continue

            out_val = node.outputs[0]

            if inp.name not in folded:
                # Derive shape of the transposed tensor from the Transpose output.
                # Fall back to computing from the input shape if shape inference
                # did not propagate to this new node (e.g. after stage-2 rewrites).
                t_shape = out_val.shape
                if t_shape is None and inp.shape is not None:
                    perm = list(perm_attr.value)
                    t_shape = ir.Shape([inp.shape[p] for p in perm])

                new_val = ir.Value(name=f"{inp.name}_t", shape=t_shape, type=inp.type)

                if inp.const_value is not None:
                    # Use a LazyTensor to defer the transpose until serialization,
                    # avoiding holding a second copy of the weight in memory.
                    src = inp  # captured for the closure below
                    new_val.const_value = ir.LazyTensor(
                        lambda s=src: ir.tensor(s.const_value.numpy().T),
                        dtype=inp.dtype or ir.DataType.FLOAT,
                        shape=t_shape,
                        name=new_val.name,
                    )
                else:
                    # No data yet — leave const_value=None and record the source
                    # so fold_initializers_after_weights() can fill it in later.
                    new_val.metadata_props["_fold_source"] = inp.name

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
