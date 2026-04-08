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
"""

from __future__ import annotations

import logging

import onnx_ir as ir

logger = logging.getLogger(__name__)


class FoldTransposedInitializerPass(ir.passes.InPlacePass):
    """Fold ``Transpose(initializer, perm=[1, 0])`` into a pre-transposed weight.

    For each ``Transpose`` node whose sole input is a graph initializer and
    whose ``perm`` attribute is ``[1, 0]``:

    1. A new initializer ``{original_name}_t`` is registered whose
       :class:`~onnx_ir.LazyTensor` value lazily transposes the original data.
    2. All consumers of the ``Transpose`` output are rewired to the new
       initializer.
    3. The ``Transpose`` node is removed from the graph.

    If multiple ``Transpose`` nodes reference the same initializer, the
    transposed initializer is created only once and shared among all consumers.

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
            if inp.name not in model.graph.initializers:
                continue

            perm_attr = node.attributes.get("perm")
            if perm_attr is None or list(perm_attr.value) != [1, 0]:
                continue

            out_val = node.outputs[0]

            if inp.name not in folded:
                # Derive shape of the transposed tensor from the Transpose output.
                t_shape = out_val.shape  # already transposed by shape inference

                # Create a LazyTensor that transposes the original data on demand.
                original = inp  # captured by closure — updated when weights load

                def _make_transposed(w: ir.Value = original) -> ir.TensorProtocol:
                    assert w.const_value is not None, (
                        f"Initializer {w.name!r} has no const_value. "
                        "FoldTransposedInitializerPass must run after weights are loaded."
                    )
                    return ir.tensor(w.const_value.numpy().T)

                new_val = ir.Value(name=f"{inp.name}_t", shape=t_shape, type=inp.type)
                new_val.const_value = ir.LazyTensor(
                    _make_transposed,
                    dtype=inp.dtype or ir.DataType.FLOAT,
                    shape=t_shape,
                )
                model.graph.register_initializer(new_val)
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
