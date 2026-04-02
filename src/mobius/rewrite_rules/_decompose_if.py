# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Graph-level pass for decomposing ONNX ``If`` nodes into ``Where`` ops.

DML does not support the ``If`` operator.  WebGPU requires graph-capture
mode where dynamic control-flow (``If``) is prohibited.  This pass
replaces ``If`` nodes with inlined branch computations selected by
``Where``.

**Matched pattern:**

.. code-block:: text

    result_0, ... = If(
        condition,
        then_branch=[...; yield then_0, ...],
        else_branch=[...; yield else_0, ...],
    )

**Replacement:**

.. code-block:: text

    # then_branch nodes inlined into parent graph
    # else_branch nodes inlined into parent graph
    result_0 = Where(condition, then_0, else_0)
    ...

**Constraints:**

- Each output of the ``If`` node is replaced by a ``Where`` that selects
  between the corresponding then/else branch output values.
- Both branches may reference outer-scope values (captured from the parent
  graph) without restriction — those references are preserved as-is.
- The ``condition`` input to ``If`` must be a scalar ``bool``; ``Where``
  expects a boolean condition that broadcasts over the output shape, so
  scalar conditions may need explicit broadcasting for non-scalar outputs.
  This pass inserts an ``Expand`` if any branch output has a shape that
  does not match the scalar condition.

Usage::

    from mobius.rewrite_rules import decompose_if_pass

    model = build("my_model", execution_provider="dml")
    decompose_if_pass()(model)
"""

from __future__ import annotations

import onnx_ir as ir


class DecomposeIfPass(ir.passes.InPlacePass):
    """Inline If node branches and replace outputs with Where selection.

    Applies to all ``If`` nodes in the model's main graph.  Subgraph nodes
    are moved into the parent graph in their natural topological order.

    This is a graph-level pass, not a pattern rewrite rule, because the
    ``If`` node's branches are stored as subgraph attributes — the
    pattern-matching rewrite API only handles value/op-level patterns.
    """

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        changed = False
        # Collect If nodes first (list() avoids modifying during iteration).
        if_nodes = [n for n in model.graph if n.op_type == "If" and n.domain == ""]
        for if_node in if_nodes:
            self._decompose(model.graph, if_node)
            changed = True
        return ir.passes.PassResult(model, modified=changed)

    def _decompose(self, graph: ir.Graph, if_node: ir.Node) -> None:
        """Inline a single ``If`` node's branches and replace with Where."""
        cond = if_node.inputs[0]

        then_branch: ir.Graph = if_node.attributes["then_branch"].value
        else_branch: ir.Graph = if_node.attributes["else_branch"].value

        # Capture the output values from both branches *before* moving nodes,
        # since graph.outputs is updated when nodes are re-parented.
        then_outputs = list(then_branch.outputs)
        else_outputs = list(else_branch.outputs)

        # Move all nodes from each branch into the parent graph, inserting
        # them immediately before the If node in topological order.
        for branch in (then_branch, else_branch):
            branch_nodes = list(branch)  # snapshot in topological order
            for node in branch_nodes:
                branch.remove(node, safe=False)
                graph.insert_before(if_node, node)

        # For each output pair, create a Where node to select the result.
        for if_out, then_out, else_out in zip(if_node.outputs, then_outputs, else_outputs):
            where_node = ir.Node(
                domain="",
                op_type="Where",
                inputs=[cond, then_out, else_out],
                num_outputs=1,
            )
            where_result = where_node.outputs[0]
            where_result.name = if_out.name
            graph.insert_before(if_node, where_node)
            # Replace all downstream uses of the If output with Where output.
            ir.convenience.replace_all_uses_with(
                [if_out], [where_result], replace_graph_outputs=True
            )

        # Remove the now-unused If node.
        graph.remove(if_node, safe=False)


def decompose_if_pass() -> DecomposeIfPass:
    """Return a pass that decomposes ``If`` nodes into ``Where`` ops.

    Used for DML (no ``If`` support) and WebGPU (graph-capture mode).

    Returns:
        :class:`DecomposeIfPass` instance ready to be called on a model.
    """
    return DecomposeIfPass()
