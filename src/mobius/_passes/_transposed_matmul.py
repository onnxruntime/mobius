# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Pass that hands projection MatMuls their weight in the checkpoint's layout.

:class:`~mobius.components.Linear` emits ``Transpose(weight, perm=[1, 0]) →
MatMul(x, w_t)``, and :class:`~mobius._passes.FoldTransposedInitializerPass`
later stores the ``[in, out]`` result as the initializer. At batch 1 that is
the slower layout for cuBLAS. This pass rewrites the pair into
``com.microsoft::FusedMatMul(x, weight, transB=1)``, so the weight stays
``[out, in]`` and the transpose happens inside the GEMM. Unlike ``Gemm``,
``FusedMatMul`` accepts the rank-3 activations a decoder layer feeds.

The source of the ``Transpose`` must be a weight: an initializer, or a
``Concat`` of initializers along axis 0, which is the packed Q/K/V weight
:func:`~mobius.rewrite_rules.pack_qkv_for_gqa_rules` emits and
:class:`~mobius._passes.FoldConcatInitializersPass` folds once weights are
loaded. Transposed activations are left alone.

The pass is structural and runs before weights are loaded, after the
fusion stage, so the ``MatMul`` patterns the GQA and PackQKV rewrites match
are still there when they run.
"""

from __future__ import annotations

import logging

import onnx_ir as ir

logger = logging.getLogger(__name__)


def _is_weight(value: ir.Value) -> bool:
    """Whether ``value`` is an initializer or an axis-0 Concat of initializers."""
    if value.is_initializer():
        return True
    producer = value.producer()
    if producer is None or producer.op_type != "Concat" or producer.domain != "":
        return False
    axis = producer.attributes.get("axis")
    if axis is None or axis.value != 0:
        return False
    return all(inp is not None and inp.is_initializer() for inp in producer.inputs)


class TransposedWeightMatMulPass(ir.passes.InPlacePass):
    """Rewrite ``MatMul(x, Transpose(weight))`` into ``FusedMatMul(transB=1)``.

    A ``MatMul`` is rewritten when its second input is the sole use of a
    ``Transpose(perm=[1, 0])`` whose input is a weight (see the module
    docstring). The ``Transpose`` is removed with it. ``converted`` counts the
    rewritten nodes.
    """

    def __init__(self) -> None:
        super().__init__()
        self.converted = 0

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        self.converted = 0
        for node in list(model.graph.all_nodes()):
            if node.op_type != "MatMul" or node.domain != "" or len(node.inputs) != 2:
                continue
            x, w_t = node.inputs
            if x is None or w_t is None or len(w_t.uses()) != 1 or w_t.is_graph_output():
                continue
            transpose = w_t.producer()
            if transpose is None or transpose.op_type != "Transpose" or transpose.domain != "":
                continue
            perm = transpose.attributes.get("perm")
            if perm is None or list(perm.value) != [1, 0]:
                continue
            weight = transpose.inputs[0]
            if weight is None or not _is_weight(weight):
                continue

            graph = node.graph
            fused = ir.Node(
                "com.microsoft",
                "FusedMatMul",
                inputs=[x, weight],
                attributes=[
                    ir.Attr("alpha", ir.AttributeType.FLOAT, 1.0),
                    ir.Attr("transB", ir.AttributeType.INT, 1),
                ],
                num_outputs=1,
                name=node.name,
            )
            out = node.outputs[0]
            result = fused.outputs[0]
            result.shape = out.shape
            result.type = out.type
            graph.insert_before(node, fused)
            out.replace_all_uses_with(result, replace_graph_outputs=True)
            name = out.name
            graph.remove(node, safe=True)
            transpose.graph.remove(transpose, safe=True)
            # Keep the name: the LM head's MatMul output is the ``logits``
            # graph output.
            result.name = name
            self.converted += 1

        if self.converted:
            model.opset_imports.setdefault("com.microsoft", 1)
            logger.debug("TransposedWeightMatMulPass: rewrote %d MatMul nodes", self.converted)
        return ir.passes.PassResult(model, modified=bool(self.converted))
