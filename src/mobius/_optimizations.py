# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Graph metadata analysis and post-weight materialization utilities."""

from __future__ import annotations

import logging

import onnx_ir as ir
import onnx_shape_inference
from onnx_ir.passes import common as common_passes

from mobius._passes import FoldConcatInitializersPass, FoldTransposedInitializerPass

logger = logging.getLogger(__name__)

__all__ = [
    "DEBUG_METADATA_KEYS",
    "DEBUG_METADATA_PREFIXES",
    "FUNCTIONAL_METADATA_PREFIX",
    "StripDebugMetadataPass",
    "SymbolicShapeInferencePass",
    "fold_initializers_after_weights",
    "strip_debug_metadata",
]


class SymbolicShapeInferencePass(ir.passes.InPlacePass):
    """Apply symbolic shape inference without changing graph computation."""

    def __init__(self, policy: onnx_shape_inference.ShapeMergePolicy = "refine"):
        super().__init__()
        self.policy = policy

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        try:
            onnx_shape_inference.infer_symbolic_shapes(model, policy=self.policy)
        except Exception:
            logger.warning(
                "Symbolic shape inference failed; preserving existing metadata", exc_info=True
            )
            return ir.passes.PassResult(model, modified=False)
        return ir.passes.PassResult(model, modified=True)


DEBUG_METADATA_PREFIXES: tuple[str, ...] = (
    "pkg.onnxscript.",
    "pkg.onnx_shape_inference.",
)
DEBUG_METADATA_KEYS: tuple[str, ...] = ("namespace",)
FUNCTIONAL_METADATA_PREFIX = "mobius."


def _is_debug_metadata(key: str) -> bool:
    return key.startswith(DEBUG_METADATA_PREFIXES) or key in DEBUG_METADATA_KEYS


class StripDebugMetadataPass(ir.passes.InPlacePass):
    """Remove recognized build-time provenance metadata."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        removed = 0

        def strip(props) -> None:
            nonlocal removed
            for key in [key for key in props if _is_debug_metadata(key)]:
                assert not key.startswith(FUNCTIONAL_METADATA_PREFIX), (
                    f"{key!r} is both functional and debug metadata; the two "
                    "namespaces must stay disjoint"
                )
                del props[key]
                removed += 1

        seen: set[int] = set()
        for node in model.graph.all_nodes():
            strip(node.metadata_props)
            for value in (*node.inputs, *node.outputs):
                if value is not None and id(value) not in seen:
                    seen.add(id(value))
                    strip(value.metadata_props)

        return ir.passes.PassResult(model, modified=bool(removed))


def strip_debug_metadata(model: ir.Model) -> ir.Model:
    """Strip recognized build-time provenance metadata in place."""
    return StripDebugMetadataPass()(model).model


def fold_initializers_after_weights(model: ir.Model) -> None:
    """Materialize weight-only Transpose and Concat operations after loading."""
    ir.passes.PassManager(
        [
            common_passes.LiftConstantsToInitializersPass(),
            FoldConcatInitializersPass(),
            FoldTransposedInitializerPass(),
            common_passes.RemoveUnusedNodesPass(),
        ]
    )(model)
