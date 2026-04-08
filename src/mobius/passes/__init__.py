# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

"""Graph optimization passes for mobius ONNX models.

This module provides IR passes that operate on the ONNX graph structure,
complementing the rewrite-rule infrastructure in :mod:`mobius.rewrite_rules`.

Passes here are designed to run **after weights are loaded** so that
initializer data is available for pre-computation.  All passes use
:class:`onnx_ir.LazyTensor` to defer tensor evaluation until the model is
serialized, avoiding memory spikes from materializing many large weight
tensors simultaneously.

Available passes
----------------
:class:`FoldTransposedInitializerPass`
    Converts ``Transpose(initializer, perm=[1, 0])`` patterns into
    pre-transposed initializers.  Eliminates runtime transpose overhead from
    every :class:`~mobius.components.Linear` layer.

:class:`FoldConcatInitializersPass`
    Folds ``Concat(init_0, init_1, ...)`` patterns (where every input is a
    graph initializer) into a single pre-concatenated initializer.  Used to
    pre-pack QKV weight matrices after weight loading.
"""

from __future__ import annotations

__all__ = [
    "FoldTransposedInitializerPass",
    "FoldConcatInitializersPass",
]

from mobius.passes._fold_concat import FoldConcatInitializersPass
from mobius.passes._fold_transpose import FoldTransposedInitializerPass
