# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Pass that removes unused graph inputs.

After EP-aware optimization (e.g. GQA fusion absorbs RoPE), some graph
inputs may have zero consumers.  For example, ``position_ids`` becomes
dead when all attention layers use ``GroupQueryAttention`` with
``do_rotary=1``.  Removing dead inputs produces cleaner models and
avoids requiring the runtime to provide dummy feed values.

KV cache inputs (``past_key_values.*``) are always retained even if
they appear unused in the graph, because ORT GenAI manages them
externally via the KV cache protocol.

Sequence length inputs (``total_sequence_length``, ``past_sequence_length``)
are also retained because ORT GenAI uses them for graph capture bookkeeping
even though they may not be consumed by any node in the graph.
"""

from __future__ import annotations

import logging

import onnx_ir as ir

logger = logging.getLogger(__name__)

# Inputs managed by ORT GenAI that should not be removed even if unused
_RUNTIME_MANAGED_INPUTS = frozenset({
    "total_sequence_length",
})


class RemoveDeadGraphInputsPass(ir.passes.InPlacePass):
    """Remove graph inputs that have no consumers.

    Skips inputs whose name starts with ``past_key_values.`` (KV cache
    entries managed by the runtime), runtime-managed sequence length
    inputs, and inputs with ``None`` names.
    """

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        dead = [
            inp
            for inp in model.graph.inputs
            if inp.name is not None
            and not inp.name.startswith("past_key_values.")
            and inp.name not in _RUNTIME_MANAGED_INPUTS
            and len(inp.uses()) == 0
        ]
        for inp in dead:
            model.graph.inputs.remove(inp)
            logger.debug("Removed dead graph input: %s", inp.name)

        modified = len(dead) > 0
        return ir.passes.PassResult(model, modified=modified)
