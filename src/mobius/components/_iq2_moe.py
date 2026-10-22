# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Sparse non-gated ReLU2 experts with byte-exact GGML IQ2_XXS banks.

Inputs are native rank-2/3 activations, selected INT64 expert IDs and final
FP32 mixing weights. Routing, shared experts and latent projections belong
to the caller. Requires the companion CUDA QMoE extension, not stock ORT.
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius._build_context import ep_capabilities


class GGMLIQ2XXSMoE(nn.Module):
    """Selected-expert execution without dense expansion or affine requantization."""

    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int, k: int):
        super().__init__()
        if (
            hidden_size <= 0
            or intermediate_size <= 0
            or hidden_size % 256
            or intermediate_size % 256
            or num_experts <= 0
            or not 1 <= k <= min(num_experts, 32)
        ):
            raise ValueError(
                "Sparse IQ2 MoE requires positive L/I multiples of 256, E>0 and "
                "1<=k<=min(E,32)"
            )
        self._hidden_size = hidden_size
        self._k = k
        self.up_proj = nn.Parameter(
            [num_experts, intermediate_size, hidden_size // 256, 66],
            dtype=ir.DataType.UINT8,
        )
        self.down_proj = nn.Parameter(
            [num_experts, hidden_size, intermediate_size // 256, 66],
            dtype=ir.DataType.UINT8,
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        routing_weights: ir.Value,
        selected_experts: ir.Value,
    ) -> ir.Value:
        caps = ep_capabilities()
        if hidden_states.dtype not in caps.iq2_moe_dtypes:
            raise ValueError(
                f"Sparse IQ2 MoE requires the companion CUDA QMoE extension; "
                f"EP={caps.name!r}, dtype={hidden_states.dtype} is unsupported"
            )
        if hidden_states.shape is not None:
            if len(hidden_states.shape) not in {2, 3}:
                raise ValueError("Sparse IQ2 MoE activations must have rank 2 or 3")
            width = hidden_states.shape[-1]
            if isinstance(width, int) and width != self._hidden_size:
                raise ValueError("Sparse IQ2 MoE activation width does not match its banks")
        if selected_experts.dtype != ir.DataType.INT64 or routing_weights.dtype != (
            ir.DataType.FLOAT
        ):
            raise ValueError(
                "Sparse IQ2 MoE requires INT64 IDs and final FLOAT routing weights"
            )
        for value in (selected_experts, routing_weights):
            if value.shape is not None:
                if len(value.shape) not in {2, 3}:
                    raise ValueError("Sparse IQ2 MoE selected routing must have rank 2 or 3")
                if isinstance(value.shape[-1], int) and value.shape[-1] != self._k:
                    raise ValueError("Sparse IQ2 MoE selected routing width does not match k")
        if (
            selected_experts.shape is not None
            and routing_weights.shape is not None
            and selected_experts.shape != routing_weights.shape
        ):
            raise ValueError(
                "Sparse IQ2 MoE selected IDs and weights must have matching shapes"
            )

        # The kernel preserves rank-2/3 X, but selected routing is always [T,k].
        selected = op.Reshape(selected_experts, [-1, self._k])
        weights = op.Reshape(routing_weights, [-1, self._k])
        inputs: list[ir.Value | nn.Parameter | None] = [None] * 23
        inputs[0] = hidden_states
        inputs[2] = self.up_proj
        inputs[5] = self.down_proj
        inputs[21] = selected
        inputs[22] = weights
        op.builder.graph.opset_imports["com.microsoft"] = 1
        result = op.QMoE(
            *inputs,
            quant_type="iq2_xxs",
            expert_weight_bits=2,
            block_size=256,
            activation_type="relu2",
            k=self._k,
            normalize_routing_weights=0,
            use_sparse_mixer=0,
            swiglu_fusion=0,
            weights_prepacked=0,
            _domain="com.microsoft",
        )
        # Custom contrib shape inference is unavailable in stock ONNX tooling.
        result.dtype = hidden_states.dtype
        if hidden_states.shape is not None:
            result.shape = ir.Shape(hidden_states.shape)
        node = result.producer()
        assert node is not None
        node.metadata_props["mobius.sparse_iq2_moe.runtime_requirement"] = (
            "companion CUDA QMoE IQ2_XXS extension; SM80+; no CUDA graph capture"
        )
        return result
