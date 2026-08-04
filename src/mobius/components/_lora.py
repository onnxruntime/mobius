# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""LoRA (Low-Rank Adaptation) components."""

from __future__ import annotations

import numpy
import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius.components._common import Linear


class LoRALinear(Linear):
    """Linear layer with LoRA (Low-Rank Adaptation) adapters.

    Each adapter adds a low-rank contribution: ``(x @ A^T) @ B^T * scale``.
    Multiple named adapters (e.g. "vision", "speech") are summed.

    **Per-modality gating.** Models such as Phi4MM activate exactly one
    adapter (or none) depending on the input modality — HuggingFace selects
    ``vision`` for image/image+audio inputs, ``speech`` for audio-only
    inputs, and *no* adapter for text-only inputs.  To replicate this in a
    single static ONNX graph, an optional ``gate_holder`` maps adapter name
    to a runtime scalar ``ir.Value`` (``1.0`` = active, ``0.0`` = inactive).
    Each adapter's contribution is multiplied by its gate before being
    summed into the result.  When ``gate_holder`` is ``None`` (or a given
    adapter has no gate), the adapter is applied ungated (legacy behaviour).

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        bias: Whether to include a bias term.
        lora_adapters: List of ``(name, rank, scale)`` tuples.
            Each adapter creates ``lora_A.{name}.weight`` and
            ``lora_B.{name}.weight`` parameters matching HuggingFace naming.
        gate_holder: Optional shared mapping ``{adapter_name: ir.Value}``.
            Populated by the owning model's ``forward`` (before the decoder
            layer loop) so every ``LoRALinear`` instance reads the same
            runtime gate values.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        lora_adapters: list[tuple[str, int, float]] | None = None,
        gate_holder: dict[str, ir.Value] | None = None,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self._adapters: list[tuple[str, nn.Parameter]] = []
        self._gate_holder = gate_holder
        if lora_adapters:
            for name, rank, scale in lora_adapters:
                # Parameters are named to match HuggingFace convention:
                #   lora_A.{name}.weight  and  lora_B.{name}.weight
                setattr(
                    self,
                    f"_lora_A_{name}",
                    nn.Parameter([rank, in_features], name=f"lora_A.{name}.weight"),
                )
                setattr(
                    self,
                    f"_lora_B_{name}",
                    nn.Parameter([out_features, rank], name=f"lora_B.{name}.weight"),
                )
                # Store scale as a named Parameter so each module gets a unique
                # initializer (avoids constant name collisions in the graph).
                scale_param = nn.Parameter(
                    [],
                    name=f"lora_scale.{name}",
                    data=ir.tensor(numpy.array(scale, dtype=numpy.float32)),
                )
                setattr(self, f"_lora_scale_{name}", scale_param)
                self._adapters.append((name, scale_param))

    def forward(self, op: OpBuilder, x: ir.Value):
        result = super().forward(op, x)
        for name, scale_param in self._adapters:
            lora_a = getattr(self, f"_lora_A_{name}")
            lora_b = getattr(self, f"_lora_B_{name}")
            h = op.MatMul(x, op.Transpose(lora_a, perm=[1, 0]))
            lora_out = op.MatMul(h, op.Transpose(lora_b, perm=[1, 0]))
            lora_out = op.Mul(lora_out, scale_param)
            # Per-modality gate: zero out this adapter when inactive for the
            # current input modality (broadcasts a scalar over the output).
            if self._gate_holder is not None:
                gate = self._gate_holder.get(name)
                if gate is not None:
                    lora_out = op.Mul(lora_out, gate)
            result = op.Add(result, lora_out)
        return result
