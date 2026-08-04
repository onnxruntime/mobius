# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components._activations import get_activation
from mobius.components._common import Linear

if TYPE_CHECKING:
    import onnx_ir as ir


class MLP(nn.Module):
    """Feed-forward network with gated linear units (GLU-style).

    Three-matrix architecture: ``gate_proj → activation → elementwise mul
    with up_proj → down_proj``.  Used by Llama, Qwen, Mistral, etc.

    Args:
        config: Architecture configuration.
        linear_class: Factory callable ``(in_features, out_features, bias=...)``
            for creating projection layers. Defaults to ``Linear``.
    """

    def __init__(self, config: ArchitectureConfig, linear_class: type | None = None):
        super().__init__()
        if linear_class is None:
            linear_class = Linear
        self.gate_proj = linear_class(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = linear_class(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = linear_class(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias
        )
        self.act_fn = get_activation(config.hidden_act)

    def forward(self, op: OpBuilder, x: ir.Value):
        gate = self.act_fn(op, self.gate_proj(op, x))
        up = self.up_proj(op, x)
        return self.down_proj(op, op.Mul(gate, up))


class GatedMLP(nn.Module):
    """Gated MLP: ``act(gate_proj(x)) * up_proj(x) → down_proj``.

    Implements SwiGLU / GeGLU / GateMLP patterns where the hidden
    dimension is split into a gate branch and an up branch.

    **When to use GatedMLP vs MLP:** Both implement the same gated forward
    pass, but differ in how dimensions are specified at construction time.
    Use ``GatedMLP`` when the model config provides ``hidden_size`` and
    ``intermediate_size`` directly (vision encoders, codec models, etc.).
    Use :class:`MLP` when the model dimensions come from an
    :class:`~mobius._configs.ArchitectureConfig` object (language model
    decoder layers).

    Default parameter names are ``gate_proj`` / ``up_proj`` / ``down_proj``.
    Models with different HuggingFace weight names should rename in
    ``preprocess_weights()``.

    Args:
        hidden_size: Input/output dimension.
        intermediate_size: Hidden dimension of the gate/up branches.
        activation: Activation applied to the gate branch (default
            ``"silu"`` for SwiGLU).
        bias: Whether to include bias in all projection layers.
        linear_class: Factory callable for creating linear layers.
            Defaults to ``Linear``.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "silu",
        bias: bool = False,
        linear_class: type | None = None,
    ):
        super().__init__()
        if linear_class is None:
            linear_class = Linear
        self.gate_proj = linear_class(hidden_size, intermediate_size, bias=bias)
        self.up_proj = linear_class(hidden_size, intermediate_size, bias=bias)
        self.down_proj = linear_class(intermediate_size, hidden_size, bias=bias)
        self.act_fn = get_activation(activation)

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        gate = self.act_fn(op, self.gate_proj(op, x))
        return self.down_proj(op, op.Mul(gate, self.up_proj(op, x)))


class FusedGateUpMLP(nn.Module):
    """Gated MLP that keeps gate and up projections as one fused weight.

    The ``gate_up_proj`` tensor (shape ``[2*intermediate_size, hidden_size]``)
    is stored as a single parameter, matching HuggingFace checkpoints for
    Phi-3, Phi-4, and GLM family models. Activations are split after the
    fused MatMul — no weight splitting is required in ``preprocess_weights``.

    Forward pass::

        gate_up = gate_up_proj(x)           # [*, 2 * intermediate_size]
        gate, up = split(gate_up, axis=-1)  # each [*, intermediate_size]
        return down_proj(act(gate) * up)

    Use this instead of :class:`MLP` when the HuggingFace checkpoint stores
    gate and up projections as a single fused ``gate_up_proj`` weight.  This
    avoids splitting the weight tensor at load time, which fails for GPTQ
    int32-packed weights where dim 0 is ``original / pack_factor``.

    Args:
        config: Architecture configuration.
        linear_class: Factory callable ``(in_features, out_features, bias=...)``
            for creating projection layers. Defaults to ``Linear``.
    """

    def __init__(self, config: ArchitectureConfig, linear_class: type | None = None):
        super().__init__()
        if linear_class is None:
            linear_class = Linear
        # Single fused weight: [2*intermediate_size, hidden_size]
        self.gate_up_proj = linear_class(
            config.hidden_size, 2 * config.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = linear_class(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias
        )
        self.act_fn = get_activation(config.hidden_act)

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        # Fused matmul → [*, 2 * intermediate_size]
        gate_up = self.gate_up_proj(op, x)
        # Split activations at intermediate_size boundary
        gate, up = op.Split(gate_up, axis=-1, num_outputs=2, _outputs=2)
        # gated activation (e.g. SiLU(gate) * up) → down_proj
        return self.down_proj(op, op.Mul(self.act_fn(op, gate), up))


class FCMLP(nn.Module):
    """Two-layer fully-connected MLP: ``up_proj → activation → down_proj``.

    Used by models that do NOT use gated linear units (ViT, CLIP, GPT-2,
    Falcon, DistilBERT, Wav2Vec2, BART, Whisper, Nemotron, etc.).

    Default parameter names are ``up_proj`` / ``down_proj``.  Models with
    different HuggingFace weight names (e.g. ``fc1``/``fc2``,
    ``c_fc``/``c_proj``) should rename in ``preprocess_weights()``.

    Note: ``up_proj``/``down_proj`` was chosen over ``fc1``/``fc2`` for
    consistency with the gated :class:`MLP` (``gate_proj``/``up_proj``/
    ``down_proj``) and the broader LLM ecosystem (Llama, Qwen, Mistral).
    HuggingFace models use many different names (fc1/fc2, lin1/lin2,
    c_fc/c_proj, dense_h_to_4h/dense_4h_to_h, etc.) so no single choice
    avoids renames for most models — 7 of 10 consolidated models need
    ``preprocess_weights`` renames regardless.

    Args:
        hidden_size: Input/output dimension.
        intermediate_size: Hidden dimension of the inner layer.
        activation: Activation function name (e.g. ``"gelu"``,
            ``"quick_gelu"``, ``"relu"``).  Defaults to ``"gelu"``.
        bias: Whether to include bias in both linear layers.
        linear_class: Factory callable for creating linear layers.
            Defaults to ``Linear``.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "gelu",
        bias: bool = True,
        linear_class: type | None = None,
    ):
        super().__init__()
        if linear_class is None:
            linear_class = Linear
        self.up_proj = linear_class(hidden_size, intermediate_size, bias=bias)
        self.down_proj = linear_class(intermediate_size, hidden_size, bias=bias)
        self.act_fn = get_activation(activation)

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        x = self.up_proj(op, x)
        x = self.act_fn(op, x)
        return self.down_proj(op, x)
