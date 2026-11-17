# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

from onnxscript import OpBuilder, nn

if TYPE_CHECKING:
    import onnx_ir as ir


def silu(op: OpBuilder, x):
    """SiLU (Swish) activation."""
    return op.Swish(x)


def gelu(op: OpBuilder, x):
    """GELU activation using the 'none' approximation."""
    return op.Gelu(x)


def gelu_tanh(op: OpBuilder, x):
    """GELU activation using the tanh approximation."""
    return op.Gelu(x, approximate="tanh")


def relu(op: OpBuilder, x):
    """ReLU activation."""
    return op.Relu(x)


def relu2(op: OpBuilder, x):
    """Squared ReLU activation: relu(x)^2."""
    r = op.Relu(x)
    return op.Mul(r, r)


def quick_gelu(op: OpBuilder, x):
    """QuickGELU activation: x * sigmoid(1.702 * x)."""
    return op.Mul(x, op.Sigmoid(op.Mul(1.702, x)))


def mish(op: OpBuilder, x):
    """Mish activation: x * tanh(softplus(x))."""
    softplus = op.Softplus(x)
    return op.Mul(x, op.Tanh(softplus))


def sigmoid(op: OpBuilder, x):
    """Sigmoid activation."""
    return op.Sigmoid(x)


def tanh(op: OpBuilder, x):
    """Tanh activation."""
    return op.Tanh(x)


def linear(op: OpBuilder, x):
    """Linear (identity) activation."""
    return x


# Mapping from activation string names to functions.
# Each function takes (op: OpBuilder, x) and returns the activated tensor.
ACT2FN: OrderedDict[str, callable] = OrderedDict(
    {
        "gegelu": gelu,
        "gelu": gelu,
        "gelu_fast": gelu_tanh,
        "gelu_new": gelu_tanh,
        "gelu_pytorch_tanh": gelu_tanh,
        "linear": linear,
        "mish": mish,
        "quick_gelu": quick_gelu,
        "relu": relu,
        "relu2": relu2,
        "sigmoid": sigmoid,
        "silu": silu,
        "swish": silu,
        "tanh": tanh,
        # xielu is a parametric learnable activation (used by Apertus).
        # Mapped to silu here so CausalLMModel.__init__ can construct the
        # base MLP without crashing. ApertusCausalLMModel immediately
        # replaces the MLP with ApertusFCMLP that uses the real
        # XIELUActivation nn.Module (with alpha_p/alpha_n parameters).
        "xielu": silu,
    }
)


def get_activation(activation_string: str | None):
    """Get an activation function by name.

    Returns a callable with signature (op: OpBuilder, x) -> result.

    Raises:
        ValueError: If *activation_string* is ``None``.
        KeyError: If the name is not found in :data:`ACT2FN`.
    """
    if activation_string is None:
        raise ValueError(
            f"hidden_act is None — set it to one of {list(ACT2FN.keys())} in the model config"
        )
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    raise KeyError(
        f"function {activation_string!r} not found in ACT2FN mapping {list(ACT2FN.keys())}"
    )


class SiLU(nn.Module):
    """SiLU (Swish) activation as an nn.Module.

    Useful as a child in ``nn.Sequential`` where a Module is needed.
    """

    def forward(self, op: OpBuilder, x: ir.Value):
        return silu(op, x)
