# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Apertus causal language model.

Uses FCMLP with learnable xIELU activation, QK-norm, and renamed layer
norms (``attention_layernorm`` / ``feedforward_layernorm``).

Replicates HuggingFace ``ApertusForCausalLM``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius.components import RMSNormBias
from mobius.models.base import CausalLMModel, linear_class_for_config


class XIELUActivation(nn.Module):
    """xIELU activation with learnable alpha_p and alpha_n parameters.

    For x > 0: softplus(alpha_p) * x² + beta * x
    For x ≤ 0: (expm1(min(x, eps)) - x) * (beta + softplus(alpha_n)) + beta * x

    Parameters alpha_p and alpha_n are stored in Softplus-inverse space.
    """

    def __init__(
        self,
        *,
        alpha_p: float | None = None,
        alpha_n: float | None = None,
        beta: float | None = None,
        eps: float | None = None,
    ):
        super().__init__()
        values = (alpha_p, alpha_n, beta, eps)
        parameters = [
            nn.Parameter(
                [1],
                data=None
                if value is None
                else ir.tensor(np.asarray([value], dtype=np.float32)),
            )
            for value in values
        ]
        self.alpha_p, self.alpha_n, self.beta, self.eps = parameters

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        alpha_p = op.CastLike(self.alpha_p, x)
        alpha_n = op.CastLike(self.alpha_n, x)
        beta = op.CastLike(self.beta, x)
        eps = op.CastLike(self.eps, x)
        alpha_p_act = op.Softplus(alpha_p)
        alpha_n_act = op.Add(op.Softplus(alpha_n), beta)

        # Common sub-expression: beta * x
        beta_x = op.Mul(beta, x)

        # Positive branch: softplus(alpha_p) * x² + beta * x
        pos_branch = op.Add(op.Mul(alpha_p_act, op.Mul(x, x)), beta_x)

        # Negative branch: (expm1(clamp(x, max=eps)) - x) * alpha_n + beta * x
        x_clipped = op.Min(x, eps)
        neg_branch = op.Add(
            op.Mul(op.Sub(op.Sub(op.Exp(x_clipped), 1.0), x), alpha_n_act),
            beta_x,
        )

        # Select branch based on sign of x
        zero = op.CastLike(0.0, x)
        return op.Where(op.Greater(x, zero), pos_branch, neg_branch)


class ApertusFCMLP(nn.Module):
    """Apertus MLP: up_proj → xIELU → down_proj."""

    def __init__(self, config: ArchitectureConfig, layer_index: int):
        super().__init__()
        linear_class = linear_class_for_config(config)
        if linear_class is None:
            from mobius.components import Linear

            linear_class = Linear
        self.up_proj = linear_class(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = linear_class(config.intermediate_size, config.hidden_size, bias=False)

        def layer_value(values: tuple[float, ...] | None) -> float | None:
            return None if values is None else values[layer_index]

        self.act_fn = XIELUActivation(
            alpha_p=layer_value(config.xielu_alpha_p),
            alpha_n=layer_value(config.xielu_alpha_n),
            beta=layer_value(config.xielu_beta),
            eps=layer_value(config.xielu_eps),
        )

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        x = self.up_proj(op, x)
        x = self.act_fn(op, x)
        return self.down_proj(op, x)


class ApertusCausalLMModel(CausalLMModel):
    """Apertus model with xIELU activation, QK-norm, and custom norm naming."""

    def __init__(self, config: ArchitectureConfig):
        config = dataclasses.replace(config, attn_qk_norm=True)
        super().__init__(config)
        for layer_index, layer in enumerate(self.model.layers):
            layer.mlp = ApertusFCMLP(config, layer_index)
            if config.attn_q_norm_biases and config.attn_q_norm_biases[layer_index]:
                layer.self_attn.q_norm = RMSNormBias(config.head_dim, eps=config.rms_norm_eps)
            if config.attn_k_norm_biases and config.attn_k_norm_biases[layer_index]:
                layer.self_attn.k_norm = RMSNormBias(config.head_dim, eps=config.rms_norm_eps)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Rename HF Apertus weights to match ONNX model structure.

        - attention_layernorm → input_layernorm
        - feedforward_layernorm → post_attention_layernorm
        - act_fn scalars (alpha_p, alpha_n, beta, eps): reshape to [1]
        """
        new_state_dict = {}
        for name, tensor in state_dict.items():
            # Rename layer norms
            name = name.replace(".attention_layernorm.", ".input_layernorm.")
            name = name.replace(".feedforward_layernorm.", ".post_attention_layernorm.")

            # xIELU params: HF has scalar [1], ONNX nn.Parameter is [1]
            if name.endswith(
                (
                    ".mlp.act_fn.alpha_p",
                    ".mlp.act_fn.alpha_n",
                    ".mlp.act_fn.beta",
                    ".mlp.act_fn.eps",
                )
            ):
                new_state_dict[name] = tensor.reshape(1)
                continue

            new_state_dict[name] = tensor
        return super().preprocess_weights(new_state_dict)
