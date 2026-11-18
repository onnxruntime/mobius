# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CLM heads matching Contrastive-LM/CLM's make_head and HeadPair.

Reference source: bb42c6c5bf914fd449bed2f6ca65be80602cb1f7, src/clm/heads.py.
Inputs are already L2-normalized, last-token-pooled Qwen3-8B embeddings.
Outputs are separate normalized projections, scaled cosine logits and
candidate-relative probabilities. No text encoder or generation is included.
"""

from __future__ import annotations

import onnx_ir as ir
from onnxscript import OpBuilder, nn

from mobius._configs.clm import CLMConfig
from mobius.components import LayerNorm, Linear


class _CLMHead(nn.Module):
    def __init__(self, config: CLMConfig):
        super().__init__()
        self.config = config
        self.inp = Linear(config.hidden_size, config.width)
        self.hidden = nn.ModuleList(
            [Linear(config.width, config.width) for _ in range(config.depth - 2)]
        )
        self.norms = nn.ModuleList(
            [LayerNorm(config.width, eps=1e-5) for _ in self.hidden]
            if config.layernorm
            else []
        )
        self.out = Linear(config.width, config.projection_dim)

    def _activate(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        if self.config.activation == "gelu":
            return op.Gelu(x, approximate="none")
        if self.config.activation == "relu":
            return op.Relu(x)
        return op.Mul(x, op.Sigmoid(x))

    def forward(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        x = self._activate(op, self.inp(op, x))
        for index, linear in enumerate(self.hidden):
            h = linear(op, x)
            if self.config.layernorm:
                h = self.norms[index](op, h)
            h = self._activate(op, h)
            x = op.Add(x, h) if self.config.residual else h
        x = self.out(op, x)  # (rows, projection_dim)
        norm = op.Sqrt(op.ReduceSum(op.Mul(x, x), [-1], keepdims=1))
        return op.Div(x, op.Max(norm, 1e-12))


class CLMRankingModel(nn.Module):
    """Independent CLM state/action heads for candidate-relative ranking.

    Replicates upstream HeadPair projection and Engine's default-temperature
    scoring, not an AutoModelForCausalLM or the frozen Qwen3 text encoder.
    """

    default_task = "contrastive-ranking-heads"
    category = "Text Ranking"
    config_class = CLMConfig

    def __init__(self, config: CLMConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.state_head = _CLMHead(config)
        self.action_head = _CLMHead(config)

    def forward(self, op: OpBuilder, state_embeddings, action_embeddings):
        states = self.state_head(op, state_embeddings)
        actions = self.action_head(op, action_embeddings)
        # All states share the candidate set; softmax is independent per state.
        logits = op.Mul(
            op.MatMul(states, op.Transpose(actions, perm=[1, 0])), self.config.scale
        )  # (states, candidates)
        return states, actions, logits, op.Softmax(logits, axis=-1)
