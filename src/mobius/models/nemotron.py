# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from mobius._configs import ArchitectureConfig
from mobius.components import FCMLP, LayerNorm1P
from mobius.models.base import CausalLMModel


class NemotronCausalLMModel(CausalLMModel):
    """Nemotron model using LayerNorm1P and FCMLP (no gating).

    Nemotron uses ``NemotronLayerNorm1P`` where the learned weight is a delta
    around 1: ``effective_scale = weight + 1``.  The model also uses a simpler
    non-gated MLP: up → activation → down.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        # Final norm and per-layer norms all use the +1 offset variant
        self.model.norm = LayerNorm1P(config.hidden_size, eps=config.rms_norm_eps)
        for layer in self.model.layers:
            layer.mlp = FCMLP(
                config.hidden_size,
                config.intermediate_size,
                activation=config.hidden_act,
                bias=config.mlp_bias,
            )
            layer.input_layernorm = LayerNorm1P(config.hidden_size, eps=config.rms_norm_eps)
            layer.post_attention_layernorm = LayerNorm1P(
                config.hidden_size, eps=config.rms_norm_eps
            )
