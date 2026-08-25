# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Arcee causal language model.

Identical to the standard Llama-family CausalLMModel **except** the MLP
is non-gated: ``up_proj → relu² → down_proj`` (two-matrix :class:`FCMLP`
instead of three-matrix gated :class:`MLP`).

Replicates HuggingFace ``ArceeForCausalLM``.
"""

from __future__ import annotations

from mobius._configs import ArchitectureConfig
from mobius.components import FCMLP
from mobius.models.base import CausalLMModel, linear_class_for_config


class ArceeCausalLMModel(CausalLMModel):
    """Arcee model: standard Llama attention + FCMLP with relu2.

    HuggingFace weight names (up_proj / down_proj) match FCMLP convention,
    so no ``preprocess_weights`` overrides are needed.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        # Replace gated MLP → non-gated FCMLP in every decoder layer
        for layer in self.model.layers:
            layer.mlp = FCMLP(
                config.hidden_size,
                config.intermediate_size,
                activation=config.hidden_act,
                bias=config.mlp_bias,
                linear_class=linear_class_for_config(config),
            )
