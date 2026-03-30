# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from mobius.models.base import CausalLMModel


class Llama4CausalLMModel(CausalLMModel):
    """Llama4 text model.

    Extends the base CausalLMModel with a weight rename:
    HF Llama4 names MLP layers ``feed_forward.*`` while our graph builder
    uses ``mlp.*``.  ``preprocess_weights`` maps the HF names to the ONNX
    parameter names so that weight transfer works correctly.

    Replicates ``meta-llama/Llama-4-Scout-17B-16E-Instruct``.
    """

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        state_dict = super().preprocess_weights(state_dict)
        # HF uses layers.{i}.feed_forward.*; ONNX uses layers.{i}.mlp.*
        return {k.replace(".feed_forward.", ".mlp."): v for k, v in state_dict.items()}
