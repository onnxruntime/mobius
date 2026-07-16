# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""GLM-5/5.2 MoE backbone with MLA and a full-attention fallback.

GLM-5.2 uses Deep Sparse Attention (DSA) with IndexShare. This first export
increment preserves the exact MLA, MoE, shared-expert, routing, RoPE, and norm
backbone while evaluating every attention layer densely. IndexShare and the
checkpoint's final MTP layer are intentionally not exported yet.
"""

from __future__ import annotations

import re

from mobius._configs import ArchitectureConfig
from mobius.models.deepseek import DeepSeekV3CausalLMModel, DeepSeekV3TextModel


class GlmMoeDsaTextModel(DeepSeekV3TextModel):
    """GLM-MoE-DSA backbone using full MLA attention instead of sparse DSA."""


class GlmMoeDsaCausalLMModel(DeepSeekV3CausalLMModel):
    """GLM-5/5.2 causal LM with MLA, hybrid dense/MoE FFNs, and shared experts."""

    category: str = "Mixture of Experts"

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        self.model = GlmMoeDsaTextModel(config)

    def preprocess_weights(self, state_dict):
        """Map HF/GGUF GLM-MoE names to the shared DeepSeek-style modules."""
        filtered = {}
        for key, value in state_dict.items():
            match = re.search(r"\.layers\.(\d+)\.", key)
            if match is not None and int(match.group(1)) >= self.config.num_hidden_layers:
                continue
            if ".indexer." in key or ".nextn." in key:
                continue
            filtered[key] = value

        processed = super().preprocess_weights(filtered)
        return {
            key.replace(".mlp.experts.", ".mlp.moe.experts."): value
            for key, value in processed.items()
        }
