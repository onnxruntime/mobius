"""Llama4 causal language model.

Llama4 uses chunked/interleaved attention (alternating full and local attention
windows) combined with Mixture-of-Experts layers.  The implementation currently
falls back to the standard CausalLMModel; full Llama4-specific attention and
MoE routing are tracked in a future task.
"""

from __future__ import annotations

from mobius.models.base import CausalLMModel

# Llama4 shares the same baseline architecture as Llama but with additional
# chunked-attention and MoE components not yet fully implemented.
Llama4CausalLMModel = CausalLMModel
