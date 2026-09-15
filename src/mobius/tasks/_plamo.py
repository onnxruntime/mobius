# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Dynamic expanded-KV cache task for exact PLaMo GGUF imports."""

from __future__ import annotations

from mobius.tasks._causal_lm import CausalLMTask


class PlamoCausalLMTask(CausalLMTask):
    """PLaMo generation task with its source-level 40-head dynamic cache."""

    def __init__(self):
        super().__init__(static_cache=False, paged_cache=False)
