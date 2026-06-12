# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Graph-IO contract for the Gemma4-Assistant speculative-decoding drafter.

**Status: scaffolding only.**  See the implementation checklist in
``mobius/models/gemma4_assistant.py``.  This task is registered under
``"gemma4-assistant"`` and dispatches to
:class:`~mobius.models.Gemma4AssistantCausalLMModel`, but its ``build``
method raises :class:`NotImplementedError` until the model's ``forward``
is implemented.

Planned graph IO contract:

Inputs:
    - ``inputs_embeds``: ``[batch, q_len, 2 * backbone_hidden_size]`` (model dtype).
      Concatenation of the target's previous and current shared hidden
      states.
    - ``position_ids``: ``[batch, q_len]`` INT64.
    - ``attention_mask``: ``[batch, kv_len + q_len]`` INT64 OR a float
      bidirectional mask — TBD when the bidirectional masking is wired up.
    - For each layer type used by the assistant (any of ``full_attention``,
      ``sliding_attention``):
        - ``shared_kv.{layer_type}.key``: ``[batch, num_kv_heads,
          kv_len, head_dim]`` (model dtype).
        - ``shared_kv.{layer_type}.value``: same shape.

Outputs:
    - ``logits``: ``[batch, q_len, vocab_size]`` (model dtype).
    - ``projected_state``: ``[batch, q_len, backbone_hidden_size]`` (model
      dtype) — the post_projection of last_hidden_state, fed back to the
      target for the next speculative step.

No ``past_key_values.*`` / ``present.*`` — the assistant has no KV cache
of its own; every speculative step recomputes its full attention over
the (constant per step) shared target K/V plus the drafted-so-far Q.
"""

from __future__ import annotations

from onnxscript import nn

from mobius._configs import Gemma4AssistantConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask


class Gemma4AssistantTask(ModelTask):
    """Build graph for the Gemma4-Assistant draft model.

    Scaffolding only — raises ``NotImplementedError`` until the
    drafter's ``forward`` is implemented.  See module docstring for the
    planned IO contract.
    """

    def build(
        self,
        module: nn.Module,
        config: Gemma4AssistantConfig,
    ) -> ModelPackage:
        raise NotImplementedError(
            "Gemma4AssistantTask.build is not yet implemented.  Complete "
            "Gemma4AssistantCausalLMModel.forward first; see the checklist "
            "in mobius/models/gemma4_assistant.py."
        )
