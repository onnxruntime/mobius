"""Mobius port of the Gemma4-Assistant speculative-decoding draft model.

The Gemma4-Assistant family (``google/gemma-4-{E2B,E4B,12B,26B,31B}-it-assistant``)
is a small Gemma4-style decoder used as the drafter in HuggingFace's
``assistant_model=`` speculative decoding API.  Architecturally it is:

    pre_projection (2 * backbone_hidden → assistant_hidden, no bias)
    →  Gemma4TextModel-like stack of decoder layers, **every layer
       KV-shared with the target** (per-layer-type external KV inputs)
    →  final RMSNorm
    →  (logits, projected_state):
         logits           = lm_head(last_hidden_state)              # [B, q, vocab]
         projected_state  = post_projection(last_hidden_state)      # [B, q, backbone]

Where:
- ``inputs_embeds`` is fed in by the target; the size is
  ``2 * backbone_hidden_size`` because the target concatenates the previous
  and current shared hidden states before handing them to the drafter.
- ``shared_kv_states`` is a per-layer-type dict (``"full_attention"``,
  ``"sliding_attention"``) of ``(K, V)`` tensors taken from the target's
  designated KV-share-source layer of each type.  The drafter's attention
  layers attend over these shared K/V plus their own Q for the new tokens.
- The drafter uses **bidirectional** (non-causal) masks across the
  ``[shared_kv ‖ assistant_q]`` axis (see
  ``transformers.models.gemma4_assistant.modeling_gemma4_assistant.Gemma4AssistantForCausalLM.create_attention_masks``).
- The drafter has no KV cache of its own — every spec-decoding draft step
  recomputes its full attention over the (constant per step) shared
  target K/V plus the drafted-so-far Q tokens.

----------------------------------------------------------------------------
Implementation status — autonomous Phase 2 (gemma4):
----------------------------------------------------------------------------

This file ships the **scaffolding**: config wiring, model class with the
weight modules in place, and a forward stub that raises
``NotImplementedError`` with a precise checklist of the remaining work.

Why a stub instead of a full implementation?  The existing mobius Gemma4
code (``mobius/models/gemma4.py``) is ~3000 lines with delicate
assumptions baked into ``Gemma4TextAttention`` /
``Gemma4DecoderLayer`` / ``Gemma4TextModel`` — most notably the
``is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0`` guard
that returns False for the all-layers-shared case the assistant needs,
plus the internal-only ``shared_kv_states: dict`` collection that
source layers populate and shared layers consume (no path for
externally-supplied K/V).  Either route — modifying ``Gemma4TextModel``
to support external KV, or reimplementing the attention/layer stack as a
standalone Gemma4Assistant variant — is meaningful work that should not
be merged without interactive validation against the upstream PyTorch
reference (see ``archive/draft/phase2_5_drafter_parity.ipynb`` for the
DFlash parity pattern to mirror).

The DFlash drafter (commit ``d9b8b7b``) followed exactly the same pattern
and is the right reference for how to lay out the components, task IO,
and parity notebook for Gemma4-Assistant.  See its file structure:
- ``mobius/components/_dflash.py``        (cross-attention component)
- ``mobius/models/dflash.py``             (model class)
- ``mobius/tasks/_dflash.py``             (task IO contract)
- ``mobius/models/_dflash_test.py``       (unit tests)
- ``archive/draft/phase2_5_drafter_parity.ipynb`` (parity validation)

What's already done in this scaffolding:
- ``Gemma4AssistantConfig`` (in ``_configs/_base.py``): fully lifts the
  nested ``text_config`` fields onto the top level, plus the assistant-
  specific knobs.  Validates constraints (all layers shared, no MoE,
  no double-wide MLP, no per-layer inputs).  ✅
- Registry entry for the ``Gemma4AssistantForCausalLM`` architecture and
  the ``gemma4-assistant`` task name.  ✅
- ``Gemma4AssistantCausalLMModel`` class with ``pre_projection``,
  ``lm_head``, ``post_projection``, and the standard final ``RMSNorm``
  registered as ``nn.Parameter``s so weight loading is unambiguous.  ✅
- Tests for the config extraction and weight-module layout.  ✅

What's left to do (clear checklist for the follow-up commit):
- [ ] Decide between (a) extending ``Gemma4TextAttention`` to accept
      externally-supplied K/V via the existing ``shared_kv_states`` dict
      with string keys (``"full_attention"`` / ``"sliding_attention"``)
      vs (b) writing a standalone ``Gemma4AssistantAttention``.  Option
      (a) is less code but risks regressions in the existing Gemma4
      variants; option (b) is safer and recommended for the first pass.
- [ ] Implement the actual ``forward`` of ``Gemma4AssistantCausalLMModel``:
      pre_projection → attention stack with external K/V → norm →
      (lm_head, post_projection).
- [ ] Implement ``Gemma4AssistantTask`` IO contract: graph inputs are
      ``inputs_embeds``, ``position_ids``, ``attention_mask``, and per-
      layer-type ``shared_kv.full_attention.key``,
      ``shared_kv.full_attention.value``, ``shared_kv.sliding_attention.key``,
      ``shared_kv.sliding_attention.value``; outputs are ``logits`` and
      ``projected_state``.  No own KV cache (no ``past_key_values.*`` /
      ``present.*``).
- [ ] Decide whether to support ``use_ordered_embeddings`` (the
      ``Gemma4AssistantMaskedEmbedder`` centroid-routed LM head).  For
      the E2B-it-assistant the config has ``use_ordered_embeddings:
      true``, so this is required for full parity with the released
      checkpoint.  It scatters logits into a sparse vocab buffer via
      top-k centroid routing — non-trivial in ONNX.
- [ ] Build the parity notebook
      ``archive/draft/phase2_5_gemma4_assistant_parity.ipynb`` that
      mirrors the DFlash one: load real HF assistant weights into the
      mobius ONNX, generate identical inputs (run target prefill to get
      shared K/V), compare ``logits`` + ``projected_state`` against
      upstream PyTorch in fp32.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import OpBuilder, nn

from mobius._configs import Gemma4AssistantConfig
from mobius.components import Linear, RMSNorm

if TYPE_CHECKING:
    import onnx_ir as ir


class Gemma4AssistantCausalLMModel(nn.Module):
    """Gemma4-Assistant speculative-decoding draft model (scaffolding).

    See module docstring for the architecture overview and the
    implementation checklist for completing this class.

    Weight modules registered here (so weight loading from the HF
    checkpoint can be tested ahead of the forward implementation):

    - ``pre_projection``: ``Linear(2 * backbone_hidden_size, hidden_size, bias=False)``
    - ``post_projection``: ``Linear(hidden_size, backbone_hidden_size, bias=False)``
    - ``lm_head``: ``Linear(hidden_size, vocab_size, bias=False)``
    - ``norm``: ``RMSNorm(hidden_size, eps=config.rms_norm_eps)``

    The internal Gemma4-style decoder stack (``self.model.layers``,
    ``self.model.embed_tokens``, ``self.model.norm``, ``self.model.rotary_emb_*``)
    is intentionally **not** built here — adding it requires the design
    decision documented in the module-level implementation checklist.
    """

    config_class: type = Gemma4AssistantConfig
    default_task: str = "gemma4-assistant"
    category: str = "Text Generation"

    def __init__(self, config: Gemma4AssistantConfig):
        super().__init__()
        self.config = config
        # ``pre_projection`` takes the target's concatenated hidden state of
        # shape ``[B, q, 2 * backbone_hidden]`` (previous ‖ current); see
        # upstream ``Gemma4AssistantForCausalLM.__init__``.
        self.pre_projection = Linear(
            2 * config.backbone_hidden_size, config.hidden_size, bias=False
        )
        self.post_projection = Linear(
            config.hidden_size, config.backbone_hidden_size, bias=False
        )
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Tied weights: upstream sets ``lm_head.weight = model.embed_tokens.weight``.
        # When the assistant has no own embed_tokens (we don't ingest token ids
        # — we ingest pre-projected hidden states), tying happens by aliasing
        # the lm_head weight tensor with the target's embed_tokens at weight
        # load time.  See preprocess_weights when the model stack is added.

    def forward(self, op: OpBuilder, *args, **kwargs):  # noqa: ARG002
        raise NotImplementedError(
            "Gemma4AssistantCausalLMModel.forward is not yet implemented. "
            "See the module-level docstring in mobius/models/gemma4_assistant.py "
            "for the implementation checklist."
        )
