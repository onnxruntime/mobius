# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Lightning's single NextN step: fusion -> attention -> MoE -> RMSNorm.

Replicates vLLM's NemotronHMultiTokenPredictor at
ef63c23d35acccbba8e014fc88da19d541da38a9. The pinned Lightning checkpoints
store two physical MTP blocks, including dedicated shared-expert weights.
Embedding and vocabulary projection weights are shared with the target.
No positional encoding or Mamba state is used by the MTP blocks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from onnxscript import OpBuilder, nn

from mobius._configs import NemotronHConfig
from mobius.components import Embedding, Linear, RMSNorm, create_padding_mask
from mobius.models.nemotron_h import (
    NemotronHAttentionLayer,
    NemotronHCausalLMModel,
    NemotronHMoELayer,
)

if TYPE_CHECKING:
    import onnx_ir as ir


def validate_lightning_mtp_config(config: NemotronHConfig) -> None:
    """Reject unsupported NextN variants rather than guessing their architecture."""
    if config.num_nextn_predict_layers != 1:
        raise NotImplementedError("Lightning MTP requires exactly one NextN prediction step")
    if config.mtp_layers_block_type != ["attention", "moe"]:
        raise NotImplementedError(
            "Lightning MTP requires mtp_layers_block_type=['attention','moe']"
        )
    if config.n_shared_experts != 1 or config.moe_latent_size is not None:
        raise NotImplementedError("Lightning MTP requires one shared expert and no latent MoE")
    if (
        config.residual_in_fp32
        or config.attn_qkv_bias
        or config.attn_o_bias
        or config.mlp_bias
    ):
        raise NotImplementedError("Lightning MTP supports bias-free, model-dtype residuals")
    if config.hidden_act != "relu2" or not config.shared_expert_intermediate_size:
        raise NotImplementedError("Lightning MTP requires ReLU2 and an explicit shared width")


class _FusionAttentionLayer(NemotronHAttentionLayer):
    def __init__(self, config: NemotronHConfig):
        super().__init__(config)
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = Linear(2 * config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: ir.Value,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        past_key_value: tuple,
    ):
        # Condition on t_(i+1)'s embedding and the target's post-final-norm h_i.
        fused = op.Concat(
            self.enorm(op, inputs_embeds), self.hnorm(op, hidden_states), axis=-1
        )  # (batch, sequence, 2H)
        hidden_states = self.eh_proj(op, fused)  # (batch, sequence, H)
        return super().forward(op, hidden_states, attention_bias, None, past_key_value)


class _FinalMoELayer(NemotronHMoELayer):
    def __init__(self, config: NemotronHConfig):
        super().__init__(config)
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None = None,
        position_embeddings: tuple | None = None,
        past_key_value: tuple | None = None,
    ):
        hidden_states, _ = super().forward(
            op, hidden_states, attention_bias, position_embeddings, past_key_value
        )
        return self.final_layernorm(op, hidden_states)


class NemotronHMtpModel(nn.Module):
    """Standalone draft computation with target-shared embedding and LM head.

    Inputs: token IDs for t_(i+1), post-final-norm target h_i, a padding mask,
    and independent MTP attention KV. Outputs: draft logits for t_(i+2),
    normalized MTP hidden states for recursive drafting, and updated MTP KV.
    Token shifting, speculative acceptance and state rollback belong to the caller.
    """

    config_class: type = NemotronHConfig
    default_task: str = "nemotron-h-mtp"
    category: str = "Mixture of Experts"

    def __init__(self, config: NemotronHConfig):
        super().__init__()
        validate_lightning_mtp_config(config)
        self.config = config
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList([_FusionAttentionLayer(config), _FinalMoELayer(config)])
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        hidden_states: ir.Value,
        attention_mask: ir.Value,
        past_key_value: tuple,
    ):
        inputs_embeds = self.embed_tokens(op, input_ids)
        attention_bias = create_padding_mask(
            op, input_ids=input_ids, attention_mask=attention_mask
        )
        hidden_states, present = self.layers[0](
            op, inputs_embeds, hidden_states, attention_bias, past_key_value
        )
        hidden_states = self.layers[1](op, hidden_states)
        return self.lm_head(op, hidden_states), hidden_states, present


class _MtpSeedTarget(NemotronHCausalLMModel):
    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states, present = self.model(
            op, input_ids, attention_mask, position_ids, past_key_values
        )
        return self.lm_head(op, hidden_states), present, None, hidden_states


class NemotronHSpeculativeModel(nn.Module):
    """Explicit target + Lightning MTP graph package, not a generation runtime."""

    config_class: type = NemotronHConfig
    default_task: str = "nemotron-h-mtp"
    category: str = "Mixture of Experts"

    def __init__(self, config: NemotronHConfig):
        super().__init__()
        validate_lightning_mtp_config(config)
        self.config = config
        # The explicit package preserves auxiliary tensors in a separate graph.
        self.decoder = _MtpSeedTarget(config, target_decoder_only=True)
        self.mtp = NemotronHMtpModel(config)
        # Each component is its own graph root, not a nested parameter namespace.
        self.decoder._set_name("")
        self.mtp._set_name("")


def mtp_source_name(target_name: str) -> str:
    """Exact checkpoint mapping; tables share target sources, not random fallbacks."""
    if target_name == "embed_tokens.weight":
        return "backbone.embeddings.weight"
    if target_name == "lm_head.weight":
        return target_name
    if target_name.startswith("layers."):
        return "mtp." + target_name.replace(".self_attn.", ".mixer.").replace(
            ".moe.", ".mixer."
        )
    raise ValueError(f"Unexpected Lightning MTP graph parameter {target_name!r}")
