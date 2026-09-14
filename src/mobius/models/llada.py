# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""LLaDA masked-diffusion language model (GSAI-ML/LLaDA-8B).

LLaDA is a Llama backbone (RMSNorm, RoPE, SwiGLU MLP, full multi-head
attention with no bias) run **bidirectionally**: every position attends to
every other position, exactly like a BERT encoder and unlike a Llama decoder.
It is a mask predictor for discrete (masked) diffusion. The onnx-genai generic
SSA workflow repeatedly invokes the denoiser and the packaged masked-update
policy artifact while carrying token, mask, and RNG state.

Two properties distinguish it from the standard :class:`CausalLMModel`:

* **Bidirectional attention** — the ONNX ``Attention`` op is invoked with
  ``is_causal=0`` and no causal bias, so a change to a later token's input
  affects earlier tokens' logits.
* **No KV cache** — every call is a full-sequence forward
  (``input_ids [B, S]`` int64 -> ``logits [B, S, V]`` f32); the module
  neither consumes nor produces past/present key-value tensors.

The HuggingFace checkpoint (``model_type: "llada"``) nests its weights under
``model.transformer`` with OLMo-style names (``q_proj``/``k_proj``/``v_proj``,
``attn_out``, ``ff_proj``/``up_proj``/``ff_out``, ``attn_norm``/``ff_norm``,
``ln_f``, ``wte``); :meth:`LLaDAModel.preprocess_weights` renames these to the
reused Mobius component tree.
"""

from __future__ import annotations

import dataclasses

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig, CausalLMConfig
from mobius.components import Linear, RMSNorm, initialize_rope
from mobius.components._attention import Attention, _apply_attention
from mobius.components._decoder import DecoderLayer
from mobius.components._quantized_linear import make_quantized_linear_factory
from mobius.components._rotary_embedding import apply_rotary_pos_emb
from mobius.models.base import CausalLMModel, TiedQuantizedLMHead, embedding_for_config
from mobius.models.moe import (
    MoEDecoderLayer,
    MoETextModel,
    _preprocess_moe_weights,
    _quantized_linear_class,
)


@dataclasses.dataclass
class DreamConfig(CausalLMConfig):
    """Normalize the official Dream remote-code config into Mobius fields."""

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> DreamConfig:
        base = super().from_transformers(config, parent_config)
        return dataclasses.replace(
            base,
            attn_qkv_bias=True,
            attn_o_bias=False,
            mlp_bias=False,
            diffusion_shift_logits=True,
        )


@dataclasses.dataclass
class LLaDAMoEConfig(CausalLMConfig):
    """Normalize the official LLaDA-MoE config and raw-top-k routing semantics."""

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> LLaDAMoEConfig:
        base = super().from_transformers(config, parent_config)
        expert_size = getattr(config, "expert_intermediate_size", None)
        return dataclasses.replace(
            base,
            attn_qk_norm=True,
            attn_qk_norm_full=False,
            moe_intermediate_size=expert_size or base.moe_intermediate_size,
            norm_topk_prob=False,
        )


@dataclasses.dataclass
class RND1Config(CausalLMConfig):
    """Normalize RND1's per-head Q/K norms and renormalized top-k routing."""

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> RND1Config:
        base = super().from_transformers(config, parent_config)
        return dataclasses.replace(
            base,
            attn_qk_norm=True,
            attn_qk_norm_full=False,
            norm_topk_prob=True,
        )


class _LLaDAAttention(Attention):
    """Llama-style multi-head attention run bidirectionally with no KV cache.

    Reuses every projection and RoPE detail of the base
    :class:`~mobius.components._attention.Attention`, but invokes the ONNX
    ``Attention`` op with ``is_causal=0`` (and no causal bias), so a query
    at position ``i`` attends to keys at every position ``j`` — including
    ``j > i``. The KV cache path is dropped entirely: the forward returns a
    ``(None, None)`` present-KV placeholder that the caller ignores.
    """

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value | None,
        position_embeddings: tuple | None = None,
        past_key_value: tuple | None = None,
        static_cache=None,
    ):
        query_states = self.q_proj(op, hidden_states)
        key_states = self.k_proj(op, hidden_states)
        value_states = self.v_proj(op, hidden_states)

        if self.q_norm is not None and self.k_norm is not None:
            if self._qk_norm_full:
                query_states = self.q_norm(op, query_states)
                key_states = self.k_norm(op, key_states)
            else:
                query_states = op.Reshape(query_states, [0, 0, -1, self.head_dim])
                key_states = op.Reshape(key_states, [0, 0, -1, self.head_dim])
                query_states = self.q_norm(op, query_states)
                key_states = self.k_norm(op, key_states)
                query_states = op.Reshape(query_states, [0, 0, -1])
                key_states = op.Reshape(key_states, [0, 0, -1])

        if position_embeddings is not None:
            query_states = apply_rotary_pos_emb(
                op,
                x=query_states,
                position_embeddings=position_embeddings,
                num_heads=self.num_attention_heads,
                rotary_embedding_dim=self.rotary_embedding_dim,
                interleaved=self._rope_interleave,
            )
            key_states = apply_rotary_pos_emb(
                op,
                x=key_states,
                position_embeddings=position_embeddings,
                num_heads=self.num_key_value_heads,
                rotary_embedding_dim=self.rotary_embedding_dim,
                interleaved=self._rope_interleave,
            )

        attn_output, _, _ = _apply_attention(
            op,
            query_states,
            key_states,
            value_states,
            attention_bias,
            None,
            None,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            scale=self.scaling,
            softcap=self._softcap,
            is_causal=0,
        )

        attn_output = self.o_proj(op, attn_output)
        return attn_output, (None, None)


class _LLaDADecoderLayer(DecoderLayer):
    """Pre-norm Llama decoder layer with bidirectional attention.

    Identical to the base :class:`~mobius.components._decoder.DecoderLayer`
    (RMSNorm -> attention -> residual -> RMSNorm -> SwiGLU MLP -> residual)
    except that the attention module is the bidirectional
    :class:`_LLaDAAttention`.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config, linear_class=_quantized_linear_class(config))
        self.self_attn = _LLaDAAttention(config, linear_class=_quantized_linear_class(config))


class LLaDATextModel(nn.Module):
    """LLaDA transformer backbone: embedding + bidirectional decoder layers + norm.

    Produces the final hidden state ``[batch, sequence_len, hidden_size]``.
    Position ids are derived from the sequence length internally
    (``0 .. sequence_len - 1``) so the graph needs only ``input_ids``, and no
    attention mask or KV cache is threaded through.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = embedding_for_config(config)
        self.layers = nn.ModuleList(
            [_LLaDADecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(self, op: OpBuilder, input_ids: ir.Value):
        hidden_states = self.embed_tokens(op, input_ids)

        # Absolute position ids 0..S-1 for a full-sequence forward.
        seq_len = op.Shape(input_ids, start=1, end=2)
        position_ids = op.Range(
            op.Constant(value_int=0),
            op.Squeeze(seq_len),
            op.Constant(value_int=1),
        )
        position_ids = op.Cast(position_ids, to=7)  # INT64
        position_ids = op.Unsqueeze(position_ids, [0])
        position_embeddings = self.rotary_emb(op, position_ids)

        # Bidirectional: no causal bias, no padding mask.
        for layer in self.layers:
            hidden_states, _ = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=None,
                position_embeddings=position_embeddings,
                past_key_value=None,
            )

        return self.norm(op, hidden_states)


class DiffusionLMModel(CausalLMModel):
    """Dense masked-diffusion language model.

    A Llama/Qwen-style backbone run bidirectionally. The forward pass maps
    ``input_ids [batch, sequence_len]``
    (int64) to ``logits [batch, sequence_len, vocab_size]`` (float) in a
    single full-sequence pass. The task also exposes greedy token proposals for
    the onnx-genai generic masked-update workflow.
    """

    default_task: str = "masked-diffusion"
    category: str = "Text Generation"
    config_class: type = CausalLMConfig

    def __init__(self, config: ArchitectureConfig):
        _initialize_diffusion_model(self, config, LLaDATextModel(config))

    def forward(self, op: OpBuilder, input_ids: ir.Value):
        hidden_states = self.model(op, input_ids)
        return self.lm_head(op, hidden_states)


class DreamModel(DiffusionLMModel):
    """Dream dense masked-diffusion transformer.

    Dream uses Qwen2-style GQA with biased Q/K/V projections, bias-free output
    and SwiGLU projections, RMSNorm, RoPE, and full bidirectional attention.
    """

    config_class = DreamConfig


class LLaDAModel(DiffusionLMModel):
    """LLaDA dense masked-diffusion transformer.

    Replicates the ``LLaDALlamaBlock`` architecture of HuggingFace's
    ``LLaDAModelLM`` (``block_type: "llama"``, ``layer_norm_type: "rms"``,
    ``rope: true``, ``activation_type: "silu"``, full MHA).
    """

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Rename HuggingFace LLaDA weights to the Mobius component tree.

        HuggingFace ``LLaDAModelLM`` nests everything under
        ``model.transformer`` with OLMo-style names::

            model.transformer.wte.weight              -> model.embed_tokens.weight
            model.transformer.ln_f.weight             -> model.norm.weight
            model.transformer.ff_out.weight           -> lm_head.weight
            model.transformer.blocks.{i}.q_proj.*     -> model.layers.{i}.self_attn.q_proj.*
            model.transformer.blocks.{i}.k_proj.*     -> model.layers.{i}.self_attn.k_proj.*
            model.transformer.blocks.{i}.v_proj.*     -> model.layers.{i}.self_attn.v_proj.*
            model.transformer.blocks.{i}.attn_out.*   -> model.layers.{i}.self_attn.o_proj.*
            model.transformer.blocks.{i}.ff_proj.*    -> model.layers.{i}.mlp.gate_proj.*
            model.transformer.blocks.{i}.up_proj.*    -> model.layers.{i}.mlp.up_proj.*
            model.transformer.blocks.{i}.ff_out.*     -> model.layers.{i}.mlp.down_proj.*
            model.transformer.blocks.{i}.attn_norm.*  -> model.layers.{i}.input_layernorm.*
            model.transformer.blocks.{i}.ff_norm.*    -> model.layers.{i}.post_attention_layernorm.*
        """
        new_state_dict: dict[str, torch.Tensor] = {}
        for name, tensor in state_dict.items():
            if name.startswith(
                ("model.embed_tokens.", "model.layers.", "model.norm.", "lm_head.")
            ):
                new_state_dict[name] = tensor
                continue
            new_name = _rename_llada_weight(name)
            if new_name is not None:
                new_state_dict[new_name] = tensor
        return super().preprocess_weights(new_state_dict)


class _DiffusionMoEDecoderLayer(MoEDecoderLayer):
    """QK-normalized MoE block with bidirectional, cache-free attention."""

    def __init__(self, config: ArchitectureConfig, gate=None, norm_class=RMSNorm):
        super().__init__(config, gate=gate, norm_class=norm_class)
        self.self_attn = _LLaDAAttention(config, linear_class=_quantized_linear_class(config))


class DiffusionMoETextModel(MoETextModel):
    """Qwen3-style MoE backbone with full-sequence bidirectional attention."""

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config, layer_class=_DiffusionMoEDecoderLayer)

    def forward(self, op: OpBuilder, input_ids: ir.Value):
        hidden_states = self.embed_tokens(op, input_ids)
        seq_len = op.Shape(input_ids, start=1, end=2)
        position_ids = op.Range(
            op.Constant(value_int=0),
            op.Squeeze(seq_len),
            op.Constant(value_int=1),
        )
        position_ids = op.Cast(position_ids, to=7)  # INT64
        position_embeddings = self.rotary_emb(op, op.Unsqueeze(position_ids, [0]))

        for layer in self.layers:
            hidden_states, _ = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=None,
                position_embeddings=position_embeddings,
                past_key_value=None,
            )
        return self.norm(op, hidden_states)


class DiffusionMoEModel(CausalLMModel):
    """Masked-diffusion MoE denoiser with no causal mask or KV cache."""

    default_task: str = "masked-diffusion"
    category: str = "Mixture of Experts"
    config_class: type = CausalLMConfig

    def __init__(self, config: ArchitectureConfig):
        _initialize_diffusion_model(self, config, DiffusionMoETextModel(config))

    def forward(self, op: OpBuilder, input_ids: ir.Value):
        return self.lm_head(op, self.model(op, input_ids))

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return _preprocess_moe_weights(self, state_dict)


class LLaDAMoEModel(DiffusionMoEModel):
    """LLaDA-MoE denoiser with raw selected softmax routing weights."""

    config_class = LLaDAMoEConfig


class RND1Model(DiffusionMoEModel):
    """RND1 denoiser with selected softmax routing weights renormalized to one."""

    config_class = RND1Config


def _initialize_diffusion_model(
    module: CausalLMModel,
    config: ArchitectureConfig,
    text_model: nn.Module,
) -> None:
    """Initialize a no-cache model while retaining quantized embedding/head ABI."""
    nn.Module.__init__(module)
    module.config = config
    module.model = text_model

    quantization = getattr(config, "quantization", None)
    quantize_head = quantization is not None and quantization.quantize_lm_head
    quantize_embedding = quantization is not None and quantization.quantize_embeddings
    tie = config.tie_word_embeddings or (
        quantization is not None and quantization.tie_word_embeddings
    )
    if quantize_head and quantize_embedding and tie:
        module.lm_head = TiedQuantizedLMHead(
            module.model.embed_tokens, config.hidden_size, config.vocab_size
        )
    elif quantize_head:
        zero_point_dtype = config.dtype if quantization.float_zero_point else ir.DataType.UINT8
        head_class = make_quantized_linear_factory(
            bits=quantization.bits,
            block_size=quantization.group_size,
            has_zero_point=not quantization.sym,
            zero_point_dtype=zero_point_dtype,
        )
        module.lm_head = head_class(config.hidden_size, config.vocab_size, bias=False)
    else:
        module.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings and not quantize_embedding:
            module.lm_head.weight = module.model.embed_tokens.weight


# Block-level attribute renames (HF LLaDA -> Mobius component tree).
_BLOCK_RENAMES: dict[str, str] = {
    "q_proj": "self_attn.q_proj",
    "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
    "attn_out": "self_attn.o_proj",
    "ff_proj": "mlp.gate_proj",
    "up_proj": "mlp.up_proj",
    "ff_out": "mlp.down_proj",
    "attn_norm": "input_layernorm",
    "ff_norm": "post_attention_layernorm",
}


def _rename_llada_weight(name: str) -> str | None:
    """Rename a single HuggingFace LLaDA weight, or return ``None`` to drop it.

    Rotary caches are recomputed from the config, so any buffered rotary
    tensors in the checkpoint are dropped.
    """
    # Strip the HF ``model.transformer.`` prefix.
    prefix = "model.transformer."
    if name.startswith(prefix):
        name = name[len(prefix) :]
    elif name.startswith("transformer."):
        name = name[len("transformer.") :]

    # Rotary caches are regenerated, not loaded.
    if "rotary_emb" in name or "rope" in name:
        return None

    if name.startswith("wte."):
        return "model.embed_tokens." + name[len("wte.") :]
    if name.startswith("ln_f."):
        return "model.norm." + name[len("ln_f.") :]
    if name.startswith("ff_out."):
        # Top-level ff_out is the untied LM head.
        return "lm_head." + name[len("ff_out.") :]

    if name.startswith("blocks."):
        parts = name.split(".", 2)
        if len(parts) < 3:
            return None
        layer_idx, remainder = parts[1], parts[2]
        for hf_attr, mobius_attr in _BLOCK_RENAMES.items():
            if remainder.startswith(hf_attr + "."):
                suffix = remainder[len(hf_attr) + 1 :]
                return f"model.layers.{layer_idx}.{mobius_attr}.{suffix}"
        return None

    return None
