# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig
from mobius._weight_utils import pack_qmoe_expert_weights
from mobius.components._attention import Qwen35Attention
from mobius.components._common import (
    Embedding,
    Linear,
    create_attention_bias,
)
from mobius.components._gated_deltanet import GatedDeltaNet
from mobius.components._mlp import MLP
from mobius.components._moe import _supported_qmoe_quantization
from mobius.components._quantized_linear import make_quantized_linear_factory
from mobius.components._rms_norm import OffsetRMSNorm
from mobius.components._rotary_embedding import initialize_rope
from mobius.models.base import CausalLMModel
from mobius.models.moe import Qwen2MoELayer
from mobius.models.qwen_vl import (
    Qwen3VLEmbeddingModel,
    Qwen3VLVisionEncoderModel,
    _QwenVLTextMixin,
    split_deepstack_embeds,
)

# ---------------------------------------------------------------------------
# Qwen3.5 — hybrid linear/full attention
# ---------------------------------------------------------------------------


def _linear_factory(config: ArchitectureConfig) -> type | None:
    """Build a quantized-linear factory from ``config.quantization``, or None.

    Returns ``None`` (meaning "use plain ``Linear``") when the model is
    unquantized. Shared by the dense (:class:`Qwen35DecoderLayer`) and MoE
    (:class:`Qwen35MoEDecoderLayer`) variants so ``self_attn``/``linear_attn``/
    ``mlp``/``shared_expert`` are all quantized consistently.
    """
    quantization = config.quantization
    if quantization is None or quantization.quant_method == "none":
        return None
    zero_point_dtype = config.dtype if quantization.float_zero_point else ir.DataType.UINT8
    return make_quantized_linear_factory(
        bits=quantization.bits,
        block_size=quantization.group_size,
        has_zero_point=not quantization.sym,
        zero_point_dtype=zero_point_dtype,
    )


class Qwen35DecoderLayer(nn.Module):
    """Qwen3.5 decoder layer with hybrid attention.

    Each layer is either ``"linear_attention"`` (GatedDeltaNet) or
    ``"full_attention"`` (Qwen35Attention with output gating), controlled
    by ``config.layer_types[layer_idx]``.

    Both variants use :class:`OffsetRMSNorm` (the *1 + weight* variant)
    for pre-attention and post-attention normalization.

    When ``config.quantization`` is set, ``self_attn``/``linear_attn``/``mlp``
    projections are built with a quantized-linear factory (see
    :func:`_linear_factory`) so quantized checkpoints (e.g. Olive
    RTN/GPTQ) load correctly instead of hitting a dense-vs-packed shape
    mismatch.
    """

    def __init__(self, config: ArchitectureConfig, layer_idx: int):
        super().__init__()
        layer_types = config.layer_types or []
        self.layer_type: str = (
            layer_types[layer_idx] if layer_idx < len(layer_types) else "full_attention"
        )
        linear_class = _linear_factory(config)

        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(config, linear_class=linear_class)
        else:
            self.self_attn = Qwen35Attention(config, linear_class=linear_class)

        self.mlp = MLP(config, linear_class=linear_class)
        self.input_layernorm = OffsetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = OffsetRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple[ir.Value, ir.Value],
        past_key_value: tuple[ir.Value, ir.Value] | None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(op, hidden_states)

        if self.layer_type == "linear_attention":
            # DeltaNet states are passed through past_key_value as
            # (conv_state, recurrent_state), same tuple pattern as KV cache
            conv_state, recurrent_state = past_key_value

            attn_output, new_conv_state, new_recurrent_state = self.linear_attn(
                op, hidden_states, conv_state, recurrent_state
            )
            present_key_value = (new_conv_state, new_recurrent_state)
        else:
            attn_output, present_key_value = self.self_attn(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_key_value,
            )

        hidden_states = op.Add(residual, attn_output)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(op, hidden_states)
        hidden_states = self.mlp(op, hidden_states)
        hidden_states = op.Add(residual, hidden_states)

        return hidden_states, present_key_value


class Qwen35TextModel(nn.Module):
    """Qwen3.5 text model with hybrid linear/full attention layers.

    Uses :class:`OffsetRMSNorm` for the final norm and creates
    :class:`Qwen35DecoderLayer` instances that dispatch to either
    ``GatedDeltaNet`` or ``Qwen35Attention`` based on
    ``config.layer_types``.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [Qwen35DecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = OffsetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value | None,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        inputs_embeds: ir.Value | None = None,
        deepstack_embeds: list | None = None,
    ):
        # Embed tokens: (batch, seq_len) → (batch, seq_len, hidden_size)
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(op, input_ids)
        # Compute (cos, sin) for RoPE: each (batch, seq_len, rotary_dim)
        position_embeddings = self.rotary_emb(op, position_ids)

        # Causal attention mask: (batch, 1, seq_len, total_seq_len)
        attention_bias = create_attention_bias(
            op,
            input_ids=hidden_states if input_ids is None else input_ids,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )

        present_key_values: list = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer_idx, (layer, past_kv) in enumerate(zip(self.layers, past_kvs)):
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)
            # DeepStack injection (see TextModel.forward for the rationale).
            if deepstack_embeds is not None and layer_idx < len(deepstack_embeds):
                hidden_states = op.Add(hidden_states, deepstack_embeds[layer_idx])

        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values


class Qwen35CausalLMModel(CausalLMModel):
    """Qwen3.5 causal language model with hybrid linear/full attention.

    Combines standard GQA layers (with output gating) and GatedDeltaNet
    linear attention layers in a single decoder stack.  The per-layer
    attention type is controlled by ``config.layer_types``.

    Full attention layers use standard KV cache. DeltaNet layers carry
    ``conv_state`` and ``recurrent_state`` tensors, managed by
    :class:`HybridCausalLMTask`.
    """

    default_task: str = "hybrid-text-generation"

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        self.model = Qwen35TextModel(config)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Preprocess HuggingFace state dict for Qwen3.5.

        Handles:
        - Stripping ``language_model.`` prefix from HF checkpoint keys
          (HF stores weights as ``model.language_model.*`` in safetensors)
        - Dropping visual encoder keys (``model.visual.*``)
        - Dropping multi-token prediction (MTP) keys (``mtp*``):
          MTP heads are auxiliary decoding heads used only during
          HuggingFace training; they are not needed for inference.
        - Weight tying (``tie_word_embeddings``)
        """
        cleaned: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith(("mtp_", "mtp.")):
                continue

            stripped = key
            if stripped.startswith("model."):
                stripped = stripped[len("model.") :]

            if stripped.startswith("visual."):
                continue
            if stripped.startswith("language_model."):
                stripped = stripped[len("language_model.") :]
                cleaned[f"model.{stripped}"] = value
            else:
                cleaned[key] = value

        return super().preprocess_weights(cleaned)


class Qwen35MoEBlock(Qwen2MoELayer):
    """Qwen3.5-MoE sparse MoE block: top-k routing + sigmoid-gated shared expert.

    Structurally identical to Qwen2-MoE's shared-expert block
    (:class:`~mobius.models.moe.Qwen2MoELayer`): top-k expert routing whose
    weighted sum is added to a shared expert scaled by a sigmoid gate. Weight
    names match the HuggingFace convention::

        gate.weight                  → router logits
        experts.N.{gate,up,down}_proj.weight
        shared_expert.{gate,up,down}_proj.weight
        shared_expert_gate.weight    → sigmoid gate for shared expert

    Retained as a named subclass for readability within the Qwen3.5 hybrid stack.
    """


class Qwen35MoEDecoderLayer(Qwen35DecoderLayer):
    """Qwen3.5-MoE decoder layer with hybrid attention and MoE FFN.

    Same hybrid DeltaNet/full-attention architecture as
    :class:`Qwen35DecoderLayer`, but replaces the dense MLP with a
    :class:`Qwen35MoEBlock`.

    The routed experts are quantized via the fused ``com.microsoft::QMoE``
    path (see :class:`~mobius.components._moe.MoELayer`), driven directly
    by ``config.quantization``. The always-active ``shared_expert``/
    ``shared_expert_gate`` are not part of that fused op, so they are
    quantized separately via the same ``linear_class`` factory used for
    ``self_attn``/``linear_attn``/dense ``mlp`` (see :func:`_linear_factory`).
    """

    def __init__(self, config: ArchitectureConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.mlp = Qwen35MoEBlock(config, linear_class=_linear_factory(config))


class Qwen35MoETextModel(nn.Module):
    """Qwen3.5-MoE text backbone (no LM head).

    Same hybrid DeltaNet/full-attention layer structure as
    :class:`Qwen35TextModel`, but each layer uses MoE FFN
    (:class:`Qwen35MoEBlock`) instead of dense MLP.

    HuggingFace class: ``Qwen3_5MoeModel``
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self._dtype = config.dtype
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [
                Qwen35MoEDecoderLayer(config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = OffsetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value | None,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        inputs_embeds: ir.Value | None = None,
        deepstack_embeds: list | None = None,
    ):
        # Embed tokens unless caller already provided fused inputs_embeds
        # (e.g. the VL decoder, which interleaves vision features before
        # entering the text backbone).
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(op, input_ids)
        position_embeddings = self.rotary_emb(op, position_ids)

        attention_bias = create_attention_bias(
            op,
            input_ids=hidden_states if input_ids is None else input_ids,
            attention_mask=attention_mask,
            dtype=self._dtype,
        )

        present_key_values: list = []
        past_kvs = past_key_values or [None] * len(self.layers)
        for layer_idx, (layer, past_kv) in enumerate(zip(self.layers, past_kvs)):
            hidden_states, present_kv = layer(
                op,
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_kv,
            )
            present_key_values.append(present_kv)
            # DeepStack injection (see TextModel.forward for the rationale).
            if deepstack_embeds is not None and layer_idx < len(deepstack_embeds):
                hidden_states = op.Add(hidden_states, deepstack_embeds[layer_idx])

        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values


class Qwen35MoECausalLMModel(CausalLMModel):
    """Qwen3.5-MoE causal language model.

    Combines the hybrid DeltaNet/full-attention architecture of Qwen3.5
    with Mixture-of-Experts FFN layers that include a shared expert
    gated by sigmoid.
    """

    default_task: str = "hybrid-text-generation"
    category: str = "Mixture of Experts"

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)
        self.model = Qwen35MoETextModel(config)

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Preprocess HuggingFace state dict for Qwen3.5-MoE.

        Handles:
        - Dropping multi-token prediction (MTP) keys (``mtp_*``, ``mtp.*``):
          MTP heads are auxiliary decoding heads used only during
          HuggingFace training; they are not needed for inference.
        - Stripping ``language_model.`` prefix from HF checkpoint keys
          (HF stores weights as ``model.language_model.*`` in safetensors)
        - Dropping visual encoder keys (``model.visual.*``)
        - Weight tying (``tie_word_embeddings``)
        - Unpacking fused expert weights (``experts.gate_up_proj``,
          ``experts.down_proj``) into per-expert tensors for the portable
          dense fallback, OR keeping them expert-major and packing them into
          native ``com.microsoft::QMoE`` parameters when the quantization
          config matches the QMoE ABI (mirrors DeepSeek-V3's ``use_qmoe``
          path so both MoE families share one emission route).
        """
        # When the int4 block scheme matches the native QMoE ABI, keep the
        # fused expert-major tensors and route them through the QMoE repacker
        # instead of un-fusing into per-expert MLPs. Uses the same predicate
        # as MoELayer so the weights and the emitted graph never disagree.
        use_qmoe = _supported_qmoe_quantization(self.config.quantization) is not None
        if not use_qmoe:
            # The dense per-expert fallback (MoELayer.experts) only knows how
            # to consume *unquantized* fused expert tensors -- the split
            # below un-fuses a bare float [num_experts, ...] tensor into
            # per-expert nn.Linear-shaped weights. If quantization produced
            # packed Olive tensors here (suffixed "_qweight"/"_scales"/
            # "_qzeros", see preprocess_olive_weights) but the quantization
            # config doesn't match the native QMoE ABI, there is no code path
            # that splits a *packed* fused expert tensor into per-expert
            # quantized Linear initializers -- silently falling through would
            # emit a graph whose per-expert Linear modules request
            # "experts.N.{gate,up}_proj.weight_qweight" keys that were never
            # produced, since the packed tensor stays fused under
            # "experts.gate_up_proj_qweight" instead. Reject explicitly
            # rather than emit an unloadable graph.
            packed_expert_keys = [
                key
                for key in state_dict
                if ".mlp.experts." in key
                and any(key.endswith(suffix) for suffix in ("_qweight", "_scales", "_qzeros"))
            ]
            if packed_expert_keys:
                raise ValueError(
                    "Quantized MoE expert weights were found "
                    f"(e.g. {packed_expert_keys[0]!r}) but this quantization "
                    "config doesn't match the native QMoE ABI "
                    "(_supported_qmoe_quantization returned None). The dense "
                    "loop-over-experts fallback only supports unquantized "
                    "fused expert tensors, not packed quantized ones -- "
                    "there is no path to un-fuse a packed expert tensor into "
                    "per-expert quantized Linear initializers. Use a "
                    "QMoE-ABI-compatible quantization config (see "
                    "_supported_qmoe_quantization) for MoE models instead."
                )
        cleaned: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith(("mtp_", "mtp.")):
                continue

            # Strip "model." prefix, then handle sub-prefixes
            stripped = key
            if stripped.startswith("model."):
                stripped = stripped[len("model.") :]

            # Drop visual encoder weights (not used by text-only MoE)
            if stripped.startswith("visual."):
                continue

            # Strip "language_model." nesting and re-add "model." to
            # match the ONNX module hierarchy (self.model = ...)
            if stripped.startswith("language_model."):
                stripped = stripped[len("language_model.") :]
                key = f"model.{stripped}"
            else:
                key = f"model.{stripped}" if key.startswith("model.") else key

            # Unpack fused float expert weights into per-expert tensors for the
            # dense fallback. When ``use_qmoe`` is set the fused quantized
            # tensors arrive as ``_qweight``/``_scales``/``_qzeros`` (which do
            # not match these suffixes) and are kept expert-major for
            # ``pack_qmoe_expert_weights`` below.
            # HF format: [num_experts, fused_dim, hidden] with gate+up fused
            if not use_qmoe and key.endswith(".mlp.experts.gate_up_proj"):
                prefix = key[: -len("experts.gate_up_proj")]
                num_experts = value.shape[0]
                half = value.shape[1] // 2
                for i in range(num_experts):
                    cleaned[f"{prefix}experts.{i}.gate_proj.weight"] = value[i, :half]
                    cleaned[f"{prefix}experts.{i}.up_proj.weight"] = value[i, half:]
                continue

            # Stacked expert down_proj without .weight suffix (HF format)
            if not use_qmoe and key.endswith(".mlp.experts.down_proj"):
                prefix = key[: -len("experts.down_proj")]
                num_experts = value.shape[0]
                for i in range(num_experts):
                    cleaned[f"{prefix}experts.{i}.down_proj.weight"] = value[i]
                continue

            cleaned[key] = value

        processed = super().preprocess_weights(cleaned)
        if use_qmoe:
            # Qwen3.5-MoE's MoELayer lives directly at ``.mlp`` (Qwen35MoEBlock
            # is the MoELayer), unlike DeepSeek's nested ``.mlp.moe``.
            processed = pack_qmoe_expert_weights(processed, target_moe_path=".mlp")
        return processed


# ---------------------------------------------------------------------------
# Qwen3.5-VL — vision-language (3-model split)
# ---------------------------------------------------------------------------


class Qwen35VLTextModel(_QwenVLTextMixin, Qwen35CausalLMModel):
    """Qwen3.5-VL text-only decoder.

    Extracts the text backbone from the Qwen3.5-VL multimodal model.
    Strips ``language_model.`` weight prefixes and drops ``visual.`` keys.
    """


class Qwen35VL3ModelCausalLMModel(nn.Module):
    """Qwen3.5-VL vision-language model (3-model split).

    Builds three separate ONNX models for onnxruntime-genai:

    - ``decoder``: text decoder taking ``inputs_embeds`` (interleaved MRoPE)
    - ``vision_encoder``: packed-attention ViT outputting merged features
    - ``embedding``: token embedding + image feature fusion

    The vision encoder is identical to Qwen3-VL's
    :class:`Qwen3VLVisionModel`.  The text decoder uses
    :class:`Qwen35TextModel` (hybrid linear/full attention).
    """

    default_task: str = "hybrid-qwen-vl"
    category: str = "Multimodal"
    config_class: type = ArchitectureConfig

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.decoder = Qwen35VLDecoderModel(config)
        self.vision_encoder = Qwen3VLVisionEncoderModel(config)
        self.embedding = Qwen3VLEmbeddingModel(config)

    def forward(self, op: OpBuilder, **kwargs):
        raise NotImplementedError(
            "Qwen35VL3ModelCausalLMModel uses QwenVLTask "
            "which calls each sub-module separately."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Route HF weights to the correct sub-model ONNX initializer names.

        HF keys: ``model.visual.*``, ``model.language_model.*``.
        """
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            # Drop multi-token prediction (MTP) keys: MTP heads are
            # auxiliary decoding heads used only during HuggingFace
            # training; they are not needed for inference.
            if key.startswith(("mtp_", "mtp.")):
                continue

            stripped = key
            if stripped.startswith("model."):
                stripped = stripped[len("model.") :]

            if stripped.startswith("visual."):
                # Qwen3-VL uses linear_fc1/fc2; ONNX uses up_proj/down_proj
                stripped = stripped.replace(".mlp.linear_fc1.", ".mlp.up_proj.")
                stripped = stripped.replace(".mlp.linear_fc2.", ".mlp.down_proj.")
                renamed[f"vision_encoder.{stripped}"] = value
            elif stripped.startswith("language_model.embed_tokens."):
                suffix = stripped[len("language_model.") :]
                renamed[f"decoder.model.{suffix}"] = value
                renamed[f"embedding.{suffix}"] = value
                if (
                    self.config.tie_word_embeddings
                    and stripped == "language_model.embed_tokens.weight"
                ):
                    renamed["decoder.lm_head.weight"] = value
            elif stripped.startswith("language_model.lm_head."):
                renamed[f"decoder.{stripped[len('language_model.') :]}"] = value
            elif stripped.startswith("lm_head."):
                renamed[f"decoder.{stripped}"] = value
            elif stripped.startswith("language_model."):
                suffix = stripped[len("language_model.") :]
                renamed[f"decoder.model.{suffix}"] = value
        return renamed


class Qwen35VLDecoderModel(nn.Module):
    """Qwen3.5-VL text decoder taking ``inputs_embeds`` (3-model split).

    Uses interleaved MRoPE with 3D ``position_ids`` of shape
    ``(3, batch, seq_len)`` and hybrid linear/full attention layers.
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.model = Qwen35TextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        deepstack_embeds: ir.Value | None = None,
    ):
        hidden_states, present_key_values = self.model(
            op,
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            deepstack_embeds=split_deepstack_embeds(op, deepstack_embeds, self.config),
        )
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Route language_model weights for standalone decoder build."""
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            # Drop MTP heads (training-only auxiliary decoders)
            if key.startswith(("mtp_", "mtp.")):
                continue
            stripped = key
            if stripped.startswith("model."):
                stripped = stripped[len("model.") :]
            if stripped.startswith("visual."):
                continue
            if stripped.startswith("language_model."):
                stripped = stripped[len("language_model.") :]
            renamed[stripped] = value

        if self.config.tie_word_embeddings:
            if "lm_head.weight" not in renamed and "model.embed_tokens.weight" in renamed:
                renamed["lm_head.weight"] = renamed["model.embed_tokens.weight"]
        return renamed


# ---------------------------------------------------------------------------
# Qwen3.5-MoE-VL (Qwen3.6-35B-A3B and friends)
# ---------------------------------------------------------------------------


class Qwen35MoEVLDecoderModel(nn.Module):
    """Qwen3.5-MoE-VL text decoder taking ``inputs_embeds`` (3-model split).

    Same wiring as :class:`Qwen35VLDecoderModel` but the text backbone is the
    MoE variant :class:`Qwen35MoETextModel` (hybrid linear/full attention +
    sparse Mixture-of-Experts FFN with a sigmoid-gated shared expert).
    """

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.model = Qwen35MoETextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        deepstack_embeds: ir.Value | None = None,
    ):
        hidden_states, present_key_values = self.model(
            op,
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            deepstack_embeds=split_deepstack_embeds(op, deepstack_embeds, self.config),
        )
        logits = self.lm_head(op, hidden_states)
        return logits, present_key_values

    # No preprocess_weights here: this class is an internal sub-module of
    # Qwen35MoEVL3ModelCausalLMModel (constructed at line ~667 below) and is
    # never registered standalone. The wrapper's preprocess_weights handles
    # all HF weight routing for the 3-model package, including the fused
    # MoE expert unpack and the tied lm_head/embeddings hookup. Adding a
    # standalone preprocess_weights here would be dead code at best and
    # confusingly out-of-sync with the wrapper at worst.


class Qwen35MoEVL3ModelCausalLMModel(nn.Module):
    """Qwen3.5-MoE-VL vision-language model (3-model split for ORT GenAI).

    Builds three separate ONNX models:

    - ``decoder``: text decoder over hybrid linear/full attention with MoE
      FFN; consumes ``inputs_embeds`` so the embedding model can splice in
      vision features.
    - ``vision_encoder``: shared Qwen3-VL ViT (identical to the dense
      Qwen3.5-VL counterpart).
    - ``embedding``: token embedding + image-feature fusion.

    HuggingFace class: ``Qwen3_5MoeForConditionalGeneration``. The HF
    model_type string is ``qwen3_5_moe`` for both the text-only checkpoints
    and these VL checkpoints; the registry dispatches to this class when the
    HF config carries a ``vision_config`` sub-object and to
    :class:`Qwen35MoECausalLMModel` otherwise.
    """

    default_task: str = "hybrid-qwen-vl"
    category: str = "Multimodal"
    config_class: type = ArchitectureConfig

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.decoder = Qwen35MoEVLDecoderModel(config)
        self.vision_encoder = Qwen3VLVisionEncoderModel(config)
        self.embedding = Qwen3VLEmbeddingModel(config)

    def forward(self, op, **kwargs):
        raise NotImplementedError(
            "Qwen35MoEVL3ModelCausalLMModel uses HybridQwenVLTask which "
            "drives each sub-module independently."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Route HF weights to the three sub-model ONNX initializer names.

        HF key prefixes: ``model.visual.*`` (vision encoder),
        ``model.language_model.*`` (MoE text backbone + embeddings),
        ``lm_head.*`` (final projection — tied or untied to embeddings).
        Mirrors :meth:`Qwen35VL3ModelCausalLMModel.preprocess_weights` but
        the decoder uses the MoE-flavoured text model and HF stores experts
        as fused tensors that must be unpacked.
        """
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith(("mtp_", "mtp.")):
                continue
            stripped = key
            if stripped.startswith("model."):
                stripped = stripped[len("model.") :]

            if stripped.startswith("visual."):
                stripped = stripped.replace(".mlp.linear_fc1.", ".mlp.up_proj.")
                stripped = stripped.replace(".mlp.linear_fc2.", ".mlp.down_proj.")
                renamed[f"vision_encoder.{stripped}"] = value
            elif stripped.startswith("language_model.embed_tokens."):
                suffix = stripped[len("language_model.") :]
                renamed[f"decoder.model.{suffix}"] = value
                renamed[f"embedding.{suffix}"] = value
                if (
                    self.config.tie_word_embeddings
                    and stripped == "language_model.embed_tokens.weight"
                ):
                    renamed["decoder.lm_head.weight"] = value
            elif stripped.startswith("language_model.lm_head."):
                renamed[f"decoder.{stripped[len('language_model.') :]}"] = value
            elif stripped.startswith("lm_head."):
                renamed[f"decoder.{stripped}"] = value
            elif stripped.startswith("language_model."):
                suffix = stripped[len("language_model.") :]
                target = f"decoder.model.{suffix}"
                # Unpack fused expert tensors into per-expert ONNX initializer
                # names. HF stores ``experts.gate_up_proj`` as
                # ``[num_experts, 2*inter, hidden]`` (gate + up concatenated
                # along dim 1) and ``experts.down_proj`` as
                # ``[num_experts, hidden, inter]``.
                if suffix.endswith(".mlp.experts.gate_up_proj"):
                    prefix = target[: -len("experts.gate_up_proj")]
                    half = value.shape[1] // 2
                    for i in range(value.shape[0]):
                        renamed[f"{prefix}experts.{i}.gate_proj.weight"] = value[i, :half]
                        renamed[f"{prefix}experts.{i}.up_proj.weight"] = value[i, half:]
                elif suffix.endswith(".mlp.experts.down_proj"):
                    prefix = target[: -len("experts.down_proj")]
                    for i in range(value.shape[0]):
                        renamed[f"{prefix}experts.{i}.down_proj.weight"] = value[i]
                else:
                    renamed[target] = value
        return renamed
