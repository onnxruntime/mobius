# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Gemma3n text model with AltUp, Laurel, and per-layer input gating.

Gemma3n extends Gemma3 with three efficiency features:
1. **AltUp (Alternating Updates)**: Only a subset of hidden dims are updated
   each layer, reducing compute per layer.
2. **Laurel (Low-rank Residual)**: A low-rank residual augmentation added after
   attention normalization.
3. **Per-layer input gating**: Each layer receives a per-layer embedding derived
   from the input tokens, gated and projected into the hidden space.

The base attention and MLP structure is similar to Gemma3 (hybrid global+sliding
window attention, QK-norm, four-norm decoder layers).
"""

from __future__ import annotations

import copy
import math
from statistics import NormalDist

import numpy as np
import onnx_ir as ir
import torch
from onnxscript import OpBuilder, nn

from mobius._configs import Gemma3nConfig, Gemma3nMultiModalConfig
from mobius.components import (
    MLP,
    Attention,
    Embedding,
    Gemma3nAudioEncoder,
    Gemma3nMultimodalEmbedder,
    Linear,
    MobileNetV5Encoder,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
)
from mobius.components._activations import get_activation
from mobius.components._attention import _apply_attention, apply_rotary_pos_emb
from mobius.models.base import CausalLMModel


def _apply_logit_softcapping(op: OpBuilder, logits: ir.Value, cap: float) -> ir.Value:
    """Tanh-cap *logits* at +-*cap* as Gemma2/Gemma3n's LM head does.

    ``cap * tanh(logits / cap)``.  A non-positive *cap* disables the capping and
    returns *logits* unchanged, so no nodes are emitted for configs that ship
    ``final_logit_softcapping: null``.
    """
    if not cap or cap <= 0.0:
        return logits
    cap_value = op.CastLike(op.Constant(value_float=float(cap)), logits)
    return op.Mul(op.Tanh(op.Div(logits, cap_value)), cap_value)


def _strip_altup_prefix(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Drop the ``.altup.`` level from every key that carries it.

    The AltUp submodules (``correction_coefs``, ``prediction_coefs``,
    ``modality_router``, ``router_norm``, ``correct_output_scale``) are invoked
    through plain methods rather than ``__call__``, so onnxscript registers them
    on the parent :class:`Gemma3nDecoderLayer` without the ``.altup.`` level.
    """
    for key in list(state_dict.keys()):
        if ".altup." in key:
            state_dict[key.replace(".altup.", ".")] = state_dict.pop(key)
    return state_dict


def _drop_kv_shared_layer_weights(
    state_dict: dict[str, torch.Tensor],
    text_model: Gemma3nTextModel,
    prefix: str = "model.layers.",
) -> dict[str, torch.Tensor]:
    """Drop the K/V weights of KV-shared layers from *state_dict*.

    The Gemma 3n checkpoint ships ``k_proj``/``v_proj``/``k_norm`` for **every**
    layer, but HF only constructs them for the non-shared layers and discards
    the rest.  mobius likewise builds no such projections for KV-shared layers,
    so leaving these keys in place would only emit "weight not found in the
    model" warnings for tensors that are correctly unused.

    Args:
        state_dict: Weights keyed relative to the decoder root.
        text_model: The :class:`Gemma3nTextModel` whose layers declare which
            indices are KV-shared.
        prefix: Layer-key prefix, e.g. ``"model.layers."``.
    """
    for idx, layer in enumerate(text_model.layers):
        if not layer.self_attn.is_kv_shared_layer:
            continue
        for suffix in ("k_proj.weight", "v_proj.weight", "k_norm.weight"):
            state_dict.pop(f"{prefix}{idx}.self_attn.{suffix}", None)
    return state_dict


class Gemma3nAttention(Attention):
    """Gemma3n attention with per-head Q/K/V normalization and KV sharing.

    Extends the base Attention by adding a parameterless V normalization
    (``v_norm`` with ``with_scale=False`` in HF) applied after ``v_proj``.
    Q and K use the parent's per-head ``q_norm``/``k_norm``, built from
    :class:`RMSNorm` rather than ``OffsetRMSNorm`` because ``Gemma3nRMSNorm``
    scales by the gain directly with no ``1 + w`` offset.  Those two norms are
    load-bearing: HF hardcodes ``scaling = 1.0`` instead of ``head_dim**-0.5``
    precisely because they leave Q and K at unit RMS, so dropping them leaves
    the attention logits scaled by ``|q||k|`` and drives softmax to a near
    one-hot argmax.  They are enabled by ``config.attn_qk_norm``, which
    ``ArchitectureConfig.from_transformers`` keys off the model type.

    Layers at or after ``first_kv_shared_layer_idx`` are *KV-shared*: they
    borrow the already-RoPE'd K,V of the last non-shared layer of the same
    attention type instead of running their own ``k_proj``/``v_proj``.  Such
    layers own no KV cache entry and hold no K/V weights — matching HF
    ``Gemma3nTextAttention``, which does not even construct them.  Because the
    borrowed K,V already span past + current, they are passed whole and the
    layer supplies no ``past_key``/``past_value``; those layers therefore emit
    ``is_causal=0`` and lean on the explicit causal ``attn_mask`` (see
    :meth:`forward`).

    Args:
        config: Gemma3n configuration.
        layer_idx: Index of this layer (0-based).
        layer_types: Attention type per layer for all layers.
        first_kv_shared_layer_idx: First layer index that borrows K,V.
            ``0`` disables sharing entirely.
    """

    def __init__(
        self,
        config: Gemma3nConfig,
        layer_idx: int = 0,
        layer_types: list[str] | None = None,
        first_kv_shared_layer_idx: int = 0,
    ):
        # HF Gemma3nTextAttention hardcodes self.scaling = 1.0 (not head_dim**-0.5)
        super().__init__(config, rms_norm_class=RMSNorm, scale=1.0)
        self._v_norm_eps = config.rms_norm_eps
        self.layer_idx = layer_idx

        layer_types = layer_types or ["full_attention"] * config.num_hidden_layers
        # Mirrors HF modeling_gemma3n.py: shared layers reuse the K,V of the
        # LAST pre-cutoff layer whose attention type matches theirs, so sliding
        # and full-attention layers borrow from different sources.
        self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        prev_layers = layer_types[:first_kv_shared_layer_idx]
        if self.is_kv_shared_layer:
            self.kv_shared_layer_index = (
                len(prev_layers) - 1 - prev_layers[::-1].index(layer_types[layer_idx])
            )
            self.provides_shared_kv = False
        else:
            self.kv_shared_layer_index = None
            # The last non-shared layer of each type must publish its K,V.
            self.provides_shared_kv = first_kv_shared_layer_idx > 0 and (
                layer_idx
                == len(prev_layers) - 1 - prev_layers[::-1].index(layer_types[layer_idx])
            )

        if self.is_kv_shared_layer:
            # Drop the K/V projections and K norm the parent built: these layers
            # borrow K,V, and the weights are absent from what HF constructs.
            # Deleting the registered submodules keeps them out of the graph's
            # initializers and out of the expected state_dict.
            for name in ("k_proj", "v_proj", "k_norm"):
                self._modules.pop(name, None)
                object.__setattr__(self, name, None)

    def forward(
        self,
        op: OpBuilder,
        hidden_states,
        attention_bias,
        position_embeddings=None,
        past_key_value=None,
        static_cache=None,
        shared_kv_states=None,
    ):
        query_states = self.q_proj(op, hidden_states)

        # Per-head Q normalization (same order as parent Attention)
        if self.q_norm is not None:
            query_states = op.Reshape(query_states, [0, 0, -1, self.head_dim])
            query_states = self.q_norm(op, query_states)
            query_states = op.Reshape(query_states, [0, 0, -1])

        # Apply RoPE to Q (same as parent)
        if position_embeddings is not None:
            query_states = apply_rotary_pos_emb(
                op,
                x=query_states,
                position_embeddings=position_embeddings,
                num_heads=self.num_attention_heads,
                rotary_embedding_dim=self.rotary_embedding_dim,
                interleaved=self._rope_interleave,
            )

        if self.is_kv_shared_layer:
            # Borrow the source layer's ``present`` K,V — already normalized and
            # RoPE'd, and already spanning past + current positions, so this
            # layer passes no past_key/past_value of its own.
            src_key, src_value = shared_kv_states[self.kv_shared_layer_index]

            # The opset-24 ``Attention`` op's present_key/present_value outputs
            # have no shape inference in ORT, so they arrive rank-unknown; the
            # Transpose/Reshape below would then lose the head dim and make this
            # layer's o_proj MatMul fail shape inference at load time. The source
            # shares this layer's KV geometry, so pin the known BNSH shape.
            for _kv in (src_key, src_value):
                if _kv.shape is None or len(_kv.shape) != 4:
                    _kv.shape = ir.Shape(
                        [
                            "batch",
                            self.num_key_value_heads,
                            "kv_sequence_length",
                            self.head_dim,
                        ]
                    )

            kv_hidden = self.num_key_value_heads * self.head_dim
            key_states = op.Reshape(
                op.Transpose(src_key, perm=[0, 2, 1, 3]), [0, 0, kv_hidden]
            )
            value_states = op.Reshape(
                op.Transpose(src_value, perm=[0, 2, 1, 3]), [0, 0, kv_hidden]
            )
            past_key = past_value = None
        else:
            key_states = self.k_proj(op, hidden_states)
            value_states = self.v_proj(op, hidden_states)

            if self.k_norm is not None:
                key_states = op.Reshape(key_states, [0, 0, -1, self.head_dim])
                key_states = self.k_norm(op, key_states)
                key_states = op.Reshape(key_states, [0, 0, -1])

            # V normalization: parameterless RMS norm per head (with_scale=False
            # in HF). Reshape to (B, T, num_kv_heads, head_dim), normalize over
            # the last dim, reshape back.
            value_states = op.Reshape(
                value_states,
                op.Constant(value_ints=[0, 0, self.num_key_value_heads, self.head_dim]),
            )
            sq = op.Mul(value_states, value_states)
            mean_sq = op.ReduceMean(sq, [-1], keepdims=1)
            rms = op.Sqrt(op.Add(mean_sq, self._v_norm_eps))
            value_states = op.Div(value_states, rms)
            value_states = op.Reshape(value_states, [0, 0, -1])

            if position_embeddings is not None:
                key_states = apply_rotary_pos_emb(
                    op,
                    x=key_states,
                    position_embeddings=position_embeddings,
                    num_heads=self.num_key_value_heads,
                    rotary_embedding_dim=self.rotary_embedding_dim,
                    interleaved=self._rope_interleave,
                )
            past_key = past_key_value[0] if past_key_value is not None else None
            past_value = past_key_value[1] if past_key_value is not None else None

        if self.is_kv_shared_layer and attention_bias is None:
            raise ValueError(
                f"layer {self.layer_idx} is KV-shared and therefore relies on an "
                "explicit causal attn_mask, but attention_bias is None"
            )

        attn_output, present_key, present_value = _apply_attention(
            op,
            query_states,
            key_states,
            value_states,
            attention_bias,
            past_key,
            past_value,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            scale=self.scaling,
            static_cache=static_cache,
            # KV-shared layers hand K,V over WHOLE (past + current) with no
            # past_key/past_value, so at decode q_len=1 while kv_len=total.
            # The opset-24 spec aligns the built-in causal mask UPPER-LEFT --
            # "the attention masking has the form of the upper left causal bias
            # due to the alignment" -- which pins that lone query to key 0 and
            # collapses the layer to the BOS token.  ORT <= 1.27 aligned it
            # bottom-right (non-conforming) and hid this; 1.28 conforms and the
            # decode output degenerates.  ``attention_bias`` from
            # ``create_attention_bias`` already bakes causal + sliding + padding
            # keyed on absolute cumsum positions, so it carries causality on its
            # own at any q_len/kv_len ratio.  Non-shared layers keep is_causal=1:
            # they pass past_key/past_value, which takes the op's cache path
            # (bottom-right aligned in every ORT version), and at prefill
            # q_len == kv_len makes the two alignments identical.
            is_causal=0 if self.is_kv_shared_layer else 1,
        )

        # Source layers publish their K,V for the downstream shared layers.
        if self.provides_shared_kv and shared_kv_states is not None:
            shared_kv_states[self.layer_idx] = (present_key, present_value)

        attn_output = self.o_proj(op, attn_output)
        return attn_output, (present_key, present_value)


class Gemma3nMLP(MLP):
    """Gemma3n MLP with optional Gaussian-top-k activation sparsity.

    Where ``config.activation_sparsity_pattern[layer_idx]`` is non-zero, the
    gate projection is sparsified before the activation: HF's
    ``_gaussian_topk`` treats each row of ``gate_proj`` as a Gaussian sample,
    derives the value below which ``sparsity`` of the mass falls, and clamps
    everything under it to zero::

        cutoff = mean(x) + std(x) * Phi^-1(sparsity)
        x = relu(x - cutoff)

    The mean and (population, ``unbiased=False``) std are per-row over the
    intermediate dim.  ``Phi^-1(sparsity)`` — the inverse standard-normal CDF
    — depends only on the config, so it is folded into a Python float at build
    time rather than needing an ONNX erfinv.

    E4B applies 0.95 sparsity to layers 0..9, zeroing roughly 95% of each
    row's gate activations; the remaining layers use the plain MLP path.

    Args:
        config: Gemma3n configuration.
        layer_idx: Index of the owning decoder layer, used to look up this
            layer's entry in ``config.activation_sparsity_pattern``.
        linear_class: Factory for the projection layers (see :class:`MLP`).
    """

    def __init__(
        self,
        config: Gemma3nConfig,
        layer_idx: int = 0,
        linear_class: type | None = None,
    ):
        super().__init__(config, linear_class=linear_class)
        pattern = config.activation_sparsity_pattern
        if pattern and layer_idx >= len(pattern):
            raise ValueError(
                f"activation_sparsity_pattern has {len(pattern)} entries but "
                f"layer {layer_idx} needs one; it must cover every layer"
            )
        sparsity = float(pattern[layer_idx]) if pattern else 0.0
        if not 0.0 <= sparsity < 1.0:
            raise ValueError(
                f"activation_sparsity_pattern[{layer_idx}] must be in [0, 1), got {sparsity}"
            )
        self.activation_sparsity = sparsity
        # Phi^-1(sparsity): HF computes this as
        # torch.distributions.Normal(0, 1).icdf(sparsity).
        self._std_multiplier = NormalDist().inv_cdf(sparsity) if sparsity > 0.0 else 0.0

    def forward(self, op: OpBuilder, x: ir.Value):
        gate = self.gate_proj(op, x)
        if self.activation_sparsity > 0.0:
            gate = self._gaussian_topk(op, gate)
        gate = self.act_fn(op, gate)
        up = self.up_proj(op, x)
        return self.down_proj(op, op.Mul(gate, up))

    def _gaussian_topk(self, op: OpBuilder, x: ir.Value) -> ir.Value:
        """Zero the gate values below the per-row Gaussian sparsity quantile."""
        mean = op.ReduceMean(x, [-1], keepdims=1)
        centered = op.Sub(x, mean)
        # Population variance (unbiased=False), matching HF's torch.std call.
        variance = op.ReduceMean(op.Mul(centered, centered), [-1], keepdims=1)
        std = op.Sqrt(variance)
        cutoff = op.Add(mean, op.Mul(std, self._std_multiplier))
        return op.Relu(op.Sub(x, cutoff))


class Gemma3nScaledWordEmbedding(Embedding):
    """Embedding with scaling by sqrt(hidden_size)."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        embed_scale: float = 1.0,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_scale = embed_scale

    def forward(self, op: OpBuilder, input_ids: ir.Value):
        embeddings = super().forward(op, input_ids)
        return op.Mul(embeddings, self.embed_scale)


class Gemma3nLaurelBlock(nn.Module):
    """Learned Augmented Residual Layer (Laurel).

    Applies a low-rank residual: output = x + RMSNorm(W_right @ W_left @ x).
    """

    def __init__(self, hidden_size: int, laurel_rank: int, eps: float = 1e-6):
        super().__init__()
        self.linear_left = Linear(hidden_size, laurel_rank, bias=False)
        self.linear_right = Linear(laurel_rank, hidden_size, bias=False)
        self.post_laurel_norm = RMSNorm(hidden_size, eps=eps)

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        laurel_hidden = self.linear_left(op, hidden_states)
        laurel_hidden = self.linear_right(op, laurel_hidden)
        normed = self.post_laurel_norm(op, laurel_hidden)
        return op.Add(hidden_states, normed)


class Gemma3nAltUp(nn.Module):
    """Alternating Updates (AltUp).

    Wraps transformer layers with predict/correct steps that enable sparse
    dimension updates. Only the active prediction is processed through the
    transformer layer; the rest are corrected using learned coefficients.

    See: https://proceedings.neurips.cc/paper_files/paper/2023/file/
    f2059277ac6ce66e7e5543001afa8bb5-Paper-Conference.pdf
    """

    def __init__(self, config: Gemma3nConfig):
        super().__init__()
        self.altup_num_inputs = config.altup_num_inputs
        self.altup_active_idx = config.altup_active_idx
        self.hidden_size = config.hidden_size

        self.correction_coefs = Linear(
            config.altup_num_inputs, config.altup_num_inputs, bias=False
        )
        self.prediction_coefs = Linear(
            config.altup_num_inputs, config.altup_num_inputs**2, bias=False
        )
        self.modality_router = Linear(config.hidden_size, config.altup_num_inputs, bias=False)
        self.router_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.router_input_scale = float(config.hidden_size**-1.0)

    def _compute_router_modalities(self, op: OpBuilder, x):
        """Compute router modalities: tanh(router(norm(x) * scale))."""
        router_input = self.router_norm(op, x)
        router_input = op.Mul(router_input, self.router_input_scale)
        routed = self.modality_router(op, router_input)
        return op.Tanh(routed)

    def predict(self, op: OpBuilder, hidden_states_list: list):
        """Predict step: modify inputs using learned prediction coefficients.

        Args:
            hidden_states_list: List of altup_num_inputs tensors, each
                [batch, seq_len, hidden_size].

        Returns:
            List of predicted tensors.
        """
        active = hidden_states_list[self.altup_active_idx]
        modalities = self._compute_router_modalities(op, active)

        # prediction_coefs projects modalities to num_inputs^2 coefficients
        all_coefs = self.prediction_coefs(op, modalities)
        # Reshape to [batch, seq, num_inputs, num_inputs]
        all_coefs = op.Reshape(
            all_coefs,
            op.Constant(value_ints=[0, 0, self.altup_num_inputs, self.altup_num_inputs]),
        )
        # Transpose to [batch, seq, num_inputs_out, num_inputs_in]
        all_coefs = op.Transpose(all_coefs, perm=[0, 1, 3, 2])

        # Stack hidden states: [batch, seq, hidden, num_inputs]
        stacked = op.Concat(*[op.Unsqueeze(h, [-1]) for h in hidden_states_list], axis=-1)
        # matmul: [batch, seq, hidden, num_inputs] x [batch, seq, num_inputs, num_inputs]
        # -> [batch, seq, hidden, num_inputs]
        predictions_stacked = op.MatMul(stacked, all_coefs)
        # Add residual
        predictions_stacked = op.Add(predictions_stacked, stacked)

        # Split back to list
        predictions = []
        for i in range(self.altup_num_inputs):
            idx = op.Constant(value_ints=[i])
            pred_i = op.Gather(predictions_stacked, idx, axis=-1)
            pred_i = op.Squeeze(pred_i, [-1])
            predictions.append(pred_i)
        return predictions

    def correct(self, op: OpBuilder, predictions: list, activated):
        """Correct step: propagate transformer output to all predictions.

        Args:
            predictions: List of predicted tensors from predict step.
            activated: Output of the transformer layer for the active prediction.

        Returns:
            List of corrected prediction tensors.
        """
        modalities = self._compute_router_modalities(op, activated)
        innovation = op.Sub(activated, predictions[self.altup_active_idx])

        # correction_coefs: [batch, seq, num_inputs] + 1
        all_coefs = self.correction_coefs(op, modalities)
        all_coefs = op.Add(all_coefs, 1.0)

        corrected = []
        for i in range(self.altup_num_inputs):
            idx = op.Constant(value_ints=[i])
            coef_i = op.Gather(all_coefs, idx, axis=-1)
            scaled_innovation = op.Mul(innovation, coef_i)
            corrected_i = op.Add(predictions[i], scaled_innovation)
            corrected.append(corrected_i)
        return corrected

    def scale_corrected_output(self, op: OpBuilder, corrected, scale):
        """Apply per-dimension scaling to the corrected output."""
        return op.Mul(corrected, scale)


class Gemma3nDecoderLayer(nn.Module):
    """Gemma3n decoder layer with AltUp, Laurel, and per-layer input gating.

    Args:
        config: Gemma3n configuration.
        layer_idx: Index of this layer (0-based).
        layer_types: Attention type per layer for all layers. Defaults to
            ``config.layer_types``.
        first_kv_shared_layer_idx: First layer index that borrows K,V from an
            earlier layer (``0`` disables KV sharing).
    """

    def __init__(
        self,
        config: Gemma3nConfig,
        layer_idx: int,
        layer_types: list[str] | None = None,
        first_kv_shared_layer_idx: int = 0,
    ):
        super().__init__()
        self.self_attn = Gemma3nAttention(
            config,
            layer_idx=layer_idx,
            layer_types=layer_types or config.layer_types,
            first_kv_shared_layer_idx=first_kv_shared_layer_idx,
        )
        self.mlp = Gemma3nMLP(config, layer_idx=layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.altup = Gemma3nAltUp(config)
        self.laurel = Gemma3nLaurelBlock(
            config.hidden_size, config.laurel_rank, eps=config.rms_norm_eps
        )

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        self.per_layer_input_gate = Linear(
            config.hidden_size, config.hidden_size_per_layer_input, bias=False
        )
        self.per_layer_projection = Linear(
            config.hidden_size_per_layer_input, config.hidden_size, bias=False
        )
        self.post_per_layer_input_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.altup_active_idx = config.altup_active_idx
        self.altup_correct_scale = config.altup_correct_scale
        # per_layer_input_gate uses the same activation as the MLP gate projection
        self.act_fn = get_activation(config.hidden_act)

        # Placed on DecoderLayer (not AltUp) so __call__ realizes it
        self.correct_output_scale = nn.Parameter([config.hidden_size])

    def forward(
        self,
        op: OpBuilder,
        hidden_states_list: list,
        attention_bias: ir.Value,
        position_embeddings: tuple,
        per_layer_input: ir.Value,
        past_key_value: tuple | None,
        shared_kv_states: dict | None = None,
    ):
        # AltUp predict
        predictions = self.altup.predict(op, hidden_states_list)
        active = predictions[self.altup_active_idx]

        # Pre-attention norm + Laurel
        active_normed = self.input_layernorm(op, active)
        laurel_output = self.laurel(op, active_normed)

        # Self attention
        attn_output, present_key_value = self.self_attn(
            op,
            hidden_states=active_normed,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            shared_kv_states=shared_kv_states,
        )
        attn_output = self.post_attention_layernorm(op, attn_output)

        # Residual + Laurel (with sqrt(2) normalization)
        attn_gated = op.Add(active, attn_output)
        attn_laurel = op.Add(attn_gated, laurel_output)
        attn_laurel = op.Div(attn_laurel, float(math.sqrt(2)))

        # MLP
        mlp_input = self.pre_feedforward_layernorm(op, attn_laurel)
        mlp_output = self.mlp(op, mlp_input)
        mlp_output = self.post_feedforward_layernorm(op, mlp_output)
        layer_output = op.Add(attn_laurel, mlp_output)

        # AltUp correct
        corrected = self.altup.correct(op, predictions, layer_output)

        # Scale and apply per-layer input
        first = corrected[self.altup_active_idx]
        if self.altup_correct_scale:
            first = self.altup.scale_corrected_output(op, first, self.correct_output_scale)

        gated = self.per_layer_input_gate(op, first)
        # Config activation (gelu_pytorch_tanh by default) matches HF's act_fn
        gated = self.act_fn(op, gated)
        gated = op.Mul(gated, per_layer_input)
        projected = self.per_layer_projection(op, gated)
        projected = self.post_per_layer_input_norm(op, projected)

        # Add projected per-layer input to non-active predictions
        for i in range(len(corrected)):
            if i != self.altup_active_idx:
                corrected[i] = op.Add(corrected[i], projected)

        return corrected, present_key_value


class Gemma3nTextModel(nn.Module):
    """Gemma3n text model with AltUp, Laurel, and hybrid attention."""

    def __init__(self, config: Gemma3nConfig):
        super().__init__()
        self._dtype = config.dtype

        embed_scale = float(np.float16(config.hidden_size**0.5))
        self.embed_tokens = Gemma3nScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            embed_scale=embed_scale,
        )
        self.layer_types = config.layer_types or (
            ["full_attention"] * config.num_hidden_layers
        )
        if len(self.layer_types) != config.num_hidden_layers:
            raise ValueError(
                f"Gemma3nConfig.layer_types length ({len(self.layer_types)}) must match "
                f"num_hidden_layers ({config.num_hidden_layers})"
            )
        # Layers at/after this index borrow K,V from an earlier layer of the
        # same attention type and own no KV cache entry.
        self.first_kv_shared_layer_idx = config.num_hidden_layers - (
            config.num_kv_shared_layers or 0
        )
        if self.first_kv_shared_layer_idx <= 0:
            # All layers "shared" would leave no source to borrow from.
            raise ValueError(
                f"num_kv_shared_layers ({config.num_kv_shared_layers}) must be less "
                f"than num_hidden_layers ({config.num_hidden_layers}); every layer "
                "cannot borrow K,V."
            )
        self.layers = nn.ModuleList(
            [
                Gemma3nDecoderLayer(
                    config,
                    i,
                    layer_types=self.layer_types,
                    first_kv_shared_layer_idx=self.first_kv_shared_layer_idx,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.num_kv_layers = self.first_kv_shared_layer_idx
        self.sliding_window = config.sliding_window

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

        # Local RoPE for sliding window layers
        local_config = copy.deepcopy(config)
        local_config.rope_theta = config.rope_local_base_freq
        local_config.rope_type = "default"
        local_config.rope_scaling = None
        self.rotary_emb_local = initialize_rope(local_config)

        # Per-layer input embeddings
        self.embed_tokens_per_layer = Gemma3nScaledWordEmbedding(
            config.vocab_size_per_layer_input,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            config.pad_token_id,
            embed_scale=float(config.hidden_size_per_layer_input**0.5),
        )

        self.per_layer_model_projection = Linear(
            config.hidden_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            bias=False,
        )
        self.per_layer_projection_norm = RMSNorm(
            config.hidden_size_per_layer_input, eps=config.rms_norm_eps
        )

        # AltUp projections for expanding input embeddings to altup_num_inputs copies
        self.altup_num_inputs = config.altup_num_inputs
        self.altup_active_idx = config.altup_active_idx
        self.altup_projections = nn.ModuleList(
            [
                Linear(config.hidden_size, config.hidden_size, bias=False)
                for _ in range(config.altup_num_inputs - 1)
            ]
        )
        self.altup_unembed_projections = nn.ModuleList(
            [
                Linear(config.hidden_size, config.hidden_size, bias=False)
                for _ in range(config.altup_num_inputs - 1)
            ]
        )

        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        self.per_layer_projection_scale = float(config.hidden_size**-0.5)
        self.per_layer_input_scale = float(1.0 / math.sqrt(2.0))
        # Epsilon used when normalizing AltUp projection magnitudes
        self._altup_eps = 1e-5

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list | None = None,
        inputs_embeds: ir.Value | None = None,
        per_layer_inputs: ir.Value | None = None,
    ):
        if inputs_embeds is not None:
            hidden_states_0 = inputs_embeds
        else:
            hidden_states_0 = self.embed_tokens(op, input_ids)

        # Per-layer inputs.  In the multimodal split the embedding sub-model owns
        # the per-layer tables and hands the combined ``[B, S, L*D]`` tensor in
        # as a graph input; the text-only path derives it here from ``input_ids``.
        if per_layer_inputs is not None:
            per_layer_inputs = op.Reshape(
                per_layer_inputs,
                op.Constant(
                    value_ints=[
                        0,
                        0,
                        self.num_hidden_layers,
                        self.hidden_size_per_layer_input,
                    ]
                ),
            )
        else:
            per_layer_inputs = self._compute_per_layer_inputs(op, input_ids, hidden_states_0)

        position_embeddings_dict = {
            "full_attention": self.rotary_emb(op, position_ids),
            "sliding_attention": self.rotary_emb_local(op, position_ids),
        }

        # Use hidden_states_0 for query length when input_ids is None (VL decoder path)
        query_input = input_ids if input_ids is not None else hidden_states_0
        attention_bias_dict = {
            "full_attention": create_attention_bias(
                op,
                input_ids=query_input,
                attention_mask=attention_mask,
                dtype=self._dtype,
            ),
            "sliding_attention": create_attention_bias(
                op,
                input_ids=query_input,
                attention_mask=attention_mask,
                sliding_window=self.sliding_window,
                dtype=self._dtype,
            ),
        }

        # Expand to AltUp inputs with magnitude normalization.
        # HF normalizes each projection's magnitude to match hidden_states_0's RMS,
        # avoiding scale mismatches between the original embedding and its projections.
        hidden_states_list = [hidden_states_0]
        # target_magnitude = sqrt(mean(x^2, dim=-1, keepdim)) — RMS of input embedding
        tgt_sq = op.ReduceMean(op.Mul(hidden_states_0, hidden_states_0), [-1], keepdims=1)
        target_mag = op.Sqrt(tgt_sq)  # (B, T, 1)
        _altup_eps = self._altup_eps
        for proj in self.altup_projections:
            altup_proj = proj(op, hidden_states_0)
            new_sq = op.ReduceMean(op.Mul(altup_proj, altup_proj), [-1], keepdims=1)
            # Clip at epsilon to avoid division by zero, then scale to target magnitude
            new_mag = op.Sqrt(op.Max(new_sq, _altup_eps))
            altup_proj = op.Mul(altup_proj, op.Div(target_mag, new_mag))
            hidden_states_list.append(altup_proj)

        # Decoder layers.  ``shared_kv_states`` is populated by the source
        # layers and consumed by the KV-shared layers that follow them.
        shared_kv_states: dict = {}
        present_key_values = []
        # ``past_key_values`` carries only ``num_kv_layers`` entries (KV-shared
        # layers own no cache), so expand it to a full per-layer list to zip
        # over every layer without truncating the tail.
        if past_key_values is not None:
            kv_iter = iter(past_key_values)
            past_kvs: list = [
                None if layer.self_attn.is_kv_shared_layer else next(kv_iter)
                for layer in self.layers
            ]
        else:
            past_kvs = [None] * len(self.layers)

        for i, (layer, layer_type, past_kv) in enumerate(
            zip(self.layers, self.layer_types, past_kvs)
        ):
            # Extract per-layer input for this layer
            per_layer_input = self._get_per_layer_input(op, per_layer_inputs, i)

            hidden_states_list, present_kv = layer(
                op,
                hidden_states_list=hidden_states_list,
                attention_bias=attention_bias_dict[layer_type],
                position_embeddings=position_embeddings_dict[layer_type],
                per_layer_input=per_layer_input,
                past_key_value=past_kv,
                shared_kv_states=shared_kv_states,
            )
            # KV-shared layers reuse a source layer's K,V — exclude them so the
            # cache output has exactly ``num_kv_layers`` entries.
            if not layer.self_attn.is_kv_shared_layer:
                present_key_values.append(present_kv)

        # Collapse AltUp outputs back to single hidden state with magnitude normalization.
        # HF normalizes each unembed projection's magnitude to match hidden_states_list[0]'s
        # RMS before averaging, preserving the output scale.
        tgt_sq = op.ReduceMean(
            op.Mul(hidden_states_list[0], hidden_states_list[0]), [-1], keepdims=1
        )
        target_mag = op.Sqrt(tgt_sq)  # (B, T, 1)
        result_list = [hidden_states_list[0]]
        for i, proj in enumerate(self.altup_unembed_projections):
            unembed = proj(op, hidden_states_list[i + 1])
            new_sq = op.ReduceMean(op.Mul(unembed, unembed), [-1], keepdims=1)
            new_mag = op.Sqrt(op.Max(new_sq, self._altup_eps))
            unembed = op.Mul(unembed, op.Div(target_mag, new_mag))
            result_list.append(unembed)

        # Average all AltUp outputs
        hidden_states = result_list[0]
        for h in result_list[1:]:
            hidden_states = op.Add(hidden_states, h)
        scale = 1.0 / self.altup_num_inputs
        hidden_states = op.Mul(hidden_states, scale)

        hidden_states = self.norm(op, hidden_states)
        return hidden_states, present_key_values

    def _compute_per_layer_inputs(self, op, input_ids, inputs_embeds):
        """Compute per-layer input embeddings from input_ids and model projection.

        Requires ``input_ids``: the token-embedding term is not optional in HF,
        so dropping it (as an ``inputs_embeds``-only decoder would have to)
        would silently skip both the add and the ``1/sqrt(2)`` scale.  Callers
        without ``input_ids`` must pass ``per_layer_inputs`` to
        :meth:`forward` instead.
        """
        if input_ids is None:
            raise ValueError(
                "Gemma3nTextModel needs either input_ids (to look up the per-layer "
                "token embeddings) or a precomputed per_layer_inputs tensor; got "
                "neither."
            )
        per_layer_token_embed = self.embed_tokens_per_layer(op, input_ids)

        # Per-layer projection from hidden states
        per_layer_proj = self.per_layer_model_projection(op, inputs_embeds)
        per_layer_proj = op.Mul(per_layer_proj, self.per_layer_projection_scale)

        # Reshape to [batch, seq, num_layers, per_layer_dim]
        per_layer_proj = op.Reshape(
            per_layer_proj,
            op.Constant(
                value_ints=[0, 0, self.num_hidden_layers, self.hidden_size_per_layer_input]
            ),
        )
        per_layer_proj = self.per_layer_projection_norm(op, per_layer_proj)

        per_layer_token_embed = op.Reshape(
            per_layer_token_embed,
            op.Constant(
                value_ints=[0, 0, self.num_hidden_layers, self.hidden_size_per_layer_input]
            ),
        )
        per_layer_inputs = op.Add(per_layer_proj, per_layer_token_embed)
        return op.Mul(per_layer_inputs, self.per_layer_input_scale)

    def _get_per_layer_input(self, op, per_layer_inputs, layer_idx: int):
        """Extract the per-layer input for a specific layer."""
        idx = op.Constant(value_ints=[layer_idx])
        # per_layer_inputs shape: [batch, seq, num_layers, per_layer_dim]
        # Gather on axis=2 to get [batch, seq, per_layer_dim]
        result = op.Gather(per_layer_inputs, idx, axis=2)
        return op.Squeeze(result, [2])


class Gemma3nCausalLMModel(CausalLMModel):
    """Gemma3n causal LM with AltUp, Laurel, and hybrid attention.

    Extends CausalLMModel with the Gemma3n text backbone that includes
    alternating updates, learned augmented residuals, and per-layer
    input gating for mobile efficiency.
    """

    config_class: type = Gemma3nConfig

    def __init__(self, config: Gemma3nConfig):
        super().__init__(config)
        self.model = Gemma3nTextModel(config)
        self._final_logit_softcapping = config.final_logit_softcapping

    def kv_cache_layer_count(self) -> int:
        """Number of layers owning a KV cache entry (excludes KV-shared layers)."""
        return self.model.num_kv_layers

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        attention_mask: ir.Value | None,
        position_ids: ir.Value,
        past_key_values: list | None = None,
    ):
        hidden_states, present_key_values = self.model(
            op,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        logits = self.lm_head(op, hidden_states)
        return _apply_logit_softcapping(op, logits, self._final_logit_softcapping), (
            present_key_values
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Preprocess weights, handling language_model prefix from multimodal."""
        for key in list(state_dict.keys()):
            if "language_model." in key:
                new_key = key.replace("language_model.", "")
                state_dict[new_key] = state_dict.pop(key)
            elif "vision_tower" in key or "multi_modal_projector" in key:
                state_dict.pop(key)
            elif "audio_tower" in key:
                state_dict.pop(key)

        state_dict = _strip_altup_prefix(state_dict)
        state_dict = _drop_kv_shared_layer_weights(state_dict, self.model)
        return super().preprocess_weights(state_dict)


# ---------------------------------------------------------------------------
# Gemma 3n multimodal sub-models
# ---------------------------------------------------------------------------


class _Gemma3nDecoderModel(nn.Module):
    """Gemma 3n text decoder sub-model consuming ``inputs_embeds``.

    The embedding sub-model owns both the token embedding and the per-layer
    embedding tables, so this graph takes the combined ``per_layer_inputs``
    ``[B, S, L*D]`` as an input and never needs ``input_ids``.  That keeps the
    4.7 GB ``embed_tokens_per_layer`` table out of the decoder and — unlike
    deriving the per-layer inputs from ``inputs_embeds`` alone — preserves the
    token-embedding term that HF's ``project_per_layer_inputs`` adds.
    """

    def __init__(self, config: Gemma3nMultiModalConfig):
        super().__init__()
        self.config = config
        self.model = Gemma3nTextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def kv_cache_layer_count(self) -> int:
        """Number of layers owning a KV cache entry (excludes KV-shared layers)."""
        return self.model.num_kv_layers

    def forward(
        self,
        op: OpBuilder,
        inputs_embeds: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        per_layer_inputs: ir.Value | None = None,
        past_key_values: list | None = None,
    ):
        hidden_states, present_key_values = self.model(
            op,
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
        )
        logits = self.lm_head(op, hidden_states)
        logits = _apply_logit_softcapping(op, logits, self.config.final_logit_softcapping)
        return logits, present_key_values


class _Gemma3nVisionEncoderModel(nn.Module):
    """Gemma 3n vision tower sub-model: pixels -> text-space soft tokens.

    Mirrors HF ``Gemma3nModel.get_image_features``::

        pixel_values [B, 3, 768, 768]
        -> encoder                  [B, vision_hidden, 16, 16]
        -> reshape + transpose      [B, 256, vision_hidden]
        -> * sqrt(vision_hidden)
        -> embed_vision (soft)      [B, 256, text_hidden]
        -> reshape                  [B*256, text_hidden]

    Only the embedder's *soft* path is built here, so this graph carries
    ``soft_embedding_norm`` but not the 128-row hard lookup table (which the
    embedding sub-model owns and uses for the placeholder ids).
    """

    def __init__(self, config: Gemma3nMultiModalConfig):
        super().__init__()
        vision = config.vision
        if vision is None:
            raise ValueError(
                "Gemma3n vision encoder requires config.vision; got None. The "
                "gemma3n vision extractor hook populates it from vision_config."
            )
        vision_hidden = vision.hidden_size or 2048
        eps = vision.rms_norm_eps or vision.norm_eps or config.rms_norm_eps
        self.encoder = MobileNetV5Encoder(
            hidden_size=vision_hidden,
            image_size=vision.image_size or 768,
            norm_eps=eps,
        )
        self.embed_vision = Gemma3nMultimodalEmbedder(
            vision_hidden,
            config.hidden_size,
            vocab_size=vision.vocab_size or 128,
            vocab_offset=vision.vocab_offset or 0,
            eps=eps,
        )
        self._vision_hidden = vision_hidden
        self._soft_tokens = config.vision_soft_tokens_per_image
        self._text_hidden_size = config.hidden_size

    def forward(self, op: OpBuilder, pixel_values: ir.Value) -> ir.Value:
        # [B, 3, S, S] -> [B, vision_hidden, 16, 16]
        features = self.encoder(op, pixel_values)
        # Flatten the spatial grid into soft tokens, then move channels last.
        # Both extents are compile-time constants, so no Shape op is needed.
        features = op.Reshape(
            features,
            op.Constant(value_ints=[0, self._vision_hidden, self._soft_tokens]),
        )
        features = op.Transpose(features, perm=[0, 2, 1])
        features = op.Mul(features, float(self._vision_hidden**0.5))
        features = self.embed_vision(op, inputs_embeds=features)
        # [B, 256, text_hidden] -> [B*256, text_hidden] so the rows line up 1:1
        # with the image placeholder tokens the processor spliced into the
        # prompt, matching the 2-D ``image_features`` contract of the
        # embedding sub-model (which Gathers along axis 0).
        return op.Reshape(features, op.Constant(value_ints=[-1, self._text_hidden_size]))


class _Gemma3nAudioEncoderModel(nn.Module):
    """Gemma 3n USM audio tower sub-model: mel frames -> text-space soft tokens.

    Mirrors HF ``Gemma3nModel.get_audio_features`` plus the padding step
    ``Gemma3nModel.forward`` applies right after it.  The processor always
    splices exactly ``audio_soft_tokens_per_image`` placeholders into the
    prompt, while the encoder produces *at most* that many frames, so both
    padded and missing frames are filled with the embedding of the last id in
    the audio vocabulary.  The output therefore always has a fixed row count
    per clip and needs no companion validity mask.

    This is the one place a single embedder is used through *both* of its
    paths: soft for the encoder features, hard for that padding token.
    """

    def __init__(self, config: Gemma3nMultiModalConfig):
        super().__init__()
        audio = config.audio
        if audio is None:
            raise ValueError(
                "Gemma3n audio encoder requires config.audio; got None. Build "
                "the package without an audio_encoder component instead."
            )
        audio_hidden = audio.hidden_size or 1536
        eps = audio.rms_norm_eps or config.rms_norm_eps
        self.encoder = Gemma3nAudioEncoder(
            input_feat_size=audio.input_feat_size,
            hidden_size=audio_hidden,
            num_heads=audio.conf_num_attention_heads,
            num_layers=audio.conf_num_hidden_layers,
            conv_kernel_size=audio.conf_conv_kernel_size,
            conv_channel_size=audio.sscp_conv_channel_size,
            conv_kernel_size_2d=audio.sscp_conv_kernel_size,
            conv_stride_size=audio.sscp_conv_stride_size,
            conv_group_norm_eps=audio.sscp_conv_group_norm_eps,
            attention_chunk_size=audio.conf_attention_chunk_size,
            attention_context_left=audio.conf_attention_context_left,
            attention_context_right=audio.conf_attention_context_right,
            attention_logit_cap=audio.conf_attention_logit_cap,
            reduction_factor=audio.conf_reduction_factor,
            rms_norm_eps=eps,
            residual_weight=audio.conf_residual_weight,
            gradient_clipping=audio.gradient_clipping,
        )
        vocab_size = audio.vocab_size or 128
        vocab_offset = audio.vocab_offset or 0
        self.embed_audio = Gemma3nMultimodalEmbedder(
            audio_hidden,
            config.hidden_size,
            vocab_size=vocab_size,
            vocab_offset=vocab_offset,
            eps=eps,
        )
        # HF pads with ``self.vocab_size - 1``, the last id of the *audio*
        # table; the embedder rebases by ``vocab_offset`` internally, so the
        # absolute id is what gets passed in.
        self._padding_token_id = vocab_offset + vocab_size - 1
        self._soft_tokens = config.audio_soft_tokens_per_image
        self._text_hidden_size = config.hidden_size

    def forward(
        self,
        op: OpBuilder,
        input_features: ir.Value,
        input_features_mask: ir.Value,
    ) -> ir.Value:
        # [B, T, mel] -> [B, T', audio_hidden] plus the downsampled valid mask.
        # ``Gemma3nAudioEncoder`` takes True = valid, which is already the
        # polarity of the task's graph input (HF negates because its
        # ``audio_mel_mask`` marks padding).
        encodings, mask = self.encoder(op, input_features, input_features_mask)
        features = self.embed_audio(op, inputs_embeds=encodings)

        # Embedding of the audio padding token, [1, 1, text_hidden].  The id
        # tensor is rank 2 so the hard lookup broadcasts against the
        # [B, T', text_hidden] features below.
        padding_id = op.Constant(
            value=ir.tensor(np.array([[self._padding_token_id]], dtype=np.int64))
        )
        padding_embed = self.embed_audio(op, input_ids=padding_id)
        padding_embed = op.CastLike(padding_embed, features)

        # Replace padded frames, then extend to the fixed placeholder count.
        features = op.Where(op.Unsqueeze(mask, [-1]), features, padding_embed)
        features = self._pad_to_soft_tokens(op, features, padding_embed)
        # [B, soft_tokens, text_hidden] -> [B*soft_tokens, text_hidden], the
        # 2-D contract the embedding sub-model Gathers from.
        return op.Reshape(features, op.Constant(value_ints=[-1, self._text_hidden_size]))

    def _pad_to_soft_tokens(
        self, op: OpBuilder, features: ir.Value, padding_embed: ir.Value
    ) -> ir.Value:
        """Force the token axis of *features* to ``audio_soft_tokens_per_image``.

        Shorter clips are extended with *padding_embed* (HF's
        ``extra_padding_features``).  Longer ones are truncated: HF cannot hit
        that case because its encoder is fed 30 s of audio, but a longer clip
        would otherwise emit more rows than there are placeholders and
        misalign every subsequent token.
        """
        soft_tokens = op.Constant(value_ints=[self._soft_tokens])
        batch = op.Shape(features, start=0, end=1)
        length = op.Shape(features, start=1, end=2)
        # Clamp at zero so an over-long clip does not ask Expand for a
        # negative extent; the Slice below trims it instead.
        extra = op.Max(op.Sub(soft_tokens, length), op.Constant(value_ints=[0]))
        pad_shape = op.Concat(
            batch, extra, op.Constant(value_ints=[self._text_hidden_size]), axis=0
        )
        padded = op.Concat(features, op.Expand(padding_embed, pad_shape), axis=1)
        return op.Slice(
            padded,
            op.Constant(value_ints=[0]),
            soft_tokens,
            op.Constant(value_ints=[1]),
        )


class _Gemma3nEmbeddingModel(nn.Module):
    """Gemma 3n embedding sub-model: token lookup + multimodal fusion.

    Reproduces the embedding half of HF ``Gemma3nModel.forward``:

    1. scaled token embedding;
    2. **hard** placeholder embeddings — the reserved vision/audio token ids
       are embedded through their modality's 128-row lookup table and
       overwrite those positions;
    3. **soft** feature scatter — encoder output rows replace the
       ``image_token_id`` / ``audio_token_id`` positions;
    4. per-layer inputs, projected from the *fused* embeddings and added to
       the per-layer token embedding.

    Steps 2 and 3 are distinct on purpose: the ``boi``/``eoi`` style markers in
    the reserved id range get the hard embedding, while the soft tokens
    standing in for the actual image and audio content get the encoder
    features.

    Inputs are ``input_ids [B, S]`` INT64, ``image_features
    [num_image_tokens, hidden]`` and — with audio configured — ``audio_features
    [num_audio_tokens, hidden]``.  Outputs are ``inputs_embeds [B, S, hidden]``
    and ``per_layer_inputs [B, S, L*D]``.
    """

    def __init__(self, config: Gemma3nMultiModalConfig):
        super().__init__()
        self.config = config
        vision = config.vision
        if vision is None:
            raise ValueError(
                "Gemma3n embedding model requires config.vision; got None. The "
                "gemma3n vision extractor hook populates it from vision_config."
            )
        self.embed_tokens = Gemma3nScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            embed_scale=float(np.float16(config.hidden_size**0.5)),
        )

        self._num_layers = config.num_hidden_layers
        self._per_layer_dim = config.hidden_size_per_layer_input
        self.embed_tokens_per_layer = Gemma3nScaledWordEmbedding(
            config.vocab_size_per_layer_input,
            self._num_layers * self._per_layer_dim,
            config.pad_token_id,
            embed_scale=float(self._per_layer_dim**0.5),
        )
        self.per_layer_model_projection = Linear(
            config.hidden_size,
            self._num_layers * self._per_layer_dim,
            bias=False,
        )
        self.per_layer_projection_norm = RMSNorm(self._per_layer_dim, eps=config.rms_norm_eps)

        vision_eps = vision.rms_norm_eps or vision.norm_eps or config.rms_norm_eps
        self._vision_vocab_offset = vision.vocab_offset or 0
        self._vision_vocab_size = vision.vocab_size or 128
        self.embed_vision = Gemma3nMultimodalEmbedder(
            vision.hidden_size or 2048,
            config.hidden_size,
            vocab_size=self._vision_vocab_size,
            vocab_offset=self._vision_vocab_offset,
            eps=vision_eps,
        )
        self.image_token_id = vision.image_token_id or config.image_token_id

        audio = config.audio
        self.audio_token_id: int | None = None
        self._audio_vocab_offset: int | None = None
        if audio is not None:
            self._audio_vocab_offset = audio.vocab_offset or 0
            self.embed_audio = Gemma3nMultimodalEmbedder(
                audio.hidden_size or 1536,
                config.hidden_size,
                vocab_size=audio.vocab_size or 128,
                vocab_offset=self._audio_vocab_offset,
                eps=audio.rms_norm_eps or config.rms_norm_eps,
            )
            self.audio_token_id = audio.audio_token_id or config.audio_token_id

    def _embed_placeholder_ids(
        self,
        op: OpBuilder,
        hidden: ir.Value,
        input_ids: ir.Value,
        embedder: Gemma3nMultimodalEmbedder,
        lower: int,
        upper: int | None,
    ) -> ir.Value:
        """Overwrite ids in ``[lower, upper)`` with their hard-path embedding.

        Out-of-range positions are rewritten to the *last* id of the
        modality's table before the lookup, exactly as HF does: ONNX
        ``Gather`` does not bounds-check, so feeding an ordinary text id
        straight into a 128-row table would read out of bounds.
        """
        in_range = op.GreaterOrEqual(input_ids, op.CastLike(lower, input_ids))
        if upper is not None:
            in_range = op.And(in_range, op.Less(input_ids, op.CastLike(upper, input_ids)))
        dummy_id = op.CastLike(embedder.vocab_offset + embedder.vocab_size - 1, input_ids)
        safe_ids = op.Where(in_range, input_ids, dummy_id)
        embeds = op.CastLike(embedder(op, input_ids=safe_ids), hidden)
        return op.Where(op.Unsqueeze(in_range, [-1]), embeds, hidden)

    def _scatter_features(
        self,
        op: OpBuilder,
        hidden: ir.Value,
        input_ids: ir.Value,
        token_id: int,
        features: ir.Value,
    ) -> ir.Value:
        """Scatter ``features`` rows into ``hidden`` at ``token_id`` positions.

        A dummy zero row is appended before the Gather so that a text-only or
        decode step, which passes a ``[0, hidden]`` feature tensor, cannot
        fault on an empty-tensor Gather even though ``Where`` discards the
        result.  ``Constant`` + ``Unsqueeze`` rather than
        ``Expand``/``ConstantOfShape``: a dynamic shape input there blocks
        ONNX shape inference.
        """
        mask = op.Equal(input_ids, op.CastLike(token_id, input_ids))
        # CumSum -> sub-1 -> clip gives each matching position its 0-based row
        # index into ``features``.
        mask_int = op.Cast(mask, to=ir.DataType.INT64)
        cumsum = op.CumSum(mask_int, op.Constant(value_int=1))
        indices = op.Clip(op.Sub(cumsum, op.Constant(value_int=1)), op.Constant(value_int=0))
        dummy_row = op.Unsqueeze(
            op.CastLike(op.Constant(value_floats=[0.0] * self.config.hidden_size), features),
            [0],
        )
        features_safe = op.Concat(features, dummy_row, axis=0)
        gathered = op.CastLike(op.Gather(features_safe, indices, axis=0), hidden)
        return op.Where(op.Unsqueeze(mask, [-1]), gathered, hidden)

    def forward(
        self,
        op: OpBuilder,
        input_ids: ir.Value,
        image_features: ir.Value,
        audio_features: ir.Value | None = None,
    ) -> dict[str, ir.Value]:
        hidden = self.embed_tokens(op, input_ids)

        # Hard placeholder embeddings.  The vision id range ends where the
        # audio range begins (HF reads ``embed_audio.vocab_offset``); with no
        # audio tower it ends after the vision table.
        vision_upper = self._audio_vocab_offset
        if vision_upper is None:
            vision_upper = self._vision_vocab_offset + self._vision_vocab_size
        hidden = self._embed_placeholder_ids(
            op,
            hidden,
            input_ids,
            self.embed_vision,
            self._vision_vocab_offset,
            vision_upper,
        )
        if self._audio_vocab_offset is not None:
            hidden = self._embed_placeholder_ids(
                op, hidden, input_ids, self.embed_audio, self._audio_vocab_offset, None
            )

        # Soft encoder features replace their placeholder positions.
        if self.image_token_id is None:
            raise ValueError(
                "Gemma3n embedding model needs image_token_id to place image "
                "features; got None."
            )
        hidden = self._scatter_features(
            op, hidden, input_ids, self.image_token_id, image_features
        )
        if audio_features is not None:
            if self.audio_token_id is None:
                raise ValueError(
                    "Gemma3n embedding model received audio_features but no "
                    "audio_token_id is configured."
                )
            hidden = self._scatter_features(
                op, hidden, input_ids, self.audio_token_id, audio_features
            )

        return {
            "inputs_embeds": hidden,
            "per_layer_inputs": self._per_layer_inputs(op, input_ids, hidden),
        }

    def _per_layer_inputs(
        self, op: OpBuilder, input_ids: ir.Value, hidden: ir.Value
    ) -> ir.Value:
        """Build the combined ``[B, S, L*D]`` per-layer input tensor.

        Matches HF ``get_per_layer_inputs`` + ``project_per_layer_inputs``.
        The projection reads the *fused* embeddings, so the multimodal
        features feed the per-layer gates exactly as they do in HF, where
        ``project_per_layer_inputs`` runs inside the language model on the
        already-merged ``inputs_embeds``.
        """
        proj = self.per_layer_model_projection(op, hidden)
        proj = op.Mul(proj, float(self.config.hidden_size**-0.5))
        proj = op.Reshape(
            proj, op.Constant(value_ints=[0, 0, self._num_layers, self._per_layer_dim])
        )
        proj = self.per_layer_projection_norm(op, proj)

        # HF masks every id outside the per-layer vocab to 0 before the
        # lookup, which covers both modalities' reserved ranges (they sit
        # above ``vocab_size_per_layer_input``) as well as any negative id.
        in_vocab = op.And(
            op.GreaterOrEqual(input_ids, op.CastLike(0, input_ids)),
            op.Less(input_ids, op.CastLike(self.config.vocab_size_per_layer_input, input_ids)),
        )
        masked_ids = op.Where(in_vocab, input_ids, op.CastLike(0, input_ids))
        token_embed = op.Reshape(
            self.embed_tokens_per_layer(op, masked_ids),
            op.Constant(value_ints=[0, 0, self._num_layers, self._per_layer_dim]),
        )

        combined = op.Mul(op.Add(proj, token_embed), float(1.0 / math.sqrt(2.0)))
        # Flatten L*D into a single graph output; the decoder reshapes back.
        return op.Reshape(
            combined,
            op.Constant(value_ints=[0, 0, self._num_layers * self._per_layer_dim]),
        )


class Gemma3nMultiModalModel(nn.Module):
    """Gemma 3n image + audio + text model (3- or 4-model split).

    Always produced:

    - ``decoder`` — text decoder on ``inputs_embeds`` + ``per_layer_inputs``
    - ``vision_encoder`` — MobileNet-V5 tower + ``embed_vision`` soft path
    - ``embedding`` — token embedding, hard placeholder embedding, feature
      fusion, and the per-layer input tables

    Added when ``config.audio is not None``:

    - ``audio_encoder`` — USM Conformer tower + ``embed_audio``

    Registered as ``gemma3n``; the text-only ``gemma3n_text`` key keeps
    mapping to :class:`Gemma3nCausalLMModel`.
    """

    default_task: str = "gemma3n"
    category: str = "Multimodal"
    config_class: type = Gemma3nMultiModalConfig

    def __init__(self, config: Gemma3nMultiModalConfig):
        super().__init__()
        self.config = config
        self.decoder = _Gemma3nDecoderModel(config)
        self.vision_encoder = _Gemma3nVisionEncoderModel(config)
        self.embedding = _Gemma3nEmbeddingModel(config)
        self.audio_encoder: _Gemma3nAudioEncoderModel | None = (
            _Gemma3nAudioEncoderModel(config) if config.audio is not None else None
        )

    def forward(self, op: OpBuilder, **kwargs):
        raise NotImplementedError(
            "Gemma3nMultiModalModel is a multi-model split; Gemma3nTask builds each "
            "sub-module (decoder, vision_encoder, embedding, and optionally "
            "audio_encoder) separately."
        )

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Rename HuggingFace weight keys to ONNX initializer names.

        HF multimodal checkpoints prefix every key with ``model.``; after
        stripping it:

        - ``language_model.lm_head.*`` -> ``decoder.lm_head.*``
        - ``language_model.{embed_tokens_per_layer,per_layer_model_projection,
          per_layer_projection_norm}.*`` -> ``embedding.*``
        - ``language_model.*`` -> ``decoder.model.*``, with ``embed_tokens.*``
          *also* copied to ``embedding.embed_tokens.*``
        - ``vision_tower.timm_model.*`` -> ``vision_encoder.encoder.*``
        - ``audio_tower.*`` -> ``audio_encoder.encoder.*``

        Both tower maps are pure prefix swaps: unlike Gemma 4, every learned
        tensor of :class:`~mobius.components.MobileNetV5Encoder` and
        :class:`~mobius.components.Gemma3nAudioEncoder` already carries its
        checkpoint name verbatim.

        ``embed_vision.*`` goes to *both* ``vision_encoder.embed_vision.*``
        and ``embedding.embed_vision.*`` (likewise ``embed_audio.*``): each
        embedder is used through its soft path in its tower's graph and its
        hard path in the embedding graph, so both graphs need all four
        tensors.  ``embedding.weight``, the hard lookup table, is kept —
        Gemma 3n uses the hard path, unlike Gemma 4.

        Unused keys are dropped rather than passed through: the KV-shared
        layers' K/V projections (shipped by the checkpoint but built by no
        layer) and, for an audio-less config, the whole audio tower.
        """
        state_dict = {
            (key[len("model.") :] if key.startswith("model.") else key): value
            for key, value in state_dict.items()
        }

        # A tied checkpoint ships no lm_head; synthesize it from the token
        # embedding so the decoder's head initializer gets data.
        if self.config.tie_word_embeddings:
            embed_key = "language_model.embed_tokens.weight"
            head_key = "language_model.lm_head.weight"
            if head_key not in state_dict and embed_key in state_dict:
                state_dict[head_key] = state_dict[embed_key]

        per_layer_prefixes = (
            "embed_tokens_per_layer.",
            "per_layer_model_projection.",
            "per_layer_projection_norm.",
        )
        renamed: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("language_model."):
                suffix = key[len("language_model.") :]
                if suffix.startswith("lm_head"):
                    # lm_head lives directly under decoder (not decoder.model)
                    renamed["decoder." + suffix] = value
                elif suffix.startswith(per_layer_prefixes):
                    # Per-layer embedding tables → embedding sub-model
                    renamed["embedding." + suffix] = value
                else:
                    renamed["decoder.model." + suffix] = value
                    if suffix.startswith("embed_tokens."):
                        # Token embedding is shared with the embedding sub-model.
                        renamed["embedding." + suffix] = value

            elif key.startswith("vision_tower.timm_model."):
                # HF wraps the timm encoder in a TimmWrapperModel, adding one
                # name level our encoder does not have.
                suffix = key[len("vision_tower.timm_model.") :]
                renamed["vision_encoder.encoder." + suffix] = value

            elif key.startswith("embed_vision."):
                suffix = key[len("embed_vision.") :]
                renamed["vision_encoder.embed_vision." + suffix] = value
                renamed["embedding.embed_vision." + suffix] = value

            elif key.startswith("audio_tower."):
                if self.audio_encoder is not None:
                    renamed["audio_encoder.encoder." + key[len("audio_tower.") :]] = value

            elif key.startswith("embed_audio."):
                if self.audio_encoder is not None:
                    suffix = key[len("embed_audio.") :]
                    renamed["audio_encoder.embed_audio." + suffix] = value
                    renamed["embedding.embed_audio." + suffix] = value

            else:
                renamed[key] = value

        renamed = _strip_altup_prefix(renamed)
        return _drop_kv_shared_layer_weights(
            renamed, self.decoder.model, prefix="decoder.model.layers."
        )
