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

from mobius._configs import Gemma3nConfig
from mobius.components import (
    MLP,
    Attention,
    Embedding,
    Linear,
    RMSNorm,
    create_attention_bias,
    initialize_rope,
)
from mobius.components._activations import get_activation
from mobius.components._attention import _apply_attention, apply_rotary_pos_emb
from mobius.models.base import CausalLMModel


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
    Q and K use standard OffsetRMSNorm from the parent; V is divided by its
    per-head RMS without any learnable scale parameter.

    Layers at or after ``first_kv_shared_layer_idx`` are *KV-shared*: they
    borrow the already-RoPE'd K,V of the last non-shared layer of the same
    attention type instead of running their own ``k_proj``/``v_proj``.  Such
    layers own no KV cache entry and hold no K/V weights — matching HF
    ``Gemma3nTextAttention``, which does not even construct them.

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
    ):
        if inputs_embeds is not None:
            hidden_states_0 = inputs_embeds
        else:
            hidden_states_0 = self.embed_tokens(op, input_ids)

        # Compute per-layer inputs
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
        """Compute per-layer input embeddings from input_ids and model projection."""
        # Per-layer token embeddings
        if input_ids is not None:
            per_layer_token_embed = self.embed_tokens_per_layer(op, input_ids)
        else:
            per_layer_token_embed = None

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

        if per_layer_token_embed is not None:
            per_layer_token_embed = op.Reshape(
                per_layer_token_embed,
                op.Constant(
                    value_ints=[0, 0, self.num_hidden_layers, self.hidden_size_per_layer_input]
                ),
            )
            per_layer_inputs = op.Add(per_layer_proj, per_layer_token_embed)
            per_layer_inputs = op.Mul(per_layer_inputs, self.per_layer_input_scale)
        else:
            per_layer_inputs = per_layer_proj

        return per_layer_inputs

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

    def kv_cache_layer_count(self) -> int:
        """Number of layers owning a KV cache entry (excludes KV-shared layers)."""
        return self.model.num_kv_layers

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

        # AltUp submodules (correction_coefs, prediction_coefs, modality_router,
        # router_norm, correct_output_scale) are called via plain methods rather than
        # __call__, so onnxscript registers them on the parent DecoderLayer without the
        # ".altup." prefix.  Strip ".altup." from all keys that contain it.
        for key in list(state_dict.keys()):
            if ".altup." in key:
                new_key = key.replace(".altup.", ".")
                state_dict[new_key] = state_dict.pop(key)

        state_dict = _drop_kv_shared_layer_weights(state_dict, self.model)
        return super().preprocess_weights(state_dict)
