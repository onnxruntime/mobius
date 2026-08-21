# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Wav2Vec2 encoder-only audio model.

Supports: wav2vec2, hubert, wavlm (all share the same architecture).

Architecture (HF ``Wav2Vec2Model``):

1. ``feature_extractor`` — a stack of strided Conv1d layers that turn a raw
   16 kHz waveform into frames.  The stride product fixes the downsampling
   ratio (320 for the ``*-base-*`` checkpoints, i.e. one frame per 20 ms).
2. ``feature_projection`` — LayerNorm over the conv channel dim + Linear into
   ``hidden_size``.
3. ``encoder`` — a relative positional conv embedding added to the frames,
   followed by transformer layers.

Two encoder variants exist and are selected by ``config.do_stable_layer_norm``:

- ``False`` (e.g. ``facebook/wav2vec2-base-960h``): ``layer_norm`` runs *before*
  the transformer stack and each layer is **post-norm**.
- ``True`` (e.g. ``facebook/wav2vec2-large-960h-lv60-self``, MMS): each layer is
  **pre-norm** and ``layer_norm`` runs *after* the stack.

Getting this ordering wrong silently produces plausible-looking but incorrect
logits, so the variant is chosen from the checkpoint rather than assumed.

Inputs:
    ``input_values``   — (batch, num_samples) float waveform
    ``attention_mask`` — (batch, num_samples) int64 padding mask, optional

Output:
    ``last_hidden_state`` — (batch, num_frames, hidden_size)

HF weight naming is mirrored one-to-one except for the FFN (HF
``intermediate_dense``/``output_dense`` vs the shared ``FCMLP``
``up_proj``/``down_proj``) and the weight-normalized positional conv, which is
materialized in :meth:`Wav2Vec2Model.preprocess_weights`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from onnxscript import OpBuilder, nn

from mobius._configs import ArchitectureConfig, MMSConfig
from mobius.components import FCMLP
from mobius.components._common import GroupNorm, LayerNorm, Linear
from mobius.components._whisper import Conv1d

if TYPE_CHECKING:
    import onnx_ir as ir


def _conv_geometry(config: ArchitectureConfig) -> tuple[tuple[int, ...], ...]:
    """Return ``(conv_dim, conv_kernel, conv_stride)`` for *config*.

    Falls back to the wav2vec2-base geometry when a caller supplies a bare
    :class:`ArchitectureConfig` (the audio-feature-extraction path).
    """
    conv_dim = tuple(getattr(config, "conv_dim", None) or (512,) * 7)
    conv_kernel = tuple(getattr(config, "conv_kernel", None) or (10, 3, 3, 3, 3, 2, 2))
    conv_stride = tuple(getattr(config, "conv_stride", None) or (5, 2, 2, 2, 2, 2, 2))
    return conv_dim, conv_kernel, conv_stride


class _ConvLayerBlock(nn.Module):
    """One feature-encoder block: strided Conv1d → optional norm → GELU.

    ``norm`` mirrors HF's three conv-layer classes:
    ``"group"`` (``Wav2Vec2GroupNormConvLayer``), ``"layer"``
    (``Wav2Vec2LayerNormConvLayer``) and ``"none"``
    (``Wav2Vec2NoLayerNormConvLayer``).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool,
        norm: str = "none",
    ):
        super().__init__()
        # Named ``conv`` with ``.weight``/``.bias`` so HF names map through
        # unchanged.  ``bias=False`` must not materialize a bias parameter at
        # all: an unset initializer makes the exported graph unloadable.
        self.conv = Conv1d(in_channels, out_channels, kernel_size, stride=stride, bias=bias)
        self._norm = norm
        if norm == "group":
            # HF uses nn.GroupNorm(num_groups=C, num_channels=C) — one group per
            # channel — with torch's default eps of 1e-5.
            self.layer_norm = GroupNorm(out_channels, out_channels, eps=1e-5)
        elif norm == "layer":
            self.layer_norm = LayerNorm(out_channels, eps=1e-5)

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        # (batch, in_channels, time) -> (batch, out_channels, time')
        hidden_states = self.conv(op, hidden_states)
        if self._norm == "group":
            hidden_states = self.layer_norm(op, hidden_states)
        elif self._norm == "layer":
            # HF normalizes over the channel axis, which is axis 1 here, so the
            # tensor is moved to channels-last for the LayerNormalization op.
            hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])
            hidden_states = self.layer_norm(op, hidden_states)
            hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])
        return op.Gelu(hidden_states)


class _Conv1dFeatureExtractor(nn.Module):
    """CNN feature encoder: raw waveform → strided frame features.

    Matches HF ``Wav2Vec2FeatureEncoder``.  With ``feat_extract_norm="group"``
    only the first layer is normalized; with ``"layer"`` every layer is.
    """

    def __init__(
        self,
        conv_dim: tuple[int, ...],
        conv_kernel: tuple[int, ...],
        conv_stride: tuple[int, ...],
        conv_bias: bool,
        feat_extract_norm: str,
    ):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        in_channels = 1
        for i, out_channels in enumerate(conv_dim):
            if feat_extract_norm == "group":
                norm = "group" if i == 0 else "none"
            else:
                norm = "layer"
            self.conv_layers.append(
                _ConvLayerBlock(
                    in_channels,
                    out_channels,
                    conv_kernel[i],
                    conv_stride[i],
                    conv_bias,
                    norm,
                )
            )
            in_channels = out_channels

    def forward(self, op: OpBuilder, input_values: ir.Value) -> ir.Value:
        # (batch, time) -> (batch, 1, time)
        hidden_states = op.Unsqueeze(input_values, [1])
        for layer in self.conv_layers:
            hidden_states = layer(op, hidden_states)
        # Channels-first (batch, channels, frames); the caller transposes.
        return hidden_states


class _PositionalConvEmbedding(nn.Module):
    """Relative position embedding via a grouped Conv1d + GELU.

    Matches HF ``Wav2Vec2PositionalConvEmbedding``.  The convolution is
    weight-normalized in the checkpoint (``weight_g``/``weight_v``); the product
    is materialized during weight preprocessing so the graph holds one dense
    kernel.

    An even ``kernel_size`` with ``padding = kernel_size // 2`` emits one frame
    too many, which HF trims via ``Wav2Vec2SamePadLayer``.
    """

    def __init__(self, hidden_size: int, kernel_size: int, groups: int):
        super().__init__()
        self.conv = Conv1d(
            hidden_size,
            hidden_size,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=groups,
        )
        self._num_pad_remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        # (batch, frames, hidden) -> (batch, hidden, frames)
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])
        hidden_states = self.conv(op, hidden_states)
        if self._num_pad_remove:
            # Drop the trailing frame produced by the even-kernel "same" padding.
            hidden_states = op.Slice(hidden_states, [0], [-1], [2])
        hidden_states = op.Gelu(hidden_states)
        return op.Transpose(hidden_states, perm=[0, 2, 1])


class _FeatureProjection(nn.Module):
    """Projects CNN features to hidden size with LayerNorm."""

    def __init__(self, conv_dim: int, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.layer_norm = LayerNorm(conv_dim, eps=eps)
        self.projection = Linear(conv_dim, hidden_size)

    def forward(self, op: OpBuilder, hidden_states: ir.Value):
        hidden_states = self.layer_norm(op, hidden_states)
        hidden_states = self.projection(op, hidden_states)
        return hidden_states


class _Wav2Vec2Attention(nn.Module):
    """Bidirectional self-attention for the Wav2Vec2 encoder."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = Linear(hidden_size, hidden_size, bias=True)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value | None = None,
    ):
        q = self.q_proj(op, hidden_states)
        k = self.k_proj(op, hidden_states)
        v = self.v_proj(op, hidden_states)
        # ``attention_mask`` is a BOOL keep-mask broadcastable to
        # (batch, heads, q_frames, kv_frames); padded frames must not be
        # attended to or a padded batch would not match an unpadded run.
        attn_out = op.Attention(
            q,
            k,
            v,
            attention_mask,
            q_num_heads=self.num_heads,
            kv_num_heads=self.num_heads,
            is_causal=0,
            scale=float(self.head_dim**-0.5),
        )
        return self.out_proj(op, attn_out)


class _Wav2Vec2EncoderLayer(nn.Module):
    """Post-norm transformer layer (HF ``Wav2Vec2EncoderLayer``).

    Normalization runs *after* each residual add, which is the layout used by
    ``do_stable_layer_norm=False`` checkpoints.
    """

    def __init__(
        self, hidden_size: int, intermediate_size: int, num_heads: int, eps: float = 1e-5
    ):
        super().__init__()
        self.layer_norm = LayerNorm(hidden_size, eps=eps)
        head_dim = hidden_size // num_heads
        self.attention = _Wav2Vec2Attention(hidden_size, num_heads, head_dim)
        self.feed_forward = FCMLP(hidden_size, intermediate_size, activation="gelu", bias=True)
        self.final_layer_norm = LayerNorm(hidden_size, eps=eps)

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value | None = None,
    ):
        residual = hidden_states
        hidden_states = self.attention(op, hidden_states, attention_mask)
        hidden_states = op.Add(residual, hidden_states)
        hidden_states = self.layer_norm(op, hidden_states)

        hidden_states = op.Add(hidden_states, self.feed_forward(op, hidden_states))
        return self.final_layer_norm(op, hidden_states)


class _Wav2Vec2EncoderLayerStableLayerNorm(_Wav2Vec2EncoderLayer):
    """Pre-norm transformer layer (HF ``Wav2Vec2EncoderLayerStableLayerNorm``).

    Same parameters as the post-norm layer; only the application order differs,
    so the weight names stay identical.
    """

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value | None = None,
    ):
        residual = hidden_states
        hidden_states = self.layer_norm(op, hidden_states)
        hidden_states = self.attention(op, hidden_states, attention_mask)
        hidden_states = op.Add(residual, hidden_states)

        normed = self.final_layer_norm(op, hidden_states)
        return op.Add(hidden_states, self.feed_forward(op, normed))


class _Wav2Vec2Encoder(nn.Module):
    """Post-norm encoder: pos-conv → layer_norm → layers (HF ``Wav2Vec2Encoder``)."""

    layer_class: type[_Wav2Vec2EncoderLayer] = _Wav2Vec2EncoderLayer

    def __init__(self, config: ArchitectureConfig, eps: float = 1e-5):
        super().__init__()
        self.pos_conv_embed = _PositionalConvEmbedding(
            config.hidden_size,
            getattr(config, "num_conv_pos_embeddings", 128),
            getattr(config, "num_conv_pos_embedding_groups", 16),
        )
        self.layer_norm = LayerNorm(config.hidden_size, eps=eps)
        self.layers = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.layers.append(
                self.layer_class(
                    config.hidden_size,
                    config.intermediate_size,
                    config.num_attention_heads,
                    eps=eps,
                )
            )

    def _add_positions(self, op: OpBuilder, hidden_states: ir.Value) -> ir.Value:
        return op.Add(hidden_states, self.pos_conv_embed(op, hidden_states))

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value | None = None,
    ):
        hidden_states = self._add_positions(op, hidden_states)
        hidden_states = self.layer_norm(op, hidden_states)
        for layer in self.layers:
            hidden_states = layer(op, hidden_states, attention_mask)
        return hidden_states


class _Wav2Vec2EncoderStableLayerNorm(_Wav2Vec2Encoder):
    """Pre-norm encoder: pos-conv → layers → layer_norm.

    Matches HF ``Wav2Vec2EncoderStableLayerNorm``; the final normalization moves
    to the end of the stack.
    """

    layer_class: type[_Wav2Vec2EncoderLayer] = _Wav2Vec2EncoderLayerStableLayerNorm

    def forward(
        self,
        op: OpBuilder,
        hidden_states: ir.Value,
        attention_mask: ir.Value | None = None,
    ):
        hidden_states = self._add_positions(op, hidden_states)
        for layer in self.layers:
            hidden_states = layer(op, hidden_states, attention_mask)
        return self.layer_norm(op, hidden_states)


class Wav2Vec2Model(nn.Module):
    """Wav2Vec2 encoder-only audio model.

    Architecture: CNN feature extractor → feature projection → transformer encoder.
    Used for ASR (CTC), audio feature extraction, and audio classification.
    """

    default_task: str = "audio-feature-extraction"
    category: str = "Audio"
    # Declared on the module so config resolution picks the wav2vec2-shaped
    # config even when the registry entry is reached by architecture re-routing
    # rather than by ``model_type``.  Without it the convolution geometry and
    # normalization placement silently fall back to wav2vec2-base defaults.
    config_class = MMSConfig

    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config

        conv_dim, conv_kernel, conv_stride = _conv_geometry(config)
        self._conv_kernel = conv_kernel
        self._conv_stride = conv_stride
        self.feature_extractor = _Conv1dFeatureExtractor(
            conv_dim,
            conv_kernel,
            conv_stride,
            bool(getattr(config, "conv_bias", False)),
            getattr(config, "feat_extract_norm", None) or "group",
        )

        eps = getattr(config, "layer_norm_eps", None) or 1e-5
        self.feature_projection = _FeatureProjection(conv_dim[-1], config.hidden_size, eps=eps)

        encoder_class = (
            _Wav2Vec2EncoderStableLayerNorm
            if getattr(config, "do_stable_layer_norm", False)
            else _Wav2Vec2Encoder
        )
        self.encoder = encoder_class(config, eps=eps)

    @property
    def batch_padding_sensitive(self) -> bool:
        """Whether a row's outputs depend on the padded width of its batch.

        ``feat_extract_norm="group"`` puts a ``GroupNorm`` with one group per
        channel on the first convolution, which reduces over the *time* axis.
        Its statistics therefore include whatever padding was appended to reach
        the batch width, so co-batching rows of unequal length perturbs every
        frame of the shorter rows.  ``"layer"`` reduces over channels instead
        and leaves rows independent.
        """
        return getattr(self.config, "feat_extract_norm", None) == "group"

    def frame_lengths(self, op: OpBuilder, attention_mask: ir.Value) -> ir.Value:
        """Return the per-row valid frame count for a sample-level mask.

        Mirrors ``_get_feat_extract_output_lengths``: every conv contributes
        ``floor((L - kernel) / stride) + 1``.  The result lets a caller segment
        a padded batch back into per-row outputs.

        Args:
            attention_mask: (batch, num_samples) INT64, 1 for valid samples.

        Returns:
            (batch,) INT64 frame counts.
        """
        lengths = op.ReduceSum(
            op.Cast(attention_mask, to=7),  # INT64
            [1],
            keepdims=0,
        )
        for kernel, stride in zip(self._conv_kernel, self._conv_stride):
            # Integer division truncates toward zero in ONNX; lengths are
            # non-negative here so that matches Python floor division.
            lengths = op.Add(
                op.Div(
                    op.Sub(lengths, op.Constant(value_int=int(kernel))),
                    op.Constant(value_int=int(stride)),
                ),
                op.Constant(value_int=1),
            )
        return lengths

    def frame_mask(
        self, op: OpBuilder, attention_mask: ir.Value, hidden_states: ir.Value
    ) -> ir.Value:
        """Build a (batch, num_frames) BOOL keep-mask from a sample-level mask."""
        lengths = self.frame_lengths(op, attention_mask)  # (batch,)
        num_frames = op.Shape(hidden_states, start=1, end=2)  # (1,)
        positions = op.Range(
            op.Constant(value_int=0),
            op.Squeeze(num_frames, [0]),
            op.Constant(value_int=1),
        )  # (num_frames,)
        # (1, num_frames) < (batch, 1) -> (batch, num_frames)
        return op.Less(op.Unsqueeze(positions, [0]), op.Unsqueeze(lengths, [1]))

    def forward(
        self,
        op: OpBuilder,
        input_values: ir.Value,
        attention_mask: ir.Value | None = None,
    ):
        """Forward pass: raw audio → hidden states.

        Args:
            op: ONNX op builder.
            input_values: (batch, num_samples) raw waveform.
            attention_mask: (batch, num_samples) INT64 sample-level padding mask.

        Returns:
            last_hidden_state: (batch, num_frames, hidden_size)
        """
        # (batch, samples) -> (batch, channels, frames) -> (batch, frames, channels)
        hidden_states = self.feature_extractor(op, input_values)
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])
        hidden_states = self.feature_projection(op, hidden_states)

        attention_bias = None
        if attention_mask is not None:
            keep = self.frame_mask(op, attention_mask, hidden_states)  # (batch, frames)
            # Zero the padded frames exactly as HF does before the pos-conv, so
            # the convolution never mixes real frames with padding residue.
            hidden_states = op.Where(
                op.Unsqueeze(keep, [-1]),
                hidden_states,
                op.CastLike(op.Constant(value_float=0.0), hidden_states),
            )
            # The ONNX Attention op does not broadcast the query axis of
            # ``attn_mask``, so the (batch, frames) keep-mask is expanded to
            # (batch, 1, q_frames, kv_frames).  Masking only the key axis
            # matches HF ``create_bidirectional_mask``: a padded *query* row
            # still attends to the valid keys, so no row is fully masked and
            # softmax never sees an all -inf row.
            num_frames = op.Shape(hidden_states, start=1, end=2)  # (1,)
            batch = op.Shape(hidden_states, start=0, end=1)  # (1,)
            mask_shape = op.Concat(
                batch,
                op.Constant(value_ints=[1]),
                num_frames,
                num_frames,
                axis=0,
            )
            attention_bias = op.Expand(op.Unsqueeze(keep, [1, 2]), mask_shape)

        return self.encoder(op, hidden_states, attention_bias)

    @staticmethod
    def _materialize_weight_norm(
        state_dict: dict[str, torch.Tensor], prefix: str
    ) -> torch.Tensor | None:
        """Recombine a weight-normalized conv kernel into a dense tensor.

        ``torch.nn.utils.weight_norm(conv, name="weight", dim=2)`` stores a
        magnitude ``g`` of shape ``(1, 1, K)`` and a direction ``v`` of shape
        ``(C_out, C_in/groups, K)``.  The effective kernel is
        ``g * v / ||v||`` with the norm taken over every axis except ``dim``.

        Checkpoints saved by newer torch use the ``parametrizations`` names, so
        both spellings are accepted.
        """
        pairs = (
            (f"{prefix}.weight_g", f"{prefix}.weight_v"),
            (
                f"{prefix}.parametrizations.weight.original0",
                f"{prefix}.parametrizations.weight.original1",
            ),
        )
        for g_key, v_key in pairs:
            if g_key in state_dict and v_key in state_dict:
                g = state_dict[g_key].float()
                v = state_dict[v_key].float()
                norm = v.pow(2).sum(dim=(0, 1), keepdim=True).sqrt()
                return g * v / norm
        return None

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HF Wav2Vec2 weight names to our names.

        Only three families of names actually differ:

        - the ``wav2vec2.``/``hubert.``/``wavlm.`` prefix is stripped,
        - HF's ``intermediate_dense``/``output_dense`` become ``FCMLP``'s
          ``up_proj``/``down_proj``,
        - the weight-normalized positional conv is collapsed to a dense kernel.

        ``masked_spec_embed`` is a SpecAugment training parameter and has no
        inference effect, so it is dropped rather than exported.
        """
        stripped: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            new_key = key
            for prefix in ("wav2vec2.", "hubert.", "wavlm."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    break
            stripped[new_key] = value

        pos_conv = "encoder.pos_conv_embed.conv"
        dense_pos_conv = self._materialize_weight_norm(stripped, pos_conv)

        new_state_dict: dict[str, torch.Tensor] = {}
        for key, value in stripped.items():
            if key == "masked_spec_embed":
                continue
            if key.startswith((f"{pos_conv}.weight_g", f"{pos_conv}.weight_v")):
                continue
            if key.startswith(f"{pos_conv}.parametrizations."):
                continue
            new_key = key.replace(".intermediate_dense.", ".up_proj.").replace(
                ".output_dense.", ".down_proj."
            )
            new_state_dict[new_key] = value

        if dense_pos_conv is not None:
            new_state_dict[f"{pos_conv}.weight"] = dense_pos_conv
        return new_state_dict
