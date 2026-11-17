# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Wav2Vec2 CTC model — used for MMS (Massively Multilingual Speech) and CTC-based ASR.

Architecture: CNN feature extractor → feature projection → transformer encoder
              → optional language adapter → CTC lm_head

Supports:
- ``mms`` model_type (facebook/mms-1b-all, facebook/mms-1b-fl102)
- Any ``Wav2Vec2ForCTC`` checkpoint
- Per-language adapter weights (Wav2Vec2Adapter with strided Conv1d + GLU)

HF class: ``Wav2Vec2ForCTC``

The three main sub-graphs share weights with ``Wav2Vec2Model``:
  feature_extractor → feature_projection → encoder → [adapter] → lm_head

Adapter architecture (``add_adapter=True``):
  optional Linear proj + LayerNorm → N x (Conv1d + GLU)

Weight name alignment:
  After stripping ``wav2vec2.`` prefix, HF names map to ours with minor renames
  for the CNN feature extractor (bare ``nn.Parameter`` vs ``nn.Conv1d`` sub-module).
"""

from __future__ import annotations

import re

import onnx_ir as ir
import torch
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig, MMSConfig
from mobius.components._common import LayerNorm, Linear
from mobius.models.wav2vec2 import Wav2Vec2Model


class _AdapterLayer(nn.Module):
    """One MMS adapter layer: strided Conv1d(hidden → 2x hidden) → GLU.

    The Conv1d doubles the channels so GLU can split them in half,
    producing output of the same size as input (per-channel gating).

    Matches HF ``Wav2Vec2AdapterLayer``.

    Uses bare ``nn.Parameter`` to match how the wav2vec2 feature extractor
    handles conv weights.  ``preprocess_weights`` renames from HF's
    ``*.conv.weight / *.conv.bias`` to our ``*.conv / *.conv_bias``.
    """

    def __init__(self, hidden_size: int, kernel_size: int = 3, stride: int = 2):
        super().__init__()
        self._stride = stride
        self._kernel_size = kernel_size
        # Conv1d: hidden → 2x hidden, stride=stride, pad=1
        # Bare parameters (no sub-module) to match onnxscript name resolution.
        self.conv = nn.Parameter([2 * hidden_size, hidden_size, kernel_size])
        self.conv_bias = nn.Parameter([2 * hidden_size])

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value) -> ir.Value:
        """Apply strided conv then GLU activation.

        Args:
            hidden_states: (batch, hidden_size, seq_len) — channels-first

        Returns:
            (batch, hidden_size, seq_len // stride)
        """
        # Symmetric padding = kernel_size // 2 on each side, matching HF's
        # nn.Conv1d(padding=kernel_size // 2) used by Wav2Vec2AdapterLayer.
        # Hard-coding pad=1 was only correct for kernel_size=3.
        pad = self._kernel_size // 2
        hidden_states = op.Conv(
            hidden_states,
            self.conv,
            self.conv_bias,
            kernel_shape=[self._kernel_size],
            strides=[self._stride],
            pads=[pad, pad],
            group=1,
        )
        # GLU: split along channel dim, gate with sigmoid
        # hidden_states: (B, 2*H, T) → a: (B, H, T), b: (B, H, T)
        a, b = op.Split(hidden_states, axis=1, num_outputs=2, _outputs=2)
        return op.Mul(a, op.Sigmoid(b))


class _Adapter(nn.Module):
    """MMS language adapter: optional projection + N strided conv+GLU layers.

    Matches HF ``Wav2Vec2Adapter``. Applied after the main encoder to adapt
    representations to a specific language or domain.

    Weight names:
      proj.weight, proj.bias  (only when ``output_hidden_size != hidden_size``)
      proj_layer_norm.weight, proj_layer_norm.bias
      layers.N.conv, layers.N.conv_bias  (bare params, renamed from HF's .conv.weight/.conv.bias)
    """

    def __init__(
        self,
        hidden_size: int,
        output_hidden_size: int,
        num_adapter_layers: int,
        adapter_kernel_size: int = 3,
        adapter_stride: int = 2,
    ):
        super().__init__()
        self._has_proj = output_hidden_size != hidden_size

        if self._has_proj:
            self.proj = Linear(hidden_size, output_hidden_size, bias=True)
            self.proj_layer_norm = LayerNorm(output_hidden_size, eps=1e-5)

        # N stacked adapter layers operating on output_hidden_size channels
        self.layers = nn.ModuleList(
            [
                _AdapterLayer(output_hidden_size, adapter_kernel_size, adapter_stride)
                for _ in range(num_adapter_layers)
            ]
        )

    def forward(self, op: builder.OpBuilder, hidden_states: ir.Value) -> ir.Value:
        """Apply adapter to encoder output.

        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            (batch, adapted_seq_len, output_hidden_size)
        """
        if self._has_proj:
            # Down-project and normalize: (B, T, H) → (B, T, output_H)
            hidden_states = self.proj(op, hidden_states)
            hidden_states = self.proj_layer_norm(op, hidden_states)

        # Adapter layers operate on (B, C, T) channels-first
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])  # (B, H, T)
        for layer in self.layers:
            hidden_states = layer(op, hidden_states)
        hidden_states = op.Transpose(hidden_states, perm=[0, 2, 1])  # (B, T, H)
        return hidden_states


class Wav2Vec2ForCTCModel(Wav2Vec2Model):
    """Wav2Vec2 CTC model for ASR — encoder + optional adapter + CTC head.

    Extends ``Wav2Vec2Model`` (feature extractor + transformer encoder) with:
    - Optional language-specific adapter (``Wav2Vec2Adapter``)
    - CTC projection head (Linear → vocab logits)

    Used for MMS (facebook/mms-1b-all) and any ``Wav2Vec2ForCTC`` checkpoint.

    Input:  ``input_values`` — (batch, num_samples) raw audio at 16 kHz
    Output: ``logits`` — (batch, num_frames, vocab_size) CTC logit scores

    HuggingFace class: ``Wav2Vec2ForCTC``

    Language switching (MMS):
        Load the base model, call ``model.load_adapter("eng")`` to inject the
        English adapter weights, then export. The adapter + lm_head weights are
        language-specific; the encoder weights are shared across all languages.
    """

    default_task: str = "ctc-asr"
    category: str = "Speech-to-Text"
    config_class = MMSConfig

    def __init__(self, config: ArchitectureConfig):
        super().__init__(config)

        add_adapter = getattr(config, "add_adapter", False)
        output_hidden_size = getattr(config, "output_hidden_size", config.hidden_size)
        adapter_kernel_size = getattr(config, "adapter_kernel_size", 3)
        adapter_stride = getattr(config, "adapter_stride", 2)
        num_adapter_layers = getattr(config, "num_adapter_layers", 3)

        if add_adapter:
            self.adapter = _Adapter(
                hidden_size=config.hidden_size,
                output_hidden_size=output_hidden_size,
                num_adapter_layers=num_adapter_layers,
                adapter_kernel_size=adapter_kernel_size,
                adapter_stride=adapter_stride,
            )
        self._adapter_stride = adapter_stride
        self._num_adapter_layers = num_adapter_layers

        # CTC head: projects hidden states to per-frame vocabulary logits
        self.lm_head = Linear(output_hidden_size, config.vocab_size, bias=True)

    def forward(
        self,
        op: builder.OpBuilder,
        input_values: ir.Value,
        attention_mask: ir.Value | None = None,
    ) -> ir.Value:
        """Encode audio to CTC logits.

        Args:
            input_values: (batch, num_samples) raw audio waveform float32.
            attention_mask: (batch, num_samples) INT64 padding mask, where
                1 = valid sample and 0 = padding.  For non-padded inputs
                (all samples are real audio) pass all-ones:
                ``np.ones((batch, num_samples), dtype=np.int64)``.

        Returns:
            logits: (batch, num_frames, vocab_size) CTC log scores
        """
        # Encoder: (B, T) → (B, T', hidden_size)
        hidden_states = super().forward(op, input_values, attention_mask)

        # Language adapter (if loaded): (B, T', H) → (B, T'', H)
        if hasattr(self, "adapter"):
            hidden_states = self.adapter(op, hidden_states)

        # CTC head: (B, T'', H) → (B, T'', vocab_size)
        return self.lm_head(op, hidden_states)

    def frame_lengths(self, op: builder.OpBuilder, attention_mask: ir.Value) -> ir.Value:
        """Per-row valid frame count, including the MMS adapter downsampling.

        The adapter's strided convolutions shrink the frame axis further, so the
        logits' time axis is shorter than the encoder's.  Reporting the encoder
        count here would over-run the logits when segmenting a padded batch.
        """
        lengths = super().frame_lengths(op, attention_mask)
        if not hasattr(self, "adapter"):
            return lengths
        stride = op.Constant(value_int=int(self._adapter_stride))
        one = op.Constant(value_int=1)
        for _ in range(self._num_adapter_layers):
            # Adapter convs use kernel 1 semantics for length purposes: HF's
            # _get_feat_extract_output_lengths passes kernel_size=1.
            lengths = op.Add(op.Div(op.Sub(lengths, one), stride), one)
        return lengths

    def preprocess_weights(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Map HuggingFace ``Wav2Vec2ForCTC`` weight names to ONNX module names.

        The encoder names are handled by :meth:`Wav2Vec2Model.preprocess_weights`
        (prefix strip, FFN rename, positional-conv weight-norm materialization).
        Only the MMS adapter needs extra work: HF wraps its convolution in an
        ``nn.Conv1d`` (``*.conv.weight``/``*.conv.bias``) while ``_AdapterLayer``
        holds bare parameters named ``conv`` and ``conv_bias``.

        ``lm_head.weight``/``lm_head.bias`` already match.
        """
        result = super().preprocess_weights(state_dict)

        adapted: dict[str, torch.Tensor] = {}
        for key, value in result.items():
            if key.startswith("adapter.layers."):
                key = re.sub(r"\.conv\.weight$", ".conv", key)
                key = re.sub(r"\.conv\.bias$", ".conv_bias", key)
            adapted[key] = value
        return adapted
