# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

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
from onnxscript import nn
from onnxscript._internal import builder

from mobius._configs import ArchitectureConfig
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
        # Conv1d with padding=1 on both sides (matches HF padding=1)
        hidden_states = op.Conv(
            hidden_states,
            self.conv,
            self.conv_bias,
            kernel_shape=[self._kernel_size],
            strides=[self._stride],
            pads=[1, 1],
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

    def preprocess_weights(self, state_dict: dict[str, object]) -> dict[str, object]:
        """Map HuggingFace weight names to ONNX module names.

        HF layout (Wav2Vec2ForCTC):
          wav2vec2.feature_extractor.conv_layers.N.conv.weight  →  feature_extractor.conv_layers.N.conv
          wav2vec2.feature_extractor.conv_layers.N.conv.bias    →  feature_extractor.conv_layers.N.conv_bias
          wav2vec2.feature_extractor.conv_layers.0.layer_norm.* →  feature_extractor.conv_layers.0.layer_norm[_bias]
          wav2vec2.encoder.layer_norm.weight                    →  layer_norm.weight  (top-level post-encoder norm)
          wav2vec2.encoder.pos_conv_embed.*                     →  (dropped — not in our model)
          wav2vec2.encoder.layers.N.*                           →  encoder.layers.N.*
          wav2vec2.adapter.layers.N.conv.weight                 →  adapter.layers.N.conv  (bare param)
          wav2vec2.feature_projection.*                         →  feature_projection.*
          lm_head.weight / lm_head.bias                        →  lm_head.weight / lm_head.bias
        """
        result: dict[str, object] = {}
        for key, value in state_dict.items():
            k = key

            # Strip outer model prefix (wav2vec2.*, hubert.*, wavlm.*)
            for prefix in ("wav2vec2.", "hubert.", "wavlm."):
                if k.startswith(prefix):
                    k = k[len(prefix) :]
                    break

            # FFN weight renames: HF uses intermediate_dense/output_dense, we use up_proj/down_proj
            k = k.replace(".intermediate_dense.", ".up_proj.").replace(
                ".output_dense.", ".down_proj."
            )

            # Feature extractor CNN: nn.Conv1d has .weight/.bias sub-attributes;
            # our _ConvLayerBlock uses bare nn.Parameter named 'conv' and 'conv_bias'.
            if k.startswith("feature_extractor.conv_layers."):
                # conv.weight → conv (bare param)
                k = re.sub(r"\.conv\.weight$", ".conv", k)
                # conv.bias → conv_bias (bare param named differently)
                k = re.sub(r"\.conv\.bias$", ".conv_bias", k)
                # GroupNorm weight/bias → bare params layer_norm / layer_norm_bias
                k = re.sub(r"\.layer_norm\.weight$", ".layer_norm", k)
                k = re.sub(r"\.layer_norm\.bias$", ".layer_norm_bias", k)
                result[k] = value
                continue

            # Stable encoder layer_norm lives at HF's encoder.layer_norm.*;
            # our Wav2Vec2Model puts it at top-level as self.layer_norm.
            if k == "encoder.layer_norm.weight":
                result["layer_norm.weight"] = value
                continue
            if k == "encoder.layer_norm.bias":
                result["layer_norm.bias"] = value
                continue

            # Positional conv embedding — not present in our simplified encoder.
            if k.startswith("encoder.pos_conv_embed."):
                continue

            # Adapter conv weights: HF uses nn.Conv1d → .conv.weight / .conv.bias;
            # our _AdapterLayer uses bare params named 'conv' / 'conv_bias'.
            if k.startswith("adapter.layers."):
                k = re.sub(r"\.conv\.weight$", ".conv", k)
                k = re.sub(r"\.conv\.bias$", ".conv_bias", k)

            result[k] = value

        return result
