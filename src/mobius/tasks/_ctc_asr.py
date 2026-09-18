# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CTC ASR task — builds a single ONNX graph for CTC-based ASR models.

Produces one model:
  - ``"model"``: raw audio waveform → per-frame vocabulary logits

Supports MMS (facebook/mms-1b-all) and any Wav2Vec2ForCTC checkpoint.
"""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir

from mobius._configs import ArchitectureConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class CTCAsrTask(ModelTask):
    """Build ONNX graph for CTC-based ASR (raw waveform → frame logits).

    Input:
        ``input_values``   — (batch, num_samples) raw audio at 16 kHz  FLOAT
        ``attention_mask`` — (batch, num_samples) INT64 padding mask
                             (1 = valid sample, 0 = padding). Always a
                             required graph input; callers with no
                             padding should pass an all-ones mask.

    Outputs:
        ``logits``        — (batch, num_frames, vocab_size) CTC logit scores
        ``frame_lengths`` — (batch,) INT64 count of non-padded frames per row

    ``frame_lengths`` is emitted so a padded batch can be segmented back into
    per-row transcripts without the caller re-deriving the convolutional
    downsampling ratio.  It is only emitted when the module knows how to compute
    it, keeping the task usable for encoders with a different contract.
    """

    name = "ctc-asr"
    model_roles: ClassVar[dict[str, str]] = {"model": "encoder"}

    def build(
        self,
        module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        batch = ir.SymbolicDim("batch")
        time = ir.SymbolicDim("time")

        graph, builder = _make_graph(name="ctc_asr")
        input_values = builder.input(
            "input_values",
            dtype=ir.DataType.FLOAT,
            shape=[batch, time],
        )
        attention_mask = builder.input(
            "attention_mask",
            dtype=ir.DataType.INT64,
            shape=[batch, time],
        )

        logits = module(builder.op, input_values=input_values, attention_mask=attention_mask)
        builder.add_output(logits, "logits")

        frame_lengths_fn = getattr(module, "frame_lengths", None)
        if callable(frame_lengths_fn):
            frame_lengths = frame_lengths_fn(builder.op, attention_mask)
            builder.add_output(frame_lengths, "frame_lengths")

        return ModelPackage({"model": _make_model(graph)}, config=config)


class FeatureCTCAsrTask(ModelTask):
    """Build feature-input CTC ASR (log-mel features → frame logits).

    Inputs:
        ``input_features`` — (batch, frames, mel_bins) normalized log-mel values
        ``attention_mask`` — (batch, frames) BOOL valid-frame mask

    Output:
        ``logits`` — (batch, subsampled_frames, vocab_size) CTC scores
    """

    name = "feature-ctc-asr"
    model_roles: ClassVar[dict[str, str]] = {"model": "encoder"}

    def build(
        self,
        module,
        config: ArchitectureConfig,
    ) -> ModelPackage:
        batch = ir.SymbolicDim("batch")
        frames = ir.SymbolicDim("frames")

        graph, builder = _make_graph(name="feature_ctc_asr")
        input_features = builder.input(
            "input_features",
            dtype=config.dtype,
            shape=[batch, frames, config.num_mel_bins],
        )
        attention_mask = builder.input(
            "attention_mask",
            dtype=ir.DataType.BOOL,
            shape=[batch, frames],
        )

        logits = module(
            builder.op,
            input_features=input_features,
            attention_mask=attention_mask,
        )
        builder.add_output(logits, "logits")
        return ModelPackage({"model": _make_model(graph)}, config=config)
