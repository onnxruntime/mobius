# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Speaker-diarization task.

Builds a single ONNX graph for Sortformer-style diarization models that
consume a mel-spectrogram feature sequence and emit per-frame speaker
activity probabilities.
"""

from __future__ import annotations

from typing import ClassVar

from mobius._configs import BaseModelConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class DiarizationTask(ModelTask):
    """Build an ONNX graph for speaker diarization (encoder-only).

    Input:  ``input_features`` — ``[batch, feat, time]`` mel spectrogram.
    Output: ``speaker_probs`` — ``[batch, frames, num_spks]`` sigmoid
    probabilities (``frames = time / subsampling_factor``).
    """

    model_roles: ClassVar[dict[str, str]] = {"model": "encoder"}

    def build(
        self,
        module,
        config: BaseModelConfig,
    ) -> ModelPackage:
        graph, builder = _make_graph()

        feat_in = getattr(config, "feat_in", None)
        input_features = builder.input(
            "input_features",
            dtype=config.dtype,
            shape=["batch", feat_in if feat_in is not None else "feat", "time"],
        )

        speaker_probs = module(builder.op, input_features=input_features)

        builder.add_output(speaker_probs, "speaker_probs")

        return ModelPackage({"model": _make_model(graph)}, config=config)
