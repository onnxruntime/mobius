# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Speech-enhancement task.

Builds a single ONNX graph for spectral speech-enhancement models (e.g.
RE-USE / SEMamba) that map a noisy STFT magnitude and phase to an enhanced
magnitude, phase, and complex spectrogram.  The STFT and ISTFT stay outside
the graph, as they do for every other audio model in mobius.
"""

from __future__ import annotations

from typing import ClassVar

import onnx_ir as ir

from mobius._configs import BaseModelConfig
from mobius._model_package import ModelPackage
from mobius.tasks._base import ModelTask, _make_graph, _make_model


class SpeechEnhancementTask(ModelTask):
    """Build an ONNX graph for spectral speech enhancement (encoder-only).

    Inputs:
        ``noisy_mag`` — ``[batch, freq, time]`` noisy STFT magnitude.
        ``noisy_pha`` — ``[batch, freq, time]`` noisy STFT phase, in radians.

    Outputs:
        ``denoised_mag`` — ``[batch, freq, time]``
        ``denoised_pha`` — ``[batch, freq, time]``
        ``denoised_com`` — ``[batch, freq, time, 2]`` real/imaginary pair.
    """

    model_roles: ClassVar[dict[str, str]] = {"model": "encoder"}

    def build(
        self,
        module,
        config: BaseModelConfig,
    ) -> ModelPackage:
        graph, builder = _make_graph()

        # Native-rate RE-USE scales its FFT geometry from the decoded sample
        # rate, so frequency is dynamic by default. An explicit native/BWE
        # export selects a static geometry for provider partitioning.
        num_freq = getattr(config, "static_num_freq_bins", None) or "freq"
        shape = ["batch", num_freq, "time"]

        noisy_mag = builder.input("noisy_mag", dtype=config.dtype, shape=shape)
        noisy_pha = builder.input("noisy_pha", dtype=config.dtype, shape=shape)

        denoised_mag, denoised_pha, denoised_com = module(
            builder.op,
            noisy_mag=noisy_mag,
            noisy_pha=noisy_pha,
        )

        # The model is spectrally shape preserving, but symbolic shape
        # inference cannot see through the Scan-based SSM recurrence, so
        # republish the input's named dimensions rather than leaving the
        # outputs fully anonymous.
        denoised_mag.shape = ir.Shape(shape)
        denoised_pha.shape = ir.Shape(shape)
        denoised_com.shape = ir.Shape([*shape, 2])

        builder.add_output(denoised_mag, "denoised_mag")
        builder.add_output(denoised_pha, "denoised_pha")
        builder.add_output(denoised_com, "denoised_com")

        return ModelPackage({"model": _make_model(graph)}, config=config)
