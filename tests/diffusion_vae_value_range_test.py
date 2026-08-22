# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Numeric regression test for the diffusion VAE decoder's value_range.

The ``diffusion`` and ``diffusion_guided`` conformance packages declare
``value_range: negative_one_to_one`` for their VAE decoder's ``image`` output
(see ``build_diffusion_workflow_metadata`` in
``mobius.integrations.onnx_genai.workflow_metadata``). Declaring the range is
only useful if it is true: this test executes the synthetic VAE decoder graph
itself and asserts its numeric output actually stays within the declared
bound, so a future edit that removes the bounding op regresses a test rather
than silently making the metadata a false claim.
"""

from __future__ import annotations

import os

import numpy as np
import onnxruntime as ort
import pytest


@pytest.mark.parametrize("package_name", ["diffusion", "diffusion_guided"])
def test_vae_decoder_output_stays_within_declared_value_range(
    materialized_workflow_packages, package_name
):
    model_path = os.path.join(
        materialized_workflow_packages, package_name, "vae_decoder", "model.onnx"
    )
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    (latent_input,) = session.get_inputs()
    # Values far outside any real VAE's latent distribution: if the graph
    # relied on the input already being in range instead of actually bounding
    # its output, this would surface it.
    latents = (
        np.random.default_rng(0).uniform(-50.0, 50.0, size=(1, 4, 8, 8)).astype(np.float32)
    )
    (image,) = session.run(None, {latent_input.name: latents})
    assert image.min() >= -1.0
    assert image.max() <= 1.0
