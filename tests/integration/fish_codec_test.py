# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unmodified-upstream and real-checkpoint tests for the pinned waveform codec."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
from huggingface_hub import hf_hub_download

from mobius._configs.fish_codec import (
    DRAMA3_CODEC_REPO,
    DRAMA3_CODEC_REVISION,
    FishCodecConfig,
)
from mobius._testing.ort_inference import OnnxModelSession
from mobius.tasks.fish_codec import build_fish_codec_decoder, load_drama3_codec_decoder
from tests.integration._fish_codec_reference import (
    initialize_reference,
    load_reference_sources,
    make_reference,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    pytest.importorskip("einops")
    return load_reference_sources(tmp_path_factory.mktemp("fish-codec-source"))


@pytest.mark.parametrize("batch,frames", [(1, 1), (2, 7), (1, 131)])
def test_tiny_upstream_waveform_parity(sources, batch, frames):
    config = FishCodecConfig(
        latent_dim=16,
        transformer_layers=2,
        transformer_heads=2,
        transformer_intermediate_size=32,
        max_frames=160,
        decoder_dim=32,
        decoder_rates=(2, 2),
    )
    reference = initialize_reference(make_reference(config, sources))
    state = reference.state_dict()
    package = build_fish_codec_decoder(config, state_dict=state)
    # Include above-native IDs to exercise the exact upstream upper clamping.
    codes = np.random.default_rng(42).integers(
        0, 4096, size=(batch, config.num_codebooks, frames), dtype=np.int64
    )
    with torch.inference_mode():
        expected = reference(torch.from_numpy(codes)).numpy()
    session = OnnxModelSession(package["waveform_decoder"])
    actual = session.run({"codes": codes})["waveform"]
    assert actual.shape == (batch, 1, frames * config.samples_per_frame)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
    if frames > config.transformer_window:
        # Reuse the same graph with a different time dimension and verify the
        # whole decode stack is causal, not just its transformer attention.
        prefix = session.run({"codes": codes[:, :, :3]})["waveform"]
        np.testing.assert_allclose(
            prefix, actual[:, :, : 3 * config.samples_per_frame], atol=1e-5, rtol=1e-4
        )


@pytest.mark.skipif(
    os.environ.get("MOBIUS_TEST_DRAMA3_CODEC") != "1",
    reason="Opt in to the 1.87GB pinned real-codec download",
)
def test_real_checkpoint_waveform_parity(sources, tmp_path):
    """Bounded 20-frame decode of the independently generated token golden.

    Set MOBIUS_DRAMA3_CODEC_PATH to reuse a local pinned checkpoint copy.
    The default downloads only codec.pth, never the large language weights.
    """
    path = os.environ.get("MOBIUS_DRAMA3_CODEC_PATH")
    if path is None:
        path = hf_hub_download(DRAMA3_CODEC_REPO, "codec.pth", revision=DRAMA3_CODEC_REVISION)
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    config = dataclasses.replace(FishCodecConfig(), max_frames=256)
    reference = make_reference(config, sources)
    loaded = reference.load_state_dict(state, strict=False, assign=True)
    assert not loaded.missing_keys
    golden_path = (
        Path(__file__).parents[2]
        / "testdata"
        / "golden"
        / "speech"
        / "drama-3_tokens_generation.json"
    )
    golden = json.loads(golden_path.read_text())
    frames = np.asarray(golden["generated_frames"], dtype=np.int64)
    codes = np.ascontiguousarray(frames[:20, 1:].T[None])
    assert codes.shape == (1, 10, 20)
    with torch.inference_mode():
        expected = reference(torch.from_numpy(codes)).numpy()
    # Exercise the production loader with its full-size default configuration,
    # then the public export API and a fresh CPU ORT session on the saved file.
    package = load_drama3_codec_decoder(path)
    output_dir = tmp_path / "export"
    package.save(str(output_dir), progress_bar=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(
        str(output_dir / "model.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    actual = session.run(["waveform"], {"codes": codes})[0]
    assert actual.shape == (1, 1, 40960)
    assert actual.dtype == expected.dtype == np.float32
    assert np.isfinite(actual).all()
    assert np.sqrt(np.mean(expected**2)) > 1e-3
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-3)
    error = actual.astype(np.float64) - expected
    rmse = float(np.sqrt(np.mean(error**2)))
    snr = float(10 * np.log10(np.mean(expected.astype(np.float64) ** 2) / np.mean(error**2)))
    print(
        f"Drama3 codec: samples={actual.size}, rate={config.sample_rate}, "
        f"max_abs={np.max(np.abs(error)):.9g}, rmse={rmse:.9g}, snr_db={snr:.4f}"
    )
