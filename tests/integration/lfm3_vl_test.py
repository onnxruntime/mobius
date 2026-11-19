# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Pinned full-checkpoint LFM3-VL image features, logits, cache reuse and greedy generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from integration._support import _get_test_device, _make_decode_feeds, _make_prefill_feeds
from mobius import build
from mobius._testing.comparison import assert_logits_close
from mobius._testing.generation import apply_repetition_penalty
from mobius._testing.ort_inference import OnnxModelSession
from mobius._testing.torch_reference import (
    lfm3_vl_reference_compatibility,
    load_torch_multimodal_model,
)

_MODEL_ID = "LiquidAI/LFM3-VL-2B"
_REVISION = "23387beda43c60310348e83e25f78f33c3140d3a"
_TESTDATA = Path(__file__).resolve().parents[2] / "testdata"
# generation_config.json of the pinned revision.
_REPETITION_PENALTY = 1.1

pytestmark = [pytest.mark.integration, pytest.mark.integration_slow]


@pytest.fixture(scope="module")
def lfm3_vl_artifacts():
    """Load the checkpoint once; fail if requested CUDA execution falls back."""
    device = _get_test_device()
    reference, _, processor = load_torch_multimodal_model(
        _MODEL_ID, device=device, revision=_REVISION
    )
    package = build(
        _MODEL_ID,
        dtype="f32",
        execution_provider="cuda" if device == "cuda" else "onnx-standard",
        revision=_REVISION,
        trust_remote_code=True,
        load_weights=True,
    )
    sessions = {name: OnnxModelSession(package[name], device=device) for name in package}
    for session in sessions.values():
        assert session.providers[0] == (
            "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
        )
    try:
        yield reference, processor, package, sessions
    finally:
        for session in sessions.values():
            session.close()


def _embed(sessions, input_ids: np.ndarray, features: np.ndarray) -> np.ndarray:
    return sessions["embedding"].run({"input_ids": input_ids, "image_features": features})[
        "inputs_embeds"
    ]


def _prefill(package, sessions, input_ids, features):
    mask = np.ones_like(input_ids)
    positions = np.arange(input_ids.shape[1], dtype=np.int64)[None]
    feeds = _make_prefill_feeds(package.config, input_ids, mask, positions)
    del feeds["input_ids"]
    feeds["inputs_embeds"] = _embed(sessions, input_ids, features)
    return sessions["decoder"].run(feeds)


def _check_prompt(artifacts, image_names: list[str], text: str):
    """Compare features per image, then prompt logits from reference and from own features."""
    reference, processor, package, sessions = artifacts
    images = [Image.open(_TESTDATA / name).convert("RGB") for name in image_names]
    content = [{"type": "image"} for _ in images] + [{"type": "text", "text": text}]
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False
    )
    inputs = processor(text=[prompt], images=images, return_tensors="pt")
    input_ids = inputs["input_ids"].numpy().astype(np.int64)
    pixel_values = inputs["pixel_values"].numpy()
    grids = inputs["image_grid_thw"].numpy()
    inputs = inputs.to(reference.device)
    with torch.no_grad():
        expected_features = [
            feature.float().cpu().numpy()
            for feature in reference.get_image_features(
                inputs["pixel_values"], inputs["image_grid_thw"]
            ).pooler_output
        ]
        expected = reference(**inputs, use_cache=True)
    expected_logits = expected.logits.float().cpu().numpy()

    # One vision run per image, the ORT GenAI lfm3_vl contract.
    features, row = [], 0
    for grid, reference_features in zip(grids, expected_features, strict=True):
        rows = int(np.prod(grid))
        actual = sessions["vision_encoder"].run(
            {"pixel_values": pixel_values[row : row + rows], "image_grid_thw": grid[None]}
        )["image_features"]
        np.testing.assert_allclose(actual, reference_features, rtol=1e-4, atol=1e-4)
        features.append(actual)
        row += rows

    # Fed the reference's own features, every prompt position matches.
    actual = _prefill(package, sessions, input_ids, np.concatenate(expected_features))
    assert_logits_close(actual["logits"], expected_logits, rtol=1e-3, atol=1e-3)
    # With its own features, image positions inherit the 1e-5 feature noise, which a
    # few of them amplify to 1e-2 (as the reference does between CPU and CUDA); the
    # text positions that generation reads stay within the LFM3 tolerance.
    actual = _prefill(package, sessions, input_ids, np.concatenate(features))
    text = input_ids[0] != reference.config.image_token_id
    assert_logits_close(
        actual["logits"][0, text], expected_logits[0, text], rtol=1e-3, atol=1e-3
    )
    return inputs, input_ids, actual, expected.past_key_values


def test_lfm3_vl_cat_logits_cached_decode_and_greedy_generation(lfm3_vl_artifacts):
    reference, _, package, sessions = lfm3_vl_artifacts
    inputs, prompt_ids, actual, hf_cache = _check_prompt(
        lfm3_vl_artifacts,
        ["pipeline-cat-chonk.jpeg"],
        "Describe this image in one short sentence.",
    )
    assert (prompt_ids == reference.config.image_token_id).sum() == 630

    seen = prompt_ids
    no_features = np.zeros((0, package.config.hidden_size), dtype=np.float32)
    for _ in range(20):
        scores = apply_repetition_penalty(
            actual["logits"][:, -1].astype(np.float32), seen, _REPETITION_PENALTY
        )
        next_ids = scores.argmax(-1, keepdims=True).astype(np.int64)
        seen = np.concatenate([seen, next_ids], axis=1)
        mask = np.ones_like(seen)
        positions = np.array([[seen.shape[1] - 1]], dtype=np.int64)
        feeds = _make_decode_feeds(package.config, next_ids, mask, positions, actual)
        del feeds["input_ids"]
        feeds["inputs_embeds"] = _embed(sessions, next_ids, no_features)
        actual = sessions["decoder"].run(feeds)
        with torch.no_grad():
            expected = reference(
                input_ids=torch.from_numpy(next_ids).to(reference.device),
                attention_mask=torch.from_numpy(mask).to(reference.device),
                position_ids=torch.from_numpy(positions).to(reference.device),
                past_key_values=hf_cache,
                use_cache=True,
            )
        assert_logits_close(
            actual["logits"], expected.logits.float().cpu().numpy(), rtol=1e-3, atol=1e-3
        )
        hf_cache = expected.past_key_values
    generated = seen[0, prompt_ids.shape[1] :].tolist()

    with torch.no_grad(), lfm3_vl_reference_compatibility(_MODEL_ID, _REVISION):
        expected_ids = reference.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            repetition_penalty=_REPETITION_PENALTY,
        )[0, prompt_ids.shape[1] :].tolist()
    golden = json.loads(
        (_TESTDATA / "golden" / "vision-language" / "lfm3-vl-2b_generation.json").read_text()
    )
    assert len(generated) == len(expected_ids) == len(golden["generated_tokens"]) == 20
    assert generated == expected_ids == golden["generated_tokens"]


def test_lfm3_vl_two_images_wide_and_tall(lfm3_vl_artifacts):
    inputs, _, _, _ = _check_prompt(
        lfm3_vl_artifacts,
        ["pipeline-cat-chonk.jpeg", "mage-vl-dog.jpg"],
        "What do these two images show?",
    )
    assert inputs["image_grid_thw"].tolist() == [[1, 42, 60], [1, 128, 64]]


def test_lfm3_vl_largest_image_features(lfm3_vl_artifacts):
    """The processor's 4096x4096 limit: 65,536 patches, 16 x 65,536^2 float32 scores."""
    reference, processor, _, sessions = lfm3_vl_artifacts
    if reference.device.type != "cuda":
        pytest.skip("ORT's CPU Attention in the onnx-standard build would hold all 275 GB")
    image = Image.open(_TESTDATA / "pipeline-cat-chonk.jpeg").convert("RGB")
    inputs = processor.image_processor(
        images=[image.resize((4096, 4096))], return_tensors="pt"
    )
    assert inputs["image_grid_thw"].tolist() == [[1, 256, 256]]
    # Eager attention would also hold every score.
    reference.model.vision_tower.config._attn_implementation = "sdpa"
    try:
        with torch.no_grad():
            expected = reference.get_image_features(
                inputs["pixel_values"].to(reference.device),
                inputs["image_grid_thw"].to(reference.device),
            ).pooler_output[0]
    finally:
        reference.model.vision_tower.config._attn_implementation = "eager"
    actual = sessions["vision_encoder"].run(
        {
            "pixel_values": inputs["pixel_values"].numpy(),
            "image_grid_thw": inputs["image_grid_thw"].numpy(),
        }
    )["image_features"]
    np.testing.assert_allclose(actual, expected.float().cpu().numpy(), rtol=1e-4, atol=1e-4)
