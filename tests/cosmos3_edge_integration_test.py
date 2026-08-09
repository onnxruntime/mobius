# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Integration tests: NVIDIA Cosmos3-Edge Reasoner against real weights.

Verifies the exported Reasoner graphs stage by stage using the real
``nvidia/Cosmos3-Edge`` checkpoint and the PyTorch transcription of the
published ``cosmos3_edge`` modeling code in
:mod:`tests._cosmos3_edge_reference`::

    pytest tests/cosmos3_edge_integration_test.py -m integration -sv

Stages checked: SigLIP2 vision tower + merger projector (image and video),
image/video token fusion in the embedding graph, decoder logits under
interleaved 3D M-RoPE, and a cached greedy-decode generation smoke test.

.. note::
   Only the *understanding* (Reasoner) tower is verified. The Cosmos3-Edge
   Generator, Action head and Sound tower share the same checkpoint but are
   proprietary rectified-flow components with no published reference
   implementation, so their numerics remain unverifiable here.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

from tests._cosmos3_edge_reference import (
    EdgeRefConfig,
    patchify_images,
    patchify_videos,
    ref_text_decoder_logits,
    ref_vision_features,
    smart_resize,
)

MODEL_ID = "nvidia/Cosmos3-Edge"
PATCH_SIZE = 16
MERGE_SIZE = 2
IMAGE_TOKEN_ID = 19
VIDEO_TOKEN_ID = 18

pytestmark = [pytest.mark.integration, pytest.mark.integration_slow]


def _checkpoint_dir() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_ID, allow_patterns=["*.json", "*/*.safetensors"])


def _normalized(image: torch.Tensor) -> torch.Tensor:
    """Apply the checkpoint's ``image_mean``/``image_std`` of 0.5."""
    return (image - 0.5) / 0.5


def _split_image(height: int, width: int) -> torch.Tensor:
    """Left half red, right half green — a deterministic, describable image."""
    image = torch.zeros(1, 3, height, width, dtype=torch.float32)
    image[:, 0, :, : width // 2] = 1.0
    image[:, 1, :, width // 2 :] = 1.0
    return _normalized(image)


@pytest.fixture(scope="module")
def edge_package(tmp_path_factory):
    """Build the Reasoner graphs with real fp32 weights and open ORT sessions."""
    import onnx_ir as ir
    import onnxruntime as ort

    import mobius
    from mobius._weight_loading import iter_weight_shards
    from mobius.models.cosmos import Cosmos3EdgeVLModel

    snapshot = _checkpoint_dir()
    with open(os.path.join(snapshot, "config.json"), encoding="utf-8") as handle:
        ref_config = EdgeRefConfig.from_hf_config(json.load(handle))

    package = mobius.build(MODEL_ID, task="cosmos3-edge-vl", dtype="f32", load_weights=False)
    module = Cosmos3EdgeVLModel(package.config)
    reference: dict[str, torch.Tensor] = {}
    for shard in iter_weight_shards(MODEL_ID):
        package.apply_weights_partial(module.preprocess_weights(shard))
        for key, value in shard.items():
            if "k_norm_und_for_gen" in key or "moe_gen" in key:
                continue
            if key.startswith(("model.visual.", "model.projector.")):
                reference[key.removeprefix("model.")] = value.float()
            elif key.startswith("layers.") or key in (
                "embed_tokens.weight",
                "norm.weight",
                "lm_head.weight",
            ):
                reference[key] = value.float()
    package.finalize_weights()
    package.validate_weights()

    directory = tmp_path_factory.mktemp("cosmos3_edge_real")
    sessions = {}
    for name in ("vision_encoder", "embedding", "decoder"):
        path = directory / f"{name}.onnx"
        ir.save(package[name], str(path), external_data=f"{name}.onnx.data")
        sessions[name] = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return package.config, ref_config, sessions, reference


def test_vision_encoder_input_contract(edge_package):
    """The exported vision graph takes packed patches, not a single image."""
    _, _, sessions, _ = edge_package
    inputs = {value.name: value for value in sessions["vision_encoder"].get_inputs()}
    assert set(inputs) == {"pixel_values", "grid_thw"}
    assert inputs["pixel_values"].shape[1] == PATCH_SIZE * PATCH_SIZE * 3
    assert inputs["grid_thw"].shape == [3]


@pytest.mark.parametrize(("height", "width"), [(256, 256), (128, 512), (320, 192)])
def test_vision_features_match_reference(edge_package, height, width):
    _, ref_config, sessions, reference = edge_package
    image = _split_image(height, width)
    packed, grid_h, grid_w = patchify_images(
        image, patch_size=PATCH_SIZE, merge_size=MERGE_SIZE
    )
    packed = packed[0]

    got = sessions["vision_encoder"].run(
        None,
        {
            "pixel_values": packed.numpy(),
            "grid_thw": np.array([1, grid_h, grid_w], dtype=np.int64),
        },
    )[0]
    expected = ref_vision_features(
        packed, torch.tensor([[1, grid_h, grid_w]]), reference, ref_config
    ).numpy()

    assert got.shape == (grid_h * grid_w // MERGE_SIZE**2, ref_config.hidden_size)
    np.testing.assert_allclose(got, expected, atol=1e-3, rtol=1e-3)
    correlation = np.corrcoef(got.reshape(-1), expected.reshape(-1))[0, 1]
    assert correlation > 0.9999


def test_smart_resized_photo_resolution_matches_reference(edge_package):
    """Drive the processor's own ``smart_resize`` policy, not a hand-picked size.

    A natural 1000x750 photo is resized to a multiple of ``patch*merge`` (32)
    inside the checkpoint's pixel-area bounds, then patchified and encoded.
    """
    _, ref_config, sessions, reference = edge_package
    height, width = smart_resize(
        750,
        1000,
        factor=PATCH_SIZE * MERGE_SIZE,
        min_pixels=256 * 256,
        max_pixels=4096 * 4096,
    )
    assert height % (PATCH_SIZE * MERGE_SIZE) == 0
    assert width % (PATCH_SIZE * MERGE_SIZE) == 0
    assert 256 * 256 <= height * width <= 4096 * 4096

    image = _split_image(height, width)
    packed, grid_h, grid_w = patchify_images(
        image, patch_size=PATCH_SIZE, merge_size=MERGE_SIZE
    )
    got = sessions["vision_encoder"].run(
        None,
        {
            "pixel_values": packed[0].numpy(),
            "grid_thw": np.array([1, grid_h, grid_w], dtype=np.int64),
        },
    )[0]
    expected = ref_vision_features(
        packed[0], torch.tensor([[1, grid_h, grid_w]]), reference, ref_config
    ).numpy()

    assert got.shape == (grid_h * grid_w // MERGE_SIZE**2, ref_config.hidden_size)
    np.testing.assert_allclose(got, expected, atol=1e-3, rtol=1e-3)


def test_video_features_match_reference_and_token_count(edge_package):
    _, ref_config, sessions, reference = edge_package
    frames = 4
    video = torch.stack([_split_image(128, 160)[0] for _ in range(frames)]).unsqueeze(0)
    packed, grid_t, grid_h, grid_w = patchify_videos(
        video, patch_size=PATCH_SIZE, merge_size=MERGE_SIZE
    )
    packed = packed[0]

    got = sessions["vision_encoder"].run(
        None,
        {
            "pixel_values": packed.numpy(),
            "grid_thw": np.array([grid_t, grid_h, grid_w], dtype=np.int64),
        },
    )[0]
    expected = ref_vision_features(
        packed, torch.tensor([[grid_t, grid_h, grid_w]]), reference, ref_config
    ).numpy()

    tokens_per_frame = grid_h * grid_w // MERGE_SIZE**2
    assert got.shape == (frames * tokens_per_frame, ref_config.hidden_size)
    np.testing.assert_allclose(got, expected, atol=1e-3, rtol=1e-3)


def test_image_and_video_fusion_and_decoder_logits(edge_package):
    config, ref_config, sessions, reference = edge_package

    image = _split_image(256, 256)
    image_packed, img_h, img_w = patchify_images(
        image, patch_size=PATCH_SIZE, merge_size=MERGE_SIZE
    )
    image_features = sessions["vision_encoder"].run(
        None,
        {
            "pixel_values": image_packed[0].numpy(),
            "grid_thw": np.array([1, img_h, img_w], dtype=np.int64),
        },
    )[0]

    frames = 2
    video = torch.stack([_split_image(64, 96)[0] for _ in range(frames)]).unsqueeze(0)
    video_packed, vid_t, vid_h, vid_w = patchify_videos(
        video, patch_size=PATCH_SIZE, merge_size=MERGE_SIZE
    )
    video_features = sessions["vision_encoder"].run(
        None,
        {
            "pixel_values": video_packed[0].numpy(),
            "grid_thw": np.array([vid_t, vid_h, vid_w], dtype=np.int64),
        },
    )[0]

    ids = [101, 102]
    image_start = len(ids)
    ids += [IMAGE_TOKEN_ID] * image_features.shape[0]
    video_start = len(ids)
    ids += [VIDEO_TOKEN_ID] * video_features.shape[0]
    ids += [201]
    input_ids = np.array([ids], dtype=np.int64)

    inputs_embeds = sessions["embedding"].run(
        None,
        {
            "input_ids": input_ids,
            "image_features": image_features,
            "video_features": video_features,
        },
    )[0]

    expected_embeds = torch.nn.functional.embedding(
        torch.from_numpy(input_ids), reference["embed_tokens.weight"]
    )
    expected_embeds = expected_embeds.masked_scatter(
        torch.from_numpy(input_ids == IMAGE_TOKEN_ID).unsqueeze(-1),
        torch.from_numpy(image_features),
    )
    expected_embeds = expected_embeds.masked_scatter(
        torch.from_numpy(input_ids == VIDEO_TOKEN_ID).unsqueeze(-1),
        torch.from_numpy(video_features),
    )
    np.testing.assert_allclose(inputs_embeds, expected_embeds.numpy(), atol=1e-3)

    length = len(ids)
    positions = np.zeros((3, 1, length), dtype=np.int64)
    for index in range(image_start):
        positions[:, 0, index] = index
    base = image_start
    merged_w = img_w // MERGE_SIZE
    for token in range(image_features.shape[0]):
        positions[0, 0, image_start + token] = base
        positions[1, 0, image_start + token] = base + token // merged_w
        positions[2, 0, image_start + token] = base + token % merged_w
    base += max(img_h, img_w) // MERGE_SIZE
    tokens_per_frame = video_features.shape[0] // frames
    merged_vw = vid_w // MERGE_SIZE
    for token in range(video_features.shape[0]):
        frame, spatial = divmod(token, tokens_per_frame)
        positions[0, 0, video_start + token] = base + frame
        positions[1, 0, video_start + token] = base + spatial // merged_vw
        positions[2, 0, video_start + token] = base + spatial % merged_vw
    base += max(frames, vid_h // MERGE_SIZE, merged_vw)
    positions[:, 0, length - 1] = base

    feeds = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": np.ones((1, length), dtype=np.int64),
        "position_ids": positions,
    }
    empty = np.zeros((1, config.num_key_value_heads, 0, config.head_dim), dtype=np.float32)
    for layer in range(config.num_hidden_layers):
        feeds[f"past_key_values.{layer}.key"] = empty
        feeds[f"past_key_values.{layer}.value"] = empty
    logits = sessions["decoder"].run(["logits"], feeds)[0]

    expected_logits = ref_text_decoder_logits(
        expected_embeds, torch.from_numpy(positions), reference, ref_config
    ).numpy()
    np.testing.assert_allclose(logits, expected_logits, atol=5e-3, rtol=5e-3)
    assert (logits.argmax(-1) == expected_logits.argmax(-1)).all()


def test_text_only_decoder_matches_reference(edge_package):
    """Regression guard: the text path must stay correct after the vision fix."""
    config, ref_config, sessions, reference = edge_package
    input_ids = np.array([[5, 77, 900, 12, 34, 56, 78, 90]], dtype=np.int64)
    length = input_ids.shape[1]
    empty_features = np.zeros((0, config.hidden_size), dtype=np.float32)

    inputs_embeds = sessions["embedding"].run(
        None,
        {
            "input_ids": input_ids,
            "image_features": empty_features,
            "video_features": empty_features,
        },
    )[0]
    expected_embeds = torch.nn.functional.embedding(
        torch.from_numpy(input_ids), reference["embed_tokens.weight"]
    )
    np.testing.assert_allclose(inputs_embeds, expected_embeds.numpy(), atol=0)

    positions = np.tile(np.arange(length, dtype=np.int64), (3, 1, 1))
    feeds = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": np.ones((1, length), dtype=np.int64),
        "position_ids": positions,
    }
    empty = np.zeros((1, config.num_key_value_heads, 0, config.head_dim), dtype=np.float32)
    for layer in range(config.num_hidden_layers):
        feeds[f"past_key_values.{layer}.key"] = empty
        feeds[f"past_key_values.{layer}.value"] = empty
    logits = sessions["decoder"].run(["logits"], feeds)[0]

    expected = ref_text_decoder_logits(
        expected_embeds, torch.from_numpy(positions), reference, ref_config
    ).numpy()
    np.testing.assert_allclose(logits, expected, atol=5e-3, rtol=5e-3)
    assert (logits.argmax(-1) == expected.argmax(-1)).all()


def test_greedy_generation_smoke_with_image(edge_package):
    """Prefill with an image, then decode a few tokens through the KV cache."""
    config, _, sessions, _ = edge_package
    image = _split_image(256, 256)
    packed, grid_h, grid_w = patchify_images(
        image, patch_size=PATCH_SIZE, merge_size=MERGE_SIZE
    )
    image_features = sessions["vision_encoder"].run(
        None,
        {
            "pixel_values": packed[0].numpy(),
            "grid_thw": np.array([1, grid_h, grid_w], dtype=np.int64),
        },
    )[0]
    empty_features = np.zeros((0, config.hidden_size), dtype=np.float32)

    ids = [101, 20] + [IMAGE_TOKEN_ID] * image_features.shape[0] + [21, 102]
    input_ids = np.array([ids], dtype=np.int64)
    length = input_ids.shape[1]

    inputs_embeds = sessions["embedding"].run(
        None,
        {
            "input_ids": input_ids,
            "image_features": image_features,
            "video_features": empty_features,
        },
    )[0]
    feeds = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": np.ones((1, length), dtype=np.int64),
        "position_ids": np.tile(np.arange(length, dtype=np.int64), (3, 1, 1)),
    }
    empty = np.zeros((1, config.num_key_value_heads, 0, config.head_dim), dtype=np.float32)
    for layer in range(config.num_hidden_layers):
        feeds[f"past_key_values.{layer}.key"] = empty
        feeds[f"past_key_values.{layer}.value"] = empty

    names = [value.name for value in sessions["decoder"].get_outputs()]
    generated: list[int] = []
    position = length
    for _ in range(4):
        outputs = dict(zip(names, sessions["decoder"].run(None, feeds), strict=True))
        logits = outputs["logits"]
        assert np.isfinite(logits).all()
        token = int(logits[0, -1].argmax())
        assert 0 <= token < config.vocab_size
        generated.append(token)
        step_embeds = sessions["embedding"].run(
            None,
            {
                "input_ids": np.array([[token]], dtype=np.int64),
                "image_features": empty_features,
                "video_features": empty_features,
            },
        )[0]
        feeds = {
            "inputs_embeds": step_embeds,
            "attention_mask": np.ones((1, position + 1), dtype=np.int64),
            "position_ids": np.full((3, 1, 1), position, dtype=np.int64),
        }
        for layer in range(config.num_hidden_layers):
            feeds[f"past_key_values.{layer}.key"] = outputs[f"present.{layer}.key"]
            feeds[f"past_key_values.{layer}.value"] = outputs[f"present.{layer}.value"]
        position += 1

    assert len(generated) == 4
