# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Numerical parity tests for the Cosmos3-Edge Reasoner (tiny configs).

These are L2 tests: they build the real ONNX graphs from a tiny architecture
config, fill them with random weights named exactly as the HuggingFace
checkpoint names them, run them under onnxruntime and compare against
:mod:`tests._cosmos3_edge_reference` — a transcription of the published
``transformers``/vLLM ``cosmos3_edge`` modeling code.

They cover the pieces that silently corrupt image understanding when wrong:

- the ``nn.Linear`` patch embedding over channel-last, block-major patches,
- the antialiased position-embedding resample for non-square patch grids,
- the pixel-shuffle merger ordering,
- per-frame (video) attention and token counts,
- interleaved 3D M-RoPE in the text decoder.
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch

from mobius._configs import ArchitectureConfig, VisionConfig
from mobius.models.cosmos import Cosmos3EdgeVLModel
from mobius.tasks import Cosmos3EdgeVLTask
from tests._cosmos3_edge_reference import (
    EdgeRefConfig,
    patchify_images,
    patchify_videos,
    ref_text_decoder_logits,
    ref_vision_features,
    smart_resize,
)

IMAGE_TOKEN_ID = 19
VIDEO_TOKEN_ID = 18
PATCH_SIZE = 8
MERGE_SIZE = 2
NUM_PATCHES = 16  # 4x4 learned reference grid
VISION_HIDDEN = 32
TEXT_HIDDEN = 64
HEAD_DIM = 24  # mrope_section sums to head_dim // 2
MROPE_SECTION = [4, 4, 4]


def _config() -> ArchitectureConfig:
    return ArchitectureConfig(
        model_type="cosmos3_edge",
        hidden_size=TEXT_HIDDEN,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        vocab_size=64,
        max_position_embeddings=256,
        hidden_act="relu2",
        rms_norm_eps=1e-5,
        rope_theta=1_000_000.0,
        mrope_section=MROPE_SECTION,
        mrope_interleaved=True,
        image_token_id=IMAGE_TOKEN_ID,
        dtype=ir.DataType.FLOAT,
        vision=VisionConfig(
            hidden_size=VISION_HIDDEN,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=None,
            patch_size=PATCH_SIZE,
            num_patches=NUM_PATCHES,
            norm_eps=1e-6,
            spatial_merge_size=MERGE_SIZE,
            temporal_patch_size=1,
            out_hidden_size=TEXT_HIDDEN,
            projector_intermediate_size=48,
            use_postshuffle_norm=False,
            image_token_id=IMAGE_TOKEN_ID,
            video_token_id=VIDEO_TOKEN_ID,
        ),
    )


def _ref_config(config: ArchitectureConfig) -> EdgeRefConfig:
    vision = config.vision
    assert vision is not None
    return EdgeRefConfig(
        vision_hidden_size=vision.hidden_size,
        vision_intermediate_size=vision.intermediate_size,
        vision_num_layers=vision.num_hidden_layers,
        vision_num_heads=vision.num_attention_heads,
        patch_size=vision.patch_size,
        num_channels=vision.in_channels,
        num_patches=vision.num_patches,
        layer_norm_eps=vision.norm_eps,
        spatial_merge_size=vision.spatial_merge_size,
        projector_hidden_size=vision.projector_intermediate_size,
        use_postshuffle_norm=vision.use_postshuffle_norm,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=config.rope_theta,
        mrope_section=tuple(MROPE_SECTION),
        vocab_size=config.vocab_size,
    )


def _hf_weights(config: ArchitectureConfig) -> dict[str, torch.Tensor]:
    """Random weights keyed exactly as the Cosmos3-Edge checkpoint keys them."""
    vision = config.vision
    assert vision is not None
    generator = torch.Generator().manual_seed(1234)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator) * 0.05

    patch_dim = vision.patch_size * vision.patch_size * vision.in_channels
    weights: dict[str, torch.Tensor] = {
        "model.visual.embeddings.patch_embedding.weight": randn(vision.hidden_size, patch_dim),
        "model.visual.embeddings.patch_embedding.bias": randn(vision.hidden_size),
        "model.visual.embeddings.position_embedding.weight": randn(
            vision.num_patches, vision.hidden_size
        ),
        "model.visual.post_layernorm.weight": 1.0 + randn(vision.hidden_size),
        "model.visual.post_layernorm.bias": randn(vision.hidden_size),
        "model.projector.norm.weight": 1.0 + randn(vision.hidden_size),
        "model.projector.norm.bias": randn(vision.hidden_size),
        "model.projector.linear_fc1.weight": randn(
            vision.projector_intermediate_size, vision.hidden_size * MERGE_SIZE**2
        ),
        "model.projector.linear_fc1.bias": randn(vision.projector_intermediate_size),
        "model.projector.linear_fc2.weight": randn(
            config.hidden_size, vision.projector_intermediate_size
        ),
        "model.projector.linear_fc2.bias": randn(config.hidden_size),
        "embed_tokens.weight": randn(config.vocab_size, config.hidden_size),
        "norm.weight": 1.0 + randn(config.hidden_size),
        "lm_head.weight": randn(config.vocab_size, config.hidden_size),
    }
    for layer in range(vision.num_hidden_layers):
        prefix = f"model.visual.encoder.layers.{layer}"
        for norm in ("layer_norm1", "layer_norm2"):
            weights[f"{prefix}.{norm}.weight"] = 1.0 + randn(vision.hidden_size)
            weights[f"{prefix}.{norm}.bias"] = randn(vision.hidden_size)
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            weights[f"{prefix}.self_attn.{proj}.weight"] = randn(
                vision.hidden_size, vision.hidden_size
            )
            weights[f"{prefix}.self_attn.{proj}.bias"] = randn(vision.hidden_size)
        weights[f"{prefix}.mlp.fc1.weight"] = randn(
            vision.intermediate_size, vision.hidden_size
        )
        weights[f"{prefix}.mlp.fc1.bias"] = randn(vision.intermediate_size)
        weights[f"{prefix}.mlp.fc2.weight"] = randn(
            vision.hidden_size, vision.intermediate_size
        )
        weights[f"{prefix}.mlp.fc2.bias"] = randn(vision.hidden_size)

    heads = config.num_attention_heads * config.head_dim
    kv_heads = config.num_key_value_heads * config.head_dim
    for layer in range(config.num_hidden_layers):
        prefix = f"layers.{layer}"
        weights[f"{prefix}.input_layernorm.weight"] = 1.0 + randn(config.hidden_size)
        weights[f"{prefix}.post_attention_layernorm.weight"] = 1.0 + randn(config.hidden_size)
        weights[f"{prefix}.self_attn.to_q.weight"] = randn(heads, config.hidden_size)
        weights[f"{prefix}.self_attn.to_k.weight"] = randn(kv_heads, config.hidden_size)
        weights[f"{prefix}.self_attn.to_v.weight"] = randn(kv_heads, config.hidden_size)
        weights[f"{prefix}.self_attn.to_out.weight"] = randn(config.hidden_size, heads)
        # Generator-tower artifact that the Reasoner must drop.
        weights[f"{prefix}.self_attn.k_norm_und_for_gen.weight"] = randn(config.head_dim)
        weights[f"{prefix}.mlp.up_proj.weight"] = randn(
            config.intermediate_size, config.hidden_size
        )
        weights[f"{prefix}.mlp.down_proj.weight"] = randn(
            config.hidden_size, config.intermediate_size
        )
    return weights


@pytest.fixture(scope="module")
def edge_fixture(tmp_path_factory):
    """Built + weighted ONNX sessions plus the matching reference weights."""
    config = _config()
    weights = _hf_weights(config)
    package = Cosmos3EdgeVLTask().build(Cosmos3EdgeVLModel(config), config)
    package.apply_weights(Cosmos3EdgeVLModel(config).preprocess_weights(weights))
    package.validate_weights()

    directory = tmp_path_factory.mktemp("cosmos3_edge")
    sessions = {}
    for name in ("vision_encoder", "embedding", "decoder"):
        path = directory / f"{name}.onnx"
        ir.save(package[name], str(path))
        sessions[name] = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    reference = {
        key.removeprefix("model."): value.float()
        for key, value in weights.items()
        if "k_norm_und_for_gen" not in key
    }
    return config, _ref_config(config), sessions, reference


def _random_image(height: int, width: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, 3, height, width, generator=generator)


def test_patchify_is_block_major_with_channel_last_values():
    """Pin the processor's packed layout — the tower's weights depend on it.

    ``Cosmos3EdgeImageProcessor.patchify`` emits ``merge x merge`` blocks
    contiguously (block-major) and orders the values inside a patch as
    ``(patch_h, patch_w, channel)``. Reading a patch as ``(C, ph, pw)`` — the
    Conv2d kernel layout — permutes colour into space and scrambles the image.
    """
    patch, merge, channels = 2, 2, 3
    height = width = patch * merge * 2  # 2x2 blocks of 2x2 patches
    # Encode each pixel as a unique value so ordering is fully observable.
    image = torch.arange(channels * height * width, dtype=torch.float32).reshape(
        1, channels, height, width
    )
    packed, grid_h, grid_w = patchify_images(image, patch_size=patch, merge_size=merge)
    packed = packed[0]

    assert (grid_h, grid_w) == (height // patch, width // patch)
    assert packed.shape == (grid_h * grid_w, patch * patch * channels)

    def pixel(channel: int, row: int, col: int) -> float:
        return float(image[0, channel, row, col])

    for block_row in range(grid_h // merge):
        for block_col in range(grid_w // merge):
            for merge_row in range(merge):
                for merge_col in range(merge):
                    # Block-major sequence index.
                    index = (
                        ((block_row * (grid_w // merge)) + block_col) * merge + merge_row
                    ) * merge + merge_col
                    patch_row = (block_row * merge + merge_row) * patch
                    patch_col = (block_col * merge + merge_col) * patch
                    for inner_row in range(patch):
                        for inner_col in range(patch):
                            for channel in range(channels):
                                # Channel-last inside the patch.
                                offset = (inner_row * patch + inner_col) * channels + channel
                                assert packed[index, offset] == pixel(
                                    channel, patch_row + inner_row, patch_col + inner_col
                                )


@pytest.mark.parametrize(
    ("height", "width", "expected"),
    [
        # Already aligned and inside the area bounds -> unchanged.
        (256, 256, (256, 256)),
        # Rounded to the nearest multiple of patch*merge = 32.
        (750, 1000, (736, 992)),
        # Below min_pixels (256*256) -> scaled up.
        (64, 64, (256, 256)),
    ],
)
def test_smart_resize_matches_processor_policy(height, width, expected):
    """Sides are multiples of 32 and the area stays inside the processor bounds.

    Uses the real checkpoint geometry (``patch_size=16``, ``merge_size=2``),
    not this module's tiny-config values, because ``smart_resize`` describes
    ``Cosmos3EdgeImageProcessor``'s policy.
    """
    factor = 16 * 2
    resized = smart_resize(
        height,
        width,
        factor=factor,
        min_pixels=256 * 256,
        max_pixels=4096 * 4096,
    )
    assert resized == expected
    assert resized[0] % factor == 0
    assert resized[1] % factor == 0


@pytest.mark.parametrize(
    ("height", "width"),
    [
        (32, 32),  # grid 4x4 — identical to the learned reference grid
        (64, 64),  # grid 8x8 — upsampled position embeddings
        (16, 96),  # grid 2x12 — height downsampled (antialias path)
        (80, 32),  # grid 10x4 — non-square, both axes resampled
    ],
)
def test_vision_encoder_matches_reference_for_image_grids(edge_fixture, height, width):
    _, ref_config, sessions, reference = edge_fixture
    image = _random_image(height, width, seed=height * 1000 + width)
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

    assert got.shape == (grid_h * grid_w // MERGE_SIZE**2, TEXT_HIDDEN)
    np.testing.assert_allclose(got, expected, atol=1e-4, rtol=1e-4)


def test_vision_encoder_matches_reference_for_video(edge_fixture):
    _, ref_config, sessions, reference = edge_fixture
    frames = 4
    video = torch.stack(
        [_random_image(32, 64, seed=7 + index)[0] for index in range(frames)]
    ).unsqueeze(0)
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

    # One token per merged 2x2 block, per frame.
    assert got.shape == (grid_t * grid_h * grid_w // MERGE_SIZE**2, TEXT_HIDDEN)
    assert got.shape[0] == frames * (grid_h * grid_w // MERGE_SIZE**2)
    np.testing.assert_allclose(got, expected, atol=1e-4, rtol=1e-4)


def test_video_frames_are_encoded_independently(edge_fixture):
    """Attention must not leak across frames (per-frame ``cu_seqlens``)."""
    _, _, sessions, _ = edge_fixture
    frame_a = _random_image(16, 16, seed=3)
    frame_b = _random_image(16, 16, seed=4)
    packed_a, grid_h, grid_w = patchify_images(
        frame_a, patch_size=PATCH_SIZE, merge_size=MERGE_SIZE
    )
    packed_b, _, _ = patchify_images(frame_b, patch_size=PATCH_SIZE, merge_size=MERGE_SIZE)

    single = sessions["vision_encoder"].run(
        None,
        {
            "pixel_values": packed_a[0].numpy(),
            "grid_thw": np.array([1, grid_h, grid_w], dtype=np.int64),
        },
    )[0]
    both = sessions["vision_encoder"].run(
        None,
        {
            "pixel_values": torch.cat([packed_a[0], packed_b[0]]).numpy(),
            "grid_thw": np.array([2, grid_h, grid_w], dtype=np.int64),
        },
    )[0]

    np.testing.assert_allclose(both[: single.shape[0]], single, atol=1e-5, rtol=1e-5)


def test_embedding_and_decoder_match_reference_with_image_and_video(edge_fixture):
    config, ref_config, sessions, reference = edge_fixture

    image = _random_image(32, 32, seed=11)
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
    video = torch.stack(
        [_random_image(16, 16, seed=21 + index)[0] for index in range(frames)]
    ).unsqueeze(0)
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

    ids = [5, 6]
    image_start = len(ids)
    ids += [IMAGE_TOKEN_ID] * image_features.shape[0]
    video_start = len(ids)
    ids += [VIDEO_TOKEN_ID] * video_features.shape[0]
    ids += [7]
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
    np.testing.assert_allclose(inputs_embeds, expected_embeds.numpy(), atol=1e-5)

    # 3D M-RoPE position ids: visual spans get (t, h, w) axes, text is diagonal.
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
    base = base + max(img_h, img_w) // MERGE_SIZE
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
    np.testing.assert_allclose(logits, expected_logits, atol=2e-4, rtol=2e-4)


def test_text_only_decoder_matches_reference(edge_fixture):
    """Text-only inference must stay bit-comparable to the reference."""
    config, ref_config, sessions, reference = edge_fixture
    input_ids = np.array([[3, 9, 12, 40, 2, 61]], dtype=np.int64)
    length = input_ids.shape[1]

    inputs_embeds = sessions["embedding"].run(
        None,
        {
            "input_ids": input_ids,
            "image_features": np.zeros((0, TEXT_HIDDEN), dtype=np.float32),
            "video_features": np.zeros((0, TEXT_HIDDEN), dtype=np.float32),
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
    np.testing.assert_allclose(logits, expected, atol=2e-4, rtol=2e-4)


def test_decoder_kv_cache_decode_step_matches_full_prefill(edge_fixture):
    """ORT generation smoke: one cached decode step equals a full re-prefill."""
    config, _, sessions, reference = edge_fixture
    input_ids = np.array([[3, 9, 12, 40]], dtype=np.int64)
    empty_features = np.zeros((0, TEXT_HIDDEN), dtype=np.float32)

    def embed(ids: np.ndarray) -> np.ndarray:
        return sessions["embedding"].run(
            None,
            {
                "input_ids": ids,
                "image_features": empty_features,
                "video_features": empty_features,
            },
        )[0]

    length = input_ids.shape[1]
    feeds = {
        "inputs_embeds": embed(input_ids),
        "attention_mask": np.ones((1, length), dtype=np.int64),
        "position_ids": np.tile(np.arange(length, dtype=np.int64), (3, 1, 1)),
    }
    empty = np.zeros((1, config.num_key_value_heads, 0, config.head_dim), dtype=np.float32)
    for layer in range(config.num_hidden_layers):
        feeds[f"past_key_values.{layer}.key"] = empty
        feeds[f"past_key_values.{layer}.value"] = empty
    outputs = sessions["decoder"].run(None, feeds)
    names = [value.name for value in sessions["decoder"].get_outputs()]
    present = dict(zip(names, outputs, strict=True))
    next_token = int(present["logits"][0, -1].argmax())

    step_ids = np.array([[next_token]], dtype=np.int64)
    feeds = {
        "inputs_embeds": embed(step_ids),
        "attention_mask": np.ones((1, length + 1), dtype=np.int64),
        "position_ids": np.full((3, 1, 1), length, dtype=np.int64),
    }
    for layer in range(config.num_hidden_layers):
        feeds[f"past_key_values.{layer}.key"] = present[f"present.{layer}.key"]
        feeds[f"past_key_values.{layer}.value"] = present[f"present.{layer}.value"]
    cached_logits = sessions["decoder"].run(["logits"], feeds)[0]

    full_ids = np.concatenate([input_ids, step_ids], axis=1)
    feeds = {
        "inputs_embeds": embed(full_ids),
        "attention_mask": np.ones((1, length + 1), dtype=np.int64),
        "position_ids": np.tile(np.arange(length + 1, dtype=np.int64), (3, 1, 1)),
    }
    for layer in range(config.num_hidden_layers):
        feeds[f"past_key_values.{layer}.key"] = empty
        feeds[f"past_key_values.{layer}.value"] = empty
    full_logits = sessions["decoder"].run(["logits"], feeds)[0]

    np.testing.assert_allclose(cached_logits[0, 0], full_logits[0, -1], atol=2e-4, rtol=2e-4)
    assert reference  # reference weights were used to build the graphs
