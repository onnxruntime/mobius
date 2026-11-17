# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Host bridge tests, plus opt-in execution of the pinned official processor."""

from __future__ import annotations

import ast
import importlib.util
import math
import os
import pathlib
import re
import sys
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest
import torch

from mobius.integrations.transformers._nemotron_h_omni_processing import (
    OMNI_PROCESSING_SOURCE_REVISION,
    OmniVisionCall,
    compact_omni_prompt,
    run_omni_vision_calls,
)


def _call(positions, *, video=False, count=1):
    pixels = np.arange(count * 3 * 8 * 8, dtype=np.float32).reshape(count, 3, 8, 8) / 97 - 2
    return OmniVisionCall(
        pixel_values=pixels,
        placeholder_positions=np.asarray(positions, dtype=np.int64),
        is_video=video,
        patch_size=2,
    )


def _result(call, mask=None, *, offset=0, dtype=np.float32):
    if mask is None:
        mask = np.ones(call.dense_token_count, dtype=np.bool_)
    mask = np.asarray(mask, dtype=np.bool_)
    dense = (np.arange(call.dense_token_count * 3).reshape(-1, 3) + offset + 1).astype(dtype)
    return {
        "image_features": dense[mask],
        "retention_mask": mask,
        "kept_indices": np.flatnonzero(mask),
    }


def _ids(positions, length=25):
    ids = np.arange(100, 100 + length, dtype=np.int64)[None, :]
    ids.reshape(-1)[positions] = 18
    return ids


def test_pixels_are_not_normalized_repeated_or_packed_again():
    call = _call(range(12), video=True, count=5)
    seen = []

    def runner(feeds):
        seen.append(feeds)
        assert feeds["pixel_values"] is call.pixel_values
        assert feeds["pixel_values"].shape == (5, 3, 8, 8)
        assert feeds["is_video"].shape == ()
        assert feeds["is_video"].dtype == np.bool_
        assert feeds["is_video"].item() is True
        return _result(call)

    outputs = run_omni_vision_calls([call], runner)
    assert len(outputs) == len(seen) == 1
    assert call.dense_token_count == 12
    np.testing.assert_array_equal(
        call.pixel_values.reshape(-1), np.arange(960, dtype=np.float32) / np.float32(97) - 2
    )
    renamed = call.feeds(pixel_input_name="pixels", mode_input_name="mode")
    assert set(renamed) == {"pixels", "mode"}


def test_source_video_compaction_preserves_timestamp_text_and_empty_tubelet_delimiters():
    positions = np.array([2, 3, 4, 5, 9, 10, 11, 12, 16, 17, 18, 19])
    call = _call(positions, video=True, count=5)
    # Drop the complete second tubelet: its label, punctuation and delimiters
    # still remain, exactly as source retention_mask[input_ids == 18] = evs_mask.
    mask = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0], dtype=np.bool_)
    output = _result(call, mask)
    ids = _ids(positions)
    attention = np.ones_like(ids)
    attention[0, -2:] = 0
    original_ids = ids.copy()
    result = compact_omni_prompt(ids, attention, [call], [output], hidden_size=3)
    reference_keep = torch.ones_like(torch.from_numpy(ids), dtype=torch.bool)
    reference_keep[torch.from_numpy(ids) == 18] = torch.from_numpy(mask)
    expected_ids = torch.from_numpy(ids)[reference_keep].unsqueeze(0).numpy()
    np.testing.assert_array_equal(result.input_ids, expected_ids)
    np.testing.assert_array_equal(
        result.attention_mask, attention.reshape(-1)[reference_keep.numpy().reshape(-1)][None]
    )
    np.testing.assert_array_equal(result.image_features, output["image_features"])
    np.testing.assert_array_equal(
        result.kept_token_indices, np.flatnonzero(reference_keep.numpy())
    )
    np.testing.assert_array_equal(ids, original_ids)
    np.testing.assert_array_equal(result.input_ids[result.input_ids != 18], ids[ids != 18])


def test_image_only_batch_keeps_global_feature_offsets_and_prompt_shape():
    positions = [1, 2, 3, 4, 9, 10, 11, 12]
    ids = _ids(positions, length=16).reshape(2, 8)
    call = _call(positions, count=2)
    output = _result(call)
    result = compact_omni_prompt(ids, None, [call], [output], hidden_size=3)
    np.testing.assert_array_equal(result.input_ids, ids)
    np.testing.assert_array_equal(result.image_features, output["image_features"])
    assert result.input_ids.shape == (2, 8)
    assert result.attention_mask is None
    assert result.token_retention_mask.all()
    np.testing.assert_array_equal(result.image_features[4], [13, 14, 15])


def test_mixed_mode_is_explicit_and_features_follow_prompt_not_invocation_order():
    video_positions = np.array([1, 2, 3, 4, 7, 8, 9, 10])
    image_positions = np.array([14, 15, 16, 17])
    ids = _ids(np.concatenate([video_positions, image_positions]))
    video = _call(video_positions, video=True, count=3)
    image = _call(image_positions)
    video_output = _result(video, [1, 1, 1, 1, 0, 1, 0, 1], offset=100)
    image_output = _result(image, offset=900)
    # Calls are image first, but the source-expanded prompt places video first.
    calls, outputs = [image, video], [image_output, video_output]
    with pytest.raises(NotImplementedError, match="mixed image/video"):
        compact_omni_prompt(ids, None, calls, outputs, hidden_size=3)
    result = compact_omni_prompt(
        ids, None, calls, outputs, hidden_size=3, allow_provenance_mixed=True
    )
    np.testing.assert_array_equal(
        result.image_features,
        np.concatenate([video_output["image_features"], image_output["image_features"]]),
    )
    np.testing.assert_array_equal(result.input_ids[result.input_ids != 18], ids[ids != 18])


def test_multiple_clips_require_provenance_extension_and_keep_separate_first_groups():
    first = _call(range(8), video=True, count=3)
    second = _call(range(10, 18), video=True, count=3)
    ids = _ids(list(range(8)) + list(range(10, 18)))
    outputs = [_result(first, [1, 1, 1, 1, 0, 0, 0, 0]), _result(second)]
    with pytest.raises(NotImplementedError, match="multiple clips"):
        compact_omni_prompt(ids, None, [first, second], outputs, hidden_size=3)
    result = compact_omni_prompt(
        ids, None, [first, second], outputs, hidden_size=3, allow_provenance_mixed=True
    )
    assert result.image_features.shape == (12, 3)
    assert result.token_retention_mask.reshape(-1)[10:14].all()


@pytest.mark.parametrize("cached", [False, True])
def test_zero_new_media_and_cached_decode(cached):
    ids = np.array([[18 if cached else 7]], dtype=np.int64)
    attention = np.ones_like(ids)

    def forbidden(_feeds):
        raise AssertionError("absent media must not run a vision graph")

    outputs = run_omni_vision_calls([], forbidden)
    result = compact_omni_prompt(
        ids, attention, [], outputs, hidden_size=4096, cached_decode=cached
    )
    np.testing.assert_array_equal(result.input_ids, ids)
    np.testing.assert_array_equal(result.attention_mask, attention)
    assert result.image_features.shape == (0, 4096)
    assert result.image_features.dtype == np.float32


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("mask_dtype", [np.int64, np.bool_])
def test_cached_decode_preserves_past_plus_current_mask(batch_size, mask_dtype):
    ids = np.full((batch_size, 1), 18, dtype=np.int64)
    attention = np.ones((batch_size, 6), dtype=mask_dtype)
    attention[:, 1:3] = 0
    original = attention.copy()
    result = compact_omni_prompt(ids, attention, [], [], hidden_size=4096, cached_decode=True)
    np.testing.assert_array_equal(result.input_ids, ids)
    np.testing.assert_array_equal(result.attention_mask, original)
    np.testing.assert_array_equal(attention, original)
    assert result.image_features.shape == (0, 4096)
    np.testing.assert_array_equal(result.token_retention_mask, np.ones_like(ids, dtype=bool))


@pytest.mark.parametrize("shape", [(6,), (2, 6), (1, 0)])
def test_cached_decode_rejects_misaligned_attention_mask(shape):
    with pytest.raises(ValueError, match="attention_mask"):
        compact_omni_prompt(
            np.array([[7]], dtype=np.int64),
            np.ones(shape, dtype=np.int64),
            [],
            [],
            hidden_size=4096,
            cached_decode=True,
        )


def test_prefill_rejects_past_plus_current_mask():
    with pytest.raises(ValueError, match="attention_mask"):
        compact_omni_prompt(
            np.array([[7]], dtype=np.int64),
            np.ones((1, 6), dtype=np.int64),
            [],
            [],
            hidden_size=4096,
        )


@pytest.mark.parametrize(
    "mutation", ["float64", "rank5", "packed", "zero", "unaligned", "nan"]
)
def test_malformed_pixels_fail_closed(mutation):
    pixels = _call(range(4)).pixel_values
    if mutation == "float64":
        pixels = pixels.astype(np.float64)
    elif mutation == "rank5":
        pixels = pixels[:, None]
    elif mutation == "packed":
        pixels = np.concatenate([pixels, pixels], axis=1)
    elif mutation == "zero":
        pixels = pixels[:0]
    elif mutation == "unaligned":
        pixels = pixels[:, :, :-1]
    else:
        pixels = pixels.copy()
        pixels[0, 0, 0, 0] = np.nan
    with pytest.raises((ValueError, TypeError)):
        OmniVisionCall(
            pixel_values=pixels,
            placeholder_positions=np.arange(4),
            is_video=False,
            patch_size=2,
        )


@pytest.mark.parametrize(
    "positions",
    [
        np.arange(3),
        np.arange(4, dtype=np.int32),
        np.arange(4).reshape(2, 2),
        np.array([0, 1, 1, 2]),
        np.array([0, 2, 1, 3]),
        np.array([-1, 0, 1, 2]),
    ],
)
def test_malformed_call_positions(positions):
    with pytest.raises((ValueError, TypeError)):
        OmniVisionCall(
            pixel_values=_call(range(4)).pixel_values,
            placeholder_positions=positions,
            is_video=False,
            patch_size=2,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_output",
        "mask_dtype",
        "mask_count",
        "indices_dtype",
        "indices_shape",
        "indices_reverse",
        "indices_duplicate",
        "indices_outside",
        "indices_disagree",
        "feature_count",
        "feature_width",
        "feature_dtype",
        "feature_nan",
        "first_group",
    ],
)
def test_malformed_graph_results_fail_closed(mutation):
    call = _call(range(8), video=True, count=3)
    output = _result(call)
    if mutation == "missing_output":
        del output["kept_indices"]
    elif mutation == "mask_dtype":
        output["retention_mask"] = output["retention_mask"].astype(np.int64)
    elif mutation == "mask_count":
        output["retention_mask"] = output["retention_mask"][:-1]
    elif mutation == "indices_dtype":
        output["kept_indices"] = output["kept_indices"].astype(np.int32)
    elif mutation == "indices_shape":
        output["kept_indices"] = output["kept_indices"][None]
    elif mutation == "indices_reverse":
        output["kept_indices"] = output["kept_indices"][::-1]
    elif mutation == "indices_duplicate":
        output["kept_indices"][1] = 0
    elif mutation == "indices_outside":
        output["kept_indices"][-1] = 100
    elif mutation == "indices_disagree":
        output["kept_indices"] = output["kept_indices"][:-1]
    elif mutation == "feature_count":
        output["image_features"] = output["image_features"][:-1]
    elif mutation == "feature_width":
        output["image_features"] = output["image_features"][:, :2]
    elif mutation == "feature_dtype":
        output["image_features"] = output["image_features"].astype(np.float64)
    elif mutation == "feature_nan":
        output["image_features"][0, 0] = np.nan
    else:
        output = _result(call, [0, 1, 1, 1, 1, 1, 1, 1])
    with pytest.raises((ValueError, TypeError)):
        compact_omni_prompt(_ids(range(8)), None, [call], [output], hidden_size=3)


def test_binding_count_bounds_overlap_and_wrong_tokens():
    call = _call(range(4))
    output = _result(call)
    with pytest.raises(ValueError, match="exactly one output"):
        compact_omni_prompt(_ids(range(4)), None, [call], [], hidden_size=3)
    with pytest.raises(ValueError, match="overlapping"):
        compact_omni_prompt(
            _ids(range(4)), None, [call, call], [output, output], hidden_size=3
        )
    with pytest.raises(ValueError, match="outside"):
        compact_omni_prompt(_ids(range(3), length=3), None, [call], [output], hidden_size=3)
    with pytest.raises(ValueError, match="select"):
        compact_omni_prompt(_ids(range(3)), None, [call], [output], hidden_size=3)
    with pytest.raises(ValueError, match="every"):
        compact_omni_prompt(_ids(range(5)), None, [call], [output], hidden_size=3)
    with pytest.raises(ValueError, match="every"):
        compact_omni_prompt(np.array([[18]], np.int64), None, [], [], hidden_size=3)


def test_source_image_pruning_and_video_batching_are_rejected():
    image = _call(range(4))
    with pytest.raises(ValueError, match="must not prune"):
        compact_omni_prompt(
            _ids(range(4)), None, [image], [_result(image, [1, 0, 1, 1])], hidden_size=3
        )
    video = _call(range(8), video=True, count=3)
    ids = _ids(range(8), length=16).reshape(2, 8)
    with pytest.raises(ValueError, match="batch size one"):
        compact_omni_prompt(ids, None, [video], [_result(video)], hidden_size=3)
    with pytest.raises(ValueError, match="cached_decode"):
        compact_omni_prompt(
            _ids(range(4)), None, [image], [_result(image)], hidden_size=3, cached_decode=True
        )


@pytest.mark.parametrize("dtype", [np.float16, np.float32, ml_dtypes.bfloat16])
def test_features_preserve_declared_dtype(dtype):
    call = _call(range(4))
    output = _result(call, dtype=dtype)
    result = compact_omni_prompt(
        _ids(range(4)), None, [call], [output], hidden_size=3, feature_dtype=dtype
    )
    assert result.image_features.dtype == dtype
    np.testing.assert_array_equal(result.image_features, output["image_features"])


@pytest.mark.parametrize(
    "mutation",
    ["ids_dtype", "ids_rank", "attention_shape", "attention_dtype", "attention_values"],
)
def test_malformed_prompt_arrays(mutation):
    ids = np.array([[7, 8]], dtype=np.int64)
    attention = np.ones_like(ids)
    if mutation == "ids_dtype":
        ids = ids.astype(np.int32)
    elif mutation == "ids_rank":
        ids = ids.reshape(-1)
    elif mutation == "attention_shape":
        attention = attention[:, :1]
    elif mutation == "attention_dtype":
        attention = attention.astype(np.float32)
    else:
        attention[0, 0] = 2
    with pytest.raises((ValueError, TypeError)):
        compact_omni_prompt(ids, attention, [], [], hidden_size=3)


class _Tokenizer:
    def __init__(self):
        self.init_kwargs = {}

    def __call__(self, texts, **_kwargs):
        self.texts = texts
        sequences = []
        special = {"<image>": 18, "<img>": 16, "</img>": 17}
        for text in texts:
            pieces = re.findall(r"<image>|<img>|</img>|.", text, flags=re.DOTALL)
            sequences.append(
                [special[piece] if piece in special else ord(piece) + 100 for piece in pieces]
            )
        length = max(map(len, sequences))
        return {
            "input_ids": [row + [0] * (length - len(row)) for row in sequences],
            "attention_mask": [
                [1] * len(row) + [0] * (length - len(row)) for row in sequences
            ],
        }


def _load_source(path, name, monkeypatch):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    monkeypatch.setitem(sys.modules, name, module)
    specification.loader.exec_module(module)
    return module


class _SourceImage(np.ndarray):
    """Array fixture supplying the original image dimensions used by source."""

    @property
    def width(self):
        return self.shape[1]

    @property
    def height(self):
        return self.shape[0]


@pytest.mark.skipif(
    not os.environ.get("MOBIUS_OMNI_SOURCE_DIR"),
    reason="Set MOBIUS_OMNI_SOURCE_DIR to the pinned local source checkout",
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_pinned_official_processor_and_evs_compaction(monkeypatch, device):
    """Execute official processor/preprocessing/EVS bodies, without a model."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA reference requires a GPU")
    assert OMNI_PROCESSING_SOURCE_REVISION == "c6cc2e8ffa353403213d6b9b18aaf8eb684d705a"
    root = pathlib.Path(os.environ["MOBIUS_OMNI_SOURCE_DIR"])
    processor_module = _load_source(
        root / "processing_nemotron_h_omni.py", "_omni_pinned_processor", monkeypatch
    )
    # Execute exact image-processor methods without installing Pillow or the
    # unused TorchvisionBackend. Only the source's normalization/resize path
    # and target-size methods are needed for these NumPy image fixtures.
    from transformers.feature_extraction_utils import BatchFeature
    from transformers.image_utils import make_list_of_images

    image_tree = ast.parse((root / "image_processing_nemotron_h_omni.py").read_text())
    image_class = next(node for node in image_tree.body if isinstance(node, ast.ClassDef))
    image_namespace = {
        "TorchvisionBackend": object,
        "ImageInput": object,
        "BatchFeature": BatchFeature,
        "np": np,
        "torch": torch,
        "math": math,
        "make_list_of_images": make_list_of_images,
    }
    exec(
        compile(ast.Module(body=[image_class], type_ignores=[]), "pinned_omni_images", "exec"),
        image_namespace,
    )
    image_processor = image_namespace["NemotronH_Omni_Reasoning_V3ImageProcessor"]()
    image_processor.__dict__.update(
        norm_mean=[0.48145466, 0.4578275, 0.40821073],
        norm_std=[0.26862954, 0.26130258, 0.27577711],
        patch_size=2,
        _downsample_factor=2,
        min_num_patches=16,
        max_num_patches=16,
        max_model_len=64,
        video_target_num_patches=16,
        video_maintain_aspect_ratio=False,
    )

    class SourceImages:
        _is_video_mode = False

        def __call__(self, *, images, **_kwargs):
            image_processor._is_video_mode = self._is_video_mode
            return image_processor._preprocess(images, return_tensors="np")

    tokenizer = _Tokenizer()
    metadata = SimpleNamespace(fps=29.97, frames_indices=[2, 5, 9, 11, 15])
    facade = SimpleNamespace(
        tokenizer=tokenizer,
        image_processor=SourceImages(),
        image_token="<image>",
        video_token="<video>",
        image_start_token="<img>",
        image_end_token="</img>",
        video_temporal_patch_dim=2,
        _merge_kwargs=lambda *_args, **_kwargs: {
            "images_kwargs": {},
            "videos_kwargs": {"video_metadata": metadata},
            "audio_kwargs": {},
            "text_kwargs": {"return_tensors": "np"},
        },
    )
    # Only fixture adapters are synthetic. All resizing/normalization, prompt
    # expansion, timestamp rounding, and EVS execute the pinned source bodies.
    images = [
        np.fromfunction(lambda h, w, c, i=i: (23 * i + 7 * h + 3 * w + c) % 256, (8, 8, 3))
        .astype(np.uint8)
        .view(_SourceImage)
        for i in range(5)
    ]
    prepared = processor_module.NemotronH_Omni_Reasoning_V3Processor.__call__(
        facade, videos=images, text="before <video> after"
    )
    assert (
        "Frame 1 sampled at 0.07 seconds and frame 2 sampled at 0.17 seconds"
        in tokenizer.texts[0]
    )
    assert "Frame 5 sampled at 0.49 seconds" in tokenizer.texts[0]
    assert "frame 6" not in tokenizer.texts[0]
    pixels = prepared["pixel_values_videos"]
    assert pixels.shape == (5, 3, 8, 8) and pixels.dtype == np.float32
    ids = prepared["input_ids"]
    positions = np.flatnonzero(ids.reshape(-1) == 18)
    call = OmniVisionCall(
        pixel_values=pixels, placeholder_positions=positions, is_video=True, patch_size=2
    )
    assert call.dense_token_count == positions.size == 12
    np.testing.assert_allclose(
        pixels[0, :, 0, 0],
        (
            np.asarray(images[0])[0, 0] / np.float32(255)
            - np.array(image_processor.norm_mean, np.float32)
        )
        / np.array(image_processor.norm_std, np.float32),
        rtol=1e-6,
    )
    parsed = ast.parse((root / "modeling_nemotron_h_omni.py").read_text())
    function = next(
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name == "compute_retention_mask"
    )
    namespace = {"torch": torch}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "pinned_omni_evs", "exec"),
        namespace,
    )
    features = torch.tensor(
        [
            [[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]],
            [[-1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]],
            [[-1, 0, 0], [0, -1, 0], [1, -1, 0], [0, 0, 1]],
        ],
        dtype=torch.float32,
        device=device,
    )
    mask = namespace["compute_retention_mask"](
        video_embeds=features, thw=(3, 2, 2), spatial_merge_size=1, q=0.5
    )
    output = {
        "image_features": features.reshape(-1, 3)[mask].cpu().numpy(),
        "retention_mask": mask.cpu().numpy(),
        "kept_indices": mask.nonzero().flatten().cpu().numpy(),
    }
    result = compact_omni_prompt(
        ids, prepared["attention_mask"], [call], [output], hidden_size=3
    )
    keep = torch.ones_like(torch.from_numpy(ids), dtype=torch.bool)
    keep[torch.from_numpy(ids) == 18] = mask.cpu()
    np.testing.assert_array_equal(
        result.input_ids, torch.from_numpy(ids)[keep].unsqueeze(0).numpy()
    )
    np.testing.assert_array_equal(
        result.attention_mask,
        torch.from_numpy(prepared["attention_mask"])[keep].unsqueeze(0).numpy(),
    )
    np.testing.assert_array_equal(result.input_ids[result.input_ids != 18], ids[ids != 18])
    np.testing.assert_array_equal(result.image_features, output["image_features"])
    image_prepared = processor_module.NemotronH_Omni_Reasoning_V3Processor.__call__(
        facade, images=images[:2], text="alpha<image>between<image>omega"
    )
    image_ids = image_prepared["input_ids"]
    image_positions = np.flatnonzero(image_ids.reshape(-1) == 18)
    image_call = OmniVisionCall(
        pixel_values=image_prepared["pixel_values"],
        placeholder_positions=image_positions,
        is_video=False,
        patch_size=2,
    )
    image_output = _result(image_call, offset=500)
    image_result = compact_omni_prompt(
        image_ids,
        image_prepared["attention_mask"],
        [image_call],
        [image_output],
        hidden_size=3,
    )
    source_embeddings = torch.arange(image_ids.size * 3, dtype=torch.float32).reshape(1, -1, 3)
    image_mask = (torch.from_numpy(image_ids) == 18).unsqueeze(-1).expand_as(source_embeddings)
    source_embeddings = source_embeddings.masked_scatter(
        image_mask, torch.from_numpy(image_output["image_features"])
    )
    bridged = torch.arange(image_ids.size * 3, dtype=torch.float32).reshape(1, -1, 3)
    bridged[torch.from_numpy(image_result.input_ids) == 18] = torch.from_numpy(
        image_result.image_features
    )
    np.testing.assert_array_equal(bridged.numpy(), source_embeddings.numpy())

    mixed = processor_module.NemotronH_Omni_Reasoning_V3Processor.__call__(
        facade,
        images=images[:1],
        videos=images,
        text="video first <video> then image <image> end",
    )
    mixed_positions = np.flatnonzero(mixed["input_ids"].reshape(-1) == 18)
    assert mixed_positions.size == 16
    mixed_image = OmniVisionCall(
        pixel_values=mixed["pixel_values"],
        placeholder_positions=mixed_positions[12:],
        is_video=False,
        patch_size=2,
    )
    mixed_video = OmniVisionCall(
        pixel_values=mixed["pixel_values_videos"],
        placeholder_positions=mixed_positions[:12],
        is_video=True,
        patch_size=2,
    )
    mixed_image_output = _result(mixed_image, offset=900)
    # This is the official forward's first image scatter expression: it fails
    # because both media streams use ID 18 but only image features are supplied.
    shared_mask = (torch.from_numpy(mixed["input_ids"]) == 18).unsqueeze(-1).expand(-1, -1, 3)
    with pytest.raises(RuntimeError, match="source"):
        torch.zeros(shared_mask.shape).masked_scatter(
            shared_mask, torch.from_numpy(mixed_image_output["image_features"])
        )
    with pytest.raises(NotImplementedError, match="mixed image/video"):
        compact_omni_prompt(
            mixed["input_ids"],
            mixed["attention_mask"],
            [mixed_image, mixed_video],
            [mixed_image_output, output],
            hidden_size=3,
        )
    mixed_result = compact_omni_prompt(
        mixed["input_ids"],
        mixed["attention_mask"],
        [mixed_image, mixed_video],
        [mixed_image_output, output],
        hidden_size=3,
        allow_provenance_mixed=True,
    )
    np.testing.assert_array_equal(
        mixed_result.image_features,
        np.concatenate([output["image_features"], mixed_image_output["image_features"]]),
    )
    np.testing.assert_array_equal(
        mixed_result.input_ids[mixed_result.input_ids != 18],
        mixed["input_ids"][mixed["input_ids"] != 18],
    )
