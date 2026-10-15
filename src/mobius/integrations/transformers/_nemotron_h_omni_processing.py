# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Host bridge for normalized Omni media and source-position EVS compaction.

Source pin: ``c6cc2e8ffa353403213d6b9b18aaf8eb684d705a``, particularly
``processing_nemotron_h_omni.py`` and ``modeling_nemotron_h_omni.py``. Images and
video tubelets both use token 18. Modality provenance must therefore come from
the caller, not from guessing which occurrence of that ID belongs to a video.

The official mixed image+video forward scatters both streams over every image
placeholder and cannot correctly represent their separate counts. Mixed mode
here requires explicit opt-in to a provenance-defined extension; it is not a
claim of parity with that unsupported official forward.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import DTypeLike

OMNI_PROCESSING_SOURCE_REVISION = "c6cc2e8ffa353403213d6b9b18aaf8eb684d705a"
_FEATURE_DTYPES = frozenset({"float16", "float32", "bfloat16"})


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, kw_only=True)
class OmniVisionCall:
    """One image batch or one video clip, with explicit dense prompt positions.

    ``pixel_values`` is already-normalized float32 NCHW. Video N is the number
    of real frames, not tubelets: the graph owns odd-frame repetition and
    channel packing. Positions are int64, ascending, flattened row-major token
    positions, one per *unpruned projected feature*. They may span timestamp
    text between tubelets and need not form a contiguous token interval.
    """

    pixel_values: np.ndarray
    placeholder_positions: np.ndarray
    is_video: bool
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.is_video, bool):
            raise TypeError("is_video must be a Python bool")
        for name in ("patch_size", "spatial_merge_size", "temporal_patch_size"):
            _positive_integer(getattr(self, name), name)
        pixels = self.pixel_values
        if not isinstance(pixels, np.ndarray) or pixels.dtype != np.float32:
            raise TypeError("pixel_values must be an already-normalized float32 ndarray")
        if pixels.ndim != 4 or pixels.shape[1] != 3 or min(pixels.shape) <= 0:
            raise ValueError("pixel_values must be nonempty NCHW with three channels")
        divisor = self.patch_size * self.spatial_merge_size
        if pixels.shape[2] % divisor or pixels.shape[3] % divisor:
            raise ValueError("pixel H/W must be multiples of patch_size * spatial_merge_size")
        if not np.isfinite(pixels).all():
            raise ValueError("pixel_values must be finite")
        positions = self.placeholder_positions
        if not isinstance(positions, np.ndarray) or positions.dtype != np.int64:
            raise TypeError("placeholder_positions must be an int64 ndarray")
        if positions.ndim != 1 or positions.size != self.dense_token_count:
            raise ValueError("placeholder count does not match dense projected media tokens")
        if np.any(positions < 0) or np.any(positions[1:] <= positions[:-1]):
            raise ValueError(
                "placeholder_positions must be nonnegative and strictly increasing"
            )
        positions = positions.copy()
        positions.flags.writeable = False
        object.__setattr__(self, "placeholder_positions", positions)
        object.__setattr__(self, "pixel_values", np.ascontiguousarray(pixels))

    @property
    def tokens_per_group(self) -> int:
        divisor = self.patch_size * self.spatial_merge_size
        return (self.pixel_values.shape[2] // divisor) * (
            self.pixel_values.shape[3] // divisor
        )

    @property
    def dense_token_count(self) -> int:
        count = self.pixel_values.shape[0]
        if self.is_video:
            count = (count + self.temporal_patch_size - 1) // self.temporal_patch_size
        return count * self.tokens_per_group

    def feeds(
        self,
        *,
        pixel_input_name: str = "pixel_values",
        mode_input_name: str = "is_video",
    ) -> dict[str, np.ndarray]:
        """Return unmodified NCHW pixels and a rank-zero bool dispatch input."""
        if not pixel_input_name or not mode_input_name or pixel_input_name == mode_input_name:
            raise ValueError("vision input names must be distinct nonempty strings")
        return {
            pixel_input_name: self.pixel_values,
            mode_input_name: np.asarray(self.is_video, dtype=np.bool_),
        }


@dataclass(frozen=True)
class OmniPromptInputs:
    """Compacted prompt plus features ordered for its surviving token-18 slots."""

    input_ids: np.ndarray
    attention_mask: np.ndarray | None
    image_features: np.ndarray
    token_retention_mask: np.ndarray
    kept_token_indices: np.ndarray


def run_omni_vision_calls(
    calls: Sequence[OmniVisionCall],
    run_vision: Callable[[Mapping[str, np.ndarray]], Mapping[str, np.ndarray]],
    *,
    pixel_input_name: str = "pixel_values",
    mode_input_name: str = "is_video",
) -> tuple[Mapping[str, np.ndarray], ...]:
    """Invoke the graph once per call, preserving call/result pairing.

    An empty sequence invokes nothing, including on cached decode. The callback
    must return image_features, retention_mask, and kept_indices by name.
    """
    return tuple(
        run_vision(
            call.feeds(pixel_input_name=pixel_input_name, mode_input_name=mode_input_name)
        )
        for call in calls
    )


def _validate_prompt(
    input_ids: np.ndarray,
    attention_mask: np.ndarray | None,
    calls: Sequence[OmniVisionCall],
    image_token_id: int,
    cached_decode: bool,
    allow_provenance_mixed: bool,
) -> None:
    if not isinstance(input_ids, np.ndarray) or input_ids.dtype != np.int64:
        raise TypeError("input_ids must be an int64 ndarray")
    if input_ids.ndim != 2 or input_ids.shape[0] <= 0:
        raise ValueError("input_ids must have shape (batch, sequence)")
    if attention_mask is not None:
        if (
            not isinstance(attention_mask, np.ndarray)
            or attention_mask.shape != input_ids.shape
        ):
            raise ValueError("attention_mask must match input_ids shape")
        if attention_mask.dtype not in (np.dtype(np.int64), np.dtype(np.bool_)):
            raise TypeError("attention_mask must be int64 or bool")
        if not np.isin(attention_mask, [0, 1]).all():
            raise ValueError("attention_mask must contain only zero or one")
    if not isinstance(cached_decode, bool) or not isinstance(allow_provenance_mixed, bool):
        raise TypeError("cached_decode and allow_provenance_mixed must be bool")
    if isinstance(image_token_id, bool) or not isinstance(image_token_id, int):
        raise TypeError("image_token_id must be an integer")
    if cached_decode and calls:
        raise ValueError("cached_decode must not submit new media calls")
    if not all(isinstance(call, OmniVisionCall) for call in calls):
        raise TypeError("calls must contain OmniVisionCall objects")
    video_count = sum(call.is_video for call in calls)
    if video_count:
        if input_ids.shape[0] != 1:
            raise ValueError(
                "The pinned processor supports video only with prompt batch size one"
            )
        if (video_count != len(calls) or video_count > 1) and not allow_provenance_mixed:
            raise NotImplementedError(
                "Pinned HF supports one video clip, not mixed image/video or multiple clips; "
                "both streams scatter onto all token-18 positions. Use "
                "allow_provenance_mixed=True only for the explicit host extension"
            )
    flat = input_ids.reshape(-1)
    bound = np.zeros(flat.size, dtype=np.bool_)
    for call in calls:
        positions = call.placeholder_positions
        if positions[-1] >= flat.size:
            raise ValueError("placeholder position is outside the prompt")
        if bound[positions].any():
            raise ValueError("media calls have overlapping placeholder positions")
        if not np.all(flat[positions] == image_token_id):
            raise ValueError("media positions must select the source image placeholder token")
        bound[positions] = True
    # With no new media, cached decode is ordinary token embedding even if the
    # emitted token happens to have ID 18, as in the source's absent-media branch.
    if not cached_decode and not np.array_equal(bound, flat == image_token_id):
        raise ValueError("every image/video placeholder must have exactly one media binding")


def _validate_result(
    call: OmniVisionCall,
    result: Mapping[str, np.ndarray],
    hidden_size: int,
    feature_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {"image_features", "retention_mask", "kept_indices"}
    if not isinstance(result, Mapping) or not required <= result.keys():
        raise ValueError(
            "vision result must contain image_features, retention_mask, kept_indices"
        )
    features = result["image_features"]
    mask = result["retention_mask"]
    indices = result["kept_indices"]
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
        raise TypeError("retention_mask must be a bool ndarray")
    if mask.shape != (call.dense_token_count,):
        raise ValueError("retention_mask length must match dense projected token count")
    if not isinstance(indices, np.ndarray) or indices.dtype != np.int64 or indices.ndim != 1:
        raise TypeError("kept_indices must be a rank-one int64 ndarray")
    if not np.array_equal(indices, np.flatnonzero(mask)):
        raise ValueError("kept_indices must equal ascending nonzero retention-mask positions")
    if not isinstance(features, np.ndarray) or features.dtype != feature_dtype:
        raise TypeError("image_features dtype must match feature_dtype")
    if features.shape != (indices.size, hidden_size):
        raise ValueError("image_features shape must match retained count and hidden_size")
    if not np.isfinite(features).all():
        raise ValueError("image_features must be finite")
    if call.is_video:
        if not mask[: call.tokens_per_group].all():
            raise ValueError("video retention must preserve every token of the first tubelet")
    elif not mask.all():
        raise ValueError("image calls must not prune image tokens")
    return features, mask, indices


def compact_omni_prompt(
    input_ids: np.ndarray,
    attention_mask: np.ndarray | None,
    calls: Sequence[OmniVisionCall],
    vision_outputs: Sequence[Mapping[str, np.ndarray]],
    *,
    hidden_size: int,
    image_token_id: int = 18,
    feature_dtype: DTypeLike = np.float32,
    cached_decode: bool = False,
    allow_provenance_mixed: bool = False,
) -> OmniPromptInputs:
    """Validate graph metadata, compact matching prompt slots, and order features.

    Timestamp labels, newlines, image delimiters, text, and padding positions
    remain untouched. Only explicitly bound video placeholder tokens can drop.
    Image-only batches retain their original shape; video requires B=1, matching
    the processor. Kept indices refer to the original flattened prompt, useful
    for caller-owned aligned fields; stale position_ids must not be reused.

    Mixed media needs explicit opt-in because the pinned HF forward cannot
    represent it correctly. In that extension, per-call provenance defines
    membership, and final features follow surviving prompt order, not call
    order or an assumed images-first layout. No graph output is silently cast.
    """
    _positive_integer(hidden_size, "hidden_size")
    dtype = np.dtype(feature_dtype)
    if dtype.name not in _FEATURE_DTYPES:
        raise TypeError("feature_dtype must be float16, bfloat16, or float32")
    if len(calls) != len(vision_outputs):
        raise ValueError("each vision call must have exactly one output mapping")
    _validate_prompt(
        input_ids, attention_mask, calls, image_token_id, cached_decode, allow_provenance_mixed
    )
    retain = np.ones(input_ids.size, dtype=np.bool_)
    features_by_call = []
    positions_by_call = []
    for call, result in zip(calls, vision_outputs):
        features, mask, indices = _validate_result(call, result, hidden_size, dtype)
        retain[call.placeholder_positions[~mask]] = False
        features_by_call.append(features)
        positions_by_call.append(call.placeholder_positions[indices])
    if features_by_call:
        features = np.concatenate(features_by_call, axis=0)
        positions = np.concatenate(positions_by_call)
        features = features[np.argsort(positions, kind="stable")]
    else:
        features = np.empty((0, hidden_size), dtype=dtype)
    kept = np.flatnonzero(retain)
    if retain.all():
        ids = input_ids.copy()
        attention = None if attention_mask is None else attention_mask.copy()
    else:
        ids = input_ids.reshape(-1)[retain][None, :]
        attention = (
            None if attention_mask is None else attention_mask.reshape(-1)[retain][None, :]
        )
    return OmniPromptInputs(
        ids,
        attention,
        features,
        retain.reshape(input_ids.shape),
        kept,
    )
