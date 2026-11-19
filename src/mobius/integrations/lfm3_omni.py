# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Offline inference over LFM3-Omni's exported components: LFM3-Audio's pipeline plus images."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from mobius.integrations.lfm3_audio import Lfm3AudioPipeline


class Lfm3OmniPipeline(Lfm3AudioPipeline):
    """Run LFM3-Omni with images, audio clips and prompt speech codes as input.

    ``pixel_values`` and ``image_grid_thw`` come from the pinned processor; the
    vision encoder runs once per image. Video input is not supported. Use
    :meth:`generate`: LFM3-Audio's greedy ``generate_text`` and
    ``generate_speech`` ignore Omni's stop ids and speech sampling.
    """

    _feature_tokens: ClassVar[dict[str, str]] = {"image_features": "image_token_id"}

    def generate_text(self, *args, **kwargs) -> np.ndarray:
        raise ValueError(
            "LFM3-Omni decodes with generate(input_ids, max_new_tokens, generation_mode=...), "
            "which follows the checkpoint's generation_config.json."
        )

    generate_speech = generate_text

    def prefill(
        self,
        input_ids: np.ndarray,
        *,
        pixel_values: np.ndarray | None = None,
        image_grid_thw: np.ndarray | None = None,
        **kwargs: np.ndarray,
    ) -> dict[str, np.ndarray]:
        if np.any(input_ids == self.config.video_token_id):
            raise ValueError("LFM3-Omni video input is not supported; pass images or audio.")
        if (pixel_values is None) != (image_grid_thw is None):
            raise ValueError("pixel_values and image_grid_thw must be supplied together.")
        if pixel_values is None:
            return super().prefill(input_ids, **kwargs)
        sizes = np.prod(image_grid_thw, axis=-1) if image_grid_thw.ndim == 2 else None
        if (
            pixel_values.ndim != 2
            or pixel_values.dtype != np.float32
            or image_grid_thw.dtype != np.int64
            or sizes is None
            or image_grid_thw.shape[1] != 3
            or sizes.sum() != pixel_values.shape[0]
        ):
            raise ValueError(
                "pixel_values must be float32 (patches, patch_dim) and image_grid_thw int64 "
                "(images, 3) with one [t, h, w] row per image covering every patch."
            )
        features = np.concatenate(
            [
                self.sessions["vision_encoder"].run(
                    {"pixel_values": patches, "image_grid_thw": grid[None]}
                )["image_features"]
                for patches, grid in zip(
                    np.split(pixel_values, np.cumsum(sizes)[:-1]), image_grid_thw, strict=True
                )
            ]
        )
        return super().prefill(input_ids, image_features=features, **kwargs)
