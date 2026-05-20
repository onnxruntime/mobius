# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Per-model extractor hooks and config subclasses.

Importing this package registers every hook with the extractor registries
defined in :mod:`mobius._configs._extractors`. New models add a file
here instead of editing a shared switch.
"""

from __future__ import annotations

# Import for side effects: each module registers its hooks at import time.
from mobius._configs.per_model import (  # noqa: F401
    _audio_default,
    _gemma4_audio,
    _hunyuan_vl_mot_vision,
    _internvl_vision,
    _phi4mm_audio,
    _phi4mm_vision,
    _qwen3_asr_audio,
    _sensevoice_audio,
    _vision_default,
)
