# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Per-model extractor hooks.

Importing this package registers every model-specific hook with the
extractor registries defined in :mod:`mobius._configs._extractors`. New
models add a file here instead of editing a shared switch.

Model-agnostic defaults live in :mod:`mobius._configs._audio_defaults`
and :mod:`mobius._configs._vision_defaults` (one level up from this
package) and are called explicitly as the first pipeline step by
``extract_audio_config`` / ``extract_vision_config``. Keeping them out
of this package guarantees that the dispatcher can lazily import the
defaults without also triggering the per-model registrations (which
would otherwise pollute monkeypatched test registries permanently).
"""

from __future__ import annotations

# Import for side effects: each module registers its hooks at import time.
# Run order is irrelevant — hooks are filtered by ``model_type`` and the
# defaults run separately as an explicit first pass, so `ruff` / `isort`
# may freely re-sort this block.
from mobius._configs.per_model import (  # noqa: F401
    _cosmos3_edge_vision,
    _gemma3n_audio,
    _gemma3n_vision,
    _gemma4_audio,
    _gemma4_unified_audio,
    _gemma4_unified_vision,
    _glm_ocr_vision,
    _hunyuan_vl_mot_vision,
    _internvl_vision,
    _mage_vl_vision,
    _minicpmv4_6_vision,
    _muse_glimmer_vision,
    _phi4mm_audio,
    _phi4mm_vision,
    _phi_vision,
    _qwen3_asr_audio,
    _sensevoice_audio,
)
