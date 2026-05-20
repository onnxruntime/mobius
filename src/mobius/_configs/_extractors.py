# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Plugin-style registry for sub-config extractors.

Each ``extract_*`` function used to be a single mega-switch inside
:mod:`mobius._configs._base` with a chain of ``if model_type == "..."``
branches. That meant every new architecture had to edit a shared file
and risk merge conflicts with unrelated work.

This module replaces those switches with a tiny registry. Hooks are
plain functions registered via the :func:`register_audio_hook` /
:func:`register_vision_hook` decorators. Each hook is invoked on every
extraction and is responsible for guarding its own applicability
(typically by checking ``model_type`` or for the presence of a specific
HuggingFace field). A hook may:

* mutate ``fields`` to contribute key/value pairs into the default
  sub-config that the dispatcher will instantiate at the end, or
* return a fully-formed ``dict`` (e.g. ``{"audio": Gemma4AudioConfig(...)}``)
  to short-circuit — skipping all subsequent hooks and the default
  instantiation. Use this when a model needs a non-default sub-config
  subclass.

New models live in :mod:`mobius._configs.per_model`. Importing that
package is what populates the registries (each module registers its
own hooks at import time).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

Hook = Callable[[Any, Any, str, dict], dict | None]

_AUDIO_HOOKS: list[Hook] = []
_VISION_HOOKS: list[Hook] = []


def register_audio_hook(fn: Hook) -> Hook:
    """Register an audio-extractor hook. See module docstring for the protocol."""
    _AUDIO_HOOKS.append(fn)
    return fn


def register_vision_hook(fn: Hook) -> Hook:
    """Register a vision-extractor hook. See module docstring for the protocol."""
    _VISION_HOOKS.append(fn)
    return fn


def extract_audio_config(config, parent_config, model_type: str) -> dict:
    """Run every registered audio hook and assemble the result.

    Each hook either contributes to ``fields`` (which become kwargs for
    :class:`AudioConfig` at the end) or returns a dict that short-circuits
    the dispatcher.
    """
    from mobius._configs._sub_configs import AudioConfig

    fields: dict = {}
    for hook in _AUDIO_HOOKS:
        result = hook(config, parent_config, model_type, fields)
        if result is not None:
            return result
    if any(v is not None for v in fields.values()):
        return {"audio": AudioConfig(**fields)}
    return {}


def extract_vision_config(config, parent_config, model_type: str) -> dict:
    """Run every registered vision hook and assemble the result.

    Vision hooks differ from audio hooks: there is no default
    :class:`VisionConfig` autoinstantiation — vision sub-configs are
    constructed by an always-applied "default" hook so that other hooks
    can override its output.
    """
    fields: dict = {}
    for hook in _VISION_HOOKS:
        result = hook(config, parent_config, model_type, fields)
        if result is not None:
            return result
    return fields
