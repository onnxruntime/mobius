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

# Each registry entry is ``(model_type_filter, hook)``. ``filter`` is
# either a frozenset of model_type strings (only fire for those types) or
# ``None`` (fire for every model_type — hook body does its own
# conditional based on parent_config etc.).
#
# Note: model-agnostic "default" first-pass logic lives in
# apply_audio_defaults / apply_vision_defaults (called explicitly by
# extract_audio_config / extract_vision_config), not in this registry.
# Keeping defaults out of the registry makes the run order self-evident
# at the call site and removes any dependence on import order.
_AUDIO_HOOKS: list[tuple[frozenset[str] | None, Hook]] = []
_VISION_HOOKS: list[tuple[frozenset[str] | None, Hook]] = []


def _make_register(registry: list) -> Callable:
    """Build a decorator that registers a per-model hook.

    Two usages are supported:

    * Parameterised — runs only when ``model_type`` matches one of the
      supplied strings; the dispatcher filters before invocation::

          @register_audio_hook("phi4mm")
          def _phi4mm(config, parent, mt, fields): ...

          @register_audio_hook("gemma4", "gemma4_text")
          def _gemma4(config, parent, mt, fields): ...

    * Bare — runs for every model_type. Reserve this for hooks that need
      to look at ``parent_config`` to decide whether to fire (e.g.
      ``_gemma4_audio`` fires when *parent* is gemma4 even when
      model_type points at a sub-config). Hooks bodies do their own
      conditional and return ``None`` to skip::

          @register_audio_hook
          def _hook(config, parent, mt, fields):
              if parent is None or parent.model_type != "gemma4":
                  return None
              ...

    There is intentionally NO bare-decorator form for "always-runs
    defaults". Defaults live in :func:`apply_audio_defaults` /
    :func:`apply_vision_defaults` and are called explicitly as the first
    pipeline step by :func:`extract_audio_config` /
    :func:`extract_vision_config`. That keeps the run order self-evident
    at the call site and removes any dependence on import order.

    Hooks fire in registration order.
    """

    def register(*model_types):
        # Bare decorator: @register_*_hook
        if (
            len(model_types) == 1
            and callable(model_types[0])
            and not isinstance(model_types[0], str)
        ):
            fn = model_types[0]
            registry.append((None, fn))
            return fn

        types_set = frozenset(model_types) if model_types else None

        def deco(fn: Hook) -> Hook:
            registry.append((types_set, fn))
            return fn

        return deco

    return register


register_audio_hook = _make_register(_AUDIO_HOOKS)
register_vision_hook = _make_register(_VISION_HOOKS)


def _run(hooks: list, config, parent_config, model_type: str, fields: dict):
    """Apply each hook whose filter matches ``model_type``, in registration order.

    Filter ``None`` means the hook runs for every model_type (the hook
    body does its own conditional). Short-circuits on the first hook
    that returns a non-None dict.
    """
    for filter_set, hook in hooks:
        if filter_set is not None and model_type not in filter_set:
            continue
        result = hook(config, parent_config, model_type, fields)
        if result is not None:
            return result
    return None


def extract_audio_config(config, parent_config, model_type: str) -> dict:
    """Run the audio extraction pipeline and assemble the result.

    Pipeline: ``apply_audio_defaults`` (always runs, first) →
    per-model hooks (run if their model_type filter matches).
    Either step can populate ``fields`` (which become kwargs for
    :class:`AudioConfig` at the end), or a per-model hook can return a
    dict that short-circuits the rest of the pipeline.
    """
    from mobius._configs._audio_defaults import apply_audio_defaults
    from mobius._configs._sub_configs import AudioConfig

    fields: dict = {}
    apply_audio_defaults(config, parent_config, model_type, fields)
    short_circuit = _run(_AUDIO_HOOKS, config, parent_config, model_type, fields)
    if short_circuit is not None:
        return short_circuit
    if any(v is not None for v in fields.values()):
        return {"audio": AudioConfig(**fields)}
    return {}


def extract_vision_config(config, parent_config, model_type: str) -> dict:
    """Run the vision extraction pipeline and assemble the result.

    Pipeline: ``apply_vision_defaults`` (always runs, first) →
    per-model hooks (run if their model_type filter matches).
    Either step can populate ``fields`` (which become kwargs for
    :class:`VisionConfig`), or a per-model hook can return a fully-formed
    dict to short-circuit. The dispatcher also lifts a fixed set of
    "shared" vision fields (``image_token_id``, ``video_token_id``, ``spatial_merge_size``,
    ...) up to the top-level of the returned dict so callers can access
    them as ``config.image_token_id`` directly.
    """
    from mobius._configs._sub_configs import VisionConfig
    from mobius._configs._vision_defaults import apply_vision_defaults

    fields: dict = {}
    apply_vision_defaults(config, parent_config, model_type, fields)
    short_circuit = _run(_VISION_HOOKS, config, parent_config, model_type, fields)
    if short_circuit is not None:
        return short_circuit
    # A "vision present" signal: at least one field beyond the always-defaulted
    # geometric knobs (norm_eps / in_channels / spatial_merge_size /
    # temporal_patch_size) must be populated.
    has_vision = any(
        v is not None
        for k, v in fields.items()
        if k not in ("norm_eps", "in_channels", "spatial_merge_size", "temporal_patch_size")
    )
    if not has_vision:
        return {}
    out: dict = {"vision": VisionConfig(**fields)}
    for shared in (
        "mm_tokens_per_image",
        "image_token_id",
        "video_token_id",
        "spatial_merge_size",
        "temporal_patch_size",
        "deepstack_visual_indexes",
        "fullatt_block_indexes",
        "window_size",
        "mrope_section",
        "image_crop_size",
    ):
        val = fields.get(shared)
        if val is not None:
            out[shared] = val
    return out
