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

# Hook execution priority constants. Lower values run first; equal values
# preserve registration order (stable sort). The two extremes are exposed
# so call-sites read self-documenting.
DEFAULT_PRIORITY = 0  # "always-runs" hooks that fill in HF defaults
PER_MODEL_PRIORITY = 100  # per-model overrides that may stomp on defaults

# Each registry entry is ``(priority, insertion_index, model_type_filter, hook)``
# where the filter is either ``None`` (always run) or a frozenset of
# model_type strings. ``insertion_index`` makes the sort total-ordered and
# stable across equal priorities so import order is irrelevant.
_AUDIO_HOOKS: list[tuple[int, int, frozenset[str] | None, Hook]] = []
_VISION_HOOKS: list[tuple[int, int, frozenset[str] | None, Hook]] = []


def _make_register(registry: list) -> Callable:
    """Build a decorator that supports both bare and parameterised usage."""

    def register(*model_types, priority: int = PER_MODEL_PRIORITY):
        """Register a hook in *registry* with an explicit priority.

        Two usages are supported:

        * Bare decorator — runs for every model_type. Use this for default
          hooks that pull a generic HuggingFace field common to many models.
          Defaults are typically registered with
          ``priority=DEFAULT_PRIORITY`` so they run before any per-model
          overrides regardless of import order:

          .. code-block:: python

              @register_audio_hook(priority=DEFAULT_PRIORITY)
              def _default(config, parent, mt, fields): ...

          A bare ``@register_audio_hook`` form (no parentheses) is also
          supported and defaults to ``priority=PER_MODEL_PRIORITY``.

        * Parameterised decorator — runs only when ``model_type`` matches
          one of the supplied strings. The dispatcher filters before
          invocation so hook bodies don't need to repeat the
          ``if model_type != ...`` guard. Hooks that *also* need to inspect
          ``parent_config`` (e.g. Gemma4's text-config short-circuit) can
          still do that check inside.

          .. code-block:: python

              @register_audio_hook("phi4mm")
              def _phi4mm(config, parent, mt, fields): ...

              @register_audio_hook("gemma4", "gemma4_text")
              def _gemma4(config, parent, mt, fields): ...

        Notes on ordering:
            Hooks are sorted by ``(priority, insertion_index)`` before the
            dispatcher iterates. Within the same priority, the relative
            order of registration is preserved (stable sort). Crucially,
            this means a per-model hook's ``fields.update(...)`` will
            always run *after* a ``DEFAULT_PRIORITY`` hook's
            ``fields.setdefault(...)`` regardless of import order.
        """
        # Bare decorator: ``@register`` with a single callable arg
        if (
            len(model_types) == 1
            and callable(model_types[0])
            and not isinstance(model_types[0], str)
        ):
            fn = model_types[0]
            registry.append((priority, len(registry), None, fn))
            return fn

        types_set = frozenset(model_types) if model_types else None

        def deco(fn: Hook) -> Hook:
            registry.append((priority, len(registry), types_set, fn))
            return fn

        return deco

    return register


register_audio_hook = _make_register(_AUDIO_HOOKS)
register_vision_hook = _make_register(_VISION_HOOKS)


def _run(hooks: list, config, parent_config, model_type: str, fields: dict):
    """Apply each hook whose filter matches ``model_type``, in priority order.

    Sort by ``(priority, insertion_index)`` so defaults always run before
    per-model overrides, independent of the order in which hook modules
    were imported. Short-circuits on the first hook that returns a non-None
    dict.
    """
    for _priority, _index, filter_set, hook in sorted(hooks):
        if filter_set is not None and model_type not in filter_set:
            continue
        result = hook(config, parent_config, model_type, fields)
        if result is not None:
            return result
    return None


def extract_audio_config(config, parent_config, model_type: str) -> dict:
    """Run every applicable audio hook and assemble the result.

    Each hook either contributes to ``fields`` (which become kwargs for
    :class:`AudioConfig` at the end) or returns a dict that short-circuits
    the dispatcher.
    """
    from mobius._configs._sub_configs import AudioConfig

    fields: dict = {}
    short_circuit = _run(_AUDIO_HOOKS, config, parent_config, model_type, fields)
    if short_circuit is not None:
        return short_circuit
    if any(v is not None for v in fields.values()):
        return {"audio": AudioConfig(**fields)}
    return {}


def extract_vision_config(config, parent_config, model_type: str) -> dict:
    """Run every registered vision hook and assemble the result.

    Hooks may either contribute to ``fields`` (which become kwargs for
    :class:`VisionConfig`) or return a fully-formed dict to short-circuit.
    The dispatcher also lifts a fixed set of "shared" vision fields
    (``image_token_id``, ``spatial_merge_size``, ...) up to the top-level
    of the returned dict so callers can access them as
    ``config.image_token_id`` directly.
    """
    from mobius._configs._sub_configs import VisionConfig

    fields: dict = {}
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
