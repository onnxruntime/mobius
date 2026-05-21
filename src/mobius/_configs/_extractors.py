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

# Each registry entry is ``(model_type_filter, hook)`` where the filter is
# either ``None`` (always run) or a frozenset of model_type strings.
_AUDIO_HOOKS: list[tuple[frozenset[str] | None, Hook]] = []
_VISION_HOOKS: list[tuple[frozenset[str] | None, Hook]] = []


def _make_register(registry: list) -> Callable:
    """Build a decorator that supports both bare and parameterised usage."""

    def register(*model_types):
        """Register a hook in *registry*.

        Two usages are supported:

        * Bare decorator — runs for every model_type. Use this for default
          hooks that pull a generic HuggingFace field common to many models.

          .. code-block:: python

              @register_audio_hook
              def _default(config, parent, mt, fields): ...

        * Parameterised decorator — runs only when ``model_type`` matches one
          of the supplied strings. The dispatcher filters before invocation
          so hook bodies don't need to repeat the ``if model_type != ...``
          guard. Hooks that *also* need to inspect ``parent_config`` (e.g.
          Gemma4's text-config short-circuit) can still do that check inside.

          .. code-block:: python

              @register_audio_hook("phi4mm")
              def _phi4mm(config, parent, mt, fields): ...

              @register_audio_hook("gemma4", "gemma4_text")
              def _gemma4(config, parent, mt, fields): ...
        """
        # Bare decorator: ``@register`` with a single callable arg
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
    """Apply each hook whose filter matches ``model_type``.

    Short-circuits on the first hook that returns a non-None dict.
    """
    for filter_set, hook in hooks:
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

    Vision hooks differ from audio hooks: there is no default
    :class:`VisionConfig` autoinstantiation — vision sub-configs are
    constructed by an always-applied "default" hook so that other hooks
    can override its output.
    """
    fields: dict = {}
    short_circuit = _run(_VISION_HOOKS, config, parent_config, model_type, fields)
    if short_circuit is not None:
        return short_circuit
    return fields
