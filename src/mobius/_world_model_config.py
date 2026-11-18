# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Immutable, model-agnostic configuration for world-model pipeline builders.

A world-model exporter composes several independently built graphs into one
:class:`mobius.PipelinePackage`. Every such exporter needs the same three
groups of settings:

* how each component graph is built (dtype, weights, execution provider);
* how a runtime should sample tokens and drive the iterative scheduler;
* which checkpoint the package came from and which profile it implements.

These dataclasses carry those settings as frozen values so a builder passes a
single object through its private helpers instead of repeating loose keyword
arguments at every call site. Nothing here is specific to one model family:
component names, action domains, scheduler tuning values, and topology stay in
the model-specific exporter modules.
"""

from __future__ import annotations

__all__ = [
    "WORLD_MODEL_PROFILE",
    "WorldModelBuildConfig",
    "WorldModelGenerationConfig",
    "WorldModelPipelineConfig",
]

import dataclasses
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any

import onnx_ir as ir

from mobius._builder import resolve_dtype

#: Manifest ``profile`` metadata value shared by every world-model package.
WORLD_MODEL_PROFILE = "world-model"

#: Manifest profile version emitted by world-model builders.
DEFAULT_PROFILE_VERSION = "1.0"


@dataclasses.dataclass(frozen=True)
class WorldModelBuildConfig:
    """How each component graph of a world model is built.

    Attributes:
        dtype: Requested parameter dtype, as a Mobius dtype string (e.g.
            ``"f16"``), an ``ir.DataType``, or ``None`` to keep each
            component's checkpoint dtype.
        load_weights: Whether checkpoint weights are streamed into the graphs.
        execution_provider: Requested execution provider, or ``"default"`` to
            let each component advertise a dtype-appropriate preference list.
        trace_optimization: Whether graph-optimization tracing is enabled.
    """

    dtype: str | ir.DataType | None = None
    load_weights: bool = True
    execution_provider: str = "default"
    trace_optimization: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Reject settings that cannot produce a loadable package."""
        if not isinstance(self.execution_provider, str) or not self.execution_provider:
            raise ValueError("execution_provider must be a non-empty string")
        # Resolving eagerly turns an unknown dtype string into a ValueError at
        # configuration time rather than midway through a multi-graph build.
        self.resolved_dtype()

    def resolved_dtype(self) -> ir.DataType | None:
        """Return the requested dtype as an ``ir.DataType``, if one was given."""
        if self.dtype is None or isinstance(self.dtype, ir.DataType):
            return self.dtype
        return resolve_dtype(self.dtype)

    def preferred_execution_providers(self, dtype: ir.DataType) -> tuple[str, ...]:
        """Return the provider preference order for a component of *dtype*.

        An explicit request is honoured verbatim. ``"default"`` expands to the
        providers that can execute *dtype*: DirectML has no bfloat16 support,
        so bfloat16 components omit it.
        """
        if self.execution_provider != "default":
            return (self.execution_provider,)
        if dtype == ir.DataType.BFLOAT16:
            return ("cuda", "cpu")
        return ("cuda", "dml", "cpu")


@dataclasses.dataclass(frozen=True)
class WorldModelGenerationConfig:
    """Runtime sampling and scheduling defaults carried into the manifest.

    The token-sampling fields mirror the HuggingFace ``generation_config.json``
    contract; the diffusion fields describe how many scheduler steps a runtime
    should take by default and which per-mode scheduler overrides the exporter
    recorded. All of them are opaque runtime hints: no graph is built from
    them, so they only reach the manifest.
    """

    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    max_new_tokens: int | None = None
    eos_token_ids: tuple[int, ...] = ()
    default_inference_steps: int = 1
    scheduler_mode_overrides: Mapping[str, Any] = dataclasses.field(
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "eos_token_ids", tuple(self.eos_token_ids))
        object.__setattr__(
            self,
            "scheduler_mode_overrides",
            MappingProxyType(dict(self.scheduler_mode_overrides)),
        )
        self.validate()

    def validate(self) -> None:
        """Reject values a runtime could not act on."""
        for name in ("temperature", "top_p", "repetition_penalty"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a number")
            if value < 0:
                raise ValueError(f"{name} must not be negative")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
            raise TypeError("top_k must be an integer")
        if self.top_k < 0:
            raise ValueError("top_k must not be negative")
        if self.max_new_tokens is not None and self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive when set")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in self.eos_token_ids
        ):
            raise TypeError("eos_token_ids must contain integers")
        if self.default_inference_steps <= 0:
            raise ValueError("default_inference_steps must be positive")

    @classmethod
    def from_generation_config(
        cls,
        values: Mapping[str, Any] | None,
        *,
        default_inference_steps: int = 1,
        scheduler_mode_overrides: Mapping[str, Any] | None = None,
    ) -> WorldModelGenerationConfig:
        """Parse a HuggingFace ``generation_config.json`` mapping.

        Missing keys — and keys explicitly set to ``null``, which real
        checkpoints use to mean "unset" — fall back to the HuggingFace
        defaults. ``eos_token_id`` may be a single id or a list; both
        normalize to a tuple.
        """
        mapping = dict(values or {})

        def number(key: str, default: float) -> float:
            value = mapping.get(key)
            return default if value is None else float(value)

        eos: Any = mapping.get("eos_token_id", ())
        if isinstance(eos, int) and not isinstance(eos, bool):
            eos = (eos,)
        eos_token_ids: Iterable[int] = tuple(eos or ())
        max_new_tokens = mapping.get("max_new_tokens")
        top_k = mapping.get("top_k")
        return cls(
            do_sample=bool(mapping.get("do_sample")),
            temperature=number("temperature", 1.0),
            top_k=50 if top_k is None else int(top_k),
            top_p=number("top_p", 1.0),
            repetition_penalty=number("repetition_penalty", 1.0),
            max_new_tokens=None if max_new_tokens is None else int(max_new_tokens),
            eos_token_ids=tuple(eos_token_ids),
            default_inference_steps=default_inference_steps,
            scheduler_mode_overrides=scheduler_mode_overrides or {},
        )

    def sampling_manifest(self) -> dict[str, Any]:
        """Return the deterministic ``sampling`` block of a decode stage."""
        return {
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
        }

    def stop_manifest(self, *, max_sequence_length: int | None) -> dict[str, Any]:
        """Return the deterministic ``stop`` block of a decode stage."""
        return {
            "kind": "token_ids",
            "eos_token_ids": list(self.eos_token_ids),
            "max_sequence_length": max_sequence_length,
        }

    def max_tokens_manifest(self, *, limit: int | None) -> dict[str, Any]:
        """Return the deterministic ``max_tokens`` block of a decode stage.

        A checkpoint without ``max_new_tokens`` cannot bound its own decode
        loop, so the runtime must supply the budget.
        """
        return {
            "default": self.max_new_tokens,
            "required_override": self.max_new_tokens is None,
            "limit": limit,
        }

    def scheduler_mode_overrides_manifest(self) -> dict[str, Any]:
        """Return a mutable copy of the per-mode scheduler overrides."""
        return dict(self.scheduler_mode_overrides)


@dataclasses.dataclass(frozen=True)
class WorldModelPipelineConfig:
    """Identity and runtime contract of one composed world-model package.

    Attributes:
        model_id: Checkpoint the package was exported from, recorded as the
            manifest ``source``.
        model_type: Checkpoint ``model_type`` the package implements, recorded
            as manifest ``model_type`` and used to derive the profile name.
        build: How the component graphs were built.
        generation: Runtime sampling and scheduling defaults.
        extra_metadata: Additional manifest metadata entries applied last, so
            a model family can attach its own top-level block.
        profile_version: Version of the runtime profile implemented here.
    """

    model_id: str
    model_type: str
    build: WorldModelBuildConfig = dataclasses.field(
        default_factory=WorldModelBuildConfig,
    )
    generation: WorldModelGenerationConfig = dataclasses.field(
        default_factory=WorldModelGenerationConfig,
    )
    extra_metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    profile_version: str = DEFAULT_PROFILE_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "extra_metadata",
            MappingProxyType(dict(self.extra_metadata)),
        )
        self.validate()

    def validate(self) -> None:
        """Reject an unidentifiable package."""
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("model_id must be a non-empty string")
        if not isinstance(self.model_type, str) or not self.model_type:
            raise ValueError("model_type must be a non-empty string")
        if not isinstance(self.profile_version, str) or not self.profile_version:
            raise ValueError("profile_version must be a non-empty string")

    @property
    def profile_name(self) -> str:
        """Manifest profile name derived from ``model_type``."""
        return self.model_type.replace("_", "-")

    def manifest_metadata(self) -> dict[str, Any]:
        """Return the metadata entries every world-model package declares."""
        return {
            "profile": WORLD_MODEL_PROFILE,
            "model_type": self.model_type,
            "source": self.model_id,
        }
