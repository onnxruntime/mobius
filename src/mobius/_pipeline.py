# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Runtime-agnostic, model-agnostic compositional pipeline core.

This module describes how already-built ONNX graphs are wired and how a
runtime executes them. It is a runtime-neutral contract:

- it never runs, traces, or optimizes a graph;
- it serializes typed input semantics, generated-input programs, state
  lifecycle, scheduler/sampling controls, transform parameters, assets, and
  execution-provider hints;
- it never executes a :attr:`PipelineConnection.transform` — a transform is a
  registered *kind* that a runtime resolves, and the core only records it and
  its validated parameters/capabilities.

Model-specific exporters compile HuggingFace/Diffusers semantics into this
standard contract. A runtime implements the registered programs and strategies
instead of inferring behavior from tensor names.

Layers
------

``PipelineComponent``
    One :class:`onnx_ir.Model` plus a validated role/phase and optional
    presence, capability, source, and config metadata.  Its typed graph ports
    are derived from the graph signature.
``PipelineConnection``
    ``component.output -> component.input``. Fan-out is allowed; an input may
    have one initializer and one recurrent update. A recurrent connection is
    loop-carried state scoped to a looping stage. A registered transform kind
    may adapt the edge and declare additional context ports/capabilities.
``PipelineStage``
    A validated strategy kind over a set of components, with JSON-safe options
    and capabilities.
``PipelineManifest``
    The serializable, deterministic executable contract.
``PipelinePackage``
    A :class:`~mobius._model_package.ModelPackage` that also carries the
    manifest and per-component configs, and persists ``pipeline.json``.  It can
    ship opaque runtime assets (tokenizers, scheduler configs) whose
    destinations are recorded and whose contents are never interpreted.
``PipelineBuilder``
    Convenience composition front-end with full structural validation on
    :meth:`PipelineBuilder.build`.

Example::

    builder = PipelineBuilder()
    builder.add_model("encoder", encoder_model, role="encoder")
    builder.add_model("decoder", decoder_model, role="decoder")
    builder.connect("encoder.hidden", "decoder.encoder_hidden")
    builder.declare_external("encoder.pixel_values")
    builder.declare_generated("decoder.position_ids", generator="zeros")
    builder.add_stage("encode", "single_pass", ["encoder"])
    builder.add_stage("generate", "autoregressive", ["decoder"])
    builder.add_public_output("decoder.logits")
    pkg = builder.build()
    pkg.save("/out")
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_PHASE",
    "InputSource",
    "GeneratedInputDefinition",
    "GeneratedInputRule",
    "LOOP_CARRIED_STATE_CAPABILITY",
    "PIPELINE_FILENAME",
    "PIPELINE_SCHEMA_VERSION",
    "PhaseDefinition",
    "PipelineAsset",
    "PipelineBuilder",
    "PipelineComponent",
    "PipelineConnection",
    "PipelineInput",
    "PipelineManifest",
    "PipelineOutput",
    "PipelinePackage",
    "PipelinePort",
    "PipelineProfile",
    "PipelineState",
    "PipelineStage",
    "PipelineValidationError",
    "RoleDefinition",
    "StrategyDefinition",
    "StateDefinition",
    "TensorSpec",
    "TransformDefinition",
    "phase_definition",
    "generated_input_definition",
    "register_phase",
    "register_generated_input",
    "register_role",
    "register_strategy",
    "register_state",
    "register_transform",
    "role_definition",
    "strategy_definition",
    "state_definition",
    "transform_definition",
]

import dataclasses
import json
import math
import os
import shutil
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any, TypeAlias

import onnx_ir as ir

from mobius._model_package import ModelPackage

JSONValue: TypeAlias = (
    "bool | int | float | str | list[JSONValue] | dict[str, JSONValue] | None"
)

#: Schema version of :class:`PipelineManifest`. ``major.minor``; a mismatching
#: *major* version is a hard failure on load.
PIPELINE_SCHEMA_VERSION = "1.1"

#: Name of the manifest file written next to the component graphs.
PIPELINE_FILENAME = "pipeline.json"

#: Separator between a component name and a graph port name in an endpoint.
ENDPOINT_SEPARATOR = "."

#: Phase meaning "no phase restriction".
DEFAULT_PHASE = "always"

#: Capability contributed by a stage that carries loop state across iterations.
LOOP_CARRIED_STATE_CAPABILITY = "loop_carried_state"


class PipelineValidationError(ValueError):
    """Raised when a pipeline topology is structurally invalid."""


# ---------------------------------------------------------------------------
# Name validation
# ---------------------------------------------------------------------------

_INVALID_NAME_CHARS = frozenset('/\\:*?"<>|\0' + ENDPOINT_SEPARATOR)
_RESERVED_NAMES = frozenset(
    ["con", "prn", "aux", "nul"]
    + [f"com{i}" for i in range(1, 10)]
    + [f"lpt{i}" for i in range(1, 10)]
)


def _validate_component_name(name: str) -> str:
    """Validate that *name* is usable as a single, safe directory name."""
    if not isinstance(name, str) or not name or name.strip() != name:
        raise PipelineValidationError(
            f"Component name {name!r} must be a non-blank string without "
            "surrounding whitespace."
        )
    if name in {".", ".."} or ".." in name:
        raise PipelineValidationError(f"Component name {name!r} must not contain '..'.")
    bad = sorted(_INVALID_NAME_CHARS & set(name))
    if bad:
        chars = ", ".join(repr(c) for c in bad)
        raise PipelineValidationError(
            f"Component name {name!r} must not contain {chars} because it is used "
            "as a directory name and as an endpoint prefix."
        )
    if any(ord(c) < 32 for c in name):
        raise PipelineValidationError(
            f"Component name {name!r} must not contain control characters."
        )
    if name.split(".")[0].lower() in _RESERVED_NAMES:
        raise PipelineValidationError(
            f"Component name {name!r} is a reserved filesystem name."
        )
    return name


def _validate_token(name: str, what: str) -> str:
    """Validate a short identifier-like token (stage/role/strategy/phase name)."""
    if not isinstance(name, str) or not name or name.strip() != name:
        raise PipelineValidationError(
            f"{what} name {name!r} must be a non-blank string without surrounding whitespace."
        )
    if any(c in name for c in "/\\\0"):
        raise PipelineValidationError(
            f"{what} name {name!r} must not contain path separators."
        )
    return name


def _validate_port_name(name: str, component: str) -> str:
    """Validate a graph port name (ONNX value names may contain dots)."""
    if not isinstance(name, str) or not name or name.strip() != name:
        raise PipelineValidationError(
            f"Port name {name!r} on component {component!r} must be a non-blank string."
        )
    return name


def _validate_asset_path(path: str) -> str:
    """Validate a ``/``-separated relative destination inside a package directory.

    Rejects anything that could escape the package directory or collide with a
    platform-reserved name: absolute paths, drive letters, UNC prefixes,
    backslashes, ``.``/``..`` segments, empty segments, and control characters.
    """
    if not isinstance(path, str) or not path or path.strip() != path:
        raise PipelineValidationError(
            f"Asset destination {path!r} must be a non-blank string without "
            "surrounding whitespace."
        )
    if "\\" in path:
        raise PipelineValidationError(
            f"Asset destination {path!r} must use '/' separators, not backslashes."
        )
    if path.startswith(("/", "~")) or os.path.isabs(path):
        raise PipelineValidationError(
            f"Asset destination {path!r} must be relative to the package directory."
        )
    if len(path) >= 2 and path[1] == ":":
        raise PipelineValidationError(
            f"Asset destination {path!r} must not contain a drive letter."
        )
    if path.endswith("/"):
        raise PipelineValidationError(
            f"Asset destination {path!r} must name a file, not a directory."
        )
    for segment in path.split("/"):
        if not segment:
            raise PipelineValidationError(
                f"Asset destination {path!r} must not contain empty path segments."
            )
        if segment in {".", ".."}:
            raise PipelineValidationError(
                f"Asset destination {path!r} must not contain '.' or '..' segments."
            )
        if segment.strip() != segment:
            raise PipelineValidationError(
                f"Asset destination {path!r} must not have padded path segments."
            )
        if any(ord(c) < 32 for c in segment) or any(c in segment for c in ':*?"<>|'):
            raise PipelineValidationError(
                f"Asset destination {path!r} contains characters that are unsafe in a "
                "file path."
            )
        if (
            segment.lower() in _RESERVED_NAMES
            or segment.split(".")[0].lower() in _RESERVED_NAMES
        ):
            raise PipelineValidationError(
                f"Asset destination {path!r} uses reserved filesystem name {segment!r}."
            )
    return path


# ---------------------------------------------------------------------------
# JSON safety
# ---------------------------------------------------------------------------


def _ensure_json_value(value: Any, context: str) -> JSONValue:
    """Return a normalized, JSON-safe copy of *value* or raise."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PipelineValidationError(
                f"{context} must be JSON-safe; {value!r} is not a finite number."
            )
        return float(value)
    if isinstance(value, Mapping):
        result: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise PipelineValidationError(
                    f"{context} must be JSON-safe; mapping key {key!r} is not a string."
                )
            result[key] = _ensure_json_value(item, f"{context}[{key!r}]")
        return result
    if isinstance(value, (list, tuple)):
        return [_ensure_json_value(item, f"{context}[{i}]") for i, item in enumerate(value)]
    raise PipelineValidationError(
        f"{context} must be JSON-safe; got value of type {type(value).__name__}."
    )


def _ensure_json_mapping(
    value: Mapping[str, Any] | None, context: str
) -> dict[str, JSONValue]:
    if value is None:
        return {}
    normalized = _ensure_json_value(value, context)
    if not isinstance(normalized, dict):
        raise PipelineValidationError(f"{context} must be a mapping.")
    return normalized


def _string_tuple(values: Iterable[str] | None, context: str) -> tuple[str, ...]:
    """Normalize an iterable of capability-like strings: deduped and sorted."""
    if values is None:
        return ()
    if isinstance(values, str):
        raise PipelineValidationError(
            f"{context} must be a sequence of strings, not a string."
        )
    out = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise PipelineValidationError(f"{context} entries must be non-blank strings.")
        out.add(value)
    return tuple(sorted(out))


def _ordered_string_tuple(
    values: Iterable[str] | None,
    context: str,
) -> tuple[str, ...]:
    """Normalize strings while preserving preference/declaration order."""
    if values is None:
        return ()
    if isinstance(values, str):
        raise PipelineValidationError(
            f"{context} must be a sequence of strings, not a string."
        )
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise PipelineValidationError(f"{context} entries must be non-blank strings.")
        if value not in seen:
            result.append(value)
            seen.add(value)
    return tuple(result)


# ---------------------------------------------------------------------------
# Open registries with closed validation
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RoleDefinition:
    """A registered component role."""

    name: str
    description: str = ""


@dataclasses.dataclass(frozen=True)
class PhaseDefinition:
    """A registered ``run_on`` phase."""

    name: str
    description: str = ""


@dataclasses.dataclass(frozen=True)
class StrategyDefinition:
    """A registered stage strategy kind.

    Attributes:
        name: The strategy identifier used in manifests.
        description: Human-readable summary.
        loop_carried_state: Whether stages of this kind may own recurrent
            (loop-carried) connections.
    """

    name: str
    description: str = ""
    loop_carried_state: bool = False
    required_options: tuple[str, ...] = ()
    allowed_options: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "required_options",
            _string_tuple(self.required_options, f"Strategy {self.name!r} required options"),
        )
        if self.allowed_options is not None:
            allowed = _string_tuple(
                self.allowed_options, f"Strategy {self.name!r} allowed options"
            )
            if not set(self.required_options) <= set(allowed):
                raise PipelineValidationError(
                    f"Strategy {self.name!r} required options must be allowed."
                )
            object.__setattr__(self, "allowed_options", allowed)


@dataclasses.dataclass(frozen=True)
class TransformDefinition:
    """A registered edge transform kind.

    A transform is applied *by the runtime* on a connection.  The core neither
    defines nor executes the computation; it only records that the edge is
    mediated and what the runtime must be able to do.  Because a transform may
    legitimately change dtype, rank, and shape (a VAE posterior parameterization
    sampled and patchified into generator tokens, a scheduler turning a noise
    prediction into the next latent), direct port compatibility is *not*
    assumed for transformed edges.

    Attributes:
        name: The transform identifier used in manifests.
        description: Human-readable summary.
        capabilities: Capabilities the executing runtime must provide for this
            transform.  A manifest using the transform must list them in
            :attr:`PipelineManifest.required_capabilities`.
    """

    name: str
    description: str = ""
    capabilities: tuple[str, ...] = ()
    required_parameters: tuple[str, ...] = ()
    allowed_parameters: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "capabilities",
            _string_tuple(self.capabilities, f"Transform {self.name!r} capabilities"),
        )
        object.__setattr__(
            self,
            "required_parameters",
            _string_tuple(
                self.required_parameters,
                f"Transform {self.name!r} required parameters",
            ),
        )
        if self.allowed_parameters is not None:
            allowed = _string_tuple(
                self.allowed_parameters,
                f"Transform {self.name!r} allowed parameters",
            )
            if not set(self.required_parameters) <= set(allowed):
                raise PipelineValidationError(
                    f"Transform {self.name!r} required parameters must be allowed."
                )
            object.__setattr__(self, "allowed_parameters", allowed)


@dataclasses.dataclass(frozen=True)
class GeneratedInputDefinition:
    """A registered runtime program that materializes a graph input."""

    name: str
    description: str = ""
    capabilities: tuple[str, ...] = ()
    required_parameters: tuple[str, ...] = ()
    allowed_parameters: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "capabilities",
            _string_tuple(self.capabilities, f"Generator {self.name!r} capabilities"),
        )
        object.__setattr__(
            self,
            "required_parameters",
            _string_tuple(
                self.required_parameters,
                f"Generator {self.name!r} required parameters",
            ),
        )
        if self.allowed_parameters is not None:
            allowed = _string_tuple(
                self.allowed_parameters,
                f"Generator {self.name!r} allowed parameters",
            )
            if not set(self.required_parameters) <= set(allowed):
                raise PipelineValidationError(
                    f"Generator {self.name!r} required parameters must be allowed."
                )
            object.__setattr__(self, "allowed_parameters", allowed)


@dataclasses.dataclass(frozen=True)
class StateDefinition:
    """A registered recurrent-state semantic."""

    name: str
    description: str = ""


_ROLES: dict[str, RoleDefinition] = {}
_PHASES: dict[str, PhaseDefinition] = {}
_STRATEGIES: dict[str, StrategyDefinition] = {}
_TRANSFORMS: dict[str, TransformDefinition] = {}
_GENERATED_INPUTS: dict[str, GeneratedInputDefinition] = {}
_STATES: dict[str, StateDefinition] = {}


def _register(
    registry: dict[str, Any],
    definition: Any,
    what: str,
) -> Any:
    """Idempotently register *definition*, rejecting conflicting redefinition."""
    _validate_token(definition.name, what)
    existing = registry.get(definition.name)
    if existing is not None:
        if existing != definition:
            raise PipelineValidationError(
                f"{what} {definition.name!r} is already registered with a different "
                f"definition ({existing!r} != {definition!r})."
            )
        return existing
    registry[definition.name] = definition
    return definition


def register_role(name: str, *, description: str = "") -> RoleDefinition:
    """Register a component role.

    Registration is idempotent for an identical definition and raises for a
    conflicting redefinition of the same name.
    """
    return _register(_ROLES, RoleDefinition(name, description), "Role")


def register_phase(name: str, *, description: str = "") -> PhaseDefinition:
    """Register a ``run_on`` phase (idempotent for an identical definition)."""
    return _register(_PHASES, PhaseDefinition(name, description), "Phase")


def register_strategy(
    name: str,
    *,
    description: str = "",
    loop_carried_state: bool = False,
    required_options: Iterable[str] | None = None,
    allowed_options: Iterable[str] | None = None,
) -> StrategyDefinition:
    """Register a stage strategy kind (idempotent for an identical definition).

    Args:
        name: Strategy identifier.
        description: Human-readable summary.
        loop_carried_state: Whether stages of this kind may own recurrent
            connections.
    """
    definition = StrategyDefinition(
        name,
        description,
        loop_carried_state,
        tuple(required_options or ()),
        tuple(allowed_options) if allowed_options is not None else None,
    )
    return _register(_STRATEGIES, definition, "Strategy")


def register_transform(
    name: str,
    *,
    description: str = "",
    capabilities: Iterable[str] | None = None,
    required_parameters: Iterable[str] | None = None,
    allowed_parameters: Iterable[str] | None = None,
) -> TransformDefinition:
    """Register an edge transform kind (idempotent for an identical definition).

    Args:
        name: Transform identifier used by :attr:`PipelineConnection.transform`.
        description: Human-readable summary.
        capabilities: Capabilities the executing runtime must provide.  Any
            manifest that uses the transform must list them in its
            ``required_capabilities``; :class:`PipelineBuilder` adds them
            automatically.
    """
    definition = TransformDefinition(
        name,
        description,
        tuple(capabilities or ()),
        tuple(required_parameters or ()),
        tuple(allowed_parameters) if allowed_parameters is not None else None,
    )
    return _register(_TRANSFORMS, definition, "Transform")


def register_generated_input(
    name: str,
    *,
    description: str = "",
    capabilities: Iterable[str] | None = None,
    required_parameters: Iterable[str] | None = None,
    allowed_parameters: Iterable[str] | None = None,
) -> GeneratedInputDefinition:
    """Register a runtime input-generation program."""
    definition = GeneratedInputDefinition(
        name,
        description,
        tuple(capabilities or ()),
        tuple(required_parameters or ()),
        tuple(allowed_parameters) if allowed_parameters is not None else None,
    )
    return _register(_GENERATED_INPUTS, definition, "Generated input")


def register_state(name: str, *, description: str = "") -> StateDefinition:
    """Register a recurrent-state semantic."""
    return _register(_STATES, StateDefinition(name, description), "State")


def role_definition(name: str) -> RoleDefinition:
    """Return a registered role or raise :class:`PipelineValidationError`."""
    return _lookup(_ROLES, name, "role")


def phase_definition(name: str) -> PhaseDefinition:
    """Return a registered phase or raise :class:`PipelineValidationError`."""
    return _lookup(_PHASES, name, "phase")


def strategy_definition(name: str) -> StrategyDefinition:
    """Return a registered strategy or raise :class:`PipelineValidationError`."""
    return _lookup(_STRATEGIES, name, "strategy")


def transform_definition(name: str) -> TransformDefinition:
    """Return a registered transform or raise :class:`PipelineValidationError`."""
    return _lookup(_TRANSFORMS, name, "transform")


def generated_input_definition(name: str) -> GeneratedInputDefinition:
    """Return a registered generated-input program."""
    return _lookup(_GENERATED_INPUTS, name, "generated_input")


def state_definition(name: str) -> StateDefinition:
    """Return a registered recurrent-state semantic."""
    return _lookup(_STATES, name, "state")


def _validate_registered_parameters(
    *,
    kind: str,
    parameters: Mapping[str, Any],
    required: Iterable[str],
    allowed: Iterable[str] | None,
    context: str,
) -> dict[str, JSONValue]:
    normalized = _ensure_json_mapping(parameters, f"{context} parameters")
    missing = sorted(set(required) - set(normalized))
    if missing:
        raise PipelineValidationError(
            f"{context} {kind!r} is missing required parameter(s) {missing}."
        )
    if allowed is not None:
        unknown = sorted(set(normalized) - set(allowed))
        if unknown:
            raise PipelineValidationError(
                f"{context} {kind!r} has unknown parameter(s) {unknown}."
            )
    return normalized


def _lookup(registry: dict[str, Any], name: str, what: str) -> Any:
    definition = registry.get(name)
    if definition is None:
        known = ", ".join(sorted(registry)) or "<none>"
        raise PipelineValidationError(
            f"Unknown {what} {name!r}. Known {what}s: {known}. "
            f"Use register_{what}() to extend the registry."
        )
    return definition


# Built-in, deliberately model-agnostic vocabulary.
for _role, _role_doc in (
    ("encoder", "Maps raw or embedded observations into a latent representation"),
    ("decoder", "Maps latent representations back into outputs or tokens"),
    ("embedding", "Turns discrete ids into dense vectors"),
    ("projector", "Adapts one representation space into another"),
    ("dynamics", "Advances latent state given an action or control signal"),
    ("observation", "Consumes or produces environment observations"),
    ("action", "Produces or consumes action representations"),
    ("policy", "Maps state to an action distribution"),
    ("value", "Estimates a scalar value of a state"),
    ("reward", "Estimates a scalar reward"),
    ("sampler", "Turns scores into concrete selections"),
    ("transform", "Pure tensor reshaping/normalization graph"),
    ("generic", "Unclassified component"),
):
    register_role(_role, description=_role_doc)

for _phase, _phase_doc in (
    (DEFAULT_PHASE, "No phase restriction"),
    ("init", "Runs once when the pipeline is created"),
    ("warmup", "Runs once before the first real request"),
    ("prefill", "Runs on the initial/context pass"),
    ("decode", "Runs on each incremental pass"),
    ("step", "Runs on each iteration of a loop"),
    ("refine", "Runs on refinement iterations"),
    ("finalize", "Runs once after the loop terminates"),
    ("on_demand", "Runs only when its presence condition holds"),
):
    register_phase(_phase, description=_phase_doc)

for _strategy, _strategy_doc, _loop, _options in (
    ("single_pass", "Run every component exactly once, in dependency order", False, ()),
    (
        "autoregressive",
        "Repeat until a stop condition, feeding outputs back",
        True,
        ("tokenizer_asset", "sampling", "stop", "max_tokens", "state_names"),
    ),
    (
        "iterative",
        "Repeat a fixed or condition-bound number of iterations",
        True,
        (
            "scheduler",
            "default_steps",
            "timestep",
            "state_inputs",
            "initial_state_inputs",
            "prediction_type",
            "packed_modalities",
        ),
    ),
    (
        "state_transition",
        "Advance a carried state one step per invocation",
        True,
        ("state_names", "max_steps", "stop"),
    ),
    ("composite", "Group other stages; ordering delegated to the runtime", False, ("stages",)),
    (
        "on_demand",
        "Run only when the component presence condition holds",
        False,
        ("presence",),
    ),
):
    register_strategy(
        _strategy,
        description=_strategy_doc,
        loop_carried_state=_loop,
        allowed_options=_options,
    )

for _transform, _transform_doc, _transform_caps, _parameters in (
    ("cast", "Convert the tensor element type", ("tensor_cast",), ("to",)),
    (
        "reshape",
        "Rearrange dimensions without changing the element count",
        ("tensor_reshape",),
        ("shape", "input_layout", "output_layout"),
    ),
    (
        "normalize",
        "Apply a scale/shift such as a latent scaling factor",
        ("tensor_normalize",),
        ("mean", "std", "scale", "shift"),
    ),
    (
        "sample",
        "Draw a sample from a distribution parameterization, e.g. VAE moments",
        ("stochastic_sampling",),
        ("distribution", "seed_input"),
    ),
    (
        "patchify",
        "Fold spatial/temporal axes into a token sequence",
        ("tensor_patchify",),
        (
            "spatial_patch_size",
            "temporal_patch_size",
            "input_layout",
            "output_layout",
            "channel_order",
        ),
    ),
    (
        "unpatchify",
        "Unfold a token sequence back into spatial/temporal axes",
        ("tensor_patchify",),
        (
            "spatial_patch_size",
            "temporal_patch_size",
            "input_layout",
            "output_layout",
            "channel_order",
        ),
    ),
    (
        "scheduler_step",
        "Advance an iterative solver, e.g. noise prediction to the next latent",
        ("iterative_scheduler",),
        ("scheduler_asset", "stage", "state", "timestep_input"),
    ),
    ("concat", "Join several producers along one axis", ("tensor_concat",), ("axis",)),
    (
        "slice",
        "Select a sub-range of one axis",
        ("tensor_slice",),
        ("axes", "starts", "ends", "steps"),
    ),
):
    register_transform(
        _transform,
        description=_transform_doc,
        capabilities=_transform_caps,
        allowed_parameters=_parameters,
    )

for _generator, _generator_doc, _caps, _required, _allowed in (
    (
        "empty_tensor",
        "Create an empty tensor matching a declared graph port",
        (),
        (),
        ("shape", "dynamic_axes", "fill"),
    ),
    (
        "zeros",
        "Create a zero-filled tensor",
        (),
        (),
        ("shape", "shape_from", "dtype"),
    ),
    (
        "causal_attention_mask",
        "Build an autoregressive attention mask",
        ("attention_mask_program",),
        ("sequence_input",),
        ("sequence_input", "past_state", "visible_value", "masked_value"),
    ),
    (
        "multimodal_position_ids",
        "Build one- or multi-axis position ids",
        ("position_program",),
        ("source", "axes"),
        (
            "source",
            "axes",
            "mrope_sections",
            "temporal_margin",
            "reset_spatial",
            "past_state",
        ),
    ),
    (
        "packed_sequence_layout",
        "Build packed-token indexes for one modality",
        ("packed_sequence_program",),
        ("modality",),
        ("modality", "source", "layout", "understanding_prefix", "index_kind"),
    ),
    (
        "scheduler_timesteps",
        "Materialize timesteps from an iterative-stage scheduler",
        ("iterative_scheduler",),
        ("stage",),
        ("stage", "modality"),
    ),
    (
        "action_domain_ids",
        "Map an action-domain semantic name to projection-bank ids",
        ("action_domain_program",),
        ("domain_input",),
        ("domain_input", "default", "domain_map", "padded_dimension"),
    ),
):
    register_generated_input(
        _generator,
        description=_generator_doc,
        capabilities=_caps,
        required_parameters=_required,
        allowed_parameters=_allowed,
    )

for _state, _state_doc in (
    ("kv_cache", "Autoregressive key/value cache"),
    ("diffusion_latent", "Loop-carried diffusion latent"),
    ("action_state", "Loop-carried action trajectory"),
    ("recurrent", "Generic recurrent tensor state"),
):
    register_state(_state, description=_state_doc)


# ---------------------------------------------------------------------------
# Endpoints and typed graph ports
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, order=True)
class PipelinePort:
    """An endpoint: one graph port of one component.

    ``"decoder.logits"`` denotes port ``logits`` of component ``decoder``.
    Component names may not contain :data:`ENDPOINT_SEPARATOR`, so the
    qualified form is parsed by splitting on the first separator only; ONNX
    value names containing dots (``past_key_values.0.key``) round-trip
    unchanged.
    """

    component: str
    port: str

    def __post_init__(self) -> None:
        _validate_component_name(self.component)
        _validate_port_name(self.port, self.component)

    @property
    def qualified(self) -> str:
        """``"component.port"``."""
        return f"{self.component}{ENDPOINT_SEPARATOR}{self.port}"

    def __str__(self) -> str:
        return self.qualified

    @classmethod
    def parse(cls, endpoint: str | PipelinePort) -> PipelinePort:
        """Parse ``"component.port"`` (or pass through a ``PipelinePort``)."""
        if isinstance(endpoint, PipelinePort):
            return endpoint
        if not isinstance(endpoint, str) or ENDPOINT_SEPARATOR not in endpoint:
            raise PipelineValidationError(
                f"Endpoint {endpoint!r} must have the form "
                f"'component{ENDPOINT_SEPARATOR}port'."
            )
        component, _, port = endpoint.partition(ENDPOINT_SEPARATOR)
        return cls(component, port)


@dataclasses.dataclass(frozen=True)
class TensorSpec:
    """The declared type of one graph port.

    Attributes:
        name: The ONNX value name.
        dtype: :class:`onnx_ir.DataType` name, e.g. ``"FLOAT"``.
        shape: One entry per dimension. ``int`` for a static dimension,
            ``str`` for a named symbolic dimension, ``None`` for an anonymous
            dynamic dimension.  Symbolic names are *never* compared across
            components — only the rank and concrete dimensions are.
    """

    name: str
    dtype: str
    shape: tuple[int | str | None, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise PipelineValidationError("Tensor spec name must be a non-blank string.")
        try:
            ir.DataType[self.dtype]
        except KeyError as error:
            raise PipelineValidationError(
                f"Unknown dtype {self.dtype!r} for port {self.name!r}."
            ) from error
        dims: list[int | str | None] = []
        for dim in self.shape:
            if dim is None or isinstance(dim, str):
                dims.append(dim)
            elif isinstance(dim, int) and not isinstance(dim, bool):
                if dim < 0:
                    raise PipelineValidationError(
                        f"Port {self.name!r} has negative static dimension {dim}."
                    )
                dims.append(int(dim))
            else:
                raise PipelineValidationError(
                    f"Port {self.name!r} has invalid dimension {dim!r}; "
                    "expected int, str, or None."
                )
        object.__setattr__(self, "shape", tuple(dims))

    @property
    def rank(self) -> int:
        """Number of dimensions."""
        return len(self.shape)

    def to_dict(self) -> dict[str, JSONValue]:
        """Return the deterministic serializable form."""
        return {"name": self.name, "dtype": self.dtype, "shape": list(self.shape)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TensorSpec:
        """Inverse of :meth:`to_dict`."""
        body = _read_mapping(data, {"name", "dtype", "shape"}, "tensor spec")
        shape = body.get("shape", [])
        if not isinstance(shape, list):
            raise PipelineValidationError("Tensor spec 'shape' must be a list.")
        return cls(str(body["name"]), str(body["dtype"]), tuple(shape))

    @classmethod
    def from_value(cls, value: ir.Value, *, component: str, direction: str) -> TensorSpec:
        """Derive a spec from an ``ir.Value`` on a component graph signature."""
        name = value.name
        if not name:
            raise PipelineValidationError(
                f"Component {component!r} has an unnamed graph {direction}; "
                "every pipeline port must be named."
            )
        if value.dtype is None:
            raise PipelineValidationError(
                f"Component {component!r} {direction} {name!r} has no dtype; "
                "pipeline components must have fully typed signatures."
            )
        if value.shape is None:
            raise PipelineValidationError(
                f"Component {component!r} {direction} {name!r} has no shape; "
                "pipeline components must have fully typed signatures."
            )
        dims: list[int | str | None] = []
        for dim in value.shape:
            if isinstance(dim, ir.SymbolicDim):
                dims.append(dim.value)
            else:
                dims.append(int(dim))
        return cls(name, value.dtype.name, tuple(dims))


def _tensor_mismatch(source: TensorSpec, target: TensorSpec) -> str | None:
    """Return a human-readable mismatch reason, or ``None`` when compatible.

    Exact dtype and rank are required.  Static dimensions are compared only
    when *both* sides are concrete ints; symbolic names are never compared
    because two graphs may legitimately use different names for the same axis.
    """
    if source.dtype != target.dtype:
        return f"dtype {source.dtype} != {target.dtype}"
    if source.rank != target.rank:
        return f"rank {source.rank} != {target.rank}"
    for axis, (left, right) in enumerate(zip(source.shape, target.shape)):
        left_static = isinstance(left, int) and not isinstance(left, bool)
        right_static = isinstance(right, int) and not isinstance(right, bool)
        if left_static and right_static and left != right:
            return f"dim {axis}: {left} != {right}"
    return None


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PipelineComponent:
    """One :class:`onnx_ir.Model` and its typed graph ports.

    A component is a *pure graph*: it has named, typed inputs and outputs and
    no runtime semantics of its own.

    Attributes:
        name: Unique component name.  Also used as the directory name when the
            package is saved, so it must be a safe single path segment.
        role: Registered role (see :func:`register_role`).
        inputs: Typed graph inputs, in graph order.
        outputs: Typed graph outputs, in graph order.
        run_on: Registered phase (see :func:`register_phase`).
        presence: Optional opaque key naming the condition under which the
            component exists at all.  The core never evaluates it.
        capabilities: Capabilities this component contributes.
        source: Optional free-form provenance string (e.g. a model id).
        config: JSON-safe, topology-relevant configuration.  Runtime concerns
            (tokenizers, preprocessing, sampling) do **not** belong here.
        metadata: JSON-safe extension bag; unknown keys are preserved verbatim
            across serialization round-trips.
        model: The wrapped graph.  Excluded from equality and serialization —
            the manifest is topology only — and ``None`` for a manifest loaded
            from disk (the graphs then live in the owning
            :class:`PipelinePackage`).
    """

    name: str
    role: str
    inputs: tuple[TensorSpec, ...] = ()
    outputs: tuple[TensorSpec, ...] = ()
    run_on: str = DEFAULT_PHASE
    presence: str | None = None
    capabilities: tuple[str, ...] = ()
    preferred_execution_providers: tuple[str, ...] = ()
    parameter_dtype: str | None = None
    source: str | None = None
    config: dict[str, JSONValue] = dataclasses.field(default_factory=dict)
    metadata: dict[str, JSONValue] = dataclasses.field(default_factory=dict)
    model: ir.Model | None = dataclasses.field(
        default=None, compare=False, repr=False, hash=False
    )

    def __post_init__(self) -> None:
        _validate_component_name(self.name)
        role_definition(self.role)
        phase_definition(self.run_on)
        if self.presence is not None:
            _validate_token(self.presence, "Presence key")
        if self.source is not None and not isinstance(self.source, str):
            raise PipelineValidationError(f"Component {self.name!r} source must be a string.")
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(
            self, "capabilities", _string_tuple(self.capabilities, f"{self.name} capabilities")
        )
        if self.parameter_dtype is not None:
            try:
                ir.DataType[self.parameter_dtype]
            except KeyError as error:
                raise PipelineValidationError(
                    f"Component {self.name!r} has unknown parameter dtype "
                    f"{self.parameter_dtype!r}."
                ) from error
        object.__setattr__(
            self,
            "preferred_execution_providers",
            _ordered_string_tuple(
                self.preferred_execution_providers,
                f"{self.name} preferred execution providers",
            ),
        )
        object.__setattr__(
            self, "config", _ensure_json_mapping(self.config, f"{self.name} config")
        )
        object.__setattr__(
            self, "metadata", _ensure_json_mapping(self.metadata, f"{self.name} metadata")
        )
        _check_unique((spec.name for spec in self.inputs), f"Component {self.name!r} input")
        _check_unique((spec.name for spec in self.outputs), f"Component {self.name!r} output")

    @classmethod
    def from_model(
        cls,
        name: str,
        model: ir.Model,
        *,
        role: str,
        run_on: str = DEFAULT_PHASE,
        presence: str | None = None,
        capabilities: Iterable[str] | None = None,
        preferred_execution_providers: Iterable[str] | None = None,
        parameter_dtype: str | None = None,
        source: str | None = None,
        config: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PipelineComponent:
        """Derive a component (and its typed ports) from an ``ir.Model``."""
        _validate_component_name(name)
        if not isinstance(model, ir.Model):
            raise PipelineValidationError(
                f"Component {name!r} must wrap an onnx_ir.Model, got {type(model).__name__}."
            )
        inputs = tuple(
            TensorSpec.from_value(value, component=name, direction="input")
            for value in model.graph.inputs
        )
        outputs = tuple(
            TensorSpec.from_value(value, component=name, direction="output")
            for value in model.graph.outputs
        )
        return cls(
            name=name,
            role=role,
            inputs=inputs,
            outputs=outputs,
            run_on=run_on,
            presence=presence,
            capabilities=_string_tuple(capabilities, f"{name} capabilities"),
            preferred_execution_providers=_ordered_string_tuple(
                preferred_execution_providers,
                f"{name} preferred execution providers",
            ),
            parameter_dtype=parameter_dtype,
            source=source,
            config=_ensure_json_mapping(config, f"{name} config"),
            metadata=_ensure_json_mapping(metadata, f"{name} metadata"),
            model=model,
        )

    def input(self, port: str) -> TensorSpec | None:
        """Return the typed input named *port*, if any."""
        return next((spec for spec in self.inputs if spec.name == port), None)

    def output(self, port: str) -> TensorSpec | None:
        """Return the typed output named *port*, if any."""
        return next((spec for spec in self.outputs if spec.name == port), None)

    def with_model(self, model: ir.Model | None) -> PipelineComponent:
        """Return a copy bound to *model* (topology unchanged)."""
        return dataclasses.replace(self, model=model)

    def to_dict(self) -> dict[str, JSONValue]:
        """Return the deterministic serializable form (topology only)."""
        data: dict[str, JSONValue] = {
            "name": self.name,
            "role": self.role,
            "run_on": self.run_on,
            "inputs": [spec.to_dict() for spec in self.inputs],
            "outputs": [spec.to_dict() for spec in self.outputs],
        }
        if self.presence is not None:
            data["presence"] = self.presence
        if self.capabilities:
            data["capabilities"] = list(self.capabilities)
        if self.preferred_execution_providers:
            data["preferred_execution_providers"] = list(self.preferred_execution_providers)
        if self.parameter_dtype is not None:
            data["parameter_dtype"] = self.parameter_dtype
        if self.source is not None:
            data["source"] = self.source
        if self.config:
            data["config"] = dict(self.config)
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PipelineComponent:
        """Inverse of :meth:`to_dict`."""
        body = _read_mapping(
            data,
            {
                "name",
                "role",
                "run_on",
                "inputs",
                "outputs",
                "presence",
                "capabilities",
                "preferred_execution_providers",
                "parameter_dtype",
                "source",
                "config",
                "metadata",
            },
            "component",
        )
        return cls(
            name=str(body["name"]),
            role=str(body["role"]),
            inputs=tuple(TensorSpec.from_dict(s) for s in body.get("inputs", [])),
            outputs=tuple(TensorSpec.from_dict(s) for s in body.get("outputs", [])),
            run_on=str(body.get("run_on", DEFAULT_PHASE)),
            presence=body.get("presence"),
            capabilities=tuple(body.get("capabilities", ())),
            preferred_execution_providers=tuple(body.get("preferred_execution_providers", ())),
            parameter_dtype=body.get("parameter_dtype"),
            source=body.get("source"),
            config=dict(body.get("config", {})),
            metadata=dict(body.get("metadata", {})),
        )


# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PipelineConnection:
    """A directed edge ``source output -> target input``.

    Fan-out is allowed (one output may feed many inputs). An input may have at
    most one initial producer and one recurrent producer, so an encoder can
    seed state that a dynamics model subsequently updates.

    Attributes:
        source: The producing component output endpoint.
        target: The consuming component input endpoint.
        recurrent: When ``True`` this is a loop-carried (state) edge: the value
            produced by iteration *n* is consumed by iteration *n + 1*.  Such
            edges are stage-scoped — both endpoints must live in the same
            looping stage — and are excluded from the acyclicity check.
        transform: Optional *kind* of transform a runtime applies on this edge,
            drawn from the transform registry (see :func:`register_transform`).
            The core neither defines nor executes transforms.  A transform may
            legitimately change dtype, rank, and shape — a VAE posterior
            parameterization sampled, normalized, and patchified into generator
            tokens; a scheduler turning a noise prediction into the next latent
            — so direct port compatibility is *not* checked for a transformed
            edge.  Endpoint existence, the single-producer rule, and the
            transform's declared runtime capabilities are still enforced.
        context: Additional component inputs/outputs consumed by a transform.
            For example, a diffusion scheduler needs both the denoiser output
            and the current loop-carried latent. Context ports do not produce
            the target on their own and therefore do not participate in the
            single-producer rule.
    """

    source: PipelinePort
    target: PipelinePort
    recurrent: bool = False
    transform: str | None = None
    context: tuple[PipelinePort, ...] = ()
    parameters: dict[str, JSONValue] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", PipelinePort.parse(self.source))
        object.__setattr__(self, "target", PipelinePort.parse(self.target))
        object.__setattr__(self, "recurrent", bool(self.recurrent))
        transform = None
        if self.transform is not None:
            _validate_token(self.transform, "Transform")
            transform = transform_definition(self.transform)
        context = tuple(PipelinePort.parse(port) for port in self.context)
        if context and self.transform is None:
            raise PipelineValidationError(
                "Connection context is only valid when a transform is declared."
            )
        _check_unique((port.qualified for port in context), "Connection context")
        object.__setattr__(self, "context", context)
        if transform is None:
            if self.parameters:
                raise PipelineValidationError(
                    "Connection parameters are only valid when a transform is declared."
                )
            parameters: dict[str, JSONValue] = {}
        else:
            parameters = _validate_registered_parameters(
                kind=transform.name,
                parameters=self.parameters,
                required=transform.required_parameters,
                allowed=transform.allowed_parameters,
                context="Transform",
            )
        object.__setattr__(self, "parameters", parameters)

    @property
    def transform_capabilities(self) -> tuple[str, ...]:
        """Runtime capabilities required by this edge's transform, if any."""
        if self.transform is None:
            return ()
        return transform_definition(self.transform).capabilities

    @property
    def sort_key(self) -> tuple[str, str, bool, str, tuple[str, ...]]:
        """Deterministic ordering key."""
        return (
            self.target.qualified,
            self.source.qualified,
            self.recurrent,
            self.transform or "",
            tuple(port.qualified for port in self.context),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        """Return the deterministic serializable form."""
        data: dict[str, JSONValue] = {
            "source": self.source.qualified,
            "target": self.target.qualified,
        }
        if self.recurrent:
            data["recurrent"] = True
        if self.transform is not None:
            data["transform"] = self.transform
        if self.context:
            data["context"] = [port.qualified for port in self.context]
        if self.parameters:
            data["parameters"] = dict(self.parameters)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PipelineConnection:
        """Inverse of :meth:`to_dict`."""
        body = _read_mapping(
            data,
            {"source", "target", "recurrent", "transform", "context", "parameters"},
            "connection",
        )
        return cls(
            source=PipelinePort.parse(str(body["source"])),
            target=PipelinePort.parse(str(body["target"])),
            recurrent=bool(body.get("recurrent", False)),
            transform=body.get("transform"),
            context=tuple(PipelinePort.parse(str(port)) for port in body.get("context", ())),
            parameters=dict(body.get("parameters", {})),
        )


# ---------------------------------------------------------------------------
# Input sources and public outputs
# ---------------------------------------------------------------------------


class InputSource:
    """The exhaustive set of ways a component input can be satisfied.

    Every graph input must be classified by exactly one of these.  The core
    never assumes that an unconnected input is caller-supplied — that must be
    declared, so that a forgotten wire is an error rather than a silent
    external input.
    """

    #: Produced by another component in the same invocation (a connection).
    DATAFLOW = "dataflow"
    #: Supplied by the caller of the pipeline.
    EXTERNAL = "external"
    #: Produced by the runtime harness per invocation (e.g. counters, ids).
    GENERATED = "generated"
    #: Carried across iterations (a recurrent connection, or runtime-owned state).
    STATEFUL = "stateful"
    #: Filled from a constant declared in the manifest.
    DEFAULTED = "defaulted"

    #: Kinds that can be declared on a :class:`PipelineInput`.
    DECLARABLE = (EXTERNAL, GENERATED, STATEFUL, DEFAULTED)
    ALL = (DATAFLOW, EXTERNAL, GENERATED, STATEFUL, DEFAULTED)


@dataclasses.dataclass(frozen=True)
class PipelineProfile:
    """Versioned runtime semantic profile implemented by the exporter."""

    name: str
    version: str

    def __post_init__(self) -> None:
        _validate_token(self.name, "Profile")
        _parse_version_format(self.version, "Profile")

    def to_dict(self) -> dict[str, JSONValue]:
        return {"name": self.name, "version": self.version}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PipelineProfile:
        body = _read_mapping(data, {"name", "version"}, "profile")
        return cls(name=str(body["name"]), version=str(body["version"]))


@dataclasses.dataclass(frozen=True)
class GeneratedInputRule:
    """A concrete registered program for one runtime-generated input."""

    kind: str
    parameters: dict[str, JSONValue] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        definition = generated_input_definition(self.kind)
        parameters = _validate_registered_parameters(
            kind=self.kind,
            parameters=self.parameters,
            required=definition.required_parameters,
            allowed=definition.allowed_parameters,
            context="Generated input",
        )
        object.__setattr__(self, "parameters", parameters)

    @property
    def capabilities(self) -> tuple[str, ...]:
        return generated_input_definition(self.kind).capabilities

    def to_dict(self) -> dict[str, JSONValue]:
        data: dict[str, JSONValue] = {"kind": self.kind}
        if self.parameters:
            data["parameters"] = dict(self.parameters)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GeneratedInputRule:
        body = _read_mapping(data, {"kind", "parameters"}, "generated input rule")
        return cls(
            kind=str(body["kind"]),
            parameters=dict(body.get("parameters", {})),
        )


@dataclasses.dataclass(frozen=True)
class PipelineState:
    """Explicit lifecycle contract for one recurrent connection."""

    name: str
    kind: str
    input: PipelinePort
    output: PipelinePort
    lifetime: str
    release_after: str
    sequence_axis: int | None = None
    metadata: dict[str, JSONValue] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_token(self.name, "State")
        state_definition(self.kind)
        object.__setattr__(self, "input", PipelinePort.parse(self.input))
        object.__setattr__(self, "output", PipelinePort.parse(self.output))
        if self.lifetime not in {"iteration", "sequence", "request", "session"}:
            raise PipelineValidationError(
                f"State {self.name!r} has unsupported lifetime {self.lifetime!r}."
            )
        _validate_token(self.release_after, "State release stage")
        if self.sequence_axis is not None and self.sequence_axis < 0:
            raise PipelineValidationError(
                f"State {self.name!r} sequence_axis must be non-negative."
            )
        object.__setattr__(
            self,
            "metadata",
            _ensure_json_mapping(self.metadata, f"State {self.name!r} metadata"),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        data: dict[str, JSONValue] = {
            "name": self.name,
            "kind": self.kind,
            "input": self.input.qualified,
            "output": self.output.qualified,
            "lifetime": self.lifetime,
            "release_after": self.release_after,
        }
        if self.sequence_axis is not None:
            data["sequence_axis"] = self.sequence_axis
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PipelineState:
        body = _read_mapping(
            data,
            {
                "name",
                "kind",
                "input",
                "output",
                "lifetime",
                "release_after",
                "sequence_axis",
                "metadata",
            },
            "state",
        )
        return cls(
            name=str(body["name"]),
            kind=str(body["kind"]),
            input=PipelinePort.parse(str(body["input"])),
            output=PipelinePort.parse(str(body["output"])),
            lifetime=str(body["lifetime"]),
            release_after=str(body["release_after"]),
            sequence_axis=body.get("sequence_axis"),
            metadata=dict(body.get("metadata", {})),
        )


@dataclasses.dataclass(frozen=True)
class PipelineInput:
    """A declared source for a component input that no connection feeds.

    Attributes:
        port: The component input endpoint.
        kind: One of :data:`InputSource.DECLARABLE`.
        value: The constant for ``defaulted`` inputs; must be JSON-safe and
            must be ``None`` for every other kind.
        alias: Optional pipeline-level name for ``external`` inputs.
    """

    port: PipelinePort
    kind: str
    value: JSONValue = None
    alias: str | None = None
    semantic: str | None = None
    required: bool = True
    presence: str | None = None
    generator: GeneratedInputRule | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "port", PipelinePort.parse(self.port))
        if self.kind not in InputSource.DECLARABLE:
            allowed = ", ".join(InputSource.DECLARABLE)
            raise PipelineValidationError(
                f"Input {self.port.qualified!r} has unknown source kind "
                f"{self.kind!r}; expected one of: {allowed}."
            )
        if self.kind == InputSource.DEFAULTED:
            if self.value is None:
                raise PipelineValidationError(
                    f"Defaulted input {self.port.qualified!r} requires a value; "
                    "use kind 'external' or 'generated' if there is no constant."
                )
            object.__setattr__(
                self,
                "value",
                _ensure_json_value(self.value, f"Default for {self.port.qualified!r}"),
            )
            object.__setattr__(self, "required", False)
        elif self.value is not None:
            raise PipelineValidationError(
                f"Input {self.port.qualified!r} of kind {self.kind!r} must not "
                "carry a default value."
            )
        if self.kind == InputSource.GENERATED:
            if self.generator is None:
                raise PipelineValidationError(
                    f"Generated input {self.port.qualified!r} requires a generation rule."
                )
        elif self.generator is not None:
            raise PipelineValidationError(
                f"Only generated inputs may declare a generation rule; "
                f"{self.port.qualified!r} is {self.kind!r}."
            )
        if self.alias is not None:
            if self.kind != InputSource.EXTERNAL:
                raise PipelineValidationError(
                    f"Only external inputs may declare an alias; "
                    f"{self.port.qualified!r} is {self.kind!r}."
                )
            _validate_token(self.alias, "Input alias")
        if self.semantic is not None:
            _validate_token(self.semantic, "Input semantic")
        if self.presence is not None:
            _validate_token(self.presence, "Input presence")
        object.__setattr__(self, "required", bool(self.required))

    @property
    def name(self) -> str:
        """The pipeline-level name of this input."""
        return self.alias or self.port.port

    def to_dict(self) -> dict[str, JSONValue]:
        """Return the deterministic serializable form."""
        data: dict[str, JSONValue] = {"port": self.port.qualified, "kind": self.kind}
        data["required"] = self.required
        if self.kind == InputSource.DEFAULTED:
            data["value"] = self.value
        if self.alias is not None:
            data["alias"] = self.alias
        if self.semantic is not None:
            data["semantic"] = self.semantic
        if self.presence is not None:
            data["presence"] = self.presence
        if self.generator is not None:
            data["generator"] = self.generator.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PipelineInput:
        """Inverse of :meth:`to_dict`."""
        body = _read_mapping(
            data,
            {
                "port",
                "kind",
                "value",
                "alias",
                "semantic",
                "required",
                "presence",
                "generator",
            },
            "input",
        )
        return cls(
            port=PipelinePort.parse(str(body["port"])),
            kind=str(body["kind"]),
            value=body.get("value"),
            alias=body.get("alias"),
            semantic=body.get("semantic"),
            required=bool(body.get("required", True)),
            presence=body.get("presence"),
            generator=(
                GeneratedInputRule.from_dict(body["generator"])
                if body.get("generator") is not None
                else None
            ),
        )


@dataclasses.dataclass(frozen=True)
class PipelineOutput:
    """A component output or final recurrent state exposed as a result."""

    port: PipelinePort | None = None
    alias: str | None = None
    state: str | None = None

    def __post_init__(self) -> None:
        if (self.port is None) == (self.state is None):
            raise PipelineValidationError(
                "Pipeline output must reference exactly one component port or state."
            )
        if self.port is not None:
            object.__setattr__(self, "port", PipelinePort.parse(self.port))
        if self.state is not None:
            _validate_token(self.state, "Output state")
        if self.alias is not None:
            _validate_token(self.alias, "Output alias")

    @property
    def name(self) -> str:
        """The pipeline-level name of this output."""
        if self.alias is not None:
            return self.alias
        if self.port is not None:
            return self.port.port
        assert self.state is not None
        return self.state

    def to_dict(self) -> dict[str, JSONValue]:
        """Return the deterministic serializable form."""
        data: dict[str, JSONValue] = {}
        if self.port is not None:
            data["port"] = self.port.qualified
        else:
            data["state"] = self.state
        if self.alias is not None:
            data["alias"] = self.alias
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PipelineOutput:
        """Inverse of :meth:`to_dict`."""
        body = _read_mapping(data, {"port", "state", "alias"}, "output")
        return cls(
            port=(
                PipelinePort.parse(str(body["port"])) if body.get("port") is not None else None
            ),
            state=body.get("state"),
            alias=body.get("alias"),
        )


@dataclasses.dataclass(frozen=True)
class PipelineAsset:
    """An opaque runtime file that ships next to the component graphs.

    Tokenizers, scheduler configs, and processor configs are runtime assets:
    the topology core copies them and records *where* they live, and never
    opens, parses, or interprets their contents.  Only the destination is part
    of the manifest — machine-local source paths never are.

    Attributes:
        path: ``/``-separated destination relative to the package directory.
        required: Whether :meth:`PipelinePackage.load` must find the file.
    """

    path: str
    required: bool = True

    def __post_init__(self) -> None:
        _validate_asset_path(self.path)
        object.__setattr__(self, "required", bool(self.required))

    def to_dict(self) -> dict[str, JSONValue]:
        """Return the deterministic serializable form."""
        data: dict[str, JSONValue] = {"path": self.path}
        if not self.required:
            data["required"] = False
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PipelineAsset:
        """Inverse of :meth:`to_dict`."""
        body = _read_mapping(data, {"path", "required"}, "asset")
        return cls(path=str(body["path"]), required=bool(body.get("required", True)))


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PipelineStage:
    """A group of components executed under one strategy.

    A stage says what kind of control flow applies to its components.
    Strategy-specific ``options`` are validated by the strategy registry and
    carry runtime controls such as scheduler assets, sampling, and stopping.

    Attributes:
        name: Unique stage name.
        kind: Registered strategy (see :func:`register_strategy`).
        components: Names of member components (order is declaration order).
        run_on: Registered phase.
        options: JSON-safe, registry-validated strategy parameters.
        capabilities: Capabilities this stage contributes.  A stage owning a
            recurrent connection must contribute
            :data:`LOOP_CARRIED_STATE_CAPABILITY`.
        metadata: JSON-safe extension bag preserved across round-trips.
    """

    name: str
    kind: str
    components: tuple[str, ...]
    run_on: str = DEFAULT_PHASE
    options: dict[str, JSONValue] = dataclasses.field(default_factory=dict)
    capabilities: tuple[str, ...] = ()
    metadata: dict[str, JSONValue] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_token(self.name, "Stage")
        strategy = strategy_definition(self.kind)
        phase_definition(self.run_on)
        components = tuple(self.components)
        if not components:
            raise PipelineValidationError(f"Stage {self.name!r} must contain a component.")
        for component in components:
            _validate_component_name(component)
        _check_unique(components, f"Stage {self.name!r} component")
        object.__setattr__(self, "components", components)
        object.__setattr__(
            self,
            "capabilities",
            _string_tuple(self.capabilities, f"Stage {self.name!r} capabilities"),
        )
        options = _validate_registered_parameters(
            kind=self.kind,
            parameters=self.options,
            required=strategy.required_options,
            allowed=strategy.allowed_options,
            context=f"Stage {self.name!r}",
        )
        object.__setattr__(self, "options", options)
        object.__setattr__(
            self,
            "metadata",
            _ensure_json_mapping(self.metadata, f"Stage {self.name!r} metadata"),
        )

    @property
    def strategy(self) -> StrategyDefinition:
        """The registered strategy definition for :attr:`kind`."""
        return strategy_definition(self.kind)

    @property
    def supports_loop_carried_state(self) -> bool:
        """Whether this stage may own recurrent connections."""
        return self.strategy.loop_carried_state

    def to_dict(self) -> dict[str, JSONValue]:
        """Return the deterministic serializable form."""
        data: dict[str, JSONValue] = {
            "name": self.name,
            "kind": self.kind,
            "components": list(self.components),
            "run_on": self.run_on,
        }
        if self.options:
            data["options"] = dict(self.options)
        if self.capabilities:
            data["capabilities"] = list(self.capabilities)
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PipelineStage:
        """Inverse of :meth:`to_dict`."""
        body = _read_mapping(
            data,
            {"name", "kind", "components", "run_on", "options", "capabilities", "metadata"},
            "stage",
        )
        return cls(
            name=str(body["name"]),
            kind=str(body["kind"]),
            components=tuple(body.get("components", ())),
            run_on=str(body.get("run_on", DEFAULT_PHASE)),
            options=dict(body.get("options", {})),
            capabilities=tuple(body.get("capabilities", ())),
            metadata=dict(body.get("metadata", {})),
        )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PipelineManifest:
    """The topology of a composed pipeline.

    The manifest is fully validated on construction, so any instance is
    structurally sound.  It is deterministic: components, connections, and
    inputs are canonically ordered, while stages and public outputs keep their
    declaration order because that order is meaningful.

    A manifest with :attr:`profile` is an executable contract: every declared
    input has semantics, generated inputs name a registered program, recurrent
    edges have state lifecycle, iterative/autoregressive stages carry control
    parameters, assets are explicit, and components provide dtype/EP hints.
    """

    components: tuple[PipelineComponent, ...] = ()
    connections: tuple[PipelineConnection, ...] = ()
    stages: tuple[PipelineStage, ...] = ()
    inputs: tuple[PipelineInput, ...] = ()
    outputs: tuple[PipelineOutput, ...] = ()
    assets: tuple[PipelineAsset, ...] = ()
    states: tuple[PipelineState, ...] = ()
    profile: PipelineProfile | None = None
    required_capabilities: tuple[str, ...] = ()
    schema_version: str = PIPELINE_SCHEMA_VERSION
    metadata: dict[str, JSONValue] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "components", tuple(sorted(self.components, key=lambda c: c.name))
        )
        object.__setattr__(
            self, "connections", tuple(sorted(self.connections, key=lambda c: c.sort_key))
        )
        object.__setattr__(
            self, "inputs", tuple(sorted(self.inputs, key=lambda i: i.port.qualified))
        )
        object.__setattr__(self, "assets", tuple(sorted(self.assets, key=lambda a: a.path)))
        object.__setattr__(self, "states", tuple(sorted(self.states, key=lambda s: s.name)))
        object.__setattr__(self, "stages", tuple(self.stages))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(
            self,
            "required_capabilities",
            _string_tuple(self.required_capabilities, "Required capabilities"),
        )
        object.__setattr__(
            self, "metadata", _ensure_json_mapping(self.metadata, "Manifest metadata")
        )
        _parse_schema_version(self.schema_version)
        self.validate()

    # -- Lookup ------------------------------------------------------------

    @property
    def component_names(self) -> tuple[str, ...]:
        """Component names in canonical (sorted) order."""
        return tuple(component.name for component in self.components)

    def component(self, name: str) -> PipelineComponent:
        """Return the component named *name*."""
        for component in self.components:
            if component.name == name:
                return component
        raise KeyError(name)

    @property
    def external_inputs(self) -> tuple[PipelineInput, ...]:
        """Inputs the caller must supply."""
        return tuple(i for i in self.inputs if i.kind == InputSource.EXTERNAL)

    def inputs_of_kind(self, kind: str) -> tuple[PipelineInput, ...]:
        """Return declared inputs whose source classification is *kind*."""
        return tuple(i for i in self.inputs if i.kind == kind)

    def stage_of(self, component: str) -> tuple[PipelineStage, ...]:
        """Return the stages a component belongs to."""
        return tuple(stage for stage in self.stages if component in stage.components)

    def source_of(self, port: PipelinePort | str) -> str:
        """Return the :class:`InputSource` classification of an input endpoint."""
        endpoint = PipelinePort.parse(port)
        for connection in self.connections:
            if connection.target == endpoint and connection.recurrent:
                return InputSource.STATEFUL
        for connection in self.connections:
            if connection.target == endpoint:
                return InputSource.DATAFLOW
        for declared in self.inputs:
            if declared.port == endpoint:
                return declared.kind
        raise KeyError(endpoint.qualified)

    def initial_source_of(self, port: PipelinePort | str) -> str:
        """Return how a possibly recurrent input receives its first value."""
        endpoint = PipelinePort.parse(port)
        for connection in self.connections:
            if connection.target == endpoint and not connection.recurrent:
                return InputSource.DATAFLOW
        for declared in self.inputs:
            if declared.port == endpoint:
                return declared.kind
        raise KeyError(endpoint.qualified)

    # -- Validation --------------------------------------------------------

    def validate(self) -> None:
        """Run full structural validation (called on construction)."""
        by_name = {component.name: component for component in self.components}
        if len(by_name) != len(self.components):
            _check_unique((c.name for c in self.components), "Component")
        _check_unique_casefold(
            (component.name for component in self.components),
            "Component",
        )

        self._validate_connections(by_name)
        self._validate_stages(by_name)
        self._validate_input_sources(by_name)
        self._validate_states(by_name)
        self._validate_outputs(by_name)
        self._validate_assets()
        self._validate_runtime_references(by_name)
        self._validate_acyclic()
        self._validate_capabilities()
        self._validate_profile_contract()

    def _endpoint_spec(
        self,
        by_name: Mapping[str, PipelineComponent],
        port: PipelinePort,
        direction: str,
        context: str,
    ) -> TensorSpec:
        component = by_name.get(port.component)
        if component is None:
            known = ", ".join(sorted(by_name)) or "<none>"
            raise PipelineValidationError(
                f"{context} references unknown component {port.component!r}. "
                f"Known components: {known}."
            )
        spec = (
            component.input(port.port) if direction == "input" else component.output(port.port)
        )
        if spec is None:
            available = ", ".join(
                s.name
                for s in (component.inputs if direction == "input" else component.outputs)
            )
            raise PipelineValidationError(
                f"{context} references unknown {direction} {port.port!r} on component "
                f"{port.component!r}. Available {direction}s: {available or '<none>'}."
            )
        return spec

    def _validate_connections(self, by_name: Mapping[str, PipelineComponent]) -> None:
        producers: dict[tuple[str, bool], PipelineConnection] = {}
        for connection in self.connections:
            context = f"Connection {connection.source} -> {connection.target}"
            source_spec = self._endpoint_spec(by_name, connection.source, "output", context)
            target_spec = self._endpoint_spec(by_name, connection.target, "input", context)
            producer_key = (connection.target.qualified, connection.recurrent)
            existing = producers.get(producer_key)
            if existing is not None:
                lifecycle = "recurrent" if connection.recurrent else "initial"
                raise PipelineValidationError(
                    f"Input {connection.target} has more than one {lifecycle} producer "
                    f"({existing.source} and {connection.source}); an input accepts "
                    "at most one producer for each lifecycle phase "
                    "(fan-out on outputs is fine)."
                )
            producers[producer_key] = connection
            for context_port in connection.context:
                component = by_name.get(context_port.component)
                if component is None:
                    raise PipelineValidationError(
                        f"{context} transform context references unknown component "
                        f"{context_port.component!r}."
                    )
                if (
                    component.input(context_port.port) is None
                    and component.output(context_port.port) is None
                ):
                    raise PipelineValidationError(
                        f"{context} transform context references unknown port "
                        f"{context_port.qualified!r}."
                    )
            if connection.transform is not None:
                # A registered transform may legitimately change dtype, rank,
                # and shape, and the core does not execute it, so port
                # compatibility is not assumed here. The transform kind itself
                # was validated against the registry on construction, and its
                # runtime capabilities are checked in _validate_capabilities.
                continue
            mismatch = _tensor_mismatch(source_spec, target_spec)
            if mismatch is not None:
                raise PipelineValidationError(
                    f"{context} is incompatible: {mismatch} "
                    f"(source {source_spec.dtype}{list(source_spec.shape)}, "
                    f"target {target_spec.dtype}{list(target_spec.shape)}). "
                    "Declare a transform if a runtime adapts this edge."
                )

    def _validate_stages(self, by_name: Mapping[str, PipelineComponent]) -> None:
        _check_unique((stage.name for stage in self.stages), "Stage")
        staged: set[str] = set()
        for stage in self.stages:
            for name in stage.components:
                component = by_name.get(name)
                if component is None:
                    known = ", ".join(sorted(by_name)) or "<none>"
                    raise PipelineValidationError(
                        f"Stage {stage.name!r} references unknown component {name!r}. "
                        f"Known components: {known}."
                    )
                if (
                    stage.run_on != DEFAULT_PHASE
                    and component.run_on != DEFAULT_PHASE
                    and stage.run_on != component.run_on
                ):
                    raise PipelineValidationError(
                        f"Component {name!r} runs on {component.run_on!r} but stage "
                        f"{stage.name!r} runs on {stage.run_on!r}; the component could "
                        "never execute in this stage."
                    )
                staged.add(name)
            if stage.kind == "on_demand" and not any(
                by_name[name].presence is not None for name in stage.components
            ):
                raise PipelineValidationError(
                    f"On-demand stage {stage.name!r} has no component presence "
                    "condition, so a runtime cannot decide when to execute it."
                )
        missing = sorted(set(by_name) - staged)
        if missing:
            names = ", ".join(repr(n) for n in missing)
            raise PipelineValidationError(
                f"Component(s) {names} belong to no declared stage; every component "
                "must be reachable through a stage."
            )

        stages_by_component: dict[str, list[PipelineStage]] = {}
        for stage in self.stages:
            for name in stage.components:
                stages_by_component.setdefault(name, []).append(stage)

        for connection in self.connections:
            if not connection.recurrent:
                continue
            source_stages = stages_by_component.get(connection.source.component, [])
            target_stages = stages_by_component.get(connection.target.component, [])
            shared = [
                stage
                for stage in source_stages
                if stage in target_stages and stage.supports_loop_carried_state
            ]
            if not shared:
                kinds = ", ".join(
                    sorted(k for k, v in _STRATEGIES.items() if v.loop_carried_state)
                )
                raise PipelineValidationError(
                    f"Recurrent connection {connection.source} -> {connection.target} "
                    "must be scoped to a single stage that both components belong to "
                    f"and whose strategy supports loop-carried state ({kinds})."
                )
            without_capability = [
                stage.name
                for stage in shared
                if LOOP_CARRIED_STATE_CAPABILITY not in stage.capabilities
            ]
            if len(without_capability) == len(shared):
                raise PipelineValidationError(
                    f"Recurrent connection {connection.source} -> {connection.target} "
                    f"requires its stage to contribute the "
                    f"{LOOP_CARRIED_STATE_CAPABILITY!r} capability; "
                    f"stage(s) {', '.join(repr(s) for s in without_capability)} do not."
                )

    def _validate_input_sources(self, by_name: Mapping[str, PipelineComponent]) -> None:
        declared: dict[str, PipelineInput] = {}
        for entry in self.inputs:
            context = f"Declared input {entry.port}"
            self._endpoint_spec(by_name, entry.port, "input", context)
            if entry.port.qualified in declared:
                raise PipelineValidationError(
                    f"{context} is declared more than once; each input needs exactly "
                    "one source classification."
                )
            declared[entry.port.qualified] = entry

        aliases = [entry.name for entry in self.external_inputs]
        _check_unique(aliases, "External input name")

        initial_connections = {
            connection.target.qualified: connection
            for connection in self.connections
            if not connection.recurrent
        }
        recurrent_connections = {
            connection.target.qualified: connection
            for connection in self.connections
            if connection.recurrent
        }
        for component in self.components:
            for spec in component.inputs:
                endpoint = PipelinePort(component.name, spec.name)
                key = endpoint.qualified
                initial = initial_connections.get(key)
                recurrent = recurrent_connections.get(key)
                entry = declared.get(key)
                if initial is not None and entry is not None:
                    raise PipelineValidationError(
                        f"Input {key} is both initialized by {initial.source} and "
                        f"declared as {entry.kind!r}; exactly one initial source is allowed."
                    )
                if initial is None and entry is None:
                    kinds = ", ".join(InputSource.DECLARABLE)
                    prefix = "Recurrent input" if recurrent is not None else "Input"
                    raise PipelineValidationError(
                        f"{prefix} {key} has no initial source. Connect it, or declare it as one "
                        f"of: {kinds}. Unconnected inputs are never assumed to be "
                        "external."
                    )

    def _validate_outputs(self, by_name: Mapping[str, PipelineComponent]) -> None:
        state_names = {state.name for state in self.states}
        for output in self.outputs:
            if output.port is not None:
                self._endpoint_spec(
                    by_name,
                    output.port,
                    "output",
                    f"Public output {output.port}",
                )
            elif output.state not in state_names:
                raise PipelineValidationError(
                    f"Public output references unknown state {output.state!r}."
                )
        _check_unique((output.name for output in self.outputs), "Public output name")

    def _validate_states(self, by_name: Mapping[str, PipelineComponent]) -> None:
        _check_unique((state.name for state in self.states), "State")
        _check_unique((state.input.qualified for state in self.states), "State input")
        stages = {stage.name for stage in self.stages}
        recurrent = {
            (connection.source.qualified, connection.target.qualified)
            for connection in self.connections
            if connection.recurrent
        }
        declared: set[tuple[str, str]] = set()
        for state in self.states:
            input_spec = self._endpoint_spec(
                by_name,
                state.input,
                "input",
                f"State {state.name!r}",
            )
            self._endpoint_spec(
                by_name,
                state.output,
                "output",
                f"State {state.name!r}",
            )
            edge = (state.output.qualified, state.input.qualified)
            if edge not in recurrent:
                raise PipelineValidationError(
                    f"State {state.name!r} does not match a recurrent connection "
                    f"{state.output} -> {state.input}."
                )
            if state.release_after not in stages:
                raise PipelineValidationError(
                    f"State {state.name!r} releases after unknown stage "
                    f"{state.release_after!r}."
                )
            if state.sequence_axis is not None and state.sequence_axis >= input_spec.rank:
                raise PipelineValidationError(
                    f"State {state.name!r} sequence_axis {state.sequence_axis} "
                    f"is outside input rank {input_spec.rank}."
                )
            declared.add(edge)
        missing = sorted(recurrent - declared)
        if self.profile is not None and missing:
            raise PipelineValidationError(
                "Every recurrent connection requires an explicit state lifecycle; "
                f"missing declarations for {missing}."
            )

    def _validate_assets(self) -> None:
        _check_unique((asset.path for asset in self.assets), "Asset destination")
        _check_unique_casefold(
            (asset.path for asset in self.assets),
            "Asset destination",
        )
        reserved = {PIPELINE_FILENAME, *self.component_file_layout().values()}
        reserved |= {f"{path}.data" for path in self.component_file_layout().values()}
        reserved_casefold = {path.casefold() for path in reserved}
        for asset in self.assets:
            if asset.path.casefold() in reserved_casefold:
                raise PipelineValidationError(
                    f"Asset destination {asset.path!r} collides with a file written by "
                    "the package itself."
                )

    def _validate_runtime_references(
        self,
        by_name: Mapping[str, PipelineComponent],
    ) -> None:
        state_names = {state.name for state in self.states}
        stage_names = {stage.name for stage in self.stages}
        asset_paths = {asset.path for asset in self.assets}

        def validate_port(value: Any, context: str) -> None:
            if not isinstance(value, str) or "." not in value:
                raise PipelineValidationError(
                    f"{context} must reference a qualified component port."
                )
            port = PipelinePort.parse(value)
            component = by_name.get(port.component)
            if component is None or (
                component.input(port.port) is None and component.output(port.port) is None
            ):
                raise PipelineValidationError(f"{context} references unknown port {value!r}.")

        for connection in self.connections:
            parameters = connection.parameters
            if "state" in parameters and parameters["state"] not in state_names:
                raise PipelineValidationError(
                    f"Connection {connection.source} -> {connection.target} references "
                    f"unknown state {parameters['state']!r}."
                )
            if "stage" in parameters and parameters["stage"] not in stage_names:
                raise PipelineValidationError(
                    f"Connection {connection.source} -> {connection.target} references "
                    f"unknown stage {parameters['stage']!r}."
                )
            if (
                "scheduler_asset" in parameters
                and parameters["scheduler_asset"] not in asset_paths
            ):
                raise PipelineValidationError(
                    f"Connection {connection.source} -> {connection.target} references "
                    f"undeclared scheduler asset {parameters['scheduler_asset']!r}."
                )
            if "timestep_input" in parameters:
                validate_port(
                    parameters["timestep_input"],
                    f"Connection {connection.source} timestep_input",
                )

        for entry in self.inputs:
            if entry.generator is None:
                continue
            parameters = entry.generator.parameters
            if "stage" in parameters and parameters["stage"] not in stage_names:
                raise PipelineValidationError(
                    f"Generated input {entry.port} references unknown stage "
                    f"{parameters['stage']!r}."
                )
            for key in ("source", "sequence_input"):
                if key in parameters:
                    validate_port(parameters[key], f"Generated input {entry.port} {key}")
            if "past_state" in parameters:
                values = parameters["past_state"]
                if not isinstance(values, list) or not all(
                    isinstance(value, str) and value in state_names for value in values
                ):
                    raise PipelineValidationError(
                        f"Generated input {entry.port} references unknown past state."
                    )
            if "dynamic_axes" in parameters:
                dynamic_axes = parameters["dynamic_axes"]
                input_spec = self._endpoint_spec(
                    by_name,
                    entry.port,
                    "input",
                    f"Generated input {entry.port}",
                )
                symbolic_dims = {dim for dim in input_spec.shape if isinstance(dim, str)}
                if (
                    not isinstance(dynamic_axes, dict)
                    or not set(dynamic_axes) <= symbolic_dims
                ):
                    raise PipelineValidationError(
                        f"Generated input {entry.port} references unknown dynamic axis."
                    )

    def component_file_layout(self) -> dict[str, str]:
        """Return ``{component: relative onnx path}`` for the saved layout.

        Mirrors :meth:`ModelPackage.save`: a single-component package is stored
        flat as ``model.onnx``; otherwise each component gets its own folder.
        """
        names = self.component_names
        if len(names) == 1:
            return {names[0]: "model.onnx"}
        return {name: f"{name}/model.onnx" for name in names}

    @property
    def required_assets(self) -> tuple[PipelineAsset, ...]:
        """Assets that must be present for the package to load."""
        return tuple(asset for asset in self.assets if asset.required)

    def _validate_acyclic(self) -> None:
        """Non-recurrent edges must form a DAG (hence acyclic within a stage)."""
        edges: dict[str, set[str]] = {c.name: set() for c in self.components}
        for connection in self.connections:
            if connection.recurrent:
                continue
            if connection.source.component == connection.target.component:
                raise PipelineValidationError(
                    f"Connection {connection.source} -> {connection.target} makes "
                    f"component {connection.source.component!r} depend on itself; "
                    "mark the edge recurrent if it is loop-carried state."
                )
            edges[connection.source.component].add(connection.target.component)

        visiting: set[str] = set()
        visited: set[str] = set()
        stack: list[str] = []

        def visit(node: str) -> None:
            if node in visited:
                return
            if node in visiting:
                cycle = " -> ".join([*stack[stack.index(node) :], node])
                raise PipelineValidationError(
                    f"Pipeline has a cycle in non-recurrent connections: {cycle}. "
                    "Mark loop-carried edges recurrent and scope them to an "
                    "iterative stage."
                )
            visiting.add(node)
            stack.append(node)
            for successor in sorted(edges[node]):
                visit(successor)
            stack.pop()
            visiting.discard(node)
            visited.add(node)

        for name in sorted(edges):
            visit(name)

    def _validate_capabilities(self) -> None:
        provided = set()
        for component in self.components:
            provided.update(component.capabilities)
        for stage in self.stages:
            provided.update(stage.capabilities)
        for entry in self.inputs:
            if entry.generator is None:
                continue
            required = entry.generator.capabilities
            provided.update(required)
            undeclared = sorted(set(required) - set(self.required_capabilities))
            if undeclared:
                raise PipelineValidationError(
                    f"Generated input {entry.port} requires undeclared capabilities "
                    f"{undeclared}."
                )

        # A transformed edge contributes the capabilities its transform kind
        # declares, and the manifest must require them explicitly so that the
        # runtime obligation is visible without inspecting every connection.
        for connection in self.connections:
            required = connection.transform_capabilities
            if not required:
                continue
            provided.update(required)
            undeclared = sorted(set(required) - set(self.required_capabilities))
            if undeclared:
                names = ", ".join(repr(c) for c in undeclared)
                raise PipelineValidationError(
                    f"Connection {connection.source} -> {connection.target} uses "
                    f"transform {connection.transform!r}, which requires capabilities "
                    f"{names}; add them to the manifest's required_capabilities."
                )

        missing = sorted(set(self.required_capabilities) - provided)
        if missing:
            names = ", ".join(repr(m) for m in missing)
            raise PipelineValidationError(
                f"Required capabilities {names} are not provided by any component or stage."
            )

    def _validate_profile_contract(self) -> None:
        """Require executable semantics when a runtime profile is declared."""
        if self.profile is None:
            return
        missing_semantics = [
            entry.port.qualified for entry in self.inputs if entry.semantic is None
        ]
        if missing_semantics:
            raise PipelineValidationError(
                f"Profile {self.profile.name!r} requires semantic names for every "
                f"declared input; missing {missing_semantics}."
            )
        missing_ep_hints = [
            component.name
            for component in self.components
            if not component.preferred_execution_providers
        ]
        if missing_ep_hints:
            raise PipelineValidationError(
                f"Profile {self.profile.name!r} requires execution-provider hints "
                f"for every component; missing {missing_ep_hints}."
            )
        missing_dtypes = [
            component.name
            for component in self.components
            if component.parameter_dtype is None
        ]
        if missing_dtypes:
            raise PipelineValidationError(
                f"Profile {self.profile.name!r} requires parameter dtype for every "
                f"component; missing {missing_dtypes}."
            )
        for stage in self.stages:
            if stage.kind == "autoregressive":
                required = {"tokenizer_asset", "sampling", "stop"}
            elif stage.kind == "iterative":
                required = {"scheduler", "default_steps", "timestep", "state_inputs"}
            else:
                continue
            missing = sorted(required - set(stage.options))
            if missing:
                raise PipelineValidationError(
                    f"Executable profile stage {stage.name!r} ({stage.kind}) is "
                    f"missing control option(s) {missing}."
                )
        asset_paths = {asset.path for asset in self.assets}
        referenced_assets: set[str] = set()
        for stage in self.stages:
            tokenizer_asset = stage.options.get("tokenizer_asset")
            if isinstance(tokenizer_asset, str):
                referenced_assets.add(tokenizer_asset)
            scheduler = stage.options.get("scheduler")
            if isinstance(scheduler, dict):
                config_asset = scheduler.get("config_asset")
                if isinstance(config_asset, str):
                    referenced_assets.add(config_asset)
        missing_assets = sorted(referenced_assets - asset_paths)
        if missing_assets:
            raise PipelineValidationError(
                f"Profile {self.profile.name!r} references undeclared runtime assets "
                f"{missing_assets}."
            )

    def validate_models(self, models: Mapping[str, ir.Model]) -> None:
        """Check that *models* match the manifest exactly (names and signatures)."""
        manifest_names = set(self.component_names)
        model_names = set(models)
        if manifest_names != model_names:
            missing = ", ".join(repr(n) for n in sorted(manifest_names - model_names)) or "-"
            extra = ", ".join(repr(n) for n in sorted(model_names - manifest_names)) or "-"
            raise PipelineValidationError(
                f"Pipeline models do not match the manifest. Missing: {missing}. "
                f"Unexpected: {extra}."
            )
        for component in self.components:
            actual = PipelineComponent.from_model(
                component.name, models[component.name], role=component.role
            )
            if actual.inputs != component.inputs or actual.outputs != component.outputs:
                raise PipelineValidationError(
                    f"Component {component.name!r} graph signature does not match the "
                    "manifest; the manifest is out of date with the graph."
                )

    # -- Serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, JSONValue]:
        """Return the deterministic serializable form."""
        data: dict[str, JSONValue] = {
            "schema_version": self.schema_version,
            "components": [component.to_dict() for component in self.components],
            "connections": [connection.to_dict() for connection in self.connections],
            "stages": [stage.to_dict() for stage in self.stages],
            "inputs": [entry.to_dict() for entry in self.inputs],
            "outputs": [output.to_dict() for output in self.outputs],
        }
        if self.profile is not None:
            data["profile"] = self.profile.to_dict()
        if self.states:
            data["states"] = [state.to_dict() for state in self.states]
        if self.assets:
            data["assets"] = [asset.to_dict() for asset in self.assets]
        if self.required_capabilities:
            data["required_capabilities"] = list(self.required_capabilities)
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PipelineManifest:
        """Inverse of :meth:`to_dict`.

        Unknown top-level keys, unknown roles/strategies/phases, and an unknown
        schema *major* version are hard failures.  Unknown keys nested inside a
        ``metadata`` bag are preserved verbatim.
        """
        body = _read_mapping(
            data,
            {
                "schema_version",
                "components",
                "connections",
                "stages",
                "inputs",
                "outputs",
                "assets",
                "states",
                "profile",
                "required_capabilities",
                "metadata",
            },
            "manifest",
        )
        version = str(body.get("schema_version", PIPELINE_SCHEMA_VERSION))
        _parse_schema_version(version)
        return cls(
            components=tuple(
                PipelineComponent.from_dict(c) for c in body.get("components", [])
            ),
            connections=tuple(
                PipelineConnection.from_dict(c) for c in body.get("connections", [])
            ),
            stages=tuple(PipelineStage.from_dict(s) for s in body.get("stages", [])),
            inputs=tuple(PipelineInput.from_dict(i) for i in body.get("inputs", [])),
            outputs=tuple(PipelineOutput.from_dict(o) for o in body.get("outputs", [])),
            assets=tuple(PipelineAsset.from_dict(a) for a in body.get("assets", [])),
            states=tuple(PipelineState.from_dict(s) for s in body.get("states", [])),
            profile=(
                PipelineProfile.from_dict(body["profile"])
                if body.get("profile") is not None
                else None
            ),
            required_capabilities=tuple(body.get("required_capabilities", ())),
            schema_version=version,
            metadata=dict(body.get("metadata", {})),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize to a deterministic JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_json(cls, text: str) -> PipelineManifest:
        """Inverse of :meth:`to_json`."""
        return cls.from_dict(json.loads(text))


def _parse_version_format(version: str, context: str) -> tuple[int, int]:
    """Parse a non-negative ``major.minor`` version without compatibility checks."""
    if not isinstance(version, str):
        raise PipelineValidationError(f"{context} version {version!r} must be a string.")
    parts = version.split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError) as error:
        raise PipelineValidationError(
            f"{context} version {version!r} must have the form 'major.minor'."
        ) from error
    if len(parts) > 2 or major < 0 or minor < 0:
        raise PipelineValidationError(
            f"{context} version {version!r} must have the form 'major.minor'."
        )
    return major, minor


def _parse_schema_version(version: str) -> tuple[int, int]:
    """Parse and compatibility-check a pipeline schema version."""
    major, minor = _parse_version_format(version, "Schema")
    current_major = int(PIPELINE_SCHEMA_VERSION.split(".")[0])
    if major != current_major:
        raise PipelineValidationError(
            f"Unsupported pipeline schema version {version!r}; this build understands "
            f"major version {current_major}."
        )
    return major, minor


def _check_unique(names: Iterable[str], what: str) -> None:
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise PipelineValidationError(f"{what} {name!r} is declared more than once.")
        seen.add(name)


def _check_unique_casefold(names: Iterable[str], what: str) -> None:
    """Reject names that collide on case-insensitive filesystems."""
    seen: dict[str, str] = {}
    for name in names:
        folded = name.casefold()
        existing = seen.get(folded)
        if existing is not None and existing != name:
            raise PipelineValidationError(
                f"{what} names {existing!r} and {name!r} collide on "
                "case-insensitive filesystems."
            )
        seen[folded] = name


def _read_mapping(data: Mapping[str, Any], allowed: set[str], what: str) -> dict[str, Any]:
    """Return a plain dict of *data*, rejecting unknown keys."""
    if not isinstance(data, Mapping):
        raise PipelineValidationError(
            f"Expected a mapping for {what}, got {type(data).__name__}."
        )
    unknown = sorted(set(data) - allowed)
    if unknown:
        keys = ", ".join(repr(k) for k in unknown)
        raise PipelineValidationError(
            f"Unknown key(s) {keys} in {what}. Put forward-compatible extensions in "
            "the 'metadata' field."
        )
    return dict(data)


# ---------------------------------------------------------------------------
# Package
# ---------------------------------------------------------------------------


class PipelinePackage(ModelPackage):
    """A :class:`ModelPackage` that also carries a :class:`PipelineManifest`.

    The package keeps the ``ModelPackage`` on-disk layout (``model.onnx`` for a
    single component, ``{name}/model.onnx`` otherwise) and adds
    ``pipeline.json`` describing the topology and the component filenames.

    It may also carry *runtime assets* — tokenizers, scheduler configs,
    processor configs — as a mapping of safe relative destination to an
    existing local source file.  Assets are opaque: they are copied and their
    destinations are recorded, and nothing in this module ever reads them.

    Attributes:
        manifest: The validated topology.
        config: The primary configuration, exactly as on ``ModelPackage``.
        component_configs: Optional per-component configuration objects.
        assets: Mapping of manifest-declared destination to the local source
            file it is copied from.  After :meth:`load`, the sources are the
            resolved paths inside the loaded directory.
    """

    def __init__(
        self,
        models: Mapping[str, ir.Model] | None = None,
        manifest: PipelineManifest | None = None,
        config: object | None = None,
        component_configs: Mapping[str, object] | None = None,
        assets: Mapping[str, str] | None = None,
    ) -> None:
        if manifest is None:
            manifest = PipelineManifest()
        if models is None:
            models = {
                component.name: component.model
                for component in manifest.components
                if component.model is not None
            }
        super().__init__(dict(models), config=config)
        self.manifest = manifest
        self.component_configs: dict[str, object] = dict(component_configs or {})
        unknown = sorted(set(self.component_configs) - set(manifest.component_names))
        if unknown:
            names = ", ".join(repr(n) for n in unknown)
            raise PipelineValidationError(
                f"Per-component config given for unknown component(s) {names}."
            )
        self.assets: dict[str, str] = dict(assets or {})
        self._validate_assets()
        manifest.validate_models(self.data)

    def _validate_assets(self) -> None:
        """Check asset sources against the manifest's declared destinations."""
        declared = {asset.path: asset for asset in self.manifest.assets}
        undeclared = sorted(set(self.assets) - set(declared))
        if undeclared:
            names = ", ".join(repr(n) for n in undeclared)
            raise PipelineValidationError(
                f"Asset source(s) given for undeclared destination(s) {names}; declare "
                "them on the manifest so that pipeline.json stays in sync with the "
                "saved directory."
            )
        missing = sorted(
            path
            for path, asset in declared.items()
            if asset.required and path not in self.assets
        )
        if missing:
            names = ", ".join(repr(n) for n in missing)
            raise PipelineValidationError(
                f"Required asset(s) {names} are declared by the manifest but have no "
                "source file."
            )
        for destination, source in self.assets.items():
            _validate_asset_path(destination)
            if not isinstance(source, str) or not source:
                raise PipelineValidationError(
                    f"Asset {destination!r} must map to a local file path."
                )
            if not os.path.isfile(source):
                raise PipelineValidationError(
                    f"Asset {destination!r} source {source!r} does not exist."
                )

    def asset_path(self, destination: str) -> str:
        """Return the local source path for a declared asset destination."""
        if destination not in self.assets:
            known = ", ".join(sorted(self.assets)) or "<none>"
            raise KeyError(f"No asset {destination!r}; known assets: {known}.")
        return self.assets[destination]

    def __repr__(self) -> str:
        names = ", ".join(repr(k) for k in self.data)
        return f"PipelinePackage({{{names}}}, stages={len(self.manifest.stages)})"

    def config_for(self, component: str) -> object | None:
        """Return the component config, falling back to the primary config."""
        if component in self.component_configs:
            return self.component_configs[component]
        return self.config

    def component_files(self) -> dict[str, str]:
        """Return ``{component: relative onnx path}`` for the saved layout.

        Mirrors :meth:`ModelPackage.save`: a single-component package is stored
        flat as ``model.onnx``; otherwise each component gets its own folder.
        """
        return self.manifest.component_file_layout()

    def to_dict(self) -> dict[str, JSONValue]:
        """Return the ``pipeline.json`` document."""
        return {
            "format": "mobius-pipeline",
            "schema_version": self.manifest.schema_version,
            "manifest": self.manifest.to_dict(),
            "component_files": dict(sorted(self.component_files().items())),
        }

    def save(
        self,
        directory: str,
        *,
        external_data: str = "onnx",
        max_shard_size_bytes: int | None = None,
        components: Callable[[str], bool] | None = None,
        progress_bar: bool = True,
        check_weights: bool = True,
    ) -> None:
        """Save every component, every declared asset, and ``pipeline.json``.

        Assets are validated (safe relative destination, existing source) before
        anything is written, then copied one at a time via a temporary file and
        an atomic rename, so a reader never observes a half-written asset.
        ``pipeline.json`` is written last: its presence marks a complete
        directory.

        Raises:
            PipelineValidationError: If *components* is given (a partial save
                would desynchronize the manifest from the saved graphs), if the
                in-memory models no longer match the manifest, or if an asset
                destination is unsafe or its source has disappeared.
        """
        if components is not None:
            raise PipelineValidationError(
                "PipelinePackage.save() does not support partial saves: the manifest "
                "describes every component, so writing a subset would produce a "
                "directory whose pipeline.json references missing graphs. Build a "
                "smaller pipeline instead."
            )
        self.manifest.validate_models(self.data)
        self._validate_assets()
        marker = os.path.join(directory, PIPELINE_FILENAME)
        if os.path.isfile(marker):
            os.remove(marker)
        super().save(
            directory,
            external_data=external_data,
            max_shard_size_bytes=max_shard_size_bytes,
            progress_bar=progress_bar,
            check_weights=check_weights,
        )
        self._copy_assets(directory)
        document = self.to_dict()
        handle, staged = tempfile.mkstemp(dir=directory, prefix=".mobius-pipeline-")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as file:
                json.dump(document, file, indent=2)
                file.write("\n")
            os.replace(staged, marker)
        except BaseException:
            if os.path.exists(staged):
                os.remove(staged)
            raise
        files = document["component_files"]
        assert isinstance(files, dict)
        for name, relative in files.items():
            expected = os.path.join(directory, *str(relative).split("/"))
            if not os.path.isfile(expected):
                raise PipelineValidationError(
                    f"Component {name!r} was not written to {expected!r}; the saved "
                    "layout does not match pipeline.json."
                )
        for asset in self.manifest.assets:
            written = os.path.join(directory, *asset.path.split("/"))
            if asset.required and not os.path.isfile(written):
                raise PipelineValidationError(
                    f"Required asset {asset.path!r} was not written to {written!r}."
                )

    def _copy_assets(self, directory: str) -> None:
        """Copy every declared asset into *directory*, one atomic rename each."""
        root = os.path.abspath(directory)
        for destination in sorted(self.assets):
            source = self.assets[destination]
            target = os.path.abspath(os.path.join(root, *destination.split("/")))
            # Defence in depth: the destination was validated as a safe relative
            # path, so the resolved target must still be inside the package.
            if os.path.commonpath([root, target]) != root:
                raise PipelineValidationError(
                    f"Asset destination {destination!r} resolves outside the package "
                    f"directory ({target!r})."
                )
            if os.path.abspath(source) == target:
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            handle, staged = tempfile.mkstemp(
                dir=os.path.dirname(target), prefix=".mobius-asset-"
            )
            os.close(handle)
            try:
                shutil.copyfile(source, staged)
                os.replace(staged, target)
            except BaseException:
                if os.path.exists(staged):
                    os.remove(staged)
                raise

    @classmethod
    def load(cls, directory: str) -> PipelinePackage:
        """Load a pipeline directory written by :meth:`save`.

        Raises:
            PipelineValidationError: If ``pipeline.json`` is missing, refers to
                components that are absent from the directory, or declares a
                required asset that is not present.
        """
        path = os.path.join(directory, PIPELINE_FILENAME)
        if not os.path.isfile(path):
            raise PipelineValidationError(
                f"{path!r} not found; use ModelPackage.load() for plain model directories."
            )
        try:
            with open(path, encoding="utf-8") as file:
                document = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            raise PipelineValidationError(
                f"Could not read a valid {PIPELINE_FILENAME!r} from {directory!r}."
            ) from error
        body = _read_mapping(
            document,
            {"format", "schema_version", "manifest", "component_files"},
            PIPELINE_FILENAME,
        )
        if body.get("format") != "mobius-pipeline":
            raise PipelineValidationError(
                f"{PIPELINE_FILENAME!r} has unsupported format {body.get('format')!r}."
            )
        _parse_schema_version(str(body.get("schema_version", PIPELINE_SCHEMA_VERSION)))
        manifest = PipelineManifest.from_dict(body["manifest"])
        files = body.get("component_files") or {}
        if not isinstance(files, Mapping):
            raise PipelineValidationError("'component_files' must be a mapping.")
        expected_files = manifest.component_file_layout()
        if dict(files) != expected_files:
            raise PipelineValidationError(
                "'component_files' must exactly match the safe layout derived from "
                "the manifest."
            )
        models: dict[str, ir.Model] = {}
        for name in manifest.component_names:
            model_path = os.path.join(directory, *str(files[name]).split("/"))
            if not os.path.isfile(model_path):
                raise PipelineValidationError(
                    f"Component {name!r} file {model_path!r} is missing."
                )
            models[name] = ir.load(model_path)
        assets: dict[str, str] = {}
        for asset in manifest.assets:
            resolved = os.path.join(directory, *asset.path.split("/"))
            if os.path.isfile(resolved):
                assets[asset.path] = resolved
            elif asset.required:
                raise PipelineValidationError(
                    f"Required asset {asset.path!r} is missing from {directory!r}."
                )
        manifest = dataclasses.replace(
            manifest,
            components=tuple(
                component.with_model(models[component.name])
                for component in manifest.components
            ),
        )
        return cls(models, manifest, assets=assets)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class PipelineBuilder:
    """Compose already-built graphs into a validated :class:`PipelinePackage`.

    The builder never builds, traces, or optimizes a graph: every component is
    an ``ir.Model`` that already exists.  All structural validation happens in
    :meth:`build`.
    """

    def __init__(self, *, schema_version: str = PIPELINE_SCHEMA_VERSION) -> None:
        _parse_schema_version(schema_version)
        self._schema_version = schema_version
        self._components: dict[str, PipelineComponent] = {}
        self._connections: list[PipelineConnection] = []
        self._inputs: list[PipelineInput] = []
        self._states: list[PipelineState] = []
        self._outputs: list[PipelineOutput] = []
        self._stages: list[PipelineStage] = []
        self._assets: dict[str, str] = {}
        self._asset_specs: dict[str, PipelineAsset] = {}
        self._required_capabilities: set[str] = set()
        self._metadata: dict[str, JSONValue] = {}
        self._profile: PipelineProfile | None = None

    # -- Components --------------------------------------------------------

    def add_model(
        self,
        name: str,
        model: ir.Model,
        *,
        role: str,
        run_on: str = DEFAULT_PHASE,
        presence: str | None = None,
        capabilities: Iterable[str] | None = None,
        preferred_execution_providers: Iterable[str] | None = None,
        parameter_dtype: str | None = None,
        source: str | None = None,
        config: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PipelineComponent:
        """Add one graph as a component and derive its typed ports."""
        if name in self._components:
            raise PipelineValidationError(f"Component {name!r} is already registered.")
        component = PipelineComponent.from_model(
            name,
            model,
            role=role,
            run_on=run_on,
            presence=presence,
            capabilities=capabilities,
            preferred_execution_providers=preferred_execution_providers,
            parameter_dtype=parameter_dtype,
            source=source,
            config=config,
            metadata=metadata,
        )
        self._components[name] = component
        return component

    def add_package(
        self,
        package: Mapping[str, ir.Model],
        *,
        roles: Mapping[str, str] | Callable[[str], str],
        prefix: str | None = None,
        run_on: str | Mapping[str, str] = DEFAULT_PHASE,
        source: str | None = None,
        configs: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> tuple[PipelineComponent, ...]:
        """Add every model of a :class:`ModelPackage` as a component.

        Component names are derived deterministically: keys are visited in
        sorted order and, when *prefix* is given, namespaced as
        ``"{prefix}_{key}"`` — so two packages that both contain ``"model"``
        can coexist.

        Roles must be given explicitly (a mapping keyed by *package key*, or a
        callable).  The core deliberately does not infer a role from a literal
        package key such as ``"vision"``: package keys are a storage detail and
        carry no guaranteed semantics.

        Args:
            package: Mapping of package key to ``ir.Model``.
            roles: Mapping ``package key -> role`` or a callable of the key.
            prefix: Optional namespace prefix.
            run_on: A phase for every component, or a mapping keyed by package
                key.
            source: Optional provenance recorded on each component.
            configs: Optional per-key JSON-safe component configs.

        Returns:
            The created components, in sorted key order.
        """
        if prefix is not None:
            _validate_component_name(prefix)
        created: list[PipelineComponent] = []
        for key in sorted(package):
            if isinstance(roles, Mapping):
                if key not in roles:
                    known = ", ".join(sorted(roles)) or "<none>"
                    raise PipelineValidationError(
                        f"No role declared for package key {key!r}; roles were given "
                        f"for: {known}. Roles must be explicit."
                    )
                role = roles[key]
            elif callable(roles):
                role = roles(key)
            else:
                raise PipelineValidationError(
                    "roles must be a mapping of package key to role, or a callable."
                )
            name = f"{prefix}_{key}" if prefix else key
            phase = run_on[key] if isinstance(run_on, Mapping) else run_on
            created.append(
                self.add_model(
                    name,
                    package[key],
                    role=role,
                    run_on=phase,
                    source=source,
                    config=(configs or {}).get(key),
                    metadata={"package_key": key},
                )
            )
        return tuple(created)

    # -- Wiring ------------------------------------------------------------

    def connect(
        self,
        source: str | PipelinePort,
        target: str | PipelinePort,
        *,
        recurrent: bool = False,
        transform: str | None = None,
        context: Iterable[str | PipelinePort] | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> PipelineConnection:
        """Wire ``source`` output to ``target`` input.

        Args:
            source: Producing endpoint, e.g. ``"encoder.hidden"``.
            target: Consuming endpoint, e.g. ``"decoder.encoder_hidden"``.
            recurrent: Mark the edge as loop-carried state.  The owning stage
                must support it and will be given the
                :data:`LOOP_CARRIED_STATE_CAPABILITY` capability at build time.
            transform: Registered transform kind for the runtime; never
                executed here.  Its declared capabilities are added to the
                manifest's required capabilities, and the edge is exempt from
                direct dtype/rank/shape compatibility because a transform may
                legitimately change all three.
            context: Additional component input/output endpoints consumed by
                the transform, such as the current latent for a scheduler step.
            parameters: JSON-safe parameters validated by the transform
                definition and consumed directly by the runtime.
        """
        connection = PipelineConnection(
            PipelinePort.parse(source),
            PipelinePort.parse(target),
            recurrent,
            transform,
            tuple(PipelinePort.parse(port) for port in (context or ())),
            _ensure_json_mapping(parameters, "Transform parameters"),
        )
        self._connections.append(connection)
        if recurrent:
            self._required_capabilities.add(LOOP_CARRIED_STATE_CAPABILITY)
        self._required_capabilities.update(connection.transform_capabilities)
        return connection

    def declare_external(
        self,
        port: str | PipelinePort,
        *,
        alias: str | None = None,
        semantic: str | None = None,
        required: bool = True,
        presence: str | None = None,
    ) -> PipelineInput:
        """Declare that the caller supplies this input."""
        return self._declare(
            port,
            InputSource.EXTERNAL,
            alias=alias,
            semantic=semantic,
            required=required,
            presence=presence,
        )

    def declare_generated(
        self,
        port: str | PipelinePort,
        *,
        generator: str,
        parameters: Mapping[str, Any] | None = None,
        semantic: str | None = None,
        presence: str | None = None,
    ) -> PipelineInput:
        """Declare that the runtime harness produces this input per invocation."""
        rule = GeneratedInputRule(
            generator,
            _ensure_json_mapping(parameters, "Generated input parameters"),
        )
        self._required_capabilities.update(rule.capabilities)
        return self._declare(
            port,
            InputSource.GENERATED,
            semantic=semantic,
            presence=presence,
            generator=rule,
        )

    def declare_stateful(
        self,
        port: str | PipelinePort,
        *,
        semantic: str | None = None,
    ) -> PipelineInput:
        """Declare that this input is runtime-owned state carried across steps."""
        return self._declare(port, InputSource.STATEFUL, semantic=semantic)

    def declare_default(
        self,
        port: str | PipelinePort,
        value: JSONValue,
        *,
        semantic: str | None = None,
    ) -> PipelineInput:
        """Declare a JSON-safe constant for this input."""
        return self._declare(
            port,
            InputSource.DEFAULTED,
            value=value,
            semantic=semantic,
            required=False,
        )

    def _declare(
        self,
        port: str | PipelinePort,
        kind: str,
        *,
        value: JSONValue = None,
        alias: str | None = None,
        semantic: str | None = None,
        required: bool = True,
        presence: str | None = None,
        generator: GeneratedInputRule | None = None,
    ) -> PipelineInput:
        entry = PipelineInput(
            PipelinePort.parse(port),
            kind,
            value,
            alias,
            semantic,
            required,
            presence,
            generator,
        )
        self._inputs.append(entry)
        return entry

    def add_state(
        self,
        name: str,
        *,
        kind: str,
        input: str | PipelinePort,
        output: str | PipelinePort,
        lifetime: str,
        release_after: str,
        sequence_axis: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PipelineState:
        """Declare lifecycle semantics for one recurrent connection."""
        state = PipelineState(
            name=name,
            kind=kind,
            input=PipelinePort.parse(input),
            output=PipelinePort.parse(output),
            lifetime=lifetime,
            release_after=release_after,
            sequence_axis=sequence_axis,
            metadata=_ensure_json_mapping(metadata, f"State {name!r} metadata"),
        )
        self._states.append(state)
        return state

    def add_public_output(
        self, port: str | PipelinePort, *, alias: str | None = None
    ) -> PipelineOutput:
        """Expose a component output as a pipeline result."""
        output = PipelineOutput(PipelinePort.parse(port), alias)
        self._outputs.append(output)
        return output

    def add_public_state_output(
        self,
        state: str,
        *,
        alias: str | None = None,
    ) -> PipelineOutput:
        """Expose the final value of a recurrent state as a pipeline result."""
        output = PipelineOutput(state=state, alias=alias)
        self._outputs.append(output)
        return output

    def add_stage(
        self,
        name: str,
        kind: str,
        components: Sequence[str],
        *,
        run_on: str = DEFAULT_PHASE,
        options: Mapping[str, Any] | None = None,
        capabilities: Iterable[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PipelineStage:
        """Declare a stage over *components* with a registered strategy."""
        stage = PipelineStage(
            name=name,
            kind=kind,
            components=tuple(components),
            run_on=run_on,
            options=_ensure_json_mapping(options, f"Stage {name!r} options"),
            capabilities=_string_tuple(capabilities, f"Stage {name!r} capabilities"),
            metadata=_ensure_json_mapping(metadata, f"Stage {name!r} metadata"),
        )
        self._stages.append(stage)
        return stage

    def require_capability(self, capability: str) -> None:
        """Record a capability the target runtime must provide."""
        _validate_token(capability, "Capability")
        self._required_capabilities.add(capability)

    def add_asset(
        self, destination: str, source: str, *, required: bool = True
    ) -> PipelineAsset:
        """Ship an opaque runtime file with the package.

        The file's contents are never read or interpreted here — this only
        records that *source* must be copied to *destination* inside the saved
        package directory, and only *destination* reaches the manifest.

        Args:
            destination: ``/``-separated relative path inside the package, e.g.
                ``"tokenizer.json"`` or ``"scheduler/scheduler_config.json"``.
            source: Path to an existing local file to copy at save time.
            required: Whether :meth:`PipelinePackage.load` must find the file.
        """
        _validate_asset_path(destination)
        if destination in self._assets:
            raise PipelineValidationError(
                f"Asset destination {destination!r} is already registered."
            )
        if not isinstance(source, str) or not os.path.isfile(source):
            raise PipelineValidationError(
                f"Asset {destination!r} source {source!r} must be an existing file."
            )
        asset = PipelineAsset(destination, required=required)
        self._asset_specs[destination] = asset
        self._assets[destination] = source
        return asset

    def set_metadata(self, key: str, value: JSONValue) -> None:
        """Attach a JSON-safe manifest-level metadata entry."""
        _validate_token(key, "Metadata key")
        self._metadata[key] = _ensure_json_value(value, f"Metadata {key!r}")

    def set_profile(self, name: str, version: str) -> PipelineProfile:
        """Set the versioned runtime profile implemented by this package."""
        profile = PipelineProfile(name, version)
        if self._profile is not None and self._profile != profile:
            raise PipelineValidationError(
                f"Pipeline profile is already set to {self._profile!r}."
            )
        self._profile = profile
        return profile

    # -- Build -------------------------------------------------------------

    def build(
        self,
        *,
        config: object | None = None,
        component_configs: Mapping[str, object] | None = None,
    ) -> PipelinePackage:
        """Validate the topology and return the composed package.

        Stages that own a recurrent connection are given the
        :data:`LOOP_CARRIED_STATE_CAPABILITY` capability so that the loop-carried
        state is visible to a runtime inspecting the manifest.
        """
        stages = tuple(self._augment_loop_stages())
        manifest = PipelineManifest(
            components=tuple(self._components.values()),
            connections=tuple(self._connections),
            stages=stages,
            inputs=tuple(self._inputs),
            outputs=tuple(self._outputs),
            assets=tuple(self._asset_specs.values()),
            states=tuple(self._states),
            profile=self._profile,
            required_capabilities=tuple(self._required_capabilities),
            schema_version=self._schema_version,
            metadata=dict(self._metadata),
        )
        models = {name: component.model for name, component in self._components.items()}
        missing = [name for name, model in models.items() if model is None]
        if missing:
            names = ", ".join(repr(n) for n in missing)
            raise PipelineValidationError(f"Component(s) {names} have no graph.")
        return PipelinePackage(
            {name: model for name, model in models.items() if model is not None},
            manifest,
            config=config,
            component_configs=component_configs,
            assets=dict(self._assets),
        )

    def _augment_loop_stages(self) -> Iterator[PipelineStage]:
        """Add the loop-carried-state capability to stages owning recurrent edges."""
        looping: set[str] = set()
        for connection in self._connections:
            if not connection.recurrent:
                continue
            for stage in self._stages:
                if not stage.supports_loop_carried_state:
                    continue
                if (
                    connection.source.component in stage.components
                    and connection.target.component in stage.components
                ):
                    looping.add(stage.name)
        for stage in self._stages:
            if (
                stage.name in looping
                and LOOP_CARRIED_STATE_CAPABILITY not in stage.capabilities
            ):
                yield dataclasses.replace(
                    stage,
                    capabilities=(*stage.capabilities, LOOP_CARRIED_STATE_CAPABILITY),
                )
            else:
                yield stage
