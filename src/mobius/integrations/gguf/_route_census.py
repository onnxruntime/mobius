# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Generated work census for every unresolved GGUF architecture and sidecar route."""

from __future__ import annotations

__all__ = [
    "GGUFRouteWorkItem",
    "RECENT_PR_DEPENDENCIES",
    "iter_remaining_route_work",
    "render_remaining_route_batches",
]

import dataclasses
from collections import defaultdict
from typing import Literal

from mobius.integrations.gguf._arch_registry import iter_arch_specs
from mobius.integrations.gguf._draft import is_draft_architecture
from mobius.integrations.gguf._mmproj_registry import iter_projector_specs
from mobius.integrations.gguf._mtp import mtp_architecture_capabilities
from mobius.integrations.gguf._spec import Support
from mobius.integrations.gguf._tokenizer_census import tokenizer_route_census

RouteCategory = Literal[
    "immediately-implementable",
    "evidence-only",
    "dependency-or-runtime-abi-blocked",
    "artifact-unavailable",
    "intentionally-rejected",
]
RouteKind = Literal["architecture", "projector", "tokenizer", "mtp", "draft"]


@dataclasses.dataclass(frozen=True, slots=True)
class GGUFRouteWorkItem:
    """One unresolved route with an exact registry reason and executable next batch."""

    route_id: str
    kind: RouteKind
    category: RouteCategory
    batch: str
    dependencies: tuple[str, ...]
    reason: str


@dataclasses.dataclass(frozen=True, slots=True)
class RecentPRDependency:
    """A recent PR that changes the inputs or boundary of a future census batch."""

    number: int
    title: str
    state_at_audit: Literal["merged", "open", "closed"]
    dependency: str


RECENT_PR_DEPENDENCIES: tuple[RecentPRDependency, ...] = (
    RecentPRDependency(
        645,
        "Audit all GGUF tokenizer routes",
        "merged",
        "authoritative tokenizer route inventory and compiled-semantics blockers",
    ),
    RecentPRDependency(
        651,
        "Validate Gemma4 GGUF tokenizer reconstruction",
        "merged",
        "gemma4 tokenizer evidence merged into the authoritative route census",
    ),
    RecentPRDependency(
        652,
        "Support complete sharded GGUF imports",
        "closed",
        "superseded by merged PR #656",
    ),
    RecentPRDependency(
        656,
        "Add exact fail-closed Qwen4Exp GGUF support",
        "merged",
        "qwen4exp route and complete sharded import merged, superseding PR #652",
    ),
    RecentPRDependency(
        675,
        "Fix sharded GGUF evidence edge cases",
        "merged",
        "follow-up hardening for sharded GGUF evidence edge cases after PR #656",
    ),
)

# Reviewed dispositions for routes that cannot advance until Mobius owns a new
# state/package ABI. Classification must not depend on wording in the reason.
_ARCHITECTURE_ABI_BLOCKED = frozenset(
    {
        "afmoe",
        "arwkv7",
        "bailingmoe3",
        "chameleon",
        "cogvlm",
        "cohere2moe",
        "deepseek2",
        "deepseek2-ocr",
        "deepseek32",
        "deepseek4",
        "gemma3n",
        "gemma4-assistant",
        "gpt-oss",
        "granite_swa",
        "graniteswitch",
        "hunyuan_vl",
        "laguna",
        "llama4",
        "mellum",
        "mimo2",
        "minimax-m3",
        "mistral3",
        "nanbeige",
        "paddleocr",
        "plamo3",
        "pockettts",
        "qwen3tts",
        "qwen3vl",
        "qwen3vlmoe",
        "rwkv6",
        "rwkv6qwen2",
        "rwkv7",
        "step35",
        "wavtokenizer-dec",
    }
)
_ARCHITECTURE_RUNTIME_SCHEMA_BLOCKED = frozenset(
    {
        "falcon-h1",
        "granitehybrid",
        "jamba",
        "kimi-k3",
        "kimi-linear",
        "minimax-01",
        "nemotron_h_moe",
        "plamo2",
    }
)
_ARCHITECTURE_INTENTIONAL_REJECTIONS = frozenset(
    {"bailingmoe2", "dots3note", "exaone-moe", "exaone4", "glm4", "glm4moe"}
)
_PROJECTOR_ARTIFACT_UNAVAILABLE = frozenset({"phi4"})


def _architecture_dependencies(spec) -> tuple[str, ...]:
    if spec.config is not Support.SUPPORTED:
        return ("exact metadata extraction", "tensor closure", "dedicated graph and parity")
    if spec.tensor_map is not Support.SUPPORTED:
        return ("suffix-exact tensor mapping", "packed-value transform proof", "graph closure")
    if spec.graph is not Support.SUPPORTED:
        return ("dedicated graph topology", "cache/state contract", "synthetic parity")
    return (
        "immutable representative GGUF",
        "full-logit prefill and cached-decode parity",
        "deterministic generation/state evidence",
    )


def _architecture_items() -> list[GGUFRouteWorkItem]:
    items = []
    for spec in iter_arch_specs():
        if spec.runtime is Support.SUPPORTED:
            continue
        reason = spec.reason or "Registry has no reason."
        core_verdicts = (spec.config, spec.tensor_map, spec.graph, spec.runtime)
        dependencies: tuple[str, ...]
        if (
            any(verdict is Support.REJECTED for verdict in core_verdicts)
            or spec.gguf_arch in _ARCHITECTURE_INTENTIONAL_REJECTIONS
        ):
            category: RouteCategory = "intentionally-rejected"
            batch = "policy-rejections"
            dependencies = ("policy change plus independent correctness proof",)
        elif (
            spec.gguf_arch
            in (_ARCHITECTURE_ABI_BLOCKED | _ARCHITECTURE_RUNTIME_SCHEMA_BLOCKED)
            or spec.preflight_only
        ):
            category = "dependency-or-runtime-abi-blocked"
            batch = "architecture-abi-dependencies"
            if spec.gguf_arch in _ARCHITECTURE_RUNTIME_SCHEMA_BLOCKED:
                dependencies = (
                    "ORT GenAI heterogeneous-state schema (issue #605)",
                    "stateful runtime package parity",
                )
            else:
                dependencies = _architecture_dependencies(spec)
        elif spec.is_importable:
            category = "evidence-only"
            batch = "architecture-runtime-evidence"
            dependencies = _architecture_dependencies(spec)
        else:
            category = "immediately-implementable"
            batch = "architecture-implementation"
            dependencies = _architecture_dependencies(spec)
        items.append(
            GGUFRouteWorkItem(
                f"architecture:{spec.gguf_arch}",
                "architecture",
                category,
                batch,
                dependencies,
                reason,
            )
        )
    return items


def _projector_items() -> list[GGUFRouteWorkItem]:
    items = []
    for spec in iter_projector_specs():
        reason = spec.reason or "Registry has no reason."
        if spec.runtime is Support.SUPPORTED:
            continue
        dependencies: tuple[str, ...]
        if any(verdict is Support.REJECTED for verdict in spec.verdicts.values()):
            category: RouteCategory = "intentionally-rejected"
            batch = "policy-rejections"
            dependencies = ("sidecar role must become a valid projector contract",)
        elif spec.is_importable:
            category = "evidence-only"
            batch = "projector-runtime-evidence"
            dependencies = (
                "paired text target",
                "processor boundary",
                "deterministic multimodal package execution",
            )
        elif (
            spec.metadata is Support.SUPPORTED
            and spec.tensor_map is Support.SUPPORTED
            and spec.graph is Support.DEFERRED
        ):
            category = "dependency-or-runtime-abi-blocked"
            batch = "projector-runtime-abi"
            dependencies = ("dynamic processor-to-graph media shape ABI",)
        elif spec.projector_type in _PROJECTOR_ARTIFACT_UNAVAILABLE:
            category = "artifact-unavailable"
            batch = "projector-artifact-discovery"
            dependencies = ("complete immutable mmproj", "component parity oracle")
        else:
            category = "immediately-implementable"
            batch = "projector-implementation"
            dependencies = ("metadata schema", "tensor closure", "component graph parity")
        items.append(
            GGUFRouteWorkItem(
                f"projector:{spec.projector_type}",
                "projector",
                category,
                batch,
                dependencies,
                reason,
            )
        )
    return items


def _tokenizer_items() -> list[GGUFRouteWorkItem]:
    items = []
    for record in tokenizer_route_census():
        if record.current_status == "validated-pinned-source":
            continue
        reason = record.candidate_disposition or str(record.blocker_category)
        dependencies: tuple[str, ...]
        if record.blocker_category == "compiled-llama.cpp-semantic-dependency":
            category: RouteCategory = "dependency-or-runtime-abi-blocked"
            batch = "tokenizer-compiled-semantics"
            dependencies = ("compiled pinned llama.cpp oracle", "dispatch-equivalence fixture")
        elif record.blocker_category == "pinned-artifact-source-parity-pending":
            category = "evidence-only"
            batch = "tokenizer-artifact-evidence"
            dependencies = (
                "immutable GGUF/source pair",
                "ordered vocabulary and encoding parity",
            )
        else:
            category = "artifact-unavailable"
            batch = "tokenizer-artifact-replacement"
            dependencies = (
                "replacement complete artifact",
                "matching official tokenizer source",
            )
        if record.identifier == "gemma4":
            dependencies += ("PR #651",)
        items.append(
            GGUFRouteWorkItem(
                f"tokenizer:{record.identifier}",
                "tokenizer",
                category,
                batch,
                dependencies,
                reason,
            )
        )
    return items


def _mtp_items() -> list[GGUFRouteWorkItem]:
    items = []
    for architecture, capability in mtp_architecture_capabilities().items():
        dependencies: tuple[str, ...]
        if capability.support is Support.SUPPORTED:
            category: RouteCategory = "evidence-only"
            batch = "mtp-runtime-evidence"
            dependencies = ("target acceptance loop", "cache-threaded draft/target parity")
        elif capability.loader_behavior == "executed-sidecar":
            category = "dependency-or-runtime-abi-blocked"
            batch = "mtp-specialized-abi"
            dependencies = ("specialized sidecar graph", "routed/cache state ABI")
        else:
            category = "intentionally-rejected"
            batch = "policy-rejections"
            dependencies = ("upstream executable ownership change",)
        items.append(
            GGUFRouteWorkItem(
                f"mtp:{architecture}",
                "mtp",
                category,
                batch,
                dependencies,
                capability.reason,
            )
        )
    return items


def _draft_items() -> list[GGUFRouteWorkItem]:
    reason = (
        "Exact target pairing and standalone graph construction are implemented, but no "
        "runtime acceptance-loop ABI or deterministic draft/target generation evidence exists."
    )
    return [
        GGUFRouteWorkItem(
            f"draft:{architecture}",
            "draft",
            "evidence-only",
            "draft-runtime-evidence",
            (
                "target acceptance loop",
                "draft cache orchestration",
                "deterministic speedup parity",
            ),
            reason,
        )
        for architecture in ("dflash", "eagle3")
        if is_draft_architecture(architecture)
    ]


def iter_remaining_route_work() -> tuple[GGUFRouteWorkItem, ...]:
    """Return every unresolved authoritative route exactly once."""
    items = [
        *_architecture_items(),
        *_projector_items(),
        *_tokenizer_items(),
        *_mtp_items(),
        *_draft_items(),
    ]
    items.sort(key=lambda item: item.route_id)
    route_ids = [item.route_id for item in items]
    if len(route_ids) != len(set(route_ids)):
        raise RuntimeError("GGUF remaining-route census contains duplicate route ids")
    return tuple(items)


def render_remaining_route_batches() -> str:
    """Render a concise batch table while preserving exact reasons in machine-readable records."""
    grouped: dict[tuple[str, str, tuple[str, ...]], list[str]] = defaultdict(list)
    for item in iter_remaining_route_work():
        grouped[(item.category, item.batch, item.dependencies)].append(item.route_id)
    rows = [
        "| Category | Next batch | Routes | Dependencies |",
        "|---|---|---|---|",
    ]
    for (category, batch, dependencies), route_ids in sorted(grouped.items()):
        routes = ", ".join(f"`{route_id}`" for route_id in route_ids)
        rows.append(f"| `{category}` | `{batch}` | {routes} | {'; '.join(dependencies)} |")
    return "\n".join(rows)
