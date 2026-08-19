# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Corrections to runtime assets that ship broken from upstream model repos.

A model package is more than weights. It carries a chat template, a tokenizer
config, a processor config -- files that decide whether an agent client can
actually drive the model. When one of those is wrong the export is faithful
and the deployment is still broken, and the defect comes back every time
somebody re-exports.

The corrections live in ``data/<owner>/<model>/`` as unified diffs written
against a named upstream revision, next to a ``README.md`` explaining the
symptom that motivated them. A patch is selected by the sha256 of the file
already in the package, not by a model id, so it finds its target whether the
source was a Hub repo, a snapshot directory, or a local checkout -- and it can
only ever rewrite bytes it recognises.

That hash, plus the strictness of the diff applier, is how upstream is
tracked: when upstream republishes the file, the patch stops matching and
stops being applied instead of silently corrupting a package.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import pathlib
from typing import Any

from mobius.upstream_patches._diff import PatchError, apply_unified_diff

_LOGGER = logging.getLogger(__name__)

_DATA_ROOT = pathlib.Path(__file__).parent / "data"


@dataclasses.dataclass(frozen=True)
class AssetPatch:
    """One upstream correction, loaded from ``data/<owner>/<model>/patch.json``."""

    directory: pathlib.Path
    upstream_repo: str
    upstream_revision: str
    summary: str
    operations: tuple[dict[str, Any], ...]

    @property
    def name(self) -> str:
        return self.upstream_repo


def _load(directory: pathlib.Path) -> AssetPatch:
    with (directory / "patch.json").open(encoding="utf-8") as handle:
        raw = json.load(handle)
    upstream = raw["upstream"]
    return AssetPatch(
        directory=directory,
        upstream_repo=upstream["repo"],
        upstream_revision=upstream["revision"],
        summary=raw.get("summary", ""),
        operations=tuple(raw["operations"]),
    )


def available_patches(root: pathlib.Path | None = None) -> list[AssetPatch]:
    """Return every correction shipped with this package."""
    base = _DATA_ROOT if root is None else root
    return [_load(path.parent) for path in sorted(base.glob("*/*/patch.json"))]


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _apply_diff(package: pathlib.Path, patch: AssetPatch, operation: dict) -> str | None:
    target = package / operation["target"]
    if not target.is_file():
        return None
    if _sha256(target) != operation["upstream_sha256"]:
        return None

    diff = (patch.directory / operation["patch"]).read_text(encoding="utf-8")
    try:
        patched = apply_unified_diff(target.read_text(encoding="utf-8"), diff)
    except PatchError:
        # The hash matched, so the bytes are the ones this diff was written
        # against; a failure here means the diff itself is wrong.
        _LOGGER.exception("Patch for %s does not apply to %s", patch.name, target)
        return None

    target.write_text(patched, encoding="utf-8")
    return operation["target"]


def _sync_chat_template(package: pathlib.Path, operation: dict) -> str | None:
    """Copy a patched template over the duplicate embedded in a JSON config."""
    target = package / operation["target"]
    source = package / operation["source"]
    if not target.is_file() or not source.is_file():
        return None

    template = source.read_text(encoding="utf-8")
    with target.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("chat_template") == template:
        return None

    config["chat_template"] = template
    with target.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return operation["target"]


def apply_asset_patches(
    package_dir: str | pathlib.Path,
    *,
    root: pathlib.Path | None = None,
) -> list[str]:
    """Correct any known-broken upstream assets in an exported model package.

    Args:
        package_dir: directory holding the exported package's runtime assets.
        root: patch data root; defaults to the corrections shipped here.

    Returns:
        The names of the files that were rewritten, in application order.
    """
    package = pathlib.Path(package_dir)
    if not package.is_dir():
        return []

    changed: list[str] = []
    for patch in available_patches(root):
        applied: list[str] = []
        for operation in patch.operations:
            kind = operation["kind"]
            if kind == "diff":
                result = _apply_diff(package, patch, operation)
            elif kind == "sync_chat_template":
                # Only meaningful once the template it mirrors was corrected.
                result = _sync_chat_template(package, operation) if applied else None
            else:
                _LOGGER.warning("Unknown patch operation %r in %s", kind, patch.name)
                continue
            if result is not None:
                applied.append(result)

        if applied:
            _LOGGER.info(
                "Patched %s in %s (from %s@%s): %s",
                ", ".join(applied),
                package,
                patch.upstream_repo,
                patch.upstream_revision[:12],
                patch.summary,
            )
            changed.extend(applied)

    return changed
