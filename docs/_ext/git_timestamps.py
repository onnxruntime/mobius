# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Sphinx extension: adds git-based creation and last-update dates to each page.

For every documentation source file, ``git log --follow`` is queried to find:

* **Created** - date of the first commit that introduced the file.
* **Updated** - date of the most-recent commit that touched the file.

Both dates are formatted as ``YYYY-MM-DD`` (UTC) and inserted into the
doctree as a raw HTML node immediately after the page's top-level ``<h1>``
title, so they appear below the document title on every page.

Pages that have no git history (e.g. auto-generated files that are never
committed) are silently skipped.
"""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from typing import Any

from docutils import nodes
from sphinx.application import Sphinx

logger = logging.getLogger(__name__)

_DATE_FMT = "%Y-%m-%d"


@cache
def _git_repo_root(start_dir: str) -> str:
    """Return the absolute path to the git repository root, or *start_dir* on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=start_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return start_dir


@cache
def _git_file_dates(rel_path: str, repo_root: str) -> tuple[str, str]:
    """Return ``(created, updated)`` date strings (YYYY-MM-DD) for *rel_path*.

    Both values are determined from a single ``git log`` call and normalised to
    UTC.  Returns ``("", "")`` when git is unavailable or the file has no
    history.
    """
    try:
        result = subprocess.run(
            ["git", "log", "--follow", "--format=%aI", "--", rel_path],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        lines = []

    if not lines:
        return "", ""

    # git log returns commits newest-first; the first line is the most-recent
    # commit (updated) and the last line is the oldest commit (created).
    def _fmt(raw: str) -> str:
        try:
            dt = datetime.fromisoformat(raw)
            return dt.astimezone(timezone.utc).strftime(_DATE_FMT)
        except ValueError:
            return raw[:10]  # fall back to the bare date portion

    return _fmt(lines[-1]), _fmt(lines[0])


def _on_doctree_resolved(app: Sphinx, doctree: nodes.document, docname: str) -> None:
    """``doctree-resolved`` handler: insert git timestamps after the page title.

    A raw HTML node is inserted at position 1 in the first top-level section
    (index 0 is always the title node), so the dates appear directly below the
    ``<h1>`` heading in the rendered output.
    """
    if app.builder.format != "html":
        return

    env = app.env
    docpath = Path(env.doc2path(docname, base=True)).resolve()
    repo_root = _git_repo_root(app.srcdir)

    try:
        rel_path = str(docpath.relative_to(repo_root))
    except ValueError:
        logger.debug(
            "git_timestamps: %s is outside the git root %s; skipping",
            docpath,
            repo_root,
        )
        return

    created, updated = _git_file_dates(rel_path, repo_root)
    if not created and not updated:
        return

    # Build the HTML snippet
    parts: list[str] = []
    if created:
        parts.append(
            f'<span class="git-created-date">'
            f"<strong>Created:</strong> <time>{created}</time>"
            f"</span>"
        )
    if updated:
        parts.append(
            f'<span class="git-updated-date">'
            f"<strong>Last updated:</strong> <time>{updated}</time>"
            f"</span>"
        )
    html = '<div class="page-git-timestamps">' + " · ".join(parts) + "</div>"
    raw_node = nodes.raw("", html, format="html")

    # Insert after the title node (index 0) of the first section whose first
    # child is the page title node.
    for section in doctree.traverse(nodes.section):
        if section.children and isinstance(section.children[0], nodes.title):
            section.insert(1, raw_node)
            break


def setup(app: Sphinx) -> dict[str, Any]:
    app.connect("doctree-resolved", _on_doctree_resolved)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
