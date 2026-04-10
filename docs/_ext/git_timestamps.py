# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Sphinx extension: adds git-based creation and last-update dates to each page.

For every documentation source file, ``git log --follow`` is queried to find:

* **Created** - date of the first commit that introduced the file.
* **Updated** - date of the most-recent commit that touched the file.

Both dates are formatted as ``YYYY-MM-DD`` (UTC) and injected into the Jinja2
template context as ``git_created_date`` and ``git_updated_date``.  The
companion template override (``docs/_templates/page.html``) renders them at
the bottom of every article.

Pages that have no git history (e.g. auto-generated files that are never
committed) receive empty strings and the template silently omits the section.
"""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from typing import Any

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


def _on_html_page_context(
    app: Sphinx,
    pagename: str,
    templatename: str,
    context: dict[str, Any],
    doctree: Any,
) -> None:
    """``html-page-context`` handler: inject git dates into the template context."""
    env = app.env

    # Only process pages that correspond to real source files.
    if pagename not in env.all_docs:
        context["git_created_date"] = ""
        context["git_updated_date"] = ""
        return

    docpath = Path(env.doc2path(pagename, base=True)).resolve()
    repo_root = _git_repo_root(app.srcdir)

    try:
        rel_path = str(docpath.relative_to(repo_root))
    except ValueError:
        logger.debug(
            "git_timestamps: %s is outside the git root %s; skipping",
            docpath,
            repo_root,
        )
        context["git_created_date"] = ""
        context["git_updated_date"] = ""
        return

    created, updated = _git_file_dates(rel_path, repo_root)
    context["git_created_date"] = created
    context["git_updated_date"] = updated


def setup(app: Sphinx) -> dict[str, Any]:
    app.connect("html-page-context", _on_html_page_context)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
