# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""A strict unified diff applier.

Strict means no fuzz and no offset search: a hunk applies at the line it names
with the context it names, or it does not apply at all. That rigidity is the
point. These diffs are corrections written against a specific upstream file,
and a hunk that no longer fits is the signal that upstream moved and the
correction needs a human to look at it again.
"""

from __future__ import annotations

import dataclasses
import re

_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class PatchError(Exception):
    """Raised when a diff does not apply cleanly to the given text."""


@dataclasses.dataclass(frozen=True)
class _Hunk:
    old_start: int
    lines: tuple[tuple[str, str], ...]


def _parse(diff: str) -> list[_Hunk]:
    hunks: list[_Hunk] = []
    header: int | None = None
    body: list[tuple[str, str]] = []

    for line in diff.splitlines():
        match = _HUNK_HEADER.match(line)
        if match is not None:
            if header is not None:
                hunks.append(_Hunk(header, tuple(body)))
            header = int(match.group(1))
            body = []
            continue
        if header is None:
            continue
        if line.startswith("\\"):
            # "\ No newline at end of file" annotates the previous line.
            continue
        if line[:1] in {" ", "-", "+"}:
            body.append((line[0], line[1:]))
        elif not line:
            body.append((" ", ""))

    if header is not None:
        hunks.append(_Hunk(header, tuple(body)))
    if not hunks:
        raise PatchError("diff contains no hunks")
    return hunks


def apply_unified_diff(original: str, diff: str) -> str:
    """Apply ``diff`` to ``original`` and return the result.

    Raises:
        PatchError: if any hunk's context does not match ``original`` exactly.
    """
    lines = original.splitlines(keepends=True)
    result: list[str] = []
    cursor = 0

    for hunk in _parse(diff):
        start = hunk.old_start - 1
        if start < cursor:
            raise PatchError(f"hunk at line {hunk.old_start} overlaps an earlier hunk")
        if start > len(lines):
            raise PatchError(f"hunk starts at line {hunk.old_start}, past end of file")
        result.extend(lines[cursor:start])
        cursor = start

        for tag, text in hunk.lines:
            if tag == "+":
                result.append(text + "\n")
                continue
            if cursor >= len(lines):
                raise PatchError(f"hunk at line {hunk.old_start} runs past end of file")
            found = lines[cursor].rstrip("\r\n")
            if found != text:
                raise PatchError(
                    f"context mismatch at line {cursor + 1}: "
                    f"expected {text!r}, found {found!r}"
                )
            if tag == " ":
                result.append(lines[cursor])
            cursor += 1

    result.extend(lines[cursor:])
    patched = "".join(result)
    if not original.endswith("\n") and patched.endswith("\n"):
        patched = patched[:-1]
    return patched
