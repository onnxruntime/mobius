# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Report corrections whose upstream file has changed since they were written.

Every ``diff`` operation records the sha256 of the exact bytes it was written
against. When upstream republishes that file the hash stops matching, the
correction silently stops being applied, and somebody has to decide whether it
landed upstream or was restructured around.

Run it against the upstream default branch::

    python -m mobius.upstream_patches.check_upstream
    python -m mobius.upstream_patches.check_upstream meta-models/Muse-Glimmer-30B
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

from mobius.upstream_patches._patches import AssetPatch, available_patches


def _upstream_sha256(repo: str, filename: str, revision: str) -> str:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo, filename, revision=revision)
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def drifted_files(patch: AssetPatch, revision: str) -> list[str]:
    """Return one report line per file that no longer matches what was recorded."""
    reports: list[str] = []
    for operation in patch.operations:
        recorded = operation.get("upstream_sha256")
        if not recorded:
            continue
        target = operation["target"]
        try:
            current = _upstream_sha256(patch.upstream_repo, target, revision)
        except Exception as error:
            reports.append(f"{patch.upstream_repo}:{target} could not be fetched: {error}")
            continue
        if current != recorded:
            reports.append(
                f"{patch.upstream_repo}:{target} changed\n"
                f"    recorded {recorded} (at {patch.upstream_revision})\n"
                f"    upstream {current} (at {revision})"
            )
    return reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "patch",
        nargs="?",
        help="owner/model to check; every correction is checked when omitted",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="upstream revision to compare against (default: main)",
    )
    args = parser.parse_args(argv)

    patches = available_patches()
    if args.patch is not None:
        patches = [patch for patch in patches if patch.upstream_repo == args.patch]
        if not patches:
            parser.error(f"no correction for {args.patch!r}")

    reports = [line for patch in patches for line in drifted_files(patch, args.revision)]
    if reports:
        print("\n".join(reports))
        print(f"\n{len(reports)} file(s) drifted; re-verify before trusting the patch")
        return 1

    print(f"{len(patches)} correction(s) still match their recorded upstream")
    return 0


if __name__ == "__main__":
    sys.exit(main())
