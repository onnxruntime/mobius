# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Generate full GNM reference meshes from a pinned upstream CUDA checkout."""

from __future__ import annotations

import argparse
from pathlib import Path

from mobius._testing.gnm_reference import generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--level", choices=["L4", "L5"], action="append")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    generate(args.reference_root, args.level, device=args.device)


if __name__ == "__main__":
    main()
