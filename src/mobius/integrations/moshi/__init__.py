# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Internal Kyutai Moshi / Mimi native checkpoint support for Mobius.

The Moshi family (incl. ``nvidia/personaplex-7b-v1``) ships native Kyutai
``safetensors`` checkpoints. Build the supported PersonaPlex package through
the standard public API or CLI::

    from mobius import build

    pkg = build("nvidia/personaplex-7b-v1")
    pkg.save("personaplex-onnx")
"""

from __future__ import annotations

__all__: list[str] = []
