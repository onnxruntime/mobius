# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Compatibility imports for Diffusers configuration adapters.

New code should import from :mod:`mobius.integrations.diffusers._configs`.
"""

from __future__ import annotations

import sys

from mobius.integrations.diffusers import _configs as _implementation

sys.modules[__name__] = _implementation
