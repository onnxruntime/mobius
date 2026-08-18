# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Compatibility imports for the Diffusers integration.

New code should import from :mod:`mobius.integrations.diffusers`.
"""

from __future__ import annotations

import sys

from mobius.integrations.diffusers import _builder as _implementation

sys.modules[__name__] = _implementation
