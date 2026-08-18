# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Compatibility imports for Hugging Face weight loading.

New code should import from :mod:`mobius.integrations._weight_loading`.
"""

from __future__ import annotations

import sys

from mobius.integrations import _weight_loading as _implementation

sys.modules[__name__] = _implementation
