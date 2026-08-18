# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Compatibility imports for the Transformers config resolver.

New code should import from :mod:`mobius.integrations.transformers`.
"""

from __future__ import annotations

import sys

from mobius.integrations.transformers import _config_resolver as _implementation

sys.modules[__name__] = _implementation
