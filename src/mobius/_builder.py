# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Compatibility alias for the integration builder.

New code should import ecosystem builders from :mod:`mobius.integrations`
and generic module construction from :mod:`mobius.integrations._builder`.
"""

from __future__ import annotations

import sys

from mobius.integrations import _builder as _implementation

sys.modules[__name__] = _implementation
