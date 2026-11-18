# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Full-checkpoint embedding L4 and semantic retrieval L5, including MRL."""

from __future__ import annotations

import pytest

from mobius._testing.embedding_gemma2 import replay


@pytest.mark.golden
@pytest.mark.integration
def test_embedding_gemma2_l4():
    replay(level="L4")


@pytest.mark.generation
@pytest.mark.integration
def test_embedding_gemma2_l5_retrieval():
    replay(level="L5")
