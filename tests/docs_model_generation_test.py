# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for generated model-documentation descriptions."""

from __future__ import annotations

import runpy
from pathlib import Path


class _DocumentedModel:
    """A concise summary.

    ```{mermaid}
    flowchart LR
        A --> B
    ```

    Extended details remain visible exactly once.
    """


def test_model_page_keeps_summary_and_details_once():
    """Preserve the detailed model docstring without duplicating its diagram."""
    generator_path = Path(__file__).parents[1] / "docs" / "_generate_models.py"
    generator = runpy.run_path(str(generator_path))

    page = generator["_generate_model_page"]("example", _DocumentedModel)

    assert page.count("A concise summary.") == 1
    assert page.count("```{mermaid}") == 1
    assert page.count("Extended details remain visible exactly once.") == 1
