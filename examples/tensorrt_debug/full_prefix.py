"""Full-prefix control: resend all tokens instead of relying on cached history.

The caller resets the TRT and HF caches and uses position zero for this input.
This module only constructs and validates the complete token history.
"""

from __future__ import annotations

import numpy as np


def build_full_prefix(prompt_ids: np.ndarray, generated: list[int]) -> np.ndarray:
    """Join [1, prompt_length] and [1, generated_length] along the token axis."""
    generated_ids = np.asarray([generated], dtype=np.int64)
    return np.concatenate([prompt_ids, generated_ids], axis=1)


def validate_full_prefix_budget(
    prompt_length: int, max_new_tokens: int, max_chunk: int
) -> None:
    """The final prediction consumes all preceding tokens, not itself."""
    if prompt_length + max_new_tokens - 1 > max_chunk:
        raise ValueError(
            f"Full-prefix generation exceeds the engine's maximum input length "
            f"of {max_chunk}; reduce --max-new-tokens or shorten the prompt"
        )
