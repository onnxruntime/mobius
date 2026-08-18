# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Configuration for directly declared world models."""

from __future__ import annotations

import dataclasses
import math

from mobius._configs._base import BaseModelConfig


@dataclasses.dataclass
class WorldModelConfig(BaseModelConfig):
    """Configuration shared by single-step world-model graphs.

    The three shapes exclude the leading batch dimension. The default
    :class:`~mobius.models.MLPWorldModel` flattens each value internally, while
    custom modules may preserve their original ranks.
    """

    observation_shape: tuple[int, ...] = (1,)
    action_shape: tuple[int, ...] = (1,)
    state_shape: tuple[int, ...] = (1,)
    hidden_size: int = 128
    num_hidden_layers: int = 2
    hidden_act: str | None = "silu"
    residual_state: bool = True

    @property
    def observation_size(self) -> int:
        """Flattened observation size."""
        return math.prod(self.observation_shape)

    @property
    def action_size(self) -> int:
        """Flattened action size."""
        return math.prod(self.action_shape)

    @property
    def state_size(self) -> int:
        """Flattened recurrent-state size."""
        return math.prod(self.state_shape)

    def validate(self) -> None:
        """Validate dimensions required by the world-model task and reference model."""
        for name, shape in (
            ("observation_shape", self.observation_shape),
            ("action_shape", self.action_shape),
            ("state_shape", self.state_shape),
        ):
            if not shape:
                raise ValueError(f"{name} must contain at least one dimension")
            if any(
                not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in shape
            ):
                raise ValueError(f"{name} must contain only positive integer dimensions")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if self.hidden_act is None:
            raise ValueError("hidden_act must be set")
