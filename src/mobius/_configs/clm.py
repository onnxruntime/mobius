# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Configuration for CLM projection-head-only ranking exports."""

from __future__ import annotations

import dataclasses
import math

import onnx_ir as ir

from mobius._configs._base import BaseModelConfig


@dataclasses.dataclass
class CLMConfig(BaseModelConfig):
    """Head topology; the frozen text encoder is deliberately not included."""

    model_type: str | None = "clm"
    hidden_size: int = 4096
    width: int = 1536
    depth: int = 3
    projection_dim: int = 512
    activation: str = "gelu"
    layernorm: bool = True
    residual: bool = False
    scale: float = 100.0
    base_model: str = "Qwen/Qwen3-8B"
    encoder_pooling: str = "last-token"

    def validate(self) -> None:
        for name in ("hidden_size", "width", "depth", "projection_dim"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 2 <= self.depth <= 64:
            raise ValueError("CLM depth must be in [2, 64]")
        if self.activation not in {"gelu", "relu", "silu"}:
            raise ValueError("Unsupported CLM activation")
        if type(self.layernorm) is not bool or type(self.residual) is not bool:
            raise ValueError("CLM layernorm and residual must be booleans")
        if not math.isfinite(self.scale) or not 0 <= self.scale <= 100:
            raise ValueError("CLM scale must be finite and in [0, 100]")
        if self.dtype != ir.DataType.FLOAT:
            raise ValueError("CLM ranking heads currently support only dtype='f32'")
        if self.quantization is not None or self.component_quantization is not None:
            raise ValueError("CLM quantized head export is not supported")

    @classmethod
    def from_transformers(cls, config) -> CLMConfig:
        """Read the public input contract without executing custom Hub code."""
        if (
            getattr(config, "model_type", None) != "clm"
            or getattr(config, "base_model", None) != "Qwen/Qwen3-8B"
            or getattr(config, "encoder_pooling", None) != "last-token"
            or getattr(config, "embedding_dim", None) != 4096
            or getattr(config, "checkpoints", None) != ["CLM_v0.1-8B.pt"]
        ):
            raise ValueError("Unsupported CLM encoder/checkpoint contract")
        return cls()
