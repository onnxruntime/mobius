# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Configuration for Recursive Flint's dense/MoE decoder."""

from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace

from mobius._configs._base import (
    ArchitectureConfig,
    CausalLMConfig,
    _resolve_dtype_value,
    _shallow_fields,
)


@dataclasses.dataclass
class FlintConfig(CausalLMConfig):
    """Flint attention, dense-layer placement, and sigmoid routing configuration."""

    attn_qk_norm: bool = True
    rope_type: str | None = "default"
    num_local_experts: int | None = 256
    num_experts_per_tok: int | None = 6
    moe_intermediate_size: int | None = 512
    shared_expert_intermediate_size: int | None = 1024
    mlp_only_layers: tuple[int, ...] = ()
    softmax_type: str = "learnable"
    router_score_function: str = "sigmoid"

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> FlintConfig:
        if isinstance(config, dict):
            config = SimpleNamespace(**config)
        base = ArchitectureConfig.from_transformers(config, parent_config)
        options = _shallow_fields(base)
        layer_types = getattr(config, "layer_types", None)
        options.update(
            hidden_act=getattr(config, "hidden_act", "silu"),
            attn_qk_norm=True,
            attn_qk_norm_full=False,
            no_rope_layers=list(getattr(config, "no_rope_layer_indices", None) or []),
            layer_types=list(
                ["full_attention"] * base.num_hidden_layers
                if layer_types is None
                else layer_types
            ),
            sliding_window=(
                getattr(config, "sliding_window", 513)
                if getattr(config, "use_sliding_window", True)
                else None
            ),
            num_local_experts=getattr(config, "num_experts", 256),
            num_experts_per_tok=getattr(config, "num_experts_per_tok", 6),
            moe_intermediate_size=getattr(config, "moe_intermediate_size", 512),
            shared_expert_intermediate_size=getattr(
                config, "shared_expert_intermediate_size", 1024
            ),
            scoring_func=getattr(config, "router_score_function", "sigmoid"),
            dtype=(
                _resolve_dtype_value(
                    getattr(config, "dtype", None) or getattr(config, "torch_dtype", None)
                )
                or base.dtype
            ),
            rope_type=base.rope_type or "default",
            rope_theta=(
                base.rope_theta
                if base.rope_theta is not None
                else getattr(config, "rope_theta", 10000.0)
            ),
            partial_rotary_factor=(
                base.partial_rotary_factor
                if base.partial_rotary_factor is not None
                else getattr(config, "partial_rotary_factor", 1.0)
            ),
        )
        return cls(
            **options,
            mlp_only_layers=tuple(getattr(config, "mlp_only_layers", None) or ()),
            softmax_type=getattr(config, "softmax_type", "learnable"),
            router_score_function=getattr(config, "router_score_function", "sigmoid"),
        )

    def validate(self) -> None:
        super().validate()
        if self.layer_types is not None:
            if len(self.layer_types) != self.num_hidden_layers:
                raise ValueError("Flint requires one layer_types entry per decoder layer")
            if set(self.layer_types) - {"full_attention", "sliding_attention"}:
                raise ValueError("Flint supports only full_attention and sliding_attention")
            if "sliding_attention" in self.layer_types and (
                self.sliding_window is None or self.sliding_window <= 0
            ):
                raise ValueError("Flint sliding-attention layers require a positive window")
        for name, indices in (
            ("mlp_only_layers", self.mlp_only_layers),
            ("no_rope_layers", self.no_rope_layers or []),
        ):
            if len(set(indices)) != len(indices) or any(
                index < 0 or index >= self.num_hidden_layers for index in indices
            ):
                raise ValueError(f"Flint {name} must contain distinct valid layer indices")
        if (
            not self.num_local_experts
            or not self.num_experts_per_tok
            or not 0 < self.num_experts_per_tok <= self.num_local_experts
            or not self.moe_intermediate_size
            or self.moe_intermediate_size <= 0
        ):
            raise ValueError("Flint requires positive expert dimensions and a valid top-k")
        if self.shared_expert_intermediate_size is not None and (
            self.shared_expert_intermediate_size < 0
        ):
            raise ValueError("Flint shared expert size must be non-negative")
        if self.softmax_type != "learnable" or self.router_score_function != "sigmoid":
            raise ValueError("Flint requires learned attention sinks and sigmoid routing")
        if not self.norm_topk_prob:
            raise ValueError("Flint's published inference implementation normalizes top-k")
        if self.hidden_act != "silu":
            raise ValueError("Flint requires SiLU expert and dense activations")
        if not self.attn_qk_norm or self.attn_qk_norm_full:
            raise ValueError("Flint requires per-head QK RMSNorm")
        if self.rope_type != "default" or not math.isclose(
            self.partial_rotary_factor or 0.0, 1.0, rel_tol=0.0, abs_tol=0.0
        ):
            raise ValueError("Flint supports only full-head default RoPE")
        if self.quantization is not None or self.component_quantization is not None:
            raise NotImplementedError("Flint quantized checkpoint loading is not qualified")
