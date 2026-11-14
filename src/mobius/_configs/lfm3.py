# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""LiquidAI LFM3 decoder configuration, including per-operator RoPE."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

from mobius._configs._base import (
    ArchitectureConfig,
    CausalLMConfig,
    _shallow_fields,
)


def _default_rope_parameters() -> dict[str, dict]:
    return {
        "full_attention": {
            "rope_type": "default",
            "rope_theta": 1_000_000.0,
            "partial_rotary_factor": 0.5,
        },
        "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
    }


@dataclasses.dataclass
class Lfm3Config(CausalLMConfig):
    """LFM3 uses the configured FF width directly, unlike LFM2."""

    model_type: str = "lfm3"
    rope_parameters: dict[str, dict] | None = None
    use_sandwich_norm: bool = True
    short_conv_kernel: int = 7
    sliding_window: int | None = 256
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.layer_types is None:
            self.layer_types = [
                "full_attention"
                if i % 4 == 2
                else "sliding_attention"
                if i > 0 and i % 4 == 0
                else "conv"
                for i in range(self.num_hidden_layers)
            ]
        # Newer Transformers remaps custom-code "conv" to "linear_attention".
        # LFM3 still uses the checkpoint's gated short-convolution operator.
        self.layer_types = [
            "conv" if kind == "linear_attention" else kind for kind in self.layer_types
        ]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("LFM3 layer_types must contain exactly num_hidden_layers entries")
        unknown = sorted(
            set(self.layer_types) - {"conv", "full_attention", "sliding_attention"}
        )
        if unknown:
            raise ValueError(f"Unsupported LFM3 layer type(s): {unknown}")
        if "sliding_attention" in self.layer_types and (
            self.sliding_window is None or self.sliding_window <= 0
        ):
            raise ValueError("LFM3 sliding_attention requires a positive sliding_window")
        if self.short_conv_kernel < 2:
            raise ValueError("LFM3 short_conv_kernel must be at least 2")
        if self.head_dim <= 0 or self.head_dim % 2:
            raise ValueError("LFM3 head_dim must be a positive even integer")
        if self.rope_parameters is None:
            self.rope_parameters = _default_rope_parameters()
        for kind in set(self.layer_types) - {"conv"}:
            if kind not in self.rope_parameters:
                raise ValueError(f"LFM3 rope_parameters has no entry for {kind}")
            params = self.rope_parameters[kind]
            # Dynamic/scaled RoPE must not silently become ordinary prefix rotation.
            if params.get("rope_type", "default") != "default":
                raise ValueError("LFM3 export currently supports default zero-tail RoPE only")
            if params.get("rope_theta", 0) <= 0:
                raise ValueError(f"LFM3 {kind} requires a positive rope_theta")
            if not 0 <= params.get("partial_rotary_factor", 1.0) <= 1:
                raise ValueError("LFM3 partial_rotary_factor must be between 0 and 1")
        self.attn_qk_norm = True
        self.attn_qk_norm_full = False
        self.attn_qkv_bias = False
        self.attn_o_bias = False
        self.hidden_act = "silu"
        self.mlp_bias = False

    @classmethod
    def from_transformers(cls, config, parent_config=None) -> Lfm3Config:
        if isinstance(config, dict):
            config = SimpleNamespace(**config)
        base = ArchitectureConfig.from_transformers(config, parent_config)
        fields = _shallow_fields(base)
        fields.update(
            layer_types=getattr(config, "layer_types", None),
            rms_norm_eps=getattr(config, "norm_eps", 1e-5),
            short_conv_kernel=getattr(config, "conv_L_cache", 7),
            short_conv_bias=getattr(config, "conv_bias", False),
            sliding_window=getattr(config, "sliding_window", 256),
            tie_word_embeddings=getattr(
                config, "tie_embedding", getattr(config, "tie_word_embeddings", True)
            ),
        )
        return cls(
            **fields,
            rope_parameters=getattr(config, "rope_parameters", None),
            use_sandwich_norm=getattr(config, "use_sandwich_norm", True),
        )
