# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Configuration for the NVIDIA Cosmos3-Omni unified MoT transformer.

Describes ``Cosmos3OmniTransformer`` — the *unified* Mixture-of-Transformers
(MoT) backbone shipped in the ``transformer/`` folder of the diffusers
checkpoints ``nvidia/Cosmos3-Nano`` and ``nvidia/Cosmos3-Super``.  This is the
model that carries **both** towers in one stack of layers:

* the **understanding** ("und" / Reasoner) expert — causal self-attention over
  the text/understanding prefix, using ``self_attn.to_{q,k,v,out}``,
  ``mlp.*``, ``input_layernorm``, ``post_attention_layernorm`` and ``norm``;
* the **generation** ("gen" / Generator) expert — a rectified-flow diffusion
  branch whose tokens attend non-causally over *understanding + generation*
  keys/values, using ``self_attn.add_{q,k,v}_proj`` / ``to_add_out``,
  ``mlp_moe_gen.*``, ``input_layernorm_moe_gen``,
  ``post_attention_layernorm_moe_gen`` and ``norm_moe_gen``.

On top of the shared backbone the checkpoint carries per-modality projection
heads: vision latents (``proj_in`` / ``proj_out``), an optional Sound head
(``audio_proj_in`` / ``audio_proj_out`` / ``audio_modality_embed``, gated by
``sound_gen``) and an optional Action head (``action_proj_in`` /
``action_proj_out`` / ``action_modality_embed``, gated by ``action_gen``,
implemented as per-embodiment-domain ``DomainAwareLinear`` layers).

Architecture reference: ``huggingface/diffusers``
``src/diffusers/models/transformers/transformer_cosmos3.py`` (Apache-2.0,
Copyright 2025 The NVIDIA Team and The HuggingFace Team) and the public
``nvidia/Cosmos3-Nano`` ``transformer/config.json``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import onnx_ir as ir

from mobius._configs._base import BaseModelConfig

# Feed-forward variants that the Cosmos3 backbone is defined for.  ``silu``
# selects the gated (SwiGLU) ``gate_proj``/``up_proj``/``down_proj`` MLP used
# by the Qwen3 dense backbone (Cosmos3-Nano / -Super); ``relu2`` selects the
# non-gated squared-ReLU ``up_proj``/``down_proj`` MLP used by the Nemotron
# backbone.  Anything else is an architecture mismatch.
SUPPORTED_HIDDEN_ACTS: tuple[str, ...] = ("silu", "relu2")

# Compute dtypes the graph may be built in.  ``bfloat16`` is the dtype of the
# published weights; ``float32`` is used by the unit tests and ``float16``
# is accepted for EPs without bf16 kernels.
SUPPORTED_DTYPES: tuple[ir.DataType, ...] = (
    ir.DataType.FLOAT,
    ir.DataType.FLOAT16,
    ir.DataType.BFLOAT16,
)

_DTYPE_BY_NAME: dict[str, ir.DataType] = {
    "bfloat16": ir.DataType.BFLOAT16,
    "float16": ir.DataType.FLOAT16,
    "float32": ir.DataType.FLOAT,
    "fp16": ir.DataType.FLOAT16,
    "fp32": ir.DataType.FLOAT,
}


def _as_dict(config: Any) -> dict[str, Any]:
    """Normalize a diffusers ``FrozenDict`` / config object into a plain dict."""
    if hasattr(config, "to_dict"):
        return dict(config.items())
    return dict(config)


def _resolve_diffusers_dtype(value: Any) -> ir.DataType:
    """Map the ``dtype`` string in ``transformer/config.json`` to an IR dtype."""
    if value is None:
        return ir.DataType.BFLOAT16
    if isinstance(value, ir.DataType):
        return value
    name = str(value).removeprefix("torch.")
    if name not in _DTYPE_BY_NAME:
        raise ValueError(
            f"Unsupported Cosmos3-Omni dtype {value!r}. Supported: {sorted(_DTYPE_BY_NAME)}"
        )
    return _DTYPE_BY_NAME[name]


@dataclasses.dataclass
class Cosmos3OmniGeneratorConfig(BaseModelConfig):
    """Architecture configuration for ``Cosmos3OmniTransformer``.

    Field names follow the published ``transformer/config.json`` so that
    :meth:`from_diffusers` is a near-identity mapping.  Fields that only drive
    *host-side* preprocessing (position-id construction, patchify/unpatchify,
    FPS modulation) are still parsed and validated here so the exported graph
    can be paired with a correct pre/post-processor, but they are documented
    as such — the ONNX graph itself consumes already-packed tokens and
    already-computed mRoPE position IDs.
    """

    # --- Backbone -----------------------------------------------------------
    #: RMSNorm epsilon shared by every norm in the backbone.
    rms_norm_eps: float = 1e-6
    #: Rotary base.  Cosmos3 uses a very large theta (5e6) for long video.
    rope_theta: float = 5_000_000.0
    #: mRoPE channel budget ``(T, H, W)``.  These are *channel counts*, and
    #: their sum must equal ``head_dim // 2`` (the rotary dimension).  The
    #: layout is interleaved, not chunked — see
    #: :class:`~mobius.models.cosmos3_omni_generator.Cosmos3OmniRotaryEmbedding`.
    rope_axes_dim: tuple[int, int, int] = (24, 20, 20)
    #: Bias on every attention projection (both experts).  ``False`` upstream.
    attention_bias: bool = False
    #: Attention dropout.  Must be ``0.0`` — the exported graph is inference-only.
    attention_dropout: float = 0.0
    #: Whether the understanding expert applies per-head QK RMSNorm
    #: (``self_attn.norm_q`` / ``self_attn.norm_k``).
    qk_norm_for_text: bool = True
    #: Whether the generation expert applies per-head QK RMSNorm
    #: (``self_attn.norm_added_q`` / ``self_attn.norm_added_k``).  The upstream
    #: module always constructs these, so ``False`` is an architecture mismatch.
    qk_norm_for_diffusion: bool = True
    #: Adds a *separate* ``self_attn.k_norm_und_for_gen`` RMSNorm applied to the
    #: understanding keys that the generation pathway consumes.  Upstream only
    #: instantiates it when ``use_und_k_norm_for_gen and not qk_norm_for_text``
    #: — see :attr:`has_und_k_norm_for_gen`.
    use_und_k_norm_for_gen: bool = False

    # --- Vision (diffusion) head -------------------------------------------
    #: Channel count of a single video-VAE latent (pre-patchify).
    latent_channel: int = 48
    #: Spatial patch size applied to the latent grid (host-side patchify).
    latent_patch_size: int = 2
    #: Width of ``proj_in`` / ``proj_out``.  Must equal
    #: ``latent_channel * latent_patch_size ** 2``.
    patch_latent_dim: int = 192
    #: Multiplier applied to the raw timesteps before the sinusoidal
    #: projection (``timesteps * timestep_scale``).
    timestep_scale: float = 0.001
    #: Width of the sinusoidal timestep projection feeding ``time_embedder``.
    #: Upstream hardcodes ``Timesteps(num_channels=256, ...)``; it is fixed by
    #: the published ``time_embedder.linear_1.weight`` shape ``[hidden, 256]``.
    time_proj_channels: int = 256

    # --- Sound head (optional) ---------------------------------------------
    #: Enables ``audio_proj_in`` / ``audio_proj_out`` / ``audio_modality_embed``.
    sound_gen: bool = False
    #: Channel count of a sound latent frame.  Required when ``sound_gen``.
    sound_dim: int | None = None
    #: Sound latent frame rate (host-side position-id construction only).
    sound_latent_fps: float = 25.0
    #: Sound temporal compression (host-side packing only).
    temporal_compression_factor_sound: int = 1

    # --- Action head (optional) --------------------------------------------
    #: Enables ``action_proj_in`` / ``action_proj_out`` / ``action_modality_embed``.
    action_gen: bool = False
    #: Action vector width.  Required when ``action_gen``.
    action_dim: int | None = None
    #: Upper bound on ``action_dim`` across embodiments (metadata only).
    max_action_dim: int | None = None
    #: Number of embodiment domains indexed by the ``DomainAwareLinear`` heads.
    num_embodiment_domains: int = 32

    # --- Host-side (documented, not consumed by the graph) ------------------
    #: Reference FPS used by the host when building temporal position IDs.
    base_fps: int = 24
    #: Whether the host scales temporal position IDs by the clip FPS.
    enable_fps_modulation: bool = True
    #: Whether the host restarts H/W position IDs per modality segment.
    unified_3d_mrope_reset_spatial_ids: bool = True
    #: Temporal offset the host inserts between modality segments.
    unified_3d_mrope_temporal_modality_margin: int = 15_000
    #: Maximum position id the host may emit.
    max_position_embeddings: int = 262_144

    # --- Architecture assertions (parsed so mismatches fail loudly) ---------
    #: Must be ``"unified_3d_mrope"``.
    position_embedding_type: str = "unified_3d_mrope"
    #: Must be ``"two_way"`` (causal und pathway + non-causal gen pathway).
    joint_attn_implementation: str = "two_way"
    #: Must be ``True`` — the MoT (per-expert weights) structure.
    use_moe: bool = True
    #: Must be ``False`` — temporal-causal video attention is not implemented.
    video_temporal_causal: bool = False
    #: HuggingFace-style model type tag used by downstream tooling.
    model_type: str = "cosmos3_omni_generator"

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def rotary_dim(self) -> int:
        """Number of rotary frequency channels (``head_dim // 2``)."""
        return self.head_dim // 2

    @property
    def num_key_value_groups(self) -> int:
        """Query heads per key/value head (GQA group size)."""
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def is_gated_mlp(self) -> bool:
        """``True`` when the feed-forward is SwiGLU (``gate/up/down_proj``)."""
        return self.hidden_act == "silu"

    @property
    def has_und_k_norm_for_gen(self) -> bool:
        """``True`` when ``self_attn.k_norm_und_for_gen`` exists.

        Mirrors upstream exactly: the extra norm is only instantiated when
        ``use_und_k_norm_for_gen`` is set *and* the understanding pathway has
        no QK norm of its own.  When ``qk_norm_for_text`` is ``True`` the
        understanding keys are already normalized, so the flag is inert.
        """
        return self.use_und_k_norm_for_gen and not self.qk_norm_for_text

    @property
    def attention_out_size(self) -> int:
        """Flattened attention output width (``num_attention_heads * head_dim``)."""
        return self.num_attention_heads * self.head_dim

    @property
    def key_value_size(self) -> int:
        """Flattened key/value width (``num_key_value_heads * head_dim``)."""
        return self.num_key_value_heads * self.head_dim

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Validate every shape relationship the graph builder relies on.

        Raises:
            ValueError: If any field is missing, non-positive, or inconsistent
                with the rest of the architecture.
        """
        self._validate_backbone()
        self._validate_rope()
        self._validate_vision_head()
        self._validate_sound_head()
        self._validate_action_head()
        self._validate_unsupported_variants()

    def _validate_backbone(self) -> None:
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads "
                f"(got {self.num_attention_heads} and {self.num_key_value_heads})"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim must be even for rotary embedding, got {self.head_dim}"
            )
        if self.hidden_act not in SUPPORTED_HIDDEN_ACTS:
            raise ValueError(
                f"hidden_act must be one of {SUPPORTED_HIDDEN_ACTS}, got {self.hidden_act!r}"
            )
        if self.rms_norm_eps <= 0:
            raise ValueError(f"rms_norm_eps must be positive, got {self.rms_norm_eps}")
        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(
                f"dtype must be one of {[d.name for d in SUPPORTED_DTYPES]}, got {self.dtype}"
            )
        # Any non-zero dropout is a training-time config, not an inference graph.
        if self.attention_dropout:
            raise ValueError(
                "attention_dropout must be 0.0 for an inference graph, got "
                f"{self.attention_dropout}"
            )

    def _validate_rope(self) -> None:
        if len(self.rope_axes_dim) != 3:
            raise ValueError(
                f"rope_axes_dim must have exactly 3 (T, H, W) entries, got {self.rope_axes_dim!r}"
            )
        if any(
            not isinstance(d, int) or isinstance(d, bool) or d <= 0 for d in self.rope_axes_dim
        ):
            raise ValueError(
                f"rope_axes_dim entries must be positive integers, got {self.rope_axes_dim!r}"
            )
        if sum(self.rope_axes_dim) != self.rotary_dim:
            raise ValueError(
                "sum(rope_axes_dim) must equal head_dim // 2 "
                f"(got sum={sum(self.rope_axes_dim)}, head_dim // 2={self.rotary_dim})"
            )
        if self.rope_theta <= 0:
            raise ValueError(f"rope_theta must be positive, got {self.rope_theta}")
        if self.max_position_embeddings <= 0:
            raise ValueError(
                f"max_position_embeddings must be positive, got {self.max_position_embeddings}"
            )

    def _validate_vision_head(self) -> None:
        if self.latent_channel <= 0:
            raise ValueError(f"latent_channel must be positive, got {self.latent_channel}")
        if self.latent_patch_size <= 0:
            raise ValueError(
                f"latent_patch_size must be positive, got {self.latent_patch_size}"
            )
        expected = self.latent_channel * self.latent_patch_size**2
        if self.patch_latent_dim != expected:
            raise ValueError(
                "patch_latent_dim must equal latent_channel * latent_patch_size ** 2 "
                f"(got {self.patch_latent_dim}, expected {expected})"
            )
        if self.timestep_scale <= 0:
            raise ValueError(f"timestep_scale must be positive, got {self.timestep_scale}")
        if self.time_proj_channels <= 0 or self.time_proj_channels % 2 != 0:
            raise ValueError(
                "time_proj_channels must be a positive even number, got "
                f"{self.time_proj_channels}"
            )

    def _validate_sound_head(self) -> None:
        if not self.sound_gen:
            return
        if self.sound_dim is None or self.sound_dim <= 0:
            raise ValueError(
                f"sound_dim must be a positive integer when sound_gen=True, got {self.sound_dim!r}"
            )
        if self.temporal_compression_factor_sound <= 0:
            raise ValueError(
                "temporal_compression_factor_sound must be positive, got "
                f"{self.temporal_compression_factor_sound}"
            )
        if self.sound_latent_fps <= 0:
            raise ValueError(f"sound_latent_fps must be positive, got {self.sound_latent_fps}")

    def _validate_action_head(self) -> None:
        if not self.action_gen:
            return
        if self.action_dim is None or self.action_dim <= 0:
            raise ValueError(
                "action_dim must be a positive integer when action_gen=True, got "
                f"{self.action_dim!r}"
            )
        if self.num_embodiment_domains <= 0:
            raise ValueError(
                "num_embodiment_domains must be positive when action_gen=True, got "
                f"{self.num_embodiment_domains}"
            )
        if self.max_action_dim is not None and self.max_action_dim < self.action_dim:
            raise ValueError(
                f"max_action_dim ({self.max_action_dim}) must be >= action_dim ({self.action_dim})"
            )

    def _validate_unsupported_variants(self) -> None:
        if self.position_embedding_type != "unified_3d_mrope":
            raise ValueError(
                "Only position_embedding_type='unified_3d_mrope' is supported, got "
                f"{self.position_embedding_type!r}"
            )
        if self.joint_attn_implementation != "two_way":
            raise ValueError(
                "Only joint_attn_implementation='two_way' is supported, got "
                f"{self.joint_attn_implementation!r}"
            )
        if not self.use_moe:
            raise ValueError(
                "use_moe=False is not a Cosmos3-Omni MoT checkpoint — the generation "
                "expert weights (mlp_moe_gen, add_*_proj, ...) would be absent."
            )
        if self.video_temporal_causal:
            raise ValueError(
                "video_temporal_causal=True is not supported: the generation pathway is "
                "built as fully non-causal attention over understanding + generation K/V."
            )
        if not self.qk_norm_for_diffusion:
            raise ValueError(
                "qk_norm_for_diffusion=False is not supported: the published checkpoint "
                "always carries self_attn.norm_added_q / norm_added_k."
            )

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @classmethod
    def from_diffusers(cls, config: Any) -> Cosmos3OmniGeneratorConfig:
        """Build a config from a diffusers ``transformer/config.json`` dict.

        ``rope_axes_dim`` is absent from the published config; upstream derives
        it from ``rope_scaling["mrope_section"]`` (falling back to
        ``[24, 20, 20]``), which this reproduces exactly.

        Args:
            config: Parsed ``transformer/config.json`` (a plain ``dict`` or any
                mapping-like diffusers config object).

        Returns:
            A validated :class:`Cosmos3OmniGeneratorConfig`.

        Raises:
            ValueError: If the parsed architecture fails :meth:`validate`.
        """
        raw = _as_dict(config)
        rope_scaling = raw.get("rope_scaling") or {}
        rope_axes_dim = raw.get("rope_axes_dim")
        if rope_axes_dim is None:
            rope_axes_dim = rope_scaling.get("mrope_section", [24, 20, 20])
        attention_bias = bool(raw.get("attention_bias", False))

        parsed = cls(
            # BaseModelConfig fields
            vocab_size=int(raw.get("vocab_size", 151936)),
            hidden_size=int(raw.get("hidden_size", 4096)),
            intermediate_size=int(raw.get("intermediate_size", 12288)),
            num_hidden_layers=int(raw.get("num_hidden_layers", 36)),
            num_attention_heads=int(raw.get("num_attention_heads", 32)),
            num_key_value_heads=int(raw.get("num_key_value_heads", 8)),
            head_dim=int(raw.get("head_dim", 128)),
            hidden_act=raw.get("hidden_act", "silu"),
            attn_qkv_bias=attention_bias,
            attn_o_bias=attention_bias,
            dtype=_resolve_diffusers_dtype(raw.get("dtype")),
            # Backbone
            rms_norm_eps=float(raw.get("rms_norm_eps", 1e-6)),
            rope_theta=float(raw.get("rope_theta", 5_000_000.0)),
            rope_axes_dim=tuple(int(d) for d in rope_axes_dim),  # type: ignore[arg-type]
            attention_bias=attention_bias,
            attention_dropout=float(raw.get("attention_dropout", 0.0)),
            qk_norm_for_text=bool(raw.get("qk_norm_for_text", True)),
            qk_norm_for_diffusion=bool(raw.get("qk_norm_for_diffusion", True)),
            use_und_k_norm_for_gen=bool(raw.get("use_und_k_norm_for_gen", False)),
            # Vision head
            latent_channel=int(raw.get("latent_channel", 48)),
            latent_patch_size=int(raw.get("latent_patch_size", 2)),
            patch_latent_dim=int(raw.get("patch_latent_dim", 192)),
            timestep_scale=float(raw.get("timestep_scale", 0.001)),
            # Sound head
            sound_gen=bool(raw.get("sound_gen", False)),
            sound_dim=_optional_int(raw.get("sound_dim")),
            sound_latent_fps=float(raw.get("sound_latent_fps", 25.0)),
            temporal_compression_factor_sound=int(
                raw.get("temporal_compression_factor_sound", 1)
            ),
            # Action head
            action_gen=bool(raw.get("action_gen", False)),
            action_dim=_optional_int(raw.get("action_dim")),
            max_action_dim=_optional_int(raw.get("max_action_dim")),
            num_embodiment_domains=int(raw.get("num_embodiment_domains", 32)),
            # Host-side
            base_fps=int(raw.get("base_fps", 24)),
            enable_fps_modulation=bool(raw.get("enable_fps_modulation", True)),
            unified_3d_mrope_reset_spatial_ids=bool(
                raw.get("unified_3d_mrope_reset_spatial_ids", True)
            ),
            unified_3d_mrope_temporal_modality_margin=int(
                raw.get("unified_3d_mrope_temporal_modality_margin", 15_000)
            ),
            max_position_embeddings=int(raw.get("max_position_embeddings", 262_144)),
            # Architecture assertions
            position_embedding_type=str(
                raw.get("position_embedding_type", "unified_3d_mrope")
            ),
            joint_attn_implementation=str(raw.get("joint_attn_implementation", "two_way")),
            use_moe=bool(raw.get("use_moe", True)),
            video_temporal_causal=bool(raw.get("video_temporal_causal", False)),
        )
        parsed.validate()
        return parsed


def _optional_int(value: Any) -> int | None:
    """Return ``int(value)`` or ``None`` when the field is absent/null."""
    return None if value is None else int(value)


__all__ = [
    "SUPPORTED_DTYPES",
    "SUPPORTED_HIDDEN_ACTS",
    "Cosmos3OmniGeneratorConfig",
]
