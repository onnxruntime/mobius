# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Configuration for the diffusers ``AutoencoderKLWan`` 3D causal video VAE.

``AutoencoderKLWan`` is the video autoencoder shipped with the Wan 2.1 / Wan 2.2
family and re-used verbatim as the video VAE of NVIDIA's Cosmos3 models
(``nvidia/Cosmos3-Nano/vae``, which declares
``_name_or_path = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"``).

The public ``vae/config.json`` for that checkpoint is::

    {
      "_class_name": "AutoencoderKLWan",
      "attn_scales": [], "base_dim": 160, "clip_output": false,
      "decoder_base_dim": 256, "dim_mult": [1, 2, 4, 4], "dropout": 0.0,
      "in_channels": 12, "is_residual": true,
      "latents_mean": [...48 floats...], "latents_std": [...48 floats...],
      "num_res_blocks": 2, "out_channels": 12, "patch_size": 2,
      "scale_factor_spatial": 16, "scale_factor_temporal": 4,
      "temperal_downsample": [false, true, true], "z_dim": 48
    }

Two upstream naming quirks are preserved here:

``temperal_downsample``
    Upstream misspells "temporal".  :meth:`WanVAEConfig.from_diffusers` accepts
    the misspelled JSON key (and the corrected spelling as a fallback), but the
    dataclass exposes the correctly spelled :attr:`WanVAEConfig.temporal_downsample`.

``clip_output``
    Present in the public JSON but **not** a parameter of upstream
    ``AutoencoderKLWan.__init__`` — ``ConfigMixin`` drops it as an unused kwarg,
    so diffusers always clamps the decoded video to ``[-1, 1]`` regardless of its
    value.  It is parsed here for round-trip fidelity only; see
    :attr:`WanVAEConfig.clip_output`.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Sequence
from typing import Any

import onnx_ir as ir

__all__ = ["WanVAEConfig"]

#: ``latents_mean`` from Wan 2.1's ``AutoencoderKLWan`` signature (z_dim=16).
#: Used only when a config omits the field; real checkpoints always provide it.
_WAN21_LATENTS_MEAN: tuple[float, ...] = (
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
)

#: ``latents_std`` from Wan 2.1's ``AutoencoderKLWan`` signature (z_dim=16).
_WAN21_LATENTS_STD: tuple[float, ...] = (
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.9160,
)


def _as_dict(config: Any) -> dict[str, Any]:
    """Normalise a diffusers config (dict, ``FrozenDict`` or object) to a dict."""
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "items"):
        return dict(config.items())
    if hasattr(config, "to_dict"):
        return dict(config.to_dict())
    return dict(vars(config))


def _resolve_dtype(raw: Any) -> ir.DataType:
    """Map a diffusers ``dtype``/``torch_dtype`` field onto an ``ir.DataType``.

    Returns ``ir.DataType.FLOAT`` for ``None`` and for the ``"auto"`` sentinel,
    matching how diffusers materialises weights when no dtype is requested.
    """
    if raw is None or raw == "auto":
        return ir.DataType.FLOAT
    if isinstance(raw, ir.DataType):
        return raw
    name = str(raw).replace("torch.", "")
    mapping = {
        "float32": ir.DataType.FLOAT,
        "float": ir.DataType.FLOAT,
        "float16": ir.DataType.FLOAT16,
        "half": ir.DataType.FLOAT16,
        "bfloat16": ir.DataType.BFLOAT16,
    }
    if name not in mapping:
        raise ValueError(
            f"Unsupported dtype {raw!r} for AutoencoderKLWan; "
            f"expected one of {sorted(mapping)}"
        )
    return mapping[name]


@dataclasses.dataclass(frozen=True)
class WanVAEConfig:
    """Architecture configuration for ``AutoencoderKLWan``.

    Attributes mirror the upstream ``AutoencoderKLWan.__init__`` signature
    one-for-one, except that ``temperal_downsample`` is exposed under the
    corrected spelling :attr:`temporal_downsample`.

    Attributes:
        base_dim: Encoder base channel width (``160`` for Wan 2.2 / Cosmos3).
        decoder_base_dim: Decoder base channel width.  ``None`` means "same as
            :attr:`base_dim`"; :meth:`__post_init__` resolves it eagerly so the
            model never has to re-apply the default.
        z_dim: Latent channel count (``48``).  The encoder emits ``2 * z_dim``
            channels (mean ‖ logvar) which ``quant_conv`` maps 1:1.
        dim_mult: Per-stage channel multipliers applied to the base dim.
        num_res_blocks: Residual blocks per encoder stage.  Decoder stages use
            ``num_res_blocks + 1`` blocks, matching upstream.
        attn_scales: Spatial scales at which the *non-residual* (Wan 2.1)
            encoder inserts an attention block.  Empty for Wan 2.2 / Cosmos3.
        temporal_downsample: One flag per down/up-sampling stage
            (``len(dim_mult) - 1`` entries) selecting 3D (spatio-temporal)
            instead of 2D (spatial-only) resampling.  Parsed from the upstream
            misspelled key ``temperal_downsample``.
        dropout: Upstream dropout probability.  Inference-only ONNX export
            always evaluates dropout in eval mode, so any value is a no-op;
            retained for config fidelity.
        latents_mean: Per-channel latent mean used by the *pipeline* to
            normalise / denormalise latents (length ``z_dim``).
        latents_std: Per-channel latent standard deviation (length ``z_dim``).
        is_residual: ``True`` selects the Wan 2.2 residual down/up blocks with
            ``AvgDown3D`` / ``DupUp3D`` shortcuts; ``False`` selects the flat
            Wan 2.1 blocks.
        in_channels: Encoder input channels **after** patchification, i.e.
            ``3 * patch_size ** 2`` (``12`` for Cosmos3).
        out_channels: Decoder output channels **before** unpatchification
            (``12`` for Cosmos3, which unpatchifies to 3-channel RGB video).
        patch_size: Spatial patch size folded into the channel dim before the
            encoder and unfolded after the decoder.  ``None`` disables it.
        scale_factor_temporal: Total temporal compression ratio (``4``).
        scale_factor_spatial: Total spatial compression ratio (``16``), which
            includes the ``patch_size`` factor.
        clip_output: Parsed from ``vae/config.json`` for round-trip fidelity.
            Upstream ``AutoencoderKLWan.__init__`` does **not** accept it, so
            diffusers silently drops it and *always* applies
            ``torch.clamp(out, -1.0, 1.0)`` in ``_decode``.  Mobius mirrors
            diffusers and clamps unconditionally, so this flag does not change
            the exported graph.
        dtype: Element type used for the exported graph inputs/outputs.
    """

    base_dim: int = 96
    decoder_base_dim: int | None = None
    z_dim: int = 16
    dim_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attn_scales: tuple[float, ...] = ()
    temporal_downsample: tuple[bool, ...] = (False, True, True)
    dropout: float = 0.0
    latents_mean: tuple[float, ...] = _WAN21_LATENTS_MEAN
    latents_std: tuple[float, ...] = _WAN21_LATENTS_STD
    is_residual: bool = False
    in_channels: int = 3
    out_channels: int = 3
    patch_size: int | None = None
    scale_factor_temporal: int = 4
    scale_factor_spatial: int = 8
    clip_output: bool = False
    dtype: ir.DataType = ir.DataType.FLOAT

    def __post_init__(self) -> None:
        """Coerce sequence fields to tuples, resolve defaults and validate."""
        object.__setattr__(self, "dim_mult", tuple(int(v) for v in self.dim_mult))
        object.__setattr__(self, "attn_scales", tuple(float(v) for v in self.attn_scales))
        object.__setattr__(
            self, "temporal_downsample", tuple(bool(v) for v in self.temporal_downsample)
        )
        object.__setattr__(self, "latents_mean", tuple(float(v) for v in self.latents_mean))
        object.__setattr__(self, "latents_std", tuple(float(v) for v in self.latents_std))
        if self.decoder_base_dim is None:
            object.__setattr__(self, "decoder_base_dim", self.base_dim)
        object.__setattr__(self, "dtype", _resolve_dtype(self.dtype))
        self.validate()

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def temporal_upsample(self) -> tuple[bool, ...]:
        """Decoder-side flags: :attr:`temporal_downsample` reversed."""
        return tuple(reversed(self.temporal_downsample))

    @property
    def temperal_downsample(self) -> tuple[bool, ...]:
        """Upstream-spelled alias of :attr:`temporal_downsample`.

        Kept so code written against the diffusers field name keeps working.
        """
        return self.temporal_downsample

    @property
    def video_channels(self) -> int:
        """Channel count of the un-patchified video tensor (3 for RGB).

        ``in_channels`` counts *patchified* channels, so the pixel-space video
        has ``in_channels / patch_size ** 2`` channels.
        """
        if self.patch_size is None:
            return self.in_channels
        return self.in_channels // (self.patch_size * self.patch_size)

    @property
    def decoded_video_channels(self) -> int:
        """Channel count of the decoded, un-patchified video tensor."""
        if self.patch_size is None:
            return self.out_channels
        return self.out_channels // (self.patch_size * self.patch_size)

    @property
    def encoder_dims(self) -> tuple[int, ...]:
        """Encoder stage widths ``[base, base * m0, base * m1, ...]``."""
        return tuple(self.base_dim * u for u in (1, *self.dim_mult))

    @property
    def decoder_dims(self) -> tuple[int, ...]:
        """Decoder stage widths ``[dbase * m[-1], dbase * m[::-1]...]``."""
        assert self.decoder_base_dim is not None
        base = self.decoder_base_dim
        return tuple(base * u for u in (self.dim_mult[-1], *reversed(self.dim_mult)))

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Check the parsed fields for internal consistency.

        Raises:
            ValueError: If any field is out of range or inconsistent with the
                other fields (stage counts, compression ratios, latent
                statistics length, or ``AvgDown3D``/``DupUp3D`` divisibility).
        """
        if self.base_dim <= 0:
            raise ValueError(f"base_dim must be positive, got {self.base_dim}")
        assert self.decoder_base_dim is not None
        if self.decoder_base_dim <= 0:
            raise ValueError(f"decoder_base_dim must be positive, got {self.decoder_base_dim}")
        if self.z_dim <= 0:
            raise ValueError(f"z_dim must be positive, got {self.z_dim}")
        if not self.dim_mult:
            raise ValueError("dim_mult must contain at least one stage")
        if self.num_res_blocks <= 0:
            raise ValueError(f"num_res_blocks must be positive, got {self.num_res_blocks}")
        if len(self.temporal_downsample) != len(self.dim_mult) - 1:
            raise ValueError(
                "temperal_downsample must have len(dim_mult) - 1 = "
                f"{len(self.dim_mult) - 1} entries, got {len(self.temporal_downsample)}"
            )
        if self.in_channels <= 0 or self.out_channels <= 0:
            raise ValueError(
                f"in_channels/out_channels must be positive, got "
                f"{self.in_channels}/{self.out_channels}"
            )
        if len(self.latents_mean) != self.z_dim:
            raise ValueError(
                f"latents_mean must have z_dim = {self.z_dim} entries, "
                f"got {len(self.latents_mean)}"
            )
        if len(self.latents_std) != self.z_dim:
            raise ValueError(
                f"latents_std must have z_dim = {self.z_dim} entries, "
                f"got {len(self.latents_std)}"
            )
        if any(std == 0 for std in self.latents_std):
            raise ValueError("latents_std entries must be non-zero (used as a divisor)")
        self._validate_patch_size()
        self._validate_scale_factors()
        if self.is_residual:
            self._validate_residual_shortcuts()

    def _validate_patch_size(self) -> None:
        if self.patch_size is None:
            return
        if self.patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {self.patch_size}")
        fold = self.patch_size * self.patch_size
        if self.in_channels % fold != 0:
            raise ValueError(
                f"in_channels ({self.in_channels}) must be divisible by "
                f"patch_size ** 2 ({fold}); in_channels counts patchified channels"
            )
        if self.out_channels % fold != 0:
            raise ValueError(
                f"out_channels ({self.out_channels}) must be divisible by "
                f"patch_size ** 2 ({fold})"
            )

    def _validate_scale_factors(self) -> None:
        patch = self.patch_size or 1
        expected_spatial = (2 ** (len(self.dim_mult) - 1)) * patch
        if self.scale_factor_spatial != expected_spatial:
            raise ValueError(
                f"scale_factor_spatial ({self.scale_factor_spatial}) does not match the "
                f"architecture: 2 ** (len(dim_mult) - 1) * patch_size = {expected_spatial}"
            )
        expected_temporal = 2 ** sum(self.temporal_downsample)
        if self.scale_factor_temporal != expected_temporal:
            raise ValueError(
                f"scale_factor_temporal ({self.scale_factor_temporal}) does not match the "
                f"architecture: 2 ** sum(temperal_downsample) = {expected_temporal}"
            )

    def _validate_residual_shortcuts(self) -> None:
        """Check ``AvgDown3D`` / ``DupUp3D`` channel divisibility per stage.

        Upstream asserts ``in_channels * factor % out_channels == 0`` (down) and
        ``out_channels * factor % in_channels == 0`` (up); failing that, the
        residual shortcut cannot be expressed as a grouped mean / repeat.
        """
        dims = self.encoder_dims
        last = len(self.dim_mult) - 1
        for i, (in_dim, out_dim) in enumerate(itertools.pairwise(dims)):
            down = i != last
            factor_t = 2 if (down and self.temporal_downsample[i]) else 1
            factor_s = 2 if down else 1
            factor = factor_t * factor_s * factor_s
            if in_dim * factor % out_dim != 0:
                raise ValueError(
                    f"encoder stage {i}: AvgDown3D requires in_dim * factor "
                    f"({in_dim} * {factor}) to be divisible by out_dim ({out_dim})"
                )
        up_dims = self.decoder_dims
        temporal_upsample = self.temporal_upsample
        for i, (in_dim, out_dim) in enumerate(itertools.pairwise(up_dims)):
            if i == last:
                continue
            factor_t = 2 if temporal_upsample[i] else 1
            factor = factor_t * 4
            if out_dim * factor % in_dim != 0:
                raise ValueError(
                    f"decoder stage {i}: DupUp3D requires out_dim * factor "
                    f"({out_dim} * {factor}) to be divisible by in_dim ({in_dim})"
                )

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @classmethod
    def from_diffusers(cls, config: Any) -> WanVAEConfig:
        """Build a :class:`WanVAEConfig` from a diffusers ``vae/config.json``.

        Args:
            config: Parsed ``vae/config.json`` as a dict, a diffusers
                ``FrozenDict``, or any object exposing ``items()``/``to_dict()``.

        Returns:
            A validated :class:`WanVAEConfig`.

        Raises:
            ValueError: If ``_class_name`` is present but is not
                ``"AutoencoderKLWan"``, or if the parsed fields are inconsistent.
        """
        raw = _as_dict(config)
        class_name = raw.get("_class_name")
        if class_name is not None and class_name != "AutoencoderKLWan":
            raise ValueError(
                f"WanVAEConfig expects _class_name 'AutoencoderKLWan', got {class_name!r}"
            )
        # Upstream misspells "temporal"; accept the corrected spelling as a fallback
        # so hand-written configs work too.
        temporal_downsample: Sequence[Any] = raw.get(
            "temperal_downsample", raw.get("temporal_downsample", (False, True, True))
        )
        return cls(
            base_dim=int(raw.get("base_dim", 96)),
            decoder_base_dim=(
                int(raw["decoder_base_dim"])
                if raw.get("decoder_base_dim") is not None
                else None
            ),
            z_dim=int(raw.get("z_dim", 16)),
            dim_mult=tuple(raw.get("dim_mult", (1, 2, 4, 4))),
            num_res_blocks=int(raw.get("num_res_blocks", 2)),
            attn_scales=tuple(raw.get("attn_scales", ())),
            temporal_downsample=tuple(temporal_downsample),
            dropout=float(raw.get("dropout", 0.0)),
            latents_mean=tuple(raw.get("latents_mean", _WAN21_LATENTS_MEAN)),
            latents_std=tuple(raw.get("latents_std", _WAN21_LATENTS_STD)),
            is_residual=bool(raw.get("is_residual", False)),
            in_channels=int(raw.get("in_channels", 3)),
            out_channels=int(raw.get("out_channels", 3)),
            patch_size=(int(raw["patch_size"]) if raw.get("patch_size") is not None else None),
            scale_factor_temporal=int(raw.get("scale_factor_temporal") or 4),
            scale_factor_spatial=int(raw.get("scale_factor_spatial") or 8),
            clip_output=bool(raw.get("clip_output", False)),
            dtype=raw.get("dtype", raw.get("torch_dtype")),
        )
