# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Typed configuration for the pinned MatterGen GemNet-T score core.

The MatterGen Hub checkpoints store an expanded Hydra configuration rather
than a Transformers ``config.json``.  This module deliberately accepts a
plain nested mapping so reading a checkpoint configuration does not require
Hydra, OmegaConf, or PyYAML at import time.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, ClassVar

import onnx_ir as ir

from mobius._configs import BaseModelConfig
from mobius.integrations.mattergen._contract import (
    MATTERGEN_HUB_ID,
    MATTERGEN_HUB_REVISION,
    MATTERGEN_SOURCE_COMMIT,
)

MATTERGEN_MODEL_ID = MATTERGEN_HUB_ID

# ``PROPERTY_SOURCE_IDS`` in MatterGen v1.0.3.  Keep this full tuple even
# though only the listed entries occur in the published adapter checkpoints:
# it is the authoritative identifier family for expanded Hydra configs.
MATTERGEN_CONDITION_FAMILY: tuple[str, ...] = (
    "dft_mag_density",
    "dft_bulk_modulus",
    "dft_shear_modulus",
    "energy_above_hull",
    "formation_energy_per_atom",
    "space_group",
    "hhi_score",
    "ml_bulk_modulus",
    "chemical_system",
    "dft_band_gap",
)


@dataclasses.dataclass(frozen=True)
class MatterGenConditionSpec:
    """One declared MatterGen property-embedding input.

    ``kind`` names the source conditional encoder.  Scalar properties use the
    source ``NoiseLevelEncoding`` after their optional fitted standard scaler;
    chemical systems are host-provided 101-wide multi-hot vectors; space-group
    values are one-based indices and are decremented before their lookup.
    """

    name: str
    kind: str
    scaler: str = "identity"
    log10_transform: bool = False
    unconditional: str = "embedding_vector"
    is_adapter: bool = False

    @property
    def input_shape_suffix(self) -> tuple[int, ...]:
        """Required host value shape excluding the batch dimension."""
        return (101,) if self.kind == "chemical_system_multihot" else ()


# Defaults inferred from the official config groups.  The two condition names
# without public group files are still represented as scalar sources because
# they are legal identifiers in MatterGen v1.0.3's property family.
MATTERGEN_CONDITION_SPECS: tuple[MatterGenConditionSpec, ...] = (
    MatterGenConditionSpec("dft_mag_density", "scalar_sinusoidal", "standard"),
    MatterGenConditionSpec(
        "dft_bulk_modulus", "scalar_sinusoidal", "standard", log10_transform=True
    ),
    MatterGenConditionSpec("dft_shear_modulus", "scalar_sinusoidal", "standard"),
    MatterGenConditionSpec("energy_above_hull", "scalar_sinusoidal", "standard"),
    MatterGenConditionSpec("formation_energy_per_atom", "scalar_sinusoidal", "standard"),
    MatterGenConditionSpec("space_group", "space_group_index"),
    MatterGenConditionSpec("hhi_score", "scalar_sinusoidal", "standard"),
    MatterGenConditionSpec(
        "ml_bulk_modulus", "scalar_sinusoidal", "standard", log10_transform=True
    ),
    MatterGenConditionSpec("chemical_system", "chemical_system_multihot"),
    MatterGenConditionSpec("dft_band_gap", "scalar_sinusoidal", "standard"),
)

_SPEC_BY_NAME: dict[str, MatterGenConditionSpec] = {
    spec.name: spec for spec in MATTERGEN_CONDITION_SPECS
}


def _mapping(value: object | None) -> Mapping[str, Any] | None:
    """Return a string-keyed mapping, including OmegaConf-like mapping values."""
    if isinstance(value, Mapping):
        return value
    items = getattr(value, "items", None)
    if callable(items):
        raw_items = items()
        return {str(key): item for key, item in raw_items}
    return None


def _get(value: object | None, key: str, default: object | None = None) -> object | None:
    """Read *key* from a mapping or attribute-style Hydra object."""
    mapping = _mapping(value)
    if mapping is not None:
        return mapping.get(key, default)
    return getattr(value, key, default)


def _nested(value: object | None, *keys: str) -> object | None:
    """Look up nested mapping/object fields without imposing a config library."""
    current = value
    for key in keys:
        current = _get(current, key)
        if current is None:
            return None
    return current


def _as_tuple(value: object | None) -> tuple[str, ...]:
    """Normalize a Hydra list-like value to a tuple of strings."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    raise TypeError(f"Expected a condition sequence, got {type(value).__name__}")


def _target_name(value: object | None) -> str:
    """Read a Hydra ``_target_`` value, accepting absent identity declarations."""
    target = _get(value, "_target_", "")
    return target if isinstance(target, str) else ""


def _condition_spec(
    name: str,
    source: object | None,
    *,
    is_adapter: bool,
) -> MatterGenConditionSpec:
    """Parse one expanded official ``PropertyEmbedding`` mapping."""
    if name not in _SPEC_BY_NAME:
        raise ValueError(f"Unsupported MatterGen condition {name!r}")

    fallback = _SPEC_BY_NAME[name]
    conditional = _get(source, "conditional_embedding_module")
    conditional_target = _target_name(conditional)
    if not conditional_target:
        kind = fallback.kind
    elif conditional_target.endswith("NoiseLevelEncoding"):
        kind = "scalar_sinusoidal"
    elif conditional_target.endswith("ChemicalSystemMultiHotEmbedding"):
        kind = "chemical_system_multihot"
    elif conditional_target.endswith("SpaceGroupEmbeddingVector"):
        kind = "space_group_index"
    else:
        raise ValueError(
            f"Unsupported MatterGen condition encoder for {name!r}: {conditional_target!r}"
        )

    scaler_config = _get(source, "scaler")
    scaler_target = _target_name(scaler_config)
    if not scaler_target or scaler_target.endswith("Identity"):
        scaler = "identity"
    elif scaler_target.endswith("StandardScalerTorch"):
        scaler = "standard"
    else:
        raise ValueError(f"Unsupported MatterGen scaler for {name!r}: {scaler_target!r}")

    log10_transform = bool(_get(scaler_config, "log10_transform", False))
    if kind != "scalar_sinusoidal" and scaler != "identity":
        raise ValueError(f"Only scalar MatterGen conditions may use a scaler: {name!r}")
    return MatterGenConditionSpec(
        name=name,
        kind=kind,
        scaler=scaler,
        log10_transform=log10_transform,
        # GemNetTAdapter replaces this source module with ZerosEmbedding after
        # construction, so its checkpoint contains no unconditional vector.
        unconditional="zeros" if is_adapter else "embedding_vector",
        is_adapter=is_adapter,
    )


def _condition_specs(
    value: object | None, *, is_adapter: bool
) -> tuple[MatterGenConditionSpec, ...]:
    """Parse a Hydra ModuleDict mapping in source's sorted concatenation order."""
    entries = _mapping(value)
    if not entries:
        return ()
    return tuple(
        _condition_spec(str(name), source, is_adapter=is_adapter)
        for name, source in sorted(entries.items())
    )


@dataclasses.dataclass
class MatterGenConfig(BaseModelConfig):
    """Configuration of the v1.0.3 GemNet-T MatterGen neural score core."""

    model_id: str = MATTERGEN_MODEL_ID
    revision: str = MATTERGEN_HUB_REVISION
    source_commit: str = MATTERGEN_SOURCE_COMMIT
    variant: str = "mattergen_base"

    # BaseModelConfig compatibility fields.
    vocab_size: int = 101
    hidden_size: int = 512
    num_hidden_layers: int = 4
    hidden_act: str | None = "silu"
    dtype: ir.DataType = ir.DataType.FLOAT

    # GemNet-T dimensions and graph construction bounds from mattergen.yaml.
    num_targets: int = 1
    num_spherical: int = 7
    num_radial: int = 128
    num_blocks: int = 4
    emb_size_atom: int = 512
    emb_size_edge: int = 512
    emb_size_trip: int = 64
    emb_size_rbf: int = 16
    emb_size_cbf: int = 16
    emb_size_bil_trip: int = 64
    num_before_skip: int = 1
    num_after_skip: int = 2
    num_concat: int = 1
    num_atom: int = 3
    cutoff: float = 7.0
    max_neighbors: int = 50
    max_cell_images_per_dim: int = 5
    num_atom_types: int = 101
    denoise_atom_types: bool = True
    atom_type_diffusion: str = "mask"
    regress_stress: bool = True

    # ``condition_family`` documents all legal source identifiers; the two
    # embedding tuples declare only paths actually instantiated in this graph.
    condition_family: tuple[str, ...] = MATTERGEN_CONDITION_FAMILY
    condition_catalog: tuple[MatterGenConditionSpec, ...] = MATTERGEN_CONDITION_SPECS
    property_embeddings: tuple[MatterGenConditionSpec, ...] = ()
    property_embeddings_adapt: tuple[MatterGenConditionSpec, ...] = ()
    condition_on_adapt: tuple[str, ...] = ()

    # A later task consumes this stable, config-derived list to declare one
    # tensor value and one bool ``use_unconditional`` mask per condition.
    condition_input_prefix: str = "condition"

    _MODEL_PATH: ClassVar[tuple[str, ...]] = (
        "lightning_module",
        "diffusion_module",
        "model",
    )

    @property
    def latent_dim(self) -> int:
        """GemNet atom latent width: noise encoding plus base property encodings."""
        return self.hidden_size * (1 + len(self.property_embeddings))

    @property
    def condition_input_specs(self) -> tuple[MatterGenConditionSpec, ...]:
        """Config-ordered inputs: source sorts each ModuleDict lexicographically."""
        return self.property_embeddings + self.property_embeddings_adapt

    def validate(self) -> None:
        """Validate the source-compatible dimensions and property declarations."""
        if self.dtype != ir.DataType.FLOAT:
            raise ValueError(
                "MatterGen score-core exports preserve the official float32 numerical contract; "
                "f16 and bf16 are not assessed."
            )
        positive_fields = (
            "hidden_size",
            "num_targets",
            "num_spherical",
            "num_radial",
            "num_blocks",
            "emb_size_atom",
            "emb_size_edge",
            "emb_size_trip",
            "emb_size_rbf",
            "emb_size_cbf",
            "emb_size_bil_trip",
            "num_before_skip",
            "num_after_skip",
            "num_concat",
            "num_atom",
            "max_neighbors",
            "max_cell_images_per_dim",
        )
        for name in positive_fields:
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.cutoff <= 0:
            raise ValueError("cutoff must be positive")
        if self.num_targets != 1:
            raise ValueError("MatterGen v1 score checkpoints require num_targets=1")
        if self.hidden_size != self.emb_size_atom or self.hidden_size != self.emb_size_edge:
            raise ValueError(
                "MatterGen's hidden_dim, emb_size_atom, and emb_size_edge must match"
            )
        if self.num_atom_types != 101 or self.vocab_size != 101:
            raise ValueError("MatterGen mask diffusion requires exactly 101 atom logits")
        if self.atom_type_diffusion != "mask" or not self.denoise_atom_types:
            raise ValueError(
                "Only the official masked atom-type diffusion score core is supported"
            )
        if not self.regress_stress:
            raise ValueError("MatterGen score core requires the official lattice update head")

        names = [spec.name for spec in self.condition_input_specs]
        if len(names) != len(set(names)):
            raise ValueError("A MatterGen condition cannot occur in both embedding families")
        for spec in self.condition_input_specs:
            if spec.name not in self.condition_family:
                raise ValueError(
                    f"Condition {spec.name!r} is outside the MatterGen condition family"
                )
            if spec.kind not in {
                "scalar_sinusoidal",
                "chemical_system_multihot",
                "space_group_index",
            }:
                raise ValueError(f"Unsupported condition encoder kind {spec.kind!r}")
            if spec.scaler not in {"identity", "standard"}:
                raise ValueError(f"Unsupported condition scaler {spec.scaler!r}")
            if spec.kind != "scalar_sinusoidal" and (
                spec.scaler != "identity" or spec.log10_transform
            ):
                raise ValueError(f"Only scalar condition {spec.name!r} may be scaled")
        adapter_names = tuple(spec.name for spec in self.property_embeddings_adapt)
        if self.condition_on_adapt != adapter_names:
            raise ValueError(
                "condition_on_adapt must exactly match property_embeddings_adapt in sorted order"
            )

    @classmethod
    def from_hydra_config(
        cls,
        config: object,
        *,
        model_id: str = MATTERGEN_MODEL_ID,
        revision: str = MATTERGEN_HUB_REVISION,
        source_commit: str = MATTERGEN_SOURCE_COMMIT,
        variant: str | None = None,
    ) -> MatterGenConfig:
        """Parse an expanded official Hydra config without importing YAML tooling.

        ``config`` may be the whole saved configuration, the nested
        ``lightning_module.diffusion_module.model`` mapping, or an OmegaConf-like
        attribute mapping.  The method extracts only source-proven settings and
        leaves graph construction and sampling ownership to the host task.
        """
        defaults = cls()
        # Released adapter configs retain the base denoiser beneath
        # ``lightning_module`` for training, but their checkpoint belongs to
        # ``adapter.adapter``. Prefer that concrete score module when present.
        model = _nested(config, "adapter", "adapter")
        if model is None:
            model = _nested(config, *cls._MODEL_PATH)
        if model is None:
            candidate = _get(config, "model")
            model = candidate if _get(candidate, "gemnet") is not None else config
        gemnet = _get(model, "gemnet")
        if gemnet is None:
            gemnet = model

        base_properties = _condition_specs(
            _get(model, "property_embeddings"), is_adapter=False
        )
        adapter_properties = _condition_specs(
            _get(model, "property_embeddings_adapt"), is_adapter=True
        )
        declared_adapt = _as_tuple(_get(gemnet, "condition_on_adapt"))
        if not declared_adapt:
            declared_adapt = tuple(spec.name for spec in adapter_properties)
        expected_adapt = tuple(spec.name for spec in adapter_properties)
        if declared_adapt != expected_adapt:
            raise ValueError(
                "GemNet condition_on_adapt must match property_embeddings_adapt in sorted order"
            )

        atom_embedding = _get(gemnet, "atom_embedding")
        with_mask_type = _get(atom_embedding, "with_mask_type", True)
        if with_mask_type is not True:
            raise ValueError("Only official mask-type AtomEmbedding checkpoints are supported")

        def source_value(name: str, default: Any) -> Any:
            value = _get(gemnet, name)
            return default if value is None else value

        hidden_size = _get(model, "hidden_dim", defaults.hidden_size)
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, int):
            raise TypeError("MatterGen model.hidden_dim must be an integer.")
        parsed_variant = variant or _get(config, "variant") or _get(model, "variant")
        result = cls(
            model_id=model_id,
            revision=revision,
            source_commit=source_commit,
            variant=str(parsed_variant or defaults.variant),
            vocab_size=101,
            hidden_size=hidden_size,
            num_hidden_layers=int(source_value("num_blocks", defaults.num_hidden_layers)),
            num_targets=int(source_value("num_targets", defaults.num_targets)),
            num_spherical=int(source_value("num_spherical", defaults.num_spherical)),
            num_radial=int(source_value("num_radial", defaults.num_radial)),
            num_blocks=int(source_value("num_blocks", defaults.num_blocks)),
            emb_size_atom=int(source_value("emb_size_atom", defaults.emb_size_atom)),
            emb_size_edge=int(source_value("emb_size_edge", defaults.emb_size_edge)),
            emb_size_trip=int(source_value("emb_size_trip", defaults.emb_size_trip)),
            emb_size_rbf=int(source_value("emb_size_rbf", defaults.emb_size_rbf)),
            emb_size_cbf=int(source_value("emb_size_cbf", defaults.emb_size_cbf)),
            emb_size_bil_trip=int(
                source_value("emb_size_bil_trip", defaults.emb_size_bil_trip)
            ),
            num_before_skip=int(source_value("num_before_skip", defaults.num_before_skip)),
            num_after_skip=int(source_value("num_after_skip", defaults.num_after_skip)),
            num_concat=int(source_value("num_concat", defaults.num_concat)),
            num_atom=int(source_value("num_atom", defaults.num_atom)),
            cutoff=float(source_value("cutoff", defaults.cutoff)),
            max_neighbors=int(source_value("max_neighbors", defaults.max_neighbors)),
            max_cell_images_per_dim=int(
                source_value("max_cell_images_per_dim", defaults.max_cell_images_per_dim)
            ),
            num_atom_types=101,
            denoise_atom_types=bool(
                _get(model, "denoise_atom_types", defaults.denoise_atom_types)
            ),
            atom_type_diffusion=str(
                _get(model, "atom_type_diffusion", defaults.atom_type_diffusion)
            ),
            regress_stress=bool(source_value("regress_stress", defaults.regress_stress)),
            property_embeddings=base_properties,
            property_embeddings_adapt=adapter_properties,
            condition_on_adapt=declared_adapt,
        )
        result.validate()
        return result


__all__ = [
    "MATTERGEN_CONDITION_FAMILY",
    "MATTERGEN_CONDITION_SPECS",
    "MATTERGEN_HUB_REVISION",
    "MATTERGEN_MODEL_ID",
    "MATTERGEN_SOURCE_COMMIT",
    "MatterGenConditionSpec",
    "MatterGenConfig",
]
