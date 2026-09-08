# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quantization format codecs for existing packed checkpoint weights."""

from __future__ import annotations

__all__ = [
    "QuantizationCodec",
    "QuantizationCodecRegistry",
    "codec_registry",
]

from collections.abc import Mapping
from typing import Protocol

import torch

from mobius._component_manifest import ComponentDescriptor
from mobius._configs import QuantizationConfig
from mobius._weight_utils import preprocess_quantized_weights
from mobius.weights._records import (
    FloatWeight,
    PackedWeight,
    WeightBundle,
    WeightRecord,
)


class QuantizationCodec(Protocol):
    """Groups and normalizes one producer's existing packed weight layout."""

    method: str

    def group(
        self,
        component: ComponentDescriptor,
        state_dict: Mapping[str, torch.Tensor],
        config: QuantizationConfig,
    ) -> WeightBundle:
        """Group checkpoint sidecars into typed logical records."""
        ...

    def normalize(
        self,
        record: WeightRecord,
        config: QuantizationConfig,
    ) -> dict[str, torch.Tensor]:
        """Convert one packed record to Mobius's canonical parameter layout."""
        ...


class QuantizationCodecRegistry:
    """Registry keyed by serialized ``quant_method``."""

    def __init__(self) -> None:
        self._codecs: dict[str, QuantizationCodec] = {}

    def register(self, codec: QuantizationCodec) -> None:
        if not codec.method:
            raise ValueError("quantization codec method must not be empty")
        if codec.method in self._codecs:
            raise ValueError(f"quantization codec {codec.method!r} is already registered")
        self._codecs[codec.method] = codec

    def get(self, method: str) -> QuantizationCodec:
        try:
            return self._codecs[method]
        except KeyError:
            raise KeyError(
                f"No quantization codec registered for {method!r}. "
                f"Available methods: {sorted(self._codecs)}"
            ) from None

    def __contains__(self, method: str) -> bool:
        return method in self._codecs


class _LegacyAffineCodec:
    """Typed facade over the existing Olive/GPTQ/AWQ normalization helpers."""

    def __init__(self, method: str):
        self.method = method

    @staticmethod
    def _logical_name(qweight_key: str) -> tuple[str, str, str, str | None]:
        if qweight_key.endswith("_qweight"):
            stem = qweight_key[: -len("_qweight")]
            logical_name = stem if stem.endswith(".weight") else f"{stem}.weight"
            return (
                logical_name,
                f"{stem}_scales",
                qweight_key,
                f"{stem}_qzeros",
            )
        if qweight_key.endswith(".qweight"):
            stem = qweight_key[: -len(".qweight")]
            return (
                f"{stem}.weight",
                f"{stem}.scales",
                qweight_key,
                f"{stem}.qzeros",
            )
        raise ValueError(f"{qweight_key!r} is not a packed qweight key")

    def group(
        self,
        component: ComponentDescriptor,
        state_dict: Mapping[str, torch.Tensor],
        config: QuantizationConfig,
    ) -> WeightBundle:
        if config.quant_method != self.method:
            raise ValueError(
                f"codec {self.method!r} cannot group quant_method {config.quant_method!r}"
            )

        records: dict[str, WeightRecord] = {}
        consumed: set[str] = set()
        qweight_keys = sorted(
            key for key in state_dict if key.endswith(("_qweight", ".qweight"))
        )
        for qweight_key in qweight_keys:
            logical_name, scales_key, _, zero_points_key = self._logical_name(qweight_key)
            if scales_key not in state_dict:
                raise ValueError(
                    f"Packed weight {qweight_key!r} is missing scales {scales_key!r}"
                )
            zero_points = state_dict.get(zero_points_key)
            if not config.sym and zero_points is None:
                raise ValueError(
                    f"Asymmetric packed weight {qweight_key!r} is missing "
                    f"zero points {zero_points_key!r}"
                )
            storage = PackedWeight(
                qweight=state_dict[qweight_key],
                scales=state_dict[scales_key],
                zero_points=zero_points,
                qweight_key=qweight_key,
                scales_key=scales_key,
                zero_points_key=zero_points_key if zero_points is not None else None,
                method=self.method,
            )
            if logical_name in records:
                raise ValueError(
                    f"Checkpoint declares logical weight {logical_name!r} more than once"
                )
            records[logical_name] = WeightRecord(
                name=logical_name,
                component=component.name,
                storage=storage,
            )
            consumed.update(storage.source_keys)

        orphan_sidecars = sorted(
            key
            for key in state_dict
            if key.endswith(("_scales", "_qzeros", ".scales", ".qzeros"))
            and key not in consumed
        )
        if orphan_sidecars:
            raise ValueError(
                f"Packed checkpoint sidecars have no matching qweight: {orphan_sidecars}"
            )

        for key, value in state_dict.items():
            if key in consumed:
                continue
            if key in records:
                raise ValueError(f"Float and packed checkpoint values both target {key!r}")
            records[key] = WeightRecord(
                name=key,
                component=component.name,
                storage=FloatWeight(value=value, source_key=key),
            )
        return WeightBundle(component=component, records=records)

    def normalize(
        self,
        record: WeightRecord,
        config: QuantizationConfig,
    ) -> dict[str, torch.Tensor]:
        if not isinstance(record.storage, PackedWeight):
            raise TypeError(f"weight {record.name!r} is not packed")
        if record.storage.method != self.method:
            raise ValueError(
                f"weight {record.name!r} uses {record.storage.method!r}, "
                f"not codec {self.method!r}"
            )
        return preprocess_quantized_weights(
            record.storage.as_state_dict(),
            config,
            tie_embeddings=False,
            qmoe_target_path=None,
        )


codec_registry = QuantizationCodecRegistry()
for _method in ("olive", "gptq", "awq"):
    codec_registry.register(_LegacyAffineCodec(_method))
