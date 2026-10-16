"""Read-only cache probe: check writes, preserved history, and agreement with HF."""

from __future__ import annotations

import numpy as np


def snapshot_cache(runner, name, show_layout=False):
    """Validate paired cache I/O and return an independent CPU copy."""
    shape, dtype = runner.validate_tensor(name, show_layout)
    updated_shape, updated_dtype = runner.validate_tensor("updated_" + name, show_layout)
    if len(shape) != 4 or shape != updated_shape or dtype != updated_dtype:
        raise ValueError(f"{name}: cache probe requires matching [B,H,S,D] device I/O")
    return runner.read_tensor(name)


def report_cache(name, before, after, expected, position, length):
    """Report changed storage slots separately from numerical agreement with HF."""
    end = position + length
    valid = after[:, :, :end, :].astype(np.float32)
    if valid.shape != expected.shape:
        raise ValueError(f"{name}: TRT/HF cache shapes differ: {valid.shape}/{expected.shape}")
    # [B,H,S,D,bytes_per_element]: reduce every axis except the sequence slots.
    raw_before = before.view(np.uint8).reshape(*before.shape, before.dtype.itemsize)
    raw_after = after.view(np.uint8).reshape(*after.shape, after.dtype.itemsize)
    changed = np.any(raw_before != raw_after, axis=(0, 1, 3, 4))
    slots = np.flatnonzero(changed)
    print(
        f"\nCache {name}: write=[{position}:{end}], "
        f"changed_slots={slots[:32].tolist()} (total={slots.size})"
    )
    print(
        f"  Old prefix bitwise unchanged: {not changed[:position].any()}; "
        f"unused tail bitwise unchanged: {not changed[end:].any()}"
    )
    for region, actual, reference_values in (
        ("valid prefix", valid, expected),
        ("new slots", valid[:, :, position:end, :], expected[:, :, position:end, :]),
    ):
        if not np.isfinite(actual).all() or not np.isfinite(reference_values).all():
            print(f"  {region} vs HF: non-finite values; numerical comparison invalid")
            continue
        error = np.abs(actual - reference_values)
        print(f"  {region} vs HF: mean_abs={error.mean():.6f}; max_abs={error.max():.6f}")


class CacheInspector:
    """Inspect one layer's K and V; the generation loop controls snapshot timing."""

    def __init__(self, runner, layer):
        self.layer = layer
        self.names = [f"{kind}_cache.{layer}" for kind in ("key", "value")]
        if any(name not in runner.caches for name in self.names):
            raise ValueError(f"Engine does not contain cache layer {layer}")
        self.runner = runner

    def snapshot(self, show_layout=False):
        return {name: snapshot_cache(self.runner, name, show_layout) for name in self.names}

    def compare(self, before, after, reference_cache, position, length):
        reference_layer = reference_cache.layers[self.layer]
        for name, tensor in zip(self.names, (reference_layer.keys, reference_layer.values)):
            expected = tensor.detach().float().cpu().numpy().copy()
            report_cache(name, before[name], after[name], expected, position, length)
