from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import onnx_ir as ir
import yaml


def _dimension(dimension: Any) -> int | str | None:
    return getattr(dimension, "value", dimension)


def _value(value: ir.Value | None) -> Any:
    if value is None:
        return None
    return {
        "name": value.name,
        "dtype": str(value.dtype),
        "shape": (
            [_dimension(dimension) for dimension in value.shape]
            if value.shape is not None
            else None
        ),
    }


def _tensor(tensor: ir.TensorProtocol) -> dict[str, Any]:
    return {
        "dtype": str(tensor.dtype),
        "shape": [_dimension(dimension) for dimension in tensor.shape],
        "sha256": hashlib.sha256(tensor.tobytes()).hexdigest(),
    }


def _graph(graph: ir.Graph) -> dict[str, Any]:
    return {
        "inputs": [_value(value) for value in graph.inputs],
        "outputs": [_value(value) for value in graph.outputs],
        "initializers": {
            name: _tensor(value.const_value)
            for name, value in sorted(graph.initializers.items())
            if value.const_value is not None
        },
    }


def _model(path: Path) -> dict[str, Any]:
    model = ir.load(path)
    return _graph(model.graph)


# Bytes the generator produces but the tree does not carry.  They are a
# deterministic function of the generator, so a committed copy would restate
# it in a form no reviewer can read; ``_artifacts_exist`` below checks that
# every one the metadata names was actually produced.
_GENERATED_SUFFIXES = {".onnx", ".data", ".safetensors"}


def _reviewable_files(directory: Path) -> set[Path]:
    """The files that are committed, and so are the reviewable contract."""
    return {
        path.relative_to(directory)
        for path in directory.rglob("*")
        if path.is_file() and path.suffix not in _GENERATED_SUFFIXES
    }


def _artifacts_exist(committed: Path, generated: Path) -> list[str]:
    """Every artifact the committed metadata names must have been generated.

    The graphs are no longer committed, so nothing else would notice if the
    generator stopped emitting one: the metadata would still compare equal and
    the package would still look complete.
    """
    missing: list[str] = []
    for metadata_path in sorted(committed.rglob("inference_metadata.yaml")):
        package = metadata_path.parent.relative_to(committed)
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        workflow = ((metadata.get("pipeline") or {}).get("workflow")) or {}
        for name, component in (workflow.get("components") or {}).items():
            implementation = component.get("implementation") or {}
            artifact = implementation.get("artifact")
            if implementation.get("kind") != "onnx" or not artifact:
                continue
            if not (generated / package / artifact).exists():
                missing.append(f"{package}: component {name!r} -> {artifact}")
    return missing


def _content(path: Path) -> Any:
    if path.suffix == ".onnx":
        return _model(path)
    if path.suffix in {".yaml", ".yml"}:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return path.read_bytes()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    args = parser.parse_args()

    expected_files = _reviewable_files(args.expected)
    actual_files = _reviewable_files(args.actual)
    if expected_files != actual_files:
        missing = sorted(str(path) for path in expected_files - actual_files)
        extra = sorted(str(path) for path in actual_files - expected_files)
        raise SystemExit(f"package file mismatch: missing={missing}, extra={extra}")

    if absent := _artifacts_exist(args.expected, args.actual):
        raise SystemExit(f"declared artifact was not generated: {absent}")

    changed = [
        str(relative)
        for relative in sorted(expected_files)
        if _content(args.expected / relative) != _content(args.actual / relative)
    ]
    if changed:
        raise SystemExit(f"package semantic mismatch: {changed}")


if __name__ == "__main__":
    main()
