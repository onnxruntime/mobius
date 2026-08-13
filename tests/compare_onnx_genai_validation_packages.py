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


def _relative_files(directory: Path) -> set[Path]:
    return {
        path.relative_to(directory)
        for path in directory.rglob("*")
        if path.is_file() and path.name != "model.onnx.data"
    }


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

    expected_files = _relative_files(args.expected)
    actual_files = _relative_files(args.actual)
    if expected_files != actual_files:
        missing = sorted(str(path) for path in expected_files - actual_files)
        extra = sorted(str(path) for path in actual_files - expected_files)
        raise SystemExit(f"package file mismatch: missing={missing}, extra={extra}")

    changed = [
        str(relative)
        for relative in sorted(expected_files)
        if _content(args.expected / relative) != _content(args.actual / relative)
    ]
    if changed:
        raise SystemExit(f"package semantic mismatch: {changed}")


if __name__ == "__main__":
    main()
