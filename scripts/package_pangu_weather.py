#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Package and execute the published Pangu-Weather 1-hour ONNX checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import urllib.request
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import yaml

MODEL_URL = "https://paddle-org.bj.bcebos.com/paddlescience/models/Pangu/pangu_weather_1.onnx"
MODEL_SHA256 = "179e5029c453ae459dfd52f14610a6c5f5ad39f1371985744ea0ce6c546fda2a"
UPSTREAM_REVISION = "72bdd99096721e1a1f8912c37a9a3aff9ff0a4f2"
EXPECTED_PORTS = {
    "inputs": {
        "input": [5, 13, 721, 1440],
        "input_surface": [4, 721, 1440],
    },
    "outputs": {
        "output": [5, 13, 721, 1440],
        "output_surface": [4, 721, 1440],
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape(value: ir.Value) -> list[int | str | None]:
    if value.shape is None:
        raise ValueError(f"Graph value {value.name!r} has no shape")
    return [getattr(dimension, "value", dimension) for dimension in value.shape]


def _inspect_graph(path: Path) -> dict[str, object]:
    model = ir.load(path)
    inputs = {value.name: _shape(value) for value in model.graph.inputs}
    outputs = {value.name: _shape(value) for value in model.graph.outputs}
    if inputs != EXPECTED_PORTS["inputs"] or outputs != EXPECTED_PORTS["outputs"]:
        raise ValueError(
            f"Unexpected Pangu-Weather graph ports: inputs={inputs}, outputs={outputs}"
        )
    return {
        "ir_version": model.ir_version,
        "opsets": [
            {"domain": domain or "ai.onnx", "version": version}
            for domain, version in model.opset_imports.items()
        ],
        "inputs": inputs,
        "outputs": outputs,
    }


def _download(path: Path) -> None:
    request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "mobius-catalogue"})
    with urllib.request.urlopen(request) as response, path.open("wb") as output:
        while chunk := response.read(8 << 20):
            output.write(chunk)


def _request() -> tuple[np.ndarray, np.ndarray]:
    # A deterministic, physically scaled baseline verifies the real graph without
    # claiming forecast skill or substituting for an observed ERA5 analysis.
    upper = np.empty((5, 13, 721, 1440), dtype=np.float32)
    upper[0] = 50_000.0  # geopotential
    upper[1] = 0.005  # specific humidity
    upper[2] = 270.0  # temperature
    upper[3:] = 0.0  # U/V wind
    surface = np.empty((4, 721, 1440), dtype=np.float32)
    surface[0] = 101_325.0  # mean sea-level pressure
    surface[1:3] = 0.0  # U10/V10
    surface[3] = 288.0  # T2M
    return upper, surface


def _stats(array: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "finite": bool(np.isfinite(array).all()),
    }


def _run(
    model_path: Path, provider: str
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    upper, surface = _request()
    providers: list[str | tuple[str, dict[str, str]]]
    if provider == "cuda":
        providers = [
            ("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
            "CPUExecutionProvider",
        ]
    else:
        providers = ["CPUExecutionProvider"]
    options = ort.SessionOptions()
    options.enable_mem_pattern = False
    start = time.perf_counter()
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
    init_seconds = time.perf_counter() - start
    start = time.perf_counter()
    output, output_surface = session.run(None, {"input": upper, "input_surface": surface})
    inference_seconds = time.perf_counter() - start
    runtime = {
        "session_init_seconds": init_seconds,
        "inference_seconds": inference_seconds,
        "providers": session.get_providers(),
        "onnxruntime_version": ort.__version__,
        "python_version": platform.python_version(),
        "request": {
            "input": _stats(upper),
            "input_surface": _stats(surface),
            "description": (
                "Deterministic physically-scaled baseline field; not an observed "
                "ERA5 analysis."
            ),
        },
        "output": {
            "output": _stats(output),
            "output_surface": _stats(output_surface),
        },
        "samples": {
            "z_1000hpa_lat360_lon0": float(output[0, 0, 360, 0]),
            "t_850hpa_lat360_lon0": float(output[2, 2, 360, 0]),
            "mslp_lat360_lon0": float(output_surface[0, 360, 0]),
            "t2m_lat360_lon0": float(output_surface[3, 360, 0]),
        },
    }
    return runtime, upper, surface, output, output_surface


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model.onnx"
    if not args.skip_download:
        _download(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    checksum = _sha256(model_path)
    if checksum != MODEL_SHA256:
        raise ValueError(f"Model SHA-256 mismatch: {checksum}")
    graph = _inspect_graph(model_path)

    runtime, upper, surface, output, output_surface = _run(model_path, args.provider)
    np.savez_compressed(args.output_dir / "request.npz", input=upper, input_surface=surface)
    np.savez_compressed(
        args.output_dir / "output.npz", output=output, output_surface=output_surface
    )
    (args.output_dir / "runtime_output.json").write_text(
        json.dumps(runtime, indent=2) + "\n", encoding="utf-8"
    )

    contracts = {
        "input": {
            "dtype": "float32",
            "rank": 4,
            "shape": EXPECTED_PORTS["inputs"]["input"],
        },
        "input_surface": {
            "dtype": "float32",
            "rank": 3,
            "shape": EXPECTED_PORTS["inputs"]["input_surface"],
        },
        "output": {
            "dtype": "float32",
            "rank": 4,
            "shape": EXPECTED_PORTS["outputs"]["output"],
        },
        "output_surface": {
            "dtype": "float32",
            "rank": 3,
            "shape": EXPECTED_PORTS["outputs"]["output_surface"],
        },
    }
    metadata = {
        "schema_version": "v1",
        "pipeline": {
            "workflow": {
                "manifest": {"capabilities": ["workflow_ssa", "linear_effects", "typed_emit"]},
                "effects": {
                    "forecast": {
                        "retry": "pure",
                        "speculation_safety": {"kind": "clonable"},
                    }
                },
                "inputs": {
                    "request.input": {
                        "contract": contracts["input"],
                        "role": {"kind": "opaque"},
                        "source": {"kind": "application", "name": "request.input"},
                        "required": True,
                    },
                    "request.input_surface": {
                        "contract": contracts["input_surface"],
                        "role": {"kind": "opaque"},
                        "source": {
                            "kind": "application",
                            "name": "request.input_surface",
                        },
                        "required": True,
                    },
                },
                "outputs": {
                    "output": {
                        "contract": contracts["output"],
                        "role": "tensor",
                        "stage": "post_adapter",
                    },
                    "output_surface": {
                        "contract": contracts["output_surface"],
                        "role": "tensor",
                        "stage": "post_adapter",
                    },
                },
                "components": {
                    "forecast": {
                        "implementation": {"kind": "onnx", "artifact": "model.onnx"},
                    }
                },
                "steps": [
                    {
                        "kind": "invoke",
                        "component": "forecast",
                        "inputs": {
                            "input": "request.input",
                            "input_surface": "request.input_surface",
                        },
                        "outputs": {
                            "output": "forecast.output",
                            "output_surface": "forecast.output_surface",
                        },
                    },
                    {
                        "kind": "emit",
                        "value": "forecast.output",
                        "output": "output",
                        "mode": "replace",
                    },
                    {
                        "kind": "emit",
                        "value": "forecast.output_surface",
                        "output": "output_surface",
                        "mode": "replace",
                    },
                ],
            }
        },
    }
    graph_report = {
        "task": "weather-forecast",
        "forecast_horizon_hours": 1,
        "state": {"kind": "stateless"},
        "model": {
            "path": "model.onnx",
            "format": "ONNX",
            "sha256": checksum,
            "graph": graph,
        },
        "semantics": {
            "inputs": {
                "input": {
                    "variables": ["Z", "Q", "T", "U", "V"],
                    "pressure_levels_hpa": [
                        1000,
                        925,
                        850,
                        700,
                        600,
                        500,
                        400,
                        300,
                        250,
                        200,
                        150,
                        100,
                        50,
                    ],
                },
                "input_surface": {
                    "variables": ["MSLP", "U10", "V10", "T2M"],
                },
            },
            "outputs": {
                "output": {
                    "semantics": "1-hour upper-air forecast in input variable order",
                },
                "output_surface": {
                    "semantics": "1-hour surface forecast in input variable order",
                },
            },
        },
        "grid": {
            "latitude": {"count": 721, "range_degrees": [90.0, -90.0], "step": -0.25},
            "longitude": {"count": 1440, "range_degrees": [0.0, 359.75], "step": 0.25},
        },
        "artifacts": {
            "request": "request.npz",
            "output": "output.npz",
            "runtime_summary": "runtime_output.json",
            "versions": "versions.json",
            "provenance": "provenance.json",
            "inference_metadata": "inference_metadata.yaml",
        },
    }
    (args.output_dir / "inference_metadata.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8"
    )
    (args.output_dir / "graph_report.json").write_text(
        json.dumps(graph_report, indent=2) + "\n", encoding="utf-8"
    )
    provenance = {
        "model": "Pangu-Weather 1-hour",
        "source_url": MODEL_URL,
        "source_etag": '"-c4de877700578d5d942e3a1bded2c274"',
        "source_last_modified": "Mon, 03 Mar 2025 13:00:58 GMT",
        "upstream_repository": "https://github.com/198808xc/Pangu-Weather",
        "upstream_revision": UPSTREAM_REVISION,
        "model_sha256": checksum,
        "license": {
            "spdx": "CC-BY-NC-SA-4.0",
            "url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
            "commercial_use": "forbidden by upstream model card",
        },
        "gating": "none",
        "runtime": runtime,
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "onnx": importlib_metadata.version("onnx"),
        "onnx_ir": importlib_metadata.version("onnx-ir"),
        "onnxruntime": ort.__version__,
        "onnx_ir_version": graph["ir_version"],
        "onnx_opsets": graph["opsets"],
    }
    (args.output_dir / "versions.json").write_text(
        json.dumps(versions, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "LICENSE.model.txt").write_text(
        "Pangu-Weather trained parameters: CC BY-NC-SA 4.0.\n"
        "Commercial use is forbidden by the upstream model card.\n"
        "Terms: https://creativecommons.org/licenses/by-nc-sa/4.0/\n"
        f"Pinned model card: https://github.com/198808xc/Pangu-Weather/blob/"
        f"{UPSTREAM_REVISION}/README.md#license\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
