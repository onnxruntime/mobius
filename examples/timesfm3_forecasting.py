#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run the five-component TimesFM 3 ONNX forecasting pipeline.

The example builds the pinned ``google/timesfm-3.0-pytorch`` checkpoint, or
loads an already-saved ModelPackage directory, and runs:

``raw_preprocessor -> preprocessor -> model -> postprocessor -> stitcher``.

Only the learned ``model`` component runs on CUDA. The control-flow-heavy
components stay on CPU, leaving one host/device boundary on either side of the
learned core. CUDA inference uses persistent I/O binding and fixed-shape
OrtValues; ``--cuda-graph`` additionally captures that core.

The official checkpoint weights are subject to the TimesFM Non-Commercial
License v1.0. Review that license before downloading or using the weights.

Usage::

    # Build, save, and run the pinned checkpoint on CPU
    python examples/timesfm3_forecasting.py --output-dir output/timesfm3

    # Reuse a saved package and benchmark its learned core on CUDA
    python examples/timesfm3_forecasting.py --model-dir output/timesfm3 \
        --device cuda --dtype f16 --cuda-graph --benchmark

    # Compare the ONNX core and end-user forecast with official PyTorch
    python examples/timesfm3_forecasting.py --model-dir output/timesfm3 \
        --device cuda --dtype f16 --compare-pytorch --benchmark
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

MODEL_ID = "google/timesfm-3.0-pytorch"
REVISION = "900fcab43d1bfe71733a33b3fec61a41fce28a27"
COMPONENTS = (
    "raw_preprocessor",
    "preprocessor",
    "model",
    "postprocessor",
    "stitcher",
)
QUANTILES = tuple(i / 10 for i in range(1, 10))
ORT_NUMPY_DTYPES = {
    "tensor(float)": np.dtype(np.float32),
    "tensor(float16)": np.dtype(np.float16),
}


def _run_session(
    session: ort.InferenceSession, feeds: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    names = [output.name for output in session.get_outputs()]
    return dict(zip(names, session.run(names, feeds)))


def _model_path(root: Path, component: str) -> Path:
    path = root / component / "model.onnx"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing TimesFM component {component!r}: {path}. "
            "Expected <model-dir>/<component>/model.onnx."
        )
    return path


def _percentile(samples: list[float], percentile: float) -> float:
    """Return a linearly interpolated percentile without extra dependencies."""
    ordered = sorted(samples)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _latency_summary(samples: list[float]) -> str:
    return (
        f"p50/median={statistics.median(samples):.3f} ms, "
        f"p95={_percentile(samples, 0.95):.3f} ms"
    )


@dataclass(frozen=True)
class ForecastResult:
    point: np.ndarray
    quantiles: np.ndarray
    validity: np.ndarray
    model_inputs: np.ndarray
    patch_mask: np.ndarray
    raw_logits: np.ndarray
    component_ms: dict[str, float]
    end_to_end_ms: float


class TimesFM3OrtDriver:
    """Long-lived ORT sessions for a saved five-component TimesFM 3 package."""

    def __init__(
        self,
        model_dir: str | os.PathLike[str],
        *,
        device: str = "cpu",
        enable_cuda_graph: bool = False,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self.enable_cuda_graph = enable_cuda_graph
        self._binding: ort.IOBinding | None = None
        self._binding_signature: tuple[tuple[int, ...], tuple[int, ...], str] | None = None
        self._host_model_inputs: np.ndarray | None = None
        self._host_patch_mask: np.ndarray | None = None
        self._input_ortvalues: dict[str, ort.OrtValue] = {}
        self._output_ortvalue: ort.OrtValue | None = None
        self._run_options: ort.RunOptions | None = None
        self._next_graph_id = 0

        if device == "cuda" and hasattr(ort, "preload_dlls"):
            # On Windows this also finds CUDA/cuDNN DLLs bundled with PyTorch or
            # NVIDIA site packages, avoiding a dependency on their PATH ordering.
            ort.preload_dlls()
            torch_spec = importlib.util.find_spec("torch") if sys.platform == "win32" else None
            if torch_spec is not None and torch_spec.origin is not None:
                torch_lib = Path(torch_spec.origin).parent / "lib"
                if torch_lib.is_dir():
                    ort.preload_dlls(directory=str(torch_lib))
        available = ort.get_available_providers()
        if "CPUExecutionProvider" not in available:
            raise RuntimeError(
                "CPUExecutionProvider is required for the TimesFM control components; "
                f"available providers: {available}."
            )
        if device == "cuda" and "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDA requested, but CUDAExecutionProvider is unavailable. "
                f"Available providers: {available}. Install an ONNX Runtime GPU package "
                "compatible with your CUDA installation."
            )
        if enable_cuda_graph and device != "cuda":
            raise ValueError("--cuda-graph requires --device cuda.")

        self.sessions = {
            name: self._create_session(name)
            for name in COMPONENTS
            if name != "model" or not self.enable_cuda_graph
        }
        context_meta = next(
            value
            for value in self.sessions["raw_preprocessor"].get_inputs()
            if value.name == "context_values"
        )
        try:
            self.input_dtype = ORT_NUMPY_DTYPES[context_meta.type]
        except KeyError as error:
            raise TypeError(f"Unsupported TimesFM input type: {context_meta.type}.") from error
        self._validate_session_providers()

    def _create_session(
        self,
        component: str,
        fixed_dimensions: dict[str, int] | None = None,
    ) -> ort.InferenceSession:
        options = ort.SessionOptions()
        for name, value in (fixed_dimensions or {}).items():
            options.add_free_dimension_override_by_name(name, value)
        path = str(_model_path(self.model_dir, component))
        if component == "model" and self.device == "cuda":
            cuda_options = {"enable_cuda_graph": "1"} if self.enable_cuda_graph else {}
            providers: list[str | tuple[str, dict[str, str]]] = [
                ("CUDAExecutionProvider", cuda_options),
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]
        return ort.InferenceSession(path, sess_options=options, providers=providers)

    def _validate_session_providers(self) -> None:
        for name, session in self.sessions.items():
            providers = session.get_providers()
            expected = (
                "CUDAExecutionProvider"
                if name == "model" and self.device == "cuda"
                else "CPUExecutionProvider"
            )
            if not providers or providers[0] != expected:
                raise RuntimeError(
                    f"{name!r} did not select {expected}; active providers: {providers}."
                )

    @staticmethod
    def _timed_cpu(
        session: ort.InferenceSession, feeds: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], float]:
        start = time.perf_counter()
        outputs = _run_session(session, feeds)
        return outputs, (time.perf_counter() - start) * 1_000

    def _rebuild_cuda_binding(self, model_inputs: np.ndarray, patch_mask: np.ndarray) -> None:
        model_inputs = np.ascontiguousarray(model_inputs)
        patch_mask = np.ascontiguousarray(patch_mask)
        if self.enable_cuda_graph:
            batch, variates, patches = model_inputs.shape[:3]
            self.sessions["model"] = self._create_session(
                "model",
                {
                    "component.model.batch": batch,
                    "component.model.variates": variates,
                    "component.model.patches": patches,
                },
            )
        self._host_model_inputs = np.empty_like(model_inputs)
        self._host_patch_mask = np.empty_like(patch_mask)
        np.copyto(self._host_model_inputs, model_inputs)
        np.copyto(self._host_patch_mask, patch_mask)

        self._input_ortvalues = {
            "model_inputs": ort.OrtValue.ortvalue_from_numpy(
                self._host_model_inputs, "cuda", 0
            ),
            "patch_mask": ort.OrtValue.ortvalue_from_numpy(self._host_patch_mask, "cuda", 0),
        }
        output_meta = self.sessions["model"].get_outputs()[0]
        output_width = output_meta.shape[-1]
        if not isinstance(output_width, int):
            raise TypeError(f"Expected a fixed raw_logits width, got {output_meta.shape}.")
        output_shape = (*model_inputs.shape[:3], output_width)
        self._output_ortvalue = ort.OrtValue.ortvalue_from_shape_and_type(
            output_shape, model_inputs.dtype, "cuda", 0
        )

        binding = self.sessions["model"].io_binding()
        for name, value in self._input_ortvalues.items():
            binding.bind_ortvalue_input(name, value)
        binding.bind_ortvalue_output("raw_logits", self._output_ortvalue)
        self._binding = binding

        # A captured graph cannot change addresses or shapes. Give each new
        # fixed-shape binding its own graph ID rather than replaying a stale graph.
        self._run_options = None
        if self.enable_cuda_graph:
            self._run_options = ort.RunOptions()
            self._run_options.add_run_config_entry("gpu_graph_id", str(self._next_graph_id))
            self._next_graph_id += 1

    def _run_cuda_model(self, model_inputs: np.ndarray, patch_mask: np.ndarray) -> np.ndarray:
        model_inputs = np.ascontiguousarray(model_inputs)
        patch_mask = np.ascontiguousarray(patch_mask)
        signature = (model_inputs.shape, patch_mask.shape, model_inputs.dtype.str)
        if signature != self._binding_signature:
            self._rebuild_cuda_binding(model_inputs, patch_mask)
            self._binding_signature = signature
        else:
            assert self._host_model_inputs is not None
            assert self._host_patch_mask is not None
            np.copyto(self._host_model_inputs, model_inputs)
            np.copyto(self._host_patch_mask, patch_mask)
            # Both fixed-shape inputs cross the pipeline's sole host-to-device boundary.
            self._input_ortvalues["model_inputs"].update_inplace(self._host_model_inputs)
            self._input_ortvalues["patch_mask"].update_inplace(self._host_patch_mask)

        assert self._binding is not None
        self._binding.synchronize_inputs()
        self.sessions["model"].run_with_iobinding(self._binding, self._run_options)
        self._binding.synchronize_outputs()
        # This is the pipeline's sole device-to-host boundary.
        return self._binding.copy_outputs_to_cpu()[0]

    def run_model(
        self, model_inputs: np.ndarray, patch_mask: np.ndarray
    ) -> tuple[np.ndarray, float]:
        start = time.perf_counter()
        if self.device == "cuda":
            output = self._run_cuda_model(model_inputs, patch_mask)
        else:
            output = _run_session(
                self.sessions["model"],
                {"model_inputs": model_inputs, "patch_mask": patch_mask},
            )["raw_logits"]
        return output, (time.perf_counter() - start) * 1_000

    def benchmark_model(
        self,
        model_inputs: np.ndarray,
        patch_mask: np.ndarray,
        *,
        warmups: int,
        iterations: int,
    ) -> list[float]:
        """Time only learned-core execution, excluding host/device copies."""
        samples: list[float] = []
        if self.device == "cuda":
            self._run_cuda_model(model_inputs, patch_mask)
            assert self._binding is not None
            for _ in range(warmups):
                self.sessions["model"].run_with_iobinding(self._binding, self._run_options)
            self._binding.synchronize_outputs()
            for _ in range(iterations):
                self._binding.synchronize_outputs()
                start = time.perf_counter()
                self.sessions["model"].run_with_iobinding(self._binding, self._run_options)
                self._binding.synchronize_outputs()
                samples.append((time.perf_counter() - start) * 1_000)
        else:
            feeds = {"model_inputs": model_inputs, "patch_mask": patch_mask}
            for _ in range(warmups):
                _run_session(self.sessions["model"], feeds)
            for _ in range(iterations):
                start = time.perf_counter()
                _run_session(self.sessions["model"], feeds)
                samples.append((time.perf_counter() - start) * 1_000)
        return samples

    def forecast(
        self,
        context: np.ndarray,
        horizon: int,
        *,
        make_positive: bool = False,
    ) -> ForecastResult:
        """Forecast one or more variates shaped ``[variates, context]``."""
        context = np.asarray(context)
        if context.ndim == 1:
            context = context[None, :]
        if context.ndim != 2 or context.shape[-1] == 0:
            raise ValueError("context must have shape [context] or [variates, context].")
        if horizon <= 0:
            raise ValueError("horizon must be positive.")

        context = np.ascontiguousarray(context[None, ...])
        batch, variates, context_length = context.shape
        raw_feeds = {
            "context_values": np.nan_to_num(context, nan=0.0),
            "context_observed": np.isfinite(context),
            "future_values": np.zeros((batch, variates, horizon), dtype=context.dtype),
            "future_observed": np.zeros((batch, variates, horizon), dtype=np.bool_),
            "context_lengths": np.full((batch,), context_length, dtype=np.int64),
            "horizon_lengths": np.full((batch,), horizon, dtype=np.int64),
            "variate_roles": np.zeros((batch, variates), dtype=np.int64),
        }

        total_start = time.perf_counter()
        raw, raw_ms = self._timed_cpu(self.sessions["raw_preprocessor"], raw_feeds)
        pre, pre_ms = self._timed_cpu(
            self.sessions["preprocessor"],
            {
                name: raw[name]
                for name in ("values", "masks", "patch_is_target", "patch_cpm_mask")
            },
        )
        raw_logits, model_ms = self.run_model(pre["model_inputs"], pre["patch_mask"])
        post, post_ms = self._timed_cpu(
            self.sessions["postprocessor"],
            {
                "raw_logits": raw_logits,
                "revin_count": pre["revin_count"],
                "revin_mean": pre["revin_mean"],
                "revin_std": pre["revin_std"],
                "patch_cpm_mask": raw["patch_cpm_mask"],
            },
        )
        stitch, stitch_ms = self._timed_cpu(
            self.sessions["stitcher"],
            {
                "logits": post["logits"],
                "make_positive": np.asarray(make_positive, dtype=np.bool_),
                **{
                    name: raw[name]
                    for name in (
                        "trend_slope",
                        "trend_intercept",
                        "apply_detrend",
                        "target_mask",
                        "nonnegative_mask",
                        "context_lengths",
                        "horizon_lengths",
                        "context_patch_count",
                        "forecast_patch_counts",
                    )
                },
            },
        )
        end_to_end_ms = (time.perf_counter() - total_start) * 1_000
        return ForecastResult(
            point=stitch["point_forecast"],
            quantiles=stitch["quantile_forecasts"],
            validity=stitch["validity"],
            model_inputs=pre["model_inputs"],
            patch_mask=pre["patch_mask"],
            raw_logits=raw_logits,
            component_ms={
                "raw_preprocessor": raw_ms,
                "preprocessor": pre_ms,
                "model": model_ms,
                "postprocessor": post_ms,
                "stitcher": stitch_ms,
            },
            end_to_end_ms=end_to_end_ms,
        )


def _demo_signal(context_length: int, dtype: np.dtype[Any]) -> np.ndarray:
    index = np.arange(context_length, dtype=np.float32)
    signal = 0.025 * index + 1.4 * np.sin(2 * np.pi * index / 24)
    signal += 0.35 * np.cos(2 * np.pi * index / 7)
    return signal.astype(dtype)


def _build_package(args: argparse.Namespace) -> Path:
    if args.model_dir is not None:
        root = Path(args.model_dir)
        for component in COMPONENTS:
            _model_path(root, component)
        print(f"Loading saved ModelPackage from {root}")
        return root

    from mobius import build

    root = Path(args.output_dir)
    print(f"Building {args.model_id!r} at revision {args.revision} ...")
    package = build(
        args.model_id,
        revision=args.revision,
        dtype=args.dtype,
        execution_provider=args.device,
        load_weights=True,
    )
    if set(package) != set(COMPONENTS):
        raise RuntimeError(
            f"Expected TimesFM components {list(COMPONENTS)}, got {list(package)}."
        )
    package.save(str(root), external_data="onnx")
    print(f"Saved ModelPackage to {root}")
    return root


def _benchmark(
    driver: TimesFM3OrtDriver,
    signal: np.ndarray,
    horizon: int,
    warmups: int,
    iterations: int,
) -> ForecastResult:
    for _ in range(warmups):
        driver.forecast(signal, horizon)

    end_to_end: list[float] = []
    by_component = {name: [] for name in COMPONENTS}
    result: ForecastResult | None = None
    for _ in range(iterations):
        result = driver.forecast(signal, horizon)
        end_to_end.append(result.end_to_end_ms)
        for name, latency in result.component_ms.items():
            by_component[name].append(latency)

    assert result is not None
    core_samples = driver.benchmark_model(
        result.model_inputs,
        result.patch_mask,
        warmups=warmups,
        iterations=iterations,
    )
    print("\nONNX Runtime steady-state latency")
    print(f"  end-to-end:      {_latency_summary(end_to_end)}")
    for name in COMPONENTS:
        label = "model + copies" if name == "model" and driver.device == "cuda" else name
        print(f"  {label + ':':18}{_latency_summary(by_component[name])}")
    print(f"  {'model core-only:':18}{_latency_summary(core_samples)}")
    return result


def _compare_pytorch(
    result: ForecastResult,
    signal: np.ndarray,
    args: argparse.Namespace,
) -> None:
    try:
        import timesfm3
        import torch
    except ImportError as error:
        raise RuntimeError(
            "--compare-pytorch requires the official TimesFM package. "
            'Install it with: pip install "timesfm[torch]"'
        ) from error

    # The pinned upstream implementation's RoPE path promotes Q/K to float32 while V
    # remains float16, which PyTorch SDPA rejects. Use its supported float32 path when
    # comparing an fp16 ONNX package and report the difference explicitly.
    torch_dtype = torch.float32
    if result.model_inputs.dtype == np.float16:
        print("PyTorch comparison uses float32 (upstream fp16 RoPE/SDPA is unsupported).")
    print("\nLoading official PyTorch checkpoint for comparison ...")
    forecaster = timesfm3.TimesFM3Forecaster.from_pretrained(
        args.model_id,
        device=args.device,
        revision=args.revision,
        per_core_batch_size=1,
    )
    model = forecaster.model.to(device=args.device, dtype=torch_dtype).eval()

    model_inputs = torch.from_numpy(result.model_inputs).to(
        device=args.device, dtype=torch_dtype
    )
    patch_mask = torch.from_numpy(result.patch_mask).to(device=args.device)

    def synchronize() -> None:
        if args.device == "cuda":
            torch.cuda.synchronize()

    @torch.inference_mode()
    def run_core() -> Any:
        hidden = model.pre_transformer_resblock(model_inputs)
        hidden, _, _ = model.transformer_stack(hidden, patch_mask)
        return model.output_head(hidden)

    for _ in range(args.warmups):
        core_output = run_core()
    synchronize()
    core_samples: list[float] = []
    for _ in range(args.iterations):
        synchronize()
        start = time.perf_counter()
        core_output = run_core()
        synchronize()
        core_samples.append((time.perf_counter() - start) * 1_000)
    core_numpy = core_output.float().cpu().numpy()
    core_error = float(np.max(np.abs(result.raw_logits.astype(np.float32) - core_numpy)))
    print(f"PyTorch learned-core max absolute error: {core_error:.6g}")
    print(f"PyTorch learned-core latency:           {_latency_summary(core_samples)}")

    target = torch.from_numpy(signal[None, None, :]).to(device=args.device, dtype=torch_dtype)

    @torch.inference_mode()
    def run_forecast() -> Any:
        return model.decode(target=target, horizon=args.horizon)

    try:
        for _ in range(args.warmups):
            torch_forecast = run_forecast()
        synchronize()
        forecast_samples: list[float] = []
        for _ in range(args.iterations):
            synchronize()
            start = time.perf_counter()
            torch_forecast = run_forecast()
            synchronize()
            forecast_samples.append((time.perf_counter() - start) * 1_000)
        torch_quantiles = np.sort(torch_forecast.float().cpu().numpy(), axis=-1)
        onnx_quantiles = result.quantiles[:, :, : args.horizon].astype(np.float32)
        forecast_error = float(np.max(np.abs(onnx_quantiles - torch_quantiles)))
        print(f"PyTorch user-forecast max absolute error: {forecast_error:.6g}")
        print(
            f"PyTorch device-resident forecast latency: {_latency_summary(forecast_samples)}"
        )
    except (RuntimeError, TypeError) as error:
        print(f"PyTorch user-forecast comparison unavailable: {error}")


def _print_forecast(result: ForecastResult, horizon: int) -> None:
    point = result.point[0, 0, :horizon]
    quantiles = result.quantiles[0, 0, :horizon]
    middle = quantiles.shape[-1] // 2
    print("\nForecast (first target)")
    labels = (
        ("q10", "q50", "q90")
        if quantiles.shape[-1] == 9
        else (
            "first",
            "middle",
            "last",
        )
    )
    print(f" step        point {labels[0]:>12} {labels[1]:>12} {labels[2]:>12}")
    for step in range(horizon):
        print(
            f"{step + 1:5d} {point[step]:12.5f} {quantiles[step, 0]:12.5f} "
            f"{quantiles[step, middle]:12.5f} {quantiles[step, -1]:12.5f}"
        )


def _write_csv(path: str | os.PathLike[str], result: ForecastResult, horizon: int) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = result.quantiles.shape[-1]
    levels = QUANTILES if count == len(QUANTILES) else tuple(np.linspace(0.1, 0.9, count))
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "point", *(f"q{quantile:g}" for quantile in levels)])
        for step in range(horizon):
            writer.writerow(
                [
                    step + 1,
                    float(result.point[0, 0, step]),
                    *(float(value) for value in result.quantiles[0, 0, step]),
                ]
            )
    print(f"Wrote forecast CSV to {output}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build or load the five-component TimesFM 3 ONNX package and forecast a "
            "deterministic seasonal signal. Official weights use the non-commercial "
            "TimesFM license."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-id", default=MODEL_ID, help="Hugging Face model ID.")
    parser.add_argument(
        "--revision", default=REVISION, help="Pinned Hugging Face checkpoint revision."
    )
    parser.add_argument(
        "--model-dir",
        help="Previously saved ModelPackage directory; skips build when supplied.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/timesfm3",
        help="Directory for a newly built ModelPackage.",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["f32", "f16"], default="f32")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Capture the fixed-shape CUDA model core (CUDA only).",
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Report steady-state ORT latency."
    )
    parser.add_argument(
        "--compare-pytorch",
        action="store_true",
        help="Compare the same pinned weights, core tensors, and forecast with PyTorch.",
    )
    parser.add_argument("--csv-output", help="Optional path for forecast CSV output.")
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.context_length <= 0:
        parser.error("--context-length must be positive.")
    if args.horizon <= 0:
        parser.error("--horizon must be positive.")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative.")
    if args.iterations <= 0:
        parser.error("--iterations must be positive.")
    if args.cuda_graph and args.device != "cuda":
        parser.error("--cuda-graph requires --device cuda.")

    model_dir = _build_package(args)
    driver = TimesFM3OrtDriver(
        model_dir,
        device=args.device,
        enable_cuda_graph=args.cuda_graph,
    )
    signal = _demo_signal(args.context_length, driver.input_dtype)

    if args.benchmark:
        result = _benchmark(driver, signal, args.horizon, args.warmups, args.iterations)
    else:
        result = driver.forecast(signal, args.horizon)
        print(f"End-to-end latency: {result.end_to_end_ms:.3f} ms")

    _print_forecast(result, args.horizon)
    if args.csv_output:
        _write_csv(args.csv_output, result, args.horizon)
    if args.compare_pytorch:
        _compare_pytorch(result, signal, args)


if __name__ == "__main__":
    main()
