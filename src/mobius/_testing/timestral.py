# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Independent full-weight Timestral goldens and native-ONNX replay."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from mobius._testing.golden import GoldenTestCase
from mobius.models.timestral import TIMESTRAL_REFERENCE_REVISION


def generate_timestral_golden(case: GoldenTestCase, json_path: Path, device: str) -> None:
    """Collect complete quantile tensors from the pinned upstream implementation."""
    import torch
    from timestral import TimestralForecaster

    params = case.generation_params
    rng = np.random.default_rng(params["seed"])
    context = rng.normal(size=(2, 2, 64)).astype(np.float32).cumsum(axis=-1) * 0.1
    context[1] += 3
    observed = np.ones(context.shape, dtype=np.bool_)
    observed[0, 0, :32] = False
    observed[0, 1, 15:22] = False
    observed[1, 1, 32:40] = False
    reference = TimestralForecaster.from_pretrained(
        case.model_id,
        revision=case.revision,
        device=device,
        per_shot_horizon_patches=params["per_shot_horizon_patches"],
    )
    trace = []
    original_predict = reference.model.predict_block

    def record(*args, **kwargs):
        result = original_predict(*args, **kwargs)
        trace.append(result.detach().cpu().numpy().tolist())
        return result

    values = np.where(observed, context, 0).reshape(4, -1)
    masks = observed.reshape(4, -1)
    groups = np.repeat(np.arange(2, dtype=np.int64), 2)
    with torch.inference_mode():
        prefill = (
            original_predict(
                torch.from_numpy(values).to(device),
                torch.from_numpy(masks).to(device),
                torch.from_numpy(groups).to(device),
                params["per_shot_horizon_patches"],
            )
            .cpu()
            .numpy()
        )
        reference.model.predict_block = record
        generated = (
            reference.forecast(
                context,
                params["prediction_length"],
                observation_mask=observed,
            )
            .quantiles.cpu()
            .numpy()
        )
    provenance = {
        "model_id": case.model_id,
        "revision": case.revision,
        "reference_source": f"https://github.com/mistralai/timestral/tree/{TIMESTRAL_REFERENCE_REVISION}",
        "reference_revision": TIMESTRAL_REFERENCE_REVISION,
        "torch_version": torch.__version__,
        "device": device,
    }
    base = {
        "provenance": provenance,
        "context": context.tolist(),
        "observed_mask": observed.tolist(),
        "quantile_knots": list(reference.config.quantile_knots),
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({**base, "quantiles": prefill.tolist()}, indent=1) + "\n")
    generation_path = json_path.with_name(json_path.stem + "_generation.json")
    generation_path.write_text(
        json.dumps(
            {
                **base,
                "quantiles": generated.tolist(),
                "blocks": trace,
                "prediction_length": params["prediction_length"],
                "per_shot_horizon_patches": params["per_shot_horizon_patches"],
            },
            indent=1,
        )
        + "\n"
    )


def assert_timestral_golden(
    case: GoldenTestCase, json_path: Path, *, generation: bool
) -> None:
    """Verify every quantile and rollout block, including actual CUDA kernel use."""
    import onnxruntime as ort

    from mobius import ModelPackage, build
    from mobius.integrations.timestral import forecast_timestral
    from mobius.models.timestral import TimestralConfig

    data = json.loads(json_path.read_text())
    assert data["provenance"]["model_id"] == case.model_id
    assert data["provenance"]["revision"] == case.revision
    assert data["provenance"]["reference_revision"] == TIMESTRAL_REFERENCE_REVISION
    device = os.environ.get("MOBIUS_TEST_DEVICE", "cpu")
    cuda = device.lower().startswith("cuda")
    if cuda and "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(
            "Timestral GPU validation requires CUDAExecutionProvider; install a compatible onnxruntime-gpu wheel on the existing A10 runner."
        )
    package = build(case.model_id, revision=case.revision, dtype="f32")
    config = package.config
    assert isinstance(config, TimestralConfig)
    assert list(config.quantile_knots) == data["quantile_knots"]
    context = np.asarray(data["context"], dtype=np.float32)
    observed = np.asarray(data["observed_mask"], dtype=np.bool_)
    blocks = []
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.onnx"
        package.save(directory, progress_bar=False)
        restored = ModelPackage.load(directory)
        restored_config = TimestralConfig.from_onnx(restored["model"])
        assert all(
            getattr(restored_config, name) == getattr(config, name)
            for name in TimestralConfig.__annotations__
        )
        config = restored_config
        options = ort.SessionOptions()
        options.enable_profiling = cuda
        options.profile_file_prefix = str(Path(directory) / "profile")
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if cuda
            else ["CPUExecutionProvider"]
        )
        session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        if cuda:
            assert session.get_providers()[0] == "CUDAExecutionProvider"

        class Adapter:
            def run(self, feeds):
                result = session.run(["quantiles"], feeds)[0]
                blocks.append(result)
                return {"quantiles": result}

        if generation:
            actual = forecast_timestral(
                Adapter(),
                config,
                context,
                data["prediction_length"],
                observation_mask=observed,
                per_shot_horizon_patches=data["per_shot_horizon_patches"],
            )
            assert actual.shape[2] == data["prediction_length"]
            assert len(blocks) == len(data["blocks"])
            for block, expected in zip(blocks, data["blocks"]):
                np.testing.assert_allclose(
                    block,
                    np.asarray(expected, dtype=np.float32),
                    rtol=1e-3,
                    atol=1e-3,
                    strict=True,
                )
        else:
            actual = Adapter().run(
                {
                    "context": np.where(observed, context, 0).reshape(4, -1),
                    "observed_mask": observed.reshape(4, -1),
                    "group_ids": np.repeat(np.arange(2, dtype=np.int64), 2),
                    "horizon_patches": np.array(
                        [case.generation_params["per_shot_horizon_patches"]], dtype=np.int64
                    ),
                }
            )["quantiles"]
        expected = np.asarray(data["quantiles"], dtype=np.float32)
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3, strict=True)
        assert np.all(np.isfinite(actual))
        assert np.all(np.diff(actual, axis=-1) >= 0)
        if cuda:
            events = json.loads(Path(session.end_profiling()).read_text())
            kernels = [
                event["args"]
                for event in events
                if event.get("args", {}).get("provider") == "CUDAExecutionProvider"
            ]
            assert any(
                kernel.get("op_name") in {"MatMul", "FusedMatMul", "Attention", "Gemm"}
                for kernel in kernels
            ), (
                "No Timestral neural compute ran on CUDA; inspect ORT profiling and provider compatibility."
            )
            print(f"Timestral CUDA evidence: {len(kernels)} GPU kernel events")
