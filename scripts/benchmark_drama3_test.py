# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Transport, contention guards and real-graph benchmark-driver checks."""

from __future__ import annotations

import copy
import dataclasses
import json

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch
from _test_configs import FISH_OMNI_CONFIG
from benchmark_drama3 import (
    BoundSession,
    benchmark,
    codebook_view,
    parser,
    require_idle,
    sequence_view,
)
from onnxscript import GraphBuilder
from safetensors.torch import save_file

from mobius import build_from_module
from mobius._configs.fish_codec import FishCodecConfig
from mobius._configs.fish_omni import FishOmniConfig
from mobius._testing.ort_inference import _numpy_to_ort_value
from mobius.integrations.transformers._config_resolver import _dict_to_pretrained_config
from mobius.integrations.transformers._fish_omni_weights import (
    _source_name,
    stream_fish_omni_safetensors_to_package,
)
from mobius.models import FishOmniModel
from mobius.models.fish_omni_test import reduced_precision_prototype_task
from mobius.tasks.fish_codec import build_fish_codec_decoder


def test_busy_gpu_cannot_be_a_performance_pass():
    busy = {"memory_used_mib": 130000, "utilization_percent": 100}
    with pytest.raises(RuntimeError, match="refusing misleading"):
        require_idle(busy, False)
    assert require_idle(busy, True) is False
    with pytest.raises(RuntimeError, match="refusing misleading"):
        require_idle(
            {"memory_used_mib": 10, "utilization_percent": 0, "foreign_pids": [123]}, False
        )


def test_scalar_boolean_keeps_its_rank():
    value = _numpy_to_ort_value(np.array(True))
    assert value.shape() == []


def test_zero_copy_contiguous_views_bind_the_correct_bytes(tmp_path):
    graph = ir.Graph([], [], nodes=[], opset_imports={"": 24})
    builder = GraphBuilder(graph)
    value = builder.input("x", ir.DataType.FLOAT, [1, 4])
    output = builder.op.Identity(value)
    output.type, output.shape = value.type, value.shape
    builder.add_output(output, "y")
    path = tmp_path / "view.onnx"
    ir.save(ir.Model(graph, ir_version=12), path)
    session = BoundSession(path, device="cpu")
    values = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    owner = ort.OrtValue.ortvalue_from_numpy(values)
    result = session.run({"x": sequence_view(owner, 2, 1, squeeze_time=True)})
    np.testing.assert_array_equal(result["y"].numpy(), values[:, 2])
    books = ort.OrtValue.ortvalue_from_numpy(values.reshape(1, 1, 3, 4))
    result = session.run({"x": codebook_view(books, 1)})
    np.testing.assert_array_equal(result["y"].numpy(), values[:, 1])
    with pytest.raises(ValueError, match="Invalid"):
        sequence_view(owner, 2, 2)


@pytest.mark.parametrize("task", ["fish-omni", "fish-omni-generation"])
@pytest.mark.parametrize("sample", [False, True])
@pytest.mark.parametrize(
    "dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16]
)
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available()
                or "CUDAExecutionProvider" not in ort.get_available_providers(),
                reason="CUDA binding smoke test requires a GPU",
            ),
        ),
    ],
)
def test_benchmark_runs_actual_cached_graphs_without_claiming_full_model_perf(
    tmp_path, task, sample, device, dtype
):
    if dtype != ir.DataType.FLOAT and device == "cpu":
        pytest.skip("Reduced-precision benchmark routes require CUDA")
    config = FishOmniConfig.from_transformers(
        _dict_to_pretrained_config(copy.deepcopy(FISH_OMNI_CONFIG))
    )
    config = dataclasses.replace(config, dtype=dtype)
    package = build_from_module(
        FishOmniModel(config),
        config,
        task=reduced_precision_prototype_task(task) if dtype != ir.DataType.FLOAT else task,
        execution_provider={
            ir.DataType.FLOAT: "default",
            ir.DataType.FLOAT16: "cuda",
            ir.DataType.BFLOAT16: "onnx-standard",
        }[dtype],
    )
    torch.manual_seed(42)
    weights = {}
    for model in package.values():
        for name, value in model.graph.initializers.items():
            if value.const_value is not None:
                continue
            tensor = torch.randn(tuple(value.shape)) * 0.1
            source = _source_name(name)
            if ".packed_experts." in name:
                prefix, projection = source.rsplit(".experts.", 1)
                for expert, matrix in enumerate(tensor):
                    if projection == "gate_up_proj.weight":
                        gate, up = matrix.chunk(2, dim=0)
                        weights[f"{prefix}.experts.{expert}.gate_proj.weight"] = (
                            gate.contiguous()
                        )
                        weights[f"{prefix}.experts.{expert}.up_proj.weight"] = up.contiguous()
                    else:
                        weights[f"{prefix}.experts.{expert}.{projection}"] = (
                            matrix.contiguous()
                        )
            else:
                weights[source] = tensor
    # Keep the request alive: for either sign of hidden[0], a semantic row wins.
    weights["lm_head.weight"].zero_()
    weights["lm_head.weight"][config.semantic_start_token_id, 0] = 100
    weights["lm_head.weight"][config.semantic_start_token_id + 1, 0] = -100
    save_file(weights, tmp_path / "model.safetensors")
    stream_fish_omni_safetensors_to_package(package, str(tmp_path), config, revision=None)
    model_dir = tmp_path / "onnx"
    package.save(str(model_dir), max_workers=1, progress_bar=False)
    source_config = tmp_path / "config.json"
    source_config.write_text(json.dumps(FISH_OMNI_CONFIG))
    prompt = tmp_path / "prompt.npz"
    np.savez(prompt, input_frames=np.array([[1, 0, 0, 0], [3, 0, 0, 0]], np.int64))
    with_codec = task == "fish-omni-generation" and device == "cpu"
    codec_path = tmp_path / "codec"
    if with_codec:
        codec = build_fish_codec_decoder(
            FishCodecConfig(
                latent_dim=16,
                residual_codebooks=2,
                transformer_layers=1,
                transformer_heads=2,
                transformer_intermediate_size=32,
                max_frames=16,
                decoder_dim=32,
            )
        )
        rng = np.random.default_rng(42)
        for value in codec["waveform_decoder"].graph.initializers.values():
            if value.const_value is None:
                value.const_value = ir.tensor(
                    rng.normal(0, 0.02, tuple(value.shape)).astype(np.float32)
                )
        codec.save(str(codec_path), max_workers=1, progress_bar=False)
    args = parser().parse_args(
        [
            "--package",
            str(model_dir),
            "--config",
            str(source_config),
            "--prompt",
            str(prompt),
            "--output",
            str(tmp_path / "result.json"),
            "--frames",
            "4",
            "--warmup",
            "1",
            "--device",
            device,
            "--top-k",
            "1",
            *(["--sample"] if sample else []),
            *(["--allow-contended", "--profile"] if device == "cuda" else []),
            *(["--codec", str(codec_path)] if with_codec else []),
        ]
    )
    result = benchmark(args)
    assert result["generated_frames"] == 4
    assert result["single_pass_tower"] == (task == "fish-omni-generation")
    assert result["measurement_qualified"] is False
    assert result["meets_end_to_end_rtf_target"] is None
    assert result["gpu_resident_caches"] == (device == "cuda")
    if with_codec:
        assert result["end_to_end_rtf"] > 0
        assert result["waveform_seconds"] > 0
        assert result["time_to_waveform_seconds"] > result["first_frame_seconds"]
    else:
        assert result["end_to_end_rtf"] is None
    assert result["steady_frames_per_second"] > 0
    assert result["p95_frame_ms"] > 0
