# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Check full-checkpoint, Torch-free CUDA ORT generation with own BF16 KV caches."""

from __future__ import annotations

import argparse
import collections
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import ml_dtypes
import numpy as np
from onnx_ir import DataType
from tokenizers import Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive.")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    import onnxruntime as ort

    used = int(
        subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={args.gpu_index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    )
    if used > 1024:
        raise RuntimeError("Selected GPU is occupied; refusing a full-model load.")
    model_dir = args.model.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = json.loads((model_dir / "genai_config.json").read_text())["model"]
    decoder = config["decoder"]
    layers, heads, width = (
        decoder["num_hidden_layers"],
        decoder["num_key_value_heads"],
        decoder["head_size"],
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.enable_profiling = True
    options.profile_file_prefix = str(args.output.with_suffix(""))
    options.register_custom_ops_library(
        str(model_dir / "lib/libonnxruntime_source_compat_cuda.so")
    )
    started = time.perf_counter()
    session = ort.InferenceSession(
        str(model_dir / "model.onnx"),
        options,
        providers=[
            (
                "CUDAExecutionProvider",
                {"use_tf32": "0", "arena_extend_strategy": "kSameAsRequested"},
            ),
            "CPUExecutionProvider",
        ],
    )
    print("CUDA model ready in", time.perf_counter() - started, "seconds", flush=True)
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    prompt = "<|system|>Answer briefly without reasoning.<|user|>What is 2 + 2?<|assistant|>"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    cache = {
        f"past_key_values.{layer}.{kind}": np.empty((1, heads, 0, width), dtype=np.uint16)
        for layer in range(layers)
        for kind in ("key", "value")
    }
    generated, past = [], 0
    for step in range(args.max_new_tokens):
        current = prompt_ids if step == 0 else [generated[-1]]
        total = past + len(current)
        feeds = {
            "input_ids": np.asarray([current], dtype=np.int64),
            "position_ids": np.arange(past, total, dtype=np.int64)[None],
            "attention_mask": np.ones((1, total), dtype=np.int64),
            **cache,
        }
        binding = session.io_binding()
        values = {
            name: (
                ort.OrtValue.ortvalue_from_numpy_with_onnx_type(value, int(DataType.BFLOAT16))
                if value.dtype == np.uint16
                else ort.OrtValue.ortvalue_from_numpy(value)
            )
            for name, value in feeds.items()
        }
        for name, value in values.items():
            binding.bind_ortvalue_input(name, value)
        for output in session.get_outputs():
            binding.bind_output(output.name, "cpu")
        session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        actual = {}
        for output, value in zip(session.get_outputs(), binding.get_outputs(), strict=True):
            if value.data_type() == "tensor(float)":
                actual[output.name] = value.numpy().copy()
            else:
                if value.data_type() != "tensor(bfloat16)":
                    raise TypeError(value.data_type())
                raw = (ctypes.c_uint16 * int(np.prod(value.shape()))).from_address(
                    value.data_ptr()
                )
                array = np.ctypeslib.as_array(raw).copy().reshape(value.shape())
                if array.shape != (1, heads, total, width):
                    raise ValueError(f"Incorrect cache shape: {array.shape}")
                if not np.isfinite(array.view(ml_dtypes.bfloat16).astype(np.float32)).all():
                    raise ValueError("Nonfinite KV cache.")
                actual[output.name] = array
        logits = actual["logits"]
        if (
            logits.shape != (1, len(current), config["vocab_size"])
            or not np.isfinite(logits).all()
        ):
            raise ValueError("Invalid full logits.")
        generated.append(int(logits[0, -1].argmax()))
        cache = {
            f"past_key_values.{layer}.{kind}": actual[f"present.{layer}.{kind}"]
            for layer in range(layers)
            for kind in ("key", "value")
        }
        past = total
        if generated[-1] in config["eos_token_id"]:
            break
    profile = session.end_profiling()
    events = json.loads(Path(profile).read_text())
    kernels = [e["args"] for e in events if e.get("args", {}).get("provider")]
    critical = {"MoE", "QMoE", "SinkAttention", "RMSNorm", "SkipRMSNorm", "Gemm", "MatMul"}
    if any(
        e.get("op_name") in critical and e["provider"] != "CUDAExecutionProvider"
        for e in kernels
    ):
        raise RuntimeError("Critical arithmetic fell off CUDA.")
    histogram = collections.Counter(e.get("op_name") for e in kernels)
    maps = Path("/proc/self/maps").read_text()
    if "torch" in sys.modules or "libtorch" in maps or "libc10" in maps:
        raise RuntimeError("Deployment unexpectedly loaded PyTorch/LibTorch.")
    receipt = {
        "execution_status": "pass",
        "scope": "Functional host-IO own-cache generation, not a throughput benchmark.",
        "ort_version": ort.__version__,
        "ort_build_info": ort.get_build_info(),
        "header_sha256": hashlib.sha256((model_dir / "model.onnx").read_bytes()).hexdigest(),
        "prompt": prompt,
        "prompt_ids": prompt_ids,
        "selected_ids": generated,
        "selected_token_count": len(generated),
        "text": tokenizer.decode(generated, skip_special_tokens=False),
        "stopped_at_eos": generated[-1] in config["eos_token_id"],
        "all_full_logits_and_kv_caches_finite": True,
        "critical_arithmetic_cuda": True,
        "operator_execution_counts": dict(histogram),
        "torch_free": True,
        "publisher_parity": "unqualified",
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
