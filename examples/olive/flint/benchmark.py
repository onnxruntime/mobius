# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Measure warmed, single-GPU Flint GenAI TTFT, decode speed and device memory."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import multiprocessing
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


class _Memory(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in ("total", "free", "used")]


def _sample_memory(gpu_index, peak, error, stop, ready):
    library = ctypes.CDLL("libnvidia-ml.so.1")
    handle = ctypes.c_void_p()
    error.value = library.nvmlInit_v2() or library.nvmlDeviceGetHandleByIndex_v2(
        gpu_index, ctypes.byref(handle)
    )
    ready.set()
    if error.value:
        return
    try:
        while not stop.is_set():
            memory = _Memory()
            error.value = library.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory))
            if error.value:
                return
            with peak.get_lock():
                peak.value = max(peak.value, memory.used)
            stop.wait(0.02)
    finally:
        library.nvmlShutdown()


class _MemorySampler:
    def __init__(self, gpu_index: int):
        # Native GenAI calls can hold the GIL; sample in an independent process.
        context = multiprocessing.get_context("spawn")
        self.value = context.Value("Q", 0)
        self.error = context.Value("i", 0)
        self.stop_event = context.Event()
        ready = context.Event()
        self.process = context.Process(
            target=_sample_memory,
            args=(gpu_index, self.value, self.error, self.stop_event, ready),
            daemon=True,
        )
        self.process.start()
        if not ready.wait(10) or self.error.value:
            self.stop()
            raise RuntimeError("NVML sampling did not initialize.")

    @property
    def peak(self):
        return self.value.value

    def reset(self):
        with self.value.get_lock():
            self.value.value = 0

    def stop(self):
        self.stop_event.set()
        self.process.join(5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()
        if self.error.value:
            raise RuntimeError("NVML sampling failed.")


def _prompt(question: str) -> str:
    return f"<|system|>Answer briefly without reasoning.<|user|>{question}<|assistant|>"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.tokens < 50 or args.repetitions < 3:
        raise ValueError("Use at least 50 selected tokens and three warmed repetitions.")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    import onnxruntime as ort
    import onnxruntime_genai as og

    hardware = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={args.gpu_index}",
            "--query-gpu=uuid,name,memory.total,memory.used,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    if int(hardware.split(",")[3]) > 1024:
        raise RuntimeError("Selected GPU is occupied; refusing a full-model load.")
    model_dir = args.model.resolve()
    cuda = ctypes.CDLL("libcudart.so.12")

    def synchronize():
        if cuda.cudaDeviceSynchronize():
            raise RuntimeError("CUDA synchronization failed.")

    memory = _MemorySampler(args.gpu_index)
    try:
        start = time.perf_counter()
        model = og.Model(str(model_dir))
        synchronize()
        load_seconds = time.perf_counter() - start
        tokenizer = og.Tokenizer(model)
        metadata = json.loads((model_dir / "genai_config.json").read_text())["model"]

        def generate(prompt: str, count: int, force_length: bool):
            ids = tokenizer.encode(prompt).tolist()
            params = og.GeneratorParams(model)
            params.set_search_options(
                max_length=len(ids) + count,
                min_length=len(ids) + count if force_length else 0,
                do_sample=False,
                num_beams=1,
                past_present_share_buffer=False,
            )
            generator = og.Generator(model, params)
            synchronize()
            begin = time.perf_counter()
            generator.append_tokens(np.asarray([ids], dtype=np.int32))
            selected = []
            first = None
            while not generator.is_done() and len(selected) < count:
                generator.generate_next_token()
                selected.append(int(generator.get_next_tokens()[0]))
                if first is None:
                    synchronize()
                    first = time.perf_counter()
            synchronize()
            end = time.perf_counter()
            if not selected or (force_length and len(selected) != count):
                raise RuntimeError(f"Expected {count} selected tokens, got {len(selected)}.")
            return {
                "prompt_tokens": len(ids),
                "selected_ids": selected,
                "selected_tokens": len(selected),
                "ttft_seconds": first - begin,
                "decode_seconds": end - first,
                "decode_tokens_per_second": (len(selected) - 1) / (end - first),
                "text": tokenizer.decode(selected),
            }

        quality = []
        for question, expected in (
            ("What is 2 + 2?", "4"),
            ("What is 6 * 7?", "42"),
            ("What is 12 - 5?", "7"),
        ):
            run = generate(_prompt(question), 128, False)
            selected = run["selected_ids"]
            eor = metadata["eor_token_id"]
            answer_ids = (
                selected[selected.index(eor) + 1 :] if eor in selected else selected[:]
            )
            eos = set(metadata["eos_token_id"])
            if answer_ids and answer_ids[-1] in eos:
                answer_ids = answer_ids[:-1]
            answer = tokenizer.decode(answer_ids).strip()
            quality.append(
                {
                    "question": question,
                    "expected": expected,
                    "answer": answer,
                    "answer_correct_exact": answer == expected and selected[-1] in eos,
                    "generated": run,
                }
            )
        prompts = {
            "short": _prompt("What is 2 + 2?"),
            "long": _prompt(
                "Context: "
                + "The robot observes the room and plans a safe route. " * 16
                + "Question: What is 6 * 7?"
            ),
        }
        measurements = {}
        cold_start_peak_bytes = memory.peak
        memory.reset()
        for name, prompt in prompts.items():
            for _ in range(2):
                generate(prompt, args.tokens, True)
            runs = [generate(prompt, args.tokens, True) for _ in range(args.repetitions)]
            measurements[name] = {
                "prompt": prompt,
                "prompt_tokens": runs[0]["prompt_tokens"],
                "warmup_runs": 2,
                "median_ttft_seconds": statistics.median(r["ttft_seconds"] for r in runs),
                "median_decode_tokens_per_second": statistics.median(
                    r["decode_tokens_per_second"] for r in runs
                ),
                "runs": runs,
            }
        maps = Path("/proc/self/maps").read_text()
        if "torch" in sys.modules or "libtorch" in maps or "libc10" in maps:
            raise RuntimeError("Deployment benchmarking must not load PyTorch/LibTorch.")
        receipt = {
            "hardware": hardware,
            "ort_version": ort.__version__,
            "ort_build_info": ort.get_build_info(),
            "genai_version": og.__version__,
            "genai_binding_sha256": hashlib.sha256(
                Path(sys.modules["onnxruntime_genai.onnxruntime_genai"].__file__).read_bytes()
            ).hexdigest(),
            "custom_library_sha256": hashlib.sha256(
                (model_dir / "lib/libonnxruntime_source_compat_cuda.so").read_bytes()
            ).hexdigest(),
            "header_sha256": hashlib.sha256(
                (model_dir / "model.onnx").read_bytes()
            ).hexdigest(),
            "package_file_bytes": sum(
                p.stat().st_size for p in model_dir.rglob("*") if p.is_file()
            ),
            "load_seconds": load_seconds,
            "sampled_cold_start_peak_device_bytes": cold_start_peak_bytes,
            "sampled_generation_peak_device_bytes": memory.peak,
            "memory_sample_interval_seconds": 0.02,
            "memory_sampling": "Independent NVML process, not blocked by the native GIL.",
            "torch_free": True,
            "measurements": measurements,
            "quality_spot_checks": quality,
            "timing_scope": "Synchronized append-to-first-selection TTFT; decode excludes first token. "
            "Tokenization and generator construction excluded; EOS suppressed for performance runs.",
            "qualification": "Experimental performance and arithmetic spot checks, not publisher parity "
            "or a dataset-level quantization quality evaluation.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({k: v for k, v in receipt.items() if k != "measurements"}, indent=2))
        for name, result in measurements.items():
            print(
                name,
                result["prompt_tokens"],
                result["median_ttft_seconds"],
                result["median_decode_tokens_per_second"],
                flush=True,
            )
    finally:
        memory.stop()


if __name__ == "__main__":
    main()
