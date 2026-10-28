# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Validate full Drama 3 token graphs against pinned Fish Speech inference.

Run download, export, reference, and compare as separate processes. The default
reference placement requires five CUDA devices; comparison uses three. Outputs
include full reference tensors, exact frame IDs, error bounds and an ORT CUDA
profile. The waveform codec is intentionally outside this validator.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import time
import types
import urllib.request

from mobius._configs.fish_omni import DRAMA3_REVISION, DRAMA3_SOURCE_REVISION

ROOT: pathlib.Path
REVISION = DRAMA3_REVISION
SOURCE_REVISION = DRAMA3_SOURCE_REVISION
PROMPT = "Hello, this is a speech synthesis test."
REFERENCE_DEVICES: tuple[int, ...] = (0, 1, 2, 3)
REFERENCE_TOWER_DEVICE = 4
EMBEDDING_DEVICE = 0
AUDIO_DEVICE = 1
DECODER_DEVICE = 2
DTYPE = "f32"
EXPERT_EXECUTION = "dense"
DTYPE_NAMES = {"f32": "float32", "f16": "float16", "bf16": "bfloat16"}
GOLDEN = (
    pathlib.Path(__file__).resolve().parents[1]
    / "testdata"
    / "golden"
    / "speech"
    / "drama-3_tokens_generation.json"
)


def event(kind, **details):
    print(json.dumps({"event": kind, **details}), flush=True)


def empty_cache_ports(ports):
    import ml_dtypes
    import numpy as np

    caches = {}
    for port in ports:
        if not port.name.startswith("past_key_values."):
            continue
        shape = list(port.shape)
        shape[0] = 1
        if port.name.endswith((".key", ".value")):
            assert len(shape) == 4
            shape[2] = 0
        assert all(isinstance(dim, int) for dim in shape), (port.name, shape)
        type_name = getattr(port, "type", "tensor(float)")
        dtypes = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(bfloat16)": ml_dtypes.bfloat16,
        }
        if type_name not in dtypes:
            raise ValueError(f"Unsupported cache dtype {type_name}: {port.name}")
        caches[port.name] = np.zeros(shape, dtype=dtypes[type_name])
    return caches


def download():
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        "fishaudio/drama-3",
        revision=REVISION,
        cache_dir=str(ROOT / "hf-cache"),
        allow_patterns=[
            "config.json",
            "model*.safetensors",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "chat_template.jinja",
            "LICENSE",
        ],
        max_workers=8,
    )
    (ROOT / "checkpoint-path.txt").write_text(path)
    event("checkpoint_ready", revision=REVISION)


def checkpoint_path():
    path = pathlib.Path((ROOT / "checkpoint-path.txt").read_text().strip()).resolve()
    if not path.is_dir() or path.name != REVISION:
        raise ValueError(f"Expected the immutable checkpoint snapshot {REVISION}, got {path}")
    return path


def load_upstream():
    prefix = "fish_speech.models.text2semantic"
    for name in ("fish_speech", "fish_speech.models", prefix, f"{prefix}.fish_omni"):
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    result = {}
    source_dir = ROOT / "upstream" / SOURCE_REVISION
    source_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "errors",
        "sampling",
        "fish_omni.fp8_moe",
        "fish_omni.config",
        "fish_omni.layers",
        "fish_omni.moe",
        "fish_omni.backbone",
        "fish_omni.tower",
        "fish_omni.model",
        "fish_omni.generate",
    ):
        path = source_dir / (name + ".py")
        if not path.exists():
            url = (
                "https://raw.githubusercontent.com/fishaudio/fish-speech/"
                f"{SOURCE_REVISION}/fish_speech/models/text2semantic/"
                f"{name.replace('.', '/')}.py"
            )
            with urllib.request.urlopen(url, timeout=60) as response:
                path.write_bytes(response.read())
        spec = importlib.util.spec_from_file_location(f"{prefix}.{name}", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load the pinned reference module {name!r}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        result[name.rsplit(".", 1)[-1]] = module
    return result


def export(checkpoint, *, task="fish-omni", execution_provider=None, native_fp8=False):
    from mobius import build

    start = time.monotonic()
    event("export_start")
    package = build(
        str(checkpoint),
        revision=REVISION,
        dtype=DTYPE,
        keep_quantized=native_fp8,
        task=task,
        execution_provider=execution_provider
        or (
            "cuda"
            if native_fp8
            else {"f32": "default", "f16": "cuda", "bf16": "onnx-standard"}[DTYPE]
        ),
    )
    event("graph_and_bindings_ready", seconds=time.monotonic() - start)
    package.save(str(ROOT / "onnx"), max_workers=1)
    event("export_complete", seconds=time.monotonic() - start)
    (ROOT / "export-result.json").write_text(
        json.dumps(
            {
                "revision": REVISION,
                "dtype": DTYPE,
                "expert_execution": "dynamic_fp8" if native_fp8 else "dense",
                "seconds": time.monotonic() - start,
                "weight_loading_report": package.weight_loading_report,
                "files": {
                    str(path.relative_to(ROOT / "onnx")): path.stat().st_size
                    for path in (ROOT / "onnx").rglob("*")
                    if path.is_file()
                },
            },
            indent=2,
        )
    )


def validate_reference_torch_version(version: str) -> None:
    if version.split("+", 1)[0] != "2.8.0":
        raise RuntimeError(
            "The pinned Fish Speech reference requires torch==2.8.0; "
            f"found {version}. Use a separate reference environment."
        )


def reference(checkpoint, *, native_fp8=False):
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    os.environ["FISH_DRAMA3_FP8_EXPERTS"] = "1" if native_fp8 else "0"
    os.environ["FISH_DRAMA3_CUDA_GRAPH"] = "0"
    import numpy as np
    import torch

    validate_reference_torch_version(torch.__version__)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    upstream = load_upstream()
    start = time.monotonic()
    event(
        "reference_load_start",
        devices=REFERENCE_DEVICES,
        tower_device=REFERENCE_TOWER_DEVICE,
    )
    model = upstream["model"].FishOmniModel.from_pretrained(
        checkpoint,
        devices=[f"cuda:{device}" for device in REFERENCE_DEVICES],
        tower_device=f"cuda:{REFERENCE_TOWER_DEVICE}",
        dtype={"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}[DTYPE],
    )
    event("reference_loaded", seconds=time.monotonic() - start)
    backbone, tower = model.backbone, model.audio_decoder
    if any(layer.mlp.experts.fp8 != native_fp8 for layer in backbone.model.layers):
        raise AssertionError("Pinned source did not activate the requested expert execution")
    prompt = upstream["generate"].build_drama3_prompt(PROMPT, model.tokenizer, model.config)
    states = []
    active_heads: list[np.ndarray] = []
    original_embed = tower.embed_backbone
    original_forward = backbone.forward
    original_prefill = tower.prefill
    original_decode = tower.decode_step

    def array(tensor):
        return tensor.detach().float().cpu().numpy()

    def embed(frames, text):
        output = original_embed(frames, text)
        unseen = tower.embedding_sum(
            frames.to(tower.device),
            1 + tower.config.vq_backbone_codebooks,
            tower.config.num_codebooks,
        )
        unseen = torch.where(
            tower.is_semantic(frames[:, 0].to(tower.device))[:, None], unseen, 0.0
        )
        states.append(
            {
                "input_frames": frames.detach().cpu().numpy(),
                "inputs_embeds": array(output)[None],
                "shared_input": array(unseen)[None],
            }
        )
        return output

    def forward(*args, **kwargs):
        hidden = original_forward(*args, **kwargs)
        states[-1]["hidden_states"] = array(hidden)
        states[-1]["logits"] = array(backbone.compute_logits(hidden))
        states[-1]["position_ids"] = kwargs["input_pos"].cpu().numpy()[None]
        return hidden

    def prefill(*args, **kwargs):
        active_heads.clear()
        output = original_prefill(*args, **kwargs)
        states[-1]["prefill_tower_logits"] = np.stack(active_heads, axis=-2)[None]
        return output

    def decode(*args, **kwargs):
        active_heads.clear()
        output = original_decode(*args, **kwargs)
        states[-1]["tower_logits"] = np.stack(active_heads, axis=-2)[None]
        states[-1]["sampled_frame"] = output.detach().cpu().numpy()
        return output

    def capture_head(head):
        def capture(_module, _inputs, output):
            active_heads.append(array(head(output)))

        return capture

    handles = [
        layer.register_forward_hook(capture_head(head))
        for layer, head in zip(tower.residual_layers, tower.codebook_heads)
    ]
    tower.embed_backbone = embed
    backbone.forward = forward
    tower.prefill = prefill
    tower.decode_step = decode
    upstream["tower"].sample_top_k_top_p = lambda logits, *a, **kw: logits.argmax(-1)
    with torch.inference_mode():
        generated, finish = upstream["generate"].generate_drama3(
            model,
            prompt,
            max_new_tokens=20,
            temperature=1.0,
            top_k=30,
            top_p=1.0,
        )
    for handle in handles:
        handle.remove()
    output_dir = ROOT / "reference"
    output_dir.mkdir(exist_ok=True)
    for index, state in enumerate(states):
        np.savez(output_dir / f"state-{index:02}.npz", **state)
    logits = states[0]["logits"]
    top = np.argsort(logits[0, -1])[-10:][::-1]
    golden = {
        "model_id": "fishaudio/drama-3",
        "revision": REVISION,
        "source_revision": SOURCE_REVISION,
        "dtype": DTYPE_NAMES[DTYPE],
        "expert_execution": "dynamic_fp8" if native_fp8 else "dense",
        "reference_environment": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "triton": getattr(upstream["fp8_moe"].triton, "__version__", None),
        },
        "prompt": PROMPT,
        "prompt_rows": len(prompt),
        "prefill_shape": list(logits.shape),
        "prefill_last_top10": top.tolist(),
        "prefill_last_top10_logits": logits[0, -1, top].tolist(),
        "prefill_logits_sha256": hashlib.sha256(logits.tobytes()).hexdigest(),
        "max_new_tokens": 20,
        "do_sample": False,
        "finish_reason": finish,
        "generated_frames": generated.tolist(),
        "attempted_frames": len(states),
        "seconds": time.monotonic() - start,
    }
    (output_dir / "golden.json").write_text(json.dumps(golden, indent=2))
    event(
        "reference_complete",
        **{
            k: golden[k]
            for k in ("prompt_rows", "finish_reason", "attempted_frames", "seconds")
        },
    )


def validate_reference_golden(golden):
    import numpy as np

    committed = json.loads(GOLDEN.read_text())
    assert golden.get("expert_execution", "dense") == EXPERT_EXECUTION, "expert_execution"
    assert committed.get("expert_execution", "dense") == EXPERT_EXECUTION, "expert_execution"
    if "reference_environment" in golden and "reference_environment" in committed:
        assert golden["reference_environment"] == committed["reference_environment"], (
            "reference_environment"
        )
    assert golden["dtype"] == DTYPE_NAMES[DTYPE], "dtype"
    for field in (
        "model_id",
        "revision",
        "source_revision",
        "dtype",
        "prompt",
        "max_new_tokens",
        "do_sample",
        "finish_reason",
    ):
        assert golden[field] == committed[field], field
    assert len(golden["generated_frames"]) == len(committed["generated_frames"])
    np.testing.assert_array_equal(golden["generated_frames"], committed["generated_frames"])
    assert golden["prefill_shape"] == committed["prefill_shape"]
    np.testing.assert_array_equal(
        golden["prefill_last_top10"], committed["prefill_last_top10"]
    )
    np.testing.assert_allclose(
        golden["prefill_last_top10_logits"],
        committed["prefill_last_top10_logits"],
        atol=1e-2 if DTYPE != "f32" else 1e-4,
        rtol=1e-2 if DTYPE != "f32" else 1e-4,
    )


def run_session(session, feeds):
    """Bridge NumPy/BF16 feeds and outputs without changing their tensor types."""
    import numpy as np

    from mobius._testing.ort_inference import _numpy_to_ort_value, _ort_value_to_numpy

    converted = {
        name: _numpy_to_ort_value(np.ascontiguousarray(value) if value.ndim else value)
        for name, value in feeds.items()
    }
    cuda_options = session.get_provider_options().get("CUDAExecutionProvider")
    device_context: contextlib.AbstractContextManager[object] = contextlib.nullcontext()
    if cuda_options is not None:
        import torch

        # Cross-component calls can leave another GPU current on this thread;
        # ORT's stream events must be created/recorded on the session's device.
        device_context = torch.cuda.device(int(cuda_options["device_id"]))
    with device_context:
        values = session(**converted)
    ports = session.get_outputs()
    assert len(ports) == len(values)
    return {port.name: _ort_value_to_numpy(value) for port, value in zip(ports, values)}


def run_tower_session(session, feeds, *, generate=False):
    """Run teacher-forced or single-pass greedy tower against the same reference."""
    import numpy as np

    names = {port.name for port in session.get_inputs()}
    if "generate" in names:
        feeds = {
            **feeds,
            "generate": np.array(generate),
            "do_sample": np.array(False),
            "temperature": np.array([0.7], np.float32),
            "top_k": np.array([30], np.int64),
            "top_p": np.array([0.95], np.float32),
            "seed": np.array([42], np.int64),
            "counter": np.zeros(1, np.int64),
        }
    result = run_session(session, feeds)
    if "generate" in names:
        if not result["valid"].all():
            raise FloatingPointError("Single-pass tower rejected reference inputs")
        if result["next_counter"].any():
            raise AssertionError("Greedy single-pass tower advanced RNG state")
    return result


def compare(*, audio_model_name=None):
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    import numpy as np
    import onnxruntime as ort
    import onnxruntime_easy as ort_easy

    golden = json.loads((ROOT / "reference" / "golden.json").read_text())
    validate_reference_golden(golden)
    config = json.loads((checkpoint_path() / "config.json").read_text())
    ort.preload_dlls()
    sessions = {}
    if audio_model_name is None:
        audio_model_name = (
            "audio_generator"
            if (ROOT / "onnx" / "audio_generator" / "model.onnx").is_file()
            else "audio_decoder"
        )
    start = time.monotonic()
    for name, device in (
        ("embedding", EMBEDDING_DEVICE),
        ("audio_decoder", AUDIO_DEVICE),
        ("decoder", DECODER_DEVICE),
    ):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 8
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.enable_mem_pattern = False
        options.add_session_config_entry("session.disable_prepacking", "1")
        if name == "decoder":
            options.enable_profiling = True
            options.profile_file_prefix = str(ROOT / "decoder-profile")
        event("ort_session_start", component=name, device=device)
        component = audio_model_name if name == "audio_decoder" else name
        sessions[name] = ort_easy.EasySession(
            str(ROOT / "onnx" / component / "model.onnx"),
            sess_options=options,
            providers=[
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": device,
                        "use_tf32": "0",
                        "arena_extend_strategy": "kSameAsRequested",
                    },
                ),
                "CPUExecutionProvider",
            ],
        )
        event("ort_session_ready", component=name, seconds=time.monotonic() - start)

    def run(name, feeds, *, generate=False):
        if name == "audio_decoder":
            return run_tower_session(sessions[name], feeds, generate=generate)
        return run_session(sessions[name], feeds)

    expected_type = {
        "f32": "tensor(float)",
        "f16": "tensor(float16)",
        "bf16": "tensor(bfloat16)",
    }[DTYPE]
    assert sessions["decoder"].get_outputs()[0].type == expected_type
    runtime_features = (
        sessions["decoder"]
        .get_modelmeta()
        .custom_metadata_map.get("mobius.required_runtime_features", "")
    )
    actual_experts = (
        "dynamic_fp8" if "activation_quantization=dynamic_fp8" in runtime_features else "dense"
    )
    assert actual_experts == EXPERT_EXECUTION, "Graph and reference expert execution differ"

    def empty_cache(name):
        return empty_cache_ports(sessions[name].get_inputs())

    def next_cache(output):
        return {
            key.replace("present.", "past_key_values.", 1): value
            for key, value in output.items()
            if key.startswith("present.")
        }

    metrics = {}

    def check(name, actual, expected):
        assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
        actual = actual.astype(np.float32)
        assert np.isfinite(actual).all(), name
        difference = np.abs(actual - expected)
        maximum = float(difference.max()) if difference.size else 0.0
        metrics[name] = {
            "shape": list(actual.shape),
            "max_abs_diff": maximum,
            "mean_abs_diff": float(difference.mean()) if difference.size else 0.0,
        }
        event("comparison", name=name, **metrics[name])
        tolerance = 1e-2 if DTYPE != "f32" else 1e-3
        np.testing.assert_allclose(
            actual, expected, atol=tolerance, rtol=tolerance, err_msg=name
        )

    count = config["audio_decoder_config"]["num_codebooks"]
    vocab = config["audio_decoder_config"]["vocab_size"]
    semantic_start = config["semantic_start_token_id"]
    semantic_end = config["semantic_end_token_id"]
    eos = config["eos_token_id"]
    candidates = np.array([eos, *range(semantic_start, semantic_end + 1)])
    decoder_cache, tower_cache = empty_cache("decoder"), empty_cache("audio_decoder")
    generated = []
    current_frames: np.ndarray | None = None
    profile_summary = None
    for step in range(golden["attempted_frames"]):
        with np.load(ROOT / "reference" / f"state-{step:02}.npz") as state:
            if step == 0:
                current_frames = state["input_frames"]
            else:
                assert current_frames is not None
                np.testing.assert_array_equal(current_frames, state["input_frames"])
            assert current_frames is not None
            embedded = run("embedding", {"frames": current_frames[None]})
            check(f"state_{step}.embedding", embedded["inputs_embeds"], state["inputs_embeds"])
            check(
                f"state_{step}.shared_input", embedded["shared_input"], state["shared_input"]
            )
            positions = state["position_ids"]
            mask = np.ones((1, int(positions[0, -1]) + 1), dtype=np.int64)
            decoded = run(
                "decoder",
                {
                    **decoder_cache,
                    "inputs_embeds": embedded["inputs_embeds"],
                    "position_ids": positions,
                    "attention_mask": mask,
                },
            )
            if step == 0:
                profile_path = pathlib.Path(sessions["decoder"].end_profiling())
                counts: collections.Counter[tuple[str, str]] = collections.Counter()
                for entry in json.loads(profile_path.read_text()):
                    args = entry.get("args", {})
                    if "provider" in args:
                        counts[(args["provider"], args.get("op_name", "unknown"))] += 1
                profile_summary = {
                    f"{provider}/{op}": n for (provider, op), n in counts.items()
                }
                assert (
                    sum(
                        counts[("CUDAExecutionProvider", op)]
                        for op in ("MatMul", "FusedMatMul", "Gemm")
                    )
                    > 0
                )
                if EXPERT_EXECUTION == "dynamic_fp8":
                    assert (
                        counts[("CUDAExecutionProvider", "QMoE")]
                        == config["text_config"]["num_hidden_layers"]
                    ), "Native FP8 experts did not execute on CUDA"
                (ROOT / "profile-summary.json").write_text(
                    json.dumps(profile_summary, indent=2)
                )
                event("cuda_profile_verified", counts=profile_summary)
            check(f"state_{step}.hidden", decoded["hidden_states"], state["hidden_states"])
            check(f"state_{step}.logits", decoded["logits"], state["logits"])
            decoder_cache = next_cache(decoded)
            if step == 0:
                prompt = current_frames
                prefilled = run(
                    "audio_decoder",
                    {
                        **tower_cache,
                        "hidden_states": decoded["hidden_states"][:, :-1],
                        "shared_input": embedded["shared_input"][:, :-1],
                        "codes": prompt[None, 1:, 1:],
                        "semantic_mask": (
                            (prompt[None, 1:, 0] >= semantic_start)
                            & (prompt[None, 1:, 0] <= semantic_end)
                        ),
                        "position_ids": positions[:, :-1],
                        "attention_mask": mask[:, :-1],
                    },
                )
                check(
                    "prefill.tower",
                    prefilled["codebook_logits"],
                    state["prefill_tower_logits"],
                )
                tower_cache = next_cache(prefilled)
            main = int(
                candidates[decoded["logits"][0, -1, candidates].astype(np.float32).argmax()]
            )
            codes = np.zeros((1, 1, count), dtype=np.int64)
            codes[0, 0, 0] = np.clip(main - semantic_start, 0, vocab - 1)
            semantic = np.array([[semantic_start <= main <= semantic_end]])
            single_pass = audio_model_name == "audio_generator"
            for index in range(1 if single_pass else count - 1):
                tower_output = run(
                    "audio_decoder",
                    {
                        **tower_cache,
                        "hidden_states": decoded["hidden_states"][:, -1:],
                        "shared_input": embedded["shared_input"][:, -1:],
                        "codes": codes,
                        "semantic_mask": semantic,
                        "position_ids": positions[:, -1:],
                        "attention_mask": mask,
                    },
                    generate=True,
                )
                if single_pass:
                    codes = tower_output["frame_codes"]
                else:
                    codes[0, 0, index + 1] = (
                        tower_output["codebook_logits"][0, 0, index]
                        .astype(np.float32)
                        .argmax()
                    )
            check(
                f"state_{step}.tower", tower_output["codebook_logits"], state["tower_logits"]
            )
            current_frames = np.concatenate((np.array([[[main]]]), codes), axis=-1)[0]
            np.testing.assert_array_equal(current_frames, state["sampled_frame"])
            tower_cache = next_cache(tower_output)
            event("frame_matches", step=step, frame=current_frames.tolist())
            if main == eos:
                break
            generated.extend(current_frames.tolist())
    assert len(generated) == len(golden["generated_frames"])
    np.testing.assert_array_equal(generated, golden["generated_frames"])
    report = {
        "status": "passed",
        "revision": REVISION,
        "source_revision": SOURCE_REVISION,
        "dtype": DTYPE_NAMES[DTYPE],
        "expert_execution": EXPERT_EXECUTION,
        "reference_environment": golden.get("reference_environment"),
        "audio_component": audio_model_name,
        "onnxruntime_version": ort.__version__,
        "provider": "CUDAExecutionProvider",
        "ort_graph_optimization": "ORT_ENABLE_BASIC",
        "atol": 1e-2 if DTYPE != "f32" else 1e-3,
        "rtol": 1e-2 if DTYPE != "f32" else 1e-3,
        "prompt_rows": golden["prompt_rows"],
        "generated_frames": len(generated),
        "attempted_frames": golden["attempted_frames"],
        "seconds": time.monotonic() - start,
        "metrics": metrics,
        "profile_summary": profile_summary,
    }
    (ROOT / "validation-result.json").write_text(json.dumps(report, indent=2))
    event(
        "REAL_CHECKPOINT_VALIDATION_PASSED",
        **{key: report[key] for key in ("generated_frames", "attempted_frames", "seconds")},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("download", "export", "reference", "compare"))
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    parser.add_argument("--dtype", choices=("f32", "f16", "bf16"), default="f32")
    parser.add_argument("--native-fp8", action="store_true")
    parser.add_argument(
        "--task", choices=("fish-omni", "fish-omni-generation"), default="fish-omni"
    )
    parser.add_argument("--execution-provider", choices=("default", "cuda", "onnx-standard"))
    parser.add_argument("--audio-component", choices=("audio_decoder", "audio_generator"))
    parser.add_argument("--golden", type=pathlib.Path)
    parser.add_argument("--reference-devices", default="0,1,2,3")
    parser.add_argument("--reference-tower-device", default=4, type=int)
    parser.add_argument("--embedding-device", default=0, type=int)
    parser.add_argument("--audio-device", default=1, type=int)
    parser.add_argument("--decoder-device", default=2, type=int)
    args = parser.parse_args()
    ROOT = args.work_dir.resolve()
    ROOT.mkdir(parents=True, exist_ok=True)
    REFERENCE_DEVICES = tuple(int(device) for device in args.reference_devices.split(","))
    REFERENCE_TOWER_DEVICE = args.reference_tower_device
    EMBEDDING_DEVICE = args.embedding_device
    AUDIO_DEVICE = args.audio_device
    DECODER_DEVICE = args.decoder_device
    DTYPE = args.dtype
    EXPERT_EXECUTION = "dynamic_fp8" if args.native_fp8 else "dense"
    if args.golden is not None:
        GOLDEN = args.golden.resolve()
    elif DTYPE != "f32" or args.native_fp8:
        suffix = f"fp8_{DTYPE}" if args.native_fp8 else DTYPE
        GOLDEN = GOLDEN.with_name(f"drama-3_{suffix}_tokens_generation.json")
        if args.stage == "compare" and not GOLDEN.is_file():
            parser.error(
                "Native-FP8/reduced-precision comparison requires --golden pointing to the independent "
                "reference stage's golden.json until a matching golden is committed"
            )
    if args.stage == "download":
        download()
    elif args.stage == "export":
        export(
            checkpoint_path(),
            task=args.task,
            execution_provider=args.execution_provider,
            native_fp8=args.native_fp8,
        )
    elif args.stage == "reference":
        reference(checkpoint_path(), native_fp8=args.native_fp8)
    else:
        compare(audio_model_name=args.audio_component)
