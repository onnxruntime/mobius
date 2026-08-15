#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Build and validate the pinned Nemotron 3.5 Lightning Q8_0 GGUF package."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from inference import (
    _as_numpy,
    _create_session,
    _initial_states,
    _run_session,
    _token_feeds,
    _update_states,
)

GGUF_REPO = "unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF"
GGUF_REVISION = "f2d3fe3694501008786e81e5f20360cbf715496a"
GGUF_FILENAME = "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf"
GGUF_SIZE = 35_004_643_392
GGUF_SHA256 = "dc5276dd0619c04e277504d2358a793e31ccbe39e894d767d0d14f2a221e2ca4"
LLAMA_CPP_COMMIT = "9d57ce456c94d241dde672b2db9cf18879766568"
LLAMA_CPP_SERVER_COMMAND = (
    f"llama-server -m {GGUF_FILENAME} -c 128 -t 12 -tb 12 -b 64 -ub 64 "
    "-ngl 0 --host 127.0.0.1 --port 18081 --no-warmup"
)
PROMPT = "The capital of France is"
PROMPT_IDS = [1784, 8961, 1307, 5498, 1395]
EXPECTED_IDS = [6993, 1046, 1256, 1010, 1784, 8961, 1307, 10787]
EXPECTED_TEXT = " Paris.  \nThe capital of Germany"
LLAMA_CPP_REFERENCE = {
    "commit": LLAMA_CPP_COMMIT,
    "compiler": "MSVC 19.44.35228.0",
    "server_command": LLAMA_CPP_SERVER_COMMAND,
    "completion_request": {
        "prompt": PROMPT,
        "n_predict": 8,
        "temperature": 0,
        "seed": 1,
        "cache_prompt": False,
        "n_probs": 1,
    },
    "prompt_ids": PROMPT_IDS,
    "generated_ids": EXPECTED_IDS,
    "generated_text": EXPECTED_TEXT,
    "prompt_tokens_per_second": 5.58,
    "generation_tokens_per_second": 5.95,
    "peak_working_set_gib": 16.81,
}
OFFICIAL_TOKENIZER_SHA256 = {
    "chat_template.jinja": "58933db77d3099b4f78c55a38347a72e1ea05b97d6bd8f38775303dc0194e0a9",
    "special_tokens_map.json": (
        "e9435fefd6d838fd9fcbbc44b97a8e3ff322be7f6dfb7e4fd2468586574bb52b"
    ),
    "tokenizer_config.json": (
        "10f93eabcb9b1602fbb991d6308e787ce1df28ee9cd7a1c6d1e8c3f338b957bc"
    ),
    "tokenizer.json": "623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7",
}
GGUF_CHAT_TEMPLATE_SHA256 = "cbb337473ffde036fd4b6e7e7763dcb97c7cd8b4a311cd52d361d2766b00eb7c"


def _memory_sample() -> dict[str, int]:
    try:
        import psutil
    except ImportError:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        return {"rss": rss, "peak_working_set": rss}

    info = psutil.Process().memory_info()
    return {
        "rss": int(info.rss),
        "peak_working_set": int(getattr(info, "peak_wset", info.rss)),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_tokenizer_contract(
    gguf_path: Path,
    official_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    from tokenizers import Tokenizer

    from mobius.integrations.gguf._reader import GGUFModel

    metadata = GGUFModel(gguf_path).metadata
    tokens = metadata["tokenizer.ggml.tokens"]
    expected_tokens = {0: "<unk>", 1: "<s>", 2: "</s>", 11: "<|im_end|>"}
    actual_tokens = {index: tokens[index] for index in expected_tokens}
    if actual_tokens != expected_tokens:
        raise ValueError(f"Unexpected pinned special-token strings: {actual_tokens}")

    actual_hashes = {}
    for filename, expected_sha256 in OFFICIAL_TOKENIZER_SHA256.items():
        source = official_dir / filename
        if not source.is_file():
            raise FileNotFoundError(f"Missing pinned official tokenizer asset: {source}")
        actual_hashes[filename] = _sha256(source)
        if actual_hashes[filename] != expected_sha256:
            raise ValueError(
                f"Pinned official tokenizer asset {filename!r} has SHA-256 "
                f"{actual_hashes[filename]}, expected {expected_sha256}"
            )

    rebuilt_path = output_dir / "tokenizer.json"
    rebuilt = Tokenizer.from_file(str(rebuilt_path))
    official = Tokenizer.from_file(str(official_dir / "tokenizer.json"))
    if rebuilt.get_vocab(with_added_tokens=True) != official.get_vocab(with_added_tokens=True):
        raise ValueError(
            "Reconstructed tokenizer vocabulary differs from the pinned official asset"
        )
    samples = (
        PROMPT,
        EXPECTED_TEXT,
        "café déjà vu — 你好 🌍",
        "  leading\tspaces\r\nnewlines  ",
        "<|im_start|>assistant\n<think>x</think><|im_end|>",
        bytes(range(1, 128)).decode("latin1"),
    )
    for sample in samples:
        official_encoding = official.encode(sample)
        rebuilt_encoding = rebuilt.encode(sample)
        if rebuilt_encoding.ids != official_encoding.ids:
            raise ValueError(f"Reconstructed tokenizer IDs differ for {sample!r}")
        if rebuilt.decode(
            rebuilt_encoding.ids,
            skip_special_tokens=False,
        ) != official.decode(official_encoding.ids, skip_special_tokens=False):
            raise ValueError(f"Reconstructed tokenizer decode differs for {sample!r}")

    official_json = json.loads((official_dir / "tokenizer.json").read_text(encoding="utf-8"))
    rebuilt_json = json.loads(rebuilt_path.read_text(encoding="utf-8"))
    for section in ("pre_tokenizer", "decoder", "post_processor", "added_tokens"):
        if rebuilt_json[section] != official_json[section]:
            raise ValueError(f"Reconstructed tokenizer {section!r} differs from official")

    chat_template = metadata.get("tokenizer.chat_template")
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("Pinned GGUF tokenizer has no chat template")
    gguf_template_sha256 = hashlib.sha256(
        chat_template.replace("\r\n", "\n").encode()
    ).hexdigest()
    if gguf_template_sha256 != GGUF_CHAT_TEMPLATE_SHA256:
        raise ValueError(f"Pinned GGUF chat-template SHA-256 differs: {gguf_template_sha256}")
    if gguf_template_sha256 == actual_hashes["chat_template.jinja"]:
        raise ValueError(
            "GGUF and official chat templates unexpectedly have the same provenance"
        )

    for filename in (
        "chat_template.jinja",
        "special_tokens_map.json",
        "tokenizer_config.json",
    ):
        shutil.copyfile(official_dir / filename, output_dir / filename)
    return {
        "asset_bos_token_id": 1,
        "asset_eos_token_id": 11,
        "asset_padding_token_id": 11,
        "asset_unk_token_id": 0,
        "reconstructed_matches_official_tokenizer": True,
        "official_revision": "d468880b6ad3c6e0d21377ce7242adaea4cc884d",
        "official_asset_sha256": actual_hashes,
        "gguf_chat_template_sha256_rejected": gguf_template_sha256,
    }


def _graph_audit(model, expected_weight_initializers: set[str]) -> dict[str, Any]:
    op_counts = Counter(
        f"{node.domain or 'ai.onnx'}::{node.op_type}" for node in model.graph.all_nodes()
    )
    initializers = model.graph.initializers
    quantized_weights = [
        name
        for name, value in initializers.items()
        if value.dtype.name == "UINT8" and name.endswith((".weight", ".qweight"))
    ]
    unset = [name for name, value in initializers.items() if value.const_value is None]
    if unset:
        raise ValueError(f"Weighted graph has {len(unset)} unset initializers: {unset[:10]}")
    missing_weights = expected_weight_initializers - set(initializers)
    folded_gate_weights = {
        name for name in expected_weight_initializers if name.endswith(".moe.gate.weight")
    }
    if missing_weights != folded_gate_weights:
        raise ValueError(
            f"Weighted graph is missing {len(missing_weights)} mapped initializers: "
            f"{sorted(missing_weights)[:10]}"
        )
    # Gate matmuls explicitly cast their weights to float32 to preserve the
    # official routing contract. Constant folding consumes those 23 source
    # names and materializes replacement constants in the weighted graph.
    post_fold_initializers = sorted(set(initializers) - expected_weight_initializers)
    if op_counts["com.microsoft::MatMulNBits"] != 6005:
        raise ValueError(f"Unexpected MatMulNBits count: {dict(op_counts)}")
    if op_counts["com.microsoft::GatherBlockQuantized"] != 1:
        raise ValueError(f"Unexpected GatherBlockQuantized count: {dict(op_counts)}")
    if len(expected_weight_initializers) != 18_255:
        raise ValueError(
            "Expected 18,255 mapped weight initializers, got "
            f"{len(expected_weight_initializers)}"
        )
    if len(quantized_weights) != 6006:
        raise ValueError(
            f"Expected 6,006 Q8 weight initializers, got {len(quantized_weights)}"
        )
    forbidden = {
        name: count
        for name, count in op_counts.items()
        if name.endswith(("::QuantizeLinear", "::DequantizeLinear"))
    }
    if forbidden:
        raise ValueError(f"Unexpected dequantize/requantize operators: {forbidden}")
    return {
        "op_histogram": dict(sorted(op_counts.items())),
        "initializer_count": len(initializers),
        "mapped_weight_initializer_count": len(expected_weight_initializers),
        "folded_gate_weight_count": len(folded_gate_weights),
        "post_fold_initializer_count": len(post_fold_initializers),
        "post_fold_initializers": post_fold_initializers,
        "q8_weight_initializer_count": len(quantized_weights),
        "matmul_nbits_count": op_counts["com.microsoft::MatMulNBits"],
        "gather_block_quantized_count": op_counts["com.microsoft::GatherBlockQuantized"],
        "forbidden_qdq_ops": forbidden,
    }


def _mapping_audit(gguf_path: Path) -> tuple[dict[str, Any], set[str]]:
    from mobius.integrations.gguf._architecture import (
        GGUFMappingAudit,
        create_architecture_adapter,
    )
    from mobius.integrations.gguf._reader import GGUFModel

    source = GGUFModel(gguf_path)
    adapter = create_architecture_adapter(source.architecture, source)
    if adapter is None:
        raise ValueError(f"No adapter for {source.architecture!r}")
    adapter.validate_model(source=str(gguf_path))
    audit = GGUFMappingAudit()
    qtypes: Counter[str] = Counter()
    base_qtypes: Counter[str] = Counter()
    mtp_qtypes: Counter[str] = Counter()
    qtype_parameters: Counter[str] = Counter()
    base_qtype_parameters: Counter[str] = Counter()
    mtp_qtype_parameters: Counter[str] = Counter()
    q8_targets = 0
    expected_initializers: set[str] = set()
    for record in source._reader.tensors:
        shape = tuple(int(dim) for dim in reversed(record.shape))
        mapping = adapter.map_tensor(record.name, shape)
        audit.record(record.name, mapping)
        qtype = record.tensor_type.name
        parameters = math.prod(shape)
        qtypes[qtype] += 1
        qtype_parameters[qtype] += parameters
        (mtp_qtypes if record.name.startswith("blk.52.") else base_qtypes)[qtype] += 1
        (mtp_qtype_parameters if record.name.startswith("blk.52.") else base_qtype_parameters)[
            qtype
        ] += parameters
        if mapping is not None and mapping.exclusion is None and qtype == "Q8_0":
            q8_targets += len(mapping.targets)
            for target in mapping.targets:
                stem = target.initializer_name.removesuffix(".weight")
                expected_initializers.add(
                    f"{stem}.qweight"
                    if target.initializer_name == "model.embed_tokens.weight"
                    else target.initializer_name
                )
                expected_initializers.add(f"{stem}.scales")
                expected_initializers.add(f"{stem}.zero_points")
        elif mapping is not None and mapping.exclusion is None:
            expected_initializers.update(target.initializer_name for target in mapping.targets)
    adapter.validate_mapping_audit(audit)
    metadata = source.metadata
    tokens = metadata["tokenizer.ggml.tokens"]
    merges = metadata["tokenizer.ggml.merges"]
    return {
        "source_tensors": len(source.tensor_names),
        "mapped_sources": len(audit.mapped_sources),
        "mtp_exclusions": len(audit.excluded_sources),
        "logical_targets": len(audit.target_sources),
        "q8_logical_targets": q8_targets,
        "mapped_weight_initializers": len(expected_initializers),
        "qtype_tensors": dict(sorted(qtypes.items())),
        "base_qtype_tensors": dict(sorted(base_qtypes.items())),
        "mtp_qtype_tensors": dict(sorted(mtp_qtypes.items())),
        "qtype_parameters": dict(sorted(qtype_parameters.items())),
        "base_qtype_parameters": dict(sorted(base_qtype_parameters.items())),
        "mtp_qtype_parameters": dict(sorted(mtp_qtype_parameters.items())),
        "tokenizer": {
            "profile": [
                metadata["tokenizer.ggml.model"],
                metadata["tokenizer.ggml.pre"],
            ],
            "bos_token_id": metadata["tokenizer.ggml.bos_token_id"],
            "eos_token_id": metadata["tokenizer.ggml.eos_token_id"],
            "rejected_gguf_padding_token_id": metadata["tokenizer.ggml.padding_token_id"],
            "token_count": len(tokens),
            "merge_count": len(merges),
            "token_sha256": hashlib.sha256("\n".join(tokens).encode()).hexdigest(),
            "merge_sha256": hashlib.sha256("\n".join(merges).encode()).hexdigest(),
        },
    }, expected_initializers


def _build(gguf_path: Path, official_tokenizer_dir: Path, output_dir: Path) -> None:
    if gguf_path.stat().st_size != GGUF_SIZE:
        raise ValueError(
            f"Pinned GGUF size mismatch: {gguf_path.stat().st_size} != {GGUF_SIZE}"
        )
    source_sha256 = _sha256(gguf_path)
    if source_sha256 != GGUF_SHA256:
        raise ValueError(f"Pinned GGUF SHA-256 mismatch: {source_sha256}")

    from mobius import build_from_gguf
    from mobius.integrations.gguf import write_gguf_tokenizer_json

    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_memory = _memory_sample()
    build_started = time.perf_counter()
    package = build_from_gguf(
        gguf_path,
        keep_quantized=True,
        execution_provider="cpu",
    )
    build_seconds = time.perf_counter() - build_started
    build_memory = _memory_sample()
    mapping_audit, expected_initializers = _mapping_audit(gguf_path)
    expected_mapping = {
        "source_tensors": 417,
        "mapped_sources": 401,
        "mtp_exclusions": 16,
        "logical_targets": 6243,
        "q8_logical_targets": 6006,
        "mapped_weight_initializers": 18_255,
    }
    if {key: mapping_audit[key] for key in expected_mapping} != expected_mapping:
        raise ValueError(f"Unexpected mapping audit: {mapping_audit}")
    graph_audit = _graph_audit(package["model"], expected_initializers)

    save_started = time.perf_counter()
    package.save(
        str(output_dir),
        external_data="onnx",
        progress_bar=False,
    )
    save_seconds = time.perf_counter() - save_started
    save_memory = _memory_sample()
    tokenizer_path = write_gguf_tokenizer_json(gguf_path, output_dir)
    if tokenizer_path is None:
        raise ValueError("Pinned GGUF tokenizer.json was not emitted")
    tokenizer_contract = _write_tokenizer_contract(
        gguf_path,
        official_tokenizer_dir,
        output_dir,
    )

    # The GGUF metadata's PAD=999 is not a valid runtime padding contract for
    # this model. Keep the official decoder contract and the tokenizer's role
    # token separate and explicit.
    _write_json(
        output_dir / "config.json",
        {
            "model_type": "nemotron_h",
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
        },
    )
    _write_json(
        output_dir / "generation_config.json",
        {
            "bos_token_id": 1,
            "eos_token_id": [2, 11],
            "pad_token_id": 0,
        },
    )

    package_bytes = sum(
        path.stat().st_size for path in output_dir.rglob("*") if path.is_file()
    )
    del package
    gc.collect()
    released_memory = _memory_sample()
    report = {
        "phase": "build",
        "source": {
            "repo": GGUF_REPO,
            "revision": GGUF_REVISION,
            "filename": GGUF_FILENAME,
            "size": GGUF_SIZE,
            "sha256": source_sha256,
        },
        "mapping": mapping_audit,
        "tokenizer_contract": tokenizer_contract,
        "graph": graph_audit,
        "package_bytes": package_bytes,
        "timing_seconds": {
            "build": build_seconds,
            "save": save_seconds,
        },
        "memory_bytes": {
            "baseline": baseline_memory,
            "after_build": build_memory,
            "after_save": save_memory,
            "after_release": released_memory,
        },
    }
    _write_json(output_dir / "gguf_q8_build_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


def _prefill(
    session,
    output_names: list[str],
    input_ids: list[int],
    attention_mask: np.ndarray,
):
    states = _initial_states(session)
    feeds = _token_feeds(
        session,
        np.asarray([input_ids], dtype=np.int64),
        total_length=len(input_ids),
        position_ids=np.arange(len(input_ids), dtype=np.int64)[None, :],
        states=states,
    )
    feeds["attention_mask"] = attention_mask
    outputs = _run_session(session, output_names, feeds)
    _update_states(states, output_names, outputs)
    logits = _as_numpy(outputs[output_names.index("logits")]).astype(np.float32)
    return logits, states


def _run(output_dir: Path, device: str) -> None:
    import onnxruntime as ort
    from tokenizers import Tokenizer

    baseline_memory = _memory_sample()
    load_started = time.perf_counter()
    session = _create_session(output_dir / "model.onnx", device, profile=False)
    load_seconds = time.perf_counter() - load_started
    load_memory = _memory_sample()
    output_names = [output.name for output in session.get_outputs()]

    prefill_started = time.perf_counter()
    logits, states = _prefill(
        session,
        output_names,
        PROMPT_IDS,
        np.ones((1, len(PROMPT_IDS)), dtype=np.int64),
    )
    prefill_seconds = time.perf_counter() - prefill_started
    prefill_memory = _memory_sample()

    padded_ids = [*PROMPT_IDS, 0, 0]
    padded_mask = np.asarray([[1] * len(PROMPT_IDS) + [0, 0]], dtype=np.int64)
    padded_logits, _ = _prefill(session, output_names, padded_ids, padded_mask)
    padded_real_token_max_abs = float(
        np.max(np.abs(padded_logits[:, : len(PROMPT_IDS)] - logits))
    )
    np.testing.assert_allclose(
        padded_logits[:, : len(PROMPT_IDS)],
        logits,
        rtol=1e-5,
        atol=1e-5,
    )

    generated = []
    decode_seconds = []
    decode_memory = []
    past_length = len(PROMPT_IDS)
    next_logits = logits[0, -1]
    for index in range(len(EXPECTED_IDS)):
        token_id = int(np.argmax(next_logits))
        generated.append(token_id)
        if index + 1 == len(EXPECTED_IDS):
            break
        started = time.perf_counter()
        feeds = _token_feeds(
            session,
            np.asarray([[token_id]], dtype=np.int64),
            total_length=past_length + 1,
            position_ids=np.asarray([[past_length]], dtype=np.int64),
            states=states,
        )
        outputs = _run_session(session, output_names, feeds)
        _update_states(states, output_names, outputs)
        past_length += 1
        next_logits = _as_numpy(outputs[output_names.index("logits")])[0, -1].astype(
            np.float32
        )
        decode_seconds.append(time.perf_counter() - started)
        decode_memory.append(_memory_sample())

    if generated != EXPECTED_IDS:
        raise AssertionError(
            f"Greedy tokens differ from llama.cpp {LLAMA_CPP_COMMIT}: "
            f"actual={generated}, expected={EXPECTED_IDS}"
        )
    tokenizer = Tokenizer.from_file(str(output_dir / "tokenizer.json"))
    prompt_ids = tokenizer.encode(PROMPT).ids
    if prompt_ids != PROMPT_IDS:
        raise AssertionError(
            f"Tokenizer prompt IDs differ: actual={prompt_ids}, expected={PROMPT_IDS}"
        )
    generated_text = tokenizer.decode(generated)
    if generated_text != EXPECTED_TEXT:
        raise AssertionError(
            f"Decoded text differs: actual={generated_text!r}, expected={EXPECTED_TEXT!r}"
        )
    if tokenizer.token_to_id("<|im_end|>") != 11:
        raise AssertionError("Tokenizer role token <|im_end|> must remain ID 11")
    tokenizer_config = json.loads(
        (output_dir / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    asset_special_ids = {
        role: tokenizer.token_to_id(tokenizer_config[f"{config_name}_token"])
        for role, config_name in (
            ("bos", "bos"),
            ("eos", "eos"),
            ("padding", "pad"),
            ("unknown", "unk"),
        )
    }
    if asset_special_ids != {"bos": 1, "eos": 11, "padding": 11, "unknown": 0}:
        raise AssertionError(f"Tokenizer asset contract differs: {asset_special_ids}")

    report = {
        "phase": "runtime",
        "onnxruntime_version": ort.__version__,
        "onnxruntime_build_info": ort.get_build_info(),
        "available_providers": ort.get_available_providers(),
        "session_providers": session.get_providers(),
        "device": device,
        "prompt": PROMPT,
        "prompt_ids": PROMPT_IDS,
        "generated_ids": generated,
        "generated_text": generated_text,
        "llama_cpp_commit": LLAMA_CPP_COMMIT,
        "llama_cpp_reference": LLAMA_CPP_REFERENCE,
        "padding_contract": {
            "gguf_padding_id_rejected": 999,
            "runtime_padding_id": 0,
            "runtime_eos_ids": [2, 11],
            "tokenizer_asset_special_ids": asset_special_ids,
            "right_padded_real_token_logits_max_abs": padded_real_token_max_abs,
            "right_padded_real_token_logits_atol": 1e-5,
        },
        "timing_seconds": {
            "session_load": load_seconds,
            "prefill": prefill_seconds,
            "cached_decode_steps": decode_seconds,
        },
        "throughput_tokens_per_second": {
            "prefill": len(PROMPT_IDS) / prefill_seconds,
            "cached_decode_steps": len(decode_seconds) / sum(decode_seconds),
        },
        "memory_bytes": {
            "baseline": baseline_memory,
            "after_session_load": load_memory,
            "after_prefill": prefill_memory,
            "after_cached_decode_steps": decode_memory,
        },
    }
    _write_json(output_dir / "gguf_q8_runtime_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


def _all(
    gguf_path: Path,
    official_tokenizer_dir: Path,
    output_dir: Path,
    device: str,
) -> None:
    common = [sys.executable, str(Path(__file__).resolve())]
    subprocess.run(
        [
            *common,
            "--phase",
            "build",
            "--gguf",
            str(gguf_path),
            "--official-tokenizer-dir",
            str(official_tokenizer_dir),
            "--output",
            str(output_dir),
        ],
        check=True,
    )
    subprocess.run(
        [
            *common,
            "--phase",
            "run",
            "--output",
            str(output_dir),
            "--device",
            device,
        ],
        check=True,
    )
    combined = {
        "build": json.loads(
            (output_dir / "gguf_q8_build_report.json").read_text(encoding="utf-8")
        ),
        "runtime": json.loads(
            (output_dir / "gguf_q8_runtime_report.json").read_text(encoding="utf-8")
        ),
    }
    _write_json(output_dir / "gguf_q8_acceptance_report.json", combined)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["all", "build", "run"], default="all")
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--official-tokenizer-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    if args.phase in {"all", "build"} and args.gguf is None:
        parser.error("--gguf is required for build/all phases")
    if args.phase in {"all", "build"} and args.official_tokenizer_dir is None:
        parser.error("--official-tokenizer-dir is required for build/all phases")
    if args.phase == "build":
        _build(args.gguf, args.official_tokenizer_dir, args.output)
    elif args.phase == "run":
        _run(args.output, args.device)
    else:
        _all(args.gguf, args.official_tokenizer_dir, args.output, args.device)


if __name__ == "__main__":
    main()
