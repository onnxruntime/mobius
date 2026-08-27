# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Real-artifact GGUF parity for the smallest reproducible checkpoints.

The pinned SmolLM2 Q4_K_M companion (105,454,144 bytes, LFS SHA-256
ed5fa30c487b282ec156c29062f1222e5c20875a944ac98289dbd242e947f747) is intentionally
not an L4/L5 case: its preserved MatMulNBits route diverges from the same artifact
under llama.cpp, while Mobius's dequantized route matches llama.cpp. Comparing that
quantized artifact to the unquantized HuggingFace checkpoint would not establish
same-weight provenance.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from importlib.metadata import packages_distributions, version
from pathlib import Path
from unittest import mock

import huggingface_hub.constants
import numpy as np
import onnxruntime as ort
import pytest
import torch
import yaml
from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url
from transformers import AutoModelForCausalLM, AutoTokenizer

from mobius import ModelPackage, build_from_gguf
from mobius.__main__ import main
from mobius.integrations.gguf import (
    GGUFTokenizerAsset,
    GGUFTokenizerSource,
    materialize_gguf_tokenizer,
    write_gguf_runtime_package,
)
from mobius.integrations.gguf._quantization_report import (
    GGUFQuantizationReport,
    QuantizationDisposition,
)
from mobius.integrations.gguf._reader import GGUFModel
from mobius.integrations.gguf._runtime_evidence import (
    gguf_graph_package_identity,
    runtime_evidence,
)
from mobius.integrations.gguf._tokenizer import inspect_gguf_tokenizer


@dataclass(frozen=True)
class _RuntimeCase:
    name: str
    gguf_repository: str
    gguf_revision: str
    gguf_filename: str
    gguf_size: int
    gguf_sha256: str
    reference_repository: str
    reference_revision: str
    prompt: str
    tensor_qtypes: dict[str, int]
    config_sha256: str
    generated_tokens: tuple[int, ...]
    tokenizer_repository: str
    tokenizer_revision: str
    tokenizer_metadata_sha256: str
    tokenizer_assets: tuple[tuple[str, int, str], ...]
    tokenizer_identity_exact: bool


_CASES = (
    _RuntimeCase(
        name="smollm-135m-f16",
        gguf_repository="neopolita/smollm-135m-gguf",
        gguf_revision="22cca988936eafe92908e7558907c3964e10bba7",
        gguf_filename="ggml-model-f16.gguf",
        gguf_size=270_885_504,
        gguf_sha256="ec8c775c16944a7e4b5251f97b3f848500dcc3e701b0d492ce9055cea42138a2",
        reference_repository="HuggingFaceTB/SmolLM-135M",
        reference_revision="1d461723eec654e65efdc40cf49301c89c0c92f4",
        prompt="Once upon a time,",
        tensor_qtypes={"F16": 211, "F32": 61},
        config_sha256="d3f3f2abf531abde55e04a52b5c892c93943b7e260160c796c490618a2e84886",
        generated_tokens=(
            665,
            436,
            253,
            7436,
            1838,
            8180,
            3365,
            14176,
            30,
            2306,
            5732,
            3415,
            874,
            18064,
            284,
            4464,
            351,
            874,
            2428,
            30,
        ),
        tokenizer_repository="HuggingFaceTB/SmolLM-135M",
        tokenizer_revision="1d461723eec654e65efdc40cf49301c89c0c92f4",
        tokenizer_metadata_sha256=(
            "46646ba36ecae43de6f9f649d217774b889e0fd405af92205319b882927493fc"
        ),
        tokenizer_assets=(
            (
                "special_tokens_map.json",
                831,
                "e786b595b9a23148bf1630df78d9037a048ea671e48bfd3549a1e3c233742bb3",
            ),
            (
                "tokenizer.json",
                2_104_556,
                "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
            ),
            (
                "tokenizer_config.json",
                3_685,
                "238ad6b60d48e471624ea70bc79e92f2611844d5016471fee8c167854bcb98e8",
            ),
        ),
        tokenizer_identity_exact=True,
    ),
    _RuntimeCase(
        name="smollm2-135m-instruct-f16",
        gguf_repository="unsloth/SmolLM2-135M-Instruct-GGUF",
        gguf_revision="9e6855bc4be717fca1ef21360a1db4b29d5c559a",
        gguf_filename="SmolLM2-135M-Instruct-F16.gguf",
        gguf_size=270_885_952,
        gguf_sha256="5157ca60744d21631818364854ac8e4452e1b8022d2ab4c8a2f9cda2344afb30",
        reference_repository="HuggingFaceTB/SmolLM2-135M-Instruct",
        reference_revision="12fd25f77366fa6b3b4b768ec3050bf629380bac",
        prompt="Here is my poem:",
        tensor_qtypes={"F16": 211, "F32": 61},
        config_sha256="f9ceb816433d5aa32918761944be76ecdb090df8cce7cdb64d5d8f9186f7117f",
        generated_tokens=(
            198,
            198,
            18,
            504,
            2388,
            13685,
            281,
            260,
            6376,
            28,
            198,
            49,
            3091,
            10286,
            28,
            198,
            1589,
            506,
            253,
            9154,
        ),
        tokenizer_repository="HuggingFaceTB/SmolLM2-135M-Instruct",
        tokenizer_revision="12fd25f77366fa6b3b4b768ec3050bf629380bac",
        tokenizer_metadata_sha256=(
            "cb0b637d59effdc3ab02f063039e597157fa4996663848cc2178510af5880ace"
        ),
        tokenizer_assets=(
            (
                "special_tokens_map.json",
                655,
                "2b7379f3ae813529281a5c602bc5a11c1d4e0a99107aaa597fe936c1e813ca52",
            ),
            (
                "tokenizer.json",
                2_104_556,
                "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
            ),
            (
                "tokenizer_config.json",
                3_764,
                "4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df",
            ),
        ),
        tokenizer_identity_exact=False,
    ),
)

_Q4_K_M_CASE = _RuntimeCase(
    name="smollm2-135m-instruct-q4-k-m",
    gguf_repository="unsloth/SmolLM2-135M-Instruct-GGUF",
    gguf_revision="9e6855bc4be717fca1ef21360a1db4b29d5c559a",
    gguf_filename="SmolLM2-135M-Instruct-Q4_K_M.gguf",
    gguf_size=105_454_144,
    gguf_sha256="ed5fa30c487b282ec156c29062f1222e5c20875a944ac98289dbd242e947f747",
    reference_repository="HuggingFaceTB/SmolLM2-135M-Instruct",
    reference_revision="12fd25f77366fa6b3b4b768ec3050bf629380bac",
    prompt="Here is my poem:",
    tensor_qtypes={"F32": 61, "Q4_K": 16, "Q5_0": 166, "Q6_K": 14, "Q8_0": 15},
    config_sha256="f9ceb816433d5aa32918761944be76ecdb090df8cce7cdb64d5d8f9186f7117f",
    generated_tokens=(198, 198, 18, 504, 2388, 13685, 284, 5208, 28, 198),
    tokenizer_repository="HuggingFaceTB/SmolLM2-135M-Instruct",
    tokenizer_revision="12fd25f77366fa6b3b4b768ec3050bf629380bac",
    tokenizer_metadata_sha256=(
        "cb0b637d59effdc3ab02f063039e597157fa4996663848cc2178510af5880ace"
    ),
    tokenizer_assets=(
        (
            "special_tokens_map.json",
            655,
            "2b7379f3ae813529281a5c602bc5a11c1d4e0a99107aaa597fe936c1e813ca52",
        ),
        (
            "tokenizer.json",
            2_104_556,
            "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
        ),
        (
            "tokenizer_config.json",
            3_764,
            "4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df",
        ),
    ),
    tokenizer_identity_exact=False,
)


@dataclass(frozen=True)
class _PromotedRuntimeCase:
    name: str
    evidence_id: str
    reference_repository: str
    reference_revision: str
    prompt: str
    generated_tokens: tuple[int, ...]
    atol: float


_PROMOTED_RUNTIME_CASES = (
    _PromotedRuntimeCase(
        name="qwen2.5-0.5b-instruct-q8",
        evidence_id="qwen2.5-0.5b-instruct-q8-ort-genai-0.15.2",
        reference_repository="Qwen/Qwen2.5-0.5B-Instruct",
        reference_revision="7ae557604adf67be50417f59c2c2f167def9a775",
        prompt="Hello",
        generated_tokens=(
            271,
            40,
            1079,
            4460,
            311,
            1855,
            264,
            2025,
            429,
            646,
            1477,
            279,
            7192,
            897,
            304,
            264,
            2661,
            1140,
            315,
            5109,
        ),
        atol=4e-4,
    ),
    _PromotedRuntimeCase(
        name="lfm2-350m-f16",
        evidence_id="lfm2-350m-f16-ort-genai-0.15.2",
        reference_repository="LiquidAI/LFM2-350M",
        reference_revision="f37d3f5c8c5484bc01dad379a595cf4c68c4e70e",
        prompt="Hello",
        generated_tokens=(
            8227,
            24771,
            938,
            31707,
            8587,
            4427,
            896,
            938,
            7306,
            18414,
            1399,
            4903,
            8227,
            24771,
            810,
            2492,
            5992,
            768,
            3680,
            4600,
        ),
        atol=3e-4,
    ),
)


@pytest.fixture
def isolated_hf_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Keep each real-artifact test's Hub/Xet state isolated and disposable."""
    cache_root = tmp_path / "huggingface-cache"
    hub_cache = cache_root / "hub"
    xet_cache = cache_root / "xet"
    cache_root.mkdir()
    monkeypatch.setenv("HF_HOME", str(cache_root))
    monkeypatch.setenv("HF_HUB_CACHE", str(hub_cache))
    monkeypatch.setenv("HF_XET_CACHE", str(xet_cache))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(cache_root / "transformers"))
    monkeypatch.setattr(huggingface_hub.constants, "HF_HOME", str(cache_root))
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(hub_cache))
    monkeypatch.setattr(huggingface_hub.constants, "HUGGINGFACE_HUB_CACHE", str(hub_cache))
    monkeypatch.setattr(huggingface_hub.constants, "HF_XET_CACHE", str(xet_cache))
    try:
        yield cache_root
    finally:
        shutil.rmtree(cache_root, ignore_errors=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _installed_ort_genai_version() -> str:
    distributions = packages_distributions().get("onnxruntime_genai", ())
    candidates = tuple(name for name in distributions if name.startswith("onnxruntime-genai"))
    if len(candidates) != 1:
        pytest.fail(
            f"Expected exactly one onnxruntime_genai distribution, found {sorted(candidates)}"
        )
    return version(candidates[0])


def _assert_stable_import_route(route: dict[str, object], case: _RuntimeCase) -> None:
    assert route["route_schema"] == 1
    assert route["architecture"] == "llama"
    assert route["model_type"] == "llama"
    assert route["module_type"] == "llama"
    assert route["config_sha256"] == case.config_sha256
    assert route["execution_provider"] == "cpu"
    assert route["preserve_quantization"] is False
    assert route["static_cache"] is False
    assert route["tensor_map_recipe"] == ["llama"]


def _empty_cache(session: ort.InferenceSession) -> dict[str, np.ndarray]:
    return {
        value.name: np.empty([1, value.shape[1], 0, value.shape[3]], dtype=np.float32)
        for value in session.get_inputs()
        if value.name.endswith((".key", ".value"))
    }


def _run_ort(
    session: ort.InferenceSession,
    input_ids: np.ndarray,
    cache: dict[str, np.ndarray],
    past_length: int,
) -> dict[str, np.ndarray]:
    sequence_length = input_ids.shape[1]
    candidates = {
        "input_ids": input_ids,
        "attention_mask": np.ones((1, past_length + sequence_length), dtype=np.int64),
        "position_ids": np.arange(past_length, past_length + sequence_length, dtype=np.int64)[
            None
        ],
        **cache,
    }
    input_names = {value.name for value in session.get_inputs()}
    output_names = [value.name for value in session.get_outputs()]
    feeds = {name: value for name, value in candidates.items() if name in input_names}
    return dict(zip(output_names, session.run(output_names, feeds), strict=True))


def _next_cache(outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        name.replace("present.", "past_key_values."): value
        for name, value in outputs.items()
        if name.startswith("present.")
    }


def _empty_promoted_state(
    session: ort.InferenceSession, *, batch_size: int
) -> dict[str, np.ndarray]:
    state: dict[str, np.ndarray] = {}
    for value in session.get_inputs():
        if value.name.endswith((".key", ".value")):
            state[value.name] = np.empty(
                [batch_size, value.shape[1], 0, value.shape[3]], dtype=np.float32
            )
        elif value.name.endswith(".conv_state"):
            state[value.name] = np.zeros(
                [batch_size, value.shape[1], value.shape[2]], dtype=np.float32
            )
    return state


def _run_promoted_ort(
    session: ort.InferenceSession,
    input_ids: np.ndarray,
    state: dict[str, np.ndarray],
    past_length: int,
) -> dict[str, np.ndarray]:
    sequence_length = input_ids.shape[1]
    candidates = {
        "input_ids": input_ids,
        "attention_mask": np.ones(
            (input_ids.shape[0], past_length + sequence_length), dtype=np.int64
        ),
        "position_ids": np.broadcast_to(
            np.arange(past_length, past_length + sequence_length, dtype=np.int64),
            input_ids.shape,
        ),
        **state,
    }
    input_names = {value.name for value in session.get_inputs()}
    output_names = [value.name for value in session.get_outputs()]
    feeds = {name: value for name, value in candidates.items() if name in input_names}
    return dict(zip(output_names, session.run(output_names, feeds), strict=True))


def _assert_replay_rollback_and_reorder(
    session: ort.InferenceSession,
    prompt_ids: np.ndarray,
) -> None:
    prefill = _run_promoted_ort(
        session,
        prompt_ids,
        _empty_promoted_state(session, batch_size=1),
        0,
    )
    snapshot = {name: value.copy() for name, value in _next_cache(prefill).items()}
    token = np.asarray([[int(prefill["logits"][0, -1].argmax())]], dtype=np.int64)
    first = _run_promoted_ort(session, token, snapshot, prompt_ids.shape[1])
    second = _run_promoted_ort(session, token, snapshot, prompt_ids.shape[1])
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])

    next_token = np.asarray([[int(first["logits"][0, -1].argmax())]], dtype=np.int64)
    _run_promoted_ort(session, next_token, _next_cache(first), prompt_ids.shape[1] + 1)
    rolled_back = _run_promoted_ort(session, token, snapshot, prompt_ids.shape[1])
    for name in first:
        np.testing.assert_array_equal(first[name], rolled_back[name])

    alternate_ids = prompt_ids.copy()
    alternate_ids[0, -1] += 1
    batched_ids = np.concatenate([prompt_ids, alternate_ids], axis=0)
    batched = _run_promoted_ort(
        session,
        batched_ids,
        _empty_promoted_state(session, batch_size=2),
        0,
    )
    batched_state = _next_cache(batched)
    batched_tokens = batched["logits"][:, -1].argmax(axis=-1).astype(np.int64)[:, None]
    original = _run_promoted_ort(session, batched_tokens, batched_state, prompt_ids.shape[1])
    reordered = _run_promoted_ort(
        session,
        batched_tokens[::-1].copy(),
        {name: value[::-1].copy() for name, value in batched_state.items()},
        prompt_ids.shape[1],
    )
    for name in original:
        np.testing.assert_array_equal(original[name][::-1], reordered[name])


@pytest.mark.integration
@pytest.mark.integration_fast
@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_small_f16_gguf_cli_full_logit_and_generation_parity(
    case: _RuntimeCase, tmp_path: Path
) -> None:
    """Pinned GGUFs survive CLI save/reload and match HF through cached decode."""
    gguf_path = Path(
        hf_hub_download(
            repo_id=case.gguf_repository,
            revision=case.gguf_revision,
            filename=case.gguf_filename,
        )
    )
    assert gguf_path.stat().st_size == case.gguf_size
    assert _sha256(gguf_path) == case.gguf_sha256
    gguf_model = GGUFModel(gguf_path)
    qtypes = Counter(qtype.name for _, _, qtype, _ in gguf_model.tensor_items_raw())
    assert dict(sorted(qtypes.items())) == case.tensor_qtypes
    for filename, size, sha256 in case.tokenizer_assets:
        asset_path = Path(
            hf_hub_download(
                repo_id=case.tokenizer_repository,
                revision=case.tokenizer_revision,
                filename=filename,
            )
        )
        assert asset_path.stat().st_size == size
        assert _sha256(asset_path) == sha256

    output_dir = tmp_path / case.name
    captured: list[ModelPackage] = []
    original_save = ModelPackage.save

    def capture_save(package: ModelPackage, *args: object, **kwargs: object) -> None:
        captured.append(package)
        original_save(package, *args, **kwargs)

    with mock.patch.object(ModelPackage, "save", capture_save):
        options = [
            "build-gguf",
            str(gguf_path),
            "--output",
            str(output_dir),
            "--dtype",
            "f32",
            "--execution-provider",
            "cpu",
        ]
        if case.tokenizer_identity_exact:
            options.extend(
                [
                    "--runtime",
                    "onnx-genai",
                    "--runtime-version",
                    "1.29.0",
                    "--tokenizer-repository",
                    case.tokenizer_repository,
                    "--tokenizer-revision",
                    case.tokenizer_revision,
                    "--local-files-only",
                ]
            )
        main(options)

    assert len(captured) == 1
    route = json.loads(captured[0].gguf_import_route)
    _assert_stable_import_route(route, case)
    source_report = captured[0].gguf_quantization_report
    assert source_report is not None
    assert source_report.source_fidelity is True
    assert source_report.storage_quantized is False
    assert source_report.target_storage_format == "float"
    package = ModelPackage.load(output_dir)
    assert tuple(package) == ("model",)
    assert package.gguf_quantization_report == source_report
    assert (
        GGUFQuantizationReport.read_json(output_dir / "quantization_report.json")
        == source_report
    )
    assert {"model.onnx", "model.onnx.data", "quantization_report.json"} <= {
        path.name for path in output_dir.iterdir()
    }

    tokenizer = AutoTokenizer.from_pretrained(
        case.reference_repository, revision=case.reference_revision
    )
    if case.tokenizer_identity_exact:
        packaged_tokenizer = AutoTokenizer.from_pretrained(output_dir, local_files_only=True)
        assert packaged_tokenizer(case.prompt).input_ids == tokenizer(case.prompt).input_ids
    else:
        rejected_package = tmp_path / f"{case.name}-runtime"
        with pytest.raises(ValueError, match="No unique GGUF runtime evidence"):
            write_gguf_runtime_package(
                captured[0],
                gguf_path,
                rejected_package,
                runtime="onnx-genai",
                runtime_version="1.29.0",
                tokenizer_repository=case.tokenizer_repository,
                tokenizer_revision=case.tokenizer_revision,
                local_files_only=True,
            )
        assert not rejected_package.exists()
        source = GGUFTokenizerSource(
            repository=case.tokenizer_repository,
            revision=case.tokenizer_revision,
            metadata_sha256=case.tokenizer_metadata_sha256,
            assets=tuple(GGUFTokenizerAsset(*asset) for asset in case.tokenizer_assets),
        )
        rejected_output = tmp_path / f"{case.name}-tokenizer"
        assert (
            inspect_gguf_tokenizer(gguf_model.metadata, require_complete=True).metadata_sha256
            == case.tokenizer_metadata_sha256
        )
        with pytest.raises(ValueError, match="pad_token id differs from GGUF"):
            materialize_gguf_tokenizer(
                gguf_path,
                rejected_output,
                source=source,
                metadata=gguf_model.metadata,
                local_files_only=True,
            )
        assert not rejected_output.exists()
    reference = AutoModelForCausalLM.from_pretrained(
        case.reference_repository,
        revision=case.reference_revision,
        dtype=torch.float32,
    ).eval()
    prompt_ids = tokenizer(case.prompt, return_tensors="pt").input_ids
    reference_logits: list[np.ndarray] = []
    with torch.no_grad():
        reference_output = reference(prompt_ids, use_cache=True)
    reference_logits.append(reference_output.logits.numpy().copy())
    reference_cache = reference_output.past_key_values
    reference_generated: list[int] = []
    for _ in case.generated_tokens:
        token = int(reference_output.logits[0, -1].argmax())
        reference_generated.append(token)
        token_ids = torch.tensor([[token]], dtype=torch.int64)
        with torch.no_grad():
            reference_output = reference(
                token_ids,
                past_key_values=reference_cache,
                use_cache=True,
            )
        reference_cache = reference_output.past_key_values
        reference_logits.append(reference_output.logits.numpy().copy())
    assert reference_generated == list(case.generated_tokens)
    captured.clear()
    del package, gguf_model, reference, reference_output, reference_cache
    gc.collect()

    input_ids = prompt_ids.numpy()
    session = ort.InferenceSession(
        str(output_dir / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    assert {value.name for value in session.get_inputs()} >= {
        "input_ids",
        "attention_mask",
    }
    assert {value.name for value in session.get_outputs()} >= {"logits"}
    ort_output = _run_ort(session, input_ids, _empty_cache(session), 0)
    ort_logits = ort_output["logits"]
    np.testing.assert_allclose(ort_logits, reference_logits[0], rtol=1e-4, atol=2e-4)

    cache = _next_cache(ort_output)
    generated: list[int] = []
    for step in range(len(case.generated_tokens)):
        token = int(ort_logits[0, -1].argmax())
        generated.append(token)
        token_ids = np.array([[token]], dtype=np.int64)
        ort_output = _run_ort(session, token_ids, cache, input_ids.shape[1] + step)
        cache = _next_cache(ort_output)
        ort_logits = ort_output["logits"]
        np.testing.assert_allclose(
            ort_logits, reference_logits[step + 1], rtol=1e-4, atol=2e-4
        )

    expected = np.asarray(case.generated_tokens, dtype=np.int64)
    actual = np.asarray(generated, dtype=np.int64)
    assert len(actual) == len(expected)
    np.testing.assert_array_equal(actual, expected)

    repeated = []
    repeated_output = _run_ort(session, input_ids, _empty_cache(session), 0)
    cache = _next_cache(repeated_output)
    ort_logits = repeated_output["logits"]
    for step in range(len(expected)):
        token = int(ort_logits[0, -1].argmax())
        repeated.append(token)
        token_ids = np.array([[token]], dtype=np.int64)
        output = _run_ort(session, token_ids, cache, input_ids.shape[1] + step)
        cache = _next_cache(output)
        ort_logits = output["logits"]
    assert len(repeated) == len(expected)
    np.testing.assert_array_equal(repeated, expected)


@pytest.mark.integration
@pytest.mark.integration_fast
def test_smollm_q4_k_m_target_storage_and_explicit_float_fidelity(
    tmp_path: Path,
) -> None:
    """Lossy INT4 storage is reported honestly; explicit float retains full parity."""
    case = _Q4_K_M_CASE
    gguf_path = Path(
        hf_hub_download(
            repo_id=case.gguf_repository,
            revision=case.gguf_revision,
            filename=case.gguf_filename,
        )
    )
    assert gguf_path.stat().st_size == case.gguf_size
    assert _sha256(gguf_path) == case.gguf_sha256
    gguf_model = GGUFModel(gguf_path)
    qtypes = Counter(qtype.name for _, _, qtype, _ in gguf_model.tensor_items_raw())
    assert dict(sorted(qtypes.items())) == case.tensor_qtypes
    for filename, size, sha256 in case.tokenizer_assets:
        asset_path = Path(
            hf_hub_download(
                repo_id=case.tokenizer_repository,
                revision=case.tokenizer_revision,
                filename=filename,
            )
        )
        assert asset_path.stat().st_size == size
        assert _sha256(asset_path) == sha256

    quantized = build_from_gguf(gguf_path)
    quantized_report = quantized.gguf_quantization_report
    assert quantized_report is not None
    assert quantized_report.storage_quantized is True
    assert quantized_report.source_fidelity is False
    assert "INT4 affine block-32" in quantized_report.target_storage_format
    assert {
        record.qtype
        for record in quantized_report.tensor_records
        if record.disposition is QuantizationDisposition.LOSSY_REQUANTIZE
    } >= {"Q4_K", "Q5_0", "Q6_K", "Q8_0"}
    quantized_dir = tmp_path / f"{case.name}-int4-target"
    quantized.save(quantized_dir, progress_bar=False)
    quantized_reloaded = ModelPackage.load(quantized_dir)
    assert quantized_reloaded.gguf_quantization_report == quantized_report
    assert (
        GGUFQuantizationReport.read_json(quantized_dir / "quantization_report.json")
        == quantized_report
    )
    assert any(node.op_type == "MatMulNBits" for node in quantized["model"].graph)

    output_dir = tmp_path / case.name
    captured: list[ModelPackage] = []
    original_save = ModelPackage.save

    def capture_save(package: ModelPackage, *args: object, **kwargs: object) -> None:
        captured.append(package)
        original_save(package, *args, **kwargs)

    with mock.patch.object(ModelPackage, "save", capture_save):
        main(
            [
                "build-gguf",
                str(gguf_path),
                "--output",
                str(output_dir),
                "--dequantize",
                "--dtype",
                "f32",
                "--execution-provider",
                "cpu",
            ]
        )

    assert len(captured) == 1
    package = captured[0]
    route = json.loads(package.gguf_import_route)
    _assert_stable_import_route(route, case)
    assert all(node.op_type != "MatMulNBits" for node in package["model"].graph)
    float_report = package.gguf_quantization_report
    assert float_report is not None
    assert float_report.source_fidelity is False
    assert float_report.storage_quantized is False
    assert float_report.target_storage_format == "float"

    reloaded = ModelPackage.load(output_dir)
    assert tuple(reloaded) == ("model",)
    assert reloaded.gguf_quantization_report == float_report
    assert (
        GGUFQuantizationReport.read_json(output_dir / "quantization_report.json")
        == float_report
    )
    rejected_package = tmp_path / f"{case.name}-runtime"
    with pytest.raises(ValueError, match="No unique GGUF runtime evidence"):
        write_gguf_runtime_package(
            package,
            gguf_path,
            rejected_package,
            runtime="onnx-genai",
            runtime_version="1.29.0",
            tokenizer_repository=case.tokenizer_repository,
            tokenizer_revision=case.tokenizer_revision,
            local_files_only=True,
        )
    assert not rejected_package.exists()
    source = GGUFTokenizerSource(
        repository=case.tokenizer_repository,
        revision=case.tokenizer_revision,
        metadata_sha256=case.tokenizer_metadata_sha256,
        assets=tuple(GGUFTokenizerAsset(*asset) for asset in case.tokenizer_assets),
    )
    rejected_output = tmp_path / f"{case.name}-tokenizer"
    assert (
        inspect_gguf_tokenizer(gguf_model.metadata, require_complete=True).metadata_sha256
        == case.tokenizer_metadata_sha256
    )
    with pytest.raises(ValueError, match="pad_token id differs from GGUF"):
        materialize_gguf_tokenizer(
            gguf_path,
            rejected_output,
            source=source,
            metadata=gguf_model.metadata,
            local_files_only=True,
        )
    assert not rejected_output.exists()
    tokenizer = AutoTokenizer.from_pretrained(
        case.reference_repository, revision=case.reference_revision
    )
    reference = AutoModelForCausalLM.from_pretrained(
        case.gguf_repository,
        revision=case.gguf_revision,
        gguf_file=case.gguf_filename,
        dtype=torch.float32,
    ).eval()
    prompt_ids = tokenizer(case.prompt, return_tensors="pt").input_ids
    reference_logits: list[np.ndarray] = []
    with torch.no_grad():
        reference_output = reference(prompt_ids, use_cache=True)
    reference_logits.append(reference_output.logits.numpy().copy())
    reference_cache = reference_output.past_key_values
    reference_generated: list[int] = []
    for _ in case.generated_tokens:
        token = int(reference_output.logits[0, -1].argmax())
        reference_generated.append(token)
        token_ids = torch.tensor([[token]], dtype=torch.int64)
        with torch.no_grad():
            reference_output = reference(
                token_ids,
                past_key_values=reference_cache,
                use_cache=True,
            )
        reference_cache = reference_output.past_key_values
        reference_logits.append(reference_output.logits.numpy().copy())
    assert reference_generated == list(case.generated_tokens)
    captured.clear()
    del package, reloaded, gguf_model, reference, reference_output, reference_cache
    gc.collect()

    input_ids = prompt_ids.numpy()
    session = ort.InferenceSession(
        str(output_dir / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    ort_output = _run_ort(session, input_ids, _empty_cache(session), 0)
    ort_logits = ort_output["logits"]
    np.testing.assert_allclose(ort_logits, reference_logits[0], rtol=1e-4, atol=2e-4)

    cache = _next_cache(ort_output)
    generated: list[int] = []
    for step in range(len(case.generated_tokens)):
        token = int(ort_logits[0, -1].argmax())
        generated.append(token)
        token_ids = np.array([[token]], dtype=np.int64)
        ort_output = _run_ort(session, token_ids, cache, input_ids.shape[1] + step)
        cache = _next_cache(ort_output)
        ort_logits = ort_output["logits"]
        np.testing.assert_allclose(
            ort_logits, reference_logits[step + 1], rtol=1e-4, atol=2e-4
        )

    np.testing.assert_array_equal(generated, case.generated_tokens)


@pytest.mark.integration
@pytest.mark.integration_slow
@pytest.mark.ort_genai_real
def test_smollm_generic_ort_genai_generation(
    tmp_path: Path,
    isolated_hf_cache: Path,
) -> None:
    """The one evidenced GGUF route loads through ORT GenAI's generic decoder."""
    import onnxruntime_genai as ort_genai

    case = _CASES[0]
    case_yaml = Path("testdata/cases/causal-lm/smollm-135m-gguf-f16.yaml")
    metadata = yaml.safe_load(case_yaml.read_text(encoding="utf-8"))["ort_genai"]
    installed_version = _installed_ort_genai_version()
    expected_version = os.environ.get("MOBIUS_EXPECTED_ORT_GENAI_VERSION")
    if expected_version:
        assert installed_version == expected_version
    assert installed_version in metadata["runtime_versions"]
    assert metadata["released_capabilities"][installed_version] == {
        "generic_decoder": True,
        "state_groups": False,
    }
    download_specs = [
        (
            case.gguf_repository,
            case.gguf_revision,
            case.gguf_filename,
            case.gguf_size,
        ),
        *(
            (case.tokenizer_repository, case.tokenizer_revision, filename, size)
            for filename, size, _ in case.tokenizer_assets
        ),
    ]
    remote_download_bytes = 0
    for repository, revision, filename, expected_size in download_specs:
        remote = get_hf_file_metadata(
            hf_hub_url(repository, filename, revision=revision),
            timeout=30,
        )
        assert remote.commit_hash == revision
        assert remote.size == expected_size
        remote_download_bytes += remote.size
    assert remote_download_bytes <= metadata["max_download_bytes"]
    assert isolated_hf_cache.exists()
    gguf_path = Path(
        hf_hub_download(
            repo_id=case.gguf_repository,
            revision=case.gguf_revision,
            filename=case.gguf_filename,
        )
    )
    assert gguf_path.stat().st_size == case.gguf_size
    assert _sha256(gguf_path) == case.gguf_sha256
    for filename, size, sha256 in case.tokenizer_assets:
        asset_path = Path(
            hf_hub_download(
                repo_id=case.tokenizer_repository,
                revision=case.tokenizer_revision,
                filename=filename,
            )
        )
        assert asset_path.stat().st_size == size
        assert _sha256(asset_path) == sha256
    output_dir = tmp_path / "smollm-ort-genai"
    main(
        [
            "build-gguf",
            str(gguf_path),
            "--output",
            str(output_dir),
            "--dtype",
            "f32",
            "--execution-provider",
            "cpu",
            "--runtime",
            "ort-genai",
            "--runtime-version",
            installed_version,
            "--tokenizer-repository",
            case.tokenizer_repository,
            "--tokenizer-revision",
            case.tokenizer_revision,
            "--local-files-only",
        ]
    )
    config = json.loads((output_dir / "genai_config.json").read_text())
    assert config["model"]["type"] == metadata["model_type"]
    assert "state_groups" not in config["model"]["decoder"]

    model = ort_genai.Model(str(output_dir))
    tokenizer = ort_genai.Tokenizer(model)
    prompt_ids = tokenizer.encode(case.prompt)
    params = ort_genai.GeneratorParams(model)
    params.set_search_options(
        max_length=len(prompt_ids) + len(case.generated_tokens),
        do_sample=False,
    )
    generator = ort_genai.Generator(model, params)
    generator.append_tokens(prompt_ids)

    generated: list[int] = []
    for _ in case.generated_tokens:
        generator.generate_next_token()
        generated.append(generator.get_next_tokens()[0])

    assert len(generated) == len(case.generated_tokens)
    assert generated == list(case.generated_tokens)
    del generator, params, tokenizer, model
    gc.collect()


@pytest.mark.integration
@pytest.mark.integration_slow
@pytest.mark.ort_genai_real
@pytest.mark.parametrize("case", _PROMOTED_RUNTIME_CASES, ids=lambda case: case.name)
def test_promoted_gguf_full_runtime_evidence(
    case: _PromotedRuntimeCase,
    tmp_path: Path,
    isolated_hf_cache: Path,
) -> None:
    """Exact GGUF weights prove logits, state semantics, publication, and OGA decode."""
    import onnxruntime_genai as ort_genai

    evidence = runtime_evidence(case.evidence_id)
    case_yaml = Path(f"testdata/cases/causal-lm/{case.name}.yaml")
    enrollment = yaml.safe_load(case_yaml.read_text(encoding="utf-8"))["ort_genai"]
    installed_version = _installed_ort_genai_version()
    assert installed_version == evidence.runtime_version == "0.15.2"
    assert ort.__version__ == evidence.onnxruntime_version == "1.29.0"
    assert enrollment["runtime_evidence_id"] == case.evidence_id
    assert enrollment["runtime_versions"] == [installed_version]

    download_specs = [
        (
            evidence.repository,
            evidence.revision,
            evidence.filename,
            evidence.size,
        ),
        *(
            (
                evidence.tokenizer_repository,
                evidence.tokenizer_revision,
                filename,
                size,
            )
            for filename, size, _ in evidence.tokenizer_assets
        ),
    ]
    remote_download_bytes = 0
    for repository, revision, filename, expected_size in download_specs:
        remote = get_hf_file_metadata(
            hf_hub_url(repository, filename, revision=revision),
            timeout=30,
        )
        assert remote.commit_hash == revision
        assert remote.size == expected_size
        remote_download_bytes += remote.size
    assert evidence.size <= 16 * 2**30
    assert remote_download_bytes <= enrollment["max_download_bytes"]
    assert isolated_hf_cache.exists()
    assert shutil.disk_usage(isolated_hf_cache).free >= 2 * evidence.size

    cached_gguf = Path(
        hf_hub_download(
            repo_id=evidence.repository,
            revision=evidence.revision,
            filename=evidence.filename,
        )
    )
    assert cached_gguf.stat().st_size == evidence.size
    assert _sha256(cached_gguf) == evidence.lfs_sha256
    gguf_path = tmp_path / evidence.filename
    shutil.copyfile(cached_gguf, gguf_path)
    gguf_model = GGUFModel(gguf_path)
    qtypes = Counter(qtype.name for _, _, qtype, _ in gguf_model.tensor_items_raw())
    assert len(gguf_model.reader_tensors()) == evidence.tensor_count
    assert tuple(sorted(qtypes.items())) == evidence.tensor_qtypes
    del gguf_model
    gc.collect()
    for filename, size, sha256 in evidence.tokenizer_assets:
        asset_path = Path(
            hf_hub_download(
                repo_id=evidence.tokenizer_repository,
                revision=evidence.tokenizer_revision,
                filename=filename,
            )
        )
        assert asset_path.stat().st_size == size
        assert _sha256(asset_path) == sha256

    output_dir = tmp_path / "runtime-package"
    assert shutil.disk_usage(tmp_path).free >= 2 * evidence.size
    captured: list[ModelPackage] = []
    original_save = ModelPackage.save

    def capture_save(package: ModelPackage, *args: object, **kwargs: object) -> None:
        captured.append(package)
        original_save(package, *args, **kwargs)

    with mock.patch.object(ModelPackage, "save", capture_save):
        main(
            [
                "build-gguf",
                str(gguf_path),
                "--output",
                str(output_dir),
                "--dtype",
                "f32",
                "--execution-provider",
                "cpu",
                "--runtime",
                "ort-genai",
                "--runtime-version",
                installed_version,
                "--tokenizer-repository",
                evidence.tokenizer_repository,
                "--tokenizer-revision",
                evidence.tokenizer_revision,
                "--local-files-only",
            ]
        )
    assert len(captured) == 1
    assert captured[0].gguf_import_route == evidence.import_route
    source_report = captured[0].gguf_quantization_report
    assert source_report is not None
    if dict(evidence.tensor_qtypes).get("Q8_0", 0):
        assert source_report.storage_quantized is True
        assert source_report.source_fidelity is True
        graph = captured[0]["model"].graph
        op_types = Counter(node.op_type for node in graph)
        assert op_types["MatMulNBits"] > 0
        assert op_types["GatherBlockQuantized"] == 1
        assert op_types["BlockQuantizedMatMul"] == 0
        initializer_names = set(graph.initializers)
        assert "model.embed_tokens.scales" in initializer_names
        assert "lm_head.scales" in initializer_names
    package = ModelPackage.load(output_dir)
    assert tuple(package) == ("model",)
    assert package.gguf_quantization_report == source_report
    assert (
        GGUFQuantizationReport.read_json(output_dir / "quantization_report.json")
        == source_report
    )
    runtime_identity = gguf_graph_package_identity(output_dir)
    assert runtime_identity.files == evidence.runtime_package_files
    assert runtime_identity.sha256 == evidence.runtime_package_sha256
    graph_dir = tmp_path / "graph-identity"
    graph_dir.mkdir()
    for filename in evidence.graph_files:
        os.link(output_dir / filename, graph_dir / filename)
    graph_identity = gguf_graph_package_identity(graph_dir)
    assert graph_identity.files == evidence.graph_files
    assert graph_identity.sha256 == evidence.graph_sha256
    captured.clear()
    del package
    gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(
        evidence.tokenizer_repository,
        revision=evidence.tokenizer_revision,
    )
    packaged_tokenizer = AutoTokenizer.from_pretrained(output_dir, local_files_only=True)
    prompt_ids = tokenizer(case.prompt, return_tensors="pt").input_ids
    assert packaged_tokenizer(case.prompt).input_ids == prompt_ids.tolist()[0]
    reference = AutoModelForCausalLM.from_pretrained(
        evidence.repository,
        revision=evidence.revision,
        gguf_file=evidence.filename,
        dtype=torch.float32,
    ).eval()
    input_ids = prompt_ids.numpy()
    reference_logits: list[np.ndarray] = []
    with torch.no_grad():
        reference_output = reference(prompt_ids, use_cache=True)
    reference_logits.append(reference_output.logits.numpy().copy())
    reference_state = reference_output.past_key_values
    reference_generated: list[int] = []
    for _ in case.generated_tokens:
        token = int(reference_output.logits[0, -1].argmax())
        reference_generated.append(token)
        token_ids = torch.tensor([[token]], dtype=torch.int64)
        with torch.no_grad():
            reference_output = reference(
                token_ids,
                past_key_values=reference_state,
                use_cache=True,
            )
        reference_state = reference_output.past_key_values
        reference_logits.append(reference_output.logits.numpy().copy())
    assert reference_generated == list(case.generated_tokens)
    del reference, reference_output, reference_state
    gc.collect()

    session = ort.InferenceSession(
        str(output_dir / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    ort_output = _run_promoted_ort(
        session,
        input_ids,
        _empty_promoted_state(session, batch_size=1),
        0,
    )
    np.testing.assert_allclose(
        ort_output["logits"],
        reference_logits[0],
        rtol=1e-4,
        atol=case.atol,
    )

    state = _next_cache(ort_output)
    generated: list[int] = []
    for step in range(len(case.generated_tokens)):
        token = int(ort_output["logits"][0, -1].argmax())
        generated.append(token)
        token_ids = np.asarray([[token]], dtype=np.int64)
        ort_output = _run_promoted_ort(session, token_ids, state, input_ids.shape[1] + step)
        state = _next_cache(ort_output)
        np.testing.assert_allclose(
            ort_output["logits"],
            reference_logits[step + 1],
            rtol=1e-4,
            atol=case.atol,
        )
    assert len(generated) == len(case.generated_tokens)
    assert generated == list(case.generated_tokens)
    _assert_replay_rollback_and_reorder(session, input_ids)
    del session, ort_output, state, reference_logits
    gc.collect()

    compatibility = json.loads(
        (output_dir / "runtime_compatibility.json").read_text(encoding="utf-8")
    )
    assert installed_version in compatibility["tested_versions"]
    model = ort_genai.Model(str(output_dir))
    oga_tokens = prompt_ids.tolist()[0]
    params = ort_genai.GeneratorParams(model)
    params.set_search_options(
        max_length=len(oga_tokens) + len(case.generated_tokens),
        do_sample=False,
    )
    generator = ort_genai.Generator(model, params)
    generator.append_tokens(oga_tokens)
    oga_generated: list[int] = []
    for _ in case.generated_tokens:
        generator.generate_next_token()
        oga_generated.append(int(generator.get_next_tokens()[0]))
    assert len(oga_generated) == len(case.generated_tokens)
    assert oga_generated == list(case.generated_tokens)
    del generator, params, model
    gc.collect()

    shutil.rmtree(output_dir)
    gguf_path.unlink()
