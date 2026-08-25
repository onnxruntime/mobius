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

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import numpy as np
import onnxruntime as ort
import pytest
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from mobius import ModelPackage
from mobius.__main__ import main
from mobius.integrations.gguf import (
    GGUFTokenizerAsset,
    GGUFTokenizerSource,
    materialize_gguf_tokenizer,
    write_gguf_runtime_package,
)
from mobius.integrations.gguf._reader import GGUFModel
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
        config_sha256="134f95e6a635d978737d712ed61ac8959acebdf080eafae838cf97f12c416430",
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
        config_sha256="c62123baf4e95656cdc9f5b798c14319bbaafec594526c462b10555f561969f9",
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    assert route == {
        "architecture": "llama",
        "config_sha256": case.config_sha256,
        "execution_provider": "cpu",
        "model_type": "llama",
        "module_type": "llama",
        "preserve_quantization": False,
        "registry_import": {
            "config_key_map": None,
            "config_postprocessor": None,
            "llama_qk_permute": True,
            "offset_norm": False,
            "required_metadata": [],
            "rope_interleave": False,
            "tensor_processor": "llama",
            "v_head_reorder": False,
            "vlm_builder": None,
        },
        "route_schema": 1,
        "static_cache": False,
        "task": {"class": "builtins.str", "state": "text-generation"},
        "tensor_map_recipe": ["llama"],
    }
    package = ModelPackage.load(output_dir)
    assert tuple(package) == ("model",)
    assert {"model.onnx", "model.onnx.data"} <= {path.name for path in output_dir.iterdir()}
    session = ort.InferenceSession(
        str(output_dir / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    assert {value.name for value in session.get_inputs()} >= {
        "input_ids",
        "attention_mask",
    }
    assert {value.name for value in session.get_outputs()} >= {"logits"}

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
    with torch.no_grad():
        reference_output = reference(prompt_ids, use_cache=True)

    input_ids = prompt_ids.numpy()
    ort_output = _run_ort(session, input_ids, _empty_cache(session), 0)
    ort_logits = ort_output["logits"]
    reference_logits = reference_output.logits.numpy()
    np.testing.assert_allclose(ort_logits, reference_logits, rtol=1e-4, atol=2e-4)

    cache = _next_cache(ort_output)
    reference_cache = reference_output.past_key_values
    generated: list[int] = []
    for step in range(len(case.generated_tokens)):
        token = int(ort_logits[0, -1].argmax())
        generated.append(token)
        token_ids = np.array([[token]], dtype=np.int64)
        ort_output = _run_ort(session, token_ids, cache, input_ids.shape[1] + step)
        cache = _next_cache(ort_output)
        with torch.no_grad():
            reference_output = reference(
                torch.from_numpy(token_ids),
                past_key_values=reference_cache,
                use_cache=True,
            )
        reference_cache = reference_output.past_key_values
        ort_logits = ort_output["logits"]
        reference_logits = reference_output.logits.numpy()
        np.testing.assert_allclose(ort_logits, reference_logits, rtol=1e-4, atol=2e-4)

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
