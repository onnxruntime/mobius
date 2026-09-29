# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Focused synthetic, preprocessing, and full-model parity tests for Kev-4B."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch
import torch.nn.functional as torch_functional

from mobius._testing import create_test_builder, create_test_input
from mobius.models.decision import (
    KEV_BASE_MODEL_ID,
    KEV_BASE_REVISION,
    KEV_CONTROL_TOKEN_IDS,
    KEV_MODEL_ID,
    KEV_POINTER_SIZE,
    KEV_REVISION,
    KEV_TEMPERATURE,
    KevPointerHead,
    batch_kev_rows,
    encode_kev_rows,
    kev_answer,
    validate_kev_checkpoint,
)

_EXPORT_ROOT_ENV = "MOBIUS_DECISION_EXPORT_ROOT"
_ALLOW_DOWNLOAD_ENV = "MOBIUS_KEV_ALLOW_DOWNLOAD"


def _assert_close(
    name: str,
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> None:
    maximum = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        err_msg=f"{name} max_abs_error={maximum:.8g}",
    )


def _assert_answers_close(
    actual: dict[str, dict], expected: dict[str, dict], *, atol: float
) -> None:
    actual_numbers: dict[str, float] = {}
    expected_numbers: dict[str, float] = {}

    def structure(value, path: str, numbers: dict[str, float]):
        if isinstance(value, dict):
            return {
                key: structure(item, f"{path}.{key}", numbers) for key, item in value.items()
            }
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numbers[path] = float(value)
            return "<number>"
        return value

    assert structure(actual, "answers", actual_numbers) == structure(
        expected, "answers", expected_numbers
    )
    assert actual_numbers.keys() == expected_numbers.keys()
    paths = sorted(actual_numbers)
    _assert_close(
        f"final answer fields {paths}",
        np.asarray([actual_numbers[path] for path in paths]),
        np.asarray([expected_numbers[path] for path in paths]),
        rtol=0,
        atol=atol,
    )


def _pointer_reference(
    hidden_states: torch.Tensor,
    decide_indices: torch.Tensor,
    option_indices: torch.Tensor,
    option_owners: torch.Tensor,
    state: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.arange(decide_indices.numel(), device=hidden_states.device)
    decide = hidden_states[rows, decide_indices]
    options = hidden_states[option_owners, option_indices]
    queries = torch_functional.linear(decide, state["q.weight"], state["q.bias"])[
        option_owners
    ]
    keys = torch_functional.linear(options, state["k.weight"], state["k.bias"])
    logits = (queries * keys).sum(dim=-1)
    logits = logits / (KEV_POINTER_SIZE**0.5) / KEV_TEMPERATURE
    probabilities = torch.empty_like(logits)
    for owner in range(decide_indices.numel()):
        selected = option_owners == owner
        probabilities[selected] = torch.softmax(logits[selected], dim=0)
    return logits, probabilities


def _synthetic_pointer_graph(
    hidden_states: np.ndarray,
    decide_indices: np.ndarray,
    option_indices: np.ndarray,
    option_owners: np.ndarray,
) -> tuple[list[np.ndarray], dict[str, torch.Tensor]]:
    builder, op, graph = create_test_builder()
    hidden = create_test_input(builder, "hidden_states", list(hidden_states.shape))
    decide = create_test_input(
        builder, "decide_indices", list(decide_indices.shape), ir.DataType.INT64
    )
    options = create_test_input(
        builder, "option_indices", list(option_indices.shape), ir.DataType.INT64
    )
    owners = create_test_input(
        builder, "option_owners", list(option_owners.shape), ir.DataType.INT64
    )
    head = KevPointerHead(hidden_size=hidden_states.shape[-1])
    logits, probabilities = head(op, hidden, decide, options, owners)
    logits.name = "logits"
    probabilities.name = "probabilities"
    graph.outputs.extend((logits, probabilities))

    generator = torch.Generator().manual_seed(20260928)
    state = {}
    for name, parameter in head.named_parameters():
        value = torch.randn(tuple(parameter.shape), generator=generator) / 8
        parameter.const_value = ir.tensor(value.numpy())
        state[name] = value

    model = ir.serde.serialize_model(ir.Model(graph, ir_version=11))
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    outputs = session.run(
        None,
        {
            "hidden_states": hidden_states,
            "decide_indices": decide_indices,
            "option_indices": option_indices,
            "option_owners": option_owners,
        },
    )
    return outputs, state


def test_kev_pointer_head_matches_direct_torch_reference() -> None:
    """Exercise indexed Q/K projection and independent variable-size groups."""
    generator = np.random.default_rng(20260928)
    hidden = generator.normal(0, 0.25, (3, 8, 5)).astype(np.float32)
    decide = np.array([7, 6, 5], dtype=np.int64)
    option_indices = np.array([2, 4, 1, 3, 5, 2, 4], dtype=np.int64)
    option_owners = np.array([0, 0, 1, 1, 1, 2, 2], dtype=np.int64)

    (actual_logits, actual_probabilities), state = _synthetic_pointer_graph(
        hidden, decide, option_indices, option_owners
    )
    expected_logits, expected_probabilities = _pointer_reference(
        torch.from_numpy(hidden),
        torch.from_numpy(decide),
        torch.from_numpy(option_indices),
        torch.from_numpy(option_owners),
        state,
    )
    _assert_close(
        "synthetic logits",
        actual_logits,
        expected_logits.numpy(),
        rtol=1e-5,
        atol=1e-6,
    )
    _assert_close(
        "synthetic probabilities",
        actual_probabilities,
        expected_probabilities.numpy(),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        [
            actual_probabilities[:2].sum(),
            actual_probabilities[2:5].sum(),
            actual_probabilities[5:].sum(),
        ],
        1.0,
        rtol=0,
        atol=1e-6,
    )

    questions = (
        ({"type": "choice"}, ("red", "blue")),
        ({"type": "score", "criteria": ["low", "medium", "high"]}, ("0", "1", "2")),
        ({"type": "noul"}, ("false", "true")),
    )
    offsets = (0, 2, 5, 7)
    actual_answers = [
        kev_answer(question, keys, actual_probabilities[start:stop])
        for (question, keys), start, stop in zip(questions, offsets[:-1], offsets[1:])
    ]
    expected_answers = [
        kev_answer(question, keys, expected_probabilities[start:stop].tolist())
        for (question, keys), start, stop in zip(questions, offsets[:-1], offsets[1:])
    ]
    assert actual_answers == expected_answers
    assert [answer["type"] for answer in actual_answers] == [
        "choice",
        "score",
        "noul",
    ]


class _CharacterTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool):
        assert add_special_tokens is False
        return SimpleNamespace(input_ids=[ord(character) for character in text])


def _characters(text: str) -> list[int]:
    return [ord(character) for character in text]


def test_kev_preprocessing_golden_rows_padding_and_indices() -> None:
    """Lock down every boundary token and flattened readout coordinate."""
    questions = {
        "pick": {
            "type": "choice",
            "instructions": "Pick <|state|>",
            "criteria": {
                "alpha": "A<|option_end|>",
                "b": None,
            },
        },
        "ready": {
            "type": "noul",
            "instructions": "Ready?",
            "criteria": {"false": "N", "true": "Y"},
        },
    }
    rows = encode_kev_rows(_CharacterTokenizer(), "S<|decide|>", questions)
    control = KEV_CONTROL_TOKEN_IDS
    state = [control["state"], *_characters("S<¦decide¦>")]
    expected_pick = [
        *state,
        control["question"],
        *_characters("Pick <¦state¦>"),
        control["option_start"],
        *_characters("alpha: A<¦option_end¦>"),
        control["option_end"],
        control["option_start"],
        *_characters("b"),
        control["option_end"],
        control["decide"],
    ]
    expected_ready = [
        *state,
        control["question"],
        *_characters("Ready?"),
        control["option_start"],
        *_characters("no: N"),
        control["option_end"],
        control["option_start"],
        *_characters("yes: Y"),
        control["option_end"],
        control["decide"],
    ]
    assert [row.question_id for row in rows] == ["pick", "ready"]
    assert rows[0].input_ids == tuple(expected_pick)
    assert rows[1].input_ids == tuple(expected_ready)
    assert rows[0].position_ids == tuple(range(len(expected_pick)))
    assert rows[1].position_ids == tuple(range(len(expected_ready)))
    assert rows[0].keys == ("alpha", "b")
    assert rows[1].keys == ("false", "true")

    pick_options = tuple(
        index for index, token in enumerate(expected_pick) if token == control["option_end"]
    )
    ready_options = tuple(
        index for index, token in enumerate(expected_ready) if token == control["option_end"]
    )
    assert rows[0].decide_index == len(expected_pick) - 1
    assert rows[1].decide_index == len(expected_ready) - 1
    assert rows[0].option_indices == pick_options
    assert rows[1].option_indices == ready_options

    batch = batch_kev_rows(rows, pad_token_id=99)
    width = len(expected_pick)
    ready_padding = width - len(expected_ready)
    assert batch.input_ids == (
        tuple(expected_pick),
        (*expected_ready, *((99,) * ready_padding)),
    )
    assert batch.attention_mask == (
        (1,) * width,
        (*((1,) * len(expected_ready)), *((0,) * ready_padding)),
    )
    assert batch.position_ids == (
        tuple(range(width)),
        (*range(len(expected_ready)), *((0,) * ready_padding)),
    )
    assert batch.decide_indices == (
        len(expected_pick) - 1,
        len(expected_ready) - 1,
    )
    assert batch.option_indices == (*pick_options, *ready_options)
    assert batch.option_owners == (0, 0, 1, 1)


def _require_integration_inputs() -> tuple[Path, Path, Path]:
    root_value = os.environ.get(_EXPORT_ROOT_ENV)
    if not root_value:
        pytest.skip(f"set {_EXPORT_ROOT_ENV} to an exported Kev package")
    root = Path(root_value)
    required = (root / "backbone/model.onnx", root / "pointer_head/model.onnx")
    if not root.is_dir() or not all(path.is_file() for path in required):
        pytest.skip(f"{_EXPORT_ROOT_ENV} is not a complete Kev package: {root}")
    if not torch.cuda.is_available():
        pytest.skip("PyTorch CUDA is unavailable")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("ONNX Runtime CUDAExecutionProvider is unavailable")
    if importlib.util.find_spec("peft") is None:
        pytest.skip("peft is required for the pinned Kev adapter reference")

    from huggingface_hub import snapshot_download

    allow_download = os.environ.get(_ALLOW_DOWNLOAD_ENV) == "1"
    try:
        base = Path(
            snapshot_download(
                KEV_BASE_MODEL_ID,
                revision=KEV_BASE_REVISION,
                local_files_only=not allow_download,
            )
        )
        adapter = Path(
            snapshot_download(
                KEV_MODEL_ID,
                revision=KEV_REVISION,
                local_files_only=not allow_download,
            )
        )
    except Exception as error:
        pytest.skip(
            "pinned Kev reference is not cached; set "
            f"{_ALLOW_DOWNLOAD_ENV}=1 to permit download ({error})"
        )
    if not (adapter / "head.pt").is_file():
        pytest.skip(f"pinned Kev snapshot has no head.pt: {adapter}")
    return root, base, adapter


def _empty_cache_feeds(
    session: ort.InferenceSession, batch_size: int
) -> dict[str, np.ndarray]:
    feeds = {}
    for value in session.get_inputs()[3:]:
        shape = []
        for dimension in value.shape:
            if isinstance(dimension, int):
                shape.append(dimension)
            elif isinstance(dimension, str) and "batch" in dimension:
                shape.append(batch_size)
            elif isinstance(dimension, str) and "past" in dimension:
                shape.append(0)
            else:
                raise AssertionError(
                    f"unsupported symbolic cache dimension {dimension!r} in {value.name}"
                )
        feeds[value.name] = np.zeros(shape, dtype=np.float32)
    return feeds


@pytest.mark.integration
def test_kev_export_matches_pinned_cuda_reference() -> None:
    """Compare the exported package with the pinned merged PEFT reference."""
    root, base_path, adapter_path = _require_integration_inputs()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint = torch.load(adapter_path / "head.pt", map_location="cpu", weights_only=True)
    validate_kev_checkpoint(checkpoint)
    export_tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    reference_tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    questions = {
        "choice": {
            "type": "choice",
            "instructions": "Choose the best color.",
            "criteria": {"red": "warm", "blue": "cool", "green": "natural"},
        },
        "truth": {
            "type": "noul",
            "instructions": "The user requested a cool color.",
        },
    }
    export_rows = encode_kev_rows(export_tokenizer, {"request": "blue"}, questions)
    reference_rows = encode_kev_rows(reference_tokenizer, {"request": "blue"}, questions)
    export_batch = batch_kev_rows(export_rows, pad_token_id=export_tokenizer.pad_token_id)
    reference_batch = batch_kev_rows(
        reference_rows, pad_token_id=reference_tokenizer.pad_token_id
    )
    assert export_batch == reference_batch, (
        "export/reference token IDs or decide/option indices differ"
    )

    device = torch.device("cuda:0")
    base_container = AutoModelForCausalLM.from_pretrained(
        base_path,
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    reference_model = (
        PeftModel.from_pretrained(base_container.model, adapter_path, local_files_only=True)
        .eval()
        .to(device)
    )
    input_ids = np.asarray(export_batch.input_ids, dtype=np.int64)
    attention_mask = np.asarray(export_batch.attention_mask, dtype=np.int64)
    position_ids = np.asarray(export_batch.position_ids, dtype=np.int64)
    with torch.inference_mode():
        reference_output = reference_model(
            input_ids=torch.from_numpy(input_ids).to(device),
            attention_mask=torch.from_numpy(attention_mask).to(device),
            position_ids=torch.from_numpy(position_ids).to(device),
            use_cache=False,
            return_dict=True,
        )
    reference_hidden = reference_output.last_hidden_state

    backbone = ort.InferenceSession(
        str(root / "backbone/model.onnx"),
        providers=["CUDAExecutionProvider"],
    )
    assert backbone.get_providers()[0] == "CUDAExecutionProvider"
    feeds = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        **_empty_cache_feeds(backbone, input_ids.shape[0]),
    }
    ort_hidden = backbone.run(["token_hidden_states"], feeds)[0]

    decide = np.asarray(export_batch.decide_indices, dtype=np.int64)
    options = np.asarray(export_batch.option_indices, dtype=np.int64)
    owners = np.asarray(export_batch.option_owners, dtype=np.int64)
    rows = np.arange(len(decide))
    torch_rows = torch.arange(len(decide), device=device)
    torch_decide = torch.from_numpy(decide).to(device)
    torch_options = torch.from_numpy(options).to(device)
    torch_owners = torch.from_numpy(owners).to(device)
    reference_positions = (
        torch.cat(
            (
                reference_hidden[torch_rows, torch_decide],
                reference_hidden[torch_owners, torch_options],
            )
        )
        .float()
        .cpu()
        .numpy()
    )
    ort_positions = np.concatenate((ort_hidden[rows, decide], ort_hidden[owners, options]))
    _assert_close(
        "decide/option hidden states",
        ort_positions,
        reference_positions,
        rtol=2e-3,
        atol=2e-3,
    )

    head_state = {
        name: tensor.to(device=device, dtype=torch.float32)
        for name, tensor in checkpoint["head"].items()
    }
    reference_logits, reference_probabilities = _pointer_reference(
        reference_hidden.float(),
        torch_decide,
        torch_options,
        torch_owners,
        head_state,
    )
    pointer = ort.InferenceSession(
        str(root / "pointer_head/model.onnx"),
        providers=["CUDAExecutionProvider"],
    )
    assert pointer.get_providers()[0] == "CUDAExecutionProvider"
    actual_logits, actual_probabilities = pointer.run(
        None,
        {
            "hidden_states": ort_hidden,
            "decide_indices": decide,
            "option_indices": options,
            "option_owners": owners,
        },
    )
    expected_logits = reference_logits.cpu().numpy()
    expected_probabilities = reference_probabilities.cpu().numpy()
    _assert_close(
        "full-model raw logits",
        actual_logits,
        expected_logits,
        rtol=3e-3,
        atol=3e-3,
    )
    _assert_close(
        "full-model probabilities",
        actual_probabilities,
        expected_probabilities,
        rtol=3e-3,
        atol=3e-3,
    )

    counts = [len(row.keys) for row in export_rows]
    offsets = np.cumsum([0, *counts])
    actual_answers = {
        row.question_id: kev_answer(
            questions[row.question_id],
            row.keys,
            actual_probabilities[offset : offset + count],
        )
        for row, offset, count in zip(export_rows, offsets, counts)
    }
    expected_answers = {
        row.question_id: kev_answer(
            questions[row.question_id],
            row.keys,
            expected_probabilities[offset : offset + count],
        )
        for row, offset, count in zip(export_rows, offsets, counts)
    }
    _assert_answers_close(actual_answers, expected_answers, atol=5e-3)
