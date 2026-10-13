# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Focused numerical parity tests for the CLM decision-model package.

The two integration tests are intentionally opt-in.  They never download a
checkpoint merely because the integration marker was selected.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import onnx_ir as ir
import pytest
import torch
from torch.nn import functional

from mobius._configs import ArchitectureConfig
from mobius.integrations._weight_loading import apply_weights
from mobius.models.decision import (
    CLM_BASE_MODEL_ID,
    CLM_MODEL_ID,
    CLM_REVISION,
    CLMProjectionHead,
    CLMScorer,
    clm_answer,
    clm_candidates,
    clm_last_token_indices,
    clm_state_text,
    render_clm,
)
from mobius.tasks._decision import CLMTask

_FULL_PARITY_FLAG = "CLM_RUN_FULL_PARITY"
_VLLM_PARITY_FLAG = "CLM_RUN_VLLM_PARITY"
_QUESTIONS = {
    "approval": {
        "type": "noul",
        "instructions": "Approve this request?",
    },
    "route": {
        "type": "choice",
        "instructions": "Choose a route.",
        "criteria": {"safe": "Use the safe route", "fast": "Use the fast route"},
    },
}
_STATE = {"ready": True, "attempt": 2, "items": ["a", {"nested": False}]}


def _torch_projection(
    embeddings: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Independent checkpoint-order reference for the published depth-three head."""
    embeddings = functional.normalize(embeddings, dim=-1)
    value = functional.gelu(
        functional.linear(embeddings, weights["inp.weight"], weights["inp.bias"])
    )
    value = functional.linear(value, weights["hidden.0.weight"], weights["hidden.0.bias"])
    value = functional.layer_norm(
        value,
        (value.shape[-1],),
        weights["norms.0.weight"],
        weights["norms.0.bias"],
        eps=1e-5,
    )
    value = functional.gelu(value)
    value = functional.linear(value, weights["out.weight"], weights["out.bias"])
    return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(1e-12)


def _torch_score(
    states: torch.Tensor,
    actions: torch.Tensor,
    owners: torch.Tensor,
    logit_scale: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent scaled-cosine and grouped-softmax reference."""
    scale = torch.exp(logit_scale).clamp_max(100.0)
    logits = (states.index_select(0, owners) * actions).sum(-1)
    logits = logits * scale / temperature
    probabilities = torch.empty_like(logits)
    for owner in range(states.shape[0]):
        selected = owners == owner
        probabilities[selected] = torch.softmax(logits[selected], dim=0)
    return logits, probabilities


def _weights(prefix: float, hidden: int, width: int, projection: int) -> dict:
    """Create deterministic, non-symmetric weights that expose ordering errors."""
    sizes = {
        "inp.weight": (width, hidden),
        "inp.bias": (width,),
        "hidden.0.weight": (width, width),
        "hidden.0.bias": (width,),
        "norms.0.weight": (width,),
        "norms.0.bias": (width,),
        "out.weight": (projection, width),
        "out.bias": (projection,),
    }
    result = {}
    offset = prefix
    for name, shape in sizes.items():
        count = int(np.prod(shape))
        values = torch.linspace(offset, offset + 0.3, count, dtype=torch.float32)
        result[name] = values.reshape(shape)
        offset += 0.17
    return result


def _session(model: ir.Model):
    ort = pytest.importorskip("onnxruntime")
    return ort.InferenceSession(
        ir.to_proto(model).SerializeToString(),
        providers=["CPUExecutionProvider"],
    )


def _head_model(weights: Mapping[str, torch.Tensor], hidden: int):
    config = ArchitectureConfig(hidden_size=hidden, dtype=ir.DataType.FLOAT)
    head = CLMProjectionHead(
        hidden_size=hidden,
        width=next(iter(weights.values())).shape[0],
        projection_dim=weights["out.weight"].shape[0],
    )
    model = CLMTask().build_component("state_head", None, head, config)
    apply_weights(model, dict(weights))
    return model


def test_synthetic_clm_onnx_heads_and_scorer_match_direct_torch():
    """Exercise both projection graphs and scorer in the fast CPU CI lane."""
    torch.manual_seed(7)
    hidden, width, projection = 4, 5, 3
    state_weights = _weights(-0.42, hidden, width, projection)
    action_weights = _weights(0.09, hidden, width, projection)
    state_input = torch.tensor([[0.2, -0.4, 0.8, 1.1], [-0.3, 0.7, 0.5, -0.9]])
    action_input = torch.tensor(
        [
            [0.1, 0.9, -0.2, 0.4],
            [-0.5, 0.3, 1.2, -0.7],
            [0.6, -0.1, 0.2, 0.8],
            [0.4, 0.5, -0.6, 0.3],
        ]
    )
    owners = torch.tensor([0, 0, 1, 1], dtype=torch.int64)

    state_ort = _session(_head_model(state_weights, hidden)).run(
        None, {"embeddings": state_input.numpy()}
    )[0]
    action_ort = _session(_head_model(action_weights, hidden)).run(
        None, {"embeddings": action_input.numpy()}
    )[0]
    state_ref = _torch_projection(state_input, state_weights)
    action_ref = _torch_projection(action_input, action_weights)
    np.testing.assert_allclose(state_ort, state_ref.numpy(), rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(action_ort, action_ref.numpy(), rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(np.linalg.norm(state_ort, axis=-1), 1.0, atol=2e-6)
    np.testing.assert_allclose(np.linalg.norm(action_ort, axis=-1), 1.0, atol=2e-6)

    config = ArchitectureConfig(hidden_size=hidden, dtype=ir.DataType.FLOAT)
    scorer = CLMTask().build_component("scorer", None, CLMScorer(), config)
    logit_scale = torch.tensor([8.0])
    apply_weights(scorer, {"logit_scale": logit_scale})
    temperature = np.asarray(2.75, dtype=np.float32)
    logits, probabilities = _session(scorer).run(
        None,
        {
            "state_projections": state_ort,
            "action_projections": action_ort,
            "candidate_owners": owners.numpy(),
            "temperature": temperature,
        },
    )
    reference_logits, reference_probabilities = _torch_score(
        state_ref, action_ref, owners, logit_scale, float(temperature)
    )
    np.testing.assert_allclose(logits, reference_logits.numpy(), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(
        probabilities, reference_probabilities.numpy(), rtol=2e-5, atol=2e-6
    )
    assert probabilities[:2].sum() == pytest.approx(1.0)
    assert probabilities[2:].sum() == pytest.approx(1.0)
    assert clm_answer({"type": "choice"}, ["first", "second"], [0.5, 0.5])["choice"] == "first"
    assert (
        clm_answer({"type": "choice"}, ["first", "second"], [0.25, 0.75])["choice"] == "second"
    )


class _GoldenTokenizer:
    """Small deterministic tokenizer oracle; no network or model dependency."""

    _pieces: ClassVar[dict[str, list[int]]] = {
        "ready: true\n\nitems:\n  - red\n  - blue\n\nChoose.": [11, 12, 21, 22, 31],
        "alpha": [41, 42],
        "A letter": [43, 44, 45],
    }

    def __call__(self, texts: Sequence[str], *, padding: bool) -> dict[str, list]:
        assert padding
        rows = [self._pieces[text] for text in texts]
        length = max(map(len, rows))
        return {
            "input_ids": [row + [0] * (length - len(row)) for row in rows],
            "attention_mask": [[1] * len(row) + [0] * (length - len(row)) for row in rows],
        }


def test_clm_render_tokenization_and_last_token_golden():
    """Lock exact rendering, candidate text, token ids, and padding-aware pooling."""
    state_text = clm_state_text({"ready": True, "items": ["red", "blue"]}, "Choose.")
    keys, actions = clm_candidates(
        {"type": "choice", "criteria": {"alpha": None, "letter": "A letter"}}
    )
    assert state_text == "ready: true\n\nitems:\n  - red\n  - blue\n\nChoose."
    assert keys == ["alpha", "letter"]
    assert actions == ["alpha", "A letter"]
    batch = _GoldenTokenizer()([state_text, *actions], padding=True)
    assert batch == {
        "input_ids": [
            [11, 12, 21, 22, 31],
            [41, 42, 0, 0, 0],
            [43, 44, 45, 0, 0],
        ],
        "attention_mask": [
            [1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
        ],
    }
    assert clm_last_token_indices(batch["attention_mask"]) == [4, 1, 2]
    assert render_clm({"empty": None, "enabled": False}) == "empty: \n\nenabled: false"


def _require_enabled(flag: str) -> None:
    if os.getenv(flag) != "1":
        pytest.skip(f"set {flag}=1 to enable this heavyweight parity test")


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        pytest.fail(f"{name} must be set when full CLM parity is enabled")
    return value


def _component_path(root: Path, name: str) -> Path:
    nested = root / name / "model.onnx"
    direct = root / f"{name}.onnx"
    path = nested if nested.is_file() else direct
    if not path.is_file():
        pytest.fail(f"missing exported CLM {name} graph; tried {nested} and {direct}")
    return path


def _cuda_sessions(root: Path) -> dict[str, Any]:
    ort = pytest.importorskip("onnxruntime")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("CLM full parity requires ONNX Runtime CUDAExecutionProvider")
    options = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return {
        name: ort.InferenceSession(str(_component_path(root, name)), providers=options)
        for name in ("encoder", "state_head", "action_head", "scorer")
    }


def _load_reference():
    """Load pinned Qwen and official CLM heads only after explicit opt-in."""
    revision = _required_env("CLM_BASE_REVISION")
    transformers = pytest.importorskip("transformers")
    hub = pytest.importorskip("huggingface_hub")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        CLM_BASE_MODEL_ID, revision=revision
    )
    encoder = (
        transformers.AutoModel.from_pretrained(
            CLM_BASE_MODEL_ID,
            revision=revision,
            dtype=torch.float32,
        )
        .eval()
        .cuda()
    )
    filename = os.getenv("CLM_HEAD_FILENAME", "CLM_v0.1-8B.pt")
    local = os.getenv("CLM_HEAD_CHECKPOINT_PATH")
    checkpoint_path = (
        local if local else hub.hf_hub_download(CLM_MODEL_ID, filename, revision=CLM_REVISION)
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return tokenizer, encoder, checkpoint


def _texts() -> tuple[list[str], list[str], list[str], np.ndarray]:
    states, actions, keys, owners = [], [], [], []
    for owner, question in enumerate(_QUESTIONS.values()):
        state_text = clm_state_text(_STATE, question["instructions"])
        question_keys, question_actions = clm_candidates(question)
        states.append(state_text)
        actions.extend(question_actions)
        keys.extend(question_keys)
        owners.extend([owner] * len(question_actions))
    return states, actions, keys, np.asarray(owners, dtype=np.int64)


def _tokenize(tokenizer, texts: Sequence[str]) -> dict[str, torch.Tensor]:
    return tokenizer(
        list(texts),
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )


def _last_hidden(reference, tokens: Mapping[str, torch.Tensor]) -> torch.Tensor:
    cuda_tokens = {name: value.cuda() for name, value in tokens.items()}
    with torch.inference_mode():
        hidden = reference(**cuda_tokens, return_dict=True).last_hidden_state
    indices = torch.tensor(
        clm_last_token_indices(tokens["attention_mask"].tolist()),
        device=hidden.device,
    )
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), indices].float()


def _encoder_ort(session, tokens: Mapping[str, torch.Tensor]) -> np.ndarray:
    ids = tokens["input_ids"].numpy().astype(np.int64)
    mask = tokens["attention_mask"].numpy().astype(np.int64)
    position_ids = np.maximum(np.cumsum(mask, axis=1) - 1, 0).astype(np.int64)
    values = {"input_ids": ids, "attention_mask": mask, "position_ids": position_ids}
    feeds = {}
    for item in session.get_inputs():
        if item.name in values:
            feeds[item.name] = values[item.name]
            continue
        if not item.name.startswith("past_key_values."):
            pytest.fail(f"unsupported encoder input {item.name!r} in CLM parity fixture")
        shape = []
        for axis, dimension in enumerate(item.shape):
            if isinstance(dimension, int):
                shape.append(dimension)
            elif axis == 0:
                shape.append(ids.shape[0])
            else:
                shape.append(0)
        dtype = np.float16 if item.type == "tensor(float16)" else np.float32
        feeds[item.name] = np.zeros(shape, dtype=dtype)
    hidden = session.run(None, feeds)[0]
    indices = clm_last_token_indices(mask.tolist())
    return hidden[np.arange(len(indices)), indices]


def _checkpoint_head(checkpoint: Mapping[str, Any], name: str) -> dict[str, torch.Tensor]:
    value = checkpoint.get(name)
    if not isinstance(value, Mapping):
        pytest.fail(f"official CLM checkpoint has no {name!r} state dict")
    return dict(value)


def _run_full_parity(root: Path) -> dict[str, Any]:
    sessions = _cuda_sessions(root)
    tokenizer, encoder, checkpoint = _load_reference()
    state_texts, action_texts, keys, owners = _texts()
    assert state_texts == [
        (
            "ready: true\n\nattempt: 2\n\nitems:\n  - a\n  -\n    nested: false"
            "\n\nApprove this request?"
        ),
        (
            "ready: true\n\nattempt: 2\n\nitems:\n  - a\n  -\n    nested: false"
            "\n\nChoose a route."
        ),
    ], f"CLM state rendering drifted: {state_texts!r}"
    assert action_texts == [
        "false: No. This is false: Approve this request?",
        "true: Yes. This is true: Approve this request?",
        "Use the safe route",
        "Use the fast route",
    ], f"CLM action rendering drifted: {action_texts!r}"
    state_tokens = _tokenize(tokenizer, state_texts)
    action_tokens = _tokenize(tokenizer, action_texts)
    assert state_tokens["input_ids"].ndim == action_tokens["input_ids"].ndim == 2
    state_ids_unbatched = [
        tokenizer(text, add_special_tokens=False)["input_ids"] for text in state_texts
    ]
    state_ids_batched = [
        ids[mask.bool()].tolist()
        for ids, mask in zip(state_tokens["input_ids"], state_tokens["attention_mask"])
    ]
    assert state_ids_batched == state_ids_unbatched, (
        "batched CLM token IDs differ from per-text tokenization: "
        f"batched={state_ids_batched!r}, unbatched={state_ids_unbatched!r}"
    )
    state_hidden_ref = functional.normalize(_last_hidden(encoder, state_tokens), dim=-1)
    action_hidden_ref = functional.normalize(_last_hidden(encoder, action_tokens), dim=-1)
    state_hidden_ort = _encoder_ort(sessions["encoder"], state_tokens)
    action_hidden_ort = _encoder_ort(sessions["encoder"], action_tokens)
    state_hidden_ort /= np.maximum(
        np.linalg.norm(state_hidden_ort, axis=-1, keepdims=True), 1e-12
    )
    action_hidden_ort /= np.maximum(
        np.linalg.norm(action_hidden_ort, axis=-1, keepdims=True), 1e-12
    )
    np.testing.assert_allclose(state_hidden_ort, state_hidden_ref.cpu(), rtol=3e-3, atol=3e-3)
    np.testing.assert_allclose(
        action_hidden_ort, action_hidden_ref.cpu(), rtol=3e-3, atol=3e-3
    )

    state_weights = _checkpoint_head(checkpoint, "state_head")
    action_weights = _checkpoint_head(checkpoint, "action_head")
    state_ref = _torch_projection(state_hidden_ref.cpu(), state_weights)
    action_ref = _torch_projection(action_hidden_ref.cpu(), action_weights)
    state_ort = sessions["state_head"].run(
        None, {"embeddings": state_hidden_ort.astype(np.float32)}
    )[0]
    action_ort = sessions["action_head"].run(
        None, {"embeddings": action_hidden_ort.astype(np.float32)}
    )[0]
    np.testing.assert_allclose(state_ort, state_ref, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(action_ort, action_ref, rtol=2e-3, atol=2e-3)

    temperature = np.asarray(float(os.getenv("CLM_TEST_TEMPERATURE", "1.7")), np.float32)
    logits, probabilities = sessions["scorer"].run(
        None,
        {
            "state_projections": state_ort,
            "action_projections": action_ort,
            "candidate_owners": owners,
            "temperature": temperature,
        },
    )
    scale = torch.as_tensor(checkpoint["logit_scale"]).reshape(1)
    logits_ref, probabilities_ref = _torch_score(
        state_ref, action_ref, torch.from_numpy(owners), scale, float(temperature)
    )
    np.testing.assert_allclose(logits, logits_ref, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(probabilities, probabilities_ref, rtol=2e-3, atol=2e-4)
    answers, offset = [], 0
    for question in _QUESTIONS.values():
        question_keys, _ = clm_candidates(question)
        size = len(question_keys)
        answers.append(
            clm_answer(question, question_keys, probabilities[offset : offset + size])
        )
        reference_answer = clm_answer(
            question, question_keys, probabilities_ref[offset : offset + size]
        )
        assert answers[-1]["type"] == reference_answer["type"]
        if question["type"] == "choice":
            assert answers[-1]["choice"] == reference_answer["choice"], (
                f"answer mismatch for {question!r}: "
                f"onnx={answers[-1]!r}, torch={reference_answer!r}"
            )
        elif question["type"] == "noul":
            assert answers[-1]["noul"] == pytest.approx(reference_answer["noul"], abs=2e-4)
        offset += size
    return {
        "sessions": sessions,
        "tokenizer": tokenizer,
        "checkpoint": checkpoint,
        "state_texts": state_texts,
        "action_texts": action_texts,
        "state_hidden": state_hidden_ort,
        "action_hidden": action_hidden_ort,
        "probabilities": probabilities,
        "owners": owners,
        "answers": answers,
        "keys": keys,
        "temperature": temperature,
    }


@pytest.mark.integration
def test_exported_clm_package_matches_pinned_pytorch_reference():
    """Compare every externally visible full-model stage with diagnostics."""
    _require_enabled(_FULL_PARITY_FLAG)
    root = Path(_required_env("MOBIUS_DECISION_EXPORT_ROOT")).expanduser()
    result = _run_full_parity(root)
    assert result["answers"], (
        f"no answers produced; root={root}, base_revision="
        f"{os.getenv('CLM_BASE_REVISION')}, clm_revision={CLM_REVISION}"
    )


def _post_embeddings(url: str, model: str, texts: Sequence[str]) -> np.ndarray:
    body = json.dumps({"model": model, "input": list(texts), "encoding_format": "float"})
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/embeddings",
        data=body.encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    rows = sorted(payload["data"], key=lambda row: row["index"])
    return np.asarray([row["embedding"] for row in rows], dtype=np.float32)


@pytest.mark.integration
def test_official_clm_vllm_embeddings_endpoint_matches_onnx():
    """Compare an externally managed official vLLM server with the export."""
    _require_enabled(_VLLM_PARITY_FLAG)
    _require_enabled(_FULL_PARITY_FLAG)
    url = _required_env("CLM_EMBED_URL")
    model = _required_env("CLM_EMBED_MODEL")
    metadata = _required_env("CLM_VLLM_METADATA")
    root = Path(_required_env("MOBIUS_DECISION_EXPORT_ROOT")).expanduser()
    result = _run_full_parity(root)
    texts = [*result["state_texts"], *result["action_texts"]]
    endpoint = _post_embeddings(url, model, texts)
    state_count = len(result["state_texts"])
    onnx_hidden = np.concatenate([result["state_hidden"], result["action_hidden"]], axis=0)
    np.testing.assert_allclose(
        endpoint,
        onnx_hidden,
        rtol=4e-3,
        atol=4e-3,
        err_msg=f"external vLLM metadata: {metadata}",
    )
    state_endpoint = _torch_projection(
        torch.from_numpy(endpoint[:state_count]),
        _checkpoint_head(result["checkpoint"], "state_head"),
    )
    action_endpoint = _torch_projection(
        torch.from_numpy(endpoint[state_count:]),
        _checkpoint_head(result["checkpoint"], "action_head"),
    )
    _, endpoint_probabilities = _torch_score(
        state_endpoint,
        action_endpoint,
        torch.from_numpy(result["owners"]),
        torch.as_tensor(result["checkpoint"]["logit_scale"]).reshape(1),
        float(result["temperature"]),
    )
    np.testing.assert_allclose(
        endpoint_probabilities,
        result["probabilities"],
        rtol=4e-3,
        atol=4e-4,
        err_msg=(
            "vLLM version and served revision are not discoverable reliably from "
            f"/v1/embeddings; preserve them as external CLM_VLLM_METADATA: {metadata}"
        ),
    )
