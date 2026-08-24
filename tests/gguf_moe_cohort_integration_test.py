# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from huggingface_hub import hf_hub_download

from mobius.integrations.gguf import build_from_gguf
from mobius.integrations.gguf._reader import GGUFModel
from mobius.integrations.gguf._tensor_mapping import is_known_skip, map_gguf_to_hf_names


_REPO_ID = "bartowski/granite-3.0-1b-a400m-instruct-GGUF"
_REVISION = "0e1c3cecaa6e49ac0721be91ef441ec72eae62d4"
_FILENAME = "granite-3.0-1b-a400m-instruct-Q4_K_M.gguf"
_SIZE = 821_845_024
_SHA256 = "074f09e13484e54e73c93830d34e9fa9917a6319fb8bae762a22594b9b4da0dc"
_QTYPES = {"F32": 73, "Q4_K": 144, "Q6_K": 25}
_PREFILL_CHECKSUM = -587_230.6105168748
_GREEDY_TOKENS = [34, 34, 34]

# Base-model config/tokenizer provenance inspected before downloading the GGUF:
# ibm-granite/granite-3.0-1b-a400m-instruct at this immutable revision.
_BASE_REVISION = "ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _empty_cache(model_input) -> np.ndarray:
    return np.empty(
        [1, model_input.shape[1], 0, model_input.shape[3]],
        dtype=np.float32,
    )


def _generate_three(session: ort.InferenceSession) -> tuple[np.ndarray, list[int]]:
    output_names = [output.name for output in session.get_outputs()]
    cache_inputs = [
        model_input
        for model_input in session.get_inputs()
        if model_input.name.startswith("past_key_values.")
    ]
    feeds = {
        "input_ids": np.array([[1, 2]], dtype=np.int64),
        "attention_mask": np.ones((1, 2), dtype=np.int64),
        "position_ids": np.array([[0, 1]], dtype=np.int64),
        **{model_input.name: _empty_cache(model_input) for model_input in cache_inputs},
    }
    outputs = session.run(None, feeds)
    prefill_logits = outputs[0]
    output_map = dict(zip(output_names, outputs))
    logits = prefill_logits
    tokens: list[int] = []
    total_length = 2

    for _ in range(3):
        token = int(logits[0, -1].argmax())
        tokens.append(token)
        total_length += 1
        feeds = {
            "input_ids": np.array([[token]], dtype=np.int64),
            "attention_mask": np.ones((1, total_length), dtype=np.int64),
            "position_ids": np.array([[total_length - 1]], dtype=np.int64),
        }
        for model_input in cache_inputs:
            suffix = model_input.name.removeprefix("past_key_values.")
            feeds[model_input.name] = output_map[f"present.{suffix}"]
        outputs = session.run(None, feeds)
        logits = outputs[0]
        output_map = dict(zip(output_names, outputs))

    return prefill_logits, tokens


@pytest.mark.integration
def test_real_granitemoe_gguf_artifact(tmp_path: Path) -> None:
    """Pinned GraniteMoE weights preserve every router/expert and execute in ORT."""
    path = Path(
        hf_hub_download(
            repo_id=_REPO_ID,
            revision=_REVISION,
            filename=_FILENAME,
        )
    )
    assert path.stat().st_size == _SIZE
    assert _sha256(path) == _SHA256

    gguf_model = GGUFModel(path)
    assert gguf_model.architecture == "granitemoe"
    qtypes = Counter(qtype.name for _, _, qtype, _ in gguf_model.tensor_items_raw())
    assert dict(sorted(qtypes.items())) == _QTYPES
    assert sum("ffn_gate_inp" in name for name in gguf_model.tensor_names) == 24
    assert sum("_exps" in name for name in gguf_model.tensor_names) == 72
    assert not [
        name
        for name in gguf_model.tensor_names
        if map_gguf_to_hf_names(name, "granitemoe") is None and not is_known_skip(name)
    ]

    package = build_from_gguf(path)
    model = package["model"]
    initializer_names = set(model.graph.initializers)
    assert sum(".mlp.experts." in name and name.endswith(".weight") for name in initializer_names) == (
        24 * 32 * 3
    )
    assert sum(".mlp.gate.weight" in name for name in initializer_names) == 24
    assert not any("fc1_experts" in name for name in initializer_names)
    # 2,400 projection matmuls plus the tied quantized embedding/output head.
    assert sum(node.op_type == "MatMulNBits" for node in model.graph) == 2401
    assert not any(node.op_type == "QMoE" for node in model.graph)

    output_dir = tmp_path / "granitemoe"
    package.save(output_dir, progress_bar=False)
    session = ort.InferenceSession(
        str(output_dir / "model.onnx"),
        providers=["CPUExecutionProvider"],
    )
    first_logits, first_tokens = _generate_three(session)
    second_logits, second_tokens = _generate_three(session)
    assert first_logits.shape == (1, 2, 49_155)
    assert np.isfinite(first_logits).all()
    assert float(np.sum(first_logits, dtype=np.float64)) == pytest.approx(_PREFILL_CHECKSUM)
    np.testing.assert_array_equal(first_logits, second_logits)
    assert first_tokens == second_tokens == _GREEDY_TOKENS
