# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx_ir as ir
import pytest

from mobius.integrations.gguf._draft_runtime import (
    DraftPairRunner,
    _external_data_files,
    _snapshot_graph_package,
)
from mobius.integrations.gguf._runtime_evidence import gguf_graph_package_identity


class _FakeSession:
    def __init__(self, inputs, outputs):
        self._inputs = [
            SimpleNamespace(name=name, type=type_name, shape=shape)
            for name, type_name, shape in inputs
        ]
        self._outputs = outputs
        self.last_feeds = None

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return [SimpleNamespace(name=name) for name in self._outputs]

    def run(self, names, feeds):
        self.last_feeds = feeds
        return [self._outputs[name] for name in names]


class _BoundedGenerationRunner(DraftPairRunner):
    def _run_target(self, input_ids, cache):
        tokens = np.asarray(input_ids, dtype=np.int64)
        past = next(iter(cache.values())).shape[2]
        total = past + tokens.shape[1]
        self.max_target_length = max(self.max_target_length, total)
        logits = np.full((1, tokens.shape[1], 64), -1.0, dtype=np.float32)
        for index, token in enumerate(tokens[0]):
            logits[0, index, (int(token) + 1) % 64] = 1.0
        present = {
            name.replace("past_key_values.", "present."): np.zeros(
                (1, 1, total, 4),
                dtype=np.float32,
            )
            for name in cache
        }
        self._target_forwards += 1
        return {"logits": logits, **present}, {
            name: present[name.replace("past_key_values.", "present.")] for name in cache
        }

    def _target_features(self, outputs, selection=slice(None)):
        return np.zeros((1, outputs["logits"][:, selection].shape[1], 12), dtype=np.float32)

    def _outputs(self, session, feeds):
        if session is self.target_embedding:
            return {
                "inputs_embeds": np.asarray(feeds["input_ids"], dtype=np.float32)[..., None]
            }
        return super()._outputs(session, feeds)

    def _run_draft(self, feeds, cache):
        block_size = int(self.manifest["draft"]["block_size"])
        start = int(feeds["noise_embedding"][0, 0, 0])
        logits = np.full((1, block_size, 64), -1.0, dtype=np.float32)
        for index in range(1, block_size):
            logits[0, index, (start + index) % 64] = 1.0
        total = int(feeds["q_position_ids"][0, -1]) + 1
        present = {
            name.replace("past_key_values.", "present."): np.zeros(
                (1, 1, total, 4),
                dtype=np.float32,
            )
            for name in cache
        }
        self._draft_forwards += 1
        return {"draft_logits": logits, **present}, {
            name: present[name.replace("past_key_values.", "present.")] for name in cache
        }

    def _run_eagle_step(self, input_ids, fused_hidden, recycled_hidden, cache):
        tokens = np.asarray(input_ids, dtype=np.int64)
        past = next(iter(cache.values())).shape[2]
        total = past + tokens.shape[1]
        logits = np.full((1, tokens.shape[1], 64), -1.0, dtype=np.float32)
        for index, token in enumerate(tokens[0]):
            logits[0, index, (int(token) + 1) % 64] = 1.0
        present = {
            name.replace("past_key_values.", "present."): np.zeros(
                (1, 1, total, 4),
                dtype=np.float32,
            )
            for name in cache
        }
        outputs = {
            "draft_logits": logits,
            "next_hidden": np.zeros((1, tokens.shape[1], 4), dtype=np.float32),
            **present,
        }
        self._draft_forwards += 1
        return outputs, {
            name: present[name.replace("past_key_values.", "present.")] for name in cache
        }


def _bounded_runner(architecture: str) -> _BoundedGenerationRunner:
    cache_inputs = [
        ("past_key_values.0.key", "tensor(float)", [1, 1, "past", 4]),
        ("past_key_values.0.value", "tensor(float)", [1, 1, "past", 4]),
    ]
    runner = object.__new__(_BoundedGenerationRunner)
    runner.target = _FakeSession(cache_inputs, {})
    runner.draft = _FakeSession(
        [
            ("fused_hidden", "tensor(float)", [1, "sequence", 12]),
            ("recycled_hidden", "tensor(float)", [1, "sequence", 4]),
            ("q_position_ids", "tensor(int64)", [1, "sequence"]),
            *cache_inputs,
        ],
        {},
    )
    runner.target_embedding = object()
    runner.target_lm_head = None
    runner.manifest = {
        "architecture": architecture,
        "draft": {"block_size": 16, "mask_token_id": 63, "hidden_size": 4},
        "draft_to_target": None,
    }
    runner._target_forwards = 0
    runner._draft_forwards = 0
    runner.max_target_length = 0
    return runner


@pytest.mark.parametrize("max_new_tokens", range(1, 18))
def test_dflash_final_tail_never_processes_beyond_requested_tokens(
    max_new_tokens: int,
) -> None:
    runner = _bounded_runner("dflash")
    result = runner._generate_dflash(np.array([[0]], dtype=np.int64), max_new_tokens)

    assert result.tokens == tuple(range(1, max_new_tokens + 1))
    assert runner.max_target_length == max_new_tokens
    assert result.stats.proposed_tokens == (15 if max_new_tokens == 17 else 0)


@pytest.mark.parametrize("max_new_tokens", range(1, 7))
def test_eagle_final_round_never_processes_beyond_requested_tokens(
    max_new_tokens: int,
) -> None:
    runner = _bounded_runner("eagle3")
    result = runner._generate_eagle3(
        np.array([[0]], dtype=np.int64),
        max_new_tokens,
        width=4,
    )

    assert result.tokens == tuple(range(1, max_new_tokens + 1))
    assert runner.max_target_length == max_new_tokens
    assert result.stats.proposed_tokens == max(0, max_new_tokens - 2)


def test_eagle_step_uses_target_embedding_when_draft_borrows_it() -> None:
    embedding = _FakeSession(
        [("input_ids", "tensor(int64)", [1, "sequence"])],
        {"inputs_embeds": np.ones((1, 1, 4), dtype=np.float32)},
    )
    draft = _FakeSession(
        [
            ("inputs_embeds", "tensor(float)", [1, "sequence", 4]),
            ("fused_hidden", "tensor(float)", [1, "sequence", 12]),
            ("recycled_hidden", "tensor(float)", [1, "sequence", 4]),
            ("attention_mask", "tensor(int64)", [1, "total"]),
            ("past_key_values.0.key", "tensor(float)", [1, 1, "past", 4]),
            ("past_key_values.0.value", "tensor(float)", [1, 1, "past", 4]),
        ],
        {
            "draft_hidden": np.ones((1, 1, 4), dtype=np.float32),
            "next_hidden": np.ones((1, 1, 4), dtype=np.float32),
            "present.0.key": np.ones((1, 1, 1, 4), dtype=np.float32),
            "present.0.value": np.ones((1, 1, 1, 4), dtype=np.float32),
        },
    )
    runner = object.__new__(DraftPairRunner)
    runner.draft = draft
    runner.target_embedding = embedding
    runner.target_lm_head = None
    runner._draft_forwards = 0
    cache = {
        "past_key_values.0.key": np.empty((1, 1, 0, 4), dtype=np.float32),
        "past_key_values.0.value": np.empty((1, 1, 0, 4), dtype=np.float32),
    }

    runner._run_eagle_step(
        np.array([[3]], dtype=np.int64),
        np.ones((1, 1, 12), dtype=np.float32),
        np.zeros((1, 1, 4), dtype=np.float32),
        cache,
    )

    assert "inputs_embeds" in draft.last_feeds
    assert "input_ids" not in draft.last_feeds


def test_draft_logits_use_owned_head_or_target_bridge() -> None:
    runner = object.__new__(DraftPairRunner)
    runner.target_lm_head = None
    owned = np.ones((1, 2, 5), dtype=np.float32)
    assert runner._project_draft_logits({"draft_logits": owned}) is owned

    projected = np.arange(5, dtype=np.float32).reshape(1, 1, 5)
    runner.target_lm_head = _FakeSession(
        [("hidden_states", "tensor(float)", [1, "sequence", 4])],
        {"logits": projected},
    )
    hidden = np.ones((1, 1, 4), dtype=np.float32)
    np.testing.assert_array_equal(
        runner._project_draft_logits({"draft_hidden": hidden}),
        projected,
    )

    runner.target_lm_head = None
    with pytest.raises(ValueError, match="requires a target LM-head bridge"):
        runner._project_draft_logits({"draft_hidden": hidden})


def test_external_data_scan_covers_function_default_attributes(tmp_path: Path) -> None:
    data_path = tmp_path / "function.data"
    data_path.write_bytes(np.array([1.0], dtype=np.float32).tobytes())
    tensor = ir.ExternalTensor(
        data_path.name,
        0,
        4,
        ir.DataType.FLOAT,
        shape=ir.Shape([1]),
        name="default_weight",
        base_dir=tmp_path,
    )
    function = ir.Function(
        "test",
        "WithDefault",
        graph=ir.Graph([], [], nodes=[], name="function"),
        attributes={"weight": ir.AttrTensor("weight", tensor)},
    )
    model = ir.Model(
        ir.Graph([], [], nodes=[], name="main"),
        ir_version=11,
        functions=[function],
    )
    model_path = tmp_path / "model.onnx"
    ir.save(model, model_path)

    assert _external_data_files(model_path, Path("component/model.onnx")) == (
        "component/function.data",
    )


def test_graph_package_snapshot_is_immutable_after_source_replacement(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.onnx").write_bytes(b"verified")
    identity = gguf_graph_package_identity(source)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    _snapshot_graph_package(
        source,
        identity.files,
        identity.sha256,
        snapshot,
    )
    (source / "model.onnx").write_bytes(b"replaced")

    assert (snapshot / "model.onnx").read_bytes() == b"verified"
