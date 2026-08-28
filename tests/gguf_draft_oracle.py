# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Independent direct-ORT oracle for real GGUF target/draft integration tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

Mutation = Literal["cache_copy", "draft_mapping", "proposal_order", "rollback"]


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    dtype = array.dtype.str.encode()
    shape = json.dumps(array.shape, separators=(",", ":")).encode()
    digest.update(len(dtype).to_bytes(4, "big"))
    digest.update(dtype)
    digest.update(len(shape).to_bytes(4, "big"))
    digest.update(shape)
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _cache_digest(cache: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(cache):
        encoded = name.encode()
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(_array_digest(cache[name])))
    return digest.hexdigest()


def _cache_length(cache: dict[str, np.ndarray]) -> int:
    lengths = {value.shape[2] for value in cache.values()}
    if len(lengths) != 1:
        raise AssertionError(f"cache tensors have inconsistent lengths: {lengths}")
    return lengths.pop()


def _crop(cache: dict[str, np.ndarray], length: int) -> dict[str, np.ndarray]:
    return {name: value[:, :, :length, :] for name, value in cache.items()}


def _cache_max_abs(
    actual: dict[str, np.ndarray],
    expected: dict[str, np.ndarray],
) -> float:
    if actual.keys() != expected.keys():
        raise AssertionError("cache names differ")
    maximum = 0.0
    for name in actual:
        if actual[name].shape != expected[name].shape:
            return float("inf")
        if actual[name].size:
            maximum = max(
                maximum,
                float(np.max(np.abs(actual[name] - expected[name]))),
            )
    return maximum


def _assert_cache_equal(
    actual: dict[str, np.ndarray],
    expected: dict[str, np.ndarray],
) -> None:
    if actual.keys() != expected.keys() or any(
        not np.array_equal(actual[name], expected[name]) for name in actual
    ):
        raise AssertionError("independent draft cache replay differs")


def _empty_cache(session: Any) -> dict[str, np.ndarray]:
    cache = {}
    for value in session.get_inputs():
        if not value.name.startswith("past_key_values."):
            continue
        dtype = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
        }[value.type]
        shape = [
            dimension
            if isinstance(dimension, int)
            else (0 if "past" in str(dimension) else 1)
            for dimension in value.shape
        ]
        cache[value.name] = np.empty(shape, dtype=dtype)
    if not cache or len(cache) % 2:
        raise AssertionError("model has no complete dynamic KV cache")
    return cache


def _outputs(session: Any, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    names = [value.name for value in session.get_outputs()]
    return dict(zip(names, session.run(names, feeds)))


def _present(outputs: dict[str, np.ndarray], layers: int) -> dict[str, np.ndarray]:
    return {
        f"past_key_values.{layer}.{kind}": outputs[f"present.{layer}.{kind}"]
        for layer in range(layers)
        for kind in ("key", "value")
    }


def _raw_contract(draft_gguf: Path) -> dict[str, Any]:
    from gguf import GGUFReader

    reader = GGUFReader(draft_gguf, "r")
    architecture = str(reader.fields["general.architecture"].contents())
    raw_layers = [int(value) for value in reader.fields[f"{architecture}.target_layers"].contents()]
    layers = [index - 1 for index in raw_layers]
    d2t = next(
        (
            np.asarray(tensor.data).reshape(-1).astype(np.int64).tolist()
            for tensor in reader.tensors
            if tensor.name == "d2t"
        ),
        None,
    )
    return {
        "architecture": architecture,
        "raw_target_layers": raw_layers,
        "target_layers": layers,
        "draft_to_target": d2t,
        "block_size": (
            int(reader.fields[f"{architecture}.block_size"].contents())
            if architecture == "dflash"
            else None
        ),
        "mask_token_id": (
            int(reader.fields["tokenizer.ggml.mask_token_id"].contents())
            if architecture == "dflash"
            else None
        ),
    }


class IndependentDraftOracle:
    """Drive target/draft ONNX sessions without production coordinator helpers."""

    def __init__(
        self,
        package_dir: Path,
        draft_gguf: Path,
        *,
        session_options: Any,
        mutation: Mutation | None = None,
    ):
        import onnxruntime as ort

        self.package_dir = package_dir
        self.contract = _raw_contract(draft_gguf)
        self.mutation = mutation
        self.target = ort.InferenceSession(
            str(package_dir / "target" / "model.onnx"),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.draft = ort.InferenceSession(
            str(package_dir / "draft" / "model.onnx"),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        embedding = package_dir / "target_embedding" / "model.onnx"
        head = package_dir / "target_lm_head" / "model.onnx"
        self.embedding = (
            ort.InferenceSession(
                str(embedding),
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
            if embedding.is_file()
            else None
        )
        self.head = (
            ort.InferenceSession(
                str(head),
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
            if head.is_file()
            else None
        )

    def _run_target(
        self,
        tokens: np.ndarray,
        cache: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        tokens = np.asarray(tokens, dtype=np.int64)
        past = _cache_length(cache)
        feeds = {"input_ids": tokens, **cache}
        names = {value.name for value in self.target.get_inputs()}
        if "attention_mask" in names:
            feeds["attention_mask"] = np.ones((1, past + tokens.shape[1]), dtype=np.int64)
        if "position_ids" in names:
            feeds["position_ids"] = np.arange(
                past, past + tokens.shape[1], dtype=np.int64
            )[None, :]
        outputs = _outputs(self.target, feeds)
        return outputs, _present(outputs, len(cache) // 2)

    def _features(
        self,
        outputs: dict[str, np.ndarray],
        selection: slice = slice(None),
    ) -> np.ndarray:
        return np.concatenate(
            [
                outputs[f"hidden_states.{index}"][:, selection]
                for index in self.contract["target_layers"]
            ],
            axis=-1,
        )

    def _draft_logits(self, outputs: dict[str, np.ndarray]) -> np.ndarray:
        if "draft_logits" in outputs:
            return outputs["draft_logits"]
        if self.head is None:
            raise AssertionError("draft_hidden requires target LM-head component")
        return _outputs(
            self.head,
            {"hidden_states": outputs["draft_hidden"]},
        )["logits"]

    def _map_proposals(self, draft_ids: np.ndarray) -> list[int]:
        remap = self.contract["draft_to_target"]
        if remap is None:
            mapped = [int(token) for token in draft_ids]
            if self.mutation == "draft_mapping" and mapped:
                mapped[0] ^= 1
            return mapped
        mapped = [int(remap[int(token)]) for token in draft_ids]
        if self.mutation == "draft_mapping" and mapped:
            mapped[0] = int(remap[(int(draft_ids[0]) + 1) % len(remap)])
        return mapped

    def target_only(self, input_ids: np.ndarray, count: int) -> list[int]:
        cache = _empty_cache(self.target)
        outputs, cache = self._run_target(input_ids, cache)
        generated = [int(np.argmax(outputs["logits"][0, -1]))]
        while len(generated) < count:
            outputs, cache = self._run_target([[generated[-1]]], cache)
            generated.append(int(np.argmax(outputs["logits"][0, -1])))
        return generated

    def run(self, input_ids: np.ndarray, count: int) -> dict[str, Any]:
        if self.contract["architecture"] == "dflash":
            trace = self._run_dflash(input_ids, count)
        else:
            trace = self._run_eagle3(input_ids, count)
        trace["target_only_tokens"] = self.target_only(input_ids, count)
        trace["tokens_equal"] = trace["generated_tokens"] == trace["target_only_tokens"]
        return trace

    def _round(
        self,
        *,
        proposal_ids: list[int],
        proposals: list[int],
        proposal_logits: np.ndarray,
        posterior: np.ndarray,
        accepted: int,
        correction: int,
        target_before: dict[str, np.ndarray],
        target_tentative: dict[str, np.ndarray],
        target_committed: dict[str, np.ndarray],
        target_replay: dict[str, np.ndarray],
        target_replay_logits: np.ndarray,
        target_replay_tokens: list[int],
        draft_before: dict[str, np.ndarray],
        draft_tentative: dict[str, np.ndarray],
        draft_committed: dict[str, np.ndarray],
        draft_replay: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        if self.mutation == "proposal_order" and len(proposals) > 1:
            proposals = [proposals[1], proposals[0], *proposals[2:]]
        if self.mutation == "cache_copy":
            first = next(iter(draft_committed))
            draft_committed = dict(draft_committed)
            draft_committed[first] = draft_committed[first].copy()
            if draft_committed[first].size:
                draft_committed[first].flat[0] += 1
        if self.mutation == "rollback":
            target_committed = _crop(
                target_tentative,
                max(0, _cache_length(target_committed) - 1),
            )
        return {
            "proposal_ids": proposal_ids,
            "proposal_tokens": proposals,
            "proposal_logits_sha256": _array_digest(proposal_logits),
            "posterior_tokens": posterior.astype(np.int64).tolist(),
            "accepted_prefix": accepted,
            "correction_token": correction,
            "target": {
                "before_length": _cache_length(target_before),
                "before_sha256": _cache_digest(target_before),
                "tentative_length": _cache_length(target_tentative),
                "tentative_sha256": _cache_digest(target_tentative),
                "committed_length": _cache_length(target_committed),
                "committed_sha256": _cache_digest(target_committed),
                "replay_length": _cache_length(target_replay),
                "replay_sha256": _cache_digest(target_replay),
                "replay_max_abs": _cache_max_abs(target_committed, target_replay),
                "replay_logits_sha256": _array_digest(target_replay_logits),
                "replay_tokens": target_replay_tokens,
                "replay_tokens_match": (
                    target_replay_tokens == posterior[: accepted + 1].astype(np.int64).tolist()
                ),
            },
            "draft": {
                "before_length": _cache_length(draft_before),
                "before_sha256": _cache_digest(draft_before),
                "tentative_length": _cache_length(draft_tentative),
                "tentative_sha256": _cache_digest(draft_tentative),
                "committed_length": _cache_length(draft_committed),
                "committed_sha256": _cache_digest(draft_committed),
                "replay_length": _cache_length(draft_replay),
                "replay_sha256": _cache_digest(draft_replay),
                "replay_max_abs": _cache_max_abs(draft_committed, draft_replay),
            },
        }

    def _run_dflash(self, input_ids: np.ndarray, count: int) -> dict[str, Any]:
        if self.embedding is None:
            raise AssertionError("DFlash requires target embedding component")
        target_cache = _empty_cache(self.target)
        outputs, target_cache = self._run_target(input_ids, target_cache)
        generated = [int(np.argmax(outputs["logits"][0, -1]))]
        target_hidden = self._features(outputs)
        draft_cache = _empty_cache(self.draft)
        mask_id = self.contract["mask_token_id"]
        block_size = self.contract["block_size"]
        if not isinstance(mask_id, int) or not isinstance(block_size, int):
            raise AssertionError("DFlash raw GGUF contract is incomplete")
        rounds = []
        while len(generated) < count:
            start = input_ids.shape[1] + len(generated) - 1
            target_before = target_cache
            draft_before = draft_cache
            block = np.array(
                [[generated[-1], *([mask_id] * (block_size - 1))]],
                dtype=np.int64,
            )
            noise = _outputs(self.embedding, {"input_ids": block})["inputs_embeds"]
            past = _cache_length(draft_before)
            positions = np.arange(past, start + block_size, dtype=np.int64)[None, :]
            draft_outputs = _outputs(
                self.draft,
                {
                    "noise_embedding": noise,
                    "target_hidden": target_hidden,
                    "position_ids": positions,
                    "q_position_ids": positions[:, -block_size:],
                    **draft_before,
                },
            )
            draft_tentative = _present(draft_outputs, len(draft_before) // 2)
            proposal_logits = self._draft_logits(draft_outputs)[:, -block_size + 1 :]
            proposal_ids = np.argmax(proposal_logits, axis=-1).astype(np.int64)[0]
            proposals = self._map_proposals(proposal_ids)
            verified, target_tentative = self._run_target(
                [[generated[-1], *proposals]],
                target_before,
            )
            posterior = np.argmax(verified["logits"], axis=-1).astype(np.int64)[0]
            accepted = next(
                (
                    index
                    for index, (proposal, token) in enumerate(
                        zip(proposals, posterior[:-1])
                    )
                    if proposal != int(token)
                ),
                len(proposals),
            )
            correction = int(posterior[accepted])
            committed_inputs = [[generated[-1], *proposals[:accepted]]]
            replay_outputs, target_replay = self._run_target(
                committed_inputs,
                target_before,
            )
            target_committed = _crop(
                target_tentative,
                start + accepted + 1,
            )
            draft_committed = _crop(draft_tentative, start)
            replay_draft_outputs = _outputs(
                self.draft,
                {
                    "noise_embedding": noise,
                    "target_hidden": target_hidden,
                    "position_ids": positions,
                    "q_position_ids": positions[:, -block_size:],
                    **draft_before,
                },
            )
            draft_replay = _crop(
                _present(replay_draft_outputs, len(draft_before) // 2),
                start,
            )
            _assert_cache_equal(draft_committed, draft_replay)
            replay_tokens = (
                np.argmax(replay_outputs["logits"], axis=-1).astype(np.int64)[0].tolist()
            )
            if replay_tokens != posterior[: accepted + 1].astype(np.int64).tolist():
                raise AssertionError("target replay tokens differ from verification")
            rounds.append(
                self._round(
                    proposal_ids=proposal_ids.tolist(),
                    proposals=proposals,
                    proposal_logits=proposal_logits,
                    posterior=posterior,
                    accepted=accepted,
                    correction=correction,
                    target_before=target_before,
                    target_tentative=target_tentative,
                    target_committed=target_committed,
                    target_replay=target_replay,
                    target_replay_logits=replay_outputs["logits"],
                    target_replay_tokens=replay_tokens,
                    draft_before=draft_before,
                    draft_tentative=draft_tentative,
                    draft_committed=draft_committed,
                    draft_replay=draft_replay,
                )
            )
            generated.extend(proposals[:accepted])
            generated.append(correction)
            target_cache = target_committed
            draft_cache = draft_committed
            target_hidden = self._features(verified, slice(0, accepted + 1))
        return self._finish(generated, count, rounds, target_cache, draft_cache)

    def _run_eagle_step(
        self,
        tokens: np.ndarray,
        features: np.ndarray,
        recycled: np.ndarray,
        cache: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        tokens = np.asarray(tokens, dtype=np.int64)
        names = {value.name for value in self.draft.get_inputs()}
        if "input_ids" in names:
            token_feed = {"input_ids": tokens}
        else:
            if self.embedding is None:
                raise AssertionError("shared EAGLE embedding component missing")
            token_feed = {
                "inputs_embeds": _outputs(
                    self.embedding, {"input_ids": tokens}
                )["inputs_embeds"]
            }
        past = _cache_length(cache)
        feeds = {
            **token_feed,
            "fused_hidden": features,
            "recycled_hidden": recycled,
            **cache,
        }
        if "attention_mask" in names:
            feeds["attention_mask"] = np.ones((1, past + tokens.shape[1]), dtype=np.int64)
        if "position_ids" in names:
            feeds["position_ids"] = np.arange(
                past, past + tokens.shape[1], dtype=np.int64
            )[None, :]
        outputs = _outputs(self.draft, feeds)
        return outputs, _present(outputs, len(cache) // 2)

    def _run_eagle3(self, input_ids: np.ndarray, count: int) -> dict[str, Any]:
        target_cache = _empty_cache(self.target)
        outputs, target_cache = self._run_target(input_ids, target_cache)
        generated = [int(np.argmax(outputs["logits"][0, -1]))]
        all_features = self._features(outputs)
        draft_cache = _empty_cache(self.draft)
        hidden = next(
            int(value.shape[-1])
            for value in self.draft.get_inputs()
            if value.name == "recycled_hidden"
        )
        if input_ids.shape[1] > 1:
            _, draft_cache = self._run_eagle_step(
                input_ids[:, 1:],
                all_features[:, :-1],
                np.zeros((1, input_ids.shape[1] - 1, hidden), dtype=np.float32),
                draft_cache,
            )
        pending = all_features[:, -1:]
        rounds = []
        while len(generated) < count:
            start = input_ids.shape[1] + len(generated) - 1
            target_before = target_cache
            draft_before = draft_cache
            proposals, proposal_ids, proposal_logits_rows = [], [], []
            tentative = draft_before
            recycled = np.zeros((1, 1, hidden), dtype=np.float32)
            token = generated[-1]
            for step in range(4):
                features = (
                    pending
                    if step == 0
                    else np.zeros((1, 1, pending.shape[-1]), dtype=np.float32)
                )
                draft_outputs, tentative = self._run_eagle_step(
                    [[token]], features, recycled, tentative
                )
                logits = self._draft_logits(draft_outputs)
                proposal_logits_rows.append(logits[:, -1:])
                draft_id = int(np.argmax(logits[0, -1]))
                proposal_ids.append(draft_id)
                token = self._map_proposals(np.array([draft_id]))[0]
                proposals.append(token)
                recycled = draft_outputs["next_hidden"][:, -1:]
            proposal_logits = np.concatenate(proposal_logits_rows, axis=1)
            verified, target_tentative = self._run_target(
                [[generated[-1], *proposals]],
                target_before,
            )
            posterior = np.argmax(verified["logits"], axis=-1).astype(np.int64)[0]
            accepted = next(
                (
                    index
                    for index, (proposal, token) in enumerate(
                        zip(proposals, posterior[:-1])
                    )
                    if proposal != int(token)
                ),
                len(proposals),
            )
            correction = int(posterior[accepted])
            committed_inputs = [[generated[-1], *proposals[:accepted]]]
            replay_outputs, target_replay = self._run_target(
                committed_inputs,
                target_before,
            )
            target_committed = _crop(target_tentative, start + accepted + 1)
            accepted_features = (
                np.concatenate(
                    [pending, self._features(verified, slice(0, accepted))],
                    axis=1,
                )
                if accepted
                else pending
            )
            _, draft_committed = self._run_eagle_step(
                np.array(committed_inputs, dtype=np.int64),
                accepted_features,
                np.zeros((1, accepted + 1, hidden), dtype=np.float32),
                draft_before,
            )
            _, draft_replay = self._run_eagle_step(
                np.array(committed_inputs, dtype=np.int64),
                accepted_features,
                np.zeros((1, accepted + 1, hidden), dtype=np.float32),
                draft_before,
            )
            _assert_cache_equal(draft_committed, draft_replay)
            replay_tokens = (
                np.argmax(replay_outputs["logits"], axis=-1).astype(np.int64)[0].tolist()
            )
            if replay_tokens != posterior[: accepted + 1].astype(np.int64).tolist():
                raise AssertionError("target replay tokens differ from verification")
            rounds.append(
                self._round(
                    proposal_ids=proposal_ids,
                    proposals=proposals,
                    proposal_logits=proposal_logits,
                    posterior=posterior,
                    accepted=accepted,
                    correction=correction,
                    target_before=target_before,
                    target_tentative=target_tentative,
                    target_committed=target_committed,
                    target_replay=target_replay,
                    target_replay_logits=replay_outputs["logits"],
                    target_replay_tokens=replay_tokens,
                    draft_before=draft_before,
                    draft_tentative=tentative,
                    draft_committed=draft_committed,
                    draft_replay=draft_replay,
                )
            )
            generated.extend(proposals[:accepted])
            generated.append(correction)
            target_cache = target_committed
            draft_cache = draft_committed
            pending = self._features(verified, slice(accepted, accepted + 1))
        return self._finish(generated, count, rounds, target_cache, draft_cache)

    @staticmethod
    def _finish(
        generated: list[int],
        count: int,
        rounds: list[dict[str, Any]],
        target_cache: dict[str, np.ndarray],
        draft_cache: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        tokens = generated[:count]
        return {
            "generated_tokens": tokens,
            "generated_tokens_sha256": hashlib.sha256(
                json.dumps(tokens, separators=(",", ":")).encode()
            ).hexdigest(),
            "rounds": rounds,
            "counters": {
                "rounds": len(rounds),
                "proposed": sum(len(record["proposal_tokens"]) for record in rounds),
                "accepted": sum(record["accepted_prefix"] for record in rounds),
                "multi_token_rounds": sum(
                    record["accepted_prefix"] > 1 for record in rounds
                ),
                "rejections": sum(
                    record["accepted_prefix"] < len(record["proposal_tokens"])
                    for record in rounds
                ),
            },
            "final_target_cache": {
                "length": _cache_length(target_cache),
                "sha256": _cache_digest(target_cache),
            },
            "final_draft_cache": {
                "length": _cache_length(draft_cache),
                "sha256": _cache_digest(draft_cache),
            },
            "reorder": {
                "supported": False,
                "reason": "reference coordinator is batch-size 1",
            },
        }


def assert_trace_matches(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    """Compare every transition field with one actionable mismatch path."""

    def compare(left: Any, right: Any, path: str) -> None:
        if isinstance(right, dict):
            if not isinstance(left, dict):
                raise AssertionError(f"{path}: expected object")
            for key, value in right.items():
                if key not in left:
                    raise AssertionError(f"{path}.{key}: missing")
                compare(left[key], value, f"{path}.{key}")
            return
        if isinstance(right, list):
            if not isinstance(left, list) or len(left) != len(right):
                raise AssertionError(f"{path}: list length/type mismatch")
            for index, value in enumerate(right):
                compare(left[index], value, f"{path}[{index}]")
            return
        if left != right:
            raise AssertionError(f"{path}: expected {right!r}, got {left!r}")

    compare(actual, expected, "trace")
