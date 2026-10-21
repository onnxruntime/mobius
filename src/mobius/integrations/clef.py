# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Host-side Clef schema encoding and flat option tensor contracts.

Records are processed individually, without padding: the final token is the
global evidence anchor and every question attends to the complete record.
Image/video processing remains with the pinned Hugging Face processor.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import numpy as np

_SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)
_TYPES = {"noul": 0, "choice": 1, "score": 2}


def _render(value: Any) -> str:
    return (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    )


def _options(question: dict[str, Any]) -> list[tuple[str, Any]]:
    kind = question["type"]
    if kind == "noul":
        boolean_criteria = {
            "true": "The proposition is true or the answer is yes.",
            "false": "The proposition is false or the answer is no.",
        }
        criteria = question.get("criteria")
        if criteria is not None and not isinstance(criteria, dict):
            raise ValueError("Clef boolean criteria must be a mapping")
        boolean_criteria.update(criteria or {})
        return [(key, boolean_criteria[key]) for key in ("true", "false")]
    criteria = question.get("criteria")
    if kind == "choice" and isinstance(criteria, dict) and criteria:
        options = [(str(key), value) for key, value in criteria.items()]
        if len({key for key, _ in options}) != len(options):
            raise ValueError("Clef option labels must be unique after string conversion")
        return sorted(options, key=lambda option: option[0])
    if kind == "score" and isinstance(criteria, list) and criteria:
        return [(str(index), value) for index, value in enumerate(criteria)]
    raise ValueError(f"Clef {kind!r} questions require nonempty, correctly typed criteria")


@dataclasses.dataclass(frozen=True)
class ClefRecord:
    """Encoded prompt, decision feeds, processor media, and output labels."""

    input_ids: np.ndarray
    question_spans: np.ndarray
    option_spans: np.ndarray
    option_question_ids: np.ndarray
    question_types: np.ndarray
    question_ids: tuple[str, ...]
    option_ids: tuple[tuple[str, ...], ...]
    media: dict[str, Any]

    def decision_feeds(self, hidden_states: np.ndarray) -> dict[str, np.ndarray]:
        """Validate the hidden-state boundary and prepare decision-head inputs."""
        if hidden_states.ndim != 3 or hidden_states.shape[:2] != self.input_ids.shape:
            raise ValueError("Clef hidden states must have shape [1, prompt_length, hidden]")
        return {
            "hidden_states": hidden_states,
            "input_ids": self.input_ids,
            "question_spans": self.question_spans,
            "option_spans": self.option_spans,
            "option_question_ids": self.option_question_ids,
            "question_types": self.question_types,
        }

    def probabilities_by_question(
        self,
        probabilities: np.ndarray,
    ) -> dict[str, dict[str, float]]:
        """Restore question/option labels from the flat ONNX probability output."""
        if probabilities.shape != (len(self.option_spans),):
            raise ValueError("Clef probabilities must contain exactly one value per option")
        if not np.all(np.isfinite(probabilities)) or np.any(
            (probabilities < 0) | (probabilities > 1)
        ):
            raise ValueError("Clef probabilities must be finite and between zero and one")
        result = {}
        offset = 0
        for question, options in zip(self.question_ids, self.option_ids, strict=True):
            values = probabilities[offset : offset + len(options)]
            if not np.isclose(values.sum(), 1.0, atol=1e-3):
                raise ValueError(f"Clef probabilities for {question!r} do not sum to one")
            result[question] = dict(zip(options, map(float, values), strict=True))
            offset += len(options)
        return result


def encode_clef_record(
    tokenizer: Any,
    record: dict[str, Any],
    *,
    processor: Any | None = None,
    max_length: int = 16384,
    max_state_tokens: int | None = None,
) -> ClefRecord:
    """Encode the publisher's prompt and span semantics without remote execution.

    ``record`` contains ``state`` and a nonempty ``questions`` mapping. Supported
    question types are ``noul`` (boolean), ``choice`` (mapping of options), and
    ``score`` (list of levels). Optional images/videos use ``processor``;
    its packed pixels, grids, and token metadata are returned in ``media``.
    Only state text is truncated; schema spans and media tokens are preserved.
    """
    if max_length < 1 or (max_state_tokens is not None and max_state_tokens < 0):
        raise ValueError("Clef token limits must be positive (state limit may be zero)")
    questions = record.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("Clef requires a nonempty questions mapping")
    if "state" not in record:
        raise ValueError("Clef requires a state")
    if len({str(key) for key in questions}) != len(questions):
        raise ValueError("Clef question labels must be unique after string conversion")

    def tokens(text: str) -> list[int]:
        return list(tokenizer(text, add_special_tokens=False).input_ids)

    schema = tokens("\n\nSCHEMA FIELDS:\n")
    question_spans, option_spans, owners, types, labels = [], [], [], [], []
    for index, (question_id, question) in enumerate(questions.items()):
        if not isinstance(question, dict) or question.get("type") not in _TYPES:
            raise ValueError(f"Invalid Clef question type for {question_id!r}")
        kind = question["type"]
        schema.extend(
            tokens(f"\nFIELD {index + 1}\nID: {question_id}\nTYPE: {kind}\nINSTRUCTION: ")
        )
        start = len(schema)
        instructions = question.get("instructions")
        schema.extend(
            tokens(
                _render(
                    str(question_id)
                    if instructions is None or instructions == ""
                    else instructions
                )
            )
        )
        question_spans.append((start, len(schema)))
        schema.extend(tokens("\nALLOWED OPTIONS:\n"))
        options = _options(question)
        labels.append(tuple(key for key, _ in options))
        types.append(_TYPES[kind])
        for option_index, (option_id, description) in enumerate(options):
            schema.extend(tokens(f"OPTION {option_index + 1}: "))
            start = len(schema)
            semantics = {"option_id": option_id}
            if description is not None:
                semantics["description"] = description
            schema.extend(tokens(_render(semantics)))
            option_spans.append((start, len(schema)))
            owners.append(index)
            schema.extend(tokens("\n"))
        schema.extend(tokens("END FIELD\n"))
    if any(end <= start for start, end in question_spans + option_spans):
        raise ValueError("Clef question and option spans must contain at least one token")
    prefix = tokens(
        f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
    )
    media = {}
    images, videos = record.get("images") or [], record.get("videos") or []
    if images or videos:
        if processor is None:
            raise ValueError("Clef image/video records require a processor")
        text = (
            "<|vision_start|><|image_pad|><|vision_end|>" * len(images)
            + "<|vision_start|><|video_pad|><|vision_end|>" * len(videos)
            + "\n"
        )
        processed = processor(
            text=[text],
            images=images or None,
            videos=videos or None,
            return_tensors="np",
            **(record.get("media_kwargs") or {}),
        )
        offset = len(prefix)
        media_ids = processed["input_ids"][0].tolist()
        prefix.extend(media_ids)
        media = {
            key: value
            for key, value in processed.items()
            if key not in {"input_ids", "attention_mask"}
        }
        if "mm_token_type_ids" in media:
            media["mm_token_type_ids"] = np.pad(
                np.asarray(media["mm_token_type_ids"]), ((0, 0), (offset, 0))
            )
    suffix = tokens(
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"
    )
    fixed = len(prefix) + len(schema) + len(suffix)
    if fixed > max_length:
        raise ValueError(f"Clef schema/media requires {fixed} tokens; maximum is {max_length}")
    state = tokens(_render(record["state"]))
    if max_state_tokens is not None:
        state = state[:max_state_tokens]
    state = state[: max_length - fixed]
    offset = len(prefix) + len(state)
    ids = np.array([prefix + state + schema + suffix], dtype=np.int64)
    if "mm_token_type_ids" in media:
        media["mm_token_type_ids"] = np.pad(
            media["mm_token_type_ids"],
            ((0, 0), (0, ids.shape[1] - media["mm_token_type_ids"].shape[1])),
        )
    return ClefRecord(
        input_ids=ids,
        question_spans=np.array(question_spans, dtype=np.int64) + offset,
        option_spans=np.array(option_spans, dtype=np.int64) + offset,
        option_question_ids=np.array(owners, dtype=np.int64),
        question_types=np.array(types, dtype=np.int64),
        question_ids=tuple(map(str, questions)),
        option_ids=tuple(labels),
        media=media,
    )
