# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Metadata-driven reference runtime for non-generative CTC ASR packages.

This module executes an exported package using *only* the facts written into
``inference_metadata.yaml``.  Nothing here inspects the model architecture, the
checkpoint, or the model id — every decision (sample rate, normalization,
tensor names, blank id, time axis, vocabulary) is read from the document.  It
therefore doubles as an executable check that the emitted contract is complete:
if a fact is missing from the metadata, this runtime cannot produce a
transcript.

The pipeline is frame-synchronous, not autoregressive:

    encoded audio → preprocessing program → encoder (one invocation)
                  → per-frame argmax → collapse repeats → drop blank → text

Batched requests are segmented with the ``frame_lengths`` output bound by the
profile's ``decoding.lengths`` role, so a padded batch yields the same
per-row transcript as an unpadded single-row run.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import yaml


class MetadataContractError(ValueError):
    """Raised when the metadata document lacks a fact the runtime needs."""


def load_metadata(package_dir: str, filename: str = "inference_metadata.yaml") -> dict:
    """Load the one-file inference metadata document from *package_dir*."""
    path = os.path.join(package_dir, filename)
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def select_profile(metadata: dict, kind: str) -> tuple[str, dict]:
    """Return the ``(name, profile)`` of the single profile with *kind*.

    A reader that does not understand a profile may skip it only when the
    profile is ``ignorable``; a missing required profile is a contract error.
    """
    matches = [
        (name, profile)
        for name, profile in (metadata.get("profiles") or {}).items()
        if profile.get("kind") == kind
    ]
    if not matches:
        raise MetadataContractError(f"metadata declares no '{kind}' profile")
    if len(matches) > 1:
        raise MetadataContractError(
            f"metadata declares {len(matches)} '{kind}' profiles; expected exactly one"
        )
    return matches[0]


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Linearly resample a mono waveform; a no-op when the rates already agree."""
    if source_rate == target_rate:
        return samples
    duration = samples.shape[-1] / float(source_rate)
    target_length = round(duration * target_rate)
    source_positions = np.arange(samples.shape[-1], dtype=np.float64)
    target_positions = np.linspace(0.0, samples.shape[-1] - 1, target_length, dtype=np.float64)
    return np.interp(target_positions, source_positions, samples).astype(np.float32)


def run_audio_preprocessing(
    program: dict,
    waveforms: list[np.ndarray],
    sample_rate: int,
) -> dict[str, np.ndarray]:
    """Execute the declared audio preprocessing program over a batch.

    Args:
        program: The ``preprocessing.audio`` sub-document.
        waveforms: One 1-D float array per request row, at *sample_rate*.
        sample_rate: Sample rate of the supplied waveforms.

    Returns:
        A mapping from workflow SSA value name to the tensor bound to it, using
        the program's ``outputs`` bindings.

    Every transform is dispatched by its generic ``op`` name; an unknown op is
    an error rather than a silent skip, because skipping a normalization step
    would produce plausible-but-wrong logits.
    """
    rows = [np.asarray(row, dtype=np.float32).reshape(-1) for row in waveforms]
    target_rate = sample_rate
    pad_value = 0.0
    pad_side = "right"

    for transform in program.get("transforms", []):
        op = transform.get("op")
        if op == "decode":
            # The caller already decoded the container; the declared step exists
            # so a runtime that receives raw bytes knows one is required.
            pass
        elif op == "resample":
            target_rate = int(transform["sample_rate"])
            rows = [_resample(row, sample_rate, target_rate) for row in rows]
        elif op == "downmix":
            channels = int(transform.get("channels", 1))
            if channels != 1:
                raise MetadataContractError(
                    f"reference runtime only downmixes to mono, got {channels}"
                )
        elif op == "rescale":
            scale = float(transform["scale"])
            rows = [row * scale for row in rows]
        elif op == "zero_mean_unit_variance":
            epsilon = float(transform.get("epsilon", 0.0))
            # Normalized per row, matching how a feature extractor treats each
            # utterance independently — a batch-wide statistic would make a
            # padded batch disagree with a single-row run.
            rows = [(row - row.mean()) / np.sqrt(row.var() + epsilon) for row in rows]
        elif op == "pad":
            pad_value = float(transform.get("pad_value", 0.0))
            pad_side = transform.get("mode", "right")
        elif op == "trim":
            target_length = int(transform["target_length"])
            rows = [row[:target_length] for row in rows]
        else:
            raise MetadataContractError(f"unsupported audio transform op '{op}'")

    if pad_side != "right":
        raise MetadataContractError(
            f"reference runtime only pads on the right, got '{pad_side}'"
        )

    max_length = max(row.shape[0] for row in rows)
    values = np.full((len(rows), max_length), pad_value, dtype=np.float32)
    mask = np.zeros((len(rows), max_length), dtype=np.int64)
    for index, row in enumerate(rows):
        values[index, : row.shape[0]] = row
        mask[index, : row.shape[0]] = 1

    produced = {"values": values, "sample_mask": mask, "samples": values}
    bound: dict[str, np.ndarray] = {}
    for binding in program.get("outputs", []):
        source = binding["source"]
        if source not in produced:
            raise MetadataContractError(
                f"audio program output '{binding['name']}' reads undeclared value '{source}'"
            )
        tensor = produced[source]
        dtype = binding.get("dtype")
        if dtype:
            tensor = tensor.astype(_numpy_dtype(dtype), copy=False)
        if int(binding.get("rank", tensor.ndim)) != tensor.ndim:
            raise MetadataContractError(
                f"audio program output '{binding['name']}' declares rank "
                f"{binding['rank']} but produced rank {tensor.ndim}"
            )
        bound[binding["name"]] = tensor
    return bound


def _numpy_dtype(name: str) -> Any:
    mapping = {
        "float32": np.float32,
        "float16": np.float16,
        "int64": np.int64,
        "int32": np.int32,
        "uint8": np.uint8,
        "bool": np.bool_,
    }
    if name not in mapping:
        raise MetadataContractError(f"unsupported tensor dtype '{name}'")
    return mapping[name]


def run_workflow(
    metadata: dict,
    package_dir: str,
    preprocessed: dict[str, np.ndarray],
    *,
    providers: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Execute the declared workflow steps and return its emitted outputs.

    Only ``invoke`` and ``emit`` steps are supported: a CTC workflow is a plain
    sequence with no loop, no branch and no carried state.  Encountering a loop
    here means the package was mis-detected as frame-synchronous.
    """
    import onnxruntime as ort

    workflow = metadata["pipeline"]["workflow"]
    components = workflow["components"]

    # The preprocessing adapter's outputs are already materialized by the
    # caller, so its invocation binds names rather than running a session.
    ssa: dict[str, np.ndarray] = {}
    sessions: dict[str, Any] = {}
    emitted: dict[str, np.ndarray] = {}

    for step in workflow["steps"]:
        kind = step.get("kind")
        if kind == "invoke":
            name = step["component"]
            component = components[name]
            implementation = component["implementation"]
            if implementation["kind"] == "adapter":
                for port, target in step["outputs"].items():
                    if port not in preprocessed:
                        raise MetadataContractError(
                            f"adapter '{name}' output '{port}' was not produced by "
                            "the preprocessing program"
                        )
                    ssa[target] = preprocessed[port]
                continue
            if implementation["kind"] != "onnx":
                raise MetadataContractError(
                    f"component '{name}' uses unsupported implementation "
                    f"'{implementation['kind']}'"
                )
            if name not in sessions:
                sessions[name] = ort.InferenceSession(
                    os.path.join(package_dir, implementation["artifact"]),
                    providers=providers or ["CPUExecutionProvider"],
                )
            session = sessions[name]
            feed = {port: ssa[source] for port, source in step["inputs"].items()}
            requested = list(step["outputs"].keys())
            results = session.run(requested, feed)
            for port, value in zip(requested, results):
                ssa[step["outputs"][port]] = value
        elif kind == "emit":
            emitted[step["output"]] = ssa[step["value"]]
        else:
            raise MetadataContractError(
                f"CTC workflow contains unsupported step kind '{kind}'; a "
                "frame-synchronous package must be a plain sequence"
            )
    return emitted


def collapse_ctc(ids: list[int], *, blank_id: int, collapse_repeats: bool) -> list[int]:
    """Collapse a frame-argmax id sequence into CTC output tokens.

    Repeats collapse *before* blanks are removed, which is what makes a doubled
    letter representable: two identical letters separated by a blank survive as
    two tokens, while a letter held over several frames becomes one.
    """
    collapsed: list[int] = []
    previous: int | None = None
    for token in ids:
        if collapse_repeats and token == previous:
            continue
        previous = token
        if token == blank_id:
            continue
        collapsed.append(token)
    return collapsed


def decode_transcripts(
    metadata: dict,
    outputs: dict[str, np.ndarray],
    *,
    profile_kind: str = "transcription",
) -> dict[str, Any]:
    """Turn emitted workflow outputs into per-row transcripts.

    Returns a dict with ``argmax_ids``, ``collapsed_ids`` and ``transcripts``,
    one entry per batch row, so a caller can compare any decode stage against a
    reference implementation.
    """
    _, profile = select_profile(metadata, profile_kind)
    decoding = profile.get("decoding")
    if decoding is None:
        raise MetadataContractError(f"'{profile_kind}' profile declares no decoding contract")
    if decoding.get("kind") != "ctc":
        raise MetadataContractError(
            f"reference runtime decodes 'ctc' only, got '{decoding.get('kind')}'"
        )

    logits_output = profile["outputs"].get("logits")
    if logits_output is None:
        raise MetadataContractError("transcription profile binds no 'logits' output")
    logits = outputs[logits_output]

    time_axis = int(decoding["time_axis"])
    class_axis = int(decoding["class_axis"])
    if time_axis == class_axis:
        raise MetadataContractError("decoding time_axis and class_axis must differ")
    blank_id = int(decoding["blank_id"])
    collapse_repeats = bool(decoding.get("collapse_repeats", False))

    frame_ids = np.argmax(logits, axis=class_axis)  # (batch, frames)
    if time_axis != 1:
        frame_ids = np.moveaxis(frame_ids, time_axis - (time_axis > class_axis), -1)

    lengths_role = decoding.get("lengths")
    if lengths_role is not None:
        lengths_output = profile["outputs"].get(lengths_role)
        if lengths_output is None:
            raise MetadataContractError(
                f"decoding.lengths references role '{lengths_role}' that the "
                "profile does not bind"
            )
        lengths = np.asarray(outputs[lengths_output]).reshape(-1).astype(int)
    else:
        lengths = np.full(frame_ids.shape[0], frame_ids.shape[-1], dtype=int)

    vocabulary = decoding.get("vocabulary") or {}
    tokens = list(vocabulary.get("tokens") or [])
    ignored = set(vocabulary.get("ignored_tokens") or [])
    delimiter = vocabulary.get("word_delimiter")

    argmax_ids: list[list[int]] = []
    collapsed_ids: list[list[int]] = []
    transcripts: list[str] = []
    for row in range(frame_ids.shape[0]):
        valid = int(min(lengths[row], frame_ids.shape[-1]))
        row_ids = frame_ids[row, :valid].tolist()
        argmax_ids.append(row_ids)
        collapsed = collapse_ctc(row_ids, blank_id=blank_id, collapse_repeats=collapse_repeats)
        collapsed_ids.append(collapsed)
        if tokens:
            pieces = [
                tokens[token_id] for token_id in collapsed if tokens[token_id] not in ignored
            ]
            if delimiter:
                # The delimiter separates words rather than emitting whitespace,
                # so empty groups from leading/trailing/repeated delimiters are
                # dropped instead of becoming stray spaces.
                words = "".join(pieces).split(delimiter)
                transcripts.append(" ".join(word for word in words if word))
            else:
                transcripts.append("".join(pieces))
        else:
            transcripts.append("")
    return {
        "argmax_ids": argmax_ids,
        "collapsed_ids": collapsed_ids,
        "transcripts": transcripts,
    }


def transcribe(
    package_dir: str,
    waveforms: list[np.ndarray],
    sample_rate: int,
    *,
    providers: list[str] | None = None,
) -> dict[str, Any]:
    """Run the full metadata-driven CTC pipeline over a batch of waveforms."""
    metadata = load_metadata(package_dir)
    program = (metadata.get("preprocessing") or {}).get("audio")
    if program is None:
        raise MetadataContractError("metadata declares no preprocessing.audio program")
    preprocessed = run_audio_preprocessing(program, waveforms, sample_rate)
    outputs = run_workflow(metadata, package_dir, preprocessed, providers=providers)
    result = decode_transcripts(metadata, outputs)
    result["logits"] = outputs[
        select_profile(metadata, "transcription")[1]["outputs"]["logits"]
    ]
    return result
