# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Direct ONNX Runtime coordinator for target-coupled GGUF draft packages."""

from __future__ import annotations

__all__ = ["DraftGenerationResult", "DraftGenerationStats", "DraftPairRunner"]

import dataclasses
import errno
import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from typing_extensions import Self

from mobius.integrations.gguf._reader import _descriptor_identity


@dataclasses.dataclass(frozen=True, slots=True)
class DraftGenerationStats:
    """Observable work and rollback counters from one speculative generation."""

    rounds: int
    proposed_tokens: int
    accepted_tokens: int
    multi_token_rounds: int
    rollback_events: tuple[tuple[int, int, int, int], ...]
    target_forwards: int
    draft_forwards: int
    beam_reorder_supported: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class DraftGenerationResult:
    """Generated target-vocabulary tokens and their speculative work counters."""

    tokens: tuple[int, ...]
    stats: DraftGenerationStats


def _read_bounded_json(
    package_dir: Path,
    filename: str,
    *,
    limit: int,
) -> dict[str, Any]:
    if package_dir.is_symlink() or not package_dir.is_dir():
        raise ValueError("draft package root must be a real directory")
    path = package_dir / filename
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"draft package requires a regular {filename}") from error
    try:
        before = _descriptor_identity(descriptor)
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError(f"{filename} must be a regular file")
        if file_stat.st_size > limit:
            raise ValueError(f"{filename} exceeds its {limit}-byte limit")
        chunks = []
        remaining = file_stat.st_size + 1
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) != file_stat.st_size:
            raise ValueError(f"{filename} changed size while it was read")
        if _descriptor_identity(descriptor) != before:
            raise ValueError(f"{filename} changed while it was read")
        current = os.open(path, flags)
        try:
            if _descriptor_identity(current) != before:
                raise ValueError(f"{filename} was replaced while it was read")
        finally:
            os.close(current)
    finally:
        os.close(descriptor)
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise TypeError(f"{filename} root must be an object")
    return value


def _read_manifest(package_dir: Path) -> tuple[dict[str, Any], tuple[str, ...], str]:
    value = _read_bounded_json(
        package_dir,
        "draft_manifest.json",
        limit=8 * 1024 * 1024,
    )
    if not isinstance(value, dict) or value.get("kind") != "speculative-draft":
        raise ValueError("draft_manifest.json is not a speculative-draft manifest")
    graph_package = value.get("graph_package")
    if not isinstance(graph_package, dict):
        raise TypeError("draft_manifest.json has no graph_package identity")
    files = graph_package.get("files")
    sha256 = graph_package.get("sha256")
    if (
        not isinstance(files, list)
        or any(not isinstance(name, str) for name in files)
        or not isinstance(sha256, str)
    ):
        raise ValueError("draft_manifest.json has an invalid graph_package identity")
    selected = tuple(files)
    if selected != tuple(sorted(selected)) or len(selected) != len(set(selected)):
        raise ValueError("draft graph_package files must be sorted and unique")
    if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
        raise ValueError("draft graph_package SHA-256 must be lowercase hexadecimal")
    return value, selected, sha256


def _read_runtime_status(package_dir: Path) -> tuple[dict[str, Any], tuple[str, ...], str]:
    value = _read_bounded_json(
        package_dir,
        "draft_runtime_status.json",
        limit=1024 * 1024,
    )
    if value.get("schema_version") != 1 or value.get("status") != "runtime_unvalidated":
        raise ValueError("draft_runtime_status.json has an invalid status contract")
    runtime_payload = value.get("runtime_payload")
    if not isinstance(runtime_payload, dict):
        raise TypeError("draft runtime status has no runtime_payload identity")
    files = runtime_payload.get("files")
    sha256 = runtime_payload.get("sha256")
    if (
        not isinstance(files, list)
        or any(not isinstance(name, str) for name in files)
        or not isinstance(sha256, str)
        or runtime_payload.get("excludes") != "draft_runtime_status.json"
    ):
        raise ValueError("draft runtime status has an invalid runtime_payload identity")
    selected = tuple(files)
    if selected != tuple(sorted(selected)) or len(selected) != len(set(selected)):
        raise ValueError("draft runtime payload files must be sorted and unique")
    if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
        raise ValueError("draft runtime payload SHA-256 must be lowercase hexadecimal")
    return value, selected, sha256


def _clone_descriptor(source: int, destination: Path) -> bool:
    if sys.platform != "darwin":
        return False
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    fclonefileat = getattr(libc, "fclonefileat", None)
    if fclonefileat is None:
        return False
    fclonefileat.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    fclonefileat.restype = ctypes.c_int
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    directory = os.open(destination.parent, directory_flags)
    try:
        result = fclonefileat(source, directory, os.fsencode(destination.name), 0)
    finally:
        os.close(directory)
    if result == 0:
        return True
    error_number = ctypes.get_errno()
    if error_number in {
        errno.ENOTSUP,
        errno.EXDEV,
        errno.EINVAL,
        errno.ENOSYS,
        errno.EPERM,
    }:
        return False
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _copy_descriptor(source: int, destination: Path) -> None:
    if _clone_descriptor(source, destination):
        return
    destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    output = os.open(destination, destination_flags, 0o600)
    try:
        os.lseek(source, 0, os.SEEK_SET)
        while chunk := os.read(source, 8 * 1024 * 1024):
            view = memoryview(chunk)
            while view:
                written = os.write(output, view)
                view = view[written:]
    finally:
        os.close(output)


def _snapshot_graph_package(
    package_dir: Path,
    files: tuple[str, ...],
    expected_sha256: str,
    destination: Path,
) -> None:
    root = package_dir.resolve()
    for name in files:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("draft graph package path escapes its root")
        source_path = package_dir / relative
        current = package_dir
        for part in relative.parts[:-1]:
            current /= part
            current_stat = current.lstat()
            if current.is_symlink() or not stat.S_ISDIR(current_stat.st_mode):
                raise ValueError("draft graph package path traverses a non-directory or link")
        try:
            source_path.resolve().relative_to(root)
        except ValueError as error:
            raise ValueError("draft graph package path resolves outside its root") from error
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        source = os.open(source_path, flags)
        try:
            if not stat.S_ISREG(os.fstat(source).st_mode):
                raise ValueError("draft graph package entries must be regular files")
            destination_path = destination / relative
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            _copy_descriptor(source, destination_path)
        finally:
            os.close(source)
    from mobius.integrations.gguf._runtime_evidence import gguf_graph_package_identity

    identity = gguf_graph_package_identity(destination, files=files)
    if identity.sha256 != expected_sha256:
        raise ValueError(
            "draft package graph identity mismatch: "
            f"expected {expected_sha256}, got {identity.sha256}"
        )


def _external_data_files(
    model_path: Path,
    relative_model_path: Path,
) -> tuple[str, ...]:
    import onnx_ir as ir

    model = ir.load(model_path)
    external_files = set()
    tensors: list[Any] = [
        initializer.const_value
        for graph in model.graphs()
        for initializer in graph.initializers.values()
        if initializer.const_value is not None
    ]
    graph_nodes = [node for graph in model.graphs() for node in graph]
    function_nodes = [
        node
        for function in model.functions.values()
        for node in ir.traversal.RecursiveGraphIterator(function)
    ]

    def collect_attributes(attributes: Any) -> None:
        for attribute in attributes.values():
            if attribute.type == ir.AttributeType.TENSOR:
                tensors.append(attribute.as_tensor())
            elif attribute.type == ir.AttributeType.TENSORS:
                tensors.extend(attribute.as_tensors())
            elif attribute.type == ir.AttributeType.SPARSE_TENSOR:
                sparse = attribute.value
                tensors.extend((sparse.values, sparse.indices))
            elif attribute.type == ir.AttributeType.SPARSE_TENSORS:
                for sparse in attribute.value:
                    tensors.extend((sparse.values, sparse.indices))

    for node in (*graph_nodes, *function_nodes):
        collect_attributes(node.attributes)
    for function in model.functions.values():
        collect_attributes(function.attributes)
    for tensor in tensors:
        if not isinstance(tensor, ir.ExternalTensor):
            continue
        location = Path(tensor.location)
        if location.is_absolute() or ".." in location.parts:
            raise ValueError(
                f"draft component external data escapes its directory: {location}"
            )
        external_files.add((relative_model_path.parent / location).as_posix())
    return tuple(sorted(external_files))


def _session_dtype(session: Any, input_name: str) -> np.dtype:
    type_name = next(value.type for value in session.get_inputs() if value.name == input_name)
    if type_name == "tensor(float)":
        return np.dtype(np.float32)
    if type_name == "tensor(float16)":
        return np.dtype(np.float16)
    raise TypeError(f"Unsupported coordinator input type for {input_name!r}: {type_name}")


def _empty_cache(session: Any) -> dict[str, np.ndarray]:
    cache: dict[str, np.ndarray] = {}
    for value in session.get_inputs():
        if not value.name.startswith("past_key_values."):
            continue
        shape = [
            dimension if isinstance(dimension, int) else (0 if "past" in str(dimension) else 1)
            for dimension in value.shape
        ]
        cache[value.name] = np.empty(shape, dtype=_session_dtype(session, value.name))
    if not cache or len(cache) % 2:
        raise ValueError("draft coordinator requires complete dynamic key/value cache inputs")
    return cache


def _crop_cache(
    cache: dict[str, np.ndarray],
    sequence_length: int,
) -> dict[str, np.ndarray]:
    return {name: value[:, :, :sequence_length, :] for name, value in cache.items()}


class DraftPairRunner:
    """Run greedy DFlash or EAGLE3 with independent target and draft ORT sessions."""

    def __init__(
        self,
        package_dir: str | Path,
        *,
        providers: list[str] | None = None,
        session_options: Any | None = None,
    ):
        import onnxruntime as ort

        source_package_dir = Path(package_dir)
        self.runtime_status, runtime_files, runtime_sha256 = _read_runtime_status(
            source_package_dir
        )
        self._snapshot = tempfile.TemporaryDirectory(prefix="mobius-draft-pair-")
        self.package_dir = Path(self._snapshot.name)
        _snapshot_graph_package(
            source_package_dir,
            runtime_files,
            runtime_sha256,
            self.package_dir,
        )
        self.manifest, graph_files, graph_sha256 = _read_manifest(self.package_dir)
        graph_package = {
            "files": list(graph_files),
            "sha256": graph_sha256,
        }
        if self.runtime_status.get("graph_package") != graph_package:
            raise ValueError("draft manifest and runtime status graph identities disagree")
        config_hashes = self.runtime_status.get("config_sha256")
        if not isinstance(config_hashes, dict):
            raise TypeError("draft runtime status has no config_sha256 map")
        from mobius.integrations.gguf._runtime_evidence import (
            gguf_graph_package_identity,
        )
        from mobius.integrations.gguf._runtime_package import _sha256_file

        manifest_sha256 = _sha256_file(self.package_dir / "draft_manifest.json")
        if config_hashes.get("draft_manifest.json") != manifest_sha256:
            raise ValueError("draft manifest hash does not match runtime status")
        graph_identity = gguf_graph_package_identity(
            self.package_dir,
            files=graph_files,
        )
        if graph_identity.sha256 != graph_sha256:
            raise ValueError("draft graph package does not match its manifest identity")
        self._verified_files = frozenset(graph_files)
        components = self.manifest.get("components")
        if not isinstance(components, dict):
            raise TypeError("draft manifest has no component map")
        providers = providers or ["CPUExecutionProvider"]

        def load(name: str):
            record = components.get(name)
            if not isinstance(record, dict) or not isinstance(record.get("artifact"), str):
                raise TypeError(f"draft manifest has no {name!r} artifact")
            relative = Path(record["artifact"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"draft component path escapes the package: {relative}")
            relative_name = relative.as_posix()
            if relative_name not in self._verified_files:
                raise ValueError(
                    f"draft component is outside the verified graph package: {relative}"
                )
            path = self.package_dir / relative
            for external_file in _external_data_files(path, relative):
                if external_file not in self._verified_files:
                    raise ValueError(
                        "draft component external data is outside the verified graph "
                        f"package: {external_file}"
                    )
            return ort.InferenceSession(
                str(path),
                sess_options=session_options,
                providers=providers,
            )

        self.target = load("target")
        self.draft = load("draft")
        self.target_embedding = (
            load("target_embedding") if "target_embedding" in components else None
        )
        self.target_lm_head = (
            load("target_lm_head") if "target_lm_head" in components else None
        )
        self._target_forwards = 0
        self._draft_forwards = 0

    def close(self) -> None:
        """Release the private verified package snapshot."""
        self._snapshot.cleanup()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @staticmethod
    def _outputs(session: Any, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        names = [value.name for value in session.get_outputs()]
        return dict(zip(names, session.run(names, feeds)))

    @staticmethod
    def _present_cache(
        outputs: dict[str, np.ndarray],
        layer_count: int,
    ) -> dict[str, np.ndarray]:
        return {
            f"past_key_values.{layer}.{kind}": outputs[f"present.{layer}.{kind}"]
            for layer in range(layer_count)
            for kind in ("key", "value")
        }

    def _run_target(
        self,
        input_ids: np.ndarray,
        cache: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        input_ids = np.asarray(input_ids, dtype=np.int64)
        past_length = next(iter(cache.values())).shape[2]
        feeds = {"input_ids": input_ids, **cache}
        input_names = {value.name for value in self.target.get_inputs()}
        if "attention_mask" in input_names:
            feeds["attention_mask"] = np.ones(
                (1, past_length + input_ids.shape[1]),
                dtype=np.int64,
            )
        if "position_ids" in input_names:
            feeds["position_ids"] = np.arange(
                past_length,
                past_length + input_ids.shape[1],
                dtype=np.int64,
            )[None, :]
        outputs = self._outputs(self.target, feeds)
        self._target_forwards += 1
        layer_count = len(cache) // 2
        return outputs, self._present_cache(outputs, layer_count)

    def _target_features(
        self,
        outputs: dict[str, np.ndarray],
        selection: slice = slice(None),
    ) -> np.ndarray:
        layer_ids = self.manifest["target"]["target_layers"]
        return np.concatenate(
            [outputs[f"hidden_states.{index}"][:, selection] for index in layer_ids],
            axis=-1,
        )

    def _run_draft(
        self,
        feeds: dict[str, np.ndarray],
        cache: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        outputs = self._outputs(self.draft, {**feeds, **cache})
        self._draft_forwards += 1
        return outputs, self._present_cache(outputs, len(cache) // 2)

    def _project_draft_logits(
        self,
        outputs: dict[str, np.ndarray],
    ) -> np.ndarray:
        if "draft_logits" in outputs:
            return outputs["draft_logits"]
        if self.target_lm_head is None or "draft_hidden" not in outputs:
            raise ValueError("draft_hidden output requires a target LM-head bridge")
        return self._outputs(
            self.target_lm_head,
            {"hidden_states": outputs["draft_hidden"]},
        )["logits"]

    def generate_target_only(
        self,
        input_ids: np.ndarray,
        *,
        max_new_tokens: int,
    ) -> tuple[int, ...]:
        """Generate the deterministic target-only baseline."""
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        input_ids = np.asarray(input_ids, dtype=np.int64)
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("draft coordinator requires non-empty batch-size-1 input_ids")
        cache = _empty_cache(self.target)
        outputs, cache = self._run_target(input_ids, cache)
        generated = [int(np.argmax(outputs["logits"][0, -1]))]
        while len(generated) < max_new_tokens:
            outputs, cache = self._run_target(
                np.array([[generated[-1]]], dtype=np.int64),
                cache,
            )
            generated.append(int(np.argmax(outputs["logits"][0, -1])))
        return tuple(generated)

    def generate(
        self,
        input_ids: np.ndarray,
        *,
        max_new_tokens: int,
        max_draft_tokens: int | None = None,
    ) -> DraftGenerationResult:
        """Generate with the package's DFlash or EAGLE3 coordinator."""
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        input_ids = np.asarray(input_ids, dtype=np.int64)
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("draft coordinator requires non-empty batch-size-1 input_ids")
        self._target_forwards = 0
        self._draft_forwards = 0
        architecture = self.manifest["architecture"]
        if architecture == "dflash":
            return self._generate_dflash(input_ids, max_new_tokens)
        if architecture == "eagle3":
            width = 4 if max_draft_tokens is None else max_draft_tokens
            if width <= 0:
                raise ValueError("max_draft_tokens must be positive")
            return self._generate_eagle3(input_ids, max_new_tokens, width)
        raise ValueError(f"Unsupported draft architecture: {architecture!r}")

    def _result(
        self,
        generated: list[int],
        max_new_tokens: int,
        rounds: list[tuple[int, int]],
        rollbacks: list[tuple[int, int, int, int]],
    ) -> DraftGenerationResult:
        return DraftGenerationResult(
            tuple(generated[:max_new_tokens]),
            DraftGenerationStats(
                rounds=len(rounds),
                proposed_tokens=sum(proposed for _, proposed in rounds),
                accepted_tokens=sum(accepted for accepted, _ in rounds),
                multi_token_rounds=sum(accepted > 1 for accepted, _ in rounds),
                rollback_events=tuple(rollbacks),
                target_forwards=self._target_forwards,
                draft_forwards=self._draft_forwards,
            ),
        )

    def _finish_target_only(
        self,
        generated: list[int],
        target_cache: dict[str, np.ndarray],
        max_new_tokens: int,
    ) -> dict[str, np.ndarray]:
        while len(generated) < max_new_tokens:
            outputs, target_cache = self._run_target(
                np.array([[generated[-1]]], dtype=np.int64),
                target_cache,
            )
            generated.append(int(np.argmax(outputs["logits"][0, -1])))
        return target_cache

    def _generate_dflash(
        self,
        input_ids: np.ndarray,
        max_new_tokens: int,
    ) -> DraftGenerationResult:
        if self.target_embedding is None:
            raise ValueError("DFlash package requires a target embedding bridge")
        target_cache = _empty_cache(self.target)
        outputs, target_cache = self._run_target(input_ids, target_cache)
        generated = [int(np.argmax(outputs["logits"][0, -1]))]
        target_hidden = self._target_features(outputs)
        draft_cache = _empty_cache(self.draft)
        draft = self.manifest["draft"]
        block_size = int(draft["block_size"])
        mask_token_id = draft.get("mask_token_id")
        if block_size <= 1 or not isinstance(mask_token_id, int):
            raise ValueError("DFlash package requires block_size > 1 and mask_token_id")
        rounds: list[tuple[int, int]] = []
        rollbacks: list[tuple[int, int, int, int]] = []

        while len(generated) < max_new_tokens:
            if max_new_tokens - len(generated) < block_size:
                target_cache = self._finish_target_only(
                    generated,
                    target_cache,
                    max_new_tokens,
                )
                break
            start = input_ids.shape[1] + len(generated) - 1
            block = np.array(
                [[generated[-1], *([mask_token_id] * (block_size - 1))]],
                dtype=np.int64,
            )
            noise_embedding = self._outputs(
                self.target_embedding,
                {"input_ids": block},
            )["inputs_embeds"]
            past_length = next(iter(draft_cache.values())).shape[2]
            position_ids: np.ndarray = np.arange(
                past_length,
                start + block_size,
                dtype=np.int64,
            )[None, :]
            draft_inputs = {value.name for value in self.draft.get_inputs()}
            position_feeds: dict[str, np.ndarray] = {}
            if "position_ids" in draft_inputs:
                position_feeds["position_ids"] = position_ids
            if "q_position_ids" in draft_inputs:
                position_feeds["q_position_ids"] = position_ids[:, -block_size:]
            draft_outputs, draft_present = self._run_draft(
                {
                    "noise_embedding": noise_embedding,
                    "target_hidden": target_hidden,
                    **position_feeds,
                },
                draft_cache,
            )
            draft_cache = _crop_cache(draft_present, start)
            proposed_logits = self._project_draft_logits(draft_outputs)[:, -block_size + 1 :]
            proposal_ids = np.argmax(proposed_logits, axis=-1).astype(np.int64)[0]
            remap = self.manifest.get("draft_to_target")
            proposals = (
                np.asarray([remap[int(token)] for token in proposal_ids], dtype=np.int64)
                if isinstance(remap, list)
                else proposal_ids
            )
            verify = np.array([[generated[-1], *proposals.tolist()]], dtype=np.int64)
            verified, target_present = self._run_target(verify, target_cache)
            posterior = np.argmax(verified["logits"], axis=-1).astype(np.int64)[0]
            accepted = 0
            for proposal, target_token in zip(proposals, posterior[:-1]):
                if int(proposal) != int(target_token):
                    break
                accepted += 1
            generated.extend(int(token) for token in proposals[:accepted])
            generated.append(int(posterior[accepted]))
            new_length = start + accepted + 1
            if accepted < len(proposals):
                rollbacks.append(
                    (
                        next(iter(draft_present.values())).shape[2],
                        next(iter(draft_cache.values())).shape[2],
                        next(iter(target_present.values())).shape[2],
                        new_length,
                    )
                )
            target_cache = _crop_cache(target_present, new_length)
            target_hidden = self._target_features(
                verified,
                slice(0, accepted + 1),
            )
            rounds.append((accepted, len(proposals)))
        return self._result(generated, max_new_tokens, rounds, rollbacks)

    def _generate_eagle3(
        self,
        input_ids: np.ndarray,
        max_new_tokens: int,
        width: int,
    ) -> DraftGenerationResult:
        target_cache = _empty_cache(self.target)
        outputs, target_cache = self._run_target(input_ids, target_cache)
        generated = [int(np.argmax(outputs["logits"][0, -1]))]
        target_features = self._target_features(outputs)
        draft_cache = _empty_cache(self.draft)
        hidden_size = int(self.manifest["draft"]["hidden_size"])
        target_feature_size = target_features.shape[-1]
        draft_dtype = _session_dtype(self.draft, "fused_hidden")
        if input_ids.shape[1] > 1:
            _, draft_cache = self._run_eagle_step(
                input_ids[:, 1:],
                target_features[:, :-1],
                np.zeros(
                    (1, input_ids.shape[1] - 1, hidden_size),
                    dtype=draft_dtype,
                ),
                draft_cache,
            )
        pending_features = target_features[:, -1:]
        remap = self.manifest.get("draft_to_target")
        if remap is not None and (not isinstance(remap, list) or not remap):
            raise ValueError("EAGLE3 draft_to_target must be null or a non-empty map")
        rounds: list[tuple[int, int]] = []
        rollbacks: list[tuple[int, int, int, int]] = []

        while len(generated) < max_new_tokens:
            remaining = max_new_tokens - len(generated)
            if remaining == 1:
                target_cache = self._finish_target_only(
                    generated,
                    target_cache,
                    max_new_tokens,
                )
                break
            round_width = min(width, remaining - 1)
            start = input_ids.shape[1] + len(generated) - 1
            base_cache = draft_cache
            proposals: list[int] = []
            recycled = np.zeros((1, 1, hidden_size), dtype=draft_dtype)
            token = generated[-1]
            tentative = draft_cache
            for step in range(round_width):
                features = (
                    pending_features
                    if step == 0
                    else np.zeros((1, 1, target_feature_size), dtype=draft_dtype)
                )
                draft_outputs, tentative = self._run_eagle_step(
                    np.array([[token]], dtype=np.int64),
                    features,
                    recycled,
                    tentative,
                )
                logits = self._project_draft_logits(draft_outputs)
                draft_id = int(np.argmax(logits[0, -1]))
                token = int(remap[draft_id]) if isinstance(remap, list) else draft_id
                proposals.append(token)
                recycled = draft_outputs["next_hidden"][:, -1:]

            verified, target_present = self._run_target(
                np.array([[generated[-1], *proposals]], dtype=np.int64),
                target_cache,
            )
            posterior = np.argmax(verified["logits"], axis=-1).astype(np.int64)[0]
            accepted = 0
            for proposal, target_token in zip(proposals, posterior[:-1]):
                if proposal != int(target_token):
                    break
                accepted += 1
            prior_token = generated[-1]
            generated.extend(proposals[:accepted])
            generated.append(int(posterior[accepted]))
            new_length = start + accepted + 1
            target_cache = _crop_cache(target_present, new_length)

            accepted_tokens = np.array(
                [[prior_token, *proposals[:accepted]]],
                dtype=np.int64,
            )
            accepted_features = (
                np.concatenate(
                    [
                        pending_features,
                        self._target_features(verified, slice(0, accepted)),
                    ],
                    axis=1,
                )
                if accepted
                else pending_features
            )
            _, draft_cache = self._run_eagle_step(
                accepted_tokens,
                accepted_features,
                np.zeros((1, accepted + 1, hidden_size), dtype=draft_dtype),
                base_cache,
            )
            pending_features = self._target_features(
                verified,
                slice(accepted, accepted + 1),
            )
            if accepted < round_width:
                rollbacks.append(
                    (
                        next(iter(tentative.values())).shape[2],
                        next(iter(draft_cache.values())).shape[2],
                        next(iter(target_present.values())).shape[2],
                        new_length,
                    )
                )
            rounds.append((accepted, round_width))
        return self._result(generated, max_new_tokens, rounds, rollbacks)

    def _run_eagle_step(
        self,
        input_ids: np.ndarray,
        fused_hidden: np.ndarray,
        recycled_hidden: np.ndarray,
        cache: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        past_length = next(iter(cache.values())).shape[2]
        input_names = {value.name for value in self.draft.get_inputs()}
        token_feeds: dict[str, np.ndarray]
        if "input_ids" in input_names:
            token_feeds = {"input_ids": input_ids}
        elif "inputs_embeds" in input_names:
            if self.target_embedding is None:
                raise ValueError(
                    "target-shared EAGLE3 embeddings require a target embedding bridge"
                )
            token_feeds = {
                "inputs_embeds": self._outputs(
                    self.target_embedding,
                    {"input_ids": input_ids},
                )["inputs_embeds"]
            }
        else:
            raise ValueError("EAGLE3 draft graph has no token input")
        feeds = {
            **token_feeds,
            "fused_hidden": fused_hidden,
            "recycled_hidden": recycled_hidden,
            **cache,
        }
        if "attention_mask" in input_names:
            feeds["attention_mask"] = np.ones(
                (1, past_length + input_ids.shape[1]),
                dtype=np.int64,
            )
        if "position_ids" in input_names:
            feeds["position_ids"] = np.arange(
                past_length,
                past_length + input_ids.shape[1],
                dtype=np.int64,
            )[None, :]
        return self._run_draft(feeds, cache)
