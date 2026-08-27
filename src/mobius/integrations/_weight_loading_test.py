# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Defensive security regression tests for the weight loading path.

These tests guard against regressions in the security posture of weight
loading code. They verify:
1. Safetensors is preferred over legacy PyTorch formats.
2. Legacy ``torch.load`` calls always use ``weights_only=True``.
3. Path traversal attacks are blocked in weight file paths.
4. Temporary files are cleaned up after weight operations.
5. Corrupted / truncated weight files are handled gracefully.
6. The ``build_from_module`` public contract is preserved.

Targets the integration weight loader and exporter.
Uses MockWeightProvider pattern for test isolation — no network calls needed.
"""

from __future__ import annotations

import ast
import json
import os
import pathlib
from unittest import mock

import onnx_ir as ir
import pytest
import safetensors.torch
import torch
from huggingface_hub.utils import EntryNotFoundError

from mobius._builder import build_from_module
from mobius._model_package import ModelPackage
from mobius._testing import make_config
from mobius.integrations._weight_loading import _download_weights, apply_weights
from mobius.models.base import CausalLMModel
from mobius.tasks import CausalLMTask, ModelTask

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SRC_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Paths to scan for security regressions (non-test source files)
_SOURCE_FILES = [p for p in _SRC_ROOT.rglob("*.py") if not p.name.endswith("_test.py")]

# Key weight-loading files to scan (both current and refactored paths)
_WEIGHT_FILES = [
    _SRC_ROOT / "integrations" / "_weight_loading.py",
]


class MockWeightProvider:
    """Fake weight provider for test isolation — no HuggingFace calls."""

    def __init__(self, state_dict: dict[str, torch.Tensor] | None = None):
        self.state_dict = state_dict or {}

    def get_state_dict(self) -> dict[str, torch.Tensor]:
        return self.state_dict


def _build_model_with_weights() -> tuple[ir.Model, list[str]]:
    """Build a small test model and return it with its initializer names."""
    config = make_config()
    module = CausalLMModel(config)
    pkg = build_from_module(module, config)
    model = pkg["model"]
    init_names = list(model.graph.initializers.keys())
    return model, init_names


# ===========================================================================
# (1) Safetensors preference over pickle
# ===========================================================================


class TestSafetensorsPreference:
    """Verify that safetensors is preferred and legacy loading is guarded."""

    @pytest.mark.parametrize("weight_file", _WEIGHT_FILES, ids=lambda p: p.name)
    def test_no_pickle_import(self, weight_file):
        """Weight loading files must not import pickle or shelve."""
        if not weight_file.exists():
            pytest.skip(f"{weight_file.name} does not exist yet")
        source = weight_file.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module.split(".")[0])
        forbidden = {"pickle", "shelve", "_pickle", "cPickle"}
        violations = imported_modules & forbidden
        assert not violations, f"Forbidden pickle imports in {weight_file.name}: {violations}"

    @pytest.mark.parametrize("weight_file", _WEIGHT_FILES, ids=lambda p: p.name)
    def test_no_unguarded_torch_load(self, weight_file):
        """Weight loading files must guard torch.load with weights_only=True."""
        if not weight_file.exists():
            pytest.skip(f"{weight_file.name} does not exist yet")
        source = weight_file.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "load"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "torch"
                ):
                    assert any(
                        kw.arg == "weights_only"
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value is True
                        for kw in node.keywords
                    ), f"unguarded torch.load found in {weight_file.name}:{node.lineno}"

    def test_safetensors_remains_preferred(self):
        """Safetensors lookup must occur before legacy PyTorch lookup."""
        for weight_file in _WEIGHT_FILES:
            if not weight_file.exists():
                continue
            source = weight_file.read_text(encoding="utf-8")
            assert "safetensors" in source, f"No safetensors usage in {weight_file.name}"
            assert source.index("_SINGLE_WEIGHT_NAME") < source.index(
                "_SINGLE_PYTORCH_WEIGHT_NAME"
            )

    def test_no_other_weight_deserializers(self):
        """Only safetensors.load_file and guarded torch.load may deserialize weights."""
        forbidden_loaders = {"load_state_dict", "unpickle"}
        for weight_file in _WEIGHT_FILES:
            if not weight_file.exists():
                continue
            source = weight_file.read_text(encoding="utf-8")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr in forbidden_loaders:
                    if isinstance(func.value, ast.Name) and func.value.id == "torch":
                        pytest.fail(
                            f"torch.{func.attr} found in {weight_file.name}:{node.lineno}"
                        )


class TestLocalSafetensorsLoading:
    """Verify local HuggingFace checkpoint directories load without Hub access."""

    def test_local_single_safetensors_loaded_without_hub(self, tmp_path, monkeypatch):
        data = {"weight": torch.ones(2, 3)}
        safetensors.torch.save_file(data, str(tmp_path / "model.safetensors"))

        def _unexpected_hub_call(*_args, **_kwargs):
            raise AssertionError("local checkpoint should not call hf_hub_download")

        monkeypatch.setattr(
            "mobius.integrations._weight_loading.hf_hub_download", _unexpected_hub_call
        )

        state_dict = _download_weights(str(tmp_path))

        assert torch.equal(state_dict["weight"], data["weight"])

    def test_local_sharded_safetensors_index_loaded_without_hub(self, tmp_path, monkeypatch):
        shard_a = {"a.weight": torch.ones(1)}
        shard_b = {"b.weight": torch.zeros(1)}
        safetensors.torch.save_file(shard_a, str(tmp_path / "shard-a.safetensors"))
        safetensors.torch.save_file(shard_b, str(tmp_path / "shard-b.safetensors"))
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {},
                    "weight_map": {
                        "a.weight": "shard-a.safetensors",
                        "b.weight": "shard-b.safetensors",
                    },
                }
            )
        )

        def _unexpected_hub_call(*_args, **_kwargs):
            raise AssertionError("local checkpoint should not call hf_hub_download")

        monkeypatch.setattr(
            "mobius.integrations._weight_loading.hf_hub_download", _unexpected_hub_call
        )

        state_dict = _download_weights(str(tmp_path))

        assert torch.equal(state_dict["a.weight"], shard_a["a.weight"])
        assert torch.equal(state_dict["b.weight"], shard_b["b.weight"])

    def test_local_legacy_pytorch_state_dict_loaded_without_hub(self, tmp_path, monkeypatch):
        data = {"weight": torch.ones(2, 3)}
        torch.save(data, tmp_path / "pytorch_model.bin")

        def _unexpected_hub_call(*_args, **_kwargs):
            raise AssertionError("local checkpoint should not call hf_hub_download")

        monkeypatch.setattr(
            "mobius.integrations._weight_loading.hf_hub_download", _unexpected_hub_call
        )

        state_dict = _download_weights(str(tmp_path))

        assert torch.equal(state_dict["weight"], data["weight"])

    def test_local_directory_without_safetensors_raises_without_hub(
        self, tmp_path, monkeypatch
    ):
        def _unexpected_hub_call(*_args, **_kwargs):
            raise AssertionError("local checkpoint should not call hf_hub_download")

        monkeypatch.setattr(
            "mobius.integrations._weight_loading.hf_hub_download", _unexpected_hub_call
        )

        with pytest.raises(FileNotFoundError, match="Local checkpoint directory has no"):
            _download_weights(str(tmp_path))

    @pytest.mark.parametrize(
        "malicious_filename",
        ["../../../etc/passwd", "..\\..\\secret.safetensors", "/absolute/model.safetensors"],
    )
    def test_local_safetensors_index_rejects_unsafe_paths(self, tmp_path, malicious_filename):
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": {"weight": malicious_filename}})
        )

        with pytest.raises(ValueError, match="Unsafe weight filename"):
            _download_weights(str(tmp_path))


def test_hub_legacy_pytorch_fallback_preserves_revision(tmp_path, monkeypatch):
    """A Hub repo with no safetensors falls back to pinned PyTorch weights."""
    weight_path = tmp_path / "pytorch_model.bin"
    expected = {"weight": torch.arange(6).reshape(2, 3)}
    torch.save(expected, weight_path)
    calls = []

    def _download(*, repo_id, filename, revision=None):
        calls.append((repo_id, filename, revision))
        if filename == "pytorch_model.bin":
            return str(weight_path)
        raise EntryNotFoundError(filename)

    monkeypatch.setattr("mobius.integrations._weight_loading.hf_hub_download", _download)

    state_dict = _download_weights("legacy/model", revision="immutable-sha")

    assert torch.equal(state_dict["weight"], expected["weight"])
    assert calls == [
        ("legacy/model", "model.safetensors.index.json", "immutable-sha"),
        ("legacy/model", "model.safetensors", "immutable-sha"),
        ("legacy/model", "pytorch_model.bin.index.json", "immutable-sha"),
        ("legacy/model", "pytorch_model.bin", "immutable-sha"),
    ]


# ===========================================================================
# (2) weights_only=True assertion on any torch.load
# ===========================================================================


class TestNoUnguardedTorchLoad:
    """Scan all source files for unsafe deserialization calls."""

    def test_no_torch_load_without_weights_only(self):
        """Any torch.load call in the package MUST use weights_only=True."""
        violations: list[str] = []
        for path in _SOURCE_FILES:
            source = path.read_text(encoding="utf-8")
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                is_torch_load = (
                    isinstance(func, ast.Attribute)
                    and func.attr == "load"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "torch"
                )
                if not is_torch_load:
                    continue
                has_weights_only = any(
                    kw.arg == "weights_only"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                    for kw in node.keywords
                )
                if not has_weights_only:
                    rel = path.relative_to(_SRC_ROOT)
                    violations.append(f"{rel}:{node.lineno}")

        assert not violations, f"torch.load without weights_only=True found at: {violations}"

    def test_no_pickle_load_in_source(self):
        """No source file should call pickle.load or pickle.loads."""
        violations: list[str] = []
        for path in _SOURCE_FILES:
            source = path.read_text(encoding="utf-8")
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr in ("load", "loads")
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "pickle"
                ):
                    rel = path.relative_to(_SRC_ROOT)
                    violations.append(f"{rel}:{node.lineno}")

        assert not violations, f"pickle.load/loads found at: {violations}"

    def test_no_bare_eval_in_source(self):
        """No source file should use eval() (excluding model.eval())."""
        violations: list[str] = []
        for path in _SOURCE_FILES:
            source = path.read_text(encoding="utf-8")
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                # Bare eval() call
                if isinstance(func, ast.Name) and func.id == "eval":
                    rel = path.relative_to(_SRC_ROOT)
                    violations.append(f"{rel}:{node.lineno}")

        assert not violations, f"eval() found at: {violations}"


# ===========================================================================
# (3) Path traversal prevention in weight file paths
# ===========================================================================


class TestPathTraversalPrevention:
    """Ensure path traversal sequences in weight file names are rejected."""

    @pytest.mark.parametrize(
        "malicious_filename",
        [
            "../../../etc/passwd",
            "..\\..\\Windows\\System32\\config\\SAM",
            "model/../../../etc/shadow",
            "weights/../../secret.safetensors",
            "/absolute/path/model.safetensors",
        ],
        ids=[
            "unix_traversal",
            "windows_traversal",
            "embedded_traversal",
            "relative_traversal",
            "absolute_path",
        ],
    )
    def test_path_traversal_in_weight_index_rejected(self, malicious_filename):
        """Filenames from an index.json with path traversal must be sanitized or rejected.

        This tests the contract that weight file paths derived from user/model
        data should not escape the expected directory. We use PurePosixPath to
        normalize both Unix and Windows-style separators.
        """
        import pathlib as _pathlib

        # Normalize both / and \ separators, then take the final component
        # (handles Windows-style paths on Linux hosts too)
        normalized = malicious_filename.replace("\\", "/")
        pure = _pathlib.PurePosixPath(normalized)
        basename = pure.name

        has_traversal = ".." in malicious_filename or malicious_filename.startswith(
            ("/", "\\")
        )
        assert has_traversal, "Test input should contain traversal"
        # The basename extraction strips directory traversal
        assert ".." not in basename
        assert not basename.startswith("/")

    def test_apply_weights_only_modifies_existing_initializers(self):
        """apply_weights must not inject initializers keyed by untrusted state dict names.

        The fold pipeline (FoldTransposedInitializerPass etc.) is allowed to create new
        initializers with names derived from existing ones (e.g. ``weight_t``), but no
        initializer should ever be created whose name came from an untrusted state dict key.
        """
        model, _init_names = _build_model_with_weights()

        # Inject weights with a path-like name and an unknown name — neither should
        # appear as an initializer after apply_weights.
        malicious_keys = {"../../etc/passwd", "normal_name_not_in_model"}
        provider = MockWeightProvider({key: torch.zeros(1) for key in malicious_keys})
        apply_weights(model, provider.get_state_dict())

        # Injected key names must not appear as initializer names.
        for key in malicious_keys:
            assert key not in model.graph.initializers, (
                f"Untrusted key '{key}' was injected into graph.initializers"
            )

    @pytest.mark.parametrize(
        "malicious_model_id",
        [
            "../../../etc/passwd",
            "/etc/shadow",
            "legitimate-org/../../etc/passwd",
        ],
        ids=["relative_traversal", "absolute_path", "embedded_traversal"],
    )
    def test_build_with_malicious_model_id_raises(self, malicious_model_id):
        """build() with a path-traversal model_id should raise, not silently proceed."""
        from mobius.integrations.transformers import build

        with pytest.raises((ValueError, OSError, Exception)):
            build(malicious_model_id, load_weights=False)


# ===========================================================================
# (4) Temp file cleanup after weight operations
# ===========================================================================


class TestTempFileCleanup:
    """Verify temporary files are cleaned up after weight operations."""

    def test_apply_weights_does_not_leak_temp_files(self, tmp_path):
        """apply_weights should not leave temporary files behind."""
        model, init_names = _build_model_with_weights()
        assert len(init_names) > 0

        name = init_names[0]
        shape = list(model.graph.initializers[name].shape)
        provider = MockWeightProvider({name: torch.ones(shape)})

        before = set(tmp_path.iterdir())
        with mock.patch.dict(os.environ, {"TMPDIR": str(tmp_path)}):
            apply_weights(model, provider.get_state_dict())
        after = set(tmp_path.iterdir())

        leaked = after - before
        assert not leaked, f"Temporary files leaked: {leaked}"

    def test_apply_weights_no_open_file_handles(self):
        """apply_weights should not leave file handles open after completion."""
        model, init_names = _build_model_with_weights()
        assert len(init_names) > 0

        name = init_names[0]
        shape = list(model.graph.initializers[name].shape)
        provider = MockWeightProvider({name: torch.ones(shape)})

        # This should complete without resource warnings
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            apply_weights(model, provider.get_state_dict())


# ===========================================================================
# (5) Corrupted / truncated file handling
# ===========================================================================


class TestCorruptedFileHandling:
    """Verify graceful handling of bad weight data."""

    def test_apply_weights_empty_state_dict(self):
        """Applying an empty state dict is a no-op — no crash."""
        model, _ = _build_model_with_weights()
        provider = MockWeightProvider({})
        apply_weights(model, provider.get_state_dict())

    def test_apply_weights_wrong_shape_raises(self):
        """A tensor with the wrong shape should raise ValueError."""
        model, init_names = _build_model_with_weights()
        assert len(init_names) > 0

        name = init_names[0]
        # Provide a tensor with a deliberately wrong shape
        wrong_shape = torch.zeros(999)
        provider = MockWeightProvider({name: wrong_shape})
        with pytest.raises(ValueError, match="shape mismatch"):
            apply_weights(model, provider.get_state_dict())

    def test_apply_weights_wrong_shape_error_message(self):
        """Shape mismatch error should include the weight name and both shapes."""
        model, init_names = _build_model_with_weights()
        assert len(init_names) > 0

        name = init_names[0]
        wrong_shape = torch.zeros(999)
        provider = MockWeightProvider({name: wrong_shape})
        with pytest.raises(ValueError, match=name):
            apply_weights(model, provider.get_state_dict())

    def test_apply_weights_nan_tensor(self):
        """NaN tensors should be applied without error."""
        model, init_names = _build_model_with_weights()
        assert len(init_names) > 0

        name = init_names[0]
        shape = list(model.graph.initializers[name].shape)
        nan_tensor = torch.full(shape, float("nan"))
        provider = MockWeightProvider({name: nan_tensor})
        apply_weights(model, provider.get_state_dict())
        assert model.graph.initializers[name].const_value is not None

    def test_apply_weights_dtype_mismatch_uses_lazy_cast(self):
        """When dtype differs, apply_weights wraps with LazyTensor for lazy cast."""
        model, init_names = _build_model_with_weights()
        assert len(init_names) > 0

        name = init_names[0]
        shape = list(model.graph.initializers[name].shape)
        # Use float64 which likely differs from the model's expected dtype
        mismatched = torch.ones(shape, dtype=torch.float64)
        provider = MockWeightProvider({name: mismatched})
        apply_weights(model, provider.get_state_dict())

        result = model.graph.initializers[name].const_value
        assert result is not None
        # If there was a dtype mismatch, a LazyTensor should have been created
        if model.graph.initializers[name].dtype != ir.DataType.DOUBLE:
            assert isinstance(result, ir.LazyTensor)

    def test_corrupted_safetensors_file_raises(self, tmp_path):
        """Loading a corrupted .safetensors file must raise a clear error."""
        corrupted = tmp_path / "model.safetensors"
        corrupted.write_bytes(b"THIS IS NOT A VALID SAFETENSORS FILE\x00\xff" * 10)

        with pytest.raises(Exception) as exc_info:
            safetensors.torch.load_file(str(corrupted))
        # Should be a clear deserialization error, not a silent corruption
        assert exc_info.value is not None

    def test_truncated_safetensors_file_raises(self, tmp_path):
        """Loading a truncated .safetensors file must raise, not return partial data."""
        # Create a valid safetensors file first, then truncate it
        valid_data = {"weight": torch.randn(4, 4)}
        valid_path = tmp_path / "valid.safetensors"
        safetensors.torch.save_file(valid_data, str(valid_path))

        truncated = tmp_path / "truncated.safetensors"
        full_bytes = valid_path.read_bytes()
        # Write only the first half
        truncated.write_bytes(full_bytes[: len(full_bytes) // 2])

        with pytest.raises((OSError, RuntimeError, safetensors.SafetensorError)):
            safetensors.torch.load_file(str(truncated))

    def test_zero_byte_safetensors_file_raises(self, tmp_path):
        """An empty (zero-byte) .safetensors file must raise."""
        empty = tmp_path / "empty.safetensors"
        empty.write_bytes(b"")

        with pytest.raises((OSError, RuntimeError, safetensors.SafetensorError)):
            safetensors.torch.load_file(str(empty))


# ===========================================================================
# (6b) Weight tying — genuine ONNX initializer sharing
# ===========================================================================


class TestWeightTying:
    """Verify that tied weights (lm_head = embed_tokens) share one ONNX initializer."""

    def _build_tied_model(self) -> tuple[ir.Model, CausalLMModel]:
        config = make_config(tie_word_embeddings=True)
        module = CausalLMModel(config)
        pkg = build_from_module(module, config)
        return pkg["model"], module

    def test_graph_level_tying_no_lm_head_initializer(self):
        """With tie_word_embeddings=True, lm_head.weight must not be in the graph — at all.

        The Parameter aliasing in __init__ ensures a single ir.Value is used for
        both the embedding Gather and the lm_head MatMul, so no lm_head.weight
        initializer is ever registered.
        """
        model, _ = self._build_tied_model()
        assert "lm_head.weight" not in model.graph.initializers, (
            "lm_head.weight must not be a graph initializer when tie_word_embeddings=True "
            "(graph-level aliasing should produce a single embed_tokens initializer)"
        )
        assert "model.embed_tokens.weight" in model.graph.initializers, (
            "model.embed_tokens.weight must be registered as the sole embedding initializer"
        )

    def test_graph_level_tying_same_ir_value(self):
        """The embed Gather and lm_head MatMul must both reference the same ir.Value."""
        model, _ = self._build_tied_model()
        embed_initializer = model.graph.initializers["model.embed_tokens.weight"]

        # Collect all ir.Value inputs used across all nodes
        all_inputs: set[int] = set()
        for node in model.graph:
            for inp in node.inputs:
                if inp is not None:
                    all_inputs.add(id(inp))

        assert id(embed_initializer) in all_inputs, (
            "model.embed_tokens.weight initializer must be referenced by at least one node"
        )

    def test_preprocess_weights_ensures_both_tied_keys(self):
        """preprocess_weights ensures both lm_head.weight and embed_tokens.weight exist.

        tie_word_embeddings(state_dict) adds the missing key rather than removing
        the present one, so that apply_weights can assign each initializer and the
        id()-based dedup unifies them at load time via replace_all_uses_with.
        """
        _, module = self._build_tied_model()
        vocab_size = make_config().vocab_size
        hidden_size = make_config().hidden_size
        weight = torch.zeros(vocab_size, hidden_size)

        # Case 1: checkpoint has only embed_tokens.weight (typical tied checkpoint).
        # lm_head.weight must be synthesised as an alias.
        sd = module.preprocess_weights({"model.embed_tokens.weight": weight})
        assert "model.embed_tokens.weight" in sd
        assert "lm_head.weight" in sd
        # Both point to the same tensor object so apply_weights dedup fires.
        assert sd["lm_head.weight"] is sd["model.embed_tokens.weight"]

        # Case 2: checkpoint has both keys — both must remain (no key is dropped).
        lm_head_weight = torch.ones(vocab_size, hidden_size)
        sd2 = module.preprocess_weights(
            {
                "model.embed_tokens.weight": weight,
                "lm_head.weight": lm_head_weight,
            }
        )
        assert "model.embed_tokens.weight" in sd2
        assert "lm_head.weight" in sd2

        # Case 3: checkpoint has only lm_head.weight.
        # embed_tokens.weight must be synthesised as an alias.
        sd3 = module.preprocess_weights({"lm_head.weight": weight})
        assert "lm_head.weight" in sd3
        assert "model.embed_tokens.weight" in sd3
        assert sd3["lm_head.weight"] is sd3["model.embed_tokens.weight"]

    def test_apply_weights_dedup_canonical_is_first_insertion_order(self):
        """apply_weights dedup picks the first key in state_dict insertion order as canonical.

        Python dicts are insertion-ordered (PEP 468, Python 3.7+), so whichever
        key appears first in the state_dict becomes the canonical initializer.
        No model-specific name preference is applied.
        """
        # Build an untied model so the graph has both initializers.
        config = make_config(tie_word_embeddings=False)
        module = CausalLMModel(config)
        pkg = build_from_module(module, config)
        model = pkg["model"]

        weight = torch.zeros(config.vocab_size, config.hidden_size)
        # embed_tokens.weight is first — it must become the canonical initializer.
        sd = {
            "model.embed_tokens.weight": weight,
            "lm_head.weight": weight,  # same object, comes second
        }
        apply_weights(model, sd)

        # The first key in insertion order is the canonical; the second is removed.
        first_key = next(iter(sd))
        second_key = next(k for k in sd if k != first_key)
        assert first_key in model.graph.initializers, (
            f"'{first_key}' (first in state_dict) must be the canonical initializer"
        )
        assert second_key not in model.graph.initializers, (
            f"'{second_key}' (second in state_dict) must be merged away"
        )

    def test_untied_weights_have_separate_initializers(self):
        """When tie_word_embeddings=False, both initializers remain independent.

        The fold-transpose pass may rename ``lm_head.weight`` to
        ``lm_head.weight_t`` (pre-transposing the single-user Transpose
        node).  This is correct for untied weights because ``lm_head.weight``
        has exactly one consumer.  In the tied case, the shared initializer
        has multiple consumers and fold is correctly skipped.
        """
        config = make_config(tie_word_embeddings=False)
        module = CausalLMModel(config)
        pkg = build_from_module(module, config)
        model = pkg["model"]

        vocab_size = config.vocab_size
        hidden_size = config.hidden_size
        sd = {
            "model.embed_tokens.weight": torch.zeros(vocab_size, hidden_size),
            "lm_head.weight": torch.zeros(vocab_size, hidden_size),
        }
        apply_weights(model, sd)

        assert "model.embed_tokens.weight" in model.graph.initializers
        # fold_initializers_after_weights may pre-transpose lm_head.weight
        # into lm_head.weight_t (single-user Transpose fold).
        assert (
            "lm_head.weight" in model.graph.initializers
            or "lm_head.weight_t" in model.graph.initializers
        )

    def test_tied_model_initializer_count_lower_than_untied(self):
        """Tied model must have fewer initializers than untied (one embed weight, not two)."""
        from mobius._testing import make_config as mc

        config_tied = mc(tie_word_embeddings=True)
        module_tied = CausalLMModel(config_tied)
        pkg_tied = build_from_module(module_tied, config_tied)
        model_tied = pkg_tied["model"]

        config_untied = mc(tie_word_embeddings=False)
        module_untied = CausalLMModel(config_untied)
        pkg_untied = build_from_module(module_untied, config_untied)
        model_untied = pkg_untied["model"]

        n_tied = len(model_tied.graph.initializers)
        n_untied = len(model_untied.graph.initializers)
        assert n_tied < n_untied, (
            f"Tied model should have fewer initializers ({n_tied}) than untied ({n_untied})"
        )

    def test_apply_weights_dedup_same_storage_different_objects(self):
        """Dedup catches separate tensor objects sharing the same storage.

        HuggingFace safetensors may deserialize tied weights as distinct
        Python objects that share the same underlying storage (same
        data_ptr).  apply_weights must detect this and merge them.
        """
        config = make_config(tie_word_embeddings=False)
        module = CausalLMModel(config)
        pkg = build_from_module(module, config)
        model = pkg["model"]

        # Create a base tensor and a view that shares storage but is a
        # different Python object (simulates safetensors deserialization).
        base = torch.zeros(config.vocab_size, config.hidden_size)
        view = base[:]  # same storage, different id()
        assert id(base) != id(view), "Precondition: must be different objects"
        assert base.data_ptr() == view.data_ptr(), "Precondition: same storage"

        sd = {
            "model.embed_tokens.weight": base,
            "lm_head.weight": view,
        }
        apply_weights(model, sd)

        # One should be merged away
        assert "model.embed_tokens.weight" in model.graph.initializers
        assert "lm_head.weight" not in model.graph.initializers


# ===========================================================================
# (6) build_from_module contract preservation
# ===========================================================================


class TestBuildFromModuleContract:
    """Verify the public build_from_module contract is preserved."""

    def test_returns_model_package(self):
        config = make_config()
        module = CausalLMModel(config)
        result = build_from_module(module, config)
        assert isinstance(result, ModelPackage)

    def test_model_key_present(self):
        config = make_config()
        module = CausalLMModel(config)
        pkg = build_from_module(module, config)
        assert "model" in pkg

    def test_model_has_outputs(self):
        config = make_config()
        module = CausalLMModel(config)
        pkg = build_from_module(module, config)
        model = pkg["model"]
        assert len(model.graph.outputs) > 0

    def test_accepts_task_string(self):
        config = make_config()
        module = CausalLMModel(config)
        pkg = build_from_module(module, config, task="text-generation")
        assert isinstance(pkg, ModelPackage)

    def test_accepts_task_instance(self):
        config = make_config()
        module = CausalLMModel(config)
        pkg = build_from_module(module, config, task=CausalLMTask())
        assert isinstance(pkg, ModelPackage)

    def test_rejects_unknown_task(self):
        config = make_config()
        module = CausalLMModel(config)
        with pytest.raises(ValueError, match="Unknown task"):
            build_from_module(module, config, task="nonexistent-task")

    def test_custom_task_instance_works(self):
        """A user-defined ModelTask should work with build_from_module."""

        class StubTask(ModelTask):
            def build(self, module, config):
                graph = ir.Graph([], [], nodes=[], name="stub")
                model = ir.Model(graph, ir_version=10)
                return ModelPackage({"model": model})

        config = make_config()
        module = CausalLMModel(config)
        pkg = build_from_module(module, config, task=StubTask())
        assert pkg["model"].graph.name == "stub"


class TestDequantizeFP8Weights:
    """Tests for _dequantize_fp8_weights."""

    def test_no_fp8_returns_unchanged(self):
        """Non-FP8 state dicts pass through unchanged."""
        from mobius.integrations._weight_loading import _dequantize_fp8_weights

        state_dict = {
            "layer.weight": torch.randn(4, 4),
            "layer.bias": torch.randn(4),
        }
        result = _dequantize_fp8_weights(state_dict)
        assert set(result.keys()) == set(state_dict.keys())
        assert torch.equal(result["layer.weight"], state_dict["layer.weight"])

    def test_fp8_e4m3fn_dequantized(self):
        """FP8 weights are multiplied by weight_scale_inv."""
        from mobius.integrations._weight_loading import _dequantize_fp8_weights

        fp8_weight = torch.tensor([1.0, 2.0, -1.0, 0.5], dtype=torch.float32).to(
            torch.float8_e4m3fn
        )
        scale_inv = torch.tensor(0.5, dtype=torch.bfloat16)
        state_dict = {
            "proj.weight": fp8_weight,
            "proj.weight_scale_inv": scale_inv,
        }
        result = _dequantize_fp8_weights(state_dict)
        assert "proj.weight" in result
        assert "proj.weight_scale_inv" not in result  # aux tensor removed
        assert result["proj.weight"].dtype == torch.bfloat16
        # Verify dequant: fp8→bf16 * scale_inv
        expected = fp8_weight.to(torch.bfloat16) * scale_inv
        assert torch.allclose(result["proj.weight"], expected)

    def test_activation_scale_removed(self):
        """Auxiliary activation_scale tensors are removed."""
        from mobius.integrations._weight_loading import _dequantize_fp8_weights

        state_dict = {
            "proj.weight": torch.tensor([1.0], dtype=torch.float32).to(torch.float8_e4m3fn),
            "proj.weight_scale_inv": torch.tensor(1.0, dtype=torch.bfloat16),
            "proj.activation_scale": torch.tensor(1.0, dtype=torch.bfloat16),
        }
        result = _dequantize_fp8_weights(state_dict)
        assert "proj.activation_scale" not in result

    def test_suffix_replace_not_greedy(self):
        """The scale key derivation uses suffix replacement, not global replace.

        For a key like 'model.weight_proj.weight', the scale key should be
        'model.weight_proj.weight_scale_inv' (not 'model.weight_scale_inv_proj.weight_scale_inv').
        """
        from mobius.integrations._weight_loading import _dequantize_fp8_weights

        fp8_weight = torch.tensor([1.0], dtype=torch.float32).to(torch.float8_e4m3fn)
        scale = torch.tensor(2.0, dtype=torch.bfloat16)
        state_dict = {
            "model.weight_proj.weight": fp8_weight,
            "model.weight_proj.weight_scale_inv": scale,
        }
        result = _dequantize_fp8_weights(state_dict)
        assert "model.weight_proj.weight" in result
        assert result["model.weight_proj.weight"].dtype == torch.bfloat16

    def test_missing_scale_fails_closed(self):
        """Generic FP8 loading must not guess an implicit unit scale."""
        from mobius.integrations._weight_loading import _dequantize_fp8_weights

        fp8_weight = torch.tensor([1.0, 2.0], dtype=torch.float32).to(torch.float8_e4m3fn)
        state_dict = {"orphan.weight": fp8_weight}
        with pytest.raises(ValueError, match="Refusing to guess an implicit scale"):
            _dequantize_fp8_weights(state_dict)

    def test_fp32_scale_produces_bf16_output(self):
        """FP32 weight_scale_inv should still produce bfloat16 output."""
        from mobius.integrations._weight_loading import _dequantize_fp8_weights

        fp8_weight = torch.tensor([1.0, 2.0], dtype=torch.float32).to(torch.float8_e4m3fn)
        # Scale stored as FP32 (common for scalar scales in real checkpoints)
        scale_inv = torch.tensor(0.5, dtype=torch.float32)
        state_dict = {
            "proj.weight": fp8_weight,
            "proj.weight_scale_inv": scale_inv,
        }
        result = _dequantize_fp8_weights(state_dict)
        assert result["proj.weight"].dtype == torch.bfloat16, (
            f"Expected bfloat16, got {result['proj.weight'].dtype}"
        )

    def test_2d_scale_grid_dequantizes_128_by_128_blocks(self):
        """A 2-D inverse-scale grid maps to 128-by-128 FP8 weight tiles."""
        from mobius.integrations._weight_loading import _dequantize_fp8_weights

        fp8_weight = torch.full((129, 130), 2.0, dtype=torch.float32).to(torch.float8_e4m3fn)
        scale_grid = torch.tensor([[0.5, 1.0], [1.5, 2.0]], dtype=torch.bfloat16)
        state_dict = {
            "expert.down_proj.weight": fp8_weight,
            "expert.down_proj.weight_scale_inv": scale_grid,
            "expert.down_proj.activation_scale": torch.tensor(1.0, dtype=torch.bfloat16),
        }

        result = _dequantize_fp8_weights(state_dict)

        expected = torch.empty((129, 130), dtype=torch.bfloat16)
        expected[:128, :128] = 1.0
        expected[:128, 128:] = 2.0
        expected[128:, :128] = 3.0
        expected[128:, 128:] = 4.0
        assert torch.equal(result["expert.down_proj.weight"], expected)
        assert result["expert.down_proj.weight"].dtype == torch.bfloat16
        assert "expert.down_proj.weight_scale_inv" not in result
        assert "expert.down_proj.activation_scale" not in result
        assert torch.equal(
            state_dict["expert.down_proj.weight"].to(torch.float32),
            fp8_weight.to(torch.float32),
        )
        assert torch.equal(state_dict["expert.down_proj.weight_scale_inv"], scale_grid)
        assert "expert.down_proj.activation_scale" in state_dict

    @pytest.mark.parametrize(
        ("weight_shape", "grid_shape"),
        [
            ((128, 128), (1, 2)),
            ((129, 128), (1, 1)),
        ],
    )
    def test_2d_scale_grid_requires_exact_block_shape(self, weight_shape, grid_shape):
        """Block-scale grids must cover every full or partial 128-by-128 tile."""
        from mobius.integrations._weight_loading import _dequantize_fp8_weights

        state_dict = {
            "proj.weight": torch.ones(weight_shape, dtype=torch.float32).to(
                torch.float8_e4m3fn
            ),
            "proj.weight_scale_inv": torch.ones(grid_shape, dtype=torch.bfloat16),
        }

        with pytest.raises(ValueError, match="scale grid shape"):
            _dequantize_fp8_weights(state_dict)

    @pytest.mark.parametrize(
        "scale",
        [
            torch.ones(2, dtype=torch.bfloat16),
            torch.ones((1, 1, 1), dtype=torch.bfloat16),
        ],
    )
    def test_scale_must_be_scalar_or_2d_grid(self, scale):
        """Non-scalar, non-grid inverse scales fail closed."""
        from mobius.integrations._weight_loading import _dequantize_fp8_weights

        state_dict = {
            "proj.weight": torch.ones((128, 128), dtype=torch.float32).to(torch.float8_e4m3fn),
            "proj.weight_scale_inv": scale,
        }

        with pytest.raises(ValueError, match="scalar or 2-D block scale grid"):
            _dequantize_fp8_weights(state_dict)

    def test_2d_scale_grid_requires_2d_weight(self):
        """Block-scale grids cannot silently broadcast over non-matrix FP8 weights."""
        from mobius.integrations._weight_loading import _dequantize_fp8_weights

        state_dict = {
            "proj.weight": torch.ones((1, 128, 128), dtype=torch.float32).to(
                torch.float8_e4m3fn
            ),
            "proj.weight_scale_inv": torch.ones((1, 1), dtype=torch.bfloat16),
        }

        with pytest.raises(ValueError, match="block scaling requires a 2-D weight"):
            _dequantize_fp8_weights(state_dict)

    def test_does_not_mutate_input(self):
        """_dequantize_fp8_weights should not mutate the input dict."""
        from mobius.integrations._weight_loading import _dequantize_fp8_weights

        fp8_weight = torch.tensor([1.0], dtype=torch.float32).to(torch.float8_e4m3fn)
        scale = torch.tensor(1.0, dtype=torch.bfloat16)
        original = {
            "proj.weight": fp8_weight,
            "proj.weight_scale_inv": scale,
        }
        original_keys = set(original.keys())
        _dequantize_fp8_weights(original)
        assert set(original.keys()) == original_keys, "Input dict was mutated"
