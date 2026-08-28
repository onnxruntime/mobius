# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import onnx_ir as ir
import pytest

from mobius._model_package import ModelPackage
from mobius.integrations.gguf._draft import _tokenizer_digest
from mobius.integrations.gguf._draft_pair import (
    _validate_target_artifact,
    build_draft_pair_from_gguf,
    write_draft_pair_package,
)
from mobius.integrations.gguf._draft_runtime import DraftPairRunner
from mobius.integrations.gguf._runtime_evidence import gguf_graph_package_identity


def _empty_model(name: str) -> ir.Model:
    return ir.Model(ir.Graph([], [], nodes=[], name=name), ir_version=11)


def _external_data_model(name: str) -> ir.Model:
    weight = ir.Value(
        name="weight",
        const_value=ir.tensor(np.ones(300, dtype=np.float32)),
    )
    output = ir.Value(name="output", type=weight.type, shape=weight.shape)
    node = ir.Node("", "Identity", inputs=[weight], outputs=[output])
    graph = ir.Graph(
        [],
        [output],
        nodes=[node],
        initializers=[weight],
        name=name,
        opset_imports={"": 24},
    )
    return ir.Model(graph, ir_version=11)


def test_target_artifact_validation_binds_config_and_tokenizer() -> None:
    tokens = ["a", "b", "c"]
    manifest = {
        "target": {
            "model_type": "qwen3",
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "vocab_size": 3,
            "tokenizer_tokens_sha256": _tokenizer_digest(tokens),
        }
    }
    config = SimpleNamespace(
        model_type="qwen3",
        hidden_size=8,
        num_hidden_layers=2,
        vocab_size=3,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None,
    )
    model = SimpleNamespace(metadata={"tokenizer.ggml.tokens": tokens})

    _validate_target_artifact(model, config, manifest)

    model.metadata["tokenizer.ggml.tokens"] = list(reversed(tokens))
    with pytest.raises(ValueError, match="tokenizer does not match"):
        _validate_target_artifact(model, config, manifest)


def test_draft_pair_rejects_unfeedable_bfloat16_before_build() -> None:
    with pytest.raises(ValueError, match="not bfloat16"):
        build_draft_pair_from_gguf(
            "target.gguf",
            "draft.gguf",
            target_config="target-config",
            dtype="bf16",
        )


def test_write_draft_pair_package_persists_manifest(tmp_path) -> None:
    package = ModelPackage({"target": _empty_model("target"), "draft": _empty_model("draft")})
    package.draft_manifest = {
        "format_version": 1,
        "kind": "speculative-draft",
        "architecture": "eagle3",
    }
    output = tmp_path / "package"

    result = write_draft_pair_package(
        package,
        output,
        progress_bar=False,
        check_weights=False,
    )

    assert result["package"] == str(output)
    assert json.loads((output / "draft_manifest.json").read_text()) == (package.draft_manifest)
    assert package.draft_manifest["graph_package"]["files"]
    assert len(package.draft_manifest["graph_package"]["sha256"]) == 64
    assert (output / "target" / "model.onnx").is_file()
    assert (output / "draft" / "model.onnx").is_file()


def test_runner_rejects_component_outside_verified_file_set(tmp_path) -> None:
    package = ModelPackage({"target": _empty_model("target"), "draft": _empty_model("draft")})
    package.draft_manifest = {
        "format_version": 1,
        "kind": "speculative-draft",
        "architecture": "eagle3",
        "components": {
            "target": {"artifact": "target/model.onnx"},
            "draft": {"artifact": "draft/model.onnx"},
        },
    }
    output = tmp_path / "package"
    write_draft_pair_package(
        package,
        output,
        progress_bar=False,
        check_weights=False,
    )
    manifest_path = output / "draft_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["components"]["target"]["artifact"] = "unverified.onnx"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="outside the verified graph package"):
        DraftPairRunner(output)


def test_runner_rejects_unverified_external_data(tmp_path) -> None:
    package = ModelPackage(
        {
            "target": _external_data_model("target"),
            "draft": _empty_model("draft"),
        }
    )
    package.draft_manifest = {
        "format_version": 1,
        "kind": "speculative-draft",
        "architecture": "eagle3",
        "components": {
            "target": {"artifact": "target/model.onnx"},
            "draft": {"artifact": "draft/model.onnx"},
        },
    }
    output = tmp_path / "package"
    write_draft_pair_package(package, output, progress_bar=False)
    manifest_path = output / "draft_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    files = tuple(
        name for name in manifest["graph_package"]["files"] if name != "target/model.onnx.data"
    )
    identity = gguf_graph_package_identity(output, files=files)
    manifest["graph_package"] = {
        "files": list(identity.files),
        "sha256": identity.sha256,
    }
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="external data is outside"):
        DraftPairRunner(output)
