# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CLM Hub dispatch regression tests; no network or encoder downloads."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import huggingface_hub
import pytest
import torch
import transformers

from mobius import build
from mobius._configs import CLMConfig
from mobius.integrations.transformers import _builder, _clm
from mobius.models.clm_test import _checkpoint
from mobius.tasks import ContrastiveRankingHeadsTask


def _hf_config(**overrides):
    return SimpleNamespace(
        **{
            "model_type": "clm",
            "base_model": "Qwen/Qwen3-8B",
            "encoder_pooling": "last-token",
            "embedding_dim": 4096,
            "checkpoints": ["CLM_v0.1-8B.pt"],
            **overrides,
        }
    )


def test_clm_default_revision_reaches_config_and_only_heads(monkeypatch, tmp_path):
    path = tmp_path / _clm.CLM_CHECKPOINT
    torch.save(_checkpoint(CLMConfig(width=6, projection_dim=4)), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(_clm, "CLM_CHECKPOINT_SHA256", digest)
    calls = []

    def config_loader(model_id, *, revision, trust_remote_code):
        calls.append(("config", model_id, revision))
        assert not trust_remote_code
        return _hf_config(), True

    def checkpoint_loader(model_id, filename, *, revision):
        calls.append((filename, model_id, revision))
        return str(path)

    monkeypatch.setattr(_builder, "_load_transformers_config", config_loader)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", checkpoint_loader)
    package = build(_clm.CLM_MODEL_ID, execution_provider="cpu")
    assert calls == [
        ("config", _clm.CLM_MODEL_ID, _clm.CLM_REVISION),
        (_clm.CLM_CHECKPOINT, _clm.CLM_MODEL_ID, _clm.CLM_REVISION),
    ]
    assert package["model"].metadata_props["mobius.checkpoint_sha256"] == digest
    monkeypatch.setattr(_clm, "CLM_CHECKPOINT_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        build(_clm.CLM_MODEL_ID)


@pytest.mark.parametrize(
    "field,value",
    [
        ("base_model", "some/other-encoder"),
        ("embedding_dim", 512),
        ("encoder_pooling", "mean"),
        ("checkpoints", ["untrusted.pt"]),
    ],
)
def test_clm_rejects_other_encoder_contracts(field, value):
    with pytest.raises(ValueError, match="contract"):
        CLMConfig.from_transformers(_hf_config(**{field: value}))


@pytest.mark.parametrize(
    "options,message",
    [
        ({"task": "text-generation"}, "not generation"),
        ({"text_only": True}, "text_only"),
        ({"dtype": "f16"}, "f32"),
        ({"revision": "main"}, "immutable"),
    ],
)
def test_clm_invalid_options_do_not_download_weights(monkeypatch, options, message):
    monkeypatch.setattr(
        _builder, "_load_transformers_config", lambda *a, **kw: (_hf_config(), True)
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid CLM build must not download checkpoint weights")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", forbidden)
    with pytest.raises(ValueError, match=message):
        build(_clm.CLM_MODEL_ID, **options)


def test_clm_never_executes_remote_code(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CLM remote code must be rejected before config resolution")

    monkeypatch.setattr(_builder, "_load_transformers_config", forbidden)
    with pytest.raises(ValueError, match="remote model code"):
        build(_clm.CLM_MODEL_ID, trust_remote_code=True)


@pytest.mark.parametrize("local", [False, True])
def test_clm_raw_preflight_rejects_remote_code_for_every_source(monkeypatch, tmp_path, local):
    model_id = str(tmp_path) if local else "alternate/clm-heads"
    revision = "a" * 40
    calls = []

    def raw_config(source, **kwargs):
        calls.append((source, kwargs))
        return {
            "model_type": "clm",
            "auto_map": {"AutoConfig": "custom_config.UntrustedConfig"},
            "_commit_hash": revision,
        }, {}

    def forbidden(*args, **kwargs):
        pytest.fail("CLM must be rejected before AutoConfig can execute custom code")

    monkeypatch.setattr(transformers.PretrainedConfig, "get_config_dict", raw_config)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", forbidden)
    with pytest.raises(ValueError, match="remote model code"):
        build(model_id, revision=revision, trust_remote_code=True)
    assert calls == [(model_id, {"revision": revision})]


def test_trusted_non_clm_config_uses_preflight_commit_for_autoconfig(monkeypatch):
    revision = "a" * 40
    calls = []

    def raw_config(source, **kwargs):
        calls.append(("raw", source, kwargs))
        return {"model_type": "custom", "_commit_hash": revision}, {}

    def auto_config(source, **kwargs):
        calls.append(("auto", source, kwargs))
        return SimpleNamespace(model_type="custom")

    monkeypatch.setattr(transformers.PretrainedConfig, "get_config_dict", raw_config)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", auto_config)
    config, from_json = _builder._load_transformers_config(
        "alternate/custom", revision="main", trust_remote_code=True
    )
    assert not from_json
    assert config.model_type == "custom"
    assert calls == [
        ("raw", "alternate/custom", {"revision": "main"}),
        ("auto", "alternate/custom", {"revision": revision, "trust_remote_code": True}),
    ]


def test_clm_preserves_task_instance_and_overridden_build(monkeypatch, tmp_path):
    path = tmp_path / _clm.CLM_CHECKPOINT
    torch.save(_checkpoint(CLMConfig(width=6, projection_dim=4)), path)
    monkeypatch.setattr(
        _builder, "_load_transformers_config", lambda *a, **kw: (_hf_config(), True)
    )

    class CustomTask(ContrastiveRankingHeadsTask):
        def __init__(self, graph_name):
            self.graph_name = graph_name
            self.called = False

        def build(self, module, config):
            self.called = True
            package = super().build(module, config)
            package["model"].graph.name = self.graph_name
            return package

    task = CustomTask("caller_configured_clm_graph")
    package = build(str(tmp_path), task=task, execution_provider="cpu")
    assert task.called
    assert package["model"].graph.name == "caller_configured_clm_graph"


def test_clm_oversized_checkpoint_rejected_before_hashing_and_loading(monkeypatch, tmp_path):
    path = tmp_path / _clm.CLM_CHECKPOINT
    path.touch()
    original_stat = _clm.Path.stat
    original_open = _clm.Path.open

    def stat(file, *args, **kwargs):
        if file == path:
            return SimpleNamespace(st_size=256 * 1024 * 1024 + 1)
        return original_stat(file, *args, **kwargs)

    def open_file(file, *args, **kwargs):
        if file == path:
            pytest.fail("Oversized checkpoint must be rejected before opening/hash I/O")
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(_clm.Path, "stat", stat)
    monkeypatch.setattr(_clm.Path, "open", open_file)
    monkeypatch.setattr(
        _builder, "_load_transformers_config", lambda *a, **kw: (_hf_config(), True)
    )
    for operation in (
        lambda: build(str(tmp_path)),
        lambda: _clm.load_clm_checkpoint(path, CLMConfig()),
    ):
        with pytest.raises(ValueError, match="256 MiB loading limit"):
            operation()
