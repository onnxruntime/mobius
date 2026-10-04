# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CLM Hub dispatch regression tests; no network or encoder downloads."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import huggingface_hub
import pytest
import torch

from mobius import build
from mobius._configs import CLMConfig
from mobius.integrations.transformers import _builder, _clm
from mobius.models.clm_test import _checkpoint


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
