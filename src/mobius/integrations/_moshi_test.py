# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import ntpath
from pathlib import Path
from unittest import mock

import onnx_ir as ir
import pytest

from mobius._model_package import ModelPackage
from mobius.integrations import _moshi
from mobius.integrations.transformers import _builder as transformers_builder

_builder = _moshi


def _component(name: str) -> ir.Model:
    graph = ir.Graph(
        inputs=[],
        outputs=[],
        nodes=[],
        name=name,
        opset_imports={"": 21},
    )
    return ir.Model(graph, ir_version=10)


def _local_personaplex_checkpoint(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    (path / "model.safetensors").touch()
    (path / "tokenizer-test.safetensors").touch()
    return path


class _SafeTensorHeader:
    def __init__(self, path, **_kwargs):
        if str(path).endswith("model.safetensors"):
            self._shapes = {
                "text_emb.weight": (32001, 4096),
                "text_linear.weight": (32000, 4096),
                "depformer_in.15.weight": (1024, 4096),
            }
        else:
            self._shapes = {
                "encoder.model.0.weight": (1,),
                "decoder.model.0.weight": (1,),
            }

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def keys(self):
        return self._shapes.keys()

    def get_slice(self, key):
        shape = self._shapes[key]
        return mock.MagicMock(get_shape=mock.MagicMock(return_value=shape))


def test_personaplex_builder_returns_one_flat_pinned_package():
    mimi = ModelPackage({"encoder": _component("encoder"), "decoder": _component("decoder")})
    moshi = ModelPackage(
        {"temporal": _component("temporal"), "depformer": _component("depformer")}
    )

    with (
        mock.patch.object(_builder, "_build_mimi", return_value=mimi) as build_mimi,
        mock.patch.object(_builder, "_build_moshi_lm", return_value=moshi) as build_moshi,
    ):
        package = _builder._build_personaplex(
            _builder._PERSONAPLEX_MODEL_ID,
            dtype="f32",
            execution_provider="cuda",
            load_weights=False,
        )

    assert set(package) == {"encoder", "decoder", "temporal", "depformer"}
    assert package.config.dep_q == 16
    build_mimi.assert_called_once_with(
        _builder._PERSONAPLEX_MODEL_ID,
        dtype=ir.DataType.FLOAT,
        execution_provider="cuda",
        revision=_builder._PERSONAPLEX_REVISION,
        load_weights=False,
    )
    build_moshi.assert_called_once_with(
        _builder._PERSONAPLEX_MODEL_ID,
        dtype=ir.DataType.FLOAT,
        execution_provider="cuda",
        revision=_builder._PERSONAPLEX_REVISION,
        load_weights=False,
        dep_q=16,
    )
    assert {model.metadata_props["mobius.source_revision"] for model in package.values()} == {
        _builder._PERSONAPLEX_REVISION
    }


def test_public_build_dispatches_before_transformers_detection():
    expected = ModelPackage({"encoder": _component("encoder")})

    with (
        mock.patch.object(
            _builder, "_build_personaplex", return_value=expected
        ) as native_build,
        mock.patch.object(
            transformers_builder, "_load_transformers_config"
        ) as transformers_probe,
    ):
        actual = transformers_builder.build_transformers_model(
            _builder._PERSONAPLEX_MODEL_ID,
            dtype="f32",
            execution_provider="cuda",
            load_weights=False,
        )

    assert actual is expected
    transformers_probe.assert_not_called()
    native_build.assert_called_once_with(
        _builder._PERSONAPLEX_MODEL_ID,
        dtype="f32",
        execution_provider="cuda",
        revision=_builder._PERSONAPLEX_REVISION,
        load_weights=False,
    )


def test_canonical_looking_local_checkpoint_dispatches_and_records_local_revision(
    tmp_path, monkeypatch
):
    checkpoint = _local_personaplex_checkpoint(tmp_path / "nvidia" / "personaplex-7b-v1")
    monkeypatch.chdir(tmp_path)
    mimi = ModelPackage({"encoder": _component("encoder"), "decoder": _component("decoder")})
    moshi = ModelPackage(
        {"temporal": _component("temporal"), "depformer": _component("depformer")}
    )

    with mock.patch("safetensors.safe_open", side_effect=_SafeTensorHeader):
        assert _builder._is_personaplex_checkpoint("nvidia/personaplex-7b-v1")
    assert _builder._personaplex_revision("nvidia/personaplex-7b-v1", "ignored") is None
    expected = ModelPackage({"encoder": _component("public-encoder")})
    with (
        mock.patch("safetensors.safe_open", side_effect=_SafeTensorHeader),
        mock.patch.object(_builder, "_build_personaplex", return_value=expected),
        mock.patch.object(
            transformers_builder, "_load_transformers_config"
        ) as transformers_probe,
    ):
        actual = transformers_builder.build_transformers_model(
            "nvidia/personaplex-7b-v1",
            load_weights=False,
        )
    assert actual is expected
    transformers_probe.assert_not_called()

    with (
        mock.patch.object(_builder, "_build_mimi", return_value=mimi),
        mock.patch.object(_builder, "_build_moshi_lm", return_value=moshi),
    ):
        package = _builder._build_personaplex(checkpoint, revision="ignored")

    assert {model.metadata_props["mobius.source_revision"] for model in package.values()} == {
        "local"
    }


def test_personaplex_hub_id_uses_pinned_revision(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert (
        _builder._personaplex_revision(_builder._PERSONAPLEX_MODEL_ID, None)
        == _builder._PERSONAPLEX_REVISION
    )


def test_other_existing_local_directory_forms_do_not_use_hub_revision(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    checkpoint = tmp_path / "checkpoints" / "personaplex"
    checkpoint.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    # Keep the relative form anchored to the temporary cwd, not the repository checkout.
    relative_checkpoint = checkpoint.relative_to(Path.cwd())
    assert relative_checkpoint == Path("checkpoints/personaplex")
    assert ntpath.normpath(relative_checkpoint.as_posix()) == r"checkpoints\personaplex"

    local_forms = [
        checkpoint,
        str(checkpoint),
        relative_checkpoint,
        "~/checkpoints/personaplex",
    ]
    for local_form in local_forms:
        assert _builder._personaplex_revision(local_form, "ignored") is None


def test_local_detection_fails_closed_for_ambiguous_checkpoint(tmp_path):
    checkpoint = _local_personaplex_checkpoint(tmp_path)
    (checkpoint / "tokenizer-second.safetensors").touch()

    assert not _builder._is_personaplex_checkpoint(checkpoint)


def test_personaplex_rejects_partial_global_dtype_override():
    with pytest.raises(ValueError, match="only supports dtype='f32'"):
        transformers_builder.build_transformers_model(
            _builder._PERSONAPLEX_MODEL_ID,
            dtype="f16",
            load_weights=False,
        )


def test_graph_only_native_builders_do_not_resolve_or_load_weights():
    graph_package = ModelPackage({"model": _component("model")})

    with (
        mock.patch("mobius._builder.build_from_module", return_value=graph_package),
        mock.patch("mobius.models.mimi.MimiModel"),
        mock.patch.object(_builder, "_resolve_mimi_checkpoint") as resolve_mimi,
    ):
        mimi = _builder._build_mimi(
            _builder._PERSONAPLEX_MODEL_ID,
            revision=_builder._PERSONAPLEX_REVISION,
            load_weights=False,
        )

    with (
        mock.patch(
            "mobius._builder.build_from_module",
            side_effect=[
                ModelPackage({"model": _component("temporal")}),
                ModelPackage({"model": _component("depformer")}),
            ],
        ),
        mock.patch("mobius.models.moshi.MoshiTemporalModel"),
        mock.patch("mobius.models.moshi.MoshiDepformerModel"),
        mock.patch.object(_builder, "_resolve_lm_checkpoint") as resolve_lm,
    ):
        moshi = _builder._build_moshi_lm(
            _builder._PERSONAPLEX_MODEL_ID,
            revision=_builder._PERSONAPLEX_REVISION,
            load_weights=False,
            dep_q=16,
        )

    assert set(mimi) == {"model"}
    assert set(moshi) == {"temporal", "depformer"}
    resolve_mimi.assert_not_called()
    resolve_lm.assert_not_called()


def test_public_graph_only_build_is_flat_without_checkpoint_downloads():
    with (
        mock.patch.object(_builder, "_resolve_mimi_checkpoint") as resolve_mimi,
        mock.patch.object(_builder, "_resolve_lm_checkpoint") as resolve_lm,
    ):
        package = transformers_builder.build_transformers_model(
            _builder._PERSONAPLEX_MODEL_ID,
            load_weights=False,
        )

    assert set(package) == {"encoder", "decoder", "temporal", "depformer"}
    resolve_mimi.assert_not_called()
    resolve_lm.assert_not_called()


def test_hub_revision_reaches_mimi_probe_and_both_downloads():
    api = mock.MagicMock()
    api.list_repo_files.return_value = ["tokenizer-checkpoint.safetensors"]

    with (
        mock.patch("huggingface_hub.HfApi", return_value=api),
        mock.patch(
            "huggingface_hub.hf_hub_download", return_value="/cached/checkpoint"
        ) as download,
    ):
        _builder._resolve_mimi_checkpoint(
            _builder._PERSONAPLEX_MODEL_ID,
            revision=_builder._PERSONAPLEX_REVISION,
        )
        _builder._resolve_lm_checkpoint(
            _builder._PERSONAPLEX_MODEL_ID,
            revision=_builder._PERSONAPLEX_REVISION,
        )

    api.list_repo_files.assert_called_once_with(
        _builder._PERSONAPLEX_MODEL_ID,
        revision=_builder._PERSONAPLEX_REVISION,
    )
    assert download.call_args_list == [
        mock.call(
            repo_id=_builder._PERSONAPLEX_MODEL_ID,
            filename="tokenizer-checkpoint.safetensors",
            revision=_builder._PERSONAPLEX_REVISION,
        ),
        mock.call(
            repo_id=_builder._PERSONAPLEX_MODEL_ID,
            filename="model.safetensors",
            revision=_builder._PERSONAPLEX_REVISION,
        ),
    ]


def test_special_native_audio_namespace_and_configs_are_not_public():
    import importlib

    import mobius.models as models

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("mobius.integrations.moshi")
    for name in (
        "mimi_default_config",
        "moshi_temporal_config",
        "moshi_depformer_config",
    ):
        assert name not in models.__all__
        assert not hasattr(models, name)
