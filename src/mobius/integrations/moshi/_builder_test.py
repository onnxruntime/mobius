# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from unittest import mock

import onnx_ir as ir

from mobius._model_package import ModelPackage
from mobius.integrations.moshi import _builder
from mobius.integrations.transformers import _builder as transformers_builder


def _component(name: str) -> ir.Model:
    graph = ir.Graph(
        inputs=[],
        outputs=[],
        nodes=[],
        name=name,
        opset_imports={"": 21},
    )
    return ir.Model(graph, ir_version=10)


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
            dtype="f16",
            execution_provider="cuda",
            load_weights=False,
        )

    assert set(package) == {"encoder", "decoder", "temporal", "depformer"}
    assert package.config.dep_q == 16
    build_mimi.assert_called_once_with(
        _builder._PERSONAPLEX_MODEL_ID,
        execution_provider="cuda",
        revision=_builder._PERSONAPLEX_REVISION,
        load_weights=False,
    )
    build_moshi.assert_called_once_with(
        _builder._PERSONAPLEX_MODEL_ID,
        dtype="f16",
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
            dtype="f16",
            execution_provider="cuda",
            load_weights=False,
        )

    assert actual is expected
    transformers_probe.assert_not_called()
    native_build.assert_called_once_with(
        _builder._PERSONAPLEX_MODEL_ID,
        dtype="f16",
        execution_provider="cuda",
        revision=_builder._PERSONAPLEX_REVISION,
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


def test_special_native_audio_builders_and_configs_are_not_public():
    import mobius.integrations.moshi as moshi_integration
    import mobius.models as models

    for name in ("build_mimi", "build_moshi_lm"):
        assert name not in moshi_integration.__all__
        assert not hasattr(moshi_integration, name)
    for name in (
        "mimi_default_config",
        "moshi_temporal_config",
        "moshi_depformer_config",
    ):
        assert name not in models.__all__
        assert not hasattr(models, name)
