# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import json
import pathlib

import pytest
import safetensors.torch
import torch

from mobius._diffusers_checkpoint import (
    component_class,
    component_shard_paths,
    component_weight_names,
    load_checkpoint_json,
    load_component_weights,
    load_optional_checkpoint_json,
    resolve_assets,
    resolve_checkpoint_file,
)


def _write(path: pathlib.Path, value: object) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_resolve_file_returns_local_path(tmp_path) -> None:
    _write(tmp_path / "scheduler" / "scheduler_config.json", {"_class_name": "Flow"})

    path = resolve_checkpoint_file(str(tmp_path), "scheduler/scheduler_config.json")

    assert path is not None
    assert pathlib.Path(path) == (tmp_path / "scheduler" / "scheduler_config.json").resolve()


def test_resolve_file_rejects_paths_escaping_the_checkpoint(tmp_path) -> None:
    _write(tmp_path.parent / "secret.json", {"token": "value"})
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="escapes model directory"):
        resolve_checkpoint_file(str(tmp_path), "../secret.json")


def test_resolve_file_reports_missing_required_file(tmp_path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Required checkpoint file not found"):
        resolve_checkpoint_file(str(tmp_path), "model_index.json")


def test_resolve_file_skips_missing_optional_file(tmp_path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    assert resolve_checkpoint_file(str(tmp_path), "tokenizer.json", required=False) is None


def test_load_json_returns_object_and_path(tmp_path) -> None:
    _write(tmp_path / "config.json", {"model_type": "example"})

    value, path = load_checkpoint_json(str(tmp_path), "config.json")

    assert value == {"model_type": "example"}
    assert pathlib.Path(path).name == "config.json"


def test_load_json_rejects_non_object_documents(tmp_path) -> None:
    _write(tmp_path / "config.json", [1, 2, 3])

    with pytest.raises(TypeError, match="must contain a JSON object"):
        load_checkpoint_json(str(tmp_path), "config.json")


def test_optional_json_defaults_to_empty_mapping(tmp_path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    assert load_optional_checkpoint_json(str(tmp_path), "generation_config.json") == {}


def test_optional_json_reads_present_file(tmp_path) -> None:
    _write(tmp_path / "generation_config.json", {"top_k": 20})

    assert load_optional_checkpoint_json(str(tmp_path), "generation_config.json") == {
        "top_k": 20
    }


def test_component_class_reads_model_index_entries() -> None:
    index = {
        "transformer": ["diffusers", "Cosmos3OmniTransformer"],
        "sound_tokenizer": [None, None],
    }

    assert component_class(index, "transformer") == "Cosmos3OmniTransformer"
    assert component_class(index, "sound_tokenizer") is None
    assert component_class(index, "absent") is None


def test_component_class_rejects_malformed_entries() -> None:
    with pytest.raises(ValueError, match=r"Invalid model_index\.json entry"):
        component_class({"vae": "AutoencoderKLWan"}, "vae")


def test_component_weight_names_reads_single_shard_metadata(tmp_path) -> None:
    component = tmp_path / "vae"
    component.mkdir()
    safetensors.torch.save_file(
        {"encoder.conv.weight": torch.zeros(1), "decoder.conv.weight": torch.zeros(1)},
        str(component / "diffusion_pytorch_model.safetensors"),
    )

    names = component_weight_names(str(tmp_path), "vae")

    assert names == {"encoder.conv.weight", "decoder.conv.weight"}


def test_component_weight_names_reads_sharded_index(tmp_path) -> None:
    _write(
        tmp_path / "transformer" / "model.safetensors.index.json",
        {"weight_map": {"a.weight": "model-00001.safetensors"}},
    )

    assert component_weight_names(str(tmp_path), "transformer") == {"a.weight"}


def test_component_weight_names_requires_a_checkpoint(tmp_path) -> None:
    (tmp_path / "vae").mkdir()

    with pytest.raises(FileNotFoundError, match="No safetensors checkpoint"):
        component_weight_names(str(tmp_path), "vae")


def test_component_weight_names_rejects_escaping_component_names(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unsafe component path"):
        component_weight_names(str(tmp_path), "../elsewhere")


def test_component_shard_paths_rejects_escaping_shard_entries(tmp_path) -> None:
    _write(
        tmp_path / "vae" / "diffusion_pytorch_model.safetensors.index.json",
        {"weight_map": {"a.weight": "../outside.safetensors"}},
    )

    with pytest.raises(ValueError, match="Unsafe component weight filename"):
        component_shard_paths(tmp_path, "vae")


def test_component_shard_paths_rejects_escaping_component_names(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unsafe component path"):
        component_shard_paths(tmp_path, "../elsewhere")


def test_component_shard_paths_orders_index_shards(tmp_path) -> None:
    component = tmp_path / "vae"
    component.mkdir()
    for name in ("model-00002.safetensors", "model-00001.safetensors"):
        safetensors.torch.save_file({"w": torch.zeros(1)}, str(component / name))
    _write(
        component / "diffusion_pytorch_model.safetensors.index.json",
        {
            "weight_map": {
                "b.weight": "model-00002.safetensors",
                "a.weight": "model-00001.safetensors",
            }
        },
    )

    paths = component_shard_paths(tmp_path, "vae")

    assert [path.name for path in paths] == [
        "model-00001.safetensors",
        "model-00002.safetensors",
    ]


def test_load_component_weights_merges_local_shards(tmp_path) -> None:
    component = tmp_path / "vae"
    component.mkdir()
    safetensors.torch.save_file(
        {"a.weight": torch.ones(2)},
        str(component / "model-00001.safetensors"),
    )
    safetensors.torch.save_file(
        {"b.weight": torch.zeros(2)},
        str(component / "model-00002.safetensors"),
    )
    _write(
        component / "model.safetensors.index.json",
        {
            "weight_map": {
                "a.weight": "model-00001.safetensors",
                "b.weight": "model-00002.safetensors",
            }
        },
    )

    weights = load_component_weights(str(tmp_path), "vae")

    assert set(weights) == {"a.weight", "b.weight"}


def test_resolve_assets_skips_absent_optional_candidates(tmp_path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    _write(tmp_path / "text_tokenizer" / "tokenizer.json", {})

    assets = resolve_assets(
        str(tmp_path),
        (
            ("config.json", True),
            ("tokenizer.json", False),
            ("text_tokenizer/tokenizer.json", False),
        ),
    )

    assert set(assets) == {"config.json", "text_tokenizer/tokenizer.json"}
    assert assets["config.json"][1] is True
    assert assets["text_tokenizer/tokenizer.json"][1] is False


def test_resolve_assets_propagates_missing_required_candidate(tmp_path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        resolve_assets(str(tmp_path), (("model_index.json", True),))
