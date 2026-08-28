# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import copy
import hashlib
import json
import lzma
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx_ir as ir
import pytest
import torch
from onnxscript import nn

from mobius.integrations.gguf import _ocr_projector
from mobius.integrations.gguf._mmproj import (
    _preflight_standalone_mmproj,
    _validate_mmproj_tensor_closure,
)
from mobius.integrations.gguf._mmproj_mapping import map_ocr_projector_to_onnx
from mobius.integrations.gguf._mmproj_registry import (
    MMPROJ_ARTIFACT_PINS,
    MMProjModelRole,
    get_projector_spec,
)

_EVIDENCE_PATH = (
    Path(__file__).resolve().parents[4]
    / "tests"
    / "data"
    / "gguf_ocr_projector_headers.json.xz"
)
_ROUTE_HEADERS = {
    "deepseekocr": "deepseekocr",
    "deepseekocr2": "deepseekocr2",
    "dots_ocr": "dots_ocr",
    "dots3note_v": "dots3note",
    "dots3note_a": "dots3note",
    "paddleocr": "paddleocr",
    "lightonocr": "lightonocr",
    "youtuvl": "youtuvl",
    "granite4_vision": "granite4_vision",
}
_TARGETS = {
    "deepseekocr": "deepseek2-ocr",
    "deepseekocr2": "deepseek2-ocr",
    "dots_ocr": "qwen2",
    "dots3note_v": "dots3note",
    "dots3note_a": "dots3note",
    "paddleocr": "paddleocr",
    "lightonocr": "qwen3",
    "youtuvl": "deepseek2",
    "granite4_vision": "granite",
}
_HEADER_FINGERPRINTS = {
    "deepseekocr": "b570a263a2aacc9597815cf2fd5dbc5bee17087565536b0444395e1dfa0f0cc2",
    "deepseekocr2": "f9c89bcde7c6e2c4285c239ab170a7039ceff6561fb95466b6f23d19ff9b6de4",
    "dots3note": "32dbda6016815d1917f6cd27fb71bcb46f14fa639f0c26a44b3dd28252a675c1",
    "dots_ocr": "10783cb4cc179dca80814a7baed4821a3fe1e765b6bdd2ba2dba2ec9ed20e584",
    "granite4_vision": "e7520559b1a65b09c0f3ecad327a102ea1c5280982ae5f68fc178341abcaad89",
    "lightonocr": "549408abf98d33ee11fcb72dda3b00fb59f3c2ad81157eb5658fe678eb85a43d",
    "paddleocr": "bc3e9022809fe995ad83c887d714f61e338146eaa8bb4e0f32bfdce68ecf7bbd",
    "youtuvl": "b42f9528fe234ed8948f714aec626a4654638409232390d5b69e9b53dde8fd4d",
}


def _evidence() -> dict[str, dict]:
    with lzma.open(_EVIDENCE_PATH, "rt", encoding="utf-8") as stream:
        return json.load(stream)


class _HeaderFixture:
    def __init__(self, item: dict):
        self.architecture = item["architecture"]
        self.metadata = copy.deepcopy(item["metadata"])
        self._tensors = {
            tensor["name"]: {
                "shape": tuple(tensor["shape"]),
                "qtype": tensor["qtype"],
            }
            for tensor in item["tensors"]
        }
        self.tensor_names = list(self._tensors)

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        return self._tensors[name]["shape"]

    def get_tensor_type(self, name: str):
        return SimpleNamespace(name=self._tensors[name]["qtype"])

    def remove(self, name: str) -> None:
        del self._tensors[name]
        self.tensor_names.remove(name)


@pytest.mark.parametrize("projector_type", tuple(_ROUTE_HEADERS))
def test_immutable_real_header_has_exact_suffix_closure(projector_type: str):
    item = _evidence()[_ROUTE_HEADERS[projector_type]]
    source = _HeaderFixture(item)

    spec = _preflight_standalone_mmproj(
        source,
        projector_type=projector_type,
        target_architecture=_TARGETS[projector_type],
    )

    assert spec.sidecar_builder == "ocr_projector"
    assert spec.model_roles
    assert spec.runtime.value == "deferred"


def test_immutable_headers_have_distinct_source_fingerprints():
    evidence = _evidence()

    actual = {
        name: hashlib.sha256(
            json.dumps(item, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        for name, item in evidence.items()
    }

    assert actual == _HEADER_FINGERPRINTS
    assert len(set(actual.values())) == len(actual)


def test_artifact_pins_cover_every_ocr_header_and_both_dots_roles():
    pins = {
        projector_type: pin
        for pin in MMPROJ_ARTIFACT_PINS
        for projector_type in pin.projector_types
        if projector_type in _ROUTE_HEADERS
    }

    assert set(pins) == set(_ROUTE_HEADERS)
    assert pins["dots3note_v"] is pins["dots3note_a"]
    assert pins["dots3note_v"].size < 16 * 1024**3
    for projector_type, pin in pins.items():
        item = _evidence()[_ROUTE_HEADERS[projector_type]]
        assert item["tensor_count"] == pin.tensor_count
        assert tuple(sorted(item["tensor_qtypes"].items())) == tuple(
            sorted(pin.tensor_qtypes)
        )


@pytest.mark.parametrize("projector_type", tuple(_ROUTE_HEADERS))
def test_real_header_rejects_unknown_extra_tensor(projector_type: str):
    source = _HeaderFixture(_evidence()[_ROUTE_HEADERS[projector_type]])
    source._tensors["future.extra.weight"] = {"shape": (1,), "qtype": "F32"}
    source.tensor_names.append("future.extra.weight")

    with pytest.raises(ValueError, match="outside the pinned suffix-exact"):
        _validate_mmproj_tensor_closure(source, get_projector_spec(projector_type))


@pytest.mark.parametrize("projector_type", tuple(_ROUTE_HEADERS))
def test_real_header_rejects_missing_required_top_tensor(projector_type: str):
    source = _HeaderFixture(_evidence()[_ROUTE_HEADERS[projector_type]])
    spec = get_projector_spec(projector_type)
    source.remove(spec.required_top_tensors[0])

    with pytest.raises(ValueError, match="missing required tensor"):
        _validate_mmproj_tensor_closure(source, spec)


@pytest.mark.parametrize("projector_type", tuple(_ROUTE_HEADERS))
def test_real_header_rejects_partial_block_family(projector_type: str):
    source = _HeaderFixture(_evidence()[_ROUTE_HEADERS[projector_type]])
    spec = get_projector_spec(projector_type)
    source.remove(f"{spec.block_prefix}0.{spec.block_suffixes[0]}")

    with pytest.raises(ValueError, match="missing required"):
        _validate_mmproj_tensor_closure(source, spec)


@pytest.mark.parametrize("projector_type", tuple(_ROUTE_HEADERS))
def test_real_header_rejects_packed_projector_import(projector_type: str):
    source = _HeaderFixture(_evidence()[_ROUTE_HEADERS[projector_type]])
    spec = get_projector_spec(projector_type)
    source._tensors[spec.required_top_tensors[0]]["qtype"] = "Q8_0"

    with pytest.raises(NotImplementedError, match="does not preserve packed"):
        _validate_mmproj_tensor_closure(source, spec)


@pytest.mark.parametrize("projector_type", tuple(_ROUTE_HEADERS))
def test_real_header_rejects_boolean_numeric_metadata(projector_type: str):
    source = _HeaderFixture(_evidence()[_ROUTE_HEADERS[projector_type]])
    prefix = get_projector_spec(projector_type).primary_modality.value
    source.metadata[f"clip.{prefix}.embedding_length"] = True

    with pytest.raises(ValueError, match="positive integer"):
        _validate_mmproj_tensor_closure(source, get_projector_spec(projector_type))


@pytest.mark.parametrize("projector_type", tuple(_ROUTE_HEADERS))
def test_real_header_requires_exact_boolean_presence(projector_type: str):
    source = _HeaderFixture(_evidence()[_ROUTE_HEADERS[projector_type]])
    modality = get_projector_spec(projector_type).primary_modality.value
    source.metadata[f"clip.has_{modality}_encoder"] = 1

    with pytest.raises(ValueError, match="requires .*true"):
        _validate_mmproj_tensor_closure(source, get_projector_spec(projector_type))


def test_youtu_and_granite_reject_malformed_schedule_arrays():
    evidence = _evidence()
    youtu = _HeaderFixture(evidence["youtuvl"])
    youtu.metadata["clip.vision.wa_layer_indexes"] = [True]
    with pytest.raises(ValueError, match="integer array"):
        _validate_mmproj_tensor_closure(youtu, get_projector_spec("youtuvl"))

    granite = _HeaderFixture(evidence["granite4_vision"])
    granite.metadata["clip.vision.projector.spatial_offsets"] = [-1]
    with pytest.raises(ValueError, match="equal length"):
        _validate_mmproj_tensor_closure(
            granite,
            get_projector_spec("granite4_vision"),
        )


def test_dots_expert_count_metadata_is_informational_not_a_production_gate():
    source = _HeaderFixture(_evidence()["dots3note"])
    source.metadata["clip.vision.expert_count_per_layer"] = [0] * 42

    _validate_mmproj_tensor_closure(source, get_projector_spec("dots3note_v"))


def test_dots_mixed_sidecar_emits_both_executable_roles():
    expected = (
        MMProjModelRole.VISION_ENCODER,
        MMProjModelRole.AUDIO_ENCODER,
    )
    assert get_projector_spec("dots3note_v").model_roles == expected
    assert get_projector_spec("dots3note_a").model_roles == expected


def test_every_real_tensor_is_mapped_or_an_explicit_upstream_compatibility_tensor():
    compatibility_only = {
        ("deepseekocr", "v.patch_embd.weight"),
        ("granite4_vision", "v.post_ln.weight"),
        ("granite4_vision", "v.post_ln.bias"),
    }
    for projector_type, header_name in _ROUTE_HEADERS.items():
        item = _evidence()[header_name]
        for tensor in item["tensors"]:
            name = tensor["name"]
            route = projector_type
            if projector_type in {"dots3note_v", "dots3note_a"}:
                route = (
                    "dots3note_a"
                    if name.startswith(("a.", "mm.a."))
                    else "dots3note_v"
                )
            if map_ocr_projector_to_onnx(name, route) is None:
                assert (route, name) in compatibility_only


@pytest.mark.parametrize(
    "projector_type",
    tuple(route for route in _ROUTE_HEADERS if route != "dots3note_a"),
)
def test_real_header_constructs_a_dedicated_vision_component(projector_type: str):
    source = _HeaderFixture(_evidence()[_ROUTE_HEADERS[projector_type]])

    module, config = _ocr_projector._vision_module(
        source,
        projector_type,
        dtype=None,
    )

    assert module.input_schema
    assert list(module.named_parameters())
    assert config.dtype.name == "FLOAT"


def test_real_header_constructs_dots_audio_component():
    source = _HeaderFixture(_evidence()["dots3note"])

    module = _ocr_projector._dots_audio(source)

    assert module.input_schema[0][0] == "input_features"
    assert list(module.named_parameters())


class _VisionStub(nn.Module):
    input_schema = (("pixel_values", ir.DataType.FLOAT, (1, 2)),)

    def forward(self, op, pixel_values):
        return pixel_values


class _AudioStub(nn.Module):
    input_schema = (("input_features", ir.DataType.FLOAT, (1, 2)),)

    def forward(self, op, input_features):
        return input_features


@pytest.mark.parametrize("projector_type", tuple(_ROUTE_HEADERS))
def test_route_builder_emits_declared_standalone_roles(projector_type: str, monkeypatch):
    config = _ocr_projector._standalone_config(2, dtype=None)
    monkeypatch.setattr(
        _ocr_projector,
        "_vision_module",
        lambda *args, **kwargs: (_VisionStub(), config),
    )
    monkeypatch.setattr(_ocr_projector, "_dots_audio", lambda *args: _AudioStub())
    monkeypatch.setattr(_ocr_projector, "_map_state", lambda *args, **kwargs: {})

    package = _ocr_projector.build_ocr_projector_from_gguf(
        "unused.gguf",
        projector_type=projector_type,
        target_architecture=_TARGETS[projector_type],
        dtype=None,
        execution_provider="default",
        _mmproj_gguf_model=object(),
    )

    assert set(package) == {
        role.value for role in get_projector_spec(projector_type).model_roles
    }


class _MapFixture:
    def __init__(self, tensors: dict[str, np.ndarray], metadata: dict | None = None):
        self._tensors = tensors
        self.tensor_names = list(tensors)
        self.metadata = metadata or {}

    def get_tensor(self, name: str) -> np.ndarray:
        return self._tensors[name]


def test_lighton_qk_rows_are_reverse_permuted_by_value():
    from mobius.integrations.gguf._tensor_processors import _reverse_permute

    stored = np.arange(8 * 4, dtype=np.float32).reshape(8, 4)
    source = _MapFixture(
        {"v.blk.0.attn_q.weight": stored},
        {"clip.vision.attention.head_count": 2},
    )

    state = _ocr_projector._map_state(source, "lightonocr", mixed=False)

    expected = _reverse_permute(torch.from_numpy(stored), 2)
    torch.testing.assert_close(
        state["vision_encoder.vision_tower.transformer.layers.0.attention.q_proj.weight"],
        expected,
    )


def test_deepseek_source_tensor_is_shared_across_global_and_local_graphs():
    source = _MapFixture(
        {"v.sam.neck.0.weight": np.ones((2, 2, 1, 1), dtype=np.float32)}
    )

    state = _ocr_projector._map_state(source, "deepseekocr2", mixed=False)

    global_weight = state["vision_encoder.global_encoder.sam.neck.0.weight"]
    local_weight = state["vision_encoder.local_encoder.sam.neck.0.weight"]
    assert global_weight is local_weight


def test_dots_mixed_state_routes_each_modality_to_its_own_component():
    source = _MapFixture(
        {
            "v.patch_embd.bias": np.ones((2,), dtype=np.float32),
            "a.conv2d.1.bias": np.ones((2,), dtype=np.float32),
        }
    )

    state = _ocr_projector._map_state(source, "dots3note_v", mixed=True)

    assert set(state) == {
        "vision_encoder.patch_embed.bias",
        "audio_encoder.conv2d.0.bias",
    }
