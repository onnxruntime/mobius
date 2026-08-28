# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mobius.integrations.gguf._mmproj import (
    _interleaved_rope_rows,
    _preflight_mmproj_quantization_report,
    _preflight_standalone_mmproj,
)
from mobius.integrations.gguf._mmproj_mapping import (
    map_mmproj_audio_projector_to_onnx,
)
from mobius.integrations.gguf._mmproj_registry import get_projector_spec


@pytest.mark.parametrize(
    ("projector_type", "source", "expected"),
    [
        (
            "ultravox",
            "a.blk.3.attn_q.weight",
            "audio_encoder.layers.3.self_attn.q_proj.weight",
        ),
        (
            "voxtral",
            "mm.a.mlp.2.weight",
            "audio_encoder.projector.linear_2.weight",
        ),
        (
            "musicflamingo",
            "mm.a.mlp.1.bias",
            "audio_encoder.projector.linear_1.bias",
        ),
        (
            "lfm2a",
            "a.blk.2.linear_pos.weight",
            "audio_encoder.layers.2.self_attn.linear_pos.weight",
        ),
        (
            "parakeet",
            "a.blk.4.conv_norm_var",
            "audio_encoder.encoder.layers.4.conv.norm.running_var",
        ),
        (
            "granite_speech",
            "a.proj_blk.1.cross_attn_k.weight",
            "audio_encoder.projector.layers.1.cross_attn.k_proj.weight",
        ),
        (
            "mimo_audio",
            "mm.a.local_blk.0.ffn_gate.weight",
            "audio_encoder.local_layers.0.gate_proj.weight",
        ),
        (
            "pockettts_spkenc",
            "a.seanet.blk.2.scale_conv.weight",
            "speaker_encoder.seanet.stages.2.scale.weight",
        ),
    ],
)
def test_audio_projector_mapping_is_route_exact(
    projector_type: str,
    source: str,
    expected: str,
):
    assert map_mmproj_audio_projector_to_onnx(source, projector_type) == expected


def test_pockettts_generator_tensors_never_enter_speaker_encoder_graph():
    assert (
        map_mmproj_audio_projector_to_onnx(
            "a.gen.flow.input_proj.weight",
            "pockettts_spkenc",
        )
        is None
    )


def test_parakeet_preprocessor_assets_are_not_misclassified_as_graph_weights():
    assert map_mmproj_audio_projector_to_onnx("a.mel_filters", "parakeet") is None
    assert map_mmproj_audio_projector_to_onnx("a.window", "parakeet") is None


def test_unknown_audio_projector_route_fails_closed():
    with pytest.raises(ValueError, match="Unknown standalone"):
        map_mmproj_audio_projector_to_onnx("a.blk.0.attn_q.weight", "future-audio")


def test_pockettts_qk_row_transform_matches_adjacent_to_halfsplit_permutation():
    values = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)

    actual = _interleaved_rope_rows(values, head_dim=4)

    np.testing.assert_array_equal(actual, values[[0, 2, 1, 3]])


def test_audio_projector_evidence_is_test_only_immutable_and_bounded():
    evidence = json.loads(
        (Path("tests") / "data" / "gguf_audio_projector_evidence.json").read_text()
    )
    assert evidence["llama_cpp_revision"] == ("8d9af256337d1a501250f9bbf4c0859a654bddd6")
    routes = evidence["routes"]
    assert set(routes) == {
        "granite_speech",
        "lfm2a",
        "mimo_audio",
        "musicflamingo",
        "parakeet",
        "pockettts_spkenc",
        "ultravox",
        "voxtral",
    }
    assert set(evidence["converter_paths"]) == set(routes)
    assert all(
        path.startswith("conversion/") and path.endswith(".py")
        for path in evidence["converter_paths"].values()
    )
    total_bytes = 0
    for projector_type, route in routes.items():
        source = route["source"]
        assert len(source["revision"]) == 40
        artifact = route["artifact"]
        if artifact is None:
            total_bytes += route["evidence_bytes"]
            continue
        assert len(artifact["revision"]) == 40
        assert len(artifact["lfs_sha256"]) == 64
        assert sum(artifact["qtypes"].values()) == artifact["tensor_count"]
        assert artifact["graph_output_size"] > 0
        spec = get_projector_spec(projector_type)
        assert set(spec.required_metadata) <= set(artifact["metadata"])
        total_bytes += artifact["size"]
    assert total_bytes <= 16 * 1024**3
    assert routes["pockettts_spkenc"]["role"] == "speaker_encoder"
    assert routes["voxtral"]["artifact"]["storage_disposition"] == ("fail-closed-packed")
    assert (
        routes["ultravox"]["artifact"]["graph_output_size"]
        != routes["ultravox"]["artifact"]["metadata"]["clip.audio.projection_dim"]
    )
    assert "pockettts_gen" not in routes


def _standalone_sidecar(projector_type: str, *, qtype: str = "F32"):
    spec = get_projector_spec(projector_type)
    metadata: dict[str, object] = {
        "general.type": "mmproj",
        "clip.has_audio_encoder": True,
        "clip.audio.embedding_length": 8,
        "clip.audio.feed_forward_length": 16,
        "clip.audio.block_count": 1,
        "clip.audio.projection_dim": 8,
        "clip.audio.attention.head_count": 2,
        "clip.audio.attention.layer_norm_epsilon": 1e-5,
        "clip.audio.num_mel_bins": 4,
    }
    if "clip.projector_type" in spec.required_metadata:
        metadata["clip.projector_type"] = projector_type
    if "clip.audio.projector_type" in spec.required_metadata:
        metadata["clip.audio.projector_type"] = projector_type
    integer_defaults = {
        "clip.audio.projector.stack_factor": 2,
        "clip.audio.chunk_size": 4,
        "clip.audio.conv_kernel_size": 3,
        "clip.audio.max_pos_emb": 4,
        "clip.audio.projector.window_size": 4,
        "clip.audio.projector.downsample_rate": 2,
        "clip.audio.projector.head_count": 2,
        "clip.audio.subsampling_factor": 8,
        "clip.audio.rvq.num_quantizers": 2,
        "clip.audio.window_size": 4,
        "clip.audio.local_block_count": 1,
        "clip.audio.local_group_size": 2,
    }
    metadata.update(
        {
            key: value
            for key, value in integer_defaults.items()
            if key in spec.required_metadata
        }
    )
    if "clip.audio.rvq.codebook_size" in spec.required_metadata:
        metadata["clip.audio.rvq.codebook_size"] = [5, 6]
    if "clip.audio.wa_pattern_mode" in spec.required_metadata:
        metadata["clip.audio.wa_pattern_mode"] = [-1]

    names = set(spec.required_top_tensors)
    names.update(
        f"{spec.block_prefix}{layer}.{suffix}"
        for layer in range(int(metadata["clip.audio.block_count"]))
        for suffix in spec.block_suffixes
    )
    if projector_type == "mimo_audio":
        names.update(
            {
                "mm.a.local_blk.0.attn_q.weight",
                "mm.a.local_blk.0.attn_q.bias",
                "mm.a.local_blk.0.attn_k.weight",
                "mm.a.local_blk.0.attn_k.bias",
                "mm.a.local_blk.0.attn_v.weight",
                "mm.a.local_blk.0.attn_v.bias",
                "mm.a.local_blk.0.attn_out.weight",
                "mm.a.local_blk.0.ffn_gate.weight",
                "mm.a.local_blk.0.ffn_up.weight",
                "mm.a.local_blk.0.ffn_down.weight",
                "mm.a.local_blk.0.ln1.weight",
                "mm.a.local_blk.0.ln2.weight",
            }
        )
    types = {name: SimpleNamespace(name=qtype) for name in names}
    return SimpleNamespace(
        architecture="clip",
        metadata=metadata,
        tensor_names=tuple(sorted(names)),
        get_tensor_type=types.__getitem__,
    )


@pytest.mark.parametrize(
    "projector_type",
    (
        "granite_speech",
        "lfm2a",
        "mimo_audio",
        "musicflamingo",
        "parakeet",
        "ultravox",
        "voxtral",
        "pockettts_spkenc",
    ),
)
def test_audio_projector_preflight_closes_every_promoted_route(projector_type: str):
    sidecar = _standalone_sidecar(projector_type)
    target = next(iter(get_projector_spec(projector_type).target_architectures))

    spec = _preflight_standalone_mmproj(
        sidecar,
        projector_type=projector_type,
        target_architecture=target,
    )

    assert spec.projector_type == projector_type
    assert spec.sidecar_builder == "audio_projector"


def test_audio_projector_preflight_rejects_packed_graph_tensor():
    sidecar = _standalone_sidecar("voxtral", qtype="Q8_0")

    with pytest.raises(NotImplementedError, match=r"packed Q8_0"):
        _preflight_standalone_mmproj(
            sidecar,
            projector_type="voxtral",
            target_architecture="llama",
        )


def test_audio_projector_fingerprint_rejects_unknown_tensor():
    sidecar = _standalone_sidecar("ultravox")
    sidecar.tensor_names = (*sidecar.tensor_names, "a.future.weight")

    with pytest.raises(ValueError, match="outside the pinned suffix-exact"):
        _preflight_standalone_mmproj(
            sidecar,
            projector_type="ultravox",
            target_architecture="llama",
        )


def test_pockettts_generator_namespace_is_quarantined_not_promoted():
    sidecar = _standalone_sidecar("pockettts_spkenc")
    sidecar.metadata["clip.has_gen_audio_encoder"] = True
    sidecar.metadata["clip.gen.audio.projector_type"] = "pockettts_gen"
    sidecar.tensor_names = (*sidecar.tensor_names, "a.gen.flow.input_proj.weight")
    tensor_types = {name: SimpleNamespace(name="F32") for name in sidecar.tensor_names}
    sidecar.get_tensor_type = tensor_types.__getitem__

    spec = _preflight_standalone_mmproj(
        sidecar,
        projector_type="pockettts_spkenc",
        target_architecture="pockettts",
    )

    assert [role.value for role in spec.model_roles] == ["speaker_encoder"]
    assert get_projector_spec("pockettts_gen").is_importable is False
    generator_role = next(
        role
        for prefix, role in spec.tensor_roles
        if "a.gen.flow.input_proj.weight".startswith(prefix)
    )
    assert generator_role.value == "generated_audio"


def test_vision_companion_quarantine_never_hides_unknown_audio_projector_tensor():
    sidecar = _standalone_sidecar("mimo_audio")
    sidecar.metadata["clip.has_vision_encoder"] = True
    sidecar.metadata["clip.vision.projector_type"] = "mimovl"
    sidecar.tensor_names = (
        *sidecar.tensor_names,
        "v.patch_embd.weight",
        "mm.0.weight",
        "mm.2.weight",
        "mm.a.future.weight",
    )
    tensor_types = {name: SimpleNamespace(name="F32") for name in sidecar.tensor_names}
    sidecar.get_tensor_type = tensor_types.__getitem__

    with pytest.raises(ValueError, match=r"mm\.a\.future\.weight"):
        _preflight_standalone_mmproj(
            sidecar,
            projector_type="mimo_audio",
            target_architecture="mimo2",
        )


@pytest.mark.parametrize(
    ("projector_type", "target_architecture", "role"),
    [
        ("ultravox", "llama", "audio_encoder"),
        ("pockettts_spkenc", "pockettts", "speaker_encoder"),
    ],
)
def test_public_standalone_dispatch_enforces_declared_model_role(
    monkeypatch,
    projector_type: str,
    target_architecture: str,
    role: str,
):
    from mobius.integrations.gguf import _builder, _mmproj

    sidecar = _standalone_sidecar(projector_type)
    monkeypatch.setattr(_builder, "_validate_gguf_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        _mmproj,
        "build_audio_projector_from_gguf",
        lambda *args, **kwargs: {role: object()},
    )

    package = _mmproj.build_mmproj_from_gguf(
        "synthetic.gguf",
        projector_type=projector_type,
        target_architecture=target_architecture,
        _mmproj_gguf_model=sidecar,
    )

    assert set(package) == {role}


def test_standalone_audio_quantization_report_records_only_graph_weights():
    from gguf import GGMLQuantizationType

    tensors = [
        SimpleNamespace(
            name="a.conv1d.1.weight",
            tensor_type=GGMLQuantizationType.F16,
            n_bytes=32,
        ),
        SimpleNamespace(
            name="mm.a.mlp.1.weight",
            tensor_type=GGMLQuantizationType.F32,
            n_bytes=64,
        ),
        SimpleNamespace(
            name="a.unmapped_processor_asset",
            tensor_type=GGMLQuantizationType.F32,
            n_bytes=16,
        ),
    ]
    sidecar = SimpleNamespace(reader_tensors=lambda: tensors)

    report = _preflight_mmproj_quantization_report(
        sidecar,
        include_audio=True,
        standalone_projector_type="ultravox",
    )

    assert {record.name for record in report.tensor_records} == {
        "mmproj:a.conv1d.1.weight",
        "mmproj:mm.a.mlp.1.weight",
    }
    assert sum(stat.source_bytes for stat in report.source_qtype_census) == 112
    assert report.target_storage_format == "float"
