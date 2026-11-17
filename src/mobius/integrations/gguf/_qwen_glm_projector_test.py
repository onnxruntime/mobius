# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Executable coverage for the Qwen/GLM GGUF projector cohort."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import numpy as np
import onnxruntime as ort
import pytest
import torch
import torch.nn.functional as functional

from mobius._testing.ort_inference import OnnxModelSession
from mobius.integrations.gguf._mmproj import build_mmproj_from_gguf
from mobius.integrations.gguf._qwen_glm_projector import (
    qwen3vl_decoder_mrope_positions,
    validate_qwen_glm_projector_metadata,
)


class _FakeSidecar:
    architecture = "clip"

    def __init__(
        self,
        metadata: dict[str, object],
        shapes: dict[str, tuple[int, ...]],
        *,
        seed: int = 0,
    ) -> None:
        self.metadata = {
            "general.architecture": "clip",
            "general.type": "mmproj",
            **metadata,
        }
        self.tensor_names = list(shapes)
        self._shapes = shapes
        rng = np.random.default_rng(seed)
        self._values = {
            name: (
                rng.standard_normal(shape).astype(np.float32) * 0.03
                if shape
                else np.array(0.0, dtype=np.float32)
            )
            for name, shape in shapes.items()
        }

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        return self._shapes[name]

    def get_tensor_type(self, name: str):
        del name
        return SimpleNamespace(name="F32")

    def get_tensor(self, name: str) -> np.ndarray:
        return self._values[name]


def _vision_metadata(
    *,
    projector_type: str,
    hidden: int,
    intermediate: int,
    output: int,
    patch: int,
    layers: int = 1,
) -> dict[str, object]:
    return {
        "clip.has_vision_encoder": True,
        "clip.projector_type": projector_type,
        "clip.vision.embedding_length": hidden,
        "clip.vision.feed_forward_length": intermediate,
        "clip.vision.block_count": layers,
        "clip.vision.projection_dim": output,
        "clip.vision.attention.head_count": 2,
        "clip.vision.attention.layer_norm_epsilon": 1e-6,
        "clip.vision.image_size": patch * 2,
        "clip.vision.patch_size": patch,
        "clip.vision.image_mean": [0.5, 0.5, 0.5],
        "clip.vision.image_std": [0.5, 0.5, 0.5],
    }


def _audio_metadata(
    *,
    projector_type: str,
    hidden: int,
    intermediate: int,
    output: int,
    mel: int,
    layers: int = 1,
    global_type: bool = False,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "clip.has_audio_encoder": True,
        "clip.audio.embedding_length": hidden,
        "clip.audio.feed_forward_length": intermediate,
        "clip.audio.block_count": layers,
        "clip.audio.projection_dim": output,
        "clip.audio.attention.head_count": 2,
        "clip.audio.attention.layer_norm_epsilon": 1e-5,
        "clip.audio.num_mel_bins": mel,
    }
    metadata["clip.projector_type" if global_type else "clip.audio.projector_type"] = (
        projector_type
    )
    return metadata


def _whisper_block_shapes(
    *,
    hidden: int,
    intermediate: int,
    key_bias: bool,
    prefix: str = "a.blk.0.",
) -> dict[str, tuple[int, ...]]:
    shapes = {
        prefix + "ln1.weight": (hidden,),
        prefix + "ln1.bias": (hidden,),
        prefix + "ln2.weight": (hidden,),
        prefix + "ln2.bias": (hidden,),
        prefix + "attn_q.weight": (hidden, hidden),
        prefix + "attn_q.bias": (hidden,),
        prefix + "attn_k.weight": (hidden, hidden),
        prefix + "attn_v.weight": (hidden, hidden),
        prefix + "attn_v.bias": (hidden,),
        prefix + "attn_out.weight": (hidden, hidden),
        prefix + "attn_out.bias": (hidden,),
        prefix + "ffn_up.weight": (intermediate, hidden),
        prefix + "ffn_up.bias": (intermediate,),
        prefix + "ffn_down.weight": (hidden, intermediate),
        prefix + "ffn_down.bias": (hidden,),
    }
    if key_bias:
        shapes[prefix + "attn_k.bias"] = (hidden,)
    return shapes


def _qwen2a_sidecar(*, alias: bool = False) -> _FakeSidecar:
    hidden, intermediate, output, mel = 8, 16, 12, 8
    metadata = _audio_metadata(
        projector_type="qwen2.5o" if alias else "qwen2a",
        hidden=hidden,
        intermediate=intermediate,
        output=output,
        mel=mel,
        global_type=True,
    )
    shapes = {
        "a.conv1d.1.weight": (hidden, mel, 3),
        "a.conv1d.1.bias": (hidden, 1),
        "a.conv1d.2.weight": (hidden, hidden, 3),
        "a.conv1d.2.bias": (hidden, 1),
        "a.position_embd.weight": (8, hidden),
        "a.post_ln.weight": (hidden,),
        "a.post_ln.bias": (hidden,),
        "mm.a.fc.weight": (output, hidden),
        "mm.a.fc.bias": (output,),
        **_whisper_block_shapes(
            hidden=hidden,
            intermediate=intermediate,
            key_bias=False,
        ),
    }
    return _FakeSidecar(metadata, shapes, seed=2)


def _qwen3a_sidecar() -> _FakeSidecar:
    hidden, intermediate, output, mel, channels = 8, 16, 12, 8, 4
    metadata = _audio_metadata(
        projector_type="qwen3a",
        hidden=hidden,
        intermediate=intermediate,
        output=output,
        mel=mel,
    )
    shapes = {
        "a.conv2d.1.weight": (channels, 1, 3, 3),
        "a.conv2d.1.bias": (channels, 1, 1),
        "a.conv2d.2.weight": (channels, channels, 3, 3),
        "a.conv2d.2.bias": (channels, 1, 1),
        "a.conv2d.3.weight": (channels, channels, 3, 3),
        "a.conv2d.3.bias": (channels, 1, 1),
        "a.conv_out.weight": (hidden, channels),
        "a.position_embd.weight": (13, hidden),
        "a.post_ln.weight": (hidden,),
        "a.post_ln.bias": (hidden,),
        "mm.a.mlp.1.weight": (hidden, hidden),
        "mm.a.mlp.1.bias": (hidden,),
        "mm.a.mlp.2.weight": (output, hidden),
        "mm.a.mlp.2.bias": (output,),
        **_whisper_block_shapes(
            hidden=hidden,
            intermediate=intermediate,
            key_bias=True,
        ),
    }
    return _FakeSidecar(metadata, shapes, seed=3)


def _qwen3vl_sidecar() -> _FakeSidecar:
    hidden, intermediate, output, patch = 8, 16, 12, 2
    metadata = {
        **_vision_metadata(
            projector_type="qwen3vl_merger",
            hidden=hidden,
            intermediate=intermediate,
            output=output,
            patch=patch,
        ),
        "clip.vision.spatial_merge_size": 2,
        "clip.vision.is_deepstack_layers": [True],
        "clip.use_gelu": True,
    }
    merged = hidden * 4
    shapes = {
        "v.patch_embd.weight": (hidden, 3, patch, patch),
        "v.patch_embd.weight.1": (hidden, 3, patch, patch),
        "v.patch_embd.bias": (hidden,),
        "v.position_embd.weight": (4, hidden),
        "v.post_ln.weight": (hidden,),
        "v.post_ln.bias": (hidden,),
        "mm.0.weight": (merged, merged),
        "mm.0.bias": (merged,),
        "mm.2.weight": (output, merged),
        "mm.2.bias": (output,),
    }
    for stem, shape in {
        "ln1.weight": (hidden,),
        "ln1.bias": (hidden,),
        "ln2.weight": (hidden,),
        "ln2.bias": (hidden,),
        "attn_qkv.weight": (3 * hidden, hidden),
        "attn_qkv.bias": (3 * hidden,),
        "attn_out.weight": (hidden, hidden),
        "attn_out.bias": (hidden,),
        "ffn_up.weight": (intermediate, hidden),
        "ffn_up.bias": (intermediate,),
        "ffn_down.weight": (hidden, intermediate),
        "ffn_down.bias": (hidden,),
    }.items():
        shapes["v.blk.0." + stem] = shape
    for stem, shape in {
        "norm.weight": (merged,),
        "norm.bias": (merged,),
        "fc1.weight": (merged, merged),
        "fc1.bias": (merged,),
        "fc2.weight": (output, merged),
        "fc2.bias": (output,),
    }.items():
        shapes["v.deepstack.0." + stem] = shape
    return _FakeSidecar(metadata, shapes, seed=4)


def _glm4v_sidecar() -> _FakeSidecar:
    hidden, intermediate, output, patch = 8, 16, 12, 2
    projector_intermediate = output * 3
    metadata = {
        **_vision_metadata(
            projector_type="glm4v",
            hidden=hidden,
            intermediate=intermediate,
            output=output,
            patch=patch,
        ),
        "clip.use_silu": True,
    }
    shapes = {
        "v.patch_embd.weight": (hidden, 3, patch, patch),
        "v.patch_embd.weight.1": (hidden, 3, patch, patch),
        "v.patch_embd.bias": (hidden,),
        "v.post_ln.weight": (hidden,),
        "mm.patch_merger.weight": (output, hidden, 2, 2),
        "mm.patch_merger.bias": (output,),
        "mm.model.fc.weight": (output, output),
        "mm.post_norm.weight": (output,),
        "mm.post_norm.bias": (output,),
        "mm.up.weight": (projector_intermediate, output),
        "mm.gate.weight": (projector_intermediate, output),
        "mm.down.weight": (output, projector_intermediate),
    }
    for stem, shape in {
        "ln1.weight": (hidden,),
        "ln2.weight": (hidden,),
        "attn_qkv.weight": (3 * hidden, hidden),
        "attn_qkv.bias": (3 * hidden,),
        "attn_q_norm.weight": (hidden // 2,),
        "attn_k_norm.weight": (hidden // 2,),
        "attn_out.weight": (hidden, hidden),
        "attn_out.bias": (hidden,),
        "ffn_gate.weight": (intermediate, hidden),
        "ffn_gate.bias": (intermediate,),
        "ffn_up.weight": (intermediate, hidden),
        "ffn_up.bias": (intermediate,),
        "ffn_down.weight": (hidden, intermediate),
        "ffn_down.bias": (hidden,),
    }.items():
        shapes["v.blk.0." + stem] = shape
    return _FakeSidecar(metadata, shapes, seed=5)


def _glm4v_learned_position_sidecar() -> _FakeSidecar:
    hidden, intermediate, output, patch = 8, 16, 12, 2
    projector_intermediate = 16
    metadata = {
        **_vision_metadata(
            projector_type="glm4v",
            hidden=hidden,
            intermediate=intermediate,
            output=output,
            patch=patch,
        ),
        "clip.vision.spatial_merge_size": 2,
        "clip.use_silu": True,
    }
    shapes = {
        "v.patch_embd.weight": (hidden, 3, patch, patch),
        "v.patch_embd.weight.1": (hidden, 3, patch, patch),
        "v.patch_embd.bias": (hidden,),
        "v.norm_embd.weight": (hidden,),
        "v.position_embd.weight": (4, hidden),
        "v.post_ln.weight": (hidden,),
        "mm.patch_merger.weight": (output, hidden, 2, 2),
        "mm.patch_merger.bias": (output,),
        "mm.model.fc.weight": (output, output),
        "mm.post_norm.weight": (output,),
        "mm.post_norm.bias": (output,),
        "mm.up.weight": (projector_intermediate, output),
        "mm.gate.weight": (projector_intermediate, output),
        "mm.down.weight": (output, projector_intermediate),
    }
    for stem, shape in {
        "ln1.weight": (hidden,),
        "ln2.weight": (hidden,),
        "attn_qkv.weight": (3 * hidden, hidden),
        "attn_out.weight": (hidden, hidden),
        "ffn_gate.weight": (output, hidden),
        "ffn_up.weight": (output, hidden),
        "ffn_down.weight": (hidden, output),
    }.items():
        shapes["v.blk.0." + stem] = shape
    return _FakeSidecar(metadata, shapes, seed=15)


def _glma_sidecar() -> _FakeSidecar:
    hidden, intermediate, output, mel, stack = 8, 16, 12, 8, 2
    metadata = {
        **_audio_metadata(
            projector_type="glma",
            hidden=hidden,
            intermediate=intermediate,
            output=output,
            mel=mel,
            global_type=True,
        ),
        "clip.audio.projector.stack_factor": stack,
    }
    shapes = {
        "a.conv1d.1.weight": (hidden, mel, 3),
        "a.conv1d.1.bias": (hidden, 1),
        "a.conv1d.2.weight": (hidden, hidden, 3),
        "a.conv1d.2.bias": (hidden, 1),
        "a.position_embd.weight": (8, hidden),
        "a.post_ln.weight": (hidden,),
        "a.post_ln.bias": (hidden,),
        "mm.a.norm_pre.weight": (hidden,),
        "mm.a.norm_pre.bias": (hidden,),
        "mm.a.mlp.1.weight": (intermediate, hidden * stack),
        "mm.a.mlp.1.bias": (intermediate,),
        "mm.a.mlp.2.weight": (output, intermediate),
        "mm.a.mlp.2.bias": (output,),
        "v.boi": (output,),
        "v.eoi": (output,),
        **_whisper_block_shapes(
            hidden=hidden,
            intermediate=intermediate,
            key_bias=False,
        ),
    }
    return _FakeSidecar(metadata, shapes, seed=6)


def _speaker_sidecar() -> _FakeSidecar:
    channels, mel, mfa, attention, output = 8, 4, 24, 4, 12
    metadata = {
        **_audio_metadata(
            projector_type="qwen3tts_spkenc",
            hidden=mfa,
            intermediate=mfa,
            output=output,
            mel=mel,
            layers=3,
        ),
        "clip.has_gen_audio_encoder": True,
        "clip.gen.audio.projector_type": "qwen3tts_gen",
    }
    shapes = {
        "a.conv1d.0.weight": (channels, mel, 5),
        "a.conv1d.0.bias": (channels,),
        "a.conv_out.weight": (mfa, 3 * channels, 1),
        "a.conv_out.bias": (mfa,),
        "a.asp_tdnn.weight": (attention, 3 * mfa, 1),
        "a.asp_tdnn.bias": (attention,),
        "a.asp_attn.weight": (mfa, attention, 1),
        "a.asp_attn.bias": (mfa,),
        "mm.a.fc.weight": (output, 2 * mfa, 1),
        "mm.a.fc.bias": (output,),
        "a.gen.code.proj_in.weight": (1,),
    }
    for block in range(1, 4):
        shapes.update(
            {
                f"a.blk.{block}.conv_pw1.weight": (channels, channels, 1),
                f"a.blk.{block}.conv_pw1.bias": (channels,),
                f"a.blk.{block}.conv_pw2.weight": (channels, channels, 1),
                f"a.blk.{block}.conv_pw2.bias": (channels,),
                f"a.blk.{block}.se_conv1.weight": (4, channels, 1),
                f"a.blk.{block}.se_conv1.bias": (4,),
                f"a.blk.{block}.se_conv2.weight": (channels, 4, 1),
                f"a.blk.{block}.se_conv2.bias": (channels,),
            }
        )
        for branch in range(7):
            shapes[f"a.blk.{block}.res2.{branch}.weight"] = (1, 1, 3)
            shapes[f"a.blk.{block}.res2.{branch}.bias"] = (1,)
    return _FakeSidecar(metadata, shapes, seed=7)


def _qwen25o_sidecar() -> _FakeSidecar:
    sidecar = _qwen2a_sidecar(alias=True)
    hidden, intermediate, output, patch = 8, 12, 12, 2
    sidecar.metadata.update(
        _vision_metadata(
            projector_type="qwen2.5o",
            hidden=hidden,
            intermediate=hidden,
            output=output,
            patch=patch,
        )
    )
    sidecar.metadata["clip.projector_type"] = "qwen2.5o"
    sidecar.metadata["clip.vision.n_wa_pattern"] = 1
    merged = hidden * 4
    vision_shapes = {
        "v.patch_embd.weight": (hidden, 3, patch, patch),
        "v.patch_embd.weight.1": (hidden, 3, patch, patch),
        "v.post_ln.weight": (hidden,),
        "mm.0.weight": (merged, merged),
        "mm.0.bias": (merged,),
        "mm.2.weight": (output, merged),
        "mm.2.bias": (output,),
    }
    for stem, shape in {
        "ln1.weight": (hidden,),
        "ln2.weight": (hidden,),
        "attn_q.weight": (hidden, hidden),
        "attn_q.bias": (hidden,),
        "attn_k.weight": (hidden, hidden),
        "attn_k.bias": (hidden,),
        "attn_v.weight": (hidden, hidden),
        "attn_v.bias": (hidden,),
        "attn_out.weight": (hidden, hidden),
        "attn_out.bias": (hidden,),
        "ffn_gate.weight": (intermediate, hidden),
        "ffn_gate.bias": (intermediate,),
        "ffn_up.weight": (intermediate, hidden),
        "ffn_up.bias": (intermediate,),
        "ffn_down.weight": (hidden, intermediate),
        "ffn_down.bias": (hidden,),
    }.items():
        vision_shapes["v.blk.0." + stem] = shape
    rng = np.random.default_rng(8)
    sidecar._shapes.update(vision_shapes)
    sidecar.tensor_names.extend(vision_shapes)
    sidecar._values.update(
        {
            name: (rng.standard_normal(shape) * 0.03).astype(np.float32)
            for name, shape in vision_shapes.items()
        }
    )
    return sidecar


_ROUTES = {
    "glm4v": (_glm4v_sidecar, "glm4", {"vision_encoder"}),
    "glma": (_glma_sidecar, "llama", {"audio_encoder"}),
    "qwen2.5o": (_qwen25o_sidecar, "qwen2vl", {"vision_encoder", "audio_encoder"}),
    "qwen2a": (_qwen2a_sidecar, "qwen2", {"audio_encoder"}),
    "qwen3a": (_qwen3a_sidecar, "qwen3vl", {"audio_encoder"}),
    "qwen3vl_merger": (_qwen3vl_sidecar, "qwen3vl", {"vision_encoder"}),
    "qwen3tts_spkenc": (_speaker_sidecar, "qwen3tts", {"speaker_encoder"}),
}


def _build_package_from_sidecar(
    projector_type: str,
    sidecar: _FakeSidecar,
    *,
    dtype: str | None = None,
):
    _, target, _ = _ROUTES[projector_type]
    with (
        mock.patch(
            "mobius.integrations.gguf._mmproj._resolve_mmproj_companion_path",
            return_value="synthetic.gguf",
        ),
        mock.patch("mobius.integrations.gguf._builder._validate_gguf_model"),
    ):
        package = build_mmproj_from_gguf(
            "synthetic.gguf",
            projector_type=projector_type,
            target_architecture=target,
            dtype=dtype,
            _mmproj_gguf_model=sidecar,
        )
    return sidecar, package


def _build_package(projector_type: str, *, dtype: str | None = None):
    factory, _, _ = _ROUTES[projector_type]
    return _build_package_from_sidecar(projector_type, factory(), dtype=dtype)


@pytest.mark.parametrize("projector_type", tuple(_ROUTES))
def test_every_route_builds_only_its_declared_components(projector_type: str) -> None:
    _, _, expected = _ROUTES[projector_type]
    _, package = _build_package(projector_type)

    assert set(package) == expected
    assert package.gguf_projector_type == projector_type
    assert package.gguf_processor_abi
    assert all(
        initializer.const_value is not None
        for graph in package.values()
        for initializer in graph.graph.initializers.values()
    )


def test_public_dispatch_preflights_qwen_route_once() -> None:
    from mobius.integrations.gguf import _mmproj

    sidecar = _qwen2a_sidecar()
    with (
        mock.patch(
            "mobius.integrations.gguf._mmproj._resolve_mmproj_companion_path",
            return_value="synthetic.gguf",
        ),
        mock.patch("mobius.integrations.gguf._builder._validate_gguf_model"),
        mock.patch(
            "mobius.integrations.gguf._mmproj._preflight_standalone_mmproj",
            wraps=_mmproj._preflight_standalone_mmproj,
        ) as preflight,
    ):
        build_mmproj_from_gguf(
            "synthetic.gguf",
            projector_type="qwen2a",
            target_architecture="qwen2",
            _mmproj_gguf_model=sidecar,
        )

    preflight.assert_called_once_with(
        sidecar,
        projector_type="qwen2a",
        target_architecture="qwen2",
    )


@pytest.mark.parametrize("projector_type", ("qwen2a", "qwen3a", "qwen2.5o", "qwen3tts_spkenc"))
def test_unrelated_silu_metadata_does_not_change_nonconsumer_graph(
    projector_type: str,
) -> None:
    factory, _, _ = _ROUTES[projector_type]
    baseline_sidecar = factory()
    unrelated_sidecar = factory()
    unrelated_sidecar.metadata["clip.use_silu"] = {"invalid": "but unrelated"}
    _, baseline = _build_package_from_sidecar(projector_type, baseline_sidecar)
    _, unrelated = _build_package_from_sidecar(projector_type, unrelated_sidecar)

    def signature(package) -> dict[str, tuple[object, ...]]:
        return {
            role: (
                tuple((node.domain, node.op_type) for node in model.graph),
                tuple(
                    (value.name, str(value.type), str(value.shape))
                    for value in model.graph.inputs
                ),
                tuple(
                    (value.name, str(value.type), str(value.shape))
                    for value in model.graph.outputs
                ),
                tuple(model.graph.initializers),
            )
            for role, model in package.items()
        }

    assert baseline.config == unrelated.config
    assert signature(baseline) == signature(unrelated)


@pytest.mark.parametrize("projector_type", ("glm4v", "glma", "qwen3vl_merger"))
@pytest.mark.parametrize("invalid", [0, 1, "true"])
def test_silu_consumers_require_boolean_metadata(
    projector_type: str,
    invalid: object,
) -> None:
    factory, _, _ = _ROUTES[projector_type]
    sidecar = factory()
    sidecar.metadata["clip.use_silu"] = invalid

    with pytest.raises(ValueError, match="must be a boolean"):
        validate_qwen_glm_projector_metadata(sidecar, projector_type)


def test_qwen3vl_silu_metadata_changes_only_transformer_block_activation() -> None:
    gelu_sidecar = _qwen3vl_sidecar()
    silu_sidecar = _qwen3vl_sidecar()
    silu_sidecar.metadata["clip.use_silu"] = True
    _, gelu_package = _build_package_from_sidecar("qwen3vl_merger", gelu_sidecar)
    _, silu_package = _build_package_from_sidecar("qwen3vl_merger", silu_sidecar)

    gelu_ops = [node.op_type for node in gelu_package["vision_encoder"].graph]
    silu_ops = [node.op_type for node in silu_package["vision_encoder"].graph]
    assert "Swish" not in gelu_ops
    assert silu_ops.count("Swish") == 1
    # Final and DeepStack mergers remain tanh-GELU in both variants.
    assert silu_ops.count("Gelu") == gelu_ops.count("Gelu") - 1


@pytest.mark.parametrize("projector_type", tuple(_ROUTES))
@pytest.mark.parametrize(
    "invalid",
    [0, -1, float("nan"), float("inf"), "1e-5", True],
    ids=["zero", "negative", "nan", "inf", "string", "bool"],
)
def test_route_metadata_rejects_invalid_epsilon_before_tensors(
    projector_type: str,
    invalid: object,
) -> None:
    factory, _, _ = _ROUTES[projector_type]
    sidecar = factory()
    modality = "vision" if projector_type in {"glm4v", "qwen3vl_merger"} else "audio"
    if projector_type == "qwen2.5o":
        modality = "vision"
    sidecar.metadata[f"clip.{modality}.attention.layer_norm_epsilon"] = invalid

    with pytest.raises(ValueError, match="positive finite number"):
        validate_qwen_glm_projector_metadata(sidecar, projector_type)


@pytest.mark.parametrize("projector_type", tuple(_ROUTES))
@pytest.mark.parametrize(
    "invalid",
    [0, -1, float("nan"), float("inf"), "8", True, 1.5],
    ids=["zero", "negative", "nan", "inf", "string", "bool", "float"],
)
def test_route_metadata_rejects_invalid_dimensions(
    projector_type: str,
    invalid: object,
) -> None:
    factory, _, _ = _ROUTES[projector_type]
    sidecar = factory()
    modality = "vision" if projector_type in {"glm4v", "qwen3vl_merger"} else "audio"
    if projector_type == "qwen2.5o":
        modality = "vision"
    sidecar.metadata[f"clip.{modality}.embedding_length"] = invalid

    with pytest.raises(ValueError, match="positive integer"):
        validate_qwen_glm_projector_metadata(sidecar, projector_type)


@pytest.mark.parametrize("projector_type", tuple(_ROUTES))
@pytest.mark.parametrize("invalid", [0, 1, "true"], ids=["zero", "one", "string"])
def test_route_metadata_requires_boolean_presence(
    projector_type: str,
    invalid: object,
) -> None:
    factory, _, _ = _ROUTES[projector_type]
    sidecar = factory()
    modality = "vision" if projector_type in {"glm4v", "qwen3vl_merger"} else "audio"
    if projector_type == "qwen2.5o":
        modality = "vision"
    sidecar.metadata[f"clip.has_{modality}_encoder"] = invalid

    with pytest.raises(ValueError, match="must be boolean True"):
        validate_qwen_glm_projector_metadata(sidecar, projector_type)


@pytest.mark.parametrize("projector_type", tuple(_ROUTES))
def test_route_metadata_requires_exact_projector_enum(projector_type: str) -> None:
    factory, _, _ = _ROUTES[projector_type]
    sidecar = factory()
    type_key = (
        "clip.audio.projector_type"
        if projector_type in {"qwen3a", "qwen3tts_spkenc"}
        else "clip.projector_type"
    )
    sidecar.metadata[type_key] = 7

    with pytest.raises(ValueError, match="must equal"):
        validate_qwen_glm_projector_metadata(sidecar, projector_type)


@pytest.mark.parametrize(
    ("projector_type", "updates"),
    [
        (
            "glm4v",
            {
                "clip.vision.attention.head_count": 8,
                "clip.vision.image_size": 2,
                "clip.vision.spatial_merge_size": 2,
            },
        ),
        (
            "glma",
            {"clip.audio.attention.head_count": 8, "clip.audio.projector.stack_factor": 1},
        ),
        (
            "qwen2.5o",
            {
                "clip.vision.attention.head_count": 8,
                "clip.vision.image_size": 2,
                "clip.vision.n_wa_pattern": 1,
                "clip.audio.attention.head_count": 8,
            },
        ),
        ("qwen2a", {"clip.audio.attention.head_count": 8}),
        (
            "qwen3a",
            {
                "clip.audio.attention.head_count": 8,
                "clip.audio.projector.window_size": 100,
            },
        ),
        (
            "qwen3vl_merger",
            {
                "clip.vision.attention.head_count": 8,
                "clip.vision.image_size": 2,
                "clip.vision.spatial_merge_size": 2,
            },
        ),
        ("qwen3tts_spkenc", {"clip.audio.attention.head_count": 24}),
    ],
)
def test_route_metadata_accepts_exact_valid_boundaries(
    projector_type: str,
    updates: dict[str, object],
) -> None:
    factory, _, _ = _ROUTES[projector_type]
    sidecar = factory()
    sidecar.metadata.update(updates)

    validate_qwen_glm_projector_metadata(sidecar, projector_type)


@pytest.mark.parametrize(
    ("projector_type", "key", "invalid", "message"),
    [
        ("glm4v", "clip.vision.spatial_merge_size", 1, "spatial_merge_size=2"),
        ("glma", "clip.audio.projector.stack_factor", 0, "positive integer"),
        ("qwen2.5o", "clip.vision.n_wa_pattern", 2, "cannot exceed"),
        ("qwen2a", "clip.audio.attention.head_count", 3, "divide by"),
        ("qwen3a", "clip.audio.projector.window_size", 150, "multiple of 100"),
        (
            "qwen3vl_merger",
            "clip.vision.is_deepstack_layers",
            [1],
            "one boolean",
        ),
        ("qwen3tts_spkenc", "clip.audio.block_count", 2, "exactly three"),
    ],
)
def test_route_metadata_rejects_route_specific_geometry(
    projector_type: str,
    key: str,
    invalid: object,
    message: str,
) -> None:
    factory, _, _ = _ROUTES[projector_type]
    sidecar = factory()
    sidecar.metadata[key] = invalid

    with pytest.raises(ValueError, match=message):
        validate_qwen_glm_projector_metadata(sidecar, projector_type)


def test_standalone_metadata_validation_precedes_tensor_closure() -> None:
    sidecar = _qwen2a_sidecar()
    sidecar.metadata["clip.audio.attention.layer_norm_epsilon"] = 0.0
    with (
        mock.patch(
            "mobius.integrations.gguf._mmproj._resolve_mmproj_companion_path",
            return_value="synthetic.gguf",
        ),
        mock.patch("mobius.integrations.gguf._builder._validate_gguf_model"),
        mock.patch(
            "mobius.integrations.gguf._mmproj._validate_mmproj_tensor_closure"
        ) as closure,
        pytest.raises(ValueError, match="positive finite number"),
    ):
        build_mmproj_from_gguf(
            "synthetic.gguf",
            projector_type="qwen2a",
            target_architecture="qwen2",
            _mmproj_gguf_model=sidecar,
        )
    closure.assert_not_called()


def _run_component(package, component: str, feeds: dict[str, np.ndarray]) -> np.ndarray:
    session = OnnxModelSession(package[component])
    try:
        return next(iter(session.run(feeds).values()))
    finally:
        session.close()


def _load_qwen2_hf_state(sidecar: _FakeSidecar, encoder, projector) -> None:
    state = {}
    direct = {
        "a.conv1d.1.weight": "conv1.weight",
        "a.conv1d.1.bias": "conv1.bias",
        "a.conv1d.2.weight": "conv2.weight",
        "a.conv1d.2.bias": "conv2.bias",
        "a.position_embd.weight": "embed_positions.weight",
        "a.post_ln.weight": "layer_norm.weight",
        "a.post_ln.bias": "layer_norm.bias",
    }
    block = {
        "ln1": "self_attn_layer_norm",
        "ln2": "final_layer_norm",
        "attn_q": "self_attn.q_proj",
        "attn_k": "self_attn.k_proj",
        "attn_v": "self_attn.v_proj",
        "attn_out": "self_attn.out_proj",
        "ffn_up": "fc1",
        "ffn_down": "fc2",
    }
    for source, target in direct.items():
        value = sidecar.get_tensor(source)
        state[target] = torch.from_numpy(
            value.reshape(-1) if source.endswith(".bias") else value
        )
    for source, value in sidecar._values.items():
        match = __import__("re").match(
            r"^a\.blk\.0\.([^.]+)\.(weight|bias)$",
            source,
        )
        if match is None:
            continue
        stem, kind = match.groups()
        state[f"layers.0.{block[stem]}.{kind}"] = torch.from_numpy(value)
    encoder.load_state_dict(state, strict=True)
    projector.linear.weight.data.copy_(torch.from_numpy(sidecar.get_tensor("mm.a.fc.weight")))
    projector.linear.bias.data.copy_(torch.from_numpy(sidecar.get_tensor("mm.a.fc.bias")))


def test_qwen2a_projector_matches_transformers() -> None:
    from transformers.models.qwen2_audio.configuration_qwen2_audio import (
        Qwen2AudioEncoderConfig,
    )
    from transformers.models.qwen2_audio.modeling_qwen2_audio import (
        Qwen2AudioEncoder,
        Qwen2AudioMultiModalProjector,
    )

    sidecar, package = _build_package("qwen2a")
    config = Qwen2AudioEncoderConfig(
        num_mel_bins=8,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=16,
        d_model=8,
        max_source_positions=8,
    )
    encoder = Qwen2AudioEncoder(config).eval()
    projector = Qwen2AudioMultiModalProjector(
        SimpleNamespace(
            audio_config=SimpleNamespace(d_model=8),
            text_config=SimpleNamespace(hidden_size=12),
        )
    ).eval()
    _load_qwen2_hf_state(sidecar, encoder, projector)

    features = np.linspace(-1.0, 1.0, 128, dtype=np.float32).reshape(1, 8, 16)
    with torch.no_grad():
        expected = projector(encoder(torch.from_numpy(features)).last_hidden_state).numpy()
    actual = _run_component(
        package,
        "audio_encoder",
        {"input_features": features},
    )
    np.testing.assert_allclose(actual, expected[0], rtol=2e-4, atol=2e-4)


def _load_qwen3_audio_hf_state(sidecar: _FakeSidecar, encoder) -> None:
    state = {}
    direct = {
        "a.conv2d.1.weight": "conv2d1.weight",
        "a.conv2d.1.bias": "conv2d1.bias",
        "a.conv2d.2.weight": "conv2d2.weight",
        "a.conv2d.2.bias": "conv2d2.bias",
        "a.conv2d.3.weight": "conv2d3.weight",
        "a.conv2d.3.bias": "conv2d3.bias",
        "a.conv_out.weight": "conv_out.weight",
        "a.post_ln.weight": "ln_post.weight",
        "a.post_ln.bias": "ln_post.bias",
        "mm.a.mlp.1.weight": "proj1.weight",
        "mm.a.mlp.1.bias": "proj1.bias",
        "mm.a.mlp.2.weight": "proj2.weight",
        "mm.a.mlp.2.bias": "proj2.bias",
    }
    block = {
        "ln1": "self_attn_layer_norm",
        "ln2": "final_layer_norm",
        "attn_q": "self_attn.q_proj",
        "attn_k": "self_attn.k_proj",
        "attn_v": "self_attn.v_proj",
        "attn_out": "self_attn.out_proj",
        "ffn_up": "fc1",
        "ffn_down": "fc2",
    }
    for source, target in direct.items():
        value = sidecar.get_tensor(source)
        state[target] = torch.from_numpy(
            value.reshape(-1) if source.endswith(".bias") else value
        )
    for source, value in sidecar._values.items():
        match = __import__("re").match(
            r"^a\.blk\.0\.([^.]+)\.(weight|bias)$",
            source,
        )
        if match is None:
            continue
        stem, kind = match.groups()
        state[f"layers.0.{block[stem]}.{kind}"] = torch.from_numpy(value)
    encoder.load_state_dict(state, strict=True)


def test_qwen3a_projector_matches_transformers() -> None:
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoderConfig,
    )
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoder,
    )

    sidecar = _qwen3a_sidecar()
    config = Qwen3OmniMoeAudioEncoderConfig(
        num_mel_bins=8,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=16,
        d_model=8,
        max_source_positions=13,
        n_window=50,
        n_window_infer=800,
        conv_chunksize=1,
        output_dim=12,
        downsample_hidden_size=4,
    )
    encoder = Qwen3OmniMoeAudioEncoder(config).eval()
    positional = encoder.positional_embedding.positional_embedding.detach().numpy()
    sidecar._values["a.position_embd.weight"] = positional.astype(np.float32)
    _load_qwen3_audio_hf_state(sidecar, encoder)
    # Rebuild with the exact generated position table used by the HF reference.
    with (
        mock.patch(
            "mobius.integrations.gguf._mmproj._resolve_mmproj_companion_path",
            return_value="synthetic.gguf",
        ),
        mock.patch("mobius.integrations.gguf._builder._validate_gguf_model"),
    ):
        package = build_mmproj_from_gguf(
            "synthetic.gguf",
            projector_type="qwen3a",
            target_architecture="qwen3vl",
            _mmproj_gguf_model=sidecar,
        )

    features = np.linspace(-0.4, 0.6, 800, dtype=np.float32).reshape(1, 8, 100)
    with torch.no_grad():
        expected = encoder(
            torch.from_numpy(features[0]),
            feature_lens=torch.tensor([100]),
        ).last_hidden_state.numpy()
    actual = _run_component(
        package,
        "audio_encoder",
        {
            "input_features": features,
            "input_features_mask": np.ones((1, 100), dtype=np.int32),
        },
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-4)

    padded = np.concatenate(
        [features, np.linspace(1.0, 2.0, 800, dtype=np.float32).reshape(1, 8, 100)],
        axis=2,
    )
    padded_mask = np.concatenate(
        [np.ones((1, 100), dtype=np.int64), np.zeros((1, 100), dtype=np.int64)],
        axis=1,
    )
    session = OnnxModelSession(package["audio_encoder"])
    try:
        padded_outputs = session.run(
            {
                "input_features": padded,
                "input_features_mask": padded_mask.astype(np.int32),
            }
        )
    finally:
        session.close()
    np.testing.assert_allclose(
        padded_outputs["audio_features"],
        expected,
        rtol=3e-4,
        atol=3e-4,
    )
    np.testing.assert_allclose(
        padded_outputs["audio_features"],
        actual,
        rtol=3e-4,
        atol=3e-4,
    )
    np.testing.assert_array_equal(
        padded_outputs["audio_feature_lengths"],
        np.array([13], dtype=np.int64),
    )


def test_qwen3a_graph_io_and_processor_metadata_publish_lengths() -> None:
    _, package = _build_package("qwen3a")
    graph = package["audio_encoder"].graph

    assert [value.name for value in graph.inputs] == [
        "input_features",
        "input_features_mask",
    ]
    assert [value.name for value in graph.outputs] == [
        "audio_features",
        "audio_feature_lengths",
    ]
    assert package.gguf_processor_abi["input_features_mask"] == (
        "int32[1,frames], binary and right-padded"
    )
    assert "audio_feature_lengths" in package.gguf_processor_abi["output"]


def test_qwen3a_accepts_processor_outputs_without_adapter() -> None:
    from transformers import WhisperFeatureExtractor
    from transformers.models.qwen3_asr.processing_qwen3_asr import Qwen3ASRProcessor

    _, package = _build_package("qwen3a")
    # Construct the pinned processor class with a tiny, network-free feature
    # extractor. Text is omitted, so only these attributes are consulted.
    processor = object.__new__(Qwen3ASRProcessor)
    processor.feature_extractor = WhisperFeatureExtractor(
        feature_size=8,
        sampling_rate=16_000,
        hop_length=160,
        chunk_length=1,
        n_fft=400,
        padding_value=0.0,
        return_attention_mask=True,
    )
    processor.tokenizer = SimpleNamespace(init_kwargs={})
    processor.chat_template = None
    processor.timestamp_segment_time = 80
    processor.audio_token = "<audio>"
    processed = processor(
        text=None,
        audio=[np.zeros(16_000, dtype=np.float32)],
        sampling_rate=16_000,
        return_tensors="pt",
    )
    input_names = {value.name for value in package["audio_encoder"].graph.inputs}
    feeds = {name: processed[name].numpy() for name in input_names}

    assert set(feeds) == {"input_features", "input_features_mask"}
    assert feeds["input_features_mask"].dtype == np.int32
    session = OnnxModelSession(package["audio_encoder"])
    try:
        outputs = session.run(feeds)
    finally:
        session.close()
    assert outputs["audio_features"].shape == (13, 12)
    np.testing.assert_array_equal(
        outputs["audio_feature_lengths"],
        np.array([13], dtype=np.int64),
    )


@pytest.mark.parametrize(
    "input_features_mask",
    [
        np.zeros((1, 100), dtype=np.int64),
        np.array([[1] * 49 + [0] + [1] * 50], dtype=np.int64),
        np.array([[1] * 99 + [2]], dtype=np.int64),
    ],
    ids=["empty", "not-right-padded", "non-binary"],
)
def test_qwen3a_processor_mask_fails_closed(
    input_features_mask: np.ndarray,
) -> None:
    _, package = _build_package("qwen3a")
    features = np.zeros((1, 8, 100), dtype=np.float32)

    with pytest.raises(
        (
            ort.capi.onnxruntime_pybind11_state.Fail,
            ort.capi.onnxruntime_pybind11_state.InvalidArgument,
            ort.capi.onnxruntime_pybind11_state.RuntimeException,
        ),
        match="indices element out of data bounds",
    ):
        _run_component(
            package,
            "audio_encoder",
            {
                "input_features": features,
                "input_features_mask": input_features_mask.astype(np.int32),
            },
        )


def _qwen3vl_hf_state(sidecar: _FakeSidecar) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {
        "patch_embed.proj.weight": torch.from_numpy(
            np.stack(
                [
                    sidecar.get_tensor("v.patch_embd.weight"),
                    sidecar.get_tensor("v.patch_embd.weight.1"),
                ],
                axis=2,
            )
        ),
        "patch_embed.proj.bias": torch.from_numpy(sidecar.get_tensor("v.patch_embd.bias")),
        "pos_embed.weight": torch.from_numpy(sidecar.get_tensor("v.position_embd.weight")),
        "merger.norm.weight": torch.from_numpy(sidecar.get_tensor("v.post_ln.weight")),
        "merger.norm.bias": torch.from_numpy(sidecar.get_tensor("v.post_ln.bias")),
        "merger.linear_fc1.weight": torch.from_numpy(sidecar.get_tensor("mm.0.weight")),
        "merger.linear_fc1.bias": torch.from_numpy(sidecar.get_tensor("mm.0.bias")),
        "merger.linear_fc2.weight": torch.from_numpy(sidecar.get_tensor("mm.2.weight")),
        "merger.linear_fc2.bias": torch.from_numpy(sidecar.get_tensor("mm.2.bias")),
    }
    block_map = {
        "ln1": "norm1",
        "ln2": "norm2",
        "attn_qkv": "attn.qkv",
        "attn_out": "attn.proj",
        "ffn_up": "mlp.linear_fc1",
        "ffn_down": "mlp.linear_fc2",
    }
    for source, value in sidecar._values.items():
        match = __import__("re").match(
            r"^v\.blk\.0\.([^.]+)\.(weight|bias)$",
            source,
        )
        if match is not None:
            stem, kind = match.groups()
            state[f"blocks.0.{block_map[stem]}.{kind}"] = torch.from_numpy(value)
            continue
        match = __import__("re").match(
            r"^v\.deepstack\.0\.(norm|fc1|fc2)\.(weight|bias)$",
            source,
        )
        if match is not None:
            stem, kind = match.groups()
            target = {
                "norm": "norm",
                "fc1": "linear_fc1",
                "fc2": "linear_fc2",
            }[stem]
            state[f"deepstack_merger_list.0.{target}.{kind}"] = torch.from_numpy(value)
    return state


def test_qwen3vl_projector_matches_transformers() -> None:
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLVisionConfig,
    )
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    sidecar, package = _build_package("qwen3vl_merger")
    config = Qwen3VLVisionConfig(
        depth=1,
        hidden_size=8,
        intermediate_size=16,
        num_heads=2,
        in_channels=3,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=12,
        num_position_embeddings=4,
        deepstack_visual_indexes=[0],
    )
    reference = Qwen3VLVisionModel(config).eval()
    reference.load_state_dict(_qwen3vl_hf_state(sidecar), strict=True)
    # llama.cpp hardcodes its tanh-approximate GELU for both merger families.
    reference.merger.act_fn = torch.nn.GELU(approximate="tanh")
    reference.deepstack_merger_list[0].act_fn = torch.nn.GELU(approximate="tanh")

    pixels = np.linspace(-0.5, 0.7, 96, dtype=np.float32).reshape(4, 24)
    grid = np.array([[1, 2, 2]], dtype=np.int64)
    with torch.no_grad():
        output = reference(
            torch.from_numpy(pixels),
            torch.from_numpy(grid),
        )
        expected = torch.cat(
            [output.pooler_output, *output.deepstack_features],
            dim=1,
        ).numpy()
    actual = _run_component(
        package,
        "vision_encoder",
        {"pixel_values": pixels, "image_grid_thw": grid},
    )
    np.testing.assert_allclose(actual, expected, rtol=4e-4, atol=4e-4)


def _glm4v_hf_state(sidecar: _FakeSidecar) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {
        "patch_embed.proj.weight": torch.from_numpy(
            np.stack(
                [
                    sidecar.get_tensor("v.patch_embd.weight"),
                    sidecar.get_tensor("v.patch_embd.weight.1"),
                ],
                axis=2,
            )
        ),
        "patch_embed.proj.bias": torch.from_numpy(sidecar.get_tensor("v.patch_embd.bias")),
        "post_layernorm.weight": torch.from_numpy(sidecar.get_tensor("v.post_ln.weight")),
        "downsample.weight": torch.from_numpy(sidecar.get_tensor("mm.patch_merger.weight")),
        "downsample.bias": torch.from_numpy(sidecar.get_tensor("mm.patch_merger.bias")),
        "merger.proj.weight": torch.from_numpy(sidecar.get_tensor("mm.model.fc.weight")),
        "merger.post_projection_norm.weight": torch.from_numpy(
            sidecar.get_tensor("mm.post_norm.weight")
        ),
        "merger.post_projection_norm.bias": torch.from_numpy(
            sidecar.get_tensor("mm.post_norm.bias")
        ),
        "merger.up_proj.weight": torch.from_numpy(sidecar.get_tensor("mm.up.weight")),
        "merger.gate_proj.weight": torch.from_numpy(sidecar.get_tensor("mm.gate.weight")),
        "merger.down_proj.weight": torch.from_numpy(sidecar.get_tensor("mm.down.weight")),
    }
    block_map = {
        "ln1": "norm1",
        "ln2": "norm2",
        "attn_qkv": "attn.qkv",
        "attn_q_norm": "attn.q_norm",
        "attn_k_norm": "attn.k_norm",
        "attn_out": "attn.proj",
        "ffn_gate": "mlp.gate_proj",
        "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj",
    }
    for source, value in sidecar._values.items():
        match = __import__("re").match(
            r"^v\.blk\.0\.([^.]+)\.(weight|bias)$",
            source,
        )
        if match is None:
            continue
        stem, kind = match.groups()
        state[f"blocks.0.{block_map[stem]}.{kind}"] = torch.from_numpy(value)
    return state


def test_glm4v_projector_matches_transformers() -> None:
    from transformers.models.glm_ocr.configuration_glm_ocr import (
        GlmOcrVisionConfig,
    )
    from transformers.models.glm_ocr.modeling_glm_ocr import GlmOcrVisionModel

    sidecar, package = _build_package("glm4v")
    config = GlmOcrVisionConfig(
        depth=1,
        hidden_size=8,
        intermediate_size=16,
        num_heads=2,
        in_channels=3,
        image_size=4,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=12,
    )
    reference = GlmOcrVisionModel(config).eval()
    reference.load_state_dict(_glm4v_hf_state(sidecar), strict=True)
    pixels = np.linspace(-0.8, 0.9, 96, dtype=np.float32).reshape(4, 24)
    grid = np.array([[1, 2, 2]], dtype=np.int64)
    with torch.no_grad():
        expected = reference(
            torch.from_numpy(pixels),
            torch.from_numpy(grid),
        ).pooler_output.numpy()
    actual = _run_component(
        package,
        "vision_encoder",
        {"pixel_values": pixels, "image_grid_thw": grid},
    )
    np.testing.assert_allclose(actual, expected, rtol=4e-4, atol=4e-4)


def test_glm4v_learned_position_variant_matches_transformers() -> None:
    from transformers.models.glm4v.configuration_glm4v import Glm4vVisionConfig
    from transformers.models.glm4v.modeling_glm4v import Glm4vVisionModel

    sidecar = _glm4v_learned_position_sidecar()
    with (
        mock.patch(
            "mobius.integrations.gguf._mmproj._resolve_mmproj_companion_path",
            return_value="synthetic.gguf",
        ),
        mock.patch("mobius.integrations.gguf._builder._validate_gguf_model"),
    ):
        package = build_mmproj_from_gguf(
            "synthetic.gguf",
            projector_type="glm4v",
            target_architecture="glm4",
            _mmproj_gguf_model=sidecar,
        )

    config = Glm4vVisionConfig(
        depth=1,
        hidden_size=8,
        intermediate_size=16,
        num_heads=2,
        in_channels=3,
        image_size=4,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=12,
    )
    reference = Glm4vVisionModel(config).eval()
    state = _glm4v_hf_state(sidecar)
    state["post_conv_layernorm.weight"] = torch.from_numpy(
        sidecar.get_tensor("v.norm_embd.weight")
    )
    state["embeddings.position_embedding.weight"] = torch.from_numpy(
        sidecar.get_tensor("v.position_embd.weight")
    )
    reference.load_state_dict(state, strict=True)
    pixels = np.linspace(-0.6, 0.8, 96, dtype=np.float32).reshape(4, 24)
    grid = np.array([[1, 2, 2]], dtype=np.int64)
    with torch.no_grad():
        expected = reference(
            torch.from_numpy(pixels),
            torch.from_numpy(grid),
        ).pooler_output.numpy()
    actual = _run_component(
        package,
        "vision_encoder",
        {"pixel_values": pixels, "image_grid_thw": grid},
    )
    np.testing.assert_allclose(actual, expected, rtol=5e-4, atol=5e-4)


def _qwen25o_hf_vision_state(sidecar: _FakeSidecar) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {
        "patch_embed.proj.weight": torch.from_numpy(
            np.stack(
                [
                    sidecar.get_tensor("v.patch_embd.weight"),
                    sidecar.get_tensor("v.patch_embd.weight.1"),
                ],
                axis=2,
            )
        ),
        "merger.ln_q.weight": torch.from_numpy(sidecar.get_tensor("v.post_ln.weight")),
        "merger.mlp.0.weight": torch.from_numpy(sidecar.get_tensor("mm.0.weight")),
        "merger.mlp.0.bias": torch.from_numpy(sidecar.get_tensor("mm.0.bias")),
        "merger.mlp.2.weight": torch.from_numpy(sidecar.get_tensor("mm.2.weight")),
        "merger.mlp.2.bias": torch.from_numpy(sidecar.get_tensor("mm.2.bias")),
    }
    block_map = {
        "ln1": "norm1",
        "ln2": "norm2",
        "attn_q": "attn.q",
        "attn_k": "attn.k",
        "attn_v": "attn.v",
        "attn_out": "attn.proj",
        "ffn_gate": "mlp.gate_proj",
        "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj",
    }
    for source, value in sidecar._values.items():
        match = __import__("re").match(
            r"^v\.blk\.0\.([^.]+)\.(weight|bias)$",
            source,
        )
        if match is None:
            continue
        stem, kind = match.groups()
        state[f"blocks.0.{block_map[stem]}.{kind}"] = torch.from_numpy(value)
    return state


def test_qwen25o_alias_vision_matches_transformers() -> None:
    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
        Qwen2_5OmniVisionEncoderConfig,
    )
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniVisionEncoder,
    )

    sidecar, package = _build_package("qwen2.5o")
    config = Qwen2_5OmniVisionEncoderConfig(
        depth=1,
        hidden_size=8,
        intermediate_size=12,
        num_heads=2,
        in_channels=3,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=2,
        window_size=8,
        out_hidden_size=12,
        fullatt_block_indexes=[0],
    )
    reference = Qwen2_5OmniVisionEncoder(config).eval()
    reference.load_state_dict(_qwen25o_hf_vision_state(sidecar), strict=True)
    pixels = np.linspace(-0.3, 0.6, 96, dtype=np.float32).reshape(4, 24)
    grid = np.array([[1, 2, 2]], dtype=np.int64)
    with torch.no_grad():
        expected = reference(
            torch.from_numpy(pixels),
            torch.from_numpy(grid),
        ).pooler_output.numpy()
    actual = _run_component(
        package,
        "vision_encoder",
        {"pixel_values": pixels, "image_grid_thw": grid},
    )
    np.testing.assert_allclose(actual, expected, rtol=4e-4, atol=4e-4)


def _linear(
    value: torch.Tensor,
    sidecar: _FakeSidecar,
    stem: str,
) -> torch.Tensor:
    weight = torch.from_numpy(sidecar.get_tensor(stem + ".weight"))
    bias = (
        torch.from_numpy(sidecar.get_tensor(stem + ".bias").reshape(-1))
        if stem + ".bias" in sidecar._values
        else None
    )
    return functional.linear(value, weight, bias)


def _layer_norm(
    value: torch.Tensor,
    sidecar: _FakeSidecar,
    stem: str,
    *,
    eps: float = 1e-5,
) -> torch.Tensor:
    return functional.layer_norm(
        value,
        (value.shape[-1],),
        torch.from_numpy(sidecar.get_tensor(stem + ".weight")),
        torch.from_numpy(sidecar.get_tensor(stem + ".bias")),
        eps,
    )


def test_glma_projector_matches_independent_torch_reference() -> None:
    sidecar, package = _build_package("glma")
    features = np.linspace(-0.7, 0.8, 128, dtype=np.float32).reshape(1, 8, 16)
    x = torch.from_numpy(features)

    x = functional.gelu(
        functional.conv1d(
            x,
            torch.from_numpy(sidecar.get_tensor("a.conv1d.1.weight")),
            torch.from_numpy(sidecar.get_tensor("a.conv1d.1.bias").reshape(-1)),
            padding=1,
        )
    )
    x = functional.gelu(
        functional.conv1d(
            x,
            torch.from_numpy(sidecar.get_tensor("a.conv1d.2.weight")),
            torch.from_numpy(sidecar.get_tensor("a.conv1d.2.bias").reshape(-1)),
            stride=2,
            padding=1,
        )
    ).transpose(1, 2)
    x = x + torch.from_numpy(sidecar.get_tensor("a.position_embd.weight"))[: x.shape[1]]

    residual = x
    normed = _layer_norm(x, sidecar, "a.blk.0.ln1")
    q = _linear(normed, sidecar, "a.blk.0.attn_q")
    k = _linear(normed, sidecar, "a.blk.0.attn_k")
    v = _linear(normed, sidecar, "a.blk.0.attn_v")
    batch, sequence, hidden = q.shape
    heads, head_dim = 2, hidden // 2
    q = q.reshape(batch, sequence, heads, head_dim).transpose(1, 2)
    k = k.reshape(batch, sequence, heads, head_dim).transpose(1, 2)
    v = v.reshape(batch, sequence, heads, head_dim).transpose(1, 2)
    attention = torch.softmax((q * (head_dim**-0.5)) @ k.transpose(-1, -2), dim=-1)
    attention = (attention @ v).transpose(1, 2).reshape(batch, sequence, hidden)
    x = residual + _linear(attention, sidecar, "a.blk.0.attn_out")

    residual = x
    x = _layer_norm(x, sidecar, "a.blk.0.ln2")
    x = _linear(
        functional.gelu(_linear(x, sidecar, "a.blk.0.ffn_up")),
        sidecar,
        "a.blk.0.ffn_down",
    )
    x = residual + x
    x = _layer_norm(x, sidecar, "a.post_ln")
    x = _layer_norm(x, sidecar, "mm.a.norm_pre")
    x = x.reshape(1, -1, 16)
    x = _linear(
        functional.gelu(_linear(x, sidecar, "mm.a.mlp.1")),
        sidecar,
        "mm.a.mlp.2",
    )
    expected = torch.cat(
        [
            torch.from_numpy(sidecar.get_tensor("v.boi")).reshape(1, 1, -1),
            x,
            torch.from_numpy(sidecar.get_tensor("v.eoi")).reshape(1, 1, -1),
        ],
        dim=1,
    )[0].numpy()

    actual = _run_component(
        package,
        "audio_encoder",
        {"input_features": features},
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-4)


def _speaker_conv(
    value: torch.Tensor,
    sidecar: _FakeSidecar,
    stem: str,
    *,
    dilation: int = 1,
) -> torch.Tensor:
    weight = torch.from_numpy(sidecar.get_tensor(stem + ".weight"))
    bias = torch.from_numpy(sidecar.get_tensor(stem + ".bias"))
    padding = (weight.shape[-1] - 1) * dilation // 2
    if padding:
        value = functional.pad(value, (padding, padding), mode="reflect")
    return functional.conv1d(value, weight, bias, dilation=dilation)


def _speaker_block_reference(
    value: torch.Tensor,
    sidecar: _FakeSidecar,
    block: int,
    dilation: int,
) -> torch.Tensor:
    prefix = f"a.blk.{block}."
    residual = value
    value = functional.relu(_speaker_conv(value, sidecar, prefix + "conv_pw1"))
    chunks = value.chunk(8, dim=1)
    outputs = [chunks[0]]
    previous = None
    for index in range(1, 8):
        branch_input = chunks[index] if index == 1 else chunks[index] + previous
        previous = functional.relu(
            _speaker_conv(
                branch_input,
                sidecar,
                prefix + f"res2.{index - 1}",
                dilation=dilation,
            )
        )
        outputs.append(previous)
    value = torch.cat(outputs, dim=1)
    value = functional.relu(_speaker_conv(value, sidecar, prefix + "conv_pw2"))
    gate = value.mean(dim=2, keepdim=True)
    gate = functional.relu(_speaker_conv(gate, sidecar, prefix + "se_conv1"))
    gate = torch.sigmoid(_speaker_conv(gate, sidecar, prefix + "se_conv2"))
    return residual + value * gate


def test_qwen3tts_speaker_matches_independent_torch_reference() -> None:
    sidecar, package = _build_package("qwen3tts_spkenc")
    mel = np.linspace(-0.5, 0.6, 44, dtype=np.float32).reshape(11, 4)
    value = torch.from_numpy(mel).T.unsqueeze(0)
    value = functional.relu(_speaker_conv(value, sidecar, "a.conv1d.0"))
    block_outputs = []
    for block, dilation in zip(range(1, 4), (2, 3, 4)):
        value = _speaker_block_reference(value, sidecar, block, dilation)
        block_outputs.append(value)
    value = functional.relu(
        _speaker_conv(
            torch.cat(block_outputs, dim=1),
            sidecar,
            "a.conv_out",
        )
    )

    mean = value.mean(dim=2, keepdim=True)
    std = torch.sqrt(((value - mean) ** 2).mean(dim=2, keepdim=True) + 1e-12)
    attention_input = torch.cat(
        [value, mean.expand_as(value), std.expand_as(value)],
        dim=1,
    )
    attention = functional.relu(_speaker_conv(attention_input, sidecar, "a.asp_tdnn"))
    attention = torch.tanh(attention)
    attention = torch.softmax(
        _speaker_conv(attention, sidecar, "a.asp_attn"),
        dim=2,
    )
    weighted_mean = (attention * value).sum(dim=2)
    weighted_std = torch.sqrt(
        (attention * (value - weighted_mean.unsqueeze(2)) ** 2).sum(dim=2) + 1e-12
    )
    statistics = torch.cat([weighted_mean, weighted_std], dim=1).unsqueeze(2)
    expected = _speaker_conv(statistics, sidecar, "mm.a.fc").squeeze(2).numpy()

    actual = _run_component(
        package,
        "speaker_encoder",
        {"mel_features": mel},
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-4)


def test_qwen25o_alias_has_two_distinct_processor_abis() -> None:
    contract = _ROUTES["qwen2.5o"][0]().metadata
    assert contract["clip.projector_type"] == "qwen2.5o"
    assert contract["clip.has_vision_encoder"] is True
    assert contract["clip.has_audio_encoder"] is True


def test_qwen3tts_quarantines_only_generated_audio_namespace() -> None:
    sidecar = _speaker_sidecar()
    sidecar._shapes["unexpected.weight"] = (1,)
    sidecar._values["unexpected.weight"] = np.ones((1,), dtype=np.float32)
    sidecar.tensor_names.append("unexpected.weight")
    with (
        mock.patch(
            "mobius.integrations.gguf._mmproj._resolve_mmproj_companion_path",
            return_value="synthetic.gguf",
        ),
        mock.patch("mobius.integrations.gguf._builder._validate_gguf_model"),
        pytest.raises(ValueError, match="outside the pinned suffix-exact"),
    ):
        build_mmproj_from_gguf(
            "synthetic.gguf",
            projector_type="qwen3tts_spkenc",
            target_architecture="qwen3tts",
            _mmproj_gguf_model=sidecar,
        )


def test_qwen3tts_export_warns_that_runtime_prompt_assembly_is_unvalidated(
    caplog,
) -> None:
    _build_package("qwen3tts_spkenc")
    assert "downstream runtime orchestration is deferred" in caplog.text
    assert "tts_pad" in caplog.text


def test_qwen3vl_rejects_non_two_spatial_merge() -> None:
    sidecar = _qwen3vl_sidecar()
    sidecar.metadata["clip.vision.spatial_merge_size"] = 3
    with (
        mock.patch(
            "mobius.integrations.gguf._mmproj._resolve_mmproj_companion_path",
            return_value="synthetic.gguf",
        ),
        mock.patch("mobius.integrations.gguf._builder._validate_gguf_model"),
        pytest.raises(ValueError, match="spatial_merge_size=2"),
    ):
        build_mmproj_from_gguf(
            "synthetic.gguf",
            projector_type="qwen3vl_merger",
            target_architecture="qwen3vl",
            _mmproj_gguf_model=sidecar,
        )


def test_qwen3vl_multi_media_cardinality_and_deepstack_width() -> None:
    _, package = _build_package("qwen3vl_merger")
    pixels = np.arange(12 * 24, dtype=np.float32).reshape(12, 24) / 100
    grid = np.array([[1, 2, 2], [1, 2, 4]], dtype=np.int64)

    actual = _run_component(
        package,
        "vision_encoder",
        {"pixel_values": pixels, "image_grid_thw": grid},
    )

    # One 2x2 image contributes one merged row; one 2x4 image contributes two.
    assert actual.shape == (3, 24)
    assert np.isfinite(actual).all()


@pytest.mark.parametrize(
    ("pixels", "grid"),
    [
        (np.zeros((3, 24), np.float32), np.array([[1, 2, 2]], np.int64)),
        (np.zeros((6, 24), np.float32), np.array([[1, 2, 3]], np.int64)),
    ],
)
def test_qwen3vl_runtime_grid_contract_fails_closed(pixels, grid) -> None:
    _, package = _build_package("qwen3vl_merger")
    with pytest.raises(
        (
            ort.capi.onnxruntime_pybind11_state.Fail,
            ort.capi.onnxruntime_pybind11_state.InvalidArgument,
            ort.capi.onnxruntime_pybind11_state.RuntimeException,
        ),
        match="indices element out of data bounds",
    ):
        _run_component(
            package,
            "vision_encoder",
            {"pixel_values": pixels, "image_grid_thw": grid},
        )


def test_qwen3vl_decoder_positions_are_four_section_y_then_x() -> None:
    positions, next_position = qwen3vl_decoder_mrope_positions(
        merged_height=2,
        merged_width=3,
        start_position=7,
    )

    np.testing.assert_array_equal(
        positions,
        np.array(
            [
                [7, 7, 7, 7, 7, 7],
                [7, 7, 7, 8, 8, 8],
                [7, 8, 9, 7, 8, 9],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=np.int64,
        ),
    )
    assert next_position == 10


def test_packed_qtype_is_rejected_before_graph_construction() -> None:
    sidecar = _qwen2a_sidecar()
    sidecar.get_tensor_type = lambda name: SimpleNamespace(  # type: ignore[method-assign]
        name="Q8_0" if name == "mm.a.fc.weight" else "F32"
    )
    with (
        mock.patch(
            "mobius.integrations.gguf._mmproj._resolve_mmproj_companion_path",
            return_value="synthetic.gguf",
        ),
        mock.patch("mobius.integrations.gguf._builder._validate_gguf_model"),
        pytest.raises(NotImplementedError, match="packed Q8_0"),
    ):
        build_mmproj_from_gguf(
            "synthetic.gguf",
            projector_type="qwen2a",
            target_architecture="qwen2",
            _mmproj_gguf_model=sidecar,
        )


@pytest.mark.parametrize(
    ("projector_type", "frames"),
    [
        ("qwen2a", 16),
        ("qwen3a", 100),
        ("glma", 16),
        ("qwen2.5o", 16),
    ],
)
def test_reduced_precision_audio_casts_float_processor_input(
    projector_type: str,
    frames: int,
) -> None:
    _, package = _build_package(projector_type, dtype="f16")
    features = np.linspace(-0.2, 0.3, 8 * frames, dtype=np.float32).reshape(
        1,
        8,
        frames,
    )

    feeds = {"input_features": features}
    if projector_type == "qwen3a":
        feeds["input_features_mask"] = np.ones((1, frames), dtype=np.int32)
    actual = _run_component(package, "audio_encoder", feeds)

    assert actual.dtype == np.float16
    assert np.isfinite(actual).all()
