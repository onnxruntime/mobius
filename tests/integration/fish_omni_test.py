# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Reduced Drama 3 parity against the pinned Fish Speech inference implementation."""

from __future__ import annotations

import copy
import importlib.util
import sys
import types
import urllib.request

import numpy as np
import pytest
import torch
from _test_configs import FISH_OMNI_CONFIG
from safetensors.torch import save_file

from mobius import build_from_module
from mobius._configs.fish_omni import DRAMA3_SOURCE_REVISION, FishOmniConfig
from mobius._model_package import ModelPackage
from mobius._testing.ort_inference import OnnxModelSession
from mobius.integrations.transformers._config_resolver import _dict_to_pretrained_config
from mobius.integrations.transformers._fish_omni_weights import (
    stream_fish_omni_safetensors_to_package,
)
from mobius.models import FishOmniModel

pytestmark = pytest.mark.integration
_PREFIX = "fish_speech.models.text2semantic.fish_omni"


@pytest.fixture
def upstream(tmp_path, monkeypatch):
    """Load unmodified reference files in an isolated temporary package namespace."""
    for name in (
        "fish_speech",
        "fish_speech.models",
        "fish_speech.models.text2semantic",
        _PREFIX,
    ):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    modules = {}
    for name in ("fp8_moe", "config", "layers", "moe", "backbone", "tower"):
        url = (
            f"https://raw.githubusercontent.com/fishaudio/fish-speech/"
            f"{DRAMA3_SOURCE_REVISION}/fish_speech/models/text2semantic/fish_omni/{name}.py"
        )
        with urllib.request.urlopen(url, timeout=60) as response:
            path = tmp_path / f"{name}.py"
            path.write_bytes(response.read())
        spec = importlib.util.spec_from_file_location(f"{_PREFIX}.{name}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


def _checkpoint(backbone, tower):
    weights = {}
    for name, value in backbone.state_dict().items():
        if name.startswith("model."):
            name = "model.language_model." + name.removeprefix("model.")
        if name.endswith(".experts.w13_weight"):
            prefix = name.removesuffix("w13_weight")
            for expert, pair in enumerate(value):
                gate, up = pair.chunk(2)
                weights[f"{prefix}{expert}.gate_proj.weight"] = gate.contiguous()
                weights[f"{prefix}{expert}.up_proj.weight"] = up.contiguous()
        elif name.endswith(".experts.w2_weight"):
            prefix = name.removesuffix("w2_weight")
            for expert, down in enumerate(value):
                weights[f"{prefix}{expert}.down_proj.weight"] = down.contiguous()
        else:
            weights[name] = value.contiguous()
    for name, value in tower.state_dict().items():
        for local, source in (
            ("codebook_embeddings.", "audio_decoder.codebook_embeddings."),
            ("shared_layers.", "audio_decoder_mtp_shared.layers."),
            ("residual_layers.", "audio_decoder_mtp.layers."),
            ("codebook_heads.", "audio_decoder_mtp.heads."),
        ):
            if name.startswith(local):
                weights[source + name.removeprefix(local)] = value.contiguous()
                break
        else:
            raise AssertionError(name)
    return weights


def _empty_cache(model):
    result = {}
    for value in model.graph.inputs:
        if not value.name.startswith("past_key_values."):
            continue
        shape = list(value.shape)
        shape[0] = 1
        if value.name.endswith((".key", ".value")):
            shape[2] = 0
        assert all(isinstance(dim, int) for dim in shape), (value.name, shape)
        result[value.name] = np.zeros(shape, dtype=np.float32)
    return result


def _next_cache(outputs):
    return {
        key.replace("present.", "past_key_values.", 1): value
        for key, value in outputs.items()
        if key.startswith("present.")
    }


@pytest.mark.parametrize("backbone_codebooks", [0, 1])
def test_fish_omni_pipeline_prefill_and_cached_decode(
    upstream, tmp_path, monkeypatch, backbone_codebooks
):
    raw = copy.deepcopy(FISH_OMNI_CONFIG)
    raw["audio_decoder_config"]["vq_backbone_codebooks"] = backbone_codebooks
    reference_config = upstream["config"].FishOmniConfig.from_dict(raw)
    torch.manual_seed(42)
    tolerance = 1e-4
    backbone = upstream["backbone"].FishOmniBackbone(
        reference_config.text, dtype=torch.float32
    )
    tower = upstream["tower"].FishOmniAudioDecoder(reference_config, dtype=torch.float32)
    for module in (backbone, tower):
        for parameter in module.parameters():
            torch.nn.init.normal_(parameter, mean=0.0, std=0.1)
        module.eval()
    config = FishOmniConfig.from_transformers(_dict_to_pretrained_config(raw))
    package = build_from_module(FishOmniModel(config), config, task="fish-omni")
    save_file(_checkpoint(backbone, tower), tmp_path / "model.safetensors")
    stream_fish_omni_safetensors_to_package(package, str(tmp_path), config, revision=None)
    output = tmp_path / "onnx"
    package.save(str(output), max_workers=1, progress_bar=False)
    package = ModelPackage.load(str(output))
    sessions = {name: OnnxModelSession(model) for name, model in package.items()}
    backbone.setup_caches(1, 32)
    tower.setup_caches(32)
    caches = {name: _empty_cache(model) for name, model in package.items()}
    frames = torch.tensor([[1, 0, 0, 0], [16, 0, 4, 7], [30, 14, 3, 5], [3, 0, 0, 0]])
    next_frames = torch.tensor([[16, 0, 2, 8], [20, 4, 3, 1], [2, 0, 0, 0], [18, 2, 6, 9]])
    start = 0
    sampled_logits = []

    def greedy(logits, *args, **kwargs):
        sampled_logits.append(logits.detach().float().numpy())
        return logits.argmax(dim=-1)

    monkeypatch.setattr(
        upstream["tower"],
        "sample_top_k_top_p",
        greedy,
    )
    try:
        # Multi-token prompt followed by 20 reused-state steps exercises both
        # DeltaNet state and all shared/residual tower caches across time.
        with torch.no_grad():
            for length in [4] + [1] * 20:
                positions = torch.arange(start, start + length)
                if start:
                    frames = next_frames[-1:].clone()
                embedded = sessions["embedding"].run({"frames": frames.numpy()[None]})
                ref_embed = tower.embed_backbone(frames, backbone.embed(frames[:, 0]))
                np.testing.assert_allclose(
                    embedded["inputs_embeds"][0].astype(np.float32),
                    ref_embed.float().numpy(),
                    atol=tolerance,
                    rtol=tolerance,
                )
                mask = np.ones((1, start + length), dtype=np.int64)
                outputs = sessions["decoder"].run(
                    {
                        **caches["decoder"],
                        "inputs_embeds": embedded["inputs_embeds"],
                        "position_ids": positions.numpy()[None],
                        "attention_mask": mask,
                    }
                )
                hidden = backbone(inputs_embeds=ref_embed[None], input_pos=positions)
                np.testing.assert_allclose(
                    outputs["hidden_states"].astype(np.float32),
                    hidden.float().numpy(),
                    atol=tolerance,
                    rtol=tolerance,
                )
                np.testing.assert_allclose(
                    outputs["logits"].astype(np.float32),
                    backbone.compute_logits(hidden).float().numpy(),
                    atol=tolerance,
                    rtol=tolerance,
                )
                if start:
                    candidates = [
                        config.eos_token_id,
                        *range(
                            config.semantic_start_token_id, config.semantic_end_token_id + 1
                        ),
                    ]
                    main = candidates[
                        int(outputs["logits"][0, -1, candidates].astype(np.float32).argmax())
                    ]
                    next_codes = np.zeros((1, 1, config.num_codebooks), dtype=np.int64)
                    next_codes[0, 0, 0] = np.clip(
                        main - config.semantic_start_token_id, 0, config.tower.vocab_size - 1
                    )
                    semantic = np.array(
                        [
                            [
                                config.semantic_start_token_id
                                <= main
                                <= config.semantic_end_token_id
                            ]
                        ]
                    )
                    for i in range(config.num_codebooks - 1):
                        tower_output = sessions["audio_decoder"].run(
                            {
                                **caches["audio_decoder"],
                                "hidden_states": outputs["hidden_states"],
                                "shared_input": embedded["shared_input"],
                                "codes": next_codes,
                                "semantic_mask": semantic,
                                "position_ids": positions.numpy()[None],
                                "attention_mask": mask,
                            }
                        )
                        next_codes[0, 0, i + 1] = (
                            tower_output["codebook_logits"][0, 0, i]
                            .astype(np.float32)
                            .argmax()
                        )
                    sampled_logits.clear()
                    next_frames = tower.decode_step(
                        backbone.compute_logits(hidden)[0],
                        hidden[0],
                        positions,
                        frames,
                        torch.zeros(1, 16, dtype=torch.int64),
                        temperature=1.0,
                        top_k=16,
                        top_p=1.0,
                    )
                    np.testing.assert_array_equal(
                        np.concatenate((np.array([[[main]]]), next_codes), axis=-1)[0],
                        next_frames.numpy(),
                    )
                    np.testing.assert_allclose(
                        tower_output["codebook_logits"].astype(np.float32),
                        np.stack(sampled_logits[2:], axis=-2)[None],
                        atol=tolerance,
                        rtol=tolerance,
                    )
                    caches["decoder"] = _next_cache(outputs)
                    caches["audio_decoder"] = _next_cache(tower_output)
                    start += length
                    continue
                semantic = tower.is_semantic(next_frames[:, 0])
                tower_output = sessions["audio_decoder"].run(
                    {
                        **caches["audio_decoder"],
                        "hidden_states": outputs["hidden_states"],
                        "shared_input": embedded["shared_input"],
                        "codes": next_frames[:, 1:].numpy()[None],
                        "semantic_mask": semantic.numpy()[None],
                        "position_ids": positions.numpy()[None],
                        "attention_mask": mask,
                    }
                )
                h = tower.run_shared(hidden[0], next_frames[:, 1], semantic, positions, frames)
                expected = []
                for i, (layer, head) in enumerate(
                    zip(tower.residual_layers, tower.codebook_heads)
                ):
                    h = layer(h, next_frames[:, i + 1], semantic, positions)
                    expected.append(head(h))
                expected = torch.stack(expected, dim=-2)[None].float().numpy()
                np.testing.assert_allclose(
                    tower_output["codebook_logits"].astype(np.float32),
                    expected,
                    atol=tolerance,
                    rtol=tolerance,
                )
                caches["decoder"] = _next_cache(outputs)
                caches["audio_decoder"] = _next_cache(tower_output)
                start += length
    finally:
        for session in sessions.values():
            session.close()
