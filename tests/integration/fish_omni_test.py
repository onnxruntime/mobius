# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Reduced Drama 3 parity against the pinned Fish Speech inference implementation."""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import sys
import types
import urllib.request
from pathlib import Path

import ml_dtypes
import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch
from _test_configs import FISH_OMNI_CONFIG
from safetensors.torch import save_file

from mobius import build_from_module
from mobius._configs.fish_omni import DRAMA3_SOURCE_REVISION, FishOmniConfig
from mobius._model_package import ModelPackage
from mobius._testing.ort_inference import OnnxModelSession
from mobius.generation._fish_omni import build_fish_codebook_sampler
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
        dtype = ml_dtypes.bfloat16 if value.dtype == ir.DataType.BFLOAT16 else np.float32
        result[value.name] = np.zeros(shape, dtype=dtype)
    return result


def _next_cache(outputs):
    return {
        key.replace("present.", "past_key_values.", 1): value
        for key, value in outputs.items()
        if key.startswith("present.")
    }


@pytest.mark.parametrize("backbone_codebooks", [0, 1])
@pytest.mark.parametrize("task", ["fish-omni", "fish-omni-generation"])
@pytest.mark.parametrize(
    "dtype,execution_provider",
    [
        (ir.DataType.FLOAT, "default"),
        pytest.param(
            ir.DataType.BFLOAT16,
            "onnx-standard",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available()
                or "CUDAExecutionProvider" not in ort.get_available_providers(),
                reason="BF16 numerical parity requires PyTorch and ORT CUDA",
            ),
        ),
        pytest.param(
            ir.DataType.FLOAT,
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available()
                or "CUDAExecutionProvider" not in ort.get_available_providers(),
                reason="Fused sparse experts require CUDA",
            ),
        ),
    ],
)
def test_fish_omni_pipeline_prefill_and_cached_decode(
    upstream, tmp_path, monkeypatch, backbone_codebooks, dtype, execution_provider, task
):
    raw = copy.deepcopy(FISH_OMNI_CONFIG)
    raw["audio_decoder_config"]["vq_backbone_codebooks"] = backbone_codebooks
    reference_config = upstream["config"].FishOmniConfig.from_dict(raw)
    torch.manual_seed(42)
    tolerance = 1e-4 if dtype == ir.DataType.FLOAT else 1e-2
    torch_dtype = torch.float32 if dtype == ir.DataType.FLOAT else torch.bfloat16
    device = "cuda:0" if dtype != ir.DataType.FLOAT or execution_provider == "cuda" else "cpu"
    backbone = upstream["backbone"].FishOmniBackbone(
        reference_config.text, dtype=torch_dtype, device=device
    )
    tower = upstream["tower"].FishOmniAudioDecoder(
        reference_config, dtype=torch_dtype, device=device
    )
    for module in (backbone, tower):
        for parameter in module.parameters():
            torch.nn.init.normal_(parameter, mean=0.0, std=0.1)
        module.eval()
    config = dataclasses.replace(
        FishOmniConfig.from_transformers(_dict_to_pretrained_config(raw)), dtype=dtype
    )
    package = build_from_module(
        FishOmniModel(config),
        config,
        task=task,
        execution_provider=execution_provider,
    )
    save_file(_checkpoint(backbone, tower), tmp_path / "model.safetensors")
    stream_fish_omni_safetensors_to_package(package, str(tmp_path), config, revision=None)
    output = tmp_path / "onnx"
    package.save(str(output), max_workers=1, progress_bar=False)
    package = ModelPackage.load(str(output))
    assert set(package.policy_components) == {"main_sampler", "codebook_sampler"}
    if device != "cpu":
        ort.preload_dlls()
    sessions = {
        name: OnnxModelSession(
            model,
            device="cpu" if device == "cpu" else "cuda",
            enable_profiling=name == "decoder" and execution_provider == "cuda",
            profile_file_prefix=str(tmp_path / f"{name}_profile"),
        )
        for name, model in package.items()
    }
    fast = task == "fish-omni-generation"
    audio_name = "audio_generator" if fast else "audio_decoder"
    sampling = (
        {
            "generate": np.array(False),
            "do_sample": np.array(False),
            "temperature": np.ones(1, np.float32),
            "top_k": np.array([config.tower.vocab_size], np.int64),
            "top_p": np.ones(1, np.float32),
            "seed": np.array([42], np.int64),
            "counter": np.zeros(1, np.int64),
        }
        if fast
        else {}
    )
    backbone.setup_caches(1, 32)
    tower.setup_caches(32)
    caches = {name: _empty_cache(model) for name, model in package.items()}
    frames = torch.tensor(
        [[1, 0, 0, 0], [16, 0, 4, 7], [30, 14, 3, 5], [3, 0, 0, 0]], device=device
    )
    next_frames = torch.tensor(
        [[16, 0, 2, 8], [20, 4, 3, 1], [2, 0, 0, 0], [18, 2, 6, 9]], device=device
    )
    start = 0
    sampled_logits = []

    def greedy(logits, *args, **kwargs):
        sampled_logits.append(logits.detach().float().cpu().numpy())
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
                positions = torch.arange(start, start + length, device=device)
                positions_np = positions.cpu().numpy()[None]
                if start:
                    frames = next_frames[-1:].clone()
                embedded = sessions["embedding"].run({"frames": frames.cpu().numpy()[None]})
                ref_embed = tower.embed_backbone(frames, backbone.embed(frames[:, 0]))
                np.testing.assert_allclose(
                    embedded["inputs_embeds"][0].astype(np.float32),
                    ref_embed.float().cpu().numpy(),
                    atol=tolerance,
                    rtol=tolerance,
                )
                mask = np.ones((1, start + length), dtype=np.int64)
                outputs = sessions["decoder"].run(
                    {
                        **caches["decoder"],
                        "inputs_embeds": embedded["inputs_embeds"],
                        "position_ids": positions_np,
                        "attention_mask": mask,
                    }
                )
                hidden = backbone(inputs_embeds=ref_embed[None], input_pos=positions)
                np.testing.assert_allclose(
                    outputs["hidden_states"].astype(np.float32),
                    hidden.float().cpu().numpy(),
                    atol=tolerance,
                    rtol=tolerance,
                    err_msg=f"decoder hidden states at position {start}",
                )
                np.testing.assert_allclose(
                    outputs["logits"].astype(np.float32),
                    backbone.compute_logits(hidden).float().cpu().numpy(),
                    atol=tolerance,
                    rtol=tolerance,
                    err_msg=f"decoder logits at position {start}",
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
                    for i in range(1 if fast else config.num_codebooks - 1):
                        tower_output = sessions[audio_name].run(
                            {
                                **sampling,
                                **({"generate": np.array(True)} if fast else {}),
                                **caches[audio_name],
                                "hidden_states": outputs["hidden_states"],
                                "shared_input": embedded["shared_input"],
                                "codes": next_codes,
                                "semantic_mask": semantic,
                                "position_ids": positions_np,
                                "attention_mask": mask,
                            }
                        )
                        if fast:
                            assert tower_output["valid"].all()
                            next_codes = tower_output["frame_codes"]
                        else:
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
                        torch.zeros(1, 16, dtype=torch.int64, device=device),
                        temperature=1.0,
                        top_k=16,
                        top_p=1.0,
                    )
                    np.testing.assert_array_equal(
                        np.concatenate((np.array([[[main]]]), next_codes), axis=-1)[0],
                        next_frames.cpu().numpy(),
                    )
                    np.testing.assert_allclose(
                        tower_output["codebook_logits"].astype(np.float32),
                        np.stack(sampled_logits[2:], axis=-2)[None],
                        atol=tolerance,
                        rtol=tolerance,
                    )
                    caches["decoder"] = _next_cache(outputs)
                    caches[audio_name] = _next_cache(tower_output)
                    start += length
                    continue
                semantic = tower.is_semantic(next_frames[:, 0])
                tower_output = sessions[audio_name].run(
                    {
                        **sampling,
                        **caches[audio_name],
                        "hidden_states": outputs["hidden_states"],
                        "shared_input": embedded["shared_input"],
                        "codes": next_frames[:, 1:].cpu().numpy()[None],
                        "semantic_mask": semantic.cpu().numpy()[None],
                        "position_ids": positions_np,
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
                expected = torch.stack(expected, dim=-2)[None].float().cpu().numpy()
                np.testing.assert_allclose(
                    tower_output["codebook_logits"].astype(np.float32),
                    expected,
                    atol=tolerance,
                    rtol=tolerance,
                )
                caches["decoder"] = _next_cache(outputs)
                caches[audio_name] = _next_cache(tower_output)
                start += length
        if execution_provider == "cuda":
            profile = json.loads(
                Path(sessions["decoder"]._session.end_profiling()).read_text()
            )
            actual_moe = sum(
                item.get("args", {}).get("op_name") == "MoE"
                and item.get("args", {}).get("provider") == "CUDAExecutionProvider"
                for item in profile
            )
            assert actual_moe == 21 * config.num_hidden_layers
    finally:
        for session in sessions.values():
            session.close()


def test_sampled_tower_matches_sequential_replays_and_rejects_multiframe_generate(
    upstream, tmp_path
):
    raw = copy.deepcopy(FISH_OMNI_CONFIG)
    native_config = upstream["config"].FishOmniConfig.from_dict(raw)
    torch.manual_seed(42)
    backbone = upstream["backbone"].FishOmniBackbone(native_config.text, dtype=torch.float32)
    tower = upstream["tower"].FishOmniAudioDecoder(native_config, dtype=torch.float32)
    for module in (backbone, tower):
        for parameter in module.parameters():
            torch.nn.init.normal_(parameter, mean=0, std=0.1)
    config = FishOmniConfig.from_transformers(_dict_to_pretrained_config(raw))
    save_file(_checkpoint(backbone, tower), tmp_path / "model.safetensors")
    models = {}
    for task, name in (
        ("fish-omni", "audio_decoder"),
        ("fish-omni-generation", "audio_generator"),
    ):
        package = build_from_module(FishOmniModel(config), config, task=task)
        stream_fish_omni_safetensors_to_package(package, str(tmp_path), config, revision=None)
        models[name] = package[name]
    sessions = {
        name: OnnxModelSession(
            model,
            enable_profiling=name == "audio_generator",
            profile_file_prefix=str(tmp_path / f"{name}_profile"),
        )
        for name, model in models.items()
    }
    sampler = OnnxModelSession(build_fish_codebook_sampler(config).model)
    rng = np.random.default_rng(42)
    inputs = {
        "hidden_states": rng.normal(size=(1, 1, config.hidden_size)).astype(np.float32),
        "shared_input": rng.normal(size=(1, 1, config.hidden_size)).astype(np.float32),
        "semantic_mask": np.ones((1, 1), bool),
        "attention_mask": np.ones((1, 1), np.int64),
        "position_ids": np.zeros((1, 1), np.int64),
    }
    codes = np.zeros((1, 1, config.num_codebooks), np.int64)
    codes[0, 0, 0] = 3
    controls = {
        "temperature": np.array([0.8], np.float32),
        "top_k": np.array([4], np.int64),
        "top_p": np.array([0.9], np.float32),
        "seed": np.array([42], np.int64),
    }
    initial_counter = np.array([17], np.int64)
    try:
        expected_codes = codes.copy()
        counter = initial_counter
        for index in range(config.num_codebooks - 1):
            expected = sessions["audio_decoder"].run(
                {**_empty_cache(models["audio_decoder"]), **inputs, "codes": expected_codes}
            )
            sample = sampler.run(
                {
                    **controls,
                    "counter": counter,
                    "logits": expected["codebook_logits"][:, 0, index],
                }
            )
            expected_codes[0, 0, index + 1] = sample["token"][0]
            counter = sample["next_counter"]
        actual = sessions["audio_generator"].run(
            {
                **_empty_cache(models["audio_generator"]),
                **inputs,
                **controls,
                "codes": codes,
                "counter": initial_counter,
                "generate": np.array(True),
                "do_sample": np.array(True),
            }
        )
        assert actual["valid"].all()
        np.testing.assert_array_equal(actual["frame_codes"], expected_codes)
        np.testing.assert_array_equal(actual["next_counter"], counter)
        for name, value in expected.items():
            np.testing.assert_allclose(actual[name], value, atol=1e-4, rtol=1e-4)
        doubled = {key: np.repeat(value, 2, axis=1) for key, value in inputs.items()}
        doubled["position_ids"] = np.array([[0, 1]], np.int64)
        invalid = sessions["audio_generator"].run(
            {
                **_empty_cache(models["audio_generator"]),
                **doubled,
                **controls,
                "codes": np.repeat(codes, 2, axis=1),
                "counter": initial_counter,
                "generate": np.array(True),
                "do_sample": np.array(False),
            }
        )
        assert not invalid["valid"].any()
        assert (invalid["frame_codes"] == -1).all()
        np.testing.assert_array_equal(invalid["next_counter"], initial_counter)
        prefill = sessions["audio_generator"].run(
            {
                **_empty_cache(models["audio_generator"]),
                **doubled,
                **controls,
                "temperature": np.array([np.nan], np.float32),
                "top_p": np.array([np.nan], np.float32),
                "codes": np.repeat(codes, 2, axis=1),
                "counter": initial_counter,
                "generate": np.array(False),
                "do_sample": np.array(True),
            }
        )
        assert prefill["valid"].all()
        np.testing.assert_array_equal(prefill["next_counter"], initial_counter)
        profile = json.loads(
            Path(sessions["audio_generator"]._session.end_profiling()).read_text()
        )
        assert (
            sum(item.get("args", {}).get("op_name") == "TopK" for item in profile)
            == config.num_codebooks - 1
        )
    finally:
        sampler.close()
        for session in sessions.values():
            session.close()
